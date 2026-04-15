"""POST /v1/chat/completions — proxies requests to the configured LLM provider."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from llm_security_gateway.config import GatewaySettings, get_settings
from llm_security_gateway.dependencies import get_detection_engine, get_response_filter
from llm_security_gateway.detection.engine import DetectionEngine
from llm_security_gateway.detection.data_leakage.response_filter import ResponseFilter
from llm_security_gateway.llm_clients.base import BaseLLMClient, Message
from llm_security_gateway.llm_clients.factory import create_client

router = APIRouter(prefix="/v1", tags=["chat"])


# ── Request / Response models ─────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


# ── Route ─────────────────────────────────────────────────────

@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest,
    request: Request,
    settings: GatewaySettings = Depends(get_settings),
    engine: DetectionEngine = Depends(get_detection_engine),
    resp_filter: ResponseFilter = Depends(get_response_filter),
) -> ChatResponse:
    if body.stream:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming is not yet supported. Coming in Phase 4.",
        )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # ── Request Detection ─────────────────────────────────────
    if settings.detection_enabled:
        full_text = " ".join(m.content for m in body.messages)
        detection_result = engine.analyze(full_text, request_id=request_id)

        # Expose risk score on request.state for audit middleware.
        request.state.detection_result = detection_result

        if detection_result.is_blocked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "request_blocked",
                    "reason": list(detection_result.labels),
                    "risk_score": detection_result.risk_score,
                },
                headers={"X-Risk-Score": str(detection_result.risk_score)},
            )

    # ── Forward to LLM ────────────────────────────────────────
    client: BaseLLMClient = create_client(settings.default_provider, settings)
    try:
        llm_response = await client.chat(
            messages=[Message(role=m.role, content=m.content) for m in body.messages],
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        ) from exc
    finally:
        await client.close()

    # ── Response Filtering (PII / Secret) ────────────────────
    content = llm_response.content
    if settings.detection_enabled:
        content, was_filtered = resp_filter.filter(content, request_id=request_id)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=llm_response.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            total_tokens=llm_response.usage.total_tokens,
        ),
    )
