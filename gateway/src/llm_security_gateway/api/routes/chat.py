"""POST /v1/chat/completions — proxies requests to the configured LLM provider."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from llm_security_gateway.config import GatewaySettings, get_settings
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
) -> ChatResponse:
    if body.stream:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming is not yet supported. Coming in Phase 4.",
        )

    # ── Detection ─────────────────────────────────────────────
    # Detection engine will be wired in Phase 3.
    # For now, check the state set by future middleware.
    detection_result = getattr(request.state, "detection_result", None)
    if detection_result is not None and detection_result.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "request_blocked",
                "reason": detection_result.labels,
                "risk_score": detection_result.risk_score,
            },
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

    import uuid
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=llm_response.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=llm_response.content),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            total_tokens=llm_response.usage.total_tokens,
        ),
    )
