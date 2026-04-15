"""POST /v1/chat/completions — proxies requests to the configured LLM provider."""

import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_security_gateway.config import GatewaySettings, get_settings
from llm_security_gateway.dependencies import get_detection_engine, get_response_filter
from llm_security_gateway.detection.data_leakage.response_filter import ResponseFilter
from llm_security_gateway.detection.engine import DetectionEngine
from llm_security_gateway.llm_clients.base import BaseLLMClient, Message
from llm_security_gateway.llm_clients.factory import create_client
from llm_security_gateway.metrics import llm_latency_seconds, llm_requests_total

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

@router.post("/chat/completions")
async def chat_completions(
    body: ChatRequest,
    request: Request,
    settings: GatewaySettings = Depends(get_settings),
    engine: DetectionEngine = Depends(get_detection_engine),
    resp_filter: ResponseFilter = Depends(get_response_filter),
) -> ChatResponse | StreamingResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # ── Request Detection ─────────────────────────────────────
    if settings.detection_enabled:
        full_text = " ".join(m.content for m in body.messages)
        detection_result = engine.analyze(full_text, request_id=request_id)
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

    # ── Build LLM client ──────────────────────────────────────
    client: BaseLLMClient = create_client(settings.default_provider, settings)

    try:
        if body.stream:
            return await _handle_streaming(
                client=client,
                body=body,
                resp_filter=resp_filter,
                settings=settings,
                request_id=request_id,
            )
        return await _handle_blocking(
            client=client,
            body=body,
            resp_filter=resp_filter,
            settings=settings,
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        llm_requests_total.labels(provider=settings.default_provider, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        ) from exc
    finally:
        await client.close()


# ── Blocking (non-streaming) ──────────────────────────────────

async def _handle_blocking(
    *,
    client: BaseLLMClient,
    body: ChatRequest,
    resp_filter: ResponseFilter,
    settings: GatewaySettings,
    request_id: str,
) -> ChatResponse:
    t0 = time.perf_counter()
    llm_response = await client.chat(
        messages=[Message(role=m.role, content=m.content) for m in body.messages],
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    llm_latency_seconds.observe(time.perf_counter() - t0)
    llm_requests_total.labels(provider=settings.default_provider, status="success").inc()

    content = llm_response.content
    if settings.detection_enabled:
        content, _ = resp_filter.filter(content, request_id=request_id)

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


# ── Streaming (SSE) ───────────────────────────────────────────

async def _handle_streaming(
    *,
    client: BaseLLMClient,
    body: ChatRequest,
    resp_filter: ResponseFilter,
    settings: GatewaySettings,
    request_id: str,
) -> StreamingResponse:
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    provider = settings.default_provider

    async def _generate() -> AsyncIterator[str]:
        chunks: list[str] = []
        t0 = time.perf_counter()

        try:
            async for delta in await client.stream_chat(
                messages=[Message(role=m.role, content=m.content) for m in body.messages],
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                chunks.append(delta)
                event = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as exc:
            llm_requests_total.labels(provider=provider, status="error").inc()
            error_event = {"error": {"message": str(exc), "type": "provider_error"}}
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"
            return

        llm_latency_seconds.observe(time.perf_counter() - t0)
        llm_requests_total.labels(provider=provider, status="success").inc()

        # Apply response filter to accumulated content and emit correction if needed.
        if settings.detection_enabled and chunks:
            accumulated = "".join(chunks)
            filtered, was_filtered = resp_filter.filter(accumulated, request_id=request_id)
            if was_filtered:
                correction = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": None,
                            "filter_applied": True,
                        }
                    ],
                }
                yield f"data: {json.dumps(correction)}\n\n"

        # Final SSE done marker.
        done_event = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
