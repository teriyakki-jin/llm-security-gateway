"""Audit logging middleware — records every request/response to DB asynchronously."""

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from llm_security_gateway.storage.models import AuditLog

logger = structlog.get_logger(__name__)

# Background queue for fire-and-forget DB writes.
_audit_queue: asyncio.Queue[AuditLog] = asyncio.Queue(maxsize=10_000)


class AuditLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, get_session: Callable, enabled: bool = True) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._get_session = get_session
        self._enabled = enabled

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if not self._enabled:
            return await call_next(request)  # type: ignore[misc]

        start = time.perf_counter()
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Read body for hashing (re-inject so downstream can still read it).
        body = await request.body()
        body_hash = hashlib.sha256(body).hexdigest() if body else None

        response: Response = await call_next(request)  # type: ignore[misc]

        latency_ms = (time.perf_counter() - start) * 1000
        is_blocked = response.status_code == 403
        risk_score = float(response.headers.get("X-Risk-Score", 0.0))

        log_entry = AuditLog(
            request_id=request_id,
            client_ip=_get_client_ip(request),
            path=request.url.path,
            request_body_hash=body_hash,
            response_status=response.status_code,
            latency_ms=round(latency_ms, 2),
            llm_provider=response.headers.get("X-LLM-Provider"),
            tokens_used=_parse_int(response.headers.get("X-Tokens-Used")),
            is_blocked=is_blocked,
            risk_score=risk_score if risk_score > 0 else None,
        )

        # Enqueue for background writing — never block the response.
        try:
            _audit_queue.put_nowait(log_entry)
        except asyncio.QueueFull:
            logger.warning("audit_queue_full", request_id=request_id)

        logger.info(
            "request",
            request_id=request_id,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(latency_ms, 2),
            blocked=is_blocked,
        )

        return response


async def audit_log_worker(get_session: Callable) -> None:
    """Background task: drains the audit queue and writes to DB in batches."""
    while True:
        batch: list[AuditLog] = []
        try:
            entry = await asyncio.wait_for(_audit_queue.get(), timeout=5.0)
            batch.append(entry)
            # Drain up to 49 more without waiting.
            while not _audit_queue.empty() and len(batch) < 50:
                batch.append(_audit_queue.get_nowait())
        except asyncio.TimeoutError:
            continue

        try:
            async with get_session() as session:
                session: AsyncSession
                session.add_all(batch)
                await session.commit()
        except Exception:
            logger.exception("audit_log_write_failed", batch_size=len(batch))


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
