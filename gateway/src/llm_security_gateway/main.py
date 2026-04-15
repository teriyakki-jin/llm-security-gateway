"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis

from llm_security_gateway.api.middleware.audit_log import AuditLogMiddleware, audit_log_worker
from llm_security_gateway.api.middleware.rate_limit import RateLimitMiddleware
from llm_security_gateway.api.middleware.request_id import RequestIDMiddleware
from llm_security_gateway.api.routes import chat, health
from llm_security_gateway.config import GatewaySettings
from llm_security_gateway.storage.database import dispose_db, get_session, init_db

logger = structlog.get_logger(__name__)


def create_app(settings: GatewaySettings | None = None) -> FastAPI:
    if settings is None:
        from llm_security_gateway.config import get_settings
        settings = get_settings()

    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("startup", provider=settings.default_provider)

        # Initialize DB connection pool.
        init_db(settings)

        # Initialize Redis.
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        app.state.redis = redis

        # Start background audit log writer.
        audit_task = asyncio.create_task(audit_log_worker(get_session))
        app.state.audit_task = audit_task

        logger.info("startup_complete")
        yield

        # Shutdown: cancel background task, close connections.
        audit_task.cancel()
        try:
            await audit_task
        except asyncio.CancelledError:
            pass

        await redis.aclose()
        await dispose_db()
        logger.info("shutdown_complete")

    app = FastAPI(
        title="LLM Security Gateway",
        description="PQC-secured LLM proxy with prompt injection and jailbreak detection",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # ── Middleware (outermost = last applied = first executed) ─
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        AuditLogMiddleware,
        get_session=get_session,
        enabled=settings.audit_log_enabled,
    )
    # Rate limiter is added after startup (needs Redis instance).
    # Wired via a startup event below.

    @app.on_event("startup")  # type: ignore[misc]
    async def _add_rate_limit_middleware() -> None:
        app.add_middleware(
            RateLimitMiddleware,
            redis=app.state.redis,
            rpm=settings.rate_limit_rpm,
            burst=settings.rate_limit_burst,
        )

    # ── Routers ───────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router)

    return app


def _configure_logging(settings: GatewaySettings) -> None:
    import logging
    import structlog

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
