"""Health and readiness endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from llm_security_gateway.config import GatewaySettings, get_settings
from llm_security_gateway.llm_clients.base import BaseLLMClient
from llm_security_gateway.llm_clients.factory import create_client

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    llm_reachable: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(settings: GatewaySettings = Depends(get_settings)) -> ReadinessResponse:
    client: BaseLLMClient = create_client(settings.default_provider, settings)
    try:
        reachable = await client.health()
    finally:
        await client.close()

    return ReadinessResponse(
        status="ready" if reachable else "degraded",
        llm_reachable=reachable,
    )
