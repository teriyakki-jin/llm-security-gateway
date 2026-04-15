"""GET /metrics — Prometheus metrics endpoint."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["observability"])


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def prometheus_metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for scraping."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
