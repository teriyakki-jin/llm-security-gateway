"""Admin API — shadow mode toggle, threshold adjustment, live stats."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from llm_security_gateway.config import GatewaySettings, get_settings
from llm_security_gateway.dependencies import get_detection_engine
from llm_security_gateway.detection.engine import DetectionEngine
from llm_security_gateway.metrics import detection_threshold_current, shadow_mode_active

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Auth dependency ────────────────────────────────────────────

def _verify_admin_key(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    settings: GatewaySettings = Depends(get_settings),
) -> None:
    if settings.admin_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API is not configured (set ADMIN_API_KEY env var)",
        )
    if x_admin_key != settings.admin_api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin key",
        )


# ── Request models ─────────────────────────────────────────────

class ShadowModeRequest(BaseModel):
    enabled: bool


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0, description="Risk score threshold (0.0–1.0)")


# ── Endpoints ─────────────────────────────────────────────────

@router.post(
    "/shadow-mode",
    dependencies=[Depends(_verify_admin_key)],
    summary="Toggle shadow mode on the detection engine",
)
async def set_shadow_mode(
    body: ShadowModeRequest,
    engine: DetectionEngine = Depends(get_detection_engine),
) -> dict[str, bool]:
    engine.set_shadow_mode(body.enabled)
    shadow_mode_active.set(1.0 if body.enabled else 0.0)
    return {"shadow_mode": body.enabled}


@router.post(
    "/threshold",
    dependencies=[Depends(_verify_admin_key)],
    summary="Adjust the detection blocking threshold at runtime",
)
async def set_threshold(
    body: ThresholdRequest,
    engine: DetectionEngine = Depends(get_detection_engine),
) -> dict[str, float]:
    engine.set_threshold(body.threshold)
    detection_threshold_current.set(body.threshold)
    return {"threshold": body.threshold}


@router.get(
    "/stats",
    dependencies=[Depends(_verify_admin_key)],
    summary="Return live detection engine stats",
)
async def get_stats(
    engine: DetectionEngine = Depends(get_detection_engine),
) -> dict[str, object]:
    return {
        "shadow_mode": engine.shadow_mode,
        "threshold": engine.threshold,
    }
