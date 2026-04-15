from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Server ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # ── LLM Providers ─────────────────────────────────────────
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    default_provider: Literal["openai", "anthropic"] = "openai"
    llm_timeout_sec: int = 60

    # ── Detection ─────────────────────────────────────────────
    detection_enabled: bool = True
    # Risk score threshold above which a request is blocked (0.0 – 1.0).
    detection_threshold: float = 0.85
    # When True, log detections but never block (shadow mode for tuning).
    detection_shadow_mode: bool = False

    # ── Rate Limiting ─────────────────────────────────────────
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10

    # ── Storage ───────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://gateway:gateway@localhost:5432/gateway"
    redis_url: str = "redis://localhost:6379"

    # ── Logging ───────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    audit_log_enabled: bool = True

    # ── Admin ─────────────────────────────────────────────────
    admin_api_key: SecretStr | None = None

    @field_validator("detection_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"detection_threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("rate_limit_rpm")
    @classmethod
    def validate_rpm(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rate_limit_rpm must be >= 1")
        return v


def get_settings() -> GatewaySettings:
    """Return cached settings instance (override in tests via dependency injection)."""
    return GatewaySettings()
