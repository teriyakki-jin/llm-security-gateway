"""SQLAlchemy ORM models for audit logs and detection events."""

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    client_ip: Mapped[str] = mapped_column(String(45), nullable=False)
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    # SHA-256 of the raw request body — never store the body itself.
    request_body_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response_status: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_blocked: Mapped[bool] = mapped_column(default=False, nullable=False)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    detection_events: Mapped[list["DetectionEvent"]] = relationship(
        back_populates="audit_log", cascade="all, delete-orphan"
    )


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    audit_log_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("audit_logs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    detector_name: Mapped[str] = mapped_column(String(100), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON list of label strings e.g. ["instruction_override", "role_manipulation"]
    labels: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    matched_patterns: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    is_blocked: Mapped[bool] = mapped_column(default=False, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    audit_log: Mapped["AuditLog"] = relationship(back_populates="detection_events")


class RateLimitRule(Base):
    __tablename__ = "rate_limit_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    identifier_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "ip" | "api_key"
    identifier_value: Mapped[str] = mapped_column(String(255), nullable=False)
    rpm: Mapped[int] = mapped_column(Integer, nullable=False)
    burst: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
