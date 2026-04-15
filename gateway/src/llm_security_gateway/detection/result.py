"""Detection result data classes — immutable by design."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectorOutput:
    """Output from a single detector in the pipeline."""
    detector_name: str
    risk_score: float          # 0.0 – 1.0
    labels: tuple[str, ...]   # e.g. ("instruction_override", "role_manipulation")
    matched_patterns: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class DetectionResult:
    """
    Final aggregated result from the DetectionEngine.

    is_blocked reflects the engine's decision AFTER applying shadow_mode:
    - shadow_mode=False: is_blocked=True triggers HTTP 403.
    - shadow_mode=True:  is_blocked is always False regardless of risk_score
                         (requests pass through, but would_block=True is logged).
    """
    is_blocked: bool
    risk_score: float
    labels: tuple[str, ...]
    details: tuple[DetectorOutput, ...]
    latency_ms: float
    # would_block records what the engine *would* have decided without shadow mode.
    # Allows offline model evaluation without impacting traffic.
    would_block: bool = False

    @property
    def is_suspicious(self) -> bool:
        """True when risk is elevated but below blocking threshold."""
        return not self.is_blocked and self.risk_score >= 0.5
