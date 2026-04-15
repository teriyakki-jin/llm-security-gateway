"""
DetectionEngine — orchestrates the multi-layer detection pipeline.

Pipeline order (cheapest → most expensive):
  1. RuleDetector      — regex/keyword  (~0.1ms)
  2. HeuristicDetector — statistics     (~0.5ms)
  3. MLDetector        — ONNX DeBERTa   (~20-50ms, skipped on early exit)

Shadow Mode:
  When shadow_mode=True the engine always returns is_blocked=False so traffic
  is never interrupted. However would_block=True is set and a structured log
  entry is emitted. This lets operators:
    1. Validate the model's precision/recall on real traffic before going live.
    2. Tune thresholds without risking false positives in production.
    3. Build a ground-truth dataset by reviewing shadow-mode alerts.

  Recommended rollout:
    shadow_mode=True (1–2 weeks) → review FP/FN → shadow_mode=False
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from llm_security_gateway.detection.result import DetectionResult, DetectorOutput

if TYPE_CHECKING:
    from llm_security_gateway.detection.base import BaseDetector

logger = structlog.get_logger(__name__)

# Detector weights for weighted-average score combination.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "rule": 1.0,
    "heuristic": 0.3,
    "ml": 1.2,
}


class DetectionEngine:
    """
    Orchestrates the detection pipeline and applies Shadow Mode logic.

    Args:
        detectors: Ordered list of detectors (rule → heuristic → ml).
        threshold: Risk score above which a request is blocked.
        shadow_mode: If True, never block — only log what would have been blocked.
        weights: Per-detector score weights for weighted average combination.
    """

    def __init__(
        self,
        detectors: list[BaseDetector],
        threshold: float = 0.85,
        shadow_mode: bool = False,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._detectors = detectors
        self._threshold = threshold
        self._shadow_mode = shadow_mode
        self._weights = weights or _DEFAULT_WEIGHTS

    def analyze(self, text: str, *, request_id: str = "") -> DetectionResult:
        """
        Run all detectors and return a unified DetectionResult.

        Early exit: if a rule-based detector returns risk_score >= 0.99
        (i.e., exact pattern match with critical severity) the ML detector
        is skipped to save latency.
        """
        start = time.perf_counter()
        outputs: list[DetectorOutput] = []
        total_weight = 0.0
        weighted_sum = 0.0

        for detector in self._detectors:
            output = detector.detect(text)
            outputs.append(output)

            weight = self._weights.get(detector.detector_type, 1.0)
            weighted_sum += output.risk_score * weight
            total_weight += weight

            # Early exit: definitive rule match skips expensive ML inference.
            if detector.detector_type == "rule" and output.risk_score >= 0.99:
                logger.debug(
                    "detection_early_exit",
                    request_id=request_id,
                    detector=detector.detector_type,
                    score=output.risk_score,
                )
                break

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        all_labels = tuple(label for o in outputs for label in o.labels)
        would_block = final_score >= self._threshold

        latency_ms = (time.perf_counter() - start) * 1000

        if would_block:
            self._emit_detection_log(
                request_id=request_id,
                risk_score=final_score,
                labels=all_labels,
                shadow_mode=self._shadow_mode,
                latency_ms=latency_ms,
            )

        return DetectionResult(
            is_blocked=would_block and not self._shadow_mode,
            risk_score=round(final_score, 4),
            labels=all_labels,
            details=tuple(outputs),
            latency_ms=round(latency_ms, 2),
            would_block=would_block,
        )

    @property
    def shadow_mode(self) -> bool:
        return self._shadow_mode

    def set_shadow_mode(self, enabled: bool) -> None:
        """Toggle shadow mode at runtime (e.g., via admin API)."""
        self._shadow_mode = enabled
        logger.info("shadow_mode_changed", enabled=enabled)

    @staticmethod
    def _emit_detection_log(
        *,
        request_id: str,
        risk_score: float,
        labels: tuple[str, ...],
        shadow_mode: bool,
        latency_ms: float,
    ) -> None:
        """
        Emit a structured log event for every would-be block.

        In shadow mode this is the PRIMARY signal for model evaluation:
          - Collect these logs → review FP/FN → adjust threshold or retrain.
          - Feed into a monitoring dashboard (Grafana alert on shadow_block rate).
        """
        logger.warning(
            "detection_block" if not shadow_mode else "detection_shadow_block",
            request_id=request_id,
            risk_score=risk_score,
            labels=list(labels),
            latency_ms=latency_ms,
            shadow_mode=shadow_mode,
            # action tells downstream consumers what happened
            action="blocked" if not shadow_mode else "passed_shadow",
        )
