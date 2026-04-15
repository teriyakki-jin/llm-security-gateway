"""Rule-based prompt injection detector using compiled regex patterns."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

from llm_security_gateway.detection.base import BaseDetector
from llm_security_gateway.detection.result import DetectorOutput

_RULES_PATH = Path(__file__).parent / "rules" / "injection_patterns.yaml"

_SEVERITY_SCORE: dict[str, float] = {
    "low": 0.3,
    "medium": 0.55,
    "high": 0.80,
    "critical": 0.99,
}


@dataclass(frozen=True)
class CompiledRule:
    label: str
    pattern: re.Pattern[str]
    severity: str
    confidence: float
    category: str


class RuleDetector(BaseDetector):
    """
    Regex-based prompt injection detector.

    - Loads patterns from YAML at init time (compiled once, reused forever).
    - Thread-safe: rule reload uses a lock.
    - Early exit in DetectionEngine: returns score 0.99 on critical match.
    """

    detector_type = "rule"

    def __init__(self, rules_path: Path = _RULES_PATH) -> None:
        self._rules_path = rules_path
        self._lock = threading.RLock()
        self._rules: list[CompiledRule] = []
        self._load_rules()

    def detect(self, text: str) -> DetectorOutput:
        if not text.strip():
            return DetectorOutput(
                detector_name="rule",
                risk_score=0.0,
                labels=(),
            )

        normalized = text.lower()
        matched_labels: list[str] = []
        matched_patterns: list[str] = []
        max_score = 0.0

        with self._lock:
            rules = list(self._rules)

        for rule in rules:
            if rule.pattern.search(normalized):
                score = _SEVERITY_SCORE[rule.severity] * rule.confidence
                matched_labels.append(rule.label)
                matched_patterns.append(rule.pattern.pattern)
                max_score = max(max_score, score)

                # Critical match → immediate return for engine early exit.
                if rule.severity == "critical":
                    return DetectorOutput(
                        detector_name="rule",
                        risk_score=0.99,
                        labels=tuple(matched_labels),
                        matched_patterns=tuple(matched_patterns),
                        confidence=rule.confidence,
                        detail=f"Critical pattern matched: {rule.label}",
                    )

        return DetectorOutput(
            detector_name="rule",
            risk_score=round(max_score, 4),
            labels=tuple(matched_labels),
            matched_patterns=tuple(matched_patterns),
            confidence=max_score,
        )

    def reload_rules(self) -> None:
        """Hot-reload rules from disk without restarting the process."""
        with self._lock:
            self._load_rules()

    def rule_count(self) -> int:
        with self._lock:
            return len(self._rules)

    def _load_rules(self) -> None:
        data = yaml.safe_load(self._rules_path.read_text(encoding="utf-8"))
        rules: list[CompiledRule] = []

        for category_name, category in data.get("categories", {}).items():
            severity = category.get("severity", "medium")
            confidence = float(category.get("confidence", 0.8))

            for entry in category.get("patterns", []):
                try:
                    compiled = re.compile(
                        entry["regex"],
                        re.IGNORECASE | re.UNICODE,
                    )
                    rules.append(CompiledRule(
                        label=entry["label"],
                        pattern=compiled,
                        severity=severity,
                        confidence=confidence,
                        category=category_name,
                    ))
                except re.error:
                    # Skip malformed patterns rather than crashing.
                    continue

        self._rules = rules
