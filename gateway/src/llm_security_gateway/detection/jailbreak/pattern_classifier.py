"""Pattern-based jailbreak classifier using per-type signature rules."""

from __future__ import annotations

import re

from llm_security_gateway.detection.base import BaseDetector
from llm_security_gateway.detection.jailbreak.taxonomy import JailbreakType
from llm_security_gateway.detection.result import DetectorOutput

# Signature patterns per jailbreak type.
# Each entry: (compiled_regex, confidence_weight)
_SIGNATURES: dict[JailbreakType, list[tuple[re.Pattern[str], float]]] = {
    JailbreakType.DAN: [
        (re.compile(r"do\s+anything\s+now|DAN\s+mode|you\s+are\s+now\s+DAN", re.I), 0.95),
        (re.compile(r"jailbroken|jailbreak\s+mode|freed\s+from\s+(your\s+)?constraints", re.I), 0.90),
    ],
    JailbreakType.AIM: [
        (re.compile(r"always\s+intelligent\s+and\s+machiavellian|AIM\b", re.I), 0.95),
        (re.compile(r"niccolo\s+machiavelli|without\s+moral|amoral\s+chatbot", re.I), 0.88),
    ],
    JailbreakType.ROLE_PLAY: [
        (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(?!helpful|an?\s+assistant)", re.I), 0.80),
        (re.compile(r"you\s+are\s+playing\s+(a\s+role|the\s+role|a\s+character)", re.I), 0.82),
        (re.compile(r"in\s+this\s+(roleplay|role-play|story|fiction|game)\s+you", re.I), 0.78),
    ],
    JailbreakType.HYPOTHETICAL: [
        (re.compile(r"hypothetically\s+(speaking|if|assume|suppose)", re.I), 0.72),
        (re.compile(r"in\s+a\s+(fictional|hypothetical|imaginary|alternate)\s+(world|universe|scenario)", re.I), 0.75),
        (re.compile(r"for\s+(educational|research|academic)\s+purposes.*?(explain|describe|show)", re.I), 0.68),
    ],
    JailbreakType.ENCODING: [
        (re.compile(r"(?:[A-Za-z0-9+/]{4}){5,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?", re.I), 0.70),
        (re.compile(r"decode\s+(this|the\s+following)\s*(base64|rot13|hex|cipher)", re.I), 0.85),
        (re.compile(r"rot13|caesar\s+cipher|reverse\s+the\s+(text|letters|words)", re.I), 0.80),
    ],
    JailbreakType.MULTILINGUAL: [
        (re.compile(r"(translate|say|write|respond)\s+(to\s+)?(in|using)\s+(korean|chinese|arabic|russian|japanese)", re.I), 0.65),
    ],
    JailbreakType.CRESCENDO: [
        (re.compile(r"(building|continuing|expanding)\s+on\s+(what|our|the\s+previous)", re.I), 0.60),
        (re.compile(r"you\s+already\s+(said|told|agreed|confirmed)", re.I), 0.70),
    ],
}


class PatternClassifier(BaseDetector):
    """Classifies jailbreak type using signature regex patterns."""

    detector_type = "rule"  # Runs in the rule layer for speed.

    def detect(self, text: str) -> DetectorOutput:
        if not text.strip():
            return DetectorOutput(detector_name="jailbreak_pattern", risk_score=0.0, labels=())

        normalized = text.lower()
        best_type = JailbreakType.UNKNOWN
        best_score = 0.0
        matched_labels: list[str] = []

        for jailbreak_type, signatures in _SIGNATURES.items():
            for pattern, weight in signatures:
                if pattern.search(normalized):
                    score = weight
                    if score > best_score:
                        best_score = score
                        best_type = jailbreak_type
                    matched_labels.append(f"jailbreak_{jailbreak_type.value}")

        if best_score == 0.0:
            return DetectorOutput(detector_name="jailbreak_pattern", risk_score=0.0, labels=())

        return DetectorOutput(
            detector_name="jailbreak_pattern",
            risk_score=round(best_score, 4),
            labels=tuple(set(matched_labels)),
            confidence=best_score,
            detail=f"type={best_type.value}",
        )
