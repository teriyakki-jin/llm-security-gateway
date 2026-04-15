"""PII detection in LLM responses using Microsoft Presidio."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PIIEntity:
    entity_type: str
    start: int
    end: int
    score: float
    text: str


class PIIScanner:
    """
    Detects PII in text using Presidio AnalyzerEngine.

    Falls back to a minimal regex-based scanner if Presidio is unavailable.
    Presidio supports: EMAIL, PHONE, CREDIT_CARD, KR_RRNO (주민번호), LOCATION, etc.
    """

    def __init__(self) -> None:
        self._engine = None
        self._available = False
        self._try_load()

    def scan(self, text: str) -> list[PIIEntity]:
        if not text.strip():
            return []

        if self._available:
            return self._scan_presidio(text)
        return self._scan_regex_fallback(text)

    def _scan_presidio(self, text: str) -> list[PIIEntity]:
        try:
            results = self._engine.analyze(  # type: ignore[union-attr]
                text=text,
                language="en",
                entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                    "IBAN_CODE", "IP_ADDRESS", "LOCATION",
                    "PERSON", "US_SSN",
                ],
            )
            return [
                PIIEntity(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                    text=text[r.start:r.end],
                )
                for r in results
                if r.score >= 0.7
            ]
        except Exception as exc:
            logger.warning("presidio scan failed: %s", exc)
            return self._scan_regex_fallback(text)

    @staticmethod
    def _scan_regex_fallback(text: str) -> list[PIIEntity]:
        """Minimal regex-based PII detection for when Presidio is unavailable."""
        entities: list[PIIEntity] = []
        patterns = [
            ("EMAIL_ADDRESS", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
            ("PHONE_NUMBER", re.compile(r"(\+?82|0)\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}")),
            ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
            ("KR_RRNO", re.compile(r"\d{6}[-\s]?[1-4]\d{6}")),  # 주민등록번호
        ]
        for entity_type, pattern in patterns:
            for m in pattern.finditer(text):
                entities.append(PIIEntity(
                    entity_type=entity_type,
                    start=m.start(),
                    end=m.end(),
                    score=0.85,
                    text=m.group(),
                ))
        return entities

    def _try_load(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            self._engine = AnalyzerEngine()
            self._available = True
        except ImportError:
            logger.warning("presidio-analyzer not installed — using regex PII fallback")
