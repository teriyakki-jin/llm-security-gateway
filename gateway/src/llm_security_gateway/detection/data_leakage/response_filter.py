"""Filters PII and secrets from LLM responses before returning to clients."""

from __future__ import annotations

import structlog

from llm_security_gateway.detection.data_leakage.pii_scanner import PIIEntity, PIIScanner
from llm_security_gateway.detection.data_leakage.secret_scanner import SecretEntity, SecretScanner

logger = structlog.get_logger(__name__)


class ResponseFilter:
    """
    Scans LLM response text and masks detected PII and secrets.

    Original text is never logged — only entity types and masked values.
    """

    def __init__(self) -> None:
        self._pii = PIIScanner()
        self._secrets = SecretScanner()

    def filter(
        self,
        text: str,
        *,
        request_id: str = "",
    ) -> tuple[str, bool]:
        """
        Scan and mask sensitive data in text.

        Returns:
            (filtered_text, was_modified)
        """
        if not text.strip():
            return text, False

        pii_entities = self._pii.scan(text)
        secret_entities = self._secrets.scan(text)

        if not pii_entities and not secret_entities:
            return text, False

        # Log what was found (masked values only).
        if pii_entities:
            logger.warning(
                "pii_detected_in_response",
                request_id=request_id,
                types=[e.entity_type for e in pii_entities],
                count=len(pii_entities),
            )
        if secret_entities:
            logger.warning(
                "secret_detected_in_response",
                request_id=request_id,
                types=[e.secret_type for e in secret_entities],
                masked=[e.masked for e in secret_entities],
                count=len(secret_entities),
            )

        filtered = self._apply_masks(text, pii_entities, secret_entities)
        return filtered, True

    @staticmethod
    def _apply_masks(
        text: str,
        pii: list[PIIEntity],
        secrets: list[SecretEntity],
    ) -> str:
        """Replace detected spans with [REDACTED:TYPE] markers."""
        # Collect all spans with their labels, sort by start descending
        # so replacements don't shift offsets.
        spans: list[tuple[int, int, str]] = []
        for e in pii:
            spans.append((e.start, e.end, e.entity_type))
        for e in secrets:
            spans.append((e.start, e.end, e.secret_type))

        spans.sort(key=lambda x: x[0], reverse=True)

        result = text
        for start, end, label in spans:
            result = result[:start] + f"[REDACTED:{label}]" + result[end:]

        return result
