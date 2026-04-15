"""Detects secrets (API keys, tokens, private keys) in LLM responses."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretEntity:
    secret_type: str
    start: int
    end: int
    # Never store the raw value — only a masked version for logging.
    masked: str


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OPENAI_API_KEY",    re.compile(r"sk-[a-zA-Z0-9]{20,50}")),
    ("ANTHROPIC_API_KEY", re.compile(r"sk-ant-[a-zA-Z0-9\-_]{20,60}")),
    ("AWS_ACCESS_KEY",    re.compile(r"AKIA[0-9A-Z]{16}")),
    ("AWS_SECRET_KEY",    re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]")),
    ("GCP_API_KEY",       re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("GITHUB_TOKEN",      re.compile(r"ghp_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z_]{82}")),
    ("JWT",               re.compile(r"eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_.+/]+")),
    ("RSA_PRIVATE_KEY",   re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----")),
    ("GENERIC_SECRET",    re.compile(r"(?i)(secret|password|passwd|token|api_key)\s*[:=]\s*['\"]?[a-zA-Z0-9/+\-_]{16,}['\"]?")),
]


class SecretScanner:
    """Scans text for exposed secrets using regex patterns."""

    def scan(self, text: str) -> list[SecretEntity]:
        if not text.strip():
            return []

        entities: list[SecretEntity] = []
        for secret_type, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                raw = m.group()
                entities.append(SecretEntity(
                    secret_type=secret_type,
                    start=m.start(),
                    end=m.end(),
                    masked=self._mask(raw),
                ))
        return entities

    @staticmethod
    def _mask(value: str) -> str:
        """Show first 4 and last 4 chars, mask the rest."""
        if len(value) <= 8:
            return "*" * len(value)
        return value[:4] + "*" * (len(value) - 8) + value[-4:]
