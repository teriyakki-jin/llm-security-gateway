"""Unit tests for ResponseFilter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_security_gateway.detection.data_leakage.pii_scanner import PIIEntity
from llm_security_gateway.detection.data_leakage.response_filter import ResponseFilter
from llm_security_gateway.detection.data_leakage.secret_scanner import SecretEntity


@pytest.fixture()
def rf() -> ResponseFilter:
    return ResponseFilter()


# ── Clean text ────────────────────────────────────────────────

def test_clean_text_not_modified(rf: ResponseFilter) -> None:
    text = "The sum of two plus two is four."
    filtered, modified = rf.filter(text)
    assert filtered == text
    assert modified is False


def test_empty_text_not_modified(rf: ResponseFilter) -> None:
    filtered, modified = rf.filter("")
    assert filtered == ""
    assert modified is False


def test_whitespace_not_modified(rf: ResponseFilter) -> None:
    filtered, modified = rf.filter("   \n  ")
    assert modified is False


# ── Secret masking ────────────────────────────────────────────

def test_secret_in_response_is_redacted(rf: ResponseFilter) -> None:
    text = "Here is your key: sk-abcdefghijklmnopqrstuvwxyz1234567"
    filtered, modified = rf.filter(text)
    assert modified is True
    assert "sk-ab" not in filtered
    assert "[REDACTED:OPENAI_API_KEY]" in filtered


def test_aws_key_in_response_is_redacted(rf: ResponseFilter) -> None:
    text = "AWS key: AKIAIOSFODNN7EXAMPLE"
    filtered, modified = rf.filter(text)
    assert modified is True
    assert "[REDACTED:AWS_ACCESS_KEY]" in filtered


# ── Mocked PII + secret ───────────────────────────────────────

def test_pii_entity_is_redacted() -> None:
    mock_pii_scanner = MagicMock()
    mock_secret_scanner = MagicMock()

    pii_entity = PIIEntity(entity_type="EMAIL_ADDRESS", start=5, end=22, score=0.99, text="user@example.com")
    mock_pii_scanner.scan.return_value = [pii_entity]
    mock_secret_scanner.scan.return_value = []

    rf = ResponseFilter.__new__(ResponseFilter)
    rf._pii = mock_pii_scanner
    rf._secrets = mock_secret_scanner

    text = "mail user@example.com here"
    filtered, modified = rf.filter(text)
    assert modified is True
    assert "[REDACTED:EMAIL_ADDRESS]" in filtered
    assert "user@example.com" not in filtered


def test_overlapping_spans_handled() -> None:
    """Spans sorted by start desc so replacements don't corrupt offsets."""
    mock_pii = MagicMock()
    mock_sec = MagicMock()

    # Two non-overlapping spans
    mock_pii.scan.return_value = [
        PIIEntity("EMAIL_ADDRESS", start=0, end=16, score=0.99, text="user@example.com"),
    ]
    mock_sec.scan.return_value = [
        SecretEntity("OPENAI_API_KEY", start=20, end=55, masked="sk-a****567"),
    ]

    rf = ResponseFilter.__new__(ResponseFilter)
    rf._pii = mock_pii
    rf._secrets = mock_sec

    text = "user@example.com    sk-abcdefghijklmnopqrstuvwxyz1234567"
    filtered, modified = rf.filter(text)
    assert modified is True
    assert "[REDACTED:EMAIL_ADDRESS]" in filtered
    assert "[REDACTED:OPENAI_API_KEY]" in filtered


def test_no_pii_no_secret_returns_unchanged() -> None:
    mock_pii = MagicMock()
    mock_sec = MagicMock()
    mock_pii.scan.return_value = []
    mock_sec.scan.return_value = []

    rf = ResponseFilter.__new__(ResponseFilter)
    rf._pii = mock_pii
    rf._secrets = mock_sec

    text = "Hello world"
    filtered, modified = rf.filter(text)
    assert filtered == text
    assert modified is False


# ── request_id passed through ─────────────────────────────────

def test_request_id_accepted(rf: ResponseFilter) -> None:
    text = "sk-abcdefghijklmnopqrstuvwxyz1234567"
    filtered, modified = rf.filter(text, request_id="req-123")
    assert modified is True


# ── _apply_masks static method ────────────────────────────────

def test_apply_masks_replaces_spans() -> None:
    text = "hello world foo"
    pii = [PIIEntity("NAME", start=6, end=11, score=0.9, text="world")]  # "world"
    result = ResponseFilter._apply_masks(text, pii, [])
    assert "[REDACTED:NAME]" in result
    assert "world" not in result


def test_apply_masks_empty_spans_unchanged() -> None:
    text = "nothing to mask"
    result = ResponseFilter._apply_masks(text, [], [])
    assert result == text
