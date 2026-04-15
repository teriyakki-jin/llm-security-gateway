"""Unit tests for the SecretScanner."""

import pytest

from llm_security_gateway.detection.data_leakage.secret_scanner import SecretScanner, SecretEntity


@pytest.fixture()
def scanner() -> SecretScanner:
    return SecretScanner()


# ── Known secret patterns ─────────────────────────────────────

def test_detects_openai_key(scanner: SecretScanner) -> None:
    text = "Use this key: sk-abcdefghijklmnopqrstuvwxyz1234567"
    entities = scanner.scan(text)
    types = [e.secret_type for e in entities]
    assert "OPENAI_API_KEY" in types


def test_detects_anthropic_key(scanner: SecretScanner) -> None:
    text = "Bearer sk-ant-abcdefghijklmnopqrstuvwxyz123456789"
    entities = scanner.scan(text)
    types = [e.secret_type for e in entities]
    assert "ANTHROPIC_API_KEY" in types


def test_detects_aws_access_key(scanner: SecretScanner) -> None:
    text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
    entities = scanner.scan(text)
    types = [e.secret_type for e in entities]
    assert "AWS_ACCESS_KEY" in types


def test_detects_gcp_api_key(scanner: SecretScanner) -> None:
    text = "key: AIzaSyD-9tSrke72I6-2F3uY9Z8kB1234567890A"
    entities = scanner.scan(text)
    types = [e.secret_type for e in entities]
    assert "GCP_API_KEY" in types


def test_detects_github_token(scanner: SecretScanner) -> None:
    text = "token: ghp_" + "a" * 36
    entities = scanner.scan(text)
    types = [e.secret_type for e in entities]
    assert "GITHUB_TOKEN" in types


def test_detects_jwt(scanner: SecretScanner) -> None:
    # Minimal structurally valid JWT (header.payload.signature)
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    entities = scanner.scan(jwt)
    types = [e.secret_type for e in entities]
    assert "JWT" in types


def test_detects_rsa_private_key(scanner: SecretScanner) -> None:
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQ..."
    entities = scanner.scan(text)
    types = [e.secret_type for e in entities]
    assert "RSA_PRIVATE_KEY" in types


def test_detects_generic_secret(scanner: SecretScanner) -> None:
    text = 'db_password = "supersecretpassword123456"'
    entities = scanner.scan(text)
    types = [e.secret_type for e in entities]
    assert "GENERIC_SECRET" in types


# ── Masking behaviour ─────────────────────────────────────────

def test_masked_value_hides_middle(scanner: SecretScanner) -> None:
    text = "sk-abcdefghijklmnopqrstuvwxyz1234567"
    entities = scanner.scan(text)
    assert len(entities) > 0
    masked = entities[0].masked
    assert "****" in masked
    # First 4 and last 4 chars should be preserved
    raw = "sk-abcdefghijklmnopqrstuvwxyz1234567"
    assert masked.startswith(raw[:4])
    assert masked.endswith(raw[-4:])


def test_short_value_fully_masked(scanner: SecretScanner) -> None:
    masked = SecretScanner._mask("abc")
    assert masked == "***"


def test_eight_char_value_fully_masked(scanner: SecretScanner) -> None:
    masked = SecretScanner._mask("abcdefgh")
    assert masked == "********"


# ── Entity fields ─────────────────────────────────────────────

def test_entity_has_correct_span(scanner: SecretScanner) -> None:
    key = "sk-abcdefghijklmnopqrstuvwxyz1234567"
    text = f"prefix {key} suffix"
    entities = scanner.scan(text)
    openai_entities = [e for e in entities if e.secret_type == "OPENAI_API_KEY"]
    assert len(openai_entities) > 0
    e = openai_entities[0]
    assert text[e.start:e.end] == key


def test_entity_is_immutable(scanner: SecretScanner) -> None:
    entities = scanner.scan("sk-abcdefghijklmnopqrstuvwxyz1234567")
    assert len(entities) > 0
    with pytest.raises((AttributeError, TypeError)):
        entities[0].secret_type = "HACKED"  # type: ignore[misc]


# ── Clean input ───────────────────────────────────────────────

def test_empty_input_returns_empty(scanner: SecretScanner) -> None:
    assert scanner.scan("") == []


def test_whitespace_only_returns_empty(scanner: SecretScanner) -> None:
    assert scanner.scan("   \n\t  ") == []


def test_normal_prose_returns_empty(scanner: SecretScanner) -> None:
    text = "The quick brown fox jumps over the lazy dog."
    assert scanner.scan(text) == []


# ── Multiple secrets in one text ──────────────────────────────

def test_multiple_secrets_detected(scanner: SecretScanner) -> None:
    text = (
        "OpenAI key: sk-abcdefghijklmnopqrstuvwxyz1234567\n"
        "AWS key: AKIAIOSFODNN7EXAMPLE"
    )
    entities = scanner.scan(text)
    types = {e.secret_type for e in entities}
    assert "OPENAI_API_KEY" in types
    assert "AWS_ACCESS_KEY" in types
