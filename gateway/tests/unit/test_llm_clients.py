"""Unit tests for LLM client base, factory, and OpenAI client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic import SecretStr

from llm_security_gateway.config import GatewaySettings
from llm_security_gateway.llm_clients.base import LLMResponse, Message, Usage
from llm_security_gateway.llm_clients.factory import create_client


# ── Dataclass tests ───────────────────────────────────────────

def test_message_is_immutable() -> None:
    msg = Message(role="user", content="hello")
    with pytest.raises((AttributeError, TypeError)):
        msg.role = "assistant"  # type: ignore[misc]


def test_usage_total_tokens() -> None:
    u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert u.total_tokens == 15


def test_llm_response_is_immutable() -> None:
    r = LLMResponse(content="hi", model="gpt-4o", usage=Usage(1, 1, 2))
    with pytest.raises((AttributeError, TypeError)):
        r.content = "bye"  # type: ignore[misc]


def test_llm_response_default_raw_response() -> None:
    r = LLMResponse(content="hi", model="gpt-4o", usage=Usage(1, 1, 2))
    assert r.raw_response == {}


# ── Factory tests ─────────────────────────────────────────────

def _settings(**kwargs) -> GatewaySettings:
    defaults = dict(
        openai_api_key=SecretStr("sk-test"),
        anthropic_api_key=SecretStr("sk-ant-test"),
        default_provider="openai",
        database_url="postgresql+asyncpg://x:x@localhost/x",
        redis_url="redis://localhost",
    )
    defaults.update(kwargs)
    return GatewaySettings.model_construct(**defaults)


def test_factory_creates_openai_client() -> None:
    from llm_security_gateway.llm_clients.openai_client import OpenAIClient
    client = create_client("openai", _settings())
    assert isinstance(client, OpenAIClient)


def test_factory_creates_anthropic_client() -> None:
    from llm_security_gateway.llm_clients.anthropic_client import AnthropicClient
    client = create_client("anthropic", _settings())
    assert isinstance(client, AnthropicClient)


def test_factory_raises_on_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_client("unknown", _settings())


def test_factory_raises_when_openai_key_missing() -> None:
    s = _settings(openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_client("openai", s)


def test_factory_raises_when_anthropic_key_missing() -> None:
    s = _settings(anthropic_api_key=None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        create_client("anthropic", s)


# ── OpenAI client ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openai_client_chat() -> None:
    from llm_security_gateway.llm_clients.openai_client import OpenAIClient

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "model": "gpt-4o",
        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    client = OpenAIClient(api_key="sk-test")
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.chat([Message(role="user", content="Hi")])
    assert result.content == "Hello!"
    assert result.model == "gpt-4o"
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_openai_client_health_true() -> None:
    from llm_security_gateway.llm_clients.openai_client import OpenAIClient

    mock_response = MagicMock()
    mock_response.status_code = 200

    client = OpenAIClient(api_key="sk-test")
    client._client = AsyncMock()
    client._client.get = AsyncMock(return_value=mock_response)

    assert await client.health() is True


@pytest.mark.asyncio
async def test_openai_client_health_false_on_error() -> None:
    from llm_security_gateway.llm_clients.openai_client import OpenAIClient

    client = OpenAIClient(api_key="sk-test")
    client._client = AsyncMock()
    client._client.get = AsyncMock(side_effect=Exception("connection error"))

    assert await client.health() is False


@pytest.mark.asyncio
async def test_openai_client_close() -> None:
    from llm_security_gateway.llm_clients.openai_client import OpenAIClient

    client = OpenAIClient(api_key="sk-test")
    client._client = AsyncMock()
    client._client.aclose = AsyncMock()

    await client.close()
    client._client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_openai_client_chat_with_max_tokens() -> None:
    from llm_security_gateway.llm_clients.openai_client import OpenAIClient

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "model": "gpt-4o",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    client = OpenAIClient(api_key="sk-test")
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.chat(
        [Message(role="user", content="Hi")],
        model="gpt-4o-mini",
        max_tokens=100,
    )
    assert result.content == "ok"
    call_kwargs = client._client.post.call_args
    assert call_kwargs[1]["json"]["max_tokens"] == 100


@pytest.mark.asyncio
async def test_openai_client_chat_finish_reason() -> None:
    from llm_security_gateway.llm_clients.openai_client import OpenAIClient

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "model": "gpt-4o",
        "choices": [{"message": {"content": "..."}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
    }

    client = OpenAIClient(api_key="sk-test")
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.chat([Message(role="user", content="long prompt")])
    assert result.finish_reason == "length"


@pytest.mark.asyncio
async def test_openai_client_stream_chat_yields_text() -> None:
    from llm_security_gateway.llm_clients.openai_client import OpenAIClient
    import json as _json

    client = OpenAIClient(api_key="sk-test")

    sse_lines = [
        'data: ' + _json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        'data: ' + _json.dumps({"choices": [{"delta": {"content": " world"}}]}),
        'data: ' + _json.dumps({"choices": [{"delta": {}}]}),  # no content
        'data: [DONE]',
        'not-data-line',  # should be skipped
    ]

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    async def _aiter_lines():
        for line in sse_lines:
            yield line

    mock_response.aiter_lines = _aiter_lines

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    client._client = MagicMock()
    client._client.stream = MagicMock(return_value=mock_stream_ctx)

    chunks = []
    async for chunk in client.stream_chat([Message(role="user", content="Hi")]):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]
