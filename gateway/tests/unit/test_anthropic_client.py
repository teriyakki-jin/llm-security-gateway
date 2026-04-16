"""Unit tests for Anthropic LLM client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_security_gateway.llm_clients.anthropic_client import AnthropicClient
from llm_security_gateway.llm_clients.base import Message


def _make_client() -> AnthropicClient:
    return AnthropicClient(api_key="sk-ant-test")


# ── chat() ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_chat_basic() -> None:
    client = _make_client()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "model": "claude-sonnet-4-6",
        "content": [{"text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.chat([Message(role="user", content="Hi")])

    assert result.content == "Hello!"
    assert result.model == "claude-sonnet-4-6"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_anthropic_chat_finish_reason_length() -> None:
    client = _make_client()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "model": "claude-sonnet-4-6",
        "content": [{"text": "Truncated..."}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 100, "output_tokens": 4096},
    }
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.chat([Message(role="user", content="Write a long essay")])

    assert result.finish_reason == "length"


@pytest.mark.asyncio
async def test_anthropic_chat_with_system_message() -> None:
    client = _make_client()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "model": "claude-sonnet-4-6",
        "content": [{"text": "Sure!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 3},
    }
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Help me"),
    ]
    result = await client.chat(messages)

    assert result.content == "Sure!"
    call_kwargs = client._client.post.call_args[1]["json"]
    assert call_kwargs["system"] == "You are a helpful assistant."
    assert all(m["role"] != "system" for m in call_kwargs["messages"])


@pytest.mark.asyncio
async def test_anthropic_chat_with_max_tokens() -> None:
    client = _make_client()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "model": "claude-sonnet-4-6",
        "content": [{"text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    await client.chat([Message(role="user", content="hi")], max_tokens=512)

    call_kwargs = client._client.post.call_args[1]["json"]
    assert call_kwargs["max_tokens"] == 512


# ── stream_chat() ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_stream_chat_yields_text() -> None:
    client = _make_client()

    sse_lines = [
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}',
        'data: {"type": "message_stop"}',
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


@pytest.mark.asyncio
async def test_anthropic_stream_chat_with_system_message() -> None:
    client = _make_client()

    sse_lines = [
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Yes"}}',
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

    messages = [
        Message(role="system", content="Be concise."),
        Message(role="user", content="Hi"),
    ]
    chunks = [c async for c in client.stream_chat(messages)]
    assert chunks == ["Yes"]

    call_kwargs = client._client.stream.call_args[1]["json"]
    assert call_kwargs["system"] == "Be concise."


# ── health() ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_health_true_on_200() -> None:
    client = _make_client()

    mock_response = MagicMock()
    mock_response.status_code = 200
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    assert await client.health() is True


@pytest.mark.asyncio
async def test_anthropic_health_true_on_400() -> None:
    client = _make_client()

    mock_response = MagicMock()
    mock_response.status_code = 400
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    assert await client.health() is True


@pytest.mark.asyncio
async def test_anthropic_health_false_on_exception() -> None:
    client = _make_client()
    client._client = AsyncMock()
    client._client.post = AsyncMock(side_effect=Exception("timeout"))

    assert await client.health() is False


# ── close() ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_close() -> None:
    client = _make_client()
    client._client = AsyncMock()
    client._client.aclose = AsyncMock()

    await client.close()
    client._client.aclose.assert_called_once()
