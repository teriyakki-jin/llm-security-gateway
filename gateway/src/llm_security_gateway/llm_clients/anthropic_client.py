"""Anthropic (Claude) API client."""

import json
from collections.abc import AsyncIterator

import httpx

from llm_security_gateway.llm_clients.base import BaseLLMClient, LLMResponse, Message, Usage


class AnthropicClient(BaseLLMClient):
    BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-sonnet-4-6"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, timeout: int = 60) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self.API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        self._default_model = self.DEFAULT_MODEL

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        # Anthropic separates system messages from the messages array.
        system_parts = [m.content for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        payload: dict = {
            "model": model or self._default_model,
            "messages": [{"role": m.role, "content": m.content} for m in non_system],
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        response = await self._client.post("/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        content_block = data["content"][0]
        usage_data = data.get("usage", {})
        # Map Anthropic stop_reason → OpenAI-compatible finish_reason.
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = "length" if stop_reason == "max_tokens" else "stop"

        return LLMResponse(
            content=content_block["text"],
            model=data["model"],
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            ),
            raw_response=data,
        )

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        system_parts = [m.content for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        payload: dict = {
            "model": model or self._default_model,
            "messages": [{"role": m.role, "content": m.content} for m in non_system],
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
            "stream": True,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        async with self._client.stream("POST", "/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                event = json.loads(raw)
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

    async def health(self) -> bool:
        try:
            # Anthropic has no dedicated health endpoint; send a minimal request.
            response = await self._client.post(
                "/messages",
                json={
                    "model": self._default_model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=5.0,
            )
            return response.status_code in {200, 400}  # 400 = bad request, but API is reachable
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
