"""Abstract base class for LLM provider clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: Usage
    raw_response: dict = field(default_factory=dict)


class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return the response."""

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the upstream API is reachable."""

    @abstractmethod
    async def close(self) -> None:
        """Release underlying HTTP connection pool."""
