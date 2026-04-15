"""LLM client factory."""

from llm_security_gateway.config import GatewaySettings
from llm_security_gateway.llm_clients.anthropic_client import AnthropicClient
from llm_security_gateway.llm_clients.base import BaseLLMClient
from llm_security_gateway.llm_clients.openai_client import OpenAIClient


def create_client(provider: str, settings: GatewaySettings) -> BaseLLMClient:
    """Create and return an LLM client for the given provider."""
    match provider:
        case "openai":
            if settings.openai_api_key is None:
                raise ValueError("OPENAI_API_KEY is not configured")
            return OpenAIClient(
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.llm_timeout_sec,
            )
        case "anthropic":
            if settings.anthropic_api_key is None:
                raise ValueError("ANTHROPIC_API_KEY is not configured")
            return AnthropicClient(
                api_key=settings.anthropic_api_key.get_secret_value(),
                timeout=settings.llm_timeout_sec,
            )
        case _:
            raise ValueError(f"Unknown LLM provider: {provider!r}. Choose 'openai' or 'anthropic'.")
