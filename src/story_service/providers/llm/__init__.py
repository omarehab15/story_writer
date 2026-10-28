"""LLM providers. `build_llm()` returns the client for the provider chosen by LLM_PROVIDER."""
from story_service.core.config import Settings, get_settings
from story_service.providers.llm.base import LLMClient, ProviderError
from story_service.providers.llm.openai_compat import PROVIDERS, OpenAICompatLLM


def build_llm(settings: Settings | None = None, **kwargs) -> OpenAICompatLLM:
    return OpenAICompatLLM(settings or get_settings(), **kwargs)


__all__ = ["PROVIDERS", "LLMClient", "OpenAICompatLLM", "ProviderError", "build_llm"]
