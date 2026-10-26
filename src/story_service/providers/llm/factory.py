"""Create the configured LLM client.

Configuration is deliberately environment based, so deployments can switch
provider or model without changing application code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .google_client import GoogleStudioLLM
from .openai_compatible import OpenAICompatibleLLM

load_dotenv()

_PROVIDERS = {"openrouter", "google", "groq"}
_DEFAULT_MODELS = {
    "openrouter": {"large": "google/gemini-2.5-flash", "small": "google/gemini-2.5-flash-lite", "vision": "google/gemini-2.5-flash"},
    "google": {"large": "gemini-2.5-flash", "small": "gemini-2.5-flash-lite", "vision": "gemini-2.5-flash"},
    "groq": {"large": "llama-3.3-70b-versatile", "small": "llama-3.1-8b-instant", "vision": "meta-llama/llama-4-scout-17b-16e-instruct"},
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    models: dict[str, str]

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider = os.getenv("LLM_ACTIVE_PROVIDER", "openrouter").strip().lower()
        if provider not in _PROVIDERS:
            raise ValueError(f"LLM_ACTIVE_PROVIDER must be one of {', '.join(sorted(_PROVIDERS))}; got {provider!r}")
        prefix = {"openrouter": "OPENROUTER", "google": "GOOGLE", "groq": "GROQ"}[provider]
        api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
        if not api_key:
            raise ValueError(f"Missing {prefix}_API_KEY for active provider {provider!r}. Add it to .env.")
        models = {
            role: os.getenv(f"{prefix}_MODEL_{role.upper()}", default)
            for role, default in _DEFAULT_MODELS[provider].items()
        }
        return cls(provider=provider, api_key=api_key, models=models)


def create_llm(settings: LLMSettings | None = None):
    """Return a client for the active provider from the current environment."""
    settings = settings or LLMSettings.from_env()
    if settings.provider == "google":
        return GoogleStudioLLM(api_key=settings.api_key, models=settings.models)
    base_url = "https://openrouter.ai/api/v1" if settings.provider == "openrouter" else "https://api.groq.com/openai/v1"
    return OpenAICompatibleLLM(api_key=settings.api_key, base_url=base_url, models=settings.models, provider=settings.provider)
