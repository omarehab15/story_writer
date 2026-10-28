"""Unified settings. One `.env` drives everything; real environment variables win over the file."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openrouter", "groq", "google"]
PROVIDER_NAMES: tuple[str, ...] = ("openrouter", "groq", "google")
ModelRole = Literal["large", "small", "vision"]

REPO_ROOT = Path(__file__).resolve().parents[3]


def find_env_file() -> Path | None:
    """`ENV_FILE` if set, else ./.env, else <repo>/.env."""
    explicit = os.getenv("ENV_FILE")
    if explicit:
        return Path(explicit).expanduser()
    for candidate in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        if candidate.is_file():
            return candidate
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_env_file(), env_file_encoding="utf-8", extra="ignore", case_sensitive=False,
        populate_by_name=True
    )

    # ---- active provider -------------------------------------------------
    llm_provider: ProviderName = "google"

    # ---- keys ------------------------------------------------------------
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    google_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )

    # ---- models: generic override, then per provider ---------------------
    llm_model_large: str | None = None
    llm_model_small: str | None = None
    llm_model_vision: str | None = None

    google_model_large: str | None = None
    google_model_small: str | None = None
    google_model_vision: str | None = None
    openrouter_model_large: str | None = None
    openrouter_model_small: str | None = None
    openrouter_model_vision: str | None = None
    groq_model_large: str | None = None
    groq_model_small: str | None = None
    groq_model_vision: str | None = None

    # ---- tuning -----------------------------------------------------------
    llm_json_mode: Literal["json_object", "none"] = "json_object"
    llm_reasoning_effort: Literal["low", "medium", "high"] | None = None
    llm_timeout_sec: float = 120.0
    openrouter_base_url: str | None = None
    groq_base_url: str | None = None
    google_base_url: str | None = None
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None

    # ---- pipeline ---------------------------------------------------------
    max_stage_attempts: int = 3
    llm_max_concurrency: int = 8
    max_tokens_per_story: int = 600_000
    log_level: str = "INFO"
    catalog_path: str | None = None

    # ---- notebook / dev runner (all optional; blank = built-in default) ----
    pipeline_mode: Literal["fake", "real"] | None = None          # default "fake"
    s5_age_group: Literal["6-8", "9-11", "12-14", "15-18"] | None = None  # default "9-11"
    s5_script_path: str | None = None                             # default: golden script.json
    s5_brief_path: str | None = None                              # default: golden brief.json
    runs_dir: str | None = None                                   # default: <repo>/runs

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_none(cls, v):
        # `KEY=` in a .env file means "not set". `KEY=   # comment` also means "not set":
        # some dotenv parsers return the comment text as the value.
        if isinstance(v, str) and (not v.strip() or v.strip().startswith("#")):
            return None
        return v.strip() if isinstance(v, str) else v

    @field_validator("pipeline_mode", mode="before")
    @classmethod
    def _mode_lower(cls, v):
        return v.lower() if isinstance(v, str) else v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _provider_lower(cls, v):
        return v.lower() if isinstance(v, str) else v

    def provider_value(self, provider: str, field: str):
        return getattr(self, f"{provider}_{field}")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
