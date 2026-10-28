"""One OpenAI-compatible chat-completions client for OpenRouter, Groq and Google AI Studio.

All three expose `/chat/completions`, so a provider is just data (`ProviderSpec`). Switching the
active provider is a one-line change in `.env` (`LLM_PROVIDER=...`).
"""
from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from story_service.core.config import Settings
from story_service.providers.llm.base import ProviderError

RETRY_DELAYS = (2, 4, 8, 16, 32)  # spec 12.2: exponential, +/-20% jitter, at most 5 retries
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    default_base_url: str
    key_vars: tuple[str, ...]  # env var names, for error messages
    default_models: dict[str, str]
    max_tokens_param: str = "max_tokens"
    reasoning_style: str = "openai"  # "openai": reasoning_effort=..., "openrouter": reasoning={"effort":...}


PROVIDERS: dict[str, ProviderSpec] = {
    "google": ProviderSpec(
        name="google",
        label="Google AI Studio (Gemini API)",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        default_models={"large": "gemini-3.5-flash", "small": "gemini-3.5-flash-lite", "vision": "gemini-3.5-flash"},
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        label="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        key_vars=("OPENROUTER_API_KEY",),
        default_models={
            "large": "google/gemini-3.5-flash",
            "small": "google/gemini-3.5-flash-lite",
            "vision": "google/gemini-3.5-flash",
        },
        reasoning_style="openrouter",
    ),
    "groq": ProviderSpec(
        name="groq",
        label="Groq",
        default_base_url="https://api.groq.com/openai/v1",
        key_vars=("GROQ_API_KEY",),
        default_models={"large": "openai/gpt-oss-120b", "small": "openai/gpt-oss-20b", "vision": "openai/gpt-oss-120b"},
        max_tokens_param="max_completion_tokens",
    ),
}


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in `text` (tolerates ```json fences and <think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("top-level JSON is not an object", text, 0)
    return value


class OpenAICompatLLM:
    """Implements `LLMClient` for any provider in `PROVIDERS`."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        self.settings = settings
        self.spec = PROVIDERS[settings.llm_provider]
        self.base_url = (settings.provider_value(self.spec.name, "base_url") or self.spec.default_base_url).rstrip("/")
        self.api_key = settings.provider_value(self.spec.name, "api_key")
        self._sleep, self._rng = sleep, rng or random.Random()
        self._client = httpx.Client(timeout=httpx.Timeout(settings.llm_timeout_sec), transport=transport)
        self._json_mode_off: set[str] = set()  # models that rejected response_format
        self.usage: list[dict[str, Any]] = []

    # -- configuration -----------------------------------------------------
    def model_for(self, role: str) -> str:
        s = self.settings
        return (
            getattr(s, f"llm_model_{role}")
            or s.provider_value(self.spec.name, f"model_{role}")
            or self.spec.default_models[role]
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.spec.name,
            "label": self.spec.label,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key),
            "models": {r: self.model_for(r) for r in ("large", "small", "vision")},
            "json_mode": self.settings.llm_json_mode,
            "reasoning_effort": self.settings.llm_reasoning_effort,
        }

    def _require_key(self) -> str:
        if not self.api_key:
            raise ProviderError(
                f"No API key for provider '{self.spec.name}'. Set {' or '.join(self.spec.key_vars)} in .env "
                f"(or change LLM_PROVIDER)."
            )
        return self.api_key

    # -- request -------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self._require_key()}", "Content-Type": "application/json"}
        if self.spec.name == "openrouter":
            if self.settings.openrouter_site_url:
                h["HTTP-Referer"] = self.settings.openrouter_site_url
            if self.settings.openrouter_app_name:
                h["X-Title"] = self.settings.openrouter_app_name
        return h

    def _payload(self, model: str, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool):
        p: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                             self.spec.max_tokens_param: max_tokens}
        if json_mode:
            p["response_format"] = {"type": "json_object"}
        effort = self.settings.llm_reasoning_effort
        if effort:
            if self.spec.reasoning_style == "openrouter":
                p["reasoning"] = {"effort": effort}
            else:
                p["reasoning_effort"] = effort
        return p

    def _delay(self, attempt: int, resp: httpx.Response | None) -> float:
        base = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
        if resp is not None:
            try:
                base = max(base, min(float(resp.headers.get("retry-after", 0)), 60))
            except ValueError:
                pass
        return base * (1 + self._rng.uniform(-0.2, 0.2))

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """POST with transient retries. Returns the final response (possibly an error status)."""
        url = f"{self.base_url}/chat/completions"
        headers = self._headers()
        last_exc: Exception | None = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            resp: httpx.Response | None = None
            try:
                resp = self._client.post(url, headers=headers, json=payload)
                if resp.status_code not in TRANSIENT_STATUS:
                    return resp
                last_exc = None
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
            if attempt == len(RETRY_DELAYS):
                break
            self._sleep(self._delay(attempt, resp))
        if last_exc is not None:
            raise ProviderError(f"{self.spec.name}: network error after retries: {last_exc!r}") from last_exc
        raise ProviderError(f"{self.spec.name}: still failing after retries (HTTP {resp.status_code}): {resp.text[:300]}")

    def _chat(self, messages: list[dict], role: str, temperature: float, max_tokens: int) -> str:
        model = self.model_for(role)
        json_mode = self.settings.llm_json_mode == "json_object" and model not in self._json_mode_off
        started = time.monotonic()
        resp = self._post(self._payload(model, messages, temperature, max_tokens, json_mode))
        if resp.status_code == 400 and json_mode and re.search(r"response_format|json_object|json mode|json_schema", resp.text, re.IGNORECASE):
            self._json_mode_off.add(model)  # model/provider does not support JSON mode: ask in the prompt only
            resp = self._post(self._payload(model, messages, temperature, max_tokens, False))
        if resp.status_code >= 400:
            hint = {401: "check the API key", 403: "key not allowed for this model", 404: "unknown model id or base URL"}.get(resp.status_code, "")
            raise ProviderError(f"{self.spec.name} HTTP {resp.status_code} for model '{model}'"
                                f"{' (' + hint + ')' if hint else ''}: {resp.text[:400]}")
        try:
            data = resp.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError) as exc:
            raise ProviderError(f"{self.spec.name}: unexpected response shape: {resp.text[:300]}") from exc
        usage = data.get("usage") or {}
        self.usage.append({"provider": self.spec.name, "model": model,
                           "tokens_in": usage.get("prompt_tokens", 0), "tokens_out": usage.get("completion_tokens", 0),
                           "latency_ms": int((time.monotonic() - started) * 1000)})
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if choice.get("finish_reason") == "length" or not content:
            raise ProviderError(
                f"{self.spec.name}: output truncated or empty for '{model}' (finish_reason={choice.get('finish_reason')}). "
                f"Reasoning models spend output tokens on thinking: set LLM_REASONING_EFFORT=low or raise max_tokens."
            )
        return content

    # -- LLMClient -------------------------------------------------------------
    def complete_json(self, system, user, schema, model_role="large", temperature=0.2, max_tokens=4096, images=None):
        if images:
            raise ProviderError("image input is not supported by this adapter yet (needed from module 6 on)")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._chat(messages, model_role, temperature, max_tokens)
        try:
            return extract_json(text)
        except json.JSONDecodeError as exc:  # spec 12.2: one repair attempt
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That was not a valid JSON object ({exc.msg}). Reply with ONLY the corrected JSON object."},
            ]
            try:
                return extract_json(self._chat(messages, model_role, 0.0, max_tokens))
            except json.JSONDecodeError as exc2:
                raise ProviderError(f"{self.spec.name}: model did not return valid JSON twice: {exc2.msg}") from exc2
