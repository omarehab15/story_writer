import json

import httpx
import pytest

from story_service.core.config import Settings
from story_service.providers.llm import PROVIDERS, OpenAICompatLLM, ProviderError, build_llm
from story_service.providers.llm.openai_compat import extract_json


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def ok_body(content='{"ok": true}', finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def make(s: Settings, handler, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return OpenAICompatLLM(s, transport=httpx.MockTransport(handler), sleep=sleeps.append), sleeps


def call(llm, **kw):
    args = {"system": "sys", "user": "usr", "schema": None, "model_role": "large", "temperature": 0.3, "max_tokens": 500}
    return llm.complete_json(**{**args, **kw})


# ----------------------------- .env and active provider ------------------------------- #
def test_env_file_selects_provider_and_keys(tmp_path):
    f = tmp_path / ".env"
    f.write_text("LLM_PROVIDER=groq\nGROQ_API_KEY=gsk_x\nGOOGLE_API_KEY=   # inline comment, blank\nLLM_REASONING_EFFORT=\n")
    s = Settings(_env_file=f)
    assert s.llm_provider == "groq" and s.groq_api_key == "gsk_x"
    assert s.google_api_key is None and s.llm_reasoning_effort is None  # blank means not set


def test_real_env_vars_win_over_the_file(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("LLM_PROVIDER=groq\n")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert Settings(_env_file=f).llm_provider == "openrouter"


def test_provider_name_is_case_insensitive_and_validated():
    assert settings(llm_provider="OpenRouter").llm_provider == "openrouter"
    with pytest.raises(ValueError):
        settings(llm_provider="openai")


def test_gemini_api_key_alias(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert Settings(_env_file=None).google_api_key == "g-key"


@pytest.mark.parametrize("provider,url,default_large,default_small", [
    ("google", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash", "gemini-3.5-flash-lite"),
    ("openrouter", "https://openrouter.ai/api/v1", "google/gemini-3.5-flash", "google/gemini-3.5-flash-lite"),
    ("groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "openai/gpt-oss-20b"),
])
def test_each_provider_defaults(provider, url, default_large, default_small):
    llm = build_llm(settings(llm_provider=provider))
    d = llm.describe()
    assert d["base_url"] == url and d["models"]["large"] == default_large and d["models"]["small"] == default_small
    assert set(PROVIDERS) == {"google", "openrouter", "groq"}


def test_model_precedence_generic_over_provider_over_default():
    assert build_llm(settings(llm_provider="groq")).model_for("large") == "openai/gpt-oss-120b"
    assert build_llm(settings(llm_provider="groq", groq_model_large="llama-x")).model_for("large") == "llama-x"
    assert build_llm(settings(llm_provider="groq", groq_model_large="llama-x", llm_model_large="global")).model_for("large") == "global"


def test_base_url_override():
    llm = build_llm(settings(llm_provider="groq", groq_base_url="http://localhost:9/v1/"))
    assert llm.base_url == "http://localhost:9/v1"


# ----------------------------- request shape per provider ----------------------------- #
@pytest.mark.parametrize("provider,key_kw,url,token_param", [
    ("google", {"GOOGLE_API_KEY": "gk"}, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "max_tokens"),
    ("openrouter", {"openrouter_api_key": "ork"}, "https://openrouter.ai/api/v1/chat/completions", "max_tokens"),
    ("groq", {"groq_api_key": "gsk"}, "https://api.groq.com/openai/v1/chat/completions", "max_completion_tokens"),
])
def test_request_shape(provider, key_kw, url, token_param):
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers["authorization"], json.loads(req.content)
        return httpx.Response(200, json=ok_body())

    llm, _ = make(settings(llm_provider=provider, **key_kw), handler)
    assert call(llm) == {"ok": True}
    assert seen["url"] == url
    assert seen["auth"] == "Bearer " + next(iter(key_kw.values()))
    body = seen["body"]
    assert body["model"] == llm.model_for("large") and body["temperature"] == 0.3 and body[token_param] == 500
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert body["response_format"] == {"type": "json_object"}
    assert llm.usage[-1]["tokens_in"] == 11 and llm.usage[-1]["tokens_out"] == 7 and llm.usage[-1]["provider"] == provider


def test_small_role_uses_small_model():
    seen = {}

    def handler(req):
        seen["model"] = json.loads(req.content)["model"]
        return httpx.Response(200, json=ok_body())

    llm, _ = make(settings(llm_provider="groq", groq_api_key="k"), handler)
    call(llm, model_role="small")
    assert seen["model"] == "openai/gpt-oss-20b"


def test_openrouter_headers_and_reasoning_shape():
    seen = {}

    def handler(req):
        seen["h"], seen["b"] = req.headers, json.loads(req.content)
        return httpx.Response(200, json=ok_body())

    s = settings(llm_provider="openrouter", openrouter_api_key="k", openrouter_site_url="https://x.test",
                 openrouter_app_name="story", llm_reasoning_effort="low")
    llm, _ = make(s, handler)
    call(llm)
    assert seen["h"]["http-referer"] == "https://x.test" and seen["h"]["x-title"] == "story"
    assert seen["b"]["reasoning"] == {"effort": "low"} and "reasoning_effort" not in seen["b"]


@pytest.mark.parametrize("provider,key", [("google", "GOOGLE_API_KEY"), ("groq", "groq_api_key")])
def test_reasoning_effort_for_openai_style_providers(provider, key):
    seen = {}

    def handler(req):
        seen["b"] = json.loads(req.content)
        return httpx.Response(200, json=ok_body())

    llm, _ = make(settings(llm_provider=provider, llm_reasoning_effort="high", **{key: "k"}), handler)
    call(llm)
    assert seen["b"]["reasoning_effort"] == "high" and "reasoning" not in seen["b"]


def test_json_mode_none_sends_no_response_format():
    seen = {}

    def handler(req):
        seen["b"] = json.loads(req.content)
        return httpx.Response(200, json=ok_body())

    llm, _ = make(settings(llm_provider="groq", groq_api_key="k", llm_json_mode="none"), handler)
    call(llm)
    assert "response_format" not in seen["b"]


# ----------------------------- errors and retries -------------------------------------- #
def test_missing_key_names_the_variable():
    llm = build_llm(settings(llm_provider="google"))
    with pytest.raises(ProviderError, match="GOOGLE_API_KEY"):
        call(llm)


@pytest.mark.parametrize("status,hint", [(401, "API key"), (404, "unknown model")])
def test_non_retryable_status_raises_immediately(status, hint):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(status, text="nope")

    llm, sleeps = make(settings(llm_provider="groq", groq_api_key="k"), handler)
    with pytest.raises(ProviderError, match=hint):
        call(llm)
    assert len(calls) == 1 and sleeps == []


def test_transient_errors_retry_with_backoff_then_succeed():
    codes = iter([429, 503, 200])

    def handler(req):
        c = next(codes)
        return httpx.Response(c, json=ok_body()) if c == 200 else httpx.Response(c, text="busy")

    llm, sleeps = make(settings(llm_provider="groq", groq_api_key="k"), handler)
    assert call(llm) == {"ok": True}
    assert len(sleeps) == 2
    assert 1.6 <= sleeps[0] <= 2.4 and 3.2 <= sleeps[1] <= 4.8  # 2 s and 4 s, +/-20% jitter


def test_retry_after_header_is_honored():
    codes = iter([429, 200])

    def handler(req):
        return httpx.Response(429, headers={"retry-after": "10"}, text="slow") if next(codes) == 429 else httpx.Response(200, json=ok_body())

    llm, sleeps = make(settings(llm_provider="groq", groq_api_key="k"), handler)
    call(llm)
    assert 8 <= sleeps[0] <= 12


def test_gives_up_after_five_retries():
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(500, text="boom")

    llm, sleeps = make(settings(llm_provider="groq", groq_api_key="k"), handler)
    with pytest.raises(ProviderError, match="500"):
        call(llm)
    assert len(n) == 6 and len(sleeps) == 5


def test_network_errors_are_retried():
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        if state["n"] < 3:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json=ok_body())

    llm, sleeps = make(settings(llm_provider="groq", groq_api_key="k"), handler)
    assert call(llm) == {"ok": True} and len(sleeps) == 2


def test_response_format_rejection_falls_back_once_and_is_remembered():
    bodies = []

    def handler(req):
        b = json.loads(req.content)
        bodies.append(b)
        if "response_format" in b:
            return httpx.Response(400, text='{"error": "response_format json_object is not supported by this model"}')
        return httpx.Response(200, json=ok_body())

    llm, _ = make(settings(llm_provider="openrouter", openrouter_api_key="k"), handler)
    assert call(llm) == {"ok": True}
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
    call(llm)  # the model is remembered: no third request with response_format
    assert "response_format" not in bodies[2]


def test_other_400_is_not_swallowed():
    llm, _ = make(settings(llm_provider="groq", groq_api_key="k"), lambda r: httpx.Response(400, text="bad temperature"))
    with pytest.raises(ProviderError, match="bad temperature"):
        call(llm)


def test_invalid_json_gets_one_repair_attempt():
    replies = iter(["Sure! here you go: {not json", '```json\n{"ok": 1}\n```'])
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=ok_body(next(replies)))

    llm, _ = make(settings(llm_provider="groq", groq_api_key="k"), handler)
    assert call(llm) == {"ok": 1}
    assert seen[1]["temperature"] == 0.0 and seen[1]["messages"][-1]["role"] == "user"
    assert len(seen[1]["messages"]) == 4


def test_two_invalid_json_replies_raise():
    llm, _ = make(settings(llm_provider="groq", groq_api_key="k"), lambda r: httpx.Response(200, json=ok_body("no json here")))
    with pytest.raises(ProviderError, match="valid JSON twice"):
        call(llm)


def test_truncated_output_explains_reasoning_budget():
    llm, _ = make(settings(llm_provider="google", GOOGLE_API_KEY="k"),
                  lambda r: httpx.Response(200, json=ok_body('{"a":', finish="length")))
    with pytest.raises(ProviderError, match="LLM_REASONING_EFFORT"):
        call(llm)


def test_empty_content_is_an_error():
    llm, _ = make(settings(llm_provider="groq", groq_api_key="k"),
                  lambda r: httpx.Response(200, json={"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}))
    with pytest.raises(ProviderError):
        call(llm)


def test_content_parts_list_is_joined():
    body = {"choices": [{"message": {"content": [{"type": "text", "text": '{"a":'}, {"type": "text", "text": " 1}"}]},
                         "finish_reason": "stop"}]}
    llm, _ = make(settings(llm_provider="groq", groq_api_key="k"), lambda r: httpx.Response(200, json=body))
    assert call(llm) == {"a": 1}


def test_images_are_rejected_for_now():
    llm, _ = make(settings(llm_provider="groq", groq_api_key="k"), lambda r: httpx.Response(200, json=ok_body()))
    with pytest.raises(ProviderError, match="image"):
        call(llm, images=[b"x"])


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('<think>hmm {"x": 2}</think>{"a": 1}', {"a": 1}),
    ('Here: {"a": {"b": [1, 2]}} done', {"a": {"b": [1, 2]}}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_rejects_non_objects():
    for bad in ("[1, 2]", "plain text", ""):
        with pytest.raises(json.JSONDecodeError):
            extract_json(bad)


def test_switching_provider_needs_only_the_env_value(tmp_path):
    """Same code path, only LLM_PROVIDER changes."""
    f = tmp_path / ".env"
    seen = []

    def handler(req):
        seen.append(str(req.url.host))
        return httpx.Response(200, json=ok_body())

    for provider in ("google", "openrouter", "groq"):
        f.write_text(f"LLM_PROVIDER={provider}\nGOOGLE_API_KEY=a\nOPENROUTER_API_KEY=b\nGROQ_API_KEY=c\n")
        llm = OpenAICompatLLM(Settings(_env_file=f), transport=httpx.MockTransport(handler), sleep=lambda s: None)
        call(llm)
    assert seen == ["generativelanguage.googleapis.com", "openrouter.ai", "api.groq.com"]
