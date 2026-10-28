# story-service: Module 5 (S5 Cast and Setting) + multi-provider LLM layer

Standalone slice of the Interactive Lesson Stories service. It takes the `script.json` from module 4 and
produces `cast.json`: one visual design per character and one background/props setting per scene.
The LLM can be OpenRouter, Groq or Google AI Studio, chosen with one line in `.env`.

## Quick start (uv)

```bash
cp .env.example .env            # the ONE unified .env
# edit .env:  LLM_PROVIDER=google|groq|openrouter  and the matching *_API_KEY
uv sync                         # creates .venv from uv.lock
uv run llm-check                # one tiny request: verifies provider, key, model, JSON mode
uv run run-s5                   # runs S5 on the bundled golden script and prints the cast
```

Use the output of module 4 as input:

```bash
uv run run-s5 --script ../story-service/script_out.json --brief my_brief.json --age-group 9-11 --out cast_out.json
```

Switch provider without touching code: change `LLM_PROVIDER` in `.env`, or one-off:
`LLM_PROVIDER=groq uv run run-s5` (real environment variables win over the file).
Share one file between module folders: `ENV_FILE=../.env uv run run-s5`.

## Providers

| `LLM_PROVIDER` | Key variable | Endpoint | Default large / small |
| --- | --- | --- | --- |
| `google` (AI Studio) | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | `generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3.5-flash` / `gemini-3.5-flash-lite` |
| `openrouter` | `OPENROUTER_API_KEY` | `openrouter.ai/api/v1` | `google/gemini-3.5-flash` / `google/gemini-3.5-flash-lite` |
| `groq` | `GROQ_API_KEY` | `api.groq.com/openai/v1` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` |

All three speak the OpenAI chat-completions protocol, so there is one client (`providers/llm/openai_compat.py`).
Model precedence: `LLM_MODEL_<ROLE>` > `<PROVIDER>_MODEL_<ROLE>` > built-in default. Transient errors
(429, 5xx, timeouts) retry with 2/4/8/16/32 s backoff (+/-20%), `Retry-After` is honored, JSON mode falls back
automatically if a model rejects `response_format`, invalid JSON gets one repair attempt.
Model ids change often: if you get HTTP 404, set the model in `.env`.

## Layout

```
.env.example                          unified settings (providers, models, limits, other modules' vars)
src/story_service/core/config.py      Settings (reads .env)
src/story_service/providers/llm/      base.py, openai_compat.py (3 providers), fake.py (tests)
src/story_service/pipeline/stages/s5_cast.py    stage: cast LLM -> code checks -> reviewer LLM -> retries
src/story_service/validators/cast.py            all S5 code checks
src/story_service/pipeline/prompts/             s5_cast.md, s5_review.md
src/story_service/assets/catalog.json           STAND-IN catalog (see docs/DECISIONS.md)
tests/                                          uv run pytest -q   (82 tests, no network)
```

The LLM layer implements the same `complete_json(...)` interface as module 4, so it can be dropped into
module 4's `scripts/run_s4.py` (`from story_service.providers.llm import build_llm`).
