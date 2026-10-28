# DECISIONS (Module 5: S5 Cast and Setting + LLM provider layer)

## Scope
- Only S5 plus what it needs: settings, LLM layer, catalog loader, FakeLLM. No orchestrator, DB or other stages.
  `s5_cast.run()` returns `Cast`; after 3 failed attempts it raises `ValidationExhausted` (code
  `VALIDATION_EXHAUSTED`, `.issues`, `.last_output`, `.stage == "S5_cast"`). `on_attempt` gives the orchestrator one
  `AttemptRecord` per attempt for `stage_runs`. `compute_input_hash()` is the cache key.
- Package name stays `story_service` so the folders merge into the full repo layout. Files shared with module 4
  (`prompting.py`, `text_utils.py`, `fake.py`) are identical copies. Module 4's `providers/llm/base.py` gains
  `ProviderError`; module 5's S5 models live in `models_s5.py` to avoid clashing with module 4's `models.py`.

## Catalog
- The real `assets/catalog.json` is generated from the asset library (module 6), which does not exist yet. The bundled one
  is a STAND-IN: character domains are exact (spec 6.3), the 3 backgrounds and 4 props of the spec excerpt are included, and
  the other background and prop ids are placeholders so other stories can be tried. Replace it with the generated file or set
  `CATALOG_PATH`.

## Validation
- Domain values are checked by named code checks, not by the Pydantic schema, so the retry feedback tells the LLM what to fix.
  The schema only fixes structure (a missing field is a `schema` blocker).
- "Differ in at least two of: skin tone, hair style or color, top color" = three dimensions (hair counts once, whether style
  or color differs). Checked for every pair of characters that SPEAK in the same scene.
- Added: `voice_profile.gender == presentation` and `voice_profile.age_band == build` (spec 9.2 selects voices that way).
- Added (major, still fails the stage): the setting's `time_of_day` must equal the script scene's `time_of_day`, so S5 never
  silently overrides S4. The spec's "background supports time of day" check is a separate blocker.
- Raw colors: any `color_token` outside the palette fails, and any hex / `rgb()` / `hsl()` anywhere in the cast fails.
- Reviewer (small LLM) output is `{"flags": [{"character_key", "issue", "fix_hint"}]}`; any flag is a blocker. It only runs when
  the code checks passed. Malformed reviewer JSON fails the attempt (`reviewer_schema`, major).
- Temperatures: cast 0.4, reviewer 0.2 (the spec gives none for S5).
- `head_covering` and clothing vs `clothing_norms` can only be judged by the reviewer, not by code.

## LLM layer and .env
- One OpenAI-compatible client serves OpenRouter, Groq and Google AI Studio (all three expose `/chat/completions`).
  Base URLs and model ids were checked against each provider's docs on 2026-10-25.
- Groq gets `max_completion_tokens` (its documented parameter); the others get `max_tokens`.
- `LLM_REASONING_EFFORT` (optional): `reasoning_effort` for Google and Groq, `reasoning: {effort}` for OpenRouter.
- Truncated or empty output raises `ProviderError` with a hint (reasoning models spend output tokens on thinking).
- Image input is rejected for now (`ProviderError`); the VLM stages (module 6) need it.
- `.env` lookup: `ENV_FILE`, then `./.env`, then the repo root. A blank value (`KEY=`) means "not set". Variables of other
  modules are ignored (`extra="ignore"`), so one file can serve the whole service.
