"""Run tracking helpers for notebooks and dev scripts.

Everything the notebook needs lives here, so the notebook only imports and calls:
    settings_table, make_run_dir, save, show_json, checks_table, TracingLLM, build_pipeline_llm

`pandas` and `IPython` are only needed by the display helpers and are imported lazily.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from story_service.core.config import REPO_ROOT, Settings
from story_service.providers.llm.fake import FakeLLM

DEFAULT_MODE = "fake"
DEFAULT_AGE_GROUP = "9-11"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden"


# ----------------------------- settings (from .env) ------------------------------ #
def resolve_run_settings(settings: Settings) -> dict[str, Any]:
    """The notebook-level settings, with defaults applied. Everything comes from .env / environment."""
    return {
        "mode": settings.pipeline_mode or DEFAULT_MODE,
        "age_group": settings.s5_age_group or DEFAULT_AGE_GROUP,
        "script_path": Path(settings.s5_script_path) if settings.s5_script_path else GOLDEN / "script.json",
        "brief_path": Path(settings.s5_brief_path) if settings.s5_brief_path else GOLDEN / "brief.json",
        "catalog_path": settings.catalog_path,  # None = bundled stand-in
        "runs_dir": Path(settings.runs_dir) if settings.runs_dir else REPO_ROOT / "runs",
        "max_attempts": settings.max_stage_attempts,
    }


def settings_table(settings: Settings, run: dict[str, Any] | None = None):
    """DataFrame of the effective settings. API keys are masked."""
    import pandas as pd

    run = run or resolve_run_settings(settings)
    prov = settings.llm_provider

    def mask(v: str | None) -> str:
        return "MISSING" if not v else f"set ({len(v)} chars)"

    rows = [("PIPELINE_MODE", run["mode"]), ("LLM_PROVIDER", prov),
            (f"{prov.upper()}_API_KEY", mask(settings.provider_value(prov, "api_key"))),
            ("LLM_JSON_MODE", settings.llm_json_mode), ("LLM_REASONING_EFFORT", settings.llm_reasoning_effort or "(provider default)"),
            ("LLM_TIMEOUT_SEC", settings.llm_timeout_sec), ("MAX_STAGE_ATTEMPTS", run["max_attempts"]),
            ("S5_AGE_GROUP", run["age_group"]), ("S5_SCRIPT_PATH", run["script_path"]), ("S5_BRIEF_PATH", run["brief_path"]),
            ("CATALOG_PATH", run["catalog_path"] or "(bundled stand-in)"), ("RUNS_DIR", run["runs_dir"])]
    return pd.DataFrame(rows, columns=["setting", "value"]).set_index("setting")


def build_pipeline_llm(settings: Settings):
    """FakeLLM (offline fixtures) or the real provider, according to PIPELINE_MODE in .env."""
    mode = settings.pipeline_mode or DEFAULT_MODE
    if mode == "fake":
        return FakeLLM("s5_cast.json", reviewer="s5_review_clean.json")
    from story_service.providers.llm import build_llm

    return build_llm(settings)


# ----------------------------- saving / display ---------------------------------- #
def make_run_dir(base: str | Path) -> Path:
    path = Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(run_dir: Path, name: str, obj: Any) -> Path:
    """Save a dict/list as JSON or a str as text into the run folder."""
    path = Path(run_dir) / name
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def show_json(title: str, obj: Any, max_chars: int = 4000) -> None:
    from IPython.display import Markdown, display

    txt = json.dumps(obj, ensure_ascii=False, indent=2)
    display(Markdown(f"**{title}**  ·  {len(txt):,} chars"))
    print(txt if len(txt) <= max_chars else txt[:max_chars] + f"\n... [{len(txt) - max_chars:,} more chars, see the file in runs/]")


def checks_table(val: dict[str, Any]) -> None:
    """Table of every check in a validation result, with a pass/fail verdict."""
    import pandas as pd
    from IPython.display import Markdown, display

    rows = [{"": "✅" if c["passed"] else "❌", "severity": c["severity"], "check": c["name"], "detail": c["detail"]}
            for c in val["checks"]]
    df = pd.DataFrame(rows)
    verdict = "PASSED ✅" if val["passed"] else "FAILED ❌"
    display(Markdown(f"**Verdict: {verdict}**  ({sum(c['passed'] for c in val['checks'])}/{len(val['checks'])} checks passed)"))
    try:
        display(df.style.hide(axis="index").set_properties(**{"text-align": "left"}))
    except Exception:  # jinja2 missing
        display(df)


# ----------------------------- LLM tracing --------------------------------------- #
class TracingLLM:
    """Wraps any LLM client (fake or real) and records every call: prompt, response, time, tokens.

    Each call is written to `<run_dir>/callNN_<Schema>_request.md` and `..._response.json`.
    """

    def __init__(self, inner, run_dir: Path, verbose: bool = True):
        self.inner, self.run_dir, self.verbose = inner, Path(run_dir), verbose
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace: list[dict[str, Any]] = []

    def complete_json(self, system, user, schema, model_role="large", temperature=0.2, max_tokens=4096, images=None):
        n = len(self.trace) + 1
        sname = getattr(schema, "__name__", str(schema))
        used_before = len(getattr(self.inner, "usage", []))
        rec: dict[str, Any] = {"n": n, "schema": sname, "role": model_role, "temp": temperature, "max_tokens": max_tokens,
                               "prompt_chars": len(system) + len(user), "user": user, "ok": True, "error": None}
        stem = f"call{n:02d}_{sname}"
        (self.run_dir / f"{stem}_request.md").write_text(f"# SYSTEM\n\n{system}\n\n# USER\n\n{user}", encoding="utf-8")
        t0 = time.monotonic()
        try:
            out = self.inner.complete_json(system, user, schema, model_role, temperature, max_tokens, images)
        except Exception as exc:
            rec.update(ok=False, error=repr(exc), ms=int((time.monotonic() - t0) * 1000))
            self.trace.append(rec)
            if self.verbose:
                print(f"  ✗ call {n} [{sname}/{model_role}] FAILED: {exc}")
            raise
        rec["ms"] = int((time.monotonic() - t0) * 1000)
        new_usage = getattr(self.inner, "usage", [])[used_before:]
        rec["model"] = new_usage[-1]["model"] if new_usage else "fake"
        rec["tokens_in"] = sum(u["tokens_in"] for u in new_usage)
        rec["tokens_out"] = sum(u["tokens_out"] for u in new_usage)
        rec["response"] = out
        self.trace.append(rec)
        (self.run_dir / f"{stem}_response.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.verbose:
            print(f"  ✓ call {n} [{sname}/{model_role}] model={rec['model']} {rec['ms']} ms "
                  f"in={rec['tokens_in']} out={rec['tokens_out']}  -> {self.run_dir.parent.name}/{self.run_dir.name}/{stem}_*")
        return out

    def __getattr__(self, item):  # describe(), usage, ...
        return getattr(self.inner, item)

    def trace_df(self):
        import pandas as pd

        cols = ["n", "schema", "role", "model", "temp", "ms", "tokens_in", "tokens_out", "prompt_chars", "ok", "error"]
        return pd.DataFrame(self.trace).reindex(columns=cols)
