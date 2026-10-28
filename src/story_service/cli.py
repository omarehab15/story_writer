"""Command line helpers (run with `uv run <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from story_service.core.config import REPO_ROOT, get_settings
from story_service.pipeline.catalog import load_catalog
from story_service.pipeline.errors import ValidationExhausted
from story_service.pipeline.stages import s5_cast
from story_service.providers.llm import ProviderError, build_llm
from story_service.validators.common import failed_checks

GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden"


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _describe(llm) -> None:
    d = llm.describe()
    print(f"provider : {d['label']}  (LLM_PROVIDER={d['provider']})")
    print(f"base url : {d['base_url']}")
    print(f"api key  : {'set' if d['api_key_set'] else 'MISSING'}")
    print("models   : " + ", ".join(f"{r}={m}" for r, m in d["models"].items()))


def llm_check() -> None:
    """Send one tiny request to the ACTIVE provider to verify key, model and JSON mode."""
    ap = argparse.ArgumentParser(description="Check the active LLM provider from .env")
    ap.add_argument("--role", choices=["large", "small", "vision"], default="large")
    args = ap.parse_args()
    llm = build_llm()
    _describe(llm)
    try:
        out = llm.complete_json('Reply with exactly this JSON object: {"ok": true}', "ping", None, args.role, 0.0, 256)
    except ProviderError as exc:
        print(f"\nFAILED: {exc}")
        sys.exit(1)
    u = llm.usage[-1]
    print(f"\nOK -> {out}   ({u['model']}, {u['latency_ms']} ms, in={u['tokens_in']} out={u['tokens_out']})")


def _show_attempt(rec: s5_cast.AttemptRecord) -> None:
    print(f"\n=== attempt {rec.attempt}: {'PASSED' if rec.passed else 'FAILED'} ===")
    for val in (rec.code_validation, rec.review_validation):
        for c in failed_checks(val or {"checks": []}):
            print(f"  [{c['severity']}] {c['name']}: {c['detail']}")


def run_s5() -> None:
    """Run stage S5 (Cast and Setting) with the ACTIVE provider and print the result."""
    ap = argparse.ArgumentParser(description="Run S5 Cast and Setting")
    ap.add_argument("--script", default=str(GOLDEN / "script.json"), help="script.json from module 4")
    ap.add_argument("--brief", default=str(GOLDEN / "brief.json"), help="brief.json (clothing_norms matters)")
    ap.add_argument("--catalog", default=None, help="catalog.json (default: CATALOG_PATH or the bundled stand-in)")
    ap.add_argument("--age-group", default="9-11", choices=["6-8", "9-11", "12-14", "15-18"])
    ap.add_argument("--out", default="cast_out.json")
    args = ap.parse_args()

    settings = get_settings()
    llm = build_llm(settings)
    _describe(llm)
    try:
        cast = s5_cast.run(
            script=_load(args.script), brief=_load(args.brief),
            catalog=load_catalog(args.catalog or settings.catalog_path), age_group=args.age_group,
            llm=llm, max_attempts=settings.max_stage_attempts, on_attempt=_show_attempt,
        )
    except ProviderError as exc:
        print(f"\nPROVIDER ERROR: {exc}")
        sys.exit(2)
    except ValidationExhausted as exc:
        print(f"\nNEEDS REVIEW ({exc.code}) after {exc.attempts} attempts:")
        for i in exc.issues:
            print(f"  - [{i['severity']}] {i['message']}")
        if exc.last_output is not None:
            Path(args.out).write_text(exc.last_output.model_dump_json(indent=2), encoding="utf-8")
            print(f"last attempt saved to {args.out}")
        sys.exit(1)

    Path(args.out).write_text(cast.model_dump_json(indent=2), encoding="utf-8")
    print("\nCharacters")
    for c in cast.characters:
        acc = f" +{','.join(c.accessories)}" if c.accessories else ""
        print(f"  {c.key:>8}: {c.build}/{c.presentation} skin={c.skin_tone} hair={c.hair_style}/{c.hair_color} "
              f"top={c.top.style}({c.top.color_token}) bottom={c.bottom.style}({c.bottom.color_token}) "
              f"shoes={c.shoes} cover={c.head_covering}{acc} voice={c.voice_profile.energy}")
    print("Settings")
    for s in cast.settings:
        custom = f"  custom={[p.key for p in s.custom_props]}" if s.custom_props else ""
        print(f"  {s.scene_id}: {s.background_id} @ {s.time_of_day}  props={s.prop_ids}{custom}")
    print(f"\nTokens: in={sum(u['tokens_in'] for u in llm.usage)} out={sum(u['tokens_out'] for u in llm.usage)}")
    print(f"Saved to {args.out}")
