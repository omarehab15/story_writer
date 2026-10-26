"""Run stage S4 against a real LLM and print what happens.

    uv run python scripts/run_s4.py                 # sample request (fractions, Egypt, 9-11)
    uv run python scripts/run_s4.py --inputs my.json --out my_script.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from story_service.pipeline.stages import s4_writer
from story_service.providers.llm import create_llm
from story_service.validators.script import failed_checks


def show_attempt(rec):
    print(f"\n=== attempt {rec.attempt}: {'PASSED' if rec.passed else 'FAILED'} ===")
    print(f"estimated duration: {rec.code_validation.get('estimated_duration_sec')} s")
    for c in failed_checks(rec.code_validation):
        print(f"  code  [{c['severity']}] {c['name']}: {c['detail']}")
    if rec.critic_validation:
        print(f"  critic scores: {rec.critic_validation.get('scores')}")
        for c in failed_checks(rec.critic_validation):
            print(f"  critic [{c['severity']}] {c['name']}: {c['detail']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(ROOT / "tests/fixtures/s4/golden_inputs.json"))
    ap.add_argument("--out", default="script_out.json")
    args = ap.parse_args()
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    llm = create_llm()
    records = []

    def on_attempt(rec):
        records.append(rec)
        show_attempt(rec)

    try:
        script = s4_writer.run(**inputs, llm=llm, on_attempt=on_attempt)
    except s4_writer.ValidationExhausted as exc:
        print(f"\nNEEDS REVIEW ({exc.code}) after {exc.attempts} attempts:")
        for i in exc.issues:
            print(f"  - [{i['severity']}] {i['message']}")
        if exc.last_script:
            Path(args.out).write_text(exc.last_script.model_dump_json(indent=2), encoding="utf-8")
            print(f"last attempt saved to {args.out}")
        sys.exit(1)
    chosen = next((r for r in records if r.selected), None)
    if chosen:
        print(f"\nADVISORY: the Critic never fully passed; best attempt ({chosen.attempt}) was kept.")
    Path(args.out).write_text(script.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nTitle: {script.title}\n{script.logline}\n")
    for s in script.scenes:
        print(f"[{s.id}] {s.summary}")
        for l in s.lines:
            print(f"   {l.speaker:>8}: {l.text_display}")
        if s.interaction:
            print(f"   >> interaction {s.interaction.type}: {s.interaction.prompt}")
    print(f"\nTokens: in={sum(u['tokens_in'] for u in llm.usage)} out={sum(u['tokens_out'] for u in llm.usage)}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()