"""S5 Cast and Setting.

Per attempt (max `max_attempts`, default 3 = MAX_STAGE_ATTEMPTS):
    cast LLM (large) -> schema + code validation (blockers)
        -> [only if code validation passed] small LLM reviewer (a flag is a blocker)
On failure the failed checks and reviewer flags are fed back to the cast LLM. When attempts run out
`ValidationExhausted` is raised (the orchestrator makes the version `needs_review`).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from story_service.pipeline.catalog import Catalog
from story_service.pipeline.errors import ValidationExhausted
from story_service.pipeline.models_s5 import BriefIn, Cast, CastReview, ScriptIn
from story_service.pipeline.prompting import input_hash, load_prompt
from story_service.providers.llm.base import LLMClient
from story_service.validators.cast import build_for_age, evaluate_review, validate_raw_cast
from story_service.validators.common import failed_checks

STAGE = "S5_cast"
CAST_PROMPT, REVIEW_PROMPT = "s5_cast.md", "s5_review.md"
CAST_TEMPERATURE, REVIEW_TEMPERATURE = 0.4, 0.2
CAST_MAX_TOKENS, REVIEW_MAX_TOKENS = 8000, 2000


@dataclass
class AttemptRecord:
    """One attempt, for the orchestrator to persist as a `stage_runs` row."""

    attempt: int
    cast: dict[str, Any] | None
    code_validation: dict[str, Any]
    review: dict[str, Any] | None = None
    review_validation: dict[str, Any] | None = None
    passed: bool = False
    feedback_for_next: str | None = None


def _script_view(script: ScriptIn) -> dict[str, Any]:
    return {
        "title": script.title,
        "characters": [
            {**c.model_dump(mode="json"), "expected_build": build_for_age(c.age_years)} for c in script.characters
        ],
        "scenes": [
            {"id": s.id, "summary": s.summary, "location": s.location, "time_of_day": s.time_of_day,
             "speakers": sorted(s.speakers())}
            for s in script.scenes
        ],
    }


def _brief_view(brief: BriefIn) -> dict[str, Any]:
    return {"clothing_norms": brief.clothing_norms, "school_context": brief.school_context,
            "humor_notes": brief.humor_notes, "sensitivities": brief.sensitivities}


def build_cast_messages(script: ScriptIn, brief: BriefIn, catalog: Catalog, age_group: str, feedback: str | None):
    payload = {
        "age_group": age_group,
        "brief": _brief_view(brief),
        "catalog": {
            "character": catalog.character,
            "backgrounds": [b.model_dump(include={"id", "label", "supports"}) for b in catalog.backgrounds],
            "props": [p.model_dump(include={"id", "label", "tags"}) for p in catalog.props],
        },
    }
    user = (
        "Design the cast and the scene settings as one JSON object.\n\n"
        f"<inputs>\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n</inputs>\n\n"
        f"<script_data>\n{json.dumps(_script_view(script), ensure_ascii=False, indent=1)}\n</script_data>"
    )
    if feedback:
        user += f"\n\n<feedback>\n{feedback}\n</feedback>"
    return load_prompt(CAST_PROMPT), user


def build_review_messages(script: ScriptIn, brief: BriefIn, age_group: str, cast: dict[str, Any]):
    payload = {"age_group": age_group, "brief": _brief_view(brief), "script": _script_view(script), "cast": cast}
    user = f"Review this cast.\n\n<cast_data>\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n</cast_data>"
    return load_prompt(REVIEW_PROMPT), user


def _format_feedback(code_val: dict, review: dict | None, review_val: dict | None) -> str:
    parts = []
    for title, val in (("Code checks that failed", code_val), ("Cast reviewer flags", review_val)):
        if val is not None and failed_checks(val):
            parts.append(title + ":\n" + "\n".join(f"- [{c['severity']}] {c['name']}: {c['detail']}" for c in failed_checks(val)))
    return "\n\n".join(parts)


def _issues(*validations: dict | None) -> list[dict[str, Any]]:
    return [
        {"code": "VALIDATION_EXHAUSTED", "severity": c["severity"], "stage": STAGE, "scene_id": None,
         "message": f"{c['name']}: {c['detail']}"}
        for v in validations if v
        for c in failed_checks(v) if c["severity"] in ("blocker", "major")
    ]


def compute_input_hash(inputs: dict[str, Any], model_id: str) -> str:
    return input_hash(inputs, (CAST_PROMPT, REVIEW_PROMPT), model_id)


def run(
    *,
    script: dict | ScriptIn,
    brief: dict | BriefIn,
    catalog: dict | Catalog,
    age_group: str,
    feedback: str | None = None,
    llm: LLMClient,
    max_attempts: int = 3,
    on_attempt: Callable[[AttemptRecord], None] | None = None,
) -> Cast:
    script_m, brief_m, cat_m = ScriptIn.model_validate(script), BriefIn.model_validate(brief), Catalog.model_validate(catalog)
    pending, last_code, last_rev_val, last_cast = feedback, {"checks": []}, None, None

    for attempt in range(1, max_attempts + 1):
        system, user = build_cast_messages(script_m, brief_m, cat_m, age_group, pending)
        raw = llm.complete_json(system, user, Cast, "large", CAST_TEMPERATURE, CAST_MAX_TOKENS)
        cast, code_val = validate_raw_cast(raw, script=script_m, catalog=cat_m)
        rec = AttemptRecord(attempt=attempt, cast=raw, code_validation=code_val)
        last_cast, last_code, last_rev_val = cast or last_cast, code_val, None

        if code_val["passed"]:
            assert cast is not None
            rsys, ruser = build_review_messages(script_m, brief_m, age_group, raw)
            try:
                rev_raw = llm.complete_json(rsys, ruser, CastReview, "small", REVIEW_TEMPERATURE, REVIEW_MAX_TOKENS)
                review = CastReview.model_validate(rev_raw)
                rev_val = evaluate_review(review)
                rec.review = review.model_dump(mode="json")
            except ValidationError as exc:
                rev_val = {"passed": False, "checks": [{"name": "reviewer_schema", "passed": False, "severity": "major",
                                                        "detail": f"cast reviewer returned invalid JSON: {exc.error_count()} error(s)"}]}
            rec.review_validation = last_rev_val = rev_val
            if rev_val["passed"]:
                rec.passed = True
                if on_attempt:
                    on_attempt(rec)
                return cast

        rec.feedback_for_next = pending = _format_feedback(code_val, rec.review, rec.review_validation)
        if on_attempt:
            on_attempt(rec)

    raise ValidationExhausted(_issues(last_code, last_rev_val), max_attempts, last_cast, STAGE)
