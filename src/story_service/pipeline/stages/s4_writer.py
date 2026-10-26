"""S4 Story Writer and Critic.

Flow per attempt (max `max_attempts`, default 3 = MAX_STAGE_ATTEMPTS, or env S4_MAX_ATTEMPTS):

    writer LLM (temp 0.7) -> schema + code validation (all blockers)
        -> [only if code validation passed] Story Critic LLM (temp 0.2, separate prompt)
        -> critic pass rule

Any failure feeds the failed checks and the critic report back to the writer. When attempts run
out `ValidationExhausted` is raised; the orchestrator turns it into a `needs_review` version with
`VALIDATION_EXHAUSTED` and skips S5 onward.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from story_service.pipeline.models import (
    AgeSpec,
    Brief,
    Concepts,
    CriticReport,
    Idea,
    Intake,
    Script,
    SpeechStyle,
)
from story_service.pipeline.prompting import (
    canonical_json,
    input_hash,
    load_prompt,
    wrap_lesson_text,
)
from story_service.providers.llm.base import LLMClient
from story_service.validators.script import (
    evaluate_critic,
    failed_checks,
    validate_raw_script,
)

STAGE = "S4_script"
WRITER_PROMPT = "s4_writer.md"
CRITIC_PROMPT = "s4_critic.md"
WRITER_TEMPERATURE = 0.7
CRITIC_TEMPERATURE = 0.2

def _positive_env_int(name: str, default: int) -> int:
    """Read an output-token budget with a clear error for invalid settings."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    return _positive_env_int(name, 0) if value else None


# Optional cap on the estimated INPUT tokens of a writer request (e.g. 7000 for Groq's free tier,
# which rejects requests over 8000 tokens per minute with HTTP 413). Unset = no cap.
MAX_INPUT_TOKENS = _optional_env_int("S4_MAX_INPUT_TOKENS")
CHARS_PER_TOKEN = 3.0  # conservative for Arabic JSON (measured ~3.5)


def estimate_tokens(*texts: str) -> int:
    return round(sum(len(t) for t in texts) / CHARS_PER_TOKEN)


def _compact_previous(previous: dict[str, Any]) -> dict[str, Any]:
    """Shrink a failed script: one short string per line, no `text_spoken` (the writer rebuilds it)."""
    scenes = []
    for s in previous.get("scenes", []):
        if not isinstance(s, dict):
            continue
        lines = [
            f"{l.get('id')}|{l.get('speaker')}|{l.get('emotion', 'neutral')}|{l.get('text_display')}"
            for l in s.get("lines", [])
            if isinstance(l, dict)
        ]
        scenes.append({**{k: v for k, v in s.items() if k != "lines"}, "lines": lines})
    return {**{k: v for k, v in previous.items() if k != "scenes"}, "scenes": scenes}


# Kept below common OpenRouter credit ceilings; override in .env for longer scripts.
WRITER_MAX_TOKENS = _positive_env_int("S4_WRITER_MAX_TOKENS", 16000)
CRITIC_MAX_TOKENS = _positive_env_int("S4_CRITIC_MAX_TOKENS", 3000)
DEFAULT_WPS = {"ar": 2.4, "en": 2.7}


class ValidationExhausted(Exception):
    """All attempts failed. `issues` use the API shape {code, severity, stage, scene_id, message}."""

    code = "VALIDATION_EXHAUSTED"

    def __init__(self, issues: list[dict[str, Any]], attempts: int, last_script: Script | None):
        details = "; ".join(issue["message"] for issue in issues) or "no blocker details recorded"
        super().__init__(
            f"S4 failed after {attempts} attempts: {len(issues)} unresolved issue(s): {details}"
        )
        self.issues = issues
        self.attempts = attempts
        self.last_script = last_script


@dataclass
class AttemptRecord:
    """One writer attempt, for the orchestrator to persist in `stage_runs`."""

    attempt: int
    script: dict[str, Any] | None
    code_validation: dict[str, Any]
    critic_report: dict[str, Any] | None = None
    critic_validation: dict[str, Any] | None = None
    passed: bool = False
    feedback_for_next: str | None = None
    selected: bool = False  # advisory mode: this attempt was the best one and became the result


@dataclass
class _Ctx:
    intake: Intake
    concepts: Concepts
    brief: Brief
    idea: Idea
    age_spec: AgeSpec
    speech_style: SpeechStyle
    wps: dict[str, float]
    display_digits: str = "western"
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def _dump(model: Any) -> Any:
    return model.model_dump(mode="json") if hasattr(model, "model_dump") else model


def _brief_for_prompt(brief: Brief) -> dict[str, Any]:
    d = _dump(brief)
    d.pop("speech_style", None)  # passed separately
    return d


def _word_plan(ctx: _Ctx) -> tuple[int, int, int, int]:
    """(target_words, word_low, word_high, avg_words_per_line) for long scripts."""
    target = ctx.age_spec.target_duration_sec
    wps = ctx.wps.get(ctx.intake.language, 2.5)
    target_words = round(target * wps - 0.3 * 39)
    return (
        target_words,
        round(target_words * 0.85),
        round(target_words * 1.10),
        max(1, round(target_words / 39)),
    )


def _writer_execution_budget(ctx: _Ctx) -> str:
    """Give the writer a concrete size plan, not just a duration formula.

    Models tend to interpret a narrated-story request as a short outline.  The
    validator correctly rejects that, but retry feedback alone is often not
    enough to make the model expand the structure.  This plan is intentionally
    derived from the same duration assumptions used by the validator.
    """
    target = ctx.age_spec.target_duration_sec
    max_chars = max(1, ctx.age_spec.max_speakers - 1)
    shared = (
        f"SPEAKERS: at most {max_chars} characters may speak in total (plus the narrator). "
        "Other people from the idea may be mentioned but must never have a line. "
        "EVERY scene, including the last one, has 3-9 lines; the recap is the final 1-2 lines "
        "of the last scene, which must still have at least 3 lines. "
    )
    if target >= 150:
        _, word_low, word_high, avg = _word_plan(ctx)
        return (
            "EXECUTION BUDGET (mandatory): write exactly 6 scenes, with 6-7 lines in "
            f"each scene (36-42 lines total). Aim for {word_low}-{word_high} spoken words "
            f"in total, to meet the {target}s target: that is about {avg} words in EVERY line, "
            f"so do not write short 3-5 word lines. Include 2-4 interactions, placed in "
            "middle scenes only (s02-s05), in different scenes, with at least 8 lines between "
            "any two of them. " + shared + "This is a full narrated script, not an outline."
        )
    return (
        "EXECUTION BUDGET (mandatory): write 5-6 scenes with 4-7 lines in each scene. "
        "Use the required number of interactions in middle scenes only, in different scenes, "
        "with at least 6 lines between any two of them. " + shared +
        "This is a full narrated script, not an outline; expand the plot until it meets "
        "the requested duration."
    )


def build_writer_messages(
    ctx: _Ctx, feedback: str | None, previous: dict[str, Any] | None = None
) -> tuple[str, str]:
    lesson = (
        f"Topic title: {ctx.intake.topic_title}\n\n"
        f"Topic text (teach ONLY this):\n{ctx.intake.topic_text}\n\n"
        f"Earlier text of the lesson (assumed known, do not teach):\n{ctx.intake.prior_text or '-'}\n\n"
        f"Later topic (do NOT teach, do not preview):\n{ctx.intake.later_preview or '-'}"
    )
    payload = {
        "language": ctx.intake.language,
        "dialect_applied": ctx.intake.dialect_applied,
        "country": ctx.intake.country,
        "grade": ctx.intake.grade,
        "subject": ctx.intake.subject,
        "age_spec": _dump(ctx.age_spec),
        "display_digits": ctx.display_digits,
        "wps": ctx.wps,
        "speech_style": _dump(ctx.speech_style),
        "concepts": _dump(ctx.concepts),
        "selected_idea": _dump(ctx.idea),
        "localization_brief": _brief_for_prompt(ctx.brief),
    }
    user = (
        "Write the script as one JSON object.\n\n"
        f"{_writer_execution_budget(ctx)}\n\n"
        f"<inputs>\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n</inputs>\n\n"
        f"{wrap_lesson_text(lesson)}"
    )
    if feedback:
        user += f"\n\n<feedback>\n{feedback}\n</feedback>"
        system = load_prompt(WRITER_PROMPT)
        if previous:
            # Repair, don't rewrite: a fresh script each time fixes one problem and
            # introduces another. Showing the failed script keeps what already works.
            # Tiers: full script -> compact script -> nothing, whichever first fits the cap.
            tail = (
                "Return the COMPLETE corrected JSON object. Change only what is needed to fix "
                "the problems in <feedback>; keep every other scene, line and field as it is."
            )
            full = (
                f"\n\n<previous_script>\n{json.dumps(previous, ensure_ascii=False)}\n"
                f"</previous_script>\n{tail}"
            )
            compact = (
                "\n\n<previous_script format=\"compact\">\n"
                f"{json.dumps(_compact_previous(previous), ensure_ascii=False)}\n</previous_script>\n"
                "Lines are written as id|speaker|emotion|text_display. Rebuild every line as a full "
                "object (id, speaker, emotion, text_display, text_spoken; text_spoken follows "
                f"text_display with numbers spelled out). {tail}"
            )
            for extra in (full, compact):
                if MAX_INPUT_TOKENS is None or estimate_tokens(system, user, extra) <= MAX_INPUT_TOKENS:
                    user += extra
                    break
        return system, user
    return load_prompt(WRITER_PROMPT), user


def build_critic_messages(ctx: _Ctx, script: dict[str, Any]) -> tuple[str, str]:
    payload = {
        "language": ctx.intake.language,
        "dialect_applied": ctx.intake.dialect_applied,
        "country": ctx.intake.country,
        "age_spec": _dump(ctx.age_spec),
        "speech_style": _dump(ctx.speech_style),
        "concepts": _dump(ctx.concepts),
        "localization_brief": _brief_for_prompt(ctx.brief),
    }
    lesson = (
        f"Topic text:\n{ctx.intake.topic_text}\n\n"
        f"Later topic (must NOT be taught):\n{ctx.intake.later_preview or '-'}"
    )
    user = (
        "Review this script.\n\n"
        f"<context>\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n</context>\n\n"
        f"{wrap_lesson_text(lesson)}\n\n"
        f"<script_json>\n{json.dumps(script, ensure_ascii=False)}\n</script_json>"
    )
    return load_prompt(CRITIC_PROMPT), user


# --------------------------------------------------------------------------- #
# feedback formatting
# --------------------------------------------------------------------------- #
def _format_failures(validation: dict[str, Any]) -> str:
    return "\n".join(
        f"- [{c['severity']}] {c['name']}: {c['detail']}" for c in failed_checks(validation)
    )


def _remedy_hints(ctx: _Ctx, code_validation: dict[str, Any]) -> str:
    """Turn failed code checks into concrete, numeric instructions for the next attempt."""
    names = {c["name"] for c in failed_checks(code_validation)}
    hints: list[str] = []
    est = code_validation.get("estimated_duration_sec")
    target = ctx.age_spec.target_duration_sec
    if "duration" in names and isinstance(est, (int, float)):
        wps = ctx.wps.get(ctx.intake.language, 2.5)
        diff = target - est
        words = round(abs(diff) * wps)
        hints.append(
            f"- Duration: the script is ~{est:.0f}s, the target is {target}s. "
            + (
                f"ADD about {words} spoken words (longer lines of ~9-12 words and more lines), "
                "keeping every line within the word limit."
                if diff > 0
                else f"REMOVE about {words} spoken words."
            )
        )
    if "speaker_limit" in names:
        hints.append(
            f"- Speakers: define at most {max(1, ctx.age_spec.max_speakers - 1)} speaking "
            "characters. Merge or drop the others; they may be mentioned but not speak."
        )
    if "lines_per_scene" in names:
        hints.append(
            "- Lines per scene: every scene (the last one too) needs 3-9 lines. Put the "
            "1-2 recap lines at the end of a normal scene instead of a separate tiny scene."
        )
    if "interaction_gap" in names or "interaction_position" in names:
        hints.append(
            "- Interactions: put them in different middle scenes (not first/last), with at "
            "least 8 lines between two interactions."
        )
    if "schema" in names:
        hints.append(
            "- Schema: the JSON shape is wrong. Use exactly the English key names from the "
            "Output section and include every required key. Each interaction needs id, type, "
            "after_line, concept_id and prompt, plus the fields of its type (targets / options / "
            "pairs / expected_option_id / reveal_text / wrong_feedback)."
        )
    if "spoken_display_similarity" in names:
        hints.append(
            "- text_spoken must be text_display copied word for word, changing only numbers and "
            "symbols into spelled-out words. Rewrite the listed lines that way."
        )
    if "recap" in names:
        hints.append("- Recap: recap_line_ids must hold 1-2 line ids from the LAST scene.")
    return "How to fix:\n" + "\n".join(hints) if hints else ""


def _critic_hints(critic_validation: dict[str, Any]) -> str:
    names = {c["name"] for c in failed_checks(critic_validation)}
    hints: list[str] = []
    if "critic_must_as_plot_tool" in names:
        hints.append(
            "- A concept was delivered as a lecture/mention. Where a character states the rule, "
            "replace it with ACTION: the character cuts, counts, compares or measures, sees a "
            "result that changes the plot, and only then says ONE short line in their own words."
        )
    if "critic_explanatory_ratio" in names:
        hints.append(
            "- Too many explanatory lines. Keep at most 20% of lines explanatory; turn the rest "
            "into action, reaction, a decision or light humor."
        )
    if "critic_issues" in names:
        hints.append(
            "- Fix every listed issue using its fix_hint. Anything inside an interaction (objects, "
            "pieces, numbers) must already have appeared in the story lines before it; never "
            "introduce new quantities or items in an interaction."
        )
    if names & {"critic_each_score", "critic_mean_score"}:
        hints.append(
            "- Scores are low: tighten the plot (one clear problem -> attempts -> result), make "
            "the dialogue sound like children talking, and give the story one small funny moment."
        )
    return "How to fix the Story Critic problems:\n" + "\n".join(hints) if hints else ""


def _compact_critic_report(report: dict[str, Any]) -> str:
    """Only what the writer can act on: the issues with their fix hints (scores and coverage are
    already summarised by the failed rules, and the full report wastes input tokens)."""
    issues = [
        {k: i.get(k) for k in ("severity", "scene_id", "line_id", "message", "fix_hint") if i.get(k)}
        for i in report.get("issues", [])
    ]
    return json.dumps({"issues": issues}, ensure_ascii=False, indent=1) if issues else ""


def _format_feedback(
    code_validation: dict[str, Any],
    critic_report: dict[str, Any] | None,
    critic_validation: dict[str, Any] | None,
    ctx: _Ctx | None = None,
) -> str:
    parts = []
    code_fail = _format_failures(code_validation)
    if code_fail:
        parts.append("Code checks that failed:\n" + code_fail)
        hints = _remedy_hints(ctx, code_validation) if ctx is not None else ""
        if hints:
            parts.append(hints)
    if critic_validation is not None:
        crit_fail = _format_failures(critic_validation)
        if crit_fail:
            parts.append("Story Critic result (failed rules):\n" + crit_fail)
            hints = _critic_hints(critic_validation)
            if hints:
                parts.append(hints)
    if critic_report is not None:
        compact = _compact_critic_report(critic_report)
        if compact:
            parts.append("Story Critic issues to fix:\n" + compact)
    return "\n\n".join(parts)


def _issues_from(
    code_validation: dict[str, Any], critic_validation: dict[str, Any] | None
) -> list[dict[str, Any]]:
    issues = []
    for source in (code_validation, critic_validation):
        if source is None:
            continue
        for c in failed_checks(source):
            if c["severity"] in ("blocker", "major"):
                issues.append(
                    {
                        "code": "VALIDATION_EXHAUSTED",
                        "severity": c["severity"],
                        "stage": STAGE,
                        "scene_id": None,
                        "message": f"{c['name']}: {c['detail']}",
                    }
                )
    return issues


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def compute_input_hash(inputs: dict[str, Any], model_id: str) -> str:
    """Cache key for `stage_runs.input_hash` (spec 5.0)."""
    return input_hash(inputs, (WRITER_PROMPT, CRITIC_PROMPT), model_id)


def _fill_missing_recap(raw: dict[str, Any]) -> dict[str, Any]:
    """If the model forgot `recap_line_ids`, derive it from the last two lines of the last scene.

    The writer prompt already requires the recap to be the final 1-2 lines of the last scene, so
    this only restores a field that is fully determined by that rule. Anything else stays untouched
    and is judged by the validator.
    """
    if not isinstance(raw, dict) or raw.get("recap_line_ids"):
        return raw
    scenes = raw.get("scenes")
    if not isinstance(scenes, list) or not scenes or not isinstance(scenes[-1], dict):
        return raw
    lines = scenes[-1].get("lines")
    if not isinstance(lines, list) or not lines:
        return raw
    ids = [l.get("id") for l in lines[-2:] if isinstance(l, dict) and isinstance(l.get("id"), str)]
    return {**raw, "recap_line_ids": ids} if ids else raw


def _critic(llm: LLMClient, ctx: _Ctx, script: dict[str, Any]) -> CriticReport:
    system, user = build_critic_messages(ctx, script)
    last: Exception | None = None
    for _ in range(2):  # one extra call if the critic returns malformed output
        raw = llm.complete_json(
            system, user, CriticReport, "large", CRITIC_TEMPERATURE, CRITIC_MAX_TOKENS
        )
        try:
            return CriticReport.model_validate(raw)
        except ValidationError as exc:
            last = exc
    assert last is not None
    raise last


def run(
    *,
    intake: dict | Intake,
    concepts: dict | Concepts,
    brief: dict | Brief,
    idea: dict | Idea,
    age_spec: dict | AgeSpec,
    speech_style: dict | SpeechStyle,
    wps: dict[str, float] | None = None,
    feedback: str | None = None,
    llm: LLMClient,
    max_attempts: int | None = None,
    display_digits: str = "western",
    on_attempt: Callable[[AttemptRecord], None] | None = None,
) -> Script:
    ctx = _Ctx(
        intake=Intake.model_validate(intake),
        concepts=Concepts.model_validate(concepts),
        brief=Brief.model_validate(brief),
        idea=Idea.model_validate(idea),
        age_spec=AgeSpec.model_validate(age_spec),
        speech_style=SpeechStyle.model_validate(speech_style),
        wps={**DEFAULT_WPS, **(wps or {})},
        display_digits=display_digits,
    )
    language = ctx.intake.language
    must_ids = set(ctx.concepts.ids("must"))
    assert must_ids, "S2 must provide at least one 'must' concept"

    max_attempts = max_attempts or _optional_env_int("S4_MAX_ATTEMPTS") or 3
    advisory = os.getenv("S4_CRITIC_ADVISORY", "").strip().lower() in ("1", "true", "yes")
    # Advisory mode: a failing Critic triggers a revision (same loop, same attempt budget) and, when
    # attempts run out, the best code-valid script is returned instead of raising.
    candidates: list[tuple[tuple[float, float, int], Script, AttemptRecord]] = []

    pending_feedback = feedback
    last_script: Script | None = None
    last_code: dict[str, Any] = {"checks": []}
    last_crit_val: dict[str, Any] | None = None
    last_raw: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        system, user = build_writer_messages(ctx, pending_feedback, last_raw)
        try:
            raw = llm.complete_json(
                system, user, Script, "large", WRITER_TEMPERATURE, WRITER_MAX_TOKENS
            )
        except Exception as exc:
            # Provider-side structured-output validation can reject a model
            # response before it is available to our JSON/schema validator.
            # Treat it as a failed writer attempt so transient failures do not
            # terminate the whole stage and the remaining attempts still run.
            code_val = {
                "passed": False,
                "checks": [
                    {
                        "name": "writer_request",
                        "passed": False,
                        "severity": "blocker",
                        "detail": f"Writer request failed: {type(exc).__name__}: {exc}",
                    }
                ],
            }
            rec = AttemptRecord(attempt=attempt, script=None, code_validation=code_val)
            last_code, last_crit_val = code_val, None
            rec.feedback_for_next = pending_feedback = (
                "Writer request failed before a script was returned. "
                "Return one complete JSON object that includes every required field, "
                "especially recap_line_ids.\n\n"
                + _format_feedback(code_val, None, None)
            )
            if on_attempt:
                on_attempt(rec)
            continue
        raw = _fill_missing_recap(raw)
        last_raw = raw
        script, code_val = validate_raw_script(
            raw,
            concepts=ctx.concepts,
            age_spec=ctx.age_spec,
            language=language,
            wps=ctx.wps,
            display_digits=display_digits,
        )
        rec = AttemptRecord(attempt=attempt, script=raw, code_validation=code_val)
        last_script, last_code, last_crit_val = script or last_script, code_val, None

        if code_val["passed"]:
            assert script is not None
            try:
                report = _critic(llm, ctx, raw)
            except ValidationError as exc:
                crit_val = {
                    "passed": False,
                    "checks": [
                        {
                            "name": "critic_schema",
                            "passed": False,
                            "severity": "major",
                            "detail": f"Story Critic returned invalid JSON twice: {exc.error_count()} error(s)",
                        }
                    ],
                }
                rec.critic_validation = last_crit_val = crit_val
            else:
                crit_val = evaluate_critic(report, concepts=ctx.concepts)
                rec.critic_report = report.model_dump(mode="json")
                rec.critic_validation = last_crit_val = crit_val
            if crit_val["passed"]:
                rec.passed = True
                if on_attempt:
                    on_attempt(rec)
                return script
            if advisory:
                # Rank by fewest failed Critic rules first, then by mean score, then latest attempt.
                n_failed = sum(1 for c in failed_checks(crit_val) if c["severity"] != "minor")
                mean = (crit_val.get("scores") or {}).get("mean", -1.0)
                candidates.append(((-n_failed, mean, attempt), script, rec))

        rec.feedback_for_next = pending_feedback = _format_feedback(
            code_val, rec.critic_report, rec.critic_validation, ctx
        )
        if on_attempt:
            on_attempt(rec)

    if candidates:
        _, best, best_rec = max(candidates, key=lambda c: c[0])
        best_rec.selected = True
        return best
    raise ValidationExhausted(_issues_from(last_code, last_crit_val), max_attempts, last_script)


__all__ = [
    "AttemptRecord",
    "ValidationExhausted",
    "build_critic_messages",
    "build_writer_messages",
    "canonical_json",
    "compute_input_hash",
    "run",
]