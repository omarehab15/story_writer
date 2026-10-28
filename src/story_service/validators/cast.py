"""Code validation for S5 (all blockers unless noted) and the LLM reviewer rule."""
from __future__ import annotations

import itertools
import re
from typing import Any

from pydantic import ValidationError

from story_service.pipeline.catalog import Catalog
from story_service.pipeline.models_s5 import Cast, CastReview, ScriptIn
from story_service.validators.common import check, result

MAX_CUSTOM_PROPS = 2
MAX_CUSTOM_PROP_WORDS = 20
MIN_DIFFERENCES = 2
INDEX_RANGE = range(6)
_RAW_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(")
_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def build_for_age(age_years: int) -> str:
    """Spec S5: up to 11 child, 12-17 teen, 18-59 adult, 60+ elder."""
    if age_years <= 11:
        return "child"
    if age_years <= 17:
        return "teen"
    return "adult" if age_years <= 59 else "elder"


def differences(a: Any, b: Any) -> list[str]:
    """Visual dimensions in which two characters differ: skin tone, hair (style or color), top color."""
    d = []
    if a.skin_tone != b.skin_tone:
        d.append("skin_tone")
    if a.hair_style != b.hair_style or a.hair_color != b.hair_color:
        d.append("hair")
    if a.top.color_token != b.top.color_token:
        d.append("top_color")
    return d


def _strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def validate_cast(cast: Cast, *, script: ScriptIn, catalog: Catalog) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    cc = catalog.character
    chars = {c.key: c for c in cast.characters}
    script_chars = {c.key: c for c in script.characters}

    # ---- keys -------------------------------------------------------------
    keys = [c.key for c in cast.characters]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    unknown = sorted(set(keys) - set(script_chars))
    missing = sorted(set(script_chars) - set(keys))
    checks.append(check(
        "character_keys", not (dup or unknown or missing),
        f"unknown keys (not in script): {unknown}; missing characters: {missing}; duplicates: {dup}"))

    # ---- catalog references ------------------------------------------------
    bad: list[str] = []

    def need(label: str, value: str, domain_key: str):
        if value not in cc.get(domain_key, []):
            bad.append(f"{label}='{value}' not in catalog {domain_key} {cc.get(domain_key)}")

    for c in cast.characters:
        need(f"{c.key}.build", c.build, "build")
        need(f"{c.key}.hair_style", c.hair_style, "hair_style")
        need(f"{c.key}.top.style", c.top.style, "top_style")
        need(f"{c.key}.bottom.style", c.bottom.style, "bottom_style")
        need(f"{c.key}.shoes", c.shoes, "shoes")
        need(f"{c.key}.head_covering", c.head_covering, "head_covering")
        for acc in c.accessories:
            need(f"{c.key}.accessories", acc, "accessories")
        if len(set(c.accessories)) != len(c.accessories):
            bad.append(f"{c.key}.accessories has duplicates")
        if c.presentation not in ("f", "m"):
            bad.append(f"{c.key}.presentation='{c.presentation}' must be 'f' or 'm'")
    checks.append(check("catalog_references", not bad, "; ".join(bad[:8])))

    # ---- colors: palette tokens only ---------------------------------------
    tok_bad = [
        f"{c.key}.{part}.color_token='{o.color_token}'"
        for c in cast.characters for part, o in (("top", c.top), ("bottom", c.bottom))
        if o.color_token not in cc.get("color_token", [])
    ]
    raw = [s for s in _strings(cast.model_dump()) if _RAW_COLOR.search(s)]
    checks.append(check(
        "palette_tokens_only", not (tok_bad or raw),
        f"not a palette token: {tok_bad}; raw color values found: {raw}. Allowed: {cc.get('color_token')}"))

    # ---- number ranges -------------------------------------------------------
    rng = [f"{c.key}.skin_tone={c.skin_tone}" for c in cast.characters if c.skin_tone not in INDEX_RANGE]
    rng += [f"{c.key}.hair_color={c.hair_color}" for c in cast.characters if c.hair_color not in INDEX_RANGE]
    checks.append(check("color_indices", not rng, f"must be integers 0 to 5: {rng}"))

    # ---- build / presentation / voice follow the script -----------------------
    wrong_build, wrong_pres, wrong_voice = [], [], []
    for c in cast.characters:
        s = script_chars.get(c.key)
        if s is None:
            continue
        if c.build != build_for_age(s.age_years):
            wrong_build.append(f"{c.key}: build '{c.build}' but age_years {s.age_years} needs '{build_for_age(s.age_years)}'")
        if c.presentation != s.gender:
            wrong_pres.append(f"{c.key}: presentation '{c.presentation}' but script gender is '{s.gender}'")
        if c.voice_profile.gender != c.presentation or c.voice_profile.age_band != c.build:
            wrong_voice.append(f"{c.key}: voice_profile gender/age_band must equal presentation/build")
    checks.append(check("build_follows_age", not wrong_build, "; ".join(wrong_build)))
    checks.append(check("presentation_follows_gender", not wrong_pres, "; ".join(wrong_pres)))
    checks.append(check("voice_profile_consistent", not wrong_voice, "; ".join(wrong_voice)))

    # ---- distinctness inside each scene ---------------------------------------
    clashes: list[str] = []
    seen: set[tuple[str, str]] = set()
    for scene in script.scenes:
        present = sorted(k for k in scene.speakers() if k in chars)
        for a, b in itertools.combinations(present, 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            diff = differences(chars[a], chars[b])
            if len(diff) < MIN_DIFFERENCES:
                clashes.append(f"{a} and {b} (together in {scene.id}) differ only in {diff or 'nothing'}")
    checks.append(check(
        "distinct_characters", not clashes,
        f"{'; '.join(clashes)}. Characters in the same scene must differ in at least {MIN_DIFFERENCES} of: "
        f"skin tone, hair style or color, top color"))

    # ---- settings ---------------------------------------------------------------
    script_ids = [s.id for s in script.scenes]
    setting_ids = [s.scene_id for s in cast.settings]
    miss = [i for i in script_ids if i not in setting_ids]
    extra = sorted(set(setting_ids) - set(script_ids))
    dups = sorted({i for i in setting_ids if setting_ids.count(i) > 1})
    checks.append(check(
        "settings_complete", not (miss or extra or dups),
        f"every scene needs exactly one setting. missing: {miss}; unknown scene ids: {extra}; duplicated: {dups}"))

    by_scene = {s.id: s for s in script.scenes}
    support_bad, tod_bad, prop_bad, custom_bad = [], [], [], []
    known_props = catalog.prop_ids()
    for st in cast.settings:
        bg = catalog.bg(st.background_id)
        if bg is None:
            support_bad.append(f"{st.scene_id}: unknown background '{st.background_id}'")
        elif st.time_of_day not in bg.supports:
            support_bad.append(f"{st.scene_id}: '{st.background_id}' does not support '{st.time_of_day}' (supports {bg.supports})")
        sc = by_scene.get(st.scene_id)
        if sc is not None and st.time_of_day != sc.time_of_day:
            tod_bad.append(f"{st.scene_id}: setting says '{st.time_of_day}' but the script scene is '{sc.time_of_day}'")
        unk = [p for p in st.prop_ids if p not in known_props]
        if unk:
            prop_bad.append(f"{st.scene_id}: unknown props {unk}")
        if len(set(st.prop_ids)) != len(st.prop_ids):
            prop_bad.append(f"{st.scene_id}: duplicate prop ids")
        if len(st.custom_props) > MAX_CUSTOM_PROPS:
            custom_bad.append(f"{st.scene_id}: {len(st.custom_props)} custom props (max {MAX_CUSTOM_PROPS})")
        ckeys = [cp.key for cp in st.custom_props]
        if len(set(ckeys)) != len(ckeys):
            custom_bad.append(f"{st.scene_id}: duplicate custom prop keys")
        for cp in st.custom_props:
            n = len(cp.description.split())
            if n == 0 or n > MAX_CUSTOM_PROP_WORDS:
                custom_bad.append(f"{st.scene_id}.{cp.key}: description has {n} words (1 to {MAX_CUSTOM_PROP_WORDS})")
            if not _SNAKE.match(cp.key):
                custom_bad.append(f"{st.scene_id}.{cp.key}: key must be snake_case ascii")
    checks.append(check("background_supports_time_of_day", not support_bad, "; ".join(support_bad)))
    checks.append(check("time_of_day_matches_script", not tod_bad, "; ".join(tod_bad), severity="major"))
    checks.append(check("props_known", not prop_bad, "; ".join(prop_bad)))
    checks.append(check("custom_props", not custom_bad, "; ".join(custom_bad)))
    return result(checks)


def validate_raw_cast(raw: dict[str, Any], *, script: ScriptIn, catalog: Catalog) -> tuple[Cast | None, dict[str, Any]]:
    try:
        cast = Cast.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8])
        return None, result([check("schema", False, details)])
    return cast, validate_cast(cast, script=script, catalog=catalog)


def evaluate_review(review: CastReview) -> dict[str, Any]:
    """A reviewer flag is a blocker (stereotyping, caricature, clothing norms)."""
    flags = "; ".join(
        f"{f.character_key or 'cast'}: {f.issue}" + (f" (fix: {f.fix_hint})" if f.fix_hint else "") for f in review.flags
    )
    return result([check("reviewer_flags", not review.flags, flags)])
