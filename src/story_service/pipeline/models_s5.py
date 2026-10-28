"""Pydantic models for stage S5 (Cast and Setting).

Domain values (catalog ids, number ranges, word limits) are checked by `validators/cast.py` so that
failures come back to the LLM as named, explained checks. The schema only fixes the structure.
"""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

TimeOfDay = Literal["day", "evening", "night"]
Build = Literal["child", "teen", "adult", "elder"]  # also the voice age bands


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


# ----------------------------- output: cast.json ----------------------------- #
class Outfit(_Strict):
    style: str
    color_token: str


class VoiceProfile(_Strict):
    gender: Literal["f", "m"]
    age_band: Build
    energy: Literal["calm", "normal", "lively"]


class CastCharacter(_Strict):
    key: str
    build: str
    presentation: str
    skin_tone: int
    hair_style: str
    hair_color: int
    top: Outfit
    bottom: Outfit
    shoes: str
    accessories: list[str] = Field(default_factory=list)
    head_covering: str
    voice_profile: VoiceProfile
    idle_style: Literal["calm", "normal", "lively"]


class CustomProp(_Strict):
    key: str
    description: str


class SceneSetting(_Strict):
    scene_id: str
    background_id: str
    time_of_day: TimeOfDay
    prop_ids: list[str] = Field(default_factory=list)
    custom_props: list[CustomProp] = Field(default_factory=list)


class Cast(_Strict):
    ROUTE: ClassVar[str] = "writer"  # FakeLLM routing key

    characters: list[CastCharacter]
    settings: list[SceneSetting]


# ----------------------------- LLM reviewer ----------------------------------- #
class ReviewFlag(_Strict):
    character_key: str | None = None
    issue: str
    fix_hint: str | None = None


class CastReview(_Strict):
    """`{"flags": []}` means clean. Any flag is a blocker."""

    ROUTE: ClassVar[str] = "reviewer"

    flags: list[ReviewFlag] = Field(default_factory=list)


# ----------------------------- input: script.json (module 4) ------------------ #
class ScriptCharacter(_Loose):
    key: str
    name: str = ""
    role: str = ""
    gender: Literal["f", "m"]
    age_years: int
    traits: list[str] = Field(default_factory=list)


class ScriptLine(_Loose):
    id: str
    speaker: str


class ScriptScene(_Loose):
    id: str
    summary: str = ""
    location: str = ""
    time_of_day: TimeOfDay
    lines: list[ScriptLine] = Field(default_factory=list)

    def speakers(self) -> set[str]:
        return {l.speaker for l in self.lines if l.speaker != "narrator"}


class ScriptIn(_Loose):
    title: str = ""
    characters: list[ScriptCharacter]
    scenes: list[ScriptScene]


class BriefIn(_Loose):
    clothing_norms: str = ""
    school_context: str = ""
    humor_notes: str = ""
    sensitivities: list[str] = Field(default_factory=list)
    family_terms: list[str] = Field(default_factory=list)
