"""Catalog loading. The catalog is the ONLY source of allowed ids and values (spec 6.7)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "assets" / "catalog.json"


class Background(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    label: str = ""
    ground_y: int = 640
    supports: list[str]


class Prop(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    label: str = ""
    tags: list[str] = []


class Catalog(BaseModel):
    model_config = ConfigDict(extra="allow")
    character: dict[str, list[str]]
    backgrounds: list[Background]
    props: list[Prop]

    def bg(self, bg_id: str) -> Background | None:
        return next((b for b in self.backgrounds if b.id == bg_id), None)

    def prop_ids(self) -> set[str]:
        return {p.id for p in self.props}


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or DEFAULT_CATALOG).read_text(encoding="utf-8"))
