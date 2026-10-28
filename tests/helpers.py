import copy
import json
from pathlib import Path

from story_service.pipeline.catalog import Catalog, load_catalog
from story_service.pipeline.models_s5 import ScriptIn
from story_service.validators.cast import validate_raw_cast

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG = load_catalog()


def load(name: str) -> dict:
    """`load("golden/script.json")` -> fixtures/golden/script.json"""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def golden_cast() -> dict:
    return copy.deepcopy(json.loads((FIXTURES / "llm" / "s5_cast.json").read_text(encoding="utf-8")))


def validate(raw: dict, script: dict | None = None):
    return validate_raw_cast(
        raw, script=ScriptIn.model_validate(script or load("golden/script.json")), catalog=Catalog.model_validate(CATALOG)
    )


def failed_names(res: dict) -> set[str]:
    return {c["name"] for c in res["checks"] if not c["passed"] and c["severity"] != "minor"}


def char(raw: dict, key: str) -> dict:
    return next(c for c in raw["characters"] if c["key"] == key)


def setting(raw: dict, sid: str) -> dict:
    return next(s for s in raw["settings"] if s["scene_id"] == sid)
