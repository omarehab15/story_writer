from __future__ import annotations

from typing import Any


def check(name: str, passed: bool, detail: str = "", severity: str = "blocker") -> dict[str, Any]:
    return {"name": name, "passed": passed, "severity": severity, "detail": detail}


def result(checks: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Spec 5.0: a stage passes when no blocker or major check failed."""
    passed = not any(not c["passed"] and c["severity"] in ("blocker", "major") for c in checks)
    return {"passed": passed, "checks": checks, **extra}


def failed_checks(res: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in res["checks"] if not c["passed"]]
