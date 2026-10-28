from __future__ import annotations

from typing import Any


class ValidationExhausted(Exception):
    """A stage ran out of attempts. `issues` use the API shape {code, severity, stage, scene_id, message}.
    The orchestrator turns this into a `needs_review` version (VALIDATION_EXHAUSTED)."""

    code = "VALIDATION_EXHAUSTED"

    def __init__(self, issues: list[dict[str, Any]], attempts: int, last_output: Any = None, stage: str = ""):
        super().__init__(f"{stage or 'stage'} failed after {attempts} attempts: {len(issues)} unresolved issue(s)")
        self.issues, self.attempts, self.last_output, self.stage = issues, attempts, last_output, stage
