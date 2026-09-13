"""Fail-closed exceptions for the twin effect engine (CLAUDE.md anti-mocking).

Standard library only: the tool composer imports this module, and a heavier import here
would load on every tool registration (pinned by ``test_effect_causes_2021``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping, Optional, Union


class EffectCause(StrEnum):
    """Why no effect could be estimated (#2021). Closed set; the tool maps each value to a
    reason code, keyed by the string value."""

    INTERVENTION_NOT_IDENTIFIED = "intervention_not_identified"
    EMPTY_COHORT = "empty_cohort"
    REQUIRED_COLUMN_MISSING = "required_column_missing"
    TOO_FEW_USABLE_ROWS = "too_few_usable_rows"
    NO_TREATMENT_CONTRAST = "no_treatment_contrast"
    TARGET_REGION_NOT_COVERED = "target_region_not_covered"
    ESTIMATION_FAILED = "estimation_failed"
    TARGET_INFERENCE_FAILED = "target_inference_failed"


class EffectDataUnavailable(RuntimeError):
    """Raised when no real labeled (treatment, outcome, confounders) frame is available.

    The estimator MUST NOT fall back to synthetic plausible values or the old
    INTERVENTION_EFFECTS heuristic. Callers surface this as a failed simulation.

    ``cause`` and ``details`` (counts and flags only) say why, for the refusal's reason code.
    Both are optional so the error stays picklable: ``BaseException.__reduce__`` replays the
    message positionally and restores them from ``__dict__``.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[EffectCause] = None,
        details: Optional[Mapping[str, Union[int, float, bool]]] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.details: dict[str, Union[int, float, bool]] = dict(details or {})
