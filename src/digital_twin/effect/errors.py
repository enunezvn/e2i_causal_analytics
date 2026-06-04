"""Fail-closed exceptions for the twin effect engine (CLAUDE.md anti-mocking)."""


class EffectDataUnavailable(RuntimeError):
    """Raised when no real labeled (treatment, outcome, confounders) frame is available.

    The estimator MUST NOT fall back to synthetic plausible values or the old
    INTERVENTION_EFFECTS heuristic. Callers surface this as a failed simulation.
    """
