"""
E2I Causal Engine - Structured Error Types

Fail-closed exceptions for causal estimation and refutation. These errors
surface to the agent execute() handler, are recorded as `error_message`,
and propagate to chat UI as "Service unavailable, retry" rather than
producing silent-wrong outputs from mock/placeholder code paths.

This file replaces silent-fallback patterns (e.g., `np.corrcoef`-based mock
estimators in `agents/causal_impact/nodes/estimation.py`; `_mock_*` paths in
`refutation_runner.py`) with structured failures per `CLAUDE.md` §
"CRITICAL — Anti-Mocking & Verification Discipline".

Reference issues: #354 (parent), #416 (F-014 refutation mocks), #417 (F-006
estimation legacy mocks).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CausalEngineError(Exception):
    """Base exception for causal engine errors.

    All structured causal engine errors inherit from this class for unified
    catch-and-surface in agent nodes.

    Attributes:
        message: Human-readable error message (surfaces to chat UI).
        details: Diagnostic context (logged but not surfaced verbatim).
        original_error: Underlying exception, if wrapping one.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result: Dict[str, Any] = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }
        if self.original_error is not None:
            result["original_error"] = str(self.original_error)
            result["original_error_type"] = type(self.original_error).__name__
        return result


class EstimationError(CausalEngineError):
    """Raised when causal effect estimation cannot produce a real result.

    This is a FAIL-CLOSED error: it explicitly refuses to return mock/placeholder
    values when the real (energy-score / EconML / DoWhy) path fails.

    Surfaces to chat as:
        "Causal estimation unavailable for this query. Service unavailable, retry."

    Triggers:
    - Energy score estimator selection failed (all wrappers returned success=False)
    - Explicit method requested but real implementation is unavailable
    - Data preparation failed in a way that can't be silently recovered

    See: #417 (F-006), #354 plan §"silent-fallback trapdoors".
    """


class RefutationError(CausalEngineError):
    """Raised when refutation analysis cannot run against a real CausalModel.

    This is a FAIL-CLOSED error: it explicitly refuses to dispatch to the
    `_mock_*` test paths in `refutation_runner.py` when DoWhy is unavailable
    OR when `CausalModel` reconstruction from estimation outputs fails.

    Surfaces to chat as:
        "Refutation analysis unavailable for this query, retry without refutation."

    Triggers:
    - DoWhy library not installed (DOWHY_AVAILABLE=False)
    - CausalModel reconstruction failed (missing data/treatment/outcome inputs,
      DoWhy internal error, etc.)
    - identify_effect / estimate_effect threw during preparation

    See: #416 (F-014), #354 plan §"silent-fallback trapdoors".
    """
