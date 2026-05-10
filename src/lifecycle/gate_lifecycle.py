"""Canonical lifecycle-state enum + Pydantic declaration model.

Plan v4 Gate N2 (codex-rescue MEDIUM-1) implementation.

The advisory invariant must guard BEHAVIOR, not SPELLING. A naive scan for
``advisory_mode: true`` misses configs that use other names (``mode: advisory``,
``enforcement: false``, ``shadow: true``, ``warn_only: true``). This module
defines the canonical vocabulary every gate-relevant location must use.

Five lifecycle states cover the full ML-gate lifecycle:

* ``DEVELOPMENT`` — gate code exists but is not connected to production.
  Used for new features whose plumbing is in place but whose calibration
  window has not yet started. NEVER blocks anything.
* ``ADVISORY`` — gate emits structured signals (logs, validation_metrics
  keys, denial reasons) but does NOT block deployment, halt the pipeline,
  or drop features. Operator-facing observability. The 5-pass progression
  (T2.2 / T2.3 / T2.6a / T2.6b / T2.4) all currently sit here.
* ``CALIBRATING`` — calibration window is in progress. Gate emits same
  signals as ADVISORY plus a structured "would-be reject" rate metric so
  operators can size the future enforcement impact. Still does NOT block.
* ``ENFORCED`` — gate is connected to production. Verdict drops a feature,
  halts a pipeline, or denies a promotion. Transitions INTO this state
  require a signed lifecycle-change doc (start_date, end_date, drift_summary,
  signing_reviewer per Gate N2 acceptance #3).
* ``DEPRECATED`` — gate has been superseded by a newer gate. Code may
  remain for backward-compat reads but new emissions stop. Transitions
  out of any state INTO ``DEPRECATED`` are the terminal lifecycle move.

The transition policy is enforced by ``scripts/check_lifecycle_state.py``
+ ``.github/workflows/lifecycle_state_guard.yml``.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class GateLifecycleState(str, Enum):
    """Canonical lifecycle-state enum for gate-relevant code paths.

    Subclasses ``str`` so it serializes naturally to YAML/JSON as the
    lowercase string value.  Values are stable forever — DO NOT change
    or remove a value without coordinating with all checked-in
    declarations + the lifecycle-change-doc workflow.
    """

    DEVELOPMENT = "development"
    """Gate code exists but is not connected to production.

    Plumbing complete; calibration window not yet open. Never blocks.
    """

    ADVISORY = "advisory"
    """Gate emits structured signals but does NOT block.

    Operator-facing observability only. The 5-pass v3 features
    (T2.2 perm-anchored AUC floor, T2.3 cohort-derived honest band,
    T2.6a deployer input metrics, T2.6b shadow reporting, T2.4
    imputation audit) all currently live here.
    """

    CALIBRATING = "calibrating"
    """Calibration window is in progress.

    Gate emits ADVISORY signals plus a structured ``would_be_reject_rate``
    metric so operators can size the future enforcement impact. Still
    does NOT block. Bridge state between ADVISORY and ENFORCED.
    """

    ENFORCED = "enforced"
    """Gate is connected to production.

    Verdict drops a feature, halts a pipeline, or denies a promotion.
    Transitions INTO this state require a signed lifecycle-change doc.
    """

    DEPRECATED = "deprecated"
    """Gate has been superseded by a newer gate.

    Code may remain for backward-compat reads but new emissions stop.
    Terminal lifecycle move.
    """


# Allowed transitions per the v4 plan's lifecycle policy. Each key is the
# CURRENT state; the value is the set of states the gate may move INTO.
# DEPRECATED is terminal (no out-transitions). DEVELOPMENT cannot jump
# straight to ENFORCED — it must spend at least one CALIBRATING window
# (Gate N2 acceptance #3 requires a signed doc for ENFORCED transitions
# AND the calibration window's drift_summary, which only exists if a
# CALIBRATING window ran).
_ALLOWED_TRANSITIONS: dict[GateLifecycleState, frozenset[GateLifecycleState]] = {
    GateLifecycleState.DEVELOPMENT: frozenset(
        {GateLifecycleState.ADVISORY, GateLifecycleState.DEPRECATED}
    ),
    GateLifecycleState.ADVISORY: frozenset(
        {GateLifecycleState.CALIBRATING, GateLifecycleState.DEPRECATED}
    ),
    GateLifecycleState.CALIBRATING: frozenset(
        {
            GateLifecycleState.ADVISORY,  # rollback if calibration fails
            GateLifecycleState.ENFORCED,
            GateLifecycleState.DEPRECATED,
        }
    ),
    GateLifecycleState.ENFORCED: frozenset(
        {
            GateLifecycleState.CALIBRATING,  # re-calibrate after drift
            GateLifecycleState.DEPRECATED,
        }
    ),
    GateLifecycleState.DEPRECATED: frozenset(),
}


def is_transition_allowed(from_state: GateLifecycleState, to_state: GateLifecycleState) -> bool:
    """Return True iff ``from_state -> to_state`` is allowed by the
    lifecycle policy.

    Identity transitions (``X -> X``) are not allowed — the scanner only
    fires on actual changes, so an identity transition has no doc and
    should fail.

    N2 finding L1: a future addition to ``GateLifecycleState`` without a
    matching entry in ``_ALLOWED_TRANSITIONS`` would have raised KeyError
    on direct dict access. We use ``.get(..., frozenset())`` so the
    function safely returns False for unknown ``from_state`` values
    (no out-transitions allowed by default — fail-closed).
    """
    if from_state == to_state:
        return False
    return to_state in _ALLOWED_TRANSITIONS.get(from_state, frozenset())


class LifecycleDeclaration(BaseModel):
    """Pydantic v2 model validating a single lifecycle declaration.

    Used by:

    * YAML/TOML/JSON config parsers (top-level ``lifecycle_state`` key
      plus optional metadata in a sibling ``lifecycle_metadata`` block).
    * Python AST scanner (module-level ``LIFECYCLE_STATE_*`` constants
      plus optional companion ``LIFECYCLE_METADATA_*`` dict).
    * Lifecycle-change doc validator (extracts ``from_state`` /
      ``to_state`` / ``signing_reviewer`` / etc.).

    For ENFORCED transitions specifically, ``start_date``, ``end_date``,
    ``drift_summary``, and ``signing_reviewer`` MUST all be set per
    Gate N2 acceptance #3. See ``validate_enforced_transition``.
    """

    state: GateLifecycleState = Field(
        ..., description="Canonical lifecycle state. See GateLifecycleState."
    )
    gate_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable identifier (e.g., 'T2.2', 'T2.6a'). "
        "Used in lifecycle-change doc filenames.",
    )
    owner: Optional[str] = Field(
        default=None,
        description="Team or individual accountable for the gate. "
        "Required when state != DEVELOPMENT.",
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Start date of the current lifecycle window. "
        "Required for CALIBRATING + ENFORCED states.",
    )
    end_date: Optional[date] = Field(
        default=None,
        description="Planned end date of the current lifecycle window. "
        "Required for CALIBRATING + ENFORCED states.",
    )
    drift_summary: Optional[str] = Field(
        default=None,
        description="Summary of drift / calibration findings from the "
        "preceding window. Required for ENFORCED transitions.",
    )
    signing_reviewer: Optional[str] = Field(
        default=None,
        description="Name of the reviewer who signed off on the "
        "transition. Required for ENFORCED transitions.",
    )
    notes: Optional[str] = Field(default=None, description="Free-text operator notes.")

    @field_validator("gate_name")
    @classmethod
    def _validate_gate_name(cls, v: str) -> str:
        """Reject whitespace-only gate names (min_length=1 alone allows
        ``"   "``)."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("gate_name must not be whitespace-only")
        return v

    def is_complete_for_enforced_transition(self) -> bool:
        """Return True iff this declaration has all the metadata required
        for an ENFORCED lifecycle transition per Gate N2 acceptance #3.

        Required: ``start_date``, ``end_date``, ``drift_summary``,
        ``signing_reviewer``. ``owner`` and ``gate_name`` are validated
        by the model itself.
        """
        if self.state != GateLifecycleState.ENFORCED:
            return False
        return all(
            [
                self.start_date is not None,
                self.end_date is not None,
                self.drift_summary is not None and self.drift_summary.strip(),
                self.signing_reviewer is not None and self.signing_reviewer.strip(),
            ]
        )

    model_config = {
        "frozen": True,
        "use_enum_values": False,
    }


__all__ = [
    "GateLifecycleState",
    "LifecycleDeclaration",
    "is_transition_allowed",
]
