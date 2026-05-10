"""Tests for ``src.lifecycle.gate_lifecycle`` — Plan v4 Gate N2.

Covers acceptance criteria:

1. ``GateLifecycleState`` enum has all 5 values (DEVELOPMENT, ADVISORY,
   CALIBRATING, ENFORCED, DEPRECATED).
2. ``LifecycleDeclaration`` Pydantic model validates correct + rejects
   invalid input.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.lifecycle import GateLifecycleState, LifecycleDeclaration
from src.lifecycle.gate_lifecycle import is_transition_allowed


class TestGateLifecycleStateEnum:
    """Acceptance #1: enum has all 5 values with stable string codes."""

    def test_enum_has_five_canonical_states(self) -> None:
        states = list(GateLifecycleState)
        assert len(states) == 5
        names = {s.name for s in states}
        assert names == {
            "DEVELOPMENT",
            "ADVISORY",
            "CALIBRATING",
            "ENFORCED",
            "DEPRECATED",
        }

    @pytest.mark.parametrize(
        ("state", "expected_value"),
        [
            (GateLifecycleState.DEVELOPMENT, "development"),
            (GateLifecycleState.ADVISORY, "advisory"),
            (GateLifecycleState.CALIBRATING, "calibrating"),
            (GateLifecycleState.ENFORCED, "enforced"),
            (GateLifecycleState.DEPRECATED, "deprecated"),
        ],
    )
    def test_enum_value_is_lowercase_string(
        self, state: GateLifecycleState, expected_value: str
    ) -> None:
        """Subclasses ``str`` so YAML/JSON serialize idiomatically."""
        assert state.value == expected_value
        assert state == expected_value  # str equality

    def test_enum_round_trips_through_str(self) -> None:
        """``str -> Enum -> str`` is identity for every value."""
        for s in GateLifecycleState:
            assert GateLifecycleState(s.value) is s


class TestLifecycleDeclarationValidation:
    """Acceptance #2: model validates correct + rejects invalid input."""

    def test_minimal_valid_declaration(self) -> None:
        d = LifecycleDeclaration(state=GateLifecycleState.ADVISORY, gate_name="T2.2")
        assert d.state is GateLifecycleState.ADVISORY
        assert d.gate_name == "T2.2"
        # Optional fields default to None.
        assert d.owner is None
        assert d.start_date is None
        assert d.signing_reviewer is None

    def test_full_enforced_declaration(self) -> None:
        d = LifecycleDeclaration(
            state=GateLifecycleState.ENFORCED,
            gate_name="T2.6c",
            owner="ml-foundation",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 9, 1),
            drift_summary="0.5pp drift across 12 cohorts; within tolerance.",
            signing_reviewer="Erik Nunez",
            notes="Calibration window N=42 model_trainer runs.",
        )
        assert d.state is GateLifecycleState.ENFORCED
        assert d.is_complete_for_enforced_transition() is True

    def test_enforced_missing_required_field_marks_incomplete(self) -> None:
        # All fields except drift_summary set.
        d = LifecycleDeclaration(
            state=GateLifecycleState.ENFORCED,
            gate_name="T2.6c",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 9, 1),
            signing_reviewer="Erik Nunez",
        )
        assert d.is_complete_for_enforced_transition() is False

    def test_enforced_completeness_returns_false_for_non_enforced(self) -> None:
        d = LifecycleDeclaration(
            state=GateLifecycleState.ADVISORY,
            gate_name="T2.2",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 9, 1),
            drift_summary="x",
            signing_reviewer="y",
        )
        # All four fields present but state != ENFORCED -> False.
        assert d.is_complete_for_enforced_transition() is False

    def test_invalid_state_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LifecycleDeclaration(state="not_a_state", gate_name="T2.2")  # type: ignore[arg-type]

    def test_empty_gate_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LifecycleDeclaration(state=GateLifecycleState.ADVISORY, gate_name="")

    def test_whitespace_only_gate_name_rejected(self) -> None:
        # min_length=1 alone allows ``"   "`` — the validator rejects it.
        with pytest.raises(ValidationError):
            LifecycleDeclaration(state=GateLifecycleState.ADVISORY, gate_name="   ")

    def test_declaration_is_frozen(self) -> None:
        d = LifecycleDeclaration(state=GateLifecycleState.ADVISORY, gate_name="T2.2")
        with pytest.raises(ValidationError):
            d.gate_name = "T2.3"  # type: ignore[misc]

    def test_drift_summary_blank_string_marks_enforced_incomplete(self) -> None:
        d = LifecycleDeclaration(
            state=GateLifecycleState.ENFORCED,
            gate_name="T2.6c",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 9, 1),
            drift_summary="   ",  # whitespace-only
            signing_reviewer="Erik Nunez",
        )
        assert d.is_complete_for_enforced_transition() is False


class TestIsTransitionAllowed:
    """Lifecycle state-machine policy."""

    def test_development_can_advance_to_advisory(self) -> None:
        assert is_transition_allowed(GateLifecycleState.DEVELOPMENT, GateLifecycleState.ADVISORY)

    def test_development_cannot_jump_to_enforced(self) -> None:
        # Plan: must pass through CALIBRATING per Gate N2 acceptance #3.
        assert not is_transition_allowed(
            GateLifecycleState.DEVELOPMENT, GateLifecycleState.ENFORCED
        )

    def test_advisory_can_advance_to_calibrating(self) -> None:
        assert is_transition_allowed(GateLifecycleState.ADVISORY, GateLifecycleState.CALIBRATING)

    def test_calibrating_can_rollback_to_advisory(self) -> None:
        # If calibration fails, gate rolls back to advisory.
        assert is_transition_allowed(GateLifecycleState.CALIBRATING, GateLifecycleState.ADVISORY)

    def test_calibrating_can_advance_to_enforced(self) -> None:
        assert is_transition_allowed(GateLifecycleState.CALIBRATING, GateLifecycleState.ENFORCED)

    def test_enforced_can_rollback_to_calibrating(self) -> None:
        # Re-calibrate after observed drift.
        assert is_transition_allowed(GateLifecycleState.ENFORCED, GateLifecycleState.CALIBRATING)

    def test_enforced_cannot_jump_back_to_advisory(self) -> None:
        # Direct rollback from enforced to advisory not allowed (must
        # re-enter the calibration window first).
        assert not is_transition_allowed(GateLifecycleState.ENFORCED, GateLifecycleState.ADVISORY)

    def test_deprecated_is_terminal(self) -> None:
        # No outgoing transitions from deprecated.
        for to_state in GateLifecycleState:
            assert not is_transition_allowed(GateLifecycleState.DEPRECATED, to_state)

    def test_identity_transitions_disallowed(self) -> None:
        # Scanner only fires on actual changes; identity has no doc.
        for state in GateLifecycleState:
            assert not is_transition_allowed(state, state)

    def test_any_state_can_be_deprecated(self) -> None:
        for from_state in GateLifecycleState:
            if from_state is GateLifecycleState.DEPRECATED:
                continue
            assert is_transition_allowed(from_state, GateLifecycleState.DEPRECATED)

    def test_unknown_from_state_returns_false_not_keyerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """N2 finding L1: a future enum addition without a matching entry
        in ``_ALLOWED_TRANSITIONS`` must not raise KeyError. Simulate the
        gap by removing an existing key and confirming the function
        returns False (fail-closed) instead of raising.
        """
        from src.lifecycle import gate_lifecycle as _gl

        original = dict(_gl._ALLOWED_TRANSITIONS)
        truncated = {k: v for k, v in original.items() if k is not GateLifecycleState.ADVISORY}
        monkeypatch.setattr(_gl, "_ALLOWED_TRANSITIONS", truncated)
        # ADVISORY no longer has out-transitions in the table; must
        # return False, not KeyError.
        assert (
            is_transition_allowed(GateLifecycleState.ADVISORY, GateLifecycleState.CALIBRATING)
            is False
        )
