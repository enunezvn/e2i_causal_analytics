"""Unit tests for the handoff_protocols validators.

Phase 1 of data-sufficiency rollout: confirms the latent
``minimum_samples > 0`` rule (declared in
``SCOPE_TO_DATA_PROTOCOL.validation_rules`` but never enforced before)
is now actually checked by ``validate_scope_to_data_handoff``.
"""

from __future__ import annotations

from src.agents.tier_0.handoff_protocols import (
    validate_scope_to_data_handoff,
)


def _ok_scope(**overrides) -> dict:
    """Return a minimum-viable ScopeDefiner → DataPreparer handoff payload."""
    scope = {
        "experiment_id": "exp-123",
        "problem_type": "binary_classification",
        "minimum_samples": 500,
    }
    scope.update(overrides.pop("scope_spec_extras", {}))
    return {
        "experiment_id": "exp-123",
        "scope_spec": scope,
        **overrides,
    }


class TestScopeToDataHandoff:
    def test_valid_handoff_passes(self):
        is_valid, errors = validate_scope_to_data_handoff(_ok_scope())
        assert is_valid
        assert errors == []

    def test_minimum_samples_zero_rejected(self):
        is_valid, errors = validate_scope_to_data_handoff(
            _ok_scope(scope_spec_extras={"minimum_samples": 0})
        )
        assert not is_valid
        assert any("minimum_samples must be > 0" in e for e in errors)

    def test_minimum_samples_negative_rejected(self):
        is_valid, errors = validate_scope_to_data_handoff(
            _ok_scope(scope_spec_extras={"minimum_samples": -50})
        )
        assert not is_valid
        assert any("minimum_samples must be > 0" in e for e in errors)

    def test_minimum_samples_non_int_rejected(self):
        is_valid, errors = validate_scope_to_data_handoff(
            _ok_scope(scope_spec_extras={"minimum_samples": "five hundred"})
        )
        assert not is_valid
        assert any("minimum_samples" in e for e in errors)

    def test_missing_minimum_samples_does_not_block(self):
        # Backward-compat: if the producer omits minimum_samples entirely the
        # handoff still passes (the field is advisory, not required). Only an
        # affirmatively bad value (<=0 or wrong type) is rejected.
        payload = _ok_scope()
        del payload["scope_spec"]["minimum_samples"]
        is_valid, errors = validate_scope_to_data_handoff(payload)
        assert is_valid
        assert errors == []

    def test_experiment_id_mismatch_still_rejected(self):
        # Sanity check that the new rule doesn't mask existing rules.
        payload = _ok_scope()
        payload["scope_spec"]["experiment_id"] = "different-exp"
        is_valid, errors = validate_scope_to_data_handoff(payload)
        assert not is_valid
        assert any("experiment_id must match" in e for e in errors)

    def test_invalid_problem_type_still_rejected(self):
        payload = _ok_scope(scope_spec_extras={"problem_type": "made_up_problem"})
        is_valid, errors = validate_scope_to_data_handoff(payload)
        assert not is_valid
        assert any("problem_type" in e for e in errors)
