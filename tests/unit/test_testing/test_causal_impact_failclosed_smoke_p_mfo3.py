"""M-fo3 — the tier1-5 smoke quality gate must accept an HONEST sensitivity
fail-closed verdict (status='failed' driven by a failed sensitivity node), not
only a refutation BLOCK. After M-fo3, _build_output emits status='failed' with
needs_review=True for a sensitivity-node failure; gate_decision is typically
'proceed' (refutation passed), so the pre-existing _validate_causal_impact would
wrongly reject it as 'not an honest fail-closed block'. These tests pin the
corrected gate semantics.
"""

from __future__ import annotations

from src.testing.agent_quality_gates import _validate_causal_impact


def _sensitivity_failure_output() -> dict:
    """A fully-executed analysis whose SENSITIVITY node failed (M-fo3)."""
    return {
        "status": "failed",
        "ate_estimate": 0.5,
        "confidence_interval": (0.3, 0.7),
        "gate_decision": "proceed",  # refutation PASSED; only sensitivity failed
        "refutation_passed": True,
        "needs_review": True,
        "refutation_tests_passed": 4,
        "refutation_tests_total": 4,
    }


class TestQualityGateAcceptsSensitivityFailure:
    def test_semantic_validator_accepts_sensitivity_failure(self):
        passed, msg = _validate_causal_impact(_sensitivity_failure_output())
        assert passed, msg

    def test_sensitivity_failure_without_ate_still_rejected(self):
        out = _sensitivity_failure_output()
        out["ate_estimate"] = None
        passed, _ = _validate_causal_impact(out)
        assert not passed

    def test_failed_without_block_or_needs_review_still_rejected(self):
        # A 'failed' with proceed gate but NO needs_review flag is still a real
        # failure (refutation errored path), and must keep failing the gate.
        out = _sensitivity_failure_output()
        out["needs_review"] = False
        out["gate_decision"] = "proceed"
        passed, _ = _validate_causal_impact(out)
        assert not passed
