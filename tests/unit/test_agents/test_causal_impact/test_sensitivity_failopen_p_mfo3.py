"""M-fo3 (MED) — _build_output must fail-closed on a sensitivity-node failure.

The sensitivity node fail-closes by setting state['sensitivity_error'] and
status='failed' (sensitivity.py:98-106), and the interpretation node re-asserts
that (interpretation.py:92-101: status='failed', needs_review=True). But the prod
entrypoint CausalImpactAgent._build_output RECOMPUTES status/needs_review/confidence
from refutation+ATE signals only and never reads sensitivity_error, so a failed
sensitivity analysis with a passing refutation is silently surfaced as
status='completed', needs_review=False, confidence~=0.675. These tests pin the
fail-closed contract at the prod-output seam.
"""

from __future__ import annotations

import time

from src.agents.causal_impact.agent import CausalImpactAgent


def _base_state(**overrides):
    # Refutation PASSED (gate_decision='proceed', confidence_adjustment present);
    # ATE is valid and in-CI. The ONLY thing wrong is the sensitivity node.
    state = {
        "query_id": "q-sens",
        "estimation_result": {
            "ate": 0.5,
            "ate_ci_lower": 0.3,
            "ate_ci_upper": 0.7,
            "statistical_significance": True,
            "method": "linear_regression",
        },
        "refutation_results": {
            "confidence_adjustment": 0.9,
            "gate_decision": "proceed",
            "tests_passed": 4,
            "total_tests": 4,
        },
        "gate_decision": "proceed",
        "interpretation": {"narrative": "E-value of 1.00 suggests robustness."},
        "sensitivity_analysis": {},
        "causal_graph": {},
        "status": "failed",
    }
    state.update(overrides)
    return state


class TestBuildOutputSensitivityFailsClosed:
    def test_sensitivity_error_yields_failed_and_needs_review(self):
        agent = CausalImpactAgent()
        state = _base_state(sensitivity_error="E-value computation failed")
        out = agent._build_output(state, time.time())
        # A failed sensitivity node must NOT be surfaced as 'completed'.
        assert out["status"] == "failed", "sensitivity failure must not be 'completed'"
        assert out.get("needs_review") is True

    def test_sensitivity_error_penalizes_confidence(self):
        agent = CausalImpactAgent()
        state = _base_state(sensitivity_error="E-value computation failed")
        out = agent._build_output(state, time.time())
        # Confidence must be hard-capped at the unvalidated penalty (0.3),
        # never the pre-fix 0.675 (0.75 base * 0.9 refutation adjustment).
        assert out["confidence"] <= 0.3 + 1e-9, (
            f"sensitivity failure must penalize confidence, got {out['confidence']}"
        )
        assert out["requires_further_analysis"] is True

    def test_no_sensitivity_error_completes_normally(self):
        agent = CausalImpactAgent()
        # No sensitivity_error: a passing refutation + valid ATE must still complete.
        state = _base_state(sensitivity_analysis={"e_value": 2.5, "robust_to_confounding": True})
        out = agent._build_output(state, time.time())
        assert out["status"] == "completed"
        assert out.get("needs_review") in (False, None)
        assert out["confidence"] > 0.3
