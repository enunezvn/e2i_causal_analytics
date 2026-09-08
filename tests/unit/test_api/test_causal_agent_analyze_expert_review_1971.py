"""#1971 -- the API surfaces an expert-review HALT honestly.

When the RefutationNode withholds an estimate on the expert-review gate
(``expert_review_halt=True`` in the final state -- a human rejected the DAG
structure, or ``CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL`` is on and a REVIEW-band
structure holds no approval), ``_agent_state_to_response`` must report
``status='failed'`` with the halt reason (review id + how to resolve) in
``warnings`` and the gate verdict in ``refutation.expert_review_decision`` /
``expert_review_id``. Before this change the mapping keyed only on the
statistical gate, so a halted PROCEED-band run would have read ``completed``
and a halted REVIEW-band run ``needs_review`` -- an estimate that looks usable.

``failed`` is deliberately reused rather than a fourth status: the frontend
poll loop (frontend/src/api/causal.ts runCausalAgentAnalysisAndWait) treats
only completed / needs_review / failed as terminal.
"""

from __future__ import annotations

import pytest

from src.api.schemas.causal import AgentCausalAnalysisRequest


def _req() -> AgentCausalAnalysisRequest:
    return AgentCausalAnalysisRequest(treatment_var="treatment_arm", outcome_var="persistent_180d")


def _state(**overrides):
    state = {
        "causal_graph": {
            "nodes": ["treatment_arm", "persistent_180d"],
            "edges": [("treatment_arm", "persistent_180d")],
            "treatment_nodes": ["treatment_arm"],
            "outcome_nodes": ["persistent_180d"],
            "adjustment_sets": [[]],
            "dag_dot": "digraph {}",
        },
        "estimation_result": {
            "ate": 0.12,
            "ate_ci_lower": 0.05,
            "ate_ci_upper": 0.19,
            "statistical_significance": True,
            "method": "CausalForestDML",
        },
        "refutation_results": {"gate_decision": "proceed", "tests_passed": 3, "total_tests": 3},
        "interpretation": {},
        "warnings": [],
    }
    state.update(overrides)
    return state


def _response(state):
    from src.api.routes.causal import _agent_state_to_response

    return _agent_state_to_response(
        analysis_id="a-1971",
        request=_req(),
        data_source="database",
        n_rows=100,
        final_state=state,
        latency_ms=10,
    )


@pytest.mark.unit
def test_rejected_structure_halt_is_failed_with_the_verdict_in_warnings():
    halt = (
        "Estimate withheld: the DAG structure was REJECTED by expert review by Dr. No "
        "(review rev-rejected): collider."
    )
    resp = _response(
        _state(
            expert_review_halt=True,
            expert_review_decision="rejected",
            expert_review_id="rev-rejected",
            error_message=halt,
            status="failed",
        )
    )
    assert resp.status == "failed"
    assert halt in resp.warnings
    assert resp.refutation is not None
    assert resp.refutation.expert_review_decision == "rejected"
    assert resp.refutation.expert_review_id == "rev-rejected"
    # The statistical gate is reported truthfully alongside the halt.
    assert resp.refutation.gate_decision == "proceed"
    assert resp.ate == pytest.approx(0.12)


@pytest.mark.unit
def test_approval_required_halt_on_review_band_is_failed_not_needs_review():
    halt = (
        "Estimate withheld: CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL is on ... "
        "resolve via POST /expert-reviews/rev-created/resolve"
    )
    resp = _response(
        _state(
            refutation_results={"gate_decision": "review", "tests_passed": 2, "total_tests": 3},
            expert_review_halt=True,
            expert_review_decision="pending_review",
            expert_review_id="rev-created",
            error_message=halt,
            status="failed",
        )
    )
    assert resp.status == "failed"
    assert any("POST /expert-reviews/rev-created/resolve" in w for w in resp.warnings)
    assert resp.refutation.expert_review_decision == "pending_review"
    assert resp.refutation.needs_review is True


@pytest.mark.unit
def test_same_states_without_the_halt_key_keep_todays_status():
    """Positive control: the halt key is the ONLY difference."""
    assert _response(_state()).status == "completed"
    assert (
        _response(
            _state(
                refutation_results={"gate_decision": "review", "tests_passed": 2, "total_tests": 3}
            )
        ).status
        == "needs_review"
    )
