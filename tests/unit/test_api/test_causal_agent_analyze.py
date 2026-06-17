"""Unit coverage for the causal_impact agent-analyze endpoint.

Focuses on the pure mapping (``_agent_state_to_response``): the agent's final
LangGraph state -> API response (DAG, effect, fail-closed status, honest
warnings) and the estimator-override allowlist. No DB, no agent run — the
end-to-end real-engine path is covered by a faithful integration check.
"""

import pytest

from src.api.schemas.causal import (
    AGENT_FORCEABLE_ESTIMATORS,
    AgentCausalAnalysisRequest,
)


def _req() -> AgentCausalAnalysisRequest:
    return AgentCausalAnalysisRequest(treatment_var="treatment_arm", outcome_var="persistent_180d")


def _base_state(**overrides):
    state = {
        "causal_graph": {
            "nodes": ["treatment_arm", "persistent_180d", "disease_severity"],
            "edges": [
                ("treatment_arm", "persistent_180d"),
                ("disease_severity", "persistent_180d"),
            ],
            "treatment_nodes": ["treatment_arm"],
            "outcome_nodes": ["persistent_180d"],
            "adjustment_sets": [["disease_severity"]],
            "dag_dot": "digraph {}",
        },
        "estimation_result": {
            "ate": 0.12,
            "ate_ci_lower": 0.05,
            "ate_ci_upper": 0.19,
            "standard_error": 0.03,
            "p_value": 0.001,
            "statistical_significance": True,
            "method": "CausalForestDML",
        },
        "refutation_results": {"gate_decision": "proceed", "tests_passed": 3, "total_tests": 3},
        "sensitivity_analysis": {"e_value": 1.8},
        "interpretation": {
            "narrative": "Treatment raises persistence.",
            "recommendations": ["Prioritize adherence support"],
            "key_findings": ["Effect is positive"],
        },
        "overall_confidence": 0.81,
        "warnings": [],
    }
    state.update(overrides)
    return state


@pytest.mark.unit
def test_completed_run_maps_dag_effect_and_estimator():
    from src.api.routes.causal import _agent_state_to_response

    resp = _agent_state_to_response(
        analysis_id="a1",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=_base_state(),
        latency_ms=42,
    )
    assert resp.status == "completed"
    assert resp.ate == 0.12
    assert (resp.ate_ci_lower, resp.ate_ci_upper) == (0.05, 0.19)
    assert resp.statistical_significance is True
    # The estimator the agent actually used is surfaced (data-driven or forced).
    assert resp.selected_estimator == "CausalForestDML"
    # DAG mapped faithfully (raw nodes/edges, not a summary string).
    assert resp.dag.treatment_nodes == ["treatment_arm"]
    assert resp.dag.outcome_nodes == ["persistent_180d"]
    assert ("treatment_arm", "persistent_180d") in resp.dag.edges
    assert resp.dag.adjustment_sets == [["disease_severity"]]
    # Robustness surfaced from the real gate.
    assert resp.refutation.gate_decision == "proceed"
    assert resp.refutation.passed is True
    assert resp.refutation.sensitivity_e_value == 1.8
    assert resp.narrative and resp.recommendations == ["Prioritize adherence support"]
    assert resp.data_source == "synthetic" and resp.n_rows == 120


@pytest.mark.unit
def test_review_band_is_needs_review_not_passed():
    from src.api.routes.causal import _agent_state_to_response

    state = _base_state(
        refutation_results={"gate_decision": "review", "tests_passed": 1, "total_tests": 3}
    )
    resp = _agent_state_to_response(
        analysis_id="a2",
        request=_req(),
        data_source="database",
        n_rows=80,
        final_state=state,
        latency_ms=10,
    )
    assert resp.status == "needs_review"
    assert resp.refutation.needs_review is True
    assert resp.refutation.passed is False


@pytest.mark.unit
def test_no_estimate_fails_closed_with_honest_warning():
    from src.api.routes.causal import _agent_state_to_response

    # An empty estimation_result => the agent could not estimate -> fail-closed,
    # NEVER a fabricated ATE.
    resp = _agent_state_to_response(
        analysis_id="a3",
        request=_req(),
        data_source="synthetic",
        n_rows=0,
        final_state=_base_state(estimation_result={}),
        latency_ms=10,
    )
    assert resp.status == "failed"
    assert resp.ate is None
    assert any("No treatment effect" in w for w in resp.warnings)


@pytest.mark.unit
def test_blocked_gate_fails_closed():
    from src.api.routes.causal import _agent_state_to_response

    state = _base_state(
        refutation_results={"gate_decision": "block", "tests_passed": 0, "total_tests": 3}
    )
    resp = _agent_state_to_response(
        analysis_id="a4",
        request=_req(),
        data_source="synthetic",
        n_rows=50,
        final_state=state,
        latency_ms=10,
    )
    assert resp.status == "failed"
    assert any("BLOCKED" in w for w in resp.warnings)


@pytest.mark.unit
def test_forceable_estimators_match_agent_allowlist():
    # The override values the API accepts MUST be a subset of the agent's
    # _VALID_EXPLICIT_METHODS, or a forced run would fail deep in the graph.
    import inspect

    from src.agents.causal_impact.nodes import estimation as est_mod

    src = inspect.getsource(est_mod)
    for name in AGENT_FORCEABLE_ESTIMATORS:
        assert f'"{name}"' in src, f"{name} not in estimation node's valid methods"
