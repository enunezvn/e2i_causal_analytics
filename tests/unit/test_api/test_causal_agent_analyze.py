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
    assert ["treatment_arm", "persistent_180d"] in resp.dag.edges
    assert resp.dag.adjustment_sets == [["disease_severity"]]
    # Robustness surfaced from the real gate.
    assert resp.refutation.gate_decision == "proceed"
    assert resp.refutation.passed is True
    assert resp.refutation.sensitivity_e_value == 1.8
    assert resp.narrative and resp.recommendations == ["Prioritize adherence support"]
    assert resp.data_source == "synthetic" and resp.n_rows == 120


@pytest.mark.unit
def test_naive_vs_adjusted_fields_pass_through():
    """The unadjusted diff-in-means foil + the confounding bias it removes must
    reach the API response so the page can show 'adjustment removed X bias'."""
    from src.api.routes.causal import _agent_state_to_response

    state = _base_state()
    state["estimation_result"].update(
        {
            "naive_ate": 0.2815,
            "naive_ate_ci_lower": 0.26,
            "naive_ate_ci_upper": 0.30,
            # naive - adjusted = how much the naive estimate was inflated.
            "confounding_bias_removed": 0.1615,
        }
    )
    resp = _agent_state_to_response(
        analysis_id="n1",
        request=_req(),
        data_source="synthetic",
        n_rows=25000,
        final_state=state,
        latency_ms=10,
    )
    assert resp.naive_ate == 0.2815
    assert (resp.naive_ate_ci_lower, resp.naive_ate_ci_upper) == (0.26, 0.30)
    assert resp.confounding_bias_removed == pytest.approx(0.1615)


@pytest.mark.unit
def test_naive_fields_default_none_when_estimator_did_not_emit_them():
    """A non-binary treatment (or an old result) carries no naive contrast — the
    response must surface None, never a fabricated 0."""
    from src.api.routes.causal import _agent_state_to_response

    resp = _agent_state_to_response(
        analysis_id="n2",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=_base_state(),
        latency_ms=10,
    )
    assert resp.naive_ate is None
    assert resp.naive_ate_ci_lower is None
    assert resp.naive_ate_ci_upper is None
    assert resp.confounding_bias_removed is None


@pytest.mark.unit
def test_estimator_comparison_surfaced_when_multiple_evaluated():
    """The Auto path fits + energy-scores several estimators; that comparison
    lived in state but was dropped at the API boundary. Surface it so the UI can
    explain WHY the winner won — not just show its name."""
    from src.api.routes.causal import _agent_state_to_response

    estimation = {
        "ate": 0.12,
        "statistical_significance": True,
        "method": "LinearDML",
        "selected_estimator": "LinearDML",
        "selection_reason": "confounding-robust preferred over OLS at comparable energy",
        "energy_score_gap": 0.04,
        "n_estimators_evaluated": 4,
        "n_estimators_succeeded": 3,
        "requires_review": False,
        "energy_score_data": {"quality_tier": "good"},
        "all_estimators_evaluated": [
            {
                "estimator": "CausalForestDML",
                "success": True,
                "energy_score": 0.51,
                "ate": 0.10,
                "error": None,
            },
            {
                "estimator": "LinearDML",
                "success": True,
                "energy_score": 0.48,
                "ate": 0.12,
                "error": None,
            },
            {
                "estimator": "DRLearner",
                "success": True,
                "energy_score": 0.55,
                "ate": 0.09,
                "error": None,
            },
            {
                "estimator": "OLS",
                "success": False,
                "energy_score": None,
                "ate": None,
                "error": "singular",
            },
        ],
    }
    resp = _agent_state_to_response(
        analysis_id="a1",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=_base_state(estimation_result=estimation),
        latency_ms=10,
    )
    cmp = resp.estimator_comparison
    assert cmp is not None
    assert cmp.n_evaluated == 4 and cmp.n_succeeded == 3
    assert len(cmp.candidates) == 4
    assert cmp.selection_reason and "robust" in cmp.selection_reason
    assert cmp.quality_tier == "good"
    # Exactly the winner is flagged.
    selected = [c for c in cmp.candidates if c.is_selected]
    assert len(selected) == 1 and selected[0].estimator == "LinearDML"


@pytest.mark.unit
def test_estimator_comparison_none_for_single_forced_estimator():
    """A forced/explicit method evaluates exactly one estimator — a 1-row
    'comparison' conveys nothing, so it collapses to None (verifier guard)."""
    from src.api.routes.causal import _agent_state_to_response

    estimation = {
        "ate": 0.12,
        "statistical_significance": True,
        "method": "LinearDML",
        "selected_estimator": "LinearDML",
        "n_estimators_evaluated": 1,
        "all_estimators_evaluated": [
            {
                "estimator": "LinearDML",
                "success": True,
                "energy_score": 0.48,
                "ate": 0.12,
                "error": None,
            },
        ],
    }
    resp = _agent_state_to_response(
        analysis_id="a1",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=_base_state(estimation_result=estimation),
        latency_ms=10,
    )
    assert resp.estimator_comparison is None


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


@pytest.mark.unit
def test_dag_source_discovered_surfaces_confounders():
    """When guided discovery ran and the gate ACCEPTED, the DAG is reported as
    'discovered' and the data-identified adjustment set is surfaced as
    discovered_confounders (so the FE can honestly say 'learned from data')."""
    from src.api.routes.causal import _agent_state_to_response

    state = _base_state()
    state["discovery_result"] = {"n_edges": 5}  # presence => discovery actually ran
    state["causal_graph"]["discovery_gate_decision"] = "accept"
    resp = _agent_state_to_response(
        analysis_id="d1",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=state,
        latency_ms=10,
    )
    assert resp.dag_source == "discovered"
    assert resp.discovered_confounders == ["disease_severity"]


@pytest.mark.unit
def test_dag_source_domain_knowledge_when_accept_overridden_by_fallback():
    """Discovery ran and the gate ACCEPTED, but the accepted DAG contradicted a
    curated confounder so graph_builder DISCARDED it and fell back to the manual
    domain DAG (``discovery_dag_overridden=True``). Provenance must report the
    HONEST source — 'domain_knowledge', NOT 'discovered' — so the FE never claims
    a human-curated DAG was learned from data, and no data-identified confounders
    are attributed to a structure the data never produced."""
    from src.api.routes.causal import _agent_state_to_response

    state = _base_state()
    state["discovery_result"] = {"n_edges": 5}  # discovery ran
    state["causal_graph"]["discovery_gate_decision"] = "accept"  # gate did accept
    state["causal_graph"]["discovery_dag_overridden"] = True  # but its DAG was discarded
    resp = _agent_state_to_response(
        analysis_id="o1",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=state,
        latency_ms=10,
    )
    assert resp.dag_source == "domain_knowledge"
    assert resp.discovered_confounders == []


@pytest.mark.unit
def test_dag_source_domain_knowledge_when_discovery_absent():
    """No discovery in the state -> the agent's domain DAG; no data-identified
    confounders are claimed (discovered_confounders stays empty)."""
    from src.api.routes.causal import _agent_state_to_response

    resp = _agent_state_to_response(
        analysis_id="m1",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=_base_state(),
        latency_ms=10,
    )
    assert resp.dag_source == "domain_knowledge"
    assert resp.discovered_confounders == []


# ---------------------------------------------------------------------------
# Per-test refutation details (the drill-down table needs more than pass/total)
# ---------------------------------------------------------------------------

_INDIVIDUAL_TESTS = {
    "placebo_treatment": {
        "test_name": "placebo_treatment",
        "passed": True,
        "original_effect": 0.12,
        "new_effect": 0.001,
        "p_value": 0.61,
        "details": "placebo effect ~0",
    },
    "random_common_cause": {
        "test_name": "random_common_cause",
        "passed": True,
        "original_effect": 0.12,
        "new_effect": 0.118,
        "p_value": 0.88,
        "details": "stable to a random common cause",
    },
    "data_subset": {
        "test_name": "data_subset",
        "passed": False,
        "original_effect": 0.12,
        "new_effect": 0.04,
        "p_value": 0.02,
        "details": "subset estimate drifted",
    },
}


@pytest.mark.unit
def test_refutation_individual_tests_surfaced_in_response():
    """The agent computes per-test refutation results; the response must carry
    them (regression: they were previously dropped, so the drill-down table
    showed the misleading 'enable refutation tests' empty-state)."""
    from src.api.routes.causal import _agent_state_to_response

    state = _base_state(
        refutation_results={
            "gate_decision": "proceed",
            "tests_passed": 2,
            "total_tests": 3,
            "individual_tests": _INDIVIDUAL_TESTS,
        }
    )
    resp = _agent_state_to_response(
        analysis_id="r1",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=state,
        latency_ms=5,
    )
    tests = resp.refutation.tests
    assert {t.test_name for t in tests} == {
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
    }
    by_name = {t.test_name: t for t in tests}
    assert by_name["placebo_treatment"].passed is True
    assert by_name["placebo_treatment"].original_effect == 0.12
    assert by_name["placebo_treatment"].new_effect == 0.001
    assert by_name["placebo_treatment"].p_value == 0.61
    assert by_name["data_subset"].passed is False


@pytest.mark.unit
def test_refutation_tests_empty_when_refutation_did_not_run():
    """No individual_tests -> empty list (the FE then shows the honest 'did not
    run' state, never a fabricated row)."""
    from src.api.routes.causal import _agent_state_to_response

    resp = _agent_state_to_response(
        analysis_id="r2",
        request=_req(),
        data_source="synthetic",
        n_rows=120,
        final_state=_base_state(refutation_results={}),
        latency_ms=5,
    )
    assert resp.refutation.tests == []


@pytest.mark.unit
def test_refutation_tests_helper_skips_malformed_and_coerces_floats():
    from src.api.routes.causal import _refutation_tests_from_state

    out = _refutation_tests_from_state(
        {
            "individual_tests": {
                # name falls back to the dict key when test_name is absent
                "bootstrap": {"passed": True, "p_value": "0.30", "original_effect": "0.1"},
                # non-dict entries are skipped, not crashed on
                "garbage": "not-a-dict",
            }
        }
    )
    assert len(out) == 1
    assert out[0].test_name == "bootstrap"
    assert out[0].p_value == 0.30  # coerced from str
    assert out[0].original_effect == 0.1
    # absent numeric -> None, not 0
    assert out[0].new_effect is None


@pytest.mark.unit
def test_refutation_sensitivity_test_surfaced_under_contract_key_not_raw_enum():
    """Regression: to_legacy_format keys the sensitivity test under
    'unobserved_common_cause' but sets its inner test_name to the raw enum
    'sensitivity_e_value'. We must surface the CONTRACT KEY so the FE labels it
    'Unobserved Common Cause' — using the inner value made the FE fall back to
    'Random Common Cause' and duplicate that row."""
    from src.api.routes.causal import _refutation_tests_from_state

    out = _refutation_tests_from_state(
        {
            "individual_tests": {
                "random_common_cause": {
                    "test_name": "random_common_cause",
                    "passed": True,
                    "p_value": 0.9,
                },
                # The divergent one: key != inner test_name.
                "unobserved_common_cause": {
                    "test_name": "sensitivity_e_value",
                    "passed": True,
                    "p_value": 0.0,
                },
            }
        }
    )
    names = {t.test_name for t in out}
    assert names == {"random_common_cause", "unobserved_common_cause"}
    # The sensitivity test is NOT surfaced under the raw enum name.
    assert "sensitivity_e_value" not in names


@pytest.mark.asyncio
async def test_agent_analyze_passes_expanded_geo_dummies_as_covariates():
    """run_causal_agent_analysis must hand _run_agent_analysis_task the EXPANDED
    covariate names (geo dummies), not the raw categorical 'geographic_region'."""
    from unittest.mock import AsyncMock, patch

    import pandas as pd

    from src.api.routes import causal as causal_routes

    frame = pd.DataFrame(
        {
            "treatment_arm": [1.0, 0.0],
            "persistent_180d": [1.0, 0.0],
            "disease_severity": [2.0, 1.0],
            "academic_hcp": [1.0, 0.0],
            "geographic_region=south": [1.0, 0.0],
            "geographic_region=west": [0.0, 1.0],
        }
    )
    expanded_cols = [
        "treatment_arm",
        "persistent_180d",
        "disease_severity",
        "academic_hcp",
        "geographic_region=south",
        "geographic_region=west",
    ]

    captured: dict = {}

    async def _fake_task(analysis_id, request, df, covariates, data_source):
        captured["covariates"] = covariates

    req = AgentCausalAnalysisRequest(
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
        covariates=["disease_severity", "academic_hcp", "geographic_region"],
        limit=1500,
    )

    # Capture the background task and run it explicitly (robust to event-loop
    # scheduling): BackgroundTasks.add_task(fn, *args) stores the call.
    scheduled: list = []

    class _BG:
        def add_task(self, fn, *args):
            scheduled.append((fn, args))

    with (
        patch.object(
            causal_routes,
            "_load_agent_estimation_frame",
            AsyncMock(return_value=(frame, expanded_cols)),
        ),
        patch.object(causal_routes, "_run_agent_analysis_task", _fake_task),
        patch.object(causal_routes._agent_analysis_store, "set", AsyncMock()),
    ):
        await causal_routes.run_causal_agent_analysis(req, _BG(), user={"sub": "t"})
        # Execute the scheduled background task to capture the covariates argument.
        for fn, args in scheduled:
            await fn(*args)

    assert "geographic_region" not in captured["covariates"]
    assert "geographic_region=south" in captured["covariates"]
    assert "geographic_region=west" in captured["covariates"]
    # Treatment/outcome are never covariates.
    assert "treatment_arm" not in captured["covariates"]
    assert "persistent_180d" not in captured["covariates"]


# ---------------------------------------------------------------------------
# #1188: opt-in RCT baseline (ANCOVA) adjustment — request flag + honest
# response labeling (efficiency vs confounding).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adjust_baselines_request_field_defaults_false():
    """Baseline adjustment is OPT-IN: the default request must keep today's
    unadjusted RCT behavior."""
    req = _req()
    assert req.adjust_baselines is False
    on = AgentCausalAnalysisRequest(
        treatment_var="control_group_flag",
        outcome_var="action_taken",
        dataset="nba_triggers",
        adjust_baselines=True,
    )
    assert on.adjust_baselines is True


@pytest.mark.unit
def test_adjustment_type_and_baselines_map_to_response():
    """An efficiency run must reach the client labeled as VARIANCE REDUCTION —
    never as confounding adjustment — with the baseline set it adjusted for."""
    from src.api.routes.causal import _agent_state_to_response

    state = _base_state()
    state["estimation_result"].update(
        {
            "adjustment_type": "efficiency",
            "baseline_covariates_adjusted": ["disease_severity", "age_at_diagnosis"],
        }
    )
    resp = _agent_state_to_response(
        analysis_id="e1",
        request=_req(),
        data_source="synthetic",
        n_rows=37541,
        final_state=state,
        latency_ms=10,
    )
    assert resp.adjustment_type == "efficiency"
    assert resp.baseline_covariates == ["disease_severity", "age_at_diagnosis"]


@pytest.mark.unit
def test_adjustment_type_defaults_none_for_legacy_states():
    """Old agent states carry no adjustment_type — the response must surface
    None (unknown), never fabricate a label."""
    from src.api.routes.causal import _agent_state_to_response

    resp = _agent_state_to_response(
        analysis_id="l1",
        request=_req(),
        data_source="synthetic",
        n_rows=100,
        final_state=_base_state(),
        latency_ms=5,
    )
    assert resp.adjustment_type is None
    assert resp.baseline_covariates == []


@pytest.mark.unit
def test_variables_response_carries_baseline_candidates_field():
    from src.api.schemas.causal import CausalVariablesResponse

    resp = CausalVariablesResponse(
        dataset="nba_triggers",
        treatment_candidates=["control_group_flag"],
        outcome_candidates=["action_taken"],
        covariate_candidates=[],
        baseline_candidates=["disease_severity", "age_at_diagnosis"],
        columns=["control_group_flag", "action_taken"],
        labels={},
    )
    assert resp.baseline_candidates == ["disease_severity", "age_at_diagnosis"]
