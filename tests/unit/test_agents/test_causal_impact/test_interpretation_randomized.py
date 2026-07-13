# tests/unit/test_agents/test_causal_impact/test_interpretation_randomized.py
"""InterpretationNode assumptions/limitations must be design-aware
(post-#1217 e-value RCT follow-up).

Today the node UNCONDITIONALLY claims "Analysis based on observational data,
not randomized experiment" and lists "No unmeasured confounding (given
observed covariates)" as an ASSUMPTION — both false for the nba_triggers RCT,
where randomization makes no-confounding a design GUARANTEE, not an
assumption. Fail-closed: absent flag → the observational wording stays.
"""

import pytest

from src.agents.causal_impact.nodes.interpretation import InterpretationNode
from src.agents.causal_impact.state import (
    CausalGraph,
    CausalImpactState,
    EstimationResult,
    SensitivityAnalysis,
)


def _state(**overrides) -> CausalImpactState:
    causal_graph: CausalGraph = {
        "nodes": ["control_group_flag", "action_taken"],
        "edges": [("control_group_flag", "action_taken")],
        "treatment_nodes": ["control_group_flag"],
        "outcome_nodes": ["action_taken"],
        "adjustment_sets": [[]],
        "dag_dot": "digraph {}",
        "confidence": 0.9,
    }
    estimation_result: EstimationResult = {
        "method": "linear_regression",
        "ate": 0.08,
        "ate_ci_lower": 0.06,
        "ate_ci_upper": 0.10,
        "effect_size": "small",
        "statistical_significance": True,
        "p_value": 0.001,
        "sample_size": 4000,
        "covariates_adjusted": [],
        "heterogeneity_detected": False,
    }
    sensitivity_analysis: SensitivityAnalysis = {
        "e_value": 1.4,
        "e_value_ci": 1.3,
        "interpretation": "informational",
        "robust_to_confounding": True,
        "unmeasured_confounder_strength": "not_applicable_randomized",
    }
    state: CausalImpactState = {
        "query": "what is the effect of the NBA holdout on action rates?",
        "query_id": "test-rct-interp",
        "treatment_var": "control_group_flag",
        "outcome_var": "action_taken",
        "confounders": [],
        "data_source": "live",
        "causal_graph": causal_graph,
        "estimation_result": estimation_result,
        "refutation_results": {
            "tests_passed": 4,
            "tests_failed": 0,
            "total_tests": 4,
            "overall_robust": True,
            "individual_tests": {},
            "confidence_adjustment": 1.0,
        },
        "sensitivity_analysis": sensitivity_analysis,
        "interpretation_depth": "standard",
        "status": "pending",
        "errors": [],
        "warnings": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class TestRandomizedDesignInterpretation:
    @pytest.mark.asyncio
    async def test_randomized_design_drops_observational_limitation(self):
        node = InterpretationNode()
        result = await node.execute(_state(randomized_design=True))
        limitations = result["interpretation"]["limitations"]
        joined = " ".join(limitations).lower()
        assert "observational data" not in joined
        assert "randomized" in joined

    @pytest.mark.asyncio
    async def test_randomized_design_states_no_confounding_by_design(self):
        node = InterpretationNode()
        result = await node.execute(_state(randomized_design=True))
        assumptions = " ".join(result["interpretation"]["assumptions_made"]).lower()
        assert "randomiz" in assumptions

    @pytest.mark.asyncio
    async def test_observational_default_keeps_existing_wording(self):
        """Guard: without the flag the observational caveats stay verbatim."""
        node = InterpretationNode()
        result = await node.execute(_state())
        joined = " ".join(result["interpretation"]["limitations"]).lower()
        assert "observational data" in joined
