# tests/unit/test_agents/test_causal_impact/test_sensitivity_randomized.py
"""SensitivityNode must not narrate an unmeasured-confounding risk for a
genuinely randomized design (post-#1217 e-value RCT follow-up).

The E-value numbers stay computed and reported (informational), but the
user-visible interpretation/robustness fields must reflect that randomization
excludes confounding of assignment by construction — instead of telling the
user a randomized effect "could be explained by moderate unmeasured
confounding" (misleading, and the words that accompanied the live gate-block).
"""

import pytest

from src.agents.causal_impact.nodes.sensitivity import SensitivityNode
from src.agents.causal_impact.state import CausalImpactState, EstimationResult

# Weak standardized effect: e_value_ci < 1.5 → "weak" / robust=False on the
# observational path. Exactly the profile of the live nba_triggers RCT run.
_WEAK_ESTIMATION: EstimationResult = {
    "method": "linear_regression",
    "ate": 0.05,
    "ate_ci_lower": 0.04,
    "ate_ci_upper": 0.06,
    "effect_size": "small",
    "statistical_significance": True,
    "p_value": 0.01,
    "sample_size": 4000,
    "covariates_adjusted": [],
    "heterogeneity_detected": False,
}


def _state(**overrides) -> CausalImpactState:
    state: CausalImpactState = {
        "query": "rct question",
        "query_id": "test-rct",
        "estimation_result": dict(_WEAK_ESTIMATION),
        "status": "pending",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class TestRandomizedDesignSensitivity:
    @pytest.mark.asyncio
    async def test_randomized_design_interpretation_names_the_design(self):
        node = SensitivityNode()
        result = await node.execute(_state(randomized_design=True))
        sens = result["sensitivity_analysis"]
        assert "randomized" in sens["interpretation"].lower()
        # The informational number is still there, not erased.
        assert sens["e_value"] >= 1.0
        assert sens["e_value_ci"] >= 1.0

    @pytest.mark.asyncio
    async def test_randomized_design_is_robust_by_design(self):
        node = SensitivityNode()
        result = await node.execute(_state(randomized_design=True))
        sens = result["sensitivity_analysis"]
        assert sens["robust_to_confounding"] is True
        assert sens["unmeasured_confounder_strength"] == "not_applicable_randomized"

    @pytest.mark.asyncio
    async def test_observational_default_unchanged(self):
        """Guard: without the flag the weak-effect classification is untouched."""
        node = SensitivityNode()
        result = await node.execute(_state())
        sens = result["sensitivity_analysis"]
        assert sens["robust_to_confounding"] is False
        assert sens["unmeasured_confounder_strength"] == "weak"
        assert "randomized" not in sens["interpretation"].lower()
