"""P11 (MED) — interpretation must NOT mask a sensitivity-node failure.

The sensitivity node fail-closes by setting state["sensitivity_error"] and
status="failed" (with a defaulted E-value). The interpretation node previously
overwrote status to "completed" unconditionally — masking the failure. It must
now surface it (status stays "failed", needs_review=True).
"""

from __future__ import annotations

import pytest

from src.agents.causal_impact.nodes.interpretation import InterpretationNode


def _state(**overrides):
    state = {
        "query": "impact of t on y",
        "query_id": "q-fail",
        "treatment_var": "t",
        "outcome_var": "y",
        "confounders": ["region"],
        "data_source": "synthetic",
        "causal_graph": {
            "nodes": ["t", "y", "region"],
            "edges": [("region", "t"), ("t", "y")],
            "treatment_nodes": ["t"],
            "outcome_nodes": ["y"],
            "adjustment_sets": [["region"]],
            "dag_dot": "digraph { }",
            "confidence": 0.85,
        },
        "estimation_result": {
            "method": "ols",
            "ate": 0.5,
            "ate_ci_lower": 0.4,
            "ate_ci_upper": 0.6,
            "standard_error": 0.05,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
        },
        "refutation_results": {"tests_passed": 3, "total_tests": 4, "overall_robust": True},
        "sensitivity_analysis": {"e_value": 1.0, "robust_to_confounding": False},
        "interpretation_depth": "minimal",
        "user_context": {"expertise": "analyst"},
        "status": "failed",
        "errors": [],
        "warnings": [],
    }
    state.update(overrides)
    return state


class TestInterpretationSensitivityFailClosed:
    @pytest.mark.asyncio
    async def test_sensitivity_failure_is_not_masked_as_completed(self):
        node = InterpretationNode()
        result = await node.execute(_state(sensitivity_error="E-value computation failed"))
        assert result["status"] == "failed", "a sensitivity failure must NOT become 'completed'"
        assert result.get("needs_review") is True

    @pytest.mark.asyncio
    async def test_no_sensitivity_error_completes_normally(self):
        node = InterpretationNode()
        result = await node.execute(_state(status="in_progress"))
        assert result["status"] == "completed"
