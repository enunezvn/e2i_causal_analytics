"""
E2I Resource Optimizer Agent - Scenario Analyzer Node Tests
"""

import pytest

from src.agents.resource_optimizer.nodes.scenario_analyzer import (
    ScenarioAnalyzerNode,
)


class TestScenarioAnalyzerNode:
    """Tests for ScenarioAnalyzerNode."""

    @pytest.mark.asyncio
    async def test_analyze_scenarios(self, optimized_state, sample_targets):
        """Test scenario generation."""
        optimized_state["run_scenarios"] = True
        optimized_state["allocation_targets"] = sample_targets
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        assert result["scenarios"] is not None
        assert len(result["scenarios"]) >= 2
        assert result["status"] == "projecting"

    @pytest.mark.asyncio
    async def test_analyze_baseline_scenario(
        self, optimized_state, sample_targets, budget_constraint
    ):
        """Test baseline scenario generation."""
        optimized_state["run_scenarios"] = True
        optimized_state["allocation_targets"] = sample_targets
        optimized_state["constraints"] = budget_constraint
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        baseline = next(
            (s for s in result["scenarios"] if "Baseline" in s["scenario_name"]),
            None,
        )
        assert baseline is not None
        assert baseline["total_allocation"] > 0

    @pytest.mark.asyncio
    async def test_analyze_optimized_scenario(
        self, optimized_state, sample_targets, budget_constraint
    ):
        """Test optimized scenario generation."""
        optimized_state["run_scenarios"] = True
        optimized_state["allocation_targets"] = sample_targets
        optimized_state["constraints"] = budget_constraint
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        optimized = next(
            (s for s in result["scenarios"] if "Optimized" in s["scenario_name"]),
            None,
        )
        assert optimized is not None
        assert optimized["projected_outcome"] > 0

    @pytest.mark.asyncio
    async def test_analyze_equal_distribution_scenario(
        self, optimized_state, sample_targets, budget_constraint
    ):
        """Test equal distribution scenario."""
        optimized_state["run_scenarios"] = True
        optimized_state["scenario_count"] = 3
        optimized_state["allocation_targets"] = sample_targets
        optimized_state["constraints"] = budget_constraint
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        equal = next(
            (s for s in result["scenarios"] if "Equal" in s["scenario_name"]),
            None,
        )
        assert equal is not None

    @pytest.mark.asyncio
    async def test_analyze_sensitivity(self, optimized_state, sample_targets):
        """Test sensitivity analysis."""
        optimized_state["run_scenarios"] = True
        optimized_state["allocation_targets"] = sample_targets
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        assert result["sensitivity_analysis"] is not None
        assert len(result["sensitivity_analysis"]) == len(sample_targets)

    @pytest.mark.asyncio
    async def test_sensitivity_is_marginal_at_optimum(self, optimized_state, sample_targets):
        """Sensitivity = d(outcome)/d(allocation) at the OPTIMIZED allocation.

        With a linear response (no response_model in the problem) the marginal
        is the raw coefficient; with a concave power curve it must be the
        curve's derivative at the optimized point — NOT the raw coefficient.
        """
        from src.agents.resource_optimizer.response_model import response_marginal

        optimized_state["run_scenarios"] = True
        optimized_state["allocation_targets"] = sample_targets
        gamma = 0.6
        optimized_state["_problem"] = {
            **(optimized_state.get("_problem") or {}),
            "response_model": {"type": "power", "gamma": gamma},
        }
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        sens = result["sensitivity_analysis"]
        opt_by_id = {
            a["entity_id"]: a["optimized_allocation"]
            for a in optimized_state["optimal_allocations"]
        }
        for t in sample_targets:
            tid = t["entity_id"] if isinstance(t, dict) else t.entity_id
            r = t["expected_response"] if isinstance(t, dict) else t.expected_response
            cur = t["current_allocation"] if isinstance(t, dict) else t.current_allocation
            x_opt = opt_by_id.get(tid, cur)
            assert sens[tid] == pytest.approx(response_marginal(r, cur, x_opt, gamma))
            if x_opt != cur:
                assert sens[tid] != pytest.approx(r)

    @pytest.mark.asyncio
    async def test_sensitivity_current_is_marginal_at_current_allocation(
        self, optimized_state, sample_targets
    ):
        """The paired 'current' series = d(outcome)/d(allocation) at the CURRENT
        allocation, so the UI can render the before->after equalization instead
        of a wall of identical optimized-marginal bars. With a concave curve it
        must differ from the optimized-allocation marginal wherever money moved.
        """
        from src.agents.resource_optimizer.response_model import response_marginal

        optimized_state["run_scenarios"] = True
        optimized_state["allocation_targets"] = sample_targets
        gamma = 0.6
        optimized_state["_problem"] = {
            **(optimized_state.get("_problem") or {}),
            "response_model": {"type": "power", "gamma": gamma},
        }
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        sens_cur = result["sensitivity_analysis_current"]
        sens_opt = result["sensitivity_analysis"]
        assert sens_cur is not None
        # Same entities in both series so the UI can pair them by key.
        assert set(sens_cur) == set(sens_opt)

        opt_by_id = {
            a["entity_id"]: a["optimized_allocation"]
            for a in optimized_state["optimal_allocations"]
        }
        for t in sample_targets:
            tid = t["entity_id"] if isinstance(t, dict) else t.entity_id
            r = t["expected_response"] if isinstance(t, dict) else t.expected_response
            cur = t["current_allocation"] if isinstance(t, dict) else t.current_allocation
            # 'current' marginal is evaluated AT the current allocation.
            assert sens_cur[tid] == pytest.approx(response_marginal(r, cur, cur, gamma))
            # ...and diverges from the optimized-allocation marginal where the
            # solver moved money (the dispersion the before->after view shows).
            if opt_by_id.get(tid, cur) != cur:
                assert sens_cur[tid] != pytest.approx(sens_opt[tid])

    @pytest.mark.asyncio
    async def test_analyze_skip_if_not_requested(self, optimized_state):
        """Test skipping scenario analysis when not requested."""
        optimized_state["run_scenarios"] = False
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        assert result["scenarios"] is None
        assert result["status"] == "projecting"

    @pytest.mark.asyncio
    async def test_analyze_already_failed_passthrough(self, optimized_state):
        """Test that already failed state passes through."""
        optimized_state["status"] = "failed"
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_analyze_scenario_count(self, optimized_state, sample_targets, budget_constraint):
        """Test scenario count limiting."""
        optimized_state["run_scenarios"] = True
        optimized_state["scenario_count"] = 2
        optimized_state["allocation_targets"] = sample_targets
        optimized_state["constraints"] = budget_constraint
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        assert len(result["scenarios"]) <= 2

    @pytest.mark.asyncio
    async def test_analyze_roi_calculation(
        self, optimized_state, sample_targets, budget_constraint
    ):
        """Test ROI calculation in scenarios."""
        optimized_state["run_scenarios"] = True
        optimized_state["allocation_targets"] = sample_targets
        optimized_state["constraints"] = budget_constraint
        node = ScenarioAnalyzerNode()
        result = await node.execute(optimized_state)

        for scenario in result["scenarios"]:
            if scenario["total_allocation"] > 0:
                expected_roi = scenario["projected_outcome"] / scenario["total_allocation"]
                assert scenario["roi"] == pytest.approx(expected_roi, rel=0.01)
