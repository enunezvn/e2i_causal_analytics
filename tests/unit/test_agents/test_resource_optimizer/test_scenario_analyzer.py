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
    async def test_analyze_scenario_count(
        self, optimized_state, sample_targets, budget_constraint
    ):
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
                expected_roi = (
                    scenario["projected_outcome"] / scenario["total_allocation"]
                )
                assert scenario["roi"] == pytest.approx(expected_roi, rel=0.01)
