"""
E2I Resource Optimizer Agent - Integration Tests
"""

import pytest

from src.agents.resource_optimizer import (
    ResourceOptimizerAgent,
    ResourceOptimizerInput,
    ResourceOptimizerOutput,
    build_resource_optimizer_graph,
    optimize_allocation,
)


class TestResourceOptimizerAgent:
    """Integration tests for ResourceOptimizerAgent."""

    @pytest.mark.asyncio
    async def test_full_optimization_pipeline(self, sample_targets, budget_constraint):
        """Test complete optimization pipeline."""
        agent = ResourceOptimizerAgent()

        result = await agent.optimize(
            allocation_targets=sample_targets,
            constraints=budget_constraint,
            objective="maximize_outcome",
        )

        assert isinstance(result, ResourceOptimizerOutput)
        assert result.status == "completed"
        assert result.solver_status == "optimal"
        assert result.objective_value is not None
        assert len(result.optimal_allocations) == 4
        assert result.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_quick_optimize(self, sample_targets, budget_constraint):
        """Test quick optimization without scenarios."""
        agent = ResourceOptimizerAgent()

        result = await agent.quick_optimize(
            allocation_targets=sample_targets,
            constraints=budget_constraint,
        )

        assert result.status == "completed"
        assert result.scenarios is None

    @pytest.mark.asyncio
    async def test_optimization_with_scenarios(self, sample_targets, budget_constraint):
        """Test optimization with scenario analysis."""
        agent = ResourceOptimizerAgent()

        result = await agent.optimize(
            allocation_targets=sample_targets,
            constraints=budget_constraint,
            run_scenarios=True,
            scenario_count=3,
        )

        assert result.status == "completed"
        assert result.scenarios is not None
        assert len(result.scenarios) >= 2

    @pytest.mark.asyncio
    async def test_handoff_generation(self, sample_targets, budget_constraint):
        """Test handoff generation for orchestrator."""
        agent = ResourceOptimizerAgent()

        result = await agent.optimize(
            allocation_targets=sample_targets,
            constraints=budget_constraint,
        )

        handoff = agent.get_handoff(result)

        assert handoff["agent"] == "resource_optimizer"
        assert handoff["analysis_type"] == "resource_optimization"
        assert "key_findings" in handoff
        assert "allocations" in handoff
        assert handoff["allocations"]["increases"] > 0 or handoff["allocations"]["decreases"] > 0

    @pytest.mark.asyncio
    async def test_different_objectives(self, sample_targets, budget_constraint):
        """Test different optimization objectives."""
        agent = ResourceOptimizerAgent()

        for objective in ["maximize_outcome", "maximize_roi", "minimize_cost"]:
            result = await agent.optimize(
                allocation_targets=sample_targets,
                constraints=budget_constraint,
                objective=objective,
            )
            assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_different_solvers(self, sample_targets, budget_constraint):
        """Test different solver types."""
        agent = ResourceOptimizerAgent()

        for solver in ["linear", "nonlinear"]:
            result = await agent.optimize(
                allocation_targets=sample_targets,
                constraints=budget_constraint,
                solver_type=solver,
            )
            assert result.status == "completed"


class TestResourceOptimizerGraph:
    """Tests for LangGraph workflow."""

    @pytest.mark.asyncio
    async def test_full_graph_execution(self, base_state):
        """Test full graph execution."""
        graph = build_resource_optimizer_graph()
        result = await graph.ainvoke(base_state)

        assert result["status"] == "completed"
        assert result["optimal_allocations"] is not None
        assert result["projected_roi"] is not None

    @pytest.mark.asyncio
    async def test_error_handling_path(self):
        """Test error handling path in graph."""
        graph = build_resource_optimizer_graph()

        # No targets, no budget = should fail
        initial_state = {
            "query": "Optimize",
            "resource_type": "budget",
            "allocation_targets": [],
            "constraints": [],
            "objective": "maximize_outcome",
            "solver_type": "linear",
            "time_limit_seconds": 30,
            "gap_tolerance": 0.01,
            "run_scenarios": False,
            "scenario_count": 3,
            "optimal_allocations": None,
            "objective_value": None,
            "solver_status": None,
            "solve_time_ms": 0,
            "scenarios": None,
            "sensitivity_analysis": None,
            "projected_total_outcome": None,
            "projected_roi": None,
            "impact_by_segment": None,
            "optimization_summary": None,
            "recommendations": None,
            "timestamp": "",
            "formulation_latency_ms": 0,
            "optimization_latency_ms": 0,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        result = await graph.ainvoke(initial_state)
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_optimize_allocation_function(self, sample_targets, budget_constraint):
        """Test optimize_allocation convenience function."""
        result = await optimize_allocation(
            allocation_targets=sample_targets,
            constraints=budget_constraint,
        )

        assert isinstance(result, ResourceOptimizerOutput)
        assert result.status == "completed"


class TestInputOutputContracts:
    """Tests for Pydantic contracts."""

    def test_input_contract_defaults(self):
        """Test input contract default values."""
        input_data = ResourceOptimizerInput(
            allocation_targets=[],
            constraints=[],
        )

        assert input_data.resource_type == "budget"
        assert input_data.objective == "maximize_outcome"
        assert input_data.solver_type == "linear"
        assert input_data.run_scenarios is False

    def test_output_contract_defaults(self):
        """Test output contract default values."""
        output_data = ResourceOptimizerOutput()

        assert output_data.optimal_allocations == []
        assert output_data.status == "pending"

    def test_output_contract_serialization(self):
        """Test output contract JSON serialization."""
        output_data = ResourceOptimizerOutput(
            status="completed",
            objective_value=100000.0,
            optimization_summary="Test summary",
        )

        json_data = output_data.model_dump()

        assert json_data["status"] == "completed"
        assert json_data["objective_value"] == 100000.0


class TestHCPLevelOptimization:
    """Tests for HCP-level optimization."""

    @pytest.mark.asyncio
    async def test_hcp_optimization(self, hcp_targets, hcp_constraints):
        """Test HCP-level optimization."""
        agent = ResourceOptimizerAgent()

        result = await agent.optimize(
            allocation_targets=hcp_targets,
            constraints=hcp_constraints,
            resource_type="rep_time",
        )

        assert result.status == "completed"
        assert len(result.optimal_allocations) == 3

    @pytest.mark.asyncio
    async def test_hcp_impact_by_segment(self, hcp_targets, hcp_constraints):
        """Test impact by segment for HCPs."""
        agent = ResourceOptimizerAgent()

        result = await agent.optimize(
            allocation_targets=hcp_targets,
            constraints=hcp_constraints,
        )

        assert result.impact_by_segment is not None
        assert "hcp" in result.impact_by_segment


class TestLazyLoading:
    """Tests for lazy graph loading."""

    def test_graph_lazy_loading(self):
        """Test that graphs are lazily loaded."""
        agent = ResourceOptimizerAgent()

        # Graphs should not be built yet
        assert agent._full_graph is None
        assert agent._simple_graph is None

    def test_full_graph_builds_on_access(self):
        """Test full graph builds when accessed."""
        agent = ResourceOptimizerAgent()

        # Access the graph
        _ = agent.full_graph

        assert agent._full_graph is not None

    def test_simple_graph_builds_on_access(self):
        """Test simple graph builds when accessed."""
        agent = ResourceOptimizerAgent()

        # Access the graph
        _ = agent.simple_graph

        assert agent._simple_graph is not None
