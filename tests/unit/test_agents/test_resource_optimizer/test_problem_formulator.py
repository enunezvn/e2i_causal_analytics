"""
E2I Resource Optimizer Agent - Problem Formulator Node Tests
"""

import pytest

from src.agents.resource_optimizer.nodes.problem_formulator import (
    ProblemFormulatorNode,
)


class TestProblemFormulatorNode:
    """Tests for ProblemFormulatorNode."""

    @pytest.mark.asyncio
    async def test_formulate_valid_problem(self, base_state):
        """Test formulation with valid inputs."""
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        assert result["status"] == "optimizing"
        assert "_problem" in result
        assert result["formulation_latency_ms"] >= 0

        problem = result["_problem"]
        assert problem["n"] == 4
        assert len(problem["c"]) == 4
        assert len(problem["lb"]) == 4
        assert len(problem["ub"]) == 4

    @pytest.mark.asyncio
    async def test_formulate_objective_coefficients(self, base_state, sample_targets):
        """Test that objective coefficients match response coefficients."""
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        problem = result["_problem"]
        expected_responses = [t["expected_response"] for t in sample_targets]
        assert problem["c"] == expected_responses

    @pytest.mark.asyncio
    async def test_formulate_maximize_roi(self, base_state):
        """Test ROI maximization objective."""
        base_state["objective"] = "maximize_roi"
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        problem = result["_problem"]
        # ROI coefficients should be response / current_allocation
        assert len(problem["c"]) == 4
        assert problem["c"][0] == pytest.approx(3.0 / 50000.0)

    @pytest.mark.asyncio
    async def test_formulate_minimize_cost(self, base_state):
        """Test cost minimization objective."""
        base_state["objective"] = "minimize_cost"
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        problem = result["_problem"]
        # All coefficients should be -1
        assert all(c == -1.0 for c in problem["c"])

    @pytest.mark.asyncio
    async def test_formulate_no_targets_fails(self, base_state):
        """Test failure when no targets provided."""
        base_state["allocation_targets"] = []
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert any("No allocation targets" in e["error"] for e in result["errors"])

    @pytest.mark.asyncio
    async def test_formulate_no_budget_constraint_fails(self, base_state):
        """Test failure when no budget constraint."""
        base_state["constraints"] = []
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        assert result["status"] == "failed"
        assert any("budget constraint" in e["error"].lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_formulate_negative_response_fails(self, base_state, sample_targets):
        """Test failure with negative response coefficient."""
        sample_targets[0]["expected_response"] = -1.0
        base_state["allocation_targets"] = sample_targets
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        assert result["status"] == "failed"
        assert any("Negative response" in e["error"] for e in result["errors"])

    @pytest.mark.asyncio
    async def test_formulate_selects_linear_solver(self, base_state):
        """Test that linear solver is selected by default."""
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        assert result["solver_type"] == "linear"

    @pytest.mark.asyncio
    async def test_formulate_respects_requested_solver(self, base_state):
        """Test that requested solver type is respected."""
        base_state["solver_type"] = "nonlinear"
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        assert result["solver_type"] == "nonlinear"

    @pytest.mark.asyncio
    async def test_formulate_already_failed_passthrough(self, base_state):
        """Test that already failed state passes through."""
        base_state["status"] = "failed"
        base_state["errors"] = [{"error": "Previous error"}]
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_formulate_bounds_from_targets(self, base_state, sample_targets):
        """Test that bounds are extracted from targets."""
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        problem = result["_problem"]
        expected_lb = [t["min_allocation"] for t in sample_targets]
        expected_ub = [t["max_allocation"] for t in sample_targets]

        assert problem["lb"] == expected_lb
        assert problem["ub"] == expected_ub

    @pytest.mark.asyncio
    async def test_formulate_multiple_constraints(self, base_state, multiple_constraints):
        """Test formulation with multiple constraints."""
        base_state["constraints"] = multiple_constraints
        node = ProblemFormulatorNode()
        result = await node.execute(base_state)

        problem = result["_problem"]
        # Should have 2 inequality constraints
        assert len(problem["a_ub"]) == 2
        assert len(problem["b_ub"]) == 2
