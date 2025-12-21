"""
E2I Resource Optimizer Agent - Optimizer Node Tests
"""

import pytest

from src.agents.resource_optimizer.nodes.optimizer import OptimizerNode


class TestOptimizerNode:
    """Tests for OptimizerNode."""

    @pytest.mark.asyncio
    async def test_optimize_linear(self, formulated_state):
        """Test linear optimization."""
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        assert result["solver_status"] == "optimal"
        assert result["objective_value"] is not None
        assert result["objective_value"] > 0
        assert len(result["optimal_allocations"]) == 4
        assert result["optimization_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_optimize_allocations_respect_budget(self, formulated_state):
        """Test that allocations respect budget constraint."""
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        total_allocation = sum(a["optimized_allocation"] for a in result["optimal_allocations"])
        budget = formulated_state["_problem"]["b_ub"][0]

        assert total_allocation <= budget + 0.01  # Allow small tolerance

    @pytest.mark.asyncio
    async def test_optimize_allocations_respect_bounds(self, formulated_state):
        """Test that allocations respect variable bounds."""
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        lb = formulated_state["_problem"]["lb"]
        ub = formulated_state["_problem"]["ub"]

        for _i, alloc in enumerate(result["optimal_allocations"]):
            # Find matching target by entity_id
            target_idx = None
            for j, t in enumerate(formulated_state["_problem"]["targets"]):
                if t["entity_id"] == alloc["entity_id"]:
                    target_idx = j
                    break
            if target_idx is not None:
                assert alloc["optimized_allocation"] >= lb[target_idx] - 0.01
                assert alloc["optimized_allocation"] <= ub[target_idx] + 0.01

    @pytest.mark.asyncio
    async def test_optimize_change_calculation(self, formulated_state):
        """Test that change is calculated correctly."""
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        for alloc in result["optimal_allocations"]:
            expected_change = alloc["optimized_allocation"] - alloc["current_allocation"]
            assert alloc["change"] == pytest.approx(expected_change, abs=0.01)

    @pytest.mark.asyncio
    async def test_optimize_nonlinear(self, formulated_state):
        """Test nonlinear optimization."""
        formulated_state["solver_type"] = "nonlinear"
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        assert result["solver_status"] == "optimal"
        assert result["objective_value"] is not None

    @pytest.mark.asyncio
    async def test_optimize_milp_fallback(self, formulated_state):
        """Test MILP falls back to linear."""
        formulated_state["solver_type"] = "milp"
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        assert result["solver_status"] == "optimal"

    @pytest.mark.asyncio
    async def test_optimize_no_problem_fails(self, base_state):
        """Test failure when no problem formulated."""
        base_state["status"] = "optimizing"
        node = OptimizerNode()
        result = await node.execute(base_state)

        assert result["status"] == "failed"
        assert any("No problem formulated" in e["error"] for e in result["errors"])

    @pytest.mark.asyncio
    async def test_optimize_already_failed_passthrough(self, formulated_state):
        """Test that already failed state passes through."""
        formulated_state["status"] = "failed"
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_optimize_status_transitions(self, formulated_state):
        """Test status transition after optimization."""
        formulated_state["run_scenarios"] = False
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        assert result["status"] == "projecting"

    @pytest.mark.asyncio
    async def test_optimize_status_with_scenarios(self, formulated_state):
        """Test status transition with scenarios enabled."""
        formulated_state["run_scenarios"] = True
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        assert result["status"] == "analyzing"

    @pytest.mark.asyncio
    async def test_optimize_expected_impact(self, formulated_state):
        """Test expected impact calculation."""
        node = OptimizerNode()
        result = await node.execute(formulated_state)

        for alloc in result["optimal_allocations"]:
            # Find response coefficient for this entity
            for i, t in enumerate(formulated_state["_problem"]["targets"]):
                if t["entity_id"] == alloc["entity_id"]:
                    c = formulated_state["_problem"]["c"][i]
                    expected_impact = c * alloc["optimized_allocation"]
                    assert alloc["expected_impact"] == pytest.approx(expected_impact, rel=0.01)
                    break


class TestOptimizerProportional:
    """Tests for proportional fallback solver."""

    @pytest.mark.asyncio
    async def test_proportional_allocation(self, formulated_state):
        """Test proportional allocation when scipy unavailable."""
        node = OptimizerNode()
        # Use proportional solver directly
        result = node._solve_proportional(formulated_state["_problem"])

        assert result["status"] == "optimal"
        assert result["x"] is not None
        assert len(result["x"]) == 4

    @pytest.mark.asyncio
    async def test_proportional_respects_budget(self, formulated_state):
        """Test proportional solver respects budget."""
        node = OptimizerNode()
        result = node._solve_proportional(formulated_state["_problem"])

        total = sum(result["x"])
        budget = formulated_state["_problem"]["b_ub"][0]
        assert total <= budget + 0.01
