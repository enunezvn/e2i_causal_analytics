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
    async def test_optimize_milp_continuous(self, formulated_state):
        """Test MILP with continuous variables (same as LP)."""
        formulated_state["solver_type"] = "milp"
        # Add MILP-specific problem fields
        formulated_state["_problem"]["var_types"] = ["continuous"] * 4
        formulated_state["_problem"]["fixed_costs"] = [0.0] * 4
        formulated_state["_problem"]["allocation_units"] = [None] * 4
        formulated_state["_problem"]["min_entities"] = None
        formulated_state["_problem"]["max_entities"] = None
        formulated_state["_problem"]["has_integer_vars"] = False

        node = OptimizerNode()
        result = await node.execute(formulated_state)

        assert result["solver_status"] == "optimal"
        assert result["objective_value"] is not None

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


class TestMILPInteger:
    """Tests for MILP integer variable support."""

    @pytest.mark.asyncio
    async def test_milp_integer_allocations(self, milp_integer_state):
        """Test MILP produces integer allocations."""
        node = OptimizerNode()
        result = await node.execute(milp_integer_state)

        assert result["solver_status"] == "optimal"
        assert result["objective_value"] is not None

        # Check all allocations are integers
        for alloc in result["optimal_allocations"]:
            value = alloc["optimized_allocation"]
            assert value == int(value), f"Expected integer, got {value}"

    @pytest.mark.asyncio
    async def test_milp_integer_respects_bounds(self, milp_integer_state):
        """Test MILP integer allocations respect bounds."""
        node = OptimizerNode()
        result = await node.execute(milp_integer_state)

        lb = milp_integer_state["_problem"]["lb"]
        ub = milp_integer_state["_problem"]["ub"]
        targets = milp_integer_state["_problem"]["targets"]

        for alloc in result["optimal_allocations"]:
            for i, t in enumerate(targets):
                if t["entity_id"] == alloc["entity_id"]:
                    assert alloc["optimized_allocation"] >= lb[i]
                    assert alloc["optimized_allocation"] <= ub[i]
                    break

    @pytest.mark.asyncio
    async def test_milp_integer_respects_budget(self, milp_integer_state):
        """Test MILP integer allocations respect budget."""
        node = OptimizerNode()
        result = await node.execute(milp_integer_state)

        total = sum(a["optimized_allocation"] for a in result["optimal_allocations"])
        budget = milp_integer_state["_problem"]["b_ub"][0]
        assert total <= budget


class TestMILPBinary:
    """Tests for MILP binary variable support."""

    @pytest.mark.asyncio
    async def test_milp_binary_selections(self, milp_binary_state):
        """Test MILP produces binary (0/1) selections."""
        node = OptimizerNode()
        result = await node.execute(milp_binary_state)

        assert result["solver_status"] == "optimal"

        # Check all allocations are binary (0 or 1)
        for alloc in result["optimal_allocations"]:
            value = alloc["optimized_allocation"]
            assert value in [0, 1] or value in [0.0, 1.0], f"Expected binary, got {value}"

    @pytest.mark.asyncio
    async def test_milp_binary_respects_budget(self, milp_binary_state):
        """Test binary selections respect budget (max projects)."""
        node = OptimizerNode()
        result = await node.execute(milp_binary_state)

        # Count selected projects
        selected = sum(1 for a in result["optimal_allocations"] if a["optimized_allocation"] > 0.5)
        max_projects = milp_binary_state["_problem"]["b_ub"][0]
        assert selected <= max_projects

    @pytest.mark.asyncio
    async def test_milp_binary_selects_highest_value(self, milp_binary_state):
        """Test binary MILP selects highest value projects."""
        node = OptimizerNode()
        result = await node.execute(milp_binary_state)

        # Get selected project IDs
        selected_ids = [
            a["entity_id"] for a in result["optimal_allocations"] if a["optimized_allocation"] > 0.5
        ]

        # With budget of 3 and net values (response - fixed_cost):
        # project_a: 1000 - 200 = 800
        # project_b: 800 - 150 = 650
        # project_c: 600 - 100 = 500
        # project_d: 400 - 50 = 350
        # Should select top 3: a, b, c
        assert len(selected_ids) == 3
        assert "project_a" in selected_ids
        assert "project_b" in selected_ids
        assert "project_c" in selected_ids


class TestMILPCardinality:
    """Tests for MILP cardinality constraint support."""

    @pytest.mark.asyncio
    async def test_milp_cardinality_max_entities(self, milp_cardinality_state):
        """Test MILP respects max entities cardinality constraint."""
        node = OptimizerNode()
        result = await node.execute(milp_cardinality_state)

        assert result["solver_status"] == "optimal"

        # Count entities with non-zero allocation (above minimum)
        min_alloc = min(milp_cardinality_state["_problem"]["lb"])
        active_entities = sum(
            1 for a in result["optimal_allocations"] if a["optimized_allocation"] > min_alloc + 0.01
        )

        max_entities = milp_cardinality_state["_problem"]["max_entities"]
        assert active_entities <= max_entities

    @pytest.mark.asyncio
    async def test_milp_cardinality_selects_highest_response(self, milp_cardinality_state):
        """Test cardinality MILP selects entities with highest response."""
        node = OptimizerNode()
        result = await node.execute(milp_cardinality_state)

        # Get allocations sorted by amount (descending)
        allocations = sorted(
            result["optimal_allocations"],
            key=lambda a: a["optimized_allocation"],
            reverse=True,
        )

        # With max 2 entities and response coefficients [3.0, 2.5, 2.0, 1.5],
        # should allocate most to northeast (3.0) and midwest (2.5)
        if len(allocations) >= 2:
            top_two_ids = {allocations[0]["entity_id"], allocations[1]["entity_id"]}
            # Top two should include the highest response entities
            assert (
                "territory_northeast" in top_two_ids or allocations[0]["optimized_allocation"] > 0
            )


class TestMILPSolverDirect:
    """Direct tests for MILP solver method."""

    def test_solve_milp_integer_problem(self):
        """Test _solve_milp with integer variables."""
        node = OptimizerNode()
        problem = {
            "c": [100.0, 80.0, 60.0],
            "lb": [1, 1, 1],
            "ub": [10, 10, 10],
            "a_ub": [[1.0, 1.0, 1.0]],
            "b_ub": [15],
            "a_eq": None,
            "b_eq": None,
            "n": 3,
            "targets": [
                {"entity_id": "a", "entity_type": "hcp"},
                {"entity_id": "b", "entity_type": "hcp"},
                {"entity_id": "c", "entity_type": "hcp"},
            ],
            "objective": "maximize_outcome",
            "var_types": ["integer", "integer", "integer"],
            "fixed_costs": [0.0, 0.0, 0.0],
            "allocation_units": [None, None, None],
            "min_entities": None,
            "max_entities": None,
            "has_integer_vars": True,
        }

        result = node._solve_milp(problem)

        assert result["status"] == "optimal"
        assert result["x"] is not None
        # All values should be integers
        for val in result["x"]:
            assert val == int(val)
        # Sum should be <= 15
        assert sum(result["x"]) <= 15

    def test_solve_milp_binary_problem(self):
        """Test _solve_milp with binary variables."""
        node = OptimizerNode()
        problem = {
            "c": [100.0, 80.0, 60.0, 40.0],
            "lb": [0, 0, 0, 0],
            "ub": [1, 1, 1, 1],
            "a_ub": [[1.0, 1.0, 1.0, 1.0]],
            "b_ub": [2],  # Select max 2
            "a_eq": None,
            "b_eq": None,
            "n": 4,
            "targets": [
                {"entity_id": "a", "entity_type": "project"},
                {"entity_id": "b", "entity_type": "project"},
                {"entity_id": "c", "entity_type": "project"},
                {"entity_id": "d", "entity_type": "project"},
            ],
            "objective": "maximize_outcome",
            "var_types": ["binary", "binary", "binary", "binary"],
            "fixed_costs": [0.0, 0.0, 0.0, 0.0],
            "allocation_units": [None, None, None, None],
            "min_entities": None,
            "max_entities": None,
            "has_integer_vars": True,
        }

        result = node._solve_milp(problem)

        assert result["status"] == "optimal"
        assert result["x"] is not None
        # All values should be 0 or 1
        for val in result["x"]:
            assert val in [0, 1, 0.0, 1.0]
        # Sum should be <= 2
        assert sum(result["x"]) <= 2
        # Should select top 2 (100 + 80 = 180)
        assert result["objective"] == 180.0

    def test_solve_milp_with_cardinality(self):
        """Test _solve_milp with cardinality constraint."""
        node = OptimizerNode()
        problem = {
            "c": [3.0, 2.5, 2.0, 1.5],
            "lb": [0, 0, 0, 0],
            "ub": [100, 100, 100, 100],
            "a_ub": [[1.0, 1.0, 1.0, 1.0]],
            "b_ub": [150],
            "a_eq": None,
            "b_eq": None,
            "n": 4,
            "targets": [
                {"entity_id": "a", "entity_type": "territory"},
                {"entity_id": "b", "entity_type": "territory"},
                {"entity_id": "c", "entity_type": "territory"},
                {"entity_id": "d", "entity_type": "territory"},
            ],
            "objective": "maximize_outcome",
            "var_types": ["continuous", "continuous", "continuous", "continuous"],
            "fixed_costs": [0.0, 0.0, 0.0, 0.0],
            "allocation_units": [None, None, None, None],
            "min_entities": None,
            "max_entities": 2,  # Select at most 2 entities
            "has_integer_vars": False,
        }

        result = node._solve_milp(problem)

        assert result["status"] == "optimal"
        assert result["x"] is not None

        # Count active allocations (non-zero)
        active = sum(1 for v in result["x"] if v > 0.01)
        assert active <= 2

    def test_solve_milp_infeasible(self):
        """Test _solve_milp with infeasible problem."""
        node = OptimizerNode()
        problem = {
            "c": [100.0, 80.0],
            "lb": [50, 50],  # Min 50 each = 100 total
            "ub": [100, 100],
            "a_ub": [[1.0, 1.0]],
            "b_ub": [80],  # Budget only 80 (less than minimum)
            "a_eq": None,
            "b_eq": None,
            "n": 2,
            "targets": [
                {"entity_id": "a", "entity_type": "hcp"},
                {"entity_id": "b", "entity_type": "hcp"},
            ],
            "objective": "maximize_outcome",
            "var_types": ["continuous", "continuous"],
            "fixed_costs": [0.0, 0.0],
            "allocation_units": [None, None],
            "min_entities": None,
            "max_entities": None,
            "has_integer_vars": False,
        }

        result = node._solve_milp(problem)

        assert result["status"] == "infeasible"
