"""
E2I Resource Optimizer Agent - Test Fixtures
"""

import pytest
from typing import Any, Dict, List

from src.agents.resource_optimizer.state import (
    AllocationTarget,
    Constraint,
)


# ============================================================================
# ALLOCATION TARGET FIXTURES
# ============================================================================


@pytest.fixture
def sample_targets() -> List[AllocationTarget]:
    """Sample allocation targets for testing."""
    return [
        AllocationTarget(
            entity_id="territory_northeast",
            entity_type="territory",
            current_allocation=50000.0,
            min_allocation=10000.0,
            max_allocation=100000.0,
            expected_response=3.0,
        ),
        AllocationTarget(
            entity_id="territory_midwest",
            entity_type="territory",
            current_allocation=40000.0,
            min_allocation=10000.0,
            max_allocation=100000.0,
            expected_response=2.5,
        ),
        AllocationTarget(
            entity_id="territory_south",
            entity_type="territory",
            current_allocation=35000.0,
            min_allocation=10000.0,
            max_allocation=100000.0,
            expected_response=2.0,
        ),
        AllocationTarget(
            entity_id="territory_west",
            entity_type="territory",
            current_allocation=25000.0,
            min_allocation=10000.0,
            max_allocation=100000.0,
            expected_response=1.5,
        ),
    ]


@pytest.fixture
def hcp_targets() -> List[AllocationTarget]:
    """HCP-level allocation targets."""
    return [
        AllocationTarget(
            entity_id="hcp_001",
            entity_type="hcp",
            current_allocation=5000.0,
            min_allocation=1000.0,
            max_allocation=15000.0,
            expected_response=4.0,
        ),
        AllocationTarget(
            entity_id="hcp_002",
            entity_type="hcp",
            current_allocation=4000.0,
            min_allocation=1000.0,
            max_allocation=15000.0,
            expected_response=3.5,
        ),
        AllocationTarget(
            entity_id="hcp_003",
            entity_type="hcp",
            current_allocation=3000.0,
            min_allocation=1000.0,
            max_allocation=15000.0,
            expected_response=2.0,
        ),
    ]


# ============================================================================
# CONSTRAINT FIXTURES
# ============================================================================


@pytest.fixture
def budget_constraint() -> List[Constraint]:
    """Budget constraint for testing."""
    return [
        Constraint(
            constraint_type="budget",
            value=150000.0,
            scope="global",
        )
    ]


@pytest.fixture
def multiple_constraints() -> List[Constraint]:
    """Multiple constraints for testing."""
    return [
        Constraint(
            constraint_type="budget",
            value=150000.0,
            scope="global",
        ),
        Constraint(
            constraint_type="min_total",
            value=100000.0,
            scope="global",
        ),
    ]


@pytest.fixture
def hcp_constraints() -> List[Constraint]:
    """Constraints for HCP allocation."""
    return [
        Constraint(
            constraint_type="budget",
            value=15000.0,
            scope="global",
        )
    ]


# ============================================================================
# STATE FIXTURES
# ============================================================================


@pytest.fixture
def base_state(sample_targets, budget_constraint) -> Dict[str, Any]:
    """Base state for testing."""
    return {
        "query": "Optimize budget allocation across territories",
        "resource_type": "budget",
        "allocation_targets": sample_targets,
        "constraints": budget_constraint,
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


@pytest.fixture
def state_with_scenarios(sample_targets, budget_constraint) -> Dict[str, Any]:
    """State with scenarios enabled."""
    return {
        "query": "Optimize budget with scenario analysis",
        "resource_type": "budget",
        "allocation_targets": sample_targets,
        "constraints": budget_constraint,
        "objective": "maximize_outcome",
        "solver_type": "linear",
        "time_limit_seconds": 30,
        "gap_tolerance": 0.01,
        "run_scenarios": True,
        "scenario_count": 4,
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


@pytest.fixture
def formulated_state(base_state, sample_targets, budget_constraint) -> Dict[str, Any]:
    """State after problem formulation."""
    return {
        **base_state,
        "_problem": {
            "c": [3.0, 2.5, 2.0, 1.5],
            "lb": [10000.0, 10000.0, 10000.0, 10000.0],
            "ub": [100000.0, 100000.0, 100000.0, 100000.0],
            "a_ub": [[1.0, 1.0, 1.0, 1.0]],
            "b_ub": [150000.0],
            "a_eq": None,
            "b_eq": None,
            "n": 4,
            "targets": sample_targets,
            "objective": "maximize_outcome",
        },
        "formulation_latency_ms": 5,
        "status": "optimizing",
    }


@pytest.fixture
def optimized_state(formulated_state) -> Dict[str, Any]:
    """State after optimization."""
    return {
        **formulated_state,
        "optimal_allocations": [
            {
                "entity_id": "territory_northeast",
                "entity_type": "territory",
                "current_allocation": 50000.0,
                "optimized_allocation": 100000.0,
                "change": 50000.0,
                "change_percentage": 100.0,
                "expected_impact": 300000.0,
            },
            {
                "entity_id": "territory_midwest",
                "entity_type": "territory",
                "current_allocation": 40000.0,
                "optimized_allocation": 30000.0,
                "change": -10000.0,
                "change_percentage": -25.0,
                "expected_impact": 75000.0,
            },
            {
                "entity_id": "territory_south",
                "entity_type": "territory",
                "current_allocation": 35000.0,
                "optimized_allocation": 10000.0,
                "change": -25000.0,
                "change_percentage": -71.4,
                "expected_impact": 20000.0,
            },
            {
                "entity_id": "territory_west",
                "entity_type": "territory",
                "current_allocation": 25000.0,
                "optimized_allocation": 10000.0,
                "change": -15000.0,
                "change_percentage": -60.0,
                "expected_impact": 15000.0,
            },
        ],
        "objective_value": 410000.0,
        "solver_status": "optimal",
        "solve_time_ms": 10,
        "optimization_latency_ms": 15,
        "status": "projecting",
    }
