"""
E2I Resource Optimizer Agent - State Definitions
Version: 4.2
Purpose: LangGraph state for resource allocation optimization
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict
from uuid import UUID


class AllocationTarget(TypedDict, total=False):
    """Target entity for resource allocation."""

    # Required fields
    entity_id: str
    entity_type: str  # "hcp", "territory", "region"
    current_allocation: float
    expected_response: float  # Response coefficient

    # Bounds
    min_allocation: Optional[float]
    max_allocation: Optional[float]

    # MILP extensions (discrete allocation support)
    is_integer: bool  # Whether allocation must be integer (e.g., rep visits)
    is_binary: bool  # Whether to include/exclude entity (0/1 decision)
    allocation_unit: Optional[float]  # Discrete step size (e.g., 1000 for $1K increments)
    fixed_cost: Optional[float]  # Fixed cost if entity is selected (for binary)


class Constraint(TypedDict, total=False):
    """Optimization constraint."""

    # Required fields
    constraint_type: str  # "budget", "capacity", "min_coverage", "max_frequency", "cardinality"
    value: float

    # Optional fields
    scope: str  # "global", "regional", "entity"

    # MILP extensions
    min_entities: Optional[int]  # Minimum entities to select (cardinality)
    max_entities: Optional[int]  # Maximum entities to select (cardinality)
    entity_ids: Optional[List[str]]  # Specific entities this constraint applies to


class AllocationResult(TypedDict):
    """Optimized allocation result for an entity."""

    entity_id: str
    entity_type: str
    current_allocation: float
    optimized_allocation: float
    change: float
    change_percentage: float
    expected_impact: float


class ScenarioResult(TypedDict):
    """Result of a scenario analysis."""

    scenario_name: str
    total_allocation: float
    projected_outcome: float
    roi: float
    constraint_violations: List[str]


class ResourceOptimizerState(TypedDict):
    """Complete state for Resource Optimizer agent."""

    # === INPUT (NotRequired - provided by caller) ===
    query: NotRequired[str]
    resource_type: NotRequired[str]  # "budget", "rep_time", "samples", "calls"
    allocation_targets: NotRequired[List[AllocationTarget]]
    constraints: NotRequired[List[Constraint]]
    objective: NotRequired[Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"]]

    # === CONFIGURATION (NotRequired - has defaults) ===
    solver_type: NotRequired[Literal["linear", "milp", "nonlinear"]]
    time_limit_seconds: NotRequired[int]
    gap_tolerance: NotRequired[float]  # For MILP
    run_scenarios: NotRequired[bool]
    scenario_count: NotRequired[int]

    # === INTERNAL (problem formulation) ===
    _problem: NotRequired[Dict[str, Any] | None]

    # === OPTIMIZATION OUTPUTS (populated during execution) ===
    optimal_allocations: NotRequired[List[AllocationResult] | None]
    objective_value: NotRequired[float | None]
    solver_status: NotRequired[str | None]
    solve_time_ms: NotRequired[int]

    # === SCENARIO OUTPUTS (NotRequired - only if run_scenarios=True) ===
    scenarios: NotRequired[List[ScenarioResult] | None]
    sensitivity_analysis: NotRequired[Dict[str, float] | None]

    # === IMPACT OUTPUTS (NotRequired - computed after optimization) ===
    projected_total_outcome: NotRequired[float | None]
    projected_roi: NotRequired[float | None]
    impact_by_segment: NotRequired[Dict[str, float] | None]

    # === SUMMARY (populated during execution) ===
    optimization_summary: NotRequired[str | None]
    recommendations: NotRequired[List[str] | None]

    # === EXECUTION METADATA (NotRequired - populated during execution) ===
    timestamp: NotRequired[str]
    formulation_latency_ms: NotRequired[int]
    optimization_latency_ms: NotRequired[int]
    total_latency_ms: NotRequired[int]

    # === ERROR HANDLING (Required outputs) ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal[
        "pending",
        "formulating",
        "optimizing",
        "analyzing",
        "projecting",
        "completed",
        "failed",
    ]

    # === AUDIT CHAIN ===
    audit_workflow_id: NotRequired[UUID]
