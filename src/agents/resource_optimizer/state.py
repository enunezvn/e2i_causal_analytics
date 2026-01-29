"""
E2I Resource Optimizer Agent - State Definitions
Version: 4.2
Purpose: LangGraph state for resource allocation optimization
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
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

    # === INPUT ===
    query: str
    resource_type: str  # "budget", "rep_time", "samples", "calls"
    allocation_targets: List[AllocationTarget]
    constraints: List[Constraint]
    objective: Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"]

    # === CONFIGURATION ===
    solver_type: Literal["linear", "milp", "nonlinear"]
    time_limit_seconds: int
    gap_tolerance: float  # For MILP
    run_scenarios: bool
    scenario_count: int

    # === INTERNAL (problem formulation) ===
    _problem: Optional[Dict[str, Any]]

    # === OPTIMIZATION OUTPUTS ===
    # Note: Required outputs from optimization node
    optimal_allocations: List[AllocationResult]
    objective_value: float
    solver_status: str
    solve_time_ms: int

    # === SCENARIO OUTPUTS ===
    scenarios: Optional[List[ScenarioResult]]
    sensitivity_analysis: Optional[Dict[str, float]]

    # === IMPACT OUTPUTS ===
    projected_total_outcome: Optional[float]
    projected_roi: Optional[float]
    impact_by_segment: Optional[Dict[str, float]]

    # === SUMMARY ===
    # Note: Required output from summary generation
    optimization_summary: str
    recommendations: Optional[List[str]]

    # === EXECUTION METADATA ===
    timestamp: str
    formulation_latency_ms: int
    optimization_latency_ms: int
    total_latency_ms: int

    # === ERROR HANDLING ===
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
    audit_workflow_id: Optional[UUID]
