"""
E2I Resource Optimizer Agent - Main Agent Class
Version: 4.2
Purpose: Resource allocation optimization for pharmaceutical operations
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .state import (
    AllocationResult,
    AllocationTarget,
    Constraint,
    ResourceOptimizerState,
    ScenarioResult,
)
from .graph import build_resource_optimizer_graph, build_simple_optimizer_graph

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT/OUTPUT CONTRACTS
# ============================================================================


class ResourceOptimizerInput(BaseModel):
    """Input contract for Resource Optimizer agent."""

    query: str = ""
    resource_type: str = "budget"  # budget, rep_time, samples, calls
    allocation_targets: List[AllocationTarget] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    objective: Literal[
        "maximize_outcome", "maximize_roi", "minimize_cost", "balance"
    ] = "maximize_outcome"
    solver_type: Literal["linear", "milp", "nonlinear"] = "linear"
    run_scenarios: bool = False
    scenario_count: int = 3


class ResourceOptimizerOutput(BaseModel):
    """Output contract for Resource Optimizer agent."""

    optimal_allocations: List[AllocationResult] = Field(default_factory=list)
    objective_value: Optional[float] = None
    solver_status: Optional[str] = None
    projected_total_outcome: Optional[float] = None
    projected_roi: Optional[float] = None
    impact_by_segment: Optional[Dict[str, float]] = None
    scenarios: Optional[List[ScenarioResult]] = None
    sensitivity_analysis: Optional[Dict[str, float]] = None
    optimization_summary: str = ""
    recommendations: List[str] = Field(default_factory=list)
    total_latency_ms: int = 0
    timestamp: str = ""
    status: str = "pending"
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# AGENT CLASS
# ============================================================================


class ResourceOptimizerAgent:
    """
    Tier 4 Resource Optimizer Agent.

    Responsibilities:
    - Optimize budget allocation across territories/HCPs
    - Handle rep time and sample optimization
    - Perform what-if scenario analysis
    - Project allocation impact
    """

    def __init__(self):
        """Initialize Resource Optimizer agent."""
        self._full_graph = None
        self._simple_graph = None

    @property
    def full_graph(self):
        """Lazy-load full optimization graph with scenarios."""
        if self._full_graph is None:
            self._full_graph = build_resource_optimizer_graph()
        return self._full_graph

    @property
    def simple_graph(self):
        """Lazy-load simple optimization graph without scenarios."""
        if self._simple_graph is None:
            self._simple_graph = build_simple_optimizer_graph()
        return self._simple_graph

    async def optimize(
        self,
        allocation_targets: List[AllocationTarget],
        constraints: List[Constraint],
        resource_type: str = "budget",
        objective: str = "maximize_outcome",
        solver_type: str = "linear",
        run_scenarios: bool = False,
        scenario_count: int = 3,
        query: str = "",
    ) -> ResourceOptimizerOutput:
        """
        Optimize resource allocation.

        Args:
            allocation_targets: Entities to allocate resources to
            constraints: Budget and other constraints
            resource_type: Type of resource (budget, rep_time, samples)
            objective: Optimization objective
            solver_type: Solver to use
            run_scenarios: Whether to run scenario analysis
            scenario_count: Number of scenarios to generate
            query: Original query text

        Returns:
            ResourceOptimizerOutput with optimal allocations
        """
        initial_state: ResourceOptimizerState = {
            "query": query,
            "resource_type": resource_type,
            "allocation_targets": allocation_targets,
            "constraints": constraints,
            "objective": objective,
            "solver_type": solver_type,
            "time_limit_seconds": 30,
            "gap_tolerance": 0.01,
            "run_scenarios": run_scenarios,
            "scenario_count": scenario_count,
            "_problem": None,
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

        # Choose graph based on scenario requirement
        graph = self.full_graph if run_scenarios else self.simple_graph

        logger.info(
            f"Starting resource optimization: {len(allocation_targets)} targets, "
            f"objective={objective}, solver={solver_type}"
        )

        result = await graph.ainvoke(initial_state)

        return ResourceOptimizerOutput(
            optimal_allocations=result.get("optimal_allocations") or [],
            objective_value=result.get("objective_value"),
            solver_status=result.get("solver_status"),
            projected_total_outcome=result.get("projected_total_outcome"),
            projected_roi=result.get("projected_roi"),
            impact_by_segment=result.get("impact_by_segment"),
            scenarios=result.get("scenarios"),
            sensitivity_analysis=result.get("sensitivity_analysis"),
            optimization_summary=result.get("optimization_summary", ""),
            recommendations=result.get("recommendations") or [],
            total_latency_ms=result.get("total_latency_ms", 0),
            timestamp=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
            status=result.get("status", "failed"),
            errors=result.get("errors") or [],
            warnings=result.get("warnings") or [],
        )

    async def quick_optimize(
        self,
        allocation_targets: List[AllocationTarget],
        constraints: List[Constraint],
        objective: str = "maximize_outcome",
    ) -> ResourceOptimizerOutput:
        """
        Quick optimization without scenario analysis.

        Args:
            allocation_targets: Entities to allocate resources to
            constraints: Budget and other constraints
            objective: Optimization objective

        Returns:
            ResourceOptimizerOutput with optimal allocations
        """
        return await self.optimize(
            allocation_targets=allocation_targets,
            constraints=constraints,
            objective=objective,
            run_scenarios=False,
        )

    def get_handoff(self, output: ResourceOptimizerOutput) -> Dict[str, Any]:
        """
        Generate handoff for orchestrator.

        Args:
            output: Optimization output

        Returns:
            Handoff dictionary for other agents
        """
        allocations = output.optimal_allocations or []
        increases = [a for a in allocations if a.get("change", 0) > 0]
        decreases = [a for a in allocations if a.get("change", 0) < 0]

        # Find top change
        top_change = None
        if allocations:
            top_change = max(allocations, key=lambda a: abs(a.get("change", 0)))

        recommendations = []
        if output.status == "failed":
            recommendations.append("Review constraints for feasibility")
            recommendations.append("Check allocation targets for validity")
        elif output.projected_roi and output.projected_roi < 1.0:
            recommendations.append(
                "ROI below 1.0 - consider revising allocation strategy"
            )

        recommendations.extend(output.recommendations[:2])

        return {
            "agent": "resource_optimizer",
            "analysis_type": "resource_optimization",
            "key_findings": {
                "objective_value": output.objective_value,
                "projected_outcome": output.projected_total_outcome,
                "projected_roi": output.projected_roi,
            },
            "allocations": {
                "increases": len(increases),
                "decreases": len(decreases),
                "top_change": top_change.get("entity_id") if top_change else None,
            },
            "recommendations": recommendations,
            "requires_further_analysis": output.status == "failed",
            "suggested_next_agent": "gap_analyzer" if output.status == "completed" else None,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def optimize_allocation(
    allocation_targets: List[AllocationTarget],
    constraints: List[Constraint],
    objective: str = "maximize_outcome",
    run_scenarios: bool = False,
) -> ResourceOptimizerOutput:
    """
    Convenience function for resource optimization.

    Args:
        allocation_targets: Entities to allocate resources to
        constraints: Budget and other constraints
        objective: Optimization objective
        run_scenarios: Whether to run scenario analysis

    Returns:
        ResourceOptimizerOutput
    """
    agent = ResourceOptimizerAgent()
    return await agent.optimize(
        allocation_targets=allocation_targets,
        constraints=constraints,
        objective=objective,
        run_scenarios=run_scenarios,
    )
