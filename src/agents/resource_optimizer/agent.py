"""
E2I Resource Optimizer Agent - Main Agent Class
Version: 4.3
Purpose: Resource allocation optimization for pharmaceutical operations
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .memory_hooks import ResourceOptimizerMemoryHooks
    from .opik_tracer import ResourceOptimizerOpikTracer

from .graph import build_resource_optimizer_graph, build_simple_optimizer_graph
from .state import (
    AllocationResult,
    AllocationTarget,
    Constraint,
    ResourceOptimizerState,
    ScenarioResult,
)

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
    objective: Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"] = (
        "maximize_outcome"
    )
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

    def __init__(self, enable_opik: bool = True, enable_memory: bool = True):
        """
        Initialize Resource Optimizer agent.

        Args:
            enable_opik: Whether to enable Opik distributed tracing (default: True)
            enable_memory: Whether to enable memory integration (default: True)
        """
        self._full_graph = None
        self._simple_graph = None
        self.enable_opik = enable_opik
        self.enable_memory = enable_memory
        self._opik_tracer: Optional["ResourceOptimizerOpikTracer"] = None
        self._memory_hooks: Optional["ResourceOptimizerMemoryHooks"] = None

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

    def _get_opik_tracer(self) -> Optional["ResourceOptimizerOpikTracer"]:
        """Get or create Opik tracer instance (lazy initialization)."""
        if not self.enable_opik:
            return None

        if self._opik_tracer is None:
            try:
                from .opik_tracer import get_resource_optimizer_tracer

                self._opik_tracer = get_resource_optimizer_tracer()
            except ImportError:
                logger.warning("Opik tracer not available")
                return None

        return self._opik_tracer

    @property
    def memory_hooks(self) -> Optional["ResourceOptimizerMemoryHooks"]:
        """Lazy-load memory hooks."""
        if self._memory_hooks is None and self.enable_memory:
            try:
                from .memory_hooks import get_resource_optimizer_memory_hooks

                self._memory_hooks = get_resource_optimizer_memory_hooks()
            except ImportError:
                logger.warning("Memory hooks not available")
                return None
        return self._memory_hooks

    async def optimize(
        self,
        allocation_targets: List[AllocationTarget],
        constraints: List[Constraint],
        resource_type: str = "budget",
        objective: Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"] = "maximize_outcome",
        solver_type: Literal["linear", "milp", "nonlinear"] = "linear",
        run_scenarios: bool = False,
        scenario_count: int = 3,
        query: str = "",
        session_id: Optional[str] = None,
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
            session_id: Optional session identifier for memory context

        Returns:
            ResourceOptimizerOutput with optimal allocations
        """
        start_time = time.time()

        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Retrieve memory context if enabled
        memory_context = None
        if self.enable_memory and self.memory_hooks:
            try:
                memory_context = await self.memory_hooks.get_context(
                    session_id=session_id,
                    resource_type=resource_type,
                    objective=objective,
                    constraints=[dict(c) for c in constraints] if constraints else None,  # type: ignore[arg-type]
                )
                logger.debug(
                    f"Retrieved memory context: "
                    f"cached={memory_context.cached_optimization is not None}, "
                    f"similar={len(memory_context.similar_optimizations)}, "
                    f"patterns={len(memory_context.learned_patterns)}"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve memory context: {e}")

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

        # Get Opik tracer
        opik_tracer = self._get_opik_tracer()

        async def execute_and_build_output() -> tuple[ResourceOptimizerOutput, Dict[str, Any]]:
            """Execute workflow and build output."""
            result = await graph.ainvoke(initial_state)

            allocations = result.get("optimal_allocations") or []
            output = ResourceOptimizerOutput(
                optimal_allocations=allocations,
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
            return output, result

        async def contribute_memory(
            output: ResourceOptimizerOutput, final_state: Dict[str, Any]
        ) -> None:
            """Contribute optimization results to memory systems."""
            if self.enable_memory and self.memory_hooks and output.status != "failed":
                try:
                    from .memory_hooks import contribute_to_memory

                    await contribute_to_memory(
                        result=output.model_dump(),
                        state=final_state,
                        memory_hooks=self.memory_hooks,
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to contribute to memory: {e}")

        if opik_tracer:
            async with opik_tracer.trace_optimization(
                resource_type=resource_type,
                objective=objective,
                solver_type=solver_type,
                query=query,
            ) as trace_ctx:
                trace_ctx.log_optimization_started(
                    resource_type=resource_type,
                    objective=objective,
                    solver_type=solver_type,
                    target_count=len(allocation_targets),
                    constraint_count=len(constraints),
                    run_scenarios=run_scenarios,
                )

                output, final_state = await execute_and_build_output()

                # Contribute to memory
                await contribute_memory(output, final_state)

                # Count allocation changes
                allocations = output.optimal_allocations or []
                increases = len([a for a in allocations if a.get("change", 0) > 0])
                decreases = len([a for a in allocations if a.get("change", 0) < 0])

                # Log completion
                elapsed_ms = int((time.time() - start_time) * 1000)
                trace_ctx.log_optimization_complete(
                    status=output.status,
                    success=output.status == "completed",
                    total_duration_ms=output.total_latency_ms,
                    objective_value=output.objective_value,
                    solver_status=output.solver_status,
                    projected_outcome=output.projected_total_outcome,
                    projected_roi=output.projected_roi,
                    allocations_count=len(allocations),
                    increases_count=increases,
                    decreases_count=decreases,
                    recommendations=output.recommendations,
                    errors=output.errors,
                    warnings=output.warnings,
                )

                logger.info(
                    f"Optimization complete: status={output.status}, "
                    f"allocations={len(allocations)}, latency={elapsed_ms}ms"
                )

                return output
        else:
            # Execute without Opik tracing
            output, final_state = await execute_and_build_output()

            # Contribute to memory
            await contribute_memory(output, final_state)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Optimization complete: status={output.status}, "
                f"allocations={len(output.optimal_allocations or [])}, latency={elapsed_ms}ms"
            )

            return output

    async def quick_optimize(
        self,
        allocation_targets: List[AllocationTarget],
        constraints: List[Constraint],
        objective: Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"] = "maximize_outcome",
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
            recommendations.append("ROI below 1.0 - consider revising allocation strategy")

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
    objective: Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"] = "maximize_outcome",
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
