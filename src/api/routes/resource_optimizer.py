"""
E2I Resource Optimizer API
==========================

FastAPI endpoints for resource allocation optimization.

Phase: Agent Output Routing

Endpoints:
- POST /resources/optimize: Run resource optimization
- GET  /resources/{optimization_id}: Get optimization results
- GET  /resources/scenarios: List scenario analyses
- GET  /resources/health: Service health check

Integration Points:
- Resource Optimizer Agent (Tier 4)
- scipy for linear/nonlinear optimization
- MILP solvers for discrete optimization

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/resources",
    tags=["Resource Optimization"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS
# =============================================================================


class OptimizationObjective(str, Enum):
    """Optimization objectives."""

    MAXIMIZE_OUTCOME = "maximize_outcome"
    MAXIMIZE_ROI = "maximize_roi"
    MINIMIZE_COST = "minimize_cost"
    BALANCE = "balance"


class SolverType(str, Enum):
    """Available solver types."""

    LINEAR = "linear"
    MILP = "milp"
    NONLINEAR = "nonlinear"


class OptimizationStatus(str, Enum):
    """Status of optimization."""

    PENDING = "pending"
    FORMULATING = "formulating"
    OPTIMIZING = "optimizing"
    ANALYZING = "analyzing"
    PROJECTING = "projecting"
    COMPLETED = "completed"
    FAILED = "failed"


class ResourceType(str, Enum):
    """Types of resources to optimize."""

    BUDGET = "budget"
    REP_TIME = "rep_time"
    SAMPLES = "samples"
    CALLS = "calls"


class ConstraintType(str, Enum):
    """Types of optimization constraints."""

    BUDGET = "budget"
    CAPACITY = "capacity"
    MIN_COVERAGE = "min_coverage"
    MAX_FREQUENCY = "max_frequency"


class ConstraintScope(str, Enum):
    """Scope of constraints."""

    GLOBAL = "global"
    REGIONAL = "regional"
    ENTITY = "entity"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class AllocationTarget(BaseModel):
    """Target entity for resource allocation."""

    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(..., description="Entity type (hcp, territory, region)")
    current_allocation: float = Field(..., description="Current allocation amount")
    min_allocation: Optional[float] = Field(default=None, description="Minimum allowed allocation")
    max_allocation: Optional[float] = Field(default=None, description="Maximum allowed allocation")
    expected_response: float = Field(default=1.0, description="Response coefficient")


class Constraint(BaseModel):
    """Optimization constraint."""

    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    value: float = Field(..., description="Constraint value")
    scope: ConstraintScope = Field(default=ConstraintScope.GLOBAL, description="Constraint scope")


class RunOptimizationRequest(BaseModel):
    """Request to run resource optimization."""

    query: str = Field(..., description="Natural language query")
    resource_type: ResourceType = Field(..., description="Type of resource to optimize")
    allocation_targets: List[AllocationTarget] = Field(
        ..., description="Entities to allocate resources to"
    )
    constraints: List[Constraint] = Field(
        default_factory=list, description="Optimization constraints"
    )
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MAXIMIZE_OUTCOME,
        description="Optimization objective",
    )

    # Configuration
    solver_type: SolverType = Field(default=SolverType.LINEAR, description="Solver type")
    time_limit_seconds: int = Field(default=60, description="Solver time limit", ge=1, le=300)
    gap_tolerance: float = Field(default=0.01, description="MILP gap tolerance", gt=0.0, lt=1.0)
    run_scenarios: bool = Field(default=False, description="Run what-if scenarios")
    scenario_count: int = Field(default=3, description="Number of scenarios", ge=1, le=10)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Optimize budget allocation across territories",
                "resource_type": "budget",
                "allocation_targets": [
                    {
                        "entity_id": "territory_northeast",
                        "entity_type": "territory",
                        "current_allocation": 50000,
                        "min_allocation": 30000,
                        "max_allocation": 80000,
                        "expected_response": 1.3,
                    }
                ],
                "constraints": [{"constraint_type": "budget", "value": 200000, "scope": "global"}],
                "objective": "maximize_outcome",
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class AllocationResult(BaseModel):
    """Optimized allocation result for an entity."""

    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(..., description="Entity type")
    current_allocation: float = Field(..., description="Current allocation")
    optimized_allocation: float = Field(..., description="Optimized allocation")
    change: float = Field(..., description="Change from current")
    change_percentage: float = Field(..., description="Change percentage")
    expected_impact: float = Field(..., description="Expected outcome impact")


class ScenarioResult(BaseModel):
    """Result of a scenario analysis."""

    scenario_name: str = Field(..., description="Scenario name")
    total_allocation: float = Field(..., description="Total allocation in scenario")
    projected_outcome: float = Field(..., description="Projected outcome")
    roi: float = Field(..., description="Return on investment")
    constraint_violations: List[str] = Field(
        default_factory=list, description="Any constraint violations"
    )


class OptimizationResponse(BaseModel):
    """Response from resource optimization."""

    optimization_id: str = Field(..., description="Unique optimization identifier")
    status: OptimizationStatus = Field(..., description="Optimization status")
    resource_type: ResourceType = Field(..., description="Resource type optimized")
    objective: OptimizationObjective = Field(..., description="Objective used")

    # Optimization results
    optimal_allocations: List[AllocationResult] = Field(
        default_factory=list, description="Optimized allocations"
    )
    objective_value: Optional[float] = Field(default=None, description="Optimized objective value")
    solver_status: Optional[str] = Field(default=None, description="Solver termination status")
    solve_time_ms: int = Field(default=0, description="Solver time (ms)")

    # Scenario results
    scenarios: List[ScenarioResult] = Field(
        default_factory=list, description="Scenario analysis results"
    )
    sensitivity_analysis: Optional[Dict[str, float]] = Field(
        default=None, description="Sensitivity of objective to constraints"
    )

    # Impact projections
    projected_total_outcome: Optional[float] = Field(
        default=None, description="Total projected outcome"
    )
    projected_roi: Optional[float] = Field(default=None, description="Projected ROI")
    impact_by_segment: Optional[Dict[str, float]] = Field(
        default=None, description="Impact breakdown by segment"
    )

    # Summary
    optimization_summary: Optional[str] = Field(default=None, description="Executive summary")
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )

    # Metadata
    formulation_latency_ms: int = Field(default=0, description="Problem formulation time")
    optimization_latency_ms: int = Field(default=0, description="Optimization time")
    total_latency_ms: int = Field(default=0, description="Total workflow time")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Optimization timestamp",
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "optimization_id": "opt_abc123",
                "status": "completed",
                "resource_type": "budget",
                "objective": "maximize_roi",
                "objective_value": 450000,
                "projected_roi": 2.25,
            }
        }
    )


class ScenarioListResponse(BaseModel):
    """Response for listing scenario analyses."""

    total_count: int = Field(..., description="Total scenarios")
    scenarios: List[ScenarioResult] = Field(..., description="Scenario results")


class ResourceHealthResponse(BaseModel):
    """Health check response for resource optimization service."""

    status: str = Field(..., description="Service status")
    agent_available: bool = Field(..., description="Resource Optimizer agent status")
    scipy_available: bool = Field(default=True, description="scipy availability")
    last_optimization: Optional[datetime] = Field(
        default=None, description="Last optimization timestamp"
    )
    optimizations_24h: int = Field(default=0, description="Optimizations in last 24 hours")


# =============================================================================
# IN-MEMORY STORAGE (replace with Supabase in production)
# =============================================================================

_optimizations_store: Dict[str, OptimizationResponse] = {}


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    summary="Run resource optimization",
    operation_id="run_optimization",
    description="Optimize resource allocation across entities.",
)
async def run_optimization(
    request: RunOptimizationRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(
        default=True, description="Run asynchronously (returns immediately with ID)"
    ),
) -> OptimizationResponse:
    """
    Run resource optimization.

    This endpoint invokes the Resource Optimizer agent (Tier 4) to:
    1. Formulate optimization problem
    2. Solve using appropriate solver
    3. Run optional scenario analysis
    4. Project allocation impact

    Args:
        request: Optimization parameters
        background_tasks: FastAPI background tasks
        async_mode: If True, returns immediately with optimization ID

    Returns:
        Optimization results or pending status if async
    """
    optimization_id = f"opt_{uuid4().hex[:12]}"

    # Create initial response
    response = OptimizationResponse(
        optimization_id=optimization_id,
        status=OptimizationStatus.PENDING if async_mode else OptimizationStatus.FORMULATING,
        resource_type=request.resource_type,
        objective=request.objective,
    )

    if async_mode:
        # Store pending optimization
        _optimizations_store[optimization_id] = response

        # Schedule background task
        background_tasks.add_task(
            _run_optimization_task,
            optimization_id=optimization_id,
            request=request,
        )

        logger.info(f"Optimization {optimization_id} queued for background execution")
        return response

    # Synchronous execution
    try:
        result = await _execute_optimization(request)
        result.optimization_id = optimization_id
        _optimizations_store[optimization_id] = result
        return result
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        response.status = OptimizationStatus.FAILED
        response.warnings.append(str(e))
        _optimizations_store[optimization_id] = response
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")


@router.get(
    "/{optimization_id}",
    response_model=OptimizationResponse,
    summary="Get optimization results",
    operation_id="get_optimization",
    description="Retrieve results of an optimization by ID.",
)
async def get_optimization(optimization_id: str) -> OptimizationResponse:
    """
    Get optimization results by ID.

    Args:
        optimization_id: Unique optimization identifier

    Returns:
        Optimization results

    Raises:
        HTTPException: If optimization not found
    """
    if optimization_id not in _optimizations_store:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization {optimization_id} not found",
        )

    return _optimizations_store[optimization_id]


@router.get(
    "/scenarios",
    response_model=ScenarioListResponse,
    summary="List scenario analyses",
    operation_id="list_scenarios",
    description="List scenario analyses from all optimizations.",
)
async def list_scenarios(
    min_roi: Optional[float] = Query(default=None, description="Minimum ROI threshold"),
    limit: int = Query(default=20, description="Maximum results", ge=1, le=100),
) -> ScenarioListResponse:
    """
    List scenario analyses from optimizations.

    Args:
        min_roi: Minimum ROI threshold
        limit: Maximum number of results

    Returns:
        List of scenario analyses
    """
    all_scenarios: List[ScenarioResult] = []

    for opt in _optimizations_store.values():
        if opt.status != OptimizationStatus.COMPLETED:
            continue

        for scenario in opt.scenarios:
            if min_roi and scenario.roi < min_roi:
                continue
            all_scenarios.append(scenario)

    # Sort by ROI and limit
    all_scenarios.sort(key=lambda x: x.roi, reverse=True)
    all_scenarios = all_scenarios[:limit]

    return ScenarioListResponse(
        total_count=len(all_scenarios),
        scenarios=all_scenarios,
    )


@router.get(
    "/health",
    response_model=ResourceHealthResponse,
    summary="Resource optimization service health",
    operation_id="get_resource_health",
    description="Check health status of the resource optimization service.",
)
async def get_resource_health() -> ResourceHealthResponse:
    """
    Get health status of resource optimization service.

    Returns:
        Service health information
    """
    # Check agent availability
    agent_available = True
    try:
        from src.agents.resource_optimizer import ResourceOptimizerAgent  # noqa: F401

        agent_available = True
    except ImportError:
        agent_available = False

    # Check scipy availability
    scipy_available = True
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        scipy_available = False

    # Count recent optimizations
    now = datetime.now(timezone.utc)
    optimizations_24h = sum(
        1 for o in _optimizations_store.values() if (now - o.timestamp).total_seconds() < 86400
    )

    # Get last optimization
    last_optimization = None
    if _optimizations_store:
        last_optimization = max(o.timestamp for o in _optimizations_store.values())

    status = "healthy"
    if not agent_available:
        status = "degraded"
    elif not scipy_available:
        status = "partial"

    return ResourceHealthResponse(
        status=status,
        agent_available=agent_available,
        scipy_available=scipy_available,
        last_optimization=last_optimization,
        optimizations_24h=optimizations_24h,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _run_optimization_task(
    optimization_id: str,
    request: RunOptimizationRequest,
) -> None:
    """Background task to run optimization."""
    try:
        logger.info(f"Starting optimization task {optimization_id}")

        # Update status
        if optimization_id in _optimizations_store:
            _optimizations_store[optimization_id].status = OptimizationStatus.FORMULATING

        # Execute optimization
        result = await _execute_optimization(request)
        result.optimization_id = optimization_id

        # Store result
        _optimizations_store[optimization_id] = result

        logger.info(f"Optimization {optimization_id} completed successfully")

    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {e}")
        if optimization_id in _optimizations_store:
            _optimizations_store[optimization_id].status = OptimizationStatus.FAILED
            _optimizations_store[optimization_id].warnings.append(str(e))


async def _execute_optimization(
    request: RunOptimizationRequest,
) -> OptimizationResponse:
    """
    Execute optimization using Resource Optimizer agent.

    This function orchestrates the Resource Optimizer agent (Tier 4) to:
    1. Formulate optimization problem via problem_formulator node
    2. Solve via optimizer node
    3. Analyze scenarios via scenario_analyzer node
    4. Project impact via impact_projector node
    """
    import time

    start_time = time.time()

    try:
        # Try to use the actual Resource Optimizer agent
        from src.agents.resource_optimizer.graph import (
            create_resource_optimizer_graph,
        )
        from src.agents.resource_optimizer.state import ResourceOptimizerState

        # Convert request targets to state format
        allocation_targets = [
            {
                "entity_id": t.entity_id,
                "entity_type": t.entity_type,
                "current_allocation": t.current_allocation,
                "min_allocation": t.min_allocation,
                "max_allocation": t.max_allocation,
                "expected_response": t.expected_response,
            }
            for t in request.allocation_targets
        ]

        # Convert constraints
        constraints = [
            {
                "constraint_type": c.constraint_type.value,
                "value": c.value,
                "scope": c.scope.value,
            }
            for c in request.constraints
        ]

        # Initialize state
        initial_state: ResourceOptimizerState = {
            "query": request.query,
            "resource_type": request.resource_type.value,
            "allocation_targets": allocation_targets,
            "constraints": constraints,
            "objective": request.objective.value,
            "solver_type": request.solver_type.value,
            "time_limit_seconds": request.time_limit_seconds,
            "gap_tolerance": request.gap_tolerance,
            "run_scenarios": request.run_scenarios,
            "scenario_count": request.scenario_count,
            "status": "pending",
            "errors": [],
            "warnings": [],
            "formulation_latency_ms": 0,
            "optimization_latency_ms": 0,
            "total_latency_ms": 0,
        }

        # Create and run graph
        graph = create_resource_optimizer_graph()
        result = await graph.ainvoke(initial_state)

        # Convert agent output to API response
        total_latency = int((time.time() - start_time) * 1000)

        return OptimizationResponse(
            optimization_id="",  # Will be set by caller
            status=OptimizationStatus.COMPLETED
            if result.get("status") == "completed"
            else OptimizationStatus.FAILED,
            resource_type=request.resource_type,
            objective=request.objective,
            optimal_allocations=_convert_allocations(result.get("optimal_allocations", [])),
            objective_value=result.get("objective_value"),
            solver_status=result.get("solver_status"),
            solve_time_ms=result.get("solve_time_ms", 0),
            scenarios=_convert_scenarios(result.get("scenarios", [])),
            sensitivity_analysis=result.get("sensitivity_analysis"),
            projected_total_outcome=result.get("projected_total_outcome"),
            projected_roi=result.get("projected_roi"),
            impact_by_segment=result.get("impact_by_segment"),
            optimization_summary=result.get("optimization_summary"),
            recommendations=result.get("recommendations", []),
            formulation_latency_ms=result.get("formulation_latency_ms", 0),
            optimization_latency_ms=result.get("optimization_latency_ms", 0),
            total_latency_ms=total_latency,
            warnings=result.get("warnings", []),
        )

    except ImportError as e:
        logger.warning(f"Resource Optimizer agent not available: {e}, using mock data")
        return _generate_mock_response(request, start_time)

    except Exception as e:
        logger.error(f"Optimization execution failed: {e}")
        raise


def _convert_allocations(
    allocations: List[Dict[str, Any]],
) -> List[AllocationResult]:
    """Convert agent allocation output to API response format."""
    result = []
    for alloc in allocations:
        try:
            result.append(
                AllocationResult(
                    entity_id=alloc.get("entity_id", ""),
                    entity_type=alloc.get("entity_type", ""),
                    current_allocation=alloc.get("current_allocation", 0.0),
                    optimized_allocation=alloc.get("optimized_allocation", 0.0),
                    change=alloc.get("change", 0.0),
                    change_percentage=alloc.get("change_percentage", 0.0),
                    expected_impact=alloc.get("expected_impact", 0.0),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert allocation: {e}")
    return result


def _convert_scenarios(
    scenarios: List[Dict[str, Any]],
) -> List[ScenarioResult]:
    """Convert agent scenario output to API response format."""
    result = []
    for scenario in scenarios:
        try:
            result.append(
                ScenarioResult(
                    scenario_name=scenario.get("scenario_name", ""),
                    total_allocation=scenario.get("total_allocation", 0.0),
                    projected_outcome=scenario.get("projected_outcome", 0.0),
                    roi=scenario.get("roi", 0.0),
                    constraint_violations=scenario.get("constraint_violations", []),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert scenario: {e}")
    return result


def _generate_mock_response(
    request: RunOptimizationRequest,
    start_time: float,
) -> OptimizationResponse:
    """Generate mock response when agent is not available."""
    import time

    # Calculate mock optimizations
    total_current = sum(t.current_allocation for t in request.allocation_targets)
    total_budget = total_current

    # Find budget constraint if exists
    for c in request.constraints:
        if c.constraint_type == ConstraintType.BUDGET:
            total_budget = c.value
            break

    # Generate mock allocations
    mock_allocations = []
    for target in request.allocation_targets:
        # Increase high responders, decrease low responders
        if target.expected_response > 1.1:
            change_pct = 0.2
        elif target.expected_response < 0.9:
            change_pct = -0.15
        else:
            change_pct = 0.05

        optimized = target.current_allocation * (1 + change_pct)

        # Apply constraints
        if target.min_allocation and optimized < target.min_allocation:
            optimized = target.min_allocation
        if target.max_allocation and optimized > target.max_allocation:
            optimized = target.max_allocation

        change = optimized - target.current_allocation

        mock_allocations.append(
            AllocationResult(
                entity_id=target.entity_id,
                entity_type=target.entity_type,
                current_allocation=target.current_allocation,
                optimized_allocation=round(optimized, 2),
                change=round(change, 2),
                change_percentage=round(change / target.current_allocation * 100, 1)
                if target.current_allocation > 0
                else 0.0,
                expected_impact=round(optimized * target.expected_response, 2),
            )
        )

    # Mock scenarios
    mock_scenarios = []
    if request.run_scenarios:
        mock_scenarios = [
            ScenarioResult(
                scenario_name="Conservative",
                total_allocation=total_budget * 0.9,
                projected_outcome=total_budget * 0.9 * 1.8,
                roi=1.8,
                constraint_violations=[],
            ),
            ScenarioResult(
                scenario_name="Aggressive",
                total_allocation=total_budget * 1.1,
                projected_outcome=total_budget * 1.1 * 2.1,
                roi=2.1,
                constraint_violations=["budget_exceeded"],
            ),
            ScenarioResult(
                scenario_name="Balanced",
                total_allocation=total_budget,
                projected_outcome=total_budget * 2.0,
                roi=2.0,
                constraint_violations=[],
            ),
        ]

    total_optimized = sum(a.optimized_allocation for a in mock_allocations)
    total_impact = sum(a.expected_impact for a in mock_allocations)
    projected_roi = total_impact / total_optimized if total_optimized > 0 else 0

    total_latency = int((time.time() - start_time) * 1000)

    increases = sum(1 for a in mock_allocations if a.change > 0)
    decreases = sum(1 for a in mock_allocations if a.change < 0)

    return OptimizationResponse(
        optimization_id="",
        status=OptimizationStatus.COMPLETED,
        resource_type=request.resource_type,
        objective=request.objective,
        optimal_allocations=mock_allocations,
        objective_value=round(total_impact, 2),
        solver_status="optimal",
        solve_time_ms=150,
        scenarios=mock_scenarios,
        sensitivity_analysis={
            "budget": 0.85,
            "capacity": 0.42,
        },
        projected_total_outcome=round(total_impact, 2),
        projected_roi=round(projected_roi, 2),
        impact_by_segment={
            "high_responders": round(total_impact * 0.6, 2),
            "medium_responders": round(total_impact * 0.3, 2),
            "low_responders": round(total_impact * 0.1, 2),
        },
        optimization_summary=f"Optimization complete. Projected outcome: {total_impact:.0f} (ROI: {projected_roi:.2f}). Recommended changes: {increases} increases, {decreases} decreases.",
        recommendations=[
            f"Increase allocation to high-response entities (+{increases} entities)",
            f"Decrease allocation to low-response entities (-{decreases} entities)",
            f"Total reallocation: ${abs(sum(a.change for a in mock_allocations)):,.0f}",
        ],
        formulation_latency_ms=50,
        optimization_latency_ms=150,
        total_latency_ms=total_latency,
        warnings=["Using mock data - Resource Optimizer agent not available"],
    )
