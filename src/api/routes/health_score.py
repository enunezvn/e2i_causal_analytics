"""
E2I Health Score API
====================

FastAPI endpoints for system health monitoring and scoring.

Phase: Agent Output Routing

Endpoints:
- GET  /health-score/check: Run health check
- GET  /health-score/quick: Quick health check
- GET  /health-score/full: Full health check
- GET  /health-score/components: Get component health
- GET  /health-score/models: Get model health
- GET  /health-score/pipelines: Get pipeline health
- GET  /health-score/agents: Get agent health
- GET  /health-score/history: Get health check history
- GET  /health-score/status: Service status

Integration Points:
- Health Score Agent (Tier 3)
- Fast path design - no LLM usage
- Dashboard-ready metrics

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health-score",
    tags=["Health Score"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS
# =============================================================================


class CheckScope(str, Enum):
    """Scope of health check."""

    FULL = "full"
    QUICK = "quick"
    MODELS = "models"
    PIPELINES = "pipelines"
    AGENTS = "agents"


class ComponentStatus(str, Enum):
    """Status of a system component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ModelStatus(str, Enum):
    """Status of a model."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class PipelineStatus(str, Enum):
    """Status of a data pipeline."""

    HEALTHY = "healthy"
    STALE = "stale"
    FAILED = "failed"


class HealthGrade(str, Enum):
    """Health letter grade."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class ComponentHealth(BaseModel):
    """Status of a system component."""

    component_name: str = Field(..., description="Component identifier")
    status: ComponentStatus = Field(..., description="Component status")
    latency_ms: Optional[int] = Field(default=None, description="Check latency in ms")
    last_check: str = Field(..., description="Last check timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class ModelHealth(BaseModel):
    """Model performance metrics."""

    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model display name")
    accuracy: Optional[float] = Field(default=None, description="Model accuracy")
    precision: Optional[float] = Field(default=None, description="Model precision")
    recall: Optional[float] = Field(default=None, description="Model recall")
    f1_score: Optional[float] = Field(default=None, description="Model F1 score")
    auc_roc: Optional[float] = Field(default=None, description="AUC-ROC score")
    prediction_latency_p50_ms: Optional[int] = Field(
        default=None, description="50th percentile prediction latency"
    )
    prediction_latency_p99_ms: Optional[int] = Field(
        default=None, description="99th percentile prediction latency"
    )
    predictions_last_24h: int = Field(default=0, description="Predictions in last 24 hours")
    error_rate: float = Field(default=0.0, description="Error rate (0-1)")
    status: ModelStatus = Field(..., description="Model health status")


class PipelineHealth(BaseModel):
    """Data pipeline status."""

    pipeline_name: str = Field(..., description="Pipeline identifier")
    last_run: str = Field(..., description="Last run timestamp")
    last_success: str = Field(..., description="Last successful run timestamp")
    rows_processed: int = Field(default=0, description="Rows processed in last run")
    freshness_hours: float = Field(..., description="Data freshness in hours")
    status: PipelineStatus = Field(..., description="Pipeline status")


class AgentHealth(BaseModel):
    """Agent availability status."""

    agent_name: str = Field(..., description="Agent identifier")
    tier: int = Field(..., description="Agent tier (0-5)")
    available: bool = Field(..., description="Whether agent is available")
    avg_latency_ms: int = Field(default=0, description="Average response latency")
    success_rate: float = Field(default=1.0, description="Success rate (0-1)")
    last_invocation: Optional[str] = Field(default=None, description="Last invocation timestamp")
    invocations_24h: int = Field(default=0, description="Invocations in last 24 hours")


class HealthScoreResponse(BaseModel):
    """Response from health check."""

    check_id: str = Field(..., description="Unique check identifier")
    check_scope: CheckScope = Field(..., description="Scope of this check")

    # Overall score
    overall_health_score: float = Field(..., description="Overall health score (0-100)")
    health_grade: HealthGrade = Field(..., description="Letter grade (A-F)")

    # Component scores (0-1). F1 (Codex #1): Optional — None means the dimension
    # was NOT measured (no real backend). The dashboard renders None as
    # "Unknown"/"—", never as 0% or healthy.
    component_health_score: Optional[float] = Field(
        default=None, description="Component health score (0-1), null if unmeasured"
    )
    model_health_score: Optional[float] = Field(
        default=None, description="Model health score (0-1), null if unmeasured"
    )
    pipeline_health_score: Optional[float] = Field(
        default=None, description="Pipeline health score (0-1), null if unmeasured"
    )
    agent_health_score: Optional[float] = Field(
        default=None, description="Agent health score (0-1), null if unmeasured"
    )

    # Details (included based on scope)
    component_statuses: Optional[List[ComponentHealth]] = Field(
        default=None, description="Component status details"
    )
    model_metrics: Optional[List[ModelHealth]] = Field(
        default=None, description="Model health details"
    )
    pipeline_statuses: Optional[List[PipelineHealth]] = Field(
        default=None, description="Pipeline status details"
    )
    agent_statuses: Optional[List[AgentHealth]] = Field(
        default=None, description="Agent status details"
    )

    # Issues
    critical_issues: List[str] = Field(
        default_factory=list, description="Critical issues requiring attention"
    )
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")

    # Summary
    health_summary: str = Field(..., description="Human-readable health summary")

    # Metadata
    check_latency_ms: int = Field(..., description="Check duration in ms")
    timestamp: str = Field(..., description="Check timestamp")

    # F1 fail-closed: provenance of the agent's score so the dashboard never
    # presents an unmeasured score as a real measurement. Defaults to "unknown"
    # so any path that forgets to set it fails closed, not open.
    data_provenance: str = Field(
        default="unknown",
        description="Provenance of the score: measured | partial | unknown | placeholder",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "check_id": "hs_abc123",
                "check_scope": "full",
                "overall_health_score": 85.5,
                "health_grade": "B",
                "component_health_score": 0.9,
                "model_health_score": 0.8,
                "pipeline_health_score": 0.85,
                "agent_health_score": 0.9,
                "critical_issues": [],
                "warnings": ["Model 'churn_predictor' has degraded accuracy (0.72)"],
                "health_summary": "System health is good (Grade: B, Score: 85.5/100).",
                "check_latency_ms": 1250,
                "timestamp": "2026-02-06T12:00:00Z",
            }
        }
    )


class DataProvenance(str, Enum):
    """Where the health data in a response came from.

    ``measured`` — produced by the real Health Score agent.
    ``placeholder`` — fabricated sample data served only because the agent is
    unavailable AND mock-fallback is explicitly permitted (dev/test). Surfaced
    so consumers never mistake placeholder values for real measurements.
    """

    MEASURED = "measured"
    PLACEHOLDER = "placeholder"


class ComponentHealthResponse(BaseModel):
    """Response for component health check."""

    component_health_score: float = Field(..., description="Aggregate score (0-1)")
    total_components: int = Field(..., description="Total components checked")
    healthy_count: int = Field(..., description="Healthy component count")
    degraded_count: int = Field(..., description="Degraded component count")
    unhealthy_count: int = Field(..., description="Unhealthy component count")
    components: List[ComponentHealth] = Field(..., description="Component details")
    check_latency_ms: int = Field(..., description="Check duration")
    data_provenance: DataProvenance = Field(
        default=DataProvenance.PLACEHOLDER,
        description="Provenance of this sample data; placeholder unless a real backend measured it",
    )


class ModelHealthResponse(BaseModel):
    """Response for model health check."""

    model_health_score: float = Field(..., description="Aggregate score (0-1)")
    total_models: int = Field(..., description="Total models checked")
    healthy_count: int = Field(..., description="Healthy model count")
    degraded_count: int = Field(..., description="Degraded model count")
    unhealthy_count: int = Field(..., description="Unhealthy model count")
    models: List[ModelHealth] = Field(..., description="Model details")
    check_latency_ms: int = Field(..., description="Check duration")
    data_provenance: DataProvenance = Field(
        default=DataProvenance.PLACEHOLDER,
        description="Provenance of this sample data; placeholder unless a real backend measured it",
    )


class PipelineHealthResponse(BaseModel):
    """Response for pipeline health check."""

    pipeline_health_score: float = Field(..., description="Aggregate score (0-1)")
    total_pipelines: int = Field(..., description="Total pipelines checked")
    healthy_count: int = Field(..., description="Healthy pipeline count")
    stale_count: int = Field(..., description="Stale pipeline count")
    failed_count: int = Field(..., description="Failed pipeline count")
    pipelines: List[PipelineHealth] = Field(..., description="Pipeline details")
    check_latency_ms: int = Field(..., description="Check duration")
    data_provenance: DataProvenance = Field(
        default=DataProvenance.PLACEHOLDER,
        description="Provenance of this sample data; placeholder unless a real backend measured it",
    )


class AgentHealthResponse(BaseModel):
    """Response for agent health check."""

    agent_health_score: float = Field(..., description="Aggregate score (0-1)")
    total_agents: int = Field(..., description="Total agents checked")
    available_count: int = Field(..., description="Available agent count")
    unavailable_count: int = Field(..., description="Unavailable agent count")
    agents: List[AgentHealth] = Field(..., description="Agent details")
    by_tier: Dict[str, int] = Field(..., description="Agent count by tier")
    check_latency_ms: int = Field(..., description="Check duration")
    data_provenance: DataProvenance = Field(
        default=DataProvenance.PLACEHOLDER,
        description="Provenance of this sample data; placeholder unless a real backend measured it",
    )


class HealthHistoryItem(BaseModel):
    """Historical health check record."""

    check_id: str = Field(..., description="Check identifier")
    timestamp: str = Field(..., description="Check timestamp")
    overall_health_score: float = Field(..., description="Score at time of check")
    health_grade: HealthGrade = Field(..., description="Grade at time of check")
    critical_issues_count: int = Field(..., description="Number of critical issues")


class HealthHistoryResponse(BaseModel):
    """Response for health check history."""

    total_checks: int = Field(..., description="Total checks in history")
    checks: List[HealthHistoryItem] = Field(..., description="Historical records")
    avg_health_score: float = Field(..., description="Average health score")
    trend: str = Field(..., description="Trend direction (improving, stable, declining)")


class HealthServiceStatus(BaseModel):
    """Service status response."""

    status: str = Field(..., description="Service status")
    agent_available: bool = Field(..., description="Health Score agent available")
    last_check: Optional[str] = Field(default=None, description="Last health check")
    checks_24h: int = Field(default=0, description="Checks in last 24 hours")
    avg_check_latency_ms: int = Field(default=0, description="Average check latency")


# =============================================================================
# IN-MEMORY STORAGE (replace with persistence in production)
# =============================================================================

_health_history: List[HealthScoreResponse] = []


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "/check",
    response_model=HealthScoreResponse,
    summary="Run health check",
    operation_id="run_health_check",
    description="Run a health check with specified scope.",
)
async def run_health_check(
    scope: CheckScope = Query(default=CheckScope.FULL, description="Check scope"),
) -> HealthScoreResponse:
    """
    Run a health check.

    This endpoint invokes the Health Score agent (Tier 3) which is a
    Fast Path agent with no LLM usage.

    Args:
        scope: Scope of health check (full, quick, models, pipelines, agents)

    Returns:
        Health check results with scores and details
    """
    try:
        import time

        start_time = time.time()

        result = await _execute_health_check(scope)
        check_latency = int((time.time() - start_time) * 1000)

        result.check_latency_ms = check_latency
        result.check_id = f"hs_{uuid4().hex[:12]}"

        # Store in history
        _health_history.append(result)
        # Keep only last 100 checks
        while len(_health_history) > 100:
            _health_history.pop(0)

        return result

    except HTTPException:
        # F-010-backend (#429, codex iter-1 M1): preserve 503 from
        # agent-import guard.
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@router.get(
    "/quick",
    response_model=HealthScoreResponse,
    summary="Quick health check",
    operation_id="quick_health_check",
    description="Run a quick health check (<1s target).",
)
async def quick_health_check() -> HealthScoreResponse:
    """
    Run a quick health check focused on components only.

    Target latency: <1 second

    Returns:
        Basic health check results
    """
    return await run_health_check(scope=CheckScope.QUICK)


@router.get(
    "/full",
    response_model=HealthScoreResponse,
    summary="Full health check",
    operation_id="full_health_check",
    description="Run a comprehensive health check (<5s target).",
)
async def full_health_check() -> HealthScoreResponse:
    """
    Run a full health check across all dimensions.

    Target latency: <5 seconds

    Returns:
        Comprehensive health check results
    """
    return await run_health_check(scope=CheckScope.FULL)


@router.get(
    "/components",
    response_model=ComponentHealthResponse,
    summary="Component health",
    operation_id="get_component_health",
    description="Check health of system components.",
)
async def get_component_health() -> ComponentHealthResponse:
    """
    Get detailed component health information.

    Checks: Database, Cache (Redis), Vector Store, API, Message Queue

    Behavior (F1, Codex #3): this endpoint serves HARDCODED placeholder sample
    data — it does NOT invoke the agent or query a real backend. The response is
    therefore ALWAYS tagged ``data_provenance="placeholder"`` so callers never
    mistake the sample values for real measurements. Mere importability of the
    agent class is not a measurement. The #429 import guard still fails-closed
    (503) in production if the agent cannot be imported at all; otherwise the
    placeholder data is returned and clearly tagged as such.

    For genuinely measured component health, use ``GET /health-score/check`` (or
    ``/quick``), which runs the real Health Score agent + SupabaseHealthClient.

    Returns:
        Component health details (placeholder sample data)
    """
    import time

    start_time = time.time()

    provenance = _resolve_health_provenance(agent_name="Health Score")

    components = _get_mock_component_health()
    check_latency = int((time.time() - start_time) * 1000)

    healthy = sum(1 for c in components if c.status == ComponentStatus.HEALTHY)
    degraded = sum(1 for c in components if c.status == ComponentStatus.DEGRADED)
    unhealthy = sum(1 for c in components if c.status == ComponentStatus.UNHEALTHY)

    score = (healthy * 1.0 + degraded * 0.5) / len(components) if components else 0.0

    return ComponentHealthResponse(
        component_health_score=score,
        total_components=len(components),
        healthy_count=healthy,
        degraded_count=degraded,
        unhealthy_count=unhealthy,
        components=components,
        check_latency_ms=check_latency,
        data_provenance=provenance,
    )


@router.get(
    "/models",
    response_model=ModelHealthResponse,
    summary="Model health",
    operation_id="get_model_health",
    description="Check health of deployed ML models.",
)
async def get_model_health() -> ModelHealthResponse:
    """
    Get detailed model health information.

    Checks model accuracy, latency, error rates, and prediction volume.

    Behavior (F1, Codex #3): mirrors ``/components`` — this endpoint serves
    HARDCODED placeholder sample data (e.g. accuracy=0.89, auc=0.92), NOT real
    measurements. The response is ALWAYS tagged ``data_provenance="placeholder"``
    so fabricated model accuracy is never presented as real in the dashboard.
    The #429 import guard still fails-closed (503) in production if the agent
    cannot be imported at all; otherwise placeholder data is returned and clearly
    tagged. For measured health, use ``GET /health-score/check``.

    Returns:
        Model health details (placeholder sample data)
    """
    import time

    start_time = time.time()

    provenance = _resolve_health_provenance(agent_name="Health Score")

    models = _get_mock_model_health()
    check_latency = int((time.time() - start_time) * 1000)

    healthy = sum(1 for m in models if m.status == ModelStatus.HEALTHY)
    degraded = sum(1 for m in models if m.status == ModelStatus.DEGRADED)
    unhealthy = sum(1 for m in models if m.status == ModelStatus.UNHEALTHY)

    score = (healthy * 1.0 + degraded * 0.5) / len(models) if models else 1.0

    return ModelHealthResponse(
        model_health_score=score,
        total_models=len(models),
        healthy_count=healthy,
        degraded_count=degraded,
        unhealthy_count=unhealthy,
        models=models,
        check_latency_ms=check_latency,
        data_provenance=provenance,
    )


@router.get(
    "/pipelines",
    response_model=PipelineHealthResponse,
    summary="Pipeline health",
    operation_id="get_pipeline_health",
    description="Check health of data pipelines.",
)
async def get_pipeline_health() -> PipelineHealthResponse:
    """
    Get detailed pipeline health information.

    Checks data freshness, processing success, and row counts.

    Behavior (F1, Codex #3): mirrors ``/components`` and ``/models`` — this
    endpoint serves HARDCODED placeholder sample data, NOT real measurements.
    The response is ALWAYS tagged ``data_provenance="placeholder"`` so fabricated
    pipeline freshness/row counts are never presented as real in the dashboard.
    The #429 import guard still fails-closed (503) in production if the agent
    cannot be imported at all; otherwise placeholder data is returned and clearly
    tagged. For measured health, use ``GET /health-score/check``.

    Returns:
        Pipeline health details (placeholder sample data)
    """
    import time

    start_time = time.time()

    provenance = _resolve_health_provenance(agent_name="Health Score")

    pipelines = _get_mock_pipeline_health()
    check_latency = int((time.time() - start_time) * 1000)

    healthy = sum(1 for p in pipelines if p.status == PipelineStatus.HEALTHY)
    stale = sum(1 for p in pipelines if p.status == PipelineStatus.STALE)
    failed = sum(1 for p in pipelines if p.status == PipelineStatus.FAILED)

    score = (healthy * 1.0 + stale * 0.5) / len(pipelines) if pipelines else 1.0

    return PipelineHealthResponse(
        pipeline_health_score=score,
        total_pipelines=len(pipelines),
        healthy_count=healthy,
        stale_count=stale,
        failed_count=failed,
        pipelines=pipelines,
        check_latency_ms=check_latency,
        data_provenance=provenance,
    )


@router.get(
    "/agents",
    response_model=AgentHealthResponse,
    summary="Agent health",
    operation_id="get_agent_health",
    description="Check health of system agents.",
)
async def get_agent_health() -> AgentHealthResponse:
    """
    Get detailed agent health information.

    Checks agent availability, success rates, and latency.

    Behavior (F1, Codex #3): mirrors ``/components`` and ``/models`` — this
    endpoint serves HARDCODED placeholder sample data, NOT real measurements.
    The response is ALWAYS tagged ``data_provenance="placeholder"`` so fabricated
    agent success-rates/latency are never presented as real in the dashboard.
    The #429 import guard still fails-closed (503) in production if the agent
    cannot be imported at all; otherwise placeholder data is returned and clearly
    tagged. For measured health, use ``GET /health-score/check``.

    Returns:
        Agent health details (placeholder sample data)
    """
    import time

    start_time = time.time()

    provenance = _resolve_health_provenance(agent_name="Health Score")

    agents = _get_mock_agent_health()
    check_latency = int((time.time() - start_time) * 1000)

    available = sum(1 for a in agents if a.available)
    unavailable = len(agents) - available

    score = available / len(agents) if agents else 1.0

    by_tier: Dict[str, int] = {}
    for agent in agents:
        tier_key = f"tier_{agent.tier}"
        by_tier[tier_key] = by_tier.get(tier_key, 0) + 1

    return AgentHealthResponse(
        agent_health_score=score,
        total_agents=len(agents),
        available_count=available,
        unavailable_count=unavailable,
        agents=agents,
        by_tier=by_tier,
        check_latency_ms=check_latency,
        data_provenance=provenance,
    )


@router.get(
    "/history",
    response_model=HealthHistoryResponse,
    summary="Health check history",
    operation_id="get_health_history",
    description="Get history of health checks.",
)
async def get_health_history(
    limit: int = Query(default=20, description="Maximum records to return", ge=1, le=100),
) -> HealthHistoryResponse:
    """
    Get historical health check records.

    Returns recent health check results with trend analysis.

    Args:
        limit: Maximum number of records to return

    Returns:
        Historical health check data
    """
    history = _health_history[-limit:] if _health_history else []

    checks = [
        HealthHistoryItem(
            check_id=h.check_id,
            timestamp=h.timestamp,
            overall_health_score=h.overall_health_score,
            health_grade=h.health_grade,
            critical_issues_count=len(h.critical_issues),
        )
        for h in history
    ]

    avg_score = sum(h.overall_health_score for h in history) / len(history) if history else 0.0

    # Calculate trend
    trend = "stable"
    if len(history) >= 3:
        recent_avg = sum(h.overall_health_score for h in history[-3:]) / 3
        earlier_avg = sum(h.overall_health_score for h in history[:3]) / 3
        if recent_avg > earlier_avg + 5:
            trend = "improving"
        elif recent_avg < earlier_avg - 5:
            trend = "declining"

    return HealthHistoryResponse(
        total_checks=len(history),
        checks=checks,
        avg_health_score=avg_score,
        trend=trend,
    )


@router.get(
    "/status",
    response_model=HealthServiceStatus,
    summary="Service status",
    operation_id="get_health_service_status",
    description="Get health score service status.",
)
async def get_service_status() -> HealthServiceStatus:
    """
    Get health score service status.

    Returns:
        Service status information
    """
    # Check agent availability
    agent_available = True
    try:
        from src.agents.health_score import HealthScoreAgent  # noqa: F401

        agent_available = True
    except ImportError:
        agent_available = False

    # Get last check
    last_check = _health_history[-1].timestamp if _health_history else None

    # Count recent checks
    now = datetime.now(timezone.utc)
    checks_24h = sum(
        1
        for h in _health_history
        if (now - datetime.fromisoformat(h.timestamp.replace("Z", "+00:00"))).total_seconds()
        < 86400
    )

    # Calculate average latency
    avg_latency = (
        sum(h.check_latency_ms for h in _health_history) // len(_health_history)
        if _health_history
        else 0
    )

    return HealthServiceStatus(
        status="healthy" if agent_available else "degraded",
        agent_available=agent_available,
        last_check=last_check,
        checks_24h=checks_24h,
        avg_check_latency_ms=avg_latency,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _resolve_health_provenance(*, agent_name: str) -> DataProvenance:
    """Decide the provenance of the per-dimension health endpoints.

    F1b correction: ``/components``, ``/models``, ``/pipelines`` and ``/agents``
    serve HARDCODED ``_get_mock_*_health()`` sample data — they do NOT invoke
    the agent or any real backend. Mere *importability* of the agent class is
    NOT a measurement, so this MUST NOT return ``measured``. The honest
    provenance for these endpoints is ALWAYS ``placeholder``.

    The #429 fail-closed import guard still applies: in a fail-closed
    environment (``E2I_REQUIRE_AGENT_IMPORT=1`` or non-dev ``ENVIRONMENT``) a
    failed agent import raises ``HTTPException(503)`` so placeholder data is not
    served at all in production; otherwise we return ``placeholder`` so dev/test
    callers can plainly see the data is sample data, not a measurement.
    """
    from src.api.utils.agent_import_guard import guard_or_raise

    try:
        from src.agents.health_score import HealthScoreAgent

        # Construct (trackers off — we discard the instance) only to confirm the
        # agent imports in production; a broken/partial deployment raises
        # ImportError, which the #429 guard turns into a 503. Importability does
        # NOT upgrade the served sample data to "measured".
        HealthScoreAgent(enable_mlflow=False, enable_opik=False)
        return DataProvenance.PLACEHOLDER
    except ImportError as e:
        # Raises 503 in fail-closed environments; returns placeholder otherwise.
        guard_or_raise(e, agent_name=agent_name)
        return DataProvenance.PLACEHOLDER


async def _execute_health_check(scope: CheckScope) -> HealthScoreResponse:
    """Execute health check using Health Score agent."""
    import time

    start_time = time.time()

    try:
        # Try to use the actual Health Score agent.
        # F1: wire the REAL SupabaseHealthClient so component health is genuinely
        # measured (not the fail-open mock path). The other dimensions have no
        # real backend yet, so the agent fails them closed to "unknown" and the
        # composite provenance becomes "partial" — surfaced honestly below.
        from src.agents.health_score import HealthScoreAgent, SupabaseHealthClient

        agent = HealthScoreAgent(health_client=SupabaseHealthClient())

        if scope == CheckScope.QUICK:
            result = await agent.quick_check()
        else:
            result = await agent.check_health(scope=scope.value)

        return HealthScoreResponse(
            check_id="",  # Will be set by caller
            check_scope=scope,
            overall_health_score=result.overall_health_score,
            health_grade=HealthGrade(result.health_grade),
            component_health_score=result.component_health_score,
            model_health_score=result.model_health_score,
            pipeline_health_score=result.pipeline_health_score,
            agent_health_score=result.agent_health_score,
            critical_issues=result.critical_issues,
            warnings=result.warnings,
            # F1 (Codex #1): use the agent's own recommendations — they already
            # encode the None/unmeasured semantics ("wire a real <dim> backend",
            # never "system is healthy" while a dim is unmeasured). Re-deriving
            # here from Optional scores would crash on the None comparison.
            recommendations=result.recommendations,
            health_summary=result.health_summary,
            check_latency_ms=result.total_latency_ms,
            timestamp=result.timestamp,
            # F1: propagate the agent's ACTUAL provenance, not a hardcoded value.
            data_provenance=result.data_provenance,
        )

    except ImportError as e:
        # F-010-backend (#429): fail-closed in production unless mock-fallback
        # is explicitly enabled (E2I_REQUIRE_AGENT_IMPORT=0 or ENVIRONMENT!=production).
        from src.api.utils.agent_import_guard import guard_or_raise

        guard_or_raise(e, agent_name="Health Score")
        return _generate_mock_health_response(scope, start_time)

    except Exception as e:
        logger.error(f"Health check execution failed: {e}")
        raise


def _generate_mock_health_response(
    scope: CheckScope,
    start_time: float,
) -> HealthScoreResponse:
    """Generate mock response when agent is not available."""
    import time

    # Mock component health
    components = (
        _get_mock_component_health() if scope in [CheckScope.FULL, CheckScope.QUICK] else None
    )
    models = _get_mock_model_health() if scope in [CheckScope.FULL, CheckScope.MODELS] else None
    pipelines = (
        _get_mock_pipeline_health() if scope in [CheckScope.FULL, CheckScope.PIPELINES] else None
    )
    agents = _get_mock_agent_health() if scope in [CheckScope.FULL, CheckScope.AGENTS] else None

    # Calculate scores
    component_score = 0.9
    model_score = 0.85
    pipeline_score = 0.88
    agent_score = 0.95

    # Weighted overall score
    overall = (
        0.30 * component_score + 0.30 * model_score + 0.25 * pipeline_score + 0.15 * agent_score
    ) * 100

    # Determine grade
    if overall >= 90:
        grade = HealthGrade.A
    elif overall >= 80:
        grade = HealthGrade.B
    elif overall >= 70:
        grade = HealthGrade.C
    elif overall >= 60:
        grade = HealthGrade.D
    else:
        grade = HealthGrade.F

    check_latency = int((time.time() - start_time) * 1000)

    return HealthScoreResponse(
        check_id="",
        check_scope=scope,
        overall_health_score=overall,
        health_grade=grade,
        component_health_score=component_score,
        model_health_score=model_score,
        pipeline_health_score=pipeline_score,
        agent_health_score=agent_score,
        component_statuses=components,
        model_metrics=models,
        pipeline_statuses=pipelines,
        agent_statuses=agents,
        critical_issues=[],
        warnings=["Using mock data - Health Score agent not available"],
        recommendations=_generate_recommendations(
            component_score, model_score, pipeline_score, agent_score
        ),
        health_summary=f"System health is good (Grade: {grade.value}, Score: {overall:.1f}/100). "
        "All core systems operational.",
        check_latency_ms=check_latency,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _get_mock_component_health() -> List[ComponentHealth]:
    """Get mock component health data."""
    now = datetime.now(timezone.utc).isoformat()
    return [
        ComponentHealth(
            component_name="postgresql",
            status=ComponentStatus.HEALTHY,
            latency_ms=12,
            last_check=now,
            error_message=None,
        ),
        ComponentHealth(
            component_name="redis",
            status=ComponentStatus.HEALTHY,
            latency_ms=3,
            last_check=now,
            error_message=None,
        ),
        ComponentHealth(
            component_name="falkordb",
            status=ComponentStatus.HEALTHY,
            latency_ms=8,
            last_check=now,
            error_message=None,
        ),
        ComponentHealth(
            component_name="mlflow",
            status=ComponentStatus.HEALTHY,
            latency_ms=45,
            last_check=now,
            error_message=None,
        ),
        ComponentHealth(
            component_name="opik",
            status=ComponentStatus.DEGRADED,
            latency_ms=250,
            last_check=now,
            error_message="High latency detected",
        ),
    ]


def _get_mock_model_health() -> List[ModelHealth]:
    """Get mock model health data."""
    return [
        ModelHealth(
            model_id="churn_predictor_v2",
            model_name="Churn Predictor",
            accuracy=0.89,
            precision=0.87,
            recall=0.85,
            f1_score=0.86,
            auc_roc=0.92,
            prediction_latency_p50_ms=45,
            prediction_latency_p99_ms=120,
            predictions_last_24h=1250,
            error_rate=0.02,
            status=ModelStatus.HEALTHY,
        ),
        ModelHealth(
            model_id="conversion_model_v1",
            model_name="Conversion Model",
            accuracy=0.82,
            precision=0.80,
            recall=0.78,
            f1_score=0.79,
            auc_roc=0.85,
            prediction_latency_p50_ms=38,
            prediction_latency_p99_ms=95,
            predictions_last_24h=890,
            error_rate=0.03,
            status=ModelStatus.HEALTHY,
        ),
        ModelHealth(
            model_id="uplift_model_v3",
            model_name="Uplift Model",
            accuracy=0.72,
            precision=0.70,
            recall=0.68,
            f1_score=0.69,
            auc_roc=0.75,
            prediction_latency_p50_ms=65,
            prediction_latency_p99_ms=180,
            predictions_last_24h=450,
            error_rate=0.05,
            status=ModelStatus.DEGRADED,
        ),
    ]


def _get_mock_pipeline_health() -> List[PipelineHealth]:
    """Get mock pipeline health data."""
    now = datetime.now(timezone.utc)
    return [
        PipelineHealth(
            pipeline_name="hcp_data_ingestion",
            last_run=(now.isoformat()),
            last_success=(now.isoformat()),
            rows_processed=15420,
            freshness_hours=1.5,
            status=PipelineStatus.HEALTHY,
        ),
        PipelineHealth(
            pipeline_name="trx_aggregation",
            last_run=(now.isoformat()),
            last_success=(now.isoformat()),
            rows_processed=28750,
            freshness_hours=2.0,
            status=PipelineStatus.HEALTHY,
        ),
        PipelineHealth(
            pipeline_name="feature_engineering",
            last_run=(now.isoformat()),
            last_success=(now.isoformat()),
            rows_processed=45000,
            freshness_hours=4.0,
            status=PipelineStatus.HEALTHY,
        ),
        PipelineHealth(
            pipeline_name="kpi_calculations",
            last_run=(now.isoformat()),
            last_success=(now.isoformat()),
            rows_processed=8500,
            freshness_hours=6.5,
            status=PipelineStatus.STALE,
        ),
    ]


def _get_mock_agent_health() -> List[AgentHealth]:
    """Get mock agent health data."""
    now = datetime.now(timezone.utc).isoformat()
    return [
        AgentHealth(
            agent_name="orchestrator",
            tier=1,
            available=True,
            avg_latency_ms=150,
            success_rate=0.98,
            last_invocation=now,
            invocations_24h=450,
        ),
        AgentHealth(
            agent_name="causal_impact",
            tier=2,
            available=True,
            avg_latency_ms=2500,
            success_rate=0.95,
            last_invocation=now,
            invocations_24h=125,
        ),
        AgentHealth(
            agent_name="gap_analyzer",
            tier=2,
            available=True,
            avg_latency_ms=1800,
            success_rate=0.96,
            last_invocation=now,
            invocations_24h=89,
        ),
        AgentHealth(
            agent_name="drift_monitor",
            tier=3,
            available=True,
            avg_latency_ms=800,
            success_rate=0.99,
            last_invocation=now,
            invocations_24h=240,
        ),
        AgentHealth(
            agent_name="health_score",
            tier=3,
            available=True,
            avg_latency_ms=450,
            success_rate=1.0,
            last_invocation=now,
            invocations_24h=180,
        ),
        AgentHealth(
            agent_name="prediction_synthesizer",
            tier=4,
            available=True,
            avg_latency_ms=350,
            success_rate=0.97,
            last_invocation=now,
            invocations_24h=320,
        ),
        AgentHealth(
            agent_name="explainer",
            tier=5,
            available=True,
            avg_latency_ms=1200,
            success_rate=0.94,
            last_invocation=now,
            invocations_24h=210,
        ),
        AgentHealth(
            agent_name="feedback_learner",
            tier=5,
            available=True,
            avg_latency_ms=3500,
            success_rate=0.92,
            last_invocation=now,
            invocations_24h=45,
        ),
    ]


def _generate_recommendations(
    component_score: float,
    model_score: float,
    pipeline_score: float,
    agent_score: float,
) -> List[str]:
    """Generate recommendations based on health scores."""
    recommendations = []

    if component_score < 0.8:
        recommendations.append("Investigate unhealthy components and restore services")

    if model_score < 0.8:
        recommendations.append("Review model performance metrics and consider retraining")

    if pipeline_score < 0.8:
        recommendations.append("Check data pipeline freshness and resolve any failures")

    if agent_score < 0.8:
        recommendations.append("Verify agent availability and address connectivity issues")

    if not recommendations:
        recommendations.append("Continue monitoring - system is healthy")

    return recommendations
