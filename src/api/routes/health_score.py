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
import re
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
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
    # Optional: None = UNMEASURED (no ml_performance_metrics source), not a real
    # zero. A 0/0.0 default here would fabricate a count/rate never observed.
    predictions_last_24h: Optional[int] = Field(
        default=None, description="Predictions in last 24 hours, null if unmeasured"
    )
    error_rate: Optional[float] = Field(
        default=None, description="Error rate (0-1), null if unmeasured"
    )
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
    # Optional: None means UNMEASURED (no recent runtime telemetry), NOT a
    # measured zero. A 1.0/0.0 default here would fabricate a success rate the
    # system never observed; null is the honest value when provenance=partial.
    avg_latency_ms: Optional[int] = Field(
        default=None, description="Average response latency, null if unmeasured"
    )
    success_rate: Optional[float] = Field(
        default=None, description="Success rate (0-1), null if unmeasured"
    )
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

    ``measured`` — every field came from a real backend query.
    ``partial`` — the dimension is real-backed (rows came from a live table) but
    some sub-fields have no source yet and are left null/empty (e.g. model
    accuracy when ``ml_performance_metrics`` is empty, or agent runtime metrics
    when no recent telemetry exists). Honest middle ground — never fabricated.
    ``unknown`` — no real source returned rows; an empty, non-fabricated result.
    ``placeholder`` — hardcoded sample data served only because the real backend
    is unavailable AND mock-fallback is explicitly permitted (dev/test).
    Surfaced so consumers never mistake placeholder values for real measurements.
    """

    MEASURED = "measured"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
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
    # Same fail-closed contract as the live payloads: a recorded score is only
    # as trustworthy as the check that produced it, so the wire carries its
    # provenance and defaults to "unknown" (untrusted) if a writer forgets it.
    data_provenance: str = Field(
        default="unknown",
        description="Provenance of the recorded score: measured | partial | unknown | placeholder",
    )


class HealthHistoryDailyPoint(BaseModel):
    """One UTC-day aggregate of recorded full health checks.

    Aggregates carry their own provenance: 'measured' only when EVERY
    contributing check was measured, else 'partial'. The durable table's CHECK
    constraint means untrusted checks can never contribute at all.
    """

    date: str = Field(..., description="UTC day (YYYY-MM-DD)")
    avg_score: float = Field(..., description="Mean overall health score that day")
    min_score: float = Field(..., description="Lowest overall health score that day")
    max_score: float = Field(..., description="Highest overall health score that day")
    checks_count: int = Field(..., description="Recorded checks contributing to the day")
    data_provenance: str = Field(
        default="partial",
        description="'measured' when every contributing check was measured, else 'partial'",
    )


class HealthHistoryResponse(BaseModel):
    """Response for health check history."""

    total_checks: int = Field(..., description="Total checks in history")
    checks: List[HealthHistoryItem] = Field(..., description="Historical records")
    avg_health_score: Optional[float] = Field(
        default=None, description="Average health score, null when there is no history"
    )
    trend: str = Field(
        default="unknown",
        description="Trend direction (improving, stable, declining, unknown)",
    )
    # Daily aggregates over the requested window, ascending by date. Empty when
    # no durable history exists yet — the chart renders honestly empty rather
    # than replotting the minutes-scale in-memory fallback as a month of data.
    daily: List[HealthHistoryDailyPoint] = Field(
        default_factory=list, description="Per-UTC-day aggregates over the window"
    )
    window_days: Optional[int] = Field(
        default=None,
        description="Days window served from durable history; null when serving the in-memory fallback",
    )


class HealthServiceStatus(BaseModel):
    """Service status response."""

    status: str = Field(..., description="Service status")
    agent_available: bool = Field(..., description="Health Score agent available")
    last_check: Optional[str] = Field(default=None, description="Last health check")
    checks_24h: int = Field(default=0, description="Checks in last 24 hours")
    avg_check_latency_ms: int = Field(default=0, description="Average check latency")


# =============================================================================
# HISTORY STORAGE
# =============================================================================
# Durable source of truth: the health_check_history table (migration 096).
# The in-memory list remains as (a) the read fallback when the DB is
# unreachable and (b) a per-process cache for /status; it is per-gunicorn-
# worker and reset on restart, so it must never be presented as a multi-day
# history — /history only serves it when the durable read fails.

_health_history: List[HealthScoreResponse] = []

# One durable row per 10 minutes at most: the dashboard polls /full every 60s
# while open, and per-minute rows would bloat the table without adding trend
# resolution. The 6h lifespan heartbeat (src/api/main.py) guarantees points
# even when nobody has the page open. The probe below is only a cheap
# short-circuit for the 60s polls — the dedup GUARANTEE is the table's
# UNIQUE(time_bucket): the writer upserts with ON CONFLICT DO NOTHING, so
# workers that pass the probe concurrently still yield one row per bucket.
_HISTORY_WRITE_MIN_INTERVAL_S = 600
_HISTORY_RETENTION_DAYS = 90


def _record_history_durable(result: HealthScoreResponse) -> None:
    """Best-effort durable write of a trusted full check (never raises).

    Failures only log: recording history must never fail the health check
    itself. The table's CHECK constraint re-enforces the trusted-provenance
    gate at the DB layer, so no future writer can regress it.
    """
    db = _health_source_client()
    if db is None:
        return
    try:
        now = datetime.now(timezone.utc)
        last = (
            db.table("health_check_history")
            .select("checked_at")
            .order("checked_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        if (
            last
            and _hours_since(last[0].get("checked_at"), now) * 3600.0
            < _HISTORY_WRITE_MIN_INTERVAL_S
        ):
            return
        # ON CONFLICT (time_bucket) DO NOTHING makes the write itself atomic:
        # the probe above is not (SELECT then INSERT are separate round-trips),
        # so concurrent workers can all pass it — the UNIQUE bucket then keeps
        # exactly one of their rows (codex rounds 1-2 LOW). Residual duplicates
        # need two writes inside probe latency STRADDLING a 600s boundary —
        # vanishingly rare and benign (a real check a few seconds early).
        bucket = int(now.timestamp()) // _HISTORY_WRITE_MIN_INTERVAL_S
        db.table("health_check_history").upsert(
            {
                "check_id": result.check_id,
                "checked_at": result.timestamp,
                "time_bucket": bucket,
                "overall_health_score": round(float(result.overall_health_score), 2),
                "health_grade": str(result.health_grade.value),
                "component_health_score": result.component_health_score,
                "model_health_score": result.model_health_score,
                "pipeline_health_score": result.pipeline_health_score,
                "agent_health_score": result.agent_health_score,
                "critical_issues_count": len(result.critical_issues),
                "warnings_count": len(result.warnings),
                "data_provenance": result.data_provenance,
                "check_scope": result.check_scope.value,
            },
            on_conflict="time_bucket",
            ignore_duplicates=True,
        ).execute()
        # Retention sweep piggybacks on the rate-limited write path (at most a
        # handful of deletes per day — no separate cleanup job needed).
        cutoff = (now - timedelta(days=_HISTORY_RETENTION_DAYS)).isoformat()
        db.table("health_check_history").delete().lt("checked_at", cutoff).execute()
    except Exception as e:
        logger.warning(f"health history: durable write failed ({e})")


def _record_full_check(result: HealthScoreResponse) -> None:
    """Record a health check in history — ONLY trusted full-scope checks.

    A QUICK check measures components only (model/pipeline/agent UNMEASURED),
    so its overall score is component-only (e.g. 100/A) and would pollute the
    health TREND with a misleadingly-rosy flat line; single-dimension scopes
    (models/pipelines/agents) are not an overall measurement either. Only the
    FULL all-dimension check is a faithful overall data point for the trend.

    Provenance guard: a full check whose score is placeholder (dev mock
    fallback) or unknown (fail-closed default) is a fabricated data point —
    recording it would replot as historical truth the very score the live
    dashboard refuses to render.
    """
    if result.check_scope != CheckScope.FULL:
        return
    if result.data_provenance not in (
        DataProvenance.MEASURED.value,
        DataProvenance.PARTIAL.value,
    ):
        return
    _health_history.append(result)
    # Keep only last 100 checks in the in-memory fallback
    while len(_health_history) > 100:
        _health_history.pop(0)
    _record_history_durable(result)


async def run_scheduled_full_check() -> None:
    """Run a FULL health check and record it (lifespan heartbeat entry point).

    Mirrors GET /check?scope=full's recording semantics exactly (same trusted-
    provenance gate, same rate-limited durable write), so scheduled points are
    indistinguishable from organic ones and multi-worker firings dedup.
    """
    result = await _execute_health_check(CheckScope.FULL)
    result.check_id = f"hs_{uuid4().hex[:12]}"
    _record_full_check(result)


def _trend_from_scores(scores: List[float]) -> str:
    """Trend over an ascending score series: last-3 avg vs first-3 avg.

    "unknown" below 3 points — a default "stable" would fabricate a trend
    from zero/one/two data points.
    """
    if len(scores) < 3:
        return "unknown"
    recent_avg = sum(scores[-3:]) / 3
    earlier_avg = sum(scores[:3]) / 3
    if recent_avg > earlier_avg + 5:
        return "improving"
    if recent_avg < earlier_avg - 5:
        return "declining"
    return "stable"


def _read_history_durable(
    days: int, limit: int
) -> Optional[tuple[List[HealthHistoryItem], List[HealthHistoryDailyPoint]]]:
    """Read recent checks + daily aggregates from the durable table.

    Returns None on ANY failure (unreachable DB, malformed row) so the caller
    falls back to the in-memory list — fail closed to the old behavior, never
    500. Both lists come back ascending (oldest first), matching the
    in-memory contract the frontend charts already consume.
    """
    db = _health_source_client()
    if db is None:
        return None
    try:
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=days)).isoformat()
        raw = (
            db.table("health_check_history")
            .select(
                "check_id, checked_at, overall_health_score, health_grade, "
                "critical_issues_count, data_provenance"
            )
            .gte("checked_at", cutoff)
            .order("checked_at", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        checks = [
            HealthHistoryItem(
                check_id=str(r.get("check_id") or ""),
                timestamp=str(r.get("checked_at") or ""),
                overall_health_score=float(r["overall_health_score"]),
                health_grade=HealthGrade(str(r.get("health_grade"))),
                critical_issues_count=int(r.get("critical_issues_count") or 0),
                data_provenance=str(r.get("data_provenance") or "unknown"),
            )
            for r in reversed(raw)
        ]
        # The daily view (migration 096) aggregates in SQL, so this reads at
        # most `days` rows — no PostgREST row-cap concerns. days-1 because the
        # gte cutoff is inclusive and today's partial bucket counts as day 1:
        # days=30 must yield at most 30 dates (today plus the 29 before it),
        # not 31 (codex round-2 LOW).
        cutoff_day = (now - timedelta(days=days - 1)).date().isoformat()
        daily_rows = (
            db.table("health_check_history_daily")
            .select("day, avg_score, min_score, max_score, checks_count, data_provenance")
            .gte("day", cutoff_day)
            .order("day", desc=False)
            .limit(days)
            .execute()
            .data
            or []
        )
        daily = [
            HealthHistoryDailyPoint(
                date=str(d.get("day")),
                avg_score=float(d["avg_score"]),
                min_score=float(d["min_score"]),
                max_score=float(d["max_score"]),
                checks_count=int(d.get("checks_count") or 0),
                data_provenance=str(d.get("data_provenance") or "partial"),
            )
            for d in daily_rows
        ]
        return checks, daily
    except Exception as e:
        logger.warning(f"health history: durable read failed ({e})")
        return None


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

        # Store in history (in-memory + durable table, migration 096) — the
        # recorder enforces the full-scope + trusted-provenance gates and
        # rate-limits the durable write; see _record_full_check.
        _record_full_check(result)

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

    Behavior: probes the five components DIRECTLY via SupabaseHealthClient
    (``_fetch_component_health``) and tags the response
    ``data_provenance="measured"`` — mirroring the /models, /pipelines, /agents
    readers. This replaces the prior composite-agent path, whose
    ``HealthScoreOutput`` never surfaced the per-component status list (so this
    endpoint always fell back to placeholder and the dashboard's Services card
    showed 0/0). Only when every probe fails does it fall back to the #429 guard:
    a 503 in production, or clearly-tagged ``placeholder`` sample data in dev/test.

    Returns:
        Component health details (measured live, or honest placeholder in dev)
    """
    import time

    start_time = time.time()

    components, provenance = await _fetch_component_health()
    if provenance is None:
        # Backend unreachable: fail closed in prod (503), clearly-tagged
        # placeholder only in an explicit dev/test environment.
        _fail_closed_if_no_backend()
        provenance = DataProvenance.PLACEHOLDER
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

    Behavior: reads REAL model health from ml_model_health_dashboard (production
    stage). Status/drift/alert signals are measured; performance sub-fields
    (accuracy/latency) are left null while ml_performance_metrics is empty, so
    the response is tagged ``data_provenance="partial"`` (never a fabricated
    accuracy). Only when the backend is unreachable does it fall back to the #429
    guard: 503 in production, clearly-tagged placeholder in dev/test.

    Returns:
        Model health details (measured/partial from the live registry)
    """
    import time

    start_time = time.time()

    models, provenance = _fetch_model_health()
    if provenance is None:
        # Backend unreachable: fail closed in prod (503), clearly-tagged
        # placeholder only in an explicit dev/test environment.
        _fail_closed_if_no_backend()
        provenance = DataProvenance.PLACEHOLDER
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

    Behavior: reads REAL pipeline health from etl_pipeline_metrics (latest run
    per pipeline; freshness computed from run_end; synthetic showcase rows
    excluded), tagged ``data_provenance="measured"``. An empty-but-reachable
    table yields an empty roster tagged ``unknown``. Only when the backend is
    unreachable does it fall back to the #429 guard: 503 in production,
    clearly-tagged placeholder in dev/test.

    Returns:
        Pipeline health details (measured from etl_pipeline_metrics)
    """
    import time

    start_time = time.time()

    pipelines, provenance = _fetch_pipeline_health()
    if provenance is None:
        _fail_closed_if_no_backend()
        provenance = DataProvenance.PLACEHOLDER
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

    Behavior: reads the REAL agent roster from agent_registry (availability/tier
    measured) and runtime metrics from audit_chain_entries within the last
    30 days (the configured look-back window). Where an agent has no recent telemetry, its success_rate and
    latency are left null (NOT a fabricated 1.0/0.0) and the response is tagged
    ``data_provenance="partial"``. Only when the backend is unreachable does it
    fall back to the #429 guard: 503 in production, clearly-tagged placeholder in
    dev/test.

    Returns:
        Agent health details (measured roster + measured/null runtime metrics)
    """
    import time

    start_time = time.time()

    agents, provenance = _fetch_agent_health()
    if provenance is None:
        _fail_closed_if_no_backend()
        provenance = DataProvenance.PLACEHOLDER
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
    days: int = Query(
        default=30,
        ge=1,
        le=90,
        description="Window in days for durable history and daily aggregates",
    ),
) -> HealthHistoryResponse:
    """
    Get historical health check records.

    Serves the durable health_check_history table (migration 096): the most
    recent ``limit`` checks within the ``days`` window plus per-UTC-day
    aggregates for trend charts. Falls back to the process-local in-memory
    list ONLY when the durable read fails (DB unreachable) — flagged by
    ``window_days`` being null so consumers can tell the two apart.

    Args:
        limit: Maximum number of raw check records to return
        days: Window in days for the durable read and daily aggregates

    Returns:
        Historical health check data with trend analysis
    """
    durable = _read_history_durable(days, limit)
    if durable is not None:
        checks, daily = durable
        # Average over the WHOLE window (weighted by checks per day), not just
        # the `limit` newest raw rows — it captions the daily trend chart.
        total_n = sum(p.checks_count for p in daily)
        avg_score = sum(p.avg_score * p.checks_count for p in daily) / total_n if total_n else None
        # Trend from daily averages once 3 days exist; before that, from the
        # raw checks (first hours after rollout); "unknown" below 3 points.
        trend = (
            _trend_from_scores([p.avg_score for p in daily])
            if len(daily) >= 3
            else _trend_from_scores([c.overall_health_score for c in checks])
        )
        return HealthHistoryResponse(
            total_checks=total_n if total_n else len(checks),
            checks=checks,
            avg_health_score=avg_score,
            trend=trend,
            daily=daily,
            window_days=days,
        )

    # Fallback: durable read failed — serve the in-memory list (per-worker,
    # reset on restart). daily stays EMPTY and window_days null: repackaging
    # minutes-scale process history as day buckets would fabricate a month.
    history = _health_history[-limit:] if _health_history else []

    checks = [
        HealthHistoryItem(
            check_id=h.check_id,
            timestamp=h.timestamp,
            overall_health_score=h.overall_health_score,
            health_grade=h.health_grade,
            critical_issues_count=len(h.critical_issues),
            data_provenance=h.data_provenance,
        )
        for h in history
    ]

    # No history -> no average (None), NOT a fabricated 0.0 the dashboard would
    # render as a real metric.
    avg_score = sum(h.overall_health_score for h in history) / len(history) if history else None

    return HealthHistoryResponse(
        total_checks=len(history),
        checks=checks,
        avg_health_score=avg_score,
        trend=_trend_from_scores([h.overall_health_score for h in history]),
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


# =============================================================================
# REAL-DATA SOURCES for the per-dimension endpoints
#
# These replace the hardcoded `_get_mock_*_health()` placeholders for the
# user-facing /models, /pipelines, /agents endpoints. They read live tables via
# the service-role Supabase client (the ml_* tables are GRANTed to service_role
# only; `get_supabase()` resolves SUPABASE_SERVICE_ROLE_KEY/SUPABASE_SERVICE_KEY
# per #926, so these reads are not denied 42501). Each returns
# ``(items, provenance)``: MEASURED when every field is sourced, PARTIAL when the
# dimension is real-backed but some sub-fields have no source yet (left
# null/empty — never fabricated), UNKNOWN when no rows exist, and ``None``
# provenance to signal the backend was unreachable (caller fails closed).
# =============================================================================

# Look-back window for agent runtime telemetry (audit_chain_entries).
_AGENT_TELEMETRY_WINDOW_DAYS = 30
# A pipeline is stale if its last successful run is older than this.
_PIPELINE_STALE_HOURS = 48.0


def _health_source_client() -> Any:
    """Service-role Supabase client for health-source reads (or None)."""
    try:
        from src.api.dependencies.supabase_client import get_supabase

        return get_supabase()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"health sources: could not get Supabase client: {e}")
        return None


def _hours_since(ts: Any, now: datetime) -> float:
    """Hours between an ISO timestamp and ``now`` (large sentinel if unparseable)."""
    if not ts:
        return 1e9
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (now - parsed).total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return 1e9


def _map_model_status(health_status: Any) -> ModelStatus:
    s = (health_status or "").strip().lower()
    if s in ("healthy", "ok", "good", "nominal"):
        return ModelStatus.HEALTHY
    if s in ("unhealthy", "critical", "failed", "error", "down"):
        return ModelStatus.UNHEALTHY
    # attention / warning / degraded / unknown / anything else -> degraded
    return ModelStatus.DEGRADED


# agent_registry.agent_tier is a TEXT category enum; this is its numeric tier
# (the same number tier_v2 embeds: 'tier_2_causal' -> 2). Used as the fallback
# when tier_v2 is null (it is nullable in the schema).
_AGENT_TIER_BY_CATEGORY = {
    "coordination": 1,
    "causal_analytics": 2,
    "monitoring": 3,
    "ml_predictions": 4,
    "self_improvement": 5,
}


def _agent_tier_int(row: Dict[str, Any]) -> int:
    """Numeric tier (0-5) from agent_registry. ``tier_v2`` ('tier_2_causal')
    embeds the number; when it is null, map the ``agent_tier`` TEXT category
    ('coordination', 'causal_analytics', ...). Returns 0 only when neither
    resolves (genuinely unknown tier)."""
    tier_v2 = row.get("tier_v2")
    if tier_v2 is not None:
        m = re.search(r"(\d+)", str(tier_v2))
        if m:
            return int(m.group(1))
    category = row.get("agent_tier")
    if category is not None:
        return _AGENT_TIER_BY_CATEGORY.get(str(category).strip().lower(), 0)
    return 0


def _map_pipeline_status(status: Any, freshness_hours: float) -> PipelineStatus:
    s = (status or "").strip().lower()
    if s in ("failed", "error", "failure"):
        return PipelineStatus.FAILED
    if freshness_hours > _PIPELINE_STALE_HOURS:
        return PipelineStatus.STALE
    return PipelineStatus.HEALTHY


def _fail_closed_if_no_backend() -> None:
    """A dimension's real backend is unreachable (fetcher returned None).

    Fail closed in production (503) so fabricated placeholder data is NEVER served
    as real when the source is down. In an explicit dev/test environment, return
    so the caller serves clearly-tagged placeholder (mirrors the #429 import-guard
    policy, extended from 'agent import failed' to 'data source unavailable')."""
    from src.api.utils.agent_import_guard import should_fail_closed_on_import_error

    if should_fail_closed_on_import_error():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "health_source_unavailable",
                "message": (
                    "Health data source is unavailable; refusing to serve "
                    "placeholder data as real in production."
                ),
            },
        )


# The five infrastructure components, mirroring the agent's
# ``ComponentHealthNode.DEFAULT_COMPONENTS`` so /components reports the SAME set
# (database / cache / vector store / API gateway / message queue).
_HEALTH_COMPONENTS = [
    ("database", "/health/db"),
    ("cache", "/health/cache"),
    ("vector_store", "/health/vectors"),
    ("api_gateway", "/health/api"),
    ("message_queue", "/health/queue"),
]


async def _fetch_component_health() -> tuple[List[ComponentHealth], Optional[DataProvenance]]:
    """Real component health: live probes via ``SupabaseHealthClient``.

    A DIRECT measured reader (mirrors ``_fetch_model_health`` / ``_fetch_pipeline_health``
    / ``_fetch_agent_health``), NOT the composite agent. The composite path could
    not feed this endpoint: the agent's ``HealthScoreOutput`` carries the component
    SCORE but never the per-component STATUS list, so ``/components`` always fell
    back to placeholder and the dashboard's Services card rendered 0/0. Here we
    probe each component directly: ``ok`` -> healthy, ``degraded`` -> degraded,
    else -> unhealthy; a probe that RAISES -> ``unknown`` (never fabricated).
    Returns ``None`` provenance ONLY when every probe raised (backend genuinely
    unreachable) so the caller fails closed; otherwise MEASURED (the per-component
    statuses are real live measurements).
    """
    import asyncio

    try:
        from src.agents.health_score.health_client import SupabaseHealthClient
    except Exception as e:  # pragma: no cover - defensive import guard
        logger.warning(f"component health: SupabaseHealthClient import failed ({e})")
        return [], None

    client = SupabaseHealthClient()
    try:
        results = await asyncio.gather(
            *[client.check(endpoint) for _, endpoint in _HEALTH_COMPONENTS],
            return_exceptions=True,
        )
    except Exception as e:
        logger.warning(f"component health: probe gather failed ({e})")
        return [], None
    finally:
        try:
            await client.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    now_iso = datetime.now(timezone.utc).isoformat()
    components: List[ComponentHealth] = []
    measured_any = False
    for (name, _endpoint), res in zip(_HEALTH_COMPONENTS, results, strict=True):
        # NOTE: the real SupabaseHealthClient.check() catches all exceptions
        # internally and ALWAYS returns a dict (a down backend -> {"ok": False}
        # -> UNHEALTHY, measured), so this BaseException branch is DEFENSIVE only
        # (e.g. a test double that raises, or a future raising client). It is the
        # sole path to the ([], None) fail-closed return below.
        if isinstance(res, BaseException):
            components.append(
                ComponentHealth(
                    component_name=name,
                    status=ComponentStatus.UNKNOWN,
                    latency_ms=None,
                    last_check=now_iso,
                    error_message=str(res),
                )
            )
            continue
        measured_any = True
        if res.get("ok"):
            status = ComponentStatus.HEALTHY
        elif res.get("degraded"):
            status = ComponentStatus.DEGRADED
        else:
            status = ComponentStatus.UNHEALTHY
        components.append(
            ComponentHealth(
                component_name=name,
                status=status,
                latency_ms=res.get("latency_ms"),
                last_check=now_iso,
                error_message=res.get("error"),
            )
        )

    if not measured_any:
        # Every probe raised -> the health backend is genuinely unreachable.
        return [], None
    return components, DataProvenance.MEASURED


def _fetch_model_health() -> tuple[List[ModelHealth], Optional[DataProvenance]]:
    """Real model health from ml_model_health_dashboard (production stage).

    Eval metrics (accuracy/auc_roc/f1) are measured: the weekly gold-standard
    retrain writes them to ml_performance_metrics and migration 103 exposes the
    latest value of each named metric on the view (the generic
    latest_metric_value LATERAL often lands on a confusion_matrix/roc_curve
    summary row, which maps to nothing displayable). Serving-side sub-fields
    (latency/predictions_last_24h/error_rate) still have NO source — the models
    are weekly-retrained batch models with no online serving telemetry
    (ml_bentoml_serving_metrics is unwritten) — so they stay null and the
    provenance is PARTIAL. The status/drift/alert signals are genuinely
    measured.
    """
    db = _health_source_client()
    if db is None:
        return [], None
    # The query AND the row->model construction are both inside the try: a bad
    # row (e.g. non-numeric metric) must fail closed (None -> 503/placeholder),
    # never 500 the endpoint.
    try:
        rows = (
            db.table("ml_model_health_dashboard")
            .select(
                "model_id, model_name, model_stage, health_status, latest_metric_value, "
                "primary_metric, has_active_drift, max_drift_severity, performance_degraded, "
                "active_alerts, critical_alerts, is_synthetic, "
                "latest_accuracy, latest_auc_roc, latest_f1"
            )
            # Exclude synthetic experiment artifacts STRUCTURALLY (migration 031
            # exposes is_synthetic on the view). The registry holds 720
            # is_synthetic=true rows under stage IN (production, staging) — the
            # noise that rendered "362/362" with blank accuracy — versus 14 real
            # models (2 production + 12 staging), ALL is_synthetic=false. This is
            # NOT gating a desired capability: the 12 gold-standard models the
            # platform surfaces are is_synthetic=false; we are removing planted
            # artifacts, not real model health.
            .eq("is_synthetic", False)
            # Of the 14 real models, surface the 12 that actually carry a
            # performance metric; the 2 legacy production rows without a metric
            # have nothing to show on a health dashboard.
            .not_.is_("latest_metric_value", "null")
            .execute()
            .data
            or []
        )

        def _num(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        models: List[ModelHealth] = []
        for r in rows:
            # Named latest metrics (migration 103) are the primary source.
            # Fall back to the generic latest_metric_value mapping only when a
            # named column is absent (pre-103 view) — never fabricate.
            acc = _num(r.get("latest_accuracy"))
            auc = _num(r.get("latest_auc_roc"))
            f1 = _num(r.get("latest_f1"))
            generic = _num(r.get("latest_metric_value"))
            primary = (r.get("primary_metric") or "").strip().lower()
            if generic is not None:
                if auc is None and "auc" in primary:
                    auc = generic
                elif acc is None and primary in ("accuracy", "acc"):
                    acc = generic
            models.append(
                ModelHealth(
                    model_id=str(r.get("model_id") or r.get("model_name") or "unknown"),
                    model_name=str(r.get("model_name") or "unknown"),
                    accuracy=acc,
                    auc_roc=auc,
                    f1_score=f1,
                    # latencies/predictions_last_24h/error_rate have NO source
                    # (no online serving telemetry) -> left null.
                    status=_map_model_status(r.get("health_status")),
                )
            )
    except Exception as e:
        logger.warning(f"model health: live query/build failed ({e})")
        return [], None
    if not models:
        return [], DataProvenance.UNKNOWN
    # ALWAYS PARTIAL: the dashboard view sources status/drift and the latest
    # eval metrics (accuracy/auc_roc/f1), but latency/predictions_last_24h/
    # error_rate have no serving-telemetry source, so the dimension is never
    # fully measured. Never MEASURED here -> we never present those
    # null/structural fields as real.
    return models, DataProvenance.PARTIAL


# =============================================================================
# NAMED EVALUATION METRICS (#1450)
#
# ``ml_model_health_dashboard`` answers "is this model HEALTHY?" — status, drift,
# alerts, and (since migration 103) the latest accuracy/auc_roc/f1. It cannot
# answer "what is this model's ROC-AUC, calibration and Brier, on which cohort,
# as of when?": the view exposes neither calibration_slope/brier_score nor the
# model version, evaluation cohort or measurement date. Those live one level
# down, in ``ml_performance_metrics`` (+ ``ml_model_registry`` for the version),
# which is what these two readers read. They are a DIFFERENT question over the
# SAME source of truth, not a second copy of the dashboard reader.
# =============================================================================

# The scalar evaluation metrics the platform records per model. Curve payloads
# (``roc_curve`` / ``confusion_matrix``, written with source ``holdout_curve``)
# are deliberately excluded — they are not scalar quality metrics.
_EVAL_METRIC_NAMES: Tuple[str, ...] = (
    "auc_roc",
    "pr_auc",
    "brier_score",
    "calibration_slope",
    "accuracy",
    "f1",
    "precision",
    "recall",
)

# Per-model row budget for the eval-metric read. One evaluation EVENT contributes
# at most ``len(_EVAL_METRIC_NAMES)`` rows, so ordering by measured_at DESC and
# taking this many rows always contains the latest event in full — the bound can
# never produce a false "no metrics recorded". (A single global read with a flat
# limit CAN: measured against the live table on 2026-08-03, a 1500-row bulk read
# truncated before reaching the oldest model's latest event.)
_EVAL_ROWS_PER_MODEL = len(_EVAL_METRIC_NAMES) * 4


def _fetch_model_registry_facts() -> Dict[str, Dict[str, Any]]:
    """Registry facts (version + stage) for the real production/staging models.

    Keyed by ``model_id``. Returns ``{}`` when the backend is unreachable — the
    caller degrades to "version not recorded", never to a fabricated version.
    """
    db = _health_source_client()
    if db is None:
        return {}
    try:
        rows = (
            db.table("ml_model_registry")
            .select("id, model_name, model_version, stage")
            # Same structural exclusions as _fetch_model_health: synthetic
            # experiment artifacts are not real models.
            .eq("is_synthetic", False)
            .in_("stage", ["production", "staging"])
            .execute()
            .data
            or []
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"model registry facts: live query failed ({e})")
        return {}
    facts: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        model_id = str(r.get("id") or "")
        if not model_id:
            continue
        stage = r.get("stage")
        facts[model_id] = {
            "model_version": (str(r["model_version"]) if r.get("model_version") else None),
            "model_stage": (str(stage) if stage else None),
        }
    return facts


def _fetch_model_eval_metrics(model_id: str) -> Dict[str, Any]:
    """The model's LATEST evaluation event, as one coherent set of metrics.

    Returns ``{"eval_metrics": {name: value}, "eval_cohort", "eval_sample_size",
    "eval_as_of"}`` or ``{}`` when nothing is recorded / the backend is down.

    "One coherent event" is the honesty property: ``ml_performance_metrics``
    holds a monthly walk-forward backtest series (``source='backtest_wf'``)
    ALONGSIDE the held-out gold-standard evaluation (``source='holdout'``).
    Taking each metric's own latest row independently could pair a fresh holdout
    ROC-AUC with a stale backtest Brier and present the mixture as one
    evaluation. Instead we pick the single latest event — ordered by
    ``(measured_at, sample_size, source)``, which reproduces the value the
    dashboard view's ``ORDER BY measured_at DESC LIMIT 1`` LATERALs return while
    resolving the same-day holdout/backtest tie deterministically toward the
    larger cohort — and report only the metrics measured in it.
    """
    db = _health_source_client()
    if db is None:
        return {}
    try:
        rows = (
            db.table("ml_performance_metrics")
            .select("metric_name, metric_value, sample_size, measured_at, source, data_split")
            .eq("model_id", model_id)
            .in_("metric_name", list(_EVAL_METRIC_NAMES))
            .order("measured_at", desc=True)
            .limit(_EVAL_ROWS_PER_MODEL)
            .execute()
            .data
            or []
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"model eval metrics: live query failed for {model_id} ({e})")
        return {}
    if not rows:
        return {}

    def _event_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
        sample_size = row.get("sample_size")
        return (
            str(row.get("measured_at") or ""),
            int(sample_size) if isinstance(sample_size, (int, float)) else -1,
            str(row.get("source") or ""),
        )

    try:
        latest = max(_event_key(r) for r in rows)
        event_rows = [r for r in rows if _event_key(r) == latest]
        metrics: Dict[str, float] = {}
        for r in event_rows:
            name = str(r.get("metric_name") or "")
            value = r.get("metric_value")
            if not name or value is None:
                continue
            metrics[name] = float(value)
    except (TypeError, ValueError) as e:
        # A malformed row must fail closed to "nothing recorded", never to a
        # partially-parsed number presented as a measurement.
        logger.warning(f"model eval metrics: unparseable row for {model_id} ({e})")
        return {}
    if not metrics:
        return {}
    measured_at, sample_size, source = latest
    return {
        "eval_metrics": metrics,
        "eval_cohort": source or None,
        "eval_sample_size": sample_size if sample_size >= 0 else None,
        "eval_as_of": measured_at or None,
    }


def _fetch_pipeline_health() -> tuple[List[PipelineHealth], Optional[DataProvenance]]:
    """Real pipeline health: latest run per non-synthetic pipeline from
    etl_pipeline_metrics."""
    db = _health_source_client()
    if db is None:
        return [], None
    # Query + build both inside the try so a malformed row fails closed, not 500.
    try:
        rows = (
            db.table("etl_pipeline_metrics")
            .select("pipeline_name, run_start, run_end, records_processed, status, created_at")
            # Exclude synthetic showcase rows, mirroring _fetch_model_health. The
            # table's only pipeline (rwd_ingest, 50k rows, ALL is_synthetic=true)
            # is planted by coverage_tables_generator for the TTR KPI and goes
            # STALE (>48h) between reseeds — scoring the dimension at 50% as if a
            # real pipeline were failing. No real ETL telemetry writes this table
            # yet; an empty result is the honest reading (UNKNOWN provenance, and
            # the pipeline node scores an empty-but-reachable fleet as measured
            # healthy).
            .eq("is_synthetic", False)
            .order("run_end", desc=True)
            .limit(2000)
            .execute()
            .data
            or []
        )
        now = datetime.now(timezone.utc)
        latest: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            name = r.get("pipeline_name") or "unknown"
            if name not in latest:
                latest[name] = r
        pipelines: List[PipelineHealth] = []
        for name, r in latest.items():
            run_end = r.get("run_end") or r.get("created_at")
            freshness = _hours_since(run_end, now)
            try:
                rows_processed = int(r.get("records_processed") or 0)
            except (ValueError, TypeError):
                rows_processed = 0
            pipelines.append(
                PipelineHealth(
                    pipeline_name=str(name),
                    last_run=str(r.get("run_start") or run_end or ""),
                    last_success=str(run_end or ""),
                    rows_processed=rows_processed,
                    freshness_hours=round(freshness, 2),
                    status=_map_pipeline_status(r.get("status"), freshness),
                )
            )
    except Exception as e:
        logger.warning(f"pipeline health: live query/build failed ({e})")
        return [], None
    if not pipelines:
        return [], DataProvenance.UNKNOWN
    return pipelines, DataProvenance.MEASURED


def _fetch_agent_health() -> tuple[List[AgentHealth], Optional[DataProvenance]]:
    """Real agent roster from agent_registry + runtime metrics from
    audit_chain_entries (last ``_AGENT_TELEMETRY_WINDOW_DAYS`` days).

    Availability/tier are measured from the registry. Runtime metrics
    (success_rate/latency/invocations) are measured only where recent telemetry
    exists; otherwise they are left null (NOT a fabricated 1.0/0.0) and the
    provenance is PARTIAL.
    """
    db = _health_source_client()
    if db is None:
        return [], None
    try:
        roster = (
            db.table("agent_registry")
            .select("agent_name, agent_tier, tier_v2, is_active")
            .execute()
            .data
            or []
        )
    except Exception as e:
        logger.warning(f"agent health: roster query failed ({e})")
        return [], None
    if not roster:
        return [], DataProvenance.UNKNOWN

    start = (datetime.now(timezone.utc) - timedelta(days=_AGENT_TELEMETRY_WINDOW_DAYS)).isoformat()
    try:
        telemetry = (
            db.table("audit_chain_entries")
            .select("agent_name, duration_ms, validation_passed, created_at")
            .gte("created_at", start)
            .execute()
            .data
            or []
        )
    except Exception as e:
        # Roster is real; telemetry is just unavailable -> partial with null metrics.
        logger.warning(f"agent health: telemetry query failed ({e})")
        telemetry = []
    # invocations_24h must mean the last 24h (its name), distinct from the wider
    # window used to aggregate success_rate/latency over more samples.
    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    try:
        agg: Dict[str, Dict[str, Any]] = {}
        for entry in telemetry:
            name = entry.get("agent_name")
            if not name:
                continue
            d = agg.setdefault(
                name, {"latencies": [], "ok": 0, "total": 0, "last": None, "count_24h": 0}
            )
            d["total"] += 1
            dur = entry.get("duration_ms")
            if isinstance(dur, (int, float)) and dur > 0:
                d["latencies"].append(dur)
            if entry.get("validation_passed") is not False:  # True or None -> counted ok
                d["ok"] += 1
            ts = entry.get("created_at")
            if ts:
                if d["last"] is None or str(ts) > str(d["last"]):
                    d["last"] = ts
                if str(ts) >= cutoff_24h:
                    d["count_24h"] += 1

        agents: List[AgentHealth] = []
        missing_telemetry = False
        for r in roster:
            name = str(r.get("agent_name") or "unknown")
            a = agg.get(name)
            lat = a["latencies"] if a else []
            if a and a["total"] > 0:
                avg_latency = int(sum(lat) / len(lat)) if lat else None
                # Fully sourced only if BOTH the rate and a latency are measured;
                # telemetry rows without any valid duration leave latency null ->
                # the agent is not fully sourced -> dimension is PARTIAL.
                if avg_latency is None:
                    missing_telemetry = True
                agents.append(
                    AgentHealth(
                        agent_name=name,
                        tier=_agent_tier_int(r),
                        available=bool(r.get("is_active")),
                        avg_latency_ms=avg_latency,
                        success_rate=round(a["ok"] / a["total"], 4),
                        last_invocation=str(a["last"]) if a["last"] else None,
                        invocations_24h=a["count_24h"],
                    )
                )
            else:
                # Registered, but no recent telemetry: availability is real, runtime
                # metrics are UNMEASURED (null), not a fabricated success rate.
                missing_telemetry = True
                agents.append(
                    AgentHealth(
                        agent_name=name,
                        tier=_agent_tier_int(r),
                        available=bool(r.get("is_active")),
                        avg_latency_ms=None,
                        success_rate=None,
                        last_invocation=None,
                        invocations_24h=0,
                    )
                )
    except Exception as e:
        logger.warning(f"agent health: build failed ({e})")
        return [], None
    # MEASURED only if EVERY roster agent had recent telemetry AND a measured
    # latency; any null runtime field -> PARTIAL (honest, not fabricated).
    prov = DataProvenance.PARTIAL if missing_telemetry else DataProvenance.MEASURED
    return agents, prov


# =============================================================================
# STORE ADAPTERS — bridge the REAL per-dimension readers into the agent graph
#
# The full health SCORE (/health-score/full) is computed by HealthScoreAgent
# whose model/pipeline/agent nodes consume *stores* satisfying the node Protocols
# (get_active_models/get_model_metrics, get_all_pipelines/get_pipeline_status,
# get_all_agents/get_agent_metrics). Before this wiring the route constructed the
# agent with only a health_client, so those three dimensions fail-closed to null.
#
# These adapters REUSE the already-wired _fetch_model_health / _fetch_pipeline_health
# / _fetch_agent_health readers (single source of truth: the same live tables the
# /models, /pipelines, /agents endpoints read) and translate their output into the
# node Protocol shape. They carry the readers' AUTHORITATIVE per-item status through
# the metrics dict so the node's aggregate scoring reproduces the per-dimension
# endpoint scores exactly — no duplicated status logic, no fabrication.
#
# Each adapter raises ``HealthSourceUnavailable`` when its reader signals the
# backend is unreachable (provenance is None), so the node fails closed
# (model_health_measured=False -> honest null) instead of fabricating a healthy
# score. An empty-but-reachable table (provenance UNKNOWN) yields an empty roster,
# which the node treats as a measured idle fleet (score 1.0) — that is a real
# measurement, distinct from "backend down".
# =============================================================================


class HealthSourceUnavailable(RuntimeError):
    """Raised by a store adapter when its real backend is unreachable.

    The agent's per-dimension node catches it, marks the dimension UNMEASURED
    (``*_health_measured=False``) and the composer emits an honest null/partial
    — never a fabricated healthy score."""


# How long an adapter may serve its cached reader result (#1450).
#
# The cache exists so ONE health check does not re-read the table once per model
# (the node's get_active_models -> get_model_metrics(id) access pattern). The
# REST route builds a fresh adapter per request, so the original load-once flag
# was bounded by the request. The CHAT path holds a single HealthScoreAgent for
# the process lifetime (the ``cognitive.get_orchestrator`` singleton), where
# load-once would pin the first reading forever and report a retrained model's
# old numbers as current. A TTL keeps the intra-check single-read intent while
# bounding staleness.
_ADAPTER_CACHE_TTL_SECONDS = 300.0


class _TTLCachedStoreAdapter:
    """Shared TTL-cache bookkeeping for the three real-table store adapters."""

    def __init__(self) -> None:
        self._loaded_at: Optional[float] = None
        # The reader's source provenance, captured at load so the route can
        # downgrade the composite to "partial" when sub-fields are unsourced
        # (PARTIAL) even though the dimension SCORE is a real measurement.
        self.provenance: Optional[DataProvenance] = None

    def _cache_is_stale(self) -> bool:
        """True when the adapter must (re-)read its backend."""
        if self._loaded_at is None:
            return True
        return (time.monotonic() - self._loaded_at) >= _ADAPTER_CACHE_TTL_SECONDS

    def _mark_loaded(self) -> None:
        self._loaded_at = time.monotonic()


class _ModelMetricsStoreAdapter(_TTLCachedStoreAdapter):
    """MetricsStore adapter backed by ``_fetch_model_health`` (ml_model_health_dashboard).

    Calls the reader ONCE per cache window and serves the node's
    get_active_models -> get_model_metrics(model_id) access pattern from it,
    carrying the reader's authoritative ``status`` so the node scores
    identically to /models.

    #1450: it ALSO carries the model's named evaluation metrics (ROC-AUC,
    calibration slope, Brier, ...) together with the model version, evaluation
    cohort and as-of date, so a chat question naming a metric can be answered
    with the measurement instead of a composite grade. Those come from
    ``_fetch_model_eval_metrics`` / ``_fetch_model_registry_facts`` — a
    different question over the same tables, read lazily per model and cached
    for the same window. When nothing is recorded the fields stay absent/None;
    the consumer must say "not recorded", never substitute a number."""

    def __init__(self) -> None:
        super().__init__()
        self._by_id: Dict[str, ModelHealth] = {}
        self._registry_facts: Dict[str, Dict[str, Any]] = {}
        self._eval_by_id: Dict[str, Dict[str, Any]] = {}

    def _load(self) -> None:
        if not self._cache_is_stale():
            return
        models, provenance = _fetch_model_health()
        if provenance is None:
            raise HealthSourceUnavailable("model health backend unreachable")
        self._by_id = {m.model_id: m for m in models}
        self.provenance = provenance
        # Registry facts are one small read for the whole fleet; eval metrics are
        # read lazily per model (see _eval_detail) since the node only asks for
        # the models it actually scores.
        self._registry_facts = _fetch_model_registry_facts()
        self._eval_by_id = {}
        self._mark_loaded()

    def _eval_detail(self, model_id: str) -> Dict[str, Any]:
        """Latest evaluation event for ``model_id`` (cached for the window)."""
        if model_id not in self._eval_by_id:
            self._eval_by_id[model_id] = _fetch_model_eval_metrics(model_id)
        return self._eval_by_id[model_id]

    async def get_active_models(self) -> List[str]:
        self._load()
        return list(self._by_id.keys())

    async def get_model_metrics(self, model_id: str, time_window: str) -> Dict[str, Any]:
        self._load()
        m = self._by_id.get(model_id)
        if m is None:
            return {"status": "unhealthy"}
        # Carry the reader's authoritative status (mapped from the dashboard
        # health_status) so the node does NOT re-derive a divergent one. Sub-fields
        # that have no source remain null/absent — never fabricated.
        # model_name rides along so composer-issued alerts can name the model
        # instead of printing a bare registry UUID.
        facts = self._registry_facts.get(model_id) or {}
        detail = self._eval_detail(model_id)
        return {
            "status": m.status.value,
            "model_name": m.model_name,
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1_score,
            "auc_roc": m.auc_roc,
            "latency_p50": m.prediction_latency_p50_ms,
            "latency_p99": m.prediction_latency_p99_ms,
            "prediction_count": m.predictions_last_24h,
            "error_rate": m.error_rate,
            # #1450 named-metric detail. All Optional: absent/None means NOT
            # RECORDED, which the consumer must state honestly.
            "model_version": facts.get("model_version"),
            "model_stage": facts.get("model_stage"),
            "eval_metrics": detail.get("eval_metrics") or {},
            "eval_cohort": detail.get("eval_cohort"),
            "eval_sample_size": detail.get("eval_sample_size"),
            "eval_as_of": detail.get("eval_as_of"),
        }


class _PipelineStoreAdapter(_TTLCachedStoreAdapter):
    """PipelineStore adapter backed by ``_fetch_pipeline_health`` (etl_pipeline_metrics)."""

    def __init__(self) -> None:
        super().__init__()
        self._by_name: Dict[str, PipelineHealth] = {}

    def _load(self) -> None:
        if not self._cache_is_stale():
            return
        pipelines, provenance = _fetch_pipeline_health()
        if provenance is None:
            raise HealthSourceUnavailable("pipeline health backend unreachable")
        self._by_name = {p.pipeline_name: p for p in pipelines}
        self.provenance = provenance
        self._mark_loaded()

    async def get_all_pipelines(self) -> List[str]:
        self._load()
        return list(self._by_name.keys())

    async def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        self._load()
        p = self._by_name.get(pipeline_name)
        if p is None:
            return {"status": "failed", "failed": True}
        # Carry the reader's authoritative status AND its already-computed
        # freshness_hours so the node honors both instead of recomputing freshness
        # from timestamps (which yields a -1 sentinel when a row lacks a usable
        # run_end, then formats as "(-1.0 hours)" in diagnosis — a false reading).
        return {
            "status": p.status.value,
            "failed": p.status == PipelineStatus.FAILED,
            "last_run": p.last_run,
            "last_success": p.last_success,
            "rows_processed": p.rows_processed,
            "freshness_hours": p.freshness_hours,
        }


class _AgentRegistryAdapter(_TTLCachedStoreAdapter):
    """AgentRegistry adapter backed by ``_fetch_agent_health`` (agent_registry +
    audit_chain_entries). The reader already measures availability and (where
    telemetry exists) success_rate/latency; unmeasured runtime metrics stay null."""

    def __init__(self) -> None:
        super().__init__()
        self._by_name: Dict[str, AgentHealth] = {}

    def _load(self) -> None:
        if not self._cache_is_stale():
            return
        agents, provenance = _fetch_agent_health()
        if provenance is None:
            raise HealthSourceUnavailable("agent health backend unreachable")
        self._by_name = {a.agent_name: a for a in agents}
        self.provenance = provenance
        self._mark_loaded()

    async def get_all_agents(self) -> List[Dict[str, Any]]:
        self._load()
        return [{"name": a.agent_name, "tier": a.tier} for a in self._by_name.values()]

    async def get_agent_metrics(self, agent_name: str) -> Dict[str, Any]:
        self._load()
        a = self._by_name.get(agent_name)
        if a is None:
            return {"available": False, "success_rate": None, "avg_latency_ms": None}
        # Pass the reader's HONEST values through. success_rate/avg_latency_ms are
        # None when the agent has no recent telemetry (UNMEASURED) — NOT fabricated
        # to 1.0/0. The agent node treats a None success_rate as "available =>
        # not penalized" (matching the /agents endpoint, which scores on
        # availability), so the score is real without inventing a measurement.
        return {
            "available": a.available,
            "success_rate": a.success_rate,
            "avg_latency_ms": a.avg_latency_ms,
            "last_invocation": a.last_invocation or "",
        }


def _build_real_health_stores() -> tuple[
    "_ModelMetricsStoreAdapter", "_PipelineStoreAdapter", "_AgentRegistryAdapter"
]:
    """Construct the three real-table store adapters for the full health graph.

    Returns ``(metrics_store, pipeline_store, agent_registry)``. Cheap to build
    (no I/O until the node first calls them); each adapter reads its live table
    lazily and fails closed (UNMEASURED -> honest null) if the backend is down."""
    return (
        _ModelMetricsStoreAdapter(),
        _PipelineStoreAdapter(),
        _AgentRegistryAdapter(),
    )


def _reconcile_full_provenance(composite: str, *adapters: Any) -> str:
    """Downgrade a "measured" composite to "partial" when any wired source was PARTIAL.

    The score composer reports "measured" once all four dimension SCORES are real.
    But a dimension's SCORE can be a genuine measurement while its underlying reader
    is PARTIAL — e.g. model status is sourced (score is real) yet accuracy/latency
    sub-fields are unsourced, and agent availability is sourced yet runtime telemetry
    is null. Surfacing "measured" then would mislead a consumer comparing /full to
    /models or /agents (which honestly report "partial"). So: if the composer says
    "measured" but any loaded adapter saw PARTIAL, report "partial". "unknown"/
    "partial" composites are left untouched (already conservative)."""
    if composite != "measured":
        return composite
    for adapter in adapters:
        if getattr(adapter, "provenance", None) == DataProvenance.PARTIAL:
            return "partial"
    return composite


async def _execute_health_check(scope: CheckScope) -> HealthScoreResponse:
    """Execute health check using Health Score agent."""
    import time

    start_time = time.time()

    try:
        # Try to use the actual Health Score agent.
        # F1: wire the REAL SupabaseHealthClient so component health is genuinely
        # measured (not the fail-open mock path).
        # Wire the REAL model/pipeline/agent stores so the full health SCORE
        # computes those three dimensions from the same live tables the
        # /models, /pipelines, /agents endpoints already read
        # (ml_model_health_dashboard / etl_pipeline_metrics / agent_registry +
        # audit_chain_entries). Each store reuses the route's _fetch_* readers
        # (DRY — single source of truth) and fails CLOSED to an honest null when
        # its backend is unreachable, never a fabricated healthy score. The QUICK
        # graph is component-only by design and ignores these stores, so /quick
        # legitimately keeps model/pipeline/agent null for its <1s budget.
        from src.agents.health_score import HealthScoreAgent, SupabaseHealthClient

        metrics_store, pipeline_store, agent_registry = _build_real_health_stores()
        agent = HealthScoreAgent(
            health_client=SupabaseHealthClient(),
            metrics_store=metrics_store,
            pipeline_store=pipeline_store,
            agent_registry=agent_registry,
        )

        if scope == CheckScope.QUICK:
            result = await agent.quick_check()
        else:
            result = await agent.check_health(scope=scope.value)

        # Honesty reconciliation: the composer reports "measured" once all four
        # dimension SCORES are real, but the model/agent readers are PARTIAL
        # (unsourced sub-fields / null telemetry). Downgrade to "partial" so /full
        # never claims a fuller provenance than the per-dimension endpoints. QUICK
        # is component-only and never touched these stores (adapters unloaded ->
        # provenance None), so this is a no-op there.
        reconciled_provenance = _reconcile_full_provenance(
            result.data_provenance, metrics_store, pipeline_store, agent_registry
        )

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
            # F1: propagate the agent's ACTUAL provenance, reconciled with the
            # adapter source provenance so /full never overclaims vs /models|/agents.
            data_provenance=reconciled_provenance,
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
        # This is the dev-offline ImportError fallback (#429): the scores above
        # are hardcoded sample data, NOT measured. Tag PLACEHOLDER explicitly so
        # it is never mislabeled "unknown" and consumers (the dashboard chart)
        # can refuse to render it as real. Production fails closed (503) before
        # reaching here.
        data_provenance=DataProvenance.PLACEHOLDER.value,
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
