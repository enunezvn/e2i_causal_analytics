"""
E2I Analytics & Metrics API
============================

FastAPI endpoints for agent performance analytics and metrics dashboards.

Provides:
- Query execution metrics over time
- Agent latency breakdown and percentiles
- Success/failure rates per agent
- Historical trends for observability

Integration Points:
- audit_chain_entries table for historical data
- CopilotKit dispatch_info for real-time metrics
- Prometheus metrics (optional)

Author: E2I Causal Analytics Team
Version: 4.3.0
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import get_current_user, require_auth
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class TimeSeriesPoint(BaseModel):
    """Single point in a time series."""

    timestamp: datetime
    value: float
    label: Optional[str] = None


class AgentMetrics(BaseModel):
    """Performance metrics for a single agent."""

    agent_name: str
    agent_tier: int
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    success_rate: float = 0.0

    # Latency metrics (in milliseconds). None == UNMEASURED (the agent has audit
    # entries in the window but none carried a real duration_ms), DISTINCT from a
    # measured 0ms. The per-agent table renders "—" for null rather than a fake
    # "0ms". Same honest-null convention as QueryMetricsSummary / AgentHealth.
    avg_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None

    # Confidence metrics
    avg_confidence: Optional[float] = None


class LatencyBreakdown(BaseModel):
    """Latency breakdown by processing stage."""

    classification_ms: float = 0.0
    rag_retrieval_ms: float = 0.0
    routing_ms: float = 0.0
    agent_dispatch_ms: float = 0.0
    synthesis_ms: float = 0.0
    # None == UNMEASURED (no timed entries), distinct from a measured 0ms total.
    # Sourced from summary.avg_latency_ms which is now nullable.
    total_ms: Optional[float] = None


class QueryMetricsSummary(BaseModel):
    """Summary of query execution metrics."""

    period_start: datetime
    period_end: datetime
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate: float = 0.0

    # Latency summary. ``None`` == UNMEASURED (no audit entry in the window
    # carried a duration_ms), which is DISTINCT from a real measured 0ms. The
    # agent graphs only began emitting per-node duration_ms once instrumented
    # (audited_node); genesis-only windows have no timed entries, so reporting
    # 0.0 here rendered a fake "instant" 0ms in the UI. Honest null lets the
    # frontend show an em-dash ("—") instead. Mirrors the AgentHealth convention
    # (avg_latency_ms nullable == unmeasured, not zero).
    avg_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None

    # Intent distribution
    intent_distribution: Dict[str, int] = Field(default_factory=dict)

    # Agent usage
    top_agents: List[str] = Field(default_factory=list)


class AnalyticsDashboardResponse(BaseModel):
    """Complete analytics dashboard response."""

    summary: QueryMetricsSummary
    agent_metrics: List[AgentMetrics]
    latency_trend: List[TimeSeriesPoint]
    query_volume_trend: List[TimeSeriesPoint]
    latency_breakdown: LatencyBreakdown
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)


class AgentPerformanceTrend(BaseModel):
    """Performance trend for a specific agent over time."""

    agent_name: str
    data_points: List[TimeSeriesPoint]
    period: str  # e.g., "7d", "30d"


class TierMetricsItem(BaseModel):
    """Per-tier performance, aggregated from audit_chain_entries."""

    tier: int = Field(..., ge=0, le=5, description="Agent tier (0-5)")
    tasks_completed: int = Field(0, description="Non-poller agent actions in window")
    # ``None`` == UNMEASURED (no timed non-poller row in this tier), DISTINCT from
    # a measured 0ms — the same honest-null convention as QueryMetricsSummary.
    avg_response_time_ms: Optional[float] = Field(
        None, description="Mean duration over timed rows; None if none timed"
    )
    # Intentionally ``None``: validation_passed is recorded for too few rows to
    # compute a representative per-tier success rate, so the UI shows "—" rather
    # than a misleading number off a tiny explicit-validation sample.
    success_rate: Optional[float] = Field(
        None, description="Always None: per-tier success is not reliably recorded"
    )


class TierMetricsResponse(BaseModel):
    """Per-tier performance metrics for all six tiers (0-5)."""

    tiers: List[TierMetricsItem]
    window_hours: int = Field(..., description="Look-back window in hours")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile from a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


# Genesis action_type written once per workflow run by create_workflow_initializer
# (and causal_impact). One workflow == one user-facing "query".
_WORKFLOW_GENESIS_ACTION = "workflow_start"


def _count_query_volume(entries: List[Dict[str, Any]]) -> tuple[int, int]:
    """Return (total_queries, successful_queries) counted at the WORKFLOW level.

    A "query" is one agent-workflow run, not one audit-chain entry. Each run
    writes exactly one ``workflow_start`` genesis entry plus N per-node entries
    (after Problem-B instrumentation, N grew from ~0 to ~3-8). Counting raw
    entries as queries would inflate volume several-fold once agents emit timed
    nodes, so we count distinct workflows instead.

    Preference order for identifying a workflow:
      1. distinct ``workflow_id`` (authoritative, when the column is selected)
      2. count of ``workflow_start`` genesis entries (fallback)

    A workflow is counted "successful" unless any of its entries failed
    (``validation_passed is False`` or an ``*_error`` action_type). When grouping
    by workflow_id is unavailable, success is approximated entry-wise on the
    genesis rows, which is honest for the genesis-only legacy windows.

    Mixed windows are partitioned: rows WITH a workflow_id are counted by
    distinct workflow; rows WITHOUT one (legacy pre-instrumentation rows that may
    coexist in the same period) are added separately via the genesis-row
    fallback — so a transition window neither double-counts nor drops the legacy
    rows.
    """
    with_id = [e for e in entries if e.get("workflow_id")]
    without_id = [e for e in entries if not e.get("workflow_id")]

    total = 0
    successful = 0

    # Partition 1: rows with a workflow_id -> count distinct workflows.
    if with_id:
        failed_ids: set = set()
        all_ids: set = set()
        for e in with_id:
            wid = e.get("workflow_id")
            all_ids.add(wid)
            action = e.get("action_type") or ""
            if e.get("validation_passed") is False or action.endswith("_error"):
                failed_ids.add(wid)
        total += len(all_ids)
        successful += len(all_ids) - len(failed_ids)

    # Partition 2: rows without a workflow_id -> genesis-row fallback (then raw
    # entries only if no genesis present, so a window with clear activity never
    # reports 0).
    if without_id:
        genesis = [e for e in without_id if e.get("action_type") == _WORKFLOW_GENESIS_ACTION]
        source = genesis if genesis else without_id
        total += len(source)
        successful += sum(1 for e in source if e.get("validation_passed") is not False)

    return total, successful


def _get_supabase_client():
    """Get Supabase client for analytics queries."""
    try:
        from src.api.dependencies.supabase_client import get_supabase

        return get_supabase()
    except Exception as e:
        logger.warning(f"Could not get Supabase client: {e}")
        return None


async def _fetch_audit_metrics(
    db,
    start_date: datetime,
    end_date: datetime,
    brand: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch aggregated metrics from audit_chain_entries."""
    try:
        query = (
            db.table("audit_chain_entries")
            .select(
                "workflow_id, agent_name, agent_tier, duration_ms, validation_passed, confidence_score, created_at, action_type"
            )
            .gte("created_at", start_date.isoformat())
            .lte("created_at", end_date.isoformat())
        )

        if brand:
            query = query.eq("brand", brand)

        result = query.execute()
        return {"success": True, "data": result.data or []}

    except Exception as e:
        logger.error(f"Failed to fetch audit metrics: {e}")
        return {"success": False, "data": [], "error": str(e)}


def _aggregate_agent_metrics(entries: List[Dict[str, Any]]) -> List[AgentMetrics]:
    """Aggregate entries into per-agent metrics."""
    agent_data: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        agent_name = entry.get("agent_name", "unknown")
        if agent_name not in agent_data:
            agent_data[agent_name] = {
                "agent_tier": entry.get("agent_tier", 0),
                "latencies": [],
                "confidences": [],
                "successful": 0,
                "failed": 0,
                "total": 0,
            }

        data = agent_data[agent_name]
        data["total"] += 1

        # Track latencies
        duration = entry.get("duration_ms")
        if duration is not None and duration > 0:
            data["latencies"].append(duration)

        # Track validation status
        validation = entry.get("validation_passed")
        if validation is True:
            data["successful"] += 1
        elif validation is False:
            data["failed"] += 1
        else:
            # No explicit validation, count as successful if action completed
            data["successful"] += 1

        # Track confidence
        confidence = entry.get("confidence_score")
        if confidence is not None:
            data["confidences"].append(confidence)

    # Build AgentMetrics list
    metrics = []
    for agent_name, data in agent_data.items():
        latencies = data["latencies"]
        confidences = data["confidences"]
        total = data["total"]
        successful = data["successful"]

        metrics.append(
            AgentMetrics(
                agent_name=agent_name,
                agent_tier=data["agent_tier"],
                total_invocations=total,
                successful_invocations=successful,
                failed_invocations=data["failed"],
                success_rate=round(successful / total * 100, 2) if total > 0 else 0.0,
                # No timed entries for this agent -> latency UNMEASURED (None),
                # not a fabricated 0ms.
                avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else None,
                p50_latency_ms=round(_calculate_percentile(latencies, 50), 2)
                if latencies
                else None,
                p95_latency_ms=round(_calculate_percentile(latencies, 95), 2)
                if latencies
                else None,
                p99_latency_ms=round(_calculate_percentile(latencies, 99), 2)
                if latencies
                else None,
                min_latency_ms=round(min(latencies), 2) if latencies else None,
                max_latency_ms=round(max(latencies), 2) if latencies else None,
                avg_confidence=round(sum(confidences) / len(confidences), 3)
                if confidences
                else None,
            )
        )

    # Sort by total invocations descending
    return sorted(metrics, key=lambda m: m.total_invocations, reverse=True)


def _build_time_series(
    entries: List[Dict[str, Any]],
    field: str,
    interval_hours: int = 1,
) -> List[TimeSeriesPoint]:
    """Build time series from entries."""
    if not entries:
        return []

    # Group by time interval
    buckets: Dict[datetime, List[float]] = {}

    for entry in entries:
        value = entry.get(field)
        # Crash-safe parse: one malformed created_at drops that point instead of
        # 500-ing the dashboard (shared with _build_volume_series via _parse_ts).
        ts = _parse_ts(entry.get("created_at"))

        if ts is not None and value is not None:
            # Round to interval
            bucket_ts = ts.replace(
                minute=0,
                second=0,
                microsecond=0,
                hour=(ts.hour // interval_hours) * interval_hours,
            )

            if bucket_ts not in buckets:
                buckets[bucket_ts] = []
            buckets[bucket_ts].append(float(value))

    # Calculate averages per bucket
    points = []
    for ts, values in sorted(buckets.items()):
        avg_value = sum(values) / len(values) if values else 0.0
        points.append(
            TimeSeriesPoint(
                timestamp=ts,
                value=round(avg_value, 2),
            )
        )

    return points


def _parse_ts(created_at: Any) -> Optional[datetime]:
    """Parse an audit-entry created_at (ISO string or datetime) -> datetime.

    Defensive: a single malformed ``created_at`` must drop that one point, NOT
    500 the whole dashboard. Returns None for anything unparseable.
    """
    if not created_at:
        return None
    if isinstance(created_at, datetime):
        return created_at
    if isinstance(created_at, str):
        try:
            return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    return None


def _build_volume_series(
    entries: List[Dict[str, Any]],
    interval_hours: int = 1,
) -> List[TimeSeriesPoint]:
    """Build query VOLUME time series, counted at the WORKFLOW level.

    Each workflow run is one point of volume, bucketed by a single representative
    timestamp (the ``workflow_start`` genesis entry's time when available, else
    the earliest entry for that workflow). Counting raw audit entries here would
    inflate the trend several-fold once per-node instrumentation writes multiple
    entries per run — the same regression fixed for ``total_queries`` in
    ``_count_query_volume``. Mixed windows are partitioned identically to
    ``_count_query_volume``: workflow-id rows by distinct workflow, no-id rows via
    the genesis-rows fallback — so legacy rows in a transition window are neither
    dropped nor double-counted.
    """
    if not entries:
        return []

    with_id = [e for e in entries if e.get("workflow_id")]
    without_id = [e for e in entries if not e.get("workflow_id")]

    # One representative timestamp per logical query.
    rep_timestamps: List[datetime] = []

    # Partition 1: rows with a workflow_id -> one ts per distinct workflow
    # (prefer the genesis ts, else the earliest entry ts for that workflow).
    if with_id:
        per_workflow: Dict[Any, datetime] = {}
        genesis_ts: Dict[Any, datetime] = {}
        for entry in with_id:
            wid = entry.get("workflow_id")
            ts = _parse_ts(entry.get("created_at"))
            if ts is None:
                continue
            if entry.get("action_type") == _WORKFLOW_GENESIS_ACTION:
                genesis_ts.setdefault(wid, ts)
            cur = per_workflow.get(wid)
            if cur is None or ts < cur:
                per_workflow[wid] = ts
        rep_timestamps.extend(
            genesis_ts.get(wid, earliest) for wid, earliest in per_workflow.items()
        )

    # Partition 2: rows without a workflow_id -> genesis-rows fallback.
    if without_id:
        genesis_entries = [
            e for e in without_id if e.get("action_type") == _WORKFLOW_GENESIS_ACTION
        ]
        source = genesis_entries if genesis_entries else without_id
        rep_timestamps.extend(
            ts for e in source if (ts := _parse_ts(e.get("created_at"))) is not None
        )

    buckets: Dict[datetime, int] = {}
    for ts in rep_timestamps:
        bucket_ts = ts.replace(
            minute=0,
            second=0,
            microsecond=0,
            hour=(ts.hour // interval_hours) * interval_hours,
        )
        buckets[bucket_ts] = buckets.get(bucket_ts, 0) + 1

    return [
        TimeSeriesPoint(timestamp=ts, value=float(count)) for ts, count in sorted(buckets.items())
    ]


# ``health_score_quick`` is the automated background health poller: it emits the
# overwhelming majority of audit rows (workflow_start / component / compose
# scaffolding). It is REAL activity, but including it would dominate per-tier
# task counts and dilute tier-3 latency toward the poller's value, so it is
# excluded from the per-tier rollup — consistent with how the Activity Feed
# treats it. Defined locally (not imported from the agents router) so this
# module stays independent.
_TIER_METRICS_POLLER_AGENTS = {"health_score_quick"}


def _aggregate_tier_metrics(entries: List[Dict[str, Any]]) -> List[TierMetricsItem]:
    """Roll audit entries up into per-tier metrics for all six tiers (0-5).

    - ``tasks_completed``: number of non-poller agent actions in the tier.
    - ``avg_response_time_ms``: mean duration over timed (>0) rows; ``None`` when
      no timed row exists (UNMEASURED, never a fabricated 0ms).
    - ``success_rate``: always ``None`` — validation_passed is too sparse to
      compute a representative per-tier rate honestly.
    """
    buckets: Dict[int, Dict[str, Any]] = {t: {"tasks": 0, "latencies": []} for t in range(6)}

    for entry in entries:
        if entry.get("agent_name") in _TIER_METRICS_POLLER_AGENTS:
            continue
        tier = int(entry.get("agent_tier") or 0)
        if tier < 0 or tier > 5:
            continue
        bucket = buckets[tier]
        bucket["tasks"] += 1
        duration = entry.get("duration_ms")
        if duration is not None and duration > 0:
            bucket["latencies"].append(float(duration))

    items: List[TierMetricsItem] = []
    for tier in range(6):
        latencies = buckets[tier]["latencies"]
        items.append(
            TierMetricsItem(
                tier=tier,
                tasks_completed=buckets[tier]["tasks"],
                avg_response_time_ms=(
                    round(sum(latencies) / len(latencies), 2) if latencies else None
                ),
                success_rate=None,
            )
        )
    return items


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "/dashboard",
    response_model=AnalyticsDashboardResponse,
    summary="Get analytics dashboard data",
    operation_id="get_analytics_dashboard",
    description="Retrieve comprehensive analytics including agent metrics, latency trends, and query volumes.",
)
async def get_analytics_dashboard(
    period: str = Query(
        default="7d",
        description="Time period: 1d, 7d, 30d, 90d",
        pattern="^(1d|7d|30d|90d)$",
    ),
    brand: Optional[str] = Query(default=None, description="Filter by brand"),
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> AnalyticsDashboardResponse:
    """Get complete analytics dashboard data."""
    db = _get_supabase_client()
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Analytics service unavailable. Database connection failed.",
        )

    # Calculate date range
    now = datetime.now(timezone.utc)
    period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
    days = period_days.get(period, 7)
    start_date = now - timedelta(days=days)

    # Fetch data
    result = await _fetch_audit_metrics(db, start_date, now, brand)

    if not result["success"]:
        # Do NOT swallow the failure into a zeroed dashboard: returning all-zero
        # metrics on a fetch error would present fabricated zeros to the client
        # as if they were real measurements (indistinguishable from a genuinely
        # idle period). Surface an honest degraded signal instead. This mirrors
        # the 503 already returned above when the DB client is unavailable, and
        # drives the frontend's existing error-render path. The endpoint stays
        # public; only the error handling changes.
        logger.warning(f"Failed to fetch metrics: {result.get('error')}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Analytics metrics are temporarily unavailable. The metrics "
                "store could not be reached; no data is shown rather than "
                "presenting fabricated zeros."
            ),
        )

    entries = result["data"]

    # Aggregate metrics
    agent_metrics = _aggregate_agent_metrics(entries)

    # Build time series (adjust interval based on period)
    interval_hours = 1 if days <= 1 else (6 if days <= 7 else 24)
    latency_trend = _build_time_series(entries, "duration_ms", interval_hours)
    volume_trend = _build_volume_series(entries, interval_hours)

    # Calculate summary. Query volume is counted at the WORKFLOW level (one run
    # == one query), NOT per audit entry — per-node instrumentation writes
    # several entries per run, so len(entries) would inflate the count.
    all_latencies = [e.get("duration_ms", 0) for e in entries if e.get("duration_ms")]
    total, successful = _count_query_volume(entries)

    # Intent distribution (from action_type)
    intent_dist: Dict[str, int] = {}
    for entry in entries:
        action = entry.get("action_type", "unknown")
        intent_dist[action] = intent_dist.get(action, 0) + 1

    # Top agents
    top_agents = [m.agent_name for m in agent_metrics[:5]]

    summary = QueryMetricsSummary(
        period_start=start_date,
        period_end=now,
        total_queries=total,
        successful_queries=successful,
        failed_queries=total - successful,
        success_rate=round(successful / total * 100, 2) if total > 0 else 0.0,
        # No timed entries -> latency UNMEASURED (None), not a fake 0ms.
        avg_latency_ms=(
            round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else None
        ),
        p50_latency_ms=round(_calculate_percentile(all_latencies, 50), 2)
        if all_latencies
        else None,
        p95_latency_ms=round(_calculate_percentile(all_latencies, 95), 2)
        if all_latencies
        else None,
        p99_latency_ms=round(_calculate_percentile(all_latencies, 99), 2)
        if all_latencies
        else None,
        intent_distribution=intent_dist,
        top_agents=top_agents,
    )

    # Latency breakdown (estimated from agent tiers)
    tier_latencies: Dict[int, List[float]] = {}
    for entry in entries:
        tier = entry.get("agent_tier") or 0
        duration = entry.get("duration_ms")
        if duration is not None and duration > 0:
            if tier not in tier_latencies:
                tier_latencies[tier] = []
            tier_latencies[tier].append(duration)

    # Estimate breakdown based on tier (tiers correspond to processing stages)
    breakdown = LatencyBreakdown(
        classification_ms=round(
            sum(tier_latencies.get(0, [0])) / max(len(tier_latencies.get(0, [1])), 1), 2
        ),
        routing_ms=round(
            sum(tier_latencies.get(1, [0])) / max(len(tier_latencies.get(1, [1])), 1), 2
        ),
        agent_dispatch_ms=round(
            sum(tier_latencies.get(2, [0])) / max(len(tier_latencies.get(2, [1])), 1), 2
        ),
        synthesis_ms=round(
            sum(tier_latencies.get(5, [0])) / max(len(tier_latencies.get(5, [1])), 1), 2
        ),
        total_ms=summary.avg_latency_ms,
    )

    return AnalyticsDashboardResponse(
        summary=summary,
        agent_metrics=agent_metrics,
        latency_trend=latency_trend,
        query_volume_trend=volume_trend,
        latency_breakdown=breakdown,
    )


@router.get(
    "/agents/{agent_name}",
    response_model=AgentMetrics,
    summary="Get metrics for a specific agent",
    operation_id="get_agent_metrics",
    description="Retrieve detailed performance metrics for a single agent.",
)
async def get_agent_metrics(
    agent_name: str,
    period: str = Query(default="7d", pattern="^(1d|7d|30d|90d)$"),
    brand: Optional[str] = Query(default=None),
    user: Dict[str, Any] = Depends(require_auth),
) -> AgentMetrics:
    """Get metrics for a specific agent."""
    db = _get_supabase_client()
    if db is None:
        raise HTTPException(status_code=503, detail="Analytics service unavailable.")

    now = datetime.now(timezone.utc)
    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 7)
    start_date = now - timedelta(days=days)

    try:
        query = (
            db.table("audit_chain_entries")
            .select("agent_name, agent_tier, duration_ms, validation_passed, confidence_score")
            .eq("agent_name", agent_name)
            .gte("created_at", start_date.isoformat())
        )

        if brand:
            query = query.eq("brand", brand)

        result = query.execute()
        entries = result.data or []

        if not entries:
            raise HTTPException(status_code=404, detail=f"No data found for agent: {agent_name}")

        metrics = _aggregate_agent_metrics(entries)
        return metrics[0] if metrics else AgentMetrics(agent_name=agent_name, agent_tier=0)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")


@router.get(
    "/agents/{agent_name}/trend",
    response_model=AgentPerformanceTrend,
    summary="Get performance trend for an agent",
    operation_id="get_agent_trend",
    description="Retrieve latency trend over time for a specific agent.",
)
async def get_agent_trend(
    agent_name: str,
    period: str = Query(default="7d", pattern="^(1d|7d|30d|90d)$"),
    brand: Optional[str] = Query(default=None),
    user: Dict[str, Any] = Depends(require_auth),
) -> AgentPerformanceTrend:
    """Get performance trend for a specific agent."""
    db = _get_supabase_client()
    if db is None:
        raise HTTPException(status_code=503, detail="Analytics service unavailable.")

    now = datetime.now(timezone.utc)
    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 7)
    start_date = now - timedelta(days=days)
    interval_hours = 1 if days <= 1 else (6 if days <= 7 else 24)

    try:
        result = (
            db.table("audit_chain_entries")
            .select("duration_ms, created_at")
            .eq("agent_name", agent_name)
            .gte("created_at", start_date.isoformat())
            .execute()
        )

        entries = result.data or []
        data_points = _build_time_series(entries, "duration_ms", interval_hours)

        return AgentPerformanceTrend(
            agent_name=agent_name,
            data_points=data_points,
            period=period,
        )

    except Exception as e:
        logger.error(f"Failed to get agent trend: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trend: {str(e)}")


@router.get(
    "/summary",
    response_model=QueryMetricsSummary,
    summary="Get quick metrics summary",
    operation_id="get_metrics_summary",
    description="Get a quick summary of query execution metrics.",
)
async def get_metrics_summary(
    period: str = Query(default="24h", pattern="^(1h|6h|24h|7d)$"),
    user: Dict[str, Any] = Depends(require_auth),
) -> QueryMetricsSummary:
    """Get quick metrics summary for header/status display."""
    db = _get_supabase_client()
    if db is None:
        raise HTTPException(status_code=503, detail="Analytics service unavailable.")

    now = datetime.now(timezone.utc)
    hours = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}.get(period, 24)
    start_date = now - timedelta(hours=hours)

    try:
        result = (
            db.table("audit_chain_entries")
            .select("workflow_id, duration_ms, validation_passed, action_type, agent_name")
            .gte("created_at", start_date.isoformat())
            .execute()
        )

        entries = result.data or []
        all_latencies = [e.get("duration_ms", 0) for e in entries if e.get("duration_ms")]
        # Query volume counted at workflow level (one run == one query), not per
        # audit entry — per-node instrumentation writes several entries per run.
        total, successful = _count_query_volume(entries)

        # Agent counts
        agent_counts: Dict[str, int] = {}
        for e in entries:
            agent = e.get("agent_name", "unknown")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        top_agents = sorted(agent_counts.keys(), key=lambda a: agent_counts[a], reverse=True)[:5]

        return QueryMetricsSummary(
            period_start=start_date,
            period_end=now,
            total_queries=total,
            successful_queries=successful,
            failed_queries=total - successful,
            success_rate=round(successful / total * 100, 2) if total > 0 else 0.0,
            # No timed entries in the window -> latency is UNMEASURED (None), NOT
            # a real 0ms. Avoids the misleading "0ms / instant" the UI showed for
            # genesis-only windows before the agent graphs were instrumented.
            avg_latency_ms=(
                round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else None
            ),
            p50_latency_ms=round(_calculate_percentile(all_latencies, 50), 2)
            if all_latencies
            else None,
            p95_latency_ms=round(_calculate_percentile(all_latencies, 95), 2)
            if all_latencies
            else None,
            p99_latency_ms=round(_calculate_percentile(all_latencies, 99), 2)
            if all_latencies
            else None,
            top_agents=top_agents,
        )

    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch summary: {str(e)}")


@router.get(
    "/tier-metrics",
    response_model=TierMetricsResponse,
    summary="Get per-tier agent performance metrics",
    operation_id="get_tier_metrics",
    description="Per-tier task counts and average response time from the audit chain.",
)
async def get_tier_metrics(
    hours: int = Query(default=24, ge=1, le=168, description="Look-back window (hours)"),
    brand: Optional[str] = Query(default=None, description="Filter by brand"),
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> TierMetricsResponse:
    """Per-tier performance, aggregated from ``audit_chain_entries``.

    Backs the Agent Orchestration "Tier Metrics" tab (Avg Response / Tasks). The
    automated health poller is excluded so "Tasks" reflects meaningful agent
    work; per-tier success rate is reported as unmeasured (``None`` -> "—") since
    validation is too sparse to compute honestly. Public like
    ``/analytics/dashboard`` (aggregate-only, no query text); on a fetch failure
    it surfaces 503 rather than presenting fabricated zeroed tiers as real.
    """
    db = _get_supabase_client()
    if db is None:
        raise HTTPException(status_code=503, detail="Analytics service unavailable.")

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(hours=hours)
    result = await _fetch_audit_metrics(db, start_date, now, brand)

    if not result["success"]:
        logger.warning(f"Failed to fetch tier metrics: {result.get('error')}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Tier metrics are temporarily unavailable; no data is shown "
                "rather than presenting fabricated zeros."
            ),
        )

    return TierMetricsResponse(
        tiers=_aggregate_tier_metrics(result["data"]),
        window_hours=hours,
    )
