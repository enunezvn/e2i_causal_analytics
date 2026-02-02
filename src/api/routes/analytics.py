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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


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

    # Latency metrics (in milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Confidence metrics
    avg_confidence: Optional[float] = None


class LatencyBreakdown(BaseModel):
    """Latency breakdown by processing stage."""

    classification_ms: float = 0.0
    rag_retrieval_ms: float = 0.0
    routing_ms: float = 0.0
    agent_dispatch_ms: float = 0.0
    synthesis_ms: float = 0.0
    total_ms: float = 0.0


class QueryMetricsSummary(BaseModel):
    """Summary of query execution metrics."""

    period_start: datetime
    period_end: datetime
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate: float = 0.0

    # Latency summary
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

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
                "agent_name, agent_tier, duration_ms, validation_passed, confidence_score, created_at, action_type"
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
                avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                p50_latency_ms=round(_calculate_percentile(latencies, 50), 2),
                p95_latency_ms=round(_calculate_percentile(latencies, 95), 2),
                p99_latency_ms=round(_calculate_percentile(latencies, 99), 2),
                min_latency_ms=round(min(latencies), 2) if latencies else 0.0,
                max_latency_ms=round(max(latencies), 2) if latencies else 0.0,
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
        created_at = entry.get("created_at")
        value = entry.get(field)

        if created_at and value is not None:
            # Parse timestamp
            if isinstance(created_at, str):
                ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                ts = created_at

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


def _build_volume_series(
    entries: List[Dict[str, Any]],
    interval_hours: int = 1,
) -> List[TimeSeriesPoint]:
    """Build query volume time series."""
    if not entries:
        return []

    buckets: Dict[datetime, int] = {}

    for entry in entries:
        created_at = entry.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                ts = created_at

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


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "/dashboard",
    response_model=AnalyticsDashboardResponse,
    summary="Get analytics dashboard data",
    description="Retrieve comprehensive analytics including agent metrics, latency trends, and query volumes.",
)
async def get_analytics_dashboard(
    period: str = Query(
        default="7d",
        description="Time period: 1d, 7d, 30d, 90d",
        regex="^(1d|7d|30d|90d)$",
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
        logger.warning(f"Failed to fetch metrics: {result.get('error')}")
        # Return empty dashboard rather than error
        entries = []
    else:
        entries = result["data"]

    # Aggregate metrics
    agent_metrics = _aggregate_agent_metrics(entries)

    # Build time series (adjust interval based on period)
    interval_hours = 1 if days <= 1 else (6 if days <= 7 else 24)
    latency_trend = _build_time_series(entries, "duration_ms", interval_hours)
    volume_trend = _build_volume_series(entries, interval_hours)

    # Calculate summary
    all_latencies = [e.get("duration_ms", 0) for e in entries if e.get("duration_ms")]
    successful = sum(1 for e in entries if e.get("validation_passed") is not False)
    total = len(entries)

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
        avg_latency_ms=round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0,
        p50_latency_ms=round(_calculate_percentile(all_latencies, 50), 2),
        p95_latency_ms=round(_calculate_percentile(all_latencies, 95), 2),
        p99_latency_ms=round(_calculate_percentile(all_latencies, 99), 2),
        intent_distribution=intent_dist,
        top_agents=top_agents,
    )

    # Latency breakdown (estimated from agent tiers)
    tier_latencies: Dict[int, List[float]] = {}
    for entry in entries:
        tier = entry.get("agent_tier", 0)
        duration = entry.get("duration_ms", 0)
        if duration > 0:
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
    description="Retrieve detailed performance metrics for a single agent.",
)
async def get_agent_metrics(
    agent_name: str,
    period: str = Query(default="7d", regex="^(1d|7d|30d|90d)$"),
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
    description="Retrieve latency trend over time for a specific agent.",
)
async def get_agent_trend(
    agent_name: str,
    period: str = Query(default="7d", regex="^(1d|7d|30d|90d)$"),
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
    description="Get a quick summary of query execution metrics.",
)
async def get_metrics_summary(
    period: str = Query(default="24h", regex="^(1h|6h|24h|7d)$"),
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
            .select("duration_ms, validation_passed, action_type, agent_name")
            .gte("created_at", start_date.isoformat())
            .execute()
        )

        entries = result.data or []
        all_latencies = [e.get("duration_ms", 0) for e in entries if e.get("duration_ms")]
        successful = sum(1 for e in entries if e.get("validation_passed") is not False)
        total = len(entries)

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
            avg_latency_ms=round(sum(all_latencies) / len(all_latencies), 2)
            if all_latencies
            else 0.0,
            p50_latency_ms=round(_calculate_percentile(all_latencies, 50), 2),
            p95_latency_ms=round(_calculate_percentile(all_latencies, 95), 2),
            p99_latency_ms=round(_calculate_percentile(all_latencies, 99), 2),
            top_agents=top_agents,
        )

    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch summary: {str(e)}")
