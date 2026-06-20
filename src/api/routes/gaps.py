"""
E2I Gap Analysis API
====================

FastAPI endpoints for gap analysis, ROI estimation, and opportunity prioritization.

Phase: Agent Output Routing

Endpoints:
- POST /gaps/analyze: Run gap analysis for a brand/segment
- GET  /gaps/{analysis_id}: Get gap analysis results
- GET  /gaps/opportunities: List prioritized opportunities
- GET  /gaps/health: Service health check

Integration Points:
- Gap Analyzer Agent (Tier 2)
- Orchestrator for agent invocation
- Supabase for persistence

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import require_analyst
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

if TYPE_CHECKING:
    from src.api.repositories.gaps_repository import GapsRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/gaps",
    tags=["Gap Analysis"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS
# =============================================================================


class GapType(str, Enum):
    """Types of performance gaps."""

    VS_TARGET = "vs_target"
    VS_BENCHMARK = "vs_benchmark"
    VS_POTENTIAL = "vs_potential"
    TEMPORAL = "temporal"
    ALL = "all"


class ImplementationDifficulty(str, Enum):
    """Difficulty levels for closing a gap."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisStatus(str, Enum):
    """Status of a gap analysis."""

    PENDING = "pending"
    DETECTING = "detecting"
    CALCULATING = "calculating"
    PRIORITIZING = "prioritizing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class RunGapAnalysisRequest(BaseModel):
    """Request to run gap analysis."""

    query: str = Field(..., description="Natural language query describing the analysis")
    brand: str = Field(..., description="Brand identifier (e.g., 'kisqali', 'fabhalta')")
    metrics: List[str] = Field(
        default=["trx", "market_share"],
        description="KPIs to analyze (e.g., ['trx', 'market_share', 'conversion_rate'])",
    )
    segments: List[str] = Field(
        default=["region"],
        description="Segmentation dimensions (e.g., ['region', 'specialty'])",
    )
    time_period: str = Field(
        default="current_quarter",
        description="Analysis period (e.g., 'current_quarter', '2024-Q3')",
    )
    indication: Optional[str] = Field(
        default=None,
        description="Indication scope for the label lookup (defaults to the brand's primary)",
    )
    label_segmentation: bool = Field(
        default=False,
        description=(
            "Opt-in label-gater: flag/de-prioritize bets whose segment is outside the "
            "FDA-indicated population. Default off = unchanged behaviour."
        ),
    )
    gap_type: GapType = Field(
        default=GapType.ALL,
        description="Type of gaps to detect",
    )
    min_gap_threshold: float = Field(
        default=5.0,
        description="Minimum gap percentage to report (e.g., 5.0 for 5%)",
        ge=0.0,
        le=100.0,
    )
    max_opportunities: int = Field(
        default=10,
        description="Maximum opportunities to return",
        ge=1,
        le=50,
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional filters (e.g., {'region': 'Northeast'})",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Identify performance gaps for Kisqali in Q4",
                "brand": "kisqali",
                "metrics": ["trx", "market_share", "conversion_rate"],
                "segments": ["region", "specialty"],
                "time_period": "current_quarter",
                "gap_type": "all",
                "min_gap_threshold": 5.0,
                "max_opportunities": 10,
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class PerformanceGap(BaseModel):
    """Individual performance gap identified."""

    gap_id: str = Field(..., description="Unique gap identifier")
    metric: str = Field(..., description="KPI name")
    segment: str = Field(..., description="Segmentation dimension")
    segment_value: str = Field(..., description="Specific segment value")
    current_value: float = Field(..., description="Current performance value")
    target_value: float = Field(..., description="Target/benchmark value")
    gap_size: float = Field(..., description="Absolute gap (target - current)")
    gap_percentage: float = Field(..., description="Gap as percentage")
    gap_type: str = Field(..., description="Type of comparison")


class ConfidenceInterval(BaseModel):
    """Bootstrap confidence interval for ROI estimates."""

    lower_bound: float = Field(..., description="2.5th percentile")
    median: float = Field(..., description="50th percentile")
    upper_bound: float = Field(..., description="97.5th percentile")
    probability_positive: float = Field(..., description="P(ROI > 1x)")
    probability_target: float = Field(..., description="P(ROI > target)")


class ROIEstimate(BaseModel):
    """ROI estimate for closing a performance gap."""

    gap_id: str = Field(..., description="References gap identifier")
    estimated_revenue_impact: float = Field(..., description="Annual revenue impact (USD)")
    estimated_cost_to_close: float = Field(..., description="One-time cost (USD)")
    expected_roi: float = Field(..., description="Base ROI ratio")
    risk_adjusted_roi: float = Field(..., description="ROI after risk adjustment")
    payback_period_months: int = Field(..., description="Months to recoup investment")
    confidence_interval: Optional[ConfidenceInterval] = Field(
        default=None, description="95% confidence interval"
    )
    attribution_level: str = Field(..., description="Attribution level")
    attribution_rate: float = Field(..., description="Attribution rate (0-1)")
    confidence: float = Field(..., description="Estimate confidence (0-1)")
    # Label-gater (codex#4 — carried end-to-end so the UI can surface it). Optional:
    # only populated when label_segmentation is enabled.
    off_label: Optional[bool] = Field(
        default=None,
        description="True if the bet's segment is outside the FDA-indicated population",
    )
    off_label_reason: Optional[str] = Field(
        default=None, description="Why the bet is off-label (label-evidenced violation)"
    )
    label_verdict: Optional[str] = Field(
        default=None, description="on_label | off_label | mixed | indeterminate"
    )
    label_evidence_confirmed: Optional[bool] = Field(
        default=None, description="Whether the verdict is confirmed by the live FDA label"
    )


class PrioritizedOpportunity(BaseModel):
    """Prioritized gap with ROI estimate and action recommendation."""

    rank: int = Field(..., description="Priority rank (1 = highest)")
    gap: PerformanceGap = Field(..., description="The identified gap")
    roi_estimate: ROIEstimate = Field(..., description="ROI analysis")
    recommended_action: str = Field(..., description="Specific action to close gap")
    implementation_difficulty: ImplementationDifficulty = Field(..., description="Difficulty level")
    time_to_impact: str = Field(..., description="Expected time to results")


class GapAnalysisResponse(BaseModel):
    """Response from gap analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    brand: str = Field(..., description="Brand analyzed")
    metrics_analyzed: List[str] = Field(..., description="KPIs analyzed")
    segments_analyzed: int = Field(..., description="Number of segments")

    # Prioritized results
    prioritized_opportunities: List[PrioritizedOpportunity] = Field(
        default_factory=list, description="All opportunities ranked by ROI"
    )
    quick_wins: List[PrioritizedOpportunity] = Field(
        default_factory=list, description="Low difficulty, high ROI (top 5)"
    )
    strategic_bets: List[PrioritizedOpportunity] = Field(
        default_factory=list, description="High impact, high difficulty (top 5)"
    )

    # Aggregate values
    total_addressable_value: float = Field(
        default=0.0, description="Total potential revenue impact"
    )
    total_gap_value: float = Field(default=0.0, description="Sum of all gap sizes")

    # Summary
    executive_summary: str = Field(default="", description="Executive-level summary")
    key_insights: List[str] = Field(default_factory=list, description="Key findings")

    # Multi-library support
    libraries_used: Optional[List[str]] = Field(default=None, description="Causal libraries used")
    library_agreement_score: Optional[float] = Field(
        default=None, description="Agreement between libraries"
    )

    # Metadata
    detection_latency_ms: int = Field(default=0, description="Detection time")
    roi_latency_ms: int = Field(default=0, description="ROI calculation time")
    total_latency_ms: int = Field(default=0, description="Total workflow time")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp",
    )
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis_id": "gap_abc123",
                "status": "completed",
                "brand": "kisqali",
                "metrics_analyzed": ["trx", "market_share"],
                "segments_analyzed": 12,
                "total_addressable_value": 2500000.0,
                "total_gap_value": 15.3,
                "executive_summary": "Identified 8 high-value opportunities...",
                "total_latency_ms": 3500,
            }
        }
    )


class OpportunityListResponse(BaseModel):
    """Response for listing opportunities."""

    total_count: int = Field(
        ..., description="Opportunities returned (after min_roi/difficulty filters and limit)"
    )
    quick_wins_count: int = Field(
        ...,
        description=(
            "Curated quick wins from the latest run per brand (prioritizer's "
            "definition: low difficulty, ROI>1). Independent of the min_roi/"
            "difficulty list filters."
        ),
    )
    strategic_bets_count: int = Field(
        ...,
        description=(
            "Curated strategic bets from the latest run per brand (prioritizer's "
            "definition: high difficulty AND ROI>2 AND cost>$50k, top 5). "
            "Independent of the min_roi/difficulty list filters."
        ),
    )
    opportunities: List[PrioritizedOpportunity] = Field(..., description="List of opportunities")
    total_addressable_value: float = Field(..., description="Total potential value")


class GapHealthResponse(BaseModel):
    """Health check response for gap analysis service."""

    status: str = Field(..., description="Service status")
    agent_available: bool = Field(..., description="Gap Analyzer agent status")
    last_analysis: Optional[datetime] = Field(default=None, description="Last analysis timestamp")
    analyses_24h: int = Field(default=0, description="Analyses in last 24 hours")


# =============================================================================
# PERSISTENCE (Supabase) with an in-memory dev fallback (M2)
# =============================================================================
# Process-local fallback — used ONLY when Supabase is unconfigured or the dev
# flag E2I_GAPS_FEEDBACK_INMEMORY=1 is set. In prod (gunicorn --workers 2) this
# is bypassed: the repo is the single source of truth shared across workers.

_analyses_store: Dict[str, GapAnalysisResponse] = {}


def _get_repo() -> "Optional[GapsRepository]":
    """Return a GapsRepository, or None if Supabase is unavailable."""
    try:
        from src.api.repositories.gaps_repository import GapsRepository

        return GapsRepository()
    except Exception as exc:  # ServiceConnectionError when unconfigured
        logger.warning("Gaps persistence unavailable, using in-memory fallback: %s", exc)
        return None


def _use_inmemory_fallback() -> bool:
    """True when we should fall back to the process-local dict (dev/offline)."""
    if os.environ.get("E2I_GAPS_FEEDBACK_INMEMORY") == "1":
        return True
    try:
        return _get_repo() is None
    except Exception:
        return True


async def _persist_analysis(response: GapAnalysisResponse) -> None:
    """Write one analysis to Supabase (or the dict fallback)."""
    if _use_inmemory_fallback():
        _analyses_store[response.analysis_id] = response
        return
    repo = _get_repo()
    if repo is None:
        _analyses_store[response.analysis_id] = response
        return
    await repo.upsert(response)


async def _load_analysis(analysis_id: str) -> Optional[GapAnalysisResponse]:
    """Read one analysis from Supabase (or the dict fallback)."""
    if _use_inmemory_fallback():
        return _analyses_store.get(analysis_id)
    repo = _get_repo()
    if repo is None:
        return _analyses_store.get(analysis_id)
    return await repo.get(analysis_id)


async def _completed_analyses(brand: Optional[str] = None) -> List[GapAnalysisResponse]:
    """Completed analyses for list_opportunities (repo or dict fallback)."""
    if _use_inmemory_fallback():
        return list(_analyses_store.values())
    repo = _get_repo()
    if repo is None:
        return list(_analyses_store.values())
    return await repo.list_completed(brand=brand)


async def _all_analyses() -> List[GapAnalysisResponse]:
    """All analyses for get_gap_health (repo or dict fallback)."""
    if _use_inmemory_fallback():
        return list(_analyses_store.values())
    repo = _get_repo()
    if repo is None:
        return list(_analyses_store.values())
    return await repo.list_all()


def _to_utc(ts: datetime) -> datetime:
    """Coerce a timestamp to UTC-aware so comparisons never raise.

    Rows are normally persisted with a tz-aware timestamp (the model default is
    ``datetime.now(timezone.utc)``), but a payload whose ``timestamp`` was stored
    without an offset round-trips through pydantic as a NAIVE datetime. Comparing
    a naive and an aware datetime raises ``TypeError`` — which would 500 the
    opportunities endpoint — so treat a naive timestamp as UTC.
    """
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _latest_completed_per_brand(
    analyses: List[GapAnalysisResponse],
) -> List[GapAnalysisResponse]:
    """Return the most recent COMPLETED analysis per (case-normalized) brand.

    A gap analysis is a point-in-time SNAPSHOT. ``list_opportunities`` previously
    SUMMED every stored run for a brand, which (a) re-counted the same recurring
    gaps — gap_ids are deterministic (``region_<value>_<metric>_<gap_type>``) so
    each "Run Analysis" re-contributed the same opportunities — and (b) resurfaced
    stale runs produced by older code (e.g. pre-fix over-performance "gaps" with
    implausible ROI). Both inflated the headline counts and duplicated cards.

    Collapsing to the latest run per brand fixes both. Brands are grouped
    case-insensitively so historical lowercase rows ("kisqali") and the canonical
    capitalized rows ("Kisqali") resolve to ONE snapshot. Ties (identical
    timestamps — e.g. a gunicorn 2-worker double-write) break by analysis_id for
    deterministic selection.
    """
    latest: Dict[str, GapAnalysisResponse] = {}
    for analysis in analyses:
        key = analysis.brand.lower()
        current = latest.get(key)
        if current is None or (_to_utc(analysis.timestamp), analysis.analysis_id) > (
            _to_utc(current.timestamp),
            current.analysis_id,
        ):
            latest[key] = analysis
    return list(latest.values())


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/analyze",
    response_model=GapAnalysisResponse,
    summary="Run gap analysis",
    operation_id="run_gap_analysis",
    description="Analyze performance gaps for a brand across segments and calculate ROI.",
)
async def run_gap_analysis(
    request: RunGapAnalysisRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(
        default=True, description="Run asynchronously (returns immediately with ID)"
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> GapAnalysisResponse:
    """
    Run gap analysis for a brand.

    This endpoint invokes the Gap Analyzer agent (Tier 2) to:
    1. Detect performance gaps across segments
    2. Calculate ROI for closing each gap
    3. Prioritize opportunities by expected value

    Args:
        request: Gap analysis parameters
        background_tasks: FastAPI background tasks
        async_mode: If True, returns immediately with analysis ID

    Returns:
        Gap analysis results or pending status if async
    """
    analysis_id = f"gap_{uuid4().hex[:12]}"

    # Create initial response
    response = GapAnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING if async_mode else AnalysisStatus.DETECTING,
        brand=request.brand,
        metrics_analyzed=request.metrics,
        segments_analyzed=0,
    )

    if async_mode:
        # Store pending analysis
        await _persist_analysis(response)

        # Schedule background task
        background_tasks.add_task(
            _run_gap_analysis_task,
            analysis_id=analysis_id,
            request=request,
        )

        logger.info(f"Gap analysis {analysis_id} queued for background execution")
        return response

    # Synchronous execution
    try:
        result = await _execute_gap_analysis(request)
        result.analysis_id = analysis_id
        await _persist_analysis(result)
        return result
    except HTTPException:
        # F-010-backend (#429, codex iter-1 M1): preserve the 503
        # raised by the agent-import guard inside _execute_gap_analysis
        # instead of masking it as a 500.
        raise
    except Exception as e:
        logger.error(f"Gap analysis failed: {e}")
        response.status = AnalysisStatus.FAILED
        response.warnings.append(str(e))
        await _persist_analysis(response)
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {e}")


@router.get(
    "/opportunities",
    response_model=OpportunityListResponse,
    summary="List prioritized opportunities",
    operation_id="list_opportunities",
    description="List all identified opportunities across analyses.",
)
async def list_opportunities(
    brand: Optional[str] = Query(default=None, description="Filter by brand"),
    min_roi: Optional[float] = Query(default=None, description="Minimum ROI threshold"),
    difficulty: Optional[ImplementationDifficulty] = Query(
        default=None, description="Filter by difficulty"
    ),
    limit: int = Query(default=20, description="Maximum results", ge=1, le=100),
) -> OpportunityListResponse:
    """
    List prioritized opportunities across all analyses.

    Args:
        brand: Optional brand filter
        min_roi: Minimum ROI threshold
        difficulty: Filter by implementation difficulty
        limit: Maximum number of results

    Returns:
        List of prioritized opportunities
    """
    analyses = await _completed_analyses(brand=brand)

    # Keep COMPLETED analyses matching the brand filter case-insensitively. The
    # repo applies an ``.ilike`` brand filter, but the in-memory fallback path does
    # not, so re-filter here for parity. Brand casing is canonically capitalized
    # ("Kisqali"); historical rows may be lowercase — compare case-insensitively so
    # grounded analyses are never silently dropped.
    completed = [
        a
        for a in analyses
        if a.status == AnalysisStatus.COMPLETED and (not brand or a.brand.lower() == brand.lower())
    ]

    # ACCUMULATION/STALENESS FIX: surface only the LATEST run per brand, not the
    # sum of every historical run (see _latest_completed_per_brand). brand=None
    # ("All Brands") therefore aggregates the latest run of EACH brand.
    latest_analyses = _latest_completed_per_brand(completed)

    all_opportunities: List[PrioritizedOpportunity] = []
    total_value = 0.0
    quick_wins_count = 0
    strategic_bets_count = 0

    for analysis in latest_analyses:
        # Headline counts come from the prioritizer's OWN curated lists — its
        # authoritative definitions (quick win = low difficulty, ROI>1; strategic
        # bet = high difficulty AND ROI>2 AND cost>$50k, capped at top 5). The page
        # previously re-derived "strategic bets" as *every* high-difficulty
        # opportunity, which both diverged from this definition and inflated the
        # number. These counts are brand-scoped (only this brand's latest run is
        # kept) and intentionally independent of the list-view min_roi/difficulty
        # filters below, which narrow the displayed cards, not the KPI totals.
        quick_wins_count += len(analysis.quick_wins)
        strategic_bets_count += len(analysis.strategic_bets)

        for opp in analysis.prioritized_opportunities:
            # List-view filters (narrow the displayed opportunities only).
            if min_roi and opp.roi_estimate.expected_roi < min_roi:
                continue
            if difficulty and opp.implementation_difficulty != difficulty:
                continue

            all_opportunities.append(opp)
            total_value += opp.roi_estimate.estimated_revenue_impact

    # Sort by ROI and limit
    all_opportunities.sort(key=lambda x: x.roi_estimate.expected_roi, reverse=True)
    all_opportunities = all_opportunities[:limit]

    return OpportunityListResponse(
        total_count=len(all_opportunities),
        quick_wins_count=quick_wins_count,
        strategic_bets_count=strategic_bets_count,
        opportunities=all_opportunities,
        total_addressable_value=total_value,
    )


@router.get(
    "/health",
    response_model=GapHealthResponse,
    summary="Gap analysis service health",
    operation_id="get_gap_health",
    description="Check health status of the gap analysis service.",
)
async def get_gap_health() -> GapHealthResponse:
    """
    Get health status of gap analysis service.

    Returns:
        Service health information
    """
    # Check agent availability
    agent_available = True
    try:
        from src.agents.gap_analyzer import GapAnalyzerAgent  # noqa: F401

        agent_available = True
    except ImportError:
        agent_available = False

    # Count recent analyses
    now = datetime.now(timezone.utc)
    analyses = await _all_analyses()
    analyses_24h = sum(1 for a in analyses if (now - a.timestamp).total_seconds() < 86400)

    # Get last analysis
    last_analysis = max((a.timestamp for a in analyses), default=None)

    return GapHealthResponse(
        status="healthy" if agent_available else "degraded",
        agent_available=agent_available,
        last_analysis=last_analysis,
        analyses_24h=analyses_24h,
    )


@router.get(
    "/{analysis_id}",
    response_model=GapAnalysisResponse,
    summary="Get gap analysis results",
    operation_id="get_gap_analysis",
    description="Retrieve results of a gap analysis by ID.",
)
async def get_gap_analysis(analysis_id: str) -> GapAnalysisResponse:
    """
    Get gap analysis results by ID.

    Args:
        analysis_id: Unique analysis identifier

    Returns:
        Gap analysis results

    Raises:
        HTTPException: If analysis not found
    """
    result = await _load_analysis(analysis_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Gap analysis {analysis_id} not found",
        )

    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _run_gap_analysis_task(
    analysis_id: str,
    request: RunGapAnalysisRequest,
) -> None:
    """Background task to run gap analysis."""
    try:
        logger.info(f"Starting gap analysis task {analysis_id}")

        # Update status (read-modify-persist; in-memory mutates in place)
        existing = await _load_analysis(analysis_id)
        if existing is not None:
            existing.status = AnalysisStatus.DETECTING
            await _persist_analysis(existing)

        # Execute analysis
        result = await _execute_gap_analysis(request)
        result.analysis_id = analysis_id

        # Store result
        await _persist_analysis(result)

        logger.info(f"Gap analysis {analysis_id} completed successfully")

    except Exception as e:
        logger.error(f"Gap analysis {analysis_id} failed: {e}")
        existing = await _load_analysis(analysis_id)
        if existing is not None:
            existing.status = AnalysisStatus.FAILED
            existing.warnings.append(str(e))
            await _persist_analysis(existing)


async def _execute_gap_analysis(
    request: RunGapAnalysisRequest,
) -> GapAnalysisResponse:
    """
    Execute gap analysis using Gap Analyzer agent.

    This function orchestrates the Gap Analyzer agent (Tier 2) to:
    1. Detect gaps via gap_detector node
    2. Calculate ROI via roi_calculator node
    3. Prioritize via prioritizer node
    """
    import time

    start_time = time.time()

    try:
        # Try to use the actual Gap Analyzer agent
        from src.agents.gap_analyzer.graph import create_gap_analyzer_graph
        from src.agents.gap_analyzer.state import GapAnalyzerState

        # Initialize state (cast partial state - remaining fields populated by graph nodes)
        initial_state = cast(
            GapAnalyzerState,
            {
                "query": request.query,
                "metrics": request.metrics,
                "segments": request.segments,
                "brand": request.brand,
                "indication": request.indication,
                "label_segmentation": request.label_segmentation,
                "time_period": request.time_period,
                "gap_type": request.gap_type.value,
                "min_gap_threshold": request.min_gap_threshold,
                "max_opportunities": request.max_opportunities,
                "filters": request.filters,
                "status": "pending",
                "errors": [],
                "warnings": [],
                "detection_latency_ms": 0,
                "roi_latency_ms": 0,
                "total_latency_ms": 0,
                "segments_analyzed": 0,
            },
        )

        # Create and run graph
        graph = create_gap_analyzer_graph()
        result = await graph.ainvoke(initial_state)

        # Convert agent output to API response
        total_latency = int((time.time() - start_time) * 1000)

        return GapAnalysisResponse(
            analysis_id="",  # Will be set by caller
            # F2 (HIGH) belt-and-suspenders: force FAILED whenever the result
            # carries errors, even if a stale/buggy node still reports
            # status="completed". Never launder an upstream error into a green
            # HTTP-200 "no significant gaps" response.
            status=AnalysisStatus.COMPLETED
            if result.get("status") == "completed" and not result.get("errors")
            else AnalysisStatus.FAILED,
            brand=request.brand,
            metrics_analyzed=request.metrics,
            segments_analyzed=result.get("segments_analyzed", 0),
            prioritized_opportunities=_convert_opportunities(
                result.get("prioritized_opportunities", [])
            ),
            quick_wins=_convert_opportunities(result.get("quick_wins", [])),
            strategic_bets=_convert_opportunities(result.get("strategic_bets", [])),
            total_addressable_value=result.get("total_addressable_value", 0.0),
            total_gap_value=result.get("total_gap_value", 0.0),
            executive_summary=result.get("executive_summary", ""),
            key_insights=result.get("key_insights", []),
            libraries_used=result.get("libraries_executed"),
            library_agreement_score=result.get("library_agreement_score"),
            detection_latency_ms=result.get("detection_latency_ms", 0),
            roi_latency_ms=result.get("roi_latency_ms", 0),
            total_latency_ms=total_latency,
            warnings=result.get("warnings", []),
        )

    except ImportError as e:
        # F-010-backend (#429): fail-closed in production unless mock-fallback
        # is explicitly enabled (E2I_REQUIRE_AGENT_IMPORT=0 or ENVIRONMENT!=production).
        from src.api.utils.agent_import_guard import guard_or_raise

        guard_or_raise(e, agent_name="Gap Analyzer")
        return _generate_mock_response(request, start_time)

    except Exception as e:
        logger.error(f"Gap analysis execution failed: {e}")
        raise


def _convert_opportunities(
    opportunities: List[Dict[str, Any]],
) -> List[PrioritizedOpportunity]:
    """Convert agent output to API response format."""
    result = []
    for opp in opportunities:
        try:
            gap_data = opp.get("gap", {})
            roi_data = opp.get("roi_estimate", {})

            gap = PerformanceGap(
                gap_id=gap_data.get("gap_id", ""),
                metric=gap_data.get("metric", ""),
                segment=gap_data.get("segment", ""),
                segment_value=gap_data.get("segment_value", ""),
                current_value=gap_data.get("current_value", 0.0),
                target_value=gap_data.get("target_value", 0.0),
                gap_size=gap_data.get("gap_size", 0.0),
                gap_percentage=gap_data.get("gap_percentage", 0.0),
                gap_type=gap_data.get("gap_type", "vs_target"),
            )

            roi = ROIEstimate(
                gap_id=roi_data.get("gap_id", ""),
                estimated_revenue_impact=roi_data.get("estimated_revenue_impact", 0.0),
                estimated_cost_to_close=roi_data.get("estimated_cost_to_close", 0.0),
                expected_roi=roi_data.get("expected_roi", 0.0),
                risk_adjusted_roi=roi_data.get("risk_adjusted_roi", 0.0),
                payback_period_months=roi_data.get("payback_period_months", 0),
                attribution_level=roi_data.get("attribution_level", "partial"),
                attribution_rate=roi_data.get("attribution_rate", 0.5),
                confidence=roi_data.get("confidence", 0.7),
                off_label=roi_data.get("off_label"),
                off_label_reason=roi_data.get("off_label_reason"),
                label_verdict=roi_data.get("label_verdict"),
                label_evidence_confirmed=roi_data.get("label_evidence_confirmed"),
            )

            result.append(
                PrioritizedOpportunity(
                    rank=opp.get("rank", 0),
                    gap=gap,
                    roi_estimate=roi,
                    recommended_action=opp.get("recommended_action", ""),
                    implementation_difficulty=ImplementationDifficulty(
                        opp.get("implementation_difficulty", "medium")
                    ),
                    time_to_impact=opp.get("time_to_impact", "3-6 months"),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert opportunity: {e}")
            continue

    return result


def _generate_mock_response(
    request: RunGapAnalysisRequest,
    start_time: float,
) -> GapAnalysisResponse:
    """Generate mock response when agent is not available."""
    import time

    # Mock gap
    mock_gap = PerformanceGap(
        gap_id=f"{request.segments[0]}_northeast_{request.metrics[0]}",
        metric=request.metrics[0],
        segment=request.segments[0],
        segment_value="Northeast",
        current_value=85.0,
        target_value=100.0,
        gap_size=15.0,
        gap_percentage=15.0,
        gap_type="vs_target",
    )

    # Mock ROI
    mock_roi = ROIEstimate(
        gap_id=mock_gap.gap_id,
        estimated_revenue_impact=500000.0,
        estimated_cost_to_close=100000.0,
        expected_roi=4.0,
        risk_adjusted_roi=3.2,
        payback_period_months=6,
        attribution_level="partial",
        attribution_rate=0.7,
        confidence=0.75,
    )

    # Mock opportunity
    mock_opp = PrioritizedOpportunity(
        rank=1,
        gap=mock_gap,
        roi_estimate=mock_roi,
        recommended_action="Increase field force coverage in Northeast region",
        implementation_difficulty=ImplementationDifficulty.MEDIUM,
        time_to_impact="3-6 months",
    )

    total_latency = int((time.time() - start_time) * 1000)

    return GapAnalysisResponse(
        analysis_id="",
        status=AnalysisStatus.COMPLETED,
        brand=request.brand,
        metrics_analyzed=request.metrics,
        segments_analyzed=len(request.segments) * 4,  # Mock 4 values per segment
        prioritized_opportunities=[mock_opp],
        quick_wins=[mock_opp],
        strategic_bets=[],
        total_addressable_value=500000.0,
        total_gap_value=15.0,
        executive_summary=f"Analysis identified 1 opportunity for {request.brand} with total addressable value of $500,000.",
        key_insights=[
            f"Northeast {request.segments[0]} shows 15% gap vs target",
            "Recommended action: Increase field force coverage",
            "Expected ROI: 4.0x with 6-month payback period",
        ],
        detection_latency_ms=100,
        roi_latency_ms=150,
        total_latency_ms=total_latency,
        warnings=["Using mock data - Gap Analyzer agent not available"],
    )
