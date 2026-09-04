"""
E2I Segment Analysis & Heterogeneous Optimization API
======================================================

FastAPI endpoints for segment-level CATE analysis and targeting optimization.

Phase: Agent Output Routing

Endpoints:
- POST /segments/analyze: Run segment analysis (CATE estimation)
- GET  /segments/{analysis_id}: Get analysis results
- GET  /segments/policies: Get targeting recommendations
- GET  /segments/health: Service health check

Integration Points:
- Heterogeneous Optimizer Agent (Tier 2)
- EconML for CATE estimation
- CausalML for uplift modeling
- Supabase for persistence

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    cast,
)
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.exceptions import RedisError

from src.api.dependencies.auth import require_analyst
from src.api.dependencies.compute import HeavyComputeSaturated, heavy_compute_slot
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.repositories.provenance import apply_provenance_filter

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/segments",
    tags=["Segment Analysis"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS
# =============================================================================


class ResponderType(str, Enum):
    """Types of treatment responders."""

    HIGH = "high"
    LOW = "low"
    AVERAGE = "average"


class SegmentationMethod(str, Enum):
    """Methods for creating segments."""

    QUANTILE = "quantile"
    KMEANS = "kmeans"
    THRESHOLD = "threshold"
    TREE = "tree"


class SegmentAnalysisStatus(str, Enum):
    """Status of segment analysis."""

    PENDING = "pending"
    ESTIMATING = "estimating"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"


class SegmentQuestionType(str, Enum):
    """Type of analysis question for library routing."""

    EFFECT_HETEROGENEITY = "effect_heterogeneity"  # EconML primary
    TARGETING = "targeting"  # CausalML primary
    SEGMENT_OPTIMIZATION = "segment_optimization"  # Both libraries
    COMPREHENSIVE = "comprehensive"  # All libraries with DoWhy validation


# =============================================================================
# REQUEST MODELS
# =============================================================================


class RunSegmentAnalysisRequest(BaseModel):
    """Request to run segment analysis.

    Clinical-HTE rebuild (2026-06-20): the page is now agent-driven over the
    curated ``patient_journeys`` gold-standard substrate. Only ``treatment_var``
    / ``outcome_var`` are selectable (from the curated allowlist, enforced
    server-side by :func:`_load_segment_hte_frame`); the clinical contract
    (``effect_modifiers`` / ``confounders`` / ``segment_vars``) is FIXED
    server-side in :func:`_execute_segment_analysis`, so those fields are now
    optional and any request-supplied values for the patient_journeys path are
    overridden. ``treatment_var`` / ``outcome_var`` default to
    ``treatment_arm`` -> ``persistent_180d`` when omitted.
    """

    query: str = Field(..., description="Natural language query describing the analysis")
    brand: Optional[str] = Field(
        default=None,
        description=(
            "Optional cohort FILTER (data-driven dropdown, like /causal/brands). "
            "Scopes the gold-standard load to one brand server-side; it is a row "
            "subset, NOT a causal variable. None => all brands."
        ),
    )
    treatment_var: Optional[str] = Field(
        default=None,
        description=(
            "Treatment variable (curated). Defaults to 'treatment_arm' when "
            "omitted. Must be in the patient_journeys allowlist AND, unless "
            "allow_unmodeled=true, the (treatment, outcome) pair must be a modeled "
            "causal_paths edge for the brand scope — GET /segments/datasets?brand= "
            "enumerates the offered pairs. Enforced server-side."
        ),
    )
    outcome_var: Optional[str] = Field(
        default=None,
        description=(
            "Outcome variable (curated). Defaults to 'persistent_180d' when "
            "omitted. Must be in the patient_journeys allowlist; "
            "GET /segments/datasets?brand= lists the outcomes each treatment has a "
            "modeled causal edge to. Enforced server-side."
        ),
    )
    allow_unmodeled: bool = Field(
        default=False,
        description=(
            "Exploratory opt-in (#1827). By default a (treatment, outcome) pair with "
            "NO modeled causal_paths edge for the brand scope is refused with 400 "
            "before any compute runs — its estimate would reflect confounding, not "
            "an effect. Set true to run it anyway on the default adjustment set; the "
            "result then carries an explicit not-a-modeled-question warning."
        ),
    )
    segment_vars: Optional[List[str]] = Field(
        default=None,
        description=(
            "Variables to segment by. FIXED server-side to the clinical "
            "allowlist for the patient_journeys path; any value supplied here is "
            "overridden. Optional (the route sets it)."
        ),
    )
    effect_modifiers: Optional[List[str]] = Field(
        default=None,
        description=(
            "Variables that modify treatment effect (X). FIXED server-side to "
            "the numeric clinical covariate set for the patient_journeys path; "
            "any value supplied here is overridden."
        ),
    )
    # NOTE: `brand` is defined once above (cohort FILTER). #1060 added a SECOND
    # `brand` field here for the label-gater, which Pydantic silently collapsed and
    # mypy flagged as [no-redef] — pushing main's mypy count 61 -> 62 (over the
    # ceiling) and blocking ALL CI. The single `brand` above already supplies the
    # gater's brand input, so the duplicate is removed.
    indication: Optional[str] = Field(
        default=None,
        description="Indication scope for the label lookup; resolved from the data when omitted",
    )
    label_segmentation: bool = Field(
        default=False,
        description=(
            "Opt-in label-gater: augment segment_vars with the brand's label-relevant "
            "columns and flag/de-prioritize segments outside the FDA-indicated population. "
            "Requires brand. Default off = unchanged behaviour."
        ),
    )
    confounders: Optional[List[str]] = Field(
        default=None,
        description=(
            "Confounders to adjust for. Routed into the DML nuisance model (W) "
            "and residualized out, NOT modeled as effect modifiers — so the "
            "reported per-segment CATE reflects the de-confounded treatment "
            "effect rather than selection bias. Distinct from segment_vars "
            "(reporting grouping) and effect_modifiers (heterogeneity features). "
            "FIXED server-side (W=engagement_score) for the patient_journeys "
            "path; any value supplied here is overridden."
        ),
    )
    data_source: str = Field(
        default="business_metrics",
        description=(
            "Data source table identifier. IGNORED by /segments/analyze: the "
            "clinical HTE path is FIXED server-side to 'patient_journeys' "
            "(with a server-prepared, banded gold-standard frame), and any "
            "value supplied here is overridden. Retained for wire "
            "compatibility with older callers only."
        ),
    )
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")

    # Configuration
    n_estimators: int = Field(default=100, description="Causal Forest trees", ge=10, le=1000)
    min_samples_leaf: int = Field(default=10, description="Minimum samples per leaf", ge=1, le=100)
    significance_level: float = Field(
        default=0.05, description="For CI calculation", gt=0.0, lt=0.5
    )
    top_segments_count: int = Field(
        default=10, description="Number of top segments to return", ge=1, le=50
    )
    question_type: Optional[SegmentQuestionType] = Field(
        default=None, description="Analysis question type for library routing"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": (
                    "Treatment effect heterogeneity of treatment_arm on "
                    "persistent_180d across clinical segments"
                ),
                "treatment_var": "treatment_arm",
                "outcome_var": "persistent_180d",
                "brand": "Remibrutinib",
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class CATEResult(BaseModel):
    """CATE estimation result for a segment."""

    segment_name: str = Field(..., description="Segment dimension name")
    segment_value: str = Field(..., description="Segment value")
    cate_estimate: float = Field(..., description="Conditional Average Treatment Effect")
    cate_ci_lower: float = Field(
        ...,
        description=(
            "CATE CI lower bound at the response's confidence_level "
            "(default 95%; see SegmentAnalysisResponse.confidence_level)"
        ),
    )
    cate_ci_upper: float = Field(
        ...,
        description=(
            "CATE CI upper bound at the response's confidence_level "
            "(default 95%; see SegmentAnalysisResponse.confidence_level)"
        ),
    )
    sample_size: int = Field(..., description="Number of observations in segment")
    statistical_significance: bool = Field(
        ..., description="Whether effect is statistically significant"
    )


class SegmentProfile(BaseModel):
    """Profile of a high/low responder segment."""

    segment_id: str = Field(..., description="Unique segment identifier")
    responder_type: ResponderType = Field(..., description="Responder classification")
    cate_estimate: float = Field(..., description="CATE for this segment")
    defining_features: List[Dict[str, Any]] = Field(
        ..., description="Features that define this segment"
    )
    size: int = Field(..., description="Segment size (observations)")
    size_percentage: float = Field(..., description="Percentage of total population")
    recommendation: str = Field(..., description="Targeting recommendation")


class PolicyRecommendation(BaseModel):
    """Treatment allocation recommendation."""

    segment: str = Field(..., description="Segment identifier")
    current_treatment_rate: float = Field(..., description="Current treatment rate (0-1)")
    recommended_treatment_rate: float = Field(..., description="Recommended treatment rate (0-1)")
    expected_incremental_outcome: float = Field(
        ..., description="Expected incremental outcome from change"
    )
    confidence: float = Field(..., description="Recommendation confidence (0-1)")
    # Label-gater (codex#4 — carried end-to-end so the UI can surface it). Optional:
    # only populated when label_segmentation is enabled.
    off_label: Optional[bool] = Field(
        default=None, description="True if the segment falls outside the FDA-indicated population"
    )
    off_label_reason: Optional[str] = Field(
        default=None, description="Why the segment is off-label (label-evidenced violation)"
    )
    label_verdict: Optional[str] = Field(
        default=None, description="on_label | off_label | mixed | indeterminate"
    )
    label_evidence_confirmed: Optional[bool] = Field(
        default=None, description="Whether the verdict is confirmed by the live FDA label"
    )


class UpliftMetrics(BaseModel):
    """Uplift modeling metrics."""

    overall_auuc: float = Field(..., description="Area Under Uplift Curve (0-1)")
    overall_qini: float = Field(..., description="Qini coefficient")
    targeting_efficiency: float = Field(..., description="How well model targets responders (0-1)")
    model_type_used: str = Field(..., description="Model type (random_forest, gradient_boosting)")


class SegmentAnalysisResponse(BaseModel):
    """Response from segment analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: SegmentAnalysisStatus = Field(..., description="Analysis status")
    question_type: Optional[SegmentQuestionType] = Field(
        default=None, description="Question type used for routing"
    )

    # Run design — echoed so consumers reading the PERSISTED record alone
    # (e.g. POST /insights/hte grounding) can state what was estimated without
    # re-posting caller figures. Optional/None so records persisted before
    # these fields existed still validate on read (the durable store fail-softs
    # unreadable records into None).
    brand: Optional[str] = Field(
        default=None, description="Brand row-filter the run used (None = all brands)"
    )
    treatment_var: Optional[str] = Field(
        default=None, description="Treatment variable the CATEs were estimated for"
    )
    outcome_var: Optional[str] = Field(
        default=None, description="Outcome variable the CATEs were estimated for"
    )

    # CATE results
    cate_by_segment: Dict[str, List[CATEResult]] = Field(
        default_factory=dict, description="CATE results grouped by segment variable"
    )
    overall_ate: Optional[float] = Field(
        default=None, description="Overall Average Treatment Effect"
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0.5,
        lt=1.0,
        description=(
            "Confidence level the CATE CIs (cate_by_segment[*].cate_ci_lower/"
            "upper) are computed at, e.g. 0.95 => a 95% CI. Derived from the "
            "request's significance_level (confidence_level = 1 - "
            "significance_level). Exposed (#27) so the UI labels the intervals "
            "truthfully instead of assuming 95%. Range mirrors the EXACT inverse "
            "of RunSegmentAnalysisRequest.significance_level (gt=0.0, lt=0.5) so "
            "no previously-valid request can fail response validation."
        ),
    )
    heterogeneity_score: Optional[float] = Field(
        default=None, description="Treatment effect heterogeneity (0-1)"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None, description="Feature importance for CATE"
    )

    # Uplift results
    uplift_metrics: Optional[UpliftMetrics] = Field(
        default=None, description="Uplift modeling metrics"
    )

    # Segment discovery
    high_responders: List[SegmentProfile] = Field(
        default_factory=list, description="High responder segments"
    )
    mid_responders: List[SegmentProfile] = Field(
        default_factory=list,
        description=(
            "Mid (average) responder segments — |CATE| in the band between the "
            "low and high thresholds (responder_type='average'). [] when none "
            "qualify. Surfaced so the page does not imply exactly two buckets."
        ),
    )
    low_responders: List[SegmentProfile] = Field(
        default_factory=list, description="Low responder segments"
    )

    # Policy recommendations
    policy_recommendations: List[PolicyRecommendation] = Field(
        default_factory=list, description="Targeting recommendations"
    )
    expected_total_lift: Optional[float] = Field(
        default=None,
        description=(
            "Expected lift from optimal targeting as a COUNT of incremental outcomes "
            "on the best single segmentation axis (de-double-counted across "
            "overlapping dimensions). Secondary to expected_lift_pp."
        ),
    )
    expected_lift_pp: Optional[float] = Field(
        default=None,
        description=(
            "HEADLINE lift metric: the best-axis expected lift as a percentage-point "
            "change in the outcome rate (count / best-axis cohort size). For binary / "
            "rate outcomes this lies in [0, 1]; it is only well-defined for such "
            "outcomes (a continuous outcome makes a percentage-point rate change "
            "ill-defined). ~0 for a homogeneous effect under the above-ATE gate."
        ),
    )
    optimal_allocation_summary: Optional[str] = Field(
        default=None, description="Summary of optimal allocation"
    )

    # Summary
    executive_summary: Optional[str] = Field(default=None, description="Executive-level summary")
    strategic_interpretation: Optional[str] = Field(
        default=None,
        description=(
            "3-tier business narrative (who responds, why, expected lift) from "
            "the profile_generator node. Mapped from the final graph state — was "
            "silently dropped at the route before the clinical-HTE rebuild."
        ),
    )
    key_insights: List[str] = Field(default_factory=list, description="Key findings")

    # Hierarchical / heterogeneity (mapped from the final graph state)
    segment_comparison: Optional[Dict[str, Any]] = Field(
        default=None,
        description="High/mid/low comparison summary (effect_ratio, counts) from segment_analyzer",
    )
    segment_heterogeneity: Optional[float] = Field(
        default=None,
        description=(
            "Between-segment heterogeneity (I^2) from the hierarchical analyzer. "
            "Maps result['segment_heterogeneity'] (note: NOT '_score' — that is "
            "the TypedDict field name; the node emits 'segment_heterogeneity')."
        ),
    )
    n_segments_analyzed: Optional[int] = Field(
        default=None, description="Number of segments analyzed by the hierarchical analyzer"
    )
    segmentation_method_used: Optional[str] = Field(
        default=None, description="Segmentation method used (quantile/kmeans/threshold/tree)"
    )
    overall_hierarchical_ate: Optional[float] = Field(
        default=None, description="Aggregate ATE from the hierarchical (nested-CI) analysis"
    )
    hierarchical_segment_results: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Per-segment hierarchical CATE results"
    )
    uplift_by_segment: Optional[Dict[str, Any]] = Field(
        default=None, description="Uplift scores grouped by segment dimension"
    )

    # Multi-library support
    libraries_used: Optional[List[str]] = Field(default=None, description="Causal libraries used")
    library_agreement_score: Optional[float] = Field(
        default=None, description="Agreement between libraries (0-1)"
    )
    validation_passed: Optional[bool] = Field(
        default=None, description="Whether cross-validation passed"
    )
    cross_library_validation: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Components behind library_agreement_score so the verdict is explainable: "
            "method, n_segments_compared, sign_agreement (direction), "
            "n_distinguishable_pairs + ordering_agreement (within-axis segment pairs "
            "whose CATE CIs are disjoint), spearman_rho (pooled diagnostic only), "
            "threshold, uplift_model. computed=False carries a reason instead."
        ),
    )

    # Metadata
    estimation_latency_ms: int = Field(default=0, description="CATE estimation time")
    analysis_latency_ms: int = Field(default=0, description="Segment analysis time")
    total_latency_ms: int = Field(default=0, description="Total workflow time")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp",
    )
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    confidence: float = Field(default=0.0, description="Overall analysis confidence")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis_id": "seg_abc123",
                "status": "completed",
                "overall_ate": 12.5,
                "heterogeneity_score": 0.65,
                "total_latency_ms": 4500,
            }
        }
    )


class PolicyListResponse(BaseModel):
    """Response for listing policy recommendations."""

    total_count: int = Field(..., description="Total recommendations")
    recommendations: List[PolicyRecommendation] = Field(..., description="Policy recommendations")
    expected_total_lift: float = Field(
        ..., description="Total expected lift if all policies adopted"
    )


class SegmentHealthResponse(BaseModel):
    """Health check response for segment analysis service."""

    status: str = Field(..., description="Service status")
    agent_available: bool = Field(..., description="Heterogeneous Optimizer agent status")
    econml_available: bool = Field(default=True, description="EconML availability")
    causalml_available: bool = Field(default=True, description="CausalML availability")
    last_analysis: Optional[datetime] = Field(default=None, description="Last analysis timestamp")
    analyses_24h: int = Field(default=0, description="Analyses in last 24 hours")
    storage_mode: str = Field(
        default="durable",
        description=(
            "Analyses-store backing: 'durable' (Redis, shared across workers) "
            "or 'degraded' (process-local in-memory fallback — Redis "
            "unreachable from this worker, so cross-worker reads may 404)."
        ),
    )


# =============================================================================
# ANALYSES STORAGE (durable / cross-worker)
# =============================================================================
#
# C21: the analyses store must be DURABLE and SHARED ACROSS WORKERS.
#
# Production runs gunicorn with ``--workers 2`` (docker/docker-compose.yml,
# docker-compose.secure.yml). A process-local dict therefore has two real,
# user-visible failures:
#   - A POST handled by worker A is invisible to a GET handled by worker B, so
#     a legitimate analysis 404s roughly half the time.
#   - All state is lost on process restart / redeploy.
#
# The store below backs the analyses in Redis, REUSING the app's existing async
# Redis client (``src.api.dependencies.redis_client.get_redis``, already wired
# into the FastAPI lifespan in ``src/api/main.py``). When Redis is unavailable
# it transparently falls back to a BOUNDED in-process dict — mirroring the
# app's existing graceful-degradation posture (``app.state.redis_available``)
# rather than failing the request. The fallback is FIFO-bounded so memory
# cannot grow without limit (a plain dict would be a slow leak / DoS vector),
# and the Redis index is bounded the same way.

# Maximum number of analyses retained (both in Redis and in the fallback dict).
ANALYSES_STORE_MAX_ENTRIES = 1000

# How long an analysis record lives in Redis. Generous relative to the 24h
# window the health endpoint reports on, while still letting abandoned records
# expire so the store cannot accumulate stale entries indefinitely.
ANALYSES_STORE_TTL_SECONDS = 7 * 24 * 60 * 60

# Redis key namespace.
_REDIS_KEY_PREFIX = "segments:analysis:"
_REDIS_INDEX_KEY = "segments:analysis:index"

# #1840 in-flight dedup marker: ``segments:inflight:<dedup_key>`` -> analysis_id
# of the run currently pending/estimating for that RESOLVED question (see
# ``_segment_analysis_dedup_key``). Written with ``SET NX EX`` so two identical
# POSTs racing on different gunicorn workers cannot both win; released when the
# run reaches a terminal state and bounded by a TTL (run budget + grace) so a
# crashed worker's marker self-heals.
_REDIS_INFLIGHT_KEY_PREFIX = "segments:inflight:"
_INFLIGHT_MARKER_GRACE_SECONDS = 120
_INFLIGHT_STATUSES = frozenset({SegmentAnalysisStatus.PENDING, SegmentAnalysisStatus.ESTIMATING})

# #1840 run budget for the heterogeneous-optimizer graph fit held under the
# per-worker ``heavy_compute_slot()``. Without it a hung / pathological run
# held the worker's ONLY slot forever and every later analysis on that worker
# was rejected "compute capacity saturated" until the worker restarted.
#
# Default 900 s (15 min), chosen from the measured deployed runtime and the
# page contract:
#   * single-brand runs measured 73-208 s (#1836) and 109-121 s (e2i_api logs
#     2026-08-30, Fabhalta copay_support -> persistent_180d); the CausalML
#     uplift fit is ~80% of it and scales with host load. 900 s is >4x the
#     slowest observed run, leaving room for an all-brands cohort (unmeasured,
#     more rows) and a loaded host.
#   * the page waits up to 300 s single-brand / 600 s all-brands
#     (SegmentAnalysis.tsx poll ceilings). The backend must never fail a run
#     the page is still willing to wait for, so the budget is bounded BELOW by
#     600 s; the margin above lets a run the page gave up on still land as a
#     durable COMPLETED record (a later GET can show it).
# Override with ``SEGMENT_ANALYSIS_BUDGET_SECONDS`` (float seconds > 0).
SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT = 900.0
_SEGMENT_ANALYSIS_BUDGET_ENV = "SEGMENT_ANALYSIS_BUDGET_SECONDS"


def _segment_analysis_budget_seconds() -> float:
    """Run budget (seconds) for one graph fit; env-overridable, read per call.

    Invalid / non-positive values fall back to the default with a warning,
    mirroring ``compute._max_concurrency_from_env``.
    """
    raw = os.environ.get(_SEGMENT_ANALYSIS_BUDGET_ENV)
    if raw is None or raw.strip() == "":
        return SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %.0f s",
            _SEGMENT_ANALYSIS_BUDGET_ENV,
            raw,
            SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT,
        )
        return SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT
    if not math.isfinite(value) or value <= 0:
        logger.warning(
            "%s=%r must be > 0; falling back to %.0f s",
            _SEGMENT_ANALYSIS_BUDGET_ENV,
            raw,
            SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT,
        )
        return SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT
    return value


class SegmentAnalysisBudgetExceeded(Exception):
    """The graph fit did not finish within ``SEGMENT_ANALYSIS_BUDGET_SECONDS``.

    Raised by :func:`_execute_segment_analysis` AFTER the run has been
    cancelled and the heavy-compute slot released. The background task records
    a FAILED analysis naming the budget; the sync route maps it to HTTP 504.

    Honest limit: cancelling the graph coroutine cannot stop a fit thread the
    nodes already handed to ``asyncio.to_thread`` — that thread runs to the end
    of its current sklearn/causalml call. What IS restored is the slot, so the
    worker accepts the next request instead of rejecting everything until a
    restart. The default budget sits >4x above the slowest measured run, so
    this path is the pathological-run escape hatch, not a routine event.
    """

    def __init__(self, budget_seconds: float) -> None:
        self.budget_seconds = budget_seconds
        super().__init__(f"segment analysis exceeded its {budget_seconds:g} s run budget")


def _budget_exceeded_warning(budget_seconds: float) -> str:
    """User-facing FAILED-record warning that NAMES the budget applied."""
    return (
        f"Segment analysis exceeded its run budget of {budget_seconds:g} s and was "
        "cancelled; the worker's compute slot was released. Retry later or narrow "
        f"the scope (budget: {_SEGMENT_ANALYSIS_BUDGET_ENV})."
    )


# Redis errors that should trigger graceful degradation rather than a 500.
#
# CRITICAL: the app's Redis client is ``redis.asyncio`` (see
# ``src/api/dependencies/redis_client.py``), whose connection/timeout failures
# are ``redis.exceptions.ConnectionError`` / ``redis.exceptions.TimeoutError``
# — these are NOT the builtin ``ConnectionError`` / ``TimeoutError`` (verified:
# ``redis.exceptions.ConnectionError is builtins.ConnectionError`` -> False).
# They both subclass ``redis.exceptions.RedisError``, so catching ``RedisError``
# covers every client-raised Redis failure (Connection/Timeout/Response/etc.).
# A previous tuple of the *builtins* would let a real mid-flight Redis outage
# escape the fallback and turn into a 500. ``OSError`` covers socket-level
# failures; ``RuntimeError`` covers "redis not initialised" from get_redis().
_REDIS_DEGRADE_ERRORS = (RedisError, OSError, RuntimeError)


# Read-side deserialisation failures that must fail SOFT (skip/None + cleanup)
# rather than 500. ``ValidationError`` is raised when a persisted record cannot
# be re-validated — most importantly when NaN/+-inf floats were serialised to
# JSON ``null`` by ``model_dump_json`` (non-finite floats are not valid for the
# non-Optional float fields on read). ``json.JSONDecodeError`` covers a
# truncated / corrupted payload. Neither is a Redis transport error, so they
# are handled separately from ``_REDIS_DEGRADE_ERRORS``.
_RECORD_DECODE_ERRORS = (ValidationError, json.JSONDecodeError, ValueError)


def _has_non_finite_floats(response: "SegmentAnalysisResponse") -> bool:
    """Return ``True`` if any float in ``response`` is NaN or +-inf.

    ``model_dump_json`` serialises non-finite floats to JSON ``null``; on read
    ``model_validate_json`` then raises ``ValidationError`` because the affected
    fields (e.g. ``CATEResult.cate_estimate``) are non-Optional floats. Such a
    record is therefore WRITTEN successfully but UNREADABLE — it would 500 every
    later ``/policies`` / ``/health`` enumeration. We detect this before
    persisting and refuse to store an unreadable record (see ``set``).
    """

    def _walk(obj: Any) -> bool:
        if isinstance(obj, float):
            return not math.isfinite(obj)
        if isinstance(obj, dict):
            return any(_walk(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(_walk(v) for v in obj)
        return False

    # mode="python" keeps native floats so NaN/inf are detectable (JSON mode
    # would have already coerced them to null).
    return _walk(response.model_dump(mode="python"))


class _BoundedAnalysesStore(Dict[str, SegmentAnalysisResponse]):
    """A dict that evicts the oldest entry once it exceeds ``max_entries``.

    Used as the in-process fallback for :class:`_DurableAnalysesStore` when
    Redis is unavailable. Python dicts preserve insertion order, so
    ``next(iter(self))`` is the oldest key. Re-assigning an existing key updates
    in place and does NOT grow the store.
    """

    def __init__(self, *args: Any, max_entries: int = ANALYSES_STORE_MAX_ENTRIES) -> None:
        super().__init__(*args)
        self.max_entries = max_entries

    def __setitem__(self, key: str, value: SegmentAnalysisResponse) -> None:
        super().__setitem__(key, value)
        while len(self) > self.max_entries:
            oldest_key = next(iter(self))
            # Guard against evicting the key we just inserted (cannot happen
            # while max_entries >= 1, but keeps the loop provably terminating).
            if oldest_key == key:
                break
            del self[oldest_key]


# Type of the zero-arg async factory that yields a Redis client. Defaults to the
# app's canonical ``get_redis`` (lazily imported to keep this module importable
# without a live Redis and to avoid import cycles).
RedisFactory = Callable[[], Awaitable[Any]]


async def _default_redis_factory() -> Any:
    """Return the app's canonical async Redis client.

    Imported lazily so the module imports cleanly in environments without Redis
    configured (tests, CLI tools); failures here are caught by the durable
    store and trigger the in-memory fallback.
    """
    from src.api.dependencies.redis_client import get_redis

    return await get_redis()


class _DurableAnalysesStore:
    """Durable, cross-worker analyses store backed by Redis.

    Each analysis is stored as a JSON string under ``segments:analysis:<id>``
    with a TTL, and indexed in a sorted set scored by CREATION time (the score
    is preserved across later status updates, see ``set``) so the full
    collection can be enumerated (for ``/policies`` and ``/health``) and TRUE
    FIFO-evicted once the count of LIVE records exceeds ``max_entries``.

    All methods are ``async``. When Redis is unavailable (or any command
    fails), they transparently fall back to a bounded in-process dict so a
    request degrades gracefully instead of 500-ing. Writes are mirrored to the
    fallback so a later read can still succeed within the same process if Redis
    is only intermittently reachable.
    """

    def __init__(
        self,
        redis_factory: Optional[RedisFactory] = None,
        max_entries: int = ANALYSES_STORE_MAX_ENTRIES,
        ttl_seconds: int = ANALYSES_STORE_TTL_SECONDS,
    ) -> None:
        self._redis_factory: RedisFactory = redis_factory or _default_redis_factory
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        # In-process fallback used when Redis is unavailable.
        self._memory: _BoundedAnalysesStore = _BoundedAnalysesStore(max_entries=max_entries)
        # #1840 in-process mirror of the in-flight dedup markers:
        # dedup_key -> (analysis_id, monotonic expiry). Serves the same role as
        # ``_memory`` for records: the fallback when Redis is unavailable, and a
        # same-process backstop when Redis drops between two identical POSTs.
        self._inflight: Dict[str, Tuple[str, float]] = {}
        # Last-observed storage mode, so /health can surface silent per-worker
        # degradation (a process serving from the in-memory fallback re-creates
        # the exact cross-worker 404 this store exists to fix). ``None`` until a
        # store operation has probed Redis at least once.
        self._last_durable: Optional[bool] = None

    async def _redis(self) -> Optional[Any]:
        """Return a live Redis client, or ``None`` if unavailable.

        Records the observed storage mode in ``self._last_durable`` so
        ``/health`` can report whether this worker is serving durable (Redis)
        or degraded (in-memory) state.
        """
        try:
            client = await self._redis_factory()
            self._last_durable = True
            return client
        except _REDIS_DEGRADE_ERRORS as e:
            self._last_durable = False
            logger.warning(f"Segments store: Redis unavailable, using in-memory fallback: {e}")
            return None
        except Exception as e:  # pragma: no cover - defensive
            self._last_durable = False
            logger.warning(f"Segments store: unexpected Redis factory error, degrading: {e}")
            return None

    async def is_durable(self) -> bool:
        """Return ``True`` if this worker can currently reach Redis.

        Actively probes the Redis factory so ``/health`` reflects the LIVE
        storage mode rather than a possibly-stale cached value.
        """
        return (await self._redis()) is not None

    @staticmethod
    def _key(analysis_id: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{analysis_id}"

    async def set(self, analysis_id: str, response: SegmentAnalysisResponse) -> None:
        """Persist ``response`` under ``analysis_id`` (Redis + memory mirror).

        Write-side guards:

        * HIGH#1 — refuse to persist a record containing NaN/+-inf floats. Such
          a record serialises to JSON ``null`` and is UNREADABLE on the way
          back (a ``ValidationError`` that would 500 every later enumeration).
          A non-finite estimate is a DEGENERATE fit, so we store an honest
          FAILED record (no fabricated finite numbers) instead.
        * #5  — key SET and index ZADD commit together in a pipeline/txn so a
          ZADD failure never leaves a record fetchable-by-id yet invisible to
          enumeration.
        * #7  — the index score preserves CREATION time (read-existing /
          add-only) so a later status update does not re-score the record into
          "newest", which would make eviction behave like LRU instead of FIFO.
        """
        # HIGH#1 write-side guard: never persist an unreadable record.
        if _has_non_finite_floats(response):
            response = self._sanitize_non_finite(analysis_id, response)

        # Always mirror to the in-process fallback first so an intermittent
        # Redis failure on a later read can still be served in-process.
        self._memory[analysis_id] = response

        client = await self._redis()
        if client is None:
            return
        try:
            payload = response.model_dump_json()
            key = self._key(analysis_id)

            # #7 FIFO: preserve the original creation score across status
            # updates. Only assign a fresh timestamp for a brand-new id.
            existing_score = await self._existing_score(client, analysis_id)
            score = (
                existing_score
                if existing_score is not None
                else datetime.now(timezone.utc).timestamp()
            )

            # #5 atomicity: SET + ZADD in ONE pipelined transaction so they
            # commit in a single round-trip. This closes the
            # connection-drops-between-the-two-writes window that previously
            # left a record fetchable-by-id yet invisible to enumeration (or an
            # index entry pointing at a missing key). Redis MULTI/EXEC does NOT
            # roll back a runtime error mid-EXEC, but our args are well-typed so
            # the only realistic failure here is transport-level (whole pipeline
            # fails -> neither write is applied). For defence-in-depth we
            # additionally run a compensating cleanup on failure so we can never
            # leave a key without its index entry.
            try:
                pipe = client.pipeline(transaction=True)
                pipe.set(key, payload, ex=self.ttl_seconds)
                pipe.zadd(_REDIS_INDEX_KEY, {analysis_id: score})
                await pipe.execute()
            except _REDIS_DEGRADE_ERRORS:
                # Restore consistency: if the key landed but the index did not,
                # drop the orphaned key so it is never fetchable-but-invisible.
                await self._restore_consistency_after_failed_write(client, analysis_id)
                raise

            await self._evict_if_needed(client, keep_id=analysis_id)
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(f"Segments store: Redis write failed for {analysis_id}, degraded: {e}")

    async def _restore_consistency_after_failed_write(self, client: Any, analysis_id: str) -> None:
        """Undo a partial write so a key never outlives its index entry (#5).

        If, despite the pipeline, the string key was applied while the index
        member is absent (Redis does not roll back a mid-EXEC runtime error),
        delete the orphan so the record is not fetchable-by-id yet invisible to
        enumeration. Best-effort; failures here are logged and swallowed (we are
        already on the degrade path).
        """
        try:
            indexed = await client.zscore(_REDIS_INDEX_KEY, analysis_id)
            if indexed is None:
                # No index entry -> any key that landed is an orphan; drop it.
                await client.delete(self._key(analysis_id))
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(
                f"Segments store: consistency restore failed for {analysis_id}, degraded: {e}"
            )

    @staticmethod
    def _sanitize_non_finite(
        analysis_id: str, response: SegmentAnalysisResponse
    ) -> SegmentAnalysisResponse:
        """Return an honest FAILED record for a degenerate (non-finite) fit.

        We do NOT fabricate finite numbers (anti-mocking) and we do NOT persist
        the unreadable original. Instead we drop the degenerate CATE / policy
        payloads and mark the analysis FAILED with a clear warning — a state the
        schema already represents (empty ``cate_by_segment`` /
        ``policy_recommendations`` + ``status=failed``).
        """
        logger.warning(
            "Segments store: analysis %s has non-finite (NaN/inf) estimates "
            "(degenerate fit); persisting as FAILED rather than an unreadable "
            "record.",
            analysis_id,
        )
        sanitized = response.model_copy(deep=True)
        sanitized.status = SegmentAnalysisStatus.FAILED
        sanitized.cate_by_segment = {}
        sanitized.high_responders = []
        sanitized.low_responders = []
        sanitized.policy_recommendations = []
        sanitized.overall_ate = None
        sanitized.heterogeneity_score = None
        # Round-2 BUG 1: ``_has_non_finite_floats`` fires for a non-finite in ANY
        # float field, so dropping only the CATE/policy payloads is INSUFFICIENT
        # — a NaN/inf in any of the fields below would survive and re-poison the
        # record (unreadable on read -> silently skipped + pruned on
        # enumeration, vanishing from durable storage). Scrub EVERY remaining
        # float-bearing field so the result is provably finite. The Optional
        # ones drop to ``None``; ``confidence`` is non-Optional so it resets to
        # the schema default ``0.0`` rather than a fabricated estimate.
        sanitized.feature_importance = None
        sanitized.uplift_metrics = None
        sanitized.library_agreement_score = None
        sanitized.cross_library_validation = None
        sanitized.expected_total_lift = None
        sanitized.expected_lift_pp = None
        sanitized.confidence = 0.0
        if "non-finite" not in " ".join(sanitized.warnings).lower():
            sanitized.warnings.append(
                "Analysis produced non-finite (NaN/inf) estimates; marked as failed."
            )
        # Guard against regression: the whole point of this method is to return
        # a record that round-trips cleanly. If a future field is added that can
        # carry a non-finite float, fail loudly here rather than silently
        # persisting another unreadable record.
        assert not _has_non_finite_floats(sanitized), (
            "sanitize_non_finite left a non-finite float; record would be unreadable"
        )
        return sanitized

    @staticmethod
    async def _existing_score(client: Any, analysis_id: str) -> Optional[float]:
        """Return the current index score for ``analysis_id`` (or None)."""
        try:
            score = await client.zscore(_REDIS_INDEX_KEY, analysis_id)
            return float(score) if score is not None else None
        except _REDIS_DEGRADE_ERRORS:
            # If we cannot read the existing score, fall back to "treat as new";
            # the caller will assign a fresh timestamp.
            return None

    async def _prune_orphans(self, client: Any) -> None:
        """Remove index members whose underlying key no longer exists (HIGH#3).

        TTL-expired records (and records evicted by Redis ``maxmemory`` before
        TTL) leave their index member behind as an orphan. Those orphans inflate
        the count used by FIFO eviction and can cause a LIVE, in-TTL,
        under-capacity record to be evicted while a dead orphan survives (data
        loss + spurious 404). We prune purely by KEY EXISTENCE.

        Round-2 BUG 2: a previous "Pass 1" pruned by SCORE
        (``zremrangebyscore(index, '-inf', now-ttl)``) on the assumption
        "score <= now-ttl => expired". That assumption is FALSE once the index
        score is frozen at CREATION time (fix #7) while every ``set()`` resets
        the key's TTL (``ex=ttl`` on the SET). A record created > ttl ago but
        UPDATED recently then has a LIVE key with a frozen creation score older
        than ``now-ttl`` — and Pass-1 would delete its index member while the
        key is alive, making it fetchable-by-id yet invisible to enumeration
        (the exact split-brain this store exists to prevent). Key-existence is
        the only correct signal regardless of score-vs-TTL divergence, so we
        rely on it alone.
        """
        try:
            members = await client.zrange(_REDIS_INDEX_KEY, 0, -1)
            if not members:
                return
            keys = [self._key(m) for m in members]
            present = await client.mget(keys)
            stale = [m for m, raw in zip(members, present, strict=False) if raw is None]
            if stale:
                await client.zrem(_REDIS_INDEX_KEY, *stale)
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(f"Segments store: Redis orphan-prune failed, degraded: {e}")

    async def _evict_if_needed(self, client: Any, keep_id: str) -> None:
        """FIFO-evict oldest LIVE entries so the Redis index stays bounded.

        Orphans are pruned FIRST (HIGH#3) so the count reflects LIVE records
        only — never evicting a live, under-capacity analysis because dead
        orphans inflated the count.
        """
        try:
            await self._prune_orphans(client)

            count = await client.zcard(_REDIS_INDEX_KEY)
            overflow = count - self.max_entries
            if overflow <= 0:
                return
            # Oldest-first members to drop. Defensive: never evict a member
            # whose key still exists is NOT applied here because, post-prune,
            # every remaining member is live, so true overflow is genuine
            # capacity pressure and FIFO (oldest creation score) is correct.
            oldest = await client.zrange(_REDIS_INDEX_KEY, 0, overflow - 1)
            for member in oldest:
                if member == keep_id:
                    continue
                await client.delete(self._key(member))
                await client.zrem(_REDIS_INDEX_KEY, member)
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(f"Segments store: Redis eviction failed, degraded: {e}")

    async def get(self, analysis_id: str) -> Optional[SegmentAnalysisResponse]:
        """Return the stored analysis, or ``None`` if absent.

        HIGH#1 read-side fail-soft: if the persisted payload cannot be decoded
        / validated (e.g. a poison record written by an older build with
        NaN->null), return ``None`` and lazily remove the poison (key + index
        member) instead of letting a ``ValidationError`` 500 the request.
        """
        client = await self._redis()
        if client is not None:
            try:
                raw = await client.get(self._key(analysis_id))
            except _REDIS_DEGRADE_ERRORS as e:
                logger.warning(
                    f"Segments store: Redis read failed for {analysis_id}, degraded: {e}"
                )
            else:
                if raw is not None:
                    try:
                        return SegmentAnalysisResponse.model_validate_json(raw)
                    except _RECORD_DECODE_ERRORS as e:
                        # Poison / corrupt record: fail soft and self-heal.
                        logger.warning(
                            "Segments store: unreadable record %s (%s); "
                            "removing poison and returning None.",
                            analysis_id,
                            type(e).__name__,
                        )
                        await self._remove_poison(client, analysis_id)
                        return None
                # Not in Redis — fall through to the in-process mirror (may hold
                # a record written while Redis was briefly down).
        return self._memory.get(analysis_id)

    async def _remove_poison(self, client: Any, analysis_id: str) -> None:
        """Lazily delete an unreadable record's key + index member."""
        try:
            await client.delete(self._key(analysis_id))
            await client.zrem(_REDIS_INDEX_KEY, analysis_id)
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(
                f"Segments store: failed to remove poison record {analysis_id}, degraded: {e}"
            )

    async def contains(self, analysis_id: str) -> bool:
        """Return ``True`` if an analysis exists for ``analysis_id``."""
        return (await self.get(analysis_id)) is not None

    async def values(self) -> List[SegmentAnalysisResponse]:
        """Return all stored analyses (Redis-backed, falling back to memory).

        #6 — batches the per-record reads with a single ``mget`` after the
        ``zrange`` (was up to ``max_entries`` sequential ``get`` round-trips on
        every ``/policies`` and ``/health`` call).

        HIGH#1 — a single unreadable (poison/corrupt) record is SKIPPED and its
        index member pruned; it never breaks enumeration of the rest.
        """
        client = await self._redis()
        if client is not None:
            try:
                ids = await client.zrange(_REDIS_INDEX_KEY, 0, -1)
                if not ids:
                    return []
                keys = [self._key(analysis_id) for analysis_id in ids]
                raws = await client.mget(keys)  # one round-trip for all records

                results: List[SegmentAnalysisResponse] = []
                stale_ids: List[str] = []
                for analysis_id, raw in zip(ids, raws, strict=False):
                    if raw is None:
                        # Expired/missing record still indexed — prune lazily.
                        stale_ids.append(analysis_id)
                        continue
                    try:
                        results.append(SegmentAnalysisResponse.model_validate_json(raw))
                    except _RECORD_DECODE_ERRORS as e:
                        # Poison record: skip it, never fail the whole listing.
                        logger.warning(
                            "Segments store: skipping unreadable record %s (%s) "
                            "during enumeration; pruning it.",
                            analysis_id,
                            type(e).__name__,
                        )
                        await client.delete(self._key(analysis_id))
                        stale_ids.append(analysis_id)
                if stale_ids:
                    await client.zrem(_REDIS_INDEX_KEY, *stale_ids)
                return results
            except _REDIS_DEGRADE_ERRORS as e:
                logger.warning(f"Segments store: Redis enumerate failed, degraded: {e}")
        return list(self._memory.values())

    # ------------------------------------------------------------------ #
    # #1840 in-flight dedup markers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inflight_key(dedup_key: str) -> str:
        return f"{_REDIS_INFLIGHT_KEY_PREFIX}{dedup_key}"

    def _memory_inflight_owner(self, dedup_key: str) -> Optional[str]:
        """Owner from the in-process mirror, honouring the marker's TTL."""
        entry = self._inflight.get(dedup_key)
        if entry is None:
            return None
        analysis_id, expires_at = entry
        if time.monotonic() >= expires_at:
            self._inflight.pop(dedup_key, None)
            return None
        return analysis_id

    def _memory_inflight_set(self, dedup_key: str, analysis_id: str, ttl_seconds: int) -> None:
        now = time.monotonic()
        # Prune expired markers so the mirror stays bounded without a sweeper;
        # cap it like ``_memory`` in case TTLs are long and traffic is bursty.
        for key, (_owner, expires_at) in list(self._inflight.items()):
            if now >= expires_at:
                self._inflight.pop(key, None)
        while len(self._inflight) >= self.max_entries:
            self._inflight.pop(next(iter(self._inflight)), None)
        self._inflight[dedup_key] = (analysis_id, now + ttl_seconds)

    async def inflight_owner(self, dedup_key: str) -> Optional[str]:
        """Return the analysis_id currently marked in flight for ``dedup_key``.

        This is the raw marker (Redis first, mirror second); it may point at a
        record that has since finished — :meth:`claim_inflight` is what decides
        whether a marker is LIVE.
        """
        client = await self._redis()
        if client is not None:
            try:
                owner = await client.get(self._inflight_key(dedup_key))
            except _REDIS_DEGRADE_ERRORS as e:
                logger.warning(f"Segments store: Redis in-flight read failed, degraded: {e}")
            else:
                if owner is not None:
                    return str(owner)
        return self._memory_inflight_owner(dedup_key)

    async def _is_inflight_record(self, analysis_id: str) -> bool:
        record = await self.get(analysis_id)
        return record is not None and record.status in _INFLIGHT_STATUSES

    async def claim_inflight(
        self, dedup_key: str, analysis_id: str, *, ttl_seconds: int
    ) -> Optional[str]:
        """Mark ``analysis_id`` as THE in-flight run for ``dedup_key``.

        Returns ``None`` when the claim succeeded (caller should queue its run),
        or the analysis_id of an EXISTING run that is still pending/estimating
        (caller should hand that record back instead of queuing a duplicate).

        A marker is honoured only while its record is in flight: a marker whose
        record has completed/failed — or vanished (evicted, TTL-expired, never
        written) — is STALE and is overwritten, so a finished twin never
        swallows a legitimate re-run and a lost record never blocks submissions
        until the marker's TTL.

        Redis path: ``SET NX EX`` is the atomic cross-worker arbiter. Only when
        NX loses do we read the owner and check its record; a stale owner is
        replaced with a plain ``SET`` (the residual race — two claimants both
        replacing the same stale marker — reproduces today's behaviour of two
        runs, never a lost run). Degrades to the in-process mirror like every
        other store operation.
        """
        redis_key = self._inflight_key(dedup_key)
        client = await self._redis()
        if client is not None:
            try:
                claimed = await client.set(redis_key, analysis_id, ex=ttl_seconds, nx=True)
                if not claimed:
                    owner = await client.get(redis_key)
                    if owner is not None and str(owner) != analysis_id:
                        if await self._is_inflight_record(str(owner)):
                            return str(owner)
                        logger.info(
                            "Segments store: in-flight marker for %s points at %s which is "
                            "no longer in flight; replacing it with %s.",
                            dedup_key[:12],
                            owner,
                            analysis_id,
                        )
                    await client.set(redis_key, analysis_id, ex=ttl_seconds)
                self._memory_inflight_set(dedup_key, analysis_id, ttl_seconds)
                return None
            except _REDIS_DEGRADE_ERRORS as e:
                logger.warning(f"Segments store: Redis in-flight claim failed, degraded: {e}")

        # In-process fallback. Single-threaded loop: no await between the read
        # and the write below, so two coroutines cannot both claim the key.
        owner = self._memory_inflight_owner(dedup_key)
        if owner is not None and owner != analysis_id:
            if await self._is_inflight_record(owner):
                return owner
        self._memory_inflight_set(dedup_key, analysis_id, ttl_seconds)
        return None

    async def release_inflight(self, dedup_key: str, analysis_id: str) -> None:
        """Drop the marker for ``dedup_key`` IF it still points at ``analysis_id``.

        Best-effort and never raises: the status check in :meth:`claim_inflight`
        plus the marker TTL already guarantee correctness; releasing just lets a
        re-run start without first reading a stale marker.
        """
        entry = self._inflight.get(dedup_key)
        if entry is not None and entry[0] == analysis_id:
            self._inflight.pop(dedup_key, None)
        client = await self._redis()
        if client is None:
            return
        redis_key = self._inflight_key(dedup_key)
        try:
            owner = await client.get(redis_key)
            if owner is not None and str(owner) == analysis_id:
                await client.delete(redis_key)
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(f"Segments store: Redis in-flight release failed, degraded: {e}")

    def clear(self) -> None:
        """Clear the in-process fallback (used by tests).

        Note: this clears only the in-process mirror, not Redis. Tests that
        exercise Redis behaviour use a fresh fake client per test.
        """
        self._memory.clear()
        self._inflight.clear()


_analyses_store: _DurableAnalysesStore = _DurableAnalysesStore()


async def get_persisted_analysis(analysis_id: str) -> Optional[SegmentAnalysisResponse]:
    """Public read accessor for the durable analyses store.

    Used by POST /insights/hte to derive its grounding SERVER-SIDE from a
    persisted run — the caller supplies only the analysis_id, never figures
    (same trust boundary as /insights/executive-brief).
    """
    return await _analyses_store.get(analysis_id)


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/analyze",
    response_model=SegmentAnalysisResponse,
    summary="Run segment analysis",
    operation_id="run_segment_analysis",
    description="Analyze treatment effect heterogeneity across segments using CATE/uplift modeling.",
)
async def run_segment_analysis(
    request: RunSegmentAnalysisRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(
        default=True, description="Run asynchronously (returns immediately with ID)"
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> SegmentAnalysisResponse:
    """
    Run segment analysis for treatment effect heterogeneity.

    This endpoint invokes the Heterogeneous Optimizer agent (Tier 2) to:
    1. Estimate CATE using EconML (Causal Forest)
    2. Run uplift modeling using CausalML
    3. Identify high/low responder segments
    4. Generate targeting policy recommendations

    Args:
        request: Segment analysis parameters
        background_tasks: FastAPI background tasks
        async_mode: If True, returns immediately with analysis ID

    Returns:
        Segment analysis results or pending status if async
    """
    analysis_id = f"seg_{uuid4().hex[:12]}"

    # #1827: resolve the question and refuse an UNMODELED pair here, BEFORE the
    # record is created or the task queued — both async modes get an immediate
    # 400 naming the modeled alternatives, and no heavy compute is spent producing
    # a plausible-looking confounded estimate (live 2026-08-30:
    # treatment_initiated -> persistent_180d ran ~40 s to an ATE of +0.076 with
    # nothing but a warning string guarding it). The resolved adjustment is
    # handed to the run so the registry is read once per request.
    treatment_var = request.treatment_var or _SEGMENT_HTE_DEFAULT_TREATMENT
    outcome_var = request.outcome_var or _SEGMENT_HTE_DEFAULT_OUTCOME
    effect_modifiers = _segment_effect_modifiers(
        request.brand, treatment_var=treatment_var, outcome_var=outcome_var
    )
    adjustment = await _segment_question_adjustment(
        treatment_var=treatment_var,
        outcome_var=outcome_var,
        brand=request.brand,
        effect_modifiers=effect_modifiers,
    )
    _refuse_unmodeled_question(request, adjustment, treatment_var, outcome_var)

    # Create initial response
    response = SegmentAnalysisResponse(
        analysis_id=analysis_id,
        status=SegmentAnalysisStatus.PENDING if async_mode else SegmentAnalysisStatus.ESTIMATING,
        question_type=request.question_type,
        # #27: carry the requested CI level (alpha=significance_level) from the
        # start so an async poller sees the level its CATE CIs will use.
        confidence_level=1.0 - request.significance_level,
    )

    if async_mode:
        # #1840 in-flight dedup: an identical question (same body AND same
        # resolved X/W) that is still pending/estimating gets the EXISTING
        # record back — 200 with its analysis_id and current status — instead
        # of a new id whose run the slot guard would reject (#1836: the page
        # then polled the duplicate's FAILED record while the original
        # completed unseen). The page's POST-then-poll flow polls whatever id
        # it is given, so no response-shape change is needed. Only in-flight
        # twins collapse; a completed/failed twin never swallows a re-run.
        # The marker's TTL covers the run budget plus grace so a crashed
        # worker's marker self-heals.
        dedup_key = _segment_analysis_dedup_key(request, adjustment, effect_modifiers)
        marker_ttl = int(_segment_analysis_budget_seconds()) + _INFLIGHT_MARKER_GRACE_SECONDS
        existing_id = await _analyses_store.claim_inflight(
            dedup_key, analysis_id, ttl_seconds=marker_ttl
        )
        if existing_id is not None:
            existing = await _analyses_store.get(existing_id)
            if existing is not None and existing.status in _INFLIGHT_STATUSES:
                logger.info(
                    f"Segment analysis request deduplicated onto in-flight {existing_id} "
                    f"(status={existing.status.value}); not queuing {analysis_id}"
                )
                return existing
            # The owner finished between claim and read: take the marker for our
            # own run (the owner is terminal now, so this claim overwrites it).
            await _analyses_store.claim_inflight(dedup_key, analysis_id, ttl_seconds=marker_ttl)

        # Store pending analysis
        await _analyses_store.set(analysis_id, response)

        # Schedule background task
        background_tasks.add_task(
            _run_segment_analysis_task,
            analysis_id=analysis_id,
            request=request,
            adjustment=adjustment,
            dedup_key=dedup_key,
        )

        logger.info(f"Segment analysis {analysis_id} queued for background execution")
        return response

    # Synchronous execution
    try:
        result = await _execute_segment_analysis(request, adjustment=adjustment)
        result.analysis_id = analysis_id
        await _analyses_store.set(analysis_id, result)
        return result
    except HTTPException:
        # F-010-backend (#429, codex iter-1 M1): preserve 503 from
        # agent-import guard.
        raise
    except HeavyComputeSaturated:
        # OOM guard (#1293): the worker rejected this heavy fit fast because it is
        # saturated. Re-raise so the app-level handler maps it to 503 + Retry-After
        # — must precede the broad handler below or it becomes a 500. Nothing was
        # started, so we store no FAILED record; the client simply retries.
        raise
    except SegmentAnalysisBudgetExceeded as e:
        # #1840: the run was cancelled at the budget and the slot released.
        # Persist an honest FAILED record naming the budget and answer 504 (the
        # server gave up on the upstream computation), not a generic 500.
        detail = _budget_exceeded_warning(e.budget_seconds)
        logger.warning(f"Segment analysis {analysis_id} {detail}")
        response.status = SegmentAnalysisStatus.FAILED
        response.warnings.append(detail)
        await _analyses_store.set(analysis_id, response)
        raise HTTPException(status_code=504, detail=detail) from e
    except Exception as e:
        logger.error(f"Segment analysis failed: {e}", exc_info=True)
        response.status = SegmentAnalysisStatus.FAILED
        # Store a generic warning on the persisted FAILED record rather than the
        # raw exception text (the record is later returned to clients via GET).
        response.warnings.append("Segment analysis failed due to an internal error.")
        await _analyses_store.set(analysis_id, response)
        # Do not echo raw exception text to the client (it can leak internal
        # paths/identifiers); the full exception is logged above with exc_info.
        raise HTTPException(
            status_code=500, detail="Segment analysis failed due to an internal error."
        ) from e


@router.get(
    "/policies",
    response_model=PolicyListResponse,
    summary="List targeting recommendations",
    operation_id="list_policies",
    description="List all targeting policy recommendations.",
)
async def list_policies(
    min_lift: Optional[float] = Query(default=None, description="Minimum expected lift threshold"),
    min_confidence: Optional[float] = Query(
        default=None, description="Minimum confidence threshold"
    ),
    limit: int = Query(default=20, description="Maximum results", ge=1, le=100),
) -> PolicyListResponse:
    """
    List targeting policy recommendations.

    Args:
        min_lift: Minimum expected lift threshold
        min_confidence: Minimum confidence threshold
        limit: Maximum number of results

    Returns:
        List of policy recommendations
    """
    all_recommendations: List[PolicyRecommendation] = []
    total_lift = 0.0

    for analysis in await _analyses_store.values():
        if analysis.status != SegmentAnalysisStatus.COMPLETED:
            continue

        for rec in analysis.policy_recommendations:
            # Apply filters
            if min_lift and rec.expected_incremental_outcome < min_lift:
                continue
            if min_confidence and rec.confidence < min_confidence:
                continue

            all_recommendations.append(rec)
            total_lift += rec.expected_incremental_outcome

    # Sort by expected outcome and limit
    all_recommendations.sort(key=lambda x: x.expected_incremental_outcome, reverse=True)
    all_recommendations = all_recommendations[:limit]

    return PolicyListResponse(
        total_count=len(all_recommendations),
        recommendations=all_recommendations,
        expected_total_lift=total_lift,
    )


@router.get(
    "/health",
    response_model=SegmentHealthResponse,
    summary="Segment analysis service health",
    operation_id="get_segment_health",
    description="Check health status of the segment analysis service.",
)
async def get_segment_health() -> SegmentHealthResponse:
    """
    Get health status of segment analysis service.

    Returns:
        Service health information
    """
    # Check agent availability
    agent_available = True
    try:
        from src.agents.heterogeneous_optimizer import HeterogeneousOptimizerAgent  # noqa: F401

        agent_available = True
    except ImportError:
        agent_available = False

    # Check library availability
    econml_available = True
    causalml_available = True
    try:
        import econml  # noqa: F401
    except ImportError:
        econml_available = False

    try:
        import causalml  # noqa: F401
    except ImportError:
        causalml_available = False

    # Count recent analyses
    now = datetime.now(timezone.utc)
    all_analyses = await _analyses_store.values()
    analyses_24h = sum(1 for a in all_analyses if (now - a.timestamp).total_seconds() < 86400)

    # Get last analysis
    last_analysis = None
    if all_analyses:
        last_analysis = max(a.timestamp for a in all_analyses)

    # #4 — surface whether this worker is serving DURABLE (Redis, cross-worker)
    # or DEGRADED (process-local in-memory) state. A silent per-worker fallback
    # re-introduces the cross-worker 404 this store exists to fix, so it must be
    # observable rather than invisible.
    durable = await _analyses_store.is_durable()
    storage_mode = "durable" if durable else "degraded"

    status = "healthy"
    if not agent_available:
        status = "degraded"
    elif not durable:
        # Storage degraded to the in-memory fallback -> cross-worker reads can
        # 404. This is a real, user-visible degradation, so report it.
        status = "degraded"
    elif not (econml_available and causalml_available):
        status = "partial"

    return SegmentHealthResponse(
        status=status,
        agent_available=agent_available,
        econml_available=econml_available,
        causalml_available=causalml_available,
        last_analysis=last_analysis,
        analyses_24h=analyses_24h,
        storage_mode=storage_mode,
    )


# =============================================================================
# DATASET CONFIG ENDPOINT
# =============================================================================
#
# ⚠️ ROUTE ORDER: this literal "/datasets" route MUST be declared BEFORE the
# greedy "/{analysis_id}" route below. FastAPI matches routes in declaration
# order with no specificity ranking, so if "/{analysis_id}" comes first a
# request to GET /segments/datasets is captured by it (analysis_id="datasets")
# -> get_segment_analysis -> 404. That silently degraded the FE config dropdowns
# to a single curated default each (brands empty -> only "All brands"). Keep all
# fixed-path GET routes ahead of the path-param route. (See test_segments.py:
# test_get_segment_datasets_route_not_shadowed_by_analysis_id.)


# Provenance of the /datasets option lists (see get_segment_datasets).
_SEGMENT_OPTIONS_SOURCE_SSOT = "causal_paths"
_SEGMENT_OPTIONS_SOURCE_FALLBACK = "curated_fallback"


class SegmentDatasetsResponse(BaseModel):
    """Curated config options for the Segment Analysis page (data-driven FE)."""

    treatments: List[str] = Field(
        ..., description="Selectable treatment columns, scoped to `brand` (curated spec order)"
    )
    outcomes: List[str] = Field(
        ..., description="Union of selectable outcome columns over `outcomes_by_treatment`"
    )
    brands: List[str] = Field(
        default_factory=list,
        description="Distinct brands present in the gold-standard cohort (filter)",
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable display labels keyed by column name",
    )
    outcomes_by_treatment: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Outcomes with a modeled causal edge from each offered treatment "
            "(causal_paths SSOT, brand-scoped). Empty when options_source is the "
            "curated fallback — the FE then offers the flat `outcomes` list."
        ),
    )
    brand: Optional[str] = Field(
        default=None, description="Brand the options are scoped to (None = all brands)"
    )
    options_source: str = Field(
        default=_SEGMENT_OPTIONS_SOURCE_SSOT,
        description=(
            f"'{_SEGMENT_OPTIONS_SOURCE_SSOT}' when derived from the causal-path registry; "
            f"'{_SEGMENT_OPTIONS_SOURCE_FALLBACK}' when the registry was unavailable and "
            "the flat curated allowlists were returned instead"
        ),
    )


async def _segment_question_options(
    brand: Optional[str],
) -> tuple[List[str], List[str], Dict[str, List[str]]]:
    """SSOT-derived ``treatment -> [outcomes]`` options for the HTE page.

    Reads the distinct ``(treatment, outcome, brand)`` questions from the
    ``causal_paths`` registry — the SAME enumeration the discovery leaderboard
    uses (``causal._discover_candidate_questions``) — restricted to this page's
    patient_journeys grain (treatment AND outcome in the curated spec, t != o),
    then gates the treatment axis per brand through ``_brand_scoped_covariates``:
    a brand-DISTINCT axis (complement_inhibitor_status / disease_stage /
    urticaria_severity_uas7) is offered only on its own brand's cohort and never
    for all-brands, exactly like the covariates (the column is NULL off-brand,
    so an off-brand run fails closed with "No usable rows").

    Why SSOT pairs and not the flat allowlists (2026-08-29 /segment-analysis
    review): the flat lists offered treatment_initiated on BOTH sides (it is a
    treatment only in the commercial grain; on the patient grain it is an
    outcome) and let a user pose treatment_initiated -> persistent_180d — a pair
    with no DGP effect and no registry edge, where the EconML/CausalML
    cross-check then honestly FAILED at 42%. Offering only modeled questions
    removes the ill-posed pairs at the source; the run-time allowlist gate in
    ``_load_segment_hte_frame`` is unchanged (security), and the API still
    accepts any allowlisted pair for programmatic callers.

    Lists follow the curated spec order (stable dropdowns). Raises on registry
    unavailability — the caller falls back to the flat curated lists.
    """
    from src.api.routes.causal import (
        _CAUSAL_DATASET_SPECS,
        _brand_scoped_covariates,
        _get_causal_path_repo,
    )

    spec = _CAUSAL_DATASET_SPECS[_SEGMENT_HTE_DATASET]
    t_order = {c: i for i, c in enumerate(spec["treatment"])}
    o_order = {c: i for i, c in enumerate(spec["outcome"])}

    repo = await _get_causal_path_repo()
    rows = await repo.get_distinct_questions(brand=brand, include_synthetic=True)

    pairs: Dict[str, set] = {}
    for r in rows:
        t, o = r.get("treatment"), r.get("outcome")
        # Grain-scope guard (same as the leaderboard): commercial-grain edges such
        # as treatment_initiated -> nrx_volume share the table and must not leak.
        if t == o or t not in t_order or o not in o_order:
            continue
        pairs.setdefault(t, set()).add(o)

    treatments = _brand_scoped_covariates(sorted(pairs, key=t_order.__getitem__), brand)
    outcomes_by_treatment = {t: sorted(pairs[t], key=o_order.__getitem__) for t in treatments}
    outcomes = sorted(
        {o for outs in outcomes_by_treatment.values() for o in outs}, key=o_order.__getitem__
    )
    return treatments, outcomes, outcomes_by_treatment


@router.get(
    "/datasets",
    response_model=SegmentDatasetsResponse,
    summary="Curated segment-analysis config options",
    operation_id="get_segment_datasets",
    description=(
        "Brand-scoped treatment/outcome options (causal_paths SSOT: only pairs "
        "with a modeled causal edge on the selected brand's cohort) + data-driven "
        "brand list for the agent-driven Segment Analysis page (patient_journeys "
        "substrate). Falls back to the flat curated allowlists when the registry "
        "is unavailable (options_source tells which)."
    ),
)
async def get_segment_datasets(
    brand: Optional[str] = Query(
        None,
        description=(
            "Brand the analysis will be scoped to. Treatment options are brand-scoped "
            "(a brand-distinct clinical axis is offered only on its own cohort) and "
            "each treatment lists only the outcomes it has a modeled causal edge to. "
            "Omitted = all brands (universal arms only)."
        ),
    ),
) -> SegmentDatasetsResponse:
    """Return brand-scoped treatment/outcome options and the live brand list.

    Treatment/outcome pairs come from the causal_paths SSOT via
    ``_segment_question_options``; brands are data-driven (distinct brands in
    the live cohort). Both are fail-soft: an unavailable registry returns the
    flat patient_journeys allowlists (brand-gated) with
    ``options_source="curated_fallback"``; an unavailable brand list returns
    ``[]`` (FE shows "All brands"). An unknown brand is a 400.
    """
    from src.api.routes.causal import (
        _CAUSAL_DATASET_SPECS,
        _COLUMN_LABELS,
        _brand_scoped_covariates,
    )

    spec = _CAUSAL_DATASET_SPECS[_SEGMENT_HTE_DATASET]
    brand = brand or None

    brands: List[str] = []
    try:
        from src.api.routes.causal import _list_dataset_brands

        brands = await _list_dataset_brands(_SEGMENT_HTE_DATASET)
    except Exception as e:  # pragma: no cover - fail-soft, FE shows "All brands"
        logger.warning(f"Segment datasets: brand list unavailable, returning []: {e}")
        brands = []

    if brand is not None and brands and brand not in brands:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown brand '{brand}' for dataset '{_SEGMENT_HTE_DATASET}'. Known: {brands}",
        )

    options_source = _SEGMENT_OPTIONS_SOURCE_SSOT
    try:
        treatments, outcomes, outcomes_by_treatment = await _segment_question_options(brand)
        if not treatments:
            raise RuntimeError("registry returned no patient-grain questions")
    except Exception as e:
        logger.warning(
            "Segment datasets: causal_paths options unavailable (%s); "
            "returning the flat curated allowlists for brand=%s",
            e,
            brand,
        )
        options_source = _SEGMENT_OPTIONS_SOURCE_FALLBACK
        treatments = _brand_scoped_covariates(list(spec["treatment"]), brand)
        outcomes = list(spec["outcome"])
        outcomes_by_treatment = {}

    offered = list(dict.fromkeys(treatments + outcomes))
    labels = {c: _COLUMN_LABELS.get(c, c.replace("_", " ").capitalize()) for c in offered}

    return SegmentDatasetsResponse(
        treatments=treatments,
        outcomes=outcomes,
        brands=brands,
        labels=labels,
        outcomes_by_treatment=outcomes_by_treatment,
        brand=brand,
        options_source=options_source,
    )


@router.get(
    "/{analysis_id}",
    response_model=SegmentAnalysisResponse,
    summary="Get segment analysis results",
    operation_id="get_segment_analysis",
    description="Retrieve results of a segment analysis by ID.",
)
async def get_segment_analysis(analysis_id: str) -> SegmentAnalysisResponse:
    """
    Get segment analysis results by ID.

    Args:
        analysis_id: Unique analysis identifier

    Returns:
        Segment analysis results

    Raises:
        HTTPException: If analysis not found
    """
    analysis = await _analyses_store.get(analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail=f"Segment analysis {analysis_id} not found",
        )

    return analysis


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
#
# Clinical-HTE rebuild (2026-06-20): the Segment Analysis page is agent-driven
# over the curated ``patient_journeys`` gold-standard substrate (the same SSOT
# the /causal pages use — see src/api/routes/causal.py ``_CAUSAL_DATASET_SPECS``).
# The route loads the frame SERVER-SIDE (provenance-aware: ``apply_provenance_filter``
# INCLUDES the is_synthetic=true gold-standard rows on this synthetic-showcase
# deployment), bands the continuous clinical columns, and passes the prepared
# frame via the ``tier0_frame_ref`` frame-registry handle (#1734 — the frame
# itself never enters graph state) so the CATE / hierarchical / uplift nodes
# all consume ONE banded frame (no connector fetch, no double-read). The
# clinical contract below is FIXED server-side; only treatment/outcome are
# selectable (curated).

# Dataset the Segment Analysis page reads (gold-standard patient cohort).
_SEGMENT_HTE_DATASET = "patient_journeys"

# Curated defaults when the request omits treatment/outcome.
_SEGMENT_HTE_DEFAULT_TREATMENT = "treatment_arm"
_SEGMENT_HTE_DEFAULT_OUTCOME = "persistent_180d"

# Effect modifiers (X -> heterogeneity model + feature_importance). NUMERIC only
# (no region — categoricals would need encoding the heterogeneity model does not
# do here). engagement_score is a PURE CONTROL (W, below), deliberately NOT in X
# so there is no X/W overlap (codex MED-2).
_SEGMENT_HTE_EFFECT_MODIFIERS = [
    "disease_severity",
    "age_at_diagnosis",
    "academic_hcp",
    "ecog_performance_status",
    "egfr",
    "proteinuria_g_day",
    "ldh_ratio",
    "urticaria_severity_uas7",
]


def _segment_effect_modifiers(
    brand: Optional[str],
    *,
    treatment_var: Optional[str] = None,
    outcome_var: Optional[str] = None,
) -> List[str]:
    """Brand-aware heterogeneity features (X) for the HTE run (Phase 2 brand-gating).

    The 5 indication-specific clinical modifiers (ecog/egfr/proteinuria/ldh/uas7) are
    populated for only their own brand's rows after gating; feeding an off-brand
    (NULL) column to CausalForestDML raises ``Input contains NaN``. Reuses the causal
    route's SSOT ``_brand_scoped_covariates`` so a single map governs both surfaces:
    universals (disease_severity/age/academic_hcp) always survive; a brand's own
    clinical modifier survives only when that brand is the row filter; brand=None
    (all brands) keeps the universals only.

    The question slots are removed from X (wave 53). Since #1321 a brand's own
    clinical column is BOTH a curated effect modifier and that brand's treatment
    axis (Remibrutinib: urticaria_severity_uas7). With the treatment inside X the
    median-split T is a deterministic function of an X column, CausalForestDML's
    propensity model is perfect and its residual is zero — live seg_05f29d1b3295
    returned ATE -0.514 on a 0/1 outcome against a planted +0.150, and dropping
    the column from X alone recovered +0.140. The causal page dedups the same way
    on its submit path.
    """
    from src.api.routes.causal import _brand_scoped_covariates

    scoped = _brand_scoped_covariates(list(_SEGMENT_HTE_EFFECT_MODIFIERS), brand)
    return [c for c in scoped if c not in (treatment_var, outcome_var)]


# Confounders (W -> pure controls routed into the DML nuisance model, NOT in X).
_SEGMENT_HTE_CONFOUNDERS = ["engagement_score"]

# Segment dimensions (post-hoc CATE breakdown). RAW categoricals that EXIST in
# the prepared frame — banded continuous columns + naturally-categorical columns.
# cate_estimator groups by ``df[seg].unique()`` and skips segments with <10 rows,
# so these must be low-cardinality strings, never raw floats (codex HIGH-4).
_SEGMENT_HTE_SEGMENT_VARS = [
    "disease_severity_band",
    "age_band",
    "geographic_region",
    "ecog_performance_status",
    "academic_hcp",
]

# NUMERIC clinical covariates loaded from patient_journeys (kept intact alongside
# the banded string columns). geographic_region is loaded RAW (categorical) and
# is therefore NOT in this set (it must not be float-coerced to None).
_SEGMENT_HTE_NUMERIC_COLUMNS = {
    "treatment_arm",
    "treatment_initiated",
    "persistent_180d",
    "discontinued_180d",
    "disease_severity",
    "engagement_score",
    "age_at_diagnosis",
    "academic_hcp",
    "ecog_performance_status",
    "egfr",
    "proteinuria_g_day",
    "ldh_ratio",
    "urticaria_severity_uas7",
    # copay_support's DGP backdoor (treatment_arm.ARM_REGISTRY): routed as W when
    # the registry edge names it (_segment_question_adjustment) — must float-coerce.
    "insurance_access_score",
}

# Raw categorical columns kept as strings for post-hoc segmentation.
_SEGMENT_HTE_CATEGORICAL_COLUMNS = {"geographic_region"}

# Max rows pulled for the gold-standard cohort (whole synthetic cohort fits well
# under this; mirrors the generous causal-loader ceiling).
_SEGMENT_HTE_ROW_LIMIT = 100_000


class _SegmentQuestionAdjustment(NamedTuple):
    """Resolved nuisance controls (W) for one segment-analysis question."""

    confounders: List[str]
    # True: a causal_paths edge backs the pair; False: none does; None: the
    # registry could not be read (fail-soft on the default W).
    modeled: Optional[bool]
    warnings: List[str]
    # Outcomes the registry DOES model for this treatment in scope (allowlisted
    # for this dataset) — only populated for an unmodeled pair, to name the
    # alternatives in the #1827 refusal.
    modeled_outcomes: Tuple[str, ...] = ()


_SEGMENT_UNMODELED_QUESTION_WARNING = (
    "'{treatment} -> {outcome}' is not a modeled causal question in the causal_paths "
    "registry for {scope}: no validated causal edge backs it, so the estimates below "
    "may reflect confounding rather than a real effect and cross-library validation "
    "is likely to fail. Choose a registered pair (GET /segments/datasets?brand=...)."
)
_SEGMENT_REGISTRY_UNAVAILABLE_WARNING = (
    "Could not verify '{treatment} -> {outcome}' against the causal_paths registry "
    "(registry unavailable); using the default adjustment set {confounders}."
)
_SEGMENT_UNMODELED_QUESTION_REFUSAL = (
    "'{treatment} -> {outcome}' is not a modeled causal question in the causal_paths "
    "registry for {scope}: no validated causal edge backs it, so an estimate would "
    "reflect confounding rather than a real effect. {alternatives} "
    "(GET /segments/datasets?brand=...). Pass allow_unmodeled=true to run it anyway "
    "(exploratory; the result carries a not-a-modeled-question warning)."
)


def _refuse_unmodeled_question(
    request: RunSegmentAnalysisRequest,
    adjustment: "_SegmentQuestionAdjustment",
    treatment_var: str,
    outcome_var: str,
) -> None:
    """#1827: 400 on a pair with NO registry edge unless the caller opted in.

    ``modeled is None`` (registry unreadable) is deliberately NOT refused — that
    path is fail-soft on the default W with its own warning, so a registry
    outage never takes the page down. ``modeled is True`` runs normally.
    """
    if adjustment.modeled is not False or request.allow_unmodeled:
        return
    scope = request.brand or "any brand"
    if adjustment.modeled_outcomes:
        alternatives = (
            f"Modeled outcomes for '{treatment_var}' on {scope}: "
            f"{', '.join(adjustment.modeled_outcomes)}"
        )
    else:
        alternatives = (
            f"'{treatment_var}' has no modeled outcome on {scope} at this grain "
            "(it may be an outcome, not a treatment)"
        )
    raise HTTPException(
        status_code=400,
        detail=_SEGMENT_UNMODELED_QUESTION_REFUSAL.format(
            treatment=treatment_var,
            outcome=outcome_var,
            scope=scope,
            alternatives=alternatives,
        ),
    )


async def _segment_question_adjustment(
    *,
    treatment_var: str,
    outcome_var: str,
    brand: Optional[str],
    effect_modifiers: List[str],
) -> _SegmentQuestionAdjustment:
    """Derive W for (treatment, outcome, brand) from the causal_paths registry.

    Why: the page's W used to be the FIXED ``_SEGMENT_HTE_CONFOUNDERS``. That is
    complete for the arms whose DGP backdoor is covered by X ∪ W (psp / rep /
    sample / trigger / treatment_arm) but NOT for copay_support, whose assignment
    depends on ``insurance_access_score`` (treatment_arm.ARM_REGISTRY) — a column
    in neither X nor W, so its estimate reported the confounded diff (measured
    2026-08-29 on Remibrutinib: raw +0.2pp vs +2.7pp within insurance-access
    bins). The registry stores each modeled edge's ``confounders_controlled``;
    this reads it — the SAME enumeration the discovery leaderboard uses
    (``causal._discover_candidate_questions``) — and routes the members that are
    numeric allowlisted covariates and NOT already in X (no X/W overlap, codex
    MED-2). Categoricals the segment loader keeps RAW for post-hoc grouping
    (``geographic_region``) are excluded: the nuisance models would only see a
    label-encoded ordinal, and no DGP arm is assigned on region.

    A pair with NO registry edge (e.g. treatment_initiated -> persistent_180d: no
    planted effect) resolves to the default W, ``modeled=False`` and an explicit
    warning, plus the outcomes the registry DOES model for that treatment. The
    route refuses such a pair with 400 unless ``allow_unmodeled`` is set (#1827,
    :func:`_refuse_unmodeled_question`); opted-in runs carry the warning — the
    case that used to surface only as an unexplained "cross-library validation
    FAILED".

    Fail-soft: if the registry cannot be read the default W is used and the run
    says so (the frame loader remains the fail-closed gate for the substrate).
    """
    from src.api.routes.causal import (
        _CAUSAL_DATASET_SPECS,
        _CAUSAL_NUMERIC_COLUMNS,
        _brand_scoped_covariates,
        _get_causal_path_repo,
    )

    default_w = list(_SEGMENT_HTE_CONFOUNDERS)
    try:
        repo = await _get_causal_path_repo()
        rows = await repo.get_distinct_questions(brand=brand, include_synthetic=True)
    except Exception as exc:
        logger.warning(
            f"causal_paths registry unavailable for segment question "
            f"{treatment_var}->{outcome_var}: {exc}; using default W {default_w}"
        )
        return _SegmentQuestionAdjustment(
            confounders=default_w,
            modeled=None,
            warnings=[
                _SEGMENT_REGISTRY_UNAVAILABLE_WARNING.format(
                    treatment=treatment_var, outcome=outcome_var, confounders=default_w
                )
            ],
        )

    modeled_rows = [
        r for r in rows if r.get("treatment") == treatment_var and r.get("outcome") == outcome_var
    ]
    spec = _CAUSAL_DATASET_SPECS[_SEGMENT_HTE_DATASET]
    if not modeled_rows:
        # Name the alternatives: outcomes this treatment IS modeled against in
        # scope, restricted to this dataset's outcome allowlist (the registry
        # also holds HCP-grain edges such as treatment_initiated -> nrx_volume
        # that this endpoint cannot run).
        allowed_outcomes = set(spec["outcome"])
        modeled_outcomes = tuple(
            sorted(
                {
                    str(r.get("outcome"))
                    for r in rows
                    if r.get("treatment") == treatment_var and r.get("outcome") in allowed_outcomes
                }
            )
        )
        return _SegmentQuestionAdjustment(
            confounders=default_w,
            modeled=False,
            warnings=[
                _SEGMENT_UNMODELED_QUESTION_WARNING.format(
                    treatment=treatment_var,
                    outcome=outcome_var,
                    scope=brand or "any brand",
                )
            ],
            modeled_outcomes=modeled_outcomes,
        )

    numeric_allowlisted = set(spec["covariate"]) & _CAUSAL_NUMERIC_COLUMNS.get(
        _SEGMENT_HTE_DATASET, set()
    )
    excluded = (
        set(effect_modifiers) | {treatment_var, outcome_var} | set(_SEGMENT_HTE_CATEGORICAL_COLUMNS)
    )
    registry_w: List[str] = []
    for row in modeled_rows:
        for col in row.get("confounders") or []:
            if col in numeric_allowlisted and col not in excluded and col not in registry_w:
                registry_w.append(col)
    confounders = _brand_scoped_covariates(list(dict.fromkeys(default_w + registry_w)), brand)
    return _SegmentQuestionAdjustment(confounders=confounders, modeled=True, warnings=[])


def _segment_analysis_dedup_key(
    request: RunSegmentAnalysisRequest,
    adjustment: "_SegmentQuestionAdjustment",
    effect_modifiers: List[str],
) -> str:
    """Identity of ONE resolved segment-analysis question (#1840 dedup key).

    Two POSTs are "identical" when the whole wire body matches (brand,
    treatment, outcome, query, filters, estimator settings, opt-ins — every
    field of the request model) AND the scoping the run will actually use
    matches: the registry-derived adjustment set W (``_segment_question_
    adjustment``) and the brand-scoped effect modifiers X (``_segment_effect_
    modifiers``). Keying on the raw body alone would collapse two submissions
    that resolve to different W (e.g. the registry gained an edge between them)
    into one run; keying on the resolved sets never does.

    Sets are order-normalised; the request is dumped in JSON mode with sorted
    keys so the digest is stable across processes.
    """
    payload = {
        "request": request.model_dump(mode="json"),
        "effect_modifiers": sorted(effect_modifiers),
        "confounders": sorted(adjustment.confounders),
        "modeled": adjustment.modeled,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _band_disease_severity(value: Any) -> Optional[str]:
    """Band disease_severity into low/medium/high matching the DGP segments.

    Matches src/ml/synthetic/generators/patient_generator.py: severity > 7 ->
    high (strong CATE), > 4 -> medium, else low. Returns None when the value is
    missing / non-numeric so the band column is honestly empty for that row
    rather than mislabeled.
    """
    try:
        sev = float(value)
    except (TypeError, ValueError):
        return None
    if sev > 7:
        return "high"
    if sev > 4:
        return "medium"
    return "low"


def _band_age(value: Any) -> Optional[str]:
    """Band age_at_diagnosis into <50 / 50-65 / >65 string buckets."""
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if age < 50:
        return "<50"
    if age <= 65:
        return "50-65"
    return ">65"


async def _load_segment_hte_frame(
    *,
    brand: Optional[str],
    treatment_var: str,
    outcome_var: str,
    effect_modifiers: Optional[List[str]] = None,
    confounders: Optional[List[str]] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined] # noqa: F821
    """Load the REAL gold-standard ``patient_journeys`` frame for the HTE agent.

    Mirrors ``causal.py._load_agent_estimation_frame`` for the patient_journeys
    dataset, with two deliberate differences for the segment-analysis use-case:

    * ``geographic_region`` is kept as a RAW string column (NOT one-hot encoded):
      it is a post-hoc SEGMENT dimension that ``cate_estimator`` groups by via
      ``df[seg].unique()``; one-hot-dropping it would destroy the segmentation.
    * the continuous clinical columns are BANDED into new low-cardinality string
      columns (``disease_severity_band`` low/medium/high matching the DGP;
      ``age_band`` <50/50-65/>65) so per-segment stratification has enough rows
      per band (cate_estimator skips segments with <10 rows — raw floats would
      all be skipped).
    * a #1321 clinical axis in a QUESTION slot is derived to its 0/1 contrast
      (``_CAUSAL_NUMERIC_DERIVATIONS``, same as the causal page); the same column
      as an effect modifier stays raw (wave 53).

    Security / honesty gates (same posture as the causal loader):
      * treatment/outcome validated against the patient_journeys allowlist
        (``_CAUSAL_DATASET_SPECS``) -> ``HTTPException`` 400 on a disallowed column.
      * ``apply_provenance_filter`` (synthetic-showcase-aware: INCLUDES the
        is_synthetic=true gold-standard rows on this deployment — reused, NOT a
        local include_synthetic flag).
      * brand ``.eq("brand", brand)`` when a brand is given (row FILTER).
      * FAIL-CLOSED: a 503 with a specific message if the frame is empty after
        coercion. NEVER returns an empty / fabricated frame silently.
    """
    import pandas as pd

    # SSOT for the curated allowlist lives in causal.py (single source of truth).
    from src.api.routes.causal import (
        _CAUSAL_DATASET_SPECS,
        _CAUSAL_NUMERIC_DERIVATIONS,
        _coerce_estimation_row,
    )

    spec = _CAUSAL_DATASET_SPECS[_SEGMENT_HTE_DATASET]
    allowed = set(spec["treatment"]) | set(spec["outcome"]) | set(spec["covariate"])
    not_allowed = [c for c in (treatment_var, outcome_var) if c not in allowed]
    if not_allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed} are not permitted for dataset "
                f"'{_SEGMENT_HTE_DATASET}'. Allowed: {sorted(allowed)}"
            ),
        )

    # Columns to pull: treatment, outcome, the numeric effect-modifiers (X), the
    # confounders (W — engagement_score is W-ONLY, not in X, so it must be loaded
    # here explicitly), and raw geographic_region. Constrained to the allowlist so
    # a typo cannot inject an arbitrary column into the select (42703 — codex HIGH-3).
    # Brand-aware effect modifiers (Phase 2): default to the full curated list, but
    # the orchestrator passes the brand-scoped subset so an off-brand (gated NULL)
    # clinical column is never selected → never reaches CausalForestDML as NaN.
    modifiers = (
        list(effect_modifiers)
        if effect_modifiers is not None
        else list(_SEGMENT_HTE_EFFECT_MODIFIERS)
    )
    # W: the run's resolved adjustment set (registry-derived via
    # _segment_question_adjustment) — default to the fixed control so direct callers
    # keep today's behaviour.
    controls = list(confounders) if confounders is not None else list(_SEGMENT_HTE_CONFOUNDERS)
    base_cols = [treatment_var, outcome_var] + modifiers + controls + ["geographic_region"]
    select_cols = [c for c in dict.fromkeys(base_cols) if c in allowed]
    # Every W column is numeric (categoricals are excluded from W upstream), so
    # a registry-derived control outside the curated numeric list still gets
    # float-coerced rather than reaching EconML as an object column.
    numeric_cols = _SEGMENT_HTE_NUMERIC_COLUMNS | {
        c for c in controls if c not in _SEGMENT_HTE_CATEGORICAL_COLUMNS
    }
    # Question-slot derivations (wave 53): when the treatment (or outcome) is one
    # of the #1321 brand-distinct clinical axes, run the SAME 0/1 contrast the
    # causal page and the causal_paths edge use — "Uncontrolled CSU (UAS7 >= 28)",
    # "Advanced line", "Prior C5-inhibitor" — instead of the raw column. Raw, the
    # numeric axis was median-split by the nodes (a different, unlabeled cut) and
    # the two TEXT axes reached cate_estimator as strings and failed closed
    # ("entirely null/non-numeric after coercion"). Scoped to the question slots
    # ONLY: as an effect modifier the raw score keeps its resolution.
    derivations = {
        col: fn
        for col, fn in _CAUSAL_NUMERIC_DERIVATIONS.get(_SEGMENT_HTE_DATASET, {}).items()
        if col in (treatment_var, outcome_var)
    }
    numeric_cols = numeric_cols | set(derivations)

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Segment analysis data store unavailable")

    fetch_cols = list(select_cols)
    if brand:
        # patient_journeys uses the standard ``brand`` column.
        fetch_cols = list(dict.fromkeys([*select_cols, "brand"]))

    query = client.table(_SEGMENT_HTE_DATASET).select(",".join(fetch_cols))
    # Provenance-aware, env-gated (mirrors causal.py's loader): apply_provenance_filter
    # skips the is_synthetic=False predicate when deployment_includes_synthetic()
    # (E2I_INCLUDE_SYNTHETIC) is set, so on this synthetic-gold showcase it LOADS the
    # gold-standard rows. Deliberately NOT include_synthetic=True — hardcoding True
    # would break the platform's env-reversibility (unset the env => the strict gate
    # returns for EVERY reader). If the env is unset there is no gold-standard
    # substrate and the fail-closed 503 below fires, same as every other causal page.
    query = apply_provenance_filter(query)
    if brand:
        query = query.eq("brand", brand)
    result = await query.limit(_SEGMENT_HTE_ROW_LIMIT).execute()
    rows = result.data or []

    records: List[Dict[str, Any]] = []
    for row in rows:
        rec = _coerce_estimation_row(
            row,
            select_cols=select_cols,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            numeric_cols=numeric_cols,
            # geographic_region passes through as a raw string (not float-coerced).
            categorical_cols=frozenset(_SEGMENT_HTE_CATEGORICAL_COLUMNS),
            derivations=derivations,
        )
        if rec is not None:
            records.append(rec)

    if not records:
        # FAIL-CLOSED: never return an empty / fabricated frame.
        scope = brand or "all brands"
        raise HTTPException(
            status_code=503,
            detail=(
                f"No usable patient_journeys rows for {scope}/{treatment_var}->"
                f"{outcome_var}. The synthetic gold-standard substrate returned "
                "no rows; not fabricating results."
            ),
        )

    frame = pd.DataFrame(records)

    # Band continuous columns into new RAW string columns for clean
    # stratification (cate_estimator skips <10-row segments). Numeric columns are
    # left intact too (they remain the effect modifiers X).
    if "disease_severity" in frame.columns:
        frame["disease_severity_band"] = frame["disease_severity"].map(_band_disease_severity)
    if "age_at_diagnosis" in frame.columns:
        frame["age_band"] = frame["age_at_diagnosis"].map(_band_age)

    return frame


async def _run_segment_analysis_task(
    analysis_id: str,
    request: RunSegmentAnalysisRequest,
    adjustment: Optional[_SegmentQuestionAdjustment] = None,
    dedup_key: Optional[str] = None,
) -> None:
    """Background task to run segment analysis.

    ``adjustment`` is the question adjustment the POST handler already resolved
    (and gated, #1827); passing it through avoids a second registry read.
    ``dedup_key`` is the in-flight marker the handler claimed for this run
    (#1840); it is released once the run reaches a terminal state, whatever
    the outcome.
    """
    try:
        await _run_segment_analysis_task_body(analysis_id, request, adjustment)
    finally:
        if dedup_key is not None:
            await _analyses_store.release_inflight(dedup_key, analysis_id)


async def _run_segment_analysis_task_body(
    analysis_id: str,
    request: RunSegmentAnalysisRequest,
    adjustment: Optional[_SegmentQuestionAdjustment],
) -> None:
    try:
        logger.info(f"Starting segment analysis task {analysis_id}")

        # Update status (read-modify-write so the change persists to the store).
        pending = await _analyses_store.get(analysis_id)
        if pending is not None:
            pending.status = SegmentAnalysisStatus.ESTIMATING
            await _analyses_store.set(analysis_id, pending)

        # Execute analysis
        result = await _execute_segment_analysis(request, adjustment=adjustment)
        result.analysis_id = analysis_id

        # Store result
        await _analyses_store.set(analysis_id, result)

        logger.info(f"Segment analysis {analysis_id} completed successfully")

    except HTTPException as e:
        # Fail-closed reasons (the loader's 503 "no usable patient_journeys rows
        # for <brand>" or the 400 disallowed-column) carry a SAFE, specific detail
        # string — surface it so the FE tells the user WHY rather than a generic
        # "internal error". (This task runs the load+graph async, so these
        # HTTPExceptions land here rather than in the POST handler's response.)
        logger.error(f"Segment analysis {analysis_id} failed: {e.detail}")
        existing = await _analyses_store.get(analysis_id)
        if existing is not None:
            existing.status = SegmentAnalysisStatus.FAILED
            detail = e.detail if isinstance(e.detail, str) else "Segment analysis failed."
            existing.warnings.append(f"Segment analysis failed: {detail}")
            await _analyses_store.set(analysis_id, existing)
    except HeavyComputeSaturated:
        # OOM guard (#1293): the fit was rejected fast because the worker is
        # saturated (this async task never gets a 503 back to the client, so record
        # a SPECIFIC, safe warning on the durable store — reject-fast, not a silent
        # hang or a generic internal error). Must precede the broad handler below.
        logger.warning(f"Segment analysis {analysis_id} rejected: heavy compute saturated")
        existing = await _analyses_store.get(analysis_id)
        if existing is not None:
            existing.status = SegmentAnalysisStatus.FAILED
            existing.warnings.append(
                "Segment analysis rejected: compute capacity saturated; retry later."
            )
            await _analyses_store.set(analysis_id, existing)
    except SegmentAnalysisBudgetExceeded as e:
        # #1840: the graph run overran its budget, was cancelled, and the slot
        # was released inside _execute_segment_analysis. Record an honest FAILED
        # that NAMES the budget (not "internal error", not "capacity saturated"
        # — nothing rejected this run; it was stopped). Must precede the broad
        # handler below.
        detail = _budget_exceeded_warning(e.budget_seconds)
        logger.warning(f"Segment analysis {analysis_id} {detail}")
        existing = await _analyses_store.get(analysis_id)
        if existing is not None:
            existing.status = SegmentAnalysisStatus.FAILED
            existing.warnings.append(detail)
            await _analyses_store.set(analysis_id, existing)
    except Exception as e:
        logger.error(f"Segment analysis {analysis_id} failed: {e}")
        existing = await _analyses_store.get(analysis_id)
        if existing is not None:
            existing.status = SegmentAnalysisStatus.FAILED
            # Store a generic warning rather than raw exception text (the record
            # is later returned to clients via GET).
            existing.warnings.append("Segment analysis failed due to an internal error.")
            await _analyses_store.set(analysis_id, existing)


async def _execute_segment_analysis(
    request: RunSegmentAnalysisRequest,
    adjustment: Optional[_SegmentQuestionAdjustment] = None,
) -> SegmentAnalysisResponse:
    """
    Execute segment analysis using Heterogeneous Optimizer agent.

    This function orchestrates the Heterogeneous Optimizer agent (Tier 2) to:
    1. Estimate CATE via cate_estimator node
    2. Analyze segments via segment_analyzer node
    3. Learn policies via policy_learner node
    4. Generate profiles via profile_generator node
    """
    import time

    start_time = time.time()

    # Clinical-HTE rebuild: resolve the curated treatment/outcome (defaults when
    # omitted) and load the gold-standard patient_journeys frame SERVER-SIDE. The
    # loader enforces the curated allowlist (400 on a disallowed column) and
    # fails CLOSED (503) on an empty frame — these HTTPExceptions must propagate,
    # NOT be swallowed into a generic 500. We resolve + load BEFORE the import
    # guard so the 400 fires even if the agent package is importable.
    treatment_var = request.treatment_var or _SEGMENT_HTE_DEFAULT_TREATMENT
    outcome_var = request.outcome_var or _SEGMENT_HTE_DEFAULT_OUTCOME
    # Brand-aware X + segment dimensions (Phase 2): the SAME scoping drives the
    # loader's column selection AND the agent's effect_modifiers/segment_vars, so the
    # prepared frame and the CATE feature/breakdown sets stay in lock-step (no
    # off-brand NULL clinical column is loaded, so none reaches EconML or the
    # post-hoc grouping). ecog is a segment dimension only for Kisqali; the banded
    # (disease_severity_band/age_band) + universal (geographic_region/academic_hcp)
    # dimensions always survive.
    from src.api.routes.causal import _brand_scoped_covariates

    effect_modifiers = _segment_effect_modifiers(
        request.brand, treatment_var=treatment_var, outcome_var=outcome_var
    )
    segment_vars = _brand_scoped_covariates(list(_SEGMENT_HTE_SEGMENT_VARS), request.brand)
    # W (nuisance controls) come from the causal_paths registry edge for this
    # question (copay_support needs insurance_access_score; no other registered
    # pair adds to the default). The SAME list drives the loader's column
    # selection and the graph state so the frame and the DML W stay in lock-step.
    # An unmodeled pair is refused (400) unless the caller opted in via
    # allow_unmodeled, in which case it keeps the default W and is warned about on
    # the run (#1827). The POST handler resolves + gates before queuing and hands
    # the result in; direct callers resolve here and are gated identically.
    if adjustment is None:
        adjustment = await _segment_question_adjustment(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            brand=request.brand,
            effect_modifiers=effect_modifiers,
        )
        _refuse_unmodeled_question(request, adjustment, treatment_var, outcome_var)
    logger.info(
        f"Segment analysis {treatment_var}->{outcome_var} ({request.brand or 'all brands'}): "
        f"W={adjustment.confounders} modeled={adjustment.modeled}"
    )
    tier0_frame = await _load_segment_hte_frame(
        brand=request.brand,
        treatment_var=treatment_var,
        outcome_var=outcome_var,
        effect_modifiers=effect_modifiers,
        confounders=adjustment.confounders,
    )

    # The heterogeneous-optimizer graph fit — EconML CATE + CausalML uplift
    # estimated WITHIN each segment — is the genuinely heavy in-process compute
    # here. Hold ONE per-worker heavy-compute slot around it (and its mock
    # fallback) so concurrent background analyses can no longer each spawn fit
    # threads and push the api container past its 5G cgroup (OOM guard, #1293).
    # Pre-#1292 the inline loop-blocking fit accidentally serialised analyses
    # one-at-a-time per worker; once #1292 offloaded the fit with asyncio.to_thread
    # that implicit bound was gone. The Tier-0 frame load above is light I/O and
    # is deliberately OUTSIDE the slot. On a saturated worker heavy_compute_slot()
    # raises HeavyComputeSaturated on ENTER (nothing is queued); both callers of
    # this helper translate that into a fast reject — the sync route re-raises it
    # to the app handler (503 + Retry-After) and the background task records a
    # FAILED 'capacity saturated' analysis rather than hanging.
    async with heavy_compute_slot():
        try:
            # Try to use the actual Heterogeneous Optimizer agent
            from src.agents.heterogeneous_optimizer.agent import calculate_confidence
            from src.agents.heterogeneous_optimizer.graph import (
                create_heterogeneous_optimizer_graph,
            )
            from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState

            # #1734: stash the prepared frame in the process-local frame registry
            # for exactly the duration of the graph run — only the string handle
            # enters graph state. The raw ainvoke input dict is streamed verbatim
            # by the top-level on_chain_start event whenever this graph runs
            # under a streaming callback context, and a frame in state would
            # re-serialize into every node event (the 377.6 MB eval-4.4 turn),
            # so the frame itself must never be part of the state dict.
            from src.utils.frame_registry import stashed_frame

            # Initialize state (cast partial state - remaining fields populated by graph nodes).
            # The clinical contract is FIXED server-side (the prepared frame is passed
            # via the tier0 frame-registry handle so cate/hierarchical/uplift all
            # consume ONE banded frame):
            #   - effect_modifiers (X): numeric clinical covariates (drives feature
            #     importance), NO region.
            #   - confounders (W): engagement_score pure control — NOT in X (no overlap).
            #   - segment_vars: banded / raw categoricals present in the frame.
            with stashed_frame(tier0_frame, label="segments-hte") as tier0_ref:
                initial_state = cast(
                    HeterogeneousOptimizerState,
                    {
                        "query": request.query,
                        "treatment_var": treatment_var,
                        "outcome_var": outcome_var,
                        "segment_vars": list(segment_vars),
                        "effect_modifiers": list(effect_modifiers),
                        # Explicit confounders take precedence-1 in cate_estimator's
                        # _resolve_confounders and are residualized as the DML W (issue
                        # #237). Registry-derived per question (see above).
                        "confounders": list(adjustment.confounders),
                        # Handle to the prepared, banded gold-standard frame —
                        # resolved as tier0 priority-1 by cate_estimator /
                        # hierarchical / uplift via resolve_state_frame().
                        "tier0_frame_ref": tier0_ref,
                        "data_source": _SEGMENT_HTE_DATASET,
                        "filters": request.filters,
                        "n_estimators": request.n_estimators,
                        "min_samples_leaf": request.min_samples_leaf,
                        "significance_level": request.significance_level,
                        "top_segments_count": request.top_segments_count,
                        # Label-gater (opt-in): brand + indication + flag thread through to
                        # cate_estimator (segment augmentation) and policy_learner (the gate).
                        "brand": request.brand,
                        "indication": request.indication,
                        "label_segmentation": request.label_segmentation,
                        "status": "pending",
                        "errors": [],
                        # Seeded FIRST so an unmodeled-question warning precedes the
                        # validator's FAILED line in the persisted run (the
                        # append_unique channel keeps seeded items and order).
                        "warnings": list(adjustment.warnings),
                        "estimation_latency_ms": 0,
                        "analysis_latency_ms": 0,
                        "total_latency_ms": 0,
                    },
                )

                # Create and run graph. The factory resolves a SINGLE shared data
                # connector (when none is supplied) and passes it to BOTH data-fetching
                # nodes (cate_estimator + hierarchical_analyzer) so they read the same
                # live substrate; hierarchical_analyzer previously had no source and
                # raised RuntimeError mid-graph in production (#30). The resolution lives
                # in the factory (not here) so this function's import-guard / mock-
                # fallback contract — and the unit tests that patch the factory — stay
                # intact.
                graph = create_heterogeneous_optimizer_graph()
                # #1840 run budget. wait_for cancels the graph coroutine at the
                # deadline; the exception then unwinds through stashed_frame and
                # heavy_compute_slot, so the frame handle AND the worker's slot
                # are released — the next analysis on this worker is accepted
                # instead of being rejected until a restart. A fit thread the
                # nodes already handed to asyncio.to_thread cannot be stopped
                # (see SegmentAnalysisBudgetExceeded); the default budget sits
                # >4x above the slowest measured run so this is the
                # pathological-run escape hatch, not a routine event.
                budget_seconds = _segment_analysis_budget_seconds()
                try:
                    result = await asyncio.wait_for(
                        graph.ainvoke(initial_state), timeout=budget_seconds
                    )
                except asyncio.TimeoutError as timeout_exc:
                    raise SegmentAnalysisBudgetExceeded(budget_seconds) from timeout_exc

            # Convert agent output to API response
            total_latency = int((time.time() - start_time) * 1000)

            return SegmentAnalysisResponse(
                analysis_id="",  # Will be set by caller
                status=SegmentAnalysisStatus.COMPLETED
                if result.get("status") == "completed"
                else SegmentAnalysisStatus.FAILED,
                question_type=request.question_type,
                brand=request.brand,
                treatment_var=treatment_var,
                outcome_var=outcome_var,
                cate_by_segment=_convert_cate_results(result.get("cate_by_segment", {})),
                overall_ate=result.get("overall_ate"),
                # #27: echo the level the CATE CIs were computed at (alpha=significance_level).
                confidence_level=1.0 - request.significance_level,
                heterogeneity_score=result.get("heterogeneity_score"),
                # _to_native: coerce numpy scalars in the Dict/Any-typed fields below so
                # the durable store's model_dump_json() can serialize them (numpy.int64
                # is not JSON/Pydantic-serializable and otherwise fails the analysis).
                feature_importance=_to_native(result.get("feature_importance")),
                uplift_metrics=_convert_uplift_metrics(result),
                high_responders=_convert_segment_profiles(result.get("high_responders", [])),
                # mid_responders (responder_type="average") — the converter already
                # accepts "average"; default [] when the graph omits the key.
                mid_responders=_convert_segment_profiles(result.get("mid_responders", [])),
                low_responders=_convert_segment_profiles(result.get("low_responders", [])),
                policy_recommendations=_convert_policies(result.get("policy_recommendations", [])),
                expected_total_lift=result.get("expected_total_lift"),
                expected_lift_pp=result.get("expected_lift_pp"),
                optimal_allocation_summary=result.get("optimal_allocation_summary"),
                executive_summary=result.get("executive_summary"),
                # Clinical-HTE rebuild: map the fields that were previously dropped at
                # the route (codex LOW-2 — read from the final graph state directly).
                strategic_interpretation=result.get("strategic_interpretation"),
                segment_comparison=_to_native(result.get("segment_comparison")),
                # NOTE: the hierarchical node emits the key 'segment_heterogeneity'
                # (NOT 'segment_heterogeneity_score', which is the TypedDict field).
                segment_heterogeneity=result.get("segment_heterogeneity"),
                n_segments_analyzed=result.get("n_segments_analyzed"),
                # The hierarchical node emits the key 'segmentation_method' (the state
                # field is named '..._used', but the node sets 'segmentation_method') —
                # read the key actually emitted, else this is always None.
                segmentation_method_used=result.get("segmentation_method"),
                overall_hierarchical_ate=result.get("overall_hierarchical_ate"),
                hierarchical_segment_results=_to_native(result.get("hierarchical_segment_results")),
                uplift_by_segment=_to_native(result.get("uplift_by_segment")),
                key_insights=result.get("key_insights", []),
                libraries_used=result.get("libraries_executed"),
                library_agreement_score=result.get("library_agreement_score"),
                validation_passed=result.get("validation_passed"),
                cross_library_validation=_to_native(result.get("cross_library_validation")),
                estimation_latency_ms=result.get("estimation_latency_ms", 0),
                analysis_latency_ms=result.get("analysis_latency_ms", 0),
                total_latency_ms=total_latency,
                warnings=result.get("warnings", []),
                # The route invokes the graph DIRECTLY (graph.ainvoke), bypassing
                # agent._build_output — and no graph node writes "confidence" into the
                # state, so result.get("confidence", 0.0) always yielded 0.0 (every
                # completed run showed Confidence 0% on the page). Compute it here from
                # the SAME SSOT the agent path uses.
                confidence=calculate_confidence(result),
            )

        except ImportError as e:
            # F-010-backend (#429): fail-closed in production unless mock-fallback
            # is explicitly enabled (E2I_REQUIRE_AGENT_IMPORT=0 or ENVIRONMENT!=production).
            from src.api.utils.agent_import_guard import guard_or_raise

            guard_or_raise(e, agent_name="Heterogeneous Optimizer")
            mock = _generate_mock_response(request, start_time)
            # "Using mock data" stays warnings[0] (test_import_error_fail_closed);
            # the question warning still rides along.
            mock.warnings.extend(adjustment.warnings)
            return mock

        except SegmentAnalysisBudgetExceeded:
            # #1840: already an expected, handled outcome — the callers log it
            # with the budget; do not also log it as an execution ERROR.
            raise
        except Exception as e:
            logger.error(f"Segment analysis execution failed: {e}")
            raise


def _to_native(obj: Any) -> Any:
    """Recursively coerce numpy scalars / arrays to native python types.

    The agent's pandas/numpy pipeline emits numpy scalars (e.g. per-segment sizes
    and counts from ``groupby``/``value_counts``). ``numpy.int64`` is NOT a python
    ``int`` subclass, so it leaks un-coerced through the response's
    ``Dict[str, Any]`` / ``List[Dict[str, Any]]`` fields (segment_comparison,
    hierarchical_segment_results, uplift_by_segment) and breaks the durable store's
    ``response.model_dump_json()`` with
    ``Unable to serialize unknown type: <class 'numpy.int64'>`` — failing the whole
    background analysis. Coerce at the route boundary so the response serializes.
    (``numpy.float64`` IS a python ``float`` subclass and serializes fine, but we
    normalise it too for consistency.)
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):  # numpy scalar: int64 / float64 / bool_ / ...
        return obj.item()
    return obj


def _convert_cate_results(
    cate_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[CATEResult]]:
    """Convert agent CATE output to API response format."""
    result: Dict[str, List[CATEResult]] = {}
    for segment_var, cate_list in cate_data.items():
        result[segment_var] = []
        for cate in cate_list:
            try:
                result[segment_var].append(
                    CATEResult(
                        segment_name=cate.get("segment_name", segment_var),
                        segment_value=cate.get("segment_value", ""),
                        cate_estimate=cate.get("cate_estimate", 0.0),
                        cate_ci_lower=cate.get("cate_ci_lower", 0.0),
                        cate_ci_upper=cate.get("cate_ci_upper", 0.0),
                        sample_size=cate.get("sample_size", 0),
                        statistical_significance=cate.get("statistical_significance", False),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to convert CATE result: {e}")
    return result


def _convert_uplift_metrics(result: Dict[str, Any]) -> Optional[UpliftMetrics]:
    """Convert agent uplift output to API response format."""
    # is-None (NOT falsey): a real overall_auuc of 0.0 is a valid (if poor) uplift
    # result and must be surfaced, not dropped. None means the uplift node produced
    # no metrics (skipped / failed) -> honestly omit the card.
    if result.get("overall_auuc") is None:
        return None

    return UpliftMetrics(
        overall_auuc=result.get("overall_auuc", 0.0),
        overall_qini=result.get("overall_qini", 0.0),
        targeting_efficiency=result.get("targeting_efficiency", 0.0),
        model_type_used=result.get("model_type_used", "random_forest"),
    )


def _convert_segment_profiles(
    profiles: List[Dict[str, Any]],
) -> List[SegmentProfile]:
    """Convert agent segment profiles to API response format."""
    result = []
    for profile in profiles:
        try:
            result.append(
                SegmentProfile(
                    segment_id=profile.get("segment_id", ""),
                    responder_type=ResponderType(profile.get("responder_type", "average")),
                    cate_estimate=profile.get("cate_estimate", 0.0),
                    defining_features=profile.get("defining_features", []),
                    size=profile.get("size", 0),
                    size_percentage=profile.get("size_percentage", 0.0),
                    recommendation=profile.get("recommendation", ""),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert segment profile: {e}")
    return result


def _convert_policies(
    policies: List[Dict[str, Any]],
) -> List[PolicyRecommendation]:
    """Convert agent policy recommendations to API response format."""
    result = []
    for policy in policies:
        try:
            result.append(
                PolicyRecommendation(
                    segment=policy.get("segment", ""),
                    current_treatment_rate=policy.get("current_treatment_rate", 0.0),
                    recommended_treatment_rate=policy.get("recommended_treatment_rate", 0.0),
                    expected_incremental_outcome=policy.get("expected_incremental_outcome", 0.0),
                    confidence=policy.get("confidence", 0.0),
                    off_label=policy.get("off_label"),
                    off_label_reason=policy.get("off_label_reason"),
                    label_verdict=policy.get("label_verdict"),
                    label_evidence_confirmed=policy.get("label_evidence_confirmed"),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert policy: {e}")
    return result


def _generate_mock_response(
    request: RunSegmentAnalysisRequest,
    start_time: float,
) -> SegmentAnalysisResponse:
    """Generate mock response when agent is not available."""
    import time

    # segment_vars is optional on the request now (the clinical contract is fixed
    # server-side); fall back to the first fixed clinical segment dimension so the
    # mock-fallback path never crashes on a minimal request.
    primary_segment = (request.segment_vars or _SEGMENT_HTE_SEGMENT_VARS)[0]

    # Mock CATE results
    mock_cate = {
        primary_segment: [
            CATEResult(
                segment_name=primary_segment,
                segment_value="Northeast",
                cate_estimate=15.2,
                cate_ci_lower=8.5,
                cate_ci_upper=21.9,
                sample_size=1250,
                statistical_significance=True,
            ),
            CATEResult(
                segment_name=primary_segment,
                segment_value="Southeast",
                cate_estimate=8.7,
                cate_ci_lower=3.2,
                cate_ci_upper=14.2,
                sample_size=980,
                statistical_significance=True,
            ),
        ]
    }

    # Mock segment profiles
    mock_high_responder = SegmentProfile(
        segment_id=f"{primary_segment}_northeast",
        responder_type=ResponderType.HIGH,
        cate_estimate=15.2,
        defining_features=[
            {"feature": primary_segment, "value": "Northeast"},
            {"feature": "specialty", "value": "Oncology"},
        ],
        size=1250,
        size_percentage=28.5,
        recommendation="Increase treatment intensity for this segment",
    )

    mock_low_responder = SegmentProfile(
        segment_id=f"{primary_segment}_southeast",
        responder_type=ResponderType.LOW,
        # LOW now means HARMFUL (CI entirely below 0) -> CATE must be negative to be
        # consistent with the "Harmful Responder" tile the frontend renders.
        cate_estimate=-3.1,
        defining_features=[
            {"feature": primary_segment, "value": "Southeast"},
        ],
        size=420,
        size_percentage=9.5,
        recommendation="Treatment is net-harmful here; reduce or reallocate resources",
    )

    # Mock policy recommendation
    mock_policy = PolicyRecommendation(
        segment="Northeast_Oncology",
        current_treatment_rate=0.35,
        recommended_treatment_rate=0.55,
        expected_incremental_outcome=125.5,
        confidence=0.82,
    )

    total_latency = int((time.time() - start_time) * 1000)

    return SegmentAnalysisResponse(
        analysis_id="",
        status=SegmentAnalysisStatus.COMPLETED,
        question_type=request.question_type,
        brand=request.brand,
        treatment_var=request.treatment_var or _SEGMENT_HTE_DEFAULT_TREATMENT,
        outcome_var=request.outcome_var or _SEGMENT_HTE_DEFAULT_OUTCOME,
        cate_by_segment=mock_cate,
        overall_ate=10.5,
        confidence_level=1.0 - request.significance_level,
        heterogeneity_score=0.65,
        feature_importance={
            primary_segment: 0.42,
            "specialty": 0.28,
            "practice_size": 0.18,
        },
        uplift_metrics=UpliftMetrics(
            overall_auuc=0.72,
            overall_qini=0.58,
            targeting_efficiency=0.68,
            model_type_used="random_forest",
        ),
        high_responders=[mock_high_responder],
        low_responders=[mock_low_responder],
        policy_recommendations=[mock_policy],
        expected_total_lift=125.5,
        optimal_allocation_summary="Reallocate 20% of resources from low-responder to high-responder segments",
        executive_summary=f"Analysis identified significant treatment effect heterogeneity across {request.segment_vars or _SEGMENT_HTE_SEGMENT_VARS}. Northeast region shows 74% higher response than average.",
        key_insights=[
            "Northeast region shows highest treatment response (CATE: 15.2)",
            "Oncology specialty is key effect modifier",
            "Optimal targeting could increase outcomes by 18%",
        ],
        estimation_latency_ms=200,
        analysis_latency_ms=150,
        total_latency_ms=total_latency,
        warnings=["Using mock data - Heterogeneous Optimizer agent not available"],
        confidence=0.75,
    )
