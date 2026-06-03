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

import json
import logging
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.exceptions import RedisError

from src.api.dependencies.auth import require_analyst
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

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


class AnalysisStatus(str, Enum):
    """Status of segment analysis."""

    PENDING = "pending"
    ESTIMATING = "estimating"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionType(str, Enum):
    """Type of analysis question for library routing."""

    EFFECT_HETEROGENEITY = "effect_heterogeneity"  # EconML primary
    TARGETING = "targeting"  # CausalML primary
    SEGMENT_OPTIMIZATION = "segment_optimization"  # Both libraries
    COMPREHENSIVE = "comprehensive"  # All libraries with DoWhy validation


# =============================================================================
# REQUEST MODELS
# =============================================================================


class RunSegmentAnalysisRequest(BaseModel):
    """Request to run segment analysis."""

    query: str = Field(..., description="Natural language query describing the analysis")
    treatment_var: str = Field(
        ..., description="Treatment variable name (e.g., 'rep_visits', 'email_campaigns')"
    )
    outcome_var: str = Field(..., description="Outcome variable name (e.g., 'trx', 'conversion')")
    segment_vars: List[str] = Field(
        ..., description="Variables to segment by (e.g., ['region', 'specialty'])"
    )
    effect_modifiers: Optional[List[str]] = Field(
        default=None, description="Variables that modify treatment effect"
    )
    data_source: str = Field(default="hcp_data", description="Data source identifier")
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
    question_type: Optional[QuestionType] = Field(
        default=None, description="Analysis question type for library routing"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Which HCP segments respond best to rep visits?",
                "treatment_var": "rep_visits",
                "outcome_var": "trx",
                "segment_vars": ["region", "specialty"],
                "effect_modifiers": ["practice_size", "years_experience"],
                "data_source": "hcp_data",
                "n_estimators": 100,
                "top_segments_count": 10,
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
    cate_ci_lower: float = Field(..., description="95% CI lower bound")
    cate_ci_upper: float = Field(..., description="95% CI upper bound")
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


class UpliftMetrics(BaseModel):
    """Uplift modeling metrics."""

    overall_auuc: float = Field(..., description="Area Under Uplift Curve (0-1)")
    overall_qini: float = Field(..., description="Qini coefficient")
    targeting_efficiency: float = Field(..., description="How well model targets responders (0-1)")
    model_type_used: str = Field(..., description="Model type (random_forest, gradient_boosting)")


class SegmentAnalysisResponse(BaseModel):
    """Response from segment analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    question_type: Optional[QuestionType] = Field(
        default=None, description="Question type used for routing"
    )

    # CATE results
    cate_by_segment: Dict[str, List[CATEResult]] = Field(
        default_factory=dict, description="CATE results grouped by segment variable"
    )
    overall_ate: Optional[float] = Field(
        default=None, description="Overall Average Treatment Effect"
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
    low_responders: List[SegmentProfile] = Field(
        default_factory=list, description="Low responder segments"
    )

    # Policy recommendations
    policy_recommendations: List[PolicyRecommendation] = Field(
        default_factory=list, description="Targeting recommendations"
    )
    expected_total_lift: Optional[float] = Field(
        default=None, description="Expected lift from optimal allocation"
    )
    optimal_allocation_summary: Optional[str] = Field(
        default=None, description="Summary of optimal allocation"
    )

    # Summary
    executive_summary: Optional[str] = Field(default=None, description="Executive-level summary")
    key_insights: List[str] = Field(default_factory=list, description="Key findings")

    # Multi-library support
    libraries_used: Optional[List[str]] = Field(default=None, description="Causal libraries used")
    library_agreement_score: Optional[float] = Field(
        default=None, description="Agreement between libraries (0-1)"
    )
    validation_passed: Optional[bool] = Field(
        default=None, description="Whether cross-validation passed"
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
        sanitized.status = AnalysisStatus.FAILED
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
        sanitized.expected_total_lift = None
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

    def clear(self) -> None:
        """Clear the in-process fallback (used by tests).

        Note: this clears only the in-process mirror, not Redis. Tests that
        exercise Redis behaviour use a fresh fake client per test.
        """
        self._memory.clear()


_analyses_store: _DurableAnalysesStore = _DurableAnalysesStore()


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

    # Create initial response
    response = SegmentAnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING if async_mode else AnalysisStatus.ESTIMATING,
        question_type=request.question_type,
    )

    if async_mode:
        # Store pending analysis
        await _analyses_store.set(analysis_id, response)

        # Schedule background task
        background_tasks.add_task(
            _run_segment_analysis_task,
            analysis_id=analysis_id,
            request=request,
        )

        logger.info(f"Segment analysis {analysis_id} queued for background execution")
        return response

    # Synchronous execution
    try:
        result = await _execute_segment_analysis(request)
        result.analysis_id = analysis_id
        await _analyses_store.set(analysis_id, result)
        return result
    except HTTPException:
        # F-010-backend (#429, codex iter-1 M1): preserve 503 from
        # agent-import guard.
        raise
    except Exception as e:
        logger.error(f"Segment analysis failed: {e}", exc_info=True)
        response.status = AnalysisStatus.FAILED
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
        if analysis.status != AnalysisStatus.COMPLETED:
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


async def _run_segment_analysis_task(
    analysis_id: str,
    request: RunSegmentAnalysisRequest,
) -> None:
    """Background task to run segment analysis."""
    try:
        logger.info(f"Starting segment analysis task {analysis_id}")

        # Update status (read-modify-write so the change persists to the store).
        pending = await _analyses_store.get(analysis_id)
        if pending is not None:
            pending.status = AnalysisStatus.ESTIMATING
            await _analyses_store.set(analysis_id, pending)

        # Execute analysis
        result = await _execute_segment_analysis(request)
        result.analysis_id = analysis_id

        # Store result
        await _analyses_store.set(analysis_id, result)

        logger.info(f"Segment analysis {analysis_id} completed successfully")

    except Exception as e:
        logger.error(f"Segment analysis {analysis_id} failed: {e}")
        existing = await _analyses_store.get(analysis_id)
        if existing is not None:
            existing.status = AnalysisStatus.FAILED
            # Store a generic warning rather than raw exception text (the record
            # is later returned to clients via GET).
            existing.warnings.append("Segment analysis failed due to an internal error.")
            await _analyses_store.set(analysis_id, existing)


async def _execute_segment_analysis(
    request: RunSegmentAnalysisRequest,
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

    try:
        # Try to use the actual Heterogeneous Optimizer agent
        from src.agents.heterogeneous_optimizer.graph import (
            create_heterogeneous_optimizer_graph,
        )
        from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState

        # Initialize state (cast partial state - remaining fields populated by graph nodes)
        initial_state = cast(
            HeterogeneousOptimizerState,
            {
                "query": request.query,
                "treatment_var": request.treatment_var,
                "outcome_var": request.outcome_var,
                "segment_vars": request.segment_vars,
                "effect_modifiers": request.effect_modifiers or [],
                "data_source": request.data_source,
                "filters": request.filters,
                "n_estimators": request.n_estimators,
                "min_samples_leaf": request.min_samples_leaf,
                "significance_level": request.significance_level,
                "top_segments_count": request.top_segments_count,
                "status": "pending",
                "errors": [],
                "warnings": [],
                "estimation_latency_ms": 0,
                "analysis_latency_ms": 0,
                "total_latency_ms": 0,
            },
        )

        # Create and run graph
        graph = create_heterogeneous_optimizer_graph()
        result = await graph.ainvoke(initial_state)

        # Convert agent output to API response
        total_latency = int((time.time() - start_time) * 1000)

        return SegmentAnalysisResponse(
            analysis_id="",  # Will be set by caller
            status=AnalysisStatus.COMPLETED
            if result.get("status") == "completed"
            else AnalysisStatus.FAILED,
            question_type=request.question_type,
            cate_by_segment=_convert_cate_results(result.get("cate_by_segment", {})),
            overall_ate=result.get("overall_ate"),
            heterogeneity_score=result.get("heterogeneity_score"),
            feature_importance=result.get("feature_importance"),
            uplift_metrics=_convert_uplift_metrics(result),
            high_responders=_convert_segment_profiles(result.get("high_responders", [])),
            low_responders=_convert_segment_profiles(result.get("low_responders", [])),
            policy_recommendations=_convert_policies(result.get("policy_recommendations", [])),
            expected_total_lift=result.get("expected_total_lift"),
            optimal_allocation_summary=result.get("optimal_allocation_summary"),
            executive_summary=result.get("executive_summary"),
            key_insights=result.get("key_insights", []),
            libraries_used=result.get("libraries_executed"),
            library_agreement_score=result.get("library_agreement_score"),
            validation_passed=result.get("validation_passed"),
            estimation_latency_ms=result.get("estimation_latency_ms", 0),
            analysis_latency_ms=result.get("analysis_latency_ms", 0),
            total_latency_ms=total_latency,
            warnings=result.get("warnings", []),
            confidence=result.get("confidence", 0.0),
        )

    except ImportError as e:
        # F-010-backend (#429): fail-closed in production unless mock-fallback
        # is explicitly enabled (E2I_REQUIRE_AGENT_IMPORT=0 or ENVIRONMENT!=production).
        from src.api.utils.agent_import_guard import guard_or_raise

        guard_or_raise(e, agent_name="Heterogeneous Optimizer")
        return _generate_mock_response(request, start_time)

    except Exception as e:
        logger.error(f"Segment analysis execution failed: {e}")
        raise


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
    if not result.get("overall_auuc"):
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

    # Mock CATE results
    mock_cate = {
        request.segment_vars[0]: [
            CATEResult(
                segment_name=request.segment_vars[0],
                segment_value="Northeast",
                cate_estimate=15.2,
                cate_ci_lower=8.5,
                cate_ci_upper=21.9,
                sample_size=1250,
                statistical_significance=True,
            ),
            CATEResult(
                segment_name=request.segment_vars[0],
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
        segment_id=f"{request.segment_vars[0]}_northeast",
        responder_type=ResponderType.HIGH,
        cate_estimate=15.2,
        defining_features=[
            {"feature": request.segment_vars[0], "value": "Northeast"},
            {"feature": "specialty", "value": "Oncology"},
        ],
        size=1250,
        size_percentage=28.5,
        recommendation="Increase treatment intensity for this segment",
    )

    mock_low_responder = SegmentProfile(
        segment_id=f"{request.segment_vars[0]}_southeast",
        responder_type=ResponderType.LOW,
        cate_estimate=3.1,
        defining_features=[
            {"feature": request.segment_vars[0], "value": "Southeast"},
        ],
        size=420,
        size_percentage=9.5,
        recommendation="Consider reducing or reallocating resources",
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
        status=AnalysisStatus.COMPLETED,
        question_type=request.question_type,
        cate_by_segment=mock_cate,
        overall_ate=10.5,
        heterogeneity_score=0.65,
        feature_importance={
            request.segment_vars[0]: 0.42,
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
        executive_summary=f"Analysis identified significant treatment effect heterogeneity across {request.segment_vars}. Northeast region shows 74% higher response than average.",
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
