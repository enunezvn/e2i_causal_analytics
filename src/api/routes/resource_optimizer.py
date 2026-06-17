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

import json
import logging
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.exceptions import RedisError

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
    storage_mode: str = Field(
        default="durable",
        description=(
            "Optimizations-store backing: 'durable' (Redis, shared across workers) "
            "or 'degraded' (process-local in-memory fallback — Redis unavailable, "
            "so cross-worker reads can 404)."
        ),
    )


# =============================================================================
# OPTIMIZATIONS STORAGE (durable / cross-worker)
# =============================================================================
#
# The optimizations store must be DURABLE and SHARED ACROSS WORKERS.
#
# Production runs gunicorn with ``--workers 2`` (docker/docker-compose.yml,
# docker-compose.secure.yml). A process-local dict therefore has two real,
# user-visible failures:
#   - A POST handled by worker A is invisible to a GET handled by worker B, so
#     a legitimate optimization 404s roughly half the time. The page polls
#     ``GET /resources/{id}`` and intermittently 404s (reproduced live: ~50%
#     of polls 404 against the same id while the other worker serves 200).
#   - All state is lost on process restart / redeploy.
#
# This mirrors the sibling fix in ``src/api/routes/segments.py`` (the segment
# analyses store, C21): back the data in Redis, REUSING the app's existing
# async Redis client (``src.api.dependencies.redis_client.get_redis``, already
# wired into the FastAPI lifespan in ``src/api/main.py``). When Redis is
# unavailable it transparently falls back to a BOUNDED in-process dict —
# mirroring the app's existing graceful-degradation posture
# (``app.state.redis_available``) rather than failing the request. The fallback
# is FIFO-bounded so memory cannot grow without limit, and the Redis index is
# bounded the same way.

# Maximum number of optimizations retained (both in Redis and in the fallback).
OPTIMIZATIONS_STORE_MAX_ENTRIES = 1000

# How long an optimization record lives in Redis. Generous relative to the 24h
# window the health endpoint reports on, while still letting abandoned records
# expire so the store cannot accumulate stale entries indefinitely.
OPTIMIZATIONS_STORE_TTL_SECONDS = 7 * 24 * 60 * 60

# Redis key namespace.
_REDIS_KEY_PREFIX = "resources:optimization:"
_REDIS_INDEX_KEY = "resources:optimization:index"

# Redis errors that should trigger graceful degradation rather than a 500.
#
# The app's Redis client is ``redis.asyncio`` (see
# ``src/api/dependencies/redis_client.py``), whose connection/timeout failures
# are ``redis.exceptions.ConnectionError`` / ``redis.exceptions.TimeoutError``
# — NOT the builtin ``ConnectionError`` / ``TimeoutError``. They both subclass
# ``redis.exceptions.RedisError``, so catching ``RedisError`` covers every
# client-raised Redis failure. ``OSError`` covers socket-level failures;
# ``RuntimeError`` covers "redis not initialised" from get_redis().
_REDIS_DEGRADE_ERRORS = (RedisError, OSError, RuntimeError)

# Read-side deserialisation failures that must fail SOFT (skip/None + cleanup)
# rather than 500: a truncated/corrupt payload or a record written by an older
# build that no longer round-trips through the current schema.
_RECORD_DECODE_ERRORS = (ValidationError, json.JSONDecodeError, ValueError)


def _has_non_finite_floats(response: "OptimizationResponse") -> bool:
    """Return ``True`` if any float in ``response`` is NaN or +-inf.

    ``model_dump_json`` serialises non-finite floats to JSON ``null``; on read
    ``model_validate_json`` then raises ``ValidationError`` because the affected
    fields (e.g. ``AllocationResult.optimized_allocation`` / ``ScenarioResult.roi``)
    are non-Optional floats. Such a record is therefore WRITTEN successfully but
    UNREADABLE — it would 404 the real optimization id on every later GET and be
    silently pruned from enumeration. We detect this before persisting (mirrors
    ``src/api/routes/segments.py``) and refuse to store an unreadable record.
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


def _sanitize_non_finite(
    optimization_id: str, response: "OptimizationResponse"
) -> "OptimizationResponse":
    """Return an honest FAILED record for a degenerate (non-finite) result.

    We do NOT fabricate finite numbers (anti-mocking) and we do NOT persist the
    unreadable original. Instead we drop the degenerate numeric payloads and
    mark the optimization FAILED with a clear warning — a state the schema
    already represents (empty ``optimal_allocations`` / ``scenarios`` +
    ``status=failed``). All Optional float fields drop to ``None``;
    ``solve_time_ms`` is a non-Optional int (already finite). The result is
    provably finite and round-trips cleanly.
    """
    logger.warning(
        "Resource optimizer store: optimization %s has non-finite (NaN/inf) "
        "values (degenerate result); persisting as FAILED rather than an "
        "unreadable record.",
        optimization_id,
    )
    sanitized = response.model_copy(deep=True)
    sanitized.status = OptimizationStatus.FAILED
    sanitized.optimal_allocations = []
    sanitized.scenarios = []
    sanitized.sensitivity_analysis = None
    sanitized.impact_by_segment = None
    sanitized.objective_value = None
    sanitized.projected_total_outcome = None
    sanitized.projected_roi = None
    if not any("non-finite" in w.lower() for w in sanitized.warnings):
        sanitized.warnings.append(
            "Optimization produced non-finite (NaN/inf) values; marked as failed."
        )
    # Guard against regression: the whole point is a record that round-trips
    # cleanly. If a future field carries a non-finite float, fail loudly here
    # rather than silently persisting another unreadable record.
    assert not _has_non_finite_floats(sanitized), (
        "sanitize_non_finite left a non-finite float; record would be unreadable"
    )
    return sanitized


class _BoundedOptimizationsStore(Dict[str, OptimizationResponse]):
    """A dict that evicts the oldest entry once it exceeds ``max_entries``.

    Used as the in-process fallback for :class:`_DurableOptimizationsStore`
    when Redis is unavailable. Python dicts preserve insertion order, so
    ``next(iter(self))`` is the oldest key. Re-assigning an existing key updates
    in place and does NOT grow the store.
    """

    def __init__(self, *args: Any, max_entries: int = OPTIMIZATIONS_STORE_MAX_ENTRIES) -> None:
        super().__init__(*args)
        self.max_entries = max_entries

    def __setitem__(self, key: str, value: OptimizationResponse) -> None:
        super().__setitem__(key, value)
        while len(self) > self.max_entries:
            oldest_key = next(iter(self))
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


class _DurableOptimizationsStore:
    """Durable, cross-worker optimizations store backed by Redis.

    Each optimization is stored as a JSON string under
    ``resources:optimization:<id>`` with a TTL, and indexed in a sorted set
    scored by CREATION time (preserved across later status updates) so the full
    collection can be enumerated (for ``/scenarios`` and ``/health``) and TRUE
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
        max_entries: int = OPTIMIZATIONS_STORE_MAX_ENTRIES,
        ttl_seconds: int = OPTIMIZATIONS_STORE_TTL_SECONDS,
    ) -> None:
        self._redis_factory: RedisFactory = redis_factory or _default_redis_factory
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        # In-process fallback used when Redis is unavailable.
        self._memory: _BoundedOptimizationsStore = _BoundedOptimizationsStore(
            max_entries=max_entries
        )

    async def _redis(self) -> Optional[Any]:
        """Return a live Redis client, or ``None`` if unavailable."""
        try:
            return await self._redis_factory()
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(
                f"Resource optimizer store: Redis unavailable, using in-memory fallback: {e}"
            )
            return None
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                f"Resource optimizer store: unexpected Redis factory error, degrading: {e}"
            )
            return None

    async def is_durable(self) -> bool:
        """Return ``True`` if this worker can currently reach Redis AND run a
        store command against it.

        A reachable client whose COMMANDS fail (stale connection, mid-flight
        outage) silently degrades reads/writes to the in-process memory mirror
        — re-creating the cross-worker 404 this store exists to fix. A
        factory-only probe would still report ``durable`` in that case, hiding
        the degradation the ``storage_mode`` field is meant to surface. So we
        exercise a real, cheap Redis command (``zcard`` on the index key — the
        same command ``set``/``values`` rely on) and treat a command failure as
        degraded, exactly as the read/write paths do.
        """
        client = await self._redis()
        if client is None:
            return False
        try:
            await client.zcard(_REDIS_INDEX_KEY)
            return True
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(
                f"Resource optimizer store: Redis durability probe failed, reporting degraded: {e}"
            )
            return False

    @staticmethod
    def _key(optimization_id: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{optimization_id}"

    @staticmethod
    async def _existing_score(client: Any, optimization_id: str) -> Optional[float]:
        """Return the current index score for ``optimization_id`` (or None)."""
        try:
            score = await client.zscore(_REDIS_INDEX_KEY, optimization_id)
            return float(score) if score is not None else None
        except _REDIS_DEGRADE_ERRORS:
            return None

    async def set(self, optimization_id: str, response: OptimizationResponse) -> None:
        """Persist ``response`` under ``optimization_id`` (Redis + memory mirror).

        Write-side guards:

        * never persist a record containing NaN/+-inf floats. Such a record
          serialises to JSON ``null`` and is UNREADABLE on the way back (a
          ``ValidationError`` that would 404 the real id on every later GET and
          silently prune it from enumeration). A non-finite result is a
          DEGENERATE solve, so we store an honest FAILED record (no fabricated
          finite numbers) instead. Mirrors ``src/api/routes/segments.py``.
        * the key SET and index ZADD commit together in a pipeline/transaction
          so a ZADD failure never leaves a record fetchable-by-id yet invisible
          to enumeration. The index score preserves CREATION time (read-existing
          / add-only) so a later status update does not re-score the record into
          "newest", which would make eviction behave like LRU instead of FIFO.
        """
        # Write-side guard: never persist an unreadable (non-finite) record.
        # ``_has_non_finite_floats`` only inspects real floats, so MagicMock /
        # non-model values used in tests pass through untouched.
        if isinstance(response, OptimizationResponse) and _has_non_finite_floats(response):
            response = _sanitize_non_finite(optimization_id, response)

        # Always mirror to the in-process fallback first so an intermittent
        # Redis failure on a later read can still be served in-process.
        self._memory[optimization_id] = response

        client = await self._redis()
        if client is None:
            return
        try:
            payload = response.model_dump_json()
            key = self._key(optimization_id)

            # FIFO: preserve the original creation score across status updates.
            existing_score = await self._existing_score(client, optimization_id)
            score = (
                existing_score
                if existing_score is not None
                else datetime.now(timezone.utc).timestamp()
            )

            try:
                pipe = client.pipeline(transaction=True)
                pipe.set(key, payload, ex=self.ttl_seconds)
                pipe.zadd(_REDIS_INDEX_KEY, {optimization_id: score})
                await pipe.execute()
            except _REDIS_DEGRADE_ERRORS:
                await self._restore_consistency_after_failed_write(client, optimization_id)
                raise

            await self._evict_if_needed(client, keep_id=optimization_id)
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(
                f"Resource optimizer store: Redis write failed for {optimization_id}, degraded: {e}"
            )

    async def _restore_consistency_after_failed_write(
        self, client: Any, optimization_id: str
    ) -> None:
        """Undo a partial write so a key never outlives its index entry."""
        try:
            indexed = await client.zscore(_REDIS_INDEX_KEY, optimization_id)
            if indexed is None:
                await client.delete(self._key(optimization_id))
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(
                f"Resource optimizer store: consistency restore failed for "
                f"{optimization_id}, degraded: {e}"
            )

    async def _prune_orphans(self, client: Any) -> None:
        """Remove index members whose underlying key no longer exists.

        TTL-expired records (and records evicted by Redis ``maxmemory`` before
        TTL) leave their index member behind as an orphan. Those orphans inflate
        the count used by FIFO eviction and can cause a LIVE, in-TTL,
        under-capacity record to be evicted while a dead orphan survives. We
        prune purely by KEY EXISTENCE, the only correct signal once the index
        score is frozen at CREATION time while every ``set()`` resets the TTL.
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
            logger.warning(f"Resource optimizer store: Redis orphan-prune failed, degraded: {e}")

    async def _evict_if_needed(self, client: Any, keep_id: str) -> None:
        """FIFO-evict oldest LIVE entries so the Redis index stays bounded.

        Orphans are pruned FIRST so the count reflects LIVE records only.
        """
        try:
            await self._prune_orphans(client)

            count = await client.zcard(_REDIS_INDEX_KEY)
            overflow = count - self.max_entries
            if overflow <= 0:
                return
            oldest = await client.zrange(_REDIS_INDEX_KEY, 0, overflow - 1)
            for member in oldest:
                if member == keep_id:
                    continue
                await client.delete(self._key(member))
                await client.zrem(_REDIS_INDEX_KEY, member)
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(f"Resource optimizer store: Redis eviction failed, degraded: {e}")

    async def get(self, optimization_id: str) -> Optional[OptimizationResponse]:
        """Return the stored optimization, or ``None`` if absent.

        When Redis is REACHABLE it is AUTHORITATIVE: a clean miss returns
        ``None`` and does NOT fall through to the in-process mirror. Falling
        through on a clean miss would let one worker serve a stale mirrored
        record by id (e.g. after TTL expiry / Redis eviction / a delete by
        another worker) while Redis-backed ``values()`` omits it — re-creating
        the very cross-worker split-brain this store exists to fix. The memory
        mirror is therefore consulted ONLY when Redis is unavailable or a Redis
        command errored.

        Read-side fail-soft: if the persisted payload cannot be decoded /
        validated (e.g. a record written by an older build), return ``None`` and
        lazily remove the poison (key + index member) instead of letting a
        ``ValidationError`` 500 the request.
        """
        client = await self._redis()
        if client is not None:
            try:
                raw = await client.get(self._key(optimization_id))
            except _REDIS_DEGRADE_ERRORS as e:
                # Redis command errored -> degrade to the in-process mirror.
                logger.warning(
                    f"Resource optimizer store: Redis read failed for "
                    f"{optimization_id}, degraded: {e}"
                )
                return self._memory.get(optimization_id)
            if raw is None:
                # Redis reachable + clean miss -> authoritative absence.
                return None
            try:
                return OptimizationResponse.model_validate_json(raw)
            except _RECORD_DECODE_ERRORS as e:
                logger.warning(
                    "Resource optimizer store: unreadable record %s (%s); "
                    "removing poison and returning None.",
                    optimization_id,
                    type(e).__name__,
                )
                await self._remove_poison(client, optimization_id)
                return None
        # Redis unavailable -> serve from the in-process fallback.
        return self._memory.get(optimization_id)

    async def _remove_poison(self, client: Any, optimization_id: str) -> None:
        """Lazily delete an unreadable record's key + index member."""
        try:
            await client.delete(self._key(optimization_id))
            await client.zrem(_REDIS_INDEX_KEY, optimization_id)
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(
                f"Resource optimizer store: failed to remove poison record "
                f"{optimization_id}, degraded: {e}"
            )

    async def contains(self, optimization_id: str) -> bool:
        """Return ``True`` if an optimization exists for ``optimization_id``."""
        return (await self.get(optimization_id)) is not None

    async def values(self) -> List[OptimizationResponse]:
        """Return all stored optimizations (Redis-backed, falling back to memory).

        Batches the per-record reads with a single ``mget`` after the
        ``zrange``. A single unreadable (poison/corrupt) record is SKIPPED and
        its index member pruned; it never breaks enumeration of the rest.
        """
        client = await self._redis()
        if client is not None:
            try:
                ids = await client.zrange(_REDIS_INDEX_KEY, 0, -1)
                if not ids:
                    return []
                keys = [self._key(optimization_id) for optimization_id in ids]
                raws = await client.mget(keys)

                results: List[OptimizationResponse] = []
                stale_ids: List[str] = []
                for optimization_id, raw in zip(ids, raws, strict=False):
                    if raw is None:
                        stale_ids.append(optimization_id)
                        continue
                    try:
                        results.append(OptimizationResponse.model_validate_json(raw))
                    except _RECORD_DECODE_ERRORS as e:
                        logger.warning(
                            "Resource optimizer store: skipping unreadable record "
                            "%s (%s) during enumeration; pruning it.",
                            optimization_id,
                            type(e).__name__,
                        )
                        await client.delete(self._key(optimization_id))
                        stale_ids.append(optimization_id)
                if stale_ids:
                    await client.zrem(_REDIS_INDEX_KEY, *stale_ids)
                return results
            except _REDIS_DEGRADE_ERRORS as e:
                logger.warning(f"Resource optimizer store: Redis enumerate failed, degraded: {e}")
        return list(self._memory.values())

    def clear(self) -> None:
        """Clear the in-process fallback (used by tests).

        Note: this clears only the in-process mirror, not Redis. Tests that
        exercise Redis behaviour use a fresh fake client per test.
        """
        self._memory.clear()


_optimizations_store: _DurableOptimizationsStore = _DurableOptimizationsStore()


# =============================================================================
# ENDPOINTS
# =============================================================================


# -----------------------------------------------------------------------------
# Synthetic-gold allocation targets (showcase substrate)
# -----------------------------------------------------------------------------
#
# The resource optimizer is a real scipy solver, but the platform has NO real
# per-entity budget/allocation data: the territory_metrics columns meant to hold
# it (market_potential, resource_allocation_score) are 100% NULL, and no other
# table carries a current budget/spend per territory. So when a caller runs an
# optimization WITHOUT supplying allocation_targets, we seed a clearly-labelled
# SYNTHETIC problem from the real-shaped (but synthetic) territory_metrics
# activity data: current_allocation is a NOTIONAL budget (HCP coverage x a
# documented per-HCP rate) and expected_response is a real-shaped productivity
# coefficient (TRx per HCP). The OPTIMIZATION MATH IS REAL; the dollar values are
# illustrative. This mirrors the digital-twin "synthetic gold standard to
# showcase capabilities before real data" posture, and every such response is
# tagged with SYNTHETIC_PROVENANCE_PREFIX so the UI can label it honestly.
#
# Pass allocation_targets explicitly to skip synthetic seeding, or wire a real
# budget source into this helper once one exists.

SYNTHETIC_PROVENANCE_PREFIX = "SYNTHETIC DATA:"
NOTIONAL_BUDGET_PER_HCP = 1500.0  # USD/period — documented notional field cost per HCP
SYNTHETIC_TERRITORY_LIMIT = 10  # top-N territories by activity (keeps LP + UI readable)
_SYNTHETIC_MIN_FACTOR = 0.5  # solver may cut a territory to 50% of current
_SYNTHETIC_MAX_FACTOR = 1.5  # ...or grow it to 150%


async def _build_synthetic_allocation_inputs(
    resource_type: ResourceType,
    limit: int = SYNTHETIC_TERRITORY_LIMIT,
) -> tuple[List[AllocationTarget], Optional[Constraint], List[str]]:
    """Seed a SYNTHETIC, clearly-labelled allocation problem from territory_metrics.

    Returns ``(targets, budget_constraint, provenance_warnings)``. On any failure
    or missing data returns ``([], None, [warning])`` so the caller still responds
    honestly (an empty problem -> honest validation failure) rather than crashing.
    """
    try:
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        if client is None:
            return (
                [],
                None,
                [
                    f"{SYNTHETIC_PROVENANCE_PREFIX} territory data store unavailable; "
                    "could not seed synthetic allocation targets."
                ],
            )

        # Latest snapshot first, then highest activity. Over-fetch recent rows so
        # we can dedupe to one (latest) row per territory below.
        resp = (
            await client.table("territory_metrics")
            .select("territory_id, total_trx, active_hcp_count, covered_lives, metric_date")
            .order("metric_date", desc=True)
            .order("total_trx", desc=True)
            .limit(max(limit * 6, 60))
            .execute()
        )
        rows = resp.data or []
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Synthetic target seeding failed: {e}")
        return (
            [],
            None,
            [
                f"{SYNTHETIC_PROVENANCE_PREFIX} could not load territory data ({e}); "
                "no allocation targets seeded."
            ],
        )

    # Dedupe to one (latest) row per territory; take the top-N usable by TRx.
    seen: set = set()
    picked: List[Dict[str, Any]] = []
    for r in rows:
        tid = r.get("territory_id")
        hcp = r.get("active_hcp_count") or 0
        trx = r.get("total_trx") or 0
        if not tid or tid in seen or hcp <= 0 or trx <= 0:
            continue
        seen.add(tid)
        picked.append(r)
        if len(picked) >= limit:
            break

    if not picked:
        return (
            [],
            None,
            [
                f"{SYNTHETIC_PROVENANCE_PREFIX} no usable territory rows found; "
                "no allocation targets seeded."
            ],
        )

    targets: List[AllocationTarget] = []
    for r in picked:
        hcp = float(r["active_hcp_count"])
        trx = float(r["total_trx"])
        current = round(hcp * NOTIONAL_BUDGET_PER_HCP, 2)
        targets.append(
            AllocationTarget(
                entity_id=str(r["territory_id"]),
                entity_type="territory",
                current_allocation=current,
                min_allocation=round(current * _SYNTHETIC_MIN_FACTOR, 2),
                max_allocation=round(current * _SYNTHETIC_MAX_FACTOR, 2),
                # Real-shaped productivity: outcome (TRx) per unit of allocation (HCP).
                expected_response=round(trx / hcp, 4),
            )
        )

    total_budget = round(sum(t.current_allocation for t in targets), 2)
    budget = Constraint(
        constraint_type=ConstraintType.BUDGET,
        value=total_budget,
        scope=ConstraintScope.GLOBAL,
    )
    warning = (
        f"{SYNTHETIC_PROVENANCE_PREFIX} no real per-entity budget source is wired, so this "
        f"optimization ran on {len(targets)} territories seeded from synthetic territory_metrics. "
        f"current_allocation is a NOTIONAL budget (${NOTIONAL_BUDGET_PER_HCP:,.0f}/HCP) and "
        f"expected_response is TRx-per-HCP; total budget ${total_budget:,.0f} "
        f"({resource_type.value}). The optimization math is real but the dollar values are "
        "illustrative."
    )
    return targets, budget, [warning]


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

    # No allocation targets supplied -> seed a clearly-labelled SYNTHETIC problem
    # from territory_metrics (no real budget substrate exists; see
    # _build_synthetic_allocation_inputs). Real targets passed by the caller are
    # respected as-is and never overwritten.
    provenance_warnings: List[str] = []
    if not request.allocation_targets:
        targets, budget, provenance_warnings = await _build_synthetic_allocation_inputs(
            request.resource_type
        )
        if targets:
            request.allocation_targets = targets
            if budget is not None and not any(
                c.constraint_type == ConstraintType.BUDGET for c in request.constraints
            ):
                request.constraints = [*request.constraints, budget]

    # Create initial response
    response = OptimizationResponse(
        optimization_id=optimization_id,
        status=OptimizationStatus.PENDING if async_mode else OptimizationStatus.FORMULATING,
        resource_type=request.resource_type,
        objective=request.objective,
        warnings=list(provenance_warnings),
    )

    if async_mode:
        # Store pending optimization (durable / cross-worker)
        await _optimizations_store.set(optimization_id, response)

        # Schedule background task
        background_tasks.add_task(
            _run_optimization_task,
            optimization_id=optimization_id,
            request=request,
            provenance_warnings=provenance_warnings,
        )

        logger.info(f"Optimization {optimization_id} queued for background execution")
        return response

    # Synchronous execution
    try:
        result = await _execute_optimization(request, provenance_warnings=provenance_warnings)
        result.optimization_id = optimization_id
        await _optimizations_store.set(optimization_id, result)
        return result
    except HTTPException:
        # F-010-backend (#429, codex iter-1 M1): preserve 503 from
        # agent-import guard.
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        response.status = OptimizationStatus.FAILED
        response.warnings.append(str(e))
        await _optimizations_store.set(optimization_id, response)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")


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

    for opt in await _optimizations_store.values():
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
    all_optimizations = await _optimizations_store.values()
    optimizations_24h = sum(
        1 for o in all_optimizations if (now - o.timestamp).total_seconds() < 86400
    )

    # Get last optimization
    last_optimization = None
    if all_optimizations:
        last_optimization = max(o.timestamp for o in all_optimizations)

    # Surface whether this worker is serving DURABLE (Redis, cross-worker) or
    # DEGRADED (process-local in-memory) state. A silent per-worker fallback
    # re-introduces the cross-worker 404 this store exists to fix, so it must
    # be observable rather than invisible.
    durable = await _optimizations_store.is_durable()
    storage_mode = "durable" if durable else "degraded"

    status = "healthy"
    if not agent_available:
        status = "degraded"
    elif not durable:
        # Storage degraded to the in-memory fallback -> cross-worker reads can
        # 404. This is a real, user-visible degradation, so report it.
        status = "degraded"
    elif not scipy_available:
        status = "partial"

    return ResourceHealthResponse(
        status=status,
        agent_available=agent_available,
        scipy_available=scipy_available,
        last_optimization=last_optimization,
        optimizations_24h=optimizations_24h,
        storage_mode=storage_mode,
    )


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
    optimization = await _optimizations_store.get(optimization_id)
    if optimization is None:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization {optimization_id} not found",
        )

    return optimization


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _run_optimization_task(
    optimization_id: str,
    request: RunOptimizationRequest,
    provenance_warnings: Optional[List[str]] = None,
) -> None:
    """Background task to run optimization."""
    try:
        logger.info(f"Starting optimization task {optimization_id}")

        # Update status (read-modify-write so the change persists to the store).
        pending = await _optimizations_store.get(optimization_id)
        if pending is not None:
            pending.status = OptimizationStatus.FORMULATING
            await _optimizations_store.set(optimization_id, pending)

        # Execute optimization
        result = await _execute_optimization(request, provenance_warnings=provenance_warnings)
        result.optimization_id = optimization_id

        # Store result
        await _optimizations_store.set(optimization_id, result)

        logger.info(f"Optimization {optimization_id} completed successfully")

    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {e}")
        existing = await _optimizations_store.get(optimization_id)
        if existing is not None:
            existing.status = OptimizationStatus.FAILED
            existing.warnings.append(str(e))
            await _optimizations_store.set(optimization_id, existing)


async def _execute_optimization(
    request: RunOptimizationRequest,
    provenance_warnings: Optional[List[str]] = None,
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
            build_resource_optimizer_graph as create_resource_optimizer_graph,
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
            "allocation_targets": allocation_targets,  # type: ignore[typeddict-item]
            "constraints": constraints,  # type: ignore[typeddict-item]
            "objective": request.objective.value,
            "solver_type": request.solver_type.value,
            "time_limit_seconds": request.time_limit_seconds,
            "gap_tolerance": request.gap_tolerance,
            "run_scenarios": request.run_scenarios,
            "scenario_count": request.scenario_count,
            "status": "pending",
            "errors": [],
            # Seed provenance (e.g. SYNTHETIC-data notice) so it flows through the
            # agent nodes (which append to, never overwrite, warnings) into the
            # final response and is surfaced honestly by the UI.
            "warnings": list(provenance_warnings or []),
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
        # F-010-backend (#429): fail-closed in production unless mock-fallback
        # is explicitly enabled (E2I_REQUIRE_AGENT_IMPORT=0 or ENVIRONMENT!=production).
        from src.api.utils.agent_import_guard import guard_or_raise

        guard_or_raise(e, agent_name="Resource Optimizer")
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
