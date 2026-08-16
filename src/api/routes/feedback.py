"""
E2I Feedback Learning API
=========================

FastAPI endpoints for feedback processing, pattern detection, and knowledge updates.

Phase: Agent Output Routing + Phase 4 (G23: Opik Feedback Loop)

Endpoints:
- POST /feedback/learn: Run feedback learning cycle
- GET  /feedback/{batch_id}: Get learning results
- POST /feedback/process: Process specific feedback items
- GET  /feedback/patterns: List detected patterns
- GET  /feedback/updates: List knowledge updates
- GET  /feedback/health: Service health check
- POST /feedback/trace: Record feedback for an Opik trace (G23)
- GET  /feedback/agent/{agent_name}/stats: Get agent feedback stats (G23)
- GET  /feedback/agent/{agent_name}/signals: Get GEPA optimization signals (G23)

Integration Points:
- Feedback Learner Agent (Tier 5)
- Orchestrator for agent invocation
- Supabase for persistence
- Opik for trace feedback (Phase 4 - G23)
- GEPA for prompt optimization (Phase 4 - G23)

Author: E2I Causal Analytics Team
Version: 4.3.0 (Opik Feedback Loop Integration)
"""

import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import require_operator
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

if TYPE_CHECKING:
    from src.api.repositories.feedback_repository import FeedbackRepository

# Opik Feedback Loop imports (Phase 4 - G23)
try:
    from src.mlops.opik_feedback import (
        AgentFeedbackStats,  # noqa: F401
        get_feedback_collector,
        get_feedback_signals_for_gepa,
        log_user_feedback,
    )
    from src.mlops.opik_feedback import (
        FeedbackRecord as OpikFeedbackRecord,  # noqa: F401
    )

    OPIK_FEEDBACK_AVAILABLE = True
except ImportError:
    OPIK_FEEDBACK_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback Learning"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS
# =============================================================================


class FeedbackType(str, Enum):
    """Types of user feedback."""

    RATING = "rating"
    CORRECTION = "correction"
    OUTCOME = "outcome"
    EXPLICIT = "explicit"


class PatternType(str, Enum):
    """Types of patterns that can be detected."""

    ACCURACY_ISSUE = "accuracy_issue"
    LATENCY_ISSUE = "latency_issue"
    RELEVANCE_ISSUE = "relevance_issue"
    FORMAT_ISSUE = "format_issue"
    COVERAGE_GAP = "coverage_gap"


class PatternSeverity(str, Enum):
    """Severity levels for detected patterns."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UpdateType(str, Enum):
    """Types of knowledge updates."""

    PROMPT_REFINEMENT = "prompt_refinement"
    EXAMPLE_ADDITION = "example_addition"
    RULE_MODIFICATION = "rule_modification"
    PARAMETER_TUNING = "parameter_tuning"
    INDEX_UPDATE = "index_update"


class UpdateStatus(str, Enum):
    """Status of knowledge updates."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class LearningStatus(str, Enum):
    """Status of a learning cycle."""

    PENDING = "pending"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    EXTRACTING = "extracting"
    UPDATING = "updating"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class FeedbackItem(BaseModel):
    """Individual feedback item to process."""

    feedback_id: Optional[str] = Field(
        default=None, description="Unique feedback identifier (auto-generated if not provided)"
    )
    timestamp: Optional[str] = Field(default=None, description="Feedback timestamp (ISO format)")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    source_agent: str = Field(..., description="Agent that generated the original response")
    query: str = Field(..., description="Original user query")
    agent_response: str = Field(..., description="Agent's response to the query")
    user_feedback: Any = Field(
        ..., description="User's feedback (rating, correction, outcome, etc.)"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feedback_type": "rating",
                "source_agent": "causal_impact",
                "query": "What drives TRx for Kisqali?",
                "agent_response": "Based on causal analysis, rep visits have the strongest effect...",
                "user_feedback": {"rating": 4, "helpful": True},
            }
        }
    )


class RunLearningRequest(BaseModel):
    """Request to run a feedback learning cycle."""

    time_range_start: Optional[str] = Field(
        default=None, description="Start of time range (ISO format, defaults to last 24h)"
    )
    time_range_end: Optional[str] = Field(
        default=None, description="End of time range (ISO format, defaults to now)"
    )
    focus_agents: Optional[List[str]] = Field(
        default=None, description="Specific agents to focus on (all if not specified)"
    )
    min_feedback_count: int = Field(
        default=10, description="Minimum feedback items required to proceed", ge=1
    )
    pattern_threshold: float = Field(
        default=0.1, description="Minimum frequency for pattern detection (0-1)", ge=0.0, le=1.0
    )
    auto_apply: bool = Field(default=False, description="Automatically apply approved updates")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time_range_start": "2024-01-01T00:00:00Z",
                "time_range_end": "2024-01-07T23:59:59Z",
                "focus_agents": ["causal_impact", "gap_analyzer"],
                "min_feedback_count": 20,
                "pattern_threshold": 0.15,
                "auto_apply": False,
            }
        }
    )


class ProcessFeedbackRequest(BaseModel):
    """Request to process specific feedback items."""

    items: List[FeedbackItem] = Field(..., description="Feedback items to process")
    detect_patterns: bool = Field(default=True, description="Whether to detect patterns")
    generate_recommendations: bool = Field(
        default=True, description="Whether to generate recommendations"
    )


class ApplyUpdateRequest(BaseModel):
    """Request to apply a knowledge update."""

    update_id: str = Field(..., description="Update identifier to apply")
    force: bool = Field(default=False, description="Force apply even if not approved")


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class DetectedPattern(BaseModel):
    """Pattern detected from feedback analysis."""

    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_type: PatternType = Field(..., description="Type of pattern")
    description: str = Field(..., description="Human-readable description")
    frequency: int = Field(..., description="Number of occurrences")
    severity: PatternSeverity = Field(..., description="Impact severity")
    affected_agents: List[str] = Field(..., description="Agents affected by this pattern")
    example_feedback_ids: List[str] = Field(..., description="Example feedback IDs")
    root_cause_hypothesis: str = Field(..., description="Hypothesized root cause")
    confidence: float = Field(..., description="Detection confidence (0-1)", ge=0.0, le=1.0)
    # #1244: when the pattern was detected. Backfilled from the persistence
    # row's created_at for payloads written before this field existed; the
    # frontend Recent Activity timestamp has no other source (agent output
    # carries no timestamp).
    detected_at: Optional[datetime] = Field(
        default=None, description="When the pattern was detected"
    )


class LearningRecommendation(BaseModel):
    """Recommendation for system improvement."""

    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    pattern_id: str = Field(..., description="Pattern this addresses")
    priority: int = Field(..., description="Priority rank (1=highest)", ge=1)
    recommendation_type: str = Field(..., description="Type of recommendation")
    description: str = Field(..., description="What should be changed")
    expected_impact: str = Field(..., description="Expected improvement")
    implementation_effort: str = Field(..., description="Low/Medium/High")
    affected_agents: List[str] = Field(..., description="Agents to modify")


class KnowledgeUpdate(BaseModel):
    """Proposed or applied knowledge update."""

    update_id: str = Field(..., description="Unique update identifier")
    update_type: UpdateType = Field(..., description="Type of update")
    status: UpdateStatus = Field(..., description="Current status")
    target_agent: str = Field(..., description="Agent to update")
    target_component: str = Field(..., description="Component being updated")
    current_value: Optional[str] = Field(default=None, description="Current configuration")
    proposed_value: str = Field(..., description="Proposed new configuration")
    rationale: str = Field(..., description="Why this update is needed")
    expected_improvement: str = Field(..., description="Expected impact")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When proposed"
    )
    applied_at: Optional[datetime] = Field(default=None, description="When applied")


class FeedbackSummary(BaseModel):
    """Summary statistics from feedback analysis."""

    total_feedback_items: int = Field(..., description="Total items processed")
    by_type: Dict[str, int] = Field(..., description="Count by feedback type")
    by_agent: Dict[str, int] = Field(..., description="Count by source agent")
    average_rating: Optional[float] = Field(
        default=None, description="Average rating (if applicable)"
    )
    positive_ratio: float = Field(..., description="Ratio of positive feedback")
    time_range_start: str = Field(..., description="Analysis start time")
    time_range_end: str = Field(..., description="Analysis end time")


class LearningResponse(BaseModel):
    """Response from feedback learning cycle."""

    batch_id: str = Field(..., description="Unique batch identifier")
    status: LearningStatus = Field(..., description="Learning status")

    # Results
    detected_patterns: List[DetectedPattern] = Field(
        default_factory=list, description="Patterns detected from feedback"
    )
    learning_recommendations: List[LearningRecommendation] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    priority_improvements: List[str] = Field(default_factory=list, description="Top priority items")
    proposed_updates: List[KnowledgeUpdate] = Field(
        default_factory=list, description="Proposed knowledge updates"
    )
    applied_updates: List[KnowledgeUpdate] = Field(
        default_factory=list, description="Updates that were applied"
    )

    # Summary
    learning_summary: str = Field(default="", description="Executive summary")
    feedback_summary: Optional[FeedbackSummary] = Field(
        default=None, description="Feedback statistics"
    )

    # Metrics
    patterns_detected: int = Field(default=0, description="Number of patterns found")
    recommendations_generated: int = Field(default=0, description="Number of recommendations")
    updates_proposed: int = Field(default=0, description="Number of updates proposed")
    updates_applied: int = Field(default=0, description="Number of updates applied")

    # Metadata
    collection_latency_ms: int = Field(default=0, description="Feedback collection time")
    analysis_latency_ms: int = Field(default=0, description="Analysis time")
    total_latency_ms: int = Field(default=0, description="Total processing time")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Completion timestamp"
    )
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "batch_id": "fb_abc123",
                "status": "completed",
                "patterns_detected": 3,
                "recommendations_generated": 5,
                "updates_proposed": 2,
                "updates_applied": 0,
                "learning_summary": "Identified 3 patterns affecting response quality...",
            }
        }
    )


class PatternListResponse(BaseModel):
    """Response for listing patterns."""

    total_count: int = Field(..., description="Total patterns")
    critical_count: int = Field(..., description="Critical severity count")
    high_count: int = Field(..., description="High severity count")
    patterns: List[DetectedPattern] = Field(..., description="List of patterns")


class UpdateListResponse(BaseModel):
    """Response for listing knowledge updates."""

    total_count: int = Field(..., description="Total updates")
    proposed_count: int = Field(..., description="Pending approval")
    applied_count: int = Field(..., description="Already applied")
    updates: List[KnowledgeUpdate] = Field(..., description="List of updates")


class OptimizerGateStatus(BaseModel):
    """State of the daily DSPy prompt-optimization trigger (#1661).

    The beat that drives prompt optimization returns ``{"status": "skipped"}``
    whenever its trigger is unsatisfied — a legitimate return, so nothing fails
    and nothing alerts while the self-improvement loop stays inert. These are
    the trigger's own inputs, so an operator watching this page can see whether
    the loop is actually doing anything.

    Counts are ``None`` (not 0) when the read fails: a fabricated zero on a
    health surface is indistinguishable from a measured one.
    """

    eligible_signals: Optional[int] = Field(
        default=None, description="feedback_learner signals clearing the reward floor"
    )
    total_signals: Optional[int] = Field(
        default=None,
        description="All feedback_learner signals ever — the yield denominator",
    )
    last_eligible_signal_at: Optional[str] = Field(
        default=None, description="When an eligible signal was last recorded"
    )
    optimization_runs: Optional[int] = Field(
        default=None, description="prompt_optimization_runs rows; 0 means never optimized"
    )
    min_signals: int = Field(..., description="Eligible signals the trigger requires")
    min_reward: float = Field(..., description="Reward floor a signal must clear")
    would_trigger: Optional[bool] = Field(
        default=None, description="Whether the count gate is satisfied right now"
    )
    reason: str = Field(..., description="Human-readable gate verdict")


class FeedbackHealthResponse(BaseModel):
    """Health check response for feedback learning service."""

    status: str = Field(..., description="Service status")
    agent_available: bool = Field(..., description="Feedback Learner agent status")
    last_learning_cycle: Optional[datetime] = Field(
        default=None, description="Last learning cycle timestamp"
    )
    cycles_24h: int = Field(default=0, description="Learning cycles in last 24 hours")
    patterns_active: int = Field(default=0, description="Active patterns being tracked")
    pending_updates: int = Field(default=0, description="Updates pending approval")
    optimizer: Optional[OptimizerGateStatus] = Field(
        default=None,
        description="Daily prompt-optimization trigger state (#1661)",
    )


# =============================================================================
# PERSISTENCE (Supabase) with an in-memory dev fallback (M2)
# =============================================================================
# Process-local fallback — used ONLY when Supabase is unconfigured or the dev
# flag E2I_GAPS_FEEDBACK_INMEMORY=1 is set. In prod (gunicorn --workers 2) this
# is bypassed: the repo is the single source of truth shared across workers.

_learning_store: Dict[str, LearningResponse] = {}
_patterns_store: Dict[str, DetectedPattern] = {}
_updates_store: Dict[str, KnowledgeUpdate] = {}
_feedback_store: List[FeedbackItem] = []


def _get_repo() -> "Optional[FeedbackRepository]":
    """Return a FeedbackRepository, or None if Supabase is unavailable."""
    try:
        from src.api.repositories.feedback_repository import FeedbackRepository

        return FeedbackRepository()
    except Exception as exc:  # ServiceConnectionError when unconfigured
        logger.warning("Feedback persistence unavailable, using in-memory fallback: %s", exc)
        return None


def _use_inmemory_fallback() -> bool:
    """True when we should fall back to the process-local dicts (dev/offline)."""
    if os.environ.get("E2I_GAPS_FEEDBACK_INMEMORY") == "1":
        return True
    try:
        return _get_repo() is None
    except Exception:
        return True


# ---- write helpers (persist to Supabase or the dict fallback) --------------
async def _persist_batch(response: LearningResponse) -> None:
    if _use_inmemory_fallback():
        _learning_store[response.batch_id] = response
        return
    repo = _get_repo()
    if repo is None:
        _learning_store[response.batch_id] = response
        return
    await repo.upsert_batch(response)


async def _persist_pattern(pattern: DetectedPattern) -> None:
    if _use_inmemory_fallback():
        _patterns_store[pattern.pattern_id] = pattern
        return
    repo = _get_repo()
    if repo is None:
        _patterns_store[pattern.pattern_id] = pattern
        return
    await repo.upsert_pattern(pattern)


async def _persist_update(update: KnowledgeUpdate) -> None:
    if _use_inmemory_fallback():
        _updates_store[update.update_id] = update
        return
    repo = _get_repo()
    if repo is None:
        _updates_store[update.update_id] = update
        return
    await repo.upsert_update(update)


async def _persist_cycle_artifacts(result: LearningResponse) -> None:
    """Persist the patterns and updates a completed learning cycle produced.

    Both the SYNC (``async_mode=False``) and ASYNC (``_run_learning_task``)
    paths run the SAME ``_execute_learning_cycle`` and must persist its
    artifacts identically — otherwise the GET /feedback/{patterns,updates}
    endpoints (and the FeedbackLearning page's Patterns/Updates tabs) see the
    detected patterns / proposed updates only for whichever path happened to
    persist them. Historically only the async task did, so the sync path the
    UI drives (``useQuickLearningCycle`` -> ``async_mode=false``) computed and
    threw the artifacts away. Centralizing the two loops here keeps the paths
    from drifting apart again.
    """
    for pattern in result.detected_patterns:
        await _persist_pattern(pattern)
    for update in result.proposed_updates:
        await _persist_update(update)


async def persist_learning_cycle_output(output: Any, batch_id: str) -> "LearningResponse":
    """Convert a ``FeedbackLearnerAgent`` output into a ``LearningResponse``
    and persist the batch + its artifacts (the tables the /feedback-learning
    page reads).

    Shared entry point for out-of-band cycle triggers — concretely the 6h
    Celery beat (``src.tasks.run_feedback_learning_cycle``), which previously
    persisted ONLY dspy training signals: the page tables stayed empty despite
    a live learning loop running 4×/day. Mirrors the ``applied_updates``
    re-hydration in ``_execute_learning_cycle`` (state carries applied update
    IDs as strings; the records live in ``proposed_updates``).
    """
    proposed = [dict(u) for u in (output.proposed_updates or [])]
    applied_ids = set(output.applied_updates or [])
    applied_records = [u for u in proposed if u.get("update_id") in applied_ids]
    response = LearningResponse(
        batch_id=batch_id,
        status=(
            LearningStatus.COMPLETED if output.status == "completed" else LearningStatus.FAILED
        ),
        detected_patterns=_convert_patterns([dict(p) for p in output.detected_patterns or []]),
        learning_recommendations=_convert_recommendations(
            [dict(r) for r in output.learning_recommendations or []]
        ),
        priority_improvements=list(output.priority_improvements or []),
        proposed_updates=_convert_updates(proposed),
        applied_updates=_convert_updates(applied_records, applied=True),
        learning_summary=output.learning_summary or "",
        patterns_detected=len(output.detected_patterns or []),
        recommendations_generated=len(output.learning_recommendations or []),
        updates_proposed=len(proposed),
        updates_applied=len(applied_ids),
        total_latency_ms=int(output.total_latency_ms or 0),
        errors=[str(e) for e in (output.errors or [])],
        warnings=list(output.warnings or []),
    )
    await _persist_cycle_artifacts(response)
    await _persist_batch(response)
    return response


async def _persist_item(item: FeedbackItem) -> None:
    if _use_inmemory_fallback():
        _feedback_store.append(item)
        return
    repo = _get_repo()
    if repo is None:
        _feedback_store.append(item)
        return
    await repo.append_item(item)


# ---- read helpers (read from Supabase or the dict fallback) ----------------
async def _load_batch(batch_id: str) -> Optional[LearningResponse]:
    if _use_inmemory_fallback():
        return _learning_store.get(batch_id)
    repo = _get_repo()
    if repo is None:
        return _learning_store.get(batch_id)
    return await repo.get_batch(batch_id)


async def _load_update(update_id: str) -> Optional[KnowledgeUpdate]:
    if _use_inmemory_fallback():
        return _updates_store.get(update_id)
    repo = _get_repo()
    if repo is None:
        return _updates_store.get(update_id)
    return await repo.get_update(update_id)


async def _all_batches() -> List[LearningResponse]:
    if _use_inmemory_fallback():
        return list(_learning_store.values())
    repo = _get_repo()
    if repo is None:
        return list(_learning_store.values())
    return await repo.count_recent_and_last()


async def _all_patterns() -> List[DetectedPattern]:
    if _use_inmemory_fallback():
        return list(_patterns_store.values())
    repo = _get_repo()
    if repo is None:
        return list(_patterns_store.values())
    return await repo.list_patterns()


async def _all_updates() -> List[KnowledgeUpdate]:
    if _use_inmemory_fallback():
        return list(_updates_store.values())
    repo = _get_repo()
    if repo is None:
        return list(_updates_store.values())
    return await repo.list_updates()


async def _get_knowledge_stores() -> Dict[str, Any]:
    """Build the real Tier-5 knowledge stores for endpoint-driven apply/rollback.

    #1243: same backend the learning cycle's auto-apply path uses
    (``SupabaseKnowledgeStore``, #837 — fail-closed, read-back confirmed).
    Returns ``{}`` when the async client can't be built; callers must treat
    that as "stores unavailable" (503), never as license to status-flip.
    """
    try:
        from src.agents.feedback_learner.knowledge_stores import build_knowledge_stores
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        return build_knowledge_stores(client)
    except Exception as e:  # noqa: BLE001 - unavailable backend => honest 503 upstream
        logger.warning(f"knowledge stores unavailable: {e}")
        return {}


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/learn",
    response_model=LearningResponse,
    summary="Run feedback learning cycle",
    description="Process accumulated feedback and extract improvement patterns.",
    operation_id="run_feedback_learning_cycle",
)
async def run_learning_cycle(
    request: RunLearningRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(
        default=True, description="Run asynchronously (returns immediately with ID)"
    ),
    user: Dict[str, Any] = Depends(require_operator),
) -> LearningResponse:
    """
    Run a feedback learning cycle.

    This endpoint invokes the Feedback Learner agent (Tier 5) to:
    1. Collect feedback from the specified time range
    2. Analyze patterns and issues
    3. Generate improvement recommendations
    4. Propose knowledge updates

    Args:
        request: Learning cycle parameters
        background_tasks: FastAPI background tasks
        async_mode: If True, returns immediately with batch ID

    Returns:
        Learning results or pending status if async
    """
    batch_id = f"fb_{uuid4().hex[:12]}"

    # Create initial response
    response = LearningResponse(
        batch_id=batch_id,
        status=LearningStatus.PENDING if async_mode else LearningStatus.COLLECTING,
    )

    if async_mode:
        # Store pending batch
        await _persist_batch(response)

        # Schedule background task
        background_tasks.add_task(
            _run_learning_task,
            batch_id=batch_id,
            request=request,
        )

        logger.info(f"Learning cycle {batch_id} queued for background execution")
        return response

    # Synchronous execution
    try:
        result = await _execute_learning_cycle(request)
        result.batch_id = batch_id
        # Persist the detected patterns / proposed updates too — not just the
        # batch — so the Patterns/Updates tabs reflect this cycle. The UI drives
        # this sync path; without this the artifacts were silently discarded.
        await _persist_cycle_artifacts(result)
        await _persist_batch(result)
        return result
    except HTTPException:
        # F-010-backend (#429, codex iter-1 M1): preserve 503 from
        # agent-import guard.
        raise
    except Exception as e:
        logger.error(f"Learning cycle failed: {e}")
        response.status = LearningStatus.FAILED
        response.errors.append(str(e))
        await _persist_batch(response)
        raise HTTPException(status_code=500, detail=f"Learning cycle failed: {e}")


@router.post(
    "/process",
    response_model=LearningResponse,
    summary="Process feedback items",
    description="Process specific feedback items and detect patterns.",
    operation_id="process_feedback_items",
)
async def process_feedback(
    request: ProcessFeedbackRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> LearningResponse:
    """
    Process specific feedback items.

    This endpoint allows processing individual feedback items
    without running a full learning cycle.

    Args:
        request: Feedback items and processing options

    Returns:
        Processing results with any detected patterns
    """
    batch_id = f"fb_{uuid4().hex[:12]}"

    try:
        import time

        start_time = time.time()

        # Store feedback items
        for item in request.items:
            if not item.feedback_id:
                item.feedback_id = f"fbi_{uuid4().hex[:8]}"
            if not item.timestamp:
                item.timestamp = datetime.now(timezone.utc).isoformat()
            await _persist_item(item)

        # Detect patterns if requested
        detected_patterns: List[DetectedPattern] = []
        recommendations: List[LearningRecommendation] = []

        if request.detect_patterns and len(request.items) >= 3:
            detected_patterns = _detect_patterns_from_items(request.items)

        if request.generate_recommendations and detected_patterns:
            recommendations = _generate_recommendations(detected_patterns)

        total_latency = int((time.time() - start_time) * 1000)

        # Build summary
        by_type: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}
        for item in request.items:
            by_type[item.feedback_type.value] = by_type.get(item.feedback_type.value, 0) + 1
            by_agent[item.source_agent] = by_agent.get(item.source_agent, 0) + 1

        # F-008 (#428): compute positive_ratio from request.items instead of
        # hardcoding 0.7. An item is positive if its rating is >= 4 OR the
        # explicit user_feedback payload signals positive sentiment
        # (e.g. {"sentiment": "positive"} / {"helpful": True} / {"positive": True}).
        positive_count = sum(1 for item in request.items if _is_positive_feedback(item))
        positive_ratio = positive_count / len(request.items) if len(request.items) > 0 else 0.0

        feedback_summary = FeedbackSummary(
            total_feedback_items=len(request.items),
            by_type=by_type,
            by_agent=by_agent,
            positive_ratio=positive_ratio,
            time_range_start=request.items[0].timestamp or "",
            time_range_end=request.items[-1].timestamp or "",
        )

        response = LearningResponse(
            batch_id=batch_id,
            status=LearningStatus.COMPLETED,
            detected_patterns=detected_patterns,
            learning_recommendations=recommendations,
            patterns_detected=len(detected_patterns),
            recommendations_generated=len(recommendations),
            feedback_summary=feedback_summary,
            learning_summary=f"Processed {len(request.items)} feedback items. Found {len(detected_patterns)} patterns.",
            total_latency_ms=total_latency,
        )

        await _persist_batch(response)
        return response

    except Exception as e:
        logger.error(f"Feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@router.get(
    "/patterns",
    response_model=PatternListResponse,
    summary="List detected patterns",
    description="List all detected patterns with optional filtering.",
    operation_id="list_feedback_patterns",
)
async def list_patterns(
    severity: Optional[PatternSeverity] = Query(default=None, description="Filter by severity"),
    pattern_type: Optional[PatternType] = Query(default=None, description="Filter by type"),
    agent: Optional[str] = Query(default=None, description="Filter by affected agent"),
    limit: int = Query(default=50, description="Maximum results", ge=1, le=200),
) -> PatternListResponse:
    """
    List all detected patterns.

    Args:
        severity: Optional severity filter
        pattern_type: Optional type filter
        agent: Optional agent filter
        limit: Maximum number of results

    Returns:
        List of patterns matching filters
    """
    patterns = await _all_patterns()

    # Apply filters
    if severity:
        patterns = [p for p in patterns if p.severity == severity]
    if pattern_type:
        patterns = [p for p in patterns if p.pattern_type == pattern_type]
    if agent:
        patterns = [p for p in patterns if agent in p.affected_agents]

    # Sort by severity and frequency
    severity_order = {
        PatternSeverity.CRITICAL: 0,
        PatternSeverity.HIGH: 1,
        PatternSeverity.MEDIUM: 2,
        PatternSeverity.LOW: 3,
    }
    patterns.sort(key=lambda p: (severity_order[p.severity], -p.frequency))

    patterns = patterns[:limit]

    critical_count = sum(1 for p in patterns if p.severity == PatternSeverity.CRITICAL)
    high_count = sum(1 for p in patterns if p.severity == PatternSeverity.HIGH)

    return PatternListResponse(
        total_count=len(patterns),
        critical_count=critical_count,
        high_count=high_count,
        patterns=patterns,
    )


@router.get(
    "/updates",
    response_model=UpdateListResponse,
    summary="List knowledge updates",
    description="List all proposed and applied knowledge updates.",
    operation_id="list_knowledge_updates",
)
async def list_updates(
    status: Optional[UpdateStatus] = Query(default=None, description="Filter by status"),
    update_type: Optional[UpdateType] = Query(default=None, description="Filter by type"),
    agent: Optional[str] = Query(default=None, description="Filter by target agent"),
    limit: int = Query(default=50, description="Maximum results", ge=1, le=200),
) -> UpdateListResponse:
    """
    List knowledge updates.

    Args:
        status: Optional status filter
        update_type: Optional type filter
        agent: Optional agent filter
        limit: Maximum number of results

    Returns:
        List of updates matching filters
    """
    updates = await _all_updates()

    # Apply filters
    if status:
        updates = [u for u in updates if u.status == status]
    if update_type:
        updates = [u for u in updates if u.update_type == update_type]
    if agent:
        updates = [u for u in updates if u.target_agent == agent]

    # Sort by created_at descending
    updates.sort(key=lambda u: u.created_at, reverse=True)
    updates = updates[:limit]

    proposed_count = sum(1 for u in updates if u.status == UpdateStatus.PROPOSED)
    applied_count = sum(1 for u in updates if u.status == UpdateStatus.APPLIED)

    return UpdateListResponse(
        total_count=len(updates),
        proposed_count=proposed_count,
        applied_count=applied_count,
        updates=updates,
    )


@router.post(
    "/updates/{update_id}/apply",
    response_model=KnowledgeUpdate,
    summary="Apply knowledge update",
    description="Apply a proposed knowledge update to the system.",
    operation_id="apply_knowledge_update",
)
async def apply_update(
    update_id: str,
    request: ApplyUpdateRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> KnowledgeUpdate:
    """
    Apply a knowledge update.

    Args:
        update_id: Update identifier
        request: Apply options

    Returns:
        Updated knowledge update record

    Raises:
        HTTPException: If update not found or not in valid state
    """
    update = await _load_update(update_id)
    if update is None:
        raise HTTPException(
            status_code=404,
            detail=f"Update {update_id} not found",
        )

    if update.status not in [UpdateStatus.PROPOSED, UpdateStatus.APPROVED]:
        if not request.force:
            raise HTTPException(
                status_code=400,
                detail=f"Update {update_id} is in status {update.status}, cannot apply",
            )

    # #1243: REAL apply — write the recorded learning to the same knowledge
    # store the learning cycle's auto-apply path uses (SupabaseKnowledgeStore,
    # #837: fail-closed, read-back confirmed). target_component IS the store's
    # knowledge_type (graph-state writers set it from knowledge_type; prod
    # U_R1 carries "prompt"). FAIL-HONEST: any failure below leaves the record
    # un-flipped — APPLIED means the store write was read-back confirmed.
    from src.agents.feedback_learner.knowledge_stores import KNOWLEDGE_TYPES

    knowledge_type = update.target_component
    if knowledge_type not in KNOWLEDGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Update {update_id} has target_component {knowledge_type!r}, which maps "
                f"to no real knowledge store (expected one of {sorted(KNOWLEDGE_TYPES)}); "
                "refusing to record a status flip that applies nothing"
            ),
        )

    stores = await _get_knowledge_stores()
    store = stores.get(knowledge_type)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge stores unavailable — update NOT applied",
        )

    try:
        logger.info(f"Applying update {update_id} to {update.target_agent}")
        # Capture the pre-apply store value so rollback can restore the true
        # prior state (None => no prior row => rollback deletes the row).
        # On a force re-apply of an already-APPLIED update the store holds
        # this update's own proposed_value — re-capturing would stomp the
        # true prior and rollback would restore the wrong state, so keep the
        # original capture. (From ROLLED_BACK the store holds the restored
        # prior, so re-capturing there is correct.)
        already_applied = update.status == UpdateStatus.APPLIED
        prior = None if already_applied else await store.get(update.target_agent)
        applied = await store.update(
            key=update.target_agent,
            value=update.proposed_value,
            justification=update.rationale,
        )
        if not applied:
            raise HTTPException(
                status_code=502,
                detail=("Knowledge-store write failed or read-back mismatch — update NOT applied"),
            )
        if not already_applied:
            update.current_value = None if prior is None else str(prior)
        update.status = UpdateStatus.APPLIED
        update.applied_at = datetime.now(timezone.utc)
        await _persist_update(update)
        return update

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply update {update_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply update: {e}")


@router.post(
    "/updates/{update_id}/rollback",
    response_model=KnowledgeUpdate,
    summary="Rollback knowledge update",
    description="Rollback a previously applied knowledge update.",
    operation_id="rollback_knowledge_update",
)
async def rollback_update(
    update_id: str,
    user: Dict[str, Any] = Depends(require_operator),
) -> KnowledgeUpdate:
    """
    Rollback a knowledge update.

    Args:
        update_id: Update identifier

    Returns:
        Updated knowledge update record

    Raises:
        HTTPException: If update not found or not applied
    """
    update = await _load_update(update_id)
    if update is None:
        raise HTTPException(
            status_code=404,
            detail=f"Update {update_id} not found",
        )

    if update.status != UpdateStatus.APPLIED:
        raise HTTPException(
            status_code=400,
            detail=f"Update {update_id} is not applied, cannot rollback",
        )

    # #1243: REAL rollback — restore the captured pre-apply value in the
    # knowledge store (or remove the row when this was a first-ever apply:
    # current_value None). FAIL-HONEST: a failed store write keeps the record
    # APPLIED — ROLLED_BACK means the store verifiably holds the prior state.
    # NB: single-current-value store — rolling back an update that was later
    # overwritten by ANOTHER apply to the same (type, key) restores THIS
    # update's prior, clobbering the newer value; operators see the store's
    # justification trail ("rollback of <id>").
    from src.agents.feedback_learner.knowledge_stores import KNOWLEDGE_TYPES

    knowledge_type = update.target_component
    if knowledge_type not in KNOWLEDGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Update {update_id} has target_component {knowledge_type!r}, which maps "
                f"to no real knowledge store (expected one of {sorted(KNOWLEDGE_TYPES)})"
            ),
        )

    stores = await _get_knowledge_stores()
    store = stores.get(knowledge_type)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge stores unavailable — update NOT rolled back",
        )

    try:
        logger.info(f"Rolling back update {update_id}")
        if update.current_value is not None:
            restored = await store.update(
                key=update.target_agent,
                value=update.current_value,
                justification=f"rollback of {update_id}",
            )
        else:
            restored = await store.delete(update.target_agent)
        if not restored:
            raise HTTPException(
                status_code=502,
                detail=("Knowledge-store rollback write failed — update stays APPLIED"),
            )
        update.status = UpdateStatus.ROLLED_BACK
        await _persist_update(update)
        return update

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rollback update {update_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rollback: {e}")


@router.get(
    "/health",
    response_model=FeedbackHealthResponse,
    summary="Feedback learning service health",
    description="Check health status of the feedback learning service.",
    operation_id="get_feedback_service_health",
)
async def get_feedback_health() -> FeedbackHealthResponse:
    """
    Get health status of feedback learning service.

    Returns:
        Service health information
    """
    # Check agent availability
    agent_available = True
    try:
        from src.agents.feedback_learner import FeedbackLearnerAgent  # noqa: F401

        agent_available = True
    except ImportError:
        agent_available = False

    # Count recent cycles
    now = datetime.now(timezone.utc)
    learning = await _all_batches()
    cycles_24h = sum(1 for lr in learning if (now - lr.timestamp).total_seconds() < 86400)

    # Get last cycle
    last_cycle = max((lr.timestamp for lr in learning), default=None)

    # Count active items
    updates = await _all_updates()
    patterns_active = len(await _all_patterns())
    pending_updates = sum(1 for u in updates if u.status == UpdateStatus.PROPOSED)

    # #1661: the optimizer half of the loop. Read here rather than left to a
    # log line, because "the daily task ran and skipped" and "the daily task
    # ran and optimized" are indistinguishable from everything else on this
    # page — and the loop has been in the first state since it was built.
    from src.agents.feedback_learner import signal_store
    from src.memory.services.factories import get_async_supabase_client

    try:
        gate_client = await get_async_supabase_client()
        optimizer = OptimizerGateStatus(**await signal_store.get_optimizer_gate_status(gate_client))
    except Exception as e:  # noqa: BLE001 - health must degrade, never 500
        logger.warning("optimizer gate status unavailable: %s", e)
        optimizer = OptimizerGateStatus(
            min_signals=signal_store.optimizer_min_signals(),
            min_reward=signal_store.OPTIMIZER_MIN_REWARD,
            reason=f"Optimizer gate status unavailable ({e})",
        )

    return FeedbackHealthResponse(
        status="healthy" if agent_available else "degraded",
        agent_available=agent_available,
        last_learning_cycle=last_cycle,
        cycles_24h=cycles_24h,
        patterns_active=patterns_active,
        pending_updates=pending_updates,
        optimizer=optimizer,
    )


@router.get(
    "/{batch_id}",
    response_model=LearningResponse,
    summary="Get learning results",
    description="Retrieve results of a learning cycle by batch ID.",
    operation_id="get_feedback_learning_results",
)
async def get_learning_results(batch_id: str) -> LearningResponse:
    """
    Get learning cycle results by batch ID.

    Args:
        batch_id: Unique batch identifier

    Returns:
        Learning results

    Raises:
        HTTPException: If batch not found
    """
    result = await _load_batch(batch_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Learning batch {batch_id} not found",
        )

    return result


# =============================================================================
# OPIK FEEDBACK LOOP ENDPOINTS (Phase 4 - G23)
# =============================================================================


class TraceFeedbackRequest(BaseModel):
    """Request to record feedback for an Opik trace."""

    trace_id: str = Field(..., description="Opik trace ID")
    score: float = Field(..., description="Feedback score (0.0 to 1.0)", ge=0.0, le=1.0)
    agent_name: str = Field(..., description="Agent that generated the response")
    feedback_type: str = Field(
        default="rating", description="Type of feedback (rating, correction, outcome, explicit)"
    )
    span_id: Optional[str] = Field(default=None, description="Optional specific span ID")
    category: Optional[str] = Field(
        default=None, description="Feedback category (accuracy, relevance, latency, format)"
    )
    user_feedback: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw user feedback data"
    )
    query: Optional[str] = Field(default=None, description="Original user query")
    response: Optional[str] = Field(default=None, description="Agent response")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "trace_abc123",
                "score": 0.8,
                "agent_name": "causal_impact",
                "feedback_type": "rating",
                "category": "accuracy",
                "user_feedback": {"rating": 4, "helpful": True, "comment": "Good analysis"},
                "query": "What drives TRx for Kisqali?",
            }
        }
    )


class TraceFeedbackResponse(BaseModel):
    """Response from recording trace feedback."""

    feedback_id: str = Field(..., description="Unique feedback identifier")
    trace_id: str = Field(..., description="Associated trace ID")
    agent_name: str = Field(..., description="Agent name")
    score: float = Field(..., description="Recorded score")
    logged_to_opik: bool = Field(..., description="Whether feedback was logged to Opik")
    timestamp: datetime = Field(..., description="Recording timestamp")


class AgentFeedbackStatsResponse(BaseModel):
    """Response with agent feedback statistics."""

    agent_name: str = Field(..., description="Agent name")
    total_feedback: int = Field(..., description="Total feedback count")
    average_score: float = Field(..., description="Average feedback score")
    positive_ratio: float = Field(..., description="Ratio of positive feedback")
    positive_count: int = Field(..., description="Positive feedback count")
    negative_count: int = Field(..., description="Negative feedback count")
    by_type: Dict[str, int] = Field(..., description="Count by feedback type")
    by_category: Dict[str, int] = Field(..., description="Count by category")
    score_trend: List[float] = Field(..., description="Recent score trend")
    last_feedback_time: Optional[datetime] = Field(default=None, description="Last feedback time")


class OptimizationSignal(BaseModel):
    """GEPA optimization signal from feedback."""

    signal_type: str = Field(..., description="Signal type (positive, negative, correction)")
    weight: float = Field(..., description="Signal weight (0.0 to 1.0)")
    feedback: str = Field(..., description="Signal description")
    suggested_action: Optional[str] = Field(default=None, description="Suggested improvement")
    confidence: float = Field(..., description="Signal confidence")


class OptimizationSignalsResponse(BaseModel):
    """Response with GEPA optimization signals."""

    agent_name: str = Field(..., description="Agent name")
    signals: List[OptimizationSignal] = Field(..., description="Optimization signals")
    total_feedback_analyzed: int = Field(..., description="Total feedback analyzed")
    ready_for_optimization: bool = Field(
        ..., description="Whether enough feedback for optimization"
    )


@router.post(
    "/trace",
    response_model=TraceFeedbackResponse,
    summary="Record feedback for Opik trace (G23)",
    description="Record user feedback and associate it with an Opik trace for observability.",
    operation_id="record_opik_trace_feedback",
)
async def record_trace_feedback(
    request: TraceFeedbackRequest,
) -> TraceFeedbackResponse:
    """
    Record user feedback for a specific Opik trace.

    This endpoint integrates with Opik to:
    1. Store feedback in the feedback collector
    2. Log feedback to the associated Opik trace
    3. Make feedback available for GEPA optimization

    Args:
        request: Trace feedback details

    Returns:
        Feedback recording confirmation
    """
    if not OPIK_FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Opik feedback integration not available",
        )

    try:
        # Record feedback using the Opik feedback collector
        record = await log_user_feedback(
            trace_id=request.trace_id,
            score=request.score,
            feedback_type=request.feedback_type,
            agent_name=request.agent_name,
            span_id=request.span_id,
            category=request.category,
            user_feedback=request.user_feedback,
            query=request.query,
            response=request.response,
            metadata=request.metadata,
        )

        # Check if logged to Opik
        collector = get_feedback_collector()
        logged_to_opik = collector.opik_enabled

        logger.info(
            f"Recorded trace feedback {record.feedback_id} for {request.agent_name}: "
            f"score={request.score:.2f}, opik={logged_to_opik}"
        )

        return TraceFeedbackResponse(
            feedback_id=record.feedback_id,
            trace_id=record.trace_id,
            agent_name=record.agent_name,
            score=record.score,
            logged_to_opik=logged_to_opik,
            timestamp=record.timestamp,
        )

    except Exception as e:
        logger.error(f"Failed to record trace feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")


@router.get(
    "/agent/{agent_name}/stats",
    response_model=AgentFeedbackStatsResponse,
    summary="Get agent feedback statistics (G23)",
    description="Get aggregated feedback statistics for an agent.",
    operation_id="get_agent_feedback_stats",
)
async def get_agent_feedback_stats(
    agent_name: str,
) -> AgentFeedbackStatsResponse:
    """
    Get aggregated feedback statistics for an agent.

    Provides insights into agent performance based on user feedback.

    Args:
        agent_name: Name of the agent

    Returns:
        Aggregated feedback statistics
    """
    if not OPIK_FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Opik feedback integration not available",
        )

    try:
        collector = get_feedback_collector()
        stats = collector.get_agent_stats(agent_name)

        return AgentFeedbackStatsResponse(
            agent_name=stats.agent_name,
            total_feedback=stats.total_feedback,
            average_score=stats.average_score,
            positive_ratio=stats.positive_ratio,
            positive_count=stats.positive_count,
            negative_count=stats.negative_count,
            by_type=stats.by_type,
            by_category=stats.by_category,
            score_trend=stats.score_trend,
            last_feedback_time=stats.last_feedback_time,
        )

    except Exception as e:
        logger.error(f"Failed to get agent stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@router.get(
    "/agent/{agent_name}/signals",
    response_model=OptimizationSignalsResponse,
    summary="Get GEPA optimization signals (G23)",
    description="Get feedback-derived optimization signals for GEPA prompt improvement.",
    operation_id="get_gepa_optimization_signals",
)
async def get_optimization_signals(
    agent_name: str,
    min_feedback_count: int = Query(default=5, description="Minimum feedback required", ge=1),
) -> OptimizationSignalsResponse:
    """
    Get GEPA optimization signals derived from user feedback.

    Analyzes accumulated feedback to generate actionable signals
    that GEPA can use to improve agent prompts.

    Args:
        agent_name: Name of the agent
        min_feedback_count: Minimum feedback required to generate signals

    Returns:
        Optimization signals for GEPA
    """
    if not OPIK_FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Opik feedback integration not available",
        )

    try:
        collector = get_feedback_collector()
        stats = collector.get_agent_stats(agent_name)
        signal_dicts = get_feedback_signals_for_gepa(agent_name, min_feedback_count)

        signals = [
            OptimizationSignal(
                signal_type=s["signal_type"],
                weight=s["weight"],
                feedback=s["feedback"],
                suggested_action=s.get("suggested_action"),
                confidence=s["confidence"],
            )
            for s in signal_dicts
        ]

        return OptimizationSignalsResponse(
            agent_name=agent_name,
            signals=signals,
            total_feedback_analyzed=stats.total_feedback,
            ready_for_optimization=stats.total_feedback >= min_feedback_count,
        )

    except Exception as e:
        logger.error(f"Failed to get optimization signals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {e}")


@router.get(
    "/agent/{agent_name}/gepa-batch",
    summary="Get GEPA training batch (G23)",
    description="Get a batch of feedback examples for GEPA training.",
    operation_id="get_gepa_training_batch",
)
async def get_gepa_training_batch(
    agent_name: str,
    batch_size: int = Query(default=50, description="Batch size", ge=1, le=200),
) -> Dict[str, Any]:
    """
    Get a batch of feedback examples formatted for GEPA training.

    Args:
        agent_name: Name of the agent
        batch_size: Number of examples to return

    Returns:
        GEPA-formatted training examples
    """
    if not OPIK_FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Opik feedback integration not available",
        )

    try:
        collector = get_feedback_collector()
        examples = collector.get_gepa_feedback_batch(agent_name, batch_size)

        return {
            "agent_name": agent_name,
            "batch_size": len(examples),
            "examples": examples,
        }

    except Exception as e:
        logger.error(f"Failed to get GEPA batch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _run_learning_task(
    batch_id: str,
    request: RunLearningRequest,
) -> None:
    """Background task to run learning cycle."""
    try:
        logger.info(f"Starting learning cycle {batch_id}")

        # Update status (read-modify-persist; in-memory mutates in place)
        existing = await _load_batch(batch_id)
        if existing is not None:
            existing.status = LearningStatus.COLLECTING
            await _persist_batch(existing)

        # Execute learning
        result = await _execute_learning_cycle(request)
        result.batch_id = batch_id

        # Store patterns and updates (shared with the sync path).
        await _persist_cycle_artifacts(result)

        # Store result
        await _persist_batch(result)

        logger.info(f"Learning cycle {batch_id} completed successfully")

    except Exception as e:
        logger.error(f"Learning cycle {batch_id} failed: {e}")
        existing = await _load_batch(batch_id)
        if existing is not None:
            existing.status = LearningStatus.FAILED
            existing.errors.append(str(e))
            await _persist_batch(existing)


async def _execute_learning_cycle(
    request: RunLearningRequest,
) -> LearningResponse:
    """
    Execute feedback learning using Feedback Learner agent.

    This function orchestrates the Feedback Learner agent (Tier 5) to:
    1. Collect feedback via collector node
    2. Analyze patterns via analyzer node
    3. Extract learnings via extractor node
    4. Propose updates via updater node
    """
    import time

    start_time = time.time()

    try:
        # Try to use the actual Feedback Learner agent
        from src.agents.feedback_learner.graph import build_feedback_learner_graph
        from src.agents.feedback_learner.state import FeedbackLearnerState

        # Set default time range
        now = datetime.now(timezone.utc)
        time_range_start = request.time_range_start or (
            datetime.fromtimestamp(now.timestamp() - 86400, tz=timezone.utc).isoformat()
        )
        time_range_end = request.time_range_end or now.isoformat()

        # Initialize state (cast partial state - remaining fields populated by graph nodes)
        initial_state = cast(
            FeedbackLearnerState,
            {
                "batch_id": "",
                "time_range_start": time_range_start,
                "time_range_end": time_range_end,
                "focus_agents": request.focus_agents or [],
                # Previously dropped here — the updater then applied every
                # update regardless of the request (see KnowledgeUpdaterNode).
                "auto_apply": request.auto_apply,
                "status": "pending",
                "errors": [],
                "warnings": [],
            },
        )

        # #837 (+F15): wire the REAL feedback source + knowledge stores so this
        # route runs a FULLY real cycle — it reports updates_applied /
        # update_effectiveness as real measured values rather than a structural
        # 0 / None. Fail-closed → (None, None) → the honest unwired path (F15).
        # persist_signals=True still persists the finalized training signal so this
        # caller is not a bypass of the persistence path.
        from src.agents.feedback_learner.agent import build_production_feedback_stores

        # #883 deferred: the builder's third element (the shared async client)
        # arms the rubric node's learning_signals persistence; the rubric
        # evaluation context is derived inside the node from the run's real
        # collected feedback. Fail-closed (None) -> rubric skips honestly.
        feedback_store, knowledge_stores, db_client = await build_production_feedback_stores()
        graph = build_feedback_learner_graph(
            feedback_store=feedback_store,
            knowledge_stores=knowledge_stores,
            db_client=db_client,
            persist_signals=True,
        )
        result = await graph.ainvoke(initial_state)

        # Convert agent output to API response
        total_latency = int((time.time() - start_time) * 1000)

        # #837: state ``applied_updates`` is a list of applied update_id STRINGS
        # (KnowledgeUpdaterNode); the full update dicts live in
        # ``proposed_updates`` (``_finalize_training_signal`` does NOT write the
        # records back into state). Re-hydrate the applied IDs to their proposed
        # dicts — mirroring ``graph.py`` — so the response's ``applied_updates``
        # LIST agrees with ``updates_applied``. Feeding the raw strings straight
        # into ``_convert_updates`` (a dict API) silently produced ``[]`` while
        # the count stayed ``N`` — a self-contradicting response on the exact
        # field this path makes real.
        _proposed_updates = result.get("proposed_updates", []) or []
        _applied_ids = set(result.get("applied_updates", []) or [])
        _applied_records = [
            u
            for u in _proposed_updates
            if isinstance(u, dict) and u.get("update_id") in _applied_ids
        ]

        return LearningResponse(
            batch_id="",  # Will be set by caller
            status=LearningStatus.COMPLETED
            if result.get("status") == "completed"
            else LearningStatus.FAILED,
            detected_patterns=_convert_patterns(result.get("detected_patterns", [])),
            learning_recommendations=_convert_recommendations(
                result.get("learning_recommendations", [])
            ),
            priority_improvements=result.get("priority_improvements", []),
            proposed_updates=_convert_updates(_proposed_updates),
            applied_updates=_convert_updates(_applied_records, applied=True),
            learning_summary=result.get("learning_summary", ""),
            patterns_detected=len(result.get("detected_patterns", [])),
            recommendations_generated=len(result.get("learning_recommendations", [])),
            updates_proposed=len(result.get("proposed_updates", [])),
            updates_applied=len(result.get("applied_updates", [])),
            collection_latency_ms=result.get("collection_latency_ms", 0),
            analysis_latency_ms=result.get("analysis_latency_ms", 0),
            total_latency_ms=total_latency,
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
        )

    except ImportError as e:
        # F-010-backend (#429): fail-closed in production unless mock-fallback
        # is explicitly enabled (E2I_REQUIRE_AGENT_IMPORT=0 or ENVIRONMENT!=production).
        from src.api.utils.agent_import_guard import guard_or_raise

        guard_or_raise(e, agent_name="Feedback Learner")
        return _generate_mock_learning_response(request, start_time)

    except Exception as e:
        logger.error(f"Learning cycle execution failed: {e}")
        raise


def _convert_patterns(patterns: List[Dict[str, Any]]) -> List[DetectedPattern]:
    """Convert agent output to API response format."""
    result = []
    for p in patterns:
        try:
            result.append(
                DetectedPattern(
                    pattern_id=p.get("pattern_id", f"pat_{uuid4().hex[:8]}"),
                    pattern_type=PatternType(p.get("pattern_type", "accuracy_issue")),
                    description=p.get("description", ""),
                    frequency=p.get("frequency", 1),
                    severity=PatternSeverity(p.get("severity", "medium")),
                    affected_agents=p.get("affected_agents", []),
                    example_feedback_ids=p.get("example_feedback_ids", []),
                    root_cause_hypothesis=p.get("root_cause_hypothesis", ""),
                    confidence=p.get("confidence", 0.7),
                    # #1256: agent output carries no timestamp — stamp detection
                    # time here so the persisted payload owns it. Without this,
                    # every pattern took the persistence row's created_at, which
                    # upserts never refresh — a recycled pattern_id served the
                    # FIRST cycle's timestamp as if it were the current one.
                    detected_at=p.get("detected_at") or datetime.now(timezone.utc),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert pattern: {e}")
    return result


def _convert_recommendations(recommendations: List[Dict[str, Any]]) -> List[LearningRecommendation]:
    """Convert agent output to API response format."""
    result = []
    for r in recommendations:
        try:
            result.append(
                LearningRecommendation(
                    recommendation_id=r.get("recommendation_id", f"rec_{uuid4().hex[:8]}"),
                    pattern_id=r.get("pattern_id", ""),
                    priority=r.get("priority", 5),
                    recommendation_type=r.get("recommendation_type", ""),
                    description=r.get("description", ""),
                    expected_impact=r.get("expected_impact", ""),
                    implementation_effort=r.get("implementation_effort", "Medium"),
                    affected_agents=r.get("affected_agents", []),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert recommendation: {e}")
    return result


# KnowledgeUpdaterNode emits the graph-state KnowledgeUpdate shape
# (knowledge_type/key/old_value/new_value/justification, see
# src/agents/feedback_learner/state.py) — map its knowledge_type onto the
# API's UpdateType so real cycle output doesn't collapse to a fabricated
# "prompt_refinement" default with empty content fields.
_KNOWLEDGE_TYPE_TO_UPDATE_TYPE: Dict[str, UpdateType] = {
    "prompt": UpdateType.PROMPT_REFINEMENT,
    "threshold": UpdateType.PARAMETER_TUNING,
    "agent_config": UpdateType.PARAMETER_TUNING,
    "baseline": UpdateType.RULE_MODIFICATION,
}


def _convert_updates(updates: List[Dict[str, Any]], applied: bool = False) -> List[KnowledgeUpdate]:
    """Convert agent output (graph-state or API-style dicts) to API format.

    #1243 (PR #1241 final-review minor a): graph-state dicts carry no
    ``status`` key, so applied_updates entries defaulted to "proposed" — a
    self-contradicting response under auto_apply. Callers converting records
    the cycle actually applied pass ``applied=True`` to stamp the default
    status/applied_at honestly (an explicit ``status`` in the dict still wins).
    """
    default_status = "applied" if applied else "proposed"
    result = []
    for u in updates:
        try:
            if "update_type" in u:
                update_type = UpdateType(u["update_type"])
            else:
                update_type = _KNOWLEDGE_TYPE_TO_UPDATE_TYPE.get(
                    str(u.get("knowledge_type", "")), UpdateType.PROMPT_REFINEMENT
                )
            current_value = u.get("old_value", u.get("current_value"))
            proposed_value = u.get("new_value", u.get("proposed_value", ""))
            status = UpdateStatus(u.get("status", default_status))
            applied_at = u.get("applied_at")
            if applied_at is None and status == UpdateStatus.APPLIED:
                # The cycle applies within the run; effective_date is the
                # closest recorded timestamp (KnowledgeUpdaterNode stamps it
                # at proposal creation, same cycle as the apply).
                applied_at = u.get("effective_date") or datetime.now(timezone.utc)
            result.append(
                KnowledgeUpdate(
                    update_id=u.get("update_id", f"upd_{uuid4().hex[:8]}"),
                    update_type=update_type,
                    status=status,
                    target_agent=str(u.get("key") or u.get("target_agent", "")),
                    target_component=str(u.get("knowledge_type") or u.get("target_component", "")),
                    current_value=None if current_value is None else str(current_value),
                    proposed_value="" if proposed_value is None else str(proposed_value),
                    rationale=str(u.get("justification") or u.get("rationale", "")),
                    expected_improvement=u.get("expected_improvement", ""),
                    applied_at=applied_at,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert update: {e}")
    return result


def _is_positive_feedback(item: FeedbackItem) -> bool:
    """Determine whether a feedback item represents positive sentiment.

    A feedback item is considered positive when ANY of the following hold:

    * ``user_feedback`` is a numeric rating (or dict containing ``rating``)
      with value >= 4 on a 1-5 scale.
    * ``user_feedback`` is a dict containing an explicit positive signal
      (``sentiment=="positive"``, ``helpful is True``, ``positive is True``,
      ``approved is True``).
    * ``user_feedback`` is a string equal (case-insensitive) to a positive
      label (``"positive"``, ``"good"``, ``"helpful"``, ``"thumbs_up"``).
    * ``user_feedback`` is a bool True.

    Returns False for any other shape (negative, neutral, malformed,
    correction without rating, or an outcome dict without a positive
    indicator). The function never raises on malformed input — feedback
    payloads originate from upstream clients and may be heterogeneous.
    """
    payload = item.user_feedback

    # Direct bool: True == positive
    if isinstance(payload, bool):
        return payload

    # Numeric rating: >= 4 on a 1-5 scale
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return payload >= 4

    # String labels
    if isinstance(payload, str):
        return payload.strip().lower() in {"positive", "good", "helpful", "thumbs_up"}

    # Dict: inspect well-known fields
    if isinstance(payload, dict):
        rating = payload.get("rating")
        if isinstance(rating, (int, float)) and not isinstance(rating, bool):
            return rating >= 4
        sentiment = payload.get("sentiment")
        if isinstance(sentiment, str) and sentiment.strip().lower() == "positive":
            return True
        for flag_key in ("helpful", "positive", "approved", "useful"):
            value = payload.get(flag_key)
            if isinstance(value, bool) and value:
                return True
        return False

    return False


def _detect_patterns_from_items(items: List[FeedbackItem]) -> List[DetectedPattern]:
    """Detect patterns from a list of feedback items."""
    patterns = []

    # Group by agent
    by_agent: Dict[str, List[FeedbackItem]] = {}
    for item in items:
        by_agent.setdefault(item.source_agent, []).append(item)

    # Check for accuracy issues (low ratings)
    for agent, agent_items in by_agent.items():
        low_rating_count = sum(
            1
            for item in agent_items
            if item.feedback_type == FeedbackType.RATING
            and isinstance(item.user_feedback, dict)
            and item.user_feedback.get("rating", 5) < 3
        )

        if low_rating_count >= 2:
            patterns.append(
                DetectedPattern(
                    pattern_id=f"pat_{uuid4().hex[:8]}",
                    pattern_type=PatternType.ACCURACY_ISSUE,
                    description=f"Multiple low ratings for {agent} responses",
                    frequency=low_rating_count,
                    severity=PatternSeverity.HIGH
                    if low_rating_count >= 5
                    else PatternSeverity.MEDIUM,
                    affected_agents=[agent],
                    example_feedback_ids=[i.feedback_id or "" for i in agent_items[:3]],
                    root_cause_hypothesis="Response quality may not meet user expectations",
                    confidence=0.7,
                )
            )

    return patterns


def _generate_recommendations(patterns: List[DetectedPattern]) -> List[LearningRecommendation]:
    """Generate recommendations from detected patterns."""
    recommendations = []

    for i, pattern in enumerate(patterns):
        recommendations.append(
            LearningRecommendation(
                recommendation_id=f"rec_{uuid4().hex[:8]}",
                pattern_id=pattern.pattern_id,
                priority=i + 1,
                recommendation_type="prompt_refinement",
                description=f"Review and refine prompts for {', '.join(pattern.affected_agents)}",
                expected_impact="Improved response accuracy and user satisfaction",
                implementation_effort="Medium",
                affected_agents=pattern.affected_agents,
            )
        )

    return recommendations


def _generate_mock_learning_response(
    request: RunLearningRequest,
    start_time: float,
) -> LearningResponse:
    """Generate mock response when agent is not available."""
    import time

    # Mock pattern
    mock_pattern = DetectedPattern(
        pattern_id=f"pat_{uuid4().hex[:8]}",
        pattern_type=PatternType.ACCURACY_ISSUE,
        description="Some responses lack specific data citations",
        frequency=15,
        severity=PatternSeverity.MEDIUM,
        affected_agents=request.focus_agents or ["causal_impact", "gap_analyzer"],
        example_feedback_ids=["fbi_001", "fbi_002", "fbi_003"],
        root_cause_hypothesis="Prompts may need more emphasis on data citation",
        confidence=0.75,
    )

    # Mock recommendation
    mock_recommendation = LearningRecommendation(
        recommendation_id=f"rec_{uuid4().hex[:8]}",
        pattern_id=mock_pattern.pattern_id,
        priority=1,
        recommendation_type="prompt_refinement",
        description="Add explicit instruction to cite data sources in responses",
        expected_impact="Improved credibility and verifiability of responses",
        implementation_effort="Low",
        affected_agents=mock_pattern.affected_agents,
    )

    # Mock update
    mock_update = KnowledgeUpdate(
        update_id=f"upd_{uuid4().hex[:8]}",
        update_type=UpdateType.PROMPT_REFINEMENT,
        status=UpdateStatus.PROPOSED,
        target_agent="causal_impact",
        target_component="system_prompt",
        current_value=None,
        proposed_value="Always cite specific data points and sources in your analysis.",
        rationale="Addresses pattern of responses lacking citations",
        expected_improvement="20% improvement in response credibility scores",
    )

    # Mock summary
    mock_summary = FeedbackSummary(
        total_feedback_items=47,
        by_type={"rating": 25, "correction": 12, "explicit": 10},
        by_agent={"causal_impact": 20, "gap_analyzer": 15, "orchestrator": 12},
        average_rating=3.8,
        positive_ratio=0.72,
        time_range_start=request.time_range_start or "2024-01-01T00:00:00Z",
        time_range_end=request.time_range_end or datetime.now(timezone.utc).isoformat(),
    )

    total_latency = int((time.time() - start_time) * 1000)

    return LearningResponse(
        batch_id="",
        status=LearningStatus.COMPLETED,
        detected_patterns=[mock_pattern],
        learning_recommendations=[mock_recommendation],
        priority_improvements=[
            "Add data citations to responses",
            "Improve response formatting consistency",
        ],
        proposed_updates=[mock_update],
        applied_updates=[],
        learning_summary="Analyzed 47 feedback items. Identified 1 pattern affecting response quality. Generated 1 recommendation and 1 proposed update.",
        feedback_summary=mock_summary,
        patterns_detected=1,
        recommendations_generated=1,
        updates_proposed=1,
        updates_applied=0,
        collection_latency_ms=150,
        analysis_latency_ms=300,
        total_latency_ms=total_latency,
        warnings=["Using mock data - Feedback Learner agent not available"],
    )
