"""
E2I Feedback Learning API
=========================

FastAPI endpoints for feedback processing, pattern detection, and knowledge updates.

Phase: Agent Output Routing

Endpoints:
- POST /feedback/learn: Run feedback learning cycle
- GET  /feedback/{batch_id}: Get learning results
- POST /feedback/process: Process specific feedback items
- GET  /feedback/patterns: List detected patterns
- GET  /feedback/updates: List knowledge updates
- GET  /feedback/health: Service health check

Integration Points:
- Feedback Learner Agent (Tier 5)
- Orchestrator for agent invocation
- Supabase for persistence

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback Learning"])


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
    timestamp: Optional[str] = Field(
        default=None, description="Feedback timestamp (ISO format)"
    )
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    source_agent: str = Field(..., description="Agent that generated the original response")
    query: str = Field(..., description="Original user query")
    agent_response: str = Field(..., description="Agent's response to the query")
    user_feedback: Any = Field(
        ..., description="User's feedback (rating, correction, outcome, etc.)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )

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
    auto_apply: bool = Field(
        default=False, description="Automatically apply approved updates"
    )

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
    detect_patterns: bool = Field(
        default=True, description="Whether to detect patterns"
    )
    generate_recommendations: bool = Field(
        default=True, description="Whether to generate recommendations"
    )


class ApplyUpdateRequest(BaseModel):
    """Request to apply a knowledge update."""

    update_id: str = Field(..., description="Update identifier to apply")
    force: bool = Field(
        default=False, description="Force apply even if not approved"
    )


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
    priority_improvements: List[str] = Field(
        default_factory=list, description="Top priority items"
    )
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


# =============================================================================
# IN-MEMORY STORAGE (replace with Supabase in production)
# =============================================================================

_learning_store: Dict[str, LearningResponse] = {}
_patterns_store: Dict[str, DetectedPattern] = {}
_updates_store: Dict[str, KnowledgeUpdate] = {}
_feedback_store: List[FeedbackItem] = []


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/learn",
    response_model=LearningResponse,
    summary="Run feedback learning cycle",
    description="Process accumulated feedback and extract improvement patterns.",
)
async def run_learning_cycle(
    request: RunLearningRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(
        default=True, description="Run asynchronously (returns immediately with ID)"
    ),
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
        _learning_store[batch_id] = response

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
        _learning_store[batch_id] = result
        return result
    except Exception as e:
        logger.error(f"Learning cycle failed: {e}")
        response.status = LearningStatus.FAILED
        response.errors.append(str(e))
        _learning_store[batch_id] = response
        raise HTTPException(status_code=500, detail=f"Learning cycle failed: {e}")


@router.get(
    "/{batch_id}",
    response_model=LearningResponse,
    summary="Get learning results",
    description="Retrieve results of a learning cycle by batch ID.",
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
    if batch_id not in _learning_store:
        raise HTTPException(
            status_code=404,
            detail=f"Learning batch {batch_id} not found",
        )

    return _learning_store[batch_id]


@router.post(
    "/process",
    response_model=LearningResponse,
    summary="Process feedback items",
    description="Process specific feedback items and detect patterns.",
)
async def process_feedback(
    request: ProcessFeedbackRequest,
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
            _feedback_store.append(item)

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

        feedback_summary = FeedbackSummary(
            total_feedback_items=len(request.items),
            by_type=by_type,
            by_agent=by_agent,
            positive_ratio=0.7,  # Mock for now
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

        _learning_store[batch_id] = response
        return response

    except Exception as e:
        logger.error(f"Feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@router.get(
    "/patterns",
    response_model=PatternListResponse,
    summary="List detected patterns",
    description="List all detected patterns with optional filtering.",
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
    patterns = list(_patterns_store.values())

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
    updates = list(_updates_store.values())

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
)
async def apply_update(
    update_id: str,
    request: ApplyUpdateRequest,
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
    if update_id not in _updates_store:
        raise HTTPException(
            status_code=404,
            detail=f"Update {update_id} not found",
        )

    update = _updates_store[update_id]

    if update.status not in [UpdateStatus.PROPOSED, UpdateStatus.APPROVED]:
        if not request.force:
            raise HTTPException(
                status_code=400,
                detail=f"Update {update_id} is in status {update.status}, cannot apply",
            )

    # Apply the update (in production, this would actually modify the agent)
    try:
        logger.info(f"Applying update {update_id} to {update.target_agent}")
        update.status = UpdateStatus.APPLIED
        update.applied_at = datetime.now(timezone.utc)
        _updates_store[update_id] = update
        return update

    except Exception as e:
        logger.error(f"Failed to apply update {update_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply update: {e}")


@router.post(
    "/updates/{update_id}/rollback",
    response_model=KnowledgeUpdate,
    summary="Rollback knowledge update",
    description="Rollback a previously applied knowledge update.",
)
async def rollback_update(update_id: str) -> KnowledgeUpdate:
    """
    Rollback a knowledge update.

    Args:
        update_id: Update identifier

    Returns:
        Updated knowledge update record

    Raises:
        HTTPException: If update not found or not applied
    """
    if update_id not in _updates_store:
        raise HTTPException(
            status_code=404,
            detail=f"Update {update_id} not found",
        )

    update = _updates_store[update_id]

    if update.status != UpdateStatus.APPLIED:
        raise HTTPException(
            status_code=400,
            detail=f"Update {update_id} is not applied, cannot rollback",
        )

    # Rollback the update
    try:
        logger.info(f"Rolling back update {update_id}")
        update.status = UpdateStatus.ROLLED_BACK
        _updates_store[update_id] = update
        return update

    except Exception as e:
        logger.error(f"Failed to rollback update {update_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rollback: {e}")


@router.get(
    "/health",
    response_model=FeedbackHealthResponse,
    summary="Feedback learning service health",
    description="Check health status of the feedback learning service.",
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
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent_available = True
    except ImportError:
        agent_available = False

    # Count recent cycles
    now = datetime.now(timezone.utc)
    cycles_24h = sum(
        1
        for lr in _learning_store.values()
        if (now - lr.timestamp).total_seconds() < 86400
    )

    # Get last cycle
    last_cycle = None
    if _learning_store:
        last_cycle = max(lr.timestamp for lr in _learning_store.values())

    # Count active items
    patterns_active = len(_patterns_store)
    pending_updates = sum(
        1 for u in _updates_store.values() if u.status == UpdateStatus.PROPOSED
    )

    return FeedbackHealthResponse(
        status="healthy" if agent_available else "degraded",
        agent_available=agent_available,
        last_learning_cycle=last_cycle,
        cycles_24h=cycles_24h,
        patterns_active=patterns_active,
        pending_updates=pending_updates,
    )


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

        # Update status
        if batch_id in _learning_store:
            _learning_store[batch_id].status = LearningStatus.COLLECTING

        # Execute learning
        result = await _execute_learning_cycle(request)
        result.batch_id = batch_id

        # Store patterns and updates
        for pattern in result.detected_patterns:
            _patterns_store[pattern.pattern_id] = pattern
        for update in result.proposed_updates:
            _updates_store[update.update_id] = update

        # Store result
        _learning_store[batch_id] = result

        logger.info(f"Learning cycle {batch_id} completed successfully")

    except Exception as e:
        logger.error(f"Learning cycle {batch_id} failed: {e}")
        if batch_id in _learning_store:
            _learning_store[batch_id].status = LearningStatus.FAILED
            _learning_store[batch_id].errors.append(str(e))


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
        from src.agents.feedback_learner.graph import create_feedback_learner_graph
        from src.agents.feedback_learner.state import FeedbackLearnerState

        # Set default time range
        now = datetime.now(timezone.utc)
        time_range_start = request.time_range_start or (
            datetime.fromtimestamp(now.timestamp() - 86400, tz=timezone.utc).isoformat()
        )
        time_range_end = request.time_range_end or now.isoformat()

        # Initialize state
        initial_state: FeedbackLearnerState = {
            "batch_id": "",
            "time_range_start": time_range_start,
            "time_range_end": time_range_end,
            "focus_agents": request.focus_agents or [],
            "min_feedback_count": request.min_feedback_count,
            "pattern_threshold": request.pattern_threshold,
            "auto_apply": request.auto_apply,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        # Create and run graph
        graph = create_feedback_learner_graph()
        result = await graph.ainvoke(initial_state)

        # Convert agent output to API response
        total_latency = int((time.time() - start_time) * 1000)

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
            proposed_updates=_convert_updates(result.get("proposed_updates", [])),
            applied_updates=_convert_updates(result.get("applied_updates", [])),
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
        logger.warning(f"Feedback Learner agent not available: {e}, using mock data")
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


def _convert_updates(updates: List[Dict[str, Any]]) -> List[KnowledgeUpdate]:
    """Convert agent output to API response format."""
    result = []
    for u in updates:
        try:
            result.append(
                KnowledgeUpdate(
                    update_id=u.get("update_id", f"upd_{uuid4().hex[:8]}"),
                    update_type=UpdateType(u.get("update_type", "prompt_refinement")),
                    status=UpdateStatus(u.get("status", "proposed")),
                    target_agent=u.get("target_agent", ""),
                    target_component=u.get("target_component", ""),
                    current_value=u.get("current_value"),
                    proposed_value=u.get("proposed_value", ""),
                    rationale=u.get("rationale", ""),
                    expected_improvement=u.get("expected_improvement", ""),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert update: {e}")
    return result


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
                    severity=PatternSeverity.HIGH if low_rating_count >= 5 else PatternSeverity.MEDIUM,
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
