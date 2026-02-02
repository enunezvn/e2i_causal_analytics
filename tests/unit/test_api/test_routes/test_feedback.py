"""
Unit tests for feedback API routes.

Tests all endpoints in src/api/routes/feedback.py including:
- Feedback learning cycle execution
- Pattern detection and recommendations
- Knowledge update management
- Opik trace feedback integration (G23)
- GEPA optimization signals
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def mock_opik_feedback():
    """Mock the Opik feedback integration module."""
    with patch("src.api.routes.feedback.OPIK_FEEDBACK_AVAILABLE", True):
        with patch("src.api.routes.feedback.log_user_feedback") as mock_log:
            with patch("src.api.routes.feedback.get_feedback_collector") as mock_get_collector:
                with patch(
                    "src.api.routes.feedback.get_feedback_signals_for_gepa"
                ) as mock_get_signals:
                    # Create async mock that returns trace_id based on input
                    async def mock_log_feedback(**kwargs):
                        mock_record = MagicMock()
                        mock_record.feedback_id = "fb_test123"
                        mock_record.trace_id = kwargs.get("trace_id", "trace_test")
                        mock_record.agent_name = kwargs.get("agent_name", "causal_impact")
                        # Always return 0.8 for score (simulates normalized score)
                        mock_record.score = 0.8
                        mock_record.timestamp = datetime.now(timezone.utc)
                        return mock_record

                    mock_log.side_effect = mock_log_feedback

                    mock_collector = MagicMock()
                    mock_collector.opik_enabled = True
                    mock_get_collector.return_value = mock_collector

                    mock_get_signals.return_value = []

                    yield {
                        "log_user_feedback": mock_log,
                        "get_feedback_collector": mock_get_collector,
                        "get_feedback_signals_for_gepa": mock_get_signals,
                    }


@pytest.fixture
def mock_feedback_learner_agent():
    """Mock the Feedback Learner agent."""
    # Import the graph module to patch it
    import src.agents.feedback_learner.graph as graph_module

    mock_graph = AsyncMock()
    mock_result = {
        "status": "completed",
        "detected_patterns": [],
        "learning_recommendations": [],
        "priority_improvements": [],
        "proposed_updates": [],
        "applied_updates": [],
        "learning_summary": "Test summary",
        "collection_latency_ms": 100,
        "analysis_latency_ms": 200,
        "errors": [],
        "warnings": [],
    }
    mock_graph.ainvoke.return_value = mock_result

    # Patch the function that the route tries to import (use create=True for non-existent attr)
    def mock_create():
        return mock_graph

    with patch.object(graph_module, "create_feedback_learner_graph", mock_create, create=True):
        with patch("src.agents.feedback_learner.state.FeedbackLearnerState", dict):
            yield mock_graph


@pytest.fixture
def sample_feedback_item():
    """Sample feedback item for testing."""
    from src.api.routes.feedback import FeedbackItem, FeedbackType

    return FeedbackItem(
        feedback_id="fbi_test123",
        timestamp=datetime.now(timezone.utc).isoformat(),
        feedback_type=FeedbackType.RATING,
        source_agent="causal_impact",
        query="What drives TRx?",
        agent_response="Rep visits drive TRx.",
        user_feedback={"rating": 4, "helpful": True},
        metadata={"session_id": "sess_123"},
    )


@pytest.fixture
def sample_run_learning_request():
    """Sample RunLearningRequest for testing."""
    from src.api.routes.feedback import RunLearningRequest

    return RunLearningRequest(
        time_range_start="2024-01-01T00:00:00Z",
        time_range_end="2024-01-07T23:59:59Z",
        focus_agents=["causal_impact", "gap_analyzer"],
        min_feedback_count=10,
        pattern_threshold=0.15,
        auto_apply=False,
    )


@pytest.fixture
def sample_detected_pattern():
    """Sample DetectedPattern for testing."""
    from src.api.routes.feedback import DetectedPattern, PatternSeverity, PatternType

    return DetectedPattern(
        pattern_id="pat_test123",
        pattern_type=PatternType.ACCURACY_ISSUE,
        description="Low ratings detected",
        frequency=5,
        severity=PatternSeverity.HIGH,
        affected_agents=["causal_impact"],
        example_feedback_ids=["fbi_1", "fbi_2"],
        root_cause_hypothesis="Quality issue",
        confidence=0.8,
    )


@pytest.fixture
def sample_knowledge_update():
    """Sample KnowledgeUpdate for testing."""
    from src.api.routes.feedback import KnowledgeUpdate, UpdateStatus, UpdateType

    return KnowledgeUpdate(
        update_id="upd_test123",
        update_type=UpdateType.PROMPT_REFINEMENT,
        status=UpdateStatus.PROPOSED,
        target_agent="causal_impact",
        target_component="system_prompt",
        proposed_value="Improved prompt",
        rationale="Better accuracy",
        expected_improvement="10% better",
    )


# =============================================================================
# TESTS - Learning Cycle
# =============================================================================


@pytest.mark.asyncio
async def test_run_learning_cycle_async(sample_run_learning_request, mock_feedback_learner_agent):
    """Test running learning cycle in async mode."""
    from fastapi import BackgroundTasks

    from src.api.routes.feedback import LearningStatus, run_learning_cycle

    background_tasks = BackgroundTasks()
    user = {"user_id": "test_user", "role": "operator"}

    result = await run_learning_cycle(
        request=sample_run_learning_request,
        background_tasks=background_tasks,
        async_mode=True,
        user=user,
    )

    assert result.status == LearningStatus.PENDING
    assert result.batch_id.startswith("fb_")


@pytest.mark.asyncio
async def test_run_learning_cycle_sync(sample_run_learning_request, mock_feedback_learner_agent):
    """Test running learning cycle synchronously."""
    from fastapi import BackgroundTasks

    from src.api.routes.feedback import LearningStatus, run_learning_cycle

    background_tasks = BackgroundTasks()
    user = {"user_id": "test_user", "role": "operator"}

    result = await run_learning_cycle(
        request=sample_run_learning_request,
        background_tasks=background_tasks,
        async_mode=False,
        user=user,
    )

    assert result.status == LearningStatus.COMPLETED
    assert result.batch_id.startswith("fb_")
    assert isinstance(result.total_latency_ms, int)


@pytest.mark.asyncio
async def test_run_learning_cycle_error(sample_run_learning_request):
    """Test learning cycle with error."""
    from fastapi import BackgroundTasks

    from src.api.routes.feedback import run_learning_cycle

    with patch("src.api.routes.feedback._execute_learning_cycle") as mock_exec:
        mock_exec.side_effect = Exception("Test error")

        background_tasks = BackgroundTasks()
        user = {"user_id": "test_user", "role": "operator"}

        with pytest.raises(HTTPException) as exc_info:
            await run_learning_cycle(
                request=sample_run_learning_request,
                background_tasks=background_tasks,
                async_mode=False,
                user=user,
            )

        assert exc_info.value.status_code == 500
        assert "failed" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_get_learning_results_success():
    """Test getting learning results by batch ID."""
    from src.api.routes.feedback import (
        LearningResponse,
        LearningStatus,
        _learning_store,
        get_learning_results,
    )

    batch_id = "fb_test123"
    _learning_store[batch_id] = LearningResponse(
        batch_id=batch_id,
        status=LearningStatus.COMPLETED,
    )

    result = await get_learning_results(batch_id)

    assert result.batch_id == batch_id
    assert result.status == LearningStatus.COMPLETED

    # Cleanup
    del _learning_store[batch_id]


@pytest.mark.asyncio
async def test_get_learning_results_not_found():
    """Test getting learning results for non-existent batch."""
    from src.api.routes.feedback import get_learning_results

    with pytest.raises(HTTPException) as exc_info:
        await get_learning_results("fb_nonexistent")

    assert exc_info.value.status_code == 404
    assert "not found" in str(exc_info.value.detail).lower()


# =============================================================================
# TESTS - Feedback Processing
# =============================================================================


@pytest.mark.asyncio
async def test_process_feedback_success(sample_feedback_item):
    """Test processing feedback items."""
    from src.api.routes.feedback import ProcessFeedbackRequest, process_feedback

    request = ProcessFeedbackRequest(
        items=[sample_feedback_item],
        detect_patterns=True,
        generate_recommendations=True,
    )

    user = {"user_id": "test_user", "role": "operator"}

    result = await process_feedback(request, user)

    assert result.batch_id.startswith("fb_")
    assert result.feedback_summary.total_feedback_items == 1
    assert result.feedback_summary.by_type.get("rating") == 1


@pytest.mark.asyncio
async def test_process_feedback_with_patterns(sample_feedback_item):
    """Test feedback processing with pattern detection."""
    from src.api.routes.feedback import ProcessFeedbackRequest, process_feedback

    # Create multiple low-rating items to trigger pattern
    items = []
    for i in range(3):
        item = sample_feedback_item.model_copy()
        item.feedback_id = f"fbi_test{i}"
        item.user_feedback = {"rating": 2, "helpful": False}
        items.append(item)

    request = ProcessFeedbackRequest(
        items=items,
        detect_patterns=True,
        generate_recommendations=True,
    )

    user = {"user_id": "test_user", "role": "operator"}
    result = await process_feedback(request, user)

    assert result.patterns_detected > 0
    assert len(result.detected_patterns) > 0


@pytest.mark.asyncio
async def test_process_feedback_no_pattern_detection(sample_feedback_item):
    """Test feedback processing without pattern detection."""
    from src.api.routes.feedback import ProcessFeedbackRequest, process_feedback

    request = ProcessFeedbackRequest(
        items=[sample_feedback_item],
        detect_patterns=False,
        generate_recommendations=False,
    )

    user = {"user_id": "test_user", "role": "operator"}
    result = await process_feedback(request, user)

    assert result.patterns_detected == 0
    assert len(result.detected_patterns) == 0


@pytest.mark.asyncio
async def test_process_feedback_error():
    """Test feedback processing with error."""
    from src.api.routes.feedback import ProcessFeedbackRequest, process_feedback

    # Invalid feedback item
    request = ProcessFeedbackRequest(items=[])
    user = {"user_id": "test_user", "role": "operator"}

    with patch("src.api.routes.feedback._feedback_store", side_effect=Exception("Storage error")):
        with pytest.raises(HTTPException) as exc_info:
            await process_feedback(request, user)

        assert exc_info.value.status_code == 500


# =============================================================================
# TESTS - Pattern Listing
# =============================================================================


@pytest.mark.asyncio
async def test_list_patterns_all(sample_detected_pattern):
    """Test listing all patterns."""
    from src.api.routes.feedback import _patterns_store, list_patterns

    _patterns_store[sample_detected_pattern.pattern_id] = sample_detected_pattern

    result = await list_patterns(severity=None, pattern_type=None, agent=None, limit=50)

    assert result.total_count >= 1
    assert any(p.pattern_id == sample_detected_pattern.pattern_id for p in result.patterns)

    # Cleanup
    del _patterns_store[sample_detected_pattern.pattern_id]


@pytest.mark.asyncio
async def test_list_patterns_filter_by_severity(sample_detected_pattern):
    """Test listing patterns filtered by severity."""
    from src.api.routes.feedback import PatternSeverity, _patterns_store, list_patterns

    _patterns_store[sample_detected_pattern.pattern_id] = sample_detected_pattern

    result = await list_patterns(
        severity=PatternSeverity.HIGH, pattern_type=None, agent=None, limit=50
    )

    assert all(p.severity == PatternSeverity.HIGH for p in result.patterns)

    # Cleanup
    del _patterns_store[sample_detected_pattern.pattern_id]


@pytest.mark.asyncio
async def test_list_patterns_filter_by_type(sample_detected_pattern):
    """Test listing patterns filtered by type."""
    from src.api.routes.feedback import PatternType, _patterns_store, list_patterns

    _patterns_store[sample_detected_pattern.pattern_id] = sample_detected_pattern

    result = await list_patterns(
        severity=None, pattern_type=PatternType.ACCURACY_ISSUE, agent=None, limit=50
    )

    assert all(p.pattern_type == PatternType.ACCURACY_ISSUE for p in result.patterns)

    # Cleanup
    del _patterns_store[sample_detected_pattern.pattern_id]


@pytest.mark.asyncio
async def test_list_patterns_filter_by_agent(sample_detected_pattern):
    """Test listing patterns filtered by agent."""
    from src.api.routes.feedback import _patterns_store, list_patterns

    _patterns_store[sample_detected_pattern.pattern_id] = sample_detected_pattern

    result = await list_patterns(severity=None, pattern_type=None, agent="causal_impact", limit=50)

    assert all("causal_impact" in p.affected_agents for p in result.patterns)

    # Cleanup
    del _patterns_store[sample_detected_pattern.pattern_id]


@pytest.mark.asyncio
async def test_list_patterns_with_limit(sample_detected_pattern):
    """Test listing patterns with limit."""
    from src.api.routes.feedback import _patterns_store, list_patterns

    # Add multiple patterns
    for i in range(5):
        pattern = sample_detected_pattern.model_copy()
        pattern.pattern_id = f"pat_test{i}"
        _patterns_store[pattern.pattern_id] = pattern

    result = await list_patterns(limit=2)

    assert len(result.patterns) <= 2

    # Cleanup
    for i in range(5):
        del _patterns_store[f"pat_test{i}"]


# =============================================================================
# TESTS - Update Management
# =============================================================================


@pytest.mark.asyncio
async def test_list_updates_all(sample_knowledge_update):
    """Test listing all knowledge updates."""
    from src.api.routes.feedback import _updates_store, list_updates

    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    result = await list_updates(status=None, update_type=None, agent=None, limit=50)

    assert result.total_count >= 1
    assert any(u.update_id == sample_knowledge_update.update_id for u in result.updates)

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_list_updates_filter_by_status(sample_knowledge_update):
    """Test listing updates filtered by status."""
    from src.api.routes.feedback import UpdateStatus, _updates_store, list_updates

    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    result = await list_updates(
        status=UpdateStatus.PROPOSED, update_type=None, agent=None, limit=50
    )

    assert all(u.status == UpdateStatus.PROPOSED for u in result.updates)

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_apply_update_success(sample_knowledge_update):
    """Test applying a knowledge update."""
    from src.api.routes.feedback import (
        ApplyUpdateRequest,
        UpdateStatus,
        _updates_store,
        apply_update,
    )

    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    request = ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=False)
    user = {"user_id": "test_user", "role": "operator"}

    result = await apply_update(sample_knowledge_update.update_id, request, user)

    assert result.status == UpdateStatus.APPLIED
    assert result.applied_at is not None

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_apply_update_not_found():
    """Test applying non-existent update."""
    from src.api.routes.feedback import ApplyUpdateRequest, apply_update

    request = ApplyUpdateRequest(update_id="upd_nonexistent", force=False)
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await apply_update("upd_nonexistent", request, user)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_apply_update_invalid_status(sample_knowledge_update):
    """Test applying update with invalid status."""
    from src.api.routes.feedback import (
        ApplyUpdateRequest,
        UpdateStatus,
        _updates_store,
        apply_update,
    )

    sample_knowledge_update.status = UpdateStatus.APPLIED
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    request = ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=False)
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await apply_update(sample_knowledge_update.update_id, request, user)

    assert exc_info.value.status_code == 400

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_apply_update_force(sample_knowledge_update):
    """Test force applying update regardless of status."""
    from src.api.routes.feedback import (
        ApplyUpdateRequest,
        UpdateStatus,
        _updates_store,
        apply_update,
    )

    sample_knowledge_update.status = UpdateStatus.ROLLED_BACK
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    request = ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=True)
    user = {"user_id": "test_user", "role": "operator"}

    result = await apply_update(sample_knowledge_update.update_id, request, user)

    assert result.status == UpdateStatus.APPLIED

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_rollback_update_success(sample_knowledge_update):
    """Test rolling back an applied update."""
    from src.api.routes.feedback import UpdateStatus, _updates_store, rollback_update

    sample_knowledge_update.status = UpdateStatus.APPLIED
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    user = {"user_id": "test_user", "role": "operator"}

    result = await rollback_update(sample_knowledge_update.update_id, user)

    assert result.status == UpdateStatus.ROLLED_BACK

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_rollback_update_not_applied(sample_knowledge_update):
    """Test rolling back update that is not applied."""
    from src.api.routes.feedback import _updates_store, rollback_update

    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await rollback_update(sample_knowledge_update.update_id, user)

    assert exc_info.value.status_code == 400

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


# =============================================================================
# TESTS - Health Check
# =============================================================================


@pytest.mark.asyncio
async def test_get_feedback_health():
    """Test feedback service health check."""
    from src.api.routes.feedback import get_feedback_health

    with patch("src.agents.feedback_learner.FeedbackLearnerAgent"):
        result = await get_feedback_health()

        assert result.status in ["healthy", "degraded"]
        assert isinstance(result.agent_available, bool)
        assert isinstance(result.cycles_24h, int)


# =============================================================================
# TESTS - Opik Trace Feedback (G23)
# =============================================================================


@pytest.mark.asyncio
async def test_record_trace_feedback_success(mock_opik_feedback):
    """Test recording trace feedback successfully."""
    from src.api.routes.feedback import TraceFeedbackRequest, record_trace_feedback

    request = TraceFeedbackRequest(
        trace_id="trace_test123",
        score=0.85,
        agent_name="causal_impact",
        feedback_type="rating",
        category="accuracy",
        query="What drives TRx?",
    )

    result = await record_trace_feedback(request)

    assert result.feedback_id == "fb_test123"
    assert result.trace_id == "trace_test123"
    assert result.score == 0.8
    assert result.logged_to_opik is True


@pytest.mark.asyncio
async def test_record_trace_feedback_unavailable():
    """Test recording trace feedback when Opik is unavailable."""
    from src.api.routes.feedback import TraceFeedbackRequest, record_trace_feedback

    with patch("src.api.routes.feedback.OPIK_FEEDBACK_AVAILABLE", False):
        request = TraceFeedbackRequest(
            trace_id="trace_test123",
            score=0.85,
            agent_name="causal_impact",
        )

        with pytest.raises(HTTPException) as exc_info:
            await record_trace_feedback(request)

        assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_record_trace_feedback_error(mock_opik_feedback):
    """Test recording trace feedback with error."""
    from src.api.routes.feedback import TraceFeedbackRequest, record_trace_feedback

    mock_opik_feedback["log_user_feedback"].side_effect = Exception("Test error")

    request = TraceFeedbackRequest(
        trace_id="trace_test123",
        score=0.85,
        agent_name="causal_impact",
    )

    with pytest.raises(HTTPException) as exc_info:
        await record_trace_feedback(request)

    assert exc_info.value.status_code == 500


# =============================================================================
# TESTS - Agent Feedback Stats (G23)
# =============================================================================


@pytest.mark.asyncio
async def test_get_agent_feedback_stats_success(mock_opik_feedback):
    """Test getting agent feedback statistics."""
    from src.api.routes.feedback import get_agent_feedback_stats

    mock_stats = MagicMock()
    mock_stats.agent_name = "causal_impact"
    mock_stats.total_feedback = 100
    mock_stats.average_score = 0.75
    mock_stats.positive_ratio = 0.8
    mock_stats.positive_count = 80
    mock_stats.negative_count = 20
    mock_stats.by_type = {"rating": 60, "correction": 40}
    mock_stats.by_category = {"accuracy": 50, "relevance": 50}
    mock_stats.score_trend = [0.7, 0.75, 0.8]
    mock_stats.last_feedback_time = datetime.now(timezone.utc)

    mock_opik_feedback[
        "get_feedback_collector"
    ].return_value.get_agent_stats.return_value = mock_stats

    result = await get_agent_feedback_stats("causal_impact")

    assert result.agent_name == "causal_impact"
    assert result.total_feedback == 100
    assert result.average_score == 0.75


@pytest.mark.asyncio
async def test_get_agent_feedback_stats_unavailable():
    """Test getting stats when Opik is unavailable."""
    from src.api.routes.feedback import get_agent_feedback_stats

    with patch("src.api.routes.feedback.OPIK_FEEDBACK_AVAILABLE", False):
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_feedback_stats("causal_impact")

        assert exc_info.value.status_code == 503


# =============================================================================
# TESTS - GEPA Optimization Signals (G23)
# =============================================================================


@pytest.mark.asyncio
async def test_get_optimization_signals_success(mock_opik_feedback):
    """Test getting GEPA optimization signals."""
    from src.api.routes.feedback import get_optimization_signals

    mock_stats = MagicMock()
    mock_stats.total_feedback = 10

    mock_opik_feedback[
        "get_feedback_collector"
    ].return_value.get_agent_stats.return_value = mock_stats
    mock_opik_feedback["get_feedback_signals_for_gepa"].return_value = [
        {
            "signal_type": "positive",
            "weight": 0.8,
            "feedback": "Good accuracy",
            "suggested_action": "Keep current approach",
            "confidence": 0.9,
        }
    ]

    result = await get_optimization_signals("causal_impact", min_feedback_count=5)

    assert result.agent_name == "causal_impact"
    assert result.total_feedback_analyzed == 10
    assert len(result.signals) == 1
    assert result.ready_for_optimization is True


@pytest.mark.asyncio
async def test_get_optimization_signals_insufficient_data(mock_opik_feedback):
    """Test getting signals with insufficient feedback."""
    from src.api.routes.feedback import get_optimization_signals

    mock_stats = MagicMock()
    mock_stats.total_feedback = 3

    mock_opik_feedback[
        "get_feedback_collector"
    ].return_value.get_agent_stats.return_value = mock_stats
    mock_opik_feedback["get_feedback_signals_for_gepa"].return_value = []

    result = await get_optimization_signals("causal_impact", min_feedback_count=5)

    assert result.ready_for_optimization is False


@pytest.mark.asyncio
async def test_get_gepa_training_batch_success(mock_opik_feedback):
    """Test getting GEPA training batch."""
    from src.api.routes.feedback import get_gepa_training_batch

    mock_examples = [
        {"query": "test1", "response": "answer1", "score": 0.8},
        {"query": "test2", "response": "answer2", "score": 0.9},
    ]

    mock_opik_feedback[
        "get_feedback_collector"
    ].return_value.get_gepa_feedback_batch.return_value = mock_examples

    result = await get_gepa_training_batch("causal_impact", batch_size=50)

    assert result["agent_name"] == "causal_impact"
    assert result["batch_size"] == 2
    assert len(result["examples"]) == 2


# =============================================================================
# TESTS - Helper Functions
# =============================================================================


def test_detect_patterns_from_items():
    """Test pattern detection from feedback items."""
    from src.api.routes.feedback import FeedbackItem, FeedbackType, _detect_patterns_from_items

    # Create low-rating items
    items = []
    for i in range(3):
        item = FeedbackItem(
            feedback_id=f"fbi_{i}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            feedback_type=FeedbackType.RATING,
            source_agent="causal_impact",
            query="test",
            agent_response="response",
            user_feedback={"rating": 2},
        )
        items.append(item)

    patterns = _detect_patterns_from_items(items)

    assert len(patterns) > 0
    assert patterns[0].frequency >= 2


def test_generate_recommendations():
    """Test recommendation generation from patterns."""
    from src.api.routes.feedback import (
        DetectedPattern,
        PatternSeverity,
        PatternType,
        _generate_recommendations,
    )

    pattern = DetectedPattern(
        pattern_id="pat_test",
        pattern_type=PatternType.ACCURACY_ISSUE,
        description="Test pattern",
        frequency=5,
        severity=PatternSeverity.HIGH,
        affected_agents=["causal_impact"],
        example_feedback_ids=["fbi_1"],
        root_cause_hypothesis="Test hypothesis",
        confidence=0.8,
    )

    recommendations = _generate_recommendations([pattern])

    assert len(recommendations) == 1
    assert recommendations[0].pattern_id == pattern.pattern_id


def test_convert_patterns():
    """Test converting patterns from agent output."""
    from src.api.routes.feedback import _convert_patterns

    patterns = [
        {
            "pattern_id": "pat_test",
            "pattern_type": "accuracy_issue",
            "description": "Test",
            "frequency": 5,
            "severity": "high",
            "affected_agents": ["causal_impact"],
            "example_feedback_ids": ["fbi_1"],
            "root_cause_hypothesis": "Test",
            "confidence": 0.8,
        }
    ]

    result = _convert_patterns(patterns)

    assert len(result) == 1
    assert result[0].pattern_id == "pat_test"


def test_convert_recommendations():
    """Test converting recommendations from agent output."""
    from src.api.routes.feedback import _convert_recommendations

    recommendations = [
        {
            "recommendation_id": "rec_test",
            "pattern_id": "pat_test",
            "priority": 1,
            "recommendation_type": "prompt_refinement",
            "description": "Test",
            "expected_impact": "Better",
            "implementation_effort": "Low",
            "affected_agents": ["causal_impact"],
        }
    ]

    result = _convert_recommendations(recommendations)

    assert len(result) == 1
    assert result[0].recommendation_id == "rec_test"


def test_convert_updates():
    """Test converting updates from agent output."""
    from src.api.routes.feedback import _convert_updates

    updates = [
        {
            "update_id": "upd_test",
            "update_type": "prompt_refinement",
            "status": "proposed",
            "target_agent": "causal_impact",
            "target_component": "prompt",
            "proposed_value": "New value",
            "rationale": "Better",
            "expected_improvement": "10%",
        }
    ]

    result = _convert_updates(updates)

    assert len(result) == 1
    assert result[0].update_id == "upd_test"


def test_generate_mock_learning_response(sample_run_learning_request):
    """Test generating mock learning response."""
    import time

    from src.api.routes.feedback import _generate_mock_learning_response

    start_time = time.time()

    result = _generate_mock_learning_response(sample_run_learning_request, start_time)

    assert result.status.value == "completed"
    assert result.patterns_detected > 0
    assert result.recommendations_generated > 0
    assert len(result.warnings) > 0  # Should warn about mock data
