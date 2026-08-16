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
def _no_live_supabase_in_route_unit_tests(monkeypatch):
    """Isolate this suite from the live DB (#883 hermetic lesson).

    On this box pytest autoloads ``.env`` (real SUPABASE creds). Historically
    ``mock_feedback_learner_agent`` patched a NONEXISTENT symbol
    (``create_feedback_learner_graph``, create=True) — a vacuous mock — so the
    sync learning-cycle test ran the REAL graph against REAL production
    stores, silently depositing one ``dspy_agent_training_signals`` row per
    run. #892 repointed the fixture at the real builder
    (``build_feedback_learner_graph``); this guard stays as defense in depth
    so no future fixture regression can reach the live DB again: pin the
    production builder to the fail-closed triple and the default persist
    factory to None. Tests that patch the builder explicitly override this.
    """

    async def _disarmed():
        return None, None, None

    monkeypatch.setattr(
        "src.agents.feedback_learner.agent.build_production_feedback_stores",
        _disarmed,
    )
    monkeypatch.setattr(
        "src.memory.services.factories.get_supabase_client",
        lambda: None,
    )


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
    """Mock the Feedback Learner graph at the symbol the route actually uses.

    ``_execute_learning_cycle`` does ``from src.agents.feedback_learner.graph
    import build_feedback_learner_graph`` at call time, so patching that module
    attribute intercepts the import. (#892: the old fixture patched a
    NONEXISTENT ``create_feedback_learner_graph`` with ``create=True`` — the
    patch bound nothing the route reads, and the sync test silently ran the
    REAL graph. Proven by canary: ``mock_graph.ainvoke.assert_awaited()``
    failed while the test passed.)
    """
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

    # The route calls build_feedback_learner_graph(feedback_store=...,
    # knowledge_stores=..., db_client=..., persist_signals=True); accept and
    # ignore those kwargs.
    with patch(
        "src.agents.feedback_learner.graph.build_feedback_learner_graph",
        return_value=mock_graph,
    ):
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
        # #1243: target_component IS the knowledge_type routed to the real
        # store (KNOWLEDGE_TYPES) — real records (e.g. prod U_R1) use "prompt".
        target_component="prompt",
        proposed_value="Improved prompt",
        rationale="Better accuracy",
        expected_improvement="10% better",
    )


class _FakeKnowledgeStore:
    """In-memory stand-in honoring the SupabaseKnowledgeStore contract:
    update() returns True only after the value is actually recorded (or False
    in fail-mode without recording); get()/delete() read/mutate the record.
    The real persist + read-back path is proven in
    tests/integration/test_feedback_learner_knowledge_stores_realdb.py."""

    def __init__(self, fail: bool = False):
        self.values: dict = {}
        self.justifications: dict = {}
        self.fail = fail

    async def update(self, key, value, justification=None):
        if self.fail:
            return False
        self.values[key] = value
        self.justifications[key] = justification
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        if self.fail:
            return False
        self.values.pop(key, None)
        return True


@pytest.fixture
def fake_knowledge_stores(monkeypatch):
    """Route apply/rollback at in-memory stores; returns the dict for asserts."""
    from src.agents.feedback_learner.knowledge_stores import KNOWLEDGE_TYPES

    stores = {kt: _FakeKnowledgeStore() for kt in KNOWLEDGE_TYPES}

    async def _stores():
        return stores

    monkeypatch.setattr("src.api.routes.feedback._get_knowledge_stores", _stores)
    return stores


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
    # #892: pin that the mocked graph is what ran (the old create=True fixture
    # bound nothing and this test silently exercised the REAL graph).
    mock_feedback_learner_agent.ainvoke.assert_awaited_once()
    assert result.learning_summary == "Test summary"


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
async def test_execute_learning_cycle_applied_updates_consistent_with_count():
    """#837: the response's applied_updates LIST must agree with updates_applied.

    KnowledgeUpdaterNode writes ``applied_updates`` into state as a list of
    update_id STRINGS (the IDs that durably persisted); the full update dicts
    live in ``proposed_updates`` (graph.py:_finalize_training_signal does NOT
    write the records back into state). Before the fix the route fed those
    strings straight into ``_convert_updates`` (which calls ``u.get(...)`` — a
    dict API), so every element raised ``AttributeError``, was swallowed, and the
    response reported ``updates_applied=N`` while ``applied_updates=[]`` — a
    self-contradicting response on the exact field this PR makes real. The route
    must re-hydrate the applied IDs to their proposed dicts (mirroring
    graph.py) so the list agrees with the count.
    """
    from src.api.routes.feedback import RunLearningRequest, _execute_learning_cycle

    proposed = [
        {
            "update_id": "U_R1",
            "knowledge_type": "baseline",
            "key": "causal_impact",
            "old_value": None,
            "new_value": "new baseline",
            "justification": "low ratings",
            "effective_date": "2024-01-01T00:00:00+00:00",
        },
        {
            "update_id": "U_R2",
            "knowledge_type": "threshold",
            "key": "gap_analyzer",
            "old_value": None,
            "new_value": "0.2",
            "justification": "drift",
            "effective_date": "2024-01-01T00:00:00+00:00",
        },
    ]
    # KnowledgeUpdaterNode emits applied_updates as a list of update_id STRINGS;
    # only U_R1 durably persisted (read-back confirmed), U_R2 did not.
    result_state = {
        "status": "completed",
        "detected_patterns": [],
        "learning_recommendations": [],
        "priority_improvements": [],
        "proposed_updates": proposed,
        "applied_updates": ["U_R1"],
        "learning_summary": "ok",
        "collection_latency_ms": 0,
        "analysis_latency_ms": 0,
        "errors": [],
        "warnings": [],
    }

    fake_graph = AsyncMock()
    fake_graph.ainvoke.return_value = result_state

    request = RunLearningRequest(
        time_range_start="2024-01-01T00:00:00Z",
        time_range_end="2024-01-07T23:59:59Z",
        focus_agents=[],
    )

    with patch(
        "src.agents.feedback_learner.graph.build_feedback_learner_graph",
        return_value=fake_graph,
    ):
        with patch(
            "src.agents.feedback_learner.agent.build_production_feedback_stores",
            new=AsyncMock(return_value=(None, None, None)),
        ):
            response = await _execute_learning_cycle(request)

    # The count was always honest; the LIST must now match it (was [] before fix).
    assert response.updates_applied == 1
    assert len(response.applied_updates) == 1
    assert response.applied_updates[0].update_id == "U_R1"
    # The non-applied proposed update must NOT appear among applied_updates.
    assert all(u.update_id != "U_R2" for u in response.applied_updates)


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
async def test_execute_learning_cycle_threads_auto_apply_into_state():
    """The request's auto_apply flag must reach the graph's initial state.

    It was silently dropped: RunLearningRequest.auto_apply existed but
    _execute_learning_cycle never put it in initial_state, so
    KnowledgeUpdaterNode applied every update regardless of the request.
    """
    from src.api.routes.feedback import RunLearningRequest, _execute_learning_cycle

    result_state = {
        "status": "completed",
        "detected_patterns": [],
        "learning_recommendations": [],
        "priority_improvements": [],
        "proposed_updates": [],
        "applied_updates": [],
        "learning_summary": "ok",
        "collection_latency_ms": 0,
        "analysis_latency_ms": 0,
        "errors": [],
        "warnings": [],
    }

    for flag in (False, True):
        fake_graph = AsyncMock()
        fake_graph.ainvoke.return_value = result_state
        request = RunLearningRequest(auto_apply=flag)
        with patch(
            "src.agents.feedback_learner.graph.build_feedback_learner_graph",
            return_value=fake_graph,
        ):
            with patch(
                "src.agents.feedback_learner.agent.build_production_feedback_stores",
                new=AsyncMock(return_value=(None, None, None)),
            ):
                await _execute_learning_cycle(request)
        initial_state = fake_graph.ainvoke.call_args.args[0]
        assert initial_state["auto_apply"] is flag


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
async def test_apply_update_success(sample_knowledge_update, fake_knowledge_stores):
    """#1243: apply performs a REAL store write, not just a status flip."""
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
    # The recorded learning actually landed in the target store.
    store = fake_knowledge_stores["prompt"]
    assert store.values.get("causal_impact") == "Improved prompt"
    assert store.justifications.get("causal_impact") == "Better accuracy"

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_apply_update_store_write_fails_stays_proposed(
    sample_knowledge_update, fake_knowledge_stores
):
    """#1243 fail-honest: a failed store write must NOT mark the update APPLIED."""
    from src.api.routes.feedback import (
        ApplyUpdateRequest,
        UpdateStatus,
        _updates_store,
        apply_update,
    )

    fake_knowledge_stores["prompt"].fail = True
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    request = ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=False)
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await apply_update(sample_knowledge_update.update_id, request, user)

    assert exc_info.value.status_code == 502
    assert sample_knowledge_update.status == UpdateStatus.PROPOSED
    assert sample_knowledge_update.applied_at is None

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_apply_update_unmapped_component_is_422(
    sample_knowledge_update, fake_knowledge_stores
):
    """#1243: a target_component with no real store cannot be honestly applied."""
    from src.api.routes.feedback import (
        ApplyUpdateRequest,
        UpdateStatus,
        _updates_store,
        apply_update,
    )

    sample_knowledge_update.target_component = "not_a_knowledge_type"
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    request = ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=False)
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await apply_update(sample_knowledge_update.update_id, request, user)

    assert exc_info.value.status_code == 422
    assert sample_knowledge_update.status == UpdateStatus.PROPOSED

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_apply_update_stores_unavailable_is_503(sample_knowledge_update, monkeypatch):
    """#1243 fail-honest: no reachable store backend => 503, never a silent flip."""
    from src.api.routes.feedback import (
        ApplyUpdateRequest,
        UpdateStatus,
        _updates_store,
        apply_update,
    )

    async def _no_stores():
        return {}

    monkeypatch.setattr("src.api.routes.feedback._get_knowledge_stores", _no_stores)
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    request = ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=False)
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await apply_update(sample_knowledge_update.update_id, request, user)

    assert exc_info.value.status_code == 503
    assert sample_knowledge_update.status == UpdateStatus.PROPOSED

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_apply_update_captures_prior_value_for_rollback(
    sample_knowledge_update, fake_knowledge_stores
):
    """#1243: the pre-apply store value is captured so rollback can restore it."""
    from src.api.routes.feedback import (
        ApplyUpdateRequest,
        UpdateStatus,
        _updates_store,
        apply_update,
    )

    fake_knowledge_stores["prompt"].values["causal_impact"] = "The prior prompt"
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    request = ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=False)
    user = {"user_id": "test_user", "role": "operator"}

    result = await apply_update(sample_knowledge_update.update_id, request, user)

    assert result.status == UpdateStatus.APPLIED
    assert result.current_value == "The prior prompt"
    assert fake_knowledge_stores["prompt"].values["causal_impact"] == "Improved prompt"

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
async def test_apply_update_force(sample_knowledge_update, fake_knowledge_stores):
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
async def test_apply_update_force_reapply_keeps_original_prior(
    sample_knowledge_update, fake_knowledge_stores
):
    """#1243 codex M: force re-applying an already-APPLIED update must not
    re-capture 'prior' from the store — the store now holds this update's own
    proposed_value, so re-capturing stomps current_value and a later rollback
    would restore the wrong (post-apply) state instead of the true original."""
    from src.api.routes.feedback import (
        ApplyUpdateRequest,
        UpdateStatus,
        _updates_store,
        apply_update,
        rollback_update,
    )

    fake_knowledge_stores["prompt"].values["causal_impact"] = "The prior prompt"
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update
    user = {"user_id": "test_user", "role": "operator"}

    first = await apply_update(
        sample_knowledge_update.update_id,
        ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=False),
        user,
    )
    assert first.current_value == "The prior prompt"
    assert fake_knowledge_stores["prompt"].values["causal_impact"] == "Improved prompt"

    forced = await apply_update(
        sample_knowledge_update.update_id,
        ApplyUpdateRequest(update_id=sample_knowledge_update.update_id, force=True),
        user,
    )
    assert forced.status == UpdateStatus.APPLIED
    assert forced.current_value == "The prior prompt", (
        "force re-apply from APPLIED must keep the original pre-apply capture, "
        "not the update's own proposed_value"
    )

    rolled = await rollback_update(sample_knowledge_update.update_id, user)
    assert rolled.status == UpdateStatus.ROLLED_BACK
    assert fake_knowledge_stores["prompt"].values["causal_impact"] == "The prior prompt"

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_rollback_update_restores_prior_value(sample_knowledge_update, fake_knowledge_stores):
    """#1243: rollback restores the captured pre-apply value in the real store."""
    from src.api.routes.feedback import UpdateStatus, _updates_store, rollback_update

    sample_knowledge_update.status = UpdateStatus.APPLIED
    sample_knowledge_update.current_value = "The prior prompt"
    fake_knowledge_stores["prompt"].values["causal_impact"] = "Improved prompt"
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    user = {"user_id": "test_user", "role": "operator"}

    result = await rollback_update(sample_knowledge_update.update_id, user)

    assert result.status == UpdateStatus.ROLLED_BACK
    assert fake_knowledge_stores["prompt"].values["causal_impact"] == "The prior prompt"

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_rollback_first_apply_removes_recorded_row(
    sample_knowledge_update, fake_knowledge_stores
):
    """#1243: rolling back a first-ever apply (no prior value) removes the row —
    restoring the true pre-apply state (no recorded learning)."""
    from src.api.routes.feedback import UpdateStatus, _updates_store, rollback_update

    sample_knowledge_update.status = UpdateStatus.APPLIED
    sample_knowledge_update.current_value = None
    fake_knowledge_stores["prompt"].values["causal_impact"] = "Improved prompt"
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    user = {"user_id": "test_user", "role": "operator"}

    result = await rollback_update(sample_knowledge_update.update_id, user)

    assert result.status == UpdateStatus.ROLLED_BACK
    assert "causal_impact" not in fake_knowledge_stores["prompt"].values

    # Cleanup
    del _updates_store[sample_knowledge_update.update_id]


@pytest.mark.asyncio
async def test_rollback_store_failure_stays_applied(sample_knowledge_update, fake_knowledge_stores):
    """#1243 fail-honest: a failed rollback write keeps the record APPLIED."""
    from src.api.routes.feedback import UpdateStatus, _updates_store, rollback_update

    sample_knowledge_update.status = UpdateStatus.APPLIED
    sample_knowledge_update.current_value = "The prior prompt"
    fake_knowledge_stores["prompt"].fail = True
    _updates_store[sample_knowledge_update.update_id] = sample_knowledge_update

    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await rollback_update(sample_knowledge_update.update_id, user)

    assert exc_info.value.status_code == 502
    assert sample_knowledge_update.status == UpdateStatus.APPLIED

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


def test_convert_updates_applied_records_stamped_applied():
    """#1243 (PR #1241 final-review minor a): applied_updates entries from a
    learning cycle must convert with status='applied', not the 'proposed'
    default — graph-state dicts carry no status key."""
    from src.api.routes.feedback import UpdateStatus, _convert_updates

    graph_update = {
        "update_id": "U_R1",
        "knowledge_type": "prompt",
        "key": "cognitive_investigator",
        "old_value": None,
        "new_value": "Update system prompts",
        "justification": "Improve relevance",
        "effective_date": "2026-07-15T22:13:29+00:00",
    }

    proposed = _convert_updates([graph_update])
    assert proposed[0].status == UpdateStatus.PROPOSED
    assert proposed[0].applied_at is None

    applied = _convert_updates([graph_update], applied=True)
    assert applied[0].status == UpdateStatus.APPLIED
    assert applied[0].applied_at is not None


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


# --- #1661: the optimizer gate must be reported here ------------------------
#
# The daily prompt-optimization beat skips at its trigger every time and
# returns a legitimate ``{"status": "skipped"}``, so nothing fails and nothing
# alerts. This page's health poll is the surface an operator actually watches;
# it must carry the gate's own numbers or the inertness stays invisible.


@pytest.fixture
def _stub_async_supabase(monkeypatch):
    """Pin the async client factory the health route resolves before reading.

    The suite's autouse guard only disarms the SYNC ``get_supabase_client``. The
    async factory RAISES ``ServiceConnectionError`` when SUPABASE_URL is unset,
    which is CI's state — the route would then degrade before the stubbed status
    reader ran, and these assertions would be graded against the fallback. This
    box hides that because pytest autoloads a real ``.env``.
    """

    async def _client():
        return object()

    monkeypatch.setattr("src.memory.services.factories.get_async_supabase_client", _client)


@pytest.mark.asyncio
async def test_health_carries_the_optimizer_gate_status(monkeypatch, _stub_async_supabase):
    from src.api.routes import feedback as feedback_routes

    async def _status(client=None):
        return {
            "eligible_signals": 8,
            "total_signals": 218,
            "last_eligible_signal_at": "2026-08-08T07:09:02.686027+00:00",
            "optimization_runs": 0,
            "min_signals": 20,
            "min_reward": 0.5,
            "would_trigger": False,
            "reason": "Optimizer inert: 8 of 218 ...",
        }

    monkeypatch.setattr(
        "src.agents.feedback_learner.signal_store.get_optimizer_gate_status", _status
    )
    with patch("src.agents.feedback_learner.FeedbackLearnerAgent"):
        result = await feedback_routes.get_feedback_health()

    assert result.optimizer is not None
    assert result.optimizer.eligible_signals == 8
    assert result.optimizer.total_signals == 218
    assert result.optimizer.min_signals == 20
    assert result.optimizer.optimization_runs == 0
    assert result.optimizer.would_trigger is False
    assert result.optimizer.reason.startswith("Optimizer inert")


@pytest.mark.asyncio
async def test_health_survives_an_unreadable_optimizer_gate(monkeypatch, _stub_async_supabase):
    """A failed gate read must degrade the block, never 500 the health check."""
    from src.api.routes import feedback as feedback_routes

    async def _boom(client=None):
        raise RuntimeError("db down")

    monkeypatch.setattr("src.agents.feedback_learner.signal_store.get_optimizer_gate_status", _boom)
    with patch("src.agents.feedback_learner.FeedbackLearnerAgent"):
        result = await feedback_routes.get_feedback_health()

    assert result.status in ["healthy", "degraded"]
    # Unknown, never a fabricated zero that reads as a measurement.
    assert result.optimizer is not None
    assert result.optimizer.eligible_signals is None
    assert result.optimizer.would_trigger is None


@pytest.mark.asyncio
async def test_health_degrades_when_the_async_client_cannot_be_built(monkeypatch):
    """CI's real shape: no SUPABASE_URL, so the client factory raises."""
    from src.api.routes import feedback as feedback_routes

    async def _raise():
        raise RuntimeError("SUPABASE_URL not configured")

    monkeypatch.setattr("src.memory.services.factories.get_async_supabase_client", _raise)
    with patch("src.agents.feedback_learner.FeedbackLearnerAgent"):
        result = await feedback_routes.get_feedback_health()

    assert result.optimizer is not None
    assert result.optimizer.eligible_signals is None
    assert result.optimizer.would_trigger is None
    assert "unavailable" in result.optimizer.reason.lower()
    # The threshold is still reported — it comes from config, not the DB.
    assert result.optimizer.min_signals == 20


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


def test_convert_patterns_stamps_detected_at():
    """#1256: agent output carries no timestamp — _convert_patterns must stamp
    detection time so the persisted payload owns it. Without the stamp every
    payload fell through to the persistence row's created_at, which upserts
    never refresh: a recycled pattern_id served the FIRST cycle's timestamp."""
    from datetime import datetime, timedelta, timezone

    from src.api.routes.feedback import _convert_patterns

    pattern = {
        "pattern_id": "P1-abc12345",
        "pattern_type": "accuracy_issue",
        "description": "Test",
        "frequency": 2,
        "severity": "medium",
        "affected_agents": ["copilotkit"],
        "example_feedback_ids": [],
        "root_cause_hypothesis": "Test",
    }

    before = datetime.now(timezone.utc)
    result = _convert_patterns([dict(pattern)])
    after = datetime.now(timezone.utc)

    assert len(result) == 1
    assert result[0].detected_at is not None
    assert before - timedelta(seconds=1) <= result[0].detected_at <= after

    # an explicit detected_at (e.g. replayed output) is preserved, not restamped
    stamped = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    kept = _convert_patterns([{**pattern, "detected_at": stamped}])
    assert kept[0].detected_at == stamped


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


# =============================================================================
# persist_learning_cycle_output (shared by the 6h Celery beat)
# =============================================================================


@pytest.mark.asyncio
async def test_persist_learning_cycle_output_converts_and_persists():
    """The beat's persistence entry point must convert the agent output to a
    LearningResponse (incl. the applied_updates rehydration from string IDs)
    and persist BOTH the batch and its artifacts — this is what fills the
    tables the /feedback-learning page reads."""
    from types import SimpleNamespace

    from src.api.routes import feedback as feedback_mod

    update = {
        "update_id": "U1",
        "update_type": "prompt_refinement",
        "status": "proposed",
        "target_agent": "gap_analyzer",
        "target_component": "prompt",
        "proposed_value": "New value",
        "rationale": "Better",
        "expected_improvement": "10%",
    }
    output = SimpleNamespace(
        status="completed",
        detected_patterns=[
            {
                "pattern_id": "P1",
                "pattern_type": "accuracy_issue",
                "description": "low reward on gap_analyzer",
                "frequency": 3,
                "severity": "high",
                "affected_agents": ["gap_analyzer"],
                "example_feedback_ids": [],
                "root_cause_hypothesis": "",
                "confidence": 0.8,
            }
        ],
        learning_recommendations=[],
        priority_improvements=["fix X"],
        proposed_updates=[update, {**update, "update_id": "U2"}],
        applied_updates=["U1"],
        learning_summary="cycle summary",
        total_latency_ms=123,
        errors=[],
        warnings=["No feedback items collected"],
    )

    with (
        patch.object(feedback_mod, "_persist_batch", new_callable=AsyncMock) as mock_batch,
        patch.object(
            feedback_mod, "_persist_cycle_artifacts", new_callable=AsyncMock
        ) as mock_artifacts,
    ):
        resp = await feedback_mod.persist_learning_cycle_output(output, "beat_task12345")

    assert resp.batch_id == "beat_task12345"
    assert resp.status.value == "completed"
    assert resp.patterns_detected == 1
    assert resp.detected_patterns[0].pattern_id == "P1"
    assert resp.updates_proposed == 2
    # applied_updates rehydrated from the string IDs (mirrors _execute_learning_cycle)
    assert resp.updates_applied == 1
    assert [u.update_id for u in resp.applied_updates] == ["U1"]
    assert resp.warnings == ["No feedback items collected"]
    mock_artifacts.assert_awaited_once_with(resp)
    mock_batch.assert_awaited_once_with(resp)


@pytest.mark.asyncio
async def test_persist_learning_cycle_output_failed_status():
    """Non-completed agent status maps to FAILED, and dict errors are stringified
    (LearningResponse.errors is List[str])."""
    from types import SimpleNamespace

    from src.api.routes import feedback as feedback_mod

    output = SimpleNamespace(
        status="failed",
        detected_patterns=[],
        learning_recommendations=[],
        priority_improvements=[],
        proposed_updates=[],
        applied_updates=[],
        learning_summary="",
        total_latency_ms=5,
        errors=[{"node": "feedback_collector", "error": "boom"}],
        warnings=[],
    )
    with (
        patch.object(feedback_mod, "_persist_batch", new_callable=AsyncMock),
        patch.object(feedback_mod, "_persist_cycle_artifacts", new_callable=AsyncMock),
    ):
        resp = await feedback_mod.persist_learning_cycle_output(output, "beat_x")

    assert resp.status.value == "failed"
    assert resp.errors and isinstance(resp.errors[0], str)


def test_convert_updates_maps_graph_state_keys():
    """_convert_updates must read the node's KnowledgeUpdate TypedDict keys.

    KnowledgeUpdaterNode emits knowledge_type/key/old_value/new_value/
    justification (src/agents/feedback_learner/state.py). The converter read
    only API-style keys, so every real proposed update rendered as a
    contentless card with a fabricated default type on the Updates tab.
    """
    from src.api.routes.feedback import UpdateType, _convert_updates

    node_update = {
        "update_id": "U_R1",
        "knowledge_type": "threshold",
        "key": "gap_analyzer",
        "old_value": 0.5,
        "new_value": "0.2",
        "justification": "drift detected in weekly ratings",
        "effective_date": "2026-07-15T00:00:00+00:00",
    }
    converted = _convert_updates([node_update])
    assert len(converted) == 1
    u = converted[0]
    assert u.update_id == "U_R1"
    assert u.update_type == UpdateType.PARAMETER_TUNING
    assert u.target_agent == "gap_analyzer"
    assert u.target_component == "threshold"
    assert u.current_value == "0.5"
    assert u.proposed_value == "0.2"
    assert u.rationale == "drift detected in weekly ratings"


def test_convert_updates_still_accepts_api_style_dicts():
    """API-style dicts (explicit update_type/target_agent/...) keep working."""
    from src.api.routes.feedback import UpdateType, _convert_updates

    api_update = {
        "update_id": "upd_1",
        "update_type": "prompt_refinement",
        "target_agent": "causal_impact",
        "proposed_value": "new prompt",
        "rationale": "clarity",
    }
    converted = _convert_updates([api_update])
    assert len(converted) == 1
    u = converted[0]
    assert u.update_type == UpdateType.PROMPT_REFINEMENT
    assert u.target_agent == "causal_impact"
    assert u.proposed_value == "new prompt"
    assert u.rationale == "clarity"
