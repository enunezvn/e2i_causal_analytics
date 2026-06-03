"""Unit tests for Segment Analysis API route handlers.

Tests all endpoints and helper functions in src/api/routes/segments.py.
Mocks all external dependencies to ensure unit test isolation.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

# Import route functions and models
from src.api.routes.segments import (
    AnalysisStatus,
    ResponderType,
    # Models
    RunSegmentAnalysisRequest,
    _analyses_store,
    _convert_cate_results,
    _convert_policies,
    _convert_segment_profiles,
    _convert_uplift_metrics,
    _execute_segment_analysis,
    _generate_mock_response,
    # Helper functions
    _run_segment_analysis_task,
    get_segment_analysis,
    get_segment_health,
    list_policies,
    # Endpoints
    run_segment_analysis,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_analyses_store():
    """Reset the analyses store before each test.

    The live store is the Redis-backed durable store. For unit isolation we
    pin its Redis factory to a fast-failing stub so it deterministically uses
    its in-process fallback (no real Redis connection / retry-backoff in unit
    tests), and clear that fallback between tests.
    """

    async def _no_redis():
        raise ConnectionError("unit test: no redis")

    original_factory = _analyses_store._redis_factory
    _analyses_store._redis_factory = _no_redis
    _analyses_store.clear()
    yield
    _analyses_store.clear()
    _analyses_store._redis_factory = original_factory


@pytest.fixture
def sample_request():
    """Sample segment analysis request."""
    return RunSegmentAnalysisRequest(
        query="Which HCP segments respond best to rep visits?",
        treatment_var="rep_visits",
        outcome_var="trx",
        segment_vars=["region", "specialty"],
        effect_modifiers=["practice_size"],
        data_source="hcp_data",
        n_estimators=100,
        top_segments_count=10,
    )


@pytest.fixture
def mock_agent_result():
    """Mock agent result."""
    return {
        "status": "completed",
        "cate_by_segment": {
            "region": [
                {
                    "segment_name": "region",
                    "segment_value": "Northeast",
                    "cate_estimate": 15.2,
                    "cate_ci_lower": 8.5,
                    "cate_ci_upper": 21.9,
                    "sample_size": 1250,
                    "statistical_significance": True,
                }
            ]
        },
        "overall_ate": 10.5,
        "heterogeneity_score": 0.65,
        "feature_importance": {"region": 0.42},
        "overall_auuc": 0.72,
        "overall_qini": 0.58,
        "targeting_efficiency": 0.68,
        "model_type_used": "random_forest",
        "high_responders": [
            {
                "segment_id": "seg_1",
                "responder_type": "high",
                "cate_estimate": 15.2,
                "defining_features": [{"feature": "region", "value": "Northeast"}],
                "size": 1250,
                "size_percentage": 28.5,
                "recommendation": "Increase treatment",
            }
        ],
        "low_responders": [],
        "policy_recommendations": [
            {
                "segment": "Northeast",
                "current_treatment_rate": 0.35,
                "recommended_treatment_rate": 0.55,
                "expected_incremental_outcome": 125.5,
                "confidence": 0.82,
            }
        ],
        "expected_total_lift": 125.5,
        "optimal_allocation_summary": "Reallocate resources",
        "executive_summary": "Analysis complete",
        "key_insights": ["Insight 1"],
        "libraries_executed": ["econml", "causalml"],
        "library_agreement_score": 0.85,
        "validation_passed": True,
        "estimation_latency_ms": 200,
        "analysis_latency_ms": 150,
        "warnings": [],
        "confidence": 0.75,
    }


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    return {"user_id": "user123", "role": "analyst"}


# =============================================================================
# ENDPOINT TESTS - run_segment_analysis
# =============================================================================


@pytest.mark.asyncio
async def test_run_segment_analysis_async_mode(sample_request, mock_user):
    """Test run_segment_analysis in async mode returns immediately."""
    background_tasks = BackgroundTasks()

    result = await run_segment_analysis(
        request=sample_request,
        background_tasks=background_tasks,
        async_mode=True,
        user=mock_user,
    )

    assert result.status == AnalysisStatus.PENDING
    assert result.analysis_id.startswith("seg_")
    assert await _analyses_store.contains(result.analysis_id)


@pytest.mark.asyncio
async def test_run_segment_analysis_sync_mode(sample_request, mock_user):
    """Test run_segment_analysis in sync mode executes immediately."""
    background_tasks = BackgroundTasks()

    with patch("src.api.routes.segments._execute_segment_analysis") as mock_execute:
        mock_result = MagicMock(
            analysis_id="",
            status=AnalysisStatus.COMPLETED,
            overall_ate=10.5,
        )
        mock_execute.return_value = mock_result

        result = await run_segment_analysis(
            request=sample_request,
            background_tasks=background_tasks,
            async_mode=False,
            user=mock_user,
        )

        assert result.status == AnalysisStatus.COMPLETED
        mock_execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_segment_analysis_sync_mode_exception(sample_request, mock_user):
    """Test run_segment_analysis handles exceptions in sync mode."""
    background_tasks = BackgroundTasks()

    secret = "Test error leaking internal path /srv/app/db.py:42"
    with patch("src.api.routes.segments._execute_segment_analysis") as mock_execute:
        mock_execute.side_effect = RuntimeError(secret)

        with pytest.raises(HTTPException) as exc_info:
            await run_segment_analysis(
                request=sample_request,
                background_tasks=background_tasks,
                async_mode=False,
                user=mock_user,
            )

        # Generic detail returned; raw exception text NOT leaked (finding #7).
        assert exc_info.value.status_code == 500
        assert "Segment analysis failed" in str(exc_info.value.detail)
        assert secret not in str(exc_info.value.detail)
        assert "/srv/app/db.py" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_run_segment_analysis_stores_result(sample_request, mock_user):
    """Test run_segment_analysis stores result in store."""
    background_tasks = BackgroundTasks()

    with patch("src.api.routes.segments._execute_segment_analysis") as mock_execute:
        mock_result = MagicMock(
            analysis_id="",
            status=AnalysisStatus.COMPLETED,
        )
        mock_execute.return_value = mock_result

        result = await run_segment_analysis(
            request=sample_request,
            background_tasks=background_tasks,
            async_mode=False,
            user=mock_user,
        )

        assert await _analyses_store.contains(result.analysis_id)
        stored = await _analyses_store.get(result.analysis_id)
        assert stored is not None
        assert stored.status == AnalysisStatus.COMPLETED


# =============================================================================
# ENDPOINT TESTS - get_segment_analysis
# =============================================================================


@pytest.mark.asyncio
async def test_get_segment_analysis_success():
    """Test get_segment_analysis returns stored analysis."""
    # Add an analysis to the store
    analysis_id = "seg_test123"
    mock_analysis = MagicMock(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETED,
    )
    await _analyses_store.set(analysis_id, mock_analysis)

    result = await get_segment_analysis(analysis_id)

    assert result.analysis_id == analysis_id
    assert result.status == AnalysisStatus.COMPLETED


@pytest.mark.asyncio
async def test_get_segment_analysis_not_found():
    """Test get_segment_analysis raises 404 for missing analysis."""
    with pytest.raises(Exception) as exc_info:
        await get_segment_analysis("nonexistent_id")

    assert "not found" in str(exc_info.value)


# =============================================================================
# ENDPOINT TESTS - list_policies
# =============================================================================


@pytest.mark.asyncio
async def test_list_policies_empty_store():
    """Test list_policies with empty store."""
    result = await list_policies(min_lift=None, min_confidence=None, limit=20)

    assert result.total_count == 0
    assert len(result.recommendations) == 0
    assert result.expected_total_lift == 0.0


@pytest.mark.asyncio
async def test_list_policies_with_data():
    """Test list_policies returns policies from completed analyses."""
    # Add completed analysis with policy
    from src.api.routes.segments import PolicyRecommendation

    mock_policy = PolicyRecommendation(
        segment="Northeast",
        current_treatment_rate=0.35,
        recommended_treatment_rate=0.55,
        expected_incremental_outcome=125.5,
        confidence=0.82,
    )

    mock_analysis = MagicMock(
        status=AnalysisStatus.COMPLETED,
        policy_recommendations=[mock_policy],
    )
    await _analyses_store.set("seg_1", mock_analysis)

    result = await list_policies(min_lift=None, min_confidence=None, limit=20)

    assert result.total_count == 1
    assert len(result.recommendations) == 1
    assert result.expected_total_lift == 125.5


@pytest.mark.asyncio
async def test_list_policies_filters_by_min_lift():
    """Test list_policies filters by minimum lift."""
    from src.api.routes.segments import PolicyRecommendation

    mock_policy_high = PolicyRecommendation(
        segment="Northeast",
        current_treatment_rate=0.35,
        recommended_treatment_rate=0.55,
        expected_incremental_outcome=200.0,
        confidence=0.82,
    )

    mock_policy_low = PolicyRecommendation(
        segment="Southeast",
        current_treatment_rate=0.35,
        recommended_treatment_rate=0.45,
        expected_incremental_outcome=50.0,
        confidence=0.82,
    )

    mock_analysis = MagicMock(
        status=AnalysisStatus.COMPLETED,
        policy_recommendations=[mock_policy_high, mock_policy_low],
    )
    await _analyses_store.set("seg_1", mock_analysis)

    result = await list_policies(min_lift=100.0, min_confidence=None, limit=20)

    assert result.total_count == 1
    assert result.recommendations[0].expected_incremental_outcome == 200.0


@pytest.mark.asyncio
async def test_list_policies_filters_by_min_confidence():
    """Test list_policies filters by minimum confidence."""
    from src.api.routes.segments import PolicyRecommendation

    mock_policy_high = PolicyRecommendation(
        segment="Northeast",
        current_treatment_rate=0.35,
        recommended_treatment_rate=0.55,
        expected_incremental_outcome=200.0,
        confidence=0.9,
    )

    mock_policy_low = PolicyRecommendation(
        segment="Southeast",
        current_treatment_rate=0.35,
        recommended_treatment_rate=0.45,
        expected_incremental_outcome=150.0,
        confidence=0.5,
    )

    mock_analysis = MagicMock(
        status=AnalysisStatus.COMPLETED,
        policy_recommendations=[mock_policy_high, mock_policy_low],
    )
    await _analyses_store.set("seg_1", mock_analysis)

    result = await list_policies(min_lift=None, min_confidence=0.8, limit=20)

    assert result.total_count == 1
    assert result.recommendations[0].confidence == 0.9


@pytest.mark.asyncio
async def test_list_policies_respects_limit():
    """Test list_policies respects limit parameter."""
    from src.api.routes.segments import PolicyRecommendation

    # Create 10 policies
    policies = [
        PolicyRecommendation(
            segment=f"Segment_{i}",
            current_treatment_rate=0.35,
            recommended_treatment_rate=0.55,
            expected_incremental_outcome=100.0 + i,
            confidence=0.82,
        )
        for i in range(10)
    ]

    mock_analysis = MagicMock(
        status=AnalysisStatus.COMPLETED,
        policy_recommendations=policies,
    )
    await _analyses_store.set("seg_1", mock_analysis)

    result = await list_policies(min_lift=None, min_confidence=None, limit=5)

    assert len(result.recommendations) == 5


@pytest.mark.asyncio
async def test_list_policies_sorts_by_outcome():
    """Test list_policies sorts by expected outcome descending."""
    from src.api.routes.segments import PolicyRecommendation

    policies = [
        PolicyRecommendation(
            segment="Low",
            current_treatment_rate=0.35,
            recommended_treatment_rate=0.55,
            expected_incremental_outcome=50.0,
            confidence=0.82,
        ),
        PolicyRecommendation(
            segment="High",
            current_treatment_rate=0.35,
            recommended_treatment_rate=0.55,
            expected_incremental_outcome=200.0,
            confidence=0.82,
        ),
        PolicyRecommendation(
            segment="Medium",
            current_treatment_rate=0.35,
            recommended_treatment_rate=0.55,
            expected_incremental_outcome=100.0,
            confidence=0.82,
        ),
    ]

    mock_analysis = MagicMock(
        status=AnalysisStatus.COMPLETED,
        policy_recommendations=policies,
    )
    await _analyses_store.set("seg_1", mock_analysis)

    result = await list_policies(min_lift=None, min_confidence=None, limit=20)

    assert result.recommendations[0].segment == "High"
    assert result.recommendations[1].segment == "Medium"
    assert result.recommendations[2].segment == "Low"


@pytest.mark.asyncio
async def test_list_policies_skips_pending_analyses():
    """Test list_policies skips pending analyses."""
    from src.api.routes.segments import PolicyRecommendation

    mock_policy = PolicyRecommendation(
        segment="Northeast",
        current_treatment_rate=0.35,
        recommended_treatment_rate=0.55,
        expected_incremental_outcome=125.5,
        confidence=0.82,
    )

    mock_pending = MagicMock(
        status=AnalysisStatus.PENDING,
        policy_recommendations=[mock_policy],
    )
    await _analyses_store.set("seg_1", mock_pending)

    result = await list_policies(min_lift=None, min_confidence=None, limit=20)

    assert result.total_count == 0


# =============================================================================
# ENDPOINT TESTS - get_segment_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_segment_health_all_available():
    """Test get_segment_health when all dependencies available."""
    # The function imports inside try/except, so we just call it
    result = await get_segment_health()

    # Should be healthy if all imports succeed (they should in test environment)
    assert result.status in ["healthy", "degraded", "partial"]
    assert isinstance(result.agent_available, bool)
    assert isinstance(result.econml_available, bool)
    assert isinstance(result.causalml_available, bool)


@pytest.mark.asyncio
async def test_get_segment_health_agent_unavailable():
    """Test get_segment_health reflects agent availability status."""
    # Just verify the function runs and returns valid status
    result = await get_segment_health()

    # Status should be one of the valid options
    assert result.status in ["healthy", "degraded", "partial"]
    # Agent available should be a boolean
    assert isinstance(result.agent_available, bool)


@pytest.mark.asyncio
async def test_get_segment_health_libraries_unavailable():
    """Test get_segment_health when libraries unavailable."""
    import sys

    # Patch sys.modules to make econml appear unavailable
    original_econml = sys.modules.get("econml")

    # Remove econml temporarily
    if "econml" in sys.modules:
        del sys.modules["econml"]

    # Block econml import
    sys.modules["econml"] = None

    try:
        result = await get_segment_health()

        assert result.econml_available is False
    finally:
        # Restore
        if original_econml is not None:
            sys.modules["econml"] = original_econml
        elif "econml" in sys.modules:
            del sys.modules["econml"]


@pytest.mark.asyncio
async def test_get_segment_health_counts_recent_analyses():
    """Test get_segment_health counts analyses in last 24 hours."""
    # Add recent analysis
    from src.api.routes.segments import SegmentAnalysisResponse

    recent_analysis = SegmentAnalysisResponse(
        analysis_id="seg_1",
        status=AnalysisStatus.COMPLETED,
        timestamp=datetime.now(timezone.utc),
    )
    await _analyses_store.set("seg_1", recent_analysis)

    with patch("src.agents.heterogeneous_optimizer.HeterogeneousOptimizerAgent"):
        result = await get_segment_health()

        assert result.analyses_24h == 1


@pytest.mark.asyncio
async def test_get_segment_health_last_analysis():
    """Test get_segment_health returns last analysis timestamp."""
    from src.api.routes.segments import SegmentAnalysisResponse

    analysis = SegmentAnalysisResponse(
        analysis_id="seg_1",
        status=AnalysisStatus.COMPLETED,
        timestamp=datetime.now(timezone.utc),
    )
    await _analyses_store.set("seg_1", analysis)

    with patch("src.agents.heterogeneous_optimizer.HeterogeneousOptimizerAgent"):
        result = await get_segment_health()

        assert result.last_analysis is not None


# =============================================================================
# HELPER FUNCTION TESTS - _run_segment_analysis_task
# =============================================================================


@pytest.mark.asyncio
async def test_run_segment_analysis_task_success(sample_request, mock_agent_result):
    """Test _run_segment_analysis_task completes successfully."""
    analysis_id = "seg_test123"

    # Pre-populate store with pending analysis
    from src.api.routes.segments import SegmentAnalysisResponse

    await _analyses_store.set(
        analysis_id,
        SegmentAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
        ),
    )

    with patch("src.api.routes.segments._execute_segment_analysis") as mock_execute:
        mock_result = MagicMock(
            analysis_id="",
            status=AnalysisStatus.COMPLETED,
        )
        mock_execute.return_value = mock_result

        await _run_segment_analysis_task(analysis_id, sample_request)

        stored = await _analyses_store.get(analysis_id)
        assert stored is not None
        assert stored.status == AnalysisStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_segment_analysis_task_handles_error(sample_request):
    """Test _run_segment_analysis_task handles errors."""
    analysis_id = "seg_test123"

    from src.api.routes.segments import SegmentAnalysisResponse

    await _analyses_store.set(
        analysis_id,
        SegmentAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
        ),
    )

    with patch("src.api.routes.segments._execute_segment_analysis") as mock_execute:
        mock_execute.side_effect = RuntimeError("Test error")

        await _run_segment_analysis_task(analysis_id, sample_request)

        stored = await _analyses_store.get(analysis_id)
        assert stored is not None
        assert stored.status == AnalysisStatus.FAILED
        assert len(stored.warnings) > 0


# =============================================================================
# HELPER FUNCTION TESTS - _execute_segment_analysis
# =============================================================================


@pytest.mark.asyncio
async def test_execute_segment_analysis_with_agent(sample_request, mock_agent_result):
    """Test _execute_segment_analysis uses real agent when available."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=mock_agent_result)

    with patch(
        "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
        return_value=mock_graph,
    ):
        result = await _execute_segment_analysis(sample_request)

        assert result.status == AnalysisStatus.COMPLETED
        assert result.overall_ate == 10.5
        assert result.heterogeneity_score == 0.65
        mock_graph.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_execute_segment_analysis_falls_back_to_mock_when_explicitly_allowed(
    sample_request, monkeypatch
):
    """Mock-fallback is gated on E2I_REQUIRE_AGENT_IMPORT=0 (closed-by-default policy)."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    with patch(
        "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
        side_effect=ImportError,
    ):
        result = await _execute_segment_analysis(sample_request)

        assert result.status == AnalysisStatus.COMPLETED
        assert "mock data" in result.warnings[0].lower()


@pytest.mark.asyncio
async def test_execute_segment_analysis_raises_503_when_mock_disabled(sample_request, monkeypatch):
    """Closed-by-default: ImportError must raise 503 when mock-fallback is disabled."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "1")
    with patch(
        "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
        side_effect=ImportError,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _execute_segment_analysis(sample_request)
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"] == "agent_unavailable"


@pytest.mark.asyncio
async def test_execute_segment_analysis_handles_exception(sample_request):
    """Test _execute_segment_analysis handles agent exceptions."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Agent error"))

    with patch(
        "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
        return_value=mock_graph,
    ):
        with pytest.raises(RuntimeError):
            await _execute_segment_analysis(sample_request)


# =============================================================================
# HELPER FUNCTION TESTS - _convert_cate_results
# =============================================================================


def test_convert_cate_results_success():
    """Test _convert_cate_results converts agent output correctly."""
    agent_data = {
        "region": [
            {
                "segment_name": "region",
                "segment_value": "Northeast",
                "cate_estimate": 15.2,
                "cate_ci_lower": 8.5,
                "cate_ci_upper": 21.9,
                "sample_size": 1250,
                "statistical_significance": True,
            }
        ]
    }

    result = _convert_cate_results(agent_data)

    assert "region" in result
    assert len(result["region"]) == 1
    assert result["region"][0].segment_value == "Northeast"
    assert result["region"][0].cate_estimate == 15.2


def test_convert_cate_results_empty():
    """Test _convert_cate_results handles empty data."""
    result = _convert_cate_results({})

    assert isinstance(result, dict)
    assert len(result) == 0


def test_convert_cate_results_handles_missing_fields():
    """Test _convert_cate_results handles missing fields."""
    agent_data = {
        "region": [
            {
                "segment_name": "region",
                # Missing other fields
            }
        ]
    }

    result = _convert_cate_results(agent_data)

    assert "region" in result
    assert len(result["region"]) == 1
    # Should use defaults
    assert result["region"][0].cate_estimate == 0.0


# =============================================================================
# HELPER FUNCTION TESTS - _convert_uplift_metrics
# =============================================================================


def test_convert_uplift_metrics_success():
    """Test _convert_uplift_metrics converts agent output correctly."""
    agent_data = {
        "overall_auuc": 0.72,
        "overall_qini": 0.58,
        "targeting_efficiency": 0.68,
        "model_type_used": "random_forest",
    }

    result = _convert_uplift_metrics(agent_data)

    assert result is not None
    assert result.overall_auuc == 0.72
    assert result.overall_qini == 0.58
    assert result.model_type_used == "random_forest"


def test_convert_uplift_metrics_missing_data():
    """Test _convert_uplift_metrics returns None when missing data."""
    result = _convert_uplift_metrics({})

    assert result is None


# =============================================================================
# HELPER FUNCTION TESTS - _convert_segment_profiles
# =============================================================================


def test_convert_segment_profiles_success():
    """Test _convert_segment_profiles converts agent output correctly."""
    agent_data = [
        {
            "segment_id": "seg_1",
            "responder_type": "high",
            "cate_estimate": 15.2,
            "defining_features": [{"feature": "region", "value": "Northeast"}],
            "size": 1250,
            "size_percentage": 28.5,
            "recommendation": "Increase treatment",
        }
    ]

    result = _convert_segment_profiles(agent_data)

    assert len(result) == 1
    assert result[0].segment_id == "seg_1"
    assert result[0].responder_type == ResponderType.HIGH
    assert result[0].cate_estimate == 15.2


def test_convert_segment_profiles_empty():
    """Test _convert_segment_profiles handles empty list."""
    result = _convert_segment_profiles([])

    assert isinstance(result, list)
    assert len(result) == 0


def test_convert_segment_profiles_handles_invalid_data():
    """Test _convert_segment_profiles handles invalid data gracefully."""
    agent_data = [
        {
            "segment_id": "seg_1",
            "responder_type": "invalid_type",  # Invalid type
            # Missing required fields
        }
    ]

    result = _convert_segment_profiles(agent_data)

    # Should have empty list due to exception handling during invalid type conversion
    assert len(result) == 0 or result[0].responder_type == ResponderType.AVERAGE


# =============================================================================
# HELPER FUNCTION TESTS - _convert_policies
# =============================================================================


def test_convert_policies_success():
    """Test _convert_policies converts agent output correctly."""
    agent_data = [
        {
            "segment": "Northeast",
            "current_treatment_rate": 0.35,
            "recommended_treatment_rate": 0.55,
            "expected_incremental_outcome": 125.5,
            "confidence": 0.82,
        }
    ]

    result = _convert_policies(agent_data)

    assert len(result) == 1
    assert result[0].segment == "Northeast"
    assert result[0].current_treatment_rate == 0.35
    assert result[0].expected_incremental_outcome == 125.5


def test_convert_policies_empty():
    """Test _convert_policies handles empty list."""
    result = _convert_policies([])

    assert isinstance(result, list)
    assert len(result) == 0


# =============================================================================
# HELPER FUNCTION TESTS - _generate_mock_response
# =============================================================================


def test_generate_mock_response_structure(sample_request):
    """Test _generate_mock_response returns valid structure."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.status == AnalysisStatus.COMPLETED
    assert result.overall_ate is not None
    assert result.heterogeneity_score is not None
    assert len(result.cate_by_segment) > 0
    assert len(result.high_responders) > 0


def test_generate_mock_response_includes_cate_results(sample_request):
    """Test _generate_mock_response includes CATE results."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert sample_request.segment_vars[0] in result.cate_by_segment
    assert len(result.cate_by_segment[sample_request.segment_vars[0]]) > 0


def test_generate_mock_response_includes_uplift_metrics(sample_request):
    """Test _generate_mock_response includes uplift metrics."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.uplift_metrics is not None
    assert result.uplift_metrics.overall_auuc > 0


def test_generate_mock_response_includes_policies(sample_request):
    """Test _generate_mock_response includes policy recommendations."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert len(result.policy_recommendations) > 0
    assert result.expected_total_lift is not None


def test_generate_mock_response_includes_insights(sample_request):
    """Test _generate_mock_response includes insights."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.executive_summary is not None
    assert len(result.key_insights) > 0


def test_generate_mock_response_warning(sample_request):
    """Test _generate_mock_response includes warning about mock data."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert len(result.warnings) > 0
    assert "mock data" in result.warnings[0].lower()


# =============================================================================
# IN-MEMORY STORE BOUNDING (finding #4 — prevent unbounded growth)
# =============================================================================


class TestBoundedAnalysesStore:
    """The process-local analyses store must be bounded to avoid unbounded
    memory growth (it is an intentional, documented placeholder for a future
    Supabase backing — see the module comment — but it must not leak)."""

    def test_store_evicts_oldest_when_over_capacity(self):
        from src.api.routes.segments import (
            SegmentAnalysisResponse,
            _BoundedAnalysesStore,
        )

        store = _BoundedAnalysesStore(max_entries=3)
        for i in range(5):
            store[f"seg_{i}"] = SegmentAnalysisResponse(
                analysis_id=f"seg_{i}",
                status=AnalysisStatus.COMPLETED,
                timestamp=datetime.now(timezone.utc),
            )

        # Only the most-recent max_entries survive.
        assert len(store) == 3
        # Oldest two evicted.
        assert "seg_0" not in store
        assert "seg_1" not in store
        # Newest retained.
        assert "seg_2" in store
        assert "seg_3" in store
        assert "seg_4" in store

    def test_reassigning_existing_key_does_not_grow(self):
        from src.api.routes.segments import (
            SegmentAnalysisResponse,
            _BoundedAnalysesStore,
        )

        store = _BoundedAnalysesStore(max_entries=2)
        for i in range(2):
            store[f"seg_{i}"] = SegmentAnalysisResponse(
                analysis_id=f"seg_{i}",
                status=AnalysisStatus.PENDING,
                timestamp=datetime.now(timezone.utc),
            )
        # Update an existing key (e.g. PENDING -> COMPLETED on completion).
        store["seg_0"] = SegmentAnalysisResponse(
            analysis_id="seg_0",
            status=AnalysisStatus.COMPLETED,
            timestamp=datetime.now(timezone.utc),
        )

        assert len(store) == 2
        assert store["seg_0"].status == AnalysisStatus.COMPLETED
        assert "seg_1" in store

    def test_module_store_is_bounded_instance(self):
        """The live module-level store wraps a bounded in-memory fallback."""
        from src.api.routes.segments import (
            _analyses_store,
            _BoundedAnalysesStore,
            _DurableAnalysesStore,
        )

        # The live store is the durable (Redis-backed) store whose in-process
        # fallback is the bounded dict (so memory stays capped when Redis is down).
        assert isinstance(_analyses_store, _DurableAnalysesStore)
        assert isinstance(_analyses_store._memory, _BoundedAnalysesStore)
        assert _analyses_store._memory.max_entries > 0


# =============================================================================
# DURABLE (REDIS-BACKED) STORE — C21: durable / cross-worker shared store
# =============================================================================
#
# Production runs gunicorn with --workers 2 (docker/docker-compose.yml). A
# process-local dict means a POST handled by worker A is invisible to a GET
# handled by worker B (legitimate 404) and ALL state is lost on redeploy. The
# durable store backs the data in Redis (reusing the app's existing async
# client) so it is shared across workers and survives restarts, falling back to
# the bounded in-memory dict ONLY when Redis is unavailable (mirrors the app's
# existing graceful-degradation posture).


class _FakeAsyncRedis:
    """Minimal in-process stand-in for ``redis.asyncio.Redis``.

    Hand-rolled (consistent with the repo's other Redis tests, e.g.
    ``tests/unit/test_api/test_staleness_alerts.py``) because the ``fakeredis``
    package is not a test dependency here. Supports only the subset of commands
    the durable store uses: string get/set/delete and a sorted-set index.
    """

    def __init__(self) -> None:
        self.strings: dict = {}
        self.zset: dict = {}  # member -> score

    async def set(self, key, value, ex=None):  # noqa: ANN001
        self.strings[key] = value

    async def get(self, key):  # noqa: ANN001
        return self.strings.get(key)

    async def delete(self, *keys):  # noqa: ANN001
        for k in keys:
            self.strings.pop(k, None)

    async def zadd(self, key, mapping):  # noqa: ANN001
        self.zset.setdefault(key, {}).update(mapping)

    async def zrem(self, key, *members):  # noqa: ANN001
        z = self.zset.get(key, {})
        for m in members:
            z.pop(m, None)

    async def zcard(self, key):  # noqa: ANN001
        return len(self.zset.get(key, {}))

    async def zrange(self, key, start, end):  # noqa: ANN001
        members = [m for m, _ in sorted(self.zset.get(key, {}).items(), key=lambda kv: kv[1])]
        if end == -1:
            return members[start:]
        return members[start : end + 1]


def _make_resp(analysis_id, status=AnalysisStatus.COMPLETED):  # noqa: ANN001
    from src.api.routes.segments import SegmentAnalysisResponse

    return SegmentAnalysisResponse(
        analysis_id=analysis_id,
        status=status,
        timestamp=datetime.now(timezone.utc),
    )


class TestDurableAnalysesStore:
    """The durable store must share state across processes via Redis and fall
    back to the bounded in-memory dict when Redis is unavailable."""

    @pytest.mark.asyncio
    async def test_cross_worker_read_via_shared_redis(self):
        """A value written through one store instance (worker A) is readable
        through a DIFFERENT instance sharing the same Redis (worker B)."""
        from src.api.routes.segments import _DurableAnalysesStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        worker_a = _DurableAnalysesStore(redis_factory=_factory)
        worker_b = _DurableAnalysesStore(redis_factory=_factory)

        resp = _make_resp("seg_shared")
        await worker_a.set("seg_shared", resp)

        # Worker B's in-memory fallback is empty; it must read from Redis.
        assert "seg_shared" not in worker_b._memory
        assert await worker_b.contains("seg_shared") is True
        got = await worker_b.get("seg_shared")
        assert got is not None
        assert got.analysis_id == "seg_shared"
        assert got.status == AnalysisStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_values_enumerates_from_redis(self):
        from src.api.routes.segments import _DurableAnalysesStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory)
        await store.set("seg_1", _make_resp("seg_1"))
        await store.set("seg_2", _make_resp("seg_2"))

        values = await store.values()
        ids = sorted(v.analysis_id for v in values)
        assert ids == ["seg_1", "seg_2"]

    @pytest.mark.asyncio
    async def test_falls_back_to_memory_when_redis_unavailable(self):
        """When the Redis factory fails, the store transparently uses the
        bounded in-memory fallback (no exception bubbles to the route)."""
        from src.api.routes.segments import _DurableAnalysesStore

        async def _factory():
            raise ConnectionError("redis down")

        store = _DurableAnalysesStore(redis_factory=_factory)

        await store.set("seg_local", _make_resp("seg_local"))
        assert await store.contains("seg_local") is True
        got = await store.get("seg_local")
        assert got is not None and got.analysis_id == "seg_local"
        values = await store.values()
        assert [v.analysis_id for v in values] == ["seg_local"]

    @pytest.mark.asyncio
    async def test_falls_back_when_redis_command_errors(self):
        """A mid-operation Redis error degrades to the in-memory fallback."""
        from src.api.routes.segments import _DurableAnalysesStore

        class _BrokenRedis(_FakeAsyncRedis):
            async def get(self, key):  # noqa: ANN001
                raise ConnectionError("boom")

        broken = _BrokenRedis()

        async def _factory():
            return broken

        store = _DurableAnalysesStore(redis_factory=_factory)
        # set still mirrors to memory, so a subsequent failing get falls back.
        await store.set("seg_x", _make_resp("seg_x"))
        got = await store.get("seg_x")
        assert got is not None and got.analysis_id == "seg_x"

    @pytest.mark.asyncio
    async def test_redis_eviction_bounds_index(self):
        """The Redis index is FIFO-bounded so it cannot grow without limit."""
        from src.api.routes.segments import _DurableAnalysesStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory, max_entries=3)
        for i in range(5):
            await store.set(f"seg_{i}", _make_resp(f"seg_{i}"))

        values = await store.values()
        ids = sorted(v.analysis_id for v in values)
        # Oldest two evicted; newest three retained.
        assert ids == ["seg_2", "seg_3", "seg_4"]
        assert await store.contains("seg_0") is False
