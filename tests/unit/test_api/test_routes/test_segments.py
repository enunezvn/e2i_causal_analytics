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
    """Faithful in-process stand-in for ``redis.asyncio.Redis``.

    Hand-rolled (consistent with the repo's other Redis tests, e.g.
    ``tests/unit/test_api/test_staleness_alerts.py``) because the ``fakeredis``
    package is not a test dependency here. The original fake hid three real
    bugs, so this version is deliberately FAITHFUL to redis 7.x behaviour:

    * ``set(..., ex=N)`` is HONORED — the key lazily expires after ``ex``
      seconds (controllable via the injectable ``_now`` clock), so the
      TTL-orphan cleanup branch is actually exercised.
    * ZSET ties on equal score break LEXICOGRAPHICALLY by member, exactly as
      real Redis does (verified against redis 7.1.0). The previous fake's
      stable sort preserved insertion order, masking the FIFO/eviction bug.
    * String reads return ``str`` (the app's client is created with
      ``decode_responses=True`` in ``redis_client.py``), not ``bytes``.
    * ``mget`` / ``exists`` / ``zremrangebyscore`` / ``pipeline`` are supported
      so the batched-read, atomic-write and TTL-prune paths can be tested.

    Tests can subclass and override individual commands to raise the REAL
    ``redis.exceptions.*`` types (``RedisError`` and subclasses) to verify the
    graceful-degradation path catches the right base.
    """

    def __init__(self, now: float = 0.0) -> None:
        self.strings: dict = {}  # key -> value
        self._expires: dict = {}  # key -> absolute expiry time (clock units)
        self.zset: dict = {}  # zkey -> {member: score}
        self._now = now  # injectable monotonic-ish clock (seconds)

    # -- clock control (tests advance time to trigger TTL expiry) ----------
    def advance(self, seconds: float) -> None:
        self._now += seconds

    def _expire_if_due(self, key) -> None:  # noqa: ANN001
        exp = self._expires.get(key)
        if exp is not None and self._now >= exp:
            self.strings.pop(key, None)
            self._expires.pop(key, None)

    # -- string commands ---------------------------------------------------
    async def set(self, key, value, ex=None):  # noqa: ANN001
        self.strings[key] = value
        if ex is not None:
            self._expires[key] = self._now + ex
        else:
            self._expires.pop(key, None)

    async def get(self, key):  # noqa: ANN001
        self._expire_if_due(key)
        return self.strings.get(key)

    async def mget(self, keys):  # noqa: ANN001
        out = []
        for k in keys:
            self._expire_if_due(k)
            out.append(self.strings.get(k))
        return out

    async def exists(self, *keys):  # noqa: ANN001
        n = 0
        for k in keys:
            self._expire_if_due(k)
            if k in self.strings:
                n += 1
        return n

    async def delete(self, *keys):  # noqa: ANN001
        removed = 0
        for k in keys:
            if k in self.strings:
                removed += 1
            self.strings.pop(k, None)
            self._expires.pop(k, None)
        return removed

    # -- sorted-set commands ----------------------------------------------
    async def zadd(self, key, mapping):  # noqa: ANN001
        self.zset.setdefault(key, {}).update(mapping)

    async def zrem(self, key, *members):  # noqa: ANN001
        z = self.zset.get(key, {})
        for m in members:
            z.pop(m, None)

    async def zcard(self, key):  # noqa: ANN001
        return len(self.zset.get(key, {}))

    async def zscore(self, key, member):  # noqa: ANN001
        return self.zset.get(key, {}).get(member)

    async def zrange(self, key, start, end):  # noqa: ANN001
        # Real Redis orders by (score, member) — ties break lexicographically.
        members = [
            m for m, _ in sorted(self.zset.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]))
        ]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    async def zremrangebyscore(self, key, min_score, max_score):  # noqa: ANN001
        z = self.zset.get(key, {})
        to_drop = [
            m
            for m, score in z.items()
            if (min_score == "-inf" or score >= float(min_score))
            and (max_score == "+inf" or score <= float(max_score))
        ]
        for m in to_drop:
            z.pop(m, None)
        return len(to_drop)

    # -- pipeline (transaction) -------------------------------------------
    def pipeline(self, transaction=True):  # noqa: ANN001
        return _FakePipeline(self)


class _FakePipeline:
    """Queue-then-execute pipeline matching ``redis.asyncio`` semantics.

    Buffered commands are applied atomically on ``execute()``. A command that
    is going to fail (overridden to raise) does so on ``execute``, mirroring
    real Redis where MULTI/EXEC reports errors at EXEC time — so a failing ZADD
    leaves the buffered SET un-applied (the atomicity property under test).
    """

    def __init__(self, parent: "_FakeAsyncRedis") -> None:
        self._parent = parent
        self._ops: list = []

    def set(self, key, value, ex=None):  # noqa: ANN001
        self._ops.append(("set", (key, value), {"ex": ex}))
        return self

    def zadd(self, key, mapping):  # noqa: ANN001
        self._ops.append(("zadd", (key, mapping), {}))
        return self

    def delete(self, *keys):  # noqa: ANN001
        self._ops.append(("delete", keys, {}))
        return self

    def zrem(self, key, *members):  # noqa: ANN001
        self._ops.append(("zrem", (key, *members), {}))
        return self

    async def execute(self):  # noqa: ANN001
        results = []
        for name, args, kwargs in self._ops:
            method = getattr(self._parent, name)
            results.append(await method(*args, **kwargs))
        return results


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


# =============================================================================
# HIGH #1 — NaN / +-inf poison must NOT 500 /policies, /health, or GET-by-id.
# =============================================================================
#
# pydantic ``model_dump_json`` serialises NaN / +-inf floats to JSON ``null``.
# The write succeeds (no re-validation), but on read ``model_validate_json``
# raises ``pydantic.ValidationError`` because CATEResult.cate_estimate (and
# friends) are non-Optional floats. ValidationError is NOT a Redis degrade
# error, so without a guard it escapes to a 500 that takes down enumeration for
# EVERY analysis. The honest fix is two-sided: (a) sanitise non-finite floats
# on write so we never persist a record we cannot read back, and (b) fail-soft
# on read so a single poison record (e.g. written by an older build) never
# breaks the collection.


def _make_poison_resp(analysis_id):  # noqa: ANN001
    """Build a response whose CATE floats are NaN / +-inf (degenerate fit)."""
    import math

    from src.api.routes.segments import (
        CATEResult,
        PolicyRecommendation,
        SegmentAnalysisResponse,
    )

    return SegmentAnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETED,
        timestamp=datetime.now(timezone.utc),
        cate_by_segment={
            "region": [
                CATEResult(
                    segment_name="region",
                    segment_value="Northeast",
                    cate_estimate=math.nan,
                    cate_ci_lower=-math.inf,
                    cate_ci_upper=math.inf,
                    sample_size=10,
                    statistical_significance=True,
                )
            ]
        },
        policy_recommendations=[
            PolicyRecommendation(
                segment="Northeast",
                current_treatment_rate=0.35,
                recommended_treatment_rate=0.55,
                expected_incremental_outcome=125.5,
                confidence=math.nan,
            )
        ],
    )


class TestNonFinitePoison:
    @pytest.mark.asyncio
    async def test_set_then_get_nan_does_not_raise(self):
        """A non-finite CATE round-trips without a ValidationError 500."""
        from src.api.routes.segments import _DurableAnalysesStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory)
        await store.set("seg_poison", _make_poison_resp("seg_poison"))

        # Must not raise. A degenerate (non-finite) fit is stored HONESTLY as a
        # FAILED record with NO fabricated finite numbers and NO unreadable
        # payload — the degenerate CATE/policy entries are dropped, not coerced.
        got = await store.get("seg_poison")
        assert got is not None, "poison write must remain readable, not 500 on read"
        assert got.status == AnalysisStatus.FAILED
        assert got.cate_by_segment == {}  # degenerate CATE dropped, not faked
        assert got.policy_recommendations == []
        assert got.overall_ate is None
        assert any("non-finite" in w.lower() for w in got.warnings)

    @pytest.mark.asyncio
    async def test_values_skips_unreadable_poison_record(self):
        """One unreadable record must not break enumeration of the rest.

        Simulates a poison record written by an OLDER build (raw NaN->null in
        Redis) alongside a healthy record. ``values()`` must return the healthy
        one and lazily ``zrem`` the poison id rather than 500-ing.
        """
        from src.api.routes.segments import (
            _REDIS_INDEX_KEY,
            _DurableAnalysesStore,
        )

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory)

        # Healthy record via the normal path.
        await store.set("seg_ok", _make_resp("seg_ok"))

        # Inject a poison record directly into Redis the way an older build
        # would have: NaN serialised to JSON null, indexed in the sorted set.
        poison_json = (
            '{"analysis_id":"seg_bad","status":"completed",'
            '"cate_by_segment":{"region":[{"segment_name":"region",'
            '"segment_value":"NE","cate_estimate":null,"cate_ci_lower":null,'
            '"cate_ci_upper":null,"sample_size":10,'
            '"statistical_significance":true}]}}'
        )
        await shared.set("segments:analysis:seg_bad", poison_json)
        await shared.zadd(_REDIS_INDEX_KEY, {"seg_bad": 1.0})

        values = await store.values()
        ids = [v.analysis_id for v in values]
        assert "seg_ok" in ids
        assert "seg_bad" not in ids  # skipped, not fatal
        # Poison id pruned from the index so it stops being enumerated.
        assert "seg_bad" not in await shared.zrange(_REDIS_INDEX_KEY, 0, -1)

    @pytest.mark.asyncio
    async def test_get_poison_lazily_removes_it(self):
        """A GET on a poison id returns None and lazily ``zrem``s it."""
        from src.api.routes.segments import (
            _REDIS_INDEX_KEY,
            _DurableAnalysesStore,
        )

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory)
        poison_json = (
            '{"analysis_id":"seg_bad","status":"completed",'
            '"cate_by_segment":{"region":[{"segment_name":"region",'
            '"segment_value":"NE","cate_estimate":null,"cate_ci_lower":null,'
            '"cate_ci_upper":null,"sample_size":10,'
            '"statistical_significance":true}]}}'
        )
        await shared.set("segments:analysis:seg_bad", poison_json)
        await shared.zadd(_REDIS_INDEX_KEY, {"seg_bad": 1.0})

        got = await store.get("seg_bad")
        assert got is None
        assert await shared.get("segments:analysis:seg_bad") is None
        assert "seg_bad" not in await shared.zrange(_REDIS_INDEX_KEY, 0, -1)


# =============================================================================
# HIGH #2 — degrade handling must catch the REAL redis.asyncio exception types.
# =============================================================================
#
# ``redis.exceptions.ConnectionError`` / ``TimeoutError`` are NOT the builtins
# of the same name (verified: ``redis.ConnectionError is builtins.ConnectionError
# -> False``); they subclass ``redis.exceptions.RedisError``. A degrade tuple
# of builtin (ConnectionError, TimeoutError, OSError, RuntimeError) therefore
# does NOT catch a real mid-flight Redis outage, so it 500s. These tests
# simulate the REAL types.


class TestRealRedisErrorDegrade:
    @pytest.mark.asyncio
    async def test_get_degrades_on_real_redis_error(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        from src.api.routes.segments import _DurableAnalysesStore

        class _BrokenRedis(_FakeAsyncRedis):
            async def get(self, key):  # noqa: ANN001
                raise RedisConnectionError("Error connecting to Redis")

        broken = _BrokenRedis()

        async def _factory():
            return broken

        store = _DurableAnalysesStore(redis_factory=_factory)
        await store.set("seg_x", _make_resp("seg_x"))  # mirrors to memory
        # Must NOT raise the real RedisError — must degrade to memory mirror.
        got = await store.get("seg_x")
        assert got is not None and got.analysis_id == "seg_x"

    @pytest.mark.asyncio
    async def test_values_degrades_on_real_redis_timeout(self):
        from redis.exceptions import TimeoutError as RedisTimeoutError

        from src.api.routes.segments import _DurableAnalysesStore

        class _BrokenRedis(_FakeAsyncRedis):
            async def zrange(self, key, start, end):  # noqa: ANN001
                raise RedisTimeoutError("Timeout reading from Redis")

        broken = _BrokenRedis()

        async def _factory():
            return broken

        store = _DurableAnalysesStore(redis_factory=_factory)
        await store.set("seg_y", _make_resp("seg_y"))
        values = await store.values()  # must not raise
        assert [v.analysis_id for v in values] == ["seg_y"]

    @pytest.mark.asyncio
    async def test_set_degrades_on_real_redis_error(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        from src.api.routes.segments import _DurableAnalysesStore

        class _BrokenRedis(_FakeAsyncRedis):
            async def set(self, key, value, ex=None):  # noqa: ANN001
                raise RedisConnectionError("write failed")

            def pipeline(self, transaction=True):  # noqa: ANN001
                raise RedisConnectionError("pipeline failed")

        broken = _BrokenRedis()

        async def _factory():
            return broken

        store = _DurableAnalysesStore(redis_factory=_factory)
        # Must NOT raise — write degrades, record still served from memory.
        await store.set("seg_z", _make_resp("seg_z"))
        got = await store.get("seg_z")
        assert got is not None and got.analysis_id == "seg_z"


# =============================================================================
# HIGH #3 — TTL expiry must not let count-based eviction delete a LIVE record.
# =============================================================================
#
# The index is scored by timestamp; entries expire from Redis after TTL but
# their index members linger (orphans). Count-based FIFO eviction over the raw
# index can therefore evict a LIVE, in-TTL, under-capacity analysis while a
# dead orphan survives -> data loss + spurious 404. Eviction must prune
# expired members (by score / by key existence) before counting.


class TestTTLOrphanEvictionSafety:
    @pytest.mark.asyncio
    async def test_orphans_do_not_evict_a_live_under_capacity_record(self):
        """The exact HIGH#3 data-loss path.

        A LIVE, in-TTL record (``seg_live_old``, the oldest by creation score)
        coexists with TWO expired ORPHAN index members. When a new record
        arrives at capacity=2, naive count-based eviction over the raw index
        sees ``zcard``=4, computes overflow=2 and evicts the two LOWEST-scored
        members — which are the live ``seg_live_old`` plus one orphan. That
        DELETES a live, under-capacity analysis (only 2 live records exist for
        a cap of 2) while a dead orphan can survive: data loss + spurious 404.

        Correct behaviour: prune expired orphan members by score FIRST, so the
        live count is 1, no eviction happens, and ``seg_live_old`` survives.
        """
        from src.api.routes.segments import (
            _REDIS_INDEX_KEY,
            _DurableAnalysesStore,
        )

        shared = _FakeAsyncRedis(now=1000.0)

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory, max_entries=2, ttl_seconds=100_000)

        # Oldest LIVE record (long TTL -> stays alive); creation score ~1000.
        await store.set("seg_live_old", _make_resp("seg_live_old"))

        # Two SHORT-TTL records become orphans after we advance time. Use a
        # second store handle with a short TTL to write them.
        short_store = _DurableAnalysesStore(redis_factory=_factory, max_entries=2, ttl_seconds=10)
        shared.advance(5)
        await short_store.set("seg_orphan0", _make_resp("seg_orphan0"))
        shared.advance(5)
        await short_store.set("seg_orphan1", _make_resp("seg_orphan1"))

        # Advance past the short TTL: the two orphan KEYS expire from Redis but
        # their index members linger.
        shared.advance(50)  # orphans (TTL 10) now expired; seg_live_old alive

        # A new live record arrives.
        await store.set("seg_new", _make_resp("seg_new"))

        # DATA-LOSS CHECK: the oldest LIVE record must NOT have been evicted.
        got = await store.get("seg_live_old")
        assert got is not None and got.analysis_id == "seg_live_old", (
            "live under-capacity record was evicted due to lingering TTL orphans"
        )

        # Both live records survive; orphans gone from the index.
        assert await store.contains("seg_new") is True
        remaining = await shared.zrange(_REDIS_INDEX_KEY, 0, -1)
        assert "seg_orphan0" not in remaining
        assert "seg_orphan1" not in remaining

    @pytest.mark.asyncio
    async def test_live_records_not_evicted_below_capacity_after_orphan_prune(self):
        """With orphans pruned, an under-capacity set of LIVE records is intact."""
        from src.api.routes.segments import _DurableAnalysesStore

        shared = _FakeAsyncRedis(now=0.0)

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory, max_entries=3, ttl_seconds=50)

        # One record that will expire.
        await store.set("seg_expire", _make_resp("seg_expire"))
        shared.advance(100)  # seg_expire now expired

        # Three live records — exactly at capacity, none should be evicted.
        await store.set("seg_a", _make_resp("seg_a"))
        await store.set("seg_b", _make_resp("seg_b"))
        await store.set("seg_c", _make_resp("seg_c"))

        for live in ("seg_a", "seg_b", "seg_c"):
            assert await store.contains(live) is True


# =============================================================================
# MEDIUM #4 — /segments/health must surface degraded (in-memory) mode.
# =============================================================================


class TestHealthDegradedVisibility:
    @pytest.mark.asyncio
    async def test_store_reports_durable_when_redis_present(self):
        from src.api.routes.segments import _DurableAnalysesStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory)
        await store.set("seg_d", _make_resp("seg_d"))
        assert await store.is_durable() is True

    @pytest.mark.asyncio
    async def test_store_reports_degraded_when_redis_down(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        from src.api.routes.segments import _DurableAnalysesStore

        async def _factory():
            raise RedisConnectionError("down")

        store = _DurableAnalysesStore(redis_factory=_factory)
        assert await store.is_durable() is False

    @pytest.mark.asyncio
    async def test_health_endpoint_reports_storage_mode(self, monkeypatch):
        """/segments/health must expose whether storage is durable or degraded."""
        from src.api.routes import segments as seg_mod

        async def _down_factory():
            from redis.exceptions import ConnectionError as RedisConnectionError

            raise RedisConnectionError("down")

        monkeypatch.setattr(seg_mod._analyses_store, "_redis_factory", _down_factory)

        result = await seg_mod.get_segment_health()
        # The health response must carry an observable storage-mode signal.
        assert getattr(result, "storage_mode", None) == "degraded"


# =============================================================================
# MEDIUM #5 — partial write atomicity (key SET + index ZADD commit together).
# =============================================================================


class TestPartialWriteAtomicity:
    @pytest.mark.asyncio
    async def test_zadd_failure_does_not_leave_unindexed_key(self):
        """If indexing fails, the record must NOT be left fetchable-but-invisible.

        A key SET that succeeds while the index ZADD fails yields a record that
        ``get`` finds but ``values`` (enumeration) never sees. A pipeline/txn
        must make them commit together, so a ZADD failure leaves NEITHER the
        key nor the index entry (and the write degrades to memory).
        """
        from src.api.routes.segments import (
            _REDIS_INDEX_KEY,
            _DurableAnalysesStore,
        )

        class _ZaddBroken(_FakeAsyncRedis):
            async def zadd(self, key, mapping):  # noqa: ANN001
                from redis.exceptions import RedisError

                raise RedisError("zadd failed")

        broken = _ZaddBroken()

        async def _factory():
            return broken

        store = _DurableAnalysesStore(redis_factory=_factory)
        await store.set("seg_atom", _make_resp("seg_atom"))

        # Atomicity: the orphaned key must NOT be left behind in Redis.
        assert await broken.get("segments:analysis:seg_atom") is None
        assert "seg_atom" not in await broken.zrange(_REDIS_INDEX_KEY, 0, -1)


# =============================================================================
# MEDIUM #6 — values() must batch reads (mget), not N sequential round-trips.
# =============================================================================


class TestValuesBatchesReads:
    @pytest.mark.asyncio
    async def test_values_uses_mget_not_n_gets(self):
        from src.api.routes.segments import _DurableAnalysesStore

        class _CountingRedis(_FakeAsyncRedis):
            def __init__(self, now=0.0):  # noqa: ANN001
                super().__init__(now=now)
                self.get_calls = 0
                self.mget_calls = 0

            async def get(self, key):  # noqa: ANN001
                self.get_calls += 1
                return await super().get(key)

            async def mget(self, keys):  # noqa: ANN001
                self.mget_calls += 1
                return await super().mget(keys)

        counting = _CountingRedis()

        async def _factory():
            return counting

        store = _DurableAnalysesStore(redis_factory=_factory)
        for i in range(5):
            await store.set(f"seg_{i}", _make_resp(f"seg_{i}"))

        counting.get_calls = 0  # reset; only count the enumeration reads
        counting.mget_calls = 0
        values = await store.values()

        assert len(values) == 5
        # Enumeration must NOT do one GET per record.
        assert counting.get_calls == 0
        # It must batch via a single mget.
        assert counting.mget_calls == 1

    @pytest.mark.asyncio
    async def test_values_mget_prunes_expired_slots(self):
        """An mget slot that is None (expired) is skipped and lazily ``zrem``'d."""
        from src.api.routes.segments import (
            _REDIS_INDEX_KEY,
            _DurableAnalysesStore,
        )

        shared = _FakeAsyncRedis(now=0.0)

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory, ttl_seconds=50)
        await store.set("seg_keep", _make_resp("seg_keep"))
        # Directly inject an indexed id whose key is absent (expired/missing).
        await shared.zadd(_REDIS_INDEX_KEY, {"seg_gone": 1.0})

        values = await store.values()
        assert [v.analysis_id for v in values] == ["seg_keep"]
        # The None-slot id is pruned from the index.
        assert "seg_gone" not in await shared.zrange(_REDIS_INDEX_KEY, 0, -1)


# =============================================================================
# LOW #7 — eviction must be true FIFO (creation order), not LRU.
# =============================================================================
#
# ``set()`` re-scores the index on every status update (PENDING->ESTIMATING->
# COMPLETED). If the score is bumped each write, an old analysis that merely
# got a status update would look "newest" and the genuinely-oldest record
# would be evicted -> effectively LRU, not FIFO. The index score must preserve
# CREATION time.


class TestFIFONotLRU:
    @pytest.mark.asyncio
    async def test_status_update_preserves_creation_order(self):
        from src.api.routes.segments import (
            _REDIS_INDEX_KEY,
            _DurableAnalysesStore,
        )

        shared = _FakeAsyncRedis(now=1000.0)

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory, max_entries=2)

        # Create seg_0 first (oldest), then seg_1.
        await store.set("seg_0", _make_resp("seg_0", status=AnalysisStatus.PENDING))
        shared.advance(10)
        await store.set("seg_1", _make_resp("seg_1", status=AnalysisStatus.PENDING))

        creation_score_0 = shared.zset[_REDIS_INDEX_KEY]["seg_0"]

        # seg_0 gets a status update much later — its index score must NOT bump.
        shared.advance(100)
        await store.set("seg_0", _make_resp("seg_0", status=AnalysisStatus.COMPLETED))
        assert shared.zset[_REDIS_INDEX_KEY]["seg_0"] == creation_score_0

        # Now a third record arrives, forcing one eviction. True FIFO evicts the
        # genuinely-oldest (seg_0), NOT seg_1 — even though seg_0 was written
        # most recently (a status update).
        shared.advance(10)
        await store.set("seg_2", _make_resp("seg_2", status=AnalysisStatus.PENDING))

        assert await store.contains("seg_0") is False  # oldest by creation
        assert await store.contains("seg_1") is True
        assert await store.contains("seg_2") is True


# =============================================================================
# ROUND-2 BUG 1 (MEDIUM) — write-side NaN/Inf guard must scrub EVERY float
# field, not just the CATE/policy payloads.
# =============================================================================
#
# ``_has_non_finite_floats`` fires for a non-finite in ANY float field
# (confidence, feature_importance, uplift_metrics, library_agreement_score,
# expected_total_lift, cate_*). But the round-1 ``_sanitize_non_finite`` only
# cleared the CATE/policy payloads + overall_ate/heterogeneity_score. A NaN in
# one of the OTHER float fields therefore SURVIVES sanitisation -> the persisted
# "FAILED" record still serialises that field to JSON ``null`` ->
# ``model_validate_json`` raises ValidationError on read -> the record is
# silently skipped + ``zrem``'d on enumeration, vanishing from durable storage
# (re-introducing the cross-worker 404 C21 exists to fix). The write-side
# GUARANTEE — "a stored record is always readable back" — is broken.


def _make_poison_resp_in_field(analysis_id, field):  # noqa: ANN001
    """Build an otherwise-valid response with a NaN in exactly ONE float field."""
    import math

    from src.api.routes.segments import (
        SegmentAnalysisResponse,
        UpliftMetrics,
    )

    resp = SegmentAnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETED,
        timestamp=datetime.now(timezone.utc),
    )
    if field == "confidence":
        resp.confidence = math.nan
    elif field == "feature_importance":
        resp.feature_importance = {"region": math.nan}
    elif field == "uplift_metrics":
        resp.uplift_metrics = UpliftMetrics(
            overall_auuc=math.nan,
            overall_qini=0.5,
            targeting_efficiency=0.5,
            model_type_used="random_forest",
        )
    elif field == "library_agreement_score":
        resp.library_agreement_score = math.inf
    elif field == "expected_total_lift":
        resp.expected_total_lift = -math.inf
    else:  # pragma: no cover - guard against typos in the parametrisation
        raise ValueError(f"unknown field {field}")
    return resp


class TestSanitizeScrubsAllFloatFields:
    @pytest.mark.parametrize(
        "field",
        [
            "confidence",
            "feature_importance",
            "uplift_metrics",
            "library_agreement_score",
            "expected_total_lift",
        ],
    )
    @pytest.mark.asyncio
    async def test_nan_in_any_float_field_round_trips_after_sanitize(self, field):
        """A NaN in ANY float field must survive set -> get without a 500.

        Round-1 only scrubbed cate/policy/overall_ate/heterogeneity_score, so a
        NaN in ``confidence`` / ``feature_importance`` / ``uplift_metrics`` /
        ``library_agreement_score`` / ``expected_total_lift`` persisted as an
        unreadable record. After the fix every such record is stored as an
        honest FAILED record that reads back cleanly.
        """
        from src.api.routes.segments import _DurableAnalysesStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory)
        await store.set(f"seg_{field}", _make_poison_resp_in_field(f"seg_{field}", field))

        # Must read back (no ValidationError 500) AND must be enumerable.
        got = await store.get(f"seg_{field}")
        assert got is not None, f"poison in {field} must remain readable, not 500"
        assert got.status == AnalysisStatus.FAILED

        listed = [v.analysis_id for v in await store.values()]
        assert f"seg_{field}" in listed, (
            f"poison in {field} vanished from enumeration (broken write guarantee)"
        )

    @pytest.mark.parametrize(
        "field",
        [
            "confidence",
            "feature_importance",
            "uplift_metrics",
            "library_agreement_score",
            "expected_total_lift",
        ],
    )
    def test_sanitized_record_is_provably_finite_and_json_round_trips(self, field):
        """The sanitised record must contain NO non-finite floats and must
        re-validate from its own JSON dump."""
        from src.api.routes.segments import (
            SegmentAnalysisResponse,
            _DurableAnalysesStore,
            _has_non_finite_floats,
        )

        poison = _make_poison_resp_in_field("seg_x", field)
        assert _has_non_finite_floats(poison) is True  # precondition

        sanitized = _DurableAnalysesStore._sanitize_non_finite("seg_x", poison)

        # Provably finite — the core invariant the guard must restore.
        assert _has_non_finite_floats(sanitized) is False, (
            f"sanitize left a non-finite float in {field}"
        )
        # And the JSON dump re-validates (the actual persistence round-trip).
        dumped = sanitized.model_dump_json()
        revalidated = SegmentAnalysisResponse.model_validate_json(dumped)
        assert revalidated.status == AnalysisStatus.FAILED


# =============================================================================
# ROUND-2 BUG 2 (MEDIUM) — TTL-orphan eviction Pass-1 must not prune a LIVE
# record whose index score is FROZEN at creation time.
# =============================================================================
#
# ``_prune_orphans`` Pass-1 does ``zremrangebyscore(index, '-inf', now-ttl)`` on
# the assumption "score <= now-ttl => key expired". Fix #7 made that assumption
# FALSE: the index score is frozen at CREATION time, but every ``set()`` resets
# the key TTL to ``now+ttl`` (the ``ex=self.ttl_seconds`` on the SET). So a
# record created > ttl ago but UPDATED recently has a LIVE key whose frozen
# creation score still satisfies ``score <= now-ttl`` -> Pass-1 deletes its
# index member while the key is alive. Result: fetchable-by-id but INVISIBLE to
# /policies & /health enumeration — the very split-brain the atomicity fix
# claims to prevent. Pass-2 (key-existence) only ADDS removals; it cannot
# restore what Pass-1 wrongly deleted.


class _FakeClock:
    """A single shared, injectable wall-clock used to drive BOTH the module's
    ``datetime.now`` (creation score + Pass-1 cutoff) AND the fake-Redis TTL,
    so the score-vs-TTL divergence under test is reproduced faithfully (one
    clock, not two)."""

    def __init__(self, redis: "_FakeAsyncRedis", start: float = 1000.0) -> None:
        self._redis = redis
        self._t = start
        self._redis._now = start

    def advance(self, seconds: float) -> None:
        self._t += seconds
        self._redis._now = self._t  # keep the Redis TTL clock in lockstep

    # Mimic the slice of the ``datetime`` API the module uses:
    #   datetime.now(timezone.utc).timestamp()
    def now(self, tz=None):  # noqa: ANN001
        clock = self

        class _Moment:
            def timestamp(self_inner) -> float:  # noqa: N805
                return clock._t

        return _Moment()


class TestTTLPruneDoesNotDropLiveUpdatedRecord:
    @pytest.mark.asyncio
    async def test_recently_updated_old_record_stays_indexed_while_key_is_live(self, monkeypatch):
        """The exact BUG-2 path: a record created > ttl ago but UPDATED recently
        keeps a LIVE key (TTL reset on the update) yet a FROZEN creation score
        older than ``now-ttl``. It must remain in the index / ``values()`` —
        Pass-1's score-based prune must not drop it."""
        from src.api.routes import segments as seg_mod
        from src.api.routes.segments import (
            _REDIS_INDEX_KEY,
            _DurableAnalysesStore,
        )

        shared = _FakeAsyncRedis()
        clock = _FakeClock(shared, start=1000.0)
        # Drive the module's wall-clock (creation score + Pass-1 cutoff) from the
        # SAME clock that controls Redis TTL expiry.
        monkeypatch.setattr(seg_mod, "datetime", clock)

        async def _factory():
            return shared

        ttl = 100
        store = _DurableAnalysesStore(redis_factory=_factory, max_entries=10, ttl_seconds=ttl)

        # Create the record at t=1000 (creation score frozen at 1000).
        await store.set("seg_updated_old", _make_resp("seg_updated_old"))
        creation_score = shared.zset[_REDIS_INDEX_KEY]["seg_updated_old"]
        assert creation_score == 1000.0

        # Advance WELL past the TTL relative to creation (t=1250, ttl=100), then
        # UPDATE the record. The update resets the key TTL (alive until t=1350)
        # but keeps the frozen creation score (1000). Pass-1 cutoff at update
        # time = 1250 - 100 = 1150, and 1000 <= 1150 -> Pass-1 would drop it.
        clock.advance(250)
        await store.set(
            "seg_updated_old", _make_resp("seg_updated_old", status=AnalysisStatus.COMPLETED)
        )

        # The key must still be alive (TTL was reset on the update).
        assert await shared.get("segments:analysis:seg_updated_old") is not None

        # BUG: Pass-1 must NOT have removed the index member of a LIVE record.
        indexed = await shared.zrange(_REDIS_INDEX_KEY, 0, -1)
        assert "seg_updated_old" in indexed, (
            "Pass-1 score-prune dropped the index member of a LIVE, recently-"
            "updated record (frozen creation score < now-ttl while key alive)"
        )

        # And it must be enumerable (the split-brain the fix exists to prevent).
        listed = [v.analysis_id for v in await store.values()]
        assert "seg_updated_old" in listed
        # Still fetchable too (consistency between get and enumerate).
        assert await store.contains("seg_updated_old") is True

    @pytest.mark.asyncio
    async def test_genuinely_expired_record_is_still_pruned(self, monkeypatch):
        """Regression guard: removing the buggy Pass-1 must NOT stop genuinely
        expired (dead-key) orphans from being pruned — Pass-2 (key existence)
        still handles them."""
        from src.api.routes import segments as seg_mod
        from src.api.routes.segments import (
            _REDIS_INDEX_KEY,
            _DurableAnalysesStore,
        )

        shared = _FakeAsyncRedis()
        clock = _FakeClock(shared, start=0.0)
        monkeypatch.setattr(seg_mod, "datetime", clock)

        async def _factory():
            return shared

        store = _DurableAnalysesStore(redis_factory=_factory, max_entries=10, ttl_seconds=50)

        # A short-TTL record that we never touch again -> its key truly expires.
        await store.set("seg_dead", _make_resp("seg_dead"))
        clock.advance(100)  # key (TTL 50) is now genuinely expired/gone

        # A fresh write triggers a prune cycle.
        await store.set("seg_fresh", _make_resp("seg_fresh"))

        # The dead orphan's index member must be gone (Pass-2 by key existence).
        indexed = await shared.zrange(_REDIS_INDEX_KEY, 0, -1)
        assert "seg_dead" not in indexed
        assert "seg_fresh" in indexed
        listed = [v.analysis_id for v in await store.values()]
        assert listed == ["seg_fresh"]
