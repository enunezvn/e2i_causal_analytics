"""Unit tests for Health Score API route handlers.

Tests all endpoints and helper functions in src/api/routes/health_score.py.
Mocks all external dependencies to ensure unit test isolation.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Dict, Any

# Import route functions and models
from src.api.routes.health_score import (
    # Endpoints
    run_health_check,
    quick_health_check,
    full_health_check,
    get_component_health,
    get_model_health,
    get_pipeline_health,
    get_agent_health,
    get_health_history,
    get_service_status,
    # Helper functions
    _execute_health_check,
    _generate_mock_health_response,
    _get_mock_component_health,
    _get_mock_model_health,
    _get_mock_pipeline_health,
    _get_mock_agent_health,
    _generate_recommendations,
    # Enums
    CheckScope,
    ComponentStatus,
    ModelStatus,
    PipelineStatus,
    HealthGrade,
    # Module-level storage
    _health_history,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_health_history():
    """Clear health history before each test."""
    _health_history.clear()
    yield
    _health_history.clear()


@pytest.fixture
def mock_agent_result():
    """Mock Health Score agent result."""
    return MagicMock(
        overall_health_score=85.5,
        health_grade="B",
        component_health_score=0.9,
        model_health_score=0.85,
        pipeline_health_score=0.88,
        agent_health_score=0.95,
        critical_issues=[],
        warnings=["Some warning"],
        health_summary="System health is good",
        check_latency_ms=1250,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# ENDPOINT TESTS - run_health_check
# =============================================================================


@pytest.mark.asyncio
async def test_run_health_check_full_scope():
    """Test run_health_check with FULL scope."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 0
        mock_execute.return_value = mock_result

        result = await run_health_check(scope=CheckScope.FULL)

        assert result.overall_health_score == 85.0
        assert result.health_grade == HealthGrade.B
        assert result.check_id.startswith("hs_")
        # Function sets check_latency_ms, so it should be > 0
        assert result.check_latency_ms >= 0
        mock_execute.assert_called_once_with(CheckScope.FULL)


@pytest.mark.asyncio
async def test_run_health_check_quick_scope():
    """Test run_health_check with QUICK scope."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.QUICK,
            overall_health_score=90.0,
            health_grade=HealthGrade.A,
        )
        mock_result.check_latency_ms = 0
        mock_execute.return_value = mock_result

        result = await run_health_check(scope=CheckScope.QUICK)

        assert result.overall_health_score == 90.0
        assert result.check_scope == CheckScope.QUICK
        mock_execute.assert_called_once_with(CheckScope.QUICK)


@pytest.mark.asyncio
async def test_run_health_check_stores_history():
    """Test that health check results are stored in history."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 0
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.FULL)

        assert len(_health_history) == 1
        assert _health_history[0].overall_health_score == 85.0


@pytest.mark.asyncio
async def test_run_health_check_limits_history_to_100():
    """Test that health history is limited to last 100 entries."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 0
        mock_execute.return_value = mock_result

        # Add 105 entries
        for _ in range(105):
            await run_health_check(scope=CheckScope.FULL)

        assert len(_health_history) == 100


@pytest.mark.asyncio
async def test_run_health_check_exception_handling():
    """Test that exceptions are properly handled."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_execute.side_effect = RuntimeError("Test error")

        with pytest.raises(Exception) as exc_info:
            await run_health_check(scope=CheckScope.FULL)

        assert "Health check failed" in str(exc_info.value)


# =============================================================================
# ENDPOINT TESTS - quick_health_check
# =============================================================================


@pytest.mark.asyncio
async def test_quick_health_check_calls_run_with_quick_scope():
    """Test quick_health_check calls run_health_check with QUICK scope."""
    with patch("src.api.routes.health_score.run_health_check") as mock_run:
        mock_result = MagicMock()
        mock_run.return_value = mock_result

        result = await quick_health_check()

        assert result == mock_result
        mock_run.assert_called_once_with(scope=CheckScope.QUICK)


# =============================================================================
# ENDPOINT TESTS - full_health_check
# =============================================================================


@pytest.mark.asyncio
async def test_full_health_check_calls_run_with_full_scope():
    """Test full_health_check calls run_health_check with FULL scope."""
    with patch("src.api.routes.health_score.run_health_check") as mock_run:
        mock_result = MagicMock()
        mock_run.return_value = mock_result

        result = await full_health_check()

        assert result == mock_result
        mock_run.assert_called_once_with(scope=CheckScope.FULL)


# =============================================================================
# ENDPOINT TESTS - get_component_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_component_health_success():
    """Test get_component_health returns component details."""
    result = await get_component_health()

    assert result.total_components > 0
    assert result.component_health_score >= 0.0
    assert result.component_health_score <= 1.0
    assert len(result.components) == result.total_components
    assert result.check_latency_ms >= 0


@pytest.mark.asyncio
async def test_get_component_health_score_calculation():
    """Test component health score is calculated correctly."""
    result = await get_component_health()

    # Score should be (healthy * 1.0 + degraded * 0.5) / total
    expected_score = (
        result.healthy_count * 1.0 + result.degraded_count * 0.5
    ) / result.total_components

    assert abs(result.component_health_score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_get_component_health_counts():
    """Test component health counts are correct."""
    result = await get_component_health()

    total = result.healthy_count + result.degraded_count + result.unhealthy_count
    assert total == result.total_components


@pytest.mark.asyncio
async def test_get_component_health_includes_all_components():
    """Test all expected components are included."""
    result = await get_component_health()

    component_names = [c.component_name for c in result.components]
    assert "postgresql" in component_names
    assert "redis" in component_names
    assert "falkordb" in component_names


# =============================================================================
# ENDPOINT TESTS - get_model_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_model_health_success():
    """Test get_model_health returns model details."""
    result = await get_model_health()

    assert result.total_models > 0
    assert result.model_health_score >= 0.0
    assert result.model_health_score <= 1.0
    assert len(result.models) == result.total_models
    assert result.check_latency_ms >= 0


@pytest.mark.asyncio
async def test_get_model_health_score_calculation():
    """Test model health score is calculated correctly."""
    result = await get_model_health()

    expected_score = (
        result.healthy_count * 1.0 + result.degraded_count * 0.5
    ) / result.total_models

    assert abs(result.model_health_score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_get_model_health_empty_models():
    """Test model health handles empty model list."""
    with patch("src.api.routes.health_score._get_mock_model_health", return_value=[]):
        result = await get_model_health()

        assert result.total_models == 0
        assert result.model_health_score == 1.0  # Default for empty list


@pytest.mark.asyncio
async def test_get_model_health_includes_metrics():
    """Test models include performance metrics."""
    result = await get_model_health()

    for model in result.models:
        assert hasattr(model, "model_id")
        assert hasattr(model, "status")
        assert model.error_rate >= 0.0


# =============================================================================
# ENDPOINT TESTS - get_pipeline_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_pipeline_health_success():
    """Test get_pipeline_health returns pipeline details."""
    result = await get_pipeline_health()

    assert result.total_pipelines > 0
    assert result.pipeline_health_score >= 0.0
    assert result.pipeline_health_score <= 1.0
    assert len(result.pipelines) == result.total_pipelines
    assert result.check_latency_ms >= 0


@pytest.mark.asyncio
async def test_get_pipeline_health_score_calculation():
    """Test pipeline health score is calculated correctly."""
    result = await get_pipeline_health()

    expected_score = (
        result.healthy_count * 1.0 + result.stale_count * 0.5
    ) / result.total_pipelines

    assert abs(result.pipeline_health_score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_get_pipeline_health_counts():
    """Test pipeline health counts are correct."""
    result = await get_pipeline_health()

    total = result.healthy_count + result.stale_count + result.failed_count
    assert total == result.total_pipelines


@pytest.mark.asyncio
async def test_get_pipeline_health_empty_pipelines():
    """Test pipeline health handles empty pipeline list."""
    with patch("src.api.routes.health_score._get_mock_pipeline_health", return_value=[]):
        result = await get_pipeline_health()

        assert result.total_pipelines == 0
        assert result.pipeline_health_score == 1.0


# =============================================================================
# ENDPOINT TESTS - get_agent_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_agent_health_success():
    """Test get_agent_health returns agent details."""
    result = await get_agent_health()

    assert result.total_agents > 0
    assert result.agent_health_score >= 0.0
    assert result.agent_health_score <= 1.0
    assert len(result.agents) == result.total_agents
    assert result.check_latency_ms >= 0


@pytest.mark.asyncio
async def test_get_agent_health_score_calculation():
    """Test agent health score is calculated correctly."""
    result = await get_agent_health()

    expected_score = result.available_count / result.total_agents
    assert abs(result.agent_health_score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_get_agent_health_by_tier():
    """Test agent health groups agents by tier."""
    result = await get_agent_health()

    assert isinstance(result.by_tier, dict)
    assert len(result.by_tier) > 0

    # Verify tier counts
    total_in_tiers = sum(result.by_tier.values())
    assert total_in_tiers == result.total_agents


@pytest.mark.asyncio
async def test_get_agent_health_empty_agents():
    """Test agent health handles empty agent list."""
    with patch("src.api.routes.health_score._get_mock_agent_health", return_value=[]):
        result = await get_agent_health()

        assert result.total_agents == 0
        assert result.agent_health_score == 1.0


# =============================================================================
# ENDPOINT TESTS - get_health_history
# =============================================================================


@pytest.mark.asyncio
async def test_get_health_history_empty():
    """Test get_health_history with empty history."""
    result = await get_health_history(limit=20)

    assert result.total_checks == 0
    assert len(result.checks) == 0
    assert result.avg_health_score == 0.0
    assert result.trend == "stable"


@pytest.mark.asyncio
async def test_get_health_history_with_data():
    """Test get_health_history returns historical data."""
    # Add some history
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 100
        mock_result.critical_issues = []
        mock_result.timestamp = datetime.now(timezone.utc).isoformat()
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.FULL)
        await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.total_checks == 2
    assert len(result.checks) == 2
    assert result.avg_health_score == 85.0


@pytest.mark.asyncio
async def test_get_health_history_respects_limit():
    """Test get_health_history respects limit parameter."""
    # Add 10 entries
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 100
        mock_result.critical_issues = []
        mock_result.timestamp = datetime.now(timezone.utc).isoformat()
        mock_execute.return_value = mock_result

        for _ in range(10):
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=5)

    assert len(result.checks) == 5


@pytest.mark.asyncio
async def test_get_health_history_trend_improving():
    """Test trend calculation for improving health."""
    # Add entries with improving scores
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        for score in [70.0, 75.0, 80.0, 85.0, 90.0]:
            mock_result = MagicMock(
                check_id="",
                check_scope=CheckScope.FULL,
                overall_health_score=score,
                health_grade=HealthGrade.B,
            )
            mock_result.check_latency_ms = 100
            mock_result.critical_issues = []
            mock_result.timestamp = datetime.now(timezone.utc).isoformat()
            mock_execute.return_value = mock_result
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.trend == "improving"


@pytest.mark.asyncio
async def test_get_health_history_trend_declining():
    """Test trend calculation for declining health."""
    # Add entries with declining scores
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        for score in [90.0, 85.0, 80.0, 75.0, 70.0]:
            mock_result = MagicMock(
                check_id="",
                check_scope=CheckScope.FULL,
                overall_health_score=score,
                health_grade=HealthGrade.B,
            )
            mock_result.check_latency_ms = 100
            mock_result.critical_issues = []
            mock_result.timestamp = datetime.now(timezone.utc).isoformat()
            mock_execute.return_value = mock_result
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.trend == "declining"


@pytest.mark.asyncio
async def test_get_health_history_trend_stable():
    """Test trend calculation for stable health."""
    # Add entries with stable scores
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        for _ in range(5):
            mock_result = MagicMock(
                check_id="",
                check_scope=CheckScope.FULL,
                overall_health_score=85.0,
                health_grade=HealthGrade.B,
            )
            mock_result.check_latency_ms = 100
            mock_result.critical_issues = []
            mock_result.timestamp = datetime.now(timezone.utc).isoformat()
            mock_execute.return_value = mock_result
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.trend == "stable"


# =============================================================================
# ENDPOINT TESTS - get_service_status
# =============================================================================


@pytest.mark.asyncio
async def test_get_service_status_agent_available():
    """Test get_service_status when agent is available."""
    with patch("src.agents.health_score.HealthScoreAgent"):
        result = await get_service_status()

        assert result.status == "healthy"
        assert result.agent_available is True


@pytest.mark.asyncio
async def test_get_service_status_agent_unavailable():
    """Test get_service_status when agent is unavailable."""
    # Patch the import to raise ImportError
    import sys

    with patch.dict(sys.modules, {"src.agents.health_score": None}):
        result = await get_service_status()

        assert result.status == "degraded"
        assert result.agent_available is False


@pytest.mark.asyncio
async def test_get_service_status_with_history():
    """Test get_service_status includes history metrics."""
    # Add some history
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 100
        mock_result.critical_issues = []
        mock_result.timestamp = datetime.now(timezone.utc).isoformat()
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.FULL)

    with patch("src.agents.health_score.HealthScoreAgent"):
        result = await get_service_status()

        assert result.last_check is not None
        assert result.checks_24h > 0
        assert result.avg_check_latency_ms >= 0


# =============================================================================
# HELPER FUNCTION TESTS - _execute_health_check
# =============================================================================


@pytest.mark.asyncio
async def test_execute_health_check_with_real_agent(mock_agent_result):
    """Test _execute_health_check uses real agent when available."""
    mock_agent = MagicMock()
    mock_agent.check_health = AsyncMock(return_value=mock_agent_result)

    with patch(
        "src.agents.health_score.HealthScoreAgent", return_value=mock_agent
    ):
        result = await _execute_health_check(CheckScope.FULL)

        assert result.overall_health_score == 85.5
        assert result.health_grade == HealthGrade.B
        mock_agent.check_health.assert_called_once_with(scope="full")


@pytest.mark.asyncio
async def test_execute_health_check_quick_mode(mock_agent_result):
    """Test _execute_health_check uses quick_check for QUICK scope."""
    mock_agent = MagicMock()
    mock_agent.quick_check = AsyncMock(return_value=mock_agent_result)

    with patch(
        "src.agents.health_score.HealthScoreAgent", return_value=mock_agent
    ):
        result = await _execute_health_check(CheckScope.QUICK)

        mock_agent.quick_check.assert_called_once()


@pytest.mark.asyncio
async def test_execute_health_check_falls_back_to_mock():
    """Test _execute_health_check falls back to mock when agent unavailable."""
    with patch(
        "src.agents.health_score.HealthScoreAgent", side_effect=ImportError
    ):
        result = await _execute_health_check(CheckScope.FULL)

        assert result.overall_health_score > 0
        assert result.warnings[0] == "Using mock data - Health Score agent not available"


@pytest.mark.asyncio
async def test_execute_health_check_agent_exception():
    """Test _execute_health_check handles agent exceptions."""
    mock_agent = MagicMock()
    mock_agent.check_health = AsyncMock(side_effect=RuntimeError("Agent error"))

    with patch(
        "src.agents.health_score.HealthScoreAgent", return_value=mock_agent
    ):
        with pytest.raises(RuntimeError):
            await _execute_health_check(CheckScope.FULL)


# =============================================================================
# HELPER FUNCTION TESTS - _generate_mock_health_response
# =============================================================================


def test_generate_mock_health_response_full_scope():
    """Test mock response generation for FULL scope."""
    import time

    start_time = time.time()
    result = _generate_mock_health_response(CheckScope.FULL, start_time)

    assert result.check_scope == CheckScope.FULL
    assert result.component_statuses is not None
    assert result.model_metrics is not None
    assert result.pipeline_statuses is not None
    assert result.agent_statuses is not None


def test_generate_mock_health_response_quick_scope():
    """Test mock response generation for QUICK scope."""
    import time

    start_time = time.time()
    result = _generate_mock_health_response(CheckScope.QUICK, start_time)

    assert result.check_scope == CheckScope.QUICK
    assert result.component_statuses is not None
    assert result.model_metrics is None


def test_generate_mock_health_response_models_scope():
    """Test mock response generation for MODELS scope."""
    import time

    start_time = time.time()
    result = _generate_mock_health_response(CheckScope.MODELS, start_time)

    assert result.check_scope == CheckScope.MODELS
    assert result.model_metrics is not None
    assert result.component_statuses is None


def test_generate_mock_health_response_grade_A():
    """Test grade A assignment for high scores."""
    import time

    start_time = time.time()

    with patch("src.api.routes.health_score._get_mock_component_health"):
        with patch("src.api.routes.health_score._get_mock_model_health"):
            with patch("src.api.routes.health_score._get_mock_pipeline_health"):
                with patch("src.api.routes.health_score._get_mock_agent_health"):
                    # Mock all scores to be 1.0
                    result = _generate_mock_health_response(CheckScope.FULL, start_time)

                    # Overall should be high
                    assert result.overall_health_score >= 80


def test_generate_mock_health_response_includes_recommendations():
    """Test mock response includes recommendations."""
    import time

    start_time = time.time()
    result = _generate_mock_health_response(CheckScope.FULL, start_time)

    assert len(result.recommendations) > 0


# =============================================================================
# HELPER FUNCTION TESTS - _get_mock_* functions
# =============================================================================


def test_get_mock_component_health():
    """Test _get_mock_component_health returns expected components."""
    components = _get_mock_component_health()

    assert len(components) > 0
    component_names = [c.component_name for c in components]
    assert "postgresql" in component_names
    assert "redis" in component_names
    assert "falkordb" in component_names
    assert "mlflow" in component_names


def test_get_mock_model_health():
    """Test _get_mock_model_health returns expected models."""
    models = _get_mock_model_health()

    assert len(models) > 0
    for model in models:
        assert model.model_id is not None
        assert model.model_name is not None
        assert model.status in [ModelStatus.HEALTHY, ModelStatus.DEGRADED, ModelStatus.UNHEALTHY]


def test_get_mock_pipeline_health():
    """Test _get_mock_pipeline_health returns expected pipelines."""
    pipelines = _get_mock_pipeline_health()

    assert len(pipelines) > 0
    for pipeline in pipelines:
        assert pipeline.pipeline_name is not None
        assert pipeline.status in [PipelineStatus.HEALTHY, PipelineStatus.STALE, PipelineStatus.FAILED]
        assert pipeline.rows_processed >= 0


def test_get_mock_agent_health():
    """Test _get_mock_agent_health returns expected agents."""
    agents = _get_mock_agent_health()

    assert len(agents) > 0
    for agent in agents:
        assert agent.agent_name is not None
        assert agent.tier >= 0
        assert agent.tier <= 5
        assert isinstance(agent.available, bool)


# =============================================================================
# HELPER FUNCTION TESTS - _generate_recommendations
# =============================================================================


def test_generate_recommendations_all_healthy():
    """Test recommendations when all scores are healthy."""
    recommendations = _generate_recommendations(0.9, 0.9, 0.9, 0.9)

    assert len(recommendations) == 1
    assert "Continue monitoring" in recommendations[0]


def test_generate_recommendations_low_component_score():
    """Test recommendations when component score is low."""
    recommendations = _generate_recommendations(0.7, 0.9, 0.9, 0.9)

    assert any("component" in r.lower() for r in recommendations)


def test_generate_recommendations_low_model_score():
    """Test recommendations when model score is low."""
    recommendations = _generate_recommendations(0.9, 0.7, 0.9, 0.9)

    assert any("model" in r.lower() for r in recommendations)


def test_generate_recommendations_low_pipeline_score():
    """Test recommendations when pipeline score is low."""
    recommendations = _generate_recommendations(0.9, 0.9, 0.7, 0.9)

    assert any("pipeline" in r.lower() for r in recommendations)


def test_generate_recommendations_low_agent_score():
    """Test recommendations when agent score is low."""
    recommendations = _generate_recommendations(0.9, 0.9, 0.9, 0.7)

    assert any("agent" in r.lower() for r in recommendations)


def test_generate_recommendations_multiple_issues():
    """Test recommendations when multiple scores are low."""
    recommendations = _generate_recommendations(0.7, 0.7, 0.7, 0.7)

    assert len(recommendations) == 4
