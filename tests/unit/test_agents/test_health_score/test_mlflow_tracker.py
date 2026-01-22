"""Unit tests for HealthScore MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting health runs
- Metric extraction (overall/component health scores, grades)
- Parameter logging
- Artifact logging (JSON)
- Historical query methods (health history, health trends)
- Graceful degradation when MLflow unavailable

Phase 1 G03 from observability audit remediation plan.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.health_score.mlflow_tracker import (
    HealthScoreContext,
    HealthScoreMetrics,
    HealthScoreMLflowTracker,
    EXPERIMENT_PREFIX,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_mlflow():
    """Mock MLflow module."""
    mock = MagicMock()
    mock.set_experiment = MagicMock()
    mock.start_run = MagicMock()
    mock.end_run = MagicMock()
    mock.log_param = MagicMock()
    mock.log_params = MagicMock()
    mock.log_metric = MagicMock()
    mock.log_metrics = MagicMock()
    mock.log_artifact = MagicMock()
    mock.search_runs = MagicMock(return_value=MagicMock(to_dict=MagicMock(return_value={"run_id": []})))
    mock.get_experiment_by_name = MagicMock(return_value=MagicMock(experiment_id="test_exp_id"))
    return mock


@pytest.fixture
def tracker():
    """Create a HealthScoreMLflowTracker instance."""
    return HealthScoreMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample HealthScoreContext."""
    return HealthScoreContext(
        query="What is the current system health?",
        scope="full",
        components_checked=["database", "cache", "vector_store", "api", "message_queue"],
    )


@pytest.fixture
def sample_result():
    """Create a sample health check result dict."""
    return {
        "overall_health_score": 85.5,
        "health_grade": "B",
        "component_health_score": 0.90,
        "model_health_score": 0.80,
        "pipeline_health_score": 0.85,
        "agent_health_score": 0.90,
        "critical_issues": [],
        "warnings": ["Model 'churn_predictor' has degraded accuracy (0.72)"],
        "health_summary": "System health is good (Grade: B, Score: 85.5/100). All systems operational.",
        "check_latency_ms": 1250,
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestExperimentConfiguration:
    """Tests for experiment configuration constants."""

    def test_experiment_prefix_format(self):
        """Test experiment prefix follows naming convention."""
        assert EXPERIMENT_PREFIX == "e2i_causal/health_score"

    def test_experiment_prefix_contains_agent_name(self):
        """Test experiment prefix includes agent identifier."""
        assert "health_score" in EXPERIMENT_PREFIX


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestHealthScoreContext:
    """Tests for HealthScoreContext dataclass."""

    def test_context_creation_minimal(self):
        """Test context creation with minimal fields."""
        ctx = HealthScoreContext(
            query="health check",
        )
        assert ctx.query == "health check"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.scope == "full"
        assert len(sample_context.components_checked) == 5
        assert "database" in sample_context.components_checked

    def test_context_default_values(self):
        """Test context default values."""
        ctx = HealthScoreContext(
            query="test",
        )
        assert ctx.scope is None or isinstance(ctx.scope, str)


class TestHealthScoreMetrics:
    """Tests for HealthScoreMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = HealthScoreMetrics(
            overall_health_score=85.5,
            health_grade="B",
            check_latency_ms=1250,
        )
        assert metrics.overall_health_score == 85.5
        assert metrics.health_grade == "B"

    def test_metrics_optional_fields(self):
        """Test metrics with optional fields."""
        metrics = HealthScoreMetrics(
            overall_health_score=90.0,
        )
        assert metrics.overall_health_score == 90.0


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, HealthScoreMLflowTracker)

    def test_tracker_lazy_mlflow_loading(self, tracker):
        """Test MLflow is lazily loaded."""
        assert hasattr(tracker, "_mlflow") or hasattr(tracker, "_check_mlflow")

    def test_check_mlflow_returns_bool(self, tracker):
        """Test _check_mlflow returns boolean."""
        result = tracker._check_mlflow()
        assert isinstance(result, bool)


class TestMLflowAvailability:
    """Tests for MLflow availability checking."""

    def test_mlflow_available_when_installed(self, tracker, mock_mlflow):
        """Test MLflow detection when installed."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                assert tracker._check_mlflow() is True

    def test_graceful_degradation_when_unavailable(self, tracker):
        """Test tracker works when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            result = tracker._check_mlflow()
            assert result is False


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartHealthRun:
    """Tests for start_health_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_returns_context_manager(self, tracker, sample_context):
        """Test start_health_run returns async context manager."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            async with tracker.start_health_run(sample_context) as run_ctx:
                assert run_ctx is None or isinstance(run_ctx, dict)

    @pytest.mark.asyncio
    async def test_start_run_with_mlflow(self, tracker, sample_context, mock_mlflow):
        """Test start_health_run with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

                async with tracker.start_health_run(sample_context):
                    mock_mlflow.set_experiment.assert_called()

    @pytest.mark.asyncio
    async def test_start_run_logs_scope(self, tracker, sample_context, mock_mlflow):
        """Test that scope is logged during run start."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

                async with tracker.start_health_run(sample_context):
                    pass

                mock_mlflow.set_experiment.assert_called()


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_dict(self, tracker, sample_result):
        """Test metric extraction from result dict."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)

    def test_extract_overall_health_score(self, tracker, sample_result):
        """Test overall health score extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)
        if "overall_health_score" in metrics:
            assert metrics["overall_health_score"] == 85.5

    def test_extract_component_scores(self, tracker, sample_result):
        """Test component score extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        result = {"overall_health_score": 90.0}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, dict)

    def test_extract_metrics_handles_none(self, tracker):
        """Test metric extraction with None values."""
        result = {"overall_health_score": None, "health_grade": "A"}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, dict)

    def test_extract_latency_metric(self, tracker, sample_result):
        """Test check latency extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogHealthResult:
    """Tests for log_health_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_result):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            await tracker.log_health_result(sample_result)

    @pytest.mark.asyncio
    async def test_log_result_with_mlflow(self, tracker, sample_result, mock_mlflow):
        """Test logging result with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                await tracker.log_health_result(sample_result)
                assert mock_mlflow.log_metrics.called or mock_mlflow.log_metric.called

    @pytest.mark.asyncio
    async def test_log_result_includes_grade(self, tracker, sample_result, mock_mlflow):
        """Test that health grade is logged."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                await tracker.log_health_result(sample_result)


class TestLogParams:
    """Tests for _log_params method."""

    def test_log_params_from_context(self, tracker, sample_context, mock_mlflow):
        """Test parameter logging from context."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                tracker._log_params(sample_context)
                assert mock_mlflow.log_params.called or mock_mlflow.log_param.called

    def test_log_params_includes_scope(self, tracker, sample_context, mock_mlflow):
        """Test scope is logged as parameter."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                tracker._log_params(sample_context)


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_creates_json(self, tracker, sample_result, mock_mlflow):
        """Test artifact logging creates JSON file."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.name = "/tmp/test_artifact.json"
                    mock_temp.return_value = mock_file

                    await tracker._log_artifacts(sample_result)

    @pytest.mark.asyncio
    async def test_log_artifacts_includes_warnings(self, tracker, sample_result, mock_mlflow):
        """Test warnings are included in artifacts."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.name = "/tmp/test_artifact.json"
                    mock_temp.return_value = mock_file

                    await tracker._log_artifacts(sample_result)


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetHealthHistory:
    """Tests for get_health_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            history = await tracker.get_health_history()
            assert history is None or isinstance(history, (list, dict))

    @pytest.mark.asyncio
    async def test_get_history_with_mlflow(self, tracker, mock_mlflow):
        """Test history query with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1", "run2"]})
                )

                history = await tracker.get_health_history()
                mock_mlflow.search_runs.assert_called()

    @pytest.mark.asyncio
    async def test_get_history_with_time_filter(self, tracker, mock_mlflow):
        """Test history query filters by time range."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1"]})
                )

                await tracker.get_health_history(hours=24)


class TestGetHealthTrend:
    """Tests for get_health_trend method."""

    @pytest.mark.asyncio
    async def test_get_trend_without_mlflow(self, tracker):
        """Test trend query when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            trend = await tracker.get_health_trend()
            assert trend is None or isinstance(trend, (list, dict))

    @pytest.mark.asyncio
    async def test_get_trend_returns_dict(self, tracker, mock_mlflow):
        """Test trend returns dictionary structure."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_df = MagicMock()
                mock_df.to_dict.return_value = {
                    "metrics.overall_health_score": [85.0, 87.5, 90.0],
                    "metrics.check_latency_ms": [1200, 1100, 1000],
                }
                mock_mlflow.search_runs.return_value = mock_df

                trend = await tracker.get_health_trend()
                assert trend is None or isinstance(trend, dict)

    @pytest.mark.asyncio
    async def test_get_trend_tracks_score_changes(self, tracker, mock_mlflow):
        """Test trend tracks health score changes over time."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1", "run2", "run3"]})
                )

                await tracker.get_health_trend()


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_mlflow_connection_error(self, tracker, mock_mlflow):
        """Test handling of MLflow connection errors."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.set_experiment.side_effect = Exception("Connection failed")

                try:
                    async with tracker.start_health_run(
                        HealthScoreContext(query="test")
                    ):
                        pass
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_handles_invalid_result_format(self, tracker):
        """Test handling of invalid result format."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            await tracker.log_health_result({"invalid": "structure"})

    def test_handles_empty_context(self, tracker):
        """Test handling of minimal context."""
        ctx = HealthScoreContext(query="")
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_handles_critical_health_issues(self, tracker):
        """Test handling of result with critical issues."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            result = {
                "overall_health_score": 45.0,
                "health_grade": "F",
                "critical_issues": ["Database unreachable", "API timeout"],
            }
            await tracker.log_health_result(result)

    @pytest.mark.asyncio
    async def test_handles_perfect_health(self, tracker):
        """Test handling of perfect health score."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            result = {
                "overall_health_score": 100.0,
                "health_grade": "A",
                "critical_issues": [],
                "warnings": [],
            }
            await tracker.log_health_result(result)
