"""Unit tests for HealthScore MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting health runs
- Metrics dataclass and to_dict conversion
- Health result logging
- Historical query methods (health history, health trends)
- Graceful degradation when MLflow unavailable

Phase 1 G03 from observability audit remediation plan.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agents.health_score.mlflow_tracker import (
    HealthScoreContext,
    HealthScoreMetrics,
    HealthScoreMLflowTracker,
    EXPERIMENT_PREFIX,
    create_tracker,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tracker():
    """Create a HealthScoreMLflowTracker instance."""
    return HealthScoreMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample HealthScoreContext."""
    return HealthScoreContext(
        run_id="run_456",
        experiment_name="production",
        check_scope="full",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def sample_metrics():
    """Create a sample HealthScoreMetrics."""
    return HealthScoreMetrics(
        overall_health_score=85.5,
        health_grade="B",
        component_health_score=0.90,
        model_health_score=0.80,
        pipeline_health_score=0.85,
        agent_health_score=0.90,
        critical_issues_count=0,
        warnings_count=1,
        check_scope="full",
        check_latency_ms=1250,
    )


@pytest.fixture
def mock_health_output():
    """Create a mock HealthScoreOutput object."""
    output = MagicMock()
    output.overall_health_score = 85.5
    output.health_grade = "B"
    output.component_health_score = 0.90
    output.model_health_score = 0.80
    output.pipeline_health_score = 0.85
    output.agent_health_score = 0.90
    output.critical_issues = []
    output.warnings = ["Model 'churn_predictor' has degraded accuracy (0.72)"]
    output.check_latency_ms = 1250
    output.health_summary = "System health is good (Grade: B, Score: 85.5/100). All systems operational."
    output.timestamp = datetime.now(timezone.utc).isoformat()
    return output


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
        """Test context creation with required fields only."""
        ctx = HealthScoreContext(
            run_id="test_run",
            experiment_name="test_experiment",
            check_scope="quick",
        )
        assert ctx.run_id == "test_run"
        assert ctx.experiment_name == "test_experiment"
        assert ctx.check_scope == "quick"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.run_id == "run_456"
        assert sample_context.experiment_name == "production"
        assert sample_context.check_scope == "full"
        assert sample_context.timestamp is not None

    def test_context_default_timestamp(self):
        """Test context has default timestamp factory."""
        ctx = HealthScoreContext(
            run_id="test",
            experiment_name="test",
            check_scope="quick",
        )
        assert ctx.timestamp is not None
        # Should be an ISO format string
        assert "T" in ctx.timestamp


class TestHealthScoreMetrics:
    """Tests for HealthScoreMetrics dataclass."""

    def test_metrics_creation_full(self, sample_metrics):
        """Test metrics dataclass creation with all fields."""
        assert sample_metrics.overall_health_score == 85.5
        assert sample_metrics.health_grade == "B"
        assert sample_metrics.component_health_score == 0.90
        assert sample_metrics.model_health_score == 0.80
        assert sample_metrics.pipeline_health_score == 0.85
        assert sample_metrics.agent_health_score == 0.90
        assert sample_metrics.critical_issues_count == 0
        assert sample_metrics.warnings_count == 1
        assert sample_metrics.check_scope == "full"
        assert sample_metrics.check_latency_ms == 1250

    def test_metrics_default_values(self):
        """Test metrics with default values."""
        metrics = HealthScoreMetrics()
        assert metrics.overall_health_score == 0.0
        assert metrics.health_grade == "F"
        assert metrics.component_health_score == 0.0
        assert metrics.model_health_score == 0.0
        assert metrics.pipeline_health_score == 0.0
        assert metrics.agent_health_score == 0.0
        assert metrics.critical_issues_count == 0
        assert metrics.warnings_count == 0
        assert metrics.check_scope == "full"
        assert metrics.check_latency_ms == 0

    def test_metrics_to_dict(self, sample_metrics):
        """Test metrics to_dict conversion."""
        result = sample_metrics.to_dict()
        assert isinstance(result, dict)
        assert result["overall_health_score"] == 85.5
        assert result["component_health_score"] == 0.90
        assert result["model_health_score"] == 0.80
        assert result["pipeline_health_score"] == 0.85
        assert result["agent_health_score"] == 0.90
        assert result["critical_issues_count"] == 0
        assert result["warnings_count"] == 1
        assert result["check_latency_ms"] == 1250

    def test_metrics_to_dict_includes_numeric_grade(self, sample_metrics):
        """Test to_dict includes numeric grade conversion."""
        result = sample_metrics.to_dict()
        assert "health_grade_numeric" in result
        # Grade B should map to 4
        assert result["health_grade_numeric"] == 4

    def test_grade_to_numeric_conversion(self):
        """Test letter grade to numeric conversion."""
        assert HealthScoreMetrics._grade_to_numeric("A") == 5
        assert HealthScoreMetrics._grade_to_numeric("B") == 4
        assert HealthScoreMetrics._grade_to_numeric("C") == 3
        assert HealthScoreMetrics._grade_to_numeric("D") == 2
        assert HealthScoreMetrics._grade_to_numeric("F") == 1
        assert HealthScoreMetrics._grade_to_numeric("X") == 0  # Unknown grade


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, HealthScoreMLflowTracker)

    def test_tracker_has_mlflow_attr(self, tracker):
        """Test tracker has _mlflow attribute."""
        assert hasattr(tracker, "_mlflow")

    def test_tracker_has_tracking_uri_attr(self, tracker):
        """Test tracker has _tracking_uri attribute."""
        assert hasattr(tracker, "_tracking_uri")

    def test_tracker_has_current_run_id_attr(self, tracker):
        """Test tracker has _current_run_id attribute."""
        assert hasattr(tracker, "_current_run_id")

    def test_get_mlflow_returns_mlflow_or_none(self, tracker):
        """Test _get_mlflow returns mlflow module or None."""
        result = tracker._get_mlflow()
        assert result is None or hasattr(result, "log_metric")

    def test_tracker_creation_with_uri(self):
        """Test tracker creation with custom tracking URI."""
        tracker = HealthScoreMLflowTracker(tracking_uri="http://localhost:5000")
        assert tracker._tracking_uri == "http://localhost:5000"


class TestMLflowAvailability:
    """Tests for MLflow availability checking."""

    def test_mlflow_starts_as_none(self, tracker):
        """Test MLflow is None initially (lazy loading)."""
        assert tracker._mlflow is None

    def test_graceful_degradation_when_unavailable(self, tracker):
        """Test tracker works when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            result = tracker._get_mlflow()
            assert result is None


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_create_tracker_returns_instance(self):
        """Test factory function returns tracker instance."""
        tracker = create_tracker()
        assert isinstance(tracker, HealthScoreMLflowTracker)

    def test_create_tracker_with_uri(self):
        """Test factory function with custom URI."""
        tracker = create_tracker(tracking_uri="http://localhost:5000")
        assert tracker._tracking_uri == "http://localhost:5000"


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartHealthRun:
    """Tests for start_health_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_returns_context_manager(self, tracker):
        """Test start_health_run returns async context manager."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_health_run(
                experiment_name="test_experiment",
                check_scope="full",
            ) as run_ctx:
                assert isinstance(run_ctx, HealthScoreContext)

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow_returns_context(self, tracker):
        """Test start_health_run returns context when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_health_run(
                experiment_name="test_experiment",
                check_scope="quick",
            ) as run_ctx:
                assert isinstance(run_ctx, HealthScoreContext)
                assert run_ctx.experiment_name == "test_experiment"
                assert run_ctx.check_scope == "quick"
                assert run_ctx.run_id == "no-mlflow"

    @pytest.mark.asyncio
    async def test_start_run_context_has_required_fields(self, tracker):
        """Test context manager returns context with required fields."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_health_run(
                experiment_name="production",
                check_scope="models",
            ) as run_ctx:
                assert run_ctx.experiment_name == "production"
                assert run_ctx.check_scope == "models"
                assert run_ctx.run_id is not None
                assert run_ctx.timestamp is not None

    @pytest.mark.asyncio
    async def test_start_run_default_params(self, tracker):
        """Test start_health_run with default parameters."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_health_run() as run_ctx:
                assert run_ctx.experiment_name == "default"
                assert run_ctx.check_scope == "full"

    @pytest.mark.asyncio
    async def test_start_run_with_different_scopes(self, tracker):
        """Test start_health_run with different check scopes."""
        scopes = ["quick", "full", "models", "pipelines", "agents"]
        for scope in scopes:
            with patch.object(tracker, "_get_mlflow", return_value=None):
                async with tracker.start_health_run(
                    experiment_name="test",
                    check_scope=scope,
                ) as run_ctx:
                    assert run_ctx.check_scope == scope


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogHealthResult:
    """Tests for log_health_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, mock_health_output):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Should not raise, just return silently
            await tracker.log_health_result(mock_health_output)

    @pytest.mark.asyncio
    async def test_log_result_without_current_run(self, tracker, mock_health_output):
        """Test logging result when no current run."""
        mock_mlflow = MagicMock()
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            # _current_run_id is None by default
            await tracker.log_health_result(mock_health_output)
            # Should not call log_metrics since no run is active
            mock_mlflow.log_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_result_with_state(self, tracker, mock_health_output):
        """Test logging result with state parameter."""
        mock_state = {
            "component_statuses": [{"name": "db", "status": "healthy"}],
            "model_metrics": [{"name": "churn", "accuracy": 0.92}],
            "pipeline_statuses": [{"name": "etl", "status": "ok"}],
            "agent_statuses": [{"name": "orchestrator", "status": "active"}],
        }
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_health_result(mock_health_output, state=mock_state)


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetHealthHistory:
    """Tests for get_health_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = await tracker.get_health_history()
            assert isinstance(history, list)
            assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_history_returns_list(self, tracker):
        """Test history query returns list structure."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = await tracker.get_health_history(experiment_name="production")
            assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_history_with_max_results(self, tracker):
        """Test history query with max_results parameter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = await tracker.get_health_history(max_results=50)
            assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_history_default_params(self, tracker):
        """Test history query with default parameters."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = await tracker.get_health_history()
            assert isinstance(history, list)


class TestGetHealthTrend:
    """Tests for get_health_trend method."""

    @pytest.mark.asyncio
    async def test_get_trend_without_mlflow(self, tracker):
        """Test trend query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = await tracker.get_health_trend()
            assert isinstance(trend, dict)

    @pytest.mark.asyncio
    async def test_get_trend_returns_dict_structure(self, tracker):
        """Test trend returns dictionary structure."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = await tracker.get_health_trend()
            assert isinstance(trend, dict)
            assert "trend" in trend
            assert "data_points" in trend

    @pytest.mark.asyncio
    async def test_get_trend_unknown_when_no_data(self, tracker):
        """Test trend returns unknown when no data available."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = await tracker.get_health_trend()
            assert trend["trend"] == "unknown"
            assert trend["data_points"] == 0

    @pytest.mark.asyncio
    async def test_get_trend_with_experiment_name(self, tracker):
        """Test trend query with experiment name."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = await tracker.get_health_trend(experiment_name="production")
            assert isinstance(trend, dict)

    @pytest.mark.asyncio
    async def test_get_trend_with_hours_param(self, tracker):
        """Test trend query with hours parameter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = await tracker.get_health_trend(hours=48)
            assert isinstance(trend, dict)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_mlflow_import_error(self, tracker):
        """Test handling when MLflow import fails."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_health_run(
                experiment_name="test",
                check_scope="quick",
            ) as ctx:
                assert ctx is not None
                assert ctx.run_id == "no-mlflow"

    @pytest.mark.asyncio
    async def test_handles_critical_health_issues(self, tracker):
        """Test handling of output with critical issues."""
        mock_output = MagicMock()
        mock_output.overall_health_score = 45.0
        mock_output.health_grade = "F"
        mock_output.component_health_score = 0.30
        mock_output.model_health_score = 0.40
        mock_output.pipeline_health_score = 0.50
        mock_output.agent_health_score = 0.60
        mock_output.critical_issues = ["Database unreachable", "API timeout"]
        mock_output.warnings = []
        mock_output.check_latency_ms = 5000

        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_health_result(mock_output)

    @pytest.mark.asyncio
    async def test_handles_perfect_health(self, tracker):
        """Test handling of perfect health score."""
        mock_output = MagicMock()
        mock_output.overall_health_score = 100.0
        mock_output.health_grade = "A"
        mock_output.component_health_score = 1.0
        mock_output.model_health_score = 1.0
        mock_output.pipeline_health_score = 1.0
        mock_output.agent_health_score = 1.0
        mock_output.critical_issues = []
        mock_output.warnings = []
        mock_output.check_latency_ms = 500

        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_health_result(mock_output)

    def test_context_with_minimal_fields(self):
        """Test context creation with minimal fields."""
        ctx = HealthScoreContext(
            run_id="test",
            experiment_name="test",
            check_scope="quick",
        )
        assert ctx.run_id == "test"
        assert ctx.experiment_name == "test"

    def test_metrics_with_edge_values(self):
        """Test metrics with edge case values."""
        # Zero scores
        metrics_zero = HealthScoreMetrics(
            overall_health_score=0.0,
            health_grade="F",
        )
        assert metrics_zero.to_dict()["overall_health_score"] == 0.0

        # Perfect scores
        metrics_perfect = HealthScoreMetrics(
            overall_health_score=100.0,
            health_grade="A",
            component_health_score=1.0,
            model_health_score=1.0,
            pipeline_health_score=1.0,
            agent_health_score=1.0,
        )
        result = metrics_perfect.to_dict()
        assert result["overall_health_score"] == 100.0
        assert result["health_grade_numeric"] == 5


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestTrackerWorkflow:
    """Tests for complete tracker workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_without_mlflow(self, tracker, mock_health_output):
        """Test complete workflow without MLflow."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Start run
            async with tracker.start_health_run(
                experiment_name="test_workflow",
                check_scope="full",
            ) as ctx:
                assert isinstance(ctx, HealthScoreContext)
                assert ctx.experiment_name == "test_workflow"

                # Log result (would normally happen after health check)
                await tracker.log_health_result(mock_health_output)

            # Query history
            history = await tracker.get_health_history(experiment_name="test_workflow")
            assert isinstance(history, list)

            # Query trend
            trend = await tracker.get_health_trend(experiment_name="test_workflow")
            assert isinstance(trend, dict)

    @pytest.mark.asyncio
    async def test_multiple_runs_workflow(self, tracker):
        """Test multiple consecutive runs."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            for i in range(3):
                async with tracker.start_health_run(
                    experiment_name=f"run_{i}",
                    check_scope="quick",
                ) as ctx:
                    assert ctx.experiment_name == f"run_{i}"

    @pytest.mark.asyncio
    async def test_different_scopes_workflow(self, tracker):
        """Test runs with different check scopes."""
        scopes = ["quick", "full", "models", "pipelines", "agents"]
        with patch.object(tracker, "_get_mlflow", return_value=None):
            for scope in scopes:
                async with tracker.start_health_run(
                    experiment_name="scope_test",
                    check_scope=scope,
                ) as ctx:
                    assert ctx.check_scope == scope
