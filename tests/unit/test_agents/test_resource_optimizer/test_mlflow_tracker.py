"""Unit tests for ResourceOptimizer MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting optimization runs
- Metric extraction and logging
- Parameter logging
- Artifact logging (JSON)
- Historical query methods
- ROI trends analysis
- Graceful degradation when MLflow unavailable

Part of observability audit remediation - G09 Phase 2.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.resource_optimizer.mlflow_tracker import (
    EXPERIMENT_PREFIX,
    OptimizationContext,
    ResourceOptimizerMetrics,
    ResourceOptimizerMLflowTracker,
    create_tracker,
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
    mock.set_tag = MagicMock()
    mock.set_tags = MagicMock()
    mock.get_experiment_by_name = MagicMock(
        return_value=MagicMock(experiment_id="test_exp_id")
    )
    mock.search_runs = MagicMock(
        return_value=MagicMock(iterrows=MagicMock(return_value=iter([])))
    )
    return mock


@pytest.fixture
def tracker():
    """Create a ResourceOptimizerMLflowTracker instance."""
    return ResourceOptimizerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample OptimizationContext."""
    return OptimizationContext(
        run_id="run_123",
        experiment_name="budget_allocation",
        resource_type="budget",
        objective="maximize_outcome",
        brand="remibrutinib",
        region="Northeast",
    )


@pytest.fixture
def sample_state():
    """Create a sample ResourceOptimizerState dict."""
    return {
        "objective_value": 450000.0,
        "projected_total_outcome": 450000.0,
        "projected_roi": 2.25,
        "solver_status": "optimal",
        "solve_time_ms": 150,
        "optimal_allocations": [
            {
                "entity_id": "territory_northeast",
                "current_allocation": 50000,
                "optimized_allocation": 65000,
                "change": 15000,
                "change_percentage": 30,
            },
            {
                "entity_id": "territory_southeast",
                "current_allocation": 40000,
                "optimized_allocation": 35000,
                "change": -5000,
                "change_percentage": -12.5,
            },
            {
                "entity_id": "territory_west",
                "current_allocation": 30000,
                "optimized_allocation": 30000,
                "change": 0,
                "change_percentage": 0,
            },
        ],
        "scenarios": [
            {"name": "conservative", "roi": 1.8},
            {"name": "moderate", "roi": 2.1},
            {"name": "aggressive", "roi": 2.5},
        ],
        "formulation_latency_ms": 50,
        "optimization_latency_ms": 100,
        "total_latency_ms": 200,
        "sensitivity_analysis": {"parameter": "budget", "impact": 0.15},
        "impact_by_segment": {"northeast": 195000},
        "recommendations": ["Increase allocation to Northeast by 15000"],
    }


# =============================================================================
# CONSTANT TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_experiment_prefix(self):
        """Test experiment prefix is set correctly."""
        assert EXPERIMENT_PREFIX == "e2i_causal/resource_optimizer"


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestOptimizationContext:
    """Tests for OptimizationContext dataclass."""

    def test_context_creation_required_fields(self):
        """Test context creation with required fields."""
        ctx = OptimizationContext(
            run_id="run_123",
            experiment_name="budget_allocation",
            resource_type="budget",
            objective="maximize_outcome",
        )
        assert ctx.run_id == "run_123"
        assert ctx.experiment_name == "budget_allocation"
        assert ctx.resource_type == "budget"
        assert ctx.objective == "maximize_outcome"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.brand == "remibrutinib"
        assert sample_context.region == "Northeast"

    def test_context_default_values(self):
        """Test context default values for optional fields."""
        ctx = OptimizationContext(
            run_id="run",
            experiment_name="exp_name",
            resource_type="budget",
            objective="maximize_outcome",
        )
        assert ctx.brand is None
        assert ctx.region is None
        assert ctx.timestamp is not None  # Auto-generated


class TestResourceOptimizerMetrics:
    """Tests for ResourceOptimizerMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = ResourceOptimizerMetrics(
            objective_value=450000.0,
            projected_outcome=450000.0,
            projected_roi=2.25,
            entities_optimized=50,
            entities_increased=20,
            entities_decreased=10,
        )
        assert metrics.objective_value == 450000.0
        assert metrics.projected_roi == 2.25
        assert metrics.entities_optimized == 50

    def test_metrics_defaults(self):
        """Test metrics default values."""
        metrics = ResourceOptimizerMetrics()
        assert metrics.objective_value is None
        assert metrics.projected_roi is None
        assert metrics.entities_optimized == 0
        assert metrics.solver_status == "unknown"
        assert metrics.total_latency_ms == 0

    def test_metrics_to_dict(self):
        """Test metrics to_dict conversion."""
        metrics = ResourceOptimizerMetrics(
            objective_value=450000.0,
            projected_roi=2.25,
            entities_optimized=50,
            entities_increased=20,
            entities_decreased=10,
            entities_unchanged=20,
            solve_time_ms=150,
            total_latency_ms=200,
        )
        result = metrics.to_dict()

        assert result["objective_value"] == 450000.0
        assert result["projected_roi"] == 2.25
        assert result["entities_optimized"] == 50
        assert result["entities_increased"] == 20
        assert result["solve_time_ms"] == 150

    def test_metrics_to_dict_excludes_none(self):
        """Test that to_dict excludes None values for optional fields."""
        metrics = ResourceOptimizerMetrics(
            entities_optimized=50,
            solve_time_ms=150,
        )
        result = metrics.to_dict()

        # objective_value is None, should not be in result
        assert "objective_value" not in result
        # Required fields should be present
        assert "entities_optimized" in result


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, ResourceOptimizerMLflowTracker)

    def test_tracker_default_config(self, tracker):
        """Test tracker default configuration."""
        assert tracker.enable_artifact_logging is True
        assert tracker._current_run_id is None

    def test_tracker_custom_config(self):
        """Test tracker with custom configuration."""
        tracker = ResourceOptimizerMLflowTracker(
            tracking_uri="http://custom:5000",
            enable_artifact_logging=False,
        )
        assert tracker._tracking_uri == "http://custom:5000"
        assert tracker.enable_artifact_logging is False

    def test_tracker_lazy_mlflow_loading(self, tracker):
        """Test MLflow is lazily loaded."""
        assert tracker._mlflow is None


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartOptimizationRun:
    """Tests for start_optimization_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow(self, tracker):
        """Test start_optimization_run when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_optimization_run(
                experiment_name="budget_allocation",
                resource_type="budget",
                objective="maximize_outcome",
            ) as ctx:
                assert ctx is not None
                assert isinstance(ctx, OptimizationContext)
                assert ctx.run_id == "no-mlflow"

    @pytest.mark.asyncio
    async def test_start_run_with_mlflow(self, tracker, mock_mlflow):
        """Test start_optimization_run with MLflow available."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_optimization_run(
                experiment_name="budget_allocation",
                resource_type="budget",
                objective="maximize_outcome",
                brand="remibrutinib",
            ) as ctx:
                assert ctx is not None
                assert ctx.run_id == "test_run_123"

    @pytest.mark.asyncio
    async def test_start_run_accepts_optional_params(self, tracker):
        """Test that start_optimization_run accepts optional parameters."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_optimization_run(
                experiment_name="budget_allocation",
                resource_type="budget",
                objective="maximize_roi",
                solver_type="milp",
                brand="remibrutinib",
                region="Northeast",
                tags={"custom_tag": "value"},
            ) as ctx:
                assert ctx is not None
                assert ctx.resource_type == "budget"
                assert ctx.objective == "maximize_roi"
                assert ctx.brand == "remibrutinib"

    @pytest.mark.asyncio
    async def test_start_run_handles_experiment_error(self, tracker, mock_mlflow):
        """Test graceful handling of experiment creation errors."""
        mock_mlflow.get_experiment_by_name.side_effect = Exception("Connection failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_optimization_run(
                experiment_name="test",
                resource_type="budget",
                objective="maximize_outcome",
            ) as ctx:
                assert ctx.run_id == "experiment-error"


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_state(self, tracker, sample_state):
        """Test metric extraction from state dict."""
        metrics = tracker._extract_metrics(sample_state)
        assert isinstance(metrics, ResourceOptimizerMetrics)
        assert metrics.objective_value == 450000.0
        assert metrics.projected_roi == 2.25
        assert metrics.solver_status == "optimal"
        assert metrics.entities_optimized == 3
        assert metrics.entities_increased == 1
        assert metrics.entities_decreased == 1
        assert metrics.entities_unchanged == 1

    def test_extract_metrics_handles_empty_state(self, tracker):
        """Test metric extraction with empty state."""
        metrics = tracker._extract_metrics({})
        assert isinstance(metrics, ResourceOptimizerMetrics)
        assert metrics.objective_value is None
        assert metrics.solver_status == "unknown"

    def test_extract_metrics_handles_missing_allocations(self, tracker):
        """Test metric extraction with missing allocations."""
        state = {"objective_value": 450000.0}
        metrics = tracker._extract_metrics(state)
        assert metrics.objective_value == 450000.0
        assert metrics.entities_optimized == 0

    def test_extract_metrics_calculates_scenario_metrics(self, tracker, sample_state):
        """Test scenario metrics calculation."""
        metrics = tracker._extract_metrics(sample_state)
        assert metrics.scenarios_analyzed == 3
        assert metrics.best_scenario_roi == 2.5


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogOptimizationResult:
    """Tests for log_optimization_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_state):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Should not raise
            await tracker.log_optimization_result(sample_state)

    @pytest.mark.asyncio
    async def test_log_result_without_run_id(self, tracker, sample_state, mock_mlflow):
        """Test logging result when no run is active."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            tracker._current_run_id = None
            # Should not raise
            await tracker.log_optimization_result(sample_state)
            mock_mlflow.log_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_result_with_mlflow(self, tracker, sample_state, mock_mlflow):
        """Test logging result with MLflow available."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            tracker._current_run_id = "test_run_123"
            tracker.enable_artifact_logging = False  # Skip artifacts

            await tracker.log_optimization_result(sample_state)

            mock_mlflow.log_metrics.assert_called_once()
            mock_mlflow.set_tags.assert_called_once()


# =============================================================================
# ARTIFACT LOGGING TESTS
# =============================================================================


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_without_mlflow(self, tracker, sample_state):
        """Test artifact logging when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Should not raise
            await tracker._log_artifacts(sample_state)

    @pytest.mark.asyncio
    async def test_log_artifacts_with_mlflow(self, tracker, sample_state, mock_mlflow):
        """Test artifact logging with MLflow available."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                mock_tmpdir.return_value.__enter__ = MagicMock(return_value="/tmp/test")
                mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

                with patch("builtins.open", MagicMock()):
                    await tracker._log_artifacts(sample_state)


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetOptimizationHistory:
    """Tests for get_optimization_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = await tracker.get_optimization_history()
            assert history == []

    @pytest.mark.asyncio
    async def test_get_history_no_experiment(self, tracker, mock_mlflow):
        """Test history query when experiment doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            history = await tracker.get_optimization_history()
            assert history == []

    @pytest.mark.asyncio
    async def test_get_history_with_results(self, tracker, mock_mlflow):
        """Test history query with results."""
        mock_df = MagicMock()
        mock_df.iterrows.return_value = iter([
            (0, {
                "run_id": "run_1",
                "start_time": datetime.now(timezone.utc),
            }),
        ])
        mock_mlflow.search_runs.return_value = mock_df

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            history = await tracker.get_optimization_history()
            assert len(history) == 1
            assert history[0]["run_id"] == "run_1"


class TestGetROITrends:
    """Tests for get_roi_trends method."""

    @pytest.mark.asyncio
    async def test_get_trends_empty_history(self, tracker):
        """Test ROI trends with empty history."""
        with patch.object(tracker, "get_optimization_history", return_value=[]):
            trends = await tracker.get_roi_trends()
            assert trends["total_optimizations"] == 0
            assert trends["avg_roi"] == 0.0
            assert trends["trend"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_get_trends_with_history(self, tracker):
        """Test ROI trends with history data."""
        history = [
            {"projected_roi": 2.0, "objective_value": 400000, "resource_type": "budget"},
            {"projected_roi": 2.2, "objective_value": 440000, "resource_type": "budget"},
            {"projected_roi": 2.4, "objective_value": 480000, "resource_type": "budget"},
            {"projected_roi": 2.6, "objective_value": 520000, "resource_type": "budget"},
        ]
        with patch.object(tracker, "get_optimization_history", return_value=history):
            trends = await tracker.get_roi_trends()
            assert trends["total_optimizations"] == 4
            assert trends["avg_roi"] == 2.3
            assert trends["max_roi"] == 2.6
            assert trends["min_roi"] == 2.0
            assert trends["trend"] == "improving"  # Second half > first half * 1.1


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_create_tracker_default(self):
        """Test create_tracker with defaults."""
        tracker = create_tracker()
        assert isinstance(tracker, ResourceOptimizerMLflowTracker)

    def test_create_tracker_custom_uri(self):
        """Test create_tracker with custom URI."""
        tracker = create_tracker(tracking_uri="http://custom:5000")
        assert tracker._tracking_uri == "http://custom:5000"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_logging_error(self, tracker, mock_mlflow):
        """Test graceful handling of logging errors."""
        mock_mlflow.log_metrics.side_effect = Exception("Logging failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            tracker._current_run_id = "test_run"
            tracker.enable_artifact_logging = False

            # Should not raise
            await tracker.log_optimization_result({"objective_value": 450000})

    @pytest.mark.asyncio
    async def test_handles_artifact_error(self, tracker, mock_mlflow):
        """Test graceful handling of artifact logging errors."""
        mock_mlflow.log_artifact.side_effect = Exception("Artifact error")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                mock_tmpdir.return_value.__enter__ = MagicMock(return_value="/tmp/test")
                mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

                # Should not raise
                await tracker._log_artifacts({"optimal_allocations": [{}]})

    @pytest.mark.asyncio
    async def test_handles_history_query_error(self, tracker, mock_mlflow):
        """Test graceful handling of history query errors."""
        mock_mlflow.search_runs.side_effect = Exception("Query failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            # Should not raise, returns empty list
            history = await tracker.get_optimization_history()
            assert history == []
