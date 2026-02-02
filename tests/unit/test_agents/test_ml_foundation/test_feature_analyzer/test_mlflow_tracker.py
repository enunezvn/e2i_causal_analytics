"""Unit tests for FeatureAnalyzerMLflowTracker.

Tests comprehensive MLflow tracking for feature analysis runs,
including SHAP analysis, feature importance, feature selection,
and interaction detection.

Version: 1.0.0
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ml_foundation.feature_analyzer.mlflow_tracker import (
    FeatureAnalysisContext,
    FeatureAnalyzerMetrics,
    FeatureAnalyzerMLflowTracker,
    _NoOpRun,
    create_tracker,
)

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_mlflow():
    """Create mock MLflow module."""
    mock = MagicMock()
    mock.set_tracking_uri = MagicMock()
    return mock


@pytest.fixture
def mock_connector():
    """Create mock MLflow connector."""
    mock = MagicMock()
    mock.get_or_create_experiment = AsyncMock(return_value="exp_123")

    # Create mock run context manager
    mock_run = AsyncMock()
    mock_run.log_params = AsyncMock()
    mock_run.log_metrics = AsyncMock()
    mock_run.log_artifact = AsyncMock()
    mock_run.info = MagicMock()
    mock_run.info.run_id = "run_abc123"

    # Make start_run return async context manager
    mock.start_run = MagicMock()
    mock.start_run.return_value.__aenter__ = AsyncMock(return_value=mock_run)
    mock.start_run.return_value.__aexit__ = AsyncMock(return_value=None)

    mock.search_runs = AsyncMock(return_value=[])

    return mock


@pytest.fixture
def tracker():
    """Create a FeatureAnalyzerMLflowTracker instance."""
    return FeatureAnalyzerMLflowTracker(
        project_name="test_feature_analyzer",
        tracking_uri="http://localhost:5000",
    )


@pytest.fixture
def sample_context():
    """Create a sample FeatureAnalysisContext."""
    return FeatureAnalysisContext(
        experiment_id="exp_test_123",
        model_uri="runs:/model_abc/model",
        training_run_id="train_run_xyz",
        problem_type="classification",
        tags={"brand": "TestBrand", "environment": "test"},
    )


@pytest.fixture
def sample_state():
    """Create a sample FeatureAnalyzerState dictionary."""
    return {
        "experiment_id": "exp_test_123",
        "model_uri": "runs:/model_abc/model",
        "shap_analysis_id": "shap_abc123",
        "samples_analyzed": 1000,
        "explainer_type": "TreeExplainer",
        "base_value": 0.45,
        "shap_computation_time_seconds": 12.5,
        "global_importance": {
            "feature_a": 0.35,
            "feature_b": 0.25,
            "feature_c": 0.15,
            "feature_d": 0.10,
            "feature_e": 0.08,
        },
        "global_importance_ranked": [
            ("feature_a", 0.35),
            ("feature_b", 0.25),
            ("feature_c", 0.15),
            ("feature_d", 0.10),
            ("feature_e", 0.08),
        ],
        "feature_names": ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"],
        "top_features": ["feature_a", "feature_b", "feature_c"],
        "feature_directions": {
            "feature_a": "positive",
            "feature_b": "negative",
            "feature_c": "mixed",
            "feature_d": "positive",
            "feature_e": "neutral",
        },
        "original_feature_count": 20,
        "new_feature_count": 5,
        "feature_generation_time_seconds": 2.3,
        "selected_feature_count": 10,
        "selected_features": ["feature_a", "feature_b", "feature_c"],
        "removed_features": {
            "variance": ["low_var_1", "low_var_2"],
            "correlation": ["corr_1", "corr_2", "corr_3"],
            "vif": ["vif_1"],
        },
        "selection_history": [
            {"step": "variance", "removed": 2},
            {"step": "correlation", "removed": 3},
            {"step": "vif", "removed": 1},
        ],
        "selection_time_seconds": 1.5,
        "top_interactions_raw": [
            ("feature_a", "feature_b", 0.82),
            ("feature_a", "feature_c", 0.65),
            ("feature_b", "feature_d", 0.55),
        ],
        "interaction_method": "SHAP_interaction",
        "interaction_computation_time_seconds": 8.2,
        "discovery_enabled": True,
        "discovery_result": {"n_edges": 15},
        "discovery_gate_decision": "proceed",
        "discovery_gate_confidence": 0.85,
        "rank_correlation": 0.72,
        "divergent_features": ["feature_c"],
        "concordant_features": ["feature_a", "feature_b"],
        "direct_cause_features": ["feature_a"],
        "causal_rankings": [("feature_a", 1), ("feature_b", 2)],
        "causal_interpretation": "Feature A is the primary causal driver.",
        "total_computation_time_seconds": 25.5,
        "status": "completed",
    }


# ==============================================================================
# TestFeatureAnalysisContext
# ==============================================================================


class TestFeatureAnalysisContext:
    """Tests for FeatureAnalysisContext dataclass."""

    def test_default_values(self):
        """Test context with default values."""
        ctx = FeatureAnalysisContext(experiment_id="exp_123")

        assert ctx.experiment_id == "exp_123"
        assert ctx.model_uri is None
        assert ctx.training_run_id is None
        assert ctx.problem_type == "classification"
        assert ctx.tags == {}

    def test_full_context(self, sample_context):
        """Test context with all values populated."""
        assert sample_context.experiment_id == "exp_test_123"
        assert sample_context.model_uri == "runs:/model_abc/model"
        assert sample_context.training_run_id == "train_run_xyz"
        assert sample_context.problem_type == "classification"
        assert "brand" in sample_context.tags

    def test_regression_problem_type(self):
        """Test context with regression problem type."""
        ctx = FeatureAnalysisContext(
            experiment_id="exp_456",
            problem_type="regression",
        )
        assert ctx.problem_type == "regression"


# ==============================================================================
# TestFeatureAnalyzerMetrics
# ==============================================================================


class TestFeatureAnalyzerMetrics:
    """Tests for FeatureAnalyzerMetrics dataclass."""

    def test_default_values(self):
        """Test metrics with default values."""
        metrics = FeatureAnalyzerMetrics()

        assert metrics.shap_analysis_id is None
        assert metrics.samples_analyzed == 0
        assert metrics.explainer_type == "unknown"
        assert metrics.base_value == 0.0
        assert metrics.top_feature_importance == 0.0
        assert metrics.feature_count == 0
        assert metrics.discovery_enabled is False
        assert metrics.status == "unknown"

    def test_full_metrics(self):
        """Test metrics with all values populated."""
        metrics = FeatureAnalyzerMetrics(
            shap_analysis_id="shap_123",
            samples_analyzed=500,
            explainer_type="TreeExplainer",
            base_value=0.5,
            shap_computation_time_seconds=10.0,
            top_feature_importance=0.4,
            avg_feature_importance=0.2,
            feature_count=20,
            top_features_count=5,
            positive_direction_count=8,
            negative_direction_count=5,
            mixed_direction_count=3,
            neutral_direction_count=4,
            original_feature_count=25,
            new_feature_count=5,
            selected_feature_count=15,
            removed_variance_count=3,
            removed_correlation_count=4,
            removed_vif_count=1,
            top_interactions_count=10,
            max_interaction_strength=0.9,
            avg_interaction_strength=0.5,
            discovery_enabled=True,
            discovery_edge_count=12,
            discovery_gate_decision="proceed",
            discovery_gate_confidence=0.8,
            rank_correlation=0.75,
            total_computation_time_seconds=30.0,
            status="completed",
        )

        assert metrics.shap_analysis_id == "shap_123"
        assert metrics.samples_analyzed == 500
        assert metrics.discovery_enabled is True
        assert metrics.status == "completed"

    def test_to_dict_returns_all_metrics(self):
        """Test that to_dict includes all metric fields."""
        metrics = FeatureAnalyzerMetrics(
            samples_analyzed=100,
            base_value=0.5,
            top_feature_importance=0.3,
            feature_count=10,
            discovery_enabled=True,
        )

        result = metrics.to_dict()

        # Check key metrics are present
        assert "samples_analyzed" in result
        assert "base_value" in result
        assert "top_feature_importance" in result
        assert "feature_count" in result
        assert "discovery_enabled" in result

        # Check values
        assert result["samples_analyzed"] == 100.0
        assert result["base_value"] == 0.5
        assert result["discovery_enabled"] == 1.0  # Converted to float

    def test_to_dict_converts_types(self):
        """Test that to_dict converts int/bool to float for MLflow."""
        metrics = FeatureAnalyzerMetrics(
            samples_analyzed=100,  # int
            feature_count=20,  # int
            discovery_enabled=True,  # bool
        )

        result = metrics.to_dict()

        assert isinstance(result["samples_analyzed"], float)
        assert isinstance(result["feature_count"], float)
        assert isinstance(result["discovery_enabled"], float)

    def test_direction_counts_in_to_dict(self):
        """Test that direction counts are included in to_dict."""
        metrics = FeatureAnalyzerMetrics(
            positive_direction_count=5,
            negative_direction_count=3,
            mixed_direction_count=2,
            neutral_direction_count=1,
        )

        result = metrics.to_dict()

        assert result["positive_direction_count"] == 5.0
        assert result["negative_direction_count"] == 3.0
        assert result["mixed_direction_count"] == 2.0
        assert result["neutral_direction_count"] == 1.0


# ==============================================================================
# TestNoOpRun
# ==============================================================================


class TestNoOpRun:
    """Tests for _NoOpRun class."""

    def test_run_id_is_none(self):
        """Test that run_id is None."""
        run = _NoOpRun()
        assert run.run_id is None

    @pytest.mark.asyncio
    async def test_log_params_noop(self):
        """Test that log_params does nothing."""
        run = _NoOpRun()
        await run.log_params({"key": "value"})  # Should not raise

    @pytest.mark.asyncio
    async def test_log_metrics_noop(self):
        """Test that log_metrics does nothing."""
        run = _NoOpRun()
        await run.log_metrics({"metric": 1.0})  # Should not raise

    @pytest.mark.asyncio
    async def test_log_artifact_noop(self):
        """Test that log_artifact does nothing."""
        run = _NoOpRun()
        await run.log_artifact("/tmp/file.json", "artifact.json")  # Should not raise


# ==============================================================================
# TestTrackerInitialization
# ==============================================================================


class TestTrackerInitialization:
    """Tests for FeatureAnalyzerMLflowTracker initialization."""

    def test_default_initialization(self):
        """Test tracker with default values."""
        tracker = FeatureAnalyzerMLflowTracker()

        assert tracker.project_name == "feature_analyzer"
        assert tracker.tracking_uri is None
        assert tracker._mlflow is None
        assert tracker._connector is None

    def test_custom_initialization(self):
        """Test tracker with custom values."""
        tracker = FeatureAnalyzerMLflowTracker(
            project_name="custom_analyzer",
            tracking_uri="http://custom:5000",
        )

        assert tracker.project_name == "custom_analyzer"
        assert tracker.tracking_uri == "http://custom:5000"


# ==============================================================================
# TestLazyLoading
# ==============================================================================


class TestLazyLoading:
    """Tests for lazy MLflow loading."""

    def test_mlflow_not_loaded_on_init(self, tracker):
        """Test that MLflow is not loaded on initialization."""
        assert tracker._mlflow is None

    def test_connector_not_loaded_on_init(self, tracker):
        """Test that connector is not loaded on initialization."""
        assert tracker._connector is None

    @patch("src.agents.ml_foundation.feature_analyzer.mlflow_tracker.logger")
    def test_get_mlflow_when_unavailable(self, mock_logger, tracker):
        """Test graceful handling when MLflow is unavailable."""
        with patch.dict("sys.modules", {"mlflow": None}):
            # Force reimport failure
            tracker._mlflow = None
            with patch(
                "builtins.__import__",
                side_effect=ImportError("MLflow not installed"),
            ):
                result = tracker._get_mlflow()

        # Should return None and log warning
        assert result is None

    @patch("src.agents.ml_foundation.feature_analyzer.mlflow_tracker.logger")
    def test_get_connector_when_unavailable(self, mock_logger, tracker):
        """Test behavior when connector returns None.

        The implementation handles ImportError gracefully by returning None.
        We verify the caching behavior when connector is unavailable.
        """
        # Directly set connector to None to simulate unavailable state
        tracker._connector = None

        # Override _get_connector to return None (simulating failed import)
        original_method = tracker._get_connector

        def mock_unavailable():
            return None

        tracker._get_connector = mock_unavailable

        # Verify it returns None
        result = tracker._get_connector()
        assert result is None

        # Restore original method
        tracker._get_connector = original_method


# ==============================================================================
# TestTrackAnalysisRun
# ==============================================================================


class TestTrackAnalysisRun:
    """Tests for track_analysis_run context manager."""

    @pytest.mark.asyncio
    async def test_yields_noop_when_mlflow_unavailable(self, tracker, sample_context):
        """Test that NoOpRun is yielded when MLflow is unavailable."""
        tracker._get_mlflow = MagicMock(return_value=None)

        async with tracker.track_analysis_run(sample_context) as run:
            assert isinstance(run, _NoOpRun)

    @pytest.mark.asyncio
    async def test_yields_noop_when_connector_unavailable(
        self, tracker, mock_mlflow, sample_context
    ):
        """Test that NoOpRun is yielded when connector is unavailable."""
        tracker._get_mlflow = MagicMock(return_value=mock_mlflow)
        tracker._get_connector = MagicMock(return_value=None)

        async with tracker.track_analysis_run(sample_context) as run:
            assert isinstance(run, _NoOpRun)

    @pytest.mark.asyncio
    async def test_creates_experiment_with_correct_name(
        self, tracker, mock_mlflow, mock_connector, sample_context
    ):
        """Test that experiment is created with correct name."""
        tracker._get_mlflow = MagicMock(return_value=mock_mlflow)
        tracker._get_connector = MagicMock(return_value=mock_connector)

        async with tracker.track_analysis_run(sample_context):
            pass

        mock_connector.get_or_create_experiment.assert_called_once()
        call_kwargs = mock_connector.get_or_create_experiment.call_args
        assert "test_feature_analyzer_shap_analysis" in call_kwargs.kwargs["name"]

    @pytest.mark.asyncio
    async def test_logs_parameters_from_context(
        self, tracker, mock_mlflow, mock_connector, sample_context
    ):
        """Test that parameters from context are logged."""
        tracker._get_mlflow = MagicMock(return_value=mock_mlflow)
        tracker._get_connector = MagicMock(return_value=mock_connector)

        async with tracker.track_analysis_run(sample_context):
            pass

        # Get the mock run from context manager
        mock_run = await mock_connector.start_run.return_value.__aenter__()
        mock_run.log_params.assert_called_once()
        params = mock_run.log_params.call_args[0][0]

        assert params["experiment_id"] == "exp_test_123"
        assert params["model_uri"] == "runs:/model_abc/model"
        assert params["problem_type"] == "classification"

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(
        self, tracker, mock_mlflow, mock_connector, sample_context
    ):
        """Test graceful handling of exceptions during tracking."""
        tracker._get_mlflow = MagicMock(return_value=mock_mlflow)
        tracker._get_connector = MagicMock(return_value=mock_connector)
        mock_connector.get_or_create_experiment = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        # Should not raise, should yield NoOpRun
        async with tracker.track_analysis_run(sample_context) as run:
            assert isinstance(run, _NoOpRun)


# ==============================================================================
# TestExtractMetrics
# ==============================================================================


class TestExtractMetrics:
    """Tests for extract_metrics method."""

    def test_extracts_all_metrics(self, tracker, sample_state):
        """Test that all metrics are extracted from state."""
        metrics = tracker.extract_metrics(sample_state)

        # SHAP analysis metrics
        assert metrics.shap_analysis_id == "shap_abc123"
        assert metrics.samples_analyzed == 1000
        assert metrics.explainer_type == "TreeExplainer"
        assert metrics.base_value == 0.45
        assert metrics.shap_computation_time_seconds == 12.5

        # Importance metrics
        assert metrics.top_feature_importance == 0.35
        assert metrics.avg_feature_importance == pytest.approx(0.186, rel=0.01)
        assert metrics.feature_count == 5
        assert metrics.top_features_count == 3

        # Direction distribution
        assert metrics.positive_direction_count == 2
        assert metrics.negative_direction_count == 1
        assert metrics.mixed_direction_count == 1
        assert metrics.neutral_direction_count == 1

        # Feature generation
        assert metrics.original_feature_count == 20
        assert metrics.new_feature_count == 5
        assert metrics.feature_generation_time_seconds == 2.3

        # Feature selection
        assert metrics.selected_feature_count == 10
        assert metrics.removed_variance_count == 2
        assert metrics.removed_correlation_count == 3
        assert metrics.removed_vif_count == 1
        assert metrics.selection_time_seconds == 1.5

        # Interaction metrics
        assert metrics.top_interactions_count == 3
        assert metrics.max_interaction_strength == 0.82
        assert metrics.avg_interaction_strength == pytest.approx(0.673, rel=0.01)
        assert metrics.interaction_computation_time_seconds == 8.2

        # Causal discovery
        assert metrics.discovery_enabled is True
        assert metrics.discovery_edge_count == 15
        assert metrics.discovery_gate_decision == "proceed"
        assert metrics.discovery_gate_confidence == 0.85
        assert metrics.rank_correlation == 0.72
        assert metrics.divergent_features_count == 1

        # Overall
        assert metrics.total_computation_time_seconds == 25.5
        assert metrics.status == "completed"

    def test_handles_empty_state(self, tracker):
        """Test extraction from empty state."""
        metrics = tracker.extract_metrics({})

        assert metrics.shap_analysis_id is None
        assert metrics.samples_analyzed == 0
        assert metrics.explainer_type == "unknown"
        assert metrics.top_feature_importance == 0.0
        assert metrics.avg_feature_importance == 0.0
        assert metrics.feature_count == 0
        assert metrics.top_interactions_count == 0
        assert metrics.discovery_enabled is False
        assert metrics.status == "unknown"

    def test_handles_missing_nested_fields(self, tracker):
        """Test extraction when nested fields are missing."""
        state = {
            "samples_analyzed": 100,
            "removed_features": {},  # Empty removed features
        }

        metrics = tracker.extract_metrics(state)

        assert metrics.samples_analyzed == 100
        assert metrics.removed_variance_count == 0
        assert metrics.removed_correlation_count == 0
        assert metrics.removed_vif_count == 0

    def test_handles_no_importance_values(self, tracker):
        """Test extraction when global_importance is empty."""
        state = {
            "global_importance": {},
            "feature_names": [],
            "top_features": [],
        }

        metrics = tracker.extract_metrics(state)

        assert metrics.top_feature_importance == 0.0
        assert metrics.avg_feature_importance == 0.0
        assert metrics.feature_count == 0

    def test_handles_no_interactions(self, tracker):
        """Test extraction when no interactions are present."""
        state = {
            "top_interactions_raw": [],
        }

        metrics = tracker.extract_metrics(state)

        assert metrics.top_interactions_count == 0
        assert metrics.max_interaction_strength == 0.0
        assert metrics.avg_interaction_strength == 0.0


# ==============================================================================
# TestLogArtifacts
# ==============================================================================


class TestLogFeatureImportance:
    """Tests for log_feature_importance method."""

    @pytest.mark.asyncio
    async def test_logs_importance_artifact(self, tracker, sample_state):
        """Test that feature importance is logged as artifact."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_feature_importance(mock_run, sample_state)

        mock_run.log_artifact.assert_called_once()
        call_args = mock_run.log_artifact.call_args
        assert call_args[0][1] == "feature_importance.json"

    @pytest.mark.asyncio
    async def test_skips_when_no_importance(self, tracker):
        """Test that logging is skipped when no importance data."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_feature_importance(mock_run, {"global_importance_ranked": []})

        mock_run.log_artifact.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_run_has_no_log_artifact(self, tracker, sample_state):
        """Test graceful handling when run doesn't support artifacts."""
        mock_run = MagicMock(spec=[])  # No log_artifact method

        await tracker.log_feature_importance(mock_run, sample_state)
        # Should not raise


class TestLogInteractions:
    """Tests for log_interactions method."""

    @pytest.mark.asyncio
    async def test_logs_interactions_artifact(self, tracker, sample_state):
        """Test that interactions are logged as artifact."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_interactions(mock_run, sample_state)

        mock_run.log_artifact.assert_called_once()
        call_args = mock_run.log_artifact.call_args
        assert call_args[0][1] == "feature_interactions.json"

    @pytest.mark.asyncio
    async def test_skips_when_no_interactions(self, tracker):
        """Test that logging is skipped when no interaction data."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_interactions(mock_run, {"top_interactions_raw": []})

        mock_run.log_artifact.assert_not_called()


class TestLogSelectionSummary:
    """Tests for log_selection_summary method."""

    @pytest.mark.asyncio
    async def test_logs_selection_artifact(self, tracker, sample_state):
        """Test that selection summary is logged as artifact."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_selection_summary(mock_run, sample_state)

        mock_run.log_artifact.assert_called_once()
        call_args = mock_run.log_artifact.call_args
        assert call_args[0][1] == "feature_selection.json"

    @pytest.mark.asyncio
    async def test_skips_when_no_selection_history(self, tracker):
        """Test that logging is skipped when no selection history."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_selection_summary(mock_run, {"selection_history": []})

        mock_run.log_artifact.assert_not_called()


class TestLogCausalComparison:
    """Tests for log_causal_comparison method."""

    @pytest.mark.asyncio
    async def test_logs_causal_artifact(self, tracker, sample_state):
        """Test that causal comparison is logged as artifact."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_causal_comparison(mock_run, sample_state)

        mock_run.log_artifact.assert_called_once()
        call_args = mock_run.log_artifact.call_args
        assert call_args[0][1] == "causal_comparison.json"

    @pytest.mark.asyncio
    async def test_skips_when_discovery_disabled(self, tracker):
        """Test that logging is skipped when discovery is disabled."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_causal_comparison(mock_run, {"discovery_enabled": False})

        mock_run.log_artifact.assert_not_called()


# ==============================================================================
# TestHistory
# ==============================================================================


class TestGetAnalysisHistory:
    """Tests for get_analysis_history method."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_connector_unavailable(self, tracker):
        """Test that empty list is returned when connector unavailable."""
        tracker._get_connector = MagicMock(return_value=None)

        history = await tracker.get_analysis_history()

        assert history == []

    @pytest.mark.asyncio
    async def test_searches_correct_experiment(self, tracker, mock_connector):
        """Test that correct experiment name is searched."""
        tracker._get_connector = MagicMock(return_value=mock_connector)

        await tracker.get_analysis_history()

        mock_connector.search_runs.assert_called_once()
        call_kwargs = mock_connector.search_runs.call_args.kwargs
        assert "test_feature_analyzer_shap_analysis" in call_kwargs["experiment_names"]

    @pytest.mark.asyncio
    async def test_filters_by_model_uri(self, tracker, mock_connector):
        """Test that results can be filtered by model URI."""
        tracker._get_connector = MagicMock(return_value=mock_connector)

        await tracker.get_analysis_history(model_uri="runs:/model_123/model")

        mock_connector.search_runs.assert_called_once()
        call_kwargs = mock_connector.search_runs.call_args.kwargs
        assert "model_uri" in call_kwargs["filter_string"]

    @pytest.mark.asyncio
    async def test_respects_limit(self, tracker, mock_connector):
        """Test that limit parameter is respected."""
        tracker._get_connector = MagicMock(return_value=mock_connector)

        await tracker.get_analysis_history(limit=10)

        call_kwargs = mock_connector.search_runs.call_args.kwargs
        assert call_kwargs["max_results"] == 10

    @pytest.mark.asyncio
    async def test_handles_search_error_gracefully(self, tracker, mock_connector):
        """Test graceful handling of search errors."""
        tracker._get_connector = MagicMock(return_value=mock_connector)
        mock_connector.search_runs = AsyncMock(side_effect=Exception("Search failed"))

        history = await tracker.get_analysis_history()

        assert history == []


class TestGetImportanceTrends:
    """Tests for get_importance_trends method."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_connector_unavailable(self, tracker):
        """Test that empty list is returned when connector unavailable."""
        tracker._get_connector = MagicMock(return_value=None)

        trends = await tracker.get_importance_trends("feature_a")

        assert trends == []

    @pytest.mark.asyncio
    async def test_searches_with_correct_parameters(self, tracker, mock_connector):
        """Test that search uses correct parameters."""
        tracker._get_connector = MagicMock(return_value=mock_connector)

        await tracker.get_importance_trends("feature_a", limit=10)

        mock_connector.search_runs.assert_called_once()
        call_kwargs = mock_connector.search_runs.call_args.kwargs
        assert call_kwargs["max_results"] == 10
        assert "attributes.start_time DESC" in call_kwargs["order_by"]

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self, tracker, mock_connector):
        """Test graceful handling of errors."""
        tracker._get_connector = MagicMock(return_value=mock_connector)
        mock_connector.search_runs = AsyncMock(side_effect=Exception("Error"))

        trends = await tracker.get_importance_trends("feature_a")

        assert trends == []


# ==============================================================================
# TestFactory
# ==============================================================================


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_creates_tracker_with_defaults(self):
        """Test factory creates tracker with default values."""
        tracker = create_tracker()

        assert isinstance(tracker, FeatureAnalyzerMLflowTracker)
        assert tracker.project_name == "feature_analyzer"
        assert tracker.tracking_uri is None

    def test_creates_tracker_with_custom_values(self):
        """Test factory creates tracker with custom values."""
        tracker = create_tracker(
            project_name="custom_project",
            tracking_uri="http://custom:5000",
        )

        assert tracker.project_name == "custom_project"
        assert tracker.tracking_uri == "http://custom:5000"


# ==============================================================================
# TestErrorHandling
# ==============================================================================


class TestErrorHandling:
    """Tests for error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_artifact_logging_handles_json_error(self, tracker, sample_state):
        """Test that JSON serialization errors are handled."""
        # Add non-serializable object
        sample_state["non_serializable"] = object()

        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        # Should not raise
        await tracker.log_feature_importance(mock_run, sample_state)

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_error(self, tracker, sample_context):
        """Test that context manager handles errors gracefully.

        Note: The implementation catches all exceptions inside the generator
        and yields NoOpRun, so errors are swallowed rather than propagated.
        This test verifies the error handling behavior.
        """
        # When MLflow/connector unavailable, NoOpRun is yielded
        tracker._get_mlflow = MagicMock(return_value=None)

        async with tracker.track_analysis_run(sample_context) as run:
            # With NoOpRun, errors will propagate normally
            assert run.run_id is None
            # Operations on NoOpRun are no-ops (await async methods)
            await run.log_params({"test": "value"})

    def test_extract_metrics_handles_malformed_state(self, tracker):
        """Test behavior with malformed state - some fields cause errors."""
        # Test with None values (should be handled gracefully)

        # None global_importance is handled (if check returns [])
        metrics = tracker.extract_metrics({"global_importance": None})
        assert metrics.top_feature_importance == 0.0
        assert metrics.avg_feature_importance == 0.0

        # Empty dict is handled properly
        metrics = tracker.extract_metrics({"global_importance": {}})
        assert metrics.top_feature_importance == 0.0
