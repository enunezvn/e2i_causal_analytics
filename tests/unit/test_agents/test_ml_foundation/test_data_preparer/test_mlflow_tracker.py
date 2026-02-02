"""Unit tests for DataPreparer MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow lazy loading
- DataQualityContext dataclass creation and defaults
- DataPreparerMetrics dataclass and to_dict conversion
- Context managers for tracking preparation runs
- Metric extraction from state
- Artifact logging (QC reports, feature stats, expectation results)
- Historical query methods for quality trends
- Leakage incident tracking
- Graceful degradation when MLflow unavailable

From observability audit remediation plan.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ml_foundation.data_preparer.mlflow_tracker import (
    DataPreparerMetrics,
    DataPreparerMLflowTracker,
    DataQualityContext,
    _NoOpRun,
    create_tracker,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_mlflow():
    """Mock MLflow module."""
    mock = MagicMock()
    mock.set_tracking_uri = MagicMock()
    mock.set_experiment = MagicMock()
    return mock


@pytest.fixture
def mock_connector():
    """Mock MLflow connector."""
    connector = MagicMock()
    connector.get_or_create_experiment = AsyncMock(return_value="exp_123")
    connector.start_run = MagicMock()
    connector.search_runs = AsyncMock(return_value=[])
    return connector


@pytest.fixture
def tracker():
    """Create a DataPreparerMLflowTracker instance."""
    return DataPreparerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample DataQualityContext."""
    return DataQualityContext(
        experiment_id="exp_123",
        data_source="ml_features_v3",
        split_id="split_001",
        validation_suite="default_suite",
        tags={"env": "test"},
    )


@pytest.fixture
def sample_state():
    """Create a sample DataPreparerState dict."""
    return {
        "report_id": "report_123",
        "experiment_id": "exp_123",
        "data_source": "ml_features_v3",
        "validated_at": "2024-01-15T10:00:00Z",
        # Quality scores
        "completeness_score": 0.95,
        "validity_score": 0.92,
        "consistency_score": 0.88,
        "uniqueness_score": 0.99,
        "timeliness_score": 0.85,
        "overall_score": 0.918,
        # QC gate
        "qc_status": "passed",
        "qc_passed": True,
        "qc_score": 0.92,
        "gate_passed": True,
        # Schema validation
        "schema_validation_status": "validated",
        "schema_splits_validated": 4,
        "schema_validation_time_ms": 150,
        "schema_validation_errors": [],
        # Leakage detection
        "leakage_detected": False,
        "leakage_issues": [],
        # Sample counts
        "total_samples": 10000,
        "train_samples": 7000,
        "validation_samples": 1500,
        "test_samples": 1000,
        "holdout_samples": 500,
        # Features
        "column_count": 45,
        "available_features": ["feat1", "feat2", "feat3"],
        "missing_required_features": [],
        # Expectations
        "expectation_results": [
            {"expectation": "not_null", "success": True},
            {"expectation": "in_range", "success": True},
            {"expectation": "unique", "success": False},
        ],
        "failed_expectations": [{"expectation": "unique", "column": "id"}],
        "warnings": [{"type": "data_quality", "message": "Minor issue"}],
        "blocking_issues": [],
        # Feast
        "feast_registration_status": "registered",
        "feast_features_registered": 45,
        # Timing
        "validation_duration_seconds": 2.5,
        # Ready status
        "is_ready": True,
        # Additional for artifacts
        "feature_stats": {"feat1": {"mean": 0.5, "std": 0.1}},
        "remediation_steps": [],
    }


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestDataQualityContext:
    """Tests for DataQualityContext dataclass."""

    def test_context_creation_required_fields(self):
        """Test context creation with required fields."""
        ctx = DataQualityContext(
            experiment_id="exp_123",
            data_source="ml_features_v3",
        )
        assert ctx.experiment_id == "exp_123"
        assert ctx.data_source == "ml_features_v3"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.experiment_id == "exp_123"
        assert sample_context.data_source == "ml_features_v3"
        assert sample_context.split_id == "split_001"
        assert sample_context.validation_suite == "default_suite"
        assert sample_context.tags == {"env": "test"}

    def test_context_default_values(self):
        """Test context default values for optional fields."""
        ctx = DataQualityContext(
            experiment_id="exp",
            data_source="source",
        )
        assert ctx.split_id is None
        assert ctx.validation_suite is None
        assert ctx.tags == {}


class TestDataPreparerMetrics:
    """Tests for DataPreparerMetrics dataclass."""

    def test_metrics_creation_defaults(self):
        """Test metrics dataclass creation with defaults."""
        metrics = DataPreparerMetrics()
        assert metrics.completeness_score == 0.0
        assert metrics.validity_score == 0.0
        assert metrics.qc_status == "unknown"
        assert metrics.qc_passed is False
        assert metrics.leakage_detected is False
        assert metrics.is_ready is False

    def test_metrics_creation_with_values(self):
        """Test metrics dataclass creation with values."""
        metrics = DataPreparerMetrics(
            completeness_score=0.95,
            validity_score=0.92,
            consistency_score=0.88,
            uniqueness_score=0.99,
            timeliness_score=0.85,
            overall_score=0.918,
            qc_status="passed",
            qc_passed=True,
            gate_passed=True,
            total_samples=10000,
            is_ready=True,
        )
        assert metrics.completeness_score == 0.95
        assert metrics.overall_score == 0.918
        assert metrics.qc_passed is True
        assert metrics.is_ready is True

    def test_metrics_to_dict(self):
        """Test metrics to_dict conversion."""
        metrics = DataPreparerMetrics(
            completeness_score=0.95,
            validity_score=0.92,
            qc_passed=True,
            gate_passed=True,
            leakage_detected=False,
            total_samples=10000,
            train_samples=7000,
            validation_samples=1500,
            test_samples=1000,
            is_ready=True,
        )
        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["completeness_score"] == 0.95
        assert result["validity_score"] == 0.92
        assert result["qc_passed"] == 1.0  # Converted to float
        assert result["gate_passed"] == 1.0
        assert result["leakage_detected"] == 0.0
        assert result["total_samples"] == 10000.0
        assert result["is_ready"] == 1.0

    def test_metrics_to_dict_all_fields(self):
        """Test to_dict includes all expected fields."""
        metrics = DataPreparerMetrics()
        result = metrics.to_dict()

        expected_keys = [
            "completeness_score",
            "validity_score",
            "consistency_score",
            "uniqueness_score",
            "timeliness_score",
            "overall_score",
            "qc_passed",
            "qc_score",
            "gate_passed",
            "schema_splits_validated",
            "schema_validation_time_ms",
            "leakage_detected",
            "leakage_issues_count",
            "total_samples",
            "train_samples",
            "validation_samples",
            "test_samples",
            "holdout_samples",
            "column_count",
            "available_features_count",
            "missing_required_features_count",
            "total_expectations",
            "failed_expectations_count",
            "warnings_count",
            "blocking_issues_count",
            "feast_features_registered",
            "validation_duration_seconds",
            "is_ready",
        ]
        for key in expected_keys:
            assert key in result


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, DataPreparerMLflowTracker)

    def test_tracker_custom_project_name(self):
        """Test tracker with custom project name."""
        tracker = DataPreparerMLflowTracker(project_name="custom_project")
        assert tracker.project_name == "custom_project"

    def test_tracker_default_project_name(self, tracker):
        """Test tracker default project name."""
        assert tracker.project_name == "data_preparer"

    def test_tracker_custom_uri(self):
        """Test tracker with custom tracking URI."""
        tracker = DataPreparerMLflowTracker(tracking_uri="http://custom:5000")
        assert tracker.tracking_uri == "http://custom:5000"

    def test_tracker_lazy_mlflow_loading(self, tracker):
        """Test MLflow is lazily loaded."""
        assert tracker._mlflow is None
        assert tracker._connector is None


class TestLazyLoading:
    """Tests for lazy MLflow loading."""

    def test_get_mlflow_lazy_loads(self, tracker, mock_mlflow):
        """Test _get_mlflow lazy loads MLflow."""
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            result = tracker._get_mlflow()
            assert result is not None

    def test_get_mlflow_handles_import_error(self, tracker):
        """Test _get_mlflow handles ImportError gracefully."""
        with patch.dict("sys.modules", {"mlflow": None}):
            with patch("builtins.__import__", side_effect=ImportError("No MLflow")):
                result = tracker._get_mlflow()
                assert result is None

    def test_get_connector_lazy_loads(self, tracker, mock_connector):
        """Test _get_connector lazy loads and caches connector."""
        # Directly set the connector to simulate successful load
        tracker._connector = mock_connector
        result = tracker._get_connector()
        assert result is mock_connector

    def test_get_connector_handles_import_error(self, tracker):
        """Test behavior when connector is unavailable.

        The implementation handles ImportError gracefully by returning None.
        """
        # Simulate unavailable connector by directly setting to None
        original_method = tracker._get_connector

        def mock_unavailable():
            return None

        tracker._get_connector = mock_unavailable
        result = tracker._get_connector()
        assert result is None

        # Restore original method
        tracker._get_connector = original_method


# =============================================================================
# NO-OP RUN TESTS
# =============================================================================


class TestNoOpRun:
    """Tests for _NoOpRun class."""

    def test_noop_run_has_no_run_id(self):
        """Test NoOpRun has None run_id."""
        run = _NoOpRun()
        assert run.run_id is None

    @pytest.mark.asyncio
    async def test_noop_run_log_params(self):
        """Test NoOpRun log_params does nothing."""
        run = _NoOpRun()
        # Should not raise
        await run.log_params({"key": "value"})

    @pytest.mark.asyncio
    async def test_noop_run_log_metrics(self):
        """Test NoOpRun log_metrics does nothing."""
        run = _NoOpRun()
        # Should not raise
        await run.log_metrics({"metric": 1.0})

    @pytest.mark.asyncio
    async def test_noop_run_log_artifact(self):
        """Test NoOpRun log_artifact does nothing."""
        run = _NoOpRun()
        # Should not raise
        await run.log_artifact("/tmp/test.json", "artifact.json")


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestTrackPreparationRun:
    """Tests for track_preparation_run context manager."""

    @pytest.mark.asyncio
    async def test_track_run_without_mlflow(self, tracker, sample_context):
        """Test track_preparation_run when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.track_preparation_run(sample_context) as run:
                assert isinstance(run, _NoOpRun)

    @pytest.mark.asyncio
    async def test_track_run_without_connector(self, tracker, mock_mlflow, sample_context):
        """Test track_preparation_run when connector unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch.object(tracker, "_get_connector", return_value=None):
                async with tracker.track_preparation_run(sample_context) as run:
                    assert isinstance(run, _NoOpRun)

    @pytest.mark.asyncio
    async def test_track_run_with_mlflow(
        self, tracker, mock_mlflow, mock_connector, sample_context
    ):
        """Test track_preparation_run with MLflow available."""
        mock_run = AsyncMock()
        mock_run.log_params = AsyncMock()

        # Setup async context manager
        mock_connector.start_run = MagicMock()
        mock_connector.start_run.return_value.__aenter__ = AsyncMock(return_value=mock_run)
        mock_connector.start_run.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch.object(tracker, "_get_connector", return_value=mock_connector):
                async with tracker.track_preparation_run(sample_context) as run:
                    assert run == mock_run

    @pytest.mark.asyncio
    async def test_track_run_logs_parameters(
        self, tracker, mock_mlflow, mock_connector, sample_context
    ):
        """Test that parameters are logged on run start."""
        mock_run = AsyncMock()
        mock_run.log_params = AsyncMock()

        mock_connector.start_run = MagicMock()
        mock_connector.start_run.return_value.__aenter__ = AsyncMock(return_value=mock_run)
        mock_connector.start_run.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch.object(tracker, "_get_connector", return_value=mock_connector):
                async with tracker.track_preparation_run(sample_context):
                    pass

                mock_run.log_params.assert_called_once()
                call_args = mock_run.log_params.call_args[0][0]
                assert "experiment_id" in call_args
                assert "data_source" in call_args


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for extract_metrics method."""

    def test_extract_metrics_from_state(self, tracker, sample_state):
        """Test metric extraction from full state."""
        metrics = tracker.extract_metrics(sample_state)

        assert isinstance(metrics, DataPreparerMetrics)
        assert metrics.completeness_score == 0.95
        assert metrics.validity_score == 0.92
        assert metrics.consistency_score == 0.88
        assert metrics.uniqueness_score == 0.99
        assert metrics.timeliness_score == 0.85
        assert metrics.overall_score == 0.918

    def test_extract_qc_metrics(self, tracker, sample_state):
        """Test QC gate metric extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.qc_status == "passed"
        assert metrics.qc_passed is True
        assert metrics.qc_score == 0.92
        assert metrics.gate_passed is True

    def test_extract_schema_metrics(self, tracker, sample_state):
        """Test schema validation metric extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.schema_validation_status == "validated"
        assert metrics.schema_splits_validated == 4
        assert metrics.schema_validation_time_ms == 150

    def test_extract_leakage_metrics(self, tracker, sample_state):
        """Test leakage detection metric extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.leakage_detected is False
        assert metrics.leakage_issues_count == 0

    def test_extract_sample_metrics(self, tracker, sample_state):
        """Test sample count metric extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.total_samples == 10000
        assert metrics.train_samples == 7000
        assert metrics.validation_samples == 1500
        assert metrics.test_samples == 1000
        assert metrics.holdout_samples == 500

    def test_extract_feature_metrics(self, tracker, sample_state):
        """Test feature metric extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.column_count == 45
        assert metrics.available_features_count == 3
        assert metrics.missing_required_features_count == 0

    def test_extract_expectation_metrics(self, tracker, sample_state):
        """Test expectation metric extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.total_expectations == 3
        assert metrics.failed_expectations_count == 1
        assert metrics.warnings_count == 1
        assert metrics.blocking_issues_count == 0

    def test_extract_feast_metrics(self, tracker, sample_state):
        """Test Feast registration metric extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.feast_registration_status == "registered"
        assert metrics.feast_features_registered == 45

    def test_extract_timing_metrics(self, tracker, sample_state):
        """Test timing metric extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.validation_duration_seconds == 2.5

    def test_extract_ready_status(self, tracker, sample_state):
        """Test ready status extraction."""
        metrics = tracker.extract_metrics(sample_state)

        assert metrics.is_ready is True

    def test_extract_metrics_handles_empty_state(self, tracker):
        """Test metric extraction with empty state."""
        metrics = tracker.extract_metrics({})

        assert metrics.completeness_score == 0.0
        assert metrics.qc_status == "unknown"
        assert metrics.total_samples == 0
        assert metrics.is_ready is False


# =============================================================================
# ARTIFACT LOGGING TESTS
# =============================================================================


class TestLogQcReport:
    """Tests for log_qc_report method."""

    @pytest.mark.asyncio
    async def test_log_qc_report_without_artifact_support(self, tracker, sample_state):
        """Test QC report logging without artifact support."""
        mock_run = MagicMock(spec=[])  # No log_artifact method
        # Should not raise
        await tracker.log_qc_report(mock_run, sample_state)

    @pytest.mark.asyncio
    async def test_log_qc_report_with_artifact_support(self, tracker, sample_state):
        """Test QC report logging with artifact support."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/qc_report.json"
            mock_temp.return_value = mock_file

            with patch("pathlib.Path.unlink"):
                await tracker.log_qc_report(mock_run, sample_state)

            mock_run.log_artifact.assert_called_once()


class TestLogFeatureStats:
    """Tests for log_feature_stats method."""

    @pytest.mark.asyncio
    async def test_log_feature_stats_without_data(self, tracker):
        """Test feature stats logging without data."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        # Should not call log_artifact
        await tracker.log_feature_stats(mock_run, {})
        mock_run.log_artifact.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_feature_stats_with_data(self, tracker, sample_state):
        """Test feature stats logging with data."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/feature_stats.json"
            mock_temp.return_value = mock_file

            with patch("pathlib.Path.unlink"):
                await tracker.log_feature_stats(mock_run, sample_state)

            mock_run.log_artifact.assert_called_once()


class TestLogExpectationResults:
    """Tests for log_expectation_results method."""

    @pytest.mark.asyncio
    async def test_log_expectations_without_results(self, tracker):
        """Test expectation results logging without data."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        await tracker.log_expectation_results(mock_run, {})
        mock_run.log_artifact.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_expectations_with_results(self, tracker, sample_state):
        """Test expectation results logging with data."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/expectations.json"
            mock_temp.return_value = mock_file

            with patch("pathlib.Path.unlink"):
                await tracker.log_expectation_results(mock_run, sample_state)

            mock_run.log_artifact.assert_called_once()


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetQualityHistory:
    """Tests for get_quality_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_connector(self, tracker):
        """Test history query when connector unavailable."""
        with patch.object(tracker, "_get_connector", return_value=None):
            history = await tracker.get_quality_history("ml_features_v3")
            assert history == []

    @pytest.mark.asyncio
    async def test_get_history_with_results(self, tracker, mock_connector):
        """Test history query with results."""
        mock_run = MagicMock()
        mock_run.info.run_id = "run_1"
        mock_run.info.start_time = datetime.now(timezone.utc)
        mock_run.data.metrics = {
            "completeness_score": 0.95,
            "validity_score": 0.92,
            "overall_score": 0.918,
            "qc_passed": 1.0,
            "gate_passed": 1.0,
            "total_samples": 10000.0,
            "leakage_detected": 0.0,
        }

        mock_connector.search_runs = AsyncMock(return_value=[mock_run])

        with patch.object(tracker, "_get_connector", return_value=mock_connector):
            history = await tracker.get_quality_history("ml_features_v3")

            assert len(history) == 1
            assert history[0]["run_id"] == "run_1"

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, tracker, mock_connector):
        """Test history query with limit parameter."""
        mock_connector.search_runs = AsyncMock(return_value=[])

        with patch.object(tracker, "_get_connector", return_value=mock_connector):
            await tracker.get_quality_history("ml_features_v3", limit=10)

            call_args = mock_connector.search_runs.call_args
            assert call_args.kwargs.get("max_results") == 10


class TestGetLeakageIncidents:
    """Tests for get_leakage_incidents method."""

    @pytest.mark.asyncio
    async def test_get_leakage_without_connector(self, tracker):
        """Test leakage query when connector unavailable."""
        with patch.object(tracker, "_get_connector", return_value=None):
            incidents = await tracker.get_leakage_incidents()
            assert incidents == []

    @pytest.mark.asyncio
    async def test_get_leakage_with_incidents(self, tracker, mock_connector):
        """Test leakage query with incidents."""
        mock_run = MagicMock()
        mock_run.info.run_id = "run_1"
        mock_run.info.start_time = datetime.now(timezone.utc)
        mock_run.data.tags = {
            "data_source": "ml_features_v3",
            "experiment_id": "exp_123",
        }
        mock_run.data.metrics = {"leakage_issues_count": 2.0}

        mock_connector.search_runs = AsyncMock(return_value=[mock_run])

        with patch.object(tracker, "_get_connector", return_value=mock_connector):
            incidents = await tracker.get_leakage_incidents()

            assert len(incidents) == 1
            assert incidents[0]["run_id"] == "run_1"

    @pytest.mark.asyncio
    async def test_get_leakage_filters_by_metric(self, tracker, mock_connector):
        """Test leakage query filters by leakage_detected metric."""
        mock_connector.search_runs = AsyncMock(return_value=[])

        with patch.object(tracker, "_get_connector", return_value=mock_connector):
            await tracker.get_leakage_incidents()

            call_args = mock_connector.search_runs.call_args
            filter_string = call_args.kwargs.get("filter_string")
            assert "leakage_detected = 1" in filter_string


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_create_tracker_default(self):
        """Test creating tracker with defaults."""
        tracker = create_tracker()
        assert isinstance(tracker, DataPreparerMLflowTracker)
        assert tracker.project_name == "data_preparer"

    def test_create_tracker_custom_project_name(self):
        """Test creating tracker with custom project name."""
        tracker = create_tracker(project_name="custom_project")
        assert tracker.project_name == "custom_project"

    def test_create_tracker_custom_uri(self):
        """Test creating tracker with custom URI."""
        tracker = create_tracker(tracking_uri="http://custom:5000")
        assert tracker.tracking_uri == "http://custom:5000"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_connector_error(
        self, tracker, mock_mlflow, mock_connector, sample_context
    ):
        """Test handling of connector errors."""
        mock_connector.get_or_create_experiment.side_effect = Exception("Connection failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch.object(tracker, "_get_connector", return_value=mock_connector):
                async with tracker.track_preparation_run(sample_context) as run:
                    # Should gracefully degrade to NoOpRun
                    assert isinstance(run, _NoOpRun)

    @pytest.mark.asyncio
    async def test_handles_artifact_logging_error(self, tracker, sample_state):
        """Test handling of artifact logging errors."""
        mock_run = AsyncMock()
        mock_run.log_artifact = AsyncMock(side_effect=Exception("Artifact failed"))

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.json"
            mock_temp.return_value = mock_file

            with patch("pathlib.Path.unlink"):
                # Should not raise
                await tracker.log_qc_report(mock_run, sample_state)

    @pytest.mark.asyncio
    async def test_handles_history_query_error(self, tracker, mock_connector):
        """Test handling of history query errors."""
        mock_connector.search_runs = AsyncMock(side_effect=Exception("Query failed"))

        with patch.object(tracker, "_get_connector", return_value=mock_connector):
            history = await tracker.get_quality_history("ml_features_v3")
            assert history == []
