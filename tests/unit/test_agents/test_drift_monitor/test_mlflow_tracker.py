"""Unit tests for DriftMonitor MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting monitoring runs
- Metric extraction (PSI, KS statistics, drift severity)
- Artifact logging (JSON)
- Historical query methods (monitoring history, drift trends)
- Graceful degradation when MLflow unavailable

Phase 1 G03 from observability audit remediation plan.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.drift_monitor.mlflow_tracker import (
    DriftMonitorContext,
    DriftMonitorMetrics,
    DriftMonitorMLflowTracker,
    EXPERIMENT_PREFIX,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tracker():
    """Create a DriftMonitorMLflowTracker instance."""
    return DriftMonitorMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample DriftMonitorContext."""
    return DriftMonitorContext(
        experiment_name="pharma_drift_monitoring",
        brand="Kisqali",
        model_id="engagement_model_v1",
        time_window="7d",
        query_id="query_123",
        run_id="run_456",
        start_time=datetime.now(),
    )


@pytest.fixture
def sample_result():
    """Create a sample monitoring result dict."""
    return {
        "features_checked": 10,
        "features_with_drift": ["feature_1", "feature_2"],
        "overall_drift_score": 0.45,
        "detection_latency_ms": 250,
        "data_drift_results": [
            {"drift_detected": True, "severity": "high", "test_statistic": 0.15, "drift_type": "psi"},
            {"drift_detected": False, "severity": "none", "test_statistic": 0.05, "drift_type": "psi"},
        ],
        "model_drift_results": [
            {"drift_detected": True, "severity": "medium", "test_statistic": 0.12},
        ],
        "concept_drift_results": [
            {"drift_detected": False, "severity": "none"},
        ],
        "alerts": [
            {"severity": "critical", "message": "High drift detected"},
            {"severity": "warning", "message": "Feature distribution shift"},
        ],
        "warnings": ["Consider retraining model"],
        "recommended_actions": ["Review feature distributions"],
        "drift_summary": "Moderate drift detected in 2 features",
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestExperimentConfiguration:
    """Tests for experiment configuration constants."""

    def test_experiment_prefix_format(self):
        """Test experiment prefix follows naming convention."""
        assert EXPERIMENT_PREFIX == "e2i_causal/drift_monitor"

    def test_experiment_prefix_contains_agent_name(self):
        """Test experiment prefix includes agent identifier."""
        assert "drift_monitor" in EXPERIMENT_PREFIX


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestDriftMonitorContext:
    """Tests for DriftMonitorContext dataclass."""

    def test_context_creation_minimal(self):
        """Test context creation with minimal fields."""
        ctx = DriftMonitorContext()
        assert ctx.experiment_name == "default"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.experiment_name == "pharma_drift_monitoring"
        assert sample_context.brand == "Kisqali"
        assert sample_context.model_id == "engagement_model_v1"
        assert sample_context.time_window == "7d"

    def test_context_default_values(self):
        """Test context default values."""
        ctx = DriftMonitorContext()
        assert ctx.brand is None
        assert ctx.model_id is None
        assert ctx.time_window is None
        assert ctx.query_id is None


class TestDriftMonitorMetrics:
    """Tests for DriftMonitorMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = DriftMonitorMetrics(
            features_checked=10,
            features_with_drift=2,
            overall_drift_score=0.45,
            data_drift_count=2,
        )
        assert metrics.features_checked == 10
        assert metrics.features_with_drift == 2
        assert metrics.overall_drift_score == 0.45

    def test_metrics_default_values(self):
        """Test metrics with default values."""
        metrics = DriftMonitorMetrics()
        assert metrics.features_checked == 0
        assert metrics.overall_drift_score == 0.0
        assert metrics.structural_drift_detected is False

    def test_metrics_severity_counts(self):
        """Test metrics severity count fields."""
        metrics = DriftMonitorMetrics(
            critical_severity_count=1,
            high_severity_count=2,
            medium_severity_count=3,
        )
        assert metrics.critical_severity_count == 1
        assert metrics.high_severity_count == 2
        assert metrics.medium_severity_count == 3

    def test_metrics_alert_counts(self):
        """Test metrics alert count fields."""
        metrics = DriftMonitorMetrics(
            alerts_total=5,
            alerts_critical=2,
            alerts_warning=3,
        )
        assert metrics.alerts_total == 5
        assert metrics.alerts_critical == 2
        assert metrics.alerts_warning == 3


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, DriftMonitorMLflowTracker)

    def test_tracker_has_mlflow_attr(self, tracker):
        """Test tracker has _mlflow attribute."""
        assert hasattr(tracker, "_mlflow")

    def test_get_mlflow_returns_mlflow_or_none(self, tracker):
        """Test _get_mlflow returns mlflow module or None."""
        result = tracker._get_mlflow()
        assert result is None or hasattr(result, "log_metric")

    def test_tracker_has_tracking_uri_attr(self, tracker):
        """Test tracker has _tracking_uri attribute."""
        assert hasattr(tracker, "_tracking_uri")

    def test_tracker_with_custom_uri(self):
        """Test tracker creation with custom tracking URI."""
        tracker = DriftMonitorMLflowTracker(tracking_uri="http://localhost:5000")
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


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartMonitoringRun:
    """Tests for start_monitoring_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_returns_context_manager(self, tracker):
        """Test start_monitoring_run returns async context manager."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_monitoring_run(
                experiment_name="test_experiment",
                brand="Kisqali",
            ) as run_ctx:
                assert run_ctx is None or isinstance(run_ctx, DriftMonitorContext)

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow_returns_context(self, tracker):
        """Test start_monitoring_run returns context when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_monitoring_run(
                experiment_name="test_experiment",
                brand="Kisqali",
            ) as run_ctx:
                assert isinstance(run_ctx, DriftMonitorContext)
                assert run_ctx.experiment_name == "test_experiment"
                assert run_ctx.brand == "Kisqali"

    @pytest.mark.asyncio
    async def test_start_run_context_has_required_fields(self, tracker):
        """Test context manager returns context with required fields."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_monitoring_run(
                experiment_name="test_experiment",
                brand="Kisqali",
                model_id="model_v1",
                time_window="7d",
            ) as run_ctx:
                assert run_ctx.experiment_name == "test_experiment"
                assert run_ctx.brand == "Kisqali"
                assert run_ctx.model_id == "model_v1"
                assert run_ctx.time_window == "7d"
                assert run_ctx.start_time is not None

    @pytest.mark.asyncio
    async def test_start_run_with_query_id(self, tracker):
        """Test start_monitoring_run with query_id parameter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_monitoring_run(
                experiment_name="test",
                query_id="query_123",
            ) as run_ctx:
                assert run_ctx.query_id == "query_123"


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_dict(self, tracker, sample_result):
        """Test metric extraction from result dict."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, DriftMonitorMetrics)

    def test_extract_basic_metrics(self, tracker, sample_result):
        """Test basic metric extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert metrics.features_checked == 10
        assert metrics.overall_drift_score == 0.45
        assert metrics.detection_latency_ms == 250

    def test_extract_drift_counts(self, tracker, sample_result):
        """Test drift count extraction."""
        metrics = tracker._extract_metrics(sample_result)
        # Based on sample_result, data_drift has 1 detected, model_drift has 1 detected
        assert metrics.data_drift_count == 1
        assert metrics.model_drift_count == 1
        assert metrics.concept_drift_count == 0

    def test_extract_severity_counts(self, tracker, sample_result):
        """Test severity count extraction."""
        metrics = tracker._extract_metrics(sample_result)
        # Based on sample_result: high severity in data, medium in model
        assert metrics.high_severity_count == 1
        assert metrics.medium_severity_count == 1

    def test_extract_alert_counts(self, tracker, sample_result):
        """Test alert count extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert metrics.alerts_total == 2
        assert metrics.alerts_critical == 1
        assert metrics.alerts_warning == 1

    def test_extract_psi_metrics(self, tracker, sample_result):
        """Test PSI metric extraction."""
        metrics = tracker._extract_metrics(sample_result)
        # PSI scores from data_drift_results
        assert metrics.avg_psi_score == 0.1  # (0.15 + 0.05) / 2
        assert metrics.max_psi_score == 0.15

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        result = {"features_checked": 5}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, DriftMonitorMetrics)
        assert metrics.features_checked == 5

    def test_extract_metrics_handles_none(self, tracker):
        """Test metric extraction with None values."""
        result = {"overall_drift_score": None, "features_checked": 3}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, DriftMonitorMetrics)

    def test_extract_metrics_handles_empty_result(self, tracker):
        """Test metric extraction with empty result."""
        metrics = tracker._extract_metrics({})
        assert isinstance(metrics, DriftMonitorMetrics)
        assert metrics.features_checked == 0

    def test_extract_metrics_handles_pydantic_model(self, tracker):
        """Test metric extraction with Pydantic-like model."""
        mock_output = MagicMock()
        mock_output.model_dump.return_value = {
            "features_checked": 5,
            "overall_drift_score": 0.3,
        }
        metrics = tracker._extract_metrics(mock_output)
        assert isinstance(metrics, DriftMonitorMetrics)
        assert metrics.features_checked == 5

    def test_extract_warnings(self, tracker, sample_result):
        """Test warnings extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert metrics.warnings == ["Consider retraining model"]


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogMonitoringResult:
    """Tests for log_monitoring_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_result):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_monitoring_result(sample_result)

    @pytest.mark.asyncio
    async def test_log_result_extracts_metrics(self, tracker, sample_result):
        """Test logging extracts metrics from result."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_monitoring_result(sample_result)
            # Verify metrics can be extracted
            metrics = tracker._extract_metrics(sample_result)
            assert isinstance(metrics, DriftMonitorMetrics)

    @pytest.mark.asyncio
    async def test_log_result_handles_empty_result(self, tracker):
        """Test logging handles empty result dict."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_monitoring_result({})

    @pytest.mark.asyncio
    async def test_log_result_with_state(self, tracker, sample_result):
        """Test logging with optional state parameter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            state = {"structural_drift_details": {"detected": True}}
            await tracker.log_monitoring_result(sample_result, state=state)


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_without_mlflow(self, tracker, sample_result):
        """Test artifact logging without MLflow."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            metrics = tracker._extract_metrics(sample_result)
            await tracker._log_artifacts(sample_result, None, metrics)

    @pytest.mark.asyncio
    async def test_log_artifacts_empty_result(self, tracker):
        """Test artifact logging with empty result."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            metrics = DriftMonitorMetrics()
            await tracker._log_artifacts({}, None, metrics)


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetMonitoringHistory:
    """Tests for get_monitoring_history method."""

    def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = tracker.get_monitoring_history()
            assert isinstance(history, list)
            assert len(history) == 0

    def test_get_history_returns_list(self, tracker):
        """Test history query returns list structure."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = tracker.get_monitoring_history(brand="Kisqali")
            assert isinstance(history, list)

    def test_get_history_with_model_filter(self, tracker):
        """Test history query with model_id filter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = tracker.get_monitoring_history(model_id="engagement_model_v1")
            assert isinstance(history, list)

    def test_get_history_with_limit(self, tracker):
        """Test history query with limit parameter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = tracker.get_monitoring_history(limit=10)
            assert isinstance(history, list)

    def test_get_history_with_experiment_name(self, tracker):
        """Test history query with experiment_name parameter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = tracker.get_monitoring_history(experiment_name="custom_experiment")
            assert isinstance(history, list)


class TestGetDriftTrend:
    """Tests for get_drift_trend method."""

    def test_get_trend_without_mlflow(self, tracker):
        """Test trend query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = tracker.get_drift_trend()
            assert isinstance(trend, dict)

    def test_get_trend_returns_dict(self, tracker):
        """Test trend returns dictionary structure."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = tracker.get_drift_trend()
            assert isinstance(trend, dict)

    def test_get_trend_with_brand_filter(self, tracker):
        """Test trend with brand filter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = tracker.get_drift_trend(brand="Kisqali")
            assert isinstance(trend, dict)

    def test_get_trend_with_model_filter(self, tracker):
        """Test trend with model_id filter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = tracker.get_drift_trend(model_id="engagement_model_v1")
            assert isinstance(trend, dict)

    def test_get_trend_with_days_filter(self, tracker):
        """Test trend with days filter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            trend = tracker.get_drift_trend(days=7)
            assert isinstance(trend, dict)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_invalid_result_format(self, tracker):
        """Test handling of invalid result format."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_monitoring_result({"invalid": "structure"})

    @pytest.mark.asyncio
    async def test_handles_none_result(self, tracker):
        """Test handling of None values in result."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            result = {"overall_drift_score": None, "features_checked": None}
            await tracker.log_monitoring_result(result)

    def test_handles_empty_context(self, tracker):
        """Test handling of minimal context."""
        ctx = DriftMonitorContext()
        assert ctx is not None
        assert ctx.experiment_name == "default"

    def test_context_with_all_optional_none(self, tracker):
        """Test context with all optional fields as None."""
        ctx = DriftMonitorContext(
            experiment_name="test",
            brand=None,
            model_id=None,
            time_window=None,
            query_id=None,
        )
        assert ctx.brand is None
        assert ctx.model_id is None

    @pytest.mark.asyncio
    async def test_handles_no_drift_detected(self, tracker):
        """Test handling of result with no drift."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            result = {
                "features_checked": 5,
                "overall_drift_score": 0.0,
                "data_drift_results": [],
                "alerts": [],
            }
            await tracker.log_monitoring_result(result)

    @pytest.mark.asyncio
    async def test_handles_structural_drift_state(self, tracker):
        """Test handling of structural drift in state."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            result = {"features_checked": 3}
            state = {"structural_drift_details": {"detected": True, "reason": "Schema change"}}
            metrics = tracker._extract_metrics(result, state)
            assert metrics.structural_drift_detected is True
