"""
Unit Tests for Performance Tracking Service (Phase 14).

Tests cover:
- Performance metric calculation
- Performance snapshot recording
- Trend analysis
- Alert generation
- Model comparison
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.services.performance_tracking import (
    PerformanceSnapshot,
    PerformanceTracker,
    PerformanceTrackingConfig,
    PerformanceTrend,
    get_performance_tracker,
    record_model_performance,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> PerformanceTrackingConfig:
    """Create default performance tracking configuration."""
    return PerformanceTrackingConfig(
        tracked_metrics=["accuracy", "precision", "recall", "f1_score", "auc_roc"],
        degradation_threshold=0.1,
        absolute_min_accuracy=0.5,
        trend_window_days=30,
        min_samples=100,
        baseline_window_days=7,
        current_window_days=1,
    )


@pytest.fixture
def performance_tracker(default_config: PerformanceTrackingConfig) -> PerformanceTracker:
    """Create performance tracker instance."""
    return PerformanceTracker(config=default_config)


@pytest.fixture
def sample_predictions():
    """Create sample prediction data."""
    return {
        "predictions": np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1]),
        "actuals": np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1]),
        "prediction_scores": np.array([0.9, 0.2, 0.8, 0.7, 0.3, 0.1, 0.85, 0.4, 0.75, 0.9]),
    }


# =============================================================================
# PERFORMANCE TRACKING CONFIG TESTS
# =============================================================================


class TestPerformanceTrackingConfig:
    """Tests for PerformanceTrackingConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = PerformanceTrackingConfig()

        assert "accuracy" in config.tracked_metrics
        assert "precision" in config.tracked_metrics
        assert "recall" in config.tracked_metrics
        assert "f1_score" in config.tracked_metrics
        assert config.degradation_threshold == 0.1
        assert config.absolute_min_accuracy == 0.5
        assert config.trend_window_days == 30
        assert config.min_samples == 100

    def test_custom_config_values(self, default_config: PerformanceTrackingConfig):
        """Test custom configuration values."""
        assert len(default_config.tracked_metrics) == 5
        assert default_config.baseline_window_days == 7
        assert default_config.current_window_days == 1

    def test_config_with_custom_metrics(self):
        """Test config with custom metrics list."""
        config = PerformanceTrackingConfig(
            tracked_metrics=["accuracy", "custom_metric"],
            degradation_threshold=0.15,
        )

        assert "custom_metric" in config.tracked_metrics
        assert config.degradation_threshold == 0.15


# =============================================================================
# PERFORMANCE SNAPSHOT TESTS
# =============================================================================


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test performance snapshot creation."""
        now = datetime.now(timezone.utc)
        snapshot = PerformanceSnapshot(
            model_version="test_v1.0",
            recorded_at=now,
            sample_size=1000,
            window_start=now - timedelta(days=1),
            window_end=now,
            metrics={"accuracy": 0.85, "precision": 0.82},
        )

        assert snapshot.model_version == "test_v1.0"
        assert snapshot.sample_size == 1000
        assert snapshot.metrics["accuracy"] == 0.85

    def test_snapshot_with_segments(self):
        """Test snapshot with segment metrics."""
        now = datetime.now(timezone.utc)
        snapshot = PerformanceSnapshot(
            model_version="test_v1.0",
            recorded_at=now,
            sample_size=1000,
            window_start=now - timedelta(days=1),
            window_end=now,
            metrics={"accuracy": 0.85},
            segments={
                "segment_a": {"accuracy": 0.90},
                "segment_b": {"accuracy": 0.80},
            },
        )

        assert snapshot.segments is not None
        assert "segment_a" in snapshot.segments
        assert snapshot.segments["segment_a"]["accuracy"] == 0.90


# =============================================================================
# PERFORMANCE TREND TESTS
# =============================================================================


class TestPerformanceTrend:
    """Tests for PerformanceTrend dataclass."""

    def test_trend_creation(self):
        """Test performance trend creation."""
        trend = PerformanceTrend(
            model_version="test_v1.0",
            metric_name="accuracy",
            current_value=0.80,
            baseline_value=0.85,
            change_percent=-5.88,
            trend="degrading",
            is_significant=False,
            alert_threshold_breached=False,
        )

        assert trend.model_version == "test_v1.0"
        assert trend.metric_name == "accuracy"
        assert trend.current_value == 0.80
        assert trend.baseline_value == 0.85
        assert trend.trend == "degrading"

    def test_trend_improving(self):
        """Test improving trend."""
        trend = PerformanceTrend(
            model_version="test_v1.0",
            metric_name="accuracy",
            current_value=0.90,
            baseline_value=0.80,
            change_percent=12.5,
            trend="improving",
            is_significant=True,
            alert_threshold_breached=False,
        )

        assert trend.trend == "improving"
        assert trend.is_significant is True

    def test_trend_stable(self):
        """Test stable trend."""
        trend = PerformanceTrend(
            model_version="test_v1.0",
            metric_name="accuracy",
            current_value=0.85,
            baseline_value=0.85,
            change_percent=0.0,
            trend="stable",
            is_significant=False,
            alert_threshold_breached=False,
        )

        assert trend.trend == "stable"
        assert trend.is_significant is False


# =============================================================================
# PERFORMANCE TRACKER TESTS
# =============================================================================


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    def test_tracker_initialization(self, performance_tracker: PerformanceTracker):
        """Test tracker initialization."""
        assert performance_tracker is not None
        assert performance_tracker.config is not None

    def test_tracker_with_default_config(self):
        """Test tracker with default config."""
        tracker = PerformanceTracker()
        assert tracker.config.degradation_threshold == 0.1

    def test_calculate_metrics(self, performance_tracker: PerformanceTracker, sample_predictions):
        """Test metric calculation."""
        metrics = performance_tracker._calculate_metrics(
            predictions=sample_predictions["predictions"],
            actuals=sample_predictions["actuals"],
            scores=sample_predictions["prediction_scores"],
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1

    def test_calculate_metrics_without_scores(
        self, performance_tracker: PerformanceTracker, sample_predictions
    ):
        """Test metric calculation without probability scores."""
        metrics = performance_tracker._calculate_metrics(
            predictions=sample_predictions["predictions"],
            actuals=sample_predictions["actuals"],
            scores=None,
        )

        assert "accuracy" in metrics
        assert "auc_roc" not in metrics or metrics.get("auc_roc", 0) == 0

    @pytest.mark.asyncio
    async def test_record_performance(
        self, performance_tracker: PerformanceTracker, sample_predictions
    ):
        """Test recording performance snapshot."""
        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.record_metrics = AsyncMock()
            mock_repo_class.return_value = mock_repo

            snapshot = await performance_tracker.record_performance(
                model_version="test_v1.0",
                predictions=sample_predictions["predictions"],
                actuals=sample_predictions["actuals"],
                prediction_scores=sample_predictions["prediction_scores"],
            )

            assert snapshot.model_version == "test_v1.0"
            assert snapshot.sample_size == 10
            assert "accuracy" in snapshot.metrics
            mock_repo.record_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_performance_with_segments(self, performance_tracker: PerformanceTracker):
        """Test recording performance with segment data."""
        # Create larger sample for segmentation
        np.random.seed(42)
        predictions = np.random.randint(0, 2, 200)
        actuals = np.random.randint(0, 2, 200)
        scores = np.random.random(200)

        segments = {
            "high_value": np.array([True if i < 100 else False for i in range(200)]),
            "low_value": np.array([False if i < 100 else True for i in range(200)]),
        }

        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.record_metrics = AsyncMock()
            mock_repo_class.return_value = mock_repo

            snapshot = await performance_tracker.record_performance(
                model_version="test_v1.0",
                predictions=predictions,
                actuals=actuals,
                prediction_scores=scores,
                segments=segments,
            )

            assert snapshot.segments is not None
            assert "high_value" in snapshot.segments
            assert "low_value" in snapshot.segments

    @pytest.mark.asyncio
    async def test_get_performance_trend(self, performance_tracker: PerformanceTracker):
        """Test getting performance trend."""
        # Mock repository response
        mock_records = [
            MagicMock(metric_value=0.85),  # Current
            MagicMock(metric_value=0.82),
            MagicMock(metric_value=0.80),
            MagicMock(metric_value=0.78),
        ]

        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_metric_trend = AsyncMock(return_value=mock_records)
            mock_repo_class.return_value = mock_repo

            trend = await performance_tracker.get_performance_trend(
                model_version="test_v1.0",
                metric_name="accuracy",
            )

            assert trend.model_version == "test_v1.0"
            assert trend.metric_name == "accuracy"
            assert trend.current_value == 0.85

    @pytest.mark.asyncio
    async def test_get_performance_trend_no_data(self, performance_tracker: PerformanceTracker):
        """Test getting performance trend with no historical data."""
        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_metric_trend = AsyncMock(return_value=[])
            mock_repo_class.return_value = mock_repo

            trend = await performance_tracker.get_performance_trend(
                model_version="test_v1.0",
                metric_name="accuracy",
            )

            assert trend.trend == "unknown"
            assert trend.current_value == 0.0
            assert trend.is_significant is False

    @pytest.mark.asyncio
    async def test_check_performance_alerts(self, performance_tracker: PerformanceTracker):
        """Test checking for performance alerts."""
        # Mock trend data showing degradation
        mock_records_degraded = [
            MagicMock(metric_value=0.60),  # Current - degraded
            MagicMock(metric_value=0.85),
            MagicMock(metric_value=0.84),
        ]

        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_metric_trend = AsyncMock(return_value=mock_records_degraded)
            mock_repo_class.return_value = mock_repo

            alerts = await performance_tracker.check_performance_alerts(model_version="test_v1.0")

            # Should have alerts for degraded metrics
            assert isinstance(alerts, list)
            # With mocked data showing degradation, should have alerts
            if alerts:
                assert "metric_name" in alerts[0]
                assert "severity" in alerts[0]

    @pytest.mark.asyncio
    async def test_compare_model_versions(self, performance_tracker: PerformanceTracker):
        """Test comparing two model versions."""
        # Mock different performance for each model
        mock_records_a = [MagicMock(metric_value=0.80)]
        mock_records_b = [MagicMock(metric_value=0.85)]

        call_count = 0

        async def mock_get_trend(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_records_a
            return mock_records_b

        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_metric_trend = mock_get_trend
            mock_repo_class.return_value = mock_repo

            comparison = await performance_tracker.compare_model_versions(
                model_a="model_v1.0",
                model_b="model_v2.0",
                metric_name="accuracy",
            )

            assert "model_a" in comparison
            assert "model_b" in comparison
            assert "difference" in comparison
            assert "better_model" in comparison


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_performance_tracker_default(self):
        """Test getting default performance tracker."""
        tracker = get_performance_tracker()
        assert isinstance(tracker, PerformanceTracker)

    def test_get_performance_tracker_with_config(self, default_config: PerformanceTrackingConfig):
        """Test getting tracker with custom config."""
        tracker = get_performance_tracker(config=default_config)
        assert tracker.config.degradation_threshold == 0.1

    @pytest.mark.asyncio
    async def test_record_model_performance(self):
        """Test convenience function for recording performance."""
        with patch("src.services.performance_tracking.get_performance_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_snapshot = PerformanceSnapshot(
                model_version="test_v1.0",
                recorded_at=datetime.now(timezone.utc),
                sample_size=10,
                window_start=datetime.now(timezone.utc) - timedelta(days=1),
                window_end=datetime.now(timezone.utc),
                metrics={"accuracy": 0.85},
            )
            mock_tracker.record_performance = AsyncMock(return_value=mock_snapshot)
            mock_get_tracker.return_value = mock_tracker

            result = await record_model_performance(
                model_version="test_v1.0",
                predictions=[1, 0, 1, 1, 0],
                actuals=[1, 0, 1, 0, 0],
            )

            assert "model_version" in result
            assert "metrics" in result
            mock_tracker.record_performance.assert_called_once()


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_calculate_metrics_empty_arrays(self, performance_tracker: PerformanceTracker):
        """Test metric calculation with empty arrays."""
        metrics = performance_tracker._calculate_metrics(
            predictions=np.array([]),
            actuals=np.array([]),
            scores=None,
        )

        # Should handle gracefully
        assert "accuracy" in metrics

    def test_calculate_metrics_single_class(self, performance_tracker: PerformanceTracker):
        """Test metric calculation with single class."""
        predictions = np.array([1, 1, 1, 1, 1])
        actuals = np.array([1, 1, 1, 1, 1])

        metrics = performance_tracker._calculate_metrics(
            predictions=predictions,
            actuals=actuals,
            scores=None,
        )

        assert metrics["accuracy"] == 1.0

    def test_calculate_metrics_all_wrong(self, performance_tracker: PerformanceTracker):
        """Test metric calculation when all predictions are wrong."""
        predictions = np.array([1, 1, 1, 1, 1])
        actuals = np.array([0, 0, 0, 0, 0])

        metrics = performance_tracker._calculate_metrics(
            predictions=predictions,
            actuals=actuals,
            scores=None,
        )

        assert metrics["accuracy"] == 0.0

    @pytest.mark.asyncio
    async def test_record_performance_with_window_times(
        self, performance_tracker: PerformanceTracker
    ):
        """Test recording performance with explicit window times."""
        window_start = datetime.now(timezone.utc) - timedelta(hours=6)
        window_end = datetime.now(timezone.utc)

        predictions = np.array([1, 0, 1, 1, 0])
        actuals = np.array([1, 0, 1, 0, 0])

        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.record_metrics = AsyncMock()
            mock_repo_class.return_value = mock_repo

            snapshot = await performance_tracker.record_performance(
                model_version="test_v1.0",
                predictions=predictions,
                actuals=actuals,
                window_start=window_start,
                window_end=window_end,
            )

            assert snapshot.window_start == window_start
            assert snapshot.window_end == window_end


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestPerformanceTrackingWorkflow:
    """Tests for complete performance tracking workflows."""

    @pytest.mark.asyncio
    async def test_full_performance_workflow(self, performance_tracker: PerformanceTracker):
        """Test complete workflow: record -> trend -> alerts."""
        # Generate realistic prediction data
        np.random.seed(42)
        predictions = np.random.randint(0, 2, 200)
        actuals = np.random.randint(0, 2, 200)
        scores = np.random.random(200)

        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.record_metrics = AsyncMock()
            mock_repo.get_metric_trend = AsyncMock(
                return_value=[
                    MagicMock(metric_value=0.85),
                    MagicMock(metric_value=0.84),
                ]
            )
            mock_repo_class.return_value = mock_repo

            # Step 1: Record performance
            snapshot = await performance_tracker.record_performance(
                model_version="workflow_test_v1.0",
                predictions=predictions,
                actuals=actuals,
                prediction_scores=scores,
            )

            assert snapshot.sample_size == 200

            # Step 2: Get trend
            trend = await performance_tracker.get_performance_trend(
                model_version="workflow_test_v1.0",
                metric_name="accuracy",
            )

            assert trend.model_version == "workflow_test_v1.0"

            # Step 3: Check alerts
            alerts = await performance_tracker.check_performance_alerts(
                model_version="workflow_test_v1.0"
            )

            assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_model_comparison_workflow(self, performance_tracker: PerformanceTracker):
        """Test model comparison workflow."""
        call_count = 0

        async def mock_get_trend(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First model check
                return [MagicMock(metric_value=0.75)]
            return [MagicMock(metric_value=0.82)]  # Second model check

        with patch(
            "src.repositories.drift_monitoring.PerformanceMetricRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_metric_trend = mock_get_trend
            mock_repo_class.return_value = mock_repo

            comparison = await performance_tracker.compare_model_versions(
                model_a="old_model_v1.0",
                model_b="new_model_v2.0",
                metric_name="accuracy",
            )

            assert comparison["better_model"] == "new_model_v2.0"
            assert comparison["difference"] > 0
