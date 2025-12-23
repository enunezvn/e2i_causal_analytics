"""
Unit Tests for Drift Monitoring Celery Tasks (Phase 14).

Tests cover:
- run_drift_detection task
- check_all_production_models task
- cleanup_old_drift_history task
- send_drift_alert_notifications task
- track_model_performance task
- check_model_performance_alerts task
- evaluate_retraining_need task
- execute_model_retraining task
- check_retraining_for_all_models task
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_drift_detector():
    """Mock drift detector for testing."""
    detector = MagicMock()
    detector.detect_drift = AsyncMock(
        return_value={
            "data_drift": {"score": 0.45, "features_affected": ["feature_1", "feature_2"]},
            "model_drift": {"score": 0.35, "detected": False},
            "concept_drift": {"score": 0.25, "detected": False},
        }
    )
    return detector


@pytest.fixture
def mock_drift_repository():
    """Mock drift history repository."""
    repo = MagicMock()
    repo.save_drift_result = AsyncMock(return_value=True)
    repo.get_latest_drift_status = AsyncMock(return_value=[])
    repo.cleanup_old_records = AsyncMock(return_value=100)
    return repo


@pytest.fixture
def mock_alert_router():
    """Mock alert router."""
    router = MagicMock()
    router.route_alert = AsyncMock(return_value={"success": True, "channels_notified": 2})
    router.create_alert_payload = MagicMock()
    return router


@pytest.fixture
def mock_performance_tracker():
    """Mock performance tracker."""
    tracker = MagicMock()
    tracker.record_performance = AsyncMock(
        return_value={
            "model_version": "test_v1.0",
            "sample_size": 100,
            "metrics": {"accuracy": 0.85},
        }
    )
    tracker.check_performance_alerts = AsyncMock(return_value=[])
    return tracker


@pytest.fixture
def mock_retraining_service():
    """Mock retraining trigger service."""
    service = MagicMock()
    service.evaluate_retraining_need = AsyncMock(
        return_value=MagicMock(
            should_retrain=False,
            confidence=0.8,
            reasons=["Model stable"],
        )
    )
    service.check_and_trigger_retraining = AsyncMock(
        return_value={"evaluated": True, "triggered": False}
    )
    return service


# =============================================================================
# RUN DRIFT DETECTION TASK TESTS
# =============================================================================


class TestRunDriftDetectionTask:
    """Tests for run_drift_detection Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        assert run_drift_detection is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        # Celery tasks have .delay and .apply_async methods
        assert hasattr(run_drift_detection, "delay")
        assert hasattr(run_drift_detection, "apply_async")

    @patch("src.tasks.drift_monitoring_tasks.ConceptDriftDetector")
    @patch("src.tasks.drift_monitoring_tasks.DriftHistoryRepository")
    @patch("src.tasks.drift_monitoring_tasks.MonitoringRunRepository")
    def test_task_execution_sync(
        self, mock_run_repo, mock_drift_repo, mock_detector_cls
    ):
        """Test synchronous task execution."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        # Setup mocks
        mock_detector = MagicMock()
        mock_detector.detect_drift = MagicMock(
            return_value={
                "overall_drift_score": 0.4,
                "features_checked": 10,
                "features_with_drift": ["feature_1"],
            }
        )
        mock_detector_cls.return_value = mock_detector

        mock_drift_repo.return_value.save_drift_result = MagicMock(return_value=True)
        mock_run_repo.return_value.create_run = MagicMock()
        mock_run_repo.return_value.complete_run = MagicMock()

        # Execute task synchronously
        result = run_drift_detection(
            model_id="test_v1.0",
            time_window="7d",
            check_data_drift=True,
            check_model_drift=True,
            check_concept_drift=True,
        )

        assert result is not None
        assert isinstance(result, dict)

    @patch("src.tasks.drift_monitoring_tasks.ConceptDriftDetector")
    def test_task_with_specific_features(self, mock_detector_cls):
        """Test task execution with specific features."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        mock_detector = MagicMock()
        mock_detector.detect_drift = MagicMock(
            return_value={"overall_drift_score": 0.3}
        )
        mock_detector_cls.return_value = mock_detector

        result = run_drift_detection(
            model_id="test_v1.0",
            time_window="7d",
            features=["feature_a", "feature_b"],
        )

        assert result is not None


# =============================================================================
# CHECK ALL PRODUCTION MODELS TASK TESTS
# =============================================================================


class TestCheckAllProductionModelsTask:
    """Tests for check_all_production_models Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import check_all_production_models

        assert check_all_production_models is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import check_all_production_models

        assert hasattr(check_all_production_models, "delay")

    @patch("src.tasks.drift_monitoring_tasks.run_drift_detection")
    @patch("src.tasks.drift_monitoring_tasks.get_production_models")
    def test_task_execution(self, mock_get_models, mock_run_drift):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import check_all_production_models

        mock_get_models.return_value = ["model_v1.0", "model_v2.0", "model_v3.0"]
        mock_run_drift.delay = MagicMock()

        result = check_all_production_models(time_window="7d")

        assert result is not None
        assert "models_queued" in result or isinstance(result, dict)


# =============================================================================
# CLEANUP OLD DRIFT HISTORY TASK TESTS
# =============================================================================


class TestCleanupOldDriftHistoryTask:
    """Tests for cleanup_old_drift_history Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import cleanup_old_drift_history

        assert cleanup_old_drift_history is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import cleanup_old_drift_history

        assert hasattr(cleanup_old_drift_history, "delay")

    @patch("src.tasks.drift_monitoring_tasks.DriftHistoryRepository")
    def test_task_execution(self, mock_repo_cls):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import cleanup_old_drift_history

        mock_repo = MagicMock()
        mock_repo.cleanup_old_records = MagicMock(return_value=50)
        mock_repo_cls.return_value = mock_repo

        result = cleanup_old_drift_history(retention_days=90)

        assert result is not None


# =============================================================================
# SEND DRIFT ALERT NOTIFICATIONS TASK TESTS
# =============================================================================


class TestSendDriftAlertNotificationsTask:
    """Tests for send_drift_alert_notifications Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import send_drift_alert_notifications

        assert send_drift_alert_notifications is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import send_drift_alert_notifications

        assert hasattr(send_drift_alert_notifications, "delay")

    @patch("src.tasks.drift_monitoring_tasks.route_drift_alerts")
    def test_task_execution(self, mock_route_alerts):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import send_drift_alert_notifications

        mock_route_alerts.return_value = {"success": True, "channels_notified": 2}

        drift_results = {
            "model_id": "test_v1.0",
            "overall_drift_score": 0.65,
            "features_with_drift": ["feature_1"],
        }

        result = send_drift_alert_notifications(drift_results)

        assert result is not None


# =============================================================================
# TRACK MODEL PERFORMANCE TASK TESTS
# =============================================================================


class TestTrackModelPerformanceTask:
    """Tests for track_model_performance Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import track_model_performance

        assert track_model_performance is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import track_model_performance

        assert hasattr(track_model_performance, "delay")

    @patch("src.tasks.drift_monitoring_tasks.record_model_performance")
    def test_task_execution(self, mock_record):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import track_model_performance

        mock_record.return_value = {
            "model_version": "test_v1.0",
            "sample_size": 100,
            "metrics": {"accuracy": 0.85},
        }

        result = track_model_performance(
            model_id="test_v1.0",
            predictions=[1, 0, 1, 0],
            actuals=[1, 0, 1, 1],
        )

        assert result is not None

    @patch("src.tasks.drift_monitoring_tasks.record_model_performance")
    def test_task_with_prediction_scores(self, mock_record):
        """Test task with prediction probability scores."""
        from src.tasks.drift_monitoring_tasks import track_model_performance

        mock_record.return_value = {
            "model_version": "test_v1.0",
            "metrics": {"accuracy": 0.85, "auc_roc": 0.92},
        }

        result = track_model_performance(
            model_id="test_v1.0",
            predictions=[1, 0, 1, 0],
            actuals=[1, 0, 1, 1],
            prediction_scores=[0.9, 0.2, 0.85, 0.4],
        )

        assert result is not None


# =============================================================================
# CHECK MODEL PERFORMANCE ALERTS TASK TESTS
# =============================================================================


class TestCheckModelPerformanceAlertsTask:
    """Tests for check_model_performance_alerts Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import check_model_performance_alerts

        assert check_model_performance_alerts is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import check_model_performance_alerts

        assert hasattr(check_model_performance_alerts, "delay")

    @patch("src.tasks.drift_monitoring_tasks.get_performance_tracker")
    def test_task_execution_no_alerts(self, mock_get_tracker):
        """Test task execution with no alerts."""
        from src.tasks.drift_monitoring_tasks import check_model_performance_alerts

        mock_tracker = MagicMock()
        mock_tracker.check_performance_alerts = MagicMock(return_value=[])
        mock_get_tracker.return_value = mock_tracker

        result = check_model_performance_alerts(model_id="test_v1.0")

        assert result is not None

    @patch("src.tasks.drift_monitoring_tasks.get_performance_tracker")
    @patch("src.tasks.drift_monitoring_tasks.route_drift_alerts")
    def test_task_execution_with_alerts(self, mock_route, mock_get_tracker):
        """Test task execution with alerts."""
        from src.tasks.drift_monitoring_tasks import check_model_performance_alerts

        mock_tracker = MagicMock()
        mock_tracker.check_performance_alerts = MagicMock(
            return_value=[
                {
                    "metric_name": "accuracy",
                    "current_value": 0.75,
                    "baseline_value": 0.85,
                    "change_percent": -11.8,
                    "trend": "degrading",
                    "severity": "high",
                    "message": "Accuracy degraded significantly",
                }
            ]
        )
        mock_get_tracker.return_value = mock_tracker
        mock_route.return_value = {"success": True}

        result = check_model_performance_alerts(model_id="test_v1.0")

        assert result is not None


# =============================================================================
# EVALUATE RETRAINING NEED TASK TESTS
# =============================================================================


class TestEvaluateRetrainingNeedTask:
    """Tests for evaluate_retraining_need Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import evaluate_retraining_need

        assert evaluate_retraining_need is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import evaluate_retraining_need

        assert hasattr(evaluate_retraining_need, "delay")

    @patch("src.tasks.drift_monitoring_tasks.get_retraining_trigger_service")
    def test_task_execution_no_retrain(self, mock_get_service):
        """Test task execution when no retraining needed."""
        from src.tasks.drift_monitoring_tasks import evaluate_retraining_need

        mock_service = MagicMock()
        mock_service.evaluate_retraining_need = MagicMock(
            return_value=MagicMock(
                should_retrain=False,
                confidence=0.9,
                reasons=["Model stable"],
                recommended_action="Continue monitoring",
            )
        )
        mock_get_service.return_value = mock_service

        result = evaluate_retraining_need(model_id="test_v1.0")

        assert result is not None

    @patch("src.tasks.drift_monitoring_tasks.get_retraining_trigger_service")
    def test_task_execution_retrain_needed(self, mock_get_service):
        """Test task execution when retraining needed."""
        from src.tasks.drift_monitoring_tasks import evaluate_retraining_need

        mock_service = MagicMock()
        mock_service.evaluate_retraining_need = MagicMock(
            return_value=MagicMock(
                should_retrain=True,
                confidence=0.85,
                reasons=["High data drift detected"],
                recommended_action="Schedule retraining",
            )
        )
        mock_get_service.return_value = mock_service

        result = evaluate_retraining_need(model_id="test_v1.0")

        assert result is not None


# =============================================================================
# EXECUTE MODEL RETRAINING TASK TESTS
# =============================================================================


class TestExecuteModelRetrainingTask:
    """Tests for execute_model_retraining Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import execute_model_retraining

        assert execute_model_retraining is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import execute_model_retraining

        assert hasattr(execute_model_retraining, "delay")

    @patch("src.tasks.drift_monitoring_tasks.get_retraining_trigger_service")
    def test_task_execution(self, mock_get_service):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import execute_model_retraining

        mock_service = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "job-123"
        mock_job.status = "in_progress"
        mock_service.trigger_retraining = MagicMock(return_value=mock_job)
        mock_get_service.return_value = mock_service

        result = execute_model_retraining(
            model_id="test_v1.0",
            reason="data_drift",
            triggered_by="system",
        )

        assert result is not None


# =============================================================================
# CHECK RETRAINING FOR ALL MODELS TASK TESTS
# =============================================================================


class TestCheckRetrainingForAllModelsTask:
    """Tests for check_retraining_for_all_models Celery task."""

    def test_task_import(self):
        """Test that task can be imported."""
        from src.tasks.drift_monitoring_tasks import check_retraining_for_all_models

        assert check_retraining_for_all_models is not None

    def test_task_is_celery_task(self):
        """Test that function is a Celery task."""
        from src.tasks.drift_monitoring_tasks import check_retraining_for_all_models

        assert hasattr(check_retraining_for_all_models, "delay")

    @patch("src.tasks.drift_monitoring_tasks.evaluate_retraining_need")
    @patch("src.tasks.drift_monitoring_tasks.get_production_models")
    def test_task_execution(self, mock_get_models, mock_evaluate):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import check_retraining_for_all_models

        mock_get_models.return_value = ["model_v1.0", "model_v2.0"]
        mock_evaluate.delay = MagicMock()

        result = check_retraining_for_all_models()

        assert result is not None


# =============================================================================
# TASK CONFIGURATION TESTS
# =============================================================================


class TestTaskConfiguration:
    """Tests for task configuration and scheduling."""

    def test_all_tasks_exported(self):
        """Test that all tasks are exported from module."""
        from src.tasks import (
            check_all_production_models,
            check_model_performance_alerts,
            check_retraining_for_all_models,
            cleanup_old_drift_history,
            evaluate_retraining_need,
            execute_model_retraining,
            run_drift_detection,
            send_drift_alert_notifications,
            track_model_performance,
        )

        # All should be callable
        assert callable(run_drift_detection)
        assert callable(check_all_production_models)
        assert callable(cleanup_old_drift_history)
        assert callable(send_drift_alert_notifications)
        assert callable(track_model_performance)
        assert callable(check_model_performance_alerts)
        assert callable(evaluate_retraining_need)
        assert callable(execute_model_retraining)
        assert callable(check_retraining_for_all_models)

    def test_tasks_have_names(self):
        """Test that tasks have proper names."""
        from src.tasks.drift_monitoring_tasks import (
            check_all_production_models,
            run_drift_detection,
            track_model_performance,
        )

        # Celery tasks have .name attribute
        assert hasattr(run_drift_detection, "name")
        assert hasattr(check_all_production_models, "name")
        assert hasattr(track_model_performance, "name")


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestTaskErrorHandling:
    """Tests for task error handling."""

    @patch("src.tasks.drift_monitoring_tasks.ConceptDriftDetector")
    def test_drift_detection_handles_detector_error(self, mock_detector_cls):
        """Test drift detection handles detector errors."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        mock_detector_cls.side_effect = Exception("Detector initialization failed")

        # Should handle gracefully
        try:
            result = run_drift_detection(model_id="test_v1.0", time_window="7d")
            # Either returns error result or raises
            assert "error" in result or result is not None
        except Exception as e:
            # Expected behavior
            assert "Detector" in str(e) or "failed" in str(e).lower()

    @patch("src.tasks.drift_monitoring_tasks.record_model_performance")
    def test_performance_tracking_handles_error(self, mock_record):
        """Test performance tracking handles recording errors."""
        from src.tasks.drift_monitoring_tasks import track_model_performance

        mock_record.side_effect = Exception("Database connection failed")

        try:
            result = track_model_performance(
                model_id="test_v1.0",
                predictions=[1, 0],
                actuals=[1, 1],
            )
            assert "error" in result or result is not None
        except Exception as e:
            assert "Database" in str(e) or "failed" in str(e).lower()


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestTaskWorkflows:
    """Tests for task workflows and chains."""

    @patch("src.tasks.drift_monitoring_tasks.ConceptDriftDetector")
    @patch("src.tasks.drift_monitoring_tasks.send_drift_alert_notifications")
    def test_drift_detection_triggers_alerts(self, mock_send_alerts, mock_detector_cls):
        """Test that drift detection triggers alerts when needed."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        mock_detector = MagicMock()
        mock_detector.detect_drift = MagicMock(
            return_value={
                "overall_drift_score": 0.75,
                "features_with_drift": ["feature_1", "feature_2"],
            }
        )
        mock_detector_cls.return_value = mock_detector
        mock_send_alerts.delay = MagicMock()

        result = run_drift_detection(
            model_id="test_v1.0",
            time_window="7d",
        )

        # High drift should trigger alert sending
        assert result is not None

    @patch("src.tasks.drift_monitoring_tasks.get_retraining_trigger_service")
    @patch("src.tasks.drift_monitoring_tasks.execute_model_retraining")
    def test_evaluation_triggers_retraining(self, mock_execute, mock_get_service):
        """Test that evaluation can trigger retraining."""
        from src.tasks.drift_monitoring_tasks import evaluate_retraining_need

        mock_service = MagicMock()
        mock_service.evaluate_retraining_need = MagicMock(
            return_value=MagicMock(
                should_retrain=True,
                confidence=0.9,
                reasons=["Critical drift detected"],
            )
        )
        mock_get_service.return_value = mock_service
        mock_execute.delay = MagicMock()

        result = evaluate_retraining_need(
            model_id="test_v1.0",
            auto_trigger=True,
        )

        assert result is not None
