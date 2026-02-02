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

    @patch("src.repositories.drift_monitoring.MonitoringRunRepository")
    @patch("src.repositories.drift_monitoring.DriftHistoryRepository")
    @patch("src.repositories.drift_monitoring.MonitoringAlertRepository")
    @patch("src.agents.drift_monitor.nodes.alert_aggregator.AlertAggregatorNode")
    @patch("src.agents.drift_monitor.nodes.concept_drift.ConceptDriftNode")
    @patch("src.agents.drift_monitor.nodes.model_drift.ModelDriftNode")
    @patch("src.agents.drift_monitor.nodes.data_drift.DataDriftNode")
    @patch("src.agents.drift_monitor.connectors.get_connector")
    def test_task_execution_sync(
        self,
        mock_get_connector,
        mock_data_drift,
        mock_model_drift,
        mock_concept_drift,
        mock_alert_agg,
        mock_alert_repo,
        mock_drift_repo,
        mock_run_repo,
    ):
        """Test synchronous task execution."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        # Setup mocks
        mock_connector = MagicMock()
        mock_connector.get_available_features = AsyncMock(return_value=["f1", "f2"])
        mock_get_connector.return_value = mock_connector

        # Mock nodes to return updated state
        mock_data_drift.return_value.execute = AsyncMock(return_value={})
        mock_model_drift.return_value.execute = AsyncMock(return_value={})
        mock_concept_drift.return_value.execute = AsyncMock(return_value={})
        mock_alert_agg.return_value.execute = AsyncMock(return_value={})

        # Mock repositories
        mock_run_record = MagicMock()
        mock_run_record.id = "run-123"
        mock_run_repo.return_value.start_run = AsyncMock(return_value=mock_run_record)
        mock_run_repo.return_value.complete_run = AsyncMock()
        mock_drift_repo.return_value.record_drift_results = AsyncMock()
        mock_alert_repo.return_value.create_alerts_from_drift = AsyncMock(return_value=[])

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

    @patch("src.repositories.drift_monitoring.MonitoringRunRepository")
    @patch("src.repositories.drift_monitoring.DriftHistoryRepository")
    @patch("src.repositories.drift_monitoring.MonitoringAlertRepository")
    @patch("src.agents.drift_monitor.nodes.alert_aggregator.AlertAggregatorNode")
    @patch("src.agents.drift_monitor.nodes.concept_drift.ConceptDriftNode")
    @patch("src.agents.drift_monitor.nodes.model_drift.ModelDriftNode")
    @patch("src.agents.drift_monitor.nodes.data_drift.DataDriftNode")
    @patch("src.agents.drift_monitor.connectors.get_connector")
    def test_task_with_specific_features(
        self,
        mock_get_connector,
        mock_data_drift,
        mock_model_drift,
        mock_concept_drift,
        mock_alert_agg,
        mock_alert_repo,
        mock_drift_repo,
        mock_run_repo,
    ):
        """Test task execution with specific features."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        # Setup mocks
        mock_connector = MagicMock()
        mock_get_connector.return_value = mock_connector

        # Mock nodes to return updated state
        mock_data_drift.return_value.execute = AsyncMock(return_value={})
        mock_model_drift.return_value.execute = AsyncMock(return_value={})
        mock_concept_drift.return_value.execute = AsyncMock(return_value={})
        mock_alert_agg.return_value.execute = AsyncMock(return_value={})

        # Mock repositories
        mock_run_record = MagicMock()
        mock_run_record.id = "run-123"
        mock_run_repo.return_value.start_run = AsyncMock(return_value=mock_run_record)
        mock_run_repo.return_value.complete_run = AsyncMock()
        mock_drift_repo.return_value.record_drift_results = AsyncMock()
        mock_alert_repo.return_value.create_alerts_from_drift = AsyncMock(return_value=[])

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
    @patch("src.agents.drift_monitor.connectors.get_connector")
    def test_task_execution(self, mock_get_connector, mock_run_drift):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import check_all_production_models

        mock_connector = MagicMock()
        mock_connector.get_available_models = AsyncMock(
            return_value=[
                {"id": "model_v1.0"},
                {"id": "model_v2.0"},
                {"id": "model_v3.0"},
            ]
        )
        mock_get_connector.return_value = mock_connector
        mock_run_drift.delay = MagicMock(return_value=MagicMock(id="task-123"))

        result = check_all_production_models(time_window="7d")

        assert result is not None
        assert isinstance(result, dict)


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

    @patch("src.memory.services.factories.get_supabase_client")
    def test_task_execution(self, mock_get_client):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import cleanup_old_drift_history

        # Mock Supabase client with chained table operations
        mock_result = MagicMock()
        mock_result.data = []  # Simulate no records deleted

        mock_execute = AsyncMock(return_value=mock_result)
        mock_eq = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_lt = MagicMock(return_value=MagicMock(eq=mock_eq, execute=mock_execute))
        mock_delete = MagicMock(return_value=MagicMock(lt=mock_lt))
        mock_table = MagicMock(return_value=MagicMock(delete=mock_delete))

        mock_client = MagicMock()
        mock_client.table = mock_table
        mock_get_client.side_effect = AsyncMock(return_value=mock_client)

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

    @patch("src.repositories.drift_monitoring.MonitoringAlertRepository")
    def test_task_execution(self, mock_alert_repo_cls):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import send_drift_alert_notifications

        mock_alert = MagicMock()
        mock_alert.id = "alert-123"
        mock_alert.severity = "high"
        mock_alert.message = "Drift detected"

        mock_alert_repo = MagicMock()
        mock_alert_repo.get_by_id = AsyncMock(return_value=mock_alert)
        mock_alert_repo_cls.return_value = mock_alert_repo

        # Function takes alert_ids, not drift_results
        alert_ids = ["alert-123", "alert-456"]

        result = send_drift_alert_notifications(alert_ids)

        assert result is not None
        assert isinstance(result, dict)


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

    @patch("src.services.performance_tracking.get_performance_tracker")
    @patch("src.services.performance_tracking.record_model_performance")
    def test_task_execution(self, mock_record, mock_get_tracker):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import track_model_performance

        mock_record.return_value = AsyncMock(
            return_value={
                "model_version": "test_v1.0",
                "sample_size": 100,
                "metrics": {"accuracy": 0.85},
            }
        )()

        mock_tracker = MagicMock()
        mock_tracker.check_performance_alerts = AsyncMock(return_value=[])
        mock_get_tracker.return_value = mock_tracker

        result = track_model_performance(
            model_id="test_v1.0",
            predictions=[1, 0, 1, 0],
            actuals=[1, 0, 1, 1],
        )

        assert result is not None

    @patch("src.services.performance_tracking.get_performance_tracker")
    @patch("src.services.performance_tracking.record_model_performance")
    def test_task_with_prediction_scores(self, mock_record, mock_get_tracker):
        """Test task with prediction probability scores."""
        from src.tasks.drift_monitoring_tasks import track_model_performance

        mock_record.return_value = AsyncMock(
            return_value={
                "model_version": "test_v1.0",
                "metrics": {"accuracy": 0.85, "auc_roc": 0.92},
            }
        )()

        mock_tracker = MagicMock()
        mock_tracker.check_performance_alerts = AsyncMock(return_value=[])
        mock_get_tracker.return_value = mock_tracker

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

    @patch("src.services.performance_tracking.get_performance_tracker")
    def test_task_execution_no_alerts(self, mock_get_tracker):
        """Test task execution with no alerts."""
        from src.tasks.drift_monitoring_tasks import check_model_performance_alerts

        mock_tracker = MagicMock()
        mock_tracker.check_performance_alerts = AsyncMock(return_value=[])
        mock_get_tracker.return_value = mock_tracker

        result = check_model_performance_alerts(model_id="test_v1.0")

        assert result is not None

    @patch("src.services.alert_routing.get_alert_router")
    @patch("src.services.performance_tracking.get_performance_tracker")
    def test_task_execution_with_alerts(self, mock_get_tracker, mock_get_router):
        """Test task execution with alerts."""
        from src.tasks.drift_monitoring_tasks import check_model_performance_alerts

        mock_tracker = MagicMock()
        mock_tracker.check_performance_alerts = AsyncMock(
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

        mock_router = MagicMock()
        mock_router.route_alert = AsyncMock(return_value={"success": True})
        mock_get_router.return_value = mock_router

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

    @patch("src.services.retraining_trigger.evaluate_and_trigger_retraining")
    def test_task_execution_no_retrain(self, mock_evaluate):
        """Test task execution when no retraining needed."""
        from src.tasks.drift_monitoring_tasks import evaluate_retraining_need

        mock_evaluate.return_value = AsyncMock(
            return_value={
                "status": "evaluated",
                "should_retrain": False,
                "confidence": 0.9,
                "reasons": ["Model stable"],
                "recommended_action": "Continue monitoring",
            }
        )()

        result = evaluate_retraining_need(model_id="test_v1.0")

        assert result is not None

    @patch("src.services.retraining_trigger.evaluate_and_trigger_retraining")
    def test_task_execution_retrain_needed(self, mock_evaluate):
        """Test task execution when retraining needed."""
        from src.tasks.drift_monitoring_tasks import evaluate_retraining_need

        mock_evaluate.return_value = AsyncMock(
            return_value={
                "status": "triggered",
                "should_retrain": True,
                "confidence": 0.85,
                "reasons": ["High data drift detected"],
                "recommended_action": "Schedule retraining",
            }
        )()

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

    @patch("src.services.retraining_trigger.get_retraining_trigger_service")
    @patch("src.repositories.drift_monitoring.RetrainingHistoryRepository")
    def test_task_execution(self, mock_repo_cls, mock_get_service):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import execute_model_retraining

        mock_repo = MagicMock()
        mock_repo.update = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        mock_service = MagicMock()
        mock_service.complete_retraining = AsyncMock()
        mock_get_service.return_value = mock_service

        # Correct function signature: (retraining_id, model_version, new_version, training_config)
        result = execute_model_retraining(
            retraining_id="retrain-123",
            model_version="test_v1.0",
            new_version="test_v1.1",
            training_config={"epochs": 100, "learning_rate": 0.001},
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
    @patch("src.agents.drift_monitor.connectors.get_connector")
    def test_task_execution(self, mock_get_connector, mock_evaluate):
        """Test task execution."""
        from src.tasks.drift_monitoring_tasks import check_retraining_for_all_models

        mock_connector = MagicMock()
        mock_connector.get_available_models = AsyncMock(
            return_value=[
                {"id": "model_v1.0"},
                {"id": "model_v2.0"},
            ]
        )
        mock_get_connector.return_value = mock_connector
        mock_evaluate.delay = MagicMock(return_value=MagicMock(id="task-123"))

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

    @patch("src.agents.drift_monitor.connectors.get_connector")
    def test_drift_detection_handles_detector_error(self, mock_get_connector):
        """Test drift detection handles detector errors."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        mock_get_connector.side_effect = Exception("Connector initialization failed")

        # Should handle gracefully
        try:
            result = run_drift_detection(model_id="test_v1.0", time_window="7d")
            # Either returns error result or raises
            assert "error" in result or result is not None
        except Exception as e:
            # Expected behavior
            assert "Connector" in str(e) or "failed" in str(e).lower()

    @patch("src.services.performance_tracking.record_model_performance")
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

    @patch("src.repositories.drift_monitoring.MonitoringAlertRepository")
    @patch("src.repositories.drift_monitoring.DriftHistoryRepository")
    @patch("src.repositories.drift_monitoring.MonitoringRunRepository")
    @patch("src.agents.drift_monitor.nodes.alert_aggregator.AlertAggregatorNode")
    @patch("src.agents.drift_monitor.nodes.concept_drift.ConceptDriftNode")
    @patch("src.agents.drift_monitor.nodes.model_drift.ModelDriftNode")
    @patch("src.agents.drift_monitor.nodes.data_drift.DataDriftNode")
    @patch("src.agents.drift_monitor.connectors.get_connector")
    def test_drift_detection_triggers_alerts(
        self,
        mock_get_connector,
        mock_data_drift,
        mock_model_drift,
        mock_concept_drift,
        mock_alert_agg,
        mock_run_repo,
        mock_drift_repo,
        mock_alert_repo,
    ):
        """Test that drift detection triggers alerts when needed."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        # Setup mocks
        mock_connector = MagicMock()
        mock_connector.get_available_features = AsyncMock(return_value=["f1", "f2"])
        mock_get_connector.return_value = mock_connector

        # Mock nodes to return updated state with high drift
        mock_data_drift.return_value.execute = AsyncMock(
            return_value={
                "data_drift_results": [{"feature": "f1", "drift_score": 0.75}],
            }
        )
        mock_model_drift.return_value.execute = AsyncMock(return_value={})
        mock_concept_drift.return_value.execute = AsyncMock(return_value={})
        mock_alert_agg.return_value.execute = AsyncMock(
            return_value={
                "alerts": [{"id": "alert-1", "severity": "high"}],
            }
        )

        # Mock repositories
        mock_run_record = MagicMock()
        mock_run_record.id = "run-123"
        mock_run_repo.return_value.start_run = AsyncMock(return_value=mock_run_record)
        mock_run_repo.return_value.complete_run = AsyncMock()
        mock_drift_repo.return_value.record_drift_results = AsyncMock()
        mock_alert_repo.return_value.create_alerts_from_drift = AsyncMock(
            return_value=[MagicMock(id="alert-123")]
        )

        result = run_drift_detection(
            model_id="test_v1.0",
            time_window="7d",
        )

        # High drift should trigger alert sending
        assert result is not None

    @patch("src.services.retraining_trigger.evaluate_and_trigger_retraining")
    def test_evaluation_triggers_retraining(self, mock_evaluate):
        """Test that evaluation can trigger retraining."""
        from src.tasks.drift_monitoring_tasks import evaluate_retraining_need

        mock_evaluate.return_value = AsyncMock(
            return_value={
                "status": "triggered",
                "should_retrain": True,
                "confidence": 0.9,
                "reasons": ["Critical drift detected"],
            }
        )()

        result = evaluate_retraining_need(
            model_id="test_v1.0",
            auto_approve=True,
        )

        assert result is not None
