"""Tests for Model Monitoring & Drift Detection API Routes.

Version: 1.0.0
Tests the monitoring endpoints for drift detection, alerting, model health,
performance tracking, and retraining triggers.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.monitoring import (
    router,
    AlertAction,
    AlertStatus,
    DriftSeverity,
    DriftType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def app():
    """Create FastAPI app with monitoring router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_drift_record():
    """Create a mock drift history record."""
    record = MagicMock()
    record.id = "drift-123"
    record.model_version = "propensity_v2.1.0"
    record.feature_name = "days_since_last_visit"
    record.drift_type = "data"
    record.drift_score = 0.45
    record.severity = "medium"
    record.test_statistic = 0.156
    record.p_value = 0.023
    record.detected_at = datetime.now(timezone.utc)
    record.baseline_start = datetime.now(timezone.utc) - timedelta(days=14)
    record.baseline_end = datetime.now(timezone.utc) - timedelta(days=7)
    record.current_start = datetime.now(timezone.utc) - timedelta(days=7)
    record.current_end = datetime.now(timezone.utc)
    return record


@pytest.fixture
def mock_alert_record():
    """Create a mock alert record."""
    record = MagicMock()
    record.id = "alert-456"
    record.model_version = "propensity_v2.1.0"
    record.alert_type = "drift"
    record.severity = "high"
    record.title = "High Drift Detected"
    record.description = "Significant drift in feature X"
    record.status = "active"
    record.triggered_at = datetime.now(timezone.utc)
    record.acknowledged_at = None
    record.acknowledged_by = None
    record.resolved_at = None
    record.resolved_by = None
    return record


@pytest.fixture
def mock_run_record():
    """Create a mock monitoring run record."""
    record = MagicMock()
    record.id = "run-789"
    record.model_version = "propensity_v2.1.0"
    record.run_type = "scheduled"
    record.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    record.completed_at = datetime.now(timezone.utc)
    record.features_checked = 25
    record.drift_detected_count = 2
    record.alerts_generated = 1
    record.duration_ms = 1250
    record.error_message = None
    return record


# =============================================================================
# DRIFT DETECTION ENDPOINT TESTS
# =============================================================================


class TestTriggerDriftDetection:
    """Test POST /monitoring/drift/detect endpoint."""

    def test_trigger_drift_async_mode(self, client):
        """Test triggering drift detection in async mode."""
        mock_task = MagicMock()
        mock_task.id = "task-abc123"

        with patch(
            "src.tasks.drift_monitoring_tasks.run_drift_detection"
        ) as mock_run:
            mock_run.delay.return_value = mock_task

            response = client.post(
                "/monitoring/drift/detect",
                params={"async_mode": True},
                json={
                    "model_id": "propensity_v2.1.0",
                    "time_window": "7d",
                    "check_data_drift": True,
                    "check_model_drift": True,
                    "check_concept_drift": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-abc123"
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["status"] == "queued"

    def test_trigger_drift_sync_mode_success(self, client):
        """Test triggering drift detection in sync mode."""
        mock_result = {
            "run_id": "sync-run-1",
            "status": "completed",
            "overall_drift_score": 0.35,
            "features_checked": 25,
            "features_with_drift": ["feature_a", "feature_b"],
            "drift_summary": "Moderate drift detected",
            "recommended_actions": ["Investigate feature drift"],
            "detection_latency_ms": 1250,
        }

        with patch(
            "src.tasks.drift_monitoring_tasks.run_drift_detection",
            return_value=mock_result,
        ):
            response = client.post(
                "/monitoring/drift/detect",
                params={"async_mode": False},
                json={
                    "model_id": "propensity_v2.1.0",
                    "time_window": "7d",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["overall_drift_score"] == 0.35
        assert data["features_checked"] == 25
        assert len(data["features_with_drift"]) == 2

    def test_trigger_drift_sync_mode_failure(self, client):
        """Test drift detection failure in sync mode returns 500."""
        with patch(
            "src.tasks.drift_monitoring_tasks.run_drift_detection",
            side_effect=RuntimeError("Detection failed"),
        ):
            response = client.post(
                "/monitoring/drift/detect",
                params={"async_mode": False},
                json={
                    "model_id": "propensity_v2.1.0",
                    "time_window": "7d",
                },
            )

        assert response.status_code == 500
        assert "Detection failed" in response.json()["detail"]

    def test_trigger_drift_with_feature_filter(self, client):
        """Test triggering drift detection with specific features."""
        mock_task = MagicMock()
        mock_task.id = "task-filtered"

        with patch(
            "src.tasks.drift_monitoring_tasks.run_drift_detection"
        ) as mock_run:
            mock_run.delay.return_value = mock_task

            response = client.post(
                "/monitoring/drift/detect",
                params={"async_mode": True},
                json={
                    "model_id": "propensity_v2.1.0",
                    "time_window": "14d",
                    "features": ["feature_a", "feature_b"],
                    "brand": "Remibrutinib",
                },
            )

        assert response.status_code == 200
        mock_run.delay.assert_called_once()
        call_args = mock_run.delay.call_args
        assert call_args.kwargs["features"] == ["feature_a", "feature_b"]
        assert call_args.kwargs["brand"] == "Remibrutinib"


class TestGetDriftStatus:
    """Test GET /monitoring/drift/status/{task_id} endpoint."""

    def test_get_pending_task_status(self, client):
        """Test getting status of a pending task."""
        mock_result = MagicMock()
        mock_result.status = "PENDING"
        mock_result.ready.return_value = False

        with patch("src.workers.celery_app.celery_app"):
            with patch(
                "celery.result.AsyncResult",
                return_value=mock_result,
            ):
                response = client.get("/monitoring/drift/status/task-123")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-123"
        assert data["status"] == "PENDING"
        assert data["ready"] is False

    def test_get_completed_task_status(self, client):
        """Test getting status of a completed task."""
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.result = {"drift_score": 0.45, "features_with_drift": ["a"]}

        with patch("src.workers.celery_app.celery_app"):
            with patch(
                "celery.result.AsyncResult",
                return_value=mock_result,
            ):
                response = client.get("/monitoring/drift/status/task-456")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["ready"] is True
        assert "result" in data
        assert data["result"]["drift_score"] == 0.45

    def test_get_failed_task_status(self, client):
        """Test getting status of a failed task."""
        mock_result = MagicMock()
        mock_result.status = "FAILURE"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = False
        mock_result.result = Exception("Task failed")

        with patch("src.workers.celery_app.celery_app"):
            with patch(
                "celery.result.AsyncResult",
                return_value=mock_result,
            ):
                response = client.get("/monitoring/drift/status/task-789")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILURE"
        assert "error" in data


class TestGetLatestDriftStatus:
    """Test GET /monitoring/drift/latest/{model_id} endpoint."""

    def test_get_latest_drift_with_results(self, client, mock_drift_record):
        """Test getting latest drift status with results."""
        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_latest_drift_status.return_value = [mock_drift_record]
            MockRepo.return_value = mock_repo

            response = client.get(
                "/monitoring/drift/latest/propensity_v2.1.0",
                params={"limit": 10},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["status"] == "retrieved"
        assert data["features_checked"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["feature"] == "days_since_last_visit"

    def test_get_latest_drift_empty(self, client):
        """Test getting latest drift status when no records exist."""
        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_latest_drift_status.return_value = []
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/drift/latest/unknown_model")

        assert response.status_code == 200
        data = response.json()
        assert data["features_checked"] == 0
        assert data["results"] == []

    def test_get_latest_drift_server_error(self, client):
        """Test server error handling."""
        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_latest_drift_status.side_effect = RuntimeError("DB error")
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/drift/latest/propensity_v2.1.0")

        assert response.status_code == 500


class TestGetDriftHistory:
    """Test GET /monitoring/drift/history/{model_id} endpoint."""

    def test_get_drift_history_success(self, client, mock_drift_record):
        """Test getting drift history successfully."""
        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_latest_drift_status.return_value = [mock_drift_record]
            MockRepo.return_value = mock_repo

            response = client.get(
                "/monitoring/drift/history/propensity_v2.1.0",
                params={"days": 30, "limit": 100},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["total_records"] == 1
        assert len(data["records"]) == 1

    def test_get_drift_history_with_feature_filter(self, client, mock_drift_record):
        """Test getting drift history filtered by feature."""
        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_drift_trend.return_value = [mock_drift_record]
            MockRepo.return_value = mock_repo

            response = client.get(
                "/monitoring/drift/history/propensity_v2.1.0",
                params={"feature_name": "days_since_last_visit", "days": 30},
            )

        assert response.status_code == 200
        mock_repo.get_drift_trend.assert_called_once_with(
            "propensity_v2.1.0", "days_since_last_visit", days=30
        )


# =============================================================================
# ALERT ENDPOINT TESTS
# =============================================================================


class TestListAlerts:
    """Test GET /monitoring/alerts endpoint."""

    def test_list_alerts_success(self, client, mock_alert_record):
        """Test listing alerts successfully."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_active_alerts.return_value = [mock_alert_record]
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/alerts")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["active_count"] == 1
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["id"] == "alert-456"

    def test_list_alerts_with_model_filter(self, client, mock_alert_record):
        """Test listing alerts filtered by model."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_active_alerts.return_value = [mock_alert_record]
            MockRepo.return_value = mock_repo

            response = client.get(
                "/monitoring/alerts",
                params={"model_id": "propensity_v2.1.0"},
            )

        assert response.status_code == 200
        mock_repo.get_active_alerts.assert_called_with(
            "propensity_v2.1.0", limit=50
        )

    def test_list_alerts_with_status_filter(self, client, mock_alert_record):
        """Test listing alerts filtered by status."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_active_alerts.return_value = [mock_alert_record]
            MockRepo.return_value = mock_repo

            response = client.get(
                "/monitoring/alerts",
                params={"status": "active"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1

    def test_list_alerts_empty(self, client):
        """Test listing alerts when none exist."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_active_alerts.return_value = []
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/alerts")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["active_count"] == 0
        assert data["alerts"] == []


class TestGetAlert:
    """Test GET /monitoring/alerts/{alert_id} endpoint."""

    def test_get_alert_success(self, client, mock_alert_record):
        """Test getting a specific alert."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = mock_alert_record
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/alerts/alert-456")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "alert-456"
        assert data["title"] == "High Drift Detected"
        assert data["status"] == "active"

    def test_get_alert_not_found(self, client):
        """Test getting a non-existent alert."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_by_id.return_value = None
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/alerts/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestUpdateAlert:
    """Test POST /monitoring/alerts/{alert_id}/action endpoint."""

    def test_acknowledge_alert(self, client, mock_alert_record):
        """Test acknowledging an alert."""
        mock_alert_record.status = "acknowledged"
        mock_alert_record.acknowledged_at = datetime.now(timezone.utc)
        mock_alert_record.acknowledged_by = "user_123"

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.acknowledge_alert.return_value = mock_alert_record
            MockRepo.return_value = mock_repo

            response = client.post(
                "/monitoring/alerts/alert-456/action",
                json={
                    "action": "acknowledge",
                    "user_id": "user_123",
                    "notes": "Investigating",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"

    def test_resolve_alert(self, client, mock_alert_record):
        """Test resolving an alert."""
        mock_alert_record.status = "resolved"
        mock_alert_record.resolved_at = datetime.now(timezone.utc)
        mock_alert_record.resolved_by = "user_456"

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.resolve_alert.return_value = mock_alert_record
            MockRepo.return_value = mock_repo

            response = client.post(
                "/monitoring/alerts/alert-456/action",
                json={
                    "action": "resolve",
                    "user_id": "user_456",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "resolved"

    def test_snooze_alert(self, client, mock_alert_record):
        """Test snoozing an alert."""
        mock_alert_record.status = "acknowledged"

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.acknowledge_alert.return_value = mock_alert_record
            MockRepo.return_value = mock_repo

            response = client.post(
                "/monitoring/alerts/alert-456/action",
                json={
                    "action": "snooze",
                    "user_id": "user_789",
                    "snooze_until": "2024-12-31T00:00:00Z",
                },
            )

        assert response.status_code == 200

    def test_update_alert_not_found(self, client):
        """Test updating a non-existent alert."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.acknowledge_alert.return_value = None
            MockRepo.return_value = mock_repo

            response = client.post(
                "/monitoring/alerts/nonexistent/action",
                json={
                    "action": "acknowledge",
                },
            )

        assert response.status_code == 404


# =============================================================================
# MONITORING RUNS ENDPOINT TESTS
# =============================================================================


class TestListMonitoringRuns:
    """Test GET /monitoring/runs endpoint."""

    def test_list_runs_success(self, client, mock_run_record):
        """Test listing monitoring runs."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringRunRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_recent_runs.return_value = [mock_run_record]
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/runs")

        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 1
        assert len(data["runs"]) == 1
        assert data["runs"][0]["id"] == "run-789"
        assert data["runs"][0]["features_checked"] == 25

    def test_list_runs_with_model_filter(self, client, mock_run_record):
        """Test listing runs filtered by model."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringRunRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_recent_runs.return_value = [mock_run_record]
            MockRepo.return_value = mock_repo

            response = client.get(
                "/monitoring/runs",
                params={"model_id": "propensity_v2.1.0", "days": 14},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"

    def test_list_runs_empty(self, client):
        """Test listing runs when none exist."""
        with patch(
            "src.repositories.drift_monitoring.MonitoringRunRepository"
        ) as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_recent_runs.return_value = []
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/runs")

        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 0
        assert data["runs"] == []


# =============================================================================
# MODEL HEALTH ENDPOINT TESTS
# =============================================================================


class TestGetModelHealth:
    """Test GET /monitoring/health/{model_id} endpoint."""

    def test_get_healthy_model(self, client, mock_drift_record, mock_run_record):
        """Test getting health of a healthy model."""
        mock_drift_record.drift_score = 0.1
        mock_drift_record.severity = "low"

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as MockDriftRepo:
            with patch(
                "src.repositories.drift_monitoring.MonitoringAlertRepository"
            ) as MockAlertRepo:
                with patch(
                    "src.repositories.drift_monitoring.MonitoringRunRepository"
                ) as MockRunRepo:
                    drift_repo = AsyncMock()
                    drift_repo.get_latest_drift_status.return_value = [mock_drift_record]
                    MockDriftRepo.return_value = drift_repo

                    alert_repo = AsyncMock()
                    alert_repo.get_active_alerts.return_value = []
                    MockAlertRepo.return_value = alert_repo

                    run_repo = AsyncMock()
                    run_repo.get_recent_runs.return_value = [mock_run_record]
                    MockRunRepo.return_value = run_repo

                    response = client.get("/monitoring/health/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["overall_health"] == "healthy"
        assert data["active_alerts"] == 0

    def test_get_warning_model(self, client, mock_drift_record, mock_alert_record, mock_run_record):
        """Test getting health of a model with warnings."""
        mock_drift_record.drift_score = 0.5
        mock_alert_record.status = "active"

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as MockDriftRepo:
            with patch(
                "src.repositories.drift_monitoring.MonitoringAlertRepository"
            ) as MockAlertRepo:
                with patch(
                    "src.repositories.drift_monitoring.MonitoringRunRepository"
                ) as MockRunRepo:
                    drift_repo = AsyncMock()
                    drift_repo.get_latest_drift_status.return_value = [mock_drift_record]
                    MockDriftRepo.return_value = drift_repo

                    alert_repo = AsyncMock()
                    alert_repo.get_active_alerts.return_value = [mock_alert_record]
                    MockAlertRepo.return_value = alert_repo

                    run_repo = AsyncMock()
                    run_repo.get_recent_runs.return_value = [mock_run_record]
                    MockRunRepo.return_value = run_repo

                    response = client.get("/monitoring/health/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["overall_health"] == "warning"
        assert data["active_alerts"] == 1
        assert len(data["recommendations"]) > 0

    def test_get_critical_model(self, client, mock_drift_record, mock_run_record):
        """Test getting health of a critical model."""
        mock_drift_record.drift_score = 0.85

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository"
        ) as MockDriftRepo:
            with patch(
                "src.repositories.drift_monitoring.MonitoringAlertRepository"
            ) as MockAlertRepo:
                with patch(
                    "src.repositories.drift_monitoring.MonitoringRunRepository"
                ) as MockRunRepo:
                    drift_repo = AsyncMock()
                    drift_repo.get_latest_drift_status.return_value = [mock_drift_record]
                    MockDriftRepo.return_value = drift_repo

                    alert_repo = AsyncMock()
                    alert_repo.get_active_alerts.return_value = []
                    MockAlertRepo.return_value = alert_repo

                    run_repo = AsyncMock()
                    run_repo.get_recent_runs.return_value = [mock_run_record]
                    MockRunRepo.return_value = run_repo

                    response = client.get("/monitoring/health/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["overall_health"] == "critical"
        assert "retraining" in data["recommendations"][0].lower()


# =============================================================================
# PERFORMANCE TRACKING ENDPOINT TESTS
# =============================================================================


class TestRecordPerformance:
    """Test POST /monitoring/performance/record endpoint."""

    def test_record_performance_async(self, client):
        """Test recording performance in async mode."""
        mock_task = MagicMock()
        mock_task.id = "perf-task-123"

        with patch(
            "src.tasks.drift_monitoring_tasks.track_model_performance"
        ) as mock_track:
            mock_track.delay.return_value = mock_task

            response = client.post(
                "/monitoring/performance/record",
                params={"async_mode": True},
                json={
                    "model_id": "propensity_v2.1.0",
                    "predictions": [1, 0, 1, 1, 0],
                    "actuals": [1, 0, 1, 0, 0],
                    "prediction_scores": [0.85, 0.23, 0.91, 0.67, 0.12],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["sample_size"] == 5

    def test_record_performance_sync(self, client):
        """Test recording performance in sync mode."""
        mock_result = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "sample_size": 5,
            "metrics": {
                "accuracy": 0.80,
                "precision": 0.75,
                "recall": 0.85,
                "f1_score": 0.80,
            },
        }

        with patch(
            "src.services.performance_tracking.record_model_performance",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = client.post(
                "/monitoring/performance/record",
                params={"async_mode": False},
                json={
                    "model_id": "propensity_v2.1.0",
                    "predictions": [1, 0, 1, 1, 0],
                    "actuals": [1, 0, 1, 0, 0],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["sample_size"] == 5
        assert "accuracy" in data["metrics"]


class TestGetPerformanceTrend:
    """Test GET /monitoring/performance/{model_id}/trend endpoint."""

    def test_get_performance_trend_success(self, client):
        """Test getting performance trend."""
        mock_trend = MagicMock()
        mock_trend.current_value = 0.85
        mock_trend.baseline_value = 0.82
        mock_trend.change_percent = 3.7
        mock_trend.trend = "improving"
        mock_trend.is_significant = False
        mock_trend.alert_threshold_breached = False

        mock_metric_record = MagicMock()
        mock_metric_record.metric_name = "accuracy"
        mock_metric_record.metric_value = 0.85
        mock_metric_record.recorded_at = datetime.now(timezone.utc)

        with patch(
            "src.services.performance_tracking.get_performance_tracker"
        ) as mock_get_tracker:
            with patch(
                "src.repositories.drift_monitoring.PerformanceMetricRepository"
            ) as MockRepo:
                tracker = AsyncMock()
                tracker.get_performance_trend.return_value = mock_trend
                mock_get_tracker.return_value = tracker

                repo = AsyncMock()
                repo.get_metric_trend.return_value = [mock_metric_record]
                MockRepo.return_value = repo

                response = client.get(
                    "/monitoring/performance/propensity_v2.1.0/trend",
                    params={"metric_name": "accuracy", "days": 30},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["metric_name"] == "accuracy"
        assert data["trend"] == "improving"
        assert len(data["history"]) == 1


class TestGetPerformanceAlerts:
    """Test GET /monitoring/performance/{model_id}/alerts endpoint."""

    def test_get_performance_alerts(self, client):
        """Test getting performance alerts."""
        mock_alerts = [
            {
                "metric_name": "accuracy",
                "current_value": 0.72,
                "baseline_value": 0.85,
                "change_percent": -15.3,
                "trend": "degrading",
                "severity": "high",
                "message": "Accuracy has degraded significantly",
            }
        ]

        with patch(
            "src.services.performance_tracking.get_performance_tracker"
        ) as mock_get_tracker:
            tracker = AsyncMock()
            tracker.check_performance_alerts.return_value = mock_alerts
            mock_get_tracker.return_value = tracker

            response = client.get(
                "/monitoring/performance/propensity_v2.1.0/alerts"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["alert_count"] == 1
        assert data["alerts"][0]["severity"] == "high"


class TestCompareModelPerformance:
    """Test GET /monitoring/performance/{model_id}/compare/{other_model_id} endpoint."""

    def test_compare_models(self, client):
        """Test comparing two model versions."""
        mock_result = {
            "model_a": "propensity_v2.0.0",
            "model_b": "propensity_v2.1.0",
            "metric_name": "accuracy",
            "model_a_value": 0.82,
            "model_b_value": 0.85,
            "difference": 0.03,
            "winner": "propensity_v2.1.0",
        }

        with patch(
            "src.services.performance_tracking.get_performance_tracker"
        ) as mock_get_tracker:
            tracker = AsyncMock()
            tracker.compare_model_versions.return_value = mock_result
            mock_get_tracker.return_value = tracker

            response = client.get(
                "/monitoring/performance/propensity_v2.0.0/compare/propensity_v2.1.0",
                params={"metric_name": "accuracy"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["winner"] == "propensity_v2.1.0"
        assert data["difference"] == 0.03


# =============================================================================
# PRODUCTION SWEEP ENDPOINT TESTS
# =============================================================================


class TestProductionSweep:
    """Test POST /monitoring/sweep/production endpoint."""

    def test_trigger_production_sweep(self, client):
        """Test triggering production sweep."""
        mock_task = MagicMock()
        mock_task.id = "sweep-task-123"

        with patch(
            "src.tasks.drift_monitoring_tasks.check_all_production_models"
        ) as mock_check:
            mock_check.delay.return_value = mock_task

            response = client.post(
                "/monitoring/sweep/production",
                params={"time_window": "14d"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "sweep-task-123"
        assert data["status"] == "queued"
        assert data["time_window"] == "14d"


# =============================================================================
# RETRAINING ENDPOINT TESTS
# =============================================================================


class TestEvaluateRetrainingNeed:
    """Test POST /monitoring/retraining/evaluate/{model_id} endpoint."""

    def test_evaluate_needs_retraining(self, client):
        """Test evaluating a model that needs retraining."""
        mock_decision = MagicMock()
        mock_decision.should_retrain = True
        mock_decision.confidence = 0.85
        mock_decision.reasons = ["High data drift", "Performance degradation"]
        mock_decision.trigger_factors = {"drift_score": 0.72, "accuracy_drop": 0.08}
        mock_decision.cooldown_active = False
        mock_decision.cooldown_ends_at = None
        mock_decision.recommended_action = "Trigger retraining"

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.evaluate_retraining_need.return_value = mock_decision
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/evaluate/propensity_v2.1.0"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["should_retrain"] is True
        assert data["confidence"] == 0.85
        assert len(data["reasons"]) == 2

    def test_evaluate_no_retraining_needed(self, client):
        """Test evaluating a model that doesn't need retraining."""
        mock_decision = MagicMock()
        mock_decision.should_retrain = False
        mock_decision.confidence = 0.95
        mock_decision.reasons = []
        mock_decision.trigger_factors = {}
        mock_decision.cooldown_active = False
        mock_decision.cooldown_ends_at = None
        mock_decision.recommended_action = "Continue monitoring"

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.evaluate_retraining_need.return_value = mock_decision
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/evaluate/propensity_v2.1.0"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["should_retrain"] is False


class TestTriggerRetraining:
    """Test POST /monitoring/retraining/trigger/{model_id} endpoint."""

    def test_trigger_retraining_success(self, client):
        """Test triggering retraining successfully."""
        mock_job = MagicMock()
        mock_job.job_id = "retrain-job-123"
        mock_job.model_version = "propensity_v2.1.0"
        mock_job.status = MagicMock(value="pending")
        mock_job.trigger_reason = MagicMock(value="data_drift")
        mock_job.triggered_at = datetime.now(timezone.utc)
        mock_job.triggered_by = "api_user"
        mock_job.approved_at = None
        mock_job.started_at = None
        mock_job.completed_at = None
        mock_job.performance_before = 0.82
        mock_job.performance_after = None
        mock_job.notes = "Triggered due to drift"

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.trigger_retraining.return_value = mock_job
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/trigger/propensity_v2.1.0",
                json={
                    "reason": "data_drift",
                    "notes": "Triggered due to drift",
                    "auto_approve": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "retrain-job-123"
        assert data["status"] == "pending"


class TestGetRetrainingStatus:
    """Test GET /monitoring/retraining/status/{job_id} endpoint."""

    def test_get_retraining_status(self, client):
        """Test getting retraining job status."""
        mock_job = MagicMock()
        mock_job.job_id = "retrain-job-123"
        mock_job.model_version = "propensity_v2.1.0"
        mock_job.status = MagicMock(value="in_progress")
        mock_job.trigger_reason = MagicMock(value="data_drift")
        mock_job.triggered_at = datetime.now(timezone.utc)
        mock_job.triggered_by = "api_user"
        mock_job.approved_at = datetime.now(timezone.utc)
        mock_job.started_at = datetime.now(timezone.utc)
        mock_job.completed_at = None
        mock_job.performance_before = 0.82
        mock_job.performance_after = None
        mock_job.notes = None

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.get_retraining_status.return_value = mock_job
            mock_get_service.return_value = service

            response = client.get("/monitoring/retraining/status/retrain-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"

    def test_get_retraining_status_not_found(self, client):
        """Test getting non-existent retraining job."""
        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.get_retraining_status.return_value = None
            mock_get_service.return_value = service

            response = client.get("/monitoring/retraining/status/nonexistent")

        assert response.status_code == 404


class TestCompleteRetraining:
    """Test POST /monitoring/retraining/{job_id}/complete endpoint."""

    def test_complete_retraining_success(self, client):
        """Test completing retraining successfully."""
        mock_job = MagicMock()
        mock_job.job_id = "retrain-job-123"
        mock_job.model_version = "propensity_v2.1.0"
        mock_job.status = MagicMock(value="completed")
        mock_job.trigger_reason = MagicMock(value="data_drift")
        mock_job.triggered_at = datetime.now(timezone.utc) - timedelta(hours=2)
        mock_job.triggered_by = "api_user"
        mock_job.approved_at = datetime.now(timezone.utc) - timedelta(hours=2)
        mock_job.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_job.completed_at = datetime.now(timezone.utc)
        mock_job.performance_before = 0.82
        mock_job.performance_after = 0.89
        mock_job.notes = "Successfully improved"

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.complete_retraining.return_value = mock_job
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/complete",
                json={
                    "performance_after": 0.89,
                    "success": True,
                    "notes": "Successfully improved",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["performance_after"] == 0.89

    def test_complete_retraining_not_found(self, client):
        """Test completing non-existent retraining job."""
        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.complete_retraining.return_value = None
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/nonexistent/complete",
                json={
                    "performance_after": 0.89,
                    "success": True,
                },
            )

        assert response.status_code == 404


class TestRollbackRetraining:
    """Test POST /monitoring/retraining/{job_id}/rollback endpoint."""

    def test_rollback_retraining_success(self, client):
        """Test rolling back retraining."""
        mock_job = MagicMock()
        mock_job.job_id = "retrain-job-123"
        mock_job.model_version = "propensity_v2.1.0"
        mock_job.status = MagicMock(value="rolled_back")
        mock_job.trigger_reason = MagicMock(value="data_drift")
        mock_job.triggered_at = datetime.now(timezone.utc)
        mock_job.triggered_by = "api_user"
        mock_job.approved_at = datetime.now(timezone.utc)
        mock_job.started_at = datetime.now(timezone.utc)
        mock_job.completed_at = datetime.now(timezone.utc)
        mock_job.performance_before = 0.82
        mock_job.performance_after = 0.75
        mock_job.notes = "Rolled back due to degradation"

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.rollback_retraining.return_value = mock_job
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/rollback",
                json={"reason": "Performance degradation on validation set"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rolled_back"


class TestRetrainingSweep:
    """Test POST /monitoring/retraining/sweep endpoint."""

    def test_trigger_retraining_sweep(self, client):
        """Test triggering retraining evaluation sweep."""
        mock_task = MagicMock()
        mock_task.id = "retrain-sweep-123"

        with patch(
            "src.tasks.drift_monitoring_tasks.check_retraining_for_all_models"
        ) as mock_check:
            mock_check.delay.return_value = mock_task

            response = client.post("/monitoring/retraining/sweep")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "retrain-sweep-123"
        assert data["status"] == "queued"


# =============================================================================
# ENUM AND MODEL TESTS
# =============================================================================


class TestEnums:
    """Test enum values."""

    def test_drift_type_values(self):
        """Test DriftType enum values."""
        assert DriftType.DATA == "data"
        assert DriftType.MODEL == "model"
        assert DriftType.CONCEPT == "concept"
        assert DriftType.ALL == "all"

    def test_drift_severity_values(self):
        """Test DriftSeverity enum values."""
        assert DriftSeverity.NONE == "none"
        assert DriftSeverity.LOW == "low"
        assert DriftSeverity.MEDIUM == "medium"
        assert DriftSeverity.HIGH == "high"
        assert DriftSeverity.CRITICAL == "critical"

    def test_alert_status_values(self):
        """Test AlertStatus enum values."""
        assert AlertStatus.ACTIVE == "active"
        assert AlertStatus.ACKNOWLEDGED == "acknowledged"
        assert AlertStatus.RESOLVED == "resolved"
        assert AlertStatus.SNOOZED == "snoozed"

    def test_alert_action_values(self):
        """Test AlertAction enum values."""
        assert AlertAction.ACKNOWLEDGE == "acknowledge"
        assert AlertAction.RESOLVE == "resolve"
        assert AlertAction.SNOOZE == "snooze"
