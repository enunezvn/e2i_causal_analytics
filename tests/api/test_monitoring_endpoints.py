"""
Tests for Model Monitoring API endpoints.

Tests the monitoring endpoints:
- POST /monitoring/drift/detect - Trigger drift detection
- GET /monitoring/drift/status/{task_id} - Get async task status
- GET /monitoring/drift/latest/{model_id} - Get latest drift status
- GET /monitoring/drift/history/{model_id} - Get drift history
- GET /monitoring/alerts - List alerts
- GET /monitoring/alerts/{alert_id} - Get specific alert
- POST /monitoring/alerts/{alert_id}/action - Update alert
- GET /monitoring/runs - List monitoring runs
- GET /monitoring/health/{model_id} - Get model health

Phase 14: Model Monitoring & Drift Detection
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_celery_task():
    """Mock Celery async task."""
    task = MagicMock()
    task.id = "task_abc123"
    task.delay = MagicMock(return_value=task)
    return task


@pytest.fixture
def mock_drift_history_repo():
    """Mock DriftHistoryRepository."""
    repo = MagicMock()
    repo.get_latest_drift_status = AsyncMock(return_value=[])
    repo.get_drift_trend = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_alert_repo():
    """Mock MonitoringAlertRepository."""
    repo = MagicMock()
    repo.get_active_alerts = AsyncMock(return_value=[])
    repo.get_by_id = AsyncMock(return_value=None)
    repo.acknowledge_alert = AsyncMock(return_value=None)
    repo.resolve_alert = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_run_repo():
    """Mock MonitoringRunRepository."""
    repo = MagicMock()
    repo.get_recent_runs = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def sample_drift_record():
    """Sample drift detection record."""
    record = MagicMock()
    record.id = "drift_001"
    record.model_version = "propensity_v2.1.0"
    record.feature_name = "days_since_last_visit"
    record.drift_type = "data"
    record.drift_score = 0.45
    record.severity = "medium"
    record.test_statistic = 0.156
    record.p_value = 0.023
    record.detected_at = datetime.now(timezone.utc)
    record.baseline_start = datetime.now(timezone.utc)
    record.baseline_end = datetime.now(timezone.utc)
    record.current_start = datetime.now(timezone.utc)
    record.current_end = datetime.now(timezone.utc)
    return record


@pytest.fixture
def sample_alert_record():
    """Sample alert record."""
    record = MagicMock()
    record.id = "alert_001"
    record.model_version = "propensity_v2.1.0"
    record.alert_type = "drift"
    record.severity = "high"
    record.title = "Data drift detected"
    record.description = "Significant drift in feature X"
    record.status = "active"
    record.triggered_at = datetime.now(timezone.utc)
    record.acknowledged_at = None
    record.acknowledged_by = None
    record.resolved_at = None
    record.resolved_by = None
    return record


@pytest.fixture
def sample_run_record():
    """Sample monitoring run record."""
    record = MagicMock()
    record.id = "run_001"
    record.model_version = "propensity_v2.1.0"
    record.run_type = "scheduled"
    record.started_at = datetime.now(timezone.utc)
    record.completed_at = datetime.now(timezone.utc)
    record.features_checked = 25
    record.drift_detected_count = 2
    record.alerts_generated = 1
    record.duration_ms = 1250
    record.error_message = None
    return record


# =============================================================================
# BATCH 1A.1: DRIFT DETECTION CORE (3 tests)
# =============================================================================


class TestTriggerDriftDetection:
    """Tests for POST /monitoring/drift/detect."""

    def test_trigger_drift_async_success(self, mock_celery_task):
        """Should queue drift detection and return task ID."""
        with patch(
            "src.tasks.drift_monitoring_tasks.run_drift_detection",
            mock_celery_task,
        ):
            response = client.post(
                "/api/monitoring/drift/detect",
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
        assert data["task_id"] == "task_abc123"
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["status"] == "queued"

    def test_trigger_drift_sync_success(self):
        """Should run drift detection synchronously when async_mode=False."""
        mock_result = {
            "run_id": "sync_run_001",
            "status": "completed",
            "overall_drift_score": 0.35,
            "features_checked": 25,
            "features_with_drift": ["days_since_last_visit"],
            "drift_summary": "Moderate drift detected",
            "recommended_actions": ["Monitor closely"],
            "detection_latency_ms": 500,
        }

        with patch(
            "src.tasks.drift_monitoring_tasks.run_drift_detection",
            return_value=mock_result,
        ):
            response = client.post(
                "/api/monitoring/drift/detect?async_mode=false",
                json={
                    "model_id": "propensity_v2.1.0",
                    "time_window": "7d",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["overall_drift_score"] == 0.35
        assert "days_since_last_visit" in data["features_with_drift"]

    def test_trigger_drift_validation_error(self):
        """Should return 422 for missing required fields."""
        response = client.post(
            "/api/monitoring/drift/detect",
            json={},  # Missing model_id
        )

        assert response.status_code == 422

    def test_trigger_drift_sync_failure(self):
        """Should return 500 when sync detection fails."""
        with patch(
            "src.tasks.drift_monitoring_tasks.run_drift_detection",
            side_effect=Exception("Detection failed"),
        ):
            response = client.post(
                "/api/monitoring/drift/detect?async_mode=false",
                json={"model_id": "propensity_v2.1.0"},
            )

        assert response.status_code == 500
        assert "Detection failed" in response.json()["detail"]


class TestGetDriftStatus:
    """Tests for GET /monitoring/drift/status/{task_id}."""

    def test_get_pending_task_status(self):
        """Should return pending status for incomplete task."""
        mock_result = MagicMock()
        mock_result.status = "PENDING"
        mock_result.ready.return_value = False

        with patch("celery.result.AsyncResult", return_value=mock_result):
            with patch("src.workers.celery_app.celery_app"):
                response = client.get("/api/monitoring/drift/status/task_123")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_123"
        assert data["status"] == "PENDING"
        assert data["ready"] is False

    def test_get_completed_task_status(self):
        """Should return results for completed task."""
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.result = {
            "overall_drift_score": 0.35,
            "features_with_drift": ["feature_a"],
        }

        with patch("celery.result.AsyncResult", return_value=mock_result):
            with patch("src.workers.celery_app.celery_app"):
                response = client.get("/api/monitoring/drift/status/task_123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["ready"] is True
        assert "result" in data

    def test_get_failed_task_status(self):
        """Should return error for failed task."""
        mock_result = MagicMock()
        mock_result.status = "FAILURE"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = False
        mock_result.result = Exception("Task failed")

        with patch("celery.result.AsyncResult", return_value=mock_result):
            with patch("src.workers.celery_app.celery_app"):
                response = client.get("/api/monitoring/drift/status/task_123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILURE"
        assert "error" in data


class TestGetLatestDriftStatus:
    """Tests for GET /monitoring/drift/latest/{model_id}."""

    def test_get_latest_drift_success(self, sample_drift_record):
        """Should return latest drift status for model."""
        mock_repo = MagicMock()
        mock_repo.get_latest_drift_status = AsyncMock(
            return_value=[sample_drift_record]
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/drift/latest/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["status"] == "retrieved"
        assert data["features_checked"] == 1

    def test_get_latest_drift_empty(self):
        """Should return empty results when no drift data exists."""
        mock_repo = MagicMock()
        mock_repo.get_latest_drift_status = AsyncMock(return_value=[])

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/drift/latest/unknown_model")

        assert response.status_code == 200
        data = response.json()
        assert data["features_checked"] == 0
        assert data["results"] == []

    def test_get_latest_drift_with_limit(self, sample_drift_record):
        """Should respect limit parameter."""
        mock_repo = MagicMock()
        mock_repo.get_latest_drift_status = AsyncMock(
            return_value=[sample_drift_record]
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/drift/latest/propensity_v2.1.0?limit=5")

        assert response.status_code == 200
        # Verify limit was passed to repo
        mock_repo.get_latest_drift_status.assert_called_once()

    def test_get_latest_drift_service_error(self):
        """Should return 500 on service error."""
        mock_repo = MagicMock()
        mock_repo.get_latest_drift_status = AsyncMock(
            side_effect=Exception("Database error")
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/drift/latest/propensity_v2.1.0")

        assert response.status_code == 500


# =============================================================================
# BATCH 1A.2: DRIFT HISTORY (3 tests)
# =============================================================================


class TestGetDriftHistory:
    """Tests for GET /monitoring/drift/history/{model_id}."""

    def test_get_drift_history_success(self, sample_drift_record):
        """Should return drift history for model."""
        mock_repo = MagicMock()
        mock_repo.get_latest_drift_status = AsyncMock(
            return_value=[sample_drift_record]
        )

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/drift/history/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["total_records"] == 1
        assert len(data["records"]) == 1

    def test_get_drift_history_with_feature_filter(self, sample_drift_record):
        """Should filter by feature name."""
        mock_repo = MagicMock()
        mock_repo.get_drift_trend = AsyncMock(return_value=[sample_drift_record])

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/api/monitoring/drift/history/propensity_v2.1.0"
                "?feature_name=days_since_last_visit"
            )

        assert response.status_code == 200
        mock_repo.get_drift_trend.assert_called_once()

    def test_get_drift_history_empty(self):
        """Should return empty history when no data exists."""
        mock_repo = MagicMock()
        mock_repo.get_latest_drift_status = AsyncMock(return_value=[])

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/drift/history/unknown_model")

        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 0
        assert data["records"] == []


# =============================================================================
# BATCH 1A.3: ALERT CRUD (4 tests)
# =============================================================================


class TestListAlerts:
    """Tests for GET /monitoring/alerts."""

    def test_list_alerts_success(self, sample_alert_record):
        """Should return list of alerts."""
        mock_repo = MagicMock()
        mock_repo.get_active_alerts = AsyncMock(return_value=[sample_alert_record])

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/alerts")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["active_count"] == 1
        assert len(data["alerts"]) == 1

    def test_list_alerts_with_filters(self, sample_alert_record):
        """Should filter alerts by status and severity."""
        mock_repo = MagicMock()
        mock_repo.get_active_alerts = AsyncMock(return_value=[sample_alert_record])

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/alerts?status=active&severity=high")

        assert response.status_code == 200

    def test_list_alerts_empty(self):
        """Should return empty list when no alerts exist."""
        mock_repo = MagicMock()
        mock_repo.get_active_alerts = AsyncMock(return_value=[])

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/alerts")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["alerts"] == []


class TestGetAlert:
    """Tests for GET /monitoring/alerts/{alert_id}."""

    def test_get_alert_success(self, sample_alert_record):
        """Should return specific alert by ID."""
        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=sample_alert_record)

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/alerts/alert_001")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "alert_001"
        assert data["severity"] == "high"

    def test_get_alert_not_found(self):
        """Should return 404 for non-existent alert."""
        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=None)

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/alerts/unknown_alert")

        assert response.status_code == 404


class TestUpdateAlert:
    """Tests for POST /monitoring/alerts/{alert_id}/action."""

    def test_acknowledge_alert_success(self, sample_alert_record):
        """Should acknowledge alert."""
        updated_record = sample_alert_record
        updated_record.status = "acknowledged"
        updated_record.acknowledged_at = datetime.now(timezone.utc)
        updated_record.acknowledged_by = "user_123"

        mock_repo = MagicMock()
        mock_repo.acknowledge_alert = AsyncMock(return_value=updated_record)

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository",
            return_value=mock_repo,
        ):
            response = client.post(
                "/api/monitoring/alerts/alert_001/action",
                json={
                    "action": "acknowledge",
                    "user_id": "user_123",
                    "notes": "Investigating",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"

    def test_resolve_alert_success(self, sample_alert_record):
        """Should resolve alert."""
        updated_record = sample_alert_record
        updated_record.status = "resolved"
        updated_record.resolved_at = datetime.now(timezone.utc)
        updated_record.resolved_by = "user_123"

        mock_repo = MagicMock()
        mock_repo.resolve_alert = AsyncMock(return_value=updated_record)

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository",
            return_value=mock_repo,
        ):
            response = client.post(
                "/api/monitoring/alerts/alert_001/action",
                json={"action": "resolve", "user_id": "user_123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "resolved"

    def test_update_alert_not_found(self):
        """Should return 404 for non-existent alert."""
        mock_repo = MagicMock()
        mock_repo.acknowledge_alert = AsyncMock(return_value=None)

        with patch(
            "src.repositories.drift_monitoring.MonitoringAlertRepository",
            return_value=mock_repo,
        ):
            response = client.post(
                "/api/monitoring/alerts/unknown_alert/action",
                json={"action": "acknowledge"},
            )

        assert response.status_code == 404


# =============================================================================
# BATCH 1A.4: MONITORING RUNS (2 tests)
# =============================================================================


class TestListMonitoringRuns:
    """Tests for GET /monitoring/runs."""

    def test_list_runs_success(self, sample_run_record):
        """Should return list of monitoring runs."""
        mock_repo = MagicMock()
        mock_repo.get_recent_runs = AsyncMock(return_value=[sample_run_record])

        with patch(
            "src.repositories.drift_monitoring.MonitoringRunRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/runs")

        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 1
        assert len(data["runs"]) == 1

    def test_list_runs_with_model_filter(self, sample_run_record):
        """Should filter runs by model ID."""
        mock_repo = MagicMock()
        mock_repo.get_recent_runs = AsyncMock(return_value=[sample_run_record])

        with patch(
            "src.repositories.drift_monitoring.MonitoringRunRepository",
            return_value=mock_repo,
        ):
            response = client.get("/api/monitoring/runs?model_id=propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"


# =============================================================================
# BATCH 1A.5: MODEL HEALTH (2 tests)
# =============================================================================


class TestGetModelHealth:
    """Tests for GET /monitoring/health/{model_id}."""

    def test_get_model_health_healthy(
        self, sample_drift_record, sample_run_record
    ):
        """Should return healthy status for model without issues."""
        sample_drift_record.drift_score = 0.2  # Low drift
        sample_drift_record.severity = "low"

        mock_drift_repo = MagicMock()
        mock_drift_repo.get_latest_drift_status = AsyncMock(
            return_value=[sample_drift_record]
        )

        mock_alert_repo = MagicMock()
        mock_alert_repo.get_active_alerts = AsyncMock(return_value=[])

        mock_run_repo = MagicMock()
        mock_run_repo.get_recent_runs = AsyncMock(return_value=[sample_run_record])

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_drift_repo,
        ):
            with patch(
                "src.repositories.drift_monitoring.MonitoringAlertRepository",
                return_value=mock_alert_repo,
            ):
                with patch(
                    "src.repositories.drift_monitoring.MonitoringRunRepository",
                    return_value=mock_run_repo,
                ):
                    response = client.get("/api/monitoring/health/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["overall_health"] == "healthy"

    def test_get_model_health_critical(
        self, sample_drift_record, sample_alert_record, sample_run_record
    ):
        """Should return critical status for model with high drift."""
        sample_drift_record.drift_score = 0.8  # High drift
        sample_drift_record.severity = "critical"

        mock_drift_repo = MagicMock()
        mock_drift_repo.get_latest_drift_status = AsyncMock(
            return_value=[sample_drift_record]
        )

        mock_alert_repo = MagicMock()
        mock_alert_repo.get_active_alerts = AsyncMock(
            return_value=[sample_alert_record, sample_alert_record, sample_alert_record]
        )

        mock_run_repo = MagicMock()
        mock_run_repo.get_recent_runs = AsyncMock(return_value=[sample_run_record])

        with patch(
            "src.repositories.drift_monitoring.DriftHistoryRepository",
            return_value=mock_drift_repo,
        ):
            with patch(
                "src.repositories.drift_monitoring.MonitoringAlertRepository",
                return_value=mock_alert_repo,
            ):
                with patch(
                    "src.repositories.drift_monitoring.MonitoringRunRepository",
                    return_value=mock_run_repo,
                ):
                    response = client.get("/api/monitoring/health/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["overall_health"] == "critical"
        assert len(data["recommendations"]) > 0


# =============================================================================
# BATCH 1A.6: PERFORMANCE & RETRAINING (6 tests)
# =============================================================================


@pytest.fixture
def sample_performance_trend():
    """Sample performance trend result."""
    trend = MagicMock()
    trend.current_value = 0.92
    trend.baseline_value = 0.90
    trend.change_percent = 2.2
    trend.trend = "stable"
    trend.is_significant = False
    trend.alert_threshold_breached = False
    return trend


@pytest.fixture
def sample_retraining_job():
    """Sample retraining job."""
    from enum import Enum

    class MockStatus(Enum):
        PENDING = "pending"

    class MockReason(Enum):
        DATA_DRIFT = "data_drift"

    job = MagicMock()
    job.job_id = "retrain_001"
    job.model_version = "propensity_v2.1.0"
    job.status = MockStatus.PENDING
    job.trigger_reason = MockReason.DATA_DRIFT
    job.triggered_at = datetime.now(timezone.utc)
    job.triggered_by = "api_user"
    job.approved_at = None
    job.started_at = None
    job.completed_at = None
    job.performance_before = 0.90
    job.performance_after = None
    job.notes = None
    return job


class TestRecordPerformance:
    """Tests for POST /monitoring/performance/record."""

    def test_record_performance_async_success(self):
        """Should queue performance recording task."""
        mock_task = MagicMock()
        mock_task.id = "perf_task_001"

        with patch(
            "src.tasks.drift_monitoring_tasks.track_model_performance"
        ) as mock_tracker:
            mock_tracker.delay = MagicMock(return_value=mock_task)
            response = client.post(
                "/api/monitoring/performance/record",
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

    def test_record_performance_validation_error(self):
        """Should return 422 for invalid request."""
        response = client.post(
            "/api/monitoring/performance/record",
            json={
                "model_id": "propensity_v2.1.0",
                # Missing predictions and actuals
            },
        )

        assert response.status_code == 422


class TestGetPerformanceTrend:
    """Tests for GET /monitoring/performance/{model_id}/trend."""

    def test_get_performance_trend_success(self, sample_performance_trend):
        """Should return performance trend."""
        mock_tracker = MagicMock()
        mock_tracker.get_performance_trend = AsyncMock(
            return_value=sample_performance_trend
        )

        mock_repo = MagicMock()
        mock_repo.get_metric_trend = AsyncMock(return_value=[])

        with patch(
            "src.services.performance_tracking.get_performance_tracker",
            return_value=mock_tracker,
        ):
            with patch(
                "src.repositories.drift_monitoring.PerformanceMetricRepository",
                return_value=mock_repo,
            ):
                response = client.get(
                    "/api/monitoring/performance/propensity_v2.1.0/trend?metric_name=accuracy"
                )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["metric_name"] == "accuracy"
        assert data["current_value"] == 0.92

    def test_get_performance_trend_service_error(self):
        """Should return 500 on service error."""
        mock_tracker = MagicMock()
        mock_tracker.get_performance_trend = AsyncMock(
            side_effect=Exception("Service unavailable")
        )

        with patch(
            "src.services.performance_tracking.get_performance_tracker",
            return_value=mock_tracker,
        ):
            response = client.get("/api/monitoring/performance/propensity_v2.1.0/trend")

        assert response.status_code == 500


class TestEvaluateRetraining:
    """Tests for POST /monitoring/retraining/evaluate/{model_id}."""

    def test_evaluate_retraining_should_retrain(self):
        """Should recommend retraining when drift is high."""
        mock_decision = MagicMock()
        mock_decision.should_retrain = True
        mock_decision.confidence = 0.85
        mock_decision.reasons = ["High data drift detected"]
        mock_decision.trigger_factors = {"drift_score": 0.75}
        mock_decision.cooldown_active = False
        mock_decision.cooldown_ends_at = None
        mock_decision.recommended_action = "Trigger retraining"

        mock_service = MagicMock()
        mock_service.evaluate_retraining_need = AsyncMock(return_value=mock_decision)

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service",
            return_value=mock_service,
        ):
            response = client.post(
                "/api/monitoring/retraining/evaluate/propensity_v2.1.0"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["should_retrain"] is True
        assert data["confidence"] == 0.85

    def test_evaluate_retraining_no_retrain_needed(self):
        """Should not recommend retraining when model is healthy."""
        mock_decision = MagicMock()
        mock_decision.should_retrain = False
        mock_decision.confidence = 0.95
        mock_decision.reasons = []
        mock_decision.trigger_factors = {"drift_score": 0.15}
        mock_decision.cooldown_active = False
        mock_decision.cooldown_ends_at = None
        mock_decision.recommended_action = "Continue monitoring"

        mock_service = MagicMock()
        mock_service.evaluate_retraining_need = AsyncMock(return_value=mock_decision)

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service",
            return_value=mock_service,
        ):
            response = client.post(
                "/api/monitoring/retraining/evaluate/propensity_v2.1.0"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["should_retrain"] is False


class TestTriggerRetraining:
    """Tests for POST /monitoring/retraining/trigger/{model_id}."""

    def test_trigger_retraining_success(self, sample_retraining_job):
        """Should create retraining job."""
        mock_service = MagicMock()
        mock_service.trigger_retraining = AsyncMock(return_value=sample_retraining_job)

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service",
            return_value=mock_service,
        ):
            response = client.post(
                "/api/monitoring/retraining/trigger/propensity_v2.1.0",
                json={
                    "reason": "data_drift",
                    "notes": "Significant drift detected",
                    "auto_approve": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "retrain_001"
        assert data["model_version"] == "propensity_v2.1.0"
        assert data["status"] == "pending"


class TestGetRetrainingStatus:
    """Tests for GET /monitoring/retraining/status/{job_id}."""

    def test_get_retraining_status_success(self, sample_retraining_job):
        """Should return retraining job status."""
        mock_service = MagicMock()
        mock_service.get_retraining_status = AsyncMock(
            return_value=sample_retraining_job
        )

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service",
            return_value=mock_service,
        ):
            response = client.get("/api/monitoring/retraining/status/retrain_001")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "retrain_001"
        assert data["status"] == "pending"

    def test_get_retraining_status_not_found(self):
        """Should return 404 for non-existent job."""
        mock_service = MagicMock()
        mock_service.get_retraining_status = AsyncMock(return_value=None)

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service",
            return_value=mock_service,
        ):
            response = client.get("/api/monitoring/retraining/status/unknown_job")

        assert response.status_code == 404
