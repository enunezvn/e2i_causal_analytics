"""Tests for Model Monitoring & Drift Detection API Routes.

Version: 1.0.0
Tests the monitoring endpoints for drift detection, alerting, model health,
performance tracking, and retraining triggers.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.monitoring import (
    AlertAction,
    AlertStatus,
    DriftSeverity,
    DriftType,
    _validation_auc_from_metrics,
    router,
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
    record.message = "High Drift Detected"
    record.recommended_action = "Significant drift in feature X"
    record.status = "active"
    # #842: ml_monitoring_alerts has no triggered_at; created_at is the fire time.
    record.created_at = datetime.now(timezone.utc)
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
    record.run_type = "full"  # DB run_type = monitoring kind
    record.trigger_type = "scheduled"  # #842: API surfaces trigger as run_type
    record.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    record.completed_at = datetime.now(timezone.utc)
    # #842: real ml_monitoring_runs columns are total_checks + duration_seconds.
    record.total_checks = 25
    record.drift_detected_count = 2
    record.alerts_generated = 1
    record.duration_seconds = 1.25  # API surfaces duration_ms = 1250
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

        with patch("src.tasks.drift_monitoring_tasks.run_drift_detection") as mock_run:
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
        """Test drift detection failure in sync mode returns 500.

        Finding 2: the raw exception text must NOT be echoed to the client;
        a generic message is returned instead (the real text is logged).
        """
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
        assert "Detection failed" not in response.json()["detail"]
        assert response.json()["detail"] == "Internal server error"

    def test_trigger_drift_with_feature_filter(self, client):
        """Test triggering drift detection with specific features."""
        mock_task = MagicMock()
        mock_task.id = "task-filtered"

        with patch("src.tasks.drift_monitoring_tasks.run_drift_detection") as mock_run:
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
        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_latest_drift_status.side_effect = RuntimeError("DB error")
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/drift/latest/propensity_v2.1.0")

        assert response.status_code == 500


class TestGetDriftHistory:
    """Test GET /monitoring/drift/history/{model_id} endpoint."""

    def test_get_drift_history_success(self, client, mock_drift_record):
        """Test getting drift history successfully."""
        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_active_alerts.return_value = [mock_alert_record]
            MockRepo.return_value = mock_repo

            response = client.get(
                "/monitoring/alerts",
                params={"model_id": "propensity_v2.1.0"},
            )

        assert response.status_code == 200
        mock_repo.get_active_alerts.assert_called_with("propensity_v2.1.0", limit=50)

    def test_list_alerts_with_status_filter(self, client, mock_alert_record):
        """Test listing alerts filtered by status."""
        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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

        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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

        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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

    def test_snooze_alert_returns_501_and_does_not_acknowledge(self, client, mock_alert_record):
        """SNOOZE must fail honestly (501), not silently masquerade as ACKNOWLEDGE.

        The backend has no snooze persistence (ml_monitoring_alerts has no
        'snoozed' status / 'snooze_until' column), so a snooze cannot be honored.
        Previously the branch called acknowledge_alert() and dropped the
        caller-supplied snooze_until while returning a 200, which misled clients
        into believing an alert was snoozed when it was only acknowledged.
        """
        mock_alert_record.status = "acknowledged"

        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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

            # Honest failure, not a fake success.
            assert response.status_code == 501
            assert "snooze" in response.json()["detail"].lower()
            # And it must NOT have silently acknowledged the alert.
            mock_repo.acknowledge_alert.assert_not_called()

    def test_update_alert_not_found(self, client):
        """Test updating a non-existent alert."""
        with patch("src.repositories.drift_monitoring.MonitoringAlertRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.MonitoringRunRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.MonitoringRunRepository") as MockRepo:
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
        with patch("src.repositories.drift_monitoring.MonitoringRunRepository") as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_recent_runs.return_value = []
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/runs")

        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 0
        assert data["runs"] == []

    def test_list_runs_honors_days_param(self, client, mock_run_record):
        """Issue #321 MED: /api/monitoring/runs must pass `days` through.

        Previously the route computed a cutoff datetime but never forwarded it
        to `get_recent_runs`, so the endpoint silently ignored the query param.
        Verify that `get_recent_runs` is called with a `since` kwarg derived
        from the requested days.
        """
        with patch("src.repositories.drift_monitoring.MonitoringRunRepository") as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_recent_runs.return_value = [mock_run_record]
            MockRepo.return_value = mock_repo

            response = client.get("/monitoring/runs", params={"days": 7})

            assert response.status_code == 200
            assert mock_repo.get_recent_runs.called
            kwargs = mock_repo.get_recent_runs.call_args.kwargs
            assert "since" in kwargs, f"expected get_recent_runs(since=...) kwarg, got {kwargs!r}"
            since = kwargs["since"]
            assert isinstance(since, datetime)
            expected = datetime.now(timezone.utc) - timedelta(days=7)
            # tolerate a few seconds of clock drift during test setup
            assert abs((since - expected).total_seconds()) < 30, (
                f"since={since!r} not within 30s of days=7 cutoff {expected!r}"
            )

    def test_list_runs_days_param_changes_cutoff(self, client, mock_run_record):
        """Calling /runs with days=7 vs days=30 must yield different `since` cutoffs."""
        captured = []

        with patch("src.repositories.drift_monitoring.MonitoringRunRepository") as MockRepo:
            mock_repo = AsyncMock()

            async def fake_get_recent_runs(*args, **kwargs):
                captured.append(kwargs.get("since"))
                return [mock_run_record]

            mock_repo.get_recent_runs.side_effect = fake_get_recent_runs
            MockRepo.return_value = mock_repo

            client.get("/monitoring/runs", params={"days": 7})
            client.get("/monitoring/runs", params={"days": 30})

        assert len(captured) == 2
        since_7, since_30 = captured
        assert since_7 is not None and since_30 is not None
        # 30-day cutoff should be earlier than 7-day cutoff
        assert since_30 < since_7, (
            f"days=30 cutoff {since_30!r} should be earlier than days=7 cutoff {since_7!r}"
        )


# =============================================================================
# MODEL HEALTH ENDPOINT TESTS
# =============================================================================


class TestGetModelHealth:
    """Test GET /monitoring/health/{model_id} endpoint."""

    def test_get_healthy_model(self, client, mock_drift_record, mock_run_record):
        """Test getting health of a healthy model."""
        mock_drift_record.severity = "low"  # maps to 0.25 drift score

        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockDriftRepo:
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
        mock_drift_record.severity = "medium"  # maps to 0.5 drift score
        mock_alert_record.status = "active"

        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockDriftRepo:
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
        mock_drift_record.severity = "critical"  # maps to 1.0 drift score

        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockDriftRepo:
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

        with patch("src.tasks.drift_monitoring_tasks.track_model_performance") as mock_track:
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

        with patch("src.services.performance_tracking.get_performance_tracker") as mock_get_tracker:
            with patch("src.repositories.drift_monitoring.PerformanceMetricRepository") as MockRepo:
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

        with patch("src.services.performance_tracking.get_performance_tracker") as mock_get_tracker:
            tracker = AsyncMock()
            tracker.check_performance_alerts.return_value = mock_alerts
            mock_get_tracker.return_value = tracker

            response = client.get("/monitoring/performance/propensity_v2.1.0/alerts")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["alert_count"] == 1
        assert data["alerts"][0]["severity"] == "high"


class TestCompareModelPerformance:
    """Test GET /monitoring/performance/{model_id}/compare/{other_model_id} endpoint."""

    def test_compare_models(self, client):
        """Route maps the tracker's NESTED comparison into the FLAT FE-facing contract.

        ``PerformanceTracker.compare_model_versions`` returns nested
        ``model_a``/``model_b`` objects plus ``metric`` /
        ``relative_difference_percent`` keys. The frontend (and its
        ``ModelComparisonResponse`` type) reads a FLAT shape:
        ``model_id`` / ``other_model_id`` / ``metric_name`` / ``model_value`` /
        ``other_model_value`` / ``difference_percent`` / ``better_model``.

        Before this fix the route returned the nested dict raw (no
        ``response_model``), so the page rendered "undefined undefined NaN%".
        This test pins the flat contract so the drift cannot recur silently.
        """
        # The REAL shape returned by tracker.compare_model_versions(...).
        mock_result = {
            "model_a": {"version": "propensity_v2.0.0", "value": 0.82, "trend": "stable"},
            "model_b": {"version": "propensity_v2.1.0", "value": 0.85, "trend": "improving"},
            "metric": "accuracy",
            "difference": 0.03,
            "relative_difference_percent": 3.6585,
            "better_model": "propensity_v2.1.0",
            "is_significant": False,
        }

        with patch("src.services.performance_tracking.get_performance_tracker") as mock_get_tracker:
            tracker = AsyncMock()
            tracker.compare_model_versions.return_value = mock_result
            mock_get_tracker.return_value = tracker

            response = client.get(
                "/monitoring/performance/propensity_v2.0.0/compare/propensity_v2.1.0",
                params={"metric_name": "accuracy"},
            )

        assert response.status_code == 200
        data = response.json()
        # Flat shape the frontend actually reads.
        assert data["model_id"] == "propensity_v2.0.0"
        assert data["other_model_id"] == "propensity_v2.1.0"
        assert data["metric_name"] == "accuracy"
        assert data["model_value"] == pytest.approx(0.82)
        assert data["other_model_value"] == pytest.approx(0.85)
        assert data["difference"] == pytest.approx(0.03)
        assert data["difference_percent"] == pytest.approx(3.6585)
        assert data["better_model"] == "propensity_v2.1.0"
        assert data["is_significant"] is False
        # The nested tracker shape must NOT leak through to the client.
        assert "model_a" not in data
        assert "relative_difference_percent" not in data


# =============================================================================
# PRODUCTION SWEEP ENDPOINT TESTS
# =============================================================================


class TestProductionSweep:
    """Test POST /monitoring/sweep/production endpoint."""

    def test_trigger_production_sweep(self, client):
        """Test triggering production sweep."""
        mock_task = MagicMock()
        mock_task.id = "sweep-task-123"

        with patch("src.tasks.drift_monitoring_tasks.check_all_production_models") as mock_check:
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
        """Test evaluating a model that needs retraining (real dataclass)."""
        from src.services.retraining_trigger import RetrainingDecision, TriggerReason

        decision = RetrainingDecision(
            should_retrain=True,
            reason=TriggerReason.DATA_DRIFT,
            confidence=0.85,
            drift_score=0.72,
            performance_score=0.91,
            details={
                "data_drift_score": 0.72,
                "model_drift_score": 0.1,
                "concept_drift_score": 0.0,
                "performance_current": 0.91,
                "performance_baseline": 0.99,
                "performance_drop": 0.08,
                "features_with_drift": 3,
            },
            requires_approval=True,
        )

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.evaluate_retraining_need.return_value = decision
            mock_get_service.return_value = service

            response = client.post("/monitoring/retraining/evaluate/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["should_retrain"] is True
        assert data["confidence"] == 0.85
        assert data["reasons"] == ["data_drift"]
        assert data["trigger_factors"]["features_with_drift"] == 3
        assert data["cooldown_active"] is False
        assert data["cooldown_ends_at"] is None
        assert data["recommended_action"] == "Trigger retraining (requires approval)"

    def test_evaluate_no_retraining_needed(self, client):
        """Test evaluating a model that doesn't need retraining (real dataclass)."""
        from src.services.retraining_trigger import RetrainingDecision

        decision = RetrainingDecision(
            should_retrain=False,
            reason=None,
            confidence=0.0,
            drift_score=0.05,
            performance_score=0.97,
            details={
                "data_drift_score": 0.05,
                "model_drift_score": 0.0,
                "concept_drift_score": 0.0,
                "performance_current": 0.97,
                "performance_baseline": 0.98,
                "performance_drop": 0.01,
                "features_with_drift": 0,
            },
            requires_approval=True,
        )

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.evaluate_retraining_need.return_value = decision
            mock_get_service.return_value = service

            response = client.post("/monitoring/retraining/evaluate/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["should_retrain"] is False
        assert data["reasons"] == []
        assert data["cooldown_active"] is False
        assert data["recommended_action"] == "Continue monitoring"

    def test_evaluate_with_real_retraining_decision(self, client):
        """Real ``RetrainingDecision`` must map to the response without
        AttributeError (the bug fixed in #547): the endpoint previously read
        five attributes the dataclass never defined (reasons, trigger_factors,
        cooldown_active, cooldown_ends_at, recommended_action), passing only
        against MagicMock doubles. Auto-trigger (no approval required) path."""
        from src.services.retraining_trigger import RetrainingDecision, TriggerReason

        decision = RetrainingDecision(
            should_retrain=True,
            reason=TriggerReason.DATA_DRIFT,
            confidence=0.92,
            drift_score=0.92,
            performance_score=0.88,
            details={
                "data_drift_score": 0.92,
                "model_drift_score": 0.3,
                "concept_drift_score": 0.1,
                "performance_current": 0.88,
                "performance_baseline": 0.95,
                "performance_drop": 0.07,
                "features_with_drift": 5,
            },
            requires_approval=False,
        )

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.evaluate_retraining_need.return_value = decision
            mock_get_service.return_value = service

            response = client.post("/monitoring/retraining/evaluate/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "propensity_v2.1.0"
        assert data["should_retrain"] is True
        assert data["confidence"] == 0.92
        assert data["reasons"] == ["data_drift"]
        assert data["trigger_factors"] == decision.details
        assert data["cooldown_active"] is False
        assert data["cooldown_ends_at"] is None
        assert data["recommended_action"] == "Auto-trigger retraining"

    def test_evaluate_cooldown_active(self, client):
        """When the service returns a cooldown decision, ``cooldown_active`` is
        derived from ``details['blocked_reason'] == 'cooldown_period'`` and
        ``cooldown_ends_at`` from real ``last_retraining`` + ``cooldown_hours``."""
        from src.services.retraining_trigger import RetrainingDecision

        last_retraining = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        decision = RetrainingDecision(
            should_retrain=False,
            reason=None,
            confidence=0.0,
            drift_score=0.6,
            performance_score=0.9,
            details={
                "blocked_reason": "cooldown_period",
                "last_retraining": last_retraining.isoformat(),
                "cooldown_hours": 24,
            },
        )

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.evaluate_retraining_need.return_value = decision
            mock_get_service.return_value = service

            response = client.post("/monitoring/retraining/evaluate/propensity_v2.1.0")

        assert response.status_code == 200
        data = response.json()
        assert data["should_retrain"] is False
        assert data["reasons"] == []
        assert data["cooldown_active"] is True
        expected_end = last_retraining + timedelta(hours=24)
        assert datetime.fromisoformat(data["cooldown_ends_at"]) == expected_end
        assert data["recommended_action"] == "Continue monitoring"


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

    def test_trigger_retraining_with_real_job_and_cohort(self, client):
        """Faithful test: the endpoint must build a valid response from a REAL
        RetrainingJob (which has created_at, NOT triggered_at/triggered_by) and
        thread the cohort contract into the service — without passing the
        unsupported triggered_by/notes/auto_approve kwargs the service rejects.

        The MagicMock-based test above masks the real dataclass mismatch; this
        one exercises the actual contract.
        """
        from src.services.retraining_trigger import (
            RetrainingJob,
            RetrainingStatus,
            TriggerReason,
        )

        real_job = RetrainingJob(
            job_id="rt-real-1",
            model_version="optum_v1",
            new_model_version="optum_v1_retrained_20260528",
            trigger_reason=TriggerReason.MANUAL,
            status=RetrainingStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            performance_before=0.66,
        )

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = MagicMock()
            service.trigger_retraining = AsyncMock(return_value=real_job)
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/trigger/optum_v1",
                json={
                    "reason": "manual",
                    "data_source": "data/rwd/optum/initiation",
                    "target_outcome": "initiated_biologic_180d",
                    "feature_manifest_source": "optum",
                    "auto_approve": True,
                },
            )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["job_id"] == "rt-real-1"
        assert data["status"] == "pending"
        assert data["triggered_at"] is not None  # sourced from job.created_at

        kwargs = service.trigger_retraining.await_args.kwargs
        # cohort identity threaded for a real retrain
        assert kwargs["cohort"]["data_source"] == "data/rwd/optum/initiation"
        assert kwargs["cohort"]["feature_manifest_source"] == "optum"
        # auto_approve=True → approved_by populated; unsupported kwargs gone
        assert kwargs["approved_by"] == "api_user"
        assert "triggered_by" not in kwargs
        assert "auto_approve" not in kwargs
        assert "notes" not in kwargs


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

    def test_get_retraining_status_training_real_job(self, client):
        """codex MEDIUM-1: a real in-progress job has domain status 'training',
        but the API enum exposes 'in_progress'. The status endpoint must map it
        (200, not 500) — the live pipeline keeps a job in 'training' for minutes,
        so this path is hit in practice."""
        from src.services.retraining_trigger import (
            RetrainingJob,
            RetrainingStatus,
            TriggerReason,
        )

        real_job = RetrainingJob(
            job_id="rt-training-1",
            model_version="optum_v1",
            new_model_version="optum_v1_retrained",
            trigger_reason=TriggerReason.MANUAL,
            status=RetrainingStatus.TRAINING,
            created_at=datetime.now(timezone.utc),
        )
        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = MagicMock()
            service.get_retraining_status = AsyncMock(return_value=real_job)
            mock_get_service.return_value = service

            response = client.get("/monitoring/retraining/status/rt-training-1")

        assert response.status_code == 200, response.text
        assert response.json()["status"] == "in_progress"

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


class TestValidationAucFromMetrics:
    """Unit tests for the #546 validation-AUC extraction helper."""

    def test_prefers_roc_auc_then_falls_through_keys(self):
        assert _validation_auc_from_metrics({"roc_auc": 0.91, "auc": 0.5}) == 0.91
        assert _validation_auc_from_metrics({"auc_roc": 0.88}) == 0.88
        assert _validation_auc_from_metrics({"auc": 0.7}) == 0.7
        assert _validation_auc_from_metrics({"val_auc": 0.66}) == 0.66

    def test_ignores_non_finite_and_non_numeric_and_bool(self):
        assert _validation_auc_from_metrics({}) is None
        assert _validation_auc_from_metrics({"roc_auc": float("nan")}) is None
        assert _validation_auc_from_metrics({"roc_auc": float("inf")}) is None
        assert _validation_auc_from_metrics({"roc_auc": "0.9"}) is None
        # bool is a subclass of int but must not count as a metric
        assert _validation_auc_from_metrics({"roc_auc": True}) is None


class _FakeTrainingRunRepo:
    """Real-ish MLTrainingRunRepository stand-in (#546).

    Holds a single real ``MLTrainingRun`` keyed by mlflow_run_id and resolves it
    via ``get_by_mlflow_run_id`` exactly like the real async repo — NOT a blanket
    MagicMock that would paper over a missing/invalid lookup.
    """

    def __init__(self, run):
        self._run = run

    def __call__(self, _client):  # constructed as MLTrainingRunRepository(client)
        return self

    async def get_by_mlflow_run_id(self, mlflow_run_id):
        if self._run is not None and self._run.mlflow_run_id == mlflow_run_id:
            return self._run
        return None


def _make_training_run(mlflow_run_id="abc123def456", *, status="finished", auc=0.89):
    """Build a real MLTrainingRun with a recorded validation AUC."""
    from src.repositories.ml_experiment import MLTrainingRun

    return MLTrainingRun(
        run_name="retrain-run",
        mlflow_run_id=mlflow_run_id,
        algorithm="xgboost",
        status=status,
        validation_metrics={"roc_auc": auc} if auc is not None else {},
    )


def _make_completed_job():
    """Build a completed-job stand-in shaped for _retraining_job_to_response."""
    job = MagicMock()
    job.job_id = "retrain-job-123"
    job.model_version = "propensity_v2.1.0"
    job.status = MagicMock(value="completed")
    job.trigger_reason = MagicMock(value="data_drift")
    job.triggered_at = datetime.now(timezone.utc) - timedelta(hours=2)
    job.triggered_by = "api_user"
    job.approved_at = datetime.now(timezone.utc) - timedelta(hours=2)
    job.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    job.completed_at = datetime.now(timezone.utc)
    job.performance_before = 0.82
    job.performance_after = 0.89
    job.notes = "Successfully improved"
    return job


def _patch_provenance(run):
    """Patch the provenance gate's dependencies to resolve `run` (or None).

    Patches the async client factory (no real Supabase) and the repo class with a
    real-ish fake keyed by mlflow_run_id.
    """
    return (
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "src.repositories.ml_experiment.MLTrainingRunRepository",
            _FakeTrainingRunRepo(run),
        ),
    )


class TestCompleteRetraining:
    """Test POST /monitoring/retraining/{job_id}/complete endpoint."""

    def test_complete_retraining_not_found(self, client):
        """Test completing non-existent retraining job (run resolves; job does not)."""
        run = _make_training_run(auc=0.89)
        client_patch, repo_patch = _patch_provenance(run)
        with (
            patch(
                "src.services.retraining_trigger.get_retraining_trigger_service"
            ) as mock_get_service,
            client_patch,
            repo_patch,
        ):
            service = AsyncMock()
            service.complete_retraining.return_value = None
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/nonexistent/complete",
                json={
                    "performance_after": 0.89,
                    "success": True,
                    "mlflow_run_id": "abc123def456",
                },
            )

        assert response.status_code == 404

    def test_complete_success_without_provenance_is_rejected(self, client):
        """#546: a SUCCESS completion with no provenance pointer must be rejected.

        Before the provenance gate, an admin caller could complete a job with any
        plausible-but-fake ``performance_after`` (e.g. 0.999) and NO evidence it
        came from a real run — the service would persist status=completed. The gate
        must reject (4xx) such a request and persist NOTHING.
        """
        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.complete_retraining.return_value = _make_completed_job()
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/complete",
                json={
                    "performance_after": 0.999,
                    "success": True,
                    # no mlflow_run_id -> no provenance
                },
            )

        # Rejected with 4xx and the service is never asked to persist.
        assert response.status_code == 422, response.text
        assert "provenance" in response.json()["detail"].lower()
        service.complete_retraining.assert_not_called()

    def test_complete_success_with_invalid_metric_is_rejected(self, client):
        """#546: a SUCCESS completion with an out-of-range metric is rejected.

        Even with a provenance pointer, a non-finite or out-of-[0,1] AUC cannot
        have come from a real validation run, so persist nothing.
        """
        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/complete",
                json={
                    "performance_after": 1.5,  # AUC can't exceed 1.0
                    "success": True,
                    "mlflow_run_id": "abc123def456",
                },
            )

        assert response.status_code == 422, response.text
        service.complete_retraining.assert_not_called()

    def test_complete_success_run_not_found_is_rejected(self, client):
        """#546: provenance pointer that resolves to NO training run -> 422.

        An invented mlflow_run_id must not certify a success completion.
        """
        client_patch, repo_patch = _patch_provenance(None)  # repo returns None
        with (
            patch(
                "src.services.retraining_trigger.get_retraining_trigger_service"
            ) as mock_get_service,
            client_patch,
            repo_patch,
        ):
            service = AsyncMock()
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/complete",
                json={
                    "performance_after": 0.89,
                    "success": True,
                    "mlflow_run_id": "invented-run-id",
                },
            )

        assert response.status_code == 422, response.text
        assert "no training run found" in response.json()["detail"].lower()
        service.complete_retraining.assert_not_called()

    def test_complete_success_run_not_finished_is_rejected(self, client):
        """#546: a run that is not 'finished' cannot certify a success completion."""
        run = _make_training_run(status="running", auc=0.89)
        client_patch, repo_patch = _patch_provenance(run)
        with (
            patch(
                "src.services.retraining_trigger.get_retraining_trigger_service"
            ) as mock_get_service,
            client_patch,
            repo_patch,
        ):
            service = AsyncMock()
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/complete",
                json={
                    "performance_after": 0.89,
                    "success": True,
                    "mlflow_run_id": "abc123def456",
                },
            )

        assert response.status_code == 422, response.text
        assert "finished" in response.json()["detail"].lower()
        service.complete_retraining.assert_not_called()

    def test_complete_success_metric_mismatch_is_rejected(self, client):
        """#546: submitted metric must MATCH the run's recorded validation AUC."""
        run = _make_training_run(auc=0.89)  # real run recorded 0.89
        client_patch, repo_patch = _patch_provenance(run)
        with (
            patch(
                "src.services.retraining_trigger.get_retraining_trigger_service"
            ) as mock_get_service,
            client_patch,
            repo_patch,
        ):
            service = AsyncMock()
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/complete",
                json={
                    "performance_after": 0.95,  # caller inflates beyond tolerance
                    "success": True,
                    "mlflow_run_id": "abc123def456",
                },
            )

        assert response.status_code == 422, response.text
        assert "does not match" in response.json()["detail"].lower()
        service.complete_retraining.assert_not_called()

    def test_complete_success_with_matching_finished_run_persists(self, client):
        """#546: SUCCESS + finished run whose validation AUC matches -> 200 + persisted."""
        run = _make_training_run(auc=0.89)
        client_patch, repo_patch = _patch_provenance(run)
        with (
            patch(
                "src.services.retraining_trigger.get_retraining_trigger_service"
            ) as mock_get_service,
            client_patch,
            repo_patch,
        ):
            service = AsyncMock()
            service.complete_retraining.return_value = _make_completed_job()
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/complete",
                json={
                    "performance_after": 0.89,  # matches the run within tol
                    "success": True,
                    "mlflow_run_id": "abc123def456",
                    "notes": "Successfully improved",
                },
            )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "completed"
        assert data["performance_after"] == 0.89
        # The provenance pointer is threaded through to the service.
        _, kwargs = service.complete_retraining.call_args
        assert kwargs["mlflow_run_id"] == "abc123def456"

    def test_complete_failure_unchanged(self, client):
        """#546 non-breaking: a FAILURE completion still works with no provenance.

        The provenance dependencies are NOT patched here — the gate must not even
        attempt a run lookup for success=False.
        """
        mock_job = MagicMock()
        mock_job.job_id = "retrain-job-123"
        mock_job.model_version = "propensity_v2.1.0"
        mock_job.status = MagicMock(value="failed")
        mock_job.trigger_reason = MagicMock(value="data_drift")
        mock_job.triggered_at = datetime.now(timezone.utc) - timedelta(hours=2)
        mock_job.triggered_by = "api_user"
        mock_job.approved_at = None
        mock_job.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_job.completed_at = datetime.now(timezone.utc)
        mock_job.performance_before = 0.82
        mock_job.performance_after = None
        mock_job.notes = "Run failed"

        with patch(
            "src.services.retraining_trigger.get_retraining_trigger_service"
        ) as mock_get_service:
            service = AsyncMock()
            service.complete_retraining.return_value = mock_job
            mock_get_service.return_value = service

            response = client.post(
                "/monitoring/retraining/retrain-job-123/complete",
                json={
                    "performance_after": 0.0,
                    "success": False,
                    # no mlflow_run_id, no plausible metric -> still allowed
                },
            )

        assert response.status_code == 200, response.text
        service.complete_retraining.assert_called_once()


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


class TestComputeEndpointAuthorization:
    """Security: mutating/compute monitoring endpoints must require >= operator.

    Finding 1 (MEDIUM privilege): the POST drift/performance/sweep endpoints
    queue Celery compute jobs. Before the fix they required only authentication
    (or nothing at all), so any VIEWER could trigger expensive background work.

    Intent (auth.py): OPERATOR "manage experiments ... digital twin" — the
    operational tier. Peer monitoring/operational routes (sentinels, experiments,
    feedback, digital_twin) gate mutating ops with ``require_operator``; the
    retraining lifecycle in this same module already uses ``require_admin``.
    Drift detection / performance recording / production sweep are operational
    monitoring compute → OPERATOR, consistent with sentinels.py.

    We override ``require_auth`` with a VIEWER user so the real
    ``require_operator`` dependency runs against a low-privilege principal and
    must reject with 403.
    """

    @staticmethod
    def _viewer_user():
        return {
            "id": "viewer-1",
            "email": "viewer@example.com",
            "app_metadata": {"role": "viewer"},
        }

    @staticmethod
    def _operator_user():
        return {
            "id": "operator-1",
            "email": "operator@example.com",
            "app_metadata": {"role": "operator"},
        }

    def _override(self, app, user):
        from src.api.dependencies.auth import require_auth

        app.dependency_overrides[require_auth] = lambda: user

    @pytest.mark.parametrize(
        ("method", "url", "json_body", "params"),
        [
            (
                "post",
                "/monitoring/drift/detect",
                {"model_id": "m1", "time_window": "7d"},
                {"async_mode": True},
            ),
            (
                "post",
                "/monitoring/performance/record",
                {"model_id": "m1", "predictions": [1, 0], "actuals": [1, 0]},
                {"async_mode": True},
            ),
            (
                "post",
                "/monitoring/sweep/production",
                None,
                {"time_window": "7d"},
            ),
        ],
    )
    def test_viewer_forbidden_on_compute_endpoints(
        self, app, client, method, url, json_body, params
    ):
        """A VIEWER must receive 403 on compute/queue endpoints."""
        self._override(app, self._viewer_user())
        try:
            resp = getattr(client, method)(url, json=json_body, params=params)
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 403, resp.text

    def test_operator_allowed_on_drift_detect(self, app, client):
        """An OPERATOR passes the gate (reaches the handler, queues a task)."""
        self._override(app, self._operator_user())
        mock_task = MagicMock()
        mock_task.id = "task-op"
        try:
            with patch("src.tasks.drift_monitoring_tasks.run_drift_detection") as mock_run:
                mock_run.delay.return_value = mock_task
                resp = client.post(
                    "/monitoring/drift/detect",
                    params={"async_mode": True},
                    json={"model_id": "m1", "time_window": "7d"},
                )
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "queued"


class TestErrorDetailDoesNotLeakInternals:
    """Security: 500 responses must not echo raw internal exception text.

    Finding 2 (LOW info-disclosure): handlers previously returned
    ``detail=str(e)`` which can surface stack/driver internals, table names,
    connection strings, etc. The handler must log server-side and return a
    generic client-facing message.
    """

    def test_drift_sync_failure_returns_generic_detail(self, client):
        secret = "DB password=hunter2 at host db.internal:5432"
        with patch(
            "src.tasks.drift_monitoring_tasks.run_drift_detection",
            side_effect=RuntimeError(secret),
        ):
            resp = client.post(
                "/monitoring/drift/detect",
                params={"async_mode": False},
                json={"model_id": "m1", "time_window": "7d"},
            )
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "hunter2" not in detail
        assert "db.internal" not in detail
        assert detail == "Internal server error"

    def test_latest_drift_server_error_returns_generic_detail(self, client):
        secret = "psycopg2.OperationalError: relation drift_history secret-table"
        with patch("src.repositories.drift_monitoring.DriftHistoryRepository") as MockRepo:
            mock_repo = AsyncMock()
            mock_repo.get_latest_drift_status.side_effect = RuntimeError(secret)
            MockRepo.return_value = mock_repo
            resp = client.get("/monitoring/drift/latest/m1")
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "secret-table" not in detail
        assert "psycopg2" not in detail
        assert detail == "Internal server error"


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
