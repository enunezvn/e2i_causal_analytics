"""GET /performance/{model_id}/trend + /alerts surface the sampling context (2026-09-07).

The trend classifier is sampling-aware; the page card needs WHY a label was
given (``reason`` / ``basis``) and the fold size the judgement rests on. The
fields are additive and defaulted so tracker doubles without them still serve.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.monitoring import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)  # the router carries its own /monitoring prefix
    return TestClient(app)


def _full_trend():
    return SimpleNamespace(
        current_value=0.7094,
        baseline_value=0.8152,
        change_percent=-12.97,
        trend="stable",
        is_significant=False,
        alert_threshold_breached=False,
        alert_threshold=0.7337,
        sample_size=134,
        standard_error=0.0495,
        z_score=-2.14,
        slope_t_stat=-0.9,
        n_points=9,
        basis="within_noise",
        reason="-13.0% vs the 8-fold baseline is within sampling noise at n=134 (z=-2.1; ±2.5 needed)",
    )


def test_trend_response_carries_sampling_fields(client):
    rec = MagicMock(metric_name="auc_roc", metric_value=0.7094)
    rec.measured_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with patch("src.services.performance_tracking.get_performance_tracker") as get_tracker:
        with patch("src.repositories.drift_monitoring.PerformanceMetricRepository") as Repo:
            tracker = AsyncMock()
            tracker.get_performance_trend.return_value = _full_trend()
            get_tracker.return_value = tracker
            repo = AsyncMock()
            repo.get_metric_trend.return_value = [rec]
            Repo.return_value = repo
            r = client.get(
                "/monitoring/performance/hcp_adoption_remibrutinib_goldstd_lr_v1/trend",
                params={"metric_name": "auc_roc", "days": 365},
            )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["trend"] == "stable"
    assert d["sample_size"] == 134
    assert d["z_score"] == pytest.approx(-2.14)
    assert d["standard_error"] == pytest.approx(0.0495)
    assert d["slope_t_stat"] == pytest.approx(-0.9)
    assert d["n_points"] == 9
    assert d["basis"] == "within_noise"
    assert "sampling noise" in d["reason"]


def test_trend_response_defaults_when_tracker_double_lacks_the_fields(client):
    """A MagicMock trend (as older tests use) yields None/""/0, never a 500."""
    mock_trend = MagicMock()
    mock_trend.current_value = 0.85
    mock_trend.baseline_value = 0.82
    mock_trend.change_percent = 3.7
    mock_trend.trend = "improving"
    mock_trend.is_significant = False
    mock_trend.alert_threshold_breached = False
    mock_trend.alert_threshold = 0.74
    with patch("src.services.performance_tracking.get_performance_tracker") as get_tracker:
        with patch("src.repositories.drift_monitoring.PerformanceMetricRepository") as Repo:
            tracker = AsyncMock()
            tracker.get_performance_trend.return_value = mock_trend
            get_tracker.return_value = tracker
            repo = AsyncMock()
            repo.get_metric_trend.return_value = []
            Repo.return_value = repo
            r = client.get("/monitoring/performance/m/trend", params={"metric_name": "accuracy"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["sample_size"] is None and d["z_score"] is None
    assert d["basis"] == "" and d["reason"] == "" and d["n_points"] == 0


def test_alert_items_carry_sampling_fields(client):
    alerts = [
        {
            "model_version": "m",
            "metric_name": "auc_roc",
            "current_value": 0.60,
            "baseline_value": 0.85,
            "change_percent": -29.4,
            "trend": "degrading",
            "severity": "high",
            "message": "auc_roc degraded by 29.4%",
            "sample_size": 230,
            "z_score": -6.1,
            "basis": "level",
            "reason": "auc_roc 0.600 is 6.1 standard errors below the 8-fold baseline 0.850 (n=230, -29.4%)",
        },
        {  # legacy producer without the new keys
            "model_version": "m",
            "metric_name": "recall",
            "current_value": 0.5,
            "baseline_value": 0.7,
            "change_percent": -28.6,
            "trend": "degrading",
            "severity": "high",
            "message": "recall degraded by 28.6%",
        },
    ]
    with patch("src.services.performance_tracking.get_performance_tracker") as get_tracker:
        tracker = AsyncMock()
        tracker.check_performance_alerts.return_value = alerts
        get_tracker.return_value = tracker
        r = client.get("/monitoring/performance/m/alerts")
    assert r.status_code == 200, r.text
    items = r.json()["alerts"]
    assert items[0]["sample_size"] == 230 and items[0]["basis"] == "level"
    assert items[0]["z_score"] == pytest.approx(-6.1)
    assert items[1]["sample_size"] is None and items[1]["basis"] == ""
