"""Unit tests for drift-monitoring repository client wiring (#845).

The five drift/monitoring/retraining repositories were constructed *without* a
Supabase client at every call site, so ``BaseRepository``'s ``if not self.client``
guards silently turned every read/write into a no-op (FAILS-OPEN — the Celery
drift sweep persisted nothing, the monitoring API read empty, retraining history
was never written). #845 wires a real async client into those constructions via
``get_drift_monitoring_client`` and makes the no-DB path **fail closed** (surface
an error) instead of fabricating an empty success.

These tests are CI-safe (no DB): they patch the resolver. They assert:

* the resolver returns the resolved client and fails closed (raises) when
  Supabase is unconfigured or hands back ``None``;
* each surface (drift API read, Celery drift task, retraining service,
  performance tracker) now *wires the resolved client into the repository* and
  *fails closed* when the client cannot be resolved — rather than silently
  no-op'ing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.repositories.drift_monitoring as dm
from src.api.routes.monitoring import router as monitoring_router
from src.memory.services.factories import ServiceConnectionError


def _failing_resolver():
    async def _raise():
        raise ServiceConnectionError("Supabase", "unconfigured (test)")

    return _raise


def _client_resolver(sentinel):
    async def _resolve():
        return sentinel

    return _resolve


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------
class TestResolver:
    async def test_returns_resolved_client(self, monkeypatch):
        import src.memory.services.factories as factories

        sentinel = object()

        async def _fake():
            return sentinel

        monkeypatch.setattr(factories, "get_async_supabase_client", _fake)
        assert await dm.get_drift_monitoring_client() is sentinel

    async def test_fails_closed_when_unconfigured(self, monkeypatch):
        import src.memory.services.factories as factories

        async def _raise():
            raise ServiceConnectionError("Supabase", "SUPABASE_URL not set")

        monkeypatch.setattr(factories, "get_async_supabase_client", _raise)
        with pytest.raises(ServiceConnectionError):
            await dm.get_drift_monitoring_client()

    async def test_fails_closed_on_none_client(self, monkeypatch):
        import src.memory.services.factories as factories

        async def _none():
            return None

        monkeypatch.setattr(factories, "get_async_supabase_client", _none)
        with pytest.raises(ServiceConnectionError):
            await dm.get_drift_monitoring_client()


# ---------------------------------------------------------------------------
# API surface: GET /monitoring/drift/latest/{model_id}
# ---------------------------------------------------------------------------
def _monitoring_client():
    app = FastAPI()
    app.include_router(monitoring_router)
    # raise_server_exceptions=False so a fail-closed 5xx is returned, not re-raised.
    return TestClient(app, raise_server_exceptions=False)


class TestDriftLatestEndpoint:
    def test_fails_closed_when_db_unconfigured(self):
        """No DB -> resolver raises -> endpoint returns 5xx (not an empty 200)."""
        client = _monitoring_client()
        with patch.object(dm, "get_drift_monitoring_client", _failing_resolver()):
            resp = client.get("/monitoring/drift/latest/propensity_v2.1.0")
        assert resp.status_code >= 500

    def test_wires_resolved_client_into_repository(self):
        """The repository is constructed WITH the resolved client (not client-less)."""
        sentinel = MagicMock(name="resolved_client")
        captured: dict = {}
        real_repo = MagicMock()
        real_repo.get_latest_drift_status = AsyncMock(return_value=[])

        def _fake_repo_cls(client=None, *a, **k):
            captured["client"] = client
            return real_repo

        client = _monitoring_client()
        with (
            patch.object(dm, "get_drift_monitoring_client", _client_resolver(sentinel)),
            patch.object(dm, "DriftHistoryRepository", _fake_repo_cls),
        ):
            resp = client.get("/monitoring/drift/latest/propensity_v2.1.0")
        assert resp.status_code == 200
        assert captured.get("client") is sentinel


# ---------------------------------------------------------------------------
# Celery task surface: run_drift_detection
# ---------------------------------------------------------------------------
class TestRunDriftDetectionTask:
    def test_fails_closed_when_db_unconfigured(self):
        """The drift task must FAIL (raise) when the DB cannot be resolved —
        not return a fabricated 'completed' run that persisted nothing."""
        from src.tasks.drift_monitoring_tasks import run_drift_detection

        with patch.object(dm, "get_drift_monitoring_client", _failing_resolver()):
            with pytest.raises(ServiceConnectionError):
                run_drift_detection(
                    model_id="propensity_v2.1.0",
                    time_window="7d",
                    features=["x"],  # skip connector.get_available_features
                    check_data_drift=False,
                    check_model_drift=False,
                    check_concept_drift=False,
                )


# ---------------------------------------------------------------------------
# Service surface: RetrainingTriggerService.evaluate_retraining_need
# ---------------------------------------------------------------------------
class TestRetrainingServiceFailClosed:
    async def test_evaluate_retraining_need_fails_closed(self):
        from src.services.retraining_trigger import RetrainingTriggerService

        with patch.object(dm, "get_drift_monitoring_client", _failing_resolver()):
            with pytest.raises(ServiceConnectionError):
                await RetrainingTriggerService().evaluate_retraining_need("propensity_v2.1.0")


# ---------------------------------------------------------------------------
# Service surface: PerformanceTracker.record_snapshot
# ---------------------------------------------------------------------------
class TestPerformanceTrackerWiring:
    async def test_record_performance_wires_resolved_client(self):
        from src.services.performance_tracking import PerformanceTracker

        sentinel = MagicMock(name="resolved_client")
        captured: dict = {}
        repo = MagicMock()
        repo.record_metrics = AsyncMock(return_value=[])

        def _fake_cls(client=None, *a, **k):
            captured["client"] = client
            return repo

        tracker = PerformanceTracker()
        with (
            patch.object(dm, "get_drift_monitoring_client", _client_resolver(sentinel)),
            patch.object(dm, "PerformanceMetricRepository", _fake_cls),
        ):
            await tracker.record_performance(
                model_version="propensity_v2.1.0",
                predictions=np.array([1, 0, 1, 0]),
                actuals=np.array([1, 0, 0, 0]),
            )
        assert captured.get("client") is sentinel

    async def test_check_performance_alerts_fails_closed(self):
        """A broad ``except Exception`` in ``check_performance_alerts`` used to
        swallow the resolver's ServiceConnectionError into an empty alert list
        (fail-OPEN: reads as 'healthy, no alerts'). It must now surface (#845)."""
        from src.services.performance_tracking import PerformanceTracker

        with patch.object(dm, "get_drift_monitoring_client", _failing_resolver()):
            with pytest.raises(ServiceConnectionError):
                await PerformanceTracker().check_performance_alerts("propensity_v2.1.0")
