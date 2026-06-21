import pytest

from src.services.performance_tracking import PerformanceTracker


class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """Minimal async PostgREST builder stub: records table, returns canned data."""

    def __init__(self, store, table):
        self._store, self._table = store, table

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def in_(self, *_a, **_k):
        return self

    async def execute(self):
        return _Resp(self._store.get(self._table, []))


class _FakeClient:
    def __init__(self, store):
        self._store = store

    def table(self, name):
        return _FakeQuery(self._store, name)


@pytest.mark.asyncio
async def test_get_brand_goldstd_summary_averages_holdout(monkeypatch):
    store = {
        "ml_model_registry": [
            {"id": "1", "model_name": "initiation_kisqali_goldstd_lr_v1"},
            {"id": "2", "model_name": "persistence_kisqali_goldstd_lr_v1"},
        ],
        "ml_performance_metrics": [
            {"model_id": "1", "metric_name": "accuracy", "metric_value": 0.69, "source": "holdout"},
            {"model_id": "2", "metric_name": "accuracy", "metric_value": 0.71, "source": "holdout"},
        ],
    }

    async def _fake_client():
        return _FakeClient(store)

    monkeypatch.setattr(
        "src.repositories.drift_monitoring.get_drift_monitoring_client", _fake_client
    )
    summary = await PerformanceTracker().get_brand_goldstd_summary("Kisqali")
    assert summary["brand"] == "Kisqali"
    assert summary["n_models"] == 2
    assert summary["accuracy"] == pytest.approx(0.70)
    assert summary["is_synthetic_cohort"] is True


@pytest.mark.asyncio
async def test_get_brand_goldstd_summary_none_when_no_models(monkeypatch):
    async def _fake_client():
        return _FakeClient({"ml_model_registry": [], "ml_performance_metrics": []})

    monkeypatch.setattr(
        "src.repositories.drift_monitoring.get_drift_monitoring_client", _fake_client
    )
    assert await PerformanceTracker().get_brand_goldstd_summary("Kisqali") is None
