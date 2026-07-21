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
async def test_get_brand_goldstd_summary_slope_detail_tolerated_by_response_model(monkeypatch):
    """B2/B3 on the async monitoring path: the calibration_slope headline uses
    the 1 + mean(|slope - 1|) fold and the summary's extra detail key must NOT
    break the /performance/brand-summary response model (extra='ignore')."""
    store = {
        "ml_model_registry": [
            {"id": "1", "model_name": "persistence_remibrutinib_goldstd_lr_v1"},
            {"id": "2", "model_name": "discontinuation_remibrutinib_goldstd_lr_v1"},
        ],
        "ml_performance_metrics": [
            {
                "model_id": "1",
                "metric_name": "calibration_slope",
                "metric_value": 0.70,
                "source": "holdout",
                "sample_size": 415,
                "ci_lower": 0.55,
                "ci_upper": 0.88,
            },
            {
                "model_id": "2",
                "metric_name": "calibration_slope",
                "metric_value": 1.30,
                "source": "holdout",
                "sample_size": 415,
                "ci_lower": 1.12,
                "ci_upper": 1.45,
            },
        ],
    }

    async def _fake_client():
        return _FakeClient(store)

    monkeypatch.setattr(
        "src.repositories.drift_monitoring.get_drift_monitoring_client", _fake_client
    )
    summary = await PerformanceTracker().get_brand_goldstd_summary("Remibrutinib")
    # Signed cancellation is dead: 0.70 & 1.30 -> 1.30, not 1.00.
    assert summary["calibration_slope"] == pytest.approx(1.30)
    assert "calibration_slope_detail" in summary
    assert len(summary["calibration_slope_detail"]["models"]) == 2

    from src.api.routes.monitoring import BrandPerformanceSummaryResponse

    resp = BrandPerformanceSummaryResponse(available=True, **summary)
    assert resp.calibration_slope == pytest.approx(1.30)


@pytest.mark.asyncio
async def test_get_brand_goldstd_summary_none_when_no_models(monkeypatch):
    async def _fake_client():
        return _FakeClient({"ml_model_registry": [], "ml_performance_metrics": []})

    monkeypatch.setattr(
        "src.repositories.drift_monitoring.get_drift_monitoring_client", _fake_client
    )
    assert await PerformanceTracker().get_brand_goldstd_summary("Kisqali") is None
