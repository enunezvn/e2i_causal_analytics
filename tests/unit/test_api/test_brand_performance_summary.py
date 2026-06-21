import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import monitoring


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(monitoring.router, prefix="/api")
    return TestClient(app)


def test_brand_summary_available(client, monkeypatch):
    class _Tracker:
        async def get_brand_goldstd_summary(self, brand):
            return {
                "brand": brand or "all",
                "n_models": 4,
                "accuracy": 0.70,
                "precision": 0.66,
                "recall": 0.55,
                "f1": 0.60,
                "auc_roc": 0.75,
                "is_synthetic_cohort": True,
            }

    monkeypatch.setattr(
        "src.services.performance_tracking.get_performance_tracker", lambda: _Tracker()
    )
    r = client.get("/api/monitoring/performance/brand-summary", params={"brand": "Kisqali"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["n_models"] == 4
    assert body["accuracy"] == 0.70
    assert body["brand"] == "Kisqali"


def test_brand_summary_honest_empty(client, monkeypatch):
    class _Tracker:
        async def get_brand_goldstd_summary(self, brand):
            return None

    monkeypatch.setattr(
        "src.services.performance_tracking.get_performance_tracker", lambda: _Tracker()
    )
    r = client.get("/api/monitoring/performance/brand-summary")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["accuracy"] is None
    assert body["brand"] == "all"
