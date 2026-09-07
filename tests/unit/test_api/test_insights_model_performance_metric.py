"""/insights/model-performance reads the SAME metric the page card shows (2026-09-07)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.routes import insights_strategic as mod


class _FakeTracker:
    def __init__(self) -> None:
        self.trend_calls: list[tuple[str, str]] = []

    async def get_performance_trend(self, model_version, metric_name="accuracy"):
        self.trend_calls.append((model_version, metric_name))
        return SimpleNamespace(
            current_value=0.7094,
            baseline_value=0.8152,
            trend="stable",
            reason="-13.0% vs the 8-fold baseline is within sampling noise at n=134",
        )

    async def get_confusion_matrix(self, model_version):
        return None

    async def get_roc_curve(self, model_version):
        return None

    async def check_performance_alerts(self, model_version):
        return []


@pytest.fixture
def fake_tracker(monkeypatch):
    tracker = _FakeTracker()
    monkeypatch.setattr(
        "src.services.performance_tracking.get_performance_tracker", lambda *a, **k: tracker
    )

    async def _no_cache(key):
        return None

    async def _no_set(key, payload):
        return None

    monkeypatch.setattr(mod, "cache_get", _no_cache)
    monkeypatch.setattr(mod, "cache_set", _no_set)
    return tracker


def test_request_defaults_to_the_page_default_metric_and_rejects_unknown_metrics():
    assert mod.ModelPerfInsightRequest(model_version="m").metric_name == "auc_roc"
    assert (
        mod.ModelPerfInsightRequest(model_version="m", metric_name="recall").metric_name == "recall"
    )
    with pytest.raises(Exception):
        mod.ModelPerfInsightRequest(model_version="m", metric_name="brier_score")


@pytest.mark.asyncio
async def test_insight_trends_the_requested_metric_and_grounds_its_reason(fake_tracker):
    resp = await mod.model_performance_insight(
        mod.ModelPerfInsightRequest(model_version="hcp_adoption_remibrutinib_goldstd_lr_v1"),
        user={"role": "analyst"},
    )
    assert fake_tracker.trend_calls == [("hcp_adoption_remibrutinib_goldstd_lr_v1", "auc_roc")]
    labels = [c.label for c in resp.grounding]
    assert labels[:2] == ["Current AUC-ROC", "Baseline AUC-ROC"]
    # No LM in tests → the honest fallback carries the tracker's reason verbatim.
    assert "within sampling noise" in resp.insight

    resp = await mod.model_performance_insight(
        mod.ModelPerfInsightRequest(model_version="m", metric_name="recall"),
        user={"role": "analyst"},
    )
    assert fake_tracker.trend_calls[-1] == ("m", "recall")
    assert resp.grounding[0].label == "Current Recall"
