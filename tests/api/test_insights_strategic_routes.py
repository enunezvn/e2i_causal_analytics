"""Route tests for the strategic-insight endpoints.

These exercise the deterministic no-LLM fallback path (offline/deterministic even
where OPENAI_API_KEY is set locally — mirrors CI). Not mocking values: the fallback
computes real grounded text from the request payload. The live LLM path and the two
externally-grounded endpoints (knowledge-graph, model-performance) are verified
manually on the droplet (plan Task 12).
"""
import pytest

from src.api.main import app
from src.api.dependencies.auth import require_analyst


@pytest.fixture(autouse=True)
def _force_fallback_and_auth(monkeypatch):
    monkeypatch.setattr(
        "src.optimization.dspy_lm.ensure_dspy_configured",
        lambda *a, **k: False,
    )
    app.dependency_overrides[require_analyst] = lambda: {"user_id": "test", "role": "analyst"}
    yield
    app.dependency_overrides.pop(require_analyst, None)


def test_causal_discovery_insight_fallback(test_client):
    body = {
        "brand": "Kisqali",
        "grain": "patient",
        "effects": [{
            "treatment": "copay_card", "outcome": "adherence_180d", "ate": 0.043,
            "ate_ci_lower": 0.02, "ate_ci_upper": 0.066, "status": "proceed",
            "selected_estimator": "CausalForestDML",
        }],
    }
    r = test_client.post("/api/insights/causal-discovery", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "copay_card" in data["insight"]
    assert {"label", "value"} <= set(data["grounding"][0].keys())
    assert data["provenance"]
    assert data["generated_at"]


def test_predictive_cohort_insight_fallback(test_client):
    body = {
        "model_version": "csu_adherence_v3", "n_scored": 250, "mean_prob": 0.34,
        "top_targets": [{"entity_id": "HCP7", "probability": 0.91}],
        "top_drivers": [{"feature": "prior_adherence", "importance": 0.4}],
    }
    r = test_client.post("/api/insights/predictive-cohort", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "HCP7" in data["insight"]


def test_resource_optimization_insight_surfaces_summary(test_client):
    body = {
        "optimization_summary": "Reallocating to high-ROI HCPs lifts projected outcome 6%.",
        "recommendations": ["Shift 12% budget to segment A"],
        "projected_lift_pct": 6.0, "solver_status": "optimal",
    }
    r = test_client.post("/api/insights/resource-optimization", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is False
    assert "high-ROI" in data["insight"]
    assert any(c["label"] == "Projected lift" for c in data["grounding"])
