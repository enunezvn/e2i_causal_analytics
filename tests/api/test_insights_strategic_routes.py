"""Route tests for the strategic-insight endpoints.

These exercise the deterministic no-LLM fallback path (offline/deterministic even
where OPENAI_API_KEY is set locally — mirrors CI). Not mocking values: the fallback
computes real grounded text from the request payload. The live LLM path and the two
externally-grounded endpoints (knowledge-graph, model-performance) are verified
manually on the droplet (plan Task 12).
"""

import pytest

from src.api.dependencies.auth import require_analyst
from src.api.main import app


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
        "effects": [
            {
                "treatment": "copay_card",
                "outcome": "adherence_180d",
                "ate": 0.043,
                "ate_ci_lower": 0.02,
                "ate_ci_upper": 0.066,
                "status": "proceed",
                "selected_estimator": "CausalForestDML",
            }
        ],
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
        "model_version": "csu_adherence_v3",
        "n_scored": 250,
        "mean_prob": 0.34,
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
        "projected_lift_pct": 6.0,
        "solver_status": "optimal",
    }
    r = test_client.post("/api/insights/resource-optimization", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    # ace4e372 made the DSPy interpretation the primary content and reclassified
    # the verbatim-summary path as the (honestly labelled) fallback; with no LM
    # in tests, surfacing the agent's summary IS the fallback.
    assert data["is_fallback"] is True
    assert "high-ROI" in data["insight"]
    assert any(c["label"] == "Projected lift" for c in data["grounding"])


def test_model_performance_insight_degrades_on_backend_error(test_client, monkeypatch):
    """Backend unreachable -> honest is_fallback response, NOT a 500 (codex BUG 1)."""

    async def _boom(*a, **k):
        raise RuntimeError("supabase unreachable")

    monkeypatch.setattr(
        "src.services.performance_tracking.PerformanceTracker.get_performance_trend", _boom
    )
    r = test_client.post("/api/insights/model-performance", json={"model_version": "m1"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "unavailable" in data["insight"].lower()


def test_knowledge_graph_insight_degrades_on_backend_error(test_client, monkeypatch):
    """FalkorDB/semantic-memory unreachable -> honest is_fallback response, NOT a 500."""

    def _boom(*a, **k):
        raise RuntimeError("falkordb unreachable")

    monkeypatch.setattr("src.memory.semantic_memory.get_semantic_memory", _boom)
    r = test_client.post("/api/insights/knowledge-graph", json={"brand": "All"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "unavailable" in data["insight"].lower()


def test_treatment_effect_insight_fallback(test_client):
    body = {
        "cohort": "hcp_adoption",
        "brand": "Remibrutinib",
        "treatment_var": "treatment_arm",
        "outcome_var": "adopted",
        "confounders": ["peer_influence_score", "influence_network_size"],
        "ate": 0.1448,
        "ci_lower": 0.1426,
        "ci_upper": 0.1470,
        "p_value": 0.0004,
        "n": 5000,
        "estimator": "linear_dml",
    }
    r = test_client.post("/api/insights/treatment-effect", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "hcp_adoption" in data["insight"]
    assert "refutation tests were not run" in data["insight"]
    assert "excludes 0" in data["insight"]
    assert any(c["label"] == "ATE" for c in data["grounding"])
    assert data["provenance"]
    assert data["generated_at"]


def test_treatment_effect_insight_ci_straddles_zero(test_client):
    body = {
        "cohort": "initiation",
        "brand": "Fabhalta",
        "treatment_var": "treatment_arm",
        "outcome_var": "initiated_180d",
        "confounders": ["disease_severity"],
        "ate": 0.01,
        "ci_lower": -0.02,
        "ci_upper": 0.04,
        "p_value": 0.5,
        "n": 1200,
        "estimator": "linear_dml",
    }
    r = test_client.post("/api/insights/treatment-effect", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "not distinguishable from no effect" in data["insight"]
    assert data["is_fallback"] is True
    assert "straddles 0" in data["insight"]


def _fake_opportunities_feed():
    """A real-shaped OpportunityListResponse for the server-derived brief."""
    from src.api.routes.gaps import (
        ImplementationDifficulty,
        OpportunityListResponse,
        PerformanceGap,
        PrioritizedOpportunity,
        ROIEstimate,
    )

    return OpportunityListResponse(
        total_count=1,
        quick_wins_count=2,
        steady_plays_count=1,
        strategic_bets_count=0,
        suppressed_count=3,
        total_addressable_value=5_000_000.0,
        opportunities=[
            PrioritizedOpportunity(
                rank=1,
                gap=PerformanceGap(
                    gap_id="g1",
                    metric="trx",
                    segment="region",
                    segment_value="Northeast",
                    current_value=85.0,
                    target_value=100.0,
                    gap_size=15.0,
                    gap_percentage=42.0,
                    gap_type="vs_target",
                ),
                roi_estimate=ROIEstimate(
                    gap_id="g1",
                    estimated_revenue_impact=1_200_000.0,
                    estimated_cost_to_close=300_000.0,
                    expected_roi=3.2,
                    risk_adjusted_roi=2.5,
                    payback_period_months=6,
                    attribution_level="partial",
                    attribution_rate=0.65,
                    confidence=0.8,
                ),
                recommended_action="Deploy field triggers to lapsed writers",
                implementation_difficulty=ImplementationDifficulty.MEDIUM,
                time_to_impact="3-6 months",
            )
        ],
    )


def test_executive_brief_insight_fallback_is_server_derived(test_client, monkeypatch):
    # The endpoint derives the figures SERVER-SIDE from the gaps read path
    # (codex PR-5 round 3): the request carries only the brand.
    async def _stub(**kwargs):
        assert kwargs.get("brand") == "Kisqali"
        return _fake_opportunities_feed()

    monkeypatch.setattr("src.api.routes.gaps.list_opportunities", _stub)
    r = test_client.post("/api/insights/executive-brief", json={"brand": "Kisqali"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "3.2x ROI" in data["insight"]
    assert "$5.0M" in data["insight"]
    assert "3 low-value opportunities were suppressed" in data["insight"]
    assert any(c["label"] == "Addressable value" for c in data["grounding"])
    assert data["provenance"] == "Gap-analyzer ROI opportunities (server-derived)"
    assert data["generated_at"]


def test_executive_brief_ignores_caller_posted_figures(test_client, monkeypatch):
    # An authenticated caller must NOT be able to mint a grounded-looking
    # brief from arbitrary posted figures (codex PR-5 round 3): extra body
    # fields are ignored and the grounding reflects the server's own feed.
    async def _stub(**kwargs):
        return _fake_opportunities_feed()

    monkeypatch.setattr("src.api.routes.gaps.list_opportunities", _stub)
    body = {
        "brand": "Kisqali",
        "total_addressable_value": 99_000_000.0,
        "opportunities": [
            {
                "rank": 1,
                "recommended_action": "Buy a superyacht",
                "expected_roi": 99.0,
                "revenue_impact": 99_000_000.0,
            }
        ],
    }
    r = test_client.post("/api/insights/executive-brief", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "$99.0M" not in data["insight"]
    assert "superyacht" not in data["insight"]
    assert "$5.0M" in data["insight"]


def test_executive_brief_insight_no_signal_is_honest(test_client, monkeypatch):
    from src.api.routes.gaps import OpportunityListResponse

    async def _stub(**kwargs):
        return OpportunityListResponse(
            total_count=0,
            quick_wins_count=0,
            steady_plays_count=0,
            strategic_bets_count=0,
            suppressed_count=0,
            total_addressable_value=0.0,
            opportunities=[],
        )

    monkeypatch.setattr("src.api.routes.gaps.list_opportunities", _stub)
    r = test_client.post("/api/insights/executive-brief", json={"brand": "Fabhalta"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "run a gap analysis" in data["insight"].lower()
    assert data["key_takeaways"] == []


def test_executive_brief_feed_outage_degrades_honestly(test_client, monkeypatch):
    # A gaps read failure is a data-source outage, NOT "no signal": the
    # response must say so and never 500 (codex PR-5 rounds 2-3).
    async def _boom(**kwargs):
        raise RuntimeError("gap store unreachable")

    monkeypatch.setattr("src.api.routes.gaps.list_opportunities", _boom)
    r = test_client.post("/api/insights/executive-brief", json={"brand": "Kisqali"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "data-source failure" in data["insight"]
    assert "run a gap analysis" not in data["insight"].lower()
    assert data["provenance"] == "Gap-analyzer ROI opportunities (unavailable)"


# ---- /insights/hte --------------------------------------------------------------


def _hte_record(status: str = "completed", **overrides):
    from src.api.routes.segments import (
        AnalysisStatus,
        CATEResult,
        SegmentAnalysisResponse,
    )

    base = {
        "analysis_id": "seg_test123",
        "status": AnalysisStatus(status),
        "brand": "Remibrutinib",
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "overall_ate": 0.1106,
        "heterogeneity_score": 0.26,
        "expected_lift_pp": 0.0,
        "optimal_allocation_summary": "No reliable differential-targeting opportunity.",
        "cate_by_segment": {
            "disease_severity_band": [
                CATEResult(
                    segment_name="disease_severity_band",
                    segment_value="high",
                    cate_estimate=0.1772,
                    cate_ci_lower=0.1267,
                    cate_ci_upper=0.2277,
                    sample_size=1385,
                    statistical_significance=True,
                ),
                CATEResult(
                    segment_name="disease_severity_band",
                    segment_value="low",
                    cate_estimate=0.0338,
                    cate_ci_lower=-0.0280,
                    cate_ci_upper=0.0955,
                    sample_size=2498,
                    statistical_significance=False,
                ),
            ]
        },
    }
    base.update(overrides)
    return SegmentAnalysisResponse(**base)


def test_hte_insight_fallback_grounds_in_persisted_record(test_client, monkeypatch):
    # The endpoint must read the SERVER-persisted record (analysis_id-only
    # request) and render its real figures in the deterministic fallback.
    record = _hte_record()

    async def _get(analysis_id):
        assert analysis_id == "seg_test123"
        return record

    monkeypatch.setattr("src.api.routes.segments.get_persisted_analysis", _get)
    r = test_client.post("/api/insights/hte", json={"analysis_id": "seg_test123"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "+11.1pp" in data["insight"]
    assert "1 of 2 segments" in data["insight"]
    assert data["provenance"] == "Segment-level CATE analysis (server-derived)"
    chips = {c["label"]: c["value"] for c in data["grounding"]}
    assert chips["Significant segments"] == "1/2"


def test_hte_insight_missing_record_degrades_honestly(test_client, monkeypatch):
    # Expired/unknown analysis_id is an honest "re-run" degrade, never a 500
    # and never a fabricated interpretation.
    async def _get(analysis_id):
        return None

    monkeypatch.setattr("src.api.routes.segments.get_persisted_analysis", _get)
    r = test_client.post("/api/insights/hte", json={"analysis_id": "seg_gone"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "not found" in data["insight"]
    assert data["provenance"] == "Persisted segment-level CATE analysis (unavailable)"
    assert data["grounding"] == []


def test_hte_insight_incomplete_run_degrades_honestly(test_client, monkeypatch):
    record = _hte_record(status="failed")

    async def _get(analysis_id):
        return record

    monkeypatch.setattr("src.api.routes.segments.get_persisted_analysis", _get)
    r = test_client.post("/api/insights/hte", json={"analysis_id": "seg_test123"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "'failed'" in data["insight"]
    assert data["provenance"] == "Persisted segment-level CATE analysis (incomplete run)"
