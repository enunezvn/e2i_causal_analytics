"""Route tests for the strategic-insight endpoints.

These exercise the deterministic no-LLM fallback path (offline/deterministic even
where OPENAI_API_KEY is set locally — mirrors CI). Not mocking values: the fallback
computes real grounded text from the request payload. The live LLM path and the two
externally-grounded endpoints (knowledge-graph, model-performance) are verified
manually on the droplet (plan Task 12).
"""

import time

import pytest

from src.api.dependencies.auth import require_analyst
from src.api.main import app


@pytest.fixture(autouse=True)
def _force_fallback_and_auth(monkeypatch):
    monkeypatch.setattr(
        "src.optimization.dspy_lm.ensure_dspy_configured",
        lambda *a, **k: False,
    )

    # Keep the strategic-insight routes HERMETIC: the exec-brief/HTE routes now
    # fetch clinical context, which fans out to real OpenFDA/ChEMBL/CT.gov/PubMed
    # REST APIs. Default to the honest fail-open result (None) so tests neither
    # hit the network nor go flaky; the dedicated clinical-context tests below
    # override this with a real payload.
    async def _no_clinical(brand, outcome="TRx"):
        return None

    monkeypatch.setattr("src.insights.clinical_context.fetch_clinical_payload", _no_clinical)
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


def test_causal_discovery_insight_gates_on_clinical_positioning(test_client):
    # The brand's labeled target population + line of therapy grounds the
    # interpretation so a modeled effect in a clinically off-target population is
    # not recommended (curated label facts — no network, hermetic in CI).
    body = {
        "brand": "Remibrutinib",
        "grain": "patient",
        "effects": [
            {
                "treatment": "treatment_arm",
                "outcome": "persistent_180d",
                "ate": 0.088,
                "ate_ci_lower": 0.087,
                "ate_ci_upper": 0.089,
                "status": "proceed",
                "selected_estimator": "LinearDML",
            }
        ],
    }
    r = test_client.post("/api/insights/causal-discovery", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert any(
        c["label"] == "Clinical positioning" and c["value"] == "applied" for c in data["grounding"]
    )
    # Remibrutinib's antihistamine-refractory positioning reaches the narrative.
    assert "treatment-naive" in data["insight"].lower()


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


def test_predictive_whatif_insight_fallback(test_client):
    body = {
        "model_version": "persistence_remibrutinib_goldstd_lr_v1",
        "features": {"disease_severity": 5.6, "academic_hcp": 0},
        "probability": 0.87,
        "confidence": 0.87,
        "cohort_mean": 0.45,
        "n_scored": 4847,
        "top_drivers": [{"feature": "disease_severity", "importance": -1.21}],
    }
    r = test_client.post("/api/insights/predictive-whatif", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    # Grounded in the entered profile + score, labeled by entity kind, and
    # explicit that a what-if is predictive, not causal.
    assert "hypothetical patient" in data["insight"]
    assert "0.87" in data["insight"]
    assert "not a causal estimate" in data["insight"]
    assert data["provenance"] == "What-if prediction + per-row SHAP"


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


# Fixture graph for the KG page-parity tests. Shapes match SemanticMemory list_*
# output (flat edge properties). Kisqali chain onboarding->nrx, a Fabhalta-only
# edge (its variable drops out of the Kisqali scope), and an untagged structural
# EXPLAINS edge kept under every brand.
_KG_NODES = [
    {"id": "var:onboarding", "name": "patient_onboarding", "type": "Variable"},
    {"id": "var:nrx", "name": "nrx_volume", "type": "Variable"},
    {"id": "var:other", "name": "other_var", "type": "Variable"},
    {"id": "kpi:trx", "name": "TRx", "type": "KPI"},
]
_KG_RELS = [
    {
        "id": "e1",
        "source_id": "var:onboarding",
        "target_id": "var:nrx",
        "type": "CAUSES",
        "brand": "Kisqali",
        "confidence": 0.9,
    },
    {
        "id": "e2",
        "source_id": "var:other",
        "target_id": "var:nrx",
        "type": "CAUSES",
        "brand": "Fabhalta",
        "confidence": 0.8,
    },
    {
        "id": "e3",
        "source_id": "var:nrx",
        "target_id": "kpi:trx",
        "type": "EXPLAINS",
        "confidence": 0.7,
    },
]


def _fake_semantic_memory(monkeypatch):
    from unittest.mock import MagicMock

    sm = MagicMock()
    sm.list_nodes.return_value = list(_KG_NODES)
    sm.list_relationships.return_value = list(_KG_RELS)
    sm.count_nodes.return_value = len(_KG_NODES)
    monkeypatch.setattr("src.memory.semantic_memory.get_semantic_memory", lambda: sm)
    return sm


def test_knowledge_graph_insight_grounds_on_page_parity_reads(test_client, monkeypatch):
    """The grounding reads use the page's exact fetch scope (entity/relationship
    type filters at the 2000 window), not an unfiltered 500-row sample."""
    from src.insights import knowledge_graph as kg

    sm = _fake_semantic_memory(monkeypatch)
    r = test_client.post("/api/insights/knowledge-graph", json={"brand": "All"})
    assert r.status_code == 200, r.text
    sm.list_nodes.assert_called_once_with(
        entity_types=kg.PAGE_ENTITY_TYPES, limit=kg.PAGE_FETCH_LIMIT, curated_only=True
    )
    sm.list_relationships.assert_called_once_with(
        relationship_types=kg.PAGE_RELATIONSHIP_TYPES,
        limit=kg.PAGE_FETCH_LIMIT,
        curated_only=True,
    )
    sm.count_nodes.assert_called_once_with(entity_types=kg.PAGE_ENTITY_TYPES, curated_only=True)
    data = r.json()
    assert data["is_fallback"] is True  # no LM in tests -> deterministic fallback
    # All scope: every node is touched by >=1 edge -> 4 nodes / 3 relationships.
    chips = {c["label"]: c["value"] for c in data["grounding"]}
    assert chips["Nodes"] == "4"
    assert chips["Relationships"] == "3"


def test_knowledge_graph_insight_variable_narrows_grounding(test_client, monkeypatch):
    """A variable narrows the grounding to its causal neighborhood under the
    brand scope — the same subgraph the page renders for that selection."""
    _fake_semantic_memory(monkeypatch)
    r = test_client.post(
        "/api/insights/knowledge-graph",
        json={"brand": "Kisqali", "variable": "var:onboarding"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    # Scope string carries brand + variable NAME into the narrative.
    assert "Kisqali / patient_onboarding neighborhood" in data["insight"]
    # Neighborhood: onboarding -> nrx (chain) + kpi:trx (structural context).
    # var:other (Fabhalta-only) is excluded by the brand scope before the BFS.
    chips = {c["label"]: c["value"] for c in data["grounding"]}
    assert chips["Nodes"] == "3"
    assert chips["Relationships"] == "2"


def test_knowledge_graph_insight_unknown_variable_degrades_honestly(test_client, monkeypatch):
    """A variable id outside the brand scope (stale or bogus) yields an honest
    'not in scope' fallback — never a grounded-looking insight, never a 500."""
    _fake_semantic_memory(monkeypatch)
    r = test_client.post(
        "/api/insights/knowledge-graph",
        json={"brand": "Kisqali", "variable": "var:other"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "var:other" in data["insight"]
    assert "not part of the Kisqali causal graph" in data["insight"]
    assert data["provenance"] == "Curated knowledge graph (variable not in scope)"
    assert data["grounding"] == []


def test_digital_twin_insight_fallback_is_server_derived(test_client, monkeypatch):
    """Grounding comes from the twin repo + availability map (server-derived);
    with no LM the deterministic fallback narrates the REAL rows and discloses
    the synthetic-gold substrate."""
    from unittest.mock import AsyncMock, MagicMock

    from src.digital_twin.effect.provider import INTERVENTION_CATALOG

    repo = MagicMock()
    repo.client = MagicMock()
    repo.list_active_models = AsyncMock(return_value=[{"model_name": "hcp_Remibrutinib_twin"}])
    repo.simulations.list_simulations = AsyncMock(
        return_value=[
            {
                "simulation_status": "completed",
                "recommendation": "deploy",
                "intervention_type": "digital_engagement",
                "simulated_ate": 0.231,
                "simulated_ci_lower": 0.198,
                "simulated_ci_upper": 0.265,
                "data_provenance": "cohort_estimated_synthetic_gold_v1",
            }
        ]
    )
    monkeypatch.setattr("src.api.routes.digital_twin._get_twin_repo", AsyncMock(return_value=repo))
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value={v: True for v, _ in INTERVENTION_CATALOG}),
    )
    # Cache must not replay a previous payload for these fabricated test rows.
    monkeypatch.setattr("src.api.routes.insights_strategic.cache_get", AsyncMock(return_value=None))
    monkeypatch.setattr("src.api.routes.insights_strategic.cache_set", AsyncMock())

    r = test_client.post("/api/insights/digital-twin", json={"brand": "Remibrutinib"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "hcp_Remibrutinib_twin" in data["insight"]
    assert "digital_engagement" in data["insight"]
    # Honesty-critical: the synthetic substrate is disclosed in the narrative.
    assert "synthetic-gold" in data["insight"]
    chips = {c["label"]: c["value"] for c in data["grounding"]}
    assert chips["Identified interventions"] == "8/8"
    assert data["provenance"] == "Digital-twin simulation program (server-derived)"


def test_digital_twin_insight_degrades_on_backend_error(test_client, monkeypatch):
    """Twin repo unreachable -> honest is_fallback response, NOT a 500."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "src.api.routes.digital_twin._get_twin_repo",
        AsyncMock(side_effect=RuntimeError("supabase unreachable")),
    )
    r = test_client.post("/api/insights/digital-twin", json={"brand": "Kisqali"})
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


def test_executive_brief_fallback_is_cached_briefly(test_client, monkeypatch):
    # A fallback payload marks a transient state (LM outage / rejected sample):
    # it must be cached for minutes, not pinned for the full hour like a real
    # insight — the "Factual summary" stickiness behind the 2026-07-05 report.
    async def _stub(**kwargs):
        return _fake_opportunities_feed()

    monkeypatch.setattr("src.api.routes.gaps.list_opportunities", _stub)
    seen = {}

    async def _cache_miss(key):
        # The dev box runs a live redis: force a miss so the generate path
        # (and its cache_set) runs regardless of what earlier tests cached.
        return None

    async def _capture_cache_set(key, value, ttl_seconds=3600):
        seen["ttl"] = ttl_seconds
        seen["is_fallback"] = value.get("is_fallback")

    monkeypatch.setattr("src.api.routes.insights_strategic.cache_get", _cache_miss)
    monkeypatch.setattr("src.api.routes.insights_strategic.cache_set", _capture_cache_set)
    r = test_client.post("/api/insights/executive-brief", json={"brand": "Kisqali"})
    assert r.status_code == 200, r.text
    assert seen == {"ttl": 300, "is_fallback": True}

    monkeypatch.setattr(
        "src.insights.executive_brief.generate_insight",
        lambda g: {
            "insight": "Lead with Northeast at 3.2x.",
            "key_takeaways": [],
            "grounding": g["grounding"],
            "is_fallback": False,
        },
    )
    r = test_client.post("/api/insights/executive-brief", json={"brand": "Kisqali"})
    assert r.status_code == 200, r.text
    assert seen == {"ttl": 3600, "is_fallback": False}


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
        CATEResult,
        SegmentAnalysisResponse,
        SegmentAnalysisStatus,
    )

    base = {
        "analysis_id": "seg_test123",
        "status": SegmentAnalysisStatus(status),
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


_CLINICAL_PAYLOAD = {
    "brand": "Remibrutinib",
    "drug_name": "remibrutinib",
    "disease": "chronic spontaneous urticaria",
    "our_outcome": "persistent_180d",
    "mapped_endpoint": "UAS7 change",
    "mechanism": {"mechanism_of_action": "BTK inhibitor", "source": "chembl"},
    "pivotal_endpoints": {"endpoints": [], "source": "clinicaltrials.gov"},
    "real_world_evidence": None,
    "seminal_real_world_evidence": None,
    "approved_indications": {
        "indications": [],
        "limitations_of_use": None,
        "boxed_warning": None,
        "source": "openfda",
    },
    "competitor_landscape": {
        "competitors": ["Xolair (omalizumab)"],
        "count": 1,
        "source": "curated",
    },
    "honesty_label": "label",
}


def test_hte_insight_fallback_surfaces_clinical_context(test_client, monkeypatch):
    # Commercial outputs don't happen in a clinical vacuum: when the clinical
    # fetch returns a payload, the (digit-free) clinical setting appears in the
    # deterministic fallback narrative and grounding chips.
    async def _get(analysis_id):
        return _hte_record()

    async def _clinical(brand, outcome="TRx"):
        assert brand == "Remibrutinib"
        assert outcome == "persistent_180d"  # the analyzed outcome, not a default
        return _CLINICAL_PAYLOAD

    monkeypatch.setattr("src.api.routes.segments.get_persisted_analysis", _get)
    monkeypatch.setattr("src.insights.clinical_context.fetch_clinical_payload", _clinical)
    _force_insight_cache_miss(monkeypatch)
    r = test_client.post("/api/insights/hte", json={"analysis_id": "seg_test123"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "BTK inhibitor" in data["insight"]
    assert any(c["label"] == "Clinical context" for c in data["grounding"])


def test_executive_brief_fallback_surfaces_clinical_context(test_client, monkeypatch):
    async def _feed(**kwargs):
        return _fake_opportunities_feed()

    async def _clinical(brand, outcome="TRx"):
        assert brand == "Kisqali"  # the requested brand reaches the fetch
        return _CLINICAL_PAYLOAD

    monkeypatch.setattr("src.api.routes.gaps.list_opportunities", _feed)
    monkeypatch.setattr("src.insights.clinical_context.fetch_clinical_payload", _clinical)
    _force_insight_cache_miss(monkeypatch)
    r = test_client.post("/api/insights/executive-brief", json={"brand": "Kisqali"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "BTK inhibitor" in data["insight"]
    assert any(c["label"] == "Clinical context" for c in data["grounding"])


# ---- /insights/home-kpis (server-derived KPI grid interpretation) --------------
def _force_insight_cache_miss(monkeypatch):
    # The dev box runs a live redis: force a miss so the generate path runs
    # regardless of what earlier runs cached (same trick as the exec-brief tests).
    async def _cache_miss(key):
        return None

    async def _cache_noop(key, value, ttl_seconds=3600):
        return None

    monkeypatch.setattr("src.api.routes.insights_strategic.cache_get", _cache_miss)
    monkeypatch.setattr("src.api.routes.insights_strategic.cache_set", _cache_noop)


def _fake_kpi_calculator(seen_contexts):
    from src.kpi.models import (
        CalculationType,
        KPIBatchResult,
        KPIMetadata,
        KPIResult,
        KPIStatus,
        Workstream,
    )

    metas = [
        KPIMetadata(
            id="WS3-BI-001",
            name="Total TRx",
            definition="Total prescriptions",
            formula="count(rx)",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.WS3_BUSINESS,
        ),
        KPIMetadata(
            id="WS1-MP-002",
            name="Holdout Accuracy",
            definition="Holdout accuracy",
            formula="acc",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_MODEL_PERFORMANCE,
            value_format="percent",
        ),
        KPIMetadata(
            id="WS2-TR-001",
            name="Trigger Precision",
            definition="Trigger precision",
            formula="tp/(tp+fp)",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS2_TRIGGERS,
        ),
        KPIMetadata(
            id="KIS-CLI-001",
            name="Kisqali - Oncologist Reach",
            definition="Oncologists reached",
            formula="count(hcp)",
            calculation_type=CalculationType.DIRECT,
            workstream=Workstream.BRAND_SPECIFIC,
            brand="Kisqali",
        ),
    ]

    class _FakeCalc:
        def list_kpis(self, workstream=None, causal_library=None):
            return metas

        def calculate_batch(self, kpi_ids=None, workstream=None, use_cache=True, context=None):
            seen_contexts.append(context)
            batch = KPIBatchResult()
            batch.add_result(
                KPIResult(kpi_id="WS3-BI-001", value=11634.0, status=KPIStatus.INFORMATIONAL)
            )
            batch.add_result(KPIResult(kpi_id="WS1-MP-002", value=0.874, status=KPIStatus.WARNING))
            # Honest not-computed row: must be EXCLUDED from the grounding.
            batch.add_result(
                KPIResult(kpi_id="WS2-TR-001", value=None, error="no view for this scope")
            )
            # Sibling-brand KPI: computes portfolio-wide, but under a different
            # brand scope it must be EXCLUDED from the grounding entirely
            # (automatic brand scoping, 2026-07-22 — mirrors the dashboard
            # grid); under 'All' it stays first-class. Returned unconditionally
            # here to prove the grounding filter drops it even when the batch
            # carries it.
            batch.add_result(KPIResult(kpi_id="KIS-CLI-001", value=2890.0, status=KPIStatus.GOOD))
            return batch

    return _FakeCalc()


def test_home_kpi_insight_fallback_is_server_derived(test_client, monkeypatch):
    # The request carries ONLY the scope; figures come from the server's own
    # calculator under the same brand/region context the dashboard batch uses.
    seen_contexts = []
    monkeypatch.setattr(
        "src.api.routes.kpi.get_kpi_calculator",
        lambda: _fake_kpi_calculator(seen_contexts),
    )
    _force_insight_cache_miss(monkeypatch)
    body = {"brand": "Fabhalta", "region": "northeast"}
    r = test_client.post("/api/insights/home-kpis", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert seen_contexts == [{"brand": "Fabhalta", "region": "northeast"}]
    assert "Fabhalta / Northeast" in data["insight"]
    assert "Total TRx [ws3_business]: 11,634" in data["insight"]
    # percent value_format renders as the dashboard does (0-1 ratio -> NN.N%)
    assert "Holdout Accuracy [ws1_model_performance]: 87.4%" in data["insight"]
    # the not-computed KPI is excluded, and coverage says so honestly
    assert "Trigger Precision" not in data["insight"]
    # another brand's hard-bound KPI is scoped out of the grounding entirely
    # (automatic brand scoping) — including the coverage denominator.
    assert "Kisqali - Oncologist Reach" not in data["insight"]
    assert "2 of 3 defined KPIs computed" in data["insight"]
    chips = {c["label"]: c["value"] for c in data["grounding"]}
    assert chips["Brand"] == "Fabhalta"
    assert chips["Territory"] == "Northeast"
    assert chips["Computed"] == "2/3"
    assert data["provenance"] == "Registry KPIs recomputed for this scope (server-derived)"


def test_home_kpi_insight_ignores_caller_posted_figures(test_client, monkeypatch):
    # Same trust boundary as executive-brief: posted figures must not be able
    # to mint a grounded-looking insight.
    seen_contexts = []
    monkeypatch.setattr(
        "src.api.routes.kpi.get_kpi_calculator",
        lambda: _fake_kpi_calculator(seen_contexts),
    )
    _force_insight_cache_miss(monkeypatch)
    body = {
        "brand": "Fabhalta",
        "region": "northeast",
        "kpis": [{"name": "Fabricated KPI", "value": 99999.0, "status": "critical"}],
    }
    r = test_client.post("/api/insights/home-kpis", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "Fabricated" not in data["insight"]
    assert "99,999" not in data["insight"]
    assert "Total TRx" in data["insight"]


def test_home_kpi_insight_all_us_portfolio_scope(test_client, monkeypatch):
    # brand=All / region omitted -> empty calculator context (portfolio view).
    seen_contexts = []
    monkeypatch.setattr(
        "src.api.routes.kpi.get_kpi_calculator",
        lambda: _fake_kpi_calculator(seen_contexts),
    )
    _force_insight_cache_miss(monkeypatch)
    r = test_client.post("/api/insights/home-kpis", json={"brand": "All"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert seen_contexts == [{}]
    assert "All brands (portfolio) / All US" in data["insight"]
    # No brand scope -> every brand's hard-bound rows are first-class.
    assert "Kisqali - Oncologist Reach [brand_specific]: 2,890 (good)" in data["insight"]
    assert "4 of 4 defined KPIs computed" not in data["insight"]  # WS2 row honest-null
    assert "3 of 4 defined KPIs computed" in data["insight"]
    chips = {c["label"]: c["value"] for c in data["grounding"]}
    assert chips["Territory"] == "All US"


def test_home_kpi_insight_carries_the_authored_mitigation_playbook(test_client, monkeypatch):
    """Item 2b (2026-07-22): the claims-lag mitigation playbook is served
    VERBATIM from the vocabulary — deterministic, present even when the
    narrative itself is the factual fallback, so the structural block stays
    actionable regardless of LM availability."""
    seen_contexts = []
    monkeypatch.setattr(
        "src.api.routes.kpi.get_kpi_calculator",
        lambda: _fake_kpi_calculator(seen_contexts),
    )
    _force_insight_cache_miss(monkeypatch)
    r = test_client.post("/api/insights/home-kpis", json={"brand": "Fabhalta"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True  # no LM in tests — playbook unaffected
    pb = data["mitigation_playbook"]
    assert pb is not None
    assert "Faster adjudicated (closed) claims are not achievable" in pb["preamble"]
    assert "not vetted or contracted suppliers" in pb["vendor_note"]
    by_name = {sc["name"]: sc for sc in pb["source_classes"]}
    assert "IQVIA" in by_name["Open (pre-adjudicated) claims"]["illustrative_vendors"]
    nowcast = by_name["Completion-factor nowcast on closed claims"]
    assert nowcast["status"] and "live" in nowcast["status"]


def test_home_kpi_insight_degrades_on_backend_error(test_client, monkeypatch):
    def _boom():
        raise RuntimeError("supabase unreachable")

    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", _boom)
    r = test_client.post("/api/insights/home-kpis", json={"brand": "Fabhalta"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "unavailable" in data["insight"]
    assert data["provenance"] == "Registry KPIs for this scope (unavailable)"
    assert data["grounding"] == []


def _clinical_payload():
    """Real-shaped ClinicalContextService payload (remibrutinib, trimmed)."""
    return {
        "brand": "Remibrutinib",
        "drug_name": "remibrutinib",
        "disease": "Chronic spontaneous urticaria",
        "our_outcome": "adopted",
        "our_treatment": "treatment_arm",
        "mapped_endpoint": None,
        "treatment_context": {
            "column": "treatment_arm",
            "label": "on remibrutinib therapy",
            "framing": "being on remibrutinib",
            "kind": "drug_therapy",
            "source": "curated",
        },
        "analysis_framing": "This analysis asks what being on remibrutinib does to prescriber adoption.",
        "analysis_grounding": None,
        "mechanism": {
            "mechanism_of_action": "Bruton tyrosine kinase (BTK) inhibitor",
            "source": "chembl",
        },
        "pivotal_endpoints": {
            "endpoints": [
                {
                    "measure": "Change from baseline in UAS7 at Week 12",
                    "time_frame": "Week 12",
                    "nct_id": "NCT05030311",
                }
            ],
            "source": "clinicaltrials.gov",
        },
        "real_world_evidence": None,
        "seminal_real_world_evidence": None,
        "approved_indications": {
            "indications": ["RHAPSIDO is indicated for chronic spontaneous urticaria in adults."],
            "limitations_of_use": None,
            "boxed_warning": None,
            "source": "openfda",
        },
        "competitor_landscape": {
            "competitors": ["Xolair (omalizumab)", "Dupixent (dupilumab)"],
            "count": 2,
            "source": "curated",
        },
        "causal_evidence": None,
        "honesty_label": "Effect estimate = a SYNTHETIC patient cohort ...",
    }


_NARRATIVE_BODY = {
    "brand": "Remibrutinib",
    "grain": "hcp",
    "treatment": "treatment_arm",
    "outcome": "adopted",
    "ate": 0.14,
    "ate_ci_lower": 0.05,
    "ate_ci_upper": 0.23,
    "gate_decision": "proceed",
}


def test_clinical_narrative_fallback_grounds_in_server_fetched_facts(test_client, monkeypatch):
    # Facts are fetched SERVER-side: stub the service, not the request.
    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context",
        lambda self,
        brand,
        outcome,
        treatment=None,
        include_causal_evidence=False: _clinical_payload(),
    )
    # Redis is LIVE on this box and the grounding-derived key repeats across
    # runs: force a miss so this exercises the generate path, not a payload
    # cached from a previous run within the 300s TTL (home-kpis precedent).
    _force_insight_cache_miss(monkeypatch)
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True  # conftest forces the no-LM path
    # Pin the DERIVED grounding, not booleans: the fallback composes the strings.
    assert "ATE +0.1400 [95% CI +0.0500, +0.2300]" in data["insight"]
    assert "survived all robustness checks" in data["insight"]
    assert "Bruton tyrosine kinase (BTK) inhibitor" in data["insight"]
    assert "Xolair (omalizumab)" in data["insight"]
    assert "Analysis grain: hcp." in data["insight"]
    chips = {c["label"]: c["value"] for c in data["grounding"]}
    assert chips["Analysis"] == "treatment_arm -> adopted"
    assert data["key_takeaways"] == []
    assert data["provenance"].startswith("LLM synthesis of the labeled clinical-context sources")


def test_clinical_narrative_unknown_brand_404(test_client):
    # The app's global http_exception_handler (src/api/main.py) rewrites EVERY
    # HTTPException(404) into a generic EndpointNotFoundError body — same
    # behavior the sibling GET /causal/clinical-context route hits (its own
    # test bypasses the app and calls the route function directly, so it never
    # exercises this handler) — so only the status code is checkable here.
    r = test_client.post(
        "/api/insights/clinical-narrative", json={**_NARRATIVE_BODY, "brand": "NotABrand"}
    )
    assert r.status_code == 404


def test_clinical_narrative_fetch_failure_degrades_to_result_only(test_client, monkeypatch):
    def _boom(self, brand, outcome, treatment=None, include_causal_evidence=False):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context", _boom
    )
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "could not be fetched" in data["insight"]
    # The result the caller supplied is still honestly summarized.
    assert "ATE +0.1400 [95% CI +0.0500, +0.2300]" in data["insight"]


def test_clinical_narrative_fallback_is_cached_briefly(test_client, monkeypatch):
    # Pin the degraded-TTL discipline (exec-brief/HTE precedent): a fallback is
    # transient — cache 300s, never the full hour.
    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context",
        lambda self,
        brand,
        outcome,
        treatment=None,
        include_causal_evidence=False: _clinical_payload(),
    )

    async def _no_cached(key):
        return None

    seen: dict = {}

    async def _capture(key, value, ttl_seconds=3600):
        seen["ttl"] = ttl_seconds
        seen["is_fallback"] = value.get("is_fallback")

    monkeypatch.setattr("src.api.routes.insights_strategic.cache_get", _no_cached)
    monkeypatch.setattr("src.api.routes.insights_strategic.cache_set", _capture)
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    assert seen == {"ttl": 300, "is_fallback": True}


def test_clinical_narrative_fetch_timeout_degrades(test_client, monkeypatch):
    # The wait_for bound converts an upstream HANG into result-only degradation.
    monkeypatch.setattr(
        "src.api.routes.insights_strategic._CLINICAL_NARRATIVE_FETCH_TIMEOUT_S", 0.05
    )

    def _slow(self, brand, outcome, treatment=None, include_causal_evidence=False):
        time.sleep(1.0)
        return _clinical_payload()

    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context", _slow
    )
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "could not be fetched" in data["insight"]


def test_clinical_narrative_cache_hit_short_circuits_generation(test_client, monkeypatch):
    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context",
        lambda self,
        brand,
        outcome,
        treatment=None,
        include_causal_evidence=False: _clinical_payload(),
    )
    canned = {
        "insight": "CACHED NARRATIVE",
        "key_takeaways": ["cached takeaway"],
        "grounding": [{"label": "Brand", "value": "Remibrutinib"}],
        "is_fallback": False,
    }

    async def _hit(key):
        return canned

    wrote: dict = {}

    async def _capture(key, value, ttl_seconds=3600):
        wrote["called"] = True

    def _must_not_generate(g):
        raise AssertionError("generate_insight must not run on a cache hit")

    monkeypatch.setattr("src.api.routes.insights_strategic.cache_get", _hit)
    monkeypatch.setattr("src.api.routes.insights_strategic.cache_set", _capture)
    monkeypatch.setattr("src.insights.clinical_narrative.generate_insight", _must_not_generate)
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["insight"] == "CACHED NARRATIVE"
    assert data["is_fallback"] is False
    assert data["key_takeaways"] == ["cached takeaway"]
    assert "called" not in wrote


def test_clinical_narrative_real_narrative_cached_for_the_hour(test_client, monkeypatch):
    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context",
        lambda self,
        brand,
        outcome,
        treatment=None,
        include_causal_evidence=False: _clinical_payload(),
    )
    monkeypatch.setattr(
        "src.insights.clinical_narrative.generate_insight",
        lambda g: {
            "insight": "REAL NARRATIVE",
            "key_takeaways": [],
            "grounding": g["grounding"],
            "is_fallback": False,
        },
    )

    async def _no_cached(key):
        return None

    seen: dict = {}

    async def _capture(key, value, ttl_seconds=3600):
        seen["ttl"] = ttl_seconds
        seen["is_fallback"] = value.get("is_fallback")

    monkeypatch.setattr("src.api.routes.insights_strategic.cache_get", _no_cached)
    monkeypatch.setattr("src.api.routes.insights_strategic.cache_set", _capture)
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    assert seen == {"ttl": 3600, "is_fallback": False}
