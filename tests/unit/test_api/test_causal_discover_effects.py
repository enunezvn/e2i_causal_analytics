"""Unit coverage for the discover-effects leaderboard ranking logic.

The leaderboard ranks the agent's VALIDATED effects by confidence (robustness
gate + significance) then impact (|ate|). These pure helpers are CI-safe (no DB,
no agent run); the end-to-end agent runs are covered by a faithful check.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.api.routes import causal as causal_routes
from src.api.routes.causal import (
    _effect_confidence_score,
    _effect_from_agent_response,
    _effect_status_from_gate,
    _rank_effects,
)
from src.api.schemas.causal import (
    AgentCausalAnalysisResponse,
    CausalDAGModel,
    DiscoveredEffect,
    RefutationSummary,
)


@pytest.mark.asyncio
async def test_candidate_questions_come_from_ssot_with_modeled_adjustment_set():
    fake_questions = [
        {
            "treatment": "treatment_arm",
            "outcome": "persistent_180d",
            "brand": "Kisqali",
            # geographic_region is non-numeric (dropped); treatment_arm is a
            # treatment-collision (dropped by the `c not in (t, o)` guard, I2).
            "confounders": [
                "disease_severity",
                "academic_hcp",
                "geographic_region",
                "treatment_arm",
            ],
        },
        {
            "treatment": "treatment_arm",
            "outcome": "treatment_initiated",
            "brand": "Fabhalta",
            "confounders": ["disease_severity", "age_at_diagnosis"],
        },
    ]
    # _get_causal_path_repo is async (returns the repo), so patch it as an
    # AsyncMock whose awaited result is the repo carrying get_distinct_questions.
    with patch.object(causal_routes, "_get_causal_path_repo", new_callable=AsyncMock) as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=fake_questions)
        qs = await causal_routes._discover_candidate_questions("patient_journeys", brand=None)
    by_outcome = {q.outcome: q for q in qs}
    # retention: geographic_region dropped (non-numeric), numeric confounders kept
    assert by_outcome["persistent_180d"].adjustment_set == ["disease_severity", "academic_hcp"]
    # I2: a treatment-collision confounder is never adjusted on (would be invalid).
    assert "treatment_arm" not in by_outcome["persistent_180d"].adjustment_set
    assert by_outcome["persistent_180d"].brand == "Kisqali"
    assert by_outcome["treatment_initiated"].adjustment_set == [
        "disease_severity",
        "age_at_diagnosis",
    ]


@pytest.mark.asyncio
async def test_brand_filter_subsets_questions():
    # Brand subsetting is the repo's job (get_distinct_questions applies a
    # {"brand": brand} DB filter), so the mock honors the brand kwarg the way the
    # real repo does. Both rows use a NON-complement outcome so the ONLY thing
    # that can drop the Fabhalta row is the brand filter being honored+forwarded
    # (a complement outcome would be dropped by _COMPLEMENT_OUTCOMES_SKIP even if
    # brand forwarding were deleted, which would mask a regression).
    all_rows = [
        {
            "treatment": "treatment_arm",
            "outcome": "persistent_180d",
            "brand": "Kisqali",
            "confounders": ["disease_severity"],
        },
        {
            "treatment": "treatment_arm",
            "outcome": "treatment_initiated",
            "brand": "Fabhalta",
            "confounders": ["disease_severity"],
        },
    ]

    async def _distinct(*, brand=None, include_synthetic=True):
        return [r for r in all_rows if brand is None or r["brand"] == brand]

    with patch.object(causal_routes, "_get_causal_path_repo", new_callable=AsyncMock) as mk:
        mk.return_value.get_distinct_questions = AsyncMock(side_effect=_distinct)
        qs = await causal_routes._discover_candidate_questions("patient_journeys", brand="Kisqali")
    assert [q.brand for q in qs] == ["Kisqali"]
    # The brand filter must be forwarded to the SSOT read (not applied only FE-side):
    # if it were not forwarded, the Fabhalta row would survive and this list would
    # contain two brands.
    mk.return_value.get_distinct_questions.assert_awaited_once_with(
        brand="Kisqali", include_synthetic=True
    )


@pytest.mark.asyncio
async def test_discover_candidate_questions_awaits_async_client(monkeypatch):
    """FAITHFUL wiring check: does NOT mock _get_causal_path_repo, so a regression
    to a SYNC client (object APIResponse can't be awaited) would surface here.
    Patches only the async-client factory + the repo's get_many.

    NOTE: _get_causal_path_repo imports get_async_supabase_client locally (lazy),
    so we patch it at its SOURCE module, not as a causal_routes attribute."""
    import src.memory.services.factories as factories
    from src.repositories.causal_path import CausalPathRepository

    monkeypatch.setattr(factories, "get_async_supabase_client", AsyncMock(return_value=object()))
    fake_rows = [
        {
            "start_node": "treatment_arm",
            "end_node": "persistent_180d",
            "brand": "Kisqali",
            "confounders_controlled": ["disease_severity"],
        }
    ]
    get_many_mock = AsyncMock(return_value=fake_rows)
    monkeypatch.setattr(CausalPathRepository, "get_many", get_many_mock)

    # Must complete WITHOUT raising (proves the async/await chain is wired).
    qs = await causal_routes._discover_candidate_questions("patient_journeys", brand="Kisqali")

    get_many_mock.assert_awaited()
    assert [q.outcome for q in qs] == ["persistent_180d"]
    assert qs[0].adjustment_set == ["disease_severity"]


@pytest.mark.unit
def test_status_from_gate_separates_blocked_from_failed():
    # An estimate that the gate BLOCKED is 'blocked' (computed, inspectable) — not 'failed'.
    assert _effect_status_from_gate(-0.006, "block", "completed") == "blocked"
    assert _effect_status_from_gate(0.18, "proceed", "completed") == "completed"
    assert _effect_status_from_gate(0.04, "review", "completed") == "needs_review"
    # A run that produced NO estimate is 'failed' (or keeps an in-flight status).
    assert _effect_status_from_gate(None, None, "failed") == "failed"
    assert _effect_status_from_gate(None, None, "running") == "running"


@pytest.mark.unit
def test_confidence_score_orders_gate_then_significance():
    proceed_sig = _effect_confidence_score("proceed", True)
    proceed = _effect_confidence_score("proceed", False)
    review = _effect_confidence_score("review", False)
    block = _effect_confidence_score("block", False)
    assert proceed_sig > proceed > review > block
    assert 0.0 <= block and proceed_sig <= 1.0


@pytest.mark.unit
def test_rank_effects_by_confidence_then_impact():
    a = DiscoveredEffect(
        treatment="t", outcome="a", status="completed", confidence_score=0.9, impact=0.05
    )
    b = DiscoveredEffect(
        treatment="t", outcome="b", status="completed", confidence_score=0.9, impact=0.20
    )
    c = DiscoveredEffect(
        treatment="t", outcome="c", status="completed", confidence_score=0.35, impact=0.99
    )
    pending = DiscoveredEffect(treatment="t", outcome="d", status="pending", confidence_score=0.0)
    ranked = _rank_effects([a, c, pending, b])
    # Same confidence -> higher impact first; lower confidence after; pending last.
    assert [e.outcome for e in ranked] == ["b", "a", "c", "d"]


@pytest.mark.unit
def test_effect_from_agent_response_maps_gate_and_impact():
    resp = AgentCausalAnalysisResponse(
        analysis_id="x1",
        status="completed",
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
        n_rows=1500,
        data_source="synthetic",
        dag=CausalDAGModel(),
        ate=-0.0875,
        statistical_significance=True,
        selected_estimator="LinearDML",
        refutation=RefutationSummary(gate_decision="proceed", passed=True),
        latency_ms=4000,
    )
    eff = _effect_from_agent_response("treatment_arm", "persistent_180d", resp, "x1")
    assert eff.gate_decision == "proceed"
    assert eff.impact == pytest.approx(0.0875)  # |ate|
    assert eff.confidence_score == pytest.approx(0.9)
    assert eff.analysis_id == "x1"
    assert eff.status == "completed"
    # Plain-language one-liner so the leaderboard reads as more than numbers:
    # direction (negative ATE -> "lowers"), robustness verdict, significance.
    assert eff.summary
    assert "lowers" in eff.summary
    assert "survived all robustness checks" in eff.summary
    assert "not statistically significant" not in eff.summary


@pytest.mark.asyncio
async def test_questions_are_fwl_preranked(monkeypatch):
    qs = [
        causal_routes._CandidateQuestion(
            "treatment_arm", "treatment_initiated", "Kisqali", ["disease_severity"]
        ),
        causal_routes._CandidateQuestion(
            "treatment_arm", "persistent_180d", "Kisqali", ["disease_severity"]
        ),
    ]
    strengths = {"treatment_initiated": 0.05, "persistent_180d": 0.60}

    async def fake_signal(dataset, q):
        return strengths[q.outcome]

    monkeypatch.setattr(causal_routes, "_prerank_signal", fake_signal)
    ordered = await causal_routes._prerank_questions("patient_journeys", qs)
    assert [q.outcome for q in ordered] == ["persistent_180d", "treatment_initiated"]


@pytest.mark.unit
def test_effect_summary_none_until_estimated():
    """A pending/failed effect (no ATE) has no summary — never a fabricated one."""
    resp = AgentCausalAnalysisResponse(
        analysis_id="x2",
        status="failed",
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
        n_rows=1500,
        data_source="synthetic",
        dag=CausalDAGModel(),
        ate=None,
        statistical_significance=False,
        refutation=RefutationSummary(),
        latency_ms=10,
    )
    eff = _effect_from_agent_response("treatment_arm", "persistent_180d", resp, "x2")
    assert eff.summary is None


@pytest.mark.asyncio
async def test_trigger_questions_from_ssot_have_empty_adjustment_set():
    """The nba_triggers grain enumerates the RCT + effect-modifier questions from
    the SSOT, each with an EMPTY modeled adjustment set (randomized / effect
    modifier). Reuses P1's _discover_candidate_questions verbatim."""
    from unittest.mock import AsyncMock, patch

    fake = [
        {
            "treatment": "control_group_flag",
            "outcome": "action_taken",
            "brand": "Kisqali",
            "confounders": [],
        },
        {
            "treatment": "acceptance_status",
            "outcome": "conversion_flag",
            "brand": "Kisqali",
            "confounders": [],
        },
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=fake)
        qs = await causal_routes._discover_candidate_questions("nba_triggers", brand="Kisqali")

    by_outcome = {q.outcome: q for q in qs}
    assert by_outcome["action_taken"].treatment == "control_group_flag"
    assert by_outcome["action_taken"].adjustment_set == []
    assert by_outcome["action_taken"].brand == "Kisqali"
    assert by_outcome["conversion_flag"].treatment == "acceptance_status"
    assert by_outcome["conversion_flag"].adjustment_set == []
