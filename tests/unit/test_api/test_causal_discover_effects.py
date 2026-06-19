"""Unit coverage for the discover-effects leaderboard ranking logic.

The leaderboard ranks the agent's VALIDATED effects by confidence (robustness
gate + significance) then impact (|ate|). These pure helpers are CI-safe (no DB,
no agent run); the end-to-end agent runs are covered by a faithful check.
"""

import pytest

from src.api.routes.causal import (
    _CAUSAL_DATASET_SPECS,
    _discover_candidate_pairs,
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


@pytest.mark.unit
def test_candidate_pairs_dedupe_complement_and_self():
    pairs = _discover_candidate_pairs(_CAUSAL_DATASET_SPECS["patient_journeys"])
    # No self-pairs, and the complement outcome (discontinued_180d) is dropped.
    assert all(t != o for t, o in pairs)
    assert all(o != "discontinued_180d" for _, o in pairs)
    assert ("treatment_arm", "persistent_180d") in pairs


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
