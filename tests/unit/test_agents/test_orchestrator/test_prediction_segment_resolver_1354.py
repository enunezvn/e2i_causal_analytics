"""prediction_synthesizer dispatch resolver — segment-ranking grounding (#1354).

Before #1354 the resolver, given an hcp_adoption ask that grounded a brand + a
unique champion but named NO specific HCP entity, fail-closed with "a ranking
over segments/entities cannot be answered ... which this route does not do".
#1354 extends it: when the ask is a SEGMENT ranking, it now binds the
segment-aggregation path (segment_by + brand) instead of dead-ending. A genuinely
under-specified single-entity ask (no entity, no segment noun) must STILL fail
closed — that regression is pinned here.
"""

from __future__ import annotations

import pytest

from src.agents.orchestrator.nodes.dispatcher import (
    NeedsStructuredInput,
    _resolve_prediction_synthesizer_input,
)

_CHAMPIONS = [("hcp_adoption_kisqali_goldstd_lr_v1", "hcp_adoption_kisqali")]


@pytest.fixture(autouse=True)
def _patch_champion_probe(monkeypatch):
    monkeypatch.setattr(
        "src.agents.orchestrator.nodes.dispatcher._probe_prediction_champions",
        lambda: list(_CHAMPIONS),
    )


def _dispatch():
    return {"agent_name": "prediction_synthesizer", "parameters": {}}


def test_segment_ranking_ask_binds_segment_path_not_fail_closed():
    agent_input = {
        "query": "which HCP segments are most likely to increase Kisqali prescriptions next quarter"
    }
    out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    assert not isinstance(out, NeedsStructuredInput)
    assert out["segment_by"] == "specialty"
    assert str(out["brand"]).lower() == "kisqali"
    assert out["prediction_target"] == "hcp_adoption_kisqali"
    assert out["time_horizon"] == "90d"  # "next quarter"


def test_region_phrased_segment_ask_binds_region_axis():
    agent_input = {
        "query": "rank the HCP regions most likely to adopt Kisqali",
    }
    out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    assert not isinstance(out, NeedsStructuredInput)
    assert out["segment_by"] == "geographic_region"


def test_underspecified_single_entity_ask_still_fails_closed():
    # No entity id AND no segment noun -> genuinely under-specified single-entity
    # ask: must still fail closed on entity_id (no silent segment fallback).
    agent_input = {"query": "predict Kisqali adoption uptake"}
    out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    assert isinstance(out, NeedsStructuredInput)
    assert "entity_id" in out.missing


def test_segment_ask_without_brand_fails_closed():
    # Segment ranking but NO brand grounded -> fail closed (no silent brand).
    agent_input = {"query": "which HCP segments are most likely to increase prescriptions"}
    out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    assert isinstance(out, NeedsStructuredInput)


def test_non_ranking_by_specialty_ask_does_not_coerce_to_ranking():
    # codex iter-1 HIGH-1: a segment noun WITHOUT ranking intent (an explanation
    # / driver ask) must NOT be coerced into a ranked segment answer — that would
    # return a confident ranking the user never asked for. Still fails closed on
    # entity_id (the pre-#1354 behavior for a non-ranking single-entity ask).
    agent_input = {"query": "explain Kisqali adoption by specialty drivers"}
    out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    assert isinstance(out, NeedsStructuredInput)
    assert "entity_id" in out.missing


def test_explain_increase_by_region_is_not_ranking():
    # codex iter-2 HIGH: "increase" is family vocabulary, NOT a ranking signal.
    # "explain the increase in Kisqali prescriptions by region" matches the
    # family + has a segment noun, but is an EXPLANATION ask — it must NOT be
    # coerced into a ranked segment answer.
    agent_input = {"query": "explain the increase in Kisqali prescriptions by region"}
    out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    assert isinstance(out, NeedsStructuredInput)
    assert "entity_id" in out.missing


def test_predict_increase_by_region_forecast_is_not_ranking():
    # codex iter-3 HIGH: bare "predict"/"forecast" is NOT a comparative ranking
    # signal. "predict the increase in <brand> prescriptions by region" is a
    # per-region forecast ask, not "which regions rank highest" — must fail
    # closed rather than return a confident ranked list.
    agent_input = {"query": "predict the increase in Kisqali prescriptions by region next quarter"}
    out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    assert isinstance(out, NeedsStructuredInput)
    assert "entity_id" in out.missing


def test_which_drivers_explain_ask_is_not_ranking():
    # codex iter-4 HIGH: an explanation ask can carry a comparative token AND a
    # segment noun ("which specialty drivers explain Kisqali adoption"). An
    # explicit explanation/driver/causal marker must veto the ranking bind —
    # returning a confident ranking to an explanation ask violates honesty.
    agent_input = {"query": "which specialty drivers explain Kisqali adoption"}
    out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    assert isinstance(out, NeedsStructuredInput)
    assert "entity_id" in out.missing


@pytest.mark.parametrize(
    "query",
    [
        # codex iter-5 HIGH: attribution/causal phrasings (routine analytics
        # language) carry a comparative + segment noun but ask for EXPLANATION,
        # not a ranking. The negative gate must veto the whole attribution class.
        "which HCP specialties account for Kisqali adoption",
        "which specialties contribute most to Kisqali adoption",
        "what are the determinants of Kisqali adoption by specialty",
        "which specialties are most associated with Kisqali adoption",
        "what factors drive the highest Kisqali adoption by region",
        # codex iter-6: predictor/predictive + linked/related association phrasing
        "which specialties are predictors of Kisqali adoption",
        "which specialties are most predictive of Kisqali adoption",
        "which specialties are linked to Kisqali adoption",
        "which specialties are related to Kisqali adoption",
        # codex iter-7: indicator/signal + the "with" association variant
        "which specialties are indicators of Kisqali adoption",
        "which specialties are the strongest signals of Kisqali adoption",
        "which specialties are linked with Kisqali adoption",
        # codex iter-8: causal-verb "influence" and "factor in/for" attribution.
        # Phrase-constrained — see the positive targeting binds below that MUST
        # still route (high-influence / influential as TARGET attributes).
        "which specialties most influence Kisqali adoption",
        "which specialties influence Kisqali adoption",
        "which specialties are the top factor in Kisqali adoption",
    ],
)
def test_attribution_asks_are_not_ranking(query):
    out = _resolve_prediction_synthesizer_input({"query": query}, _dispatch())
    assert isinstance(out, NeedsStructuredInput)
    assert "entity_id" in out.missing


@pytest.mark.parametrize(
    "query",
    [
        # codex iter-8 guard: the phrase-constrained influence/factor veto must NOT
        # reject legitimate TARGETING/ranking asks that use 'high-influence' /
        # 'influential' as an HCP ATTRIBUTE (influence is literally a model feature).
        "which high-influence specialties are most likely to adopt Kisqali",
        "which influential specialties are most likely to adopt Kisqali",
    ],
)
def test_influence_as_target_attribute_still_binds_ranking(query):
    out = _resolve_prediction_synthesizer_input({"query": query}, _dispatch())
    assert not isinstance(out, NeedsStructuredInput)
    assert out["segment_by"] == "specialty"


def test_predict_which_segments_ask_binds_segment_path():
    # "predict which ..." carries ranking intent even without "most likely".
    agent_input = {"query": "predict which HCP specialties will adopt Fabhalta"}
    # Fabhalta champion for the family:
    import src.agents.orchestrator.nodes.dispatcher as disp

    original = disp._probe_prediction_champions
    disp._probe_prediction_champions = lambda: [
        ("hcp_adoption_fabhalta_goldstd_lr_v1", "hcp_adoption_fabhalta")
    ]
    try:
        out = _resolve_prediction_synthesizer_input(agent_input, _dispatch())
    finally:
        disp._probe_prediction_champions = original
    assert not isinstance(out, NeedsStructuredInput)
    assert out["segment_by"] == "specialty"
