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
