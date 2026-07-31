"""prediction_synthesizer segment-ranking mode (#1354).

``synthesize(..., segment_by=...)`` short-circuits the single-entity ensemble
graph and delegates to the shared hcp_segment_likelihood service, returning a
``PredictionSynthesizerOutput`` whose ``prediction_summary`` carries the honest
ranked-segment narrative. Fails closed (status='failed') when no champion is
promoted — never fabricates a ranking.
"""

from __future__ import annotations

import pytest

from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent
from src.services.hcp_segment_likelihood import (
    ChampionNotPromotedError,
    SegmentLikelihoodResult,
    SegmentScore,
    SegmentScoringError,
)


def _fake_result() -> SegmentLikelihoodResult:
    return SegmentLikelihoodResult(
        brand="Kisqali",
        model_name="hcp_adoption_kisqali_goldstd_lr_v1",
        segment_by="specialty",
        n_scored=5000,
        overall_mean_propensity=0.40,
        holdout_auc=0.7677,
        segments=[
            SegmentScore(
                segment="rheumatology",
                n=257,
                mean_propensity=0.4431,
                std_propensity=0.24,
                se_propensity=0.0152,
                min_propensity=0.05,
                max_propensity=0.9,
                low_confidence=False,
            )
        ],
    )


@pytest.mark.asyncio
async def test_synthesize_segment_mode_returns_ranked_narrative(monkeypatch):
    async def fake_score(brand, *, segment_by, **kw):
        assert brand == "Kisqali"
        assert segment_by == "specialty"
        return _fake_result()

    monkeypatch.setattr("src.services.hcp_segment_likelihood.score_hcp_segments", fake_score)
    agent = PredictionSynthesizerAgent(enable_memory=False, enable_dspy=False, enable_opik=False)
    out = await agent.synthesize(
        entity_id="segment_ranking:Kisqali",
        prediction_target="hcp_adoption_kisqali",
        segment_by="specialty",
        brand="Kisqali",
        time_horizon="90d",
        query="which HCP segments are most likely to increase Kisqali prescriptions",
    )
    assert out.status == "completed"
    assert "rheumatology" in out.prediction_summary
    assert "adoption propensity" in out.prediction_summary.lower()
    assert out.models_succeeded == 1


@pytest.mark.asyncio
async def test_synthesize_segment_mode_fails_closed_without_champion(monkeypatch):
    async def fake_score(brand, *, segment_by, **kw):
        raise ChampionNotPromotedError("no production champion for Kisqali")

    monkeypatch.setattr("src.services.hcp_segment_likelihood.score_hcp_segments", fake_score)
    agent = PredictionSynthesizerAgent(enable_memory=False, enable_dspy=False, enable_opik=False)
    out = await agent.synthesize(
        entity_id="segment_ranking:Kisqali",
        prediction_target="hcp_adoption_kisqali",
        segment_by="specialty",
        brand="Kisqali",
        query="rank Kisqali HCP segments",
    )
    assert out.status == "failed"
    assert out.errors
    # honest: no fabricated ranking in the summary
    assert "champion" in out.prediction_summary.lower()


@pytest.mark.asyncio
async def test_synthesize_segment_mode_fails_closed_on_transport_error(monkeypatch):
    # codex iter-1 MED: a model-server transport failure surfaces via
    # SegmentScoringError and the agent returns a structured status='failed'.
    async def fake_score(brand, *, segment_by, **kw):
        raise SegmentScoringError("model server unreachable")

    monkeypatch.setattr("src.services.hcp_segment_likelihood.score_hcp_segments", fake_score)
    agent = PredictionSynthesizerAgent(enable_memory=False, enable_dspy=False, enable_opik=False)
    out = await agent.synthesize(
        entity_id="segment_ranking:Kisqali",
        prediction_target="hcp_adoption_kisqali",
        segment_by="specialty",
        brand="Kisqali",
        query="which HCP segments are most likely to adopt Kisqali",
    )
    assert out.status == "failed"
    assert out.errors
