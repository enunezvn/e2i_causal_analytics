"""AG-UI chat tool: per-HCP-segment likelihood-to-prescribe (#1354, demo Q3.3).

The tool wraps the shared hcp_segment_likelihood service so the real UI brain
(chat_node + tools) can answer "which HCP segments are most likely to increase
<brand> prescriptions" with real per-segment propensities — instead of
substituting a regional TRx proxy. Fails closed (success=False) with an honest
error when no champion is promoted; never fabricates a ranking.
"""

from __future__ import annotations

import pytest

from src.api.routes.chatbot_tools import (
    E2I_CHATBOT_TOOLS,
    E2I_TOOL_MAP,
    predict_hcp_segment_likelihood_tool,
)
from src.services.hcp_segment_likelihood import (
    ChampionNotPromotedError,
    SegmentLikelihoodResult,
    SegmentScore,
)


def _fake_result() -> SegmentLikelihoodResult:
    return SegmentLikelihoodResult(
        brand="Kisqali",
        model_name="hcp_adoption_kisqali_goldstd_lr_v1",
        segment_by="specialty",
        n_scored=5000,
        overall_mean_propensity=0.399,
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
            ),
            SegmentScore(
                segment="hematology",
                n=1016,
                mean_propensity=0.3713,
                std_propensity=0.23,
                se_propensity=0.0074,
                min_propensity=0.04,
                max_propensity=0.9,
                low_confidence=False,
            ),
        ],
    )


def test_tool_is_registered():
    assert predict_hcp_segment_likelihood_tool in E2I_CHATBOT_TOOLS
    assert E2I_TOOL_MAP["predict_hcp_segment_likelihood_tool"] is (
        predict_hcp_segment_likelihood_tool
    )


@pytest.mark.asyncio
async def test_tool_returns_ranked_segments(monkeypatch):
    async def fake_score(brand, *, segment_by, **kw):
        assert brand == "Kisqali"
        return _fake_result()

    monkeypatch.setattr("src.services.hcp_segment_likelihood.score_hcp_segments", fake_score)
    out = await predict_hcp_segment_likelihood_tool.ainvoke(
        {"brand": "Kisqali", "segment_by": "specialty", "time_horizon": "next quarter"}
    )
    assert out["success"] is True
    assert out["model_name"] == "hcp_adoption_kisqali_goldstd_lr_v1"
    assert out["holdout_auc"] == pytest.approx(0.7677)
    assert out["segments"][0]["segment"] == "rheumatology"
    assert out["segments"][0]["n"] == 257
    assert "narrative" in out and "rheumatology" in out["narrative"]
    assert out["segment_by"] == "specialty"


@pytest.mark.asyncio
async def test_tool_fails_closed_without_brand():
    out = await predict_hcp_segment_likelihood_tool.ainvoke({"brand": ""})
    assert out["success"] is False
    assert "brand" in out["error"].lower()


@pytest.mark.asyncio
async def test_tool_fails_closed_when_no_champion(monkeypatch):
    async def fake_score(brand, *, segment_by, **kw):
        raise ChampionNotPromotedError("no production champion for Kisqali")

    monkeypatch.setattr("src.services.hcp_segment_likelihood.score_hcp_segments", fake_score)
    out = await predict_hcp_segment_likelihood_tool.ainvoke({"brand": "Kisqali"})
    assert out["success"] is False
    assert "champion" in out["error"].lower()
    # honest: no fabricated segments
    assert out.get("segments", []) == []
