"""Unit tests for the HCP segment-likelihood serving service (#1354).

Covers the reusable, importable core: the PURE per-segment aggregation, the
fail-closed champion resolution (never scores a non-promoted model), and the
score-orchestration wiring (real scoring path, no fabricated numbers). The live
end-to-end scoring against the real champions + real feature data is verified
separately in the PR (docker-exec probe); these tests pin the logic that must
not silently regress.
"""

from __future__ import annotations

import math

import pytest

from src.services.hcp_segment_likelihood import (
    DEFAULT_MIN_CONFIDENT_N,
    HCP_SEGMENT_AXES,
    ChampionNotPromotedError,
    SegmentLikelihoodResult,
    SegmentScore,
    aggregate_by_segment,
    build_segment_ranking_narrative,
    resolve_hcp_adoption_champion,
    score_hcp_segments,
)


# ---------------------------------------------------------------------------
# Fakes (plain stand-ins — NOT MagicMock: the code uses getattr(result, "data")
# and dict access; MagicMock would fake-satisfy hasattr/attribute probes).
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """Sync chain builder + async execute() mirroring the supabase-py client."""

    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return self

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def not_(self, *_a, **_k):  # pragma: no cover - unused chain sugar
        return self

    def is_(self, *_a, **_k):  # pragma: no cover
        return self

    def limit(self, *_a, **_k):
        return self

    async def execute(self):
        return _FakeResult(self._rows)


class _FakeClient:
    """Stand-in BentoML client: returns a fixed prob per row, echoing the
    per-row order so aggregation alignment is checked."""

    def __init__(self, prob_for):
        self._prob_for = prob_for
        self.calls = []

    async def predict_batch(self, model_name, batch):
        rows = batch["raw_features"]
        self.calls.append((model_name, len(rows)))
        return {"probabilities": [self._prob_for(r) for r in rows]}


# ---------------------------------------------------------------------------
# aggregate_by_segment — PURE
# ---------------------------------------------------------------------------
def test_aggregate_ranks_segments_desc_by_mean_with_correct_n():
    rows = [
        {"specialty": "oncology"},
        {"specialty": "oncology"},
        {"specialty": "hematology"},
        {"specialty": "rheumatology"},
    ]
    probs = [0.4, 0.6, 0.2, 0.9]  # onc mean .5, heme .2, rheum .9
    out = aggregate_by_segment(rows, probs, "specialty", min_confident_n=1)
    assert [s.segment for s in out] == ["rheumatology", "oncology", "hematology"]
    onc = next(s for s in out if s.segment == "oncology")
    assert onc.n == 2
    assert onc.mean_propensity == pytest.approx(0.5)
    assert onc.min_propensity == pytest.approx(0.4)
    assert onc.max_propensity == pytest.approx(0.6)
    # SE of the mean over {0.4, 0.6}: pstd = 0.1, se = 0.1/sqrt(2)
    assert onc.se_propensity == pytest.approx(0.1 / math.sqrt(2), rel=1e-6)


def test_aggregate_marks_thin_cells_low_confidence_but_never_drops_them():
    rows = [{"specialty": "neurology"}] + [{"specialty": "oncology"}] * 40
    probs = [0.9] + [0.3] * 40
    out = aggregate_by_segment(rows, probs, "specialty", min_confident_n=30)
    neuro = next(s for s in out if s.segment == "neurology")
    onc = next(s for s in out if s.segment == "oncology")
    assert neuro.n == 1 and neuro.low_confidence is True
    assert onc.n == 40 and onc.low_confidence is False
    # thin cell is still present and still ranked (honest, just flagged)
    assert out[0].segment == "neurology"


def test_aggregate_groups_none_segment_value_under_unknown_not_crash():
    rows = [{"specialty": None}, {"specialty": "oncology"}]
    probs = [0.5, 0.5]
    out = aggregate_by_segment(rows, probs, "specialty", min_confident_n=1)
    assert {s.segment for s in out} == {"unknown", "oncology"}


def test_aggregate_rejects_unknown_axis():
    with pytest.raises(ValueError):
        aggregate_by_segment([{"specialty": "x"}], [0.5], "priority_tier")


def test_aggregate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        aggregate_by_segment([{"specialty": "x"}], [0.1, 0.2], "specialty")


# ---------------------------------------------------------------------------
# resolve_hcp_adoption_champion — FAIL CLOSED
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolver_returns_model_and_auc_for_promoted_champion():
    db = _FakeQuery(
        [
            {
                "model_name": "hcp_adoption_kisqali_goldstd_lr_v1",
                "stage": "production",
                "is_champion": True,
                "artifact_path": "/models/x.pkl",
                "is_synthetic": False,
                "auc": 0.7677,
            }
        ]
    )
    model_name, auc = await resolve_hcp_adoption_champion("Kisqali", db=db)
    assert model_name == "hcp_adoption_kisqali_goldstd_lr_v1"
    assert auc == pytest.approx(0.7677)


@pytest.mark.asyncio
async def test_resolver_fails_closed_when_no_champion_row():
    db = _FakeQuery([])
    with pytest.raises(ChampionNotPromotedError):
        await resolve_hcp_adoption_champion("Kisqali", db=db)


@pytest.mark.asyncio
async def test_resolver_fails_closed_when_only_staging_not_champion():
    # A staging (is_champion False) row must NOT be served.
    db = _FakeQuery(
        [
            {
                "model_name": "hcp_adoption_kisqali_goldstd_lr_v1",
                "stage": "staging",
                "is_champion": False,
                "artifact_path": "/models/x.pkl",
                "is_synthetic": False,
                "auc": 0.7677,
            }
        ]
    )
    with pytest.raises(ChampionNotPromotedError):
        await resolve_hcp_adoption_champion("Kisqali", db=db)


@pytest.mark.asyncio
async def test_resolver_rejects_unknown_brand():
    with pytest.raises(ValueError):
        await resolve_hcp_adoption_champion("NotABrand", db=_FakeQuery([]))


# ---------------------------------------------------------------------------
# score_hcp_segments — orchestration wiring (fakes patched in)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_score_hcp_segments_end_to_end_wiring(monkeypatch):
    import src.services.hcp_segment_likelihood as svc

    async def fake_resolve(brand, *, db):
        return "hcp_adoption_kisqali_goldstd_lr_v1", 0.77

    async def fake_load(spec, splits, db):
        import pandas as pd

        return pd.DataFrame(
            {
                "hcp_id": ["h1", "h2", "h3"],
                "peer_influence_score": [0.9, 0.1, 0.5],
                "influence_network_size": [10, 2, 5],
                "years_experience": [20, 3, 12],
                "specialty": ["rheumatology", "oncology", "oncology"],
                "geographic_region": ["west", "south", "south"],
            }
        )

    # higher peer_influence -> higher prob (monotone, like the real model)
    client = _FakeClient(prob_for=lambda r: float(r["peer_influence_score"]))
    monkeypatch.setattr(svc, "resolve_hcp_adoption_champion", fake_resolve)
    monkeypatch.setattr(svc, "_load_scoring_frame", fake_load)

    result = await score_hcp_segments(
        "Kisqali", segment_by="specialty", db=object(), model_client=client, min_confident_n=1
    )
    assert result.brand == "Kisqali"
    assert result.model_name == "hcp_adoption_kisqali_goldstd_lr_v1"
    assert result.n_scored == 3
    assert result.holdout_auc == pytest.approx(0.77)
    # rheumatology (0.9) ranks above oncology (mean of 0.1, 0.5 = 0.3)
    assert result.segments[0].segment == "rheumatology"
    assert result.segments[0].mean_propensity == pytest.approx(0.9)
    onc = next(s for s in result.segments if s.segment == "oncology")
    assert onc.n == 2 and onc.mean_propensity == pytest.approx(0.3)
    # scored via the batch client, not fabricated
    assert client.calls and client.calls[0][0] == "hcp_adoption_kisqali_goldstd_lr_v1"


@pytest.mark.asyncio
async def test_score_wraps_transport_error_as_segment_scoring_error(monkeypatch):
    # codex iter-1 MED: a model-server transport/circuit-breaker failure must
    # surface via the typed SegmentScoringError contract, not a raw exception.
    import src.services.hcp_segment_likelihood as svc

    async def fake_resolve(brand, *, db):
        return "hcp_adoption_kisqali_goldstd_lr_v1", 0.77

    async def fake_load(spec, splits, db):
        import pandas as pd

        return pd.DataFrame(
            {
                "hcp_id": ["h1"],
                "peer_influence_score": [0.5],
                "influence_network_size": [5],
                "years_experience": [10],
                "specialty": ["oncology"],
                "geographic_region": ["west"],
            }
        )

    class _BoomClient:
        async def predict_batch(self, *a, **k):
            raise RuntimeError("Circuit breaker open for model")

    monkeypatch.setattr(svc, "resolve_hcp_adoption_champion", fake_resolve)
    monkeypatch.setattr(svc, "_load_scoring_frame", fake_load)
    with pytest.raises(svc.SegmentScoringError):
        await svc.score_hcp_segments("Kisqali", db=object(), model_client=_BoomClient())


@pytest.mark.asyncio
async def test_score_does_not_mask_programming_error_as_scoring_error(monkeypatch):
    # codex iter-2 MED: only transport failures wrap as SegmentScoringError.
    # A programming defect (TypeError) must PROPAGATE, not be laundered into a
    # fail-closed scoring error (debuggability).
    import src.services.hcp_segment_likelihood as svc

    async def fake_resolve(brand, *, db):
        return "hcp_adoption_kisqali_goldstd_lr_v1", 0.77

    async def fake_load(spec, splits, db):
        import pandas as pd

        return pd.DataFrame(
            {
                "hcp_id": ["h1"],
                "peer_influence_score": [0.5],
                "influence_network_size": [5],
                "years_experience": [10],
                "specialty": ["oncology"],
                "geographic_region": ["west"],
            }
        )

    class _BuggyClient:
        async def predict_batch(self, *a, **k):
            raise TypeError("developer bug, not a transport failure")

    monkeypatch.setattr(svc, "resolve_hcp_adoption_champion", fake_resolve)
    monkeypatch.setattr(svc, "_load_scoring_frame", fake_load)
    with pytest.raises(TypeError):
        await svc.score_hcp_segments("Kisqali", db=object(), model_client=_BuggyClient())


@pytest.mark.asyncio
async def test_score_wraps_malformed_json_response_as_segment_scoring_error(monkeypatch):
    # codex iter-3 MED: a malformed 2xx body makes the client's response.json()
    # raise json.JSONDecodeError (a ValueError subclass) — a transport-class
    # failure that must surface via the typed SegmentScoringError, not a raw
    # ValueError conflated with input validation.
    import json

    import src.services.hcp_segment_likelihood as svc

    async def fake_resolve(brand, *, db):
        return "hcp_adoption_kisqali_goldstd_lr_v1", 0.77

    async def fake_load(spec, splits, db):
        import pandas as pd

        return pd.DataFrame(
            {
                "hcp_id": ["h1"],
                "peer_influence_score": [0.5],
                "influence_network_size": [5],
                "years_experience": [10],
                "specialty": ["oncology"],
                "geographic_region": ["west"],
            }
        )

    class _GarbledClient:
        async def predict_batch(self, *a, **k):
            raise json.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr(svc, "resolve_hcp_adoption_champion", fake_resolve)
    monkeypatch.setattr(svc, "_load_scoring_frame", fake_load)
    with pytest.raises(svc.SegmentScoringError):
        await svc.score_hcp_segments("Kisqali", db=object(), model_client=_GarbledClient())


@pytest.mark.asyncio
async def test_score_hcp_segments_propagates_fail_closed_champion(monkeypatch):
    import src.services.hcp_segment_likelihood as svc

    async def fake_resolve(brand, *, db):
        raise ChampionNotPromotedError("no champion")

    monkeypatch.setattr(svc, "resolve_hcp_adoption_champion", fake_resolve)
    with pytest.raises(ChampionNotPromotedError):
        await score_hcp_segments("Kisqali", db=object(), model_client=_FakeClient(lambda r: 0.5))


def test_narrative_is_honest_about_propensity_and_thin_cells():
    result = SegmentLikelihoodResult(
        brand="Kisqali",
        model_name="hcp_adoption_kisqali_goldstd_lr_v1",
        segment_by="specialty",
        n_scored=5000,
        overall_mean_propensity=0.40,
        holdout_auc=0.7677,
        segments=[
            SegmentScore(
                segment="neurology",
                n=11,
                mean_propensity=0.49,
                std_propensity=0.2,
                se_propensity=0.06,
                min_propensity=0.1,
                max_propensity=0.9,
                low_confidence=True,
            ),
            SegmentScore(
                segment="oncology",
                n=1662,
                mean_propensity=0.40,
                std_propensity=0.2,
                se_propensity=0.005,
                min_propensity=0.04,
                max_propensity=0.92,
                low_confidence=False,
            ),
        ],
    )
    text = build_segment_ranking_narrative(result, top_n=5, horizon="next quarter")
    assert "adoption propensity" in text.lower()
    assert "0.768" in text  # out-of-sample AUC surfaced
    assert "low confidence" in text.lower()  # thin neurology cell flagged
    assert "next quarter" in text  # horizon echoed
    assert "not a horizon-specific" in text  # honesty caveat


def test_segment_axes_are_covariate_backed():
    # The public axis list must only contain covariates actually served to the
    # model (grouping keys present in every scored row) — guards against adding
    # an axis that isn't in the feature payload.
    assert HCP_SEGMENT_AXES == ("specialty", "geographic_region")
    assert isinstance(DEFAULT_MIN_CONFIDENT_N, int)
    assert SegmentScore.__name__ == "SegmentScore"
