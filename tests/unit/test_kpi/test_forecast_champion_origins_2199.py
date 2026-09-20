"""The champion must be chosen on a comparison the models actually share (#2199).

Each model is backtested with ``min_train = model.min_observations`` (seasonal 24,
trend/TimesFM 8), so on a short series their MAPEs are averages over DIFFERENT sets
of origins -- not merely estimates of different precision. Measured on the real PROD
series truncated to short lengths, through the real service path:

    L= 36  seasonal 7 origins [24..30]   trend 23 origins [ 8..30]
    L= 42  seasonal 13 origins [24..36]  trend 24 origins [13..36]
    L= 48  seasonal 19 origins [24..42]  trend 24 origins [19..42]
    L>=54  all models 24 origins, identical cutoffs -- no asymmetry
    L=164  (the live length) all models 24 origins [135..158]

9 of 21 served (brand, length) cases compared unequal origin sets, and on
Remibrutinib at L=42 the champion genuinely flips: today ``seasonal_mul`` wins at
8.33% while ``trend`` shows 8.63% over 24 origins, but restricted to the 13 origins
they share ``trend`` is 8.17% and should have won.

A second source needs no short series at all: a model whose origins FAIL is scored
on the subset that succeeded, which again is not the set its rivals were scored on.

The fix records WHICH cutoffs each model was actually scored on and selects on the
intersection, while each model's reported MAPE stays its own full-evidence number.
"""

from __future__ import annotations

import pytest

from src.kpi.forecast import backtest as bt

#: The service-level tests here fit real Holt-Winters models over rolling origins.
#: 300s, not more: tests/unit/test_tests_meta/test_session_stall_watchdog_1655.py
#: requires a lane's stall timeout to be at least 2x the longest per-test budget.
pytestmark = pytest.mark.timeout(300)


def _score(name, mape, cutoffs, *, per_origin=None, horizon=2):
    """A score whose per-step errors are keyed to explicit cutoffs.

    ``per_origin`` maps cutoff -> the absolute % error at EVERY step for that origin,
    which is what lets a test plant "this model was graded on a calm stretch".
    """
    cutoffs = tuple(cutoffs)
    if per_origin is None:
        per_origin = dict.fromkeys(cutoffs, mape)
    steps = tuple(tuple(per_origin[c] for c in cutoffs) for _ in range(horizon))
    flat = [v for step in steps for v in step]
    return bt.BacktestScore(
        name=name,
        monthly_mape=sum(flat) / len(flat),
        horizon_total_error=sum(flat) / len(flat),
        n_origins=len(cutoffs),
        n_failed_origins=0,
        horizon=horizon,
        step_pct_errors=steps,
        signed_step_pct_errors=steps,
        origin_cutoffs=cutoffs,
    )


# ------------------------------------------------------- the score records its origins
def test_score_model_records_which_cutoffs_it_was_actually_graded_on():
    """Without cutoff identity, ``step_pct_errors[i][k]``'s k means a different origin
    for each model and no intersection is computable."""
    y = [10.0 + i for i in range(20)]
    score = bt.score_model(
        predict=lambda ctx, h: [float(ctx[-1])] * h,
        y=y,
        horizon=2,
        origins=4,
        name="flat",
        min_train=8,
    )
    assert score.origin_cutoffs == (15, 16, 17, 18)
    assert len(score.origin_cutoffs) == score.n_origins
    for step in score.step_pct_errors:
        assert len(step) == len(score.origin_cutoffs)


def test_a_failed_origin_is_dropped_from_the_cutoffs_not_just_counted():
    """The surviving errors compact positionally; the cutoffs must compact WITH them,
    or every error after the failure is attributed to the wrong origin."""
    y = [10.0 + i for i in range(20)]

    def flaky(ctx, h):
        # origin at cutoff 16 fails
        return None if len(ctx) == 16 else [float(ctx[-1])] * h

    score = bt.score_model(predict=flaky, y=y, horizon=2, origins=4, name="flaky", min_train=8)
    assert score.n_failed_origins == 1
    assert score.origin_cutoffs == (15, 17, 18)
    assert len(score.step_pct_errors[0]) == 3


# ------------------------------------------------------------------ the shared ground
def test_common_origins_is_the_intersection():
    a = _score("a", 5.0, [24, 25, 26, 27])
    b = _score("b", 5.0, [20, 21, 22, 23, 24, 25, 26, 27])
    assert bt.common_origins([a, b]) == (24, 25, 26, 27)


def test_common_origins_survives_a_gap_from_a_failed_origin():
    """Nested suffixes are the no-failure case; a failure breaks the nesting and the
    intersection has to be a real set operation, not 'the shortest range'."""
    a = _score("a", 5.0, [24, 25, 26, 27])
    b = _score("b", 5.0, [20, 21, 22, 23, 24, 26, 27])  # 25 failed
    assert bt.common_origins([a, b]) == (24, 26, 27)


# ------------------------------------------------------------------- the selection fix
def test_a_model_graded_on_an_easier_early_stretch_does_not_win_on_that_advantage():
    """This is the defect, planted.

    ``wide`` is graded on 8 origins, 4 of which are a calm early stretch where it was
    near-perfect. Its full MAPE therefore beats ``narrow``'s. On the 4 origins they
    SHARE, ``wide`` is plainly worse, so ``narrow`` must win.
    """
    shared = [24, 25, 26, 27]
    narrow = _score("narrow", None, shared, per_origin={24: 6.0, 25: 6.0, 26: 6.0, 27: 6.0})
    wide = _score(
        "wide",
        None,
        [20, 21, 22, 23, 24, 25, 26, 27],
        per_origin={20: 0.1, 21: 0.1, 22: 0.1, 23: 0.1, 24: 9.0, 25: 9.0, 26: 9.0, 27: 9.0},
    )
    # The premise of the test: today's rule really would pick the wrong one.
    assert wide.monthly_mape < narrow.monthly_mape

    assert bt.select_champion([narrow, wide]).name == "narrow"


def test_the_champion_is_still_the_best_when_every_model_shares_every_origin():
    """The live 164-month path: all models on identical cutoffs, so the fix is a no-op.

    Measured on the real series, all three brands: all models 24 origins on [135..158].
    """
    cutoffs = list(range(135, 159))
    a = _score("a", 7.5, cutoffs)
    b = _score("b", 6.2, cutoffs)
    c = _score("c", 9.0, cutoffs)
    assert bt.common_origins([a, b, c]) == tuple(cutoffs)
    assert bt.select_champion([a, b, c]).name == "b"


def test_the_reported_mape_stays_each_model_s_own_full_evidence_number():
    """Selection moves to shared ground; REPORTING must not. A model graded on 24
    origins has more evidence behind its number than one graded on 7, and the payload
    is what tells the reader so."""
    narrow = _score("narrow", 6.0, [24, 25, 26, 27])
    wide = _score("wide", 5.0, [20, 21, 22, 23, 24, 25, 26, 27])
    bt.select_champion([narrow, wide])
    assert narrow.n_origins == 4
    assert wide.n_origins == 8
    assert wide.monthly_mape == pytest.approx(5.0)


def test_selection_falls_back_to_full_mape_when_the_shared_ground_is_too_thin():
    """Two models overlapping on one origin cannot be honestly compared there.

    Choosing on a single origin would be worse than today's rule, not better, so the
    fix must not trade one wrong comparison for a noisier one.
    """
    a = _score("a", 6.0, [27])
    b = _score("b", 5.0, [20, 21, 22, 23, 24, 25, 26, 27])
    assert len(bt.common_origins([a, b])) < bt.MIN_COMMON_ORIGINS
    assert bt.select_champion([a, b]).name == "b"  # b's full MAPE is lower


def test_disjoint_origin_sets_fall_back_rather_than_raise():
    a = _score("a", 6.0, [1, 2, 3, 4])
    b = _score("b", 5.0, [20, 21, 22, 23])
    assert bt.common_origins([a, b]) == ()
    assert bt.select_champion([a, b]).name == "b"


def test_ties_on_the_shared_ground_still_break_on_name():
    shared = [24, 25, 26, 27]
    z = _score("zebra", 6.0, shared)
    a = _score("alpha", 6.0, shared)
    assert bt.select_champion([z, a]).name == "alpha"
    assert bt.select_champion([a, z]).name == "alpha"


def test_no_scores_still_raises_no_champion():
    with pytest.raises(bt.NoChampion):
        bt.select_champion([])


# ------------------------------------------------------- the service-level contract
def _series(values, brand="Probe"):
    from datetime import date

    from src.kpi.canonical_volume_series import (
        CanonicalVolumeSeries,
        MonthlyVolumePoint,
        month_end,
    )

    months = []
    y, m = 2013, 1
    for _ in values:
        months.append(date(y, m, 1))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    pts = tuple(MonthlyVolumePoint(mo, v, 4) for mo, v in zip(months, values, strict=True))
    return CanonicalVolumeSeries(
        metric="trx",
        brand=brand,
        region=None,
        points=pts,
        data_through=month_end(pts[-1].month),
        as_of=date(2026, 9, 20),
        dropped_in_progress=(),
        dropped_incomplete=(),
        query_id="test_2199",
    )


def _seasonal(n, *, level=100_000.0, growth=400.0, amp=0.18):
    import math as _m

    return [
        level + growth * i + level * amp * _m.sin(2 * _m.pi * (i % 12) / 12.0) for i in range(n)
    ]


def test_the_payload_reports_the_shared_ground_the_champion_was_chosen_on():
    from src.kpi.forecast import service as svc

    result = svc.forecast_series(
        _series(_seasonal(48)), horizon=6, include_timesfm=False, cache=None
    )
    payload = result.to_payload()
    counts = {s["model"]: s["origins_scored"] for s in payload["backtest"]["models"]}
    assert len(set(counts.values())) > 1, f"need an unequal case to be meaningful: {counts}"

    shared = payload["backtest"]["selection_origins"]
    assert shared == min(counts.values())
    assert shared < max(counts.values())
    # The rule text must state the ground, not merely claim "the same origins".
    assert str(shared) in payload["backtest"]["selection_rule"]
    assert "EVERY model was graded on" in payload["backtest"]["selection_rule"]


def test_the_champion_is_the_first_row_even_when_its_own_mape_is_the_worst():
    """Measured on the real series at Remibrutinib/42 months: the champion is 3rd of 3
    by its OWN MAPE, because the shared ground disagrees with the full scores.

    Ordering the payload by each model's own MAPE would then crown the worst-listed
    model with no visible reason, and a correct answer would read as a bug. The rows
    are ordered by the number that actually decided the contest, and each row carries
    that number, so the ranking is checkable by the reader.
    """
    from src.kpi.forecast import service as svc

    # seasonal models are graded from month 24, trend from month 8 -> unequal sets
    result = svc.forecast_series(
        _series(_seasonal(42)), horizon=6, include_timesfm=False, cache=None
    )
    rows = result.to_payload()["backtest"]["models"]
    assert rows[0]["model"] == result.champion, (
        f"the crowned model must be the first row: {[r['model'] for r in rows]}"
    )
    shared = [r["monthly_mape_on_shared_origins_pct"] for r in rows]
    assert all(v is not None for v in shared)
    assert shared == sorted(shared), f"rows are not in the deciding order: {shared}"
    # And the deciding number really is the champion's minimum.
    assert shared[0] == min(shared)


def test_the_shared_origin_mape_is_the_same_statistic_restricted_not_a_new_one():
    """If it were a different statistic, the fix would silently change the metric as
    well as the comparison set, and 'no-op when all origins are equal' would be false.

    Verified on real PROD data across 36 scores: max difference 0.0.
    """
    from src.kpi.forecast import service as svc

    result = svc.forecast_series(
        _series(_seasonal(60)), horizon=6, include_timesfm=False, cache=None
    )
    for score in result.scores:
        assert bt.mape_on_origins(score, score.origin_cutoffs) == pytest.approx(score.monthly_mape)


def test_the_counters_cannot_disagree_with_the_set_they_count():
    """selection_origins/selection_fell_back are derived, not stored beside the set."""
    from src.kpi.forecast import service as svc

    result = svc.forecast_series(
        _series(_seasonal(42)), horizon=6, include_timesfm=False, cache=None
    )
    assert result.selection_origins == len(result.shared_cutoffs)
    assert result.selection_fell_back == (len(result.shared_cutoffs) < bt.MIN_COMMON_ORIGINS)


def test_each_model_s_reported_mape_is_still_over_its_own_origins():
    """Selection moved to shared ground; the per-model numbers must not, or the
    reader loses the only signal of how much evidence each model has."""
    from src.kpi.forecast import service as svc

    result = svc.forecast_series(
        _series(_seasonal(48)), horizon=6, include_timesfm=False, cache=None
    )
    by_name = {s.name: s for s in result.scores}
    for entry in result.to_payload()["backtest"]["models"]:
        score = by_name[entry["model"]]
        assert entry["origins_scored"] == score.n_origins == len(score.origin_cutoffs)
        assert entry["monthly_mape_pct"] == pytest.approx(round(score.monthly_mape, 2))


def test_the_selection_basis_survives_the_cache_round_trip():
    """A cache hit that dropped these would render selection_origins as 0 and claim
    the champion was chosen on no shared ground at all."""
    from src.kpi.forecast import service as svc

    result = svc.forecast_series(
        _series(_seasonal(48)), horizon=6, include_timesfm=False, cache=None
    )
    restored = svc._from_cache_entry(svc._to_cache_entry(result))
    assert restored.selection_origins == result.selection_origins
    assert restored.selection_fell_back == result.selection_fell_back
    assert restored.origins_used == result.origins_used
    for a, b in zip(restored.scores, result.scores, strict=True):
        assert a.origin_cutoffs == b.origin_cutoffs


def test_the_cache_contract_version_was_bumped_for_the_new_champion_rule():
    """A v1 entry holds a champion chosen under the OLD rule and has no
    origin_cutoffs, so serving one would be both wrong and a KeyError."""
    from src.kpi.forecast.cache import FORECAST_CONTRACT_VERSION

    assert not FORECAST_CONTRACT_VERSION.startswith("forecast-v1-")


def test_a_long_series_selects_on_every_origin_so_nothing_changes_there():
    """The live shape: every model shares all 24 cutoffs, so the fix is a no-op.

    60 months, not the live 164: measured, both give every model the full 24 origins
    (the asymmetry closes at 60), and 164 costs three seasonal Holt-Winters refits
    over 24 origins each -- the ~40s workload that made this file time out. A test
    that is slow enough to flake is a worse guard than the cheaper one proving the
    same property.
    """
    from src.kpi.forecast import service as svc

    result = svc.forecast_series(
        _series(_seasonal(60)), horizon=6, include_timesfm=False, cache=None
    )
    counts = {s.n_origins for s in result.scores}
    assert counts == {24}
    assert result.selection_origins == 24
    assert result.selection_fell_back is False
