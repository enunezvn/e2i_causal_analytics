"""Rolling-origin backtest, champion selection and the prediction band (#2115, Lane B).

The band and the champion BOTH come from the same measured object — the per-origin,
per-step errors the model actually made on this series. That is the only thing the
tool can honestly claim about a forecast it has never seen the outcome of, and it is
what makes two models with different internals comparable: neither gets to grade
itself with its own notion of uncertainty.
"""

from __future__ import annotations

import math

import pytest

from src.kpi.forecast import backtest as bt


# --------------------------------------------------------------------------- origins
def test_origins_are_the_latest_windows_and_never_peek_past_their_own_cutoff():
    """Origin at cutoff t trains on y[:t] and is graded on y[t:t+h].

    The origins are the LAST ones the series supports, so the newest months — the
    ones whose regime the forecast will actually extend — are the ones the models
    are graded on. An earlier-origin scheme would grade 2019 behaviour.
    """
    splits = bt.rolling_origin_splits(n=100, horizon=6, origins=3)
    assert [(tr.start, tr.stop, te.start, te.stop) for tr, te in splits] == [
        (0, 92, 92, 98),
        (0, 93, 93, 99),
        (0, 94, 94, 100),
    ]


def test_every_split_is_strictly_causal():
    for train, test in bt.rolling_origin_splits(n=164, horizon=6, origins=24):
        assert train.stop == test.start, "the test window must begin where training ends"
        assert len(test) == 6
        assert train.start == 0


def test_the_last_split_ends_at_the_end_of_the_series():
    splits = bt.rolling_origin_splits(n=164, horizon=6, origins=24)
    assert splits[-1][1].stop == 164


def test_origins_shrink_rather_than_borrowing_data_that_does_not_exist():
    """A short series yields fewer origins — never a split with a train set below the floor."""
    splits = bt.rolling_origin_splits(n=30, horizon=6, origins=24, min_train=24)
    assert splits, "a 30-month series must still support at least one origin"
    assert all(len(tr) >= 24 for tr, _ in splits)
    assert len(splits) == 1


def test_a_series_too_short_for_even_one_origin_yields_no_splits():
    assert bt.rolling_origin_splits(n=25, horizon=6, origins=24, min_train=24) == []


# --------------------------------------------------------------------------- scoring
def _flat_predictor(value):
    return lambda y, h: [value] * h


def test_a_perfect_predictor_scores_zero_error():
    y = [100.0] * 60
    score = bt.score_model(_flat_predictor(100.0), y, horizon=6, origins=5, name="perfect")
    assert score.name == "perfect"
    assert score.monthly_mape == pytest.approx(0.0)
    assert score.horizon_total_error == pytest.approx(0.0)
    assert score.n_origins == 5
    assert score.horizon == 6


def test_mape_is_the_mean_absolute_percentage_error_over_every_graded_step():
    y = [100.0] * 60
    score = bt.score_model(_flat_predictor(110.0), y, horizon=6, origins=5, name="ten-high")
    assert score.monthly_mape == pytest.approx(10.0)
    assert score.horizon_total_error == pytest.approx(10.0)


def test_the_horizon_total_error_lets_opposite_signed_errors_cancel_but_mape_does_not():
    """+20%/-20% alternating: the 6-month TOTAL is right, the monthly path is not.

    This is why the champion is chosen on MAPE: a model can win the total by
    cancellation while tracking the path badly, and 6.5 asks for both quarters.
    """
    y = [100.0] * 60

    def predict(_y, h):
        return [120.0 if i % 2 == 0 else 80.0 for i in range(h)]

    score = bt.score_model(predict, y, horizon=6, origins=5, name="alternating")
    assert score.monthly_mape == pytest.approx(20.0)
    assert score.horizon_total_error == pytest.approx(0.0, abs=1e-9)


def test_the_per_step_errors_are_retained_for_the_band():
    y = [100.0] * 60
    score = bt.score_model(_flat_predictor(110.0), y, horizon=6, origins=5, name="ten-high")
    assert len(score.step_pct_errors) == 6, "one bucket per horizon step"
    assert all(len(bucket) == 5 for bucket in score.step_pct_errors), "one entry per origin"
    assert score.step_pct_errors[0] == pytest.approx((10.0,) * 5)


def test_signed_step_errors_are_retained_so_the_band_can_be_asymmetric():
    y = [100.0] * 60

    def predict(_y, h):
        return [120.0 if i % 2 == 0 else 80.0 for i in range(h)]

    score = bt.score_model(predict, y, horizon=6, origins=5, name="alternating")
    assert score.signed_step_pct_errors[0] == pytest.approx((20.0,) * 5)
    assert score.signed_step_pct_errors[1] == pytest.approx((-20.0,) * 5)


def test_a_model_that_raises_on_some_origins_is_scored_on_the_rest_and_says_so():
    y = [100.0] * 60
    calls = {"n": 0}

    def flaky(_y, h):
        calls["n"] += 1
        if calls["n"] % 2:
            raise RuntimeError("no fit")
        return [100.0] * h

    score = bt.score_model(flaky, y, horizon=6, origins=6, name="flaky")
    assert score.n_origins == 3
    assert score.n_failed_origins == 3


def test_a_model_that_never_fits_is_unscorable_rather_than_scored_zero():
    def always_fails(_y, _h):
        raise RuntimeError("no fit")

    with pytest.raises(bt.ModelUnscorable):
        bt.score_model(always_fails, [100.0] * 60, horizon=6, origins=5, name="broken")


def test_a_non_finite_prediction_is_a_failed_origin_not_a_nan_score():
    """statsmodels can converge to nan; a nan must never become the served forecast."""
    y = [100.0] * 60
    calls = {"n": 0}

    def sometimes_nan(_y, h):
        calls["n"] += 1
        return [float("nan")] * h if calls["n"] % 2 else [100.0] * h

    score = bt.score_model(sometimes_nan, y, horizon=6, origins=6, name="nanny")
    assert score.n_origins == 3 and score.n_failed_origins == 3
    assert math.isfinite(score.monthly_mape)


def test_a_zero_actual_cannot_divide_by_zero():
    y = [0.0] * 60
    score = bt.score_model(_flat_predictor(0.0), y, horizon=6, origins=5, name="zeros")
    assert math.isfinite(score.monthly_mape)


# --------------------------------------------------------------------------- champion
def _score(name, mape):
    return bt.BacktestScore(
        name=name,
        monthly_mape=mape,
        horizon_total_error=mape,
        n_origins=24,
        n_failed_origins=0,
        horizon=6,
        step_pct_errors=((mape,) * 24,) * 6,
        signed_step_pct_errors=((mape,) * 24,) * 6,
    )


def test_the_champion_is_the_lowest_monthly_mape():
    scores = [_score("a", 7.5), _score("b", 6.2), _score("c", 9.0)]
    assert bt.select_champion(scores).name == "b"


def test_a_tie_is_broken_deterministically_by_name_not_by_dict_order():
    assert bt.select_champion([_score("zebra", 6.0), _score("alpha", 6.0)]).name == "alpha"
    assert bt.select_champion([_score("alpha", 6.0), _score("zebra", 6.0)]).name == "alpha"


def test_selecting_a_champion_from_nothing_refuses():
    with pytest.raises(bt.NoChampion):
        bt.select_champion([])


# --------------------------------------------------------------------------- band
# BAND RULE: for horizon step i, take the SIGNED relative errors s = (forecast-actual)/actual
# the model actually made at that step. The interval on the ACTUAL is
#   [point*(1-hi_q), point*(1-lo_q)]   (lo_q, hi_q = the (1-q)/2 and (1+q)/2 quantiles of s)
# then widened if needed to contain the point forecast, then floored at 0 for a volume.
# It is the model's own measured miss distribution, so the same mechanism grades a
# state-space model and a pretrained transformer on equal terms.


def _band_score(name, signed_by_step, horizon):
    return bt.BacktestScore(
        name=name,
        monthly_mape=0.0,
        horizon_total_error=0.0,
        n_origins=len(signed_by_step[0]),
        n_failed_origins=0,
        horizon=horizon,
        step_pct_errors=tuple(tuple(abs(v) for v in step) for step in signed_by_step),
        signed_step_pct_errors=tuple(tuple(step) for step in signed_by_step),
    )


def test_the_band_widens_with_the_horizon_when_the_errors_do():
    score = _band_score(
        "m",
        [
            (-1.0, -0.5, 0.0, 0.5, 1.0),  # step 1: the model has missed by ~1%
            (-20.0, -10.0, 0.0, 10.0, 20.0),  # step 6: by ~20%
        ],
        horizon=2,
    )
    (lo0, hi0), (lo1, hi1) = bt.band_from_errors(score, [100.0, 100.0], quantile=1.0)
    assert (hi0 - lo0) == pytest.approx(100.0 / 0.99 - 100.0 / 1.01, rel=1e-6)
    assert (hi1 - lo1) > 5 * (hi0 - lo0), "the later step's band must be far wider"


def test_a_symmetric_error_history_gives_a_symmetric_band():
    score = _band_score("m", [(-10.0, -5.0, 0.0, 5.0, 10.0)], horizon=1)
    lo, hi = bt.band_from_errors(score, [100.0], quantile=1.0)[0]
    assert lo == pytest.approx(100.0 / 1.10)
    assert hi == pytest.approx(100.0 / 0.90)


def test_the_band_always_contains_the_point_forecast():
    """A biased model still gets a band around the number the answer shows."""
    score = _band_score("biased", [(4.0, 5.0, 6.0)], horizon=1)
    lo, hi = bt.band_from_errors(score, [100.0], quantile=1.0)[0]
    assert lo <= 100.0 <= hi


def test_the_band_is_asymmetric_when_the_model_errs_in_one_direction():
    """A model that has only ever UNDER-forecast gets a band that leans up."""
    score = _band_score("low", [(-12.0, -10.0, -8.0)], horizon=1)
    lo, hi = bt.band_from_errors(score, [100.0], quantile=1.0)[0]
    assert hi - 100.0 > 100.0 - lo, "the band must lean toward where the misses actually were"
    assert hi == pytest.approx(100.0 / 0.88)


def test_a_volume_band_never_goes_negative():
    score = _band_score("wild", [(-90.0, 0.0, 250.0)], horizon=1)
    lo, _hi = bt.band_from_errors(score, [100.0], quantile=1.0, floor_at_zero=True)[0]
    assert lo >= 0.0


def test_a_narrower_quantile_gives_a_narrower_band():
    signed = tuple(float(v) for v in range(-25, 26))
    score = _band_score("spread", [signed], horizon=1)
    lo80, hi80 = bt.band_from_errors(score, [100.0], quantile=0.8)[0]
    lo95, hi95 = bt.band_from_errors(score, [100.0], quantile=0.95)[0]
    assert (hi80 - lo80) < (hi95 - lo95)
