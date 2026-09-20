"""The swappable forecast-model interface and its in-process Holt-Winters members (#2115).

Every accuracy assertion runs the real statsmodels fit over the real PROD canonical
TRx series (tests/fixtures/forecast/canonical_trx_monthly.csv) — no stubbed model and
no synthesised curve, because a model that only has to beat a curve we generated
ourselves has been graded on the wrong question.
"""

from __future__ import annotations

import math

import pytest

from src.kpi.forecast import backtest as bt
from src.kpi.forecast import models as fm
from tests.unit.test_kpi.conftest_forecast_fixture import series_values

BRANDS = ("Kisqali", "Fabhalta", "Remibrutinib")


# --------------------------------------------------------------------- the interface
def test_every_registered_model_satisfies_the_same_interface():
    for model in fm.all_models():
        assert isinstance(model.name, str) and model.name
        assert isinstance(model.description, str) and model.description
        assert callable(model.predict)
        assert isinstance(model.min_observations, int)


def test_model_names_are_unique_and_stable():
    names = [m.name for m in fm.all_models()]
    assert len(names) == len(set(names))
    assert "holt_winters_seasonal_add" in names
    assert "holt_winters_seasonal_mul" in names
    assert "holt_winters_trend" in names
    assert "timesfm_2_5" in names


def test_the_in_process_models_are_exactly_the_ones_that_need_no_worker():
    in_process = {m.name for m in fm.in_process_models()}
    assert in_process == {
        "holt_winters_seasonal_add",
        "holt_winters_seasonal_mul",
        "holt_winters_trend",
    }
    assert "timesfm_2_5" not in in_process


def test_holt_winters_always_runs_so_a_forecast_never_depends_on_the_worker():
    """Owner decision 2026-09-15: Holt-Winters ALWAYS runs; TimesFM is the optional half."""
    assert fm.in_process_models(), "at least one model must need no external service"
    assert all(m.requires_worker is False for m in fm.in_process_models())
    assert fm.get_model("timesfm_2_5").requires_worker is True


# ------------------------------------------------------------------- shape contracts
@pytest.mark.parametrize("model", fm.in_process_models(), ids=lambda m: m.name)
def test_a_model_returns_exactly_the_horizon_it_was_asked_for(model):
    y = series_values("Kisqali")
    out = model.predict(y, 6)
    assert len(out) == 6
    assert all(math.isfinite(v) for v in out)


@pytest.mark.parametrize("model", fm.in_process_models(), ids=lambda m: m.name)
def test_a_model_refuses_a_series_shorter_than_it_can_honestly_fit(model):
    with pytest.raises(fm.SeriesTooShort):
        model.predict([1.0] * (model.min_observations - 1), 6)


def test_the_seasonal_models_need_two_full_cycles_and_the_trend_model_does_not():
    assert fm.get_model("holt_winters_seasonal_add").min_observations >= 24
    assert fm.get_model("holt_winters_seasonal_mul").min_observations >= 24
    assert fm.get_model("holt_winters_trend").min_observations < 24


def test_a_multiplicative_model_refuses_a_series_with_a_non_positive_value():
    """A multiplicative season is undefined at zero — refuse, never return nan."""
    y = series_values("Kisqali")[:60]
    y[10] = 0.0
    with pytest.raises(fm.SeriesNotPositive):
        fm.get_model("holt_winters_seasonal_mul").predict(y, 6)


# ------------------------------------------------------- real accuracy on real series
# Fitting three Holt-Winters forms over 24 rolling origins for three brands is ~40 s of
# real statsmodels work. It is scored ONCE here and asserted many times: the cost is the
# point (these are the actual numbers the tool will serve), the repetition is not.


@pytest.fixture(scope="module")
def live_scores():
    out = {}
    for brand in BRANDS:
        y = series_values(brand)
        kw = {"horizon": 6, "origins": bt.DEFAULT_ORIGINS}
        out[brand] = {
            "naive": bt.score_model(lambda a, h: [a[-1]] * h, y, name="naive", **kw),
            "snaive": bt.score_model(
                lambda a, h: [a[-12 + i % 12] for i in range(h)], y, name="snaive", **kw
            ),
            "models": {
                m.name: bt.score_model(m.predict, y, name=m.name, **kw)
                for m in fm.in_process_models()
            },
        }
    return out


@pytest.mark.timeout(300)
@pytest.mark.parametrize("brand", BRANDS)
def test_every_holt_winters_model_beats_the_naive_baselines_on_the_live_series(brand, live_scores):
    """If a model cannot beat 'the next six months look like last month', it earns nothing."""
    got = live_scores[brand]
    for name, score in got["models"].items():
        assert score.n_failed_origins == 0, f"{name} failed origins on {brand}"
        assert score.monthly_mape < got["naive"].monthly_mape, f"{name} lost to naive on {brand}"
        assert score.monthly_mape < got["snaive"].monthly_mape, f"{name} lost to snaive on {brand}"


@pytest.mark.timeout(300)
@pytest.mark.parametrize("brand", BRANDS)
def test_the_champion_tracks_the_live_series_to_under_ten_percent_monthly(brand, live_scores):
    """Measured 2026-09-20: 6.2-7.9% monthly MAPE per brand. 10% is the regression floor."""
    champion = bt.select_champion(list(live_scores[brand]["models"].values()))
    assert champion.monthly_mape < 10.0
    assert champion.horizon_total_error < 8.0


@pytest.mark.timeout(300)
def test_the_champion_is_not_the_same_model_for_every_brand(live_scores):
    """The per-request backtest is load-bearing, not ceremony.

    Measured 2026-09-20 on the live series at 24 origins: seasonal-additive wins
    Kisqali and Fabhalta, seasonal-multiplicative wins Remibrutinib; at 12 origins
    the ranking flips again. A hard-coded single model would be wrong on at least one
    brand, which is the whole reason the owner asked for a model chosen per request.
    """
    champions = {bt.select_champion(list(live_scores[b]["models"].values())).name for b in BRANDS}
    assert len(champions) > 1, f"one model won everywhere: {champions}"


@pytest.mark.timeout(120)
def test_a_seasonal_model_actually_reproduces_the_january_trough():
    """Lane A planted a January trough (~0.92 of trend); a seasonal fit must find it.

    A trend-only model cannot, which is what the seasonal members are FOR.
    """
    y = series_values("Kisqali")
    # The fixture ends 2026-08, so step 4 is 2026-12 and step 5 is 2027-01.
    forecast = fm.get_model("holt_winters_seasonal_add").predict(y, 6)
    trend_only = fm.get_model("holt_winters_trend").predict(y, 6)
    december, january = forecast[3], forecast[4]
    assert january < december, "the seasonal fit must carry the January drop"
    assert abs(trend_only[4] / trend_only[3] - 1.0) < abs(january / december - 1.0)
