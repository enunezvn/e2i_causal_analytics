"""forecast_kpi: series -> champion -> forecast + band, and what it refuses (#2115).

The series under test is built by the REAL ``shape_monthly_series`` from the REAL PROD
rows in the fixture, and the models are the real statsmodels fits. Nothing here stands
in for the thing it is testing: the numbers asserted are the numbers a chat answer
would carry.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.kpi import canonical_volume_series as cvs
from src.kpi.forecast import service as svc
from tests.unit.test_kpi.conftest_forecast_fixture import load_canonical_trx

# Real statsmodels fits over 24 rolling origins, and the worker-degradation tests
# import the Celery app (which autodiscovers the causal tasks, pulling in dowhy).
# That cost is the point of these tests, so the file gets an explicit budget rather
# than the 30 s default meant for cheap unit tests.
pytestmark = pytest.mark.timeout(600)


def build_series(
    brand: str = "Kisqali",
    months: int | None = None,
    metric: str = "trx",
    drop_newest: int = 0,
):
    """A real CanonicalVolumeSeries, shaped from real rows by the real shaper.

    ``drop_newest`` rolls ``data_through`` BACK; ``months`` only shortens the history
    behind it. They are different edits and the cache key distinguishes exactly one of
    them, so the tests need both.
    """
    points = load_canonical_trx()[brand]
    if drop_newest:
        points = points[:-drop_newest]
    if months is not None:
        points = points[-months:]
    rows = [{"metric_date": m.isoformat(), "value": v, "n_rows": 4} for m, v in points]
    return cvs.shape_monthly_series(
        rows,
        metric=metric,
        brand=brand,
        region=None,
        as_of=date(2026, 9, 20),
        query_id="canonical_volume_monthly_series",
    )


@pytest.fixture(scope="module")
def kisqali():
    # cache=None throughout this file except in the caching section: these tests assert
    # what a COMPUTED forecast does, and a process with a reachable Redis would
    # otherwise serve them an entry written by an earlier run.
    return svc.forecast_series(build_series("Kisqali"), horizon=6, cache=None)


# ------------------------------------------------------------------------- the forecast
def test_the_forecast_starts_the_month_after_the_last_complete_month(kisqali):
    assert kisqali.data_through == date(2026, 8, 31)
    assert [p.month for p in kisqali.points] == [
        date(2026, 9, 1),
        date(2026, 10, 1),
        date(2026, 11, 1),
        date(2026, 12, 1),
        date(2027, 1, 1),
        date(2027, 2, 1),
    ]


def test_two_quarters_is_six_monthly_points(kisqali):
    assert kisqali.horizon == 6 and len(kisqali.points) == 6


def test_every_point_carries_a_band_that_brackets_it(kisqali):
    for p in kisqali.points:
        assert p.lower <= p.value <= p.upper
        assert p.lower >= 0.0


def test_the_forecast_is_in_the_scale_of_the_series_it_extends(kisqali):
    """A canonical Kisqali TRx month is ~800k. A forecast off by an order of magnitude
    would still satisfy every structural assertion above, so the scale is asserted."""
    recent = [p.value for p in build_series("Kisqali").points[-12:]]
    lo, hi = min(recent) * 0.5, max(recent) * 1.8
    for p in kisqali.points:
        assert lo < p.value < hi, f"{p.month} forecast {p.value} outside {lo}..{hi}"


def test_the_band_widens_over_the_horizon(kisqali):
    """Six months out is genuinely less knowable than one, and the band must say so."""
    first = kisqali.points[0].upper - kisqali.points[0].lower
    last = kisqali.points[-1].upper - kisqali.points[-1].lower
    assert last > first


# --------------------------------------------------------------------------- the scores
def test_every_model_that_ran_is_reported_with_its_own_measured_error(kisqali):
    names = {s.name for s in kisqali.scores}
    assert {"holt_winters_seasonal_add", "holt_winters_seasonal_mul", "holt_winters_trend"} <= names
    for s in kisqali.scores:
        assert s.n_origins > 0
        assert 0.0 <= s.monthly_mape < 100.0


def test_the_champion_is_one_of_the_reported_models_and_the_lowest_error_one(kisqali):
    best = min(kisqali.scores, key=lambda s: (s.monthly_mape, s.name))
    assert kisqali.champion == best.name
    assert kisqali.champion_mape == pytest.approx(best.monthly_mape)


def test_the_backtest_is_reported_with_the_origins_it_actually_used(kisqali):
    assert kisqali.origins_used == max(s.n_origins for s in kisqali.scores)
    assert kisqali.origins_used >= 12


def test_the_forecast_comes_from_the_champion_not_from_some_other_model(kisqali):
    """The served numbers must be the champion's own full-series fit.

    A service that scored one model and forecast with another would pass every
    structural test above while reporting an error figure that belongs to neither.
    """
    from src.kpi.forecast import models as fm

    direct = fm.get_model(kisqali.champion).predict(
        [p.value for p in build_series("Kisqali").points], 6
    )
    assert [p.value for p in kisqali.points] == pytest.approx(direct, rel=1e-6)


# ------------------------------------------------------------------------- provenance
def test_the_forecast_carries_where_its_history_came_from(kisqali):
    assert kisqali.metric == "trx"
    assert kisqali.brand == "Kisqali"
    assert kisqali.region is None
    assert kisqali.series_query_id == "canonical_volume_monthly_series"
    assert kisqali.n_observations == len(build_series("Kisqali").points)


def test_a_forecast_is_never_silently_a_different_horizon_than_asked():
    out = svc.forecast_series(build_series("Kisqali"), horizon=3, cache=None)
    assert out.horizon == 3 and len(out.points) == 3


# --------------------------------------------------------------------------- refusals
def test_a_series_too_short_for_any_model_is_refused_with_its_length():
    with pytest.raises(svc.ForecastRefused) as exc:
        svc.forecast_series(build_series("Kisqali", months=12), horizon=6, cache=None)
    assert "12" in str(exc.value)


def test_a_series_with_no_estimable_season_is_served_trend_only_rather_than_refused():
    """20 months has no yearly shape but does have a trend, and a trend forecast with
    a MEASURED error is worth more than a refusal. The thinner comparison is disclosed."""
    out = svc.forecast_series(
        build_series("Kisqali", months=20), horizon=6, include_timesfm=False, cache=None
    )
    assert out.champion == "holt_winters_trend"
    assert "holt_winters_seasonal_add" in dict(out.models_skipped)


def test_a_model_whose_error_rests_on_too_few_origins_is_skipped_not_served():
    """The band IS the error distribution, so a three-origin model has no band to show.

    The boundary is exact and worth pinning on both sides: at horizon 3 the trend model
    (8-month minimum fit) gets ``n - 3 - 8 + 1`` origins, so 13 months yields 3 and is
    refused while 14 yields 4 and is served. A gate that only tested the refusing side
    would also pass if the tool refused everything.
    """
    with pytest.raises(svc.ForecastRefused) as exc:
        svc.forecast_series(
            build_series("Kisqali", months=13), horizon=3, include_timesfm=False, cache=None
        )
    assert "origins" in str(exc.value)

    served = svc.forecast_series(
        build_series("Kisqali", months=14), horizon=3, include_timesfm=False, cache=None
    )
    assert served.origins_used == svc.MIN_BACKTEST_ORIGINS == 4


def test_an_empty_series_is_refused_rather_than_forecast_from_nothing():
    empty = cvs.shape_monthly_series(
        [], metric="trx", brand="Kisqali", region=None, as_of=date(2026, 9, 20), query_id="q"
    )
    with pytest.raises(svc.ForecastRefused):
        svc.forecast_series(empty, horizon=6, cache=None)


def test_a_horizon_of_zero_or_less_is_refused():
    for bad in (0, -1):
        with pytest.raises(svc.ForecastRefused):
            svc.forecast_series(build_series("Kisqali"), horizon=bad, cache=None)


def test_a_horizon_beyond_what_the_backtest_can_grade_is_refused():
    """The tool may not serve a 36-month forecast whose error it never measured."""
    with pytest.raises(svc.ForecastRefused):
        svc.forecast_series(build_series("Kisqali"), horizon=svc.MAX_HORIZON + 1, cache=None)


# ----------------------------------------------------------------- TimesFM degradation
def test_holt_winters_still_serves_the_forecast_when_the_worker_is_absent():
    """The owner's design: Holt-Winters ALWAYS runs. An absent worker costs a model,
    never the answer — and the absence is reported, not hidden."""
    out = svc.forecast_series(build_series("Kisqali"), horizon=6, include_timesfm=True, cache=None)
    assert out.points and out.champion.startswith("holt_winters")
    skipped = dict(out.models_skipped)
    assert "timesfm_2_5" in skipped
    assert skipped["timesfm_2_5"], "the reason the model did not run must be recorded"


def test_timesfm_is_not_even_attempted_when_it_is_switched_off():
    out = svc.forecast_series(build_series("Kisqali"), horizon=6, include_timesfm=False, cache=None)
    assert "timesfm_2_5" not in dict(out.models_skipped)
    assert "timesfm_2_5" not in {s.name for s in out.scores}


def test_a_model_that_cannot_fit_this_series_is_skipped_with_its_reason_not_crashed():
    """A 14-month series cannot carry a seasonal fit; the trend model still can."""
    out = svc.forecast_series(
        build_series("Kisqali", months=14), horizon=3, include_timesfm=False, cache=None
    )
    assert out.champion == "holt_winters_trend"
    skipped = dict(out.models_skipped)
    assert "holt_winters_seasonal_add" in skipped
    assert "observations" in skipped["holt_winters_seasonal_add"].lower()


# --------------------------------------------------------------------- serialisability
def test_the_result_serialises_to_something_a_chat_payload_can_carry(kisqali):
    payload = kisqali.to_payload()
    assert payload["champion"] == kisqali.champion
    assert payload["metric"] == "trx" and payload["brand"] == "Kisqali"
    assert payload["data_through"] == "2026-08-31"
    assert len(payload["forecast"]) == 6
    first = payload["forecast"][0]
    assert set(first) == {"month", "value", "lower", "upper"}
    assert first["month"] == "2026-09"
    assert payload["horizon_total"] == pytest.approx(sum(p.value for p in kisqali.points))
    assert payload["served_from_cache"] is False
    assert payload["backtest"]["origins"] == kisqali.origins_used
    assert payload["backtest"]["champion_monthly_mape_pct"] == pytest.approx(
        kisqali.champion_mape, abs=0.005
    )
    by_name = {m["model"]: m for m in payload["backtest"]["models"]}
    assert set(by_name) == {s.name for s in kisqali.scores}

    import json

    json.dumps(payload)  # must survive the chat transport


# ------------------------------------------------------------------ caching (#2115)
class _Redis:
    def __init__(self):
        self.data = {}
        self.writes = 0

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.writes += 1
        self.data[key] = value


def test_a_second_identical_forecast_is_served_from_the_cache():
    """~40 s of real fitting per contest, for a series that changes once a month."""
    from src.kpi.forecast.cache import ForecastCache

    store = ForecastCache(client=_Redis())
    series = build_series("Kisqali")
    first = svc.forecast_series(series, horizon=6, include_timesfm=False, cache=store)
    second = svc.forecast_series(series, horizon=6, include_timesfm=False, cache=store)
    assert second.from_cache is True
    assert first.from_cache is False
    assert second.to_payload()["forecast"] == first.to_payload()["forecast"]
    assert second.champion == first.champion


def test_a_new_month_of_history_is_recomputed_rather_than_served_from_august():
    """The property the key buys: a cache hit can never be about the wrong history."""
    from src.kpi.forecast.cache import ForecastCache

    store = ForecastCache(client=_Redis())
    through_july = build_series("Kisqali", drop_newest=1)
    first = svc.forecast_series(through_july, horizon=6, include_timesfm=False, cache=store)
    assert first.from_cache is False and first.data_through == date(2026, 7, 31)

    through_august = build_series("Kisqali")
    out = svc.forecast_series(through_august, horizon=6, include_timesfm=False, cache=store)
    assert out.from_cache is False, "a new month of history must not be served from the old one"
    assert out.data_through == date(2026, 8, 31)
    assert out.points[0].month == date(2026, 9, 1)


def test_a_different_horizon_is_not_answered_from_the_six_month_entry():
    from src.kpi.forecast.cache import ForecastCache

    store = ForecastCache(client=_Redis())
    series = build_series("Kisqali")
    svc.forecast_series(series, horizon=6, include_timesfm=False, cache=store)
    out = svc.forecast_series(series, horizon=3, include_timesfm=False, cache=store)
    assert out.from_cache is False and out.horizon == 3


def test_caching_can_be_switched_off_without_disabling_the_forecast():
    out = svc.forecast_series(build_series("Kisqali"), horizon=3, include_timesfm=False, cache=None)
    assert out.points and out.from_cache is False


def test_a_cached_result_reports_the_same_models_and_skips_as_the_computed_one():
    """A cache hit must not quietly drop the 'timesfm did not run' disclosure."""
    from src.kpi.forecast.cache import ForecastCache

    store = ForecastCache(client=_Redis())
    series = build_series("Kisqali")
    first = svc.forecast_series(series, horizon=3, include_timesfm=False, cache=store)
    second = svc.forecast_series(series, horizon=3, include_timesfm=False, cache=store)
    assert second.from_cache is True
    assert {s.name for s in second.scores} == {s.name for s in first.scores}
    assert dict(second.models_skipped) == dict(first.models_skipped)
    assert second.origins_used == first.origins_used
    assert second.data_through == first.data_through
    assert second.champion_mape == pytest.approx(first.champion_mape, abs=0.005)
