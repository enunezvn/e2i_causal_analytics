"""Calendar seasonality profile for the monthly Rx-volume series (canonical TRx lane)."""

import math
import random
import types
from datetime import date

import numpy as np
import pytest

MONTHS = range(1, 13)


def _s():
    from src.ml.synthetic.generators import seasonality

    return seasonality


def test_profile_covers_every_calendar_month_once():
    assert sorted(_s().SEASONAL_DEVIATION_BP) == list(MONTHS)


def test_annual_mean_is_exactly_one():
    s = _s()
    # Integer basis points: the identity is exact, not a float approximation.
    assert sum(s.SEASONAL_DEVIATION_BP.values()) == 0
    factors = [s.seasonal_factor("trx", date(2026, m, 1)) for m in MONTHS]
    assert math.fsum(factors) / 12 == pytest.approx(1.0, abs=1e-12)


def test_max_absolute_deviation_is_eight_percent():
    assert max(abs(v) for v in _s().SEASONAL_DEVIATION_BP.values()) == 800


def test_shape_january_trough_december_peak_february_low_mild_summer_dip():
    bp = dict(_s().SEASONAL_DEVIATION_BP)
    values = list(bp.values())
    assert min(bp, key=bp.get) == 1 and values.count(bp[1]) == 1
    assert max(bp, key=bp.get) == 12 and values.count(bp[12]) == 1
    assert bp[2] < 0
    summer = [bp[6], bp[7], bp[8]]
    assert all(v < 0 for v in summer)
    assert min(summer) > bp[2] > bp[1]


@pytest.mark.parametrize("metric", ["trx", "nrx", "nbrx"])
def test_volume_metrics_carry_the_profile(metric):
    s = _s()
    assert s.seasonal_factor(metric, date(2027, 1, 15)) == pytest.approx(0.92)
    assert s.seasonal_factor(metric, date(2025, 12, 1)) == pytest.approx(1.08)
    assert s.seasonal_factor(metric, date(2026, 8, 1)) == pytest.approx(0.98)


@pytest.mark.parametrize(
    "metric", ["market_share", "conversion_rate", "hcp_engagement_score", "per_hcp_rollup"]
)
def test_non_volume_metrics_are_identity(metric):
    for m in MONTHS:
        assert _s().seasonal_factor(metric, date(2026, m, 1)) == 1.0


def test_keyed_on_calendar_month_not_year_or_day():
    s = _s()
    assert s.seasonal_factor("trx", date(2013, 7, 1)) == s.seasonal_factor("trx", date(2026, 7, 28))


def test_calling_the_factor_consumes_no_global_random_state():
    s = _s()
    np_before = np.random.get_state()
    py_before = random.getstate()
    for metric in ("trx", "nrx", "nbrx", "market_share", "conversion_rate", "hcp_engagement_score"):
        for m in MONTHS:
            s.seasonal_factor(metric, date(2026, m, 1))
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    assert np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert random.getstate() == py_before


def test_module_imports_nothing_from_numpy_or_random():
    # Module globals, not source text: a local generator would need an import.
    for name, obj in vars(_s()).items():
        if isinstance(obj, types.ModuleType):
            origin = obj.__name__
        else:
            origin = getattr(obj, "__module__", None)
        assert not str(origin).startswith(("numpy", "random")), (name, origin)
