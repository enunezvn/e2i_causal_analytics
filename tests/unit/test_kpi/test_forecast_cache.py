"""The forecast cache: keyed so a stale entry is structurally impossible (#2115).

A backtest is expensive and almost always redundant. MEASURED in the prod api image
(2 CPU, 2026-09-20): 12.3 s for the seasonal-additive fit over 24 origins, 17.8 s for
the multiplicative one, 3.6 s for the trend, 5.8 s for the batched TimesFM round trip —
about 40 s for a full contest, 34 s of it Holt-Winters. Threading them was measured and
is SLOWER (0.75x: the scipy fits hold the GIL), so the way to make a forecast cheap is
not to recompute one that cannot have changed.

And it cannot change often: the canonical series gains a month when the cron appends
one. So ``data_through`` is IN THE KEY, and an entry written against August is never
served once September closes — no TTL is doing the honest work, the key is. The key
also carries the model set that actually ran, so the answer from a day when the forecast
worker was dark is not served on a day when it is up.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.kpi.forecast import cache as fc


def _key(**kw):
    params = {
        "metric": "trx",
        "brand": "Kisqali",
        "region": None,
        "horizon": 6,
        "origins": 24,
        "data_through": date(2026, 8, 31),
        "n_observations": 164,
        "models": ("holt_winters_seasonal_add", "holt_winters_trend"),
    }
    params.update(kw)
    return fc.forecast_cache_key(**params)


def test_the_same_request_against_the_same_history_is_the_same_key():
    assert _key() == _key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("metric", "nrx"),
        ("brand", "Fabhalta"),
        ("region", "midwest"),
        ("horizon", 3),
        ("origins", 12),
        ("data_through", date(2026, 9, 30)),
        ("n_observations", 120),
        ("models", ("holt_winters_trend",)),
    ],
)
def test_anything_that_changes_the_answer_changes_the_key(field, value):
    assert _key(**{field: value}) != _key()


def test_a_new_month_of_history_cannot_be_answered_from_the_old_month():
    """The guard that makes a TTL unnecessary: the key names the history it rests on."""
    august = _key(data_through=date(2026, 8, 31))
    september = _key(data_through=date(2026, 9, 30))
    assert august != september


def test_two_histories_ending_in_the_same_month_but_of_different_length_differ():
    """``data_through`` alone is NOT enough, and this is how that was found.

    ``shape_monthly_series`` drops months whose row count falls short of the series'
    fullest month, so a backfill landing a missing region lengthens the history without
    moving its last month. Measured 2026-09-20 before ``n_observations`` was in the key:
    13, 14 and 164 months of Kisqali TRx all hashed to the same key, and a forecast
    fitted on fourteen years was served to a request whose series held thirteen months.
    """
    assert _key(n_observations=164) != _key(n_observations=13)


def test_the_model_set_is_in_the_key_so_a_dark_worker_day_is_not_served_on_a_live_one():
    without = _key(models=("holt_winters_seasonal_add",))
    with_tf = _key(models=("holt_winters_seasonal_add", "timesfm_2_5"))
    assert without != with_tf


def test_the_model_set_order_does_not_change_the_key():
    """The same contest is the same contest however the models were enumerated."""
    a = _key(models=("timesfm_2_5", "holt_winters_trend"))
    b = _key(models=("holt_winters_trend", "timesfm_2_5"))
    assert a == b


def test_the_contract_version_is_in_the_key():
    """Bumping it retires every entry at once when the meaning of a forecast changes —
    a new band rule or a changed champion metric must not be served from old entries."""
    assert fc.FORECAST_CONTRACT_VERSION in _key()


def test_the_key_is_namespaced_so_it_cannot_collide_with_another_cache():
    assert _key().startswith(fc.KEY_PREFIX)


def test_a_region_scoped_forecast_never_collides_with_the_national_one():
    assert _key(region=None) != _key(region="midwest")


# ------------------------------------------------------------------- the cache itself
def test_a_cache_with_no_redis_is_a_no_op_rather_than_an_error():
    """Redis is an optimisation here, never a dependency: a forecast must still serve."""
    store = fc.ForecastCache(client=None)
    assert store.enabled is False
    assert store.get("k") is None
    store.set("k", {"a": 1})  # must not raise
    assert store.get("k") is None


class _Redis:
    """A real dict behind the same two calls the cache makes."""

    def __init__(self, failing=False):
        self.data = {}
        self.failing = failing
        self.setex_calls = []

    def get(self, key):
        if self.failing:
            raise ConnectionError("redis down")
        return self.data.get(key)

    def setex(self, key, ttl, value):
        if self.failing:
            raise ConnectionError("redis down")
        self.setex_calls.append((key, ttl))
        self.data[key] = value


def test_a_stored_payload_round_trips():
    store = fc.ForecastCache(client=_Redis())
    payload = {"champion": "holt_winters_trend", "forecast": [{"month": "2026-09"}]}
    store.set("k", payload)
    assert store.get("k") == payload


def test_a_redis_that_errors_degrades_to_a_miss_instead_of_failing_the_forecast():
    store = fc.ForecastCache(client=_Redis(failing=True))
    store.set("k", {"a": 1})  # must not raise
    assert store.get("k") is None


def test_entries_still_expire_so_a_dead_series_cannot_be_served_forever():
    """The key makes staleness impossible; the TTL bounds unbounded key growth."""
    redis = _Redis()
    fc.ForecastCache(client=redis).set("k", {"a": 1})
    assert redis.setex_calls[0][1] == fc.DEFAULT_TTL_SECONDS
    assert fc.DEFAULT_TTL_SECONDS >= 24 * 3600


def test_a_corrupt_entry_is_a_miss_not_a_crash():
    redis = _Redis()
    redis.data["k"] = "{not json"
    assert fc.ForecastCache(client=redis).get("k") is None
