"""Cache for a computed forecast, keyed so a stale entry cannot exist (#2115, Lane B).

A backtest is expensive and almost always redundant. MEASURED in the prod api image
(2 CPU, 2026-09-20) on the 164-month live Kisqali series at 24 origins: 12.3 s for the
seasonal-additive fit, 17.8 s for the multiplicative, 3.6 s for the trend, 5.8 s for the
batched TimesFM round trip -- ~40 s for a full contest, 34 s of it Holt-Winters.
Threading the fits was measured and is SLOWER (0.75x; the scipy optimiser holds the
GIL), so the way to make a forecast cheap is not to recompute one that cannot have
changed.

And it rarely can. The canonical series gains a month when the cron appends one, so
``data_through`` IS IN THE KEY: an entry computed against August is not served once
September closes. That is what makes the result correct, not the TTL -- the TTL only
bounds key growth. The key also carries the model set that actually ran, so an answer
from a day when the forecast worker was dark is not served on a day when it is up.

Redis is an optimisation, never a dependency: with no client, or a client that errors,
every operation degrades to a miss and the forecast is computed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

KEY_PREFIX = "kpi:forecast:"

#: Bump when the MEANING of a forecast changes -- a new band rule, a different champion
#: metric, a changed default horizon. An entry written under the old meaning can then
#: never be served under the new one. (Same idea as KPI_CACHE_BASIS_VERSION.)
#: v1 -> v2 (#2199): the champion is now chosen on the origins every model SHARES,
#: not on each model's own origin set, so a v1 entry can hold a different champion
#: for the same series. A v1 entry also predates ``BacktestScore.origin_cutoffs`` and
#: would raise on deserialisation -- the version bump is what stops it being read at
#: all, rather than being caught as an error per key.
FORECAST_CONTRACT_VERSION = "forecast-v2-2026-09-20"

#: Eight days: longer than the gap between monthly appends would need, because
#: ``data_through`` already makes staleness impossible. This only stops keys for
#: brand/region/horizon combinations nobody asks for again from accumulating forever.
DEFAULT_TTL_SECONDS = int(os.getenv("E2I_FORECAST_CACHE_TTL", str(8 * 24 * 3600)))


def forecast_cache_key(
    *,
    metric: str,
    brand: Optional[str],
    region: Optional[str],
    horizon: int,
    origins: int,
    data_through: Optional[date],
    n_observations: int,
    band_quantile: float,
    models: Sequence[str],
) -> str:
    """Everything that can change the answer, and nothing that cannot.

    ``n_observations`` is in the key as well as ``data_through``, and it has to be.
    ``shape_monthly_series`` DROPS months whose row count falls short of the series'
    fullest month, so the history can lengthen or shorten while ``data_through`` stays
    exactly where it is -- a backfill landing a missing region, or an ETL run leaving
    one short. MEASURED 2026-09-20 before this field existed: 13, 14 and 164 complete
    months of Kisqali TRx all hashed to the same key, so a forecast fitted on fourteen
    years of history would have been served to a request whose series held one.

    ``band_quantile`` is here because it changes the SERVED BAND, and ``origins``
    because it changes the measured error the band is built from. No wired caller
    varies either today, so neither can collide right now -- they are in the key
    because this function's contract is "everything that can change the answer", and a
    contract that is only true by accident of the current call sites is not one.

    ``models`` is SORTED: the same contest is the same contest however the models were
    enumerated, and an unsorted key would miss on every other call for no reason.
    """
    parts = [
        FORECAST_CONTRACT_VERSION,
        metric,
        # Both dimensions are case-folded. Today every production caller canonicalises
        # brand and region through resolve_brand_label / resolve_region_label before
        # reaching here, so the asymmetry was inert — but a key that folds one
        # dimension and not the other is a trap for the first caller that does not.
        (brand or "*").lower(),
        (region or "*").lower(),
        str(horizon),
        str(origins),
        data_through.isoformat() if data_through else "none",
        str(n_observations),
        f"{band_quantile:.6f}",
        ",".join(sorted(models)),
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"{KEY_PREFIX}{FORECAST_CONTRACT_VERSION}:{digest}"


#: ``client=None`` must mean "no cache", not "go and find one" -- otherwise a caller
#: that deliberately disables caching gets whatever Redis happens to be reachable.
#: Omitting the argument entirely is the one case that resolves a default client.
_UNSET = object()


class ForecastCache:
    """Redis-backed, fail-open. Every failure is a miss."""

    def __init__(self, client: Any = _UNSET, ttl: int = DEFAULT_TTL_SECONDS):
        self._ttl = ttl
        self._redis = _default_client() if client is _UNSET else client

    @property
    def enabled(self) -> bool:
        return self._redis is not None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
        except Exception as exc:  # noqa: BLE001 — a cache is never a reason to fail
            logger.debug("forecast cache read failed (degrading to a miss): %s", exc)
            return None
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("forecast cache entry at %s is not JSON; treating as a miss", key)
            return None
        return value if isinstance(value, dict) else None

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(key, self._ttl, json.dumps(payload, default=str))
        except Exception as exc:  # noqa: BLE001
            logger.debug("forecast cache write failed (ignored): %s", exc)


def _default_client() -> Any:
    try:
        import redis
    except Exception:  # noqa: BLE001
        return None
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception as exc:  # noqa: BLE001
        logger.debug("forecast cache disabled (no Redis at %s): %s", url, exc)
        return None


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "FORECAST_CONTRACT_VERSION",
    "KEY_PREFIX",
    "ForecastCache",
    "forecast_cache_key",
]
