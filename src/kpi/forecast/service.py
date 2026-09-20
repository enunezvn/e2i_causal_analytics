"""forecast_kpi: the canonical monthly series in, a champion forecast + band out (#2115).

This is the module the chat tool, the composable tool and the API all call. It does
four things and refuses rather than improvising when it cannot:

1. take the canonical monthly TRx/NRx/NBRx series (Lane A's read API, migration 143);
2. backtest every AVAILABLE model on the SAME rolling origins;
3. serve the lowest-error one, refit on the whole series;
4. attach that model's own measured miss distribution as the prediction band.

Two properties are load-bearing and are what the tests pin.

**The served numbers are the champion's.** Scoring one model and forecasting with
another would satisfy every structural check while reporting an error figure that
belongs to neither model.

**An absent TimesFM worker costs a model, never the answer.** Holt-Winters is
in-process and always runs (owner decision 2026-09-15); the box may not have headroom
for the forecast worker on any given day, and a chat turn must not depend on that.
Every model that did not run is reported WITH ITS REASON, so a thinner comparison is
visible instead of silent.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.kpi.canonical_volume_series import (
    CanonicalVolumeSeries,
    fetch_canonical_volume_series,
    month_end,
)
from src.kpi.forecast import backtest as bt
from src.kpi.forecast import models as fm
from src.kpi.forecast.cache import ForecastCache, forecast_cache_key

logger = logging.getLogger(__name__)

#: Two quarters is the ask; a year is the most the platform will extrapolate a monthly
#: series it can still grade. Beyond that the backtest on a 165-month history has too
#: few non-overlapping origins for the error figure to mean anything, and an ungraded
#: forecast is exactly what this lane exists to avoid.
MAX_HORIZON = 12

#: Below this many graded origins the measured error -- and therefore the BAND, which
#: is built from the per-step error distribution -- rests on too few observations to
#: present. Four is already thin; it is the floor, not a target. A model that cannot
#: reach it on this series is skipped with that reason rather than served with a band
#: drawn from one or two misses.
MIN_BACKTEST_ORIGINS = 4

#: The shortest history any model here can fit at all. The series is refused outright
#: only when even that model cannot be graded on it -- NOT when the seasonal models
#: cannot fit. A 20-month series has no estimable yearly shape but does have an
#: estimable trend, and a trend forecast with a measured error beats a refusal.
MIN_OBSERVATIONS = min(m.min_observations for m in fm.all_models())


#: ``cache=None`` means "do not cache"; omitting the argument means "use the shared
#: one". They are different requests, so they need different defaults -- hence a
#: sentinel rather than None.
_USE_DEFAULT_CACHE: Any = object()
_DEFAULT_CACHE: Optional[ForecastCache] = None


def _resolve_cache(cache: Any) -> Optional[ForecastCache]:
    """Build the shared client lazily: importing this module must open no socket."""
    global _DEFAULT_CACHE
    if cache is not _USE_DEFAULT_CACHE:
        return cache
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = ForecastCache()
    return _DEFAULT_CACHE


class ForecastRefused(RuntimeError):
    """The forecast cannot be served honestly; the message says what was missing."""


@dataclass(frozen=True)
class ForecastPoint:
    month: date
    value: float
    lower: float
    upper: float


@dataclass(frozen=True)
class KpiForecast:
    metric: str
    brand: Optional[str]
    region: Optional[str]
    horizon: int
    points: Tuple[ForecastPoint, ...]
    champion: str
    champion_description: str
    champion_mape: float
    champion_total_error: float
    scores: Tuple[bt.BacktestScore, ...]
    models_skipped: Tuple[Tuple[str, str], ...]
    data_through: Optional[date]
    n_observations: int
    series_query_id: str
    band_quantile: float
    #: The CHAMPION's own origin count -- never the best any model managed. Reported
    #: beside the champion's MAPE, so borrowing a losing model's larger number would
    #: overstate the evidence behind the figure the answer leads with.
    origins_used: int
    #: Months whose raw point forecast came back below zero and were floored. A
    #: volume cannot be negative, and a model extrapolating a steep decline past zero
    #: is saying its trend has stopped being physical -- which the reader is told,
    #: rather than being shown a silent 0.
    floored_months: Tuple[date, ...] = ()
    #: True when this result was read back from the cache rather than fitted now. It is
    #: reported rather than hidden: a reader comparing two answers minutes apart should
    #: be able to tell that the second did not re-fit anything.
    from_cache: bool = False

    @property
    def horizon_total(self) -> float:
        return float(sum(p.value for p in self.points))

    def to_payload(self) -> Dict[str, Any]:
        """The chat-facing shape: JSON-safe, and every number carries what produced it."""
        return {
            "metric": self.metric,
            "brand": self.brand,
            "region": self.region,
            "horizon_months": self.horizon,
            "data_through": self.data_through.isoformat() if self.data_through else None,
            "n_observations": self.n_observations,
            "series_query_id": self.series_query_id,
            # Disclosed, not hidden: a reader comparing two answers minutes apart
            # should be able to tell that the second re-fitted nothing.
            "served_from_cache": self.from_cache,
            "champion": self.champion,
            "champion_description": self.champion_description,
            "forecast": [
                {
                    "month": p.month.strftime("%Y-%m"),
                    "value": round(p.value, 1),
                    "lower": round(p.lower, 1),
                    "upper": round(p.upper, 1),
                    # ON THE MONTH, not only in the separate list below: a synthesising
                    # model reading the forecast array has to cross-reference a
                    # top-level list of month strings to notice otherwise, and the one
                    # it drops renders as a bare "0.0" -- the silent zero the floor
                    # exists to prevent, moved one layer up.
                    "floored_at_zero": p.month in self.floored_months,
                }
                for p in self.points
            ],
            "horizon_total": round(self.horizon_total, 1),
            # Disclosed, never silent: a floored month means the model extrapolated a
            # non-negative quantity below zero, so the honest reading of that month is
            # "at or near zero and the trend has left the range the fit is valid in",
            # not "exactly zero".
            "floored_at_zero_months": [m.strftime("%Y-%m") for m in self.floored_months],
            "band": {
                "coverage": self.band_quantile,
                "basis": (
                    "the champion's own rolling-origin miss distribution at each horizon "
                    "step, not a model-internal confidence interval"
                ),
            },
            "backtest": {
                "horizon_months": self.horizon,
                "origins": self.origins_used,
                "champion_monthly_mape_pct": round(self.champion_mape, 2),
                "champion_horizon_total_error_pct": round(self.champion_total_error, 2),
                "models": [
                    {
                        "model": s.name,
                        "monthly_mape_pct": round(s.monthly_mape, 2),
                        "horizon_total_error_pct": round(s.horizon_total_error, 2),
                        "origins_scored": s.n_origins,
                        "origins_failed": s.n_failed_origins,
                    }
                    for s in sorted(self.scores, key=lambda s: (s.monthly_mape, s.name))
                ],
                "models_not_run": [
                    {"model": name, "reason": reason} for name, reason in self.models_skipped
                ],
                "selection_rule": (
                    "lowest mean monthly MAPE over the same rolling origins; the "
                    "horizon total is reported but not selected on, because opposite-"
                    "signed monthly errors cancel in a sum"
                ),
            },
        }


def _score_to_dict(score: bt.BacktestScore) -> Dict[str, Any]:
    return {
        "name": score.name,
        "monthly_mape": score.monthly_mape,
        "horizon_total_error": score.horizon_total_error,
        "n_origins": score.n_origins,
        "n_failed_origins": score.n_failed_origins,
        "horizon": score.horizon,
        "step_pct_errors": [list(step) for step in score.step_pct_errors],
        "signed_step_pct_errors": [list(step) for step in score.signed_step_pct_errors],
    }


def _score_from_dict(raw: Dict[str, Any]) -> bt.BacktestScore:
    return bt.BacktestScore(
        name=raw["name"],
        monthly_mape=raw["monthly_mape"],
        horizon_total_error=raw["horizon_total_error"],
        n_origins=raw["n_origins"],
        n_failed_origins=raw["n_failed_origins"],
        horizon=raw["horizon"],
        step_pct_errors=tuple(tuple(step) for step in raw["step_pct_errors"]),
        signed_step_pct_errors=tuple(tuple(step) for step in raw["signed_step_pct_errors"]),
    )


def _to_cache_entry(result: "KpiForecast") -> Dict[str, Any]:
    """The WHOLE result, not the rendered payload.

    Storing only ``to_payload()`` would round every number and drop the per-step error
    distributions, so a cache hit would disclose less than a fresh computation -- the
    'timesfm did not run' note and the band's own basis among it.
    """
    return {
        "metric": result.metric,
        "brand": result.brand,
        "region": result.region,
        "horizon": result.horizon,
        "points": [
            {"month": p.month.isoformat(), "value": p.value, "lower": p.lower, "upper": p.upper}
            for p in result.points
        ],
        "champion": result.champion,
        "champion_description": result.champion_description,
        "champion_mape": result.champion_mape,
        "champion_total_error": result.champion_total_error,
        "scores": [_score_to_dict(s) for s in result.scores],
        "models_skipped": [list(pair) for pair in result.models_skipped],
        "data_through": result.data_through.isoformat() if result.data_through else None,
        "n_observations": result.n_observations,
        "series_query_id": result.series_query_id,
        "band_quantile": result.band_quantile,
        "origins_used": result.origins_used,
        "floored_months": [m.isoformat() for m in result.floored_months],
    }


def _from_cache_entry(raw: Dict[str, Any]) -> "KpiForecast":
    return KpiForecast(
        metric=raw["metric"],
        brand=raw["brand"],
        region=raw["region"],
        horizon=raw["horizon"],
        points=tuple(
            ForecastPoint(
                month=date.fromisoformat(p["month"]),
                value=p["value"],
                lower=p["lower"],
                upper=p["upper"],
            )
            for p in raw["points"]
        ),
        champion=raw["champion"],
        champion_description=raw["champion_description"],
        champion_mape=raw["champion_mape"],
        champion_total_error=raw["champion_total_error"],
        scores=tuple(_score_from_dict(s) for s in raw["scores"]),
        models_skipped=tuple((n, r) for n, r in raw["models_skipped"]),
        data_through=date.fromisoformat(raw["data_through"]) if raw["data_through"] else None,
        n_observations=raw["n_observations"],
        series_query_id=raw["series_query_id"],
        band_quantile=raw["band_quantile"],
        origins_used=raw["origins_used"],
        floored_months=tuple(date.fromisoformat(m) for m in raw.get("floored_months", ())),
        from_cache=True,
    )


def _forecast_months(data_through: date, horizon: int) -> List[date]:
    """The horizon months that follow the last COMPLETE month of history."""
    months: List[date] = []
    cursor = (data_through.replace(day=28) + timedelta(days=4)).replace(day=1)
    for _ in range(horizon):
        months.append(cursor)
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def _candidate_models(include_timesfm: bool) -> Tuple[fm.ForecastModel, ...]:
    models = fm.in_process_models()
    if include_timesfm:
        models = models + fm.worker_models()
    return models


def _score_one(
    model: fm.ForecastModel,
    values: Sequence[float],
    *,
    horizon: int,
    origins: int,
    min_train: int,
) -> bt.BacktestScore:
    if model.requires_worker:
        from src.kpi.forecast.timesfm import batched_predictor, dispatch_batch

        return bt.score_model(
            predict_many=batched_predictor(dispatch_batch),
            y=values,
            horizon=horizon,
            origins=origins,
            name=model.name,
            min_train=min_train,
        )
    return bt.score_model(
        model.predict,
        values,
        horizon=horizon,
        origins=origins,
        name=model.name,
        min_train=min_train,
    )


def forecast_series(
    series: CanonicalVolumeSeries,
    *,
    horizon: int = bt.DEFAULT_HORIZON,
    origins: int = bt.DEFAULT_ORIGINS,
    include_timesfm: bool = True,
    band_quantile: float = bt.DEFAULT_BAND_QUANTILE,
    cache: Any = _USE_DEFAULT_CACHE,
) -> KpiForecast:
    """Backtest every available model on ``series`` and serve the best one's forecast.

    ``cache=None`` disables caching for this call; omit it for the shared default.
    """
    if horizon < 1:
        raise ForecastRefused(f"horizon must be at least 1 month, got {horizon}")
    if horizon > MAX_HORIZON:
        raise ForecastRefused(
            f"horizon {horizon} exceeds the {MAX_HORIZON}-month maximum: beyond that the "
            "rolling-origin backtest cannot measure the error of what it would serve"
        )
    values = [p.value for p in series.points]
    if len(values) < MIN_OBSERVATIONS + horizon:
        raise ForecastRefused(
            f"{series.metric} for {series.brand or 'all brands'} has {len(values)} complete "
            f"months; a {horizon}-month forecast needs at least {MIN_OBSERVATIONS + horizon} "
            "before any model can be fitted and backtested"
        )

    cache = _resolve_cache(cache)
    candidates = _candidate_models(include_timesfm)
    cache_key = forecast_cache_key(
        metric=series.metric,
        brand=series.brand,
        region=series.region,
        horizon=horizon,
        origins=origins,
        data_through=series.data_through,
        n_observations=len(values),
        band_quantile=band_quantile,
        models=[m.name for m in candidates],
    )
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            try:
                return _from_cache_entry(hit)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("forecast cache entry unusable, recomputing: %s", exc)

    scores: List[bt.BacktestScore] = []
    skipped: List[Tuple[str, str]] = []
    for model in candidates:
        if len(values) < model.min_observations + horizon:
            skipped.append(
                (
                    model.name,
                    f"needs {model.min_observations} observations to fit and the series "
                    f"has {len(values)}",
                )
            )
            continue
        # Each model is graded from the shortest history IT can fit, so a seasonal
        # model is never handed a 10-month train set and a trend model is never
        # denied origins it could have been graded on.
        available = len(
            bt.rolling_origin_splits(
                len(values), horizon, origins, min_train=model.min_observations
            )
        )
        if available < MIN_BACKTEST_ORIGINS:
            skipped.append(
                (
                    model.name,
                    f"only {available} rolling origins are available at horizon {horizon} "
                    f"(at least {MIN_BACKTEST_ORIGINS} are needed to measure its error)",
                )
            )
            continue
        try:
            scores.append(
                _score_one(
                    model,
                    values,
                    horizon=horizon,
                    origins=origins,
                    min_train=model.min_observations,
                )
            )
        except Exception as exc:  # noqa: BLE001 — one model failing is not the request failing
            logger.info("forecast model %s skipped: %s", model.name, exc)
            skipped.append((model.name, str(exc) or exc.__class__.__name__))

    if not scores:
        raise ForecastRefused(
            f"{series.metric} for {series.brand or 'all brands'} has {len(values)} complete "
            f"months and no forecast model could be backtested on it at horizon {horizon}: "
            + "; ".join(f"{n}: {r}" for n, r in skipped)
        )

    champion_score = bt.select_champion(scores)
    champion = fm.get_model(champion_score.name)
    try:
        point_forecast = (
            champion.predict(values, horizon)
            if not champion.requires_worker
            else _worker_point_forecast(values, horizon)
        )
    except Exception as exc:  # noqa: BLE001
        raise ForecastRefused(
            f"the champion {champion_score.name!r} scored but could not fit the full series: {exc}"
        ) from exc

    # The champion's FINAL full-series refit is a SEPARATE call from the ones the
    # backtest graded, and only the backtest checks its own shapes. Without this, a
    # model returning the wrong length reached `zip(..., strict=True)` below and raised
    # a bare ValueError -- not the typed refusal this module's contract promises, and
    # not something a caller catching ForecastRefused would see. models.py's own design
    # is that adding a model costs one object and nothing else in the lane needs to know
    # it exists, so the length discipline cannot be assumed from the two models here.
    point_forecast = list(point_forecast)
    if len(point_forecast) != horizon or not all(math.isfinite(v) for v in point_forecast):
        raise ForecastRefused(
            f"the champion {champion_score.name!r} returned {len(point_forecast)} usable "
            f"values for a {horizon}-month horizon"
        )

    # TRx / NRx / NBRx are counts: they cannot be negative. An additive Holt-Winters
    # trend is unbounded, so a steeply declining brand (a late-lifecycle or post-LOE
    # erosion curve -- an ordinary case here, not a pathology) extrapolates straight
    # through zero. Measured 2026-09-20 on such a series: [2.98, 0.96, -1.06, -3.09,
    # -5.11, -7.13]. Flooring is done HERE, where the metric's domain is known, and
    # before the band, so lo <= point <= hi holds by construction.
    floored_idx = [i for i, v in enumerate(point_forecast) if v < 0]
    point_forecast = [max(float(v), 0.0) for v in point_forecast]
    try:
        band = bt.band_from_errors(
            champion_score, point_forecast, quantile=band_quantile, floor_at_zero=True
        )
    except bt.DegenerateBand as exc:
        raise ForecastRefused(
            f"a forecast for {series.metric} / {series.brand or 'all brands'} cannot be "
            f"given an honest prediction band: {exc}"
        ) from exc
    if series.data_through is None:  # pragma: no cover - guarded by the length check
        raise ForecastRefused("the series has no data_through, so a forecast has no start month")
    months = _forecast_months(series.data_through, horizon)
    points = tuple(
        ForecastPoint(month=m, value=float(v), lower=float(lo), upper=float(hi))
        for m, v, (lo, hi) in zip(months, point_forecast, band, strict=True)
    )
    result = KpiForecast(
        metric=series.metric,
        brand=series.brand,
        region=series.region,
        horizon=horizon,
        points=points,
        champion=champion_score.name,
        champion_description=champion.description,
        champion_mape=champion_score.monthly_mape,
        champion_total_error=champion_score.horizon_total_error,
        scores=tuple(scores),
        models_skipped=tuple(skipped),
        data_through=series.data_through,
        n_observations=len(values),
        series_query_id=series.query_id,
        band_quantile=band_quantile,
        origins_used=champion_score.n_origins,
        floored_months=tuple(months[i] for i in floored_idx),
    )
    if cache is not None:
        cache.set(cache_key, _to_cache_entry(result))
    return result


def _worker_point_forecast(values: Sequence[float], horizon: int) -> List[float]:
    from src.kpi.forecast.timesfm import predict_via_worker

    return predict_via_worker(values, horizon)


def forecast_kpi(
    metric: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    *,
    horizon: int = bt.DEFAULT_HORIZON,
    origins: int = bt.DEFAULT_ORIGINS,
    include_timesfm: bool = True,
    client: Any = None,
    as_of: Optional[date] = None,
) -> KpiForecast:
    """Read the canonical monthly series and forecast it. The tool's one entry point."""
    series = fetch_canonical_volume_series(metric, brand, region, client=client, as_of=as_of)
    return forecast_series(
        series, horizon=horizon, origins=origins, include_timesfm=include_timesfm
    )


__all__ = [
    "ForecastPoint",
    "ForecastRefused",
    "KpiForecast",
    "MAX_HORIZON",
    "ForecastCache",
    "MIN_BACKTEST_ORIGINS",
    "MIN_OBSERVATIONS",
    "forecast_kpi",
    "forecast_series",
    "month_end",
]
