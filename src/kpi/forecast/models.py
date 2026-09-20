"""The swappable forecast-model interface and its members (#2115, Lane B).

Owner decision 2026-09-15: Holt-Winters (statsmodels, in-process) ALWAYS runs;
TimesFM 2.5 runs on a dedicated ``forecast`` Celery worker. Per request every
available model is backtested on the SAME origins and the lower-error one is served.

So a model here is deliberately thin: a name, how much history it needs, and a
``predict(y, horizon)`` that returns ``horizon`` numbers or raises. It owns no notion
of its own uncertainty and does no scoring — ``backtest.py`` grades all of them on
equal terms. Adding a model is adding one object to ``_MODELS``; nothing else in the
lane needs to know it exists.

Why three Holt-Winters members rather than one: measured on the live PROD series
2026-09-20, at 24 origins seasonal-additive wins Kisqali (6.85% monthly MAPE) and
Fabhalta (7.06%) while seasonal-multiplicative wins Remibrutinib (7.06%), and at 12
origins the ranking flips again. There is no single right Holt-Winters form for this
data, which is exactly the case a per-request backtest is for. ``holt_winters_trend``
is not a contender so much as the member that still fits a series too short for a
seasonal cycle.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

#: Two full seasonal cycles — below this a yearly shape is not estimable.
SEASONAL_MIN_OBSERVATIONS = 24
#: A level-plus-trend fit needs far less; this is the floor for any forecast at all.
TREND_MIN_OBSERVATIONS = 8
SEASONAL_PERIODS = 12


class ForecastModelError(RuntimeError):
    """The model cannot honestly fit what it was handed."""


class SeriesTooShort(ForecastModelError):
    """Fewer observations than this model needs to estimate its own structure."""


class SeriesNotPositive(ForecastModelError):
    """A multiplicative season is undefined at or below zero."""


@dataclass(frozen=True)
class ForecastModel:
    """One interchangeable forecaster.

    ``requires_worker`` is the only structural difference between members: an
    in-process model is always available, a worker-backed one may not be, and the
    service must be able to tell those apart without importing torch to find out.
    """

    name: str
    description: str
    min_observations: int
    _predict: Callable[[Sequence[float], int], Sequence[float]] = field(repr=False)
    requires_worker: bool = False

    def predict(self, y: Sequence[float], horizon: int) -> List[float]:
        values = np.asarray(list(y), dtype=float)
        if len(values) < self.min_observations:
            raise SeriesTooShort(
                f"{self.name} needs at least {self.min_observations} observations, got {len(values)}"
            )
        if not np.all(np.isfinite(values)):
            raise ForecastModelError(f"{self.name}: the series carries a non-finite value")
        # A list, not the ndarray: `_predict` is typed over Sequence[float] so that a
        # model added later is not obliged to accept numpy, and the members convert
        # back themselves where they need an array.
        return [float(v) for v in self._predict(values.tolist(), horizon)]


def _fit_holt_winters(
    y: Sequence[float],
    horizon: int,
    *,
    seasonal: str | None,
) -> Sequence[float]:
    # Imported lazily: backtest.py and the tool schema must import without statsmodels
    # being resolvable, so a missing optional dep degrades one MODEL rather than the
    # whole forecasting surface.
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    kwargs: Dict[str, object] = {"trend": "add", "initialization_method": "estimated"}
    if seasonal is not None:
        kwargs["seasonal"] = seasonal
        kwargs["seasonal_periods"] = SEASONAL_PERIODS
    values = np.asarray(y, dtype=float)
    with warnings.catch_warnings():
        # statsmodels warns about convergence on some origins; a genuinely bad fit
        # shows up as a non-finite forecast, which score_model counts as a FAILED
        # origin. Warning noise is not the signal, so it is not amplified here.
        warnings.simplefilter("ignore")
        fitted = ExponentialSmoothing(values, **kwargs).fit(optimized=True)
        forecast: List[float] = [float(v) for v in fitted.forecast(horizon)]
        return forecast


def _holt_winters_seasonal_add(y: Sequence[float], horizon: int) -> Sequence[float]:
    return _fit_holt_winters(y, horizon, seasonal="add")


def _holt_winters_seasonal_mul(y: Sequence[float], horizon: int) -> Sequence[float]:
    if not np.all(np.asarray(y, dtype=float) > 0):
        raise SeriesNotPositive(
            "holt_winters_seasonal_mul needs a strictly positive series; "
            "a multiplicative season is undefined at zero"
        )
    return _fit_holt_winters(y, horizon, seasonal="mul")


def _holt_winters_trend(y: Sequence[float], horizon: int) -> Sequence[float]:
    return _fit_holt_winters(y, horizon, seasonal=None)


def _timesfm_predict(y: Sequence[float], horizon: int) -> Sequence[float]:
    # Dispatch lives in timesfm.py so that importing the model catalogue never pulls
    # in Celery, torch or transformers. See that module for why the call is a task.
    from src.kpi.forecast.timesfm import predict_via_worker

    return predict_via_worker(y, horizon)


_MODELS: Tuple[ForecastModel, ...] = (
    ForecastModel(
        name="holt_winters_seasonal_add",
        description="Holt-Winters, additive trend and additive 12-month season (statsmodels)",
        min_observations=SEASONAL_MIN_OBSERVATIONS,
        _predict=_holt_winters_seasonal_add,
    ),
    ForecastModel(
        name="holt_winters_seasonal_mul",
        description="Holt-Winters, additive trend and multiplicative 12-month season (statsmodels)",
        min_observations=SEASONAL_MIN_OBSERVATIONS,
        _predict=_holt_winters_seasonal_mul,
    ),
    ForecastModel(
        name="holt_winters_trend",
        description="Holt-Winters, additive trend, no season (statsmodels)",
        min_observations=TREND_MIN_OBSERVATIONS,
        _predict=_holt_winters_trend,
    ),
    ForecastModel(
        name="timesfm_2_5",
        description=(
            "TimesFM 2.5 200M, Google's pretrained time-series foundation model "
            "(google/timesfm-2.5-200m-transformers, Apache-2.0), on the forecast worker"
        ),
        min_observations=TREND_MIN_OBSERVATIONS,
        _predict=_timesfm_predict,
        requires_worker=True,
    ),
)

_BY_NAME: Dict[str, ForecastModel] = {m.name: m for m in _MODELS}


def all_models() -> Tuple[ForecastModel, ...]:
    return _MODELS


def in_process_models() -> Tuple[ForecastModel, ...]:
    """The models that need no external service — the ones that always run."""
    return tuple(m for m in _MODELS if not m.requires_worker)


def worker_models() -> Tuple[ForecastModel, ...]:
    return tuple(m for m in _MODELS if m.requires_worker)


def get_model(name: str) -> ForecastModel:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ForecastModelError(f"unknown forecast model {name!r}") from None
