"""Rolling-origin backtest, champion selection and the prediction band (#2115, Lane B).

A forecast the platform has never seen the outcome of can honestly claim only one
thing about its own accuracy: what this model did on THIS series, at THIS horizon,
on origins it could not see past. So everything the tool reports about uncertainty
is derived here, from measured misses — never from a model's internal notion of its
own confidence.

That also makes the model interface genuinely swappable. A state-space model and a
pretrained transformer disagree about what a "prediction interval" even means; they
do not disagree about how far they missed. Grading both on the SAME origins with the
SAME metric is what lets the champion be chosen honestly, and it is why the band is
built from backtest residuals rather than from statsmodels' analytic interval or
TimesFM's quantile head.

Pure and dependency-light on purpose (numpy only): the Celery worker, the API and the
tests all import it, and nothing here may need a database or a network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

#: Two full seasonal cycles. Below this a yearly shape cannot be estimated at all,
#: and the caller is told the series is too short rather than served a trend line
#: dressed up as a seasonal forecast.
MIN_TRAIN_MONTHS = 24

#: 24 monthly origins = two years of cutoffs. Measured 2026-09-20 on the live
#: Kisqali/Fabhalta/Remibrutinib TRx series: at 12 origins the ranking among the
#: Holt-Winters variants flipped between brands, at 24 it was stable, so fewer
#: origins would make the champion an artefact of the sample.
DEFAULT_ORIGINS = 24

#: Two quarters — the horizon demo 6.5 asks for.
DEFAULT_HORIZON = 6

#: The reported band's coverage. 0.8 keeps it a decision aid rather than a
#: near-vacuous 95% interval on a series whose month-to-month noise is ~15%.
DEFAULT_BAND_QUANTILE = 0.8

Predictor = Callable[[Sequence[float], int], Sequence[float]]
#: A model that is cheaper to run once over every origin than once per origin --
#: TimesFM pays a weight load and a Celery round trip per call. It returns one
#: forecast per context, positionally aligned, with ``None`` where that context
#: failed so a single bad origin cannot lose the other twenty-three.
BatchPredictor = Callable[[Sequence[Sequence[float]], int], Sequence[Optional[Sequence[float]]]]


class ModelUnscorable(RuntimeError):
    """The model produced no usable forecast on any origin, so it has no score."""


class NoChampion(RuntimeError):
    """No model was scorable, so nothing can be served."""


class DegenerateBand(RuntimeError):
    """The champion's error distribution implies an unbounded interval.

    Raised when a signed relative error reaches -100% or beyond — the model forecast
    zero or less while the actual was positive. ``actual = point / (1 + s)`` then has a
    non-positive denominator and the implied actual is unbounded above. That is a real
    measurement, not a numerical artefact: it says this model's fit is broken for this
    series. A band of "0 to infinity" conveys nothing and would invite a reader to take
    the point estimate and ignore the interval, so the forecast is refused instead.
    """


@dataclass(frozen=True)
class BacktestScore:
    """What one model actually did on this series, at this horizon."""

    name: str
    monthly_mape: float
    horizon_total_error: float
    n_origins: int
    n_failed_origins: int
    horizon: int
    #: step_pct_errors[i][k] = |forecast-actual|/|actual| * 100 at step i, origin k.
    step_pct_errors: Tuple[Tuple[float, ...], ...]
    #: Same, signed as (forecast-actual)/actual * 100 — positive means the model
    #: ran HIGH, so the actual landed below it. The band needs the direction.
    signed_step_pct_errors: Tuple[Tuple[float, ...], ...]


def rolling_origin_splits(
    n: int,
    horizon: int,
    origins: int,
    min_train: int = MIN_TRAIN_MONTHS,
) -> List[Tuple[range, range]]:
    """The last ``origins`` causal train/test cutoffs a series of length ``n`` supports.

    The newest cutoffs are used, not the oldest: the forecast extends the CURRENT
    regime, so the models must be graded on it. Each split trains on ``y[:t]`` and is
    graded on ``y[t:t+horizon]``, so no split can see a value it is asked to predict.
    A short series yields fewer origins — never a split below ``min_train``.
    """
    if horizon < 1 or origins < 1:
        return []
    last_cutoff = n - horizon
    first_cutoff = max(min_train, last_cutoff - origins + 1)
    if last_cutoff < first_cutoff:
        return []
    return [(range(0, t), range(t, t + horizon)) for t in range(first_cutoff, last_cutoff + 1)]


def _relative_error(forecast: float, actual: float) -> float:
    """(forecast-actual)/actual as a percentage, defined when the actual is 0."""
    if actual == 0.0:
        return 0.0 if forecast == 0.0 else 100.0 * math.copysign(1.0, forecast)
    return (forecast - actual) / abs(actual) * 100.0


def _as_batch_predictor(predict: Predictor) -> BatchPredictor:
    """Run a per-origin model over a batch, isolating each origin's failure."""

    def run(contexts: Sequence[Sequence[float]], horizon: int) -> List[Optional[Sequence[float]]]:
        out: List[Optional[Sequence[float]]] = []
        for context in contexts:
            try:
                out.append(predict(context, horizon))
            except Exception:
                out.append(None)
        return out

    return run


def score_model(
    predict: Optional[Predictor] = None,
    y: Optional[Sequence[float]] = None,
    *,
    horizon: int,
    origins: int,
    name: str,
    min_train: int = MIN_TRAIN_MONTHS,
    predict_many: Optional[BatchPredictor] = None,
) -> BacktestScore:
    """Grade one model over rolling origins. Raises ``ModelUnscorable`` if none worked.

    Pass ``predict`` for a per-origin model or ``predict_many`` for one that is cheaper
    batched; both take the SAME splits, so a batched model is never graded on an easier
    question than an in-process one. Every model the tool compares goes through here.

    An origin that raises, returns the wrong length, or returns a non-finite value is a
    FAILED origin: counted, excluded, never silently scored as zero error. A statsmodels
    fit converging to nan is the realistic case, and a nan that survived into the score
    would make the broken model the champion.
    """
    if (predict is None) == (predict_many is None):
        raise TypeError("score_model takes exactly one of predict= or predict_many=")
    if y is None:
        raise TypeError("score_model needs the series y")
    run = predict_many if predict_many is not None else _as_batch_predictor(predict)

    values = np.asarray(list(y), dtype=float)
    splits = rolling_origin_splits(len(values), horizon, origins, min_train=min_train)
    signed: List[List[float]] = [[] for _ in range(horizon)]
    totals: List[float] = []
    failed = 0
    if splits:
        contexts = [values[train.start : train.stop] for train, _ in splits]
        results = list(run(contexts, horizon))
        if len(results) != len(splits):
            raise ModelUnscorable(
                f"{name!r} returned {len(results)} forecasts for {len(splits)} origins"
            )
        for (_train, test), raw in zip(splits, results, strict=True):
            actual = values[test.start : test.stop]
            if raw is None:
                failed += 1
                continue
            try:
                forecast = np.asarray(list(raw), dtype=float)
            except (TypeError, ValueError):
                failed += 1
                continue
            if forecast.shape != actual.shape or not np.all(np.isfinite(forecast)):
                failed += 1
                continue
            for i in range(horizon):
                signed[i].append(_relative_error(float(forecast[i]), float(actual[i])))
            totals.append(_relative_error(float(forecast.sum()), float(actual.sum())))
    if not totals:
        raise ModelUnscorable(
            f"{name!r} produced no usable forecast on any of {len(splits)} origins"
        )
    absolute = [[abs(v) for v in step] for step in signed]
    return BacktestScore(
        name=name,
        monthly_mape=float(np.mean([v for step in absolute for v in step])),
        horizon_total_error=float(np.mean([abs(v) for v in totals])),
        n_origins=len(totals),
        n_failed_origins=failed,
        horizon=horizon,
        step_pct_errors=tuple(tuple(step) for step in absolute),
        signed_step_pct_errors=tuple(tuple(step) for step in signed),
    )


def select_champion(scores: Sequence[BacktestScore]) -> BacktestScore:
    """Lowest monthly MAPE wins; ties break on name so the choice is reproducible.

    MAPE, not the horizon total: opposite-signed monthly errors cancel in a total, so
    a model can win the two-quarter sum while tracking the path badly — and 6.5 asks
    for the path as well as the sum. Both numbers are reported either way.
    """
    if not scores:
        raise NoChampion("no model was scorable on this series")
    return min(scores, key=lambda s: (s.monthly_mape, s.name))


def band_from_errors(
    score: BacktestScore,
    point_forecast: Sequence[float],
    *,
    quantile: float = DEFAULT_BAND_QUANTILE,
    floor_at_zero: bool = True,
) -> List[Tuple[float, float]]:
    """The champion's own measured miss distribution, as an interval on the actual.

    At step i the signed errors are s = (forecast-actual)/actual, so an actual implied
    by a miss of s is ``point/(1+s)``. Taking the empirical (1-q)/2 and (1+q)/2
    quantiles of s therefore gives a q-coverage interval on the actual, asymmetric
    exactly when the model's misses were. The interval is then widened if needed to
    contain the point forecast — a band that excluded the number the answer shows
    would be incoherent to present — and floored at zero for a volume.
    """
    lower_q = (1.0 - quantile) / 2.0
    upper_q = 1.0 - lower_q
    out: List[Tuple[float, float]] = []
    for i, point in enumerate(point_forecast):
        errors = score.signed_step_pct_errors[i] if i < len(score.signed_step_pct_errors) else ()
        if not errors:
            out.append((float(point), float(point)))
            continue
        arr = np.asarray(errors, dtype=float) / 100.0
        lo_err = float(np.quantile(arr, lower_q))
        hi_err = float(np.quantile(arr, upper_q))
        if (1.0 + lo_err) <= 0 or (1.0 + hi_err) <= 0:
            raise DegenerateBand(
                f"{score.name!r} forecast zero or less against a positive actual at "
                f"horizon step {i + 1}, so the interval implied by its own errors is "
                "unbounded above"
            )
        lo = point / (1.0 + hi_err)
        hi = point / (1.0 + lo_err)
        lo, hi = min(lo, hi), max(lo, hi)
        lo, hi = min(lo, point), max(hi, point)
        if floor_at_zero:
            # BOTH ends, and the point with them. Flooring only ``lo`` produced a band
            # with lower ABOVE upper whenever the point forecast itself was negative:
            # measured 2026-09-20 on a steeply declining series, step 3 came back
            # ``point=-1.06, lower=0.00, upper=-1.06``. A volume's band cannot go below
            # zero, but neither can its upper edge sit below its lower one, and the
            # caller is responsible for not handing a negative point to a non-negative
            # quantity -- this only guarantees the invariant holds whatever it is given.
            lo, hi = max(lo, 0.0), max(hi, 0.0)
            lo, hi = min(lo, hi), max(lo, hi)
        # Nothing non-finite may leave this function. A NaN or inf edge serialises as a
        # bare `Infinity`/`NaN` token that only Python's permissive encoder accepts:
        # `json.dumps(..., allow_nan=False)` -- which FastAPI and every orjson-backed
        # path use -- raises on it, so the failure would surface as a 500 at the chat
        # boundary rather than as anything this module could explain.
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise DegenerateBand(
                f"{score.name!r} produced a non-finite band at horizon step {i + 1}"
            )
        out.append((float(lo), float(hi)))
    return out
