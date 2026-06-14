"""WalkForwardRunner — expanding-origin, strictly out-of-sample backtest.

For each month ``M`` across the journey timeline the runner trains on rows with
``journey_start_date < M`` (expanding window) and evaluates STRICTLY out of
sample on the rows that fall *in* month ``M``.  This is the leakage-free, finer-
grained counterpart to the static ``data_split`` column — the static split is
chronological too, but a single train/holdout cut yields one number, while
walk-forward yields the real performance TREND that the Time-Series page plots.
The runner therefore IGNORES ``data_split`` entirely and re-windows by time.

Output: a list of ``(month: datetime, metrics: dict, n_eval: int)`` tuples,
which is exactly the ``points`` contract consumed by
:meth:`MetricRecorder.record_run`.

Leakage discipline (the whole point)
------------------------------------
Every training row's ``journey_start_date`` is ``< first-day-of(M)`` for the
evaluated month ``M``.  Equality (a row dated within ``M``) is NOT allowed into
training, so a month is never scored on a model that has seen its own rows.

Window modes
------------
* ``"expanding"`` (default): train on ALL rows strictly before ``M``.  Chosen by
  the Task-8 EXPERIMENT (see
  ``docs/superpowers/plans/experiments/2026-06-14-walkforward-window.md``).
* ``"rolling"``: train on rows in ``[M - rolling_months, M)`` only.  Available
  via the constructor for callers who want a fixed-width window; the experiment
  recorded the measured trade-off.

Unit-testability seam
---------------------
The fit/predict pair is injectable:

* ``fit_fn(train_df) -> (model, feature_builder)``
* ``predict_fn(model, feature_builder, eval_df) -> y_score``

When not injected the runner wires the production default internally
(``FeatureBuilder`` fit/transform + ``LogisticRegression(class_weight=
'balanced', max_iter=1000)`` + ``scorer.score``) behind these same seams, so the
unit test can substitute a trivial deterministic model while the real run uses
the genuine estimator.

Guards
------
* ``min_train_n``: skip a month whose training set has fewer rows than this.
* ``n_min``: skip a month whose eval set has fewer rows than this.

Skipped months are LOGGED (and recorded in :attr:`skipped`), never emitted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
from src.mlops.gold_standard_eval.scorer import score

logger = logging.getLogger(__name__)

# Type aliases for the injectable seam.
FitFn = Callable[[pd.DataFrame], tuple[Any, Any]]
PredictFn = Callable[[Any, Any, pd.DataFrame], "np.ndarray"]

_DATE_COL = "journey_start_date"


@dataclass(frozen=True)
class SkippedMonth:
    """A month that did not qualify, with the counts + reason that triggered it."""

    month: datetime
    reason: str  # human-readable: mentions "train" or "eval"
    train_n: int
    n_eval: int


class WalkForwardRunner:
    """Expanding- (or rolling-) origin walk-forward backtest over a raw frame.

    Parameters
    ----------
    spec:
        ``CohortSpec`` supplying the label column (used by the internal default
        fit/predict; an injected ``fit_fn``/``predict_fn`` may ignore it).
    fit_fn:
        Optional ``train_df -> (model, feature_builder)``.  Defaults to the
        production FeatureBuilder + balanced LogisticRegression wiring.
    predict_fn:
        Optional ``(model, feature_builder, eval_df) -> y_score`` returning
        positive-class scores aligned to ``eval_df``'s rows.  Defaults to the
        FeatureBuilder ``transform`` + ``model.predict_proba`` wiring.
    min_train_n:
        Skip a month with fewer than this many training rows.
    n_min:
        Skip a month with fewer than this many eval rows.
    window_mode:
        ``"expanding"`` (default; experiment-selected) or ``"rolling"``.
    rolling_months:
        Window width (in months) when ``window_mode == "rolling"``.
    on_month:
        Optional callback invoked with the eval month (a ``pd.Timestamp``) right
        BEFORE that month's fit.  Lets tests key captured training data by the
        month under evaluation; unused in production.
    """

    def __init__(
        self,
        spec: object,
        *,
        fit_fn: FitFn | None = None,
        predict_fn: PredictFn | None = None,
        min_train_n: int = 50,
        n_min: int = 20,
        window_mode: str = "expanding",
        rolling_months: int = 3,
        on_month: Callable[[pd.Timestamp], None] | None = None,
    ) -> None:
        if window_mode not in ("expanding", "rolling"):
            raise ValueError(f"window_mode must be 'expanding' or 'rolling', got {window_mode!r}")
        if window_mode == "rolling" and rolling_months <= 0:
            raise ValueError("rolling_months must be > 0 for rolling window_mode")

        self.spec = spec
        self.fit_fn: FitFn = fit_fn if fit_fn is not None else self._default_fit
        self.predict_fn: PredictFn = predict_fn if predict_fn is not None else self._default_predict
        self.min_train_n = min_train_n
        self.n_min = n_min
        self.window_mode = window_mode
        self.rolling_months = rolling_months
        self._on_month = on_month

        # Populated by run().
        self.skipped: list[SkippedMonth] = []

    # ------------------------------------------------------------------ #
    # Production default fit/predict (wired behind the injectable seam).   #
    # ------------------------------------------------------------------ #
    def _default_fit(self, train_df: pd.DataFrame) -> tuple[Any, FeatureBuilder]:
        """FIT: encode the train frame and fit a balanced LogisticRegression.

        Imported lazily so the unit test (which injects a trivial model) never
        pays the sklearn import cost and the module stays import-light.
        """
        from sklearn.linear_model import LogisticRegression

        fb = FeatureBuilder(self.spec)
        x_train, y_train = fb.build_from_frame(train_df)
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(x_train.to_numpy(dtype=float), y_train.to_numpy())
        return model, fb

    def _default_predict(
        self, model: Any, fb: FeatureBuilder, eval_df: pd.DataFrame
    ) -> "np.ndarray":
        """APPLY: align the eval frame to fitted columns, return P(class=1)."""
        x_eval = fb.transform(eval_df)
        proba = model.predict_proba(x_eval.to_numpy(dtype=float))
        # Positive-class column; robust to a degenerate single-class fit.
        if proba.shape[1] == 1:
            # Only one class was present at fit; predict_proba has a single col.
            only_class = int(model.classes_[0])
            return np.full(len(eval_df), float(only_class))
        pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
        return proba[:, pos_idx]

    # ------------------------------------------------------------------ #
    # The walk-forward loop.                                              #
    # ------------------------------------------------------------------ #
    def run(self, frame: pd.DataFrame) -> list[tuple[datetime, dict[str, float], int]]:
        """Produce the per-month out-of-sample metric trend.

        Parameters
        ----------
        frame:
            Raw rows (one per patient) with at least ``journey_start_date`` and
            ``spec.label_column``.  ``data_split`` is IGNORED — walk-forward
            re-windows strictly by time.

        Returns
        -------
        list of (month, metrics, n_eval):
            One tuple per QUALIFYING month, ascending by month.  ``month`` is the
            timezone-aware first instant of the calendar month;  ``metrics`` is
            the :func:`scorer.score` dict;  ``n_eval`` is the eval row count.
            Skipped months are recorded in :attr:`skipped`, not returned.
        """
        self.skipped = []
        if frame.empty or _DATE_COL not in frame.columns:
            logger.warning(
                "WalkForwardRunner.run: empty frame or missing %s column; returning no points.",
                _DATE_COL,
            )
            return []

        df = frame.copy()
        # Normalize the date column to tz-aware UTC datetimes, then derive the
        # month period from a tz-naive UTC view (to_period drops tz anyway; doing
        # the strip explicitly keeps the conversion lossless and silences the
        # "dropping timezone information" UserWarning in real-run logs).
        dates = pd.to_datetime(df[_DATE_COL], utc=True)
        df[_DATE_COL] = dates
        df["_period"] = dates.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")

        results: list[tuple[datetime, dict[str, float], int]] = []
        ordered_periods = sorted(df["_period"].dropna().unique())

        for period in ordered_periods:
            # First instant of the month, tz-aware — the emitted "month" key and
            # the strict-OOS cutoff.
            month_start = period.to_timestamp(how="start").tz_localize("UTC")
            eval_mask = df["_period"] == period
            eval_df = df.loc[eval_mask].drop(columns=["_period"])
            n_eval = int(len(eval_df))

            # Training window: strictly BEFORE the eval month's first instant.
            if self.window_mode == "rolling":
                window_start = (
                    (period - self.rolling_months).to_timestamp(how="start").tz_localize("UTC")
                )
                train_mask = (df[_DATE_COL] >= window_start) & (df[_DATE_COL] < month_start)
            else:  # expanding
                train_mask = df[_DATE_COL] < month_start

            train_df = df.loc[train_mask].drop(columns=["_period"])
            train_n = int(len(train_df))

            # Notify any observer of the month under evaluation BEFORE fitting.
            if self._on_month is not None:
                self._on_month(month_start)

            # --- Guards: log + record the skip, do NOT emit. ---------------- #
            if train_n < self.min_train_n:
                self._skip(
                    month_start,
                    f"train guard: train_n={train_n} < min_train_n={self.min_train_n}",
                    train_n,
                    n_eval,
                )
                continue
            if n_eval < self.n_min:
                self._skip(
                    month_start,
                    f"eval guard: n_eval={n_eval} < n_min={self.n_min}",
                    train_n,
                    n_eval,
                )
                continue

            # --- Fit on the past, score strictly out-of-sample. ------------- #
            try:
                model, fb = self.fit_fn(train_df)
                y_score = np.asarray(self.predict_fn(model, fb, eval_df), dtype=float)
                y_true = eval_df[self.spec.label_column].astype(int).to_numpy()
            except Exception as exc:  # degenerate month (e.g. single-class) -> skip, logged.
                self._skip(
                    month_start,
                    f"fit/predict failed: {type(exc).__name__}: {exc}",
                    train_n,
                    n_eval,
                )
                continue

            # A single-class eval month makes AUC undefined — skip, logged.
            if len(np.unique(y_true)) < 2:
                self._skip(
                    month_start,
                    f"degenerate eval: only one class present (n_eval={n_eval})",
                    train_n,
                    n_eval,
                )
                continue

            metrics = score(y_true, y_score)
            results.append((month_start.to_pydatetime(), metrics, n_eval))
            logger.info(
                "WalkForwardRunner: month=%s train_n=%d n_eval=%d auc_roc=%.4f",
                month_start.date(),
                train_n,
                n_eval,
                metrics.get("auc_roc", float("nan")),
            )

        logger.info(
            "WalkForwardRunner.run: emitted %d point(s), skipped %d month(s) "
            "(mode=%s, min_train_n=%d, n_min=%d).",
            len(results),
            len(self.skipped),
            self.window_mode,
            self.min_train_n,
            self.n_min,
        )
        return results

    def _skip(self, month: pd.Timestamp, reason: str, train_n: int, n_eval: int) -> None:
        self.skipped.append(
            SkippedMonth(
                month=month.to_pydatetime(),
                reason=reason,
                train_n=train_n,
                n_eval=n_eval,
            )
        )
        logger.info("WalkForwardRunner: SKIP month=%s (%s)", month.date(), reason)


def run_walk_forward(
    spec: object,
    frame: pd.DataFrame,
    *,
    min_train_n: int = 50,
    n_min: int = 20,
    window_mode: str = "expanding",
    rolling_months: int = 3,
) -> tuple[list[tuple[datetime, dict[str, float], int]], list[SkippedMonth]]:
    """Convenience: build a runner with the production default fit/predict and run.

    Returns ``(points, skipped)`` so callers (and the experiment script) get both
    the emitted trend and the skip log in one call.
    """
    runner = WalkForwardRunner(
        spec,
        min_train_n=min_train_n,
        n_min=n_min,
        window_mode=window_mode,
        rolling_months=rolling_months,
    )
    points = runner.run(frame)
    return points, runner.skipped


__all__ = ["WalkForwardRunner", "SkippedMonth", "run_walk_forward", "FitFn", "PredictFn"]
