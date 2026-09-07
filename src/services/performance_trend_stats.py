"""Sampling-aware trend classification for model-performance folds.

Why (2026-09-07)
----------------
The /model-performance "Performance trend" label used to be a single ±5%
relative-change rule: newest walk-forward fold vs the mean of the older folds
in the window, with no notion of sample size. A walk-forward fold is ONE
calendar month of rows (≈80–230 patients; ≈130 HCPs), so its AUC carries a
Hanley-McNeil standard error of 0.04–0.07 — larger than the 5% threshold
(≈0.04). Simulated under a perfectly stationary model the old rule reported
something other than "stable" 38% of the time at n=130 and 56% at n=54, and
on 2026-09-07 it flagged four gold-standard models "degrading" on single
folds that were 0.8–2.2 standard errors below baseline with no slope.

What this module does
---------------------
Pure functions (no I/O) that the :class:`PerformanceTracker` calls:

* :func:`metric_standard_error` — analytic SE of one fold's metric from its
  sample size and positive rate (Hanley-McNeil for AUC, binomial for the
  rate-like metrics).
* :func:`slope_t_stat` — OLS slope t-statistic over the window, the signal a
  level test cannot see (a steady 0.01/month slide).
* :func:`assess_trend` — the classifier: ``degrading`` / ``improving`` only
  when the change is BOTH statistically distinguishable from sampling noise
  (level z beyond ``z_threshold`` OR slope t beyond ``slope_t_threshold``)
  AND material (the historical ±5% floor, kept as a materiality gate).
  Rows without a sample size cannot be tested and fall back to the legacy
  relative rule, labelled ``basis="legacy_relative"`` so the caller can see
  which rule produced the label.

Thresholds: 2.5 standard errors (two-sided p ≈ 0.012). The page trends 12
models × 5 metrics = 60 series every week; at ±2.0 the expected false alarms
per weekly run were ≈1.4, at ±2.5 ≈0.4. Shewhart's ±3 would miss a genuine
0.10 AUC drop on a normal month; 2.5 catches it (z ≈ −2.7 at n=230).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Metrics that are rates on [0, 1] with a binomial-style sampling variance.
# ``auc_roc`` is handled separately (Hanley-McNeil) but is also a [0, 1] value.
_POSITIVE_CLASS_METRICS = frozenset({"precision", "recall", "f1"})
_RATE_METRICS = frozenset({"accuracy", "auc_roc", "auc_pr", "pr_auc"}) | _POSITIVE_CLASS_METRICS


@dataclass(frozen=True)
class TrendPoint:
    """One fold of the trend series (a walk-forward month or a daily window)."""

    value: float
    sample_size: Optional[int] = None
    positive_rate: Optional[float] = None


@dataclass(frozen=True)
class TrendAssessment:
    """Classifier output. ``basis`` says WHICH rule produced ``trend``:

    * ``level`` — newest fold vs baseline beyond ``z_threshold`` SEs and material
    * ``slope`` — OLS slope over the window beyond ``slope_t_threshold`` and material
    * ``within_noise`` — change is inside the sampling-error band
    * ``immaterial`` — statistically distinguishable but below the materiality floor
    * ``legacy_relative`` — no sample size on the newest fold; ±5% relative rule
    * ``insufficient_history`` — one fold only; nothing to compare against
    * ``no_data`` — empty window
    """

    trend: str
    change_percent: float
    is_significant: bool
    basis: str
    reason: str
    current_value: float
    baseline_value: float
    n_points: int
    sample_size: Optional[int] = None
    standard_error: Optional[float] = None
    z_score: Optional[float] = None
    slope_t_stat: Optional[float] = None


# ---------------------------------------------------------------------------
# Standard errors
# ---------------------------------------------------------------------------


def _shrink(value: float, n: int) -> float:
    """Agresti-Coull-style shrinkage toward 0.5 so a boundary value (0 or 1)
    keeps a non-zero variance instead of collapsing to SE=0 → z=∞."""
    v = min(max(float(value), 0.0), 1.0)
    return (v * n + 1.0) / (n + 2.0)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUC from the class counts."""
    n_pos = max(int(n_pos), 1)
    n_neg = max(int(n_neg), 1)
    a = _shrink(auc, n_pos + n_neg)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (
        n_pos * n_neg
    )
    return math.sqrt(max(var, 0.0))


def proportion_se(value: float, n: int) -> float:
    """Binomial standard error of a rate-like metric on ``n`` trials."""
    n = max(int(n), 1)
    v = _shrink(value, n)
    return math.sqrt(v * (1.0 - v) / n)


def metric_standard_error(
    metric_name: str,
    value: float,
    sample_size: Optional[int],
    positive_rate: Optional[float],
) -> Optional[float]:
    """Analytic SE of ``value`` for one fold, or None when it cannot be derived.

    * ``auc_roc`` with a known positive rate → Hanley-McNeil on the implied class
      counts. Without the positive rate the binomial form is used — an
      OPTIMISTIC (smaller) approximation, acceptable only until the recorder
      has populated ``positive_rate`` on the fold rows.
    * ``precision`` / ``recall`` / ``f1`` → binomial on the positive count when
      the positive rate is known (recall's exact denominator; precision's and
      F1's are approximations of the same order), else on ``n``.
    * other rate metrics → binomial on ``n``.
    * anything that is not a rate on [0, 1] (calibration slope, Brier on a
      different scale, unknown names) → None; the caller falls back to the
      legacy rule rather than inventing an interval.
    """
    if sample_size is None or sample_size <= 0:
        return None
    if value is None or not math.isfinite(value):
        return None
    if metric_name not in _RATE_METRICS or not (0.0 <= value <= 1.0):
        return None
    p = float(positive_rate) if positive_rate is not None else None
    if p is not None and not (0.0 < p < 1.0):
        p = None
    if metric_name == "auc_roc":
        if p is not None:
            n_pos = max(1, int(round(sample_size * p)))
            n_neg = max(1, sample_size - n_pos)
            return hanley_mcneil_se(value, n_pos, n_neg)
        return proportion_se(value, sample_size)
    if metric_name in _POSITIVE_CLASS_METRICS and p is not None:
        n_eff = max(1, int(round(sample_size * p)))
        return proportion_se(value, n_eff)
    return proportion_se(value, sample_size)


# ---------------------------------------------------------------------------
# Slope test
# ---------------------------------------------------------------------------


def slope_t_stat(values_oldest_first: Sequence[float]) -> Optional[tuple[float, float]]:
    """OLS of value on fold index. Returns ``(t_statistic, fitted_change)`` where
    ``fitted_change`` is the slope × window span (the total change the fitted
    line implies), or None with fewer than 4 points / a degenerate design.
    A perfect linear fit (zero residual) returns a signed infinity so a
    positive control is never masked by a 0/0."""
    y = np.asarray([float(v) for v in values_oldest_first], dtype=float)
    k = int(y.size)
    if k < 4 or not np.all(np.isfinite(y)):
        return None
    x = np.arange(k, dtype=float)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0.0:
        return None
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    resid = y - (ym + slope * (x - xm))
    s2 = float((resid**2).sum()) / (k - 2)
    se_slope = math.sqrt(s2 / sxx) if s2 > 0.0 else 0.0
    if se_slope == 0.0:
        t = 0.0 if slope == 0.0 else math.copysign(float("inf"), slope)
    else:
        t = slope / se_slope
    return t, slope * (k - 1)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _finite(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _opt_float(v: object) -> Optional[float]:
    """float for a finite real number, else None (test doubles, None, NaN)."""
    return float(v) if _finite(v) else None  # type: ignore[arg-type]


def _opt_n(v: object) -> Optional[int]:
    """A positive sample size, else None."""
    f = _opt_float(v)
    return int(f) if f is not None and f > 0 else None


def assess_trend(
    points_newest_first: Sequence[TrendPoint],
    metric_name: str,
    *,
    current_window: int = 1,
    min_change_percent: float = 5.0,
    z_threshold: float = 2.5,
    slope_t_threshold: float = 2.5,
    legacy_significance_percent: float = 10.0,
    min_points_for_slope: int = 6,
    min_points_for_dispersion: int = 6,
) -> TrendAssessment:
    """Classify the newest fold against the rest of the window.

    ``points_newest_first`` is the series exactly as the repository returns it
    (``measured_at`` descending). The first ``current_window`` points are the
    "current" side (the tracker uses 1); the remainder is the baseline.
    """
    pts = [p for p in points_newest_first if _finite(p.value)]
    if not pts:
        return TrendAssessment(
            trend="unknown",
            change_percent=0.0,
            is_significant=False,
            basis="no_data",
            reason="no metric history in the window",
            current_value=0.0,
            baseline_value=0.0,
            n_points=0,
        )

    current = pts[0]
    current_value = float(current.value)
    baseline_pts = pts[max(int(current_window), 1) :]
    n_points = len(pts)
    sample_size = _opt_n(current.sample_size)

    if not baseline_pts:
        return TrendAssessment(
            trend="stable",
            change_percent=0.0,
            is_significant=False,
            basis="insufficient_history",
            reason="only one fold in the window; nothing to compare against yet",
            current_value=current_value,
            baseline_value=current_value,
            n_points=n_points,
            sample_size=sample_size,
        )

    baseline_values = np.asarray([float(p.value) for p in baseline_pts], dtype=float)
    baseline_value = float(baseline_values.mean())
    k = int(baseline_values.size)
    change_percent = (
        float((current_value - baseline_value) / baseline_value * 100.0)
        if baseline_value > 0.0
        else 0.0
    )

    # ---- Legacy fallback: no sample size → no standard error → ±5% rule. ----
    se_current = metric_standard_error(
        metric_name, current_value, sample_size, _opt_float(current.positive_rate)
    )
    if se_current is None:
        if change_percent > min_change_percent:
            trend = "improving"
        elif change_percent < -min_change_percent:
            trend = "degrading"
        else:
            trend = "stable"
        return TrendAssessment(
            trend=trend,
            change_percent=change_percent,
            is_significant=abs(change_percent) > legacy_significance_percent,
            basis="legacy_relative",
            reason=(
                "sample size unavailable for the newest fold; classified by the "
                f"relative-change rule (±{min_change_percent:g}%)"
            ),
            current_value=current_value,
            baseline_value=baseline_value,
            n_points=n_points,
            sample_size=sample_size,
        )

    # ---- Level test: newest fold vs the baseline mean on the SE scale. --------
    baseline_ses = [
        metric_standard_error(
            metric_name, float(p.value), _opt_n(p.sample_size), _opt_float(p.positive_rate)
        )
        for p in baseline_pts
    ]
    known = [s for s in baseline_ses if s is not None]
    se_mean = math.sqrt(sum(s * s for s in known) / len(known) / k) if known else 0.0
    noise = math.sqrt(se_current * se_current + se_mean * se_mean)
    # With enough history the empirical fold-to-fold spread is a second noise
    # estimate that also captures month-composition variation; take the
    # larger of the two so a normally-scattered fold is not called a drop.
    if k >= min_points_for_dispersion:
        sd = float(baseline_values.std(ddof=1))
        noise = max(noise, sd * math.sqrt(1.0 + 1.0 / k))
    z = (current_value - baseline_value) / noise if noise > 0.0 else 0.0
    level_reject = abs(z) > z_threshold
    level_material = abs(change_percent) >= min_change_percent

    # ---- Slope test over the whole window (oldest → newest). -----------------
    slope_t: Optional[float] = None
    slope_reject = False
    slope_material = False
    fitted_change = 0.0
    if n_points >= min_points_for_slope:
        st = slope_t_stat([float(p.value) for p in reversed(pts)])
        if st is not None:
            slope_t, fitted_change = st
            slope_reject = abs(slope_t) > slope_t_threshold
            slope_material = (
                baseline_value > 0.0
                and abs(fitted_change) / baseline_value * 100.0 >= min_change_percent
            )

    # ---- Decide. Degrading takes precedence over improving (monitoring errs
    # toward surfacing drops); materiality gates both directions. --------------
    level_down = level_reject and level_material and z < 0.0
    level_up = level_reject and level_material and z > 0.0
    slope_down = slope_reject and slope_material and (slope_t or 0.0) < 0.0
    slope_up = slope_reject and slope_material and (slope_t or 0.0) > 0.0

    if level_down or slope_down:
        trend = "degrading"
        basis = "level" if level_down else "slope"
    elif level_up or slope_up:
        trend = "improving"
        basis = "level" if level_up else "slope"
    elif level_reject or slope_reject:
        trend = "stable"
        basis = "immaterial"
    else:
        trend = "stable"
        basis = "within_noise"

    n_txt = f"n={sample_size}"
    if basis == "level":
        reason = (
            f"{metric_name} {current_value:.3f} is {abs(z):.1f} standard errors "
            f"{'below' if z < 0 else 'above'} the {k}-fold baseline {baseline_value:.3f} "
            f"({n_txt}, {change_percent:+.1f}%)"
        )
    elif basis == "slope":
        reason = (
            f"{metric_name} has a significant {'downward' if (slope_t or 0.0) < 0 else 'upward'} "
            f"slope across {n_points} folds (t={slope_t:+.1f}, fitted change {fitted_change:+.3f})"
        )
    elif basis == "immaterial":
        reason = (
            f"{change_percent:+.1f}% vs the {k}-fold baseline is statistically distinguishable "
            f"but below the {min_change_percent:g}% materiality floor"
        )
    else:
        reason = (
            f"{change_percent:+.1f}% vs the {k}-fold baseline is within sampling noise at "
            f"{n_txt} (z={z:+.1f}; ±{z_threshold:g} needed)"
        )

    return TrendAssessment(
        trend=trend,
        change_percent=change_percent,
        is_significant=bool(level_reject or slope_reject),
        basis=basis,
        reason=reason,
        current_value=current_value,
        baseline_value=baseline_value,
        n_points=n_points,
        sample_size=sample_size,
        standard_error=float(noise),
        z_score=float(z),
        # A perfect linear fit yields an infinite t (see slope_t_stat); JSON
        # cannot carry it, so the statistic is reported as None in that case.
        slope_t_stat=(float(slope_t) if slope_t is not None and math.isfinite(slope_t) else None),
    )


__all__ = [
    "TrendPoint",
    "TrendAssessment",
    "assess_trend",
    "hanley_mcneil_se",
    "metric_standard_error",
    "proportion_se",
    "slope_t_stat",
]
