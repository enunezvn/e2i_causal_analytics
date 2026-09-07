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
  sample size and positive rate, ONLY where a closed form exists: binomial on
  n for accuracy, binomial on the positive count for recall, Hanley-McNeil
  for AUC. Precision (denominator = PREDICTED positives, not persisted) and
  F1 (no binomial variance) get none — an invented interval would understate
  their noise and re-create the false alarms this module exists to remove.
* :func:`slope_t_stat` — OLS slope t-statistic over the window, the signal a
  level test cannot see (a steady 0.01/month slide).
* :func:`assess_trend` — the classifier: ``degrading`` / ``improving`` only
  when the change is BOTH statistically distinguishable from noise (level z
  beyond ``z_threshold`` OR slope t beyond ``slope_t_threshold``) AND
  material (the historical ±5% floor, kept as a materiality gate).

Noise scales (the level test's denominator)
-------------------------------------------
Two independent estimates; the classifier uses the LARGER available one:

* ``analytic`` — sqrt(SE_current² + SE_baseline-mean²), available only when the
  newest fold AND every baseline fold have an analytic SE. A baseline fold of
  unknown precision is never treated as exact.
* ``empirical`` — the fold-to-fold standard deviation of the baseline
  (×sqrt(1+1/k) for a new observation), available with ≥6 baseline folds.
  Because that sd is itself estimated from k folds, it is inflated by
  t_{k-1}/z so the reported z compares with ``z_threshold`` at the intended
  tail probability (a Student-t test, not a z-test).

With neither scale the row falls back to the legacy relative rule, labelled
``basis="legacy_relative"`` with the reason, so the caller can see which rule
produced the label.

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

# Metrics with a closed-form sampling variance from (n, positive_rate).
_ANALYTIC_SE_METRICS = frozenset({"accuracy", "recall", "auc_roc"})
# Of those, the ones whose SE needs the fold's positive rate.
_NEEDS_POSITIVE_RATE = frozenset({"recall", "auc_roc"})


@dataclass(frozen=True)
class TrendPoint:
    """One fold of the trend series (a walk-forward month or a daily window)."""

    value: float
    sample_size: Optional[int] = None
    positive_rate: Optional[float] = None


@dataclass(frozen=True)
class TrendAssessment:
    """Classifier output. ``basis`` says WHICH rule produced ``trend``:

    * ``level`` — newest fold vs baseline beyond ``z_threshold`` noise units and material
    * ``slope`` — OLS slope over the window beyond ``slope_t_threshold`` and material
    * ``within_noise`` — change is inside the noise band
    * ``immaterial`` — statistically distinguishable but below the materiality floor
    * ``legacy_relative`` — no noise scale could be derived; ±5% relative rule
    * ``insufficient_history`` — one fold only; nothing to compare against
    * ``no_data`` — empty window

    ``noise_source`` names the scale the level test used: ``analytic``,
    ``empirical`` (t-adjusted fold spread), or None on the fallback paths.
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
    noise_source: Optional[str] = None


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


def student_t_quantile(z: float, df: int) -> float:
    """Student-t quantile with the same tail probability as the normal quantile
    ``z`` — Cornish-Fisher expansion in 1/df (five terms). Checked against
    ``scipy.stats.t.ppf`` 2026-09-07: within 0.3% for df ≥ 5 and within 0.7%
    at df = 4 for |z| ≤ 3 (the slope test's smallest df); kept dependency-free
    so this module stays pure and import-light."""
    if df <= 0:
        return float(z)
    d = float(df)
    z3, z5, z7, z9 = z**3, z**5, z**7, z**9
    return float(
        z
        + (z3 + z) / (4.0 * d)
        + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * d**2)
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * d**3)
        + (79.0 * z9 + 776.0 * z7 + 1482.0 * z5 - 1920.0 * z3 - 945.0 * z) / (92160.0 * d**4)
    )


def metric_standard_error(
    metric_name: str,
    value: float,
    sample_size: Optional[int],
    positive_rate: Optional[float],
) -> Optional[float]:
    """Analytic SE of ``value`` for one fold, or None when no closed form applies.

    * ``accuracy`` → binomial on ``n``.
    * ``recall`` → binomial on the actual-positive count ``n·p`` (its exact
      denominator); None without the positive rate.
    * ``auc_roc`` → Hanley-McNeil on the implied class counts; None without
      the positive rate (the binomial form understates an AUC's SE, so it is
      NOT used as a stand-in).
    * ``precision`` / ``f1`` / anything else (pr_auc, calibration slope,
      unknown names) → None. Precision's denominator is the predicted-positive
      count, which the fold rows do not carry; F1 has no binomial variance.
      The classifier then uses the empirical fold spread (≥6 folds) or the
      legacy rule — never a fabricated interval.
    """
    if sample_size is None or sample_size <= 0:
        return None
    if value is None or not math.isfinite(value):
        return None
    if metric_name not in _ANALYTIC_SE_METRICS or not (0.0 <= value <= 1.0):
        return None
    if metric_name == "accuracy":
        return proportion_se(value, sample_size)
    p = float(positive_rate) if positive_rate is not None else None
    if p is None or not (0.0 < p < 1.0):
        return None
    n_pos = max(1, int(round(sample_size * p)))
    if metric_name == "recall":
        return proportion_se(value, n_pos)
    n_neg = max(1, sample_size - n_pos)
    return hanley_mcneil_se(value, n_pos, n_neg)


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
    """A sample size of at least one row, else None (0, 0.5, NaN, doubles)."""
    f = _opt_float(v)
    if f is None:
        return None
    n = int(f)
    return n if n >= 1 else None


def _no_analytic_reason(
    metric_name: str,
    current: TrendPoint,
    se_current: Optional[float],
    baseline_ses: Sequence[Optional[float]],
) -> str:
    """Why the analytic noise scale could not be formed (for the reason text)."""
    if _opt_n(current.sample_size) is None:
        return "no sample size on the newest fold"
    if metric_name not in _ANALYTIC_SE_METRICS:
        return f"{metric_name} has no closed-form standard error"
    if se_current is None and metric_name in _NEEDS_POSITIVE_RATE:
        return f"{metric_name} needs the fold's positive rate for its standard error"
    if se_current is None:
        return f"no standard error for the newest {metric_name} value"
    missing = sum(1 for s in baseline_ses if s is None)
    return f"sampling error unknown for {missing} of {len(baseline_ses)} baseline folds"


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

    Raises ``ValueError`` for a non-finite or non-positive ``z_threshold`` /
    ``slope_t_threshold`` (a zero threshold would make every non-zero z
    "significant") or a negative ``min_change_percent``.
    """
    for name, val in (("z_threshold", z_threshold), ("slope_t_threshold", slope_t_threshold)):
        if not (_finite(val) and float(val) > 0.0):
            raise ValueError(f"{name} must be a finite positive number, got {val!r}")
    if not (_finite(min_change_percent) and float(min_change_percent) >= 0.0):
        raise ValueError(
            f"min_change_percent must be a finite non-negative number, got {min_change_percent!r}"
        )

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

    # ---- Noise scale 1: analytic — needs an SE for the newest fold AND for
    # every baseline fold (a baseline of unknown precision is never "exact").
    se_current = metric_standard_error(
        metric_name, current_value, sample_size, _opt_float(current.positive_rate)
    )
    baseline_ses = [
        metric_standard_error(
            metric_name, float(p.value), _opt_n(p.sample_size), _opt_float(p.positive_rate)
        )
        for p in baseline_pts
    ]
    analytic: Optional[float] = None
    if se_current is not None and all(s is not None for s in baseline_ses):
        se_mean = math.sqrt(sum(s * s for s in baseline_ses if s is not None)) / k
        analytic = math.sqrt(se_current * se_current + se_mean * se_mean)

    # ---- Noise scale 2: empirical — the fold-to-fold spread of the baseline,
    # widened for a new observation and t-adjusted because the sd itself is
    # estimated from only k folds.
    empirical: Optional[float] = None
    if k >= max(int(min_points_for_dispersion), 2):
        sd = float(baseline_values.std(ddof=1))
        if sd > 0.0:
            t_factor = student_t_quantile(float(z_threshold), k - 1) / float(z_threshold)
            empirical = sd * math.sqrt(1.0 + 1.0 / k) * t_factor

    # ---- Legacy fallback: no noise scale at all → the ±5% rule, labelled. ----
    if analytic is None and empirical is None:
        if change_percent > min_change_percent:
            trend = "improving"
        elif change_percent < -min_change_percent:
            trend = "degrading"
        else:
            trend = "stable"
        why_analytic = _no_analytic_reason(metric_name, current, se_current, baseline_ses)
        if k < min_points_for_dispersion:
            why_empirical = (
                f"fewer than {min_points_for_dispersion} baseline folds for the "
                f"empirical spread ({k})"
            )
        else:
            why_empirical = "the baseline folds are identical (zero empirical spread)"
        return TrendAssessment(
            trend=trend,
            change_percent=change_percent,
            is_significant=abs(change_percent) > legacy_significance_percent,
            basis="legacy_relative",
            reason=(
                f"{why_analytic}; {why_empirical}; classified by the "
                f"relative-change rule (±{min_change_percent:g}%)"
            ),
            current_value=current_value,
            baseline_value=baseline_value,
            n_points=n_points,
            sample_size=sample_size,
        )

    # ---- Level test on the larger available scale. ---------------------------
    # "Sampling noise" is claimed ONLY for the analytic scale. The empirical
    # spread also carries real month-composition shifts and any earlier
    # decline, so its band is called what it is: historical fold variation.
    volatile_note = ""
    if analytic is not None and empirical is not None:
        if analytic >= empirical:
            noise, noise_source = analytic, "analytic"
            scale_txt = (
                f"analytic sampling error at n={sample_size}, which exceeds the "
                f"{k}-fold historical variation"
            )
        else:
            noise, noise_source = empirical, "empirical"
            scale_txt = (
                f"the t-adjusted historical variation of the {k} baseline folds, which "
                f"exceeds the analytic sampling error at n={sample_size}"
            )
            z_analytic = (current_value - baseline_value) / analytic
            if abs(z_analytic) > z_threshold:
                volatile_note = (
                    f"; on the analytic scale alone z={z_analytic:+.1f} — the history is "
                    "volatile, so this fold is not called an outlier"
                )
    elif analytic is not None:
        noise, noise_source = analytic, "analytic"
        scale_txt = f"analytic sampling error at n={sample_size}"
    else:
        assert empirical is not None
        noise, noise_source = empirical, "empirical"
        scale_txt = f"the t-adjusted historical variation of the {k} baseline folds"
    z = (current_value - baseline_value) / noise
    level_reject = abs(z) > z_threshold
    level_material = abs(change_percent) >= min_change_percent

    # ---- Slope test over the whole window (oldest → newest). -----------------
    slope_t: Optional[float] = None
    slope_crit: Optional[float] = None
    slope_reject = False
    slope_material = False
    fitted_change = 0.0
    if n_points >= min_points_for_slope:
        st = slope_t_stat([float(p.value) for p in reversed(pts)])
        if st is not None:
            slope_t, fitted_change = st
            # The OLS slope t-statistic has n_points-2 degrees of freedom; the
            # threshold is a NORMAL quantile, so compare against the Student-t
            # quantile at the same tail probability (t(4) needs ≈4.3, not 2.5).
            slope_crit = student_t_quantile(float(slope_t_threshold), n_points - 2)
            slope_reject = abs(slope_t) > slope_crit
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

    if basis == "level":
        units = (
            "standard errors" if noise_source == "analytic" else "× the historical fold variation"
        )
        reason = (
            f"{metric_name} {current_value:.3f} is {abs(z):.1f} {units} "
            f"{'below' if z < 0 else 'above'} the {k}-fold baseline {baseline_value:.3f} "
            f"({change_percent:+.1f}%; scale: {scale_txt})"
        )
    elif basis == "slope":
        reason = (
            f"{metric_name} has a significant {'downward' if (slope_t or 0.0) < 0 else 'upward'} "
            f"slope across {n_points} folds (t={slope_t:+.1f} vs ±{slope_crit:.1f} needed at "
            f"{n_points - 2} df; fitted change {fitted_change:+.3f})"
        )
    elif basis == "immaterial":
        reason = (
            f"{change_percent:+.1f}% vs the {k}-fold baseline is statistically distinguishable "
            f"but below the {min_change_percent:g}% materiality floor"
        )
    else:
        band = "sampling noise" if noise_source == "analytic" else "historical fold variation"
        reason = (
            f"{change_percent:+.1f}% vs the {k}-fold baseline is within {band} "
            f"(z={z:+.1f}; ±{z_threshold:g} needed; scale: {scale_txt}{volatile_note})"
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
        noise_source=noise_source,
    )


__all__ = [
    "TrendPoint",
    "TrendAssessment",
    "assess_trend",
    "hanley_mcneil_se",
    "metric_standard_error",
    "proportion_se",
    "slope_t_stat",
    "student_t_quantile",
]
