"""Sampling-aware trend classification (pure, no I/O).

Why this exists (2026-09-07): /model-performance flagged four gold-standard
models "degrading" on a single ±5% relative-change rule applied to ONE
walk-forward fold — a fold of 54 rows whose Hanley-McNeil AUC standard error
(~0.07) dwarfed the 5% threshold (~0.04). Under a perfectly stationary model
that rule reported non-"stable" 38–56% of the time. These tests pin the
replacement: the newest fold is compared with the baseline on a sampling-error
scale (plus an OLS slope test over the window), with the ±5% floor kept only as
a MATERIALITY gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.services.performance_trend_stats import (
    TrendPoint,
    assess_trend,
    hanley_mcneil_se,
    metric_standard_error,
    proportion_se,
    slope_t_stat,
)

# ---------------------------------------------------------------------------
# Standard-error building blocks
# ---------------------------------------------------------------------------


def test_hanley_mcneil_matches_textbook_value():
    # Hanley & McNeil (1982) worked example: AUC 0.75, 50 positives, 50 negatives
    # gives SE ≈ 0.0487 (before the boundary shrinkage, which is negligible here).
    se = hanley_mcneil_se(0.75, 50, 50)
    assert se == pytest.approx(0.0487, abs=0.002)


def test_hanley_mcneil_grows_with_class_imbalance_and_shrinks_with_n():
    balanced = hanley_mcneil_se(0.80, 100, 100)
    imbalanced = hanley_mcneil_se(0.80, 20, 180)
    bigger = hanley_mcneil_se(0.80, 400, 400)
    assert imbalanced > balanced > bigger


def test_boundary_values_keep_a_nonzero_standard_error():
    # A perfect fold (AUC 1.0 / accuracy 1.0) must not produce SE=0 → z=inf.
    assert hanley_mcneil_se(1.0, 30, 30) > 0.0
    assert proportion_se(1.0, 60) > 0.0
    assert proportion_se(0.0, 60) > 0.0


def test_metric_standard_error_dispatch():
    # AUC with a known class balance uses Hanley-McNeil.
    hm = metric_standard_error("auc_roc", 0.71, 134, 0.38)
    assert hm == pytest.approx(hanley_mcneil_se(0.71, 51, 83), rel=1e-6)
    # AUC WITHOUT a class balance falls back to the (optimistic) proportion form.
    assert metric_standard_error("auc_roc", 0.71, 134, None) == pytest.approx(
        proportion_se(0.71, 134), rel=1e-6
    )
    # Recall's effective n is the positive count.
    assert metric_standard_error("recall", 0.6, 200, 0.25) == pytest.approx(
        proportion_se(0.6, 50), rel=1e-6
    )
    # Unknown n → no SE (caller falls back to the legacy rule).
    assert metric_standard_error("accuracy", 0.8, None, None) is None
    assert metric_standard_error("accuracy", 0.8, 0, None) is None
    # A metric that is not a [0, 1] rate (e.g. calibration_slope ≈ 1.2) has no
    # closed-form SE here → None, never a fabricated interval.
    assert metric_standard_error("calibration_slope", 1.2, 500, 0.3) is None


def test_slope_t_stat_positive_control_and_guards():
    rng = np.random.default_rng(7)
    decline = [0.85 - 0.01 * i + rng.normal(0, 0.005) for i in range(12)]
    t, fitted = slope_t_stat(decline)
    assert t < -5.0
    assert fitted == pytest.approx(-0.11, abs=0.03)

    flat = [0.80 + rng.normal(0, 0.02) for _ in range(12)]
    t_flat, _ = slope_t_stat(flat)
    assert abs(t_flat) < 2.5

    assert slope_t_stat([0.8, 0.81, 0.79]) is None  # fewer than 4 points
    assert slope_t_stat([0.8, 0.8, 0.8, 0.8])[0] == 0.0  # zero slope, zero residual


# ---------------------------------------------------------------------------
# assess_trend — the classifier the tracker consumes
# ---------------------------------------------------------------------------


def _pts(values_newest_first, n, p=0.4):
    return [TrendPoint(value=v, sample_size=n, positive_rate=p) for v in values_newest_first]


# The real 2026-09-07 hcp_adoption_remibrutinib auc_roc window (newest first):
# June-2026 fold 0.7094 (n=134, p=0.381) vs eight folds averaging 0.815.
_HCP_REMI = [0.7094, 0.8233, 0.8426, 0.8277, 0.7974, 0.7887, 0.8222, 0.7803, 0.8390]


def test_single_low_fold_within_noise_is_stable_not_degrading():
    """The flag that motivated this module: −13% on ONE fold, z ≈ −2.1."""
    a = assess_trend(_pts(_HCP_REMI, n=134, p=0.381), "auc_roc")
    assert a.change_percent == pytest.approx(-12.97, abs=0.05)
    assert a.trend == "stable"
    assert a.basis == "within_noise"
    assert a.z_score is not None and -2.5 < a.z_score < -1.5
    assert a.sample_size == 134
    assert "sampling noise" in a.reason


def test_partial_month_fold_is_stable():
    """persistence_fabhalta 2026-09-07: 0.7125 on n=54 vs 0.771 baseline (−7.5%)."""
    window = [0.7125] + [0.771] * 8
    a = assess_trend(_pts(window, n=54, p=0.35), "auc_roc")
    assert a.trend == "stable"
    assert a.basis == "within_noise"
    assert a.is_significant is False


def test_genuine_level_drop_is_degrading():
    """A 0.15 AUC drop on a full month (n=230) is ~4 SE — must be caught."""
    window = [0.65] + [0.80] * 8
    a = assess_trend(_pts(window, n=230, p=0.35), "auc_roc")
    assert a.trend == "degrading"
    assert a.basis == "level"
    assert a.is_significant is True
    assert a.z_score is not None and a.z_score < -2.5


def test_gradual_decline_is_caught_by_the_slope_test():
    """A steady 0.01/month decline never trips the level test but has a
    decisive slope — the second signal the level test cannot see."""
    rng = np.random.default_rng(11)
    oldest_first = [0.85 - 0.01 * i + rng.normal(0, 0.004) for i in range(12)]
    a = assess_trend(_pts(list(reversed(oldest_first)), n=230, p=0.35), "auc_roc")
    assert a.trend == "degrading"
    assert a.basis == "slope"
    assert a.slope_t_stat is not None and a.slope_t_stat < -2.5


def test_symmetric_improving_classification():
    window = [0.95] + [0.80] * 8
    a = assess_trend(_pts(window, n=230, p=0.35), "auc_roc")
    assert a.trend == "improving"
    assert a.z_score is not None and a.z_score > 2.5


def test_statistically_significant_but_immaterial_change_stays_stable():
    """At n=50k a 1% drop is ~10 SE but below the 5% materiality floor."""
    window = [0.792] + [0.800] * 8
    a = assess_trend(_pts(window, n=50_000, p=0.4), "auc_roc")
    assert a.trend == "stable"
    assert a.basis == "immaterial"
    assert a.is_significant is True


def test_legacy_relative_rule_when_sample_size_is_unknown():
    """Rows without a sample size cannot be tested → the historical ±5% rule,
    labelled as such (never silently pretend it was a significance test)."""
    pts = [TrendPoint(value=v) for v in (0.85, 0.82, 0.80, 0.78)]
    a = assess_trend(pts, "accuracy")
    assert a.trend == "improving"  # +6.25% under the legacy rule
    assert a.basis == "legacy_relative"
    assert a.z_score is None and a.standard_error is None
    assert a.is_significant is False  # legacy: |change| > 10% only

    pts = [TrendPoint(value=v) for v in (0.60, 0.85, 0.84)]
    a = assess_trend(pts, "accuracy")
    assert a.trend == "degrading"
    assert a.is_significant is True


def test_no_data_and_single_point_edges():
    a = assess_trend([], "auc_roc")
    assert a.trend == "unknown" and a.basis == "no_data" and a.n_points == 0

    a = assess_trend(_pts([0.8], n=100), "auc_roc")
    assert a.trend == "stable"
    assert a.basis == "insufficient_history"
    assert a.change_percent == 0.0


def test_empirical_fold_dispersion_widens_the_noise_scale():
    """With ≥6 baseline folds the fold-to-fold spread is a second noise
    estimate; the classifier uses the LARGER of the analytic and empirical
    scales so month-composition variation is not mistaken for degradation."""
    rng = np.random.default_rng(3)
    baseline = [0.80 + rng.normal(0, 0.06) for _ in range(12)]  # wide spread
    window = [0.70] + baseline
    a = assess_trend(_pts(window, n=5000, p=0.4), "auc_roc")  # tiny analytic SE
    assert a.standard_error is not None
    assert a.standard_error > metric_standard_error("auc_roc", 0.70, 5000, 0.4)
    assert a.trend == "stable"


def test_nan_and_non_finite_values_do_not_crash():
    pts = [TrendPoint(value=0.8, sample_size=100, positive_rate=0.4)] + [
        TrendPoint(value=float("nan"), sample_size=100, positive_rate=0.4)
    ] * 3
    a = assess_trend(pts, "auc_roc")
    assert a.trend in {"stable", "unknown"}
    assert not (a.z_score is not None and math.isnan(a.z_score))
