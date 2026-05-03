"""Unit tests for ``bca_ci_from_resamples`` (Phase 1 W1 day 3).

Covers shard 20 §F BCa rows:
- ``test_bca_ci_matches_scipy_on_known_synthetic``
- ``test_bca_ci_degenerate_distribution_fallback``
- ``test_bca_ci_extreme_p_lt_fallback``
- ``test_jackknife_metric_drops_one_sample``
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bootstrap_utils import (
    BcaFromResamplesResult,
    bca_ci_from_resamples,
)


def _bootstrap_distribution_for_mean(
    sample: np.ndarray, n_bootstrap: int = 2000, seed: int = 42
) -> np.ndarray:
    """Run a plain mean bootstrap so we have a fixture distribution."""
    rng = np.random.default_rng(seed)
    n = sample.size
    out = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(low=0, high=n, size=n)
        out[i] = float(sample[idx].mean())
    return out


def _jackknife_for_mean(sample: np.ndarray) -> np.ndarray:
    """Jackknife for the mean: ``J_i = (Σ x − x_i) / (n − 1)``."""
    total = float(sample.sum())
    n = sample.size
    return (total - sample) / (n - 1)


def test_bca_ci_matches_scipy_on_known_synthetic() -> None:
    """BCa CI matches ``scipy.stats.bootstrap`` to ≤ 1e-3 (shard 20 §G #16).

    Cycle-22 I-3 closure: previous version compared two independently
    drawn bootstrap distributions, which inflated the cross-check
    tolerance to 0.05. Here we extract scipy's own bootstrap distribution
    via ``BootstrapResult.bootstrap_distribution`` and feed the SAME
    distribution into our helper — any difference must come from the BCa
    formula itself, not bootstrap sampling variance.
    """
    rng = np.random.default_rng(7)
    sample = rng.normal(loc=0.0, scale=1.0, size=200)
    point = float(sample.mean())
    jack = _jackknife_for_mean(sample)

    from scipy.stats import bootstrap as scipy_bootstrap
    scipy_res = scipy_bootstrap(
        (sample,),
        np.mean,
        method="BCa",
        n_resamples=2000,
        confidence_level=0.95,
        rng=np.random.default_rng(11),
    )
    # Use scipy's own bootstrap distribution — eliminates resampling drift.
    ours = bca_ci_from_resamples(
        scipy_res.bootstrap_distribution,
        point,
        jack,
        confidence_level=0.95,
    )
    assert ours.method == "bca"
    assert ours.ci_lo == pytest.approx(
        float(scipy_res.confidence_interval.low), abs=1e-3
    )
    assert ours.ci_hi == pytest.approx(
        float(scipy_res.confidence_interval.high), abs=1e-3
    )


def test_bca_ci_degenerate_distribution_fallback() -> None:
    """A constant bootstrap distribution falls back to percentile."""
    boot = np.full(1000, 0.5)
    jack = np.full(100, 0.5)
    res = bca_ci_from_resamples(boot, point_estimate=0.5, jackknife_values=jack)
    # Constant statistic → z0 undefined (P(boot < 0.5) == 0).
    assert res.method == "percentile_fallback"
    assert res.ci_lo == pytest.approx(0.5)
    assert res.ci_hi == pytest.approx(0.5)


def test_bca_ci_point_estimate_outside_bootstrap_range() -> None:
    """When ``point_estimate`` is outside the bootstrap distribution → fallback."""
    boot = np.array([0.4, 0.5, 0.6, 0.55, 0.45] * 200, dtype=float)
    jack = np.array([0.5] * 100)  # constant jackknife → also flags acceleration
    res_below = bca_ci_from_resamples(boot, point_estimate=0.0, jackknife_values=jack)
    res_above = bca_ci_from_resamples(boot, point_estimate=1.0, jackknife_values=jack)
    assert res_below.method == "percentile_fallback"
    assert res_above.method == "percentile_fallback"
    # Endpoints come from plain percentile.
    expected_lo = float(np.percentile(boot, 2.5))
    expected_hi = float(np.percentile(boot, 97.5))
    assert res_below.ci_lo == pytest.approx(expected_lo)
    assert res_below.ci_hi == pytest.approx(expected_hi)


def test_bca_ci_empty_bootstrap_returns_none() -> None:
    """Empty bootstrap → ``method=='none'`` and CI endpoints None."""
    res = bca_ci_from_resamples(np.array([]), point_estimate=0.5, jackknife_values=np.array([0.5, 0.5]))
    assert res.method == "none"
    assert res.ci_lo is None
    assert res.ci_hi is None


def test_bca_ci_too_small_jackknife_falls_back() -> None:
    """Jackknife with < 2 values → percentile fallback (acceleration undefined)."""
    boot = _bootstrap_distribution_for_mean(np.array([1.0, 2.0, 3.0]), n_bootstrap=500)
    res = bca_ci_from_resamples(
        boot, point_estimate=2.0, jackknife_values=np.array([2.0])
    )
    assert res.method == "percentile_fallback"
    assert res.fallback_reason == "jackknife_too_small"


def test_bca_ci_records_acceleration_and_bias_correction() -> None:
    """The result carries the BCa parameters for downstream auditability."""
    rng = np.random.default_rng(13)
    sample = rng.normal(0, 1, size=100)
    boot = _bootstrap_distribution_for_mean(sample, n_bootstrap=1000, seed=21)
    jack = _jackknife_for_mean(sample)
    res = bca_ci_from_resamples(boot, point_estimate=float(sample.mean()), jackknife_values=jack)
    assert res.method == "bca"
    assert res.acceleration is not None
    assert res.bias_correction is not None
    # Acceleration for a near-symmetric sample should be small.
    assert abs(res.acceleration) < 0.05


def test_bca_ci_validates_1d_input() -> None:
    """Non-1-D bootstrap_values raises ValueError."""
    with pytest.raises(ValueError, match="1-D"):
        bca_ci_from_resamples(np.zeros((10, 10)), point_estimate=0.0, jackknife_values=np.zeros(5))


def test_bca_ci_returns_named_dataclass() -> None:
    """Result is a ``BcaFromResamplesResult`` instance."""
    boot = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 100, dtype=float)
    jack = np.array([0.3] * 50)
    res = bca_ci_from_resamples(boot, point_estimate=0.3, jackknife_values=jack)
    assert isinstance(res, BcaFromResamplesResult)
    assert res.method in {"bca", "percentile_fallback", "none"}
