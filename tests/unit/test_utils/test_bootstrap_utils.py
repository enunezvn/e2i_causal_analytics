"""Tests for :mod:`src.utils.bootstrap_utils` (Phase 1 W3-lite Day-5).

Verifies BCa CI behavior over small fold-value samples (k=4..50), plus the
unstable-warning trigger contract: fires when ``n_samples < 4`` OR the
jackknife acceleration magnitude exceeds 0.25 OR the bootstrap endpoints are
non-finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bootstrap_utils import BcaResult, bca_confidence_interval


class TestBcaConfidenceInterval:
    def test_returns_finite_ci_for_well_conditioned_sample(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(loc=0.5, scale=0.1, size=10)
        result = bca_confidence_interval(values, rng_seed=42)
        assert result.ci_lo is not None
        assert result.ci_hi is not None
        assert np.isfinite(result.ci_lo)
        assert np.isfinite(result.ci_hi)
        assert result.ci_lo <= np.mean(values) <= result.ci_hi
        assert result.n_samples == 10

    def test_skips_bca_when_below_min_samples(self) -> None:
        # Below default min_samples=4 → unstable_warning, None endpoints
        for n in (0, 1, 2, 3):
            values = [0.5] * n
            result = bca_confidence_interval(values)
            assert result.ci_lo is None
            assert result.ci_hi is None
            assert result.unstable_warning is True
            assert result.n_samples == n

    def test_unstable_warning_when_acceleration_above_threshold(self) -> None:
        # Skewed sample with 1 outlier → moderate acceleration magnitude.
        # Using a tightened ``instability_threshold`` to verify the gate logic
        # (the default 0.25 threshold is calibrated for k=10 nominal stability;
        # the gate itself is what we're testing here).
        values = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.99])
        result = bca_confidence_interval(values, rng_seed=42, instability_threshold=0.05)
        assert result.unstable_warning is True
        # Acceleration is finite (sample is well-formed enough for jackknife)
        # but exceeds the tightened threshold of 0.05.
        assert result.acceleration is not None
        assert abs(result.acceleration) > 0.05

    def test_unstable_warning_false_for_well_conditioned_at_default_threshold(self) -> None:
        # Symmetric small sample — acceleration should be near 0
        values = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.45, 0.40, 0.35, 0.30, 0.50])
        result = bca_confidence_interval(values, rng_seed=42)
        assert result.unstable_warning is False
        assert result.acceleration is not None
        assert abs(result.acceleration) <= 0.25

    def test_deterministic_with_same_rng_seed(self) -> None:
        values = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.45, 0.40, 0.50])
        r1 = bca_confidence_interval(values, rng_seed=42, n_resamples=500)
        r2 = bca_confidence_interval(values, rng_seed=42, n_resamples=500)
        assert r1.ci_lo == r2.ci_lo
        assert r1.ci_hi == r2.ci_hi
        assert r1.unstable_warning == r2.unstable_warning

    def test_ci_brackets_mean_for_uniform_sample(self) -> None:
        # 10 values close to the mean → BCa CI should bracket the mean
        rng = np.random.default_rng(7)
        values = rng.uniform(0.6, 0.7, size=10)
        result = bca_confidence_interval(values, rng_seed=42)
        mean = float(np.mean(values))
        assert result.ci_lo is not None and result.ci_lo <= mean
        assert result.ci_hi is not None and result.ci_hi >= mean

    def test_returns_bcaresult_dataclass(self) -> None:
        result = bca_confidence_interval([0.5, 0.6, 0.55, 0.7], rng_seed=42)
        assert isinstance(result, BcaResult)
        assert hasattr(result, "ci_lo")
        assert hasattr(result, "ci_hi")
        assert hasattr(result, "unstable_warning")
        assert hasattr(result, "n_samples")
        assert hasattr(result, "acceleration")

    def test_rejects_non_1d_input(self) -> None:
        with pytest.raises(ValueError, match=r"1-D"):
            bca_confidence_interval(np.zeros((4, 3)))

    def test_acceleration_returned_for_valid_sample(self) -> None:
        values = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
        result = bca_confidence_interval(values, rng_seed=42)
        assert result.acceleration is not None
        assert np.isfinite(result.acceleration)
