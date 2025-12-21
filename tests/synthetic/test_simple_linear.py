"""Benchmark: Simple Linear - Baseline Sanity Check.

Version: 4.3
Purpose: Verify causal estimator recovers true effect with no confounding

Data Generating Process:
    T ~ Bernoulli(0.5)
    Y = 0.50 * T + epsilon
    True ATE = +0.50

This is the most basic test - any valid causal estimator should pass.
Failure indicates a fundamental implementation error.

Reference: docs/E2I_Causal_Validation_Protocol.html
"""

import numpy as np
import pytest

from tests.synthetic.conftest import (
    SyntheticDataset,
    estimate_ate_adjusted,
    estimate_ate_naive,
    generate_simple_linear,
)


class TestSimpleLinearBenchmark:
    """Benchmark tests for simple linear DGP."""

    def test_naive_estimate_accurate(self, simple_linear_dataset: SyntheticDataset):
        """CI/CD test: Naive estimate should be accurate without confounding.

        With no confounding, the naive difference-in-means estimator
        should recover the true effect.
        """
        dataset = simple_linear_dataset

        estimated_ate = estimate_ate_naive(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
        )

        error = abs(estimated_ate - dataset.true_ate)

        assert error < dataset.tolerance, (
            f"Naive ATE estimate {estimated_ate:.4f} not within tolerance "
            f"of true ATE {dataset.true_ate:.4f} (error: {error:.4f}, "
            f"tolerance: {dataset.tolerance:.4f})"
        )

    def test_adjusted_estimate_accurate(self, simple_linear_dataset: SyntheticDataset):
        """CI/CD test: Adjusted estimate should also be accurate.

        Even though no adjustment is needed, the adjusted estimator
        should still be accurate (shouldn't hurt).
        """
        dataset = simple_linear_dataset

        estimated_ate = estimate_ate_adjusted(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        error = abs(estimated_ate - dataset.true_ate)

        assert error < dataset.tolerance, (
            f"Adjusted ATE estimate {estimated_ate:.4f} not within tolerance "
            f"of true ATE {dataset.true_ate:.4f} (error: {error:.4f})"
        )

    def test_treatment_balance(self, simple_linear_dataset: SyntheticDataset):
        """Verify treatment is balanced (random assignment)."""
        dataset = simple_linear_dataset
        data = dataset.data

        treated_prop = data[dataset.treatment_col].mean()

        # With n=10000 and p=0.5, expect ~0.50 with margin
        assert (
            0.48 < treated_prop < 0.52
        ), f"Treatment proportion {treated_prop:.3f} indicates non-random assignment"

    def test_reproducibility(self):
        """Verify DGP is reproducible with same seed."""
        dataset1 = generate_simple_linear(n=1000, seed=42)
        dataset2 = generate_simple_linear(n=1000, seed=42)

        pd_testing = pytest.importorskip("pandas.testing")
        pd_testing.assert_frame_equal(dataset1.data, dataset2.data)

    def test_different_seeds_different_data(self):
        """Verify different seeds produce different data."""
        dataset1 = generate_simple_linear(n=1000, seed=42)
        dataset2 = generate_simple_linear(n=1000, seed=123)

        # Should not be equal
        assert not dataset1.data.equals(dataset2.data)

    @pytest.mark.parametrize("true_ate", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_various_effect_sizes(self, true_ate: float):
        """CI/CD test: Verify recovery across different effect sizes."""
        dataset = generate_simple_linear(n=10000, true_ate=true_ate, seed=42)

        estimated_ate = estimate_ate_naive(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
        )

        error = abs(estimated_ate - dataset.true_ate)

        assert error < dataset.tolerance, (
            f"Failed to recover true ATE={true_ate:.2f} "
            f"(estimated: {estimated_ate:.4f}, error: {error:.4f})"
        )

    def test_sample_size_sensitivity(self):
        """Verify smaller samples have larger variance."""
        generate_simple_linear(n=500, seed=42)
        generate_simple_linear(n=50000, seed=42)

        # Estimate with multiple seeds
        small_estimates = []
        large_estimates = []

        for seed in range(10):
            small_ds = generate_simple_linear(n=500, seed=seed)
            large_ds = generate_simple_linear(n=50000, seed=seed)

            small_estimates.append(
                estimate_ate_naive(small_ds.data, small_ds.treatment_col, small_ds.outcome_col)
            )
            large_estimates.append(
                estimate_ate_naive(large_ds.data, large_ds.treatment_col, large_ds.outcome_col)
            )

        small_var = np.var(small_estimates)
        large_var = np.var(large_estimates)

        # Smaller sample should have higher variance
        assert small_var > large_var, (
            f"Expected smaller sample to have higher variance "
            f"(small: {small_var:.6f}, large: {large_var:.6f})"
        )


class TestSimpleLinearDatasetProperties:
    """Test properties of the simple linear dataset."""

    def test_dataset_structure(self, simple_linear_dataset: SyntheticDataset):
        """Verify dataset has expected structure."""
        dataset = simple_linear_dataset

        assert "T" in dataset.data.columns
        assert "Y" in dataset.data.columns
        assert len(dataset.data) == dataset.n_samples
        assert dataset.dgp_name == "simple_linear"

    def test_treatment_is_binary(self, simple_linear_dataset: SyntheticDataset):
        """Verify treatment is binary 0/1."""
        dataset = simple_linear_dataset
        T = dataset.data[dataset.treatment_col]

        assert set(T.unique()).issubset({0, 1})

    def test_outcome_distribution(self, simple_linear_dataset: SyntheticDataset):
        """Verify outcome has expected distribution."""
        dataset = simple_linear_dataset
        Y = dataset.data[dataset.outcome_col]

        # Outcome should be centered around 0 for control, true_ate for treated
        control_mean = Y[dataset.data["T"] == 0].mean()
        treated_mean = Y[dataset.data["T"] == 1].mean()

        assert abs(control_mean) < 0.05, f"Control mean {control_mean:.4f} should be ~0"
        assert (
            abs(treated_mean - dataset.true_ate) < 0.05
        ), f"Treated mean {treated_mean:.4f} should be ~{dataset.true_ate}"

    def test_to_dict_serialization(self, simple_linear_dataset: SyntheticDataset):
        """Verify dataset can be serialized."""
        dataset = simple_linear_dataset
        d = dataset.to_dict()

        assert d["true_ate"] == 0.50
        assert d["dgp_name"] == "simple_linear"
        assert d["n_samples"] == 10000
        assert d["tolerance"] == 0.05
