"""Benchmark: Confounded Moderate - Adjustment Recovery Test.

Version: 4.3
Purpose: Verify causal estimator correctly adjusts for observed confounding

Data Generating Process:
    C ~ N(0, 1)
    T ~ Bernoulli(sigmoid(0.4 * C))
    Y = 0.30 * T + 0.40 * C + epsilon
    True ATE = +0.30

The naive estimator will be biased (confounding bias).
Correct adjustment for C should recover the true ATE.

Reference: docs/E2I_Causal_Validation_Protocol.html
"""

import pytest

from tests.synthetic.conftest import (
    SyntheticDataset,
    estimate_ate_adjusted,
    estimate_ate_naive,
    generate_confounded_moderate,
)


class TestConfoundedModerateBenchmark:
    """Benchmark tests for confounded moderate DGP."""

    def test_naive_estimate_biased(self, confounded_moderate_dataset: SyntheticDataset):
        """Verify naive estimate is biased (expected behavior).

        This is a meta-test - the naive estimate SHOULD be biased.
        If it's not, the DGP may not have proper confounding.
        """
        dataset = confounded_moderate_dataset

        naive_ate = estimate_ate_naive(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
        )

        bias = abs(naive_ate - dataset.true_ate)

        # Naive should be biased by at least 0.10 (substantial confounding)
        assert bias > 0.10, (
            f"Naive estimate {naive_ate:.4f} is unexpectedly close to "
            f"true ATE {dataset.true_ate:.4f}. DGP may lack confounding."
        )

    def test_adjusted_estimate_accurate(self, confounded_moderate_dataset: SyntheticDataset):
        """CI/CD test: Adjusted estimate should recover true effect.

        This is the main benchmark test - can we recover the true effect
        after adjusting for observed confounders?
        """
        dataset = confounded_moderate_dataset

        adjusted_ate = estimate_ate_adjusted(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        error = abs(adjusted_ate - dataset.true_ate)

        assert error < dataset.tolerance, (
            f"Adjusted ATE estimate {adjusted_ate:.4f} not within tolerance "
            f"of true ATE {dataset.true_ate:.4f} (error: {error:.4f}, "
            f"tolerance: {dataset.tolerance:.4f})"
        )

    def test_adjustment_reduces_bias(self, confounded_moderate_dataset: SyntheticDataset):
        """CI/CD test: Adjustment should substantially reduce bias."""
        dataset = confounded_moderate_dataset

        naive_ate = estimate_ate_naive(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
        )

        adjusted_ate = estimate_ate_adjusted(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        naive_error = abs(naive_ate - dataset.true_ate)
        adjusted_error = abs(adjusted_ate - dataset.true_ate)

        # Adjusted should have at least 50% less error
        assert adjusted_error < naive_error * 0.5, (
            f"Adjustment didn't reduce bias enough. "
            f"Naive error: {naive_error:.4f}, Adjusted error: {adjusted_error:.4f}"
        )

    def test_confounder_treatment_correlation(self, confounded_moderate_dataset: SyntheticDataset):
        """Verify confounder is correlated with treatment."""
        dataset = confounded_moderate_dataset
        data = dataset.data

        # Calculate correlation between C and T
        correlation = data["C"].corr(data["T"])

        # Should have positive correlation (higher C → higher P(T=1))
        assert correlation > 0.15, f"Confounder-treatment correlation {correlation:.4f} is too weak"

    def test_confounder_outcome_correlation(self, confounded_moderate_dataset: SyntheticDataset):
        """Verify confounder is correlated with outcome (controlling for T)."""
        dataset = confounded_moderate_dataset
        data = dataset.data

        # Check within treatment groups
        for t_val in [0, 1]:
            subset = data[data["T"] == t_val]
            correlation = subset["C"].corr(subset["Y"])

            assert abs(correlation) > 0.30, (
                f"Confounder-outcome correlation {correlation:.4f} in T={t_val} group is too weak"
            )

    @pytest.mark.parametrize(
        "confounder_strength",
        [0.2, 0.4, 0.6, 0.8],
    )
    def test_various_confounder_strengths(self, confounder_strength: float):
        """CI/CD test: Verify recovery across confounding strengths."""
        dataset = generate_confounded_moderate(
            n=10000,
            true_ate=0.30,
            confounder_strength=confounder_strength,
            seed=42,
        )

        adjusted_ate = estimate_ate_adjusted(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        error = abs(adjusted_ate - dataset.true_ate)

        # Allow slightly more tolerance for stronger confounding
        tolerance = dataset.tolerance + (confounder_strength * 0.02)

        assert error < tolerance, (
            f"Failed to recover true ATE with confounder_strength={confounder_strength:.2f} "
            f"(estimated: {adjusted_ate:.4f}, error: {error:.4f})"
        )

    def test_reproducibility(self):
        """Verify DGP is reproducible with same seed."""
        dataset1 = generate_confounded_moderate(n=1000, seed=42)
        dataset2 = generate_confounded_moderate(n=1000, seed=42)

        pd_testing = pytest.importorskip("pandas.testing")
        pd_testing.assert_frame_equal(dataset1.data, dataset2.data)


class TestConfoundedModerateDatasetProperties:
    """Test properties of the confounded moderate dataset."""

    def test_dataset_structure(self, confounded_moderate_dataset: SyntheticDataset):
        """Verify dataset has expected structure."""
        dataset = confounded_moderate_dataset

        assert "T" in dataset.data.columns
        assert "Y" in dataset.data.columns
        assert "C" in dataset.data.columns
        assert len(dataset.data) == dataset.n_samples
        assert dataset.dgp_name == "confounded_moderate"
        assert dataset.confounder_cols == ["C"]

    def test_confounder_distribution(self, confounded_moderate_dataset: SyntheticDataset):
        """Verify confounder has expected distribution (N(0,1))."""
        dataset = confounded_moderate_dataset
        C = dataset.data["C"]

        # Check mean and std
        assert abs(C.mean()) < 0.05, f"Confounder mean {C.mean():.4f} should be ~0"
        assert abs(C.std() - 1.0) < 0.05, f"Confounder std {C.std():.4f} should be ~1"

    def test_treatment_proportion_varies_with_confounder(
        self, confounded_moderate_dataset: SyntheticDataset
    ):
        """Verify treatment probability varies with confounder."""
        dataset = confounded_moderate_dataset
        data = dataset.data

        # Split by confounder terciles
        low_c = data[data["C"] < -0.67]["T"].mean()
        high_c = data[data["C"] > 0.67]["T"].mean()

        # Higher C should have higher treatment probability
        assert high_c > low_c + 0.10, (
            f"Treatment probability should increase with C "
            f"(low_C: {low_c:.3f}, high_C: {high_c:.3f})"
        )


class TestConfoundingBiasDirection:
    """Test that confounding bias has expected direction."""

    def test_positive_confounding_positive_bias(self):
        """Positive confounding should create upward bias."""
        # C → T (positive), C → Y (positive) → upward bias
        dataset = generate_confounded_moderate(
            n=10000,
            true_ate=0.30,
            confounder_strength=0.40,
            seed=42,
        )

        naive_ate = estimate_ate_naive(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
        )

        # Naive should overestimate (positive bias)
        assert naive_ate > dataset.true_ate, (
            f"Expected upward bias: naive {naive_ate:.4f} > true {dataset.true_ate:.4f}"
        )

    def test_bias_magnitude_scales_with_confounding(self):
        """Stronger confounding should create larger bias."""
        weak = generate_confounded_moderate(n=10000, confounder_strength=0.2, seed=42)
        strong = generate_confounded_moderate(n=10000, confounder_strength=0.6, seed=42)

        weak_naive = estimate_ate_naive(weak.data, "T", "Y")
        strong_naive = estimate_ate_naive(strong.data, "T", "Y")

        weak_bias = abs(weak_naive - weak.true_ate)
        strong_bias = abs(strong_naive - strong.true_ate)

        assert strong_bias > weak_bias, (
            f"Stronger confounding should create larger bias "
            f"(weak: {weak_bias:.4f}, strong: {strong_bias:.4f})"
        )
