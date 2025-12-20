"""Benchmark: Heterogeneous CATE - Segment-Level Treatment Effects.

Version: 4.3
Purpose: Verify causal estimator correctly estimates heterogeneous effects

Data Generating Process:
    C1, C2 ~ N(0, 1)
    Segment = tercile(C1)  # low, medium, high
    T ~ Bernoulli(sigmoid(0.3*C1 + 0.2*C2))

    CATE(low)    = 0.10
    CATE(medium) = 0.20
    CATE(high)   = 0.40

    Y = CATE(segment) * T + 0.3*C1 + 0.2*C2 + epsilon

This tests the ability to detect treatment effect heterogeneity
across segments - crucial for personalized intervention strategies.

Reference: docs/E2I_Causal_Validation_Protocol.html
"""

import pytest
import numpy as np

from tests.synthetic.conftest import (
    SyntheticDataset,
    generate_heterogeneous_cate,
    estimate_ate_naive,
    estimate_ate_adjusted,
    estimate_cate_by_segment,
)


class TestHeterogeneousCateBenchmark:
    """Benchmark tests for heterogeneous CATE DGP."""

    def test_overall_ate_accurate(
        self, heterogeneous_cate_dataset: SyntheticDataset
    ):
        """CI/CD test: Overall ATE should be recovered with adjustment."""
        dataset = heterogeneous_cate_dataset

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

    def test_cate_ordering_preserved(
        self, heterogeneous_cate_dataset: SyntheticDataset
    ):
        """CI/CD test: CATE ordering should be preserved (low < medium < high)."""
        dataset = heterogeneous_cate_dataset

        estimated_cates = estimate_cate_by_segment(
            dataset.data,
            segment_col="segment",
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        # Verify ordering: low < medium < high
        assert estimated_cates["low"] < estimated_cates["medium"], (
            f"CATE(low)={estimated_cates['low']:.4f} should be < "
            f"CATE(medium)={estimated_cates['medium']:.4f}"
        )
        assert estimated_cates["medium"] < estimated_cates["high"], (
            f"CATE(medium)={estimated_cates['medium']:.4f} should be < "
            f"CATE(high)={estimated_cates['high']:.4f}"
        )

    def test_cate_magnitudes_accurate(
        self, heterogeneous_cate_dataset: SyntheticDataset
    ):
        """CI/CD test: Individual CATE magnitudes should be accurate."""
        dataset = heterogeneous_cate_dataset
        true_cates = dataset.true_cate

        estimated_cates = estimate_cate_by_segment(
            dataset.data,
            segment_col="segment",
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        for segment, true_cate in true_cates.items():
            estimated_cate = estimated_cates[segment]
            error = abs(estimated_cate - true_cate)

            # Use slightly larger tolerance for segment-level estimates
            segment_tolerance = dataset.tolerance * 1.5

            assert error < segment_tolerance, (
                f"CATE({segment}) estimate {estimated_cate:.4f} not within tolerance "
                f"of true CATE {true_cate:.4f} (error: {error:.4f})"
            )

    def test_heterogeneity_detection(
        self, heterogeneous_cate_dataset: SyntheticDataset
    ):
        """CI/CD test: Should detect significant heterogeneity."""
        dataset = heterogeneous_cate_dataset

        estimated_cates = estimate_cate_by_segment(
            dataset.data,
            segment_col="segment",
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        # Compute range of CATEs
        cate_values = list(estimated_cates.values())
        cate_range = max(cate_values) - min(cate_values)

        # Range should be substantial (true range is 0.40 - 0.10 = 0.30)
        assert cate_range > 0.20, (
            f"CATE range {cate_range:.4f} too small to indicate heterogeneity"
        )

    def test_naive_misses_true_ate(
        self, heterogeneous_cate_dataset: SyntheticDataset
    ):
        """Verify naive estimate is biased (expected behavior)."""
        dataset = heterogeneous_cate_dataset

        naive_ate = estimate_ate_naive(
            dataset.data,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
        )

        bias = abs(naive_ate - dataset.true_ate)

        # Should have substantial bias due to confounding
        assert bias > 0.05, (
            f"Naive estimate {naive_ate:.4f} unexpectedly close to "
            f"true ATE {dataset.true_ate:.4f}"
        )


class TestHeterogeneousCateDatasetProperties:
    """Test properties of the heterogeneous CATE dataset."""

    def test_dataset_structure(
        self, heterogeneous_cate_dataset: SyntheticDataset
    ):
        """Verify dataset has expected structure."""
        dataset = heterogeneous_cate_dataset

        assert "T" in dataset.data.columns
        assert "Y" in dataset.data.columns
        assert "C1" in dataset.data.columns
        assert "C2" in dataset.data.columns
        assert "segment" in dataset.data.columns
        assert len(dataset.data) == dataset.n_samples
        assert dataset.dgp_name == "heterogeneous_cate"

    def test_segment_distribution(
        self, heterogeneous_cate_dataset: SyntheticDataset
    ):
        """Verify segments have expected distribution based on fixed cutoffs.

        Note: The DGP uses fixed cutoffs (-0.67, 0.67) on a standard normal,
        which produces approximately: low=25%, medium=50%, high=25%
        """
        dataset = heterogeneous_cate_dataset
        segment_counts = dataset.data["segment"].value_counts(normalize=True)

        # Expected proportions based on N(0,1) with cutoffs at -0.67 and 0.67
        expected = {"low": 0.25, "medium": 0.50, "high": 0.25}

        for segment in ["low", "medium", "high"]:
            proportion = segment_counts.get(segment, 0)
            expected_prop = expected[segment]
            # Allow 10% relative tolerance
            assert abs(proportion - expected_prop) < 0.05, (
                f"Segment '{segment}' proportion {proportion:.3f} "
                f"outside expected range (~{expected_prop:.2f})"
            )

    def test_true_cates_defined(
        self, heterogeneous_cate_dataset: SyntheticDataset
    ):
        """Verify true CATEs are defined for all segments."""
        dataset = heterogeneous_cate_dataset

        assert dataset.true_cate is not None
        assert "low" in dataset.true_cate
        assert "medium" in dataset.true_cate
        assert "high" in dataset.true_cate

        # Check expected values
        assert abs(dataset.true_cate["low"] - 0.10) < 0.01
        assert abs(dataset.true_cate["medium"] - 0.20) < 0.01
        assert abs(dataset.true_cate["high"] - 0.40) < 0.01


class TestHeterogeneousCateCustomSegments:
    """Test with custom segment effects."""

    @pytest.mark.parametrize(
        "segment_effects",
        [
            {"low": -0.20, "medium": 0.00, "high": 0.30},  # Strong heterogeneity
            {"low": 0.00, "medium": 0.05, "high": 0.10},   # Weak heterogeneity
            {"low": 0.10, "medium": 0.10, "high": 0.10},   # No heterogeneity (homogeneous)
        ],
    )
    def test_custom_segment_effects(self, segment_effects: dict):
        """Test various heterogeneity patterns."""
        dataset = generate_heterogeneous_cate(
            n=30000,
            base_effect=0.20,
            segment_effects=segment_effects,
            seed=42,
        )

        estimated_cates = estimate_cate_by_segment(
            dataset.data,
            segment_col="segment",
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        for segment, modifier in segment_effects.items():
            true_cate = 0.20 + modifier
            estimated_cate = estimated_cates[segment]
            error = abs(estimated_cate - true_cate)

            # Allow larger tolerance for custom configurations
            tolerance = 0.10

            assert error < tolerance, (
                f"Custom CATE({segment}) estimate {estimated_cate:.4f} "
                f"too far from true {true_cate:.4f} (error: {error:.4f})"
            )

    def test_no_heterogeneity_detected_when_absent(self):
        """When CATEs are equal, should not detect heterogeneity."""
        # All segments have same effect
        dataset = generate_heterogeneous_cate(
            n=30000,
            base_effect=0.30,
            segment_effects={"low": 0.0, "medium": 0.0, "high": 0.0},
            seed=42,
        )

        estimated_cates = estimate_cate_by_segment(
            dataset.data,
            segment_col="segment",
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        cate_values = list(estimated_cates.values())
        cate_range = max(cate_values) - min(cate_values)

        # Range should be small (just noise)
        assert cate_range < 0.10, (
            f"CATE range {cate_range:.4f} too large for homogeneous effects"
        )


class TestHeterogeneousCateReproducibility:
    """Test reproducibility of heterogeneous DGP."""

    def test_reproducibility(self):
        """Verify DGP is reproducible with same seed."""
        dataset1 = generate_heterogeneous_cate(n=1000, seed=42)
        dataset2 = generate_heterogeneous_cate(n=1000, seed=42)

        pd_testing = pytest.importorskip("pandas.testing")
        pd_testing.assert_frame_equal(dataset1.data, dataset2.data)

    def test_different_seeds_different_data(self):
        """Verify different seeds produce different data."""
        dataset1 = generate_heterogeneous_cate(n=1000, seed=42)
        dataset2 = generate_heterogeneous_cate(n=1000, seed=123)

        assert not dataset1.data.equals(dataset2.data)

    def test_cate_stability_across_runs(self):
        """CATEs should be stable across different seeds (same DGP)."""
        cate_estimates = []

        for seed in range(5):
            dataset = generate_heterogeneous_cate(n=50000, seed=seed)
            cates = estimate_cate_by_segment(
                dataset.data,
                segment_col="segment",
                treatment_col=dataset.treatment_col,
                outcome_col=dataset.outcome_col,
                confounder_cols=dataset.confounder_cols,
            )
            cate_estimates.append(cates)

        # Check that high-segment CATE is consistently highest
        for cates in cate_estimates:
            assert cates["high"] > cates["low"], (
                "High segment should consistently have higher CATE"
            )


class TestHeterogeneousCateEdgeCases:
    """Test edge cases for heterogeneous CATE estimation."""

    def test_small_sample_warning(self):
        """Small samples should still work but with higher variance."""
        dataset = generate_heterogeneous_cate(n=3000, seed=42)

        # Should not raise, but estimates may be noisier
        estimated_cates = estimate_cate_by_segment(
            dataset.data,
            segment_col="segment",
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        # At minimum, ordering should still be preserved
        assert estimated_cates["high"] > estimated_cates["low"]

    def test_large_sample_precision(self):
        """Large samples should give precise estimates."""
        dataset = generate_heterogeneous_cate(n=100000, seed=42)

        estimated_cates = estimate_cate_by_segment(
            dataset.data,
            segment_col="segment",
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            confounder_cols=dataset.confounder_cols,
        )

        # With large sample, should be very close
        for segment, true_cate in dataset.true_cate.items():
            error = abs(estimated_cates[segment] - true_cate)
            assert error < 0.03, (
                f"Large sample CATE({segment}) error {error:.4f} too high"
            )
