"""Tests for NestedConfidenceInterval.

B9.3: Nested confidence interval computation tests.
"""

import numpy as np
import pytest

from src.causal_engine.hierarchical import (
    AggregationMethod,
    NestedCIConfig,
    NestedCIResult,
    NestedConfidenceInterval,
)
from src.causal_engine.hierarchical.nested_ci import SegmentEstimate


class TestNestedCIConfig:
    """Test NestedCIConfig configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = NestedCIConfig()

        assert config.confidence_level == 0.95
        assert config.aggregation_method == AggregationMethod.VARIANCE_WEIGHTED
        assert config.min_segment_size == 30
        assert config.bootstrap_iterations == 1000
        assert config.bootstrap_random_state == 42

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = NestedCIConfig(
            confidence_level=0.99,
            aggregation_method=AggregationMethod.SAMPLE_WEIGHTED,
            min_segment_size=50,
            bootstrap_iterations=500,
        )

        assert config.confidence_level == 0.99
        assert config.aggregation_method == AggregationMethod.SAMPLE_WEIGHTED
        assert config.min_segment_size == 50
        assert config.bootstrap_iterations == 500


class TestAggregationMethod:
    """Test AggregationMethod enum."""

    def test_all_methods_defined(self) -> None:
        """Test all aggregation methods are defined."""
        methods = [m.value for m in AggregationMethod]

        assert "sample_weighted" in methods
        assert "variance_weighted" in methods
        assert "equal" in methods
        assert "bootstrap" in methods


class TestSegmentEstimate:
    """Test SegmentEstimate dataclass."""

    def test_create_segment_estimate(self) -> None:
        """Test creating a segment estimate."""
        estimate = SegmentEstimate(
            segment_id=0,
            segment_name="low_uplift",
            ate=0.10,
            ate_std=0.02,
            ci_lower=0.06,
            ci_upper=0.14,
            sample_size=500,
        )

        assert estimate.segment_id == 0
        assert estimate.segment_name == "low_uplift"
        assert estimate.ate == 0.10
        assert estimate.ate_std == 0.02
        assert estimate.ci_lower == 0.06
        assert estimate.ci_upper == 0.14
        assert estimate.sample_size == 500
        assert estimate.cate is None

    def test_segment_estimate_with_cate(self) -> None:
        """Test segment estimate with individual CATEs."""
        cate = np.array([0.08, 0.10, 0.12, 0.09, 0.11])
        estimate = SegmentEstimate(
            segment_id=1,
            segment_name="high_uplift",
            ate=0.10,
            ate_std=0.015,
            ci_lower=0.07,
            ci_upper=0.13,
            sample_size=5,
            cate=cate,
        )

        assert estimate.cate is not None
        assert len(estimate.cate) == 5


class TestNestedConfidenceInterval:
    """Test NestedConfidenceInterval computations."""

    @pytest.fixture
    def calculator(self) -> NestedConfidenceInterval:
        """Create NestedConfidenceInterval instance."""
        return NestedConfidenceInterval(NestedCIConfig())

    @pytest.fixture
    def three_segments(self) -> list[SegmentEstimate]:
        """Create three segment estimates for testing."""
        return [
            SegmentEstimate(
                segment_id=0,
                segment_name="low",
                ate=0.10,
                ate_std=0.02,
                ci_lower=0.06,
                ci_upper=0.14,
                sample_size=500,
            ),
            SegmentEstimate(
                segment_id=1,
                segment_name="medium",
                ate=0.15,
                ate_std=0.03,
                ci_lower=0.09,
                ci_upper=0.21,
                sample_size=300,
            ),
            SegmentEstimate(
                segment_id=2,
                segment_name="high",
                ate=0.25,
                ate_std=0.04,
                ci_lower=0.17,
                ci_upper=0.33,
                sample_size=200,
            ),
        ]

    @pytest.fixture
    def homogeneous_segments(self) -> list[SegmentEstimate]:
        """Create segments with similar effects (low heterogeneity)."""
        return [
            SegmentEstimate(0, "a", 0.12, 0.02, 0.08, 0.16, 400),
            SegmentEstimate(1, "b", 0.13, 0.02, 0.09, 0.17, 350),
            SegmentEstimate(2, "c", 0.11, 0.02, 0.07, 0.15, 450),
        ]

    @pytest.fixture
    def heterogeneous_segments(self) -> list[SegmentEstimate]:
        """Create segments with very different effects (high heterogeneity)."""
        return [
            SegmentEstimate(0, "x", 0.05, 0.01, 0.03, 0.07, 400),
            SegmentEstimate(1, "y", 0.20, 0.02, 0.16, 0.24, 350),
            SegmentEstimate(2, "z", 0.35, 0.03, 0.29, 0.41, 300),
        ]

    def test_compute_empty_segments(
        self,
        calculator: NestedConfidenceInterval,
    ) -> None:
        """Test with empty segment list."""
        result = calculator.compute([])

        assert result.aggregate_ate == 0.0
        assert result.n_segments_included == 0
        assert "No valid segments" in result.warnings[0]

    def test_compute_single_segment(
        self,
        calculator: NestedConfidenceInterval,
    ) -> None:
        """Test with single segment."""
        segment = SegmentEstimate(0, "only", 0.15, 0.03, 0.09, 0.21, 500)
        result = calculator.compute([segment])

        assert result.aggregate_ate == 0.15
        assert result.aggregate_std == 0.03
        assert result.aggregate_ci_lower == 0.09
        assert result.aggregate_ci_upper == 0.21
        assert result.n_segments_included == 1
        assert "Single segment" in result.warnings[0]

    def test_compute_variance_weighted(
        self,
        calculator: NestedConfidenceInterval,
        three_segments: list[SegmentEstimate],
    ) -> None:
        """Test variance-weighted aggregation (default)."""
        result = calculator.compute(three_segments)

        # Should produce valid aggregate
        assert 0 < result.aggregate_ate < 0.30
        assert result.aggregate_ci_lower < result.aggregate_ate < result.aggregate_ci_upper
        assert result.aggregate_std > 0
        assert result.n_segments_included == 3
        assert result.aggregation_method == "variance_weighted"

        # Contributions should sum to 1
        total_weight = sum(result.segment_contributions.values())
        assert abs(total_weight - 1.0) < 0.001

        # Lower-variance segments should have higher weight
        # (low has std=0.02, high has std=0.04)
        assert result.segment_contributions["low"] > result.segment_contributions["high"]

    def test_compute_sample_weighted(
        self,
        three_segments: list[SegmentEstimate],
    ) -> None:
        """Test sample-weighted aggregation."""
        config = NestedCIConfig(aggregation_method=AggregationMethod.SAMPLE_WEIGHTED)
        calculator = NestedConfidenceInterval(config)
        result = calculator.compute(three_segments)

        assert result.aggregation_method == "sample_weighted"
        assert result.n_segments_included == 3

        # Larger segments should have higher weight
        # (low has n=500, high has n=200)
        assert result.segment_contributions["low"] > result.segment_contributions["high"]

        # Sample-weighted average: (0.10*500 + 0.15*300 + 0.25*200) / 1000
        expected_ate = (0.10 * 500 + 0.15 * 300 + 0.25 * 200) / 1000
        assert abs(result.aggregate_ate - expected_ate) < 0.001

    def test_compute_equal_weighted(
        self,
        three_segments: list[SegmentEstimate],
    ) -> None:
        """Test equal-weighted aggregation."""
        config = NestedCIConfig(aggregation_method=AggregationMethod.EQUAL)
        calculator = NestedConfidenceInterval(config)
        result = calculator.compute(three_segments)

        assert result.aggregation_method == "equal"

        # All weights should be equal
        for weight in result.segment_contributions.values():
            assert abs(weight - 1 / 3) < 0.001

        # Simple average
        expected_ate = (0.10 + 0.15 + 0.25) / 3
        assert abs(result.aggregate_ate - expected_ate) < 0.001

    def test_compute_bootstrap(
        self,
        three_segments: list[SegmentEstimate],
    ) -> None:
        """Test bootstrap aggregation."""
        config = NestedCIConfig(
            aggregation_method=AggregationMethod.BOOTSTRAP,
            bootstrap_iterations=500,
            bootstrap_random_state=42,
        )
        calculator = NestedConfidenceInterval(config)
        result = calculator.compute(three_segments)

        assert result.aggregation_method == "bootstrap"
        assert result.n_segments_included == 3

        # Bootstrap should produce reasonable estimate
        assert 0.05 < result.aggregate_ate < 0.30
        assert result.aggregate_ci_lower < result.aggregate_ate < result.aggregate_ci_upper

    def test_heterogeneity_low(
        self,
        calculator: NestedConfidenceInterval,
        homogeneous_segments: list[SegmentEstimate],
    ) -> None:
        """Test heterogeneity statistics with similar segments."""
        result = calculator.compute(homogeneous_segments)

        # Low heterogeneity expected
        assert result.i_squared < 50  # Should be low
        assert result.tau_squared >= 0

    def test_heterogeneity_high(
        self,
        calculator: NestedConfidenceInterval,
        heterogeneous_segments: list[SegmentEstimate],
    ) -> None:
        """Test heterogeneity statistics with different segments."""
        result = calculator.compute(heterogeneous_segments)

        # High heterogeneity expected
        assert result.i_squared > 50  # Should be high
        assert any("heterogeneity" in w.lower() for w in result.warnings)

    def test_min_segment_size_filtering(
        self,
    ) -> None:
        """Test that small segments are filtered out."""
        config = NestedCIConfig(min_segment_size=100)
        calculator = NestedConfidenceInterval(config)

        segments = [
            SegmentEstimate(0, "large", 0.10, 0.02, 0.06, 0.14, 500),
            SegmentEstimate(1, "small", 0.20, 0.05, 0.10, 0.30, 50),  # Too small
        ]

        result = calculator.compute(segments)

        # Only large segment should be included
        assert result.n_segments_included == 1
        assert "Excluded 1 segments" in result.warnings[0]
        assert result.aggregate_ate == 0.10  # Only the large segment

    def test_confidence_level_90(
        self,
        three_segments: list[SegmentEstimate],
    ) -> None:
        """Test with 90% confidence level."""
        config = NestedCIConfig(confidence_level=0.90)
        calculator = NestedConfidenceInterval(config)
        result_90 = calculator.compute(three_segments)

        config_95 = NestedCIConfig(confidence_level=0.95)
        calculator_95 = NestedConfidenceInterval(config_95)
        result_95 = calculator_95.compute(three_segments)

        # 90% CI should be narrower than 95% CI
        width_90 = result_90.aggregate_ci_upper - result_90.aggregate_ci_lower
        width_95 = result_95.aggregate_ci_upper - result_95.aggregate_ci_lower
        assert width_90 < width_95

    def test_random_effects(
        self,
        calculator: NestedConfidenceInterval,
        heterogeneous_segments: list[SegmentEstimate],
    ) -> None:
        """Test random-effects meta-analysis."""
        result = calculator.compute_random_effects(heterogeneous_segments)

        assert result.aggregation_method == "random_effects"
        assert result.n_segments_included == 3

        # Random effects should produce wider CI than fixed effects
        calculator.compute(heterogeneous_segments)

        # Aggregate should be between segment extremes
        effects = [s.ate for s in heterogeneous_segments]
        assert min(effects) <= result.aggregate_ate <= max(effects)


class TestNestedCIResult:
    """Test NestedCIResult dataclass."""

    def test_result_fields(self) -> None:
        """Test all result fields are present."""
        result = NestedCIResult(
            aggregate_ate=0.15,
            aggregate_ci_lower=0.10,
            aggregate_ci_upper=0.20,
            aggregate_std=0.025,
            confidence_level=0.95,
            aggregation_method="variance_weighted",
        )

        assert result.aggregate_ate == 0.15
        assert result.aggregate_ci_lower == 0.10
        assert result.aggregate_ci_upper == 0.20
        assert result.aggregate_std == 0.025
        assert result.confidence_level == 0.95
        assert result.aggregation_method == "variance_weighted"
        assert result.segment_contributions == {}
        assert result.warnings == []

    def test_result_with_heterogeneity(self) -> None:
        """Test result with heterogeneity measures."""
        result = NestedCIResult(
            aggregate_ate=0.15,
            aggregate_ci_lower=0.10,
            aggregate_ci_upper=0.20,
            aggregate_std=0.025,
            confidence_level=0.95,
            aggregation_method="variance_weighted",
            heterogeneity_measure=15.5,
            i_squared=68.4,
            tau_squared=0.002,
        )

        assert result.heterogeneity_measure == 15.5
        assert result.i_squared == 68.4
        assert result.tau_squared == 0.002


class TestBootstrapWithCATEs:
    """Test bootstrap aggregation with individual CATEs."""

    def test_bootstrap_from_cates(self) -> None:
        """Test bootstrap using individual-level CATEs."""
        config = NestedCIConfig(
            aggregation_method=AggregationMethod.BOOTSTRAP,
            bootstrap_iterations=200,
            bootstrap_random_state=42,
        )
        calculator = NestedConfidenceInterval(config)

        # Create segments with individual CATEs
        np.random.seed(42)
        segments = [
            SegmentEstimate(
                segment_id=0,
                segment_name="seg_a",
                ate=0.10,
                ate_std=0.02,
                ci_lower=0.06,
                ci_upper=0.14,
                sample_size=100,
                cate=np.random.normal(0.10, 0.05, 100),
            ),
            SegmentEstimate(
                segment_id=1,
                segment_name="seg_b",
                ate=0.15,
                ate_std=0.03,
                ci_lower=0.09,
                ci_upper=0.21,
                sample_size=80,
                cate=np.random.normal(0.15, 0.06, 80),
            ),
        ]

        result = calculator.compute(segments)

        assert result.aggregation_method == "bootstrap"
        assert result.n_segments_included == 2
        assert result.total_sample_size == 180

        # Bootstrap estimate should be reasonable
        assert 0.05 < result.aggregate_ate < 0.25
        assert result.aggregate_ci_lower < result.aggregate_ate < result.aggregate_ci_upper
