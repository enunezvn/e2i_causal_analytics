"""Tests for HierarchicalAnalyzer.

B9.1: Hierarchical analysis orchestration tests.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.causal_engine.hierarchical import (
    HierarchicalAnalyzer,
    HierarchicalConfig,
    HierarchicalResult,
    SegmentResult,
)
from src.causal_engine.hierarchical.analyzer import SegmentationMethod


class TestHierarchicalConfig:
    """Test HierarchicalConfig configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = HierarchicalConfig()

        assert config.n_segments == 3
        assert config.segmentation_method == SegmentationMethod.QUANTILE
        assert config.min_segment_size == 50
        assert config.estimator_type == "causal_forest"
        assert config.ci_confidence_level == 0.95
        assert config.compute_nested_ci is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = HierarchicalConfig(
            n_segments=5,
            segmentation_method=SegmentationMethod.KMEANS,
            min_segment_size=100,
            estimator_type="linear_dml",
            ci_confidence_level=0.99,
            compute_nested_ci=False,
        )

        assert config.n_segments == 5
        assert config.segmentation_method == SegmentationMethod.KMEANS
        assert config.min_segment_size == 100
        assert config.estimator_type == "linear_dml"
        assert config.ci_confidence_level == 0.99
        assert config.compute_nested_ci is False

    def test_config_to_dict(self) -> None:
        """Test configuration to dict conversion."""
        config = HierarchicalConfig(n_segments=4, estimator_type="ols")
        d = config.to_dict()

        assert d["n_segments"] == 4
        assert d["estimator_type"] == "ols"
        assert "segmentation_method" in d


class TestSegmentationMethod:
    """Test SegmentationMethod enum."""

    def test_all_methods_defined(self) -> None:
        """Test all segmentation methods are defined."""
        methods = [m.value for m in SegmentationMethod]

        assert "quantile" in methods
        assert "threshold" in methods
        assert "kmeans" in methods
        assert "tree" in methods


class TestHierarchicalAnalyzer:
    """Test HierarchicalAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> HierarchicalAnalyzer:
        """Create HierarchicalAnalyzer instance."""
        config = HierarchicalConfig(
            n_segments=3,
            min_segment_size=30,  # Low for testing
        )
        return HierarchicalAnalyzer(config)

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 300
        X = pd.DataFrame({
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randn(n),
        })
        treatment = np.random.binomial(1, 0.5, n)
        # Outcome with heterogeneous treatment effect
        base_effect = 0.1 + 0.05 * X["feature_1"].values
        outcome = (
            X["feature_1"].values * 0.3
            + X["feature_2"].values * 0.2
            + treatment * base_effect
            + np.random.randn(n) * 0.1
        )
        return X, treatment, outcome

    @pytest.fixture
    def uplift_scores(self) -> np.ndarray:
        """Create sample uplift scores."""
        np.random.seed(42)
        return np.random.uniform(0, 0.3, 300)

    def test_create_analyzer(self) -> None:
        """Test creating analyzer with config."""
        config = HierarchicalConfig(n_segments=4)
        analyzer = HierarchicalAnalyzer(config)

        assert analyzer.config.n_segments == 4

    def test_create_analyzer_default_config(self) -> None:
        """Test creating analyzer with default config."""
        analyzer = HierarchicalAnalyzer()

        assert analyzer.config.n_segments == 3

    def test_segment_by_quantile(
        self,
        analyzer: HierarchicalAnalyzer,
        uplift_scores: np.ndarray,
    ) -> None:
        """Test quantile-based segmentation."""
        segments, names = analyzer._segment_by_quantile(uplift_scores)

        # Should have 3 segments (based on config)
        assert len(np.unique(segments)) == 3

        # All values should be assigned
        assert len(segments) == len(uplift_scores)

        # Segments should be 0, 1, 2
        assert set(np.unique(segments)) == {0, 1, 2}

        # Should have named segments
        assert len(names) == 3
        assert "low" in names[0].lower()
        assert "high" in names[2].lower()

    def test_segment_by_threshold(
        self,
        analyzer: HierarchicalAnalyzer,
        uplift_scores: np.ndarray,
    ) -> None:
        """Test threshold-based segmentation."""
        # Use config's quantile_thresholds (default is None -> median split)
        segments, names = analyzer._segment_by_threshold(uplift_scores)

        # Should have 2 segments (median split)
        assert len(np.unique(segments)) <= 2

        # All values should be assigned
        assert len(segments) == len(uplift_scores)

    def test_segment_by_threshold_with_custom_thresholds(
        self,
        uplift_scores: np.ndarray,
    ) -> None:
        """Test threshold-based segmentation with custom thresholds."""
        config = HierarchicalConfig(
            segmentation_method=SegmentationMethod.THRESHOLD,
            quantile_thresholds=[0.1, 0.2],
        )
        analyzer = HierarchicalAnalyzer(config)

        segments, names = analyzer._segment_by_threshold(uplift_scores)

        # Should have 3 segments (below 0.1, 0.1-0.2, above 0.2)
        assert len(np.unique(segments)) <= 3
        assert len(segments) == len(uplift_scores)

    def test_segment_by_kmeans(
        self,
        analyzer: HierarchicalAnalyzer,
        uplift_scores: np.ndarray,
    ) -> None:
        """Test k-means segmentation."""
        segments, names = analyzer._segment_by_kmeans(uplift_scores)

        # Should have 3 segments (based on config)
        assert len(np.unique(segments)) == 3

        # All values should be assigned
        assert len(segments) == len(uplift_scores)

        # Should have named segments
        assert len(names) == 3

    @pytest.mark.asyncio
    async def test_analyze_with_uplift_scores(
        self,
        analyzer: HierarchicalAnalyzer,
        sample_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        uplift_scores: np.ndarray,
    ) -> None:
        """Test full hierarchical analysis with pre-computed uplift."""
        X, treatment, outcome = sample_data

        result = await analyzer.analyze(
            X=X,
            treatment=treatment,
            outcome=outcome,
            uplift_scores=uplift_scores,
        )

        # Should have result
        assert isinstance(result, HierarchicalResult)
        assert result.success is True
        assert result.n_segments >= 1
        assert result.n_total_samples == len(treatment)

        # Should have segment results
        assert len(result.segment_results) >= 1

        # Should have aggregate
        assert result.overall_ate is not None
        assert result.overall_ate_ci_lower is not None
        assert result.overall_ate_ci_upper is not None

    @pytest.mark.asyncio
    async def test_analyze_computes_uplift(
        self,
        sample_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test analysis computes uplift if not provided."""
        # Skip if CausalML is not installed
        pytest.importorskip("causalml", reason="CausalML not installed")

        X, treatment, outcome = sample_data

        config = HierarchicalConfig(
            n_segments=2,
            min_segment_size=50,
        )
        analyzer = HierarchicalAnalyzer(config)

        # Don't provide uplift_scores - should compute internally
        result = await analyzer.analyze(
            X=X,
            treatment=treatment,
            outcome=outcome,
        )

        # Should still succeed
        assert isinstance(result, HierarchicalResult)
        assert result.success is True
        assert result.n_segments >= 1
        assert result.uplift_model_used is not None

    @pytest.mark.asyncio
    async def test_analyze_different_estimators(
        self,
        sample_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        uplift_scores: np.ndarray,
    ) -> None:
        """Test analysis with different EconML estimators."""
        X, treatment, outcome = sample_data

        estimators = ["causal_forest", "linear_dml", "ols"]

        for estimator in estimators:
            config = HierarchicalConfig(
                n_segments=2,
                min_segment_size=50,
                estimator_type=estimator,
            )
            analyzer = HierarchicalAnalyzer(config)

            result = await analyzer.analyze(
                X=X,
                treatment=treatment,
                outcome=outcome,
                uplift_scores=uplift_scores,
            )

            assert result.success is True
            assert len(result.segment_results) >= 1

    @pytest.mark.asyncio
    async def test_analyze_with_numpy_features(
        self,
        sample_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        uplift_scores: np.ndarray,
    ) -> None:
        """Test analysis with numpy array features."""
        X, treatment, outcome = sample_data

        config = HierarchicalConfig(
            n_segments=2,
            min_segment_size=50,
        )
        analyzer = HierarchicalAnalyzer(config)

        result = await analyzer.analyze(
            X=X.values,  # Numpy array
            treatment=treatment,
            outcome=outcome,
            uplift_scores=uplift_scores,
            feature_names=["f1", "f2", "f3"],
        )

        assert result.success is True


class TestSegmentResult:
    """Test SegmentResult dataclass."""

    def test_create_segment_result(self) -> None:
        """Test creating a segment result."""
        result = SegmentResult(
            segment_id=0,
            segment_name="low_uplift",
            n_samples=500,
            uplift_range=(0.0, 0.1),
            cate_mean=0.10,
            cate_std=0.02,
            cate_ci_lower=0.06,
            cate_ci_upper=0.14,
        )

        assert result.segment_id == 0
        assert result.segment_name == "low_uplift"
        assert result.n_samples == 500
        assert result.uplift_range == (0.0, 0.1)
        assert result.cate_mean == 0.10
        assert result.cate_std == 0.02
        assert result.success is True

    def test_segment_result_with_cate_values(self) -> None:
        """Test segment result with individual CATEs."""
        cate_values = np.array([0.08, 0.10, 0.12, 0.09, 0.11])
        result = SegmentResult(
            segment_id=1,
            segment_name="high_uplift",
            n_samples=5,
            uplift_range=(0.2, 0.3),
            cate_mean=0.10,
            cate_std=0.015,
            cate_ci_lower=0.07,
            cate_ci_upper=0.13,
            cate_values=cate_values,
        )

        assert result.cate_values is not None
        assert len(result.cate_values) == 5

    def test_segment_result_failure(self) -> None:
        """Test segment result with failure."""
        result = SegmentResult(
            segment_id=2,
            segment_name="failed",
            n_samples=10,
            uplift_range=(0.1, 0.15),
            success=False,
            error_message="Insufficient samples",
        )

        assert result.success is False
        assert result.error_message == "Insufficient samples"
        assert result.cate_mean is None

    def test_segment_result_to_dict(self) -> None:
        """Test segment result to dict conversion."""
        result = SegmentResult(
            segment_id=0,
            segment_name="test",
            n_samples=100,
            uplift_range=(0.05, 0.15),
            cate_mean=0.10,
            cate_std=0.02,
            cate_ci_lower=0.06,
            cate_ci_upper=0.14,
        )

        d = result.to_dict()
        assert d["segment_id"] == 0
        assert d["segment_name"] == "test"
        assert d["n_samples"] == 100
        assert d["cate_mean"] == 0.10


class TestHierarchicalResult:
    """Test HierarchicalResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a hierarchical result."""
        segments = [
            SegmentResult(0, "low", 500, (0.0, 0.1), 0.10, 0.02, 0.06, 0.14),
            SegmentResult(1, "high", 300, (0.2, 0.3), 0.18, 0.03, 0.12, 0.24),
        ]

        result = HierarchicalResult(
            success=True,
            treatment_var="treatment",
            outcome_var="outcome",
            n_total_samples=800,
            n_segments=2,
            segment_results=segments,
            overall_ate=0.13,
            overall_ate_ci_lower=0.09,
            overall_ate_ci_upper=0.17,
            segment_heterogeneity=0.35,
        )

        assert result.success is True
        assert result.n_segments == 2
        assert len(result.segment_results) == 2
        assert result.overall_ate == 0.13
        assert result.n_total_samples == 800
        assert result.segment_heterogeneity == 0.35

    def test_result_get_segment_by_name(self) -> None:
        """Test getting segment by name."""
        segments = [
            SegmentResult(0, "low_uplift", 500, (0.0, 0.1), 0.10),
            SegmentResult(1, "high_uplift", 300, (0.2, 0.3), 0.18),
        ]

        result = HierarchicalResult(
            success=True,
            treatment_var="t",
            outcome_var="y",
            n_total_samples=800,
            n_segments=2,
            segment_results=segments,
        )

        low = result.get_segment_by_name("low_uplift")
        assert low is not None
        assert low.segment_id == 0

        high = result.get_high_uplift_segment()
        assert high is not None
        assert high.segment_id == 1

    def test_result_get_segment_summary(self) -> None:
        """Test getting segment summary DataFrame."""
        segments = [
            SegmentResult(0, "low", 500, (0.0, 0.1), 0.10, 0.02, 0.06, 0.14),
            SegmentResult(1, "high", 300, (0.2, 0.3), 0.18, 0.03, 0.12, 0.24),
        ]

        result = HierarchicalResult(
            success=True,
            treatment_var="t",
            outcome_var="y",
            n_total_samples=800,
            n_segments=2,
            segment_results=segments,
        )

        summary = result.get_segment_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "segment" in summary.columns
        assert "n_samples" in summary.columns
        assert "cate_mean" in summary.columns

    def test_result_to_dict(self) -> None:
        """Test result to dict conversion."""
        result = HierarchicalResult(
            success=True,
            treatment_var="marketing_spend",
            outcome_var="conversion",
            n_total_samples=1000,
            n_segments=3,
            segment_results=[],
            overall_ate=0.15,
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["treatment_var"] == "marketing_spend"
        assert d["n_total_samples"] == 1000
        assert d["overall_ate"] == 0.15

    def test_result_failed(self) -> None:
        """Test failed hierarchical result."""
        result = HierarchicalResult(
            success=False,
            treatment_var="t",
            outcome_var="y",
            n_total_samples=100,
            n_segments=0,
            segment_results=[],
            errors=["Analysis failed: insufficient data"],
        )

        assert result.success is False
        assert len(result.errors) == 1
        assert "insufficient data" in result.errors[0]
