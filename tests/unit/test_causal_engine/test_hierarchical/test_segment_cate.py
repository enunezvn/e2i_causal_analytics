"""Tests for SegmentCATECalculator.

B9.2: Segment-level CATE computation tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.hierarchical import (
    SegmentCATECalculator,
    SegmentCATEConfig,
    SegmentCATEResult,
)


class TestSegmentCATEConfig:
    """Test SegmentCATEConfig configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SegmentCATEConfig()

        assert config.estimator_type == "causal_forest"
        assert config.min_samples == 50
        assert config.compute_ci is True
        assert config.ci_confidence_level == 0.95
        assert config.n_bootstrap == 100
        assert config.random_state == 42

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SegmentCATEConfig(
            estimator_type="linear_dml",
            min_samples=100,
            compute_ci=False,
            ci_confidence_level=0.99,
            n_bootstrap=200,
            random_state=123,
        )

        assert config.estimator_type == "linear_dml"
        assert config.min_samples == 100
        assert config.compute_ci is False
        assert config.ci_confidence_level == 0.99
        assert config.n_bootstrap == 200
        assert config.random_state == 123

    def test_config_with_estimator_params(self) -> None:
        """Test configuration with estimator parameters."""
        config = SegmentCATEConfig(
            estimator_type="causal_forest",
            estimator_params={
                "n_estimators": 200,
                "min_samples_leaf": 20,
            },
        )

        assert config.estimator_params["n_estimators"] == 200
        assert config.estimator_params["min_samples_leaf"] == 20

    def test_valid_estimator_types(self) -> None:
        """Test valid estimator types."""
        valid_types = [
            "causal_forest",
            "linear_dml",
            "drlearner",
            "s_learner",
            "t_learner",
            "x_learner",
            "ols",
        ]

        for est_type in valid_types:
            config = SegmentCATEConfig(estimator_type=est_type)
            assert config.estimator_type == est_type

    def test_config_to_dict(self) -> None:
        """Test configuration to dict conversion."""
        config = SegmentCATEConfig(
            estimator_type="ols",
            min_samples=30,
        )
        d = config.to_dict()

        assert d["estimator_type"] == "ols"
        assert d["min_samples"] == 30
        assert "compute_ci" in d


class TestSegmentCATECalculator:
    """Test SegmentCATECalculator class."""

    @pytest.fixture
    def calculator(self) -> SegmentCATECalculator:
        """Create SegmentCATECalculator instance."""
        config = SegmentCATEConfig(
            estimator_type="ols",  # Fast for testing
            min_samples=30,
            random_state=42,
        )
        return SegmentCATECalculator(config)

    @pytest.fixture
    def segment_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Create sample segment data."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
            }
        )
        treatment = np.random.binomial(1, 0.5, n)
        # Simple linear outcome with treatment effect
        outcome = X["x1"].values * 0.3 + treatment * 0.15 + np.random.randn(n) * 0.1
        return X, treatment, outcome

    def test_create_calculator(self) -> None:
        """Test creating calculator with config."""
        config = SegmentCATEConfig(estimator_type="linear_dml")
        calculator = SegmentCATECalculator(config)

        assert calculator.config.estimator_type == "linear_dml"

    def test_create_calculator_default_config(self) -> None:
        """Test creating calculator with default config."""
        calculator = SegmentCATECalculator()

        assert calculator.config.estimator_type == "causal_forest"

    @pytest.mark.asyncio
    async def test_compute_ols(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test CATE computation with OLS estimator."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(estimator_type="ols", min_samples=30)
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=0,
            segment_name="test_segment",
        )

        assert isinstance(result, SegmentCATEResult)
        assert result.segment_id == 0
        assert result.segment_name == "test_segment"
        assert result.estimator_used == "ols"
        assert result.success is True

        # Should have CATE estimate
        assert result.cate_mean is not None
        assert isinstance(result.cate_mean, float)

        # Should have confidence interval
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower < result.cate_mean < result.ci_upper

        # Should have CATE values
        assert result.cate_values is not None
        assert len(result.cate_values) == len(treatment)

    @pytest.mark.asyncio
    async def test_compute_causal_forest(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test CATE computation with Causal Forest."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(
            estimator_type="causal_forest",
            min_samples=30,
            estimator_params={"n_estimators": 52},  # Must be divisible by subforest_size=4
        )
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=1,
            segment_name="forest_segment",
        )

        assert result.estimator_used == "causal_forest"
        assert result.success is True
        assert result.cate_mean is not None
        assert result.cate_values is not None

    @pytest.mark.asyncio
    async def test_compute_linear_dml(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test CATE computation with Linear DML."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(
            estimator_type="linear_dml",
            min_samples=30,
        )
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=2,
            segment_name="dml_segment",
        )

        assert result.estimator_used == "linear_dml"
        assert result.success is True
        assert result.cate_mean is not None

    @pytest.mark.asyncio
    async def test_compute_s_learner(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test CATE computation with S-Learner."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(
            estimator_type="s_learner",
            min_samples=30,
        )
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=3,
            segment_name="s_learner_segment",
        )

        assert result.estimator_used == "s_learner"
        assert result.success is True
        assert result.cate_mean is not None
        assert result.cate_values is not None

    @pytest.mark.asyncio
    async def test_compute_t_learner(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test CATE computation with T-Learner."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(
            estimator_type="t_learner",
            min_samples=30,
        )
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=4,
            segment_name="t_learner_segment",
        )

        assert result.estimator_used == "t_learner"
        assert result.success is True
        assert result.cate_mean is not None

    @pytest.mark.asyncio
    async def test_compute_x_learner(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test CATE computation with X-Learner."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(
            estimator_type="x_learner",
            min_samples=30,
        )
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=5,
            segment_name="x_learner_segment",
        )

        assert result.estimator_used == "x_learner"
        assert result.success is True
        assert result.cate_mean is not None

    @pytest.mark.asyncio
    async def test_compute_drlearner(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test CATE computation with DRLearner."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(
            estimator_type="drlearner",
            min_samples=30,
        )
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=6,
            segment_name="drlearner_segment",
        )

        assert result.estimator_used == "drlearner"
        assert result.success is True
        assert result.cate_mean is not None

    @pytest.mark.asyncio
    async def test_compute_small_sample_fails(
        self,
    ) -> None:
        """Test CATE computation fails with too small sample."""
        np.random.seed(42)
        n = 20  # Below min_samples
        X = pd.DataFrame({"x1": np.random.randn(n)})
        treatment = np.random.binomial(1, 0.5, n)
        outcome = treatment * 0.2 + np.random.randn(n) * 0.1

        config = SegmentCATEConfig(
            estimator_type="ols",
            min_samples=50,  # Higher than n
        )
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=0,
            segment_name="small",
        )

        # Should fail due to insufficient samples
        assert result.success is False
        assert "Insufficient samples" in result.error_message
        assert result.n_samples == n

    @pytest.mark.asyncio
    async def test_compute_insufficient_treatment_fails(
        self,
    ) -> None:
        """Test CATE computation fails with insufficient treatment/control."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({"x1": np.random.randn(n)})
        treatment = np.zeros(n, dtype=int)  # All control, no treatment
        treatment[:5] = 1  # Only 5 treated
        outcome = treatment * 0.2 + np.random.randn(n) * 0.1

        config = SegmentCATEConfig(
            estimator_type="ols",
            min_samples=30,
        )
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=0,
            segment_name="unbalanced",
        )

        # Should fail due to insufficient treatment/control
        assert result.success is False
        assert "Insufficient treatment/control" in result.error_message

    @pytest.mark.asyncio
    async def test_compute_records_latency(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test that computation records latency."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(estimator_type="ols", min_samples=30)
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=0,
            segment_name="latency_test",
        )

        assert result.estimation_time_ms is not None
        assert result.estimation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_compute_with_numpy_array(
        self,
    ) -> None:
        """Test CATE computation with numpy array input."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)  # numpy array instead of DataFrame
        treatment = np.random.binomial(1, 0.5, n)
        outcome = X[:, 0] * 0.3 + treatment * 0.15 + np.random.randn(n) * 0.1

        config = SegmentCATEConfig(estimator_type="ols", min_samples=30)
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=0,
            segment_name="numpy_test",
        )

        assert result.success is True
        assert result.cate_mean is not None
        assert result.n_samples == n

    @pytest.mark.asyncio
    async def test_compute_records_treatment_control_counts(
        self,
        segment_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test that computation records treatment and control counts."""
        X, treatment, outcome = segment_data

        config = SegmentCATEConfig(estimator_type="ols", min_samples=30)
        calculator = SegmentCATECalculator(config)

        result = await calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=0,
            segment_name="count_test",
        )

        assert result.n_treated > 0
        assert result.n_control > 0
        assert result.n_treated + result.n_control == result.n_samples


class TestSegmentCATEResult:
    """Test SegmentCATEResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a segment CATE result."""
        cate_values = np.array([0.1, 0.12, 0.15, 0.08])
        result = SegmentCATEResult(
            segment_id=0,
            segment_name="test",
            success=True,
            cate_mean=0.11,
            cate_std=0.02,
            ci_lower=0.07,
            ci_upper=0.15,
            cate_values=cate_values,
            n_samples=4,
            n_treated=2,
            n_control=2,
            estimator_used="causal_forest",
            estimation_time_ms=100,
        )

        assert result.segment_id == 0
        assert result.segment_name == "test"
        assert result.success is True
        assert result.cate_mean == 0.11
        assert result.cate_std == 0.02
        assert result.ci_lower == 0.07
        assert result.ci_upper == 0.15
        assert len(result.cate_values) == 4
        assert result.n_samples == 4
        assert result.n_treated == 2
        assert result.n_control == 2
        assert result.estimator_used == "causal_forest"
        assert result.estimation_time_ms == 100

    def test_result_defaults(self) -> None:
        """Test result with default values."""
        result = SegmentCATEResult(
            segment_id=1,
            segment_name="default_test",
            success=True,
        )

        assert result.cate_mean is None
        assert result.cate_values is None
        assert result.estimation_time_ms == 0.0
        assert result.error_message is None

    def test_result_failed(self) -> None:
        """Test failed result."""
        result = SegmentCATEResult(
            segment_id=2,
            segment_name="failed_test",
            success=False,
            error_message="Estimation failed: singular matrix",
            n_samples=50,
            n_treated=25,
            n_control=25,
        )

        assert result.success is False
        assert "singular matrix" in result.error_message
        assert result.cate_mean is None
        assert result.ci_lower is None

    def test_result_to_dict(self) -> None:
        """Test result to dict conversion."""
        result = SegmentCATEResult(
            segment_id=0,
            segment_name="dict_test",
            success=True,
            cate_mean=0.15,
            n_samples=100,
            estimator_used="ols",
        )

        d = result.to_dict()
        assert d["segment_id"] == 0
        assert d["segment_name"] == "dict_test"
        assert d["success"] is True
        assert d["cate_mean"] == 0.15
        assert d["n_samples"] == 100
        assert d["estimator_used"] == "ols"


class TestMultipleEstimatorComparison:
    """Test comparing results across different estimators."""

    @pytest.fixture
    def consistent_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Create data with known treatment effect."""
        np.random.seed(42)
        n = 300
        X = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
            }
        )
        treatment = np.random.binomial(1, 0.5, n)
        # Known treatment effect of ~0.20
        outcome = X["x1"].values * 0.3 + treatment * 0.20 + np.random.randn(n) * 0.05
        return X, treatment, outcome

    @pytest.mark.asyncio
    async def test_estimators_agree_on_direction(
        self,
        consistent_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Test that different estimators agree on effect direction."""
        X, treatment, outcome = consistent_data

        estimators = ["ols", "s_learner", "t_learner"]
        results = []

        for est_type in estimators:
            config = SegmentCATEConfig(
                estimator_type=est_type,
                min_samples=30,
            )
            calculator = SegmentCATECalculator(config)

            result = await calculator.compute(
                X=X,
                treatment=treatment,
                outcome=outcome,
                segment_id=0,
                segment_name=f"{est_type}_test",
            )
            results.append(result)

        # All should show positive effect
        for result in results:
            assert result.success is True, f"{result.estimator_used} should succeed"
            assert result.cate_mean > 0, f"{result.estimator_used} should show positive effect"

        # Effects should be in reasonable range
        for result in results:
            assert 0.05 < result.cate_mean < 0.40, f"{result.estimator_used} effect out of range"
