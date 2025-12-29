"""Unit tests for uplift base classes.

Tests cover:
- UpliftConfig creation and serialization
- UpliftResult creation and validation
- BaseUpliftModel interface
"""

import numpy as np
import pytest

from src.causal_engine.uplift.base import (
    BaseUpliftModel,
    UpliftConfig,
    UpliftModelType,
    UpliftNormalization,
    UpliftResult,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config():
    """Create default uplift config."""
    return UpliftConfig()


@pytest.fixture
def custom_config():
    """Create custom uplift config."""
    return UpliftConfig(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=50,
        min_samples_treatment=5,
        n_reg=5,
        control_name="control",
        random_state=123,
        normalize_scores=True,
        normalization_method=UpliftNormalization.ZSCORE,
    )


@pytest.fixture
def sample_uplift_scores():
    """Create sample uplift scores."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def successful_result(sample_uplift_scores):
    """Create successful uplift result."""
    return UpliftResult(
        model_type=UpliftModelType.UPLIFT_RANDOM_FOREST,
        success=True,
        uplift_scores=sample_uplift_scores,
        ate=0.05,
        att=0.08,
        atc=0.02,
        ate_std=0.02,
        ate_ci_lower=0.01,
        ate_ci_upper=0.09,
        treatment_groups=["treatment"],
        feature_importances={"feature_1": 0.3, "feature_2": 0.7},
        estimation_time_ms=150.0,
    )


# =============================================================================
# UPLIFT CONFIG TESTS
# =============================================================================


class TestUpliftConfig:
    """Tests for UpliftConfig dataclass."""

    def test_default_values(self, default_config):
        """Test default configuration values."""
        assert default_config.n_estimators == 100
        assert default_config.max_depth is None
        assert default_config.min_samples_leaf == 100
        assert default_config.min_samples_treatment == 10
        assert default_config.n_reg == 10
        assert default_config.control_name == "control"
        assert default_config.random_state == 42
        assert default_config.normalize_scores is False
        assert default_config.normalization_method == UpliftNormalization.MINMAX

    def test_custom_values(self, custom_config):
        """Test custom configuration values."""
        assert custom_config.n_estimators == 50
        assert custom_config.max_depth == 10
        assert custom_config.min_samples_leaf == 50
        assert custom_config.min_samples_treatment == 5
        assert custom_config.random_state == 123
        assert custom_config.normalize_scores is True
        assert custom_config.normalization_method == UpliftNormalization.ZSCORE

    def test_to_dict(self, custom_config):
        """Test configuration serialization."""
        config_dict = custom_config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["n_estimators"] == 50
        assert config_dict["max_depth"] == 10
        assert config_dict["normalization_method"] == "zscore"
        assert config_dict["normalize_scores"] is True

    def test_evaluation_function_options(self):
        """Test different evaluation function options."""
        for eval_func in ["KL", "ED", "Chi", "CTS", "DDP"]:
            config = UpliftConfig(evaluationFunction=eval_func)
            assert config.evaluationFunction == eval_func


# =============================================================================
# UPLIFT RESULT TESTS
# =============================================================================


class TestUpliftResult:
    """Tests for UpliftResult dataclass."""

    def test_successful_result(self, successful_result):
        """Test successful result creation."""
        assert successful_result.success is True
        assert successful_result.model_type == UpliftModelType.UPLIFT_RANDOM_FOREST
        assert successful_result.uplift_scores is not None
        assert len(successful_result.uplift_scores) == 100
        assert successful_result.ate == 0.05
        assert successful_result.att == 0.08
        assert successful_result.atc == 0.02

    def test_failed_result(self):
        """Test failed result creation."""
        result = UpliftResult(
            model_type=UpliftModelType.UPLIFT_RANDOM_FOREST,
            success=False,
            error_message="Model fitting failed",
        )

        assert result.success is False
        assert result.error_message == "Model fitting failed"
        assert result.uplift_scores is None
        assert result.ate is None

    def test_result_validation_missing_scores(self):
        """Test result validation for successful result without scores."""
        with pytest.raises(ValueError, match="must include uplift_scores"):
            UpliftResult(
                model_type=UpliftModelType.UPLIFT_RANDOM_FOREST,
                success=True,
                uplift_scores=None,
            )

    def test_result_auto_error_message(self):
        """Test automatic error message for failed results."""
        result = UpliftResult(
            model_type=UpliftModelType.UPLIFT_RANDOM_FOREST,
            success=False,
        )
        assert result.error_message == "Unknown error"

    def test_to_dict(self, successful_result):
        """Test result serialization."""
        result_dict = successful_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["model_type"] == "uplift_random_forest"
        assert result_dict["success"] is True
        assert result_dict["ate"] == 0.05
        assert isinstance(result_dict["uplift_scores"], list)
        assert len(result_dict["uplift_scores"]) == 100

    def test_result_with_metadata(self):
        """Test result with custom metadata."""
        result = UpliftResult(
            model_type=UpliftModelType.UPLIFT_TREE,
            success=True,
            uplift_scores=np.array([0.1, 0.2, 0.3]),
            metadata={"custom_key": "custom_value", "n_iterations": 100},
        )

        assert result.metadata["custom_key"] == "custom_value"
        assert result.metadata["n_iterations"] == 100


# =============================================================================
# UPLIFT MODEL TYPE TESTS
# =============================================================================


class TestUpliftModelType:
    """Tests for UpliftModelType enum."""

    def test_model_types_exist(self):
        """Test all expected model types exist."""
        assert UpliftModelType.UPLIFT_RANDOM_FOREST.value == "uplift_random_forest"
        assert UpliftModelType.UPLIFT_TREE.value == "uplift_tree"
        assert UpliftModelType.UPLIFT_GRADIENT_BOOSTING.value == "uplift_gradient_boosting"
        assert UpliftModelType.CAUSAL_TREE.value == "causal_tree"


# =============================================================================
# UPLIFT NORMALIZATION TESTS
# =============================================================================


class TestUpliftNormalization:
    """Tests for UpliftNormalization enum."""

    def test_normalization_options(self):
        """Test all normalization options exist."""
        assert UpliftNormalization.NONE.value == "none"
        assert UpliftNormalization.MINMAX.value == "minmax"
        assert UpliftNormalization.ZSCORE.value == "zscore"


# =============================================================================
# BASE UPLIFT MODEL TESTS
# =============================================================================


class TestBaseUpliftModel:
    """Tests for BaseUpliftModel abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseUpliftModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseUpliftModel()

    def test_concrete_subclass_requirements(self):
        """Test that concrete subclass must implement abstract methods."""

        class IncompleteModel(BaseUpliftModel):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_concrete_subclass_works(self):
        """Test that complete concrete subclass can be instantiated."""

        class MockUpliftModel(BaseUpliftModel):
            @property
            def model_type(self):
                return UpliftModelType.UPLIFT_TREE

            def _create_model(self):
                return None

        model = MockUpliftModel()
        assert model.model_type == UpliftModelType.UPLIFT_TREE
        assert model.is_fitted is False
        assert model.config is not None

    def test_model_with_custom_config(self):
        """Test model initialization with custom config."""

        class MockUpliftModel(BaseUpliftModel):
            @property
            def model_type(self):
                return UpliftModelType.UPLIFT_TREE

            def _create_model(self):
                return None

        config = UpliftConfig(n_estimators=200, max_depth=15)
        model = MockUpliftModel(config)

        assert model.config.n_estimators == 200
        assert model.config.max_depth == 15
