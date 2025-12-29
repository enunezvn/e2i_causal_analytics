"""Unit tests for Uplift Gradient Boosting.

Tests cover:
- Model instantiation with different meta-learners
- Gradient boosting configuration
- Fit and predict workflow
- XGBoost vs sklearn selection
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.uplift.base import UpliftModelType
from src.causal_engine.uplift.gradient_boosting import (
    GradientBoostingMetaLearner,
    GradientBoostingUpliftConfig,
    UpliftGradientBoosting,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 200

    X = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.randn(n),
    })
    treatment = np.random.binomial(1, 0.5, n)
    y = np.random.binomial(1, 0.5, n).astype(float)

    return X, treatment, y


@pytest.fixture
def t_learner_config():
    """Create T-Learner configuration."""
    return GradientBoostingUpliftConfig(
        n_estimators=10,
        max_depth=3,
        meta_learner=GradientBoostingMetaLearner.T_LEARNER,
        use_xgboost=False,  # Use sklearn for testing
        random_state=42,
    )


@pytest.fixture
def x_learner_config():
    """Create X-Learner configuration."""
    return GradientBoostingUpliftConfig(
        n_estimators=10,
        max_depth=3,
        meta_learner=GradientBoostingMetaLearner.X_LEARNER,
        use_xgboost=False,
        random_state=42,
    )


@pytest.fixture
def s_learner_config():
    """Create S-Learner configuration."""
    return GradientBoostingUpliftConfig(
        n_estimators=10,
        max_depth=3,
        meta_learner=GradientBoostingMetaLearner.S_LEARNER,
        use_xgboost=False,
        random_state=42,
    )


@pytest.fixture
def mock_causalml_modules():
    """Create mock CausalML module structure."""
    mock_causalml = MagicMock()
    mock_causalml_inference = MagicMock()
    mock_causalml_inference_meta = MagicMock()

    mock_causalml.inference = mock_causalml_inference
    mock_causalml.inference.meta = mock_causalml_inference_meta

    return {
        'causalml': mock_causalml,
        'causalml.inference': mock_causalml_inference,
        'causalml.inference.meta': mock_causalml_inference_meta,
    }


@pytest.fixture
def mock_meta_classifier(mock_causalml_modules):
    """Create mock meta-learner classifiers."""
    mock_clf = MagicMock()
    mock_clf.fit = MagicMock(return_value=mock_clf)
    mock_clf.predict = MagicMock(return_value=np.random.randn(200))

    # Mock internal treatment models with feature importances
    mock_model_0 = MagicMock()
    mock_model_0.feature_importances_ = np.array([0.3, 0.5, 0.2])
    mock_model_1 = MagicMock()
    mock_model_1.feature_importances_ = np.array([0.4, 0.4, 0.2])
    mock_clf.models_t = {"0": mock_model_0, "1": mock_model_1}

    mock_causalml_modules['causalml.inference.meta'].BaseTClassifier = MagicMock(
        return_value=mock_clf
    )
    mock_causalml_modules['causalml.inference.meta'].BaseXClassifier = MagicMock(
        return_value=mock_clf
    )
    mock_causalml_modules['causalml.inference.meta'].BaseSClassifier = MagicMock(
        return_value=mock_clf
    )

    return mock_clf


# =============================================================================
# GRADIENT BOOSTING UPLIFT CONFIG TESTS
# =============================================================================


class TestGradientBoostingUpliftConfig:
    """Tests for GradientBoostingUpliftConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GradientBoostingUpliftConfig()

        assert config.meta_learner == GradientBoostingMetaLearner.T_LEARNER
        assert config.learning_rate == 0.1
        assert config.subsample == 0.8
        assert config.use_xgboost is True
        assert config.colsample_bytree == 0.8

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GradientBoostingUpliftConfig(
            meta_learner=GradientBoostingMetaLearner.X_LEARNER,
            learning_rate=0.05,
            subsample=0.7,
            use_xgboost=False,
        )

        assert config.meta_learner == GradientBoostingMetaLearner.X_LEARNER
        assert config.learning_rate == 0.05
        assert config.subsample == 0.7
        assert config.use_xgboost is False

    def test_to_dict(self):
        """Test configuration serialization."""
        config = GradientBoostingUpliftConfig(
            meta_learner=GradientBoostingMetaLearner.S_LEARNER,
            learning_rate=0.2,
        )

        config_dict = config.to_dict()

        assert config_dict["meta_learner"] == "s_learner"
        assert config_dict["learning_rate"] == 0.2
        assert "use_xgboost" in config_dict


# =============================================================================
# GRADIENT BOOSTING META LEARNER ENUM TESTS
# =============================================================================


class TestGradientBoostingMetaLearner:
    """Tests for GradientBoostingMetaLearner enum."""

    def test_meta_learner_values(self):
        """Test all meta-learner types exist."""
        assert GradientBoostingMetaLearner.T_LEARNER.value == "t_learner"
        assert GradientBoostingMetaLearner.X_LEARNER.value == "x_learner"
        assert GradientBoostingMetaLearner.S_LEARNER.value == "s_learner"


# =============================================================================
# UPLIFT GRADIENT BOOSTING TESTS
# =============================================================================


class TestUpliftGradientBoosting:
    """Tests for UpliftGradientBoosting class."""

    def test_model_type(self, t_learner_config):
        """Test model type property returns correct type."""
        model = UpliftGradientBoosting(t_learner_config)
        assert model.model_type == UpliftModelType.UPLIFT_GRADIENT_BOOSTING

    def test_initialization_default_config(self):
        """Test model initialization with default config."""
        model = UpliftGradientBoosting()
        assert model.config is not None
        assert isinstance(model.config, GradientBoostingUpliftConfig)
        assert model.is_fitted is False

    def test_initialization_custom_config(self, x_learner_config):
        """Test model initialization with custom config."""
        model = UpliftGradientBoosting(x_learner_config)
        assert model.config.meta_learner == GradientBoostingMetaLearner.X_LEARNER
        assert model.config.n_estimators == 10

    def test_fit_t_learner(
        self, sample_data, t_learner_config, mock_causalml_modules, mock_meta_classifier
    ):
        """Test model fitting with T-Learner."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(t_learner_config)
            result = model.fit(X, treatment, y)

            assert result is model
            assert model.is_fitted is True
            mock_meta_classifier.fit.assert_called_once()

    def test_fit_x_learner(
        self, sample_data, x_learner_config, mock_causalml_modules, mock_meta_classifier
    ):
        """Test model fitting with X-Learner."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(x_learner_config)
            result = model.fit(X, treatment, y)

            assert result is model
            assert model.is_fitted is True

    def test_fit_s_learner(
        self, sample_data, s_learner_config, mock_causalml_modules, mock_meta_classifier
    ):
        """Test model fitting with S-Learner."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(s_learner_config)
            result = model.fit(X, treatment, y)

            assert result is model
            assert model.is_fitted is True

    def test_predict_before_fit_raises(self, sample_data, t_learner_config):
        """Test predict raises error if model not fitted."""
        X, _, _ = sample_data
        model = UpliftGradientBoosting(t_learner_config)

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)

    def test_predict_returns_array(
        self, sample_data, t_learner_config, mock_causalml_modules, mock_meta_classifier
    ):
        """Test predict returns numpy array."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(t_learner_config)
            model.fit(X, treatment, y)
            predictions = model.predict(X)

            assert isinstance(predictions, np.ndarray)

    def test_estimate_returns_result(
        self, sample_data, t_learner_config, mock_causalml_modules, mock_meta_classifier
    ):
        """Test estimate returns UpliftResult."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(t_learner_config)
            result = model.estimate(X, treatment, y)

            assert result.success is True
            assert result.model_type == UpliftModelType.UPLIFT_GRADIENT_BOOSTING
            assert result.uplift_scores is not None
            assert result.ate is not None
            assert "meta_learner" in result.metadata

    def test_estimate_with_propensity_scores(
        self, sample_data, t_learner_config, mock_causalml_modules, mock_meta_classifier
    ):
        """Test estimate with propensity scores."""
        X, treatment, y = sample_data
        p = np.full(len(X), 0.5)

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(t_learner_config)
            result = model.estimate(X, treatment, y, p=p)

            assert result.success is True

    def test_estimate_handles_errors(
        self, sample_data, t_learner_config, mock_causalml_modules
    ):
        """Test estimate handles fitting errors gracefully."""
        X, treatment, y = sample_data

        # Setup mock to raise error
        mock_clf = MagicMock()
        mock_clf.fit = MagicMock(side_effect=ValueError("Fitting failed"))
        mock_causalml_modules['causalml.inference.meta'].BaseTClassifier = MagicMock(
            return_value=mock_clf
        )

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(t_learner_config)
            result = model.estimate(X, treatment, y)

            assert result.success is False
            assert "Fitting failed" in result.error_message

    def test_create_base_learner_sklearn(self, t_learner_config):
        """Test sklearn gradient boosting base learner creation."""
        model = UpliftGradientBoosting(t_learner_config)
        base_learner = model._create_base_learner()

        from sklearn.ensemble import GradientBoostingClassifier
        assert isinstance(base_learner, GradientBoostingClassifier)

    def test_create_base_learner_xgboost(self):
        """Test XGBoost base learner creation when available."""
        config = GradientBoostingUpliftConfig(use_xgboost=True)
        model = UpliftGradientBoosting(config)

        # Create mock xgboost module
        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {'xgboost': mock_xgb}):
            base_learner = model._create_base_learner()
            mock_xgb.XGBClassifier.assert_called_once()

    def test_feature_importances_from_models(
        self, sample_data, t_learner_config, mock_causalml_modules, mock_meta_classifier
    ):
        """Test feature importance extraction from meta-learner."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(t_learner_config)
            result = model.estimate(X, treatment, y)

            # Should get feature importances from averaging internal models
            assert result.feature_importances is not None

    def test_causalml_import_error(self, t_learner_config):
        """Test import error handling when CausalML not available."""
        model = UpliftGradientBoosting(t_learner_config)

        with patch.dict(sys.modules, {"causalml": None, "causalml.inference.meta": None}):
            with pytest.raises(ImportError, match="CausalML"):
                model._create_model()


# =============================================================================
# INTEGRATION-STYLE TESTS (with sklearn base learners)
# =============================================================================


class TestGradientBoostingIntegration:
    """Integration-style tests using actual sklearn base learners."""

    def test_full_workflow_t_learner(
        self, sample_data, t_learner_config, mock_causalml_modules, mock_meta_classifier
    ):
        """Test complete workflow with T-Learner."""
        X, treatment, y = sample_data

        # Split data
        X_train, X_test = X[:150], X[150:]
        treatment_train, treatment_test = treatment[:150], treatment[150:]
        y_train, y_test = y[:150], y[150:]

        # Update mock to return correct size for test set
        mock_meta_classifier.predict = MagicMock(return_value=np.random.randn(50))

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftGradientBoosting(t_learner_config)
            result = model.estimate(
                X_train, treatment_train, y_train,
                X_test=X_test, treatment_test=treatment_test, y_test=y_test
            )

            assert result.success is True
            assert result.metadata["n_samples_train"] == 150
            assert result.metadata["n_samples_test"] == 50
            assert result.metadata["meta_learner"] == "t_learner"
