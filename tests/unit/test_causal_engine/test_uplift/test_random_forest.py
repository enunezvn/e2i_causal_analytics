"""Unit tests for Uplift Random Forest.

Tests cover:
- Model instantiation
- Model type property
- Fit and predict workflow
- Feature importance extraction
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.uplift.base import UpliftConfig, UpliftModelType
from src.causal_engine.uplift.random_forest import UpliftRandomForest, UpliftTree


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
def default_config():
    """Create default configuration."""
    return UpliftConfig(
        n_estimators=10,  # Small for testing
        max_depth=3,
        min_samples_leaf=10,
        min_samples_treatment=5,
        random_state=42,
    )


@pytest.fixture
def mock_causalml_modules():
    """Create mock CausalML module structure."""
    mock_causalml = MagicMock()
    mock_causalml_inference = MagicMock()
    mock_causalml_inference_tree = MagicMock()

    mock_causalml.inference = mock_causalml_inference
    mock_causalml.inference.tree = mock_causalml_inference_tree

    return {
        'causalml': mock_causalml,
        'causalml.inference': mock_causalml_inference,
        'causalml.inference.tree': mock_causalml_inference_tree,
    }


@pytest.fixture
def mock_classifier(mock_causalml_modules):
    """Create mock UpliftRandomForestClassifier."""
    mock_clf = MagicMock()
    mock_clf.fit = MagicMock(return_value=mock_clf)
    mock_clf.predict = MagicMock(
        return_value={"treatment_1": np.random.randn(200)}
    )
    mock_clf.feature_importances_ = np.array([0.3, 0.5, 0.2])

    mock_causalml_modules['causalml.inference.tree'].UpliftRandomForestClassifier = MagicMock(
        return_value=mock_clf
    )
    mock_causalml_modules['causalml.inference.tree'].UpliftTreeClassifier = MagicMock(
        return_value=mock_clf
    )

    return mock_clf


# =============================================================================
# UPLIFT RANDOM FOREST TESTS
# =============================================================================


class TestUpliftRandomForest:
    """Tests for UpliftRandomForest class."""

    def test_model_type(self, default_config):
        """Test model type property returns correct type."""
        model = UpliftRandomForest(default_config)
        assert model.model_type == UpliftModelType.UPLIFT_RANDOM_FOREST

    def test_initialization_default_config(self):
        """Test model initialization with default config."""
        model = UpliftRandomForest()
        assert model.config is not None
        assert model.is_fitted is False
        assert model.model is None

    def test_initialization_custom_config(self, default_config):
        """Test model initialization with custom config."""
        model = UpliftRandomForest(default_config)
        assert model.config.n_estimators == 10
        assert model.config.max_depth == 3

    def test_fit(self, sample_data, default_config, mock_causalml_modules, mock_classifier):
        """Test model fitting with mocked CausalML."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftRandomForest(default_config)
            result = model.fit(X, treatment, y)

            assert result is model  # Returns self
            assert model.is_fitted is True
            mock_classifier.fit.assert_called_once()

    def test_predict_before_fit_raises(self, sample_data, default_config):
        """Test predict raises error if model not fitted."""
        X, _, _ = sample_data
        model = UpliftRandomForest(default_config)

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)

    def test_predict_returns_array(
        self, sample_data, default_config, mock_causalml_modules, mock_classifier
    ):
        """Test predict returns numpy array."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftRandomForest(default_config)
            model.fit(X, treatment, y)
            predictions = model.predict(X)

            assert isinstance(predictions, np.ndarray)

    def test_estimate_returns_result(
        self, sample_data, default_config, mock_causalml_modules, mock_classifier
    ):
        """Test estimate returns UpliftResult."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftRandomForest(default_config)
            result = model.estimate(X, treatment, y)

            assert result.success is True
            assert result.model_type == UpliftModelType.UPLIFT_RANDOM_FOREST
            assert result.uplift_scores is not None
            assert result.ate is not None

    def test_estimate_with_test_data(
        self, sample_data, default_config, mock_causalml_modules, mock_classifier
    ):
        """Test estimate with separate test data."""
        X, treatment, y = sample_data

        # Split data
        X_train, X_test = X[:150], X[150:]
        treatment_train, treatment_test = treatment[:150], treatment[150:]
        y_train, y_test = y[:150], y[150:]

        # Update mock to return correct size for test set
        mock_classifier.predict = MagicMock(
            return_value={"treatment_1": np.random.randn(50)}
        )

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftRandomForest(default_config)
            result = model.estimate(
                X_train, treatment_train, y_train,
                X_test=X_test, treatment_test=treatment_test, y_test=y_test
            )

            assert result.success is True
            assert result.metadata["n_samples_train"] == 150
            assert result.metadata["n_samples_test"] == 50

    def test_estimate_handles_errors(
        self, sample_data, default_config, mock_causalml_modules
    ):
        """Test estimate handles fitting errors gracefully."""
        X, treatment, y = sample_data

        # Setup mock to raise error
        mock_clf = MagicMock()
        mock_clf.fit = MagicMock(side_effect=ValueError("Fitting failed"))
        mock_causalml_modules['causalml.inference.tree'].UpliftRandomForestClassifier = MagicMock(
            return_value=mock_clf
        )

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftRandomForest(default_config)
            result = model.estimate(X, treatment, y)

            assert result.success is False
            assert "Fitting failed" in result.error_message

    def test_feature_importances(
        self, sample_data, default_config, mock_causalml_modules, mock_classifier
    ):
        """Test feature importance extraction."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftRandomForest(default_config)
            result = model.estimate(X, treatment, y)

            assert result.feature_importances is not None
            assert "feature_1" in result.feature_importances
            assert "feature_2" in result.feature_importances
            assert "feature_3" in result.feature_importances

    def test_causalml_import_error(self, default_config):
        """Test import error handling when CausalML not available."""
        model = UpliftRandomForest(default_config)

        with patch.dict(sys.modules, {"causalml": None, "causalml.inference.tree": None}):
            with pytest.raises(ImportError, match="CausalML"):
                model._create_model()


# =============================================================================
# UPLIFT TREE TESTS
# =============================================================================


class TestUpliftTree:
    """Tests for UpliftTree class."""

    def test_model_type(self, default_config):
        """Test model type property returns correct type."""
        model = UpliftTree(default_config)
        assert model.model_type == UpliftModelType.UPLIFT_TREE

    def test_initialization(self, default_config):
        """Test model initialization."""
        model = UpliftTree(default_config)
        assert model.config is not None
        assert model.is_fitted is False

    def test_fit(self, sample_data, default_config, mock_causalml_modules, mock_classifier):
        """Test tree model fitting."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftTree(default_config)
            result = model.fit(X, treatment, y)

            assert result is model
            assert model.is_fitted is True

    def test_predict(self, sample_data, default_config, mock_causalml_modules, mock_classifier):
        """Test tree model prediction."""
        X, treatment, y = sample_data

        with patch.dict(sys.modules, mock_causalml_modules):
            model = UpliftTree(default_config)
            model.fit(X, treatment, y)
            predictions = model.predict(X)

            assert isinstance(predictions, np.ndarray)
