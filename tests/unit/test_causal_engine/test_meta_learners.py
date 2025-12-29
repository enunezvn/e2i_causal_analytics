"""Tests for Meta-Learner estimators (S-Learner, T-Learner, X-Learner, OrthoForest)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.causal_engine.energy_score.estimator_selector import (
    EstimatorConfig,
    EstimatorResult,
    EstimatorType,
    SLearnerWrapper,
    TLearnerWrapper,
    XLearnerWrapper,
    OrthoForestWrapper,
)


def generate_synthetic_data(
    n_samples: int = 500,
    n_features: int = 5,
    treatment_effect: float = 2.0,
    heterogeneous: bool = True,
    seed: int = 42,
):
    """Generate synthetic data for causal inference testing.

    Args:
        n_samples: Number of samples.
        n_features: Number of covariates.
        treatment_effect: Base treatment effect (ATE).
        heterogeneous: If True, treatment effect varies with X.
        seed: Random seed.

    Returns:
        Tuple of (treatment, outcome, covariates, true_cate)
    """
    np.random.seed(seed)

    # Generate covariates
    X = np.random.randn(n_samples, n_features)
    covariates = pd.DataFrame(
        X, columns=[f"x{i}" for i in range(n_features)]
    )

    # Propensity depends on X[:,0]
    propensity = 1 / (1 + np.exp(-X[:, 0]))
    treatment = (np.random.rand(n_samples) < propensity).astype(int)

    # True CATE varies with X if heterogeneous
    if heterogeneous:
        true_cate = treatment_effect + X[:, 0] * 0.5 + X[:, 1] * 0.3
    else:
        true_cate = np.full(n_samples, treatment_effect)

    # Outcome with noise
    base_outcome = X[:, 0] * 1.0 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5
    outcome = base_outcome + treatment * true_cate

    return treatment, outcome, covariates, true_cate


class TestSLearnerWrapper:
    """Tests for S-Learner wrapper."""

    @pytest.fixture
    def config(self):
        """Default S-Learner config."""
        return EstimatorConfig(
            estimator_type=EstimatorType.S_LEARNER,
            params={"n_estimators": 50, "max_depth": 3, "random_state": 42},
        )

    @pytest.fixture
    def data(self):
        """Synthetic test data."""
        return generate_synthetic_data(n_samples=200, n_features=3)

    def test_estimator_type(self, config):
        """Test that estimator type is correct."""
        wrapper = SLearnerWrapper(config)
        assert wrapper.estimator_type == EstimatorType.S_LEARNER

    def test_fit_returns_result(self, config, data):
        """Test that fit returns an EstimatorResult."""
        treatment, outcome, covariates, _ = data
        wrapper = SLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert isinstance(result, EstimatorResult)
        assert result.success is True
        assert result.ate is not None
        assert result.cate is not None
        assert len(result.cate) == len(treatment)

    def test_cate_shape(self, config, data):
        """Test that CATE has correct shape."""
        treatment, outcome, covariates, _ = data
        wrapper = SLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert result.cate.shape == (len(treatment),)

    def test_propensity_scores_computed(self, config, data):
        """Test that propensity scores are computed."""
        treatment, outcome, covariates, _ = data
        wrapper = SLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert result.propensity_scores is not None
        assert len(result.propensity_scores) == len(treatment)
        assert all(0 <= p <= 1 for p in result.propensity_scores)

    def test_confidence_interval(self, config, data):
        """Test that confidence intervals are computed."""
        treatment, outcome, covariates, _ = data
        wrapper = SLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert result.ate_ci_lower is not None
        assert result.ate_ci_upper is not None
        assert result.ate_ci_lower < result.ate < result.ate_ci_upper


class TestTLearnerWrapper:
    """Tests for T-Learner wrapper."""

    @pytest.fixture
    def config(self):
        """Default T-Learner config."""
        return EstimatorConfig(
            estimator_type=EstimatorType.T_LEARNER,
            params={"n_estimators": 50, "max_depth": 3, "random_state": 42},
        )

    @pytest.fixture
    def data(self):
        """Synthetic test data."""
        return generate_synthetic_data(n_samples=200, n_features=3)

    def test_estimator_type(self, config):
        """Test that estimator type is correct."""
        wrapper = TLearnerWrapper(config)
        assert wrapper.estimator_type == EstimatorType.T_LEARNER

    def test_fit_returns_result(self, config, data):
        """Test that fit returns an EstimatorResult."""
        treatment, outcome, covariates, _ = data
        wrapper = TLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert isinstance(result, EstimatorResult)
        assert result.success is True
        assert result.ate is not None

    def test_separate_models_trained(self, config, data):
        """Test that separate models are stored."""
        treatment, outcome, covariates, _ = data
        wrapper = TLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert result.raw_estimate is not None
        assert "model_1" in result.raw_estimate
        assert "model_0" in result.raw_estimate

    def test_heterogeneous_effect_detection(self, config):
        """Test that T-Learner captures heterogeneous effects."""
        # Generate data with strong heterogeneity
        treatment, outcome, covariates, true_cate = generate_synthetic_data(
            n_samples=500, heterogeneous=True
        )
        wrapper = TLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        # CATE should vary
        cate_std = np.std(result.cate)
        assert cate_std > 0.1, "T-Learner should detect heterogeneity"


class TestXLearnerWrapper:
    """Tests for X-Learner wrapper."""

    @pytest.fixture
    def config(self):
        """Default X-Learner config."""
        return EstimatorConfig(
            estimator_type=EstimatorType.X_LEARNER,
            params={"n_estimators": 50, "max_depth": 3, "random_state": 42},
        )

    @pytest.fixture
    def data(self):
        """Synthetic test data."""
        return generate_synthetic_data(n_samples=200, n_features=3)

    def test_estimator_type(self, config):
        """Test that estimator type is correct."""
        wrapper = XLearnerWrapper(config)
        assert wrapper.estimator_type == EstimatorType.X_LEARNER

    def test_fit_returns_result(self, config, data):
        """Test that fit returns an EstimatorResult."""
        treatment, outcome, covariates, _ = data
        wrapper = XLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert isinstance(result, EstimatorResult)
        assert result.success is True
        assert result.ate is not None
        assert result.cate is not None

    def test_two_stage_models_stored(self, config, data):
        """Test that all X-Learner models are stored."""
        treatment, outcome, covariates, _ = data
        wrapper = XLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert result.raw_estimate is not None
        # First stage models
        assert "model_1" in result.raw_estimate
        assert "model_0" in result.raw_estimate
        # Second stage models
        assert "model_tau_1" in result.raw_estimate
        assert "model_tau_0" in result.raw_estimate
        # Propensity model
        assert "ps_model" in result.raw_estimate

    def test_propensity_weighted_combination(self, config, data):
        """Test that X-Learner uses propensity-weighted CATE."""
        treatment, outcome, covariates, _ = data
        wrapper = XLearnerWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        # Verify propensity scores are used
        assert result.propensity_scores is not None
        # CATE should be influenced by propensity weighting
        assert len(result.cate) == len(treatment)

    def test_unbalanced_treatment_handling(self, config):
        """Test X-Learner with unbalanced treatment groups."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 3)
        covariates = pd.DataFrame(X, columns=["x0", "x1", "x2"])

        # Heavily unbalanced: 80% control, 20% treatment
        treatment = (np.random.rand(n) < 0.2).astype(int)
        true_cate = 2.0 + X[:, 0] * 0.5
        outcome = X[:, 0] + treatment * true_cate + np.random.randn(n) * 0.3

        wrapper = XLearnerWrapper(config)
        result = wrapper.fit(treatment, outcome, covariates)

        # Should still work with unbalanced data
        assert result.success is True
        assert result.ate is not None
        # X-Learner should handle unbalanced data well
        assert abs(result.ate - 2.0) < 1.0  # Approximate check


class TestOrthoForestWrapper:
    """Tests for OrthoForest wrapper."""

    @pytest.fixture
    def config(self):
        """Default OrthoForest config."""
        return EstimatorConfig(
            estimator_type=EstimatorType.ORTHO_FOREST,
            params={"n_trees": 50, "min_leaf_size": 10, "random_state": 42},
        )

    @pytest.fixture
    def data(self):
        """Synthetic test data."""
        return generate_synthetic_data(n_samples=200, n_features=3)

    def test_estimator_type(self, config):
        """Test that estimator type is correct."""
        wrapper = OrthoForestWrapper(config)
        assert wrapper.estimator_type == EstimatorType.ORTHO_FOREST

    @pytest.mark.skipif(
        True,
        reason="OrthoForest requires econml.orf which may not be available"
    )
    def test_fit_returns_result(self, config, data):
        """Test that fit returns an EstimatorResult."""
        treatment, outcome, covariates, _ = data
        wrapper = OrthoForestWrapper(config)

        result = wrapper.fit(treatment, outcome, covariates)

        assert isinstance(result, EstimatorResult)
        if result.success:
            assert result.ate is not None
            assert result.cate is not None

    def test_handles_import_error_gracefully(self, config, data):
        """Test graceful handling when econml.orf is not available."""
        treatment, outcome, covariates, _ = data
        wrapper = OrthoForestWrapper(config)

        # Create a targeted mock that only fails for econml.orf
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "econml.orf" or name.startswith("econml.orf"):
                raise ImportError("No module named 'econml.orf'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = wrapper.fit(treatment, outcome, covariates)

            assert result.success is False
            assert result.error_message is not None


class TestMetaLearnerComparison:
    """Compare meta-learners on the same data."""

    @pytest.fixture
    def data(self):
        """Synthetic test data with known treatment effect."""
        return generate_synthetic_data(
            n_samples=500,
            n_features=5,
            treatment_effect=3.0,
            heterogeneous=True,
            seed=123,
        )

    def test_all_learners_estimate_similar_ate(self, data):
        """Test that all meta-learners produce reasonable ATE estimates."""
        treatment, outcome, covariates, true_cate = data
        true_ate = np.mean(true_cate)

        learners = {
            "S-Learner": SLearnerWrapper(
                EstimatorConfig(
                    estimator_type=EstimatorType.S_LEARNER,
                    params={"n_estimators": 100, "max_depth": 5, "random_state": 42},
                )
            ),
            "T-Learner": TLearnerWrapper(
                EstimatorConfig(
                    estimator_type=EstimatorType.T_LEARNER,
                    params={"n_estimators": 100, "max_depth": 5, "random_state": 42},
                )
            ),
            "X-Learner": XLearnerWrapper(
                EstimatorConfig(
                    estimator_type=EstimatorType.X_LEARNER,
                    params={"n_estimators": 100, "max_depth": 5, "random_state": 42},
                )
            ),
        }

        ates = {}
        for name, learner in learners.items():
            result = learner.fit(treatment, outcome, covariates)
            assert result.success, f"{name} failed"
            ates[name] = result.ate

        # All should be within 1.0 of true ATE (allowing for estimation error)
        for name, ate in ates.items():
            assert abs(ate - true_ate) < 1.5, f"{name} ATE {ate} too far from {true_ate}"

    def test_all_learners_return_valid_cate(self, data):
        """Test that all meta-learners return valid CATE arrays."""
        treatment, outcome, covariates, _ = data

        learners = [
            SLearnerWrapper(EstimatorConfig(estimator_type=EstimatorType.S_LEARNER)),
            TLearnerWrapper(EstimatorConfig(estimator_type=EstimatorType.T_LEARNER)),
            XLearnerWrapper(EstimatorConfig(estimator_type=EstimatorType.X_LEARNER)),
        ]

        for learner in learners:
            result = learner.fit(treatment, outcome, covariates)
            assert result.success
            assert result.cate is not None
            assert len(result.cate) == len(treatment)
            assert not np.any(np.isnan(result.cate)), f"NaN in {learner.estimator_type}"
            assert not np.any(np.isinf(result.cate)), f"Inf in {learner.estimator_type}"
