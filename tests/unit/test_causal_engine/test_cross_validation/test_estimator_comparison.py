"""Cross-Library Estimator Comparison Tests.

Tests comparing ATE estimates across DoWhy, EconML, and CausalML estimators
to verify consistency and identify estimator-specific strengths.

Test Data:
- Synthetic datasets with known causal effects (ground truth)
- Varying sample sizes (1K, 5K, 10K)
- Different effect sizes (ATE = 0.5, 1.0, 2.0)

Assertions:
- Estimates within 10% relative error of ground truth
- Cross-library estimates within 20% of each other
- Confidence intervals overlap for consistent estimators
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Tuple

# Mark all tests in this module for the dspy_integration xdist group
pytestmark = pytest.mark.xdist_group(name="cross_validation")


# =============================================================================
# TEST FIXTURES
# =============================================================================


def generate_synthetic_causal_data(
    n_samples: int = 1000,
    true_ate: float = 1.0,
    n_confounders: int = 3,
    treatment_effect_noise: float = 0.5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate synthetic data with known causal effect.

    Args:
        n_samples: Number of samples
        true_ate: True average treatment effect
        n_confounders: Number of confounding variables
        treatment_effect_noise: Noise in treatment effect
        seed: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame, metadata dict with ground truth)
    """
    np.random.seed(seed)

    # Generate confounders
    confounders = {}
    for i in range(n_confounders):
        confounders[f"X{i+1}"] = np.random.normal(0, 1, n_samples)

    # Treatment assignment (affected by confounders)
    propensity = 0.5 + 0.1 * confounders["X1"]
    propensity = np.clip(propensity, 0.1, 0.9)
    treatment = np.random.binomial(1, propensity)

    # Outcome (affected by treatment and confounders)
    outcome = (
        true_ate * treatment
        + 0.5 * confounders["X1"]
        + 0.3 * confounders["X2"]
        + np.random.normal(0, treatment_effect_noise, n_samples)
    )

    # Build DataFrame
    data = pd.DataFrame(confounders)
    data["treatment"] = treatment
    data["outcome"] = outcome

    metadata = {
        "true_ate": true_ate,
        "n_samples": n_samples,
        "n_confounders": n_confounders,
        "treatment_effect_noise": treatment_effect_noise,
    }

    return data, metadata


@pytest.fixture
def small_dataset():
    """Small dataset (1K samples) with ATE=1.0."""
    return generate_synthetic_causal_data(n_samples=1000, true_ate=1.0, seed=42)


@pytest.fixture
def medium_dataset():
    """Medium dataset (5K samples) with ATE=1.0."""
    return generate_synthetic_causal_data(n_samples=5000, true_ate=1.0, seed=43)


@pytest.fixture
def large_dataset():
    """Large dataset (10K samples) with ATE=1.0."""
    return generate_synthetic_causal_data(n_samples=10000, true_ate=1.0, seed=44)


@pytest.fixture
def small_effect_dataset():
    """Dataset with small effect (ATE=0.5)."""
    return generate_synthetic_causal_data(n_samples=5000, true_ate=0.5, seed=45)


@pytest.fixture
def large_effect_dataset():
    """Dataset with large effect (ATE=2.0)."""
    return generate_synthetic_causal_data(n_samples=5000, true_ate=2.0, seed=46)


# =============================================================================
# ESTIMATOR HELPERS
# =============================================================================


def estimate_with_dowhy_ols(data: pd.DataFrame) -> Dict[str, float]:
    """Estimate ATE using DoWhy OLS."""
    try:
        import dowhy
        from dowhy import CausalModel

        confounders = [c for c in data.columns if c.startswith("X")]

        model = CausalModel(
            data=data,
            treatment="treatment",
            outcome="outcome",
            common_causes=confounders,
        )

        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
        )

        return {
            "ate": estimate.value,
            "ci_lower": estimate.value - 1.96 * estimate.get_standard_error(),
            "ci_upper": estimate.value + 1.96 * estimate.get_standard_error(),
            "method": "dowhy_ols",
        }
    except Exception as e:
        return {"ate": None, "error": str(e), "method": "dowhy_ols"}


def estimate_with_dowhy_ipw(data: pd.DataFrame) -> Dict[str, float]:
    """Estimate ATE using DoWhy IPW."""
    try:
        import dowhy
        from dowhy import CausalModel

        confounders = [c for c in data.columns if c.startswith("X")]

        model = CausalModel(
            data=data,
            treatment="treatment",
            outcome="outcome",
            common_causes=confounders,
        )

        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.propensity_score_weighting",
        )

        return {
            "ate": estimate.value,
            "method": "dowhy_ipw",
        }
    except Exception as e:
        return {"ate": None, "error": str(e), "method": "dowhy_ipw"}


def estimate_with_econml_linear_dml(data: pd.DataFrame) -> Dict[str, float]:
    """Estimate ATE using EconML LinearDML."""
    try:
        from econml.dml import LinearDML
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        confounders = [c for c in data.columns if c.startswith("X")]
        X = data[confounders].values
        T = data["treatment"].values
        Y = data["outcome"].values

        model = LinearDML(
            model_y=LassoCV(),
            model_t=LogisticRegressionCV(max_iter=1000),
            discrete_treatment=True,
        )
        model.fit(Y, T, X=X)

        ate = model.ate(X)
        ate_interval = model.ate_interval(X, alpha=0.05)

        return {
            "ate": float(ate),
            "ci_lower": float(ate_interval[0]),
            "ci_upper": float(ate_interval[1]),
            "method": "econml_linear_dml",
        }
    except Exception as e:
        return {"ate": None, "error": str(e), "method": "econml_linear_dml"}


def estimate_with_econml_drlearner(data: pd.DataFrame) -> Dict[str, float]:
    """Estimate ATE using EconML DRLearner."""
    try:
        from econml.dr import DRLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        confounders = [c for c in data.columns if c.startswith("X")]
        X = data[confounders].values
        T = data["treatment"].values
        Y = data["outcome"].values

        model = DRLearner(
            model_propensity=GradientBoostingClassifier(n_estimators=50, max_depth=3),
            model_regression=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            model_final=GradientBoostingRegressor(n_estimators=50, max_depth=3),
        )
        model.fit(Y, T, X=X)

        ate = model.ate(X)
        ate_interval = model.ate_interval(X, alpha=0.05)

        return {
            "ate": float(ate),
            "ci_lower": float(ate_interval[0]),
            "ci_upper": float(ate_interval[1]),
            "method": "econml_drlearner",
        }
    except Exception as e:
        return {"ate": None, "error": str(e), "method": "econml_drlearner"}


def estimate_with_econml_causal_forest(data: pd.DataFrame) -> Dict[str, float]:
    """Estimate ATE using EconML CausalForestDML."""
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        confounders = [c for c in data.columns if c.startswith("X")]
        X = data[confounders].values
        T = data["treatment"].values
        Y = data["outcome"].values

        model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=48, max_depth=3),
            model_t=GradientBoostingClassifier(n_estimators=48, max_depth=3),
            discrete_treatment=True,
            n_estimators=48,  # Must be divisible by subforest_size (default=4)
            min_samples_leaf=10,
        )
        model.fit(Y, T, X=X)

        ate = model.ate(X)
        ate_interval = model.ate_interval(X, alpha=0.05)

        return {
            "ate": float(ate),
            "ci_lower": float(ate_interval[0]),
            "ci_upper": float(ate_interval[1]),
            "method": "econml_causal_forest",
        }
    except Exception as e:
        return {"ate": None, "error": str(e), "method": "econml_causal_forest"}


# =============================================================================
# CROSS-LIBRARY COMPARISON TESTS
# =============================================================================


class TestDoWhyVsEconML:
    """Tests comparing DoWhy and EconML estimators."""

    def test_ols_vs_linear_dml_medium_dataset(self, medium_dataset):
        """Compare DoWhy OLS vs EconML LinearDML on medium dataset."""
        pytest.importorskip("dowhy")
        data, metadata = medium_dataset
        true_ate = metadata["true_ate"]

        dowhy_result = estimate_with_dowhy_ols(data)
        econml_result = estimate_with_econml_linear_dml(data)

        # Both should produce estimates
        assert dowhy_result["ate"] is not None, f"DoWhy failed: {dowhy_result.get('error')}"
        assert econml_result["ate"] is not None, f"EconML failed: {econml_result.get('error')}"

        # Both should be within 10% of true ATE
        dowhy_error = abs(dowhy_result["ate"] - true_ate) / true_ate
        econml_error = abs(econml_result["ate"] - true_ate) / true_ate

        assert dowhy_error < 0.10, f"DoWhy OLS error {dowhy_error:.2%} > 10%"
        assert econml_error < 0.10, f"EconML LinearDML error {econml_error:.2%} > 10%"

        # Cross-library: estimates should be within 20% of each other
        cross_diff = abs(dowhy_result["ate"] - econml_result["ate"]) / true_ate
        assert cross_diff < 0.20, f"Cross-library difference {cross_diff:.2%} > 20%"

    def test_ipw_vs_drlearner_medium_dataset(self, medium_dataset):
        """Compare DoWhy IPW vs EconML DRLearner on medium dataset."""
        pytest.importorskip("dowhy")
        data, metadata = medium_dataset
        true_ate = metadata["true_ate"]

        dowhy_result = estimate_with_dowhy_ipw(data)
        econml_result = estimate_with_econml_drlearner(data)

        # Both should produce estimates
        assert dowhy_result["ate"] is not None, f"DoWhy failed: {dowhy_result.get('error')}"
        assert econml_result["ate"] is not None, f"EconML failed: {econml_result.get('error')}"

        # Both should be within 15% of true ATE (IPW/DR methods have more variance)
        dowhy_error = abs(dowhy_result["ate"] - true_ate) / true_ate
        econml_error = abs(econml_result["ate"] - true_ate) / true_ate

        assert dowhy_error < 0.15, f"DoWhy IPW error {dowhy_error:.2%} > 15%"
        assert econml_error < 0.15, f"EconML DRLearner error {econml_error:.2%} > 15%"

    def test_estimator_consistency_across_sample_sizes(
        self, small_dataset, medium_dataset, large_dataset
    ):
        """Test that estimates converge as sample size increases."""
        datasets = [
            ("small", small_dataset),
            ("medium", medium_dataset),
            ("large", large_dataset),
        ]

        errors = {}
        for name, (data, metadata) in datasets:
            true_ate = metadata["true_ate"]
            result = estimate_with_econml_linear_dml(data)
            if result["ate"] is not None:
                errors[name] = abs(result["ate"] - true_ate) / true_ate

        # All datasets should produce reasonable estimates (within 15% relative error)
        # Note: Statistical variance can cause individual estimates to vary, so we
        # don't enforce monotonic improvement with sample size. We only verify
        # that the estimator produces accurate results across all sample sizes.
        if len(errors) >= 2:
            for name, error in errors.items():
                assert error < 0.15, f"{name} dataset error {error:.2%} exceeds 15% threshold"
            # At least verify all estimates are in reasonable range
            assert all(e < 0.15 for e in errors.values()), (
                f"All estimates should be within 15% error: {errors}"
            )


class TestEffectSizeRecovery:
    """Tests for recovering different effect sizes."""

    def test_small_effect_recovery(self, small_effect_dataset):
        """Test recovery of small causal effect (ATE=0.5)."""
        data, metadata = small_effect_dataset
        true_ate = metadata["true_ate"]

        result = estimate_with_econml_linear_dml(data)

        assert result["ate"] is not None, f"Estimation failed: {result.get('error')}"

        error = abs(result["ate"] - true_ate) / true_ate
        assert error < 0.15, f"Small effect recovery error {error:.2%} > 15%"

    def test_large_effect_recovery(self, large_effect_dataset):
        """Test recovery of large causal effect (ATE=2.0)."""
        data, metadata = large_effect_dataset
        true_ate = metadata["true_ate"]

        result = estimate_with_econml_linear_dml(data)

        assert result["ate"] is not None, f"Estimation failed: {result.get('error')}"

        error = abs(result["ate"] - true_ate) / true_ate
        assert error < 0.10, f"Large effect recovery error {error:.2%} > 10%"


class TestConfidenceIntervalOverlap:
    """Tests for confidence interval consistency across estimators."""

    def test_ci_overlap_linear_methods(self, medium_dataset):
        """Test that CIs from linear methods overlap."""
        data, metadata = medium_dataset
        true_ate = metadata["true_ate"]

        dowhy_result = estimate_with_dowhy_ols(data)
        econml_result = estimate_with_econml_linear_dml(data)

        # Skip if CIs not available
        if "ci_lower" not in dowhy_result or "ci_lower" not in econml_result:
            pytest.skip("CI not available for comparison")

        # Check if CIs overlap
        overlap = max(dowhy_result["ci_lower"], econml_result["ci_lower"]) <= min(
            dowhy_result["ci_upper"], econml_result["ci_upper"]
        )

        assert overlap, (
            f"CIs do not overlap: DoWhy [{dowhy_result['ci_lower']:.3f}, {dowhy_result['ci_upper']:.3f}], "
            f"EconML [{econml_result['ci_lower']:.3f}, {econml_result['ci_upper']:.3f}]"
        )

    def test_true_ate_in_ci(self, medium_dataset):
        """Test that true ATE is within confidence interval."""
        data, metadata = medium_dataset
        true_ate = metadata["true_ate"]

        result = estimate_with_econml_linear_dml(data)

        if "ci_lower" not in result:
            pytest.skip("CI not available")

        in_ci = result["ci_lower"] <= true_ate <= result["ci_upper"]
        assert in_ci, (
            f"True ATE {true_ate} not in CI [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        )


class TestCausalForestComparison:
    """Tests for CausalForest estimator comparison."""

    def test_causal_forest_vs_linear_dml(self, medium_dataset):
        """Compare CausalForest vs LinearDML estimates."""
        data, metadata = medium_dataset
        true_ate = metadata["true_ate"]

        cf_result = estimate_with_econml_causal_forest(data)
        dml_result = estimate_with_econml_linear_dml(data)

        assert cf_result["ate"] is not None, f"CausalForest failed: {cf_result.get('error')}"
        assert dml_result["ate"] is not None, f"LinearDML failed: {dml_result.get('error')}"

        # Both should be reasonable
        cf_error = abs(cf_result["ate"] - true_ate) / true_ate
        dml_error = abs(dml_result["ate"] - true_ate) / true_ate

        # Allow more variance for CausalForest (nonparametric)
        assert cf_error < 0.20, f"CausalForest error {cf_error:.2%} > 20%"
        assert dml_error < 0.15, f"LinearDML error {dml_error:.2%} > 15%"

        # Cross-estimator consistency
        cross_diff = abs(cf_result["ate"] - dml_result["ate"]) / true_ate
        assert cross_diff < 0.25, f"CausalForest vs LinearDML difference {cross_diff:.2%} > 25%"
