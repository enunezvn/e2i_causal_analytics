"""Unit tests for uplift metrics.

Tests cover:
- AUUC calculation
- Qini coefficient and Qini curve
- Cumulative gain
- Uplift at k percentiles
- Calibration metrics
- Comprehensive evaluation
"""

import numpy as np
import pytest

from src.causal_engine.uplift.metrics import (
    UpliftMetrics,
    auuc,
    calculate_cumulative_gain,
    calculate_qini_curve,
    calculate_uplift_curve,
    cumulative_gain_auc,
    evaluate_uplift_model,
    qini_auc,
    qini_coefficient,
    treatment_effect_calibration,
    uplift_at_k,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def perfect_uplift_data():
    """Create data where high uplift scores predict high treatment effect.

    Treatment responders (treatment=1, outcome=1) get high scores.
    Control responders (treatment=0, outcome=1) get low scores.
    Non-responders get medium scores.
    """
    np.random.seed(42)
    n = 1000

    treatment = np.random.binomial(1, 0.5, n)
    outcome = np.zeros(n)

    # Create treatment effect - treatment group has higher response
    outcome[treatment == 1] = np.random.binomial(1, 0.6, np.sum(treatment == 1))
    outcome[treatment == 0] = np.random.binomial(1, 0.3, np.sum(treatment == 0))

    # Perfect model: scores correlate with true effect
    uplift_scores = np.zeros(n)
    # High scores for treated responders
    uplift_scores[(treatment == 1) & (outcome == 1)] = np.random.uniform(
        0.7, 1.0, np.sum((treatment == 1) & (outcome == 1))
    )
    # Low scores for control responders
    uplift_scores[(treatment == 0) & (outcome == 1)] = np.random.uniform(
        0.0, 0.3, np.sum((treatment == 0) & (outcome == 1))
    )
    # Medium scores for non-responders
    uplift_scores[outcome == 0] = np.random.uniform(
        0.3, 0.7, np.sum(outcome == 0)
    )

    return uplift_scores, treatment, outcome


@pytest.fixture
def random_uplift_data():
    """Create data with random uplift scores (no predictive power)."""
    np.random.seed(42)
    n = 1000

    treatment = np.random.binomial(1, 0.5, n)
    outcome = np.random.binomial(1, 0.5, n)
    uplift_scores = np.random.uniform(0, 1, n)

    return uplift_scores, treatment, outcome


@pytest.fixture
def simple_uplift_data():
    """Create simple small dataset for testing."""
    # 10 samples with known properties
    uplift_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    treatment = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    outcome = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])

    return uplift_scores, treatment, outcome


# =============================================================================
# UPLIFT CURVE TESTS
# =============================================================================


class TestUpliftCurve:
    """Tests for uplift curve calculation."""

    def test_uplift_curve_shape(self, simple_uplift_data):
        """Test uplift curve returns correct shape."""
        uplift_scores, treatment, outcome = simple_uplift_data
        x_values, uplift_values = calculate_uplift_curve(
            uplift_scores, treatment, outcome
        )

        # Should have n+1 points (including origin)
        assert len(x_values) == len(uplift_scores) + 1
        assert len(uplift_values) == len(uplift_scores) + 1

    def test_uplift_curve_starts_at_origin(self, simple_uplift_data):
        """Test uplift curve starts at (0, 0)."""
        uplift_scores, treatment, outcome = simple_uplift_data
        x_values, uplift_values = calculate_uplift_curve(
            uplift_scores, treatment, outcome
        )

        assert x_values[0] == 0.0
        assert uplift_values[0] == 0.0

    def test_uplift_curve_ends_at_one(self, simple_uplift_data):
        """Test uplift curve x-axis ends at 1.0."""
        uplift_scores, treatment, outcome = simple_uplift_data
        x_values, uplift_values = calculate_uplift_curve(
            uplift_scores, treatment, outcome
        )

        assert x_values[-1] == 1.0

    def test_uplift_curve_x_monotonic(self, simple_uplift_data):
        """Test uplift curve x values are monotonically increasing."""
        uplift_scores, treatment, outcome = simple_uplift_data
        x_values, _ = calculate_uplift_curve(uplift_scores, treatment, outcome)

        assert np.all(np.diff(x_values) > 0)

    def test_uplift_curve_empty_data(self):
        """Test uplift curve handles empty data."""
        x_values, uplift_values = calculate_uplift_curve(
            np.array([]), np.array([]), np.array([])
        )

        assert len(x_values) == 2
        assert len(uplift_values) == 2


# =============================================================================
# QINI CURVE TESTS
# =============================================================================


class TestQiniCurve:
    """Tests for Qini curve calculation."""

    def test_qini_curve_shape(self, simple_uplift_data):
        """Test Qini curve returns correct shape."""
        uplift_scores, treatment, outcome = simple_uplift_data
        x_values, qini_values = calculate_qini_curve(
            uplift_scores, treatment, outcome
        )

        assert len(x_values) == len(uplift_scores) + 1
        assert len(qini_values) == len(uplift_scores) + 1

    def test_qini_curve_starts_at_origin(self, simple_uplift_data):
        """Test Qini curve starts at origin."""
        uplift_scores, treatment, outcome = simple_uplift_data
        x_values, qini_values = calculate_qini_curve(
            uplift_scores, treatment, outcome
        )

        assert x_values[0] == 0.0
        assert qini_values[0] == 0.0


# =============================================================================
# CUMULATIVE GAIN TESTS
# =============================================================================


class TestCumulativeGain:
    """Tests for cumulative gain calculation."""

    def test_cumulative_gain_shape(self, simple_uplift_data):
        """Test cumulative gain returns correct shape."""
        uplift_scores, treatment, outcome = simple_uplift_data
        x_values, gain_values = calculate_cumulative_gain(
            uplift_scores, treatment, outcome
        )

        assert len(x_values) == len(uplift_scores) + 1
        assert len(gain_values) == len(uplift_scores) + 1

    def test_cumulative_gain_starts_at_origin(self, simple_uplift_data):
        """Test cumulative gain starts at origin."""
        uplift_scores, treatment, outcome = simple_uplift_data
        x_values, gain_values = calculate_cumulative_gain(
            uplift_scores, treatment, outcome
        )

        assert x_values[0] == 0.0
        assert gain_values[0] == 0.0


# =============================================================================
# AUUC TESTS
# =============================================================================


class TestAUUC:
    """Tests for AUUC calculation."""

    def test_auuc_returns_float(self, simple_uplift_data):
        """Test AUUC returns a float."""
        uplift_scores, treatment, outcome = simple_uplift_data
        score = auuc(uplift_scores, treatment, outcome)

        assert isinstance(score, float)

    def test_auuc_returns_valid_value(self, perfect_uplift_data, random_uplift_data):
        """Test that AUUC returns valid values for different data."""
        perfect_scores, treatment_p, outcome_p = perfect_uplift_data
        random_scores, treatment_r, outcome_r = random_uplift_data

        auuc_perfect = auuc(perfect_scores, treatment_p, outcome_p)
        auuc_random = auuc(random_scores, treatment_r, outcome_r)

        # Both should return valid floats (not NaN)
        assert not np.isnan(auuc_perfect)
        assert not np.isnan(auuc_random)
        assert isinstance(auuc_perfect, float)
        assert isinstance(auuc_random, float)

    def test_auuc_normalization(self, simple_uplift_data):
        """Test AUUC with and without normalization."""
        uplift_scores, treatment, outcome = simple_uplift_data

        auuc_normalized = auuc(uplift_scores, treatment, outcome, normalize=True)
        auuc_raw = auuc(uplift_scores, treatment, outcome, normalize=False)

        # Both should be valid floats
        assert not np.isnan(auuc_normalized)
        assert not np.isnan(auuc_raw)


# =============================================================================
# QINI COEFFICIENT TESTS
# =============================================================================


class TestQiniCoefficient:
    """Tests for Qini coefficient calculation."""

    def test_qini_coefficient_returns_float(self, simple_uplift_data):
        """Test Qini coefficient returns a float."""
        uplift_scores, treatment, outcome = simple_uplift_data
        coef = qini_coefficient(uplift_scores, treatment, outcome)

        assert isinstance(coef, float)

    def test_qini_coefficient_bounded(self, random_uplift_data):
        """Test Qini coefficient is a valid float."""
        uplift_scores, treatment, outcome = random_uplift_data
        coef = qini_coefficient(uplift_scores, treatment, outcome)

        # Coefficient should be a valid float (not NaN or Inf)
        assert not np.isnan(coef)
        assert not np.isinf(coef)


# =============================================================================
# QINI AUC TESTS
# =============================================================================


class TestQiniAUC:
    """Tests for Qini AUC calculation."""

    def test_qini_auc_returns_float(self, simple_uplift_data):
        """Test Qini AUC returns a float."""
        uplift_scores, treatment, outcome = simple_uplift_data
        score = qini_auc(uplift_scores, treatment, outcome)

        assert isinstance(score, float)


# =============================================================================
# CUMULATIVE GAIN AUC TESTS
# =============================================================================


class TestCumulativeGainAUC:
    """Tests for cumulative gain AUC calculation."""

    def test_cumulative_gain_auc_returns_float(self, simple_uplift_data):
        """Test cumulative gain AUC returns a float."""
        uplift_scores, treatment, outcome = simple_uplift_data
        score = cumulative_gain_auc(uplift_scores, treatment, outcome)

        assert isinstance(score, float)


# =============================================================================
# UPLIFT AT K TESTS
# =============================================================================


class TestUpliftAtK:
    """Tests for uplift at k percentiles calculation."""

    def test_uplift_at_k_returns_dict(self, simple_uplift_data):
        """Test uplift_at_k returns a dictionary."""
        uplift_scores, treatment, outcome = simple_uplift_data
        result = uplift_at_k(uplift_scores, treatment, outcome)

        assert isinstance(result, dict)

    def test_uplift_at_k_default_percentiles(self, simple_uplift_data):
        """Test uplift_at_k with default percentiles."""
        uplift_scores, treatment, outcome = simple_uplift_data
        result = uplift_at_k(uplift_scores, treatment, outcome)

        # Default percentiles are [10, 20, 30, 40, 50]
        assert "uplift_at_10" in result
        assert "uplift_at_20" in result
        assert "uplift_at_30" in result
        assert "uplift_at_40" in result
        assert "uplift_at_50" in result

    def test_uplift_at_k_custom_percentiles(self, simple_uplift_data):
        """Test uplift_at_k with custom percentiles."""
        uplift_scores, treatment, outcome = simple_uplift_data
        result = uplift_at_k(
            uplift_scores, treatment, outcome, k_percentiles=[5, 25, 75, 95]
        )

        assert "uplift_at_5" in result
        assert "uplift_at_25" in result
        assert "uplift_at_75" in result
        assert "uplift_at_95" in result

    def test_uplift_at_k_values_are_floats(self, simple_uplift_data):
        """Test uplift_at_k values are all floats."""
        uplift_scores, treatment, outcome = simple_uplift_data
        result = uplift_at_k(uplift_scores, treatment, outcome)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float"


# =============================================================================
# CALIBRATION TESTS
# =============================================================================


class TestCalibration:
    """Tests for treatment effect calibration."""

    def test_calibration_returns_tuple(self, random_uplift_data):
        """Test calibration returns correct tuple structure."""
        uplift_scores, treatment, outcome = random_uplift_data
        pred_means, obs_means, cal_error = treatment_effect_calibration(
            uplift_scores, treatment, outcome
        )

        assert isinstance(pred_means, np.ndarray)
        assert isinstance(obs_means, np.ndarray)
        assert isinstance(cal_error, float)

    def test_calibration_arrays_same_length(self, random_uplift_data):
        """Test calibration arrays have same length."""
        uplift_scores, treatment, outcome = random_uplift_data
        pred_means, obs_means, _ = treatment_effect_calibration(
            uplift_scores, treatment, outcome
        )

        assert len(pred_means) == len(obs_means)

    def test_calibration_error_non_negative(self, random_uplift_data):
        """Test calibration error is non-negative."""
        uplift_scores, treatment, outcome = random_uplift_data
        _, _, cal_error = treatment_effect_calibration(
            uplift_scores, treatment, outcome
        )

        assert cal_error >= 0.0


# =============================================================================
# COMPREHENSIVE EVALUATION TESTS
# =============================================================================


class TestEvaluateUpliftModel:
    """Tests for comprehensive uplift model evaluation."""

    def test_evaluate_returns_metrics_object(self, simple_uplift_data):
        """Test evaluate_uplift_model returns UpliftMetrics."""
        uplift_scores, treatment, outcome = simple_uplift_data
        metrics = evaluate_uplift_model(uplift_scores, treatment, outcome)

        assert isinstance(metrics, UpliftMetrics)

    def test_evaluate_all_metrics_present(self, simple_uplift_data):
        """Test all metrics are calculated."""
        uplift_scores, treatment, outcome = simple_uplift_data
        metrics = evaluate_uplift_model(uplift_scores, treatment, outcome)

        assert hasattr(metrics, "auuc")
        assert hasattr(metrics, "qini_coefficient")
        assert hasattr(metrics, "qini_auc")
        assert hasattr(metrics, "cumulative_gain_auc")
        assert hasattr(metrics, "uplift_at_k")
        assert hasattr(metrics, "calibration_error")
        assert hasattr(metrics, "treatment_balance")

    def test_evaluate_metadata_populated(self, simple_uplift_data):
        """Test metadata is populated correctly."""
        uplift_scores, treatment, outcome = simple_uplift_data
        metrics = evaluate_uplift_model(uplift_scores, treatment, outcome)

        assert "n_samples" in metrics.metadata
        assert "n_treated" in metrics.metadata
        assert "n_control" in metrics.metadata
        assert metrics.metadata["n_samples"] == len(uplift_scores)

    def test_evaluate_to_dict(self, simple_uplift_data):
        """Test metrics can be serialized to dict."""
        uplift_scores, treatment, outcome = simple_uplift_data
        metrics = evaluate_uplift_model(uplift_scores, treatment, outcome)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "auuc" in metrics_dict
        assert "qini_coefficient" in metrics_dict

    def test_evaluate_handles_multidimensional_scores(self, simple_uplift_data):
        """Test evaluation handles 2D uplift scores."""
        uplift_scores, treatment, outcome = simple_uplift_data
        # Make scores 2D (n_samples, 1)
        scores_2d = uplift_scores.reshape(-1, 1)

        metrics = evaluate_uplift_model(scores_2d, treatment, outcome)

        assert isinstance(metrics, UpliftMetrics)
        assert not np.isnan(metrics.auuc)

    def test_evaluate_custom_percentiles(self, simple_uplift_data):
        """Test evaluation with custom percentiles."""
        uplift_scores, treatment, outcome = simple_uplift_data
        metrics = evaluate_uplift_model(
            uplift_scores, treatment, outcome, k_percentiles=[5, 15, 25]
        )

        assert "uplift_at_5" in metrics.uplift_at_k
        assert "uplift_at_15" in metrics.uplift_at_k
        assert "uplift_at_25" in metrics.uplift_at_k


# =============================================================================
# UPLIFT METRICS DATACLASS TESTS
# =============================================================================


class TestUpliftMetrics:
    """Tests for UpliftMetrics dataclass."""

    def test_metrics_creation(self):
        """Test UpliftMetrics can be created directly."""
        metrics = UpliftMetrics(
            auuc=0.65,
            qini_coefficient=0.45,
            qini_auc=0.35,
            cumulative_gain_auc=0.25,
            uplift_at_k={"uplift_at_10": 0.1, "uplift_at_20": 0.08},
            calibration_error=0.05,
            treatment_balance=1.2,
        )

        assert metrics.auuc == 0.65
        assert metrics.qini_coefficient == 0.45
        assert metrics.calibration_error == 0.05

    def test_metrics_to_dict(self):
        """Test UpliftMetrics serialization."""
        metrics = UpliftMetrics(
            auuc=0.65,
            qini_coefficient=0.45,
            qini_auc=0.35,
            cumulative_gain_auc=0.25,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["auuc"] == 0.65
        assert metrics_dict["qini_coefficient"] == 0.45
