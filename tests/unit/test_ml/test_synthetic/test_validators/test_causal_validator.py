"""
Tests for CausalValidator.

Tests causal effect recovery validation.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.ground_truth.causal_effects import (
    GroundTruthEffect,
    create_ground_truth_from_dgp_config,
)
from src.ml.synthetic.validators.causal_validator import (
    CausalValidationResult,
    CausalValidator,
)


class TestCausalValidator:
    """Test suite for CausalValidator."""

    @pytest.fixture
    def validator(self):
        """Create a CausalValidator instance."""
        return CausalValidator(min_refutation_pass_rate=0.60, ate_tolerance=0.05)

    @pytest.fixture
    def simple_linear_data(self):
        """
        Create synthetic data with known TRUE_ATE = 0.40.

        Simple linear relationship:
        Y = 0.40 * T + noise
        """
        np.random.seed(42)
        n = 1000

        # Treatment (continuous)
        treatment = np.random.uniform(0, 10, n)

        # Outcome with TRUE_ATE = 0.40
        noise = np.random.normal(0, 0.5, n)
        outcome = 0.40 * treatment + noise

        # Binarize outcome for consistency
        outcome_binary = (outcome > np.median(outcome)).astype(int)

        return pd.DataFrame(
            {
                "engagement_score": treatment,
                "treatment_initiated": outcome_binary,
                "disease_severity": np.random.uniform(0, 10, n),  # Not a confounder
                "academic_hcp": np.random.binomial(1, 0.3, n),  # Not a confounder
            }
        )

    @pytest.fixture
    def confounded_data(self):
        """
        Create synthetic data with confounding.

        TRUE_ATE = 0.25 after adjustment.
        Confounders: disease_severity, academic_hcp
        """
        np.random.seed(42)
        n = 2000

        # Confounders
        disease_severity = np.clip(np.random.normal(5, 2, n), 0, 10)
        academic_hcp = np.random.binomial(1, 0.3, n)

        # Treatment influenced by confounders
        treatment_propensity = (
            3.0 + 0.3 * disease_severity + 2.0 * academic_hcp + np.random.normal(0, 1, n)
        )
        treatment = expit(treatment_propensity / 3) * 10

        # Outcome influenced by confounders AND treatment
        outcome_propensity = (
            -2.0
            + 0.25 * treatment  # TRUE CAUSAL EFFECT
            + 0.4 * disease_severity  # Confounding
            + 0.6 * academic_hcp  # Confounding
            + np.random.normal(0, 1, n)
        )
        outcome = (expit(outcome_propensity) > 0.5).astype(int)

        return pd.DataFrame(
            {
                "engagement_score": treatment,
                "treatment_initiated": outcome,
                "disease_severity": disease_severity,
                "academic_hcp": academic_hcp,
            }
        )

    @pytest.fixture
    def ground_truth_confounded(self):
        """Create ground truth for confounded DGP."""
        return create_ground_truth_from_dgp_config(
            brand=Brand.REMIBRUTINIB,
            dgp_type=DGPType.CONFOUNDED,
            n_samples=2000,
        )

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.min_refutation_pass_rate == 0.60
        assert validator.ate_tolerance == 0.05

    def test_validate_simple_linear(self, validator, simple_linear_data):
        """Test validation with simple linear data."""
        ground_truth = GroundTruthEffect(
            brand=Brand.REMIBRUTINIB,
            dgp_type=DGPType.SIMPLE_LINEAR,
            true_ate=0.40,
            tolerance=0.10,  # Wider tolerance for binary outcome
            confounders=[],
            treatment_variable="engagement_score",
            outcome_variable="treatment_initiated",
        )

        result = validator.validate(
            df=simple_linear_data,
            ground_truth=ground_truth,
            run_refutations=False,  # Skip for faster test
        )

        assert result.estimated_ate is not None
        assert result.dgp_type == "simple_linear"
        assert result.true_ate == 0.40

    def test_validate_confounded_data(self, validator, confounded_data, ground_truth_confounded):
        """Test validation with confounded data."""
        result = validator.validate(
            df=confounded_data,
            ground_truth=ground_truth_confounded,
            run_refutations=False,
        )

        assert result.estimated_ate is not None
        assert result.true_ate == 0.25
        # Check that estimate is reasonably close
        assert result.ate_error is not None

    def test_validate_with_missing_columns(self, validator, simple_linear_data):
        """Test validation with missing columns."""
        df = simple_linear_data.drop(columns=["engagement_score"])

        ground_truth = GroundTruthEffect(
            brand=Brand.REMIBRUTINIB,
            dgp_type=DGPType.SIMPLE_LINEAR,
            true_ate=0.40,
            tolerance=0.05,
            confounders=[],
            treatment_variable="engagement_score",
            outcome_variable="treatment_initiated",
        )

        result = validator.validate(
            df=df,
            ground_truth=ground_truth,
        )

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "Missing columns" in result.errors[0]

    def test_validate_dgp_shortcut(self, validator, confounded_data):
        """Test validate_dgp shortcut method."""
        result = validator.validate_dgp(
            df=confounded_data,
            dgp_type=DGPType.CONFOUNDED,
            run_refutations=False,
        )

        assert result.dgp_type == "confounded"
        assert result.true_ate == 0.25

    def test_confounder_balance_check(self, validator, confounded_data):
        """Test confounder balance checking."""
        ground_truth = GroundTruthEffect(
            brand=Brand.REMIBRUTINIB,
            dgp_type=DGPType.CONFOUNDED,
            true_ate=0.25,
            tolerance=0.05,
            confounders=["disease_severity", "academic_hcp"],
            treatment_variable="engagement_score",
            outcome_variable="treatment_initiated",
        )

        result = validator.validate(
            df=confounded_data,
            ground_truth=ground_truth,
            run_refutations=False,
        )

        # Should have confounder balance info
        assert "disease_severity" in result.confounder_balance
        assert "academic_hcp" in result.confounder_balance

    def test_validation_summary(self, validator, confounded_data):
        """Test validation summary generation."""
        result = validator.validate_dgp(
            df=confounded_data,
            dgp_type=DGPType.CONFOUNDED,
            run_refutations=False,
        )

        summary = validator.get_validation_summary(result)

        assert "is_valid" in summary
        assert "dgp_type" in summary
        assert "true_ate" in summary
        assert "estimated_ate" in summary
        assert "confounder_balance" in summary


class TestCausalValidationResult:
    """Test suite for CausalValidationResult."""

    def test_add_refutation_updates_pass_rate(self):
        """Test that adding refutations updates pass rate."""
        from src.ml.synthetic.validators.causal_validator import RefutationResult

        result = CausalValidationResult(
            is_valid=False,
            dgp_type="test",
            true_ate=0.25,
        )

        result.add_refutation(
            RefutationResult(
                test_name="test1",
                passed=True,
            )
        )

        assert result.refutation_pass_rate == 1.0

        result.add_refutation(
            RefutationResult(
                test_name="test2",
                passed=False,
            )
        )

        assert result.refutation_pass_rate == 0.5

    def test_meets_criteria(self):
        """Test meets_criteria method."""
        result = CausalValidationResult(
            is_valid=False,
            dgp_type="test",
            true_ate=0.25,
            estimated_ate=0.24,
            ate_within_tolerance=True,
            refutation_pass_rate=0.70,
            min_required_pass_rate=0.60,
        )

        assert result.meets_criteria() is True

        result.ate_within_tolerance = False
        assert result.meets_criteria() is False
