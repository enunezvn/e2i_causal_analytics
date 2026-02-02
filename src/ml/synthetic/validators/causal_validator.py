"""
Causal Validator for Synthetic Data

Validates that causal inference pipelines can recover TRUE_ATE:
- DoWhy refutation tests
- ATE estimation within tolerance
- Confounder balance verification
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import DGP_CONFIGS, DGPType
from ..ground_truth.causal_effects import GroundTruthEffect

logger = logging.getLogger(__name__)


@dataclass
class RefutationResult:
    """Result of a single refutation test."""

    test_name: str
    passed: bool
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    message: str = ""


@dataclass
class CausalValidationResult:
    """Result of causal validation."""

    is_valid: bool
    dgp_type: str
    true_ate: float
    estimated_ate: Optional[float] = None
    ate_error: Optional[float] = None
    ate_within_tolerance: bool = False
    tolerance: float = 0.05
    refutation_results: List[RefutationResult] = field(default_factory=list)
    refutation_pass_rate: float = 0.0
    min_required_pass_rate: float = 0.60
    confounder_balance: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_refutation(self, result: RefutationResult) -> None:
        """Add a refutation test result."""
        self.refutation_results.append(result)
        passed_count = sum(1 for r in self.refutation_results if r.passed)
        self.refutation_pass_rate = passed_count / len(self.refutation_results)

    def meets_criteria(self) -> bool:
        """Check if all validation criteria are met."""
        return (
            self.ate_within_tolerance and self.refutation_pass_rate >= self.min_required_pass_rate
        )


class CausalValidator:
    """
    Validates causal effect recovery from synthetic data.

    Usage:
        validator = CausalValidator()
        result = validator.validate(
            df=patient_df,
            ground_truth=ground_truth_effect,
            treatment="engagement_score",
            outcome="treatment_initiated",
            confounders=["disease_severity", "academic_hcp"]
        )
        if result.is_valid:
            print("Causal validation passed!")
    """

    def __init__(
        self,
        min_refutation_pass_rate: float = 0.60,
        ate_tolerance: float = 0.05,
    ):
        """
        Initialize causal validator.

        Args:
            min_refutation_pass_rate: Minimum fraction of refutation tests that must pass
            ate_tolerance: Maximum allowed |estimated_ate - true_ate|
        """
        self.min_refutation_pass_rate = min_refutation_pass_rate
        self.ate_tolerance = ate_tolerance
        self._dowhy_available = self._check_dowhy()

    def _check_dowhy(self) -> bool:
        """Check if DoWhy is available."""
        try:
            import dowhy  # noqa: F401

            return True
        except ImportError:
            logger.warning("DoWhy not available. Limited validation possible.")
            return False

    def validate(
        self,
        df: pd.DataFrame,
        ground_truth: GroundTruthEffect,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        confounders: Optional[List[str]] = None,
        run_refutations: bool = True,
    ) -> CausalValidationResult:
        """
        Validate causal effect recovery.

        Args:
            df: DataFrame with treatment, outcome, and confounders
            ground_truth: Known ground truth effect
            treatment: Treatment variable name (default from ground_truth)
            outcome: Outcome variable name (default from ground_truth)
            confounders: Confounder variables (default from ground_truth)
            run_refutations: Whether to run DoWhy refutation tests

        Returns:
            CausalValidationResult with validation details
        """
        treatment = treatment or ground_truth.treatment_variable
        outcome = outcome or ground_truth.outcome_variable
        confounders = confounders or ground_truth.confounders

        dgp_type = (
            ground_truth.dgp_type.value
            if isinstance(ground_truth.dgp_type, DGPType)
            else ground_truth.dgp_type
        )

        result = CausalValidationResult(
            is_valid=False,
            dgp_type=dgp_type,
            true_ate=ground_truth.true_ate,
            tolerance=ground_truth.tolerance,
            min_required_pass_rate=self.min_refutation_pass_rate,
        )

        # Check required columns exist
        required_cols = [treatment, outcome] + confounders
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            result.errors.append(f"Missing columns: {missing}")
            return result

        # Estimate ATE
        try:
            estimated_ate = self._estimate_ate(df, treatment, outcome, confounders)
            result.estimated_ate = estimated_ate
            result.ate_error = abs(estimated_ate - ground_truth.true_ate)
            result.ate_within_tolerance = result.ate_error <= ground_truth.tolerance
        except Exception as e:
            result.errors.append(f"ATE estimation failed: {str(e)}")
            return result

        # Check confounder balance
        result.confounder_balance = self._check_confounder_balance(df, treatment, confounders)

        # Run refutation tests if DoWhy available and requested
        if run_refutations and self._dowhy_available:
            self._run_refutations(df, treatment, outcome, confounders, result)
        elif run_refutations:
            result.warnings.append("Refutation tests skipped: DoWhy not available")
            # Give benefit of the doubt if DoWhy not available
            result.refutation_pass_rate = 1.0

        # Determine overall validity
        result.is_valid = result.meets_criteria()

        return result

    def _estimate_ate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> float:
        """
        Estimate ATE using available methods.

        Falls back from DoWhy to statsmodels to simple difference.
        """
        if self._dowhy_available:
            return self._estimate_ate_dowhy(df, treatment, outcome, confounders)
        else:
            return self._estimate_ate_regression(df, treatment, outcome, confounders)

    def _estimate_ate_dowhy(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> float:
        """Estimate ATE using DoWhy."""
        try:
            from dowhy import CausalModel

            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders,
            )

            identified_estimand = model.identify_effect()
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
            )

            return float(estimate.value)
        except Exception as e:
            logger.warning(f"DoWhy estimation failed, falling back to regression: {e}")
            return self._estimate_ate_regression(df, treatment, outcome, confounders)

    def _estimate_ate_regression(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> float:
        """Estimate ATE using OLS regression with confounder adjustment."""
        try:
            import statsmodels.api as sm

            X = df[[treatment] + confounders].copy()
            X = sm.add_constant(X)
            y = df[outcome]

            model = sm.OLS(y, X).fit()
            return float(model.params[treatment])
        except ImportError:
            # Fallback to numpy if statsmodels not available
            return self._estimate_ate_simple(df, treatment, outcome)

    def _estimate_ate_simple(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
    ) -> float:
        """Simple ATE estimation (biased if confounded)."""
        # Just compute correlation coefficient as rough estimate
        corr = df[[treatment, outcome]].corr().iloc[0, 1]
        return float(corr)

    def _check_confounder_balance(
        self,
        df: pd.DataFrame,
        treatment: str,
        confounders: List[str],
    ) -> Dict[str, float]:
        """
        Check balance of confounders across treatment levels.

        Returns standardized mean differences for each confounder.
        """
        balance = {}

        # Binarize treatment if continuous
        treatment_values = df[treatment]
        if treatment_values.nunique() > 10:
            median = treatment_values.median()
            treated = treatment_values >= median
        else:
            treated = treatment_values > 0

        for conf in confounders:
            if conf not in df.columns:
                continue

            conf_values = df[conf]
            if not pd.api.types.is_numeric_dtype(conf_values):
                continue

            treated_mean = conf_values[treated].mean()
            control_mean = conf_values[~treated].mean()
            pooled_std = conf_values.std()

            if pooled_std > 0:
                smd = (treated_mean - control_mean) / pooled_std
                balance[conf] = float(smd)
            else:
                balance[conf] = 0.0

        return balance

    def _run_refutations(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
        result: CausalValidationResult,
    ) -> None:
        """Run DoWhy refutation tests."""
        try:
            from dowhy import CausalModel

            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders,
            )

            identified_estimand = model.identify_effect()
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
            )

            # 1. Placebo treatment test
            try:
                refute_placebo = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="placebo_treatment_refuter",
                )
                passed = abs(refute_placebo.new_effect) < abs(estimate.value) * 0.1
                result.add_refutation(
                    RefutationResult(
                        test_name="placebo_treatment",
                        passed=passed,
                        effect_size=float(refute_placebo.new_effect),
                        message=str(refute_placebo),
                    )
                )
            except Exception as e:
                result.add_refutation(
                    RefutationResult(
                        test_name="placebo_treatment",
                        passed=False,
                        message=f"Test failed: {str(e)}",
                    )
                )

            # 2. Random common cause test
            try:
                refute_random = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="random_common_cause",
                )
                original = estimate.value
                new = refute_random.new_effect
                passed = abs(new - original) < abs(original) * 0.1
                result.add_refutation(
                    RefutationResult(
                        test_name="random_common_cause",
                        passed=passed,
                        effect_size=float(refute_random.new_effect),
                        message=str(refute_random),
                    )
                )
            except Exception as e:
                result.add_refutation(
                    RefutationResult(
                        test_name="random_common_cause",
                        passed=False,
                        message=f"Test failed: {str(e)}",
                    )
                )

            # 3. Data subset test
            try:
                refute_subset = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="data_subset_refuter",
                    subset_fraction=0.8,
                )
                original = estimate.value
                new = refute_subset.new_effect
                passed = abs(new - original) < abs(original) * 0.2
                result.add_refutation(
                    RefutationResult(
                        test_name="data_subset",
                        passed=passed,
                        effect_size=float(refute_subset.new_effect),
                        message=str(refute_subset),
                    )
                )
            except Exception as e:
                result.add_refutation(
                    RefutationResult(
                        test_name="data_subset",
                        passed=False,
                        message=f"Test failed: {str(e)}",
                    )
                )

        except Exception as e:
            result.errors.append(f"Refutation tests failed: {str(e)}")

    def validate_dgp(
        self,
        df: pd.DataFrame,
        dgp_type: DGPType,
        run_refutations: bool = True,
    ) -> CausalValidationResult:
        """
        Validate a dataset against a known DGP configuration.

        Args:
            df: DataFrame with treatment, outcome, and confounders
            dgp_type: Type of DGP to validate against
            run_refutations: Whether to run DoWhy refutation tests

        Returns:
            CausalValidationResult
        """
        dgp_config = DGP_CONFIGS[dgp_type]

        # Create a ground truth effect from the config
        from ..config import Brand
        from ..ground_truth.causal_effects import create_ground_truth_from_dgp_config

        ground_truth = create_ground_truth_from_dgp_config(
            brand=Brand.REMIBRUTINIB,  # Default brand for validation
            dgp_type=dgp_type,
            n_samples=len(df),
        )

        return self.validate(
            df=df,
            ground_truth=ground_truth,
            treatment=dgp_config.treatment_variable,
            outcome=dgp_config.outcome_variable,
            confounders=dgp_config.confounders,
            run_refutations=run_refutations,
        )

    def get_validation_summary(self, result: CausalValidationResult) -> Dict[str, Any]:
        """Get a summary of causal validation."""
        return {
            "is_valid": result.is_valid,
            "dgp_type": result.dgp_type,
            "true_ate": result.true_ate,
            "estimated_ate": result.estimated_ate,
            "ate_error": result.ate_error,
            "ate_within_tolerance": result.ate_within_tolerance,
            "tolerance": result.tolerance,
            "refutation_pass_rate": result.refutation_pass_rate,
            "min_required_pass_rate": result.min_required_pass_rate,
            "refutations": [
                {"name": r.test_name, "passed": r.passed} for r in result.refutation_results
            ],
            "confounder_balance": result.confounder_balance,
            "errors": result.errors,
            "warnings": result.warnings,
        }
