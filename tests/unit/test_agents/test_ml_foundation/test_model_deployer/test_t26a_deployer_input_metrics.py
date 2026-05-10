"""Plan v3 §4 T2.6a — Deployer-input metric computation contract tests.

Pins ``compute_deployer_input_metrics`` and the three categorization
helpers. T2.6a is PURE COMPUTATION — no enforcement. Plan §4 T2.6 splits:
  * T2.6a (this PR, 4h): metric computation only.
  * T2.6b (separate PR, 2h): shadow reporting (emit denial reasons as
    structured warnings to metrics_at_promotion).
  * T2.6c (separate PR, 2-6h): flip from advisory to enforcing after
    one quarter of stable monitoring.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    T2_6A_CALIBRATION_EXCELLENT_ECE_MAX,
    T2_6A_CALIBRATION_GOOD_ECE_MAX,
    T2_6A_CALIBRATION_MARGINAL_ECE_MAX,
    T2_6A_CV_STABILITY_MODERATE_RATIO_MAX,
    T2_6A_CV_STABILITY_STABLE_RATIO_MAX,
    T2_6A_CV_STABILITY_UNSTABLE_RATIO_MAX,
    T2_6A_SIGNAL_GENUINE_PVALUE_MAX,
    T2_6A_SIGNAL_LIKELY_GENUINE_PVALUE_MAX,
    T2_6A_SIGNAL_MARGINAL_PVALUE_MAX,
    _categorize_calibration_quality,
    _categorize_cv_stability,
    _categorize_signal_genuineness,
    compute_deployer_input_metrics,
)

# --------------------------------------------------------------------------- #
# Module constants                                                            #
# --------------------------------------------------------------------------- #


class TestT26AConstants:
    def test_signal_genuine_thresholds(self) -> None:
        assert T2_6A_SIGNAL_GENUINE_PVALUE_MAX == 0.001
        assert T2_6A_SIGNAL_LIKELY_GENUINE_PVALUE_MAX == 0.01
        assert T2_6A_SIGNAL_MARGINAL_PVALUE_MAX == 0.05
        # Strict ordering (<).
        assert (
            T2_6A_SIGNAL_GENUINE_PVALUE_MAX
            < T2_6A_SIGNAL_LIKELY_GENUINE_PVALUE_MAX
            < T2_6A_SIGNAL_MARGINAL_PVALUE_MAX
        )

    def test_calibration_thresholds(self) -> None:
        assert T2_6A_CALIBRATION_EXCELLENT_ECE_MAX == 0.05
        assert T2_6A_CALIBRATION_GOOD_ECE_MAX == 0.10
        assert T2_6A_CALIBRATION_MARGINAL_ECE_MAX == 0.20

    def test_cv_stability_thresholds(self) -> None:
        assert T2_6A_CV_STABILITY_STABLE_RATIO_MAX == 0.05
        assert T2_6A_CV_STABILITY_MODERATE_RATIO_MAX == 0.10
        assert T2_6A_CV_STABILITY_UNSTABLE_RATIO_MAX == 0.20


# --------------------------------------------------------------------------- #
# _categorize_signal_genuineness                                              #
# --------------------------------------------------------------------------- #


class TestCategorizeSignalGenuineness:
    @pytest.mark.parametrize(
        "pvalue,expected",
        [
            (0.0, "genuine"),
            (0.0005, "genuine"),
            (0.0009, "genuine"),
            (0.001, "likely_genuine"),  # boundary: strict <
            (0.005, "likely_genuine"),
            (0.0099, "likely_genuine"),
            (0.01, "marginal"),  # boundary
            (0.03, "marginal"),
            (0.0499, "marginal"),
            (0.05, "random"),  # boundary
            (0.5, "random"),
            (1.0, "random"),
        ],
    )
    def test_pvalue_categorization(self, pvalue, expected) -> None:
        assert _categorize_signal_genuineness(pvalue) == expected

    def test_none_pvalue_is_degenerate(self) -> None:
        assert _categorize_signal_genuineness(None) == "degenerate"


# --------------------------------------------------------------------------- #
# _categorize_calibration_quality                                             #
# --------------------------------------------------------------------------- #


class TestCategorizeCalibrationQuality:
    @pytest.mark.parametrize(
        "ece,expected",
        [
            (0.0, "excellent"),
            (0.02, "excellent"),
            (0.0499, "excellent"),
            (0.05, "good"),  # boundary
            (0.07, "good"),
            (0.0999, "good"),
            (0.10, "marginal"),  # boundary
            (0.15, "marginal"),
            (0.1999, "marginal"),
            (0.20, "poor"),  # boundary
            (0.50, "poor"),
        ],
    )
    def test_ece_categorization(self, ece, expected) -> None:
        assert _categorize_calibration_quality(ece) == expected

    def test_none_ece_is_degenerate(self) -> None:
        assert _categorize_calibration_quality(None) == "degenerate"


# --------------------------------------------------------------------------- #
# _categorize_cv_stability                                                    #
# --------------------------------------------------------------------------- #


class TestCategorizeCvStability:
    @pytest.mark.parametrize(
        "ratio,expected",
        [
            (0.0, "stable"),
            (0.02, "stable"),
            (0.0499, "stable"),
            (0.05, "moderate"),  # boundary
            (0.08, "moderate"),
            (0.0999, "moderate"),
            (0.10, "unstable"),  # boundary
            (0.15, "unstable"),
            (0.1999, "unstable"),
            (0.20, "very_unstable"),  # boundary
            (0.50, "very_unstable"),
        ],
    )
    def test_ratio_categorization(self, ratio, expected) -> None:
        assert _categorize_cv_stability(ratio) == expected

    def test_none_ratio_is_degenerate(self) -> None:
        assert _categorize_cv_stability(None) == "degenerate"


# --------------------------------------------------------------------------- #
# compute_deployer_input_metrics                                              #
# --------------------------------------------------------------------------- #


def _validation_metrics_full() -> Dict[str, Any]:
    """Healthy CSU-like cohort: genuine signal, good calibration, stable CV."""
    return {
        "permutation_pvalue": 0.0,  # GENUINE
        "cv_5fold_roc_auc_mean": 0.66,
        "cv_5fold_roc_auc_std": 0.02,  # ratio 0.030 → STABLE
    }


class TestComputeDeployerInputMetricsHappyPath:
    def test_returns_canonical_keys(self) -> None:
        result = compute_deployer_input_metrics(_validation_metrics_full(), calibration_error=0.04)
        for key in (
            "signal_genuineness_category",
            "signal_genuineness_pvalue",
            "calibration_quality_category",
            "calibration_quality_ece",
            "cv_stability_category",
            "cv_stability_std_over_mean",
            "cv_stability_std",
            "cv_stability_mean",
        ):
            assert key in result, f"missing key {key!r}"

    def test_healthy_cohort_categorizes_genuine_excellent_stable(self) -> None:
        result = compute_deployer_input_metrics(_validation_metrics_full(), calibration_error=0.04)
        assert result["signal_genuineness_category"] == "genuine"
        assert result["calibration_quality_category"] == "excellent"
        assert result["cv_stability_category"] == "stable"

    def test_inputs_propagate_to_outputs(self) -> None:
        result = compute_deployer_input_metrics(_validation_metrics_full(), calibration_error=0.04)
        assert result["signal_genuineness_pvalue"] == 0.0
        assert result["calibration_quality_ece"] == 0.04
        assert result["cv_stability_std"] == 0.02
        assert result["cv_stability_mean"] == 0.66

    def test_std_over_mean_is_correct_ratio(self) -> None:
        result = compute_deployer_input_metrics(
            {"cv_5fold_roc_auc_std": 0.02, "cv_5fold_roc_auc_mean": 0.50}
        )
        assert result["cv_stability_std_over_mean"] == pytest.approx(0.04)

    def test_optum_initiation_n1294_categorizes_marginal_random(self) -> None:
        """Empirical anchor from `docs/results/optum_initiation_revalidation_20260510.md`:
        Optum n=1294 (default-window) had perm_p=0.67, cv_mean=0.6795,
        cv_std=0.0937 (ratio 0.138). Expected categorization: random
        signal, unstable CV. Calibration ECE not measured in that doc;
        defaults to None → degenerate."""
        vm = {
            "permutation_pvalue": 0.67,
            "cv_5fold_roc_auc_mean": 0.6795,
            "cv_5fold_roc_auc_std": 0.0937,
        }
        result = compute_deployer_input_metrics(vm)
        assert result["signal_genuineness_category"] == "random"
        # ratio = 0.138 → in [0.10, 0.20) → unstable
        assert result["cv_stability_category"] == "unstable"
        assert result["calibration_quality_category"] == "degenerate"

    def test_optum_relaxed_n1697_categorizes_marginal_moderate(self) -> None:
        """Empirical anchor sensitivity test: relaxed window n=1697 gave
        perm_p=0.02, cv_mean=0.7259, cv_std=0.0669 (ratio 0.092). Expected:
        marginal signal (between likely_genuine and random), moderate CV."""
        vm = {
            "permutation_pvalue": 0.02,
            "cv_5fold_roc_auc_mean": 0.7259,
            "cv_5fold_roc_auc_std": 0.0669,
        }
        result = compute_deployer_input_metrics(vm)
        assert result["signal_genuineness_category"] == "marginal"
        # ratio = 0.0922 → in [0.05, 0.10) → moderate
        assert result["cv_stability_category"] == "moderate"


class TestComputeDeployerInputMetricsDegenerateInputs:
    def test_empty_validation_metrics_emits_all_degenerate(self) -> None:
        """Plan §4 T2.6a backward compat: missing inputs are NOT errors.
        Each signal categorizes to 'degenerate' so the deployer can read
        the contract and decide what to do (T2.6b/c)."""
        result = compute_deployer_input_metrics({})
        assert result["signal_genuineness_category"] == "degenerate"
        assert result["calibration_quality_category"] == "degenerate"
        assert result["cv_stability_category"] == "degenerate"

    def test_missing_pvalue_is_signal_degenerate(self) -> None:
        vm = {"cv_5fold_roc_auc_mean": 0.66, "cv_5fold_roc_auc_std": 0.02}
        result = compute_deployer_input_metrics(vm, calibration_error=0.04)
        assert result["signal_genuineness_category"] == "degenerate"
        # Other signals still computed.
        assert result["calibration_quality_category"] == "excellent"
        assert result["cv_stability_category"] == "stable"

    def test_missing_calibration_error_is_calib_degenerate(self) -> None:
        result = compute_deployer_input_metrics(_validation_metrics_full())
        # No calibration_error kwarg → None → degenerate
        assert result["calibration_quality_category"] == "degenerate"
        assert result["calibration_quality_ece"] is None

    def test_missing_cv_std_is_cv_degenerate(self) -> None:
        vm = {"permutation_pvalue": 0.0, "cv_5fold_roc_auc_mean": 0.66}
        result = compute_deployer_input_metrics(vm, calibration_error=0.04)
        assert result["cv_stability_category"] == "degenerate"
        assert result["cv_stability_std_over_mean"] is None

    def test_zero_cv_mean_is_cv_degenerate_no_div_by_zero(self) -> None:
        """Pathological cohort with cv_mean == 0: must NOT divide by zero."""
        vm = {
            "permutation_pvalue": 0.0,
            "cv_5fold_roc_auc_mean": 0.0,
            "cv_5fold_roc_auc_std": 0.02,
        }
        result = compute_deployer_input_metrics(vm)
        assert result["cv_stability_category"] == "degenerate"
        assert result["cv_stability_std_over_mean"] is None
