"""Plan v3 §4 T2.6b — Shadow reporting (advisory-mode warnings).

Pins ``compute_advisory_denial_reasons`` and the wiring through
``validate_promotion``. T2.6b is OBSERVABILITY ONLY — does NOT mutate
``promotion_allowed``. Plan §6 T2.6 calibration window: one quarter of
shadow reporting before T2.6c flips to enforcement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pytest

from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    T2_6B_CALIBRATION_QUALITY_REJECT_CATEGORIES,
    T2_6B_CV_STABILITY_REJECT_CATEGORIES,
    T2_6B_SIGNAL_GENUINENESS_REJECT_CATEGORIES,
    compute_advisory_denial_reasons,
    compute_deployer_input_metrics,
    validate_promotion,
)


# --------------------------------------------------------------------------- #
# Reject-category constants                                                   #
# --------------------------------------------------------------------------- #


class TestT26BRejectCategories:
    def test_signal_reject_includes_random_marginal_degenerate(self) -> None:
        assert T2_6B_SIGNAL_GENUINENESS_REJECT_CATEGORIES == frozenset(
            {"random", "marginal", "degenerate"}
        )

    def test_calibration_reject_includes_poor_marginal_degenerate(self) -> None:
        assert T2_6B_CALIBRATION_QUALITY_REJECT_CATEGORIES == frozenset(
            {"poor", "marginal", "degenerate"}
        )

    def test_cv_stability_reject_includes_unstable_categories(self) -> None:
        assert T2_6B_CV_STABILITY_REJECT_CATEGORIES == frozenset(
            {"very_unstable", "unstable", "degenerate"}
        )


# --------------------------------------------------------------------------- #
# compute_advisory_denial_reasons                                             #
# --------------------------------------------------------------------------- #


def _healthy_metrics() -> Dict[str, Any]:
    """All three categories healthy → no advisory warnings."""
    return compute_deployer_input_metrics(
        {
            "permutation_pvalue": 0.0,  # genuine
            "cv_5fold_roc_auc_mean": 0.66,
            "cv_5fold_roc_auc_std": 0.02,  # ratio 0.030 → stable
        },
        calibration_error=0.04,  # excellent
    )


class TestAdvisoryDenialReasons:
    def test_no_warnings_when_all_categories_healthy(self) -> None:
        reasons = compute_advisory_denial_reasons(_healthy_metrics())
        assert reasons == []

    def test_single_warning_when_only_signal_unhealthy(self) -> None:
        metrics = compute_deployer_input_metrics(
            {
                "permutation_pvalue": 0.67,  # random
                "cv_5fold_roc_auc_mean": 0.66,
                "cv_5fold_roc_auc_std": 0.02,
            },
            calibration_error=0.04,
        )
        reasons = compute_advisory_denial_reasons(metrics)
        assert len(reasons) == 1
        assert "T2.6b ADVISORY" in reasons[0]
        assert "signal_genuineness=random" in reasons[0]
        assert "0.6700" in reasons[0]

    def test_single_warning_when_only_calibration_unhealthy(self) -> None:
        metrics = compute_deployer_input_metrics(
            {
                "permutation_pvalue": 0.0,
                "cv_5fold_roc_auc_mean": 0.66,
                "cv_5fold_roc_auc_std": 0.02,
            },
            calibration_error=0.25,  # poor
        )
        reasons = compute_advisory_denial_reasons(metrics)
        assert len(reasons) == 1
        assert "calibration_quality=poor" in reasons[0]

    def test_single_warning_when_only_cv_unhealthy(self) -> None:
        metrics = compute_deployer_input_metrics(
            {
                "permutation_pvalue": 0.0,
                "cv_5fold_roc_auc_mean": 0.50,
                "cv_5fold_roc_auc_std": 0.20,  # ratio 0.40 → very_unstable
            },
            calibration_error=0.04,
        )
        reasons = compute_advisory_denial_reasons(metrics)
        assert len(reasons) == 1
        assert "cv_stability=very_unstable" in reasons[0]

    def test_three_warnings_when_all_categories_unhealthy(self) -> None:
        """Optum n=1294 anchor: random signal + degenerate calibration +
        unstable CV."""
        metrics = compute_deployer_input_metrics(
            {
                "permutation_pvalue": 0.67,
                "cv_5fold_roc_auc_mean": 0.6795,
                "cv_5fold_roc_auc_std": 0.0937,  # ratio 0.138 → unstable
            },
            # calibration_error=None → degenerate
        )
        reasons = compute_advisory_denial_reasons(metrics)
        assert len(reasons) == 3
        assert any("signal_genuineness=random" in r for r in reasons)
        assert any("calibration_quality=degenerate" in r for r in reasons)
        assert any("cv_stability=unstable" in r for r in reasons)

    def test_marginal_signal_emits_warning(self) -> None:
        """marginal is in reject set per plan §4 T2.6b — until calibration
        proves otherwise, marginal signals should not auto-promote."""
        metrics = compute_deployer_input_metrics(
            {"permutation_pvalue": 0.02},  # marginal
        )
        reasons = compute_advisory_denial_reasons(metrics)
        assert any("signal_genuineness=marginal" in r for r in reasons)

    def test_likely_genuine_does_not_emit_warning(self) -> None:
        """likely_genuine is NOT in reject set — only random/marginal/
        degenerate trigger the advisory."""
        metrics = compute_deployer_input_metrics(
            {
                "permutation_pvalue": 0.005,  # likely_genuine
                "cv_5fold_roc_auc_mean": 0.66,
                "cv_5fold_roc_auc_std": 0.02,
            },
            calibration_error=0.04,
        )
        reasons = compute_advisory_denial_reasons(metrics)
        assert not any("signal_genuineness" in r for r in reasons)

    def test_warning_format_includes_pvalue_value(self) -> None:
        metrics = compute_deployer_input_metrics(
            {"permutation_pvalue": 0.0234},
        )
        reasons = compute_advisory_denial_reasons(metrics)
        assert "0.0234" in reasons[0]

    def test_warning_format_handles_none_inputs(self) -> None:
        """Missing inputs → degenerate category → warning formats 'None'
        cleanly (not 'NoneNone' or KeyError)."""
        metrics = compute_deployer_input_metrics({})
        reasons = compute_advisory_denial_reasons(metrics)
        assert len(reasons) == 3
        for r in reasons:
            assert "None" in r


# --------------------------------------------------------------------------- #
# validate_promotion wiring                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestValidatePromotionT26BWiring:
    """End-to-end: validate_promotion now emits t26_deployer_input_metrics
    and t26_advisory_warnings on its return dict, but promotion_allowed
    is unaffected (advisory-only invariant)."""

    async def test_returns_t26_keys_on_allowed_promotion(self):
        """Healthy metrics: promotion allowed AND advisory warnings empty."""
        state = {
            "current_stage": "None",
            "target_stage": "Staging",
            "validation_metrics": {
                "permutation_pvalue": 0.0,
                "cv_5fold_roc_auc_mean": 0.66,
                "cv_5fold_roc_auc_std": 0.02,
                "calibration_error": 0.04,
            },
        }
        result = await validate_promotion(state)
        assert result["promotion_allowed"] is True
        assert "t26_deployer_input_metrics" in result
        assert "t26_advisory_warnings" in result
        assert result["t26_advisory_warnings"] == []
        # Categories surfaced.
        deployer_metrics = result["t26_deployer_input_metrics"]
        assert deployer_metrics["signal_genuineness_category"] == "genuine"
        assert deployer_metrics["calibration_quality_category"] == "excellent"
        assert deployer_metrics["cv_stability_category"] == "stable"

    async def test_advisory_warnings_do_not_block_promotion(self):
        """Plan §6 T2.6b invariant: even all-unhealthy categories must NOT
        flip promotion_allowed to False. Only the legacy gates (allowed
        promotion paths, shadow validation) can block."""
        state = {
            "current_stage": "None",
            "target_stage": "Staging",
            "validation_metrics": {
                "permutation_pvalue": 0.67,
                "cv_5fold_roc_auc_mean": 0.6795,
                "cv_5fold_roc_auc_std": 0.0937,
                "calibration_error": 0.25,
            },
        }
        result = await validate_promotion(state)
        # Promotion still allowed despite three advisory warnings.
        assert result["promotion_allowed"] is True
        # All three categories trigger warnings.
        assert len(result["t26_advisory_warnings"]) == 3

    async def test_advisory_warnings_emit_log_warnings(self, caplog):
        """Operator monitoring (Splunk on log level) sees the warnings."""
        state = {
            "current_stage": "None",
            "target_stage": "Staging",
            "validation_metrics": {
                "permutation_pvalue": 0.67,
                "cv_5fold_roc_auc_mean": 0.50,
                "cv_5fold_roc_auc_std": 0.20,
                "calibration_error": 0.30,
            },
        }
        with caplog.at_level(logging.WARNING):
            await validate_promotion(state)
        # All three warnings hit the log.
        warning_messages = " ".join(r.message for r in caplog.records)
        assert "T2.6b ADVISORY" in warning_messages
        assert "signal_genuineness=random" in warning_messages
        assert "calibration_quality=poor" in warning_messages
        assert "cv_stability=very_unstable" in warning_messages

    async def test_returns_t26_keys_on_invalid_promotion_path(self):
        """The invalid-path branch returns early WITHOUT t26 keys — that's
        OK for advisory-only since the deployer is already denying via the
        legacy gate."""
        state = {
            "current_stage": "Production",
            "target_stage": "Staging",  # invalid: Production → Staging
            "validation_metrics": {
                "permutation_pvalue": 0.0,
                "cv_5fold_roc_auc_mean": 0.66,
                "cv_5fold_roc_auc_std": 0.02,
            },
        }
        result = await validate_promotion(state)
        assert result["promotion_allowed"] is False
        # T2.6b keys NOT present on the invalid-path return — the legacy
        # gate already denies, so the advisory wouldn't add value here.
        assert "t26_advisory_warnings" not in result

    async def test_validation_metrics_missing_emits_all_degenerate(self):
        """When validation_metrics is missing, T2.6a categorizes everything
        as degenerate; T2.6b emits 3 warnings; promotion still allowed."""
        state = {
            "current_stage": "None",
            "target_stage": "Staging",
            # No validation_metrics
        }
        result = await validate_promotion(state)
        assert result["promotion_allowed"] is True
        assert len(result["t26_advisory_warnings"]) == 3
        for r in result["t26_advisory_warnings"]:
            assert "degenerate" in r
