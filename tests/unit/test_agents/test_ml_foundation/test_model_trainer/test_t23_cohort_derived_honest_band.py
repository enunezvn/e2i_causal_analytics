"""Plan v3 §4 T2.3 — Cohort-derived honest band (advisory mode).

Pins the contract for `_emit_cohort_derived_honest_band` plus end-to-end
wiring through `evaluate_model`. Plan §4 T2.3 replaces the hardcoded
`[0.62, 0.68]` literal in `synthetic_rwd_realistic.py` with a per-cohort
derivation from `baseline_test_auc` + `permutation_null_p99` +
`permutation_auc_std`. Like T2.2 (perm-anchored AUC floor), the band is
OBSERVABILITY only — no `success_criteria` mutation, no deployer impact.
T2.6c (separate work) is where this graduates to enforcement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    T2_3_HONEST_BAND_CEILING_DEFAULT,
    T2_3_HONEST_BAND_MAX_LIFT_DEFAULT,
    T2_3_HONEST_BAND_MIN_LIFT_DEFAULT,
    T2_3_HONEST_BAND_NOISE_SIGMA_DEFAULT,
    _emit_cohort_derived_honest_band,
    evaluate_model,
)
from tests.unit.test_agents.test_ml_foundation.test_model_trainer.conftest import (
    N_FEATURES,
    N_TEST_SAMPLES,
    N_TRAIN_SAMPLES,
    N_VAL_SAMPLES,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
)


# --------------------------------------------------------------------------- #
# Module-level constants                                                      #
# --------------------------------------------------------------------------- #


def test_default_constants_match_plan() -> None:
    """Plan v3 §4 T2.3 default thresholds."""
    assert T2_3_HONEST_BAND_MIN_LIFT_DEFAULT == 0.05
    assert T2_3_HONEST_BAND_MAX_LIFT_DEFAULT == 0.30
    assert T2_3_HONEST_BAND_CEILING_DEFAULT == 0.95
    assert T2_3_HONEST_BAND_NOISE_SIGMA_DEFAULT == 1.0


# --------------------------------------------------------------------------- #
# Helper-level unit tests                                                     #
# --------------------------------------------------------------------------- #


def _perm(p99: float | None, std: float | None = 0.04) -> Dict[str, Any]:
    """Permutation result stub."""
    return {
        "permutation_null_p99": p99,
        "permutation_null_p95": None if p99 is None else p99 - 0.02,
        "permutation_pvalue": 0.02,
        "permutation_auc_mean": None if p99 is None else p99 - 0.05,
        "permutation_auc_std": std,
    }


class TestHonestBandHappyPath:
    def test_band_lo_takes_max_of_meaningful_and_distinguishable(self) -> None:
        """Lo = max(baseline + min_lift, perm_null_p99 + sigma * perm_auc_std).
        baseline=0.55, perm_null_p99=0.58, perm_auc_std=0.04, defaults:
          - meaningful = 0.55 + 0.05 = 0.60
          - distinguishable = 0.58 + 1.0 * 0.04 = 0.62
          - max = 0.62"""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr, {"roc_auc": 0.70, "baseline_test_auc": 0.55}, _perm(0.58)
        )
        val = mr["validation_metrics"]
        assert val["honest_band_lo"] == pytest.approx(0.62)

    def test_band_hi_takes_min_of_ceiling_and_baseline_plus_max_lift(self) -> None:
        """Hi = min(ceiling=0.95, baseline + max_lift).
        baseline=0.55: hi = min(0.95, 0.85) = 0.85.
        baseline=0.80: hi = min(0.95, 1.10) = 0.95."""
        mr_low_base: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr_low_base, {"roc_auc": 0.70, "baseline_test_auc": 0.55}, _perm(0.58)
        )
        assert mr_low_base["validation_metrics"]["honest_band_hi"] == pytest.approx(0.85)

        mr_high_base: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr_high_base, {"roc_auc": 0.85, "baseline_test_auc": 0.80}, _perm(0.55)
        )
        assert mr_high_base["validation_metrics"]["honest_band_hi"] == pytest.approx(0.95)

    def test_band_lo_falls_back_to_meaningful_only_when_perm_null_missing(self) -> None:
        """When perm_null_p99 is missing, lo falls back to baseline + min_lift."""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr,
            {"roc_auc": 0.70, "baseline_test_auc": 0.55},
            {"permutation_null_p99": None, "permutation_auc_std": None},
        )
        val = mr["validation_metrics"]
        assert val["honest_band_lo"] == pytest.approx(0.60)
        # Hi still computed from baseline + max_lift.
        assert val["honest_band_hi"] == pytest.approx(0.85)

    def test_position_in_band_emitted_when_test_auc_inside(self) -> None:
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr, {"roc_auc": 0.70, "baseline_test_auc": 0.55}, _perm(0.58)
        )
        val = mr["validation_metrics"]
        # band [0.62, 0.85]; test_auc 0.70 → in_band.
        assert val["honest_band_position"] == "in_band"
        assert val["honest_band_violated"] is False

    def test_position_below_emitted_when_test_auc_under_lo(self) -> None:
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr, {"roc_auc": 0.55, "baseline_test_auc": 0.55}, _perm(0.58)
        )
        val = mr["validation_metrics"]
        # band [0.62, 0.85]; test_auc 0.55 → below.
        assert val["honest_band_position"] == "below"
        assert val["honest_band_violated"] is True

    def test_position_above_emitted_when_test_auc_over_hi(self) -> None:
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr, {"roc_auc": 0.92, "baseline_test_auc": 0.55}, _perm(0.58)
        )
        val = mr["validation_metrics"]
        # band [0.62, 0.85]; test_auc 0.92 → above.
        assert val["honest_band_position"] == "above"
        assert val["honest_band_violated"] is True


# --------------------------------------------------------------------------- #
# Configuration audit + override                                              #
# --------------------------------------------------------------------------- #


class TestHonestBandConfigAudit:
    def test_constants_always_emitted_for_audit_even_with_missing_inputs(self) -> None:
        """Even when the band cannot be evaluated, the configured thresholds
        are always emitted so an operator reading validation_metrics knows
        what would have been used."""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(mr, {}, {})
        val = mr["validation_metrics"]
        assert val["honest_band_min_lift"] == 0.05
        assert val["honest_band_max_lift"] == 0.30
        assert val["honest_band_ceiling"] == 0.95
        assert val["honest_band_noise_sigma"] == 1.0

    def test_per_call_override_propagates(self) -> None:
        """Caller can override defaults per call (forward-compat for
        cohort-specific tuning during T2.6c enforcement phase)."""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr,
            {"roc_auc": 0.70, "baseline_test_auc": 0.55},
            _perm(0.58),
            min_lift=0.10,
            max_lift=0.40,
            ceiling=0.90,
            noise_sigma=2.0,
        )
        val = mr["validation_metrics"]
        # Emitted thresholds match the override.
        assert val["honest_band_min_lift"] == 0.10
        assert val["honest_band_max_lift"] == 0.40
        assert val["honest_band_ceiling"] == 0.90
        assert val["honest_band_noise_sigma"] == 2.0
        # Lo = max(0.55+0.10, 0.58+2.0*0.04) = max(0.65, 0.66) = 0.66.
        assert val["honest_band_lo"] == pytest.approx(0.66)
        # Hi = min(0.90, 0.55+0.40) = min(0.90, 0.95) = 0.90.
        assert val["honest_band_hi"] == pytest.approx(0.90)


# --------------------------------------------------------------------------- #
# Degenerate / edge cases                                                     #
# --------------------------------------------------------------------------- #


class TestHonestBandDegenerateInputs:
    def test_band_keys_are_none_when_baseline_auc_missing(self) -> None:
        """No baseline → no band can be derived; lo/hi/violated/position
        emit None. Threshold constants still emit for audit."""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(mr, {"roc_auc": 0.70}, _perm(0.58))
        val = mr["validation_metrics"]
        assert val["honest_band_lo"] is None
        assert val["honest_band_hi"] is None
        assert val["honest_band_violated"] is None
        assert val["honest_band_position"] is None
        assert val["honest_band_baseline_test_auc"] is None
        # Constants still present.
        assert val["honest_band_min_lift"] == 0.05

    def test_band_violated_is_none_when_test_auc_missing(self) -> None:
        """Baseline present, no test_auc — band can be derived but not
        evaluated against test."""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(mr, {"baseline_test_auc": 0.55}, _perm(0.58))
        val = mr["validation_metrics"]
        # Band derived (lo, hi present)
        assert val["honest_band_lo"] is not None
        assert val["honest_band_hi"] is not None
        # But violation/position not evaluated.
        assert val["honest_band_violated"] is None
        assert val["honest_band_position"] is None

    def test_band_collapses_when_baseline_too_close_to_ceiling(self, caplog) -> None:
        """Pathological cohort: baseline=0.94, ceiling=0.95, perm_null_p99=0.93.
        - hi = min(0.95, 0.94+0.30) = 0.95
        - lo_distinguishable = 0.93 + 1.0 * 0.04 = 0.97
        - lo_meaningful = 0.94 + 0.05 = 0.99
        - lo = max(0.97, 0.99) = 0.99
        - lo > hi → band collapses; surface as None and emit warning."""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        with caplog.at_level(logging.WARNING):
            _emit_cohort_derived_honest_band(
                mr,
                {"roc_auc": 0.95, "baseline_test_auc": 0.94},
                _perm(0.93),
            )
        val = mr["validation_metrics"]
        assert val["honest_band_lo"] is None
        assert val["honest_band_hi"] is None
        assert val["honest_band_violated"] is None
        assert val["honest_band_position"] is None
        assert any("honest band collapsed" in record.message for record in caplog.records)

    def test_inputs_recorded_even_when_band_collapses(self) -> None:
        """The collapsed-band branch still records baseline/perm inputs so
        an operator can diagnose."""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        _emit_cohort_derived_honest_band(
            mr, {"roc_auc": 0.95, "baseline_test_auc": 0.94}, _perm(0.93)
        )
        val = mr["validation_metrics"]
        assert val["honest_band_baseline_test_auc"] == 0.94
        assert val["honest_band_perm_null_p99"] == 0.93


# --------------------------------------------------------------------------- #
# Plan §4 T2.3 invariant: no deployer/success_criteria impact                 #
# --------------------------------------------------------------------------- #


class TestHonestBandAdvisoryInvariant:
    def test_does_not_mutate_success_criteria(self) -> None:
        """Plan §6 T2.2/T2.3: advisory keys cannot enter success_criteria."""
        mr: Dict[str, Any] = {
            "validation_metrics": {},
            "success_criteria": {"minimum_auc": 0.75},
            "success_criteria_met": True,
            "success_criteria_results": {"minimum_auc": True},
        }
        _emit_cohort_derived_honest_band(
            mr,
            {"roc_auc": 0.55, "baseline_test_auc": 0.55},  # would violate
            _perm(0.58),
        )
        # Advisory present
        assert mr["validation_metrics"]["honest_band_violated"] is True
        # success_criteria untouched
        assert mr["success_criteria"] == {"minimum_auc": 0.75}
        assert mr["success_criteria_met"] is True
        assert mr["success_criteria_results"] == {"minimum_auc": True}

    def test_logs_warning_on_below_band(self, caplog) -> None:
        mr: Dict[str, Any] = {"validation_metrics": {}}
        with caplog.at_level(logging.WARNING):
            _emit_cohort_derived_honest_band(
                mr, {"roc_auc": 0.55, "baseline_test_auc": 0.55}, _perm(0.58)
            )
        assert any(
            "T2.3 ADVISORY" in record.message and "below" in record.message
            for record in caplog.records
        )

    def test_logs_warning_on_above_band(self, caplog) -> None:
        mr: Dict[str, Any] = {"validation_metrics": {}}
        with caplog.at_level(logging.WARNING):
            _emit_cohort_derived_honest_band(
                mr, {"roc_auc": 0.92, "baseline_test_auc": 0.55}, _perm(0.58)
            )
        assert any(
            "T2.3 ADVISORY" in record.message and "above" in record.message
            for record in caplog.records
        )

    def test_does_not_log_warning_when_in_band(self, caplog) -> None:
        """Healthy cohort = no log noise (CI runs many evaluations)."""
        mr: Dict[str, Any] = {"validation_metrics": {}}
        with caplog.at_level(logging.WARNING):
            _emit_cohort_derived_honest_band(
                mr, {"roc_auc": 0.70, "baseline_test_auc": 0.55}, _perm(0.58)
            )
        assert not any("T2.3 ADVISORY" in record.message for record in caplog.records)

    def test_creates_validation_metrics_when_absent(self) -> None:
        mr: Dict[str, Any] = {}
        _emit_cohort_derived_honest_band(
            mr, {"roc_auc": 0.70, "baseline_test_auc": 0.55}, _perm(0.58)
        )
        assert "validation_metrics" in mr


# --------------------------------------------------------------------------- #
# End-to-end integration through evaluate_model                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def real_classifier_state_for_t23():
    """Same fixture pattern as T2.2 / test_evaluator.py."""
    np.random.seed(RANDOM_STATE)
    X_train = np.random.rand(N_TRAIN_SAMPLES, N_FEATURES)
    y_train = np.random.randint(0, 2, N_TRAIN_SAMPLES)
    X_val = np.random.rand(N_VAL_SAMPLES, N_FEATURES)
    y_val = np.random.randint(0, 2, N_VAL_SAMPLES)
    X_test = np.random.rand(N_TEST_SAMPLES, N_FEATURES)
    y_test = np.random.randint(0, 2, N_TEST_SAMPLES)

    model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return {
        "trained_model": model,
        "problem_type": "binary_classification",
        "X_train_preprocessed": X_train,
        "X_validation_preprocessed": X_val,
        "X_test_preprocessed": X_test,
        "train_data": {"y": y_train},
        "validation_data": {"y": y_val},
        "test_data": {"y": y_test},
        "success_criteria": {},
    }


@pytest.mark.asyncio
async def test_evaluate_model_emits_honest_band_keys_on_validation_metrics(
    real_classifier_state_for_t23,
) -> None:
    """End-to-end: evaluate_model populates the honest-band keys on
    validation_metrics after the perm test runs."""
    result = await evaluate_model(real_classifier_state_for_t23)
    val = result.get("validation_metrics", {})
    expected_keys = {
        "honest_band_lo",
        "honest_band_hi",
        "honest_band_baseline_test_auc",
        "honest_band_perm_null_p99",
        "honest_band_perm_auc_std",
        "honest_band_min_lift",
        "honest_band_max_lift",
        "honest_band_ceiling",
        "honest_band_noise_sigma",
        "honest_band_violated",
        "honest_band_position",
    }
    missing = expected_keys - set(val.keys())
    assert not missing, f"missing honest-band keys in validation_metrics: {missing}"
    # Threshold constants always present.
    assert val["honest_band_min_lift"] == 0.05
    assert val["honest_band_ceiling"] == 0.95


@pytest.mark.asyncio
async def test_evaluate_model_honest_band_does_not_block_success_criteria(
    real_classifier_state_for_t23,
) -> None:
    """Honest-band advisory does NOT enter success_criteria_results and
    does NOT flip success_criteria_met. T2.6c (separate work) graduates
    the band to enforcement."""
    result = await evaluate_model(real_classifier_state_for_t23)
    crit_results = result.get("success_criteria_results", {})
    for key in ("honest_band_lo", "honest_band_hi", "honest_band_violated"):
        assert key not in crit_results
    # Fixture has empty success_criteria → success_criteria_met stays True.
    assert result.get("success_criteria_met") is True
