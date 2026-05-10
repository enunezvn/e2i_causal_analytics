"""Plan v3 §4 T2.2 — Permutation-anchored AUC floor (advisory mode).

Pins the contract for `_emit_permutation_anchored_auc_advisory` plus the
end-to-end advisory wiring through `evaluate_model`. Plan §6 T2.2:
"emitted in advisory mode for one quarter before enforcement". The
advisory is OBSERVABILITY only — no `success_criteria` mutation, no
deployer impact. T2.6c (separate work) will graduate it to enforcement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT,
    _emit_permutation_anchored_auc_advisory,
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
# Module-level constant                                                       #
# --------------------------------------------------------------------------- #


def test_default_buffer_is_005() -> None:
    """Plan v3 §4 T2.2 default buffer: 0.05 (5pp above null p99)."""
    assert T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT == 0.05


# --------------------------------------------------------------------------- #
# Helper-level unit tests                                                     #
# --------------------------------------------------------------------------- #


def _perm(p99: float | None) -> Dict[str, Any]:
    return {
        "permutation_null_p99": p99,
        "permutation_null_p95": None if p99 is None else p99 - 0.02,
        "permutation_pvalue": 0.02,
    }


def test_advisory_violated_when_test_auc_below_p99_plus_buffer() -> None:
    """test_auc=0.55, null_p99=0.52, buffer=0.05 → margin=0.03 < buffer.
    Violation expected."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    test_metrics = {"roc_auc": 0.55}
    perm = _perm(0.52)
    _emit_permutation_anchored_auc_advisory(mr, test_metrics, perm, buffer=0.05)
    val = mr["validation_metrics"]
    assert val["auc_above_permutation_null"] == pytest.approx(0.03)
    assert val["permutation_anchored_auc_buffer"] == 0.05
    assert val["permutation_anchored_auc_advisory_violated"] is True


def test_advisory_met_when_test_auc_well_above_p99_plus_buffer() -> None:
    """test_auc=0.80, null_p99=0.55, buffer=0.05 → margin=0.25 >= buffer."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    test_metrics = {"roc_auc": 0.80}
    perm = _perm(0.55)
    _emit_permutation_anchored_auc_advisory(mr, test_metrics, perm, buffer=0.05)
    val = mr["validation_metrics"]
    assert val["auc_above_permutation_null"] == pytest.approx(0.25)
    assert val["permutation_anchored_auc_advisory_violated"] is False


def test_advisory_violated_at_exact_buffer_boundary_uses_strict_lt() -> None:
    """margin == buffer → NOT violated (strict <). margin = buffer - 2^-20 → violated.
    Uses powers-of-two values so the IEEE-754 subtraction is exact and the
    boundary case lands precisely on `==`. (Plain decimals like
    0.60 - 0.55 actually equal 0.04999... due to float rep, which would
    flip the boundary case unexpectedly.)"""
    # buffer = 2^-4 = 0.0625; auc=0.625, null_p99=0.5625 → margin=0.0625 exactly.
    mr_eq: Dict[str, Any] = {"validation_metrics": {}}
    _emit_permutation_anchored_auc_advisory(mr_eq, {"roc_auc": 0.625}, _perm(0.5625), buffer=0.0625)
    assert mr_eq["validation_metrics"]["auc_above_permutation_null"] == 0.0625
    assert mr_eq["validation_metrics"]["permutation_anchored_auc_advisory_violated"] is False

    # margin slightly under buffer → violated.
    mr_lt: Dict[str, Any] = {"validation_metrics": {}}
    _emit_permutation_anchored_auc_advisory(
        mr_lt,
        {"roc_auc": 0.625 - 2**-20},
        _perm(0.5625),
        buffer=0.0625,
    )
    assert mr_lt["validation_metrics"]["permutation_anchored_auc_advisory_violated"] is True


def test_advisory_negative_margin_violated() -> None:
    """test_auc < null_p99 → margin < 0 < buffer → violated."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    _emit_permutation_anchored_auc_advisory(mr, {"roc_auc": 0.45}, _perm(0.55), buffer=0.05)
    val = mr["validation_metrics"]
    assert val["auc_above_permutation_null"] == pytest.approx(-0.10)
    assert val["permutation_anchored_auc_advisory_violated"] is True


def test_advisory_keys_are_none_when_perm_p99_is_none() -> None:
    """Degenerate perm run: ``permutation_null_p99`` is None → advisory
    cannot be evaluated; emit None for both margin and violation flag.
    The buffer key is still present (for operator visibility)."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    _emit_permutation_anchored_auc_advisory(mr, {"roc_auc": 0.80}, {"permutation_null_p99": None})
    val = mr["validation_metrics"]
    assert val["auc_above_permutation_null"] is None
    assert val["permutation_anchored_auc_advisory_violated"] is None
    assert val["permutation_anchored_auc_buffer"] == 0.05


def test_advisory_keys_are_none_when_perm_p99_absent() -> None:
    """Older callers pre-Tier-1B-step-1: perm result missing
    ``permutation_null_p99`` entirely. Same None semantics as the
    explicit-None case — backward-compat with pre-PR-#118 perm output."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    _emit_permutation_anchored_auc_advisory(mr, {"roc_auc": 0.80}, {})
    val = mr["validation_metrics"]
    assert val["auc_above_permutation_null"] is None
    assert val["permutation_anchored_auc_advisory_violated"] is None


def test_advisory_keys_are_none_when_test_auc_absent() -> None:
    """Edge: test_metrics missing roc_auc (e.g., regression model
    routed through binary path by misconfig). Advisory soft-fails."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    _emit_permutation_anchored_auc_advisory(mr, {}, _perm(0.55))
    val = mr["validation_metrics"]
    assert val["auc_above_permutation_null"] is None
    assert val["permutation_anchored_auc_advisory_violated"] is None


def test_advisory_does_not_mutate_success_criteria() -> None:
    """Advisory mode invariant: NO write to success_criteria,
    success_criteria_met, or success_criteria_results. Plan §6 T2.2:
    'emitted in advisory mode for one quarter before enforcement.'"""
    mr: Dict[str, Any] = {
        "validation_metrics": {},
        "success_criteria": {"minimum_auc": 0.75},
        "success_criteria_met": True,
        "success_criteria_results": {"minimum_auc": True},
    }
    _emit_permutation_anchored_auc_advisory(
        mr,
        {"roc_auc": 0.45},
        _perm(0.55),  # would be a violation
    )
    # Advisory keys present...
    assert mr["validation_metrics"]["permutation_anchored_auc_advisory_violated"] is True
    # ...but success_criteria fields untouched.
    assert mr["success_criteria"] == {"minimum_auc": 0.75}
    assert mr["success_criteria_met"] is True
    assert mr["success_criteria_results"] == {"minimum_auc": True}


def test_advisory_logs_warning_on_violation(caplog) -> None:
    """A violation must emit a structured WARNING log so operator
    monitoring (Grafana/Splunk on log levels) can pick it up. The
    log message must include both test_auc and null_p99 for triage."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    with caplog.at_level(logging.WARNING):
        _emit_permutation_anchored_auc_advisory(
            mr,
            {"roc_auc": 0.55},
            _perm(0.55),  # margin = 0 < 0.05
            buffer=0.05,
        )
    assert any(
        "T2.2 ADVISORY" in record.message and "0.5500" in record.message
        for record in caplog.records
    ), "missing structured T2.2 ADVISORY warning with test AUC value"


def test_advisory_does_not_log_warning_when_met(caplog) -> None:
    """No noisy warning when the advisory is satisfied — tier0 daily runs
    on healthy cohorts must not flood the log."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    with caplog.at_level(logging.WARNING):
        _emit_permutation_anchored_auc_advisory(mr, {"roc_auc": 0.80}, _perm(0.55), buffer=0.05)
    assert not any("T2.2 ADVISORY" in record.message for record in caplog.records)


def test_advisory_buffer_can_be_overridden_per_call() -> None:
    """Forward-compat for cohort-specific overrides (e.g., a CSU
    operator electing buffer=0.10 because the cohort's 0.66 honest band
    sits closer to noise)."""
    mr: Dict[str, Any] = {"validation_metrics": {}}
    _emit_permutation_anchored_auc_advisory(mr, {"roc_auc": 0.62}, _perm(0.55), buffer=0.10)
    val = mr["validation_metrics"]
    assert val["permutation_anchored_auc_buffer"] == 0.10
    assert val["auc_above_permutation_null"] == pytest.approx(0.07)
    assert val["permutation_anchored_auc_advisory_violated"] is True


def test_advisory_creates_validation_metrics_when_absent() -> None:
    """`setdefault` semantics — caller need not pre-populate."""
    mr: Dict[str, Any] = {}
    _emit_permutation_anchored_auc_advisory(mr, {"roc_auc": 0.80}, _perm(0.55))
    assert "validation_metrics" in mr
    assert mr["validation_metrics"]["permutation_anchored_auc_advisory_violated"] is False


# --------------------------------------------------------------------------- #
# End-to-end integration through evaluate_model                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def real_classifier_state_for_t22():
    """Same fixture pattern as test_evaluator.py's ``real_classifier_state``."""
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
async def test_evaluate_model_emits_t22_advisory_on_validation_metrics(
    real_classifier_state_for_t22,
) -> None:
    """End-to-end: ``evaluate_model`` populates the three T2.2 advisory
    keys on ``validation_metrics`` after the perm test runs."""
    result = await evaluate_model(real_classifier_state_for_t22)
    val = result.get("validation_metrics", {})
    assert "auc_above_permutation_null" in val
    assert "permutation_anchored_auc_buffer" in val
    assert "permutation_anchored_auc_advisory_violated" in val
    assert val["permutation_anchored_auc_buffer"] == 0.05


@pytest.mark.asyncio
async def test_evaluate_model_t22_advisory_does_not_block_success_criteria(
    real_classifier_state_for_t22,
) -> None:
    """Even when the random-data RF triggers a T2.2 violation (likely on
    the 100-sample synthetic state), ``success_criteria_met`` is unaffected
    by the advisory — only by the legacy criteria."""
    result = await evaluate_model(real_classifier_state_for_t22)
    # Advisory key may be True or False depending on RF luck on 100 samples,
    # but success_criteria_met should NOT be derived from the advisory.
    advisory_violated = result["validation_metrics"]["permutation_anchored_auc_advisory_violated"]
    # Advisory does not appear in success_criteria_results.
    crit_results = result.get("success_criteria_results", {})
    assert "permutation_anchored_auc_advisory_violated" not in crit_results
    assert "auc_above_permutation_null" not in crit_results
    # success_criteria_met is True (no enforcement criteria configured in
    # the fixture's success_criteria={} dict — no advisory-induced flip).
    assert result.get("success_criteria_met") is True
    # Sanity: advisory key is a bool or None (not a string sentinel).
    assert advisory_violated in (True, False, None)
