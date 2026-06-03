"""Smoke + integration tests for ``evaluate_model``.

End-to-end coverage of the evaluator node: invokes ``evaluate_model``
with a populated state and asserts the high-level shape of the result
(metric blocks present, error branches reachable, sklearn classifiers
and regressors round-trip cleanly).

The focused helper-level tests live in sibling files (split out in
1A-M-6 to keep this file tractable):

* ``test_threshold_selection.py`` - threshold tuning, freezing,
  provenance, ``_select_threshold``.
* ``test_metrics_computation.py`` - ``_compute_precision_at_k``,
  ``_positive_class_proba``, business_utility from cost_matrix.
* ``test_provenance.py`` - ``_check_success_criteria`` audit fields.

Shared mocks and the minimal-state fixtures
(``binary_classification_state`` / ``regression_state``) live in
``conftest.py`` so all four files share a single source of truth.
"""

import math
from typing import Any, Dict

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    evaluate_model,
)
from tests.unit.test_agents.test_ml_foundation.test_model_trainer.conftest import (
    N_FEATURES,
    N_TEST_SAMPLES,
    N_TRAIN_SAMPLES,
    N_VAL_SAMPLES,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    MockClassifierNoProba,
)


@pytest.fixture
def real_classifier_state():
    """Create state with real trained classifier for accurate testing."""
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


@pytest.fixture
def real_regressor_state():
    """Create state with real trained regressor."""
    np.random.seed(RANDOM_STATE)
    X_train = np.random.rand(N_TRAIN_SAMPLES, N_FEATURES)
    y_train = np.random.rand(N_TRAIN_SAMPLES)
    X_val = np.random.rand(N_VAL_SAMPLES, N_FEATURES)
    y_val = np.random.rand(N_VAL_SAMPLES)
    X_test = np.random.rand(N_TEST_SAMPLES, N_FEATURES)
    y_test = np.random.rand(N_TEST_SAMPLES)

    model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return {
        "trained_model": model,
        "problem_type": "regression",
        "X_train_preprocessed": X_train,
        "X_validation_preprocessed": X_val,
        "X_test_preprocessed": X_test,
        "train_data": {"y": y_train},
        "validation_data": {"y": y_val},
        "test_data": {"y": y_test},
        "success_criteria": {},
    }


# ============================================================================
# Test evaluate_model function
# ============================================================================


@pytest.mark.asyncio
class TestEvaluateModel:
    """Test core model evaluation."""

    async def test_evaluates_on_all_splits(self, binary_classification_state):
        """Should evaluate on train, validation, and test splits."""
        result = await evaluate_model(binary_classification_state)

        assert "error" not in result
        assert "train_metrics" in result
        assert "validation_metrics" in result
        assert "test_metrics" in result

    async def test_returns_classification_metrics(self, binary_classification_state):
        """Should return classification metrics for classification problems."""
        result = await evaluate_model(binary_classification_state)

        assert "error" not in result
        assert result["auc_roc"] is not None
        assert result["precision"] is not None
        assert result["recall"] is not None
        assert result["f1_score"] is not None

    async def test_returns_regression_metrics(self, regression_state):
        """Should return regression metrics for regression problems."""
        result = await evaluate_model(regression_state)

        assert "error" not in result
        assert result["rmse"] is not None
        assert result["mae"] is not None
        assert result["r2"] is not None
        # Classification metrics should be None
        assert result["auc_roc"] is None
        assert result["precision"] is None

    async def test_checks_success_criteria(self, binary_classification_state):
        """Should check if model meets success criteria.

        Section B (pre_phase2_unblockers): the lift criterion participates
        in the aggregation alongside the other thresholds — when both
        train and test have enough samples (binary_classification_state
        uses N_TRAIN=100, N_TEST=20 — both ≥ 10), the baseline AUC is
        produced and the criterion is evaluated rather than soft-skipped.
        """
        binary_classification_state["success_criteria"] = {
            "accuracy": 0.90,
            "minimum_lift_over_baseline": 0.10,
        }

        result = await evaluate_model(binary_classification_state)

        assert "success_criteria_met" in result
        assert "success_criteria_results" in result
        assert "accuracy" in result["success_criteria_results"]
        # Lift criterion must be present in the results dict — the
        # MockBinaryClassifier's noisy predictions may or may not pass the
        # 0.10 threshold (deterministic since random_state is pinned), but
        # the criterion must have actually been checked (not soft-skipped).
        assert "minimum_lift_over_baseline" in result["success_criteria_results"]
        assert result["success_criteria_results"]["minimum_lift_over_baseline"] in (
            True,
            False,
        )
        # And the underlying metrics must be in test_metrics.
        assert "baseline_test_auc" in result["test_metrics"]
        assert "minimum_lift_over_baseline" in result["test_metrics"]

    async def test_success_criteria_met_when_threshold_passed(self, real_classifier_state):
        """Should set success_criteria_met=True when threshold is passed."""
        # Set very low threshold that should always be met
        real_classifier_state["success_criteria"] = {"accuracy": 0.1}

        result = await evaluate_model(real_classifier_state)

        assert result["success_criteria_met"] is True

    async def test_returns_confusion_matrix(self, binary_classification_state):
        """Should return confusion matrix for classification."""
        result = await evaluate_model(binary_classification_state)

        assert "confusion_matrix" in result
        assert result["confusion_matrix"] is not None

    async def test_returns_optimal_threshold(self, real_classifier_state):
        """Should compute optimal threshold for binary classification."""
        result = await evaluate_model(real_classifier_state)

        assert "optimal_threshold" in result
        # Threshold should be valid (0-1) or default (0.5)
        threshold = result["optimal_threshold"]
        assert isinstance(threshold, (int, float))
        assert 0.0 <= threshold <= 1.0 or threshold == 0.5

    async def test_returns_confidence_intervals(self, binary_classification_state):
        """Should compute bootstrap confidence intervals."""
        result = await evaluate_model(binary_classification_state)

        assert "confidence_interval" in result
        assert "bootstrap_samples" in result
        assert result["bootstrap_samples"] == 1000

    async def test_error_when_no_trained_model(self):
        """Should return error when trained_model is None."""
        state = {
            "problem_type": "binary_classification",
            "X_test_preprocessed": np.random.rand(N_TEST_SAMPLES, N_FEATURES),
            "test_data": {"y": np.random.randint(0, 2, N_TEST_SAMPLES)},
        }

        result = await evaluate_model(state)

        assert "error" in result
        assert result["error_type"] == "missing_trained_model"

    async def test_error_when_no_test_data(self, binary_classification_state):
        """Should return error when test data is missing."""
        del binary_classification_state["X_test_preprocessed"]
        del binary_classification_state["test_data"]

        result = await evaluate_model(binary_classification_state)

        assert "error" in result
        assert result["error_type"] == "missing_test_data"

    async def test_error_for_unsupported_problem_type(self, binary_classification_state):
        """Should return error for unsupported problem type."""
        binary_classification_state["problem_type"] = "unsupported_type"

        result = await evaluate_model(binary_classification_state)

        assert "error" in result
        assert result["error_type"] == "unsupported_problem_type"

    async def test_handles_model_without_predict_proba(self):
        """Should handle classifiers without predict_proba."""
        np.random.seed(RANDOM_STATE)
        state = {
            "trained_model": MockClassifierNoProba(),
            "problem_type": "binary_classification",
            "X_test_preprocessed": np.random.rand(N_TEST_SAMPLES, N_FEATURES),
            "test_data": {"y": np.random.randint(0, 2, N_TEST_SAMPLES)},
            "success_criteria": {},
        }

        result = await evaluate_model(state)

        # Should still succeed but without probability-based metrics
        assert "error" not in result
        assert result.get("auc_roc") is None  # No proba available

    async def test_evaluates_with_real_classifier(self, real_classifier_state):
        """Should evaluate real sklearn classifier correctly."""
        result = await evaluate_model(real_classifier_state)

        assert "error" not in result
        assert result["auc_roc"] is not None
        assert 0.0 <= result["auc_roc"] <= 1.0
        assert result["test_metrics"]["accuracy"] is not None

    async def test_evaluates_with_real_regressor(self, real_regressor_state):
        """Should evaluate real sklearn regressor correctly."""
        result = await evaluate_model(real_regressor_state)

        assert "error" not in result
        assert result["rmse"] is not None
        assert result["rmse"] >= 0
        assert result["mae"] is not None
        assert result["mae"] >= 0

    async def test_handles_missing_validation_data(self, binary_classification_state):
        """Should handle missing validation data gracefully.

        With no validation split, the per-split validation metrics (``roc_auc``,
        ``pr_auc``, etc.) are not produced. After backlog #18, the 5-fold CV
        summary IS still computed against the available train+test data and
        its scalar summary lands in ``validation_metrics`` as ``cv_5fold_*``
        keys. The semantic invariant is "no per-split validation metrics" —
        global / non-split-bound fields (e.g., a future calibration ECE
        applied across the whole pipeline, or the existing ``cv_5fold_*``
        summary) MAY appear here without violating the invariant
        (codex pass-1 MEDIUM-1: scope the assertion to per-split absence,
        not "only cv_5fold_*", so a future global-scoped field doesn't
        spuriously break this test).
        """
        del binary_classification_state["X_validation_preprocessed"]
        del binary_classification_state["validation_data"]

        result = await evaluate_model(binary_classification_state)

        assert "error" not in result
        val_metrics = result["validation_metrics"]
        # Per-split validation metrics must not be present — these are
        # produced inside the y_validation gate at evaluator.py:1188 and
        # would only appear if validation_data was somehow re-derived:
        for split_metric in ("roc_auc", "pr_auc", "f1_score", "precision", "recall"):
            assert split_metric not in val_metrics, (
                f"With no validation split, {split_metric!r} should not be in "
                f"validation_metrics; got {val_metrics!r}"
            )

    async def test_handles_continuous_problem_type(self, regression_state):
        """Should treat 'continuous' as regression."""
        regression_state["problem_type"] = "continuous"

        result = await evaluate_model(regression_state)

        assert "error" not in result
        assert result["rmse"] is not None


def test_check_success_criteria_skips_criteria_source_field() -> None:
    """The new ``criteria_source`` audit field must not be treated as a metric.

    Regression guard for the ADAPTIVE_CRITERIA flag plumbing in task 02 of
    .claude/plans/adaptive_success_criteria/. The validator now tags every
    ``success_criteria`` dict with a string ``criteria_source`` value;
    ``_check_success_criteria`` must route it through the existing
    skip-non-numeric branch (alongside ``experiment_id`` /
    ``baseline_model``) without recording a False result.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"roc_auc": 0.80}
    success_criteria = {
        "minimum_auc": 0.75,
        "criteria_source": "fixed",  # str, not numeric — must be skipped
        "experiment_id": "abc",  # existing precedent for skip-non-numeric
        "baseline_model": "stratified_dummy",
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")
    assert result["success_criteria_met"] is True
    assert "criteria_source" not in result["success_criteria_results"]
    assert "experiment_id" not in result["success_criteria_results"]
    assert result["success_criteria_results"]["minimum_auc"] is True


# ---------------------------------------------------------------------------
# Adaptive criteria evaluator-side support (task 04 of
# .claude/plans/adaptive_success_criteria/)
# ---------------------------------------------------------------------------


def test_check_success_criteria_records_met_None_for_adaptive_skipped() -> None:
    """v2/v3 contract: criterion names in
    ``success_criteria['_adaptive_skipped']`` are recorded as met=None in
    results, regardless of whether they were in the criteria dict at all.
    This is the EXPLICIT skip mechanism; plain None thresholds do NOT
    trigger this path.

    See .claude/plans/adaptive_success_criteria/01-design.md §"Skip semantics"
    and 04-evaluator-skip-exemption.md.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"recall": 0.60, "f1_score": 0.55}
    success_criteria = {
        "minimum_recall": 0.50,  # firing
        "minimum_f1": 0.55,  # firing
        # minimum_auc / minimum_precision NOT in dict at all (skipped)
        "_adaptive_skipped": ["minimum_auc", "minimum_precision"],
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

    # Skipped names recorded as met=None
    assert result["success_criteria_results"]["minimum_auc"] is None
    assert result["success_criteria_results"]["minimum_precision"] is None
    # Firing criteria evaluated normally
    assert result["success_criteria_results"]["minimum_recall"] is True
    assert result["success_criteria_results"]["minimum_f1"] is True
    # Aggregate: only firing criteria participate
    assert result["success_criteria_met"] is True


def test_check_success_criteria_adaptive_skipped_does_not_mask_real_fail() -> None:
    """Skipped names are excluded from aggregation, but firing criteria
    that fail still produce success_criteria_met=False."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"recall": 0.40}
    success_criteria = {
        "minimum_recall": 0.65,  # firing — and failing
        "_adaptive_skipped": ["minimum_auc"],  # explicit skip
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

    assert result["success_criteria_results"]["minimum_auc"] is None
    assert result["success_criteria_results"]["minimum_recall"] is False
    assert result["success_criteria_met"] is False


def test_check_success_criteria_plain_None_threshold_does_NOT_skip_via_v1_generalization() -> None:
    """v2 regression guard: a plain None threshold in success_criteria must
    NOT be recorded as met=None just because it's None. v1 proposed this
    generalization but it created an S4 silent-skip vulnerability — config
    typos like ``min_auc: null`` upstream would silently bypass the gate.

    The existing skip-non-numeric branch should silently drop plain None
    thresholds (continue without recording in results); the validator's
    S4 defense in ``_define_classification_criteria`` catches the typo
    upstream.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"roc_auc": 0.80}
    success_criteria = {
        "minimum_auc": None,  # plain None — NOT in _adaptive_skipped
        "minimum_recall": 0.65,
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

    # Plain None threshold MUST NOT appear as met=None in results
    # (would mask a config typo). It silently drops via the existing
    # skip-non-numeric branch.
    assert "minimum_auc" not in result["success_criteria_results"]
    # Other firing criteria still evaluate (recall is missing from
    # test_metrics, so it hard-fails — that's the existing behavior).
    assert result["success_criteria_results"]["minimum_recall"] is False


def test_check_success_criteria_missing_non_lift_metric_still_hard_fails() -> None:
    """Non-None threshold + missing metric (NOT in lift-exemption allowlist
    AND NOT in _adaptive_skipped) still hard-fails — the v2 explicit-skip
    must NOT collapse the missing-metric branch."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"roc_auc": 0.80}
    success_criteria = {
        "minimum_auc": 0.75,
        "minimum_recall": 0.65,  # threshold set but recall is MISSING from test_metrics
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

    assert result["success_criteria_results"]["minimum_auc"] is True
    assert result["success_criteria_results"]["minimum_recall"] is False
    assert result["success_criteria_met"] is False


def test_check_success_criteria_lift_exemption_still_works() -> None:
    """Section B's narrow exemption (parent branch) must keep working:
    threshold set, metric missing, criterion == minimum_lift_over_baseline ⇒ skip."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"roc_auc": 0.80}
    success_criteria = {
        "minimum_auc": 0.75,
        "minimum_lift_over_baseline": 0.10,  # threshold set, metric missing
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

    assert result["success_criteria_results"]["minimum_auc"] is True
    assert result["success_criteria_results"]["minimum_lift_over_baseline"] is None
    assert result["success_criteria_met"] is True


def test_check_success_criteria_calibration_error_alias() -> None:
    """``maximum_calibration_error`` resolves to ``calibrated_ece`` (B1 fix
    — the actual emit key in ``_compute_classification_metrics``) and
    applies lower-is-better semantics."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"calibrated_ece": 0.04}
    success_criteria = {"maximum_calibration_error": 0.05}
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")
    assert result["success_criteria_results"]["maximum_calibration_error"] is True

    test_metrics_fail = {"calibrated_ece": 0.08}
    result_fail = _check_success_criteria(
        test_metrics_fail, success_criteria, "binary_classification"
    )
    assert result_fail["success_criteria_results"]["maximum_calibration_error"] is False


def test_check_success_criteria_train_val_delta_alias() -> None:
    """``maximum_train_val_delta`` resolves to ``train_val_auc_delta`` with
    lower-is-better. The metric is emitted by the evaluator in task 05;
    this unit test seeds it directly to validate the alias resolution."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"train_val_auc_delta": 0.025}
    success_criteria = {"maximum_train_val_delta": 0.03}
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")
    assert result["success_criteria_results"]["maximum_train_val_delta"] is True

    test_metrics_fail = {"train_val_auc_delta": 0.06}
    result_fail = _check_success_criteria(
        test_metrics_fail, success_criteria, "binary_classification"
    )
    assert result_fail["success_criteria_results"]["maximum_train_val_delta"] is False


def test_check_success_criteria_adaptive_skipped_field_not_treated_as_metric() -> None:
    """The ``_adaptive_skipped`` audit field (a list) must hit the existing
    skip-non-numeric branch and NOT be treated as a numeric criterion."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"roc_auc": 0.80}
    success_criteria = {
        "minimum_auc": 0.75,
        "_adaptive_skipped": ["minimum_precision"],  # list — must be skipped
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

    # The list itself is NOT a criterion result.
    assert "_adaptive_skipped" not in result["success_criteria_results"]
    # The contents ARE recorded as met=None.
    assert result["success_criteria_results"]["minimum_precision"] is None
    assert result["success_criteria_results"]["minimum_auc"] is True


# ---------------------------------------------------------------------------
# v3 (Option C) — NEW gates and audit-field handling
# ---------------------------------------------------------------------------


def test_check_success_criteria_skips_underscore_prefix_audit_fields() -> None:
    """v3 invariant: keys starting with ``_`` are audit fields, never
    criteria. ``_adaptive_p_t`` is a float and would otherwise be evaluated
    as a numeric criterion against a missing ``adaptive_p_t`` test metric.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"roc_auc": 0.80}
    success_criteria = {
        "minimum_auc": 0.75,
        "_adaptive_p_t": 0.30,  # float audit field — must NOT be evaluated
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

    assert "_adaptive_p_t" not in result["success_criteria_results"]
    assert result["success_criteria_results"]["minimum_auc"] is True
    assert result["success_criteria_met"] is True


def test_check_success_criteria_nan_actual_value_records_met_none() -> None:
    """v3 B3 fix: a NaN actual value records met=None instead of comparing.

    Adverse-regime calibration metrics emit NaN when ``n_pos < 30`` (LR
    fit unstable); without the guard, ``nan <= 0.15`` evaluates to False
    and poisons the aggregate.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {
        "calibration_slope_deviation": float("nan"),
        "roc_auc": 0.80,
    }
    success_criteria = {
        "minimum_auc": 0.75,
        "maximum_calibration_slope_deviation": 0.15,
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

    # NaN comparison routed to met=None, not False.
    assert result["success_criteria_results"]["maximum_calibration_slope_deviation"] is None
    assert result["success_criteria_results"]["minimum_auc"] is True
    assert result["success_criteria_met"] is True


@pytest.mark.parametrize(
    "criterion,metric,actual,threshold,lower_is_better,expected_met",
    [
        # MCC (higher-is-better)
        ("minimum_mcc", "mcc", 0.50, 0.45, False, True),
        ("minimum_mcc", "mcc", 0.40, 0.45, False, False),
        # Calibration slope deviation (lower-is-better)
        (
            "maximum_calibration_slope_deviation",
            "calibration_slope_deviation",
            0.10,
            0.15,
            True,
            True,
        ),
        (
            "maximum_calibration_slope_deviation",
            "calibration_slope_deviation",
            0.20,
            0.15,
            True,
            False,
        ),
        # Calibration intercept magnitude (lower-is-better)
        (
            "maximum_calibration_intercept_magnitude",
            "calibration_intercept_magnitude",
            0.20,
            0.30,
            True,
            True,
        ),
        (
            "maximum_calibration_intercept_magnitude",
            "calibration_intercept_magnitude",
            0.40,
            0.30,
            True,
            False,
        ),
    ],
)
def test_check_success_criteria_resolves_v3_aliases(
    criterion: str,
    metric: str,
    actual: float,
    threshold: float,
    lower_is_better: bool,
    expected_met: bool,
) -> None:
    """v3 alias resolution: each new criterion resolves to the correct
    test_metrics key and applies the right comparison sense (higher-is-
    better for MCC, lower-is-better for calibration deviations).
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {metric: actual}
    success_criteria = {criterion: threshold}
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")
    assert result["success_criteria_results"][criterion] is expected_met


def test_check_success_criteria_lower_is_better_includes_v3_calibration_deviations() -> None:
    """Regression guard: calibration_slope_deviation and
    calibration_intercept_magnitude must be lower-is-better. A future change
    that drops them from the set would silently invert the gate sense.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    # Slope deviation: actual > threshold ⇒ should FAIL (lower-is-better).
    test_metrics = {"calibration_slope_deviation": 0.20}
    success_criteria = {"maximum_calibration_slope_deviation": 0.15}
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")
    assert result["success_criteria_results"]["maximum_calibration_slope_deviation"] is False

    # Intercept magnitude: actual > threshold ⇒ should FAIL.
    test_metrics2 = {"calibration_intercept_magnitude": 0.40}
    success_criteria2 = {"maximum_calibration_intercept_magnitude": 0.30}
    result2 = _check_success_criteria(test_metrics2, success_criteria2, "binary_classification")
    assert result2["success_criteria_results"]["maximum_calibration_intercept_magnitude"] is False


def test_check_success_criteria_nb_at_p_t_resolves_via_grid() -> None:
    """v3 W3 fix: ``minimum_net_benefit_at_p_t`` resolves against
    ``net_benefit_grid`` keyed on the regime's ``_adaptive_p_t``. The
    threshold is fixed at 0.0 (NB > 0 gate) and ``p_t`` is recorded on
    the audit field, NOT in the metric_aliases dict.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    # Clean regime: p_t=0.30, NB > 0 ⇒ pass.
    test_metrics = {
        "net_benefit_grid": {
            "p_t=0.05": 0.018,
            "p_t=0.20": 0.012,
            "p_t=0.30": 0.005,
        }
    }
    success_criteria = {
        "minimum_net_benefit_at_p_t": 0.0,
        "_adaptive_p_t": 0.30,
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")
    assert result["success_criteria_results"]["minimum_net_benefit_at_p_t"] is True

    # Clean regime, model worse than treat-none: NB ≤ 0 at p_t=0.30 ⇒ fail.
    test_metrics_fail = {
        "net_benefit_grid": {
            "p_t=0.30": -0.010,
        }
    }
    result_fail = _check_success_criteria(
        test_metrics_fail, success_criteria, "binary_classification"
    )
    assert result_fail["success_criteria_results"]["minimum_net_benefit_at_p_t"] is False


def test_check_success_criteria_nb_at_p_t_soft_skips_when_grid_missing() -> None:
    """NB gate soft-skips (met=None) when ``net_benefit_grid`` is absent
    from test_metrics or the regime's p_t key is missing from the grid.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    # No net_benefit_grid at all ⇒ soft-skip.
    test_metrics: Dict[str, Any] = {"roc_auc": 0.80}
    success_criteria = {
        "minimum_auc": 0.75,
        "minimum_net_benefit_at_p_t": 0.0,
        "_adaptive_p_t": 0.30,
    }
    result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")
    assert result["success_criteria_results"]["minimum_net_benefit_at_p_t"] is None
    assert result["success_criteria_results"]["minimum_auc"] is True
    # Other criteria can still aggregate to True.
    assert result["success_criteria_met"] is True

    # Grid present but missing the requested p_t key ⇒ soft-skip.
    test_metrics2 = {"net_benefit_grid": {"p_t=0.20": 0.012}}
    result2 = _check_success_criteria(
        test_metrics2,
        {"minimum_net_benefit_at_p_t": 0.0, "_adaptive_p_t": 0.30},
        "binary_classification",
    )
    assert result2["success_criteria_results"]["minimum_net_benefit_at_p_t"] is None


# ---------------------------------------------------------------------------
# Task 05 — calibration helpers, NB grid, overlay, end-to-end
# ---------------------------------------------------------------------------


def test_compute_calibration_slope_intercept_perfect_calibration() -> None:
    """Calibrated logits (slope=1, intercept=0) recover perfectly."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_calibration_slope_intercept,
    )

    rng = np.random.default_rng(42)
    n = 1000
    # Generate logits, sigmoid them, then sample y from the resulting
    # probabilities. The recovered slope/intercept will sit near 1.0/0.0
    # with sample-size-bound jitter.
    z = rng.normal(0.0, 1.5, n)
    p = 1.0 / (1.0 + np.exp(-z))
    y_true = (rng.uniform(0.0, 1.0, n) < p).astype(int)

    slope, intercept = _compute_calibration_slope_intercept(y_true, p)

    assert not math.isnan(slope)
    assert not math.isnan(intercept)
    # Generous tolerance — n=1000 with stochastic sampling.
    assert abs(slope - 1.0) < 0.20, f"slope={slope}"
    assert abs(intercept) < 0.20, f"intercept={intercept}"


def test_compute_calibration_slope_intercept_overconfident() -> None:
    """Over-confident probabilities (squished toward 0/1) → slope < 1."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_calibration_slope_intercept,
    )

    rng = np.random.default_rng(7)
    n = 1000
    z = rng.normal(0.0, 1.5, n)
    p_true = 1.0 / (1.0 + np.exp(-z))
    y_true = (rng.uniform(0.0, 1.0, n) < p_true).astype(int)
    # Sharpen the probabilities (multiply logits by 2) to simulate
    # over-confidence. The recovered slope should be < 1.
    p_overconfident = 1.0 / (1.0 + np.exp(-2.0 * z))

    slope, _ = _compute_calibration_slope_intercept(y_true, p_overconfident)

    assert not math.isnan(slope)
    assert slope < 0.85, f"expected over-confident slope < 0.85, got {slope}"


def test_compute_calibration_slope_intercept_skips_at_low_n_pos() -> None:
    """n_pos < 30 ⇒ returns (nan, nan); LR fit is unstable below that."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_calibration_slope_intercept,
    )

    rng = np.random.default_rng(0)
    # Adverse-regime synthetic: 900 rows, 18 positives.
    y_true = np.concatenate([np.ones(18), np.zeros(882)]).astype(int)
    rng.shuffle(y_true)
    p = rng.uniform(0.01, 0.10, 900)

    slope, intercept = _compute_calibration_slope_intercept(y_true, p)
    assert math.isnan(slope)
    assert math.isnan(intercept)


def test_compute_net_benefit_at_p_t_known_counts() -> None:
    """Known TP/FP counts → matches Vickers 2006 formula."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_net_benefit_at_p_t,
    )

    # 100 rows; predict positive when p >= p_t. Construct so that exactly
    # TP=20, FP=10 at p_t=0.30, FN=20, TN=50.
    n = 100
    y_true = np.concatenate([np.ones(40), np.zeros(60)]).astype(int)
    # Place the first 20 positives + 10 negatives above 0.30, the rest
    # below.
    p = np.concatenate(
        [
            np.full(20, 0.50),  # TP=20
            np.full(20, 0.10),  # FN=20
            np.full(10, 0.50),  # FP=10
            np.full(50, 0.10),  # TN=50
        ]
    )
    p_t = 0.30
    # NB = TP/n - (FP/n) * p_t / (1 - p_t)
    #    = 20/100 - 10/100 * 0.30/0.70
    #    = 0.20 - 0.0428...
    expected = 20 / n - (10 / n) * p_t / (1.0 - p_t)

    nb = _compute_net_benefit_at_p_t(y_true, p, p_t)
    assert nb == pytest.approx(expected, abs=1e-9)


def test_compute_net_benefit_at_p_t_invalid_inputs() -> None:
    """``p_t`` out of (0, 1) ⇒ NaN; empty y_true ⇒ NaN."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_net_benefit_at_p_t,
    )

    y = np.array([1, 0, 1])
    p = np.array([0.5, 0.2, 0.8])
    assert math.isnan(_compute_net_benefit_at_p_t(y, p, 0.0))
    assert math.isnan(_compute_net_benefit_at_p_t(y, p, 1.0))
    assert math.isnan(_compute_net_benefit_at_p_t(y, p, -0.1))
    assert math.isnan(_compute_net_benefit_at_p_t(np.array([]), np.array([]), 0.3))


def test_compute_classification_metrics_emits_v3_keys() -> None:
    """``_compute_classification_metrics`` emits all v3 metrics on the inner
    ``test_metrics`` dict so downstream ``_check_success_criteria`` can
    resolve the v3 active gates."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_classification_metrics,
    )

    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 600, 150, 150

    def _gen_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = rng.normal(0.0, 1.5, n)
        p = 1.0 / (1.0 + np.exp(-z))
        y = (rng.uniform(0.0, 1.0, n) < p).astype(int)
        proba = np.column_stack([1.0 - p, p])
        pred = (p >= 0.5).astype(int)
        return y, pred, proba

    y_train, y_train_pred, y_train_proba = _gen_split(n_train)
    y_val, y_val_pred, y_val_proba = _gen_split(n_val)
    y_test, y_test_pred, y_test_proba = _gen_split(n_test)

    result = _compute_classification_metrics(
        y_train,
        y_train_pred,
        y_train_proba,
        y_val,
        y_val_pred,
        y_val_proba,
        y_test,
        y_test_pred,
        y_test_proba,
    )

    test_metrics = result["test_metrics"]
    # B2: train_val_auc_delta surfaced.
    assert "train_val_auc_delta" in test_metrics
    assert test_metrics["train_val_auc_delta"] >= 0.0  # absolute value
    # MCC already present from _compute_split_classification_metrics.
    assert "mcc" in test_metrics
    # v3 calibration emits.
    assert "calibration_slope" in test_metrics
    assert "calibration_intercept" in test_metrics
    assert "calibration_slope_deviation" in test_metrics
    assert "calibration_intercept_magnitude" in test_metrics
    # v3 NB grid: 6 entries keyed by canonical p_t strings.
    assert "net_benefit_grid" in test_metrics
    grid = test_metrics["net_benefit_grid"]
    assert isinstance(grid, dict)
    assert len(grid) == 6
    expected_keys = {
        "p_t=0.05",
        "p_t=0.10",
        "p_t=0.20",
        "p_t=0.30",
        "p_t=0.40",
        "p_t=0.50",
    }
    assert set(grid.keys()) == expected_keys
    # B2 sanity: gap matches train and val AUC.
    expected_gap = abs(result["train_metrics"]["roc_auc"] - result["validation_metrics"]["roc_auc"])
    assert test_metrics["train_val_auc_delta"] == pytest.approx(expected_gap, abs=1e-9)


def test_apply_adaptive_overlay_applies_v3_tuple() -> None:
    """Overlay reads ``_adaptive_inputs`` + ``baseline_test_auc``, computes
    v3 thresholds, and applies the (thresholds, skipped) tuple. v3 invariant:
    skipped criteria are REMOVED, deprecated precision/F1 are popped, and
    ``_adaptive_p_t`` is set from the regime."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _apply_adaptive_criteria_overlay,
    )

    success_criteria = {
        "minimum_auc": 0.75,
        "minimum_precision": 0.70,  # popped (v3-deprecated)
        "minimum_recall": 0.65,
        "minimum_f1": 0.70,  # popped (v3-deprecated)
        "minimum_lift_over_baseline": 0.10,
        "_adaptive_inputs": {
            "n_samples": 900,
            "prevalence": 0.02,
            "feature_count": 14,
            "regime": "adverse",
        },
        "criteria_source": "adaptive",
        "experiment_id": "exp-overlay",
    }
    test_metrics = {"roc_auc": 0.78, "baseline_test_auc": 0.50}

    overlaid = _apply_adaptive_criteria_overlay(success_criteria, test_metrics)

    # Adverse regime, prev=0.02:
    #   thresholds = {auc=0.70, recall=0.50, NB=0.0, MCC=0.20, csd=0.15,
    #                 cim=0.30, ECE=0.10, train_val_delta=0.03}
    #   skipped = {minimum_lift_over_baseline}
    assert overlaid["minimum_auc"] == pytest.approx(0.70, abs=1e-6)
    assert overlaid["minimum_recall"] == pytest.approx(0.50, abs=1e-6)
    assert overlaid["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    assert overlaid["minimum_mcc"] == pytest.approx(0.20, abs=1e-6)
    assert overlaid["maximum_calibration_slope_deviation"] == pytest.approx(0.15, abs=1e-6)
    assert overlaid["maximum_calibration_intercept_magnitude"] == pytest.approx(0.30, abs=1e-6)
    # v3-deprecated keys popped — no precision/F1.
    assert "minimum_precision" not in overlaid
    assert "minimum_f1" not in overlaid
    # Skipped: lift only.
    assert "minimum_lift_over_baseline" not in overlaid
    assert overlaid["_adaptive_skipped"] == ["minimum_lift_over_baseline"]
    # v3 audit: p_t for adverse = 0.05.
    assert overlaid["_adaptive_p_t"] == pytest.approx(0.05, abs=1e-6)
    # Non-adaptive-managed keys preserved.
    assert overlaid["criteria_source"] == "adaptive"
    assert overlaid["experiment_id"] == "exp-overlay"


def test_apply_adaptive_overlay_noop_when_not_adaptive() -> None:
    """Without ``_adaptive_inputs``, overlay returns success_criteria unchanged."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _apply_adaptive_criteria_overlay,
    )

    success_criteria = {
        "minimum_auc": 0.75,
        "minimum_precision": 0.70,
        "criteria_source": "fixed",
    }
    test_metrics = {"roc_auc": 0.80, "baseline_test_auc": 0.50}

    overlaid = _apply_adaptive_criteria_overlay(success_criteria, test_metrics)
    assert overlaid == success_criteria  # exact equality — no overwrite


def test_apply_adaptive_overlay_noop_without_baseline_test_auc() -> None:
    """Adaptive inputs present but baseline_test_auc absent (degenerate
    split) ⇒ leave success_criteria unchanged. The validator owns
    criteria_source; the overlay does not touch it.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _apply_adaptive_criteria_overlay,
    )

    success_criteria = {
        "minimum_auc": 0.75,
        "_adaptive_inputs": {
            "n_samples": 900,
            "prevalence": 0.02,
            "feature_count": 14,
            "regime": "adverse",
        },
    }
    test_metrics = {"roc_auc": 0.78}  # no baseline_test_auc

    overlaid = _apply_adaptive_criteria_overlay(success_criteria, test_metrics)
    assert overlaid["minimum_auc"] == 0.75
    assert "_adaptive_skipped" not in overlaid
    assert "_adaptive_p_t" not in overlaid


def test_apply_adaptive_overlay_default_regime_sets_p_t_0_20() -> None:
    """Default-regime overlay sets ``_adaptive_p_t = 0.20`` (Vickers 2019
    rubric-stress cost ratio 4:1)."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _apply_adaptive_criteria_overlay,
    )

    sc = {
        "minimum_auc": 0.75,
        "_adaptive_inputs": {
            "n_samples": 900,
            "prevalence": 0.30,
            "feature_count": 14,
            "regime": "default",
        },
    }
    overlaid = _apply_adaptive_criteria_overlay(sc, {"baseline_test_auc": 0.50})
    assert overlaid["_adaptive_p_t"] == pytest.approx(0.20, abs=1e-6)
    # Default regime drops min_auc.
    assert "minimum_auc" not in overlaid
    assert "minimum_auc" in overlaid["_adaptive_skipped"]


def test_apply_adaptive_overlay_clean_regime_sets_p_t_0_30() -> None:
    """Clean-regime overlay sets ``_adaptive_p_t = 0.30``."""
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _apply_adaptive_criteria_overlay,
    )

    sc = {
        "minimum_auc": 0.75,
        "_adaptive_inputs": {
            "n_samples": 900,
            "prevalence": 0.50,
            "feature_count": 14,
            "regime": "clean",
        },
    }
    overlaid = _apply_adaptive_criteria_overlay(sc, {"baseline_test_auc": 0.50})
    assert overlaid["_adaptive_p_t"] == pytest.approx(0.30, abs=1e-6)


def test_check_success_criteria_with_adaptive_overlay_end_to_end() -> None:
    """Wire-end test: validator stashes ``_adaptive_inputs``, evaluator
    overlay rewrites the criteria dict and ``_check_success_criteria``
    aggregates correctly. Post-PR-#30 wiring fix: the overlay is now
    applied by ``evaluate_model`` BEFORE ``_check_success_criteria`` is
    called, so this test applies it explicitly to mirror the new
    contract.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _apply_adaptive_criteria_overlay,
        _check_success_criteria,
    )

    # Default regime, prev=0.30: AUC dropped (skipped), MCC=0.35,
    # NB=0.0 at p_t=0.20, calibration gates fire, lift threshold 0.10.
    success_criteria = {
        "minimum_auc": 0.75,  # popped by overlay (default skip)
        "minimum_precision": 0.70,  # popped (v3-deprecated)
        "minimum_recall": 0.65,
        "minimum_f1": 0.70,  # popped (v3-deprecated)
        "minimum_lift_over_baseline": 0.10,
        "_adaptive_inputs": {
            "n_samples": 900,
            "prevalence": 0.30,
            "feature_count": 14,
            "regime": "default",
        },
        "criteria_source": "adaptive",
        "experiment_id": "e2e",
        "baseline_model": "stratified_dummy",
    }
    test_metrics = {
        "roc_auc": 0.62,
        "precision": 0.55,
        "recall": 0.62,  # < 0.65 → fails
        "f1_score": 0.583,
        "mcc": 0.40,  # > 0.35 → passes
        "minimum_lift_over_baseline": 0.12,  # > 0.10 → passes
        "baseline_test_auc": 0.50,
        "calibrated_ece": 0.04,  # < 0.10 → passes
        "train_val_auc_delta": 0.02,  # < 0.03 → passes
        "calibration_slope_deviation": 0.10,  # < 0.15 → passes
        "calibration_intercept_magnitude": 0.20,  # < 0.30 → passes
        "net_benefit_grid": {"p_t=0.20": 0.005},  # > 0.0 → passes
    }

    # Apply the overlay first — mirrors what evaluate_model now does.
    overlaid = _apply_adaptive_criteria_overlay(success_criteria, test_metrics)
    result = _check_success_criteria(test_metrics, overlaid, "binary_classification")

    # AUC removed by adaptive ⇒ recorded as met=None via post-loop pass.
    assert result["success_criteria_results"]["minimum_auc"] is None
    # Precision/F1 popped — they don't show up in results at all.
    assert "minimum_precision" not in result["success_criteria_results"]
    assert "minimum_f1" not in result["success_criteria_results"]
    # Recall fires and FAILS.
    assert result["success_criteria_results"]["minimum_recall"] is False
    # NB > 0 at p_t=0.20: 0.005 > 0.0 → True.
    assert result["success_criteria_results"]["minimum_net_benefit_at_p_t"] is True
    assert result["success_criteria_results"]["minimum_mcc"] is True
    assert result["success_criteria_results"]["maximum_calibration_slope_deviation"] is True
    assert result["success_criteria_results"]["maximum_calibration_intercept_magnitude"] is True
    assert result["success_criteria_results"]["maximum_calibration_error"] is True
    assert result["success_criteria_results"]["maximum_train_val_delta"] is True
    assert result["success_criteria_results"]["minimum_lift_over_baseline"] is True
    # Aggregate False because recall fails.
    assert result["success_criteria_met"] is False


@pytest.mark.asyncio
class TestPostHocCalibrationGate:
    """Phase 1 W2 day-2: gate the post-hoc isotonic calibration block on
    `model_candidate.skip_post_hoc_calibration`. Calibration-native algorithms
    (NGBoost, MAPIE) ship pre-calibrated predict_proba; layering isotonic on
    top tends to over-fit small validation sets and degrade test calibration
    (Duan et al. 2020 §4). Ref: shard 19 §A.7.
    """

    async def test_isotonic_runs_when_no_model_candidate_legacy_default(
        self, real_classifier_state
    ):
        """Backward compat: state without model_candidate gets isotonic (legacy behavior)."""
        result = await evaluate_model(real_classifier_state)
        assert "post_hoc_calibration" in result
        cal = result["post_hoc_calibration"]
        # Isotonic ran (calibration_applied=True with X_val + y_val present)
        assert cal.get("calibration_applied") is True
        assert cal.get("skip_reason") != "skip_post_hoc_calibration_flag"

    async def test_isotonic_runs_when_flag_explicitly_false(self, real_classifier_state):
        """Explicit False keeps isotonic on (same as legacy)."""
        real_classifier_state["model_candidate"] = {
            "algorithm_name": "LightGBM",
            "skip_post_hoc_calibration": False,
        }
        result = await evaluate_model(real_classifier_state)
        assert "post_hoc_calibration" in result
        cal = result["post_hoc_calibration"]
        assert cal.get("calibration_applied") is True
        assert cal.get("skip_reason") != "skip_post_hoc_calibration_flag"

    async def test_isotonic_skipped_when_flag_true(self, real_classifier_state):
        """Calibration-native: skip flag prevents isotonic; metadata records skip reason."""
        real_classifier_state["model_candidate"] = {
            "algorithm_name": "NGBoost",
            "skip_post_hoc_calibration": True,
        }
        result = await evaluate_model(real_classifier_state)
        assert "post_hoc_calibration" in result
        cal = result["post_hoc_calibration"]
        assert cal.get("calibration_applied") is False
        assert cal.get("skip_reason") == "skip_post_hoc_calibration_flag"
        # No calibrated_test_metrics added because no calibrated model created.
        assert "calibrated_test_metrics" not in result

    async def test_skip_path_copies_native_ece_into_calibrated_ece_alias(
        self, real_classifier_state
    ):
        """Cycle-8 codex IMPORTANT finding fix.

        When `skip_post_hoc_calibration=True`, the alias resolution
        `maximum_calibration_error → calibrated_ece` (evaluator line 1759)
        would otherwise return None (no isotonic ECE was computed) and
        hard-fail the criterion at line 1818, even though the native
        calibration-native ECE IS available at
        `metrics_result["calibration_error"]` (line 265). The skip path
        copies the native uncalibrated ECE into both `metrics_result` and
        `test_metrics["calibrated_ece"]` so the alias resolves to the
        native value (lower-is-better; NGBoost's calibration-native ECE
        IS the best-available calibration estimate without isotonic).
        """
        real_classifier_state["model_candidate"] = {
            "algorithm_name": "NGBoost",
            "skip_post_hoc_calibration": True,
        }
        result = await evaluate_model(real_classifier_state)
        # Native uncal ECE is computed at line 265 regardless of the gate.
        native_ece = result.get("calibration_error")
        assert native_ece is not None, (
            "calibration_error should be present at metrics_result level even when "
            "isotonic is skipped (computed at evaluator.py line 265)"
        )
        # Skip path must copy native ECE into calibrated_ece alias on both layers.
        assert result.get("calibrated_ece") == native_ece
        test_metrics = result.get("test_metrics", {})
        assert test_metrics.get("calibrated_ece") == native_ece


# ============================================================================
# Tier 1B step 1 — perm-null promotion to validation_metrics (codex MEDIUM-2)
# ============================================================================


@pytest.mark.asyncio
class TestEvaluatePermNullPromotion:
    """End-to-end: when `evaluate_model` runs the binary-classification
    path, the new `permutation_null_p95`, `permutation_null_p99`, and
    `permutation_n_permutations` keys land on `validation_metrics` with
    the `DEFAULT_PERMUTATION_COUNT=200` round-tripped from the callsite."""

    async def test_validation_metrics_carries_perm_n_permutations_default(
        self, real_classifier_state
    ):
        """Asserts the evaluator callsite passes
        `n_permutations=DEFAULT_PERMUTATION_COUNT` (not a stale 100). If
        someone reverts the default at the callsite this test breaks."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            DEFAULT_PERMUTATION_COUNT,
        )

        result = await evaluate_model(real_classifier_state)
        val_metrics = result.get("validation_metrics", {})
        assert val_metrics.get("permutation_n_permutations") == DEFAULT_PERMUTATION_COUNT
        assert val_metrics["permutation_n_permutations"] == 200

    async def test_validation_metrics_carries_perm_null_percentiles(self, real_classifier_state):
        """The new `permutation_null_p95` and `permutation_null_p99` keys
        are promoted to validation_metrics and lie in the AUC bounds."""
        result = await evaluate_model(real_classifier_state)
        val_metrics = result.get("validation_metrics", {})
        p95 = val_metrics.get("permutation_null_p95")
        p99 = val_metrics.get("permutation_null_p99")
        assert p95 is not None
        assert p99 is not None
        assert 0.0 <= p95 <= 1.0
        assert 0.0 <= p99 <= 1.0
        # By construction p95 <= p99.
        assert p95 <= p99

    async def test_perm_test_subdict_preserved_after_promotion(self, real_classifier_state):
        """The sub-dict `metrics_result["permutation_test"]` retains all
        the original keys (including the legacy `n_permutations` alias and
        `actual_auc`/`signal_genuine`) — the promoter only LIFTS scalar
        keys, never moves or removes them."""
        result = await evaluate_model(real_classifier_state)
        perm = result.get("permutation_test", {})
        # Legacy alias preserved on sub-dict.
        assert "n_permutations" in perm
        assert perm["n_permutations"] == 200
        # Other expected sub-dict keys present.
        assert "actual_auc" in perm
        assert "signal_genuine" in perm
        assert "permutation_pvalue" in perm


# ============================================================================
# Findings #5 + #6 — operating-point consistency in the evaluator
# ============================================================================


def _gen_balanced_split_optimal_below_half(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a balanced split whose validation Youden optimum lands well
    BELOW 0.5, and where negatives carry probabilities in [optimal, 0.5)
    so the optimal-threshold confusion matrix has false positives the 0.5
    matrix does not — i.e. headline precision@0.5 != precision@optimal.

    Returns ``(y, y_pred_at_0.5, proba_2col, proba_pos)``.
    """
    y = rng.integers(0, 2, n)
    p = np.empty(n)
    for i in range(n):
        # positives mostly above the ~0.36 optimum (some straddle 0.5);
        # negatives spread across [0.10, 0.48) so the optimal threshold
        # picks up FPs that the 0.5 threshold does not.
        p[i] = rng.uniform(0.35, 0.60) if y[i] == 1 else rng.uniform(0.10, 0.48)
    proba = np.column_stack([1.0 - p, p])
    pred_at_half = (p >= 0.5).astype(int)
    return y, pred_at_half, proba, p


def test_balanced_headline_confusion_matrix_matches_headline_precision() -> None:
    """Findings #5: in the BALANCED path (imbalance_detected=False) the
    headline precision/recall/f1 are reported at 0.5, so the headline
    confusion_matrix and business_utility MUST be at 0.5 too. Before the fix
    they were computed at the validation-tuned optimal threshold, making
    precision recomputed from the headline confusion matrix diverge from the
    headline precision whenever the optimal threshold != 0.5.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_classification_metrics,
    )

    rng = np.random.default_rng(11)
    y_tr, ytr_pred, ytr_proba, _ = _gen_balanced_split_optimal_below_half(400, rng)
    y_val, yval_pred, yval_proba, _ = _gen_balanced_split_optimal_below_half(200, rng)
    y_te, yte_pred, yte_proba, _ = _gen_balanced_split_optimal_below_half(200, rng)

    cost_matrix = {"tp": 1.0, "fp": -1.0, "fn": -5.0, "tn": 0.0}
    result = _compute_classification_metrics(
        y_tr,
        ytr_pred,
        ytr_proba,
        y_val,
        yval_pred,
        yval_proba,
        y_te,
        yte_pred,
        yte_proba,
        imbalance_detected=False,
        cost_matrix=cost_matrix,
    )

    # Precondition: the optimal threshold is meaningfully != 0.5, otherwise
    # the bug is unobservable and the test would be vacuous.
    assert not math.isclose(result["optimal_threshold"], 0.5), (
        "test fixture must produce an optimal threshold != 0.5 to exercise the bug"
    )

    test_metrics = result["test_metrics"]
    cm = result["confusion_matrix"]
    assert set(cm) >= {"TP", "TN", "FP", "FN"}, "expected a 2x2 confusion dict"

    denom = cm["TP"] + cm["FP"]
    precision_from_cm = cm["TP"] / denom if denom > 0 else 0.0
    # Findings #5 core assertion: headline precision is computed at the SAME
    # operating point as the headline confusion matrix.
    assert precision_from_cm == pytest.approx(test_metrics["precision"], abs=1e-9)

    # And recall too, from the same matrix.
    rdenom = cm["TP"] + cm["FN"]
    recall_from_cm = cm["TP"] / rdenom if rdenom > 0 else 0.0
    assert recall_from_cm == pytest.approx(test_metrics["recall"], abs=1e-9)

    # The dedicated dual-operating-point keys still carry BOTH points: the
    # balanced headline equals the 0.5 metrics, and the optimal metrics are
    # genuinely different (the optimal threshold != 0.5 here).
    at_05 = result["test_metrics_at_05"]
    at_optimal = result["test_metrics_at_optimal"]
    assert at_05["precision"] == pytest.approx(test_metrics["precision"], abs=1e-9)
    assert at_05["recall"] == pytest.approx(test_metrics["recall"], abs=1e-9)
    assert "precision" in at_optimal and "recall" in at_optimal
    # Sanity: the two operating points genuinely differ on this fixture, so a
    # mismatch between headline and CM would have been observable pre-fix.
    assert at_05["precision"] != pytest.approx(at_optimal["precision"], abs=1e-9)


def test_imbalanced_headline_confusion_matrix_still_at_optimal() -> None:
    """Findings #5 guard: the IMBALANCED path is UNCHANGED — its headline
    precision/recall and headline confusion matrix are both at the
    validation-frozen optimal threshold (the optimal metrics), and stay
    mutually consistent.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _compute_classification_metrics,
    )

    rng = np.random.default_rng(11)
    y_tr, ytr_pred, ytr_proba, _ = _gen_balanced_split_optimal_below_half(400, rng)
    y_val, yval_pred, yval_proba, _ = _gen_balanced_split_optimal_below_half(200, rng)
    y_te, yte_pred, yte_proba, _ = _gen_balanced_split_optimal_below_half(200, rng)

    result = _compute_classification_metrics(
        y_tr,
        ytr_pred,
        ytr_proba,
        y_val,
        yval_pred,
        yval_proba,
        y_te,
        yte_pred,
        yte_proba,
        imbalance_detected=True,
    )

    test_metrics = result["test_metrics"]
    cm = result["confusion_matrix"]
    denom = cm["TP"] + cm["FP"]
    precision_from_cm = cm["TP"] / denom if denom > 0 else 0.0
    assert precision_from_cm == pytest.approx(test_metrics["precision"], abs=1e-9)
    # Imbalanced headline equals the optimal-threshold metrics, not the 0.5 ones.
    at_optimal = result["test_metrics_at_optimal"]
    assert test_metrics["precision"] == pytest.approx(at_optimal["precision"], abs=1e-9)


@pytest.mark.asyncio
async def test_deployed_calibrated_imbalanced_records_youden_threshold_source() -> None:
    """Findings #6: the deployed-calibrated imbalanced overlay re-derives the
    operating point via ``_select_threshold`` (Youden / cost-optimal only),
    which does NOT reproduce the raw path's F1-fallback. The overlaid metrics
    must therefore carry the ACTUAL ``chosen_threshold_source`` returned by
    ``_select_threshold`` (never the false ``validation_f1_fallback``), so a
    consumer is not misled into thinking the deployed point mirrors the raw
    ``chosen_threshold_source``.
    """
    np.random.seed(RANDOM_STATE)
    n_tr, n_val, n_te = 200, 80, 80
    # Low-signal, imbalanced-ish cohort: pushes validation MCC low so the raw
    # path's F1-fallback CAN engage — exactly the case the deployed-calibrated
    # overlay does not reproduce.
    X_tr = np.random.rand(n_tr, N_FEATURES)
    y_tr = (np.random.rand(n_tr) < 0.25).astype(int)
    X_val = np.random.rand(n_val, N_FEATURES)
    y_val = (np.random.rand(n_val) < 0.25).astype(int)
    X_te = np.random.rand(n_te, N_FEATURES)
    y_te = (np.random.rand(n_te) < 0.25).astype(int)

    model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)

    state = {
        "trained_model": model,
        "problem_type": "binary_classification",
        "X_train_preprocessed": X_tr,
        "X_validation_preprocessed": X_val,
        "X_test_preprocessed": X_te,
        "train_data": {"y": y_tr},
        "validation_data": {"y": y_val},
        "test_data": {"y": y_te},
        "success_criteria": {},
        "imbalance_detected": True,
        "minority_ratio": 0.25,
    }

    result = await evaluate_model(state)

    # Precondition: post-hoc calibration was applied and deployed.
    assert result.get("calibration_applied") is True
    test_metrics = result.get("test_metrics", {})
    assert test_metrics.get("deployed_model_is_calibrated") is True

    # Findings #6 core assertion: the overlaid operating-point provenance is
    # present and is one of the literals ``_select_threshold`` can return —
    # the Youden / cost-optimal / default arms — NEVER the F1-fallback literal,
    # which this overlay does not reproduce.
    source = test_metrics.get("chosen_threshold_source")
    assert source is not None, (
        "deployed-calibrated overlay must record chosen_threshold_source so the "
        "operating-point provenance is honest"
    )
    assert source in {"validation", "validation_cost_optimal", "default"}
    assert source != "validation_f1_fallback"
    # The recorded threshold is consistent with the overlaid source: it was the
    # value used to binarise the calibrated test probabilities.
    assert "chosen_threshold" in test_metrics
