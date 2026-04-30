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
        """Should handle missing validation data gracefully."""
        del binary_classification_state["X_validation_preprocessed"]
        del binary_classification_state["validation_data"]

        result = await evaluate_model(binary_classification_state)

        assert "error" not in result
        assert result["validation_metrics"] == {}

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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )
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
        "minimum_recall": 0.50,                      # firing
        "minimum_f1": 0.55,                          # firing
        # minimum_auc / minimum_precision NOT in dict at all (skipped)
        "_adaptive_skipped": ["minimum_auc", "minimum_precision"],
    }
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )

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
        "minimum_recall": 0.65,                      # firing — and failing
        "_adaptive_skipped": ["minimum_auc"],        # explicit skip
    }
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )

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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )

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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )

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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )

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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )
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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )
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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )

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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )

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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )

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
        ("maximum_calibration_slope_deviation", "calibration_slope_deviation", 0.10, 0.15, True, True),
        ("maximum_calibration_slope_deviation", "calibration_slope_deviation", 0.20, 0.15, True, False),
        # Calibration intercept magnitude (lower-is-better)
        ("maximum_calibration_intercept_magnitude", "calibration_intercept_magnitude", 0.20, 0.30, True, True),
        ("maximum_calibration_intercept_magnitude", "calibration_intercept_magnitude", 0.40, 0.30, True, False),
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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )
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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )
    assert result["success_criteria_results"]["maximum_calibration_slope_deviation"] is False

    # Intercept magnitude: actual > threshold ⇒ should FAIL.
    test_metrics2 = {"calibration_intercept_magnitude": 0.40}
    success_criteria2 = {"maximum_calibration_intercept_magnitude": 0.30}
    result2 = _check_success_criteria(
        test_metrics2, success_criteria2, "binary_classification"
    )
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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )
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
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )
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
