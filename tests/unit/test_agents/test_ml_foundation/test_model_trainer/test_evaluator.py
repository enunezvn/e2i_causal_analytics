"""Tests for model evaluator node.

Tests the evaluate_model function with various problem types and edge cases.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _check_success_criteria,
    _compute_business_utility,
    _compute_classification_metrics,
    _compute_optimal_threshold,
    _compute_precision_at_k,
    _positive_class_proba,
    _select_threshold,
    evaluate_model,
)

# ============================================================================
# Shared fixture sizing — keep splits tiny so every test fits in one pytest
# tick. Lifted from inline numerals so a single edit propagates everywhere.
# ============================================================================
N_TRAIN_SAMPLES = 100
N_VAL_SAMPLES = 30
N_TEST_SAMPLES = 20
N_FEATURES = 5
RANDOM_STATE = 42
RF_N_ESTIMATORS = 10

# ============================================================================
# Test fixtures
# ============================================================================


class MockBinaryClassifier:
    """Mock trained binary classifier."""

    def predict(self, X):
        return np.random.randint(0, 2, len(X))

    def predict_proba(self, X):
        proba = np.random.rand(len(X))
        return np.column_stack([1 - proba, proba])

    def get_params(self, deep: bool = True) -> dict:
        return {}

    def fit(self, X, y):
        return self


class MockRegressor:
    """Mock trained regressor."""

    def predict(self, X):
        return np.random.rand(len(X))


class MockClassifierNoProba:
    """Mock classifier without predict_proba."""

    def predict(self, X):
        return np.random.randint(0, 2, len(X))


@pytest.fixture
def binary_classification_state():
    """Create state for binary classification evaluation."""
    np.random.seed(RANDOM_STATE)
    model = MockBinaryClassifier()
    return {
        "trained_model": model,
        "problem_type": "binary_classification",
        "X_train_preprocessed": np.random.rand(N_TRAIN_SAMPLES, N_FEATURES),
        "X_validation_preprocessed": np.random.rand(N_VAL_SAMPLES, N_FEATURES),
        "X_test_preprocessed": np.random.rand(N_TEST_SAMPLES, N_FEATURES),
        "train_data": {"y": np.random.randint(0, 2, N_TRAIN_SAMPLES)},
        "validation_data": {"y": np.random.randint(0, 2, N_VAL_SAMPLES)},
        "test_data": {"y": np.random.randint(0, 2, N_TEST_SAMPLES)},
        "success_criteria": {},
    }


@pytest.fixture
def regression_state():
    """Create state for regression evaluation."""
    np.random.seed(RANDOM_STATE)
    model = MockRegressor()
    return {
        "trained_model": model,
        "problem_type": "regression",
        "X_train_preprocessed": np.random.rand(N_TRAIN_SAMPLES, N_FEATURES),
        "X_validation_preprocessed": np.random.rand(N_VAL_SAMPLES, N_FEATURES),
        "X_test_preprocessed": np.random.rand(N_TEST_SAMPLES, N_FEATURES),
        "train_data": {"y": np.random.rand(N_TRAIN_SAMPLES)},
        "validation_data": {"y": np.random.rand(N_VAL_SAMPLES)},
        "test_data": {"y": np.random.rand(N_TEST_SAMPLES)},
        "success_criteria": {},
    }


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
        """Should check if model meets success criteria."""
        binary_classification_state["success_criteria"] = {"accuracy": 0.90}

        result = await evaluate_model(binary_classification_state)

        assert "success_criteria_met" in result
        assert "success_criteria_results" in result
        assert "accuracy" in result["success_criteria_results"]

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


# ============================================================================
# Test helper functions
# ============================================================================


class TestComputeOptimalThreshold:
    """Test optimal threshold computation."""

    def test_returns_threshold_with_proba(self):
        """Should compute optimal threshold with probabilities."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.column_stack(
            [
                1 - np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
                np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
            ]
        )

        threshold = _compute_optimal_threshold(y_true, y_proba)

        assert 0.0 <= threshold <= 1.0

    def test_returns_default_without_proba(self):
        """Should return 0.5 when no probabilities provided."""
        y_true = np.array([0, 0, 1, 1])

        threshold = _compute_optimal_threshold(y_true, None)

        assert threshold == 0.5


class TestComputePrecisionAtK:
    """Test precision@k computation."""

    def test_computes_precision_at_k(self):
        """Should compute precision at various k values."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.column_stack(
            [
                1 - np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
                np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
            ]
        )

        result = _compute_precision_at_k(y_true, y_proba, k_values=[2, 4])

        assert 2 in result
        assert 4 in result
        assert 0.0 <= result[2] <= 1.0
        assert 0.0 <= result[4] <= 1.0

    def test_returns_empty_without_proba(self):
        """Should return empty dict without probabilities."""
        y_true = np.array([0, 0, 1, 1])

        result = _compute_precision_at_k(y_true, None, k_values=[2])

        assert result == {}

    def test_skips_k_larger_than_samples(self):
        """Should skip k values larger than sample size."""
        y_true = np.array([0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])

        result = _compute_precision_at_k(y_true, y_proba, k_values=[2, 100])

        assert 2 in result
        assert 100 not in result


class TestCheckSuccessCriteria:
    """Test success criteria checking."""

    def test_all_criteria_met(self):
        """Should return True when all criteria met."""
        test_metrics = {"accuracy": 0.85, "roc_auc": 0.90}
        success_criteria = {"accuracy": 0.80, "auc": 0.85}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is True

    def test_criteria_not_met(self):
        """Should return False when criteria not met."""
        test_metrics = {"accuracy": 0.75, "roc_auc": 0.80}
        success_criteria = {"accuracy": 0.90}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is False
        assert result["success_criteria_results"]["accuracy"] is False

    def test_lower_is_better_metrics(self):
        """Should correctly handle metrics where lower is better."""
        test_metrics = {"rmse": 0.1, "mae": 0.05}
        success_criteria = {"rmse": 0.2, "mae": 0.1}

        result = _check_success_criteria(test_metrics, success_criteria, "regression")

        assert result["success_criteria_met"] is True

    def test_empty_criteria_returns_true(self):
        """Should return True when no criteria specified."""
        result = _check_success_criteria({}, {}, "binary_classification")

        assert result["success_criteria_met"] is True

    def test_handles_missing_metrics(self):
        """Should handle missing metrics gracefully."""
        test_metrics = {"accuracy": 0.85}
        success_criteria = {"accuracy": 0.80, "nonexistent_metric": 0.5}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is False
        assert result["success_criteria_results"]["nonexistent_metric"] is False


# ============================================================================
# Block 1A — threshold must be tuned on validation, not test (finding #6)
# ============================================================================


class TestThresholdTunedOnValidationOnly:
    """Verify threshold tuning is performed on validation, never on test.

    Block 1A of the Tier-0 remediation plan: the chosen classification
    threshold must be selected on the validation set, frozen, and then
    applied to the test set. A regression to test-tuning would inflate
    apparent test performance.

    Test design: synthesize 200 rows total (100 validation, 100 test) with
    deliberately divergent score distributions so the validation-derived
    threshold and test-derived threshold are far apart (>= 0.25 gap). If
    the implementation regresses to tuning on test, the assertions on
    `chosen_threshold` would fail.
    """

    @staticmethod
    def _make_split(
        rng: np.random.Generator,
        n: int,
        positive_score_mean: float,
        negative_score_mean: float,
        spread: float = 0.05,
        positive_rate: float = 0.5,
    ):
        """Generate (y_true, y_pred, y_proba) for a single split.

        Parameters
        ----------
        rng : np.random.Generator
            Source of pseudo-randomness for reproducibility.
        n : int
            Number of rows.
        positive_score_mean : float
            Mean predicted-probability for positive samples.
        negative_score_mean : float
            Mean predicted-probability for negative samples.
        spread : float
            Standard deviation of the per-class score noise. Small
            values make the distributions tight so Youden's J optimum
            sits cleanly between the two class means.
        positive_rate : float
            Fraction of rows that are class 1.
        """
        n_pos = int(round(n * positive_rate))
        n_neg = n - n_pos

        pos_scores = np.clip(
            rng.normal(loc=positive_score_mean, scale=spread, size=n_pos),
            0.001,
            0.999,
        )
        neg_scores = np.clip(
            rng.normal(loc=negative_score_mean, scale=spread, size=n_neg),
            0.001,
            0.999,
        )

        y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
        y_proba_pos = np.concatenate([pos_scores, neg_scores])
        # Shuffle so order doesn't bias anything downstream
        order = rng.permutation(len(y_true))
        y_true = y_true[order]
        y_proba_pos = y_proba_pos[order]

        # Two-column proba matrix (column 1 = positive class)
        y_proba = np.column_stack([1.0 - y_proba_pos, y_proba_pos])
        # Default-threshold predictions (0.5)
        y_pred = (y_proba_pos >= 0.5).astype(int)
        return y_true, y_pred, y_proba

    def test_threshold_tuned_on_validation_only(self):
        """Chosen threshold must come from validation, not test.

        Construct a 200-row synthetic dataset (100 validation + 100 test)
        whose score distributions force the validation-optimal threshold
        and test-optimal threshold to live in well-separated bands.
        Verify the chosen threshold matches the validation-derived value
        and is incompatible with the test-derived value.
        """
        rng = np.random.default_rng(20260426)

        # Validation split: positives ~0.40, negatives ~0.20 → opt ~0.30
        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=100, positive_score_mean=0.40, negative_score_mean=0.20
        )
        # Test split: positives ~0.80, negatives ~0.60 → opt ~0.70
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.80, negative_score_mean=0.60
        )

        # Sanity: independently confirm the two splits yield
        # well-separated optima before invoking the function under test.
        # If these baseline expectations break the test fixture itself
        # is wrong, not the implementation.
        val_only_threshold = _compute_optimal_threshold(y_val, y_val_proba)
        test_only_threshold = _compute_optimal_threshold(y_test, y_test_proba)
        assert val_only_threshold < 0.50, (
            f"Validation-derived threshold should land in the low band, "
            f"got {val_only_threshold:.4f}"
        )
        assert test_only_threshold > 0.55, (
            f"Test-derived threshold should land in the high band, "
            f"got {test_only_threshold:.4f}"
        )
        gap = test_only_threshold - val_only_threshold
        assert gap >= 0.20, (
            f"Test fixture must produce a >= 0.20 gap between val and test "
            f"thresholds; got {gap:.4f}. A smaller gap would not catch a "
            f"regression to test-tuning."
        )

        # Real call into the function under test — no mocks of
        # _compute_classification_metrics or _compute_optimal_threshold.
        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=y_val,
            y_validation_pred=y_val_pred,
            y_validation_proba=y_val_proba,
            y_test=y_test,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
        )

        # 1) Chosen threshold matches the validation-derived value exactly.
        assert "validation_metrics" in result
        validation_metrics = result["validation_metrics"]
        assert "chosen_threshold" in validation_metrics, (
            "validation_metrics must expose `chosen_threshold` so downstream "
            "consumers (model registry, monitoring) can audit operating point."
        )
        chosen = float(validation_metrics["chosen_threshold"])
        assert chosen == pytest.approx(val_only_threshold), (
            f"chosen_threshold must equal validation-derived value "
            f"{val_only_threshold:.4f}, got {chosen:.4f}"
        )

        # 2) chosen_threshold_source flags validation provenance.
        assert validation_metrics.get("chosen_threshold_source") == "validation"

        # 3) Top-level optimal_threshold (the canonical key consumed
        # cross-codebase) mirrors the validation-tuned value, and the
        # top-level provenance flag also reports validation.
        assert result["optimal_threshold"] == pytest.approx(val_only_threshold)
        assert result["chosen_threshold_source"] == "validation"

        # 4) Negative assertion — the chosen threshold MUST NOT match
        # the test-derived value. A regression to the old behaviour
        # (`_compute_optimal_threshold(y_test, y_test_proba)`) would
        # trip this assertion since the gap is >= 0.20.
        assert chosen != pytest.approx(test_only_threshold, abs=0.05), (
            f"chosen_threshold appears tuned on test ({test_only_threshold:.4f}); "
            f"this is the leakage bug Block 1A removes."
        )

    def test_chosen_threshold_frozen_for_test_evaluation(self):
        """Test-set predictions must use the validation-tuned threshold.

        Verify that test_metrics_at_optimal is computed by applying the
        validation-tuned threshold to test probabilities (not by re-tuning
        on test). We verify this indirectly: the predicted-positive count
        on test must equal the count we get by applying the
        validation-tuned threshold to test_proba — NOT the count from
        applying a test-tuned threshold to test_proba.
        """
        rng = np.random.default_rng(20260426)

        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=100, positive_score_mean=0.40, negative_score_mean=0.20
        )
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.80, negative_score_mean=0.60
        )

        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=y_val,
            y_validation_pred=y_val_pred,
            y_validation_proba=y_val_proba,
            y_test=y_test,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            imbalance_detected=True,  # forces test_metrics = test_metrics_at_optimal
            minority_ratio=0.5,
        )

        chosen = float(result["validation_metrics"]["chosen_threshold"])
        # Independently compute predicted positives at the validation-tuned
        # threshold applied to the test set.
        val_tuned_test_predictions = (y_test_proba[:, 1] >= chosen).astype(int)
        n_pos_at_val_threshold = int(val_tuned_test_predictions.sum())

        cm = result["confusion_matrix"]
        n_pos_in_result = int(cm["TP"]) + int(cm["FP"])
        assert n_pos_in_result == n_pos_at_val_threshold, (
            f"Test confusion matrix used a threshold inconsistent with "
            f"chosen_threshold={chosen:.4f}: result says {n_pos_in_result} "
            f"positives, but applying chosen_threshold to test_proba gives "
            f"{n_pos_at_val_threshold}."
        )

        # Cross-check: a test-tuned threshold would predict a different
        # number of positives. If they happen to coincide here, the gap
        # construction above is too small.
        test_tuned = _compute_optimal_threshold(y_test, y_test_proba)
        n_pos_at_test_threshold = int((y_test_proba[:, 1] >= test_tuned).astype(int).sum())
        assert n_pos_in_result != n_pos_at_test_threshold, (
            "Test predicted-positive count matches a test-tuned threshold; "
            "either the synthetic gap is too narrow or the implementation "
            "regressed to test-tuning."
        )

    def test_falls_back_to_default_when_validation_missing(self):
        """When validation arrays are unavailable, fall back to 0.5 (not test).

        We never tune on test even when validation is absent. Falling back
        to the default 0.5 threshold trades calibration for test-set
        integrity — the test set must remain untouched for thresholding.
        """
        rng = np.random.default_rng(20260426)
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.80, negative_score_mean=0.60
        )

        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=None,
            y_validation_pred=None,
            y_validation_proba=None,
            y_test=y_test,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
        )

        # In the fallback path validation_metrics is empty (no validation
        # arrays), so the operating point lives only at the top level.
        assert result["optimal_threshold"] == 0.5
        assert result["chosen_threshold_source"] == "default"
        assert result["validation_metrics"] == {}
        # And it must NOT match the test-derived value
        test_only = _compute_optimal_threshold(y_test, y_test_proba)
        assert result["optimal_threshold"] != pytest.approx(test_only, abs=0.05)


# ============================================================================
# 1A-I-3: _select_threshold extraction — direct unit tests on the helper
# ============================================================================


class TestSelectThreshold:
    """Unit tests for the extracted ``_select_threshold`` helper.

    These tests target the helper directly (not via
    ``_compute_classification_metrics``) so they pin the contract the
    rest of the evaluator and downstream consumers (mlflow_logger,
    audit code) rely on.

    1A-M-6 will move these into ``test_threshold_selection.py``.
    """

    def test_select_threshold_clamps_inf_sentinel_within_validation_branch(self):
        """When sklearn's roc_curve returns the inf sentinel, the helper
        must surface 0.5 (not inf, NaN, or out-of-range values).

        sklearn's ``roc_curve`` prepends a sentinel threshold of ``np.inf``
        for the trivial (FPR=0, TPR=0) point. On degenerate inputs (e.g.,
        constant probabilities where every threshold is equivalent),
        Youden's J argmax lands on that sentinel.
        ``_compute_optimal_threshold`` clamps the non-finite/out-of-range
        result back to 0.5 — this test verifies the clamp survives the
        round-trip through ``_select_threshold``.

        Source string remains ``"validation"`` because validation arrays
        WERE provided; only the numeric value falls back.
        """
        # Constant 0.5 probabilities → degenerate ROC curve → argmax
        # lands on sklearn's inf sentinel → clamp triggers.
        n = 60
        np.random.seed(RANDOM_STATE)
        y_validation = np.random.randint(0, 2, n)
        y_validation_proba = np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        threshold, source = _select_threshold(y_validation, y_validation_proba)

        assert threshold == 0.5, (
            f"Non-finite/out-of-range optimal threshold must clamp to 0.5; got {threshold!r}"
        )
        assert np.isfinite(threshold)
        assert source == "validation", (
            "Source must remain 'validation' when arrays are provided — "
            "only the numeric threshold falls back, not the provenance."
        )

    def test_select_threshold_provenance_string_format(self):
        """Provenance source must be exactly 'validation' or 'default'.

        Downstream consumers (mlflow_logger, audit code, monitoring)
        match on these literal string values. Anything else (e.g.
        "VALIDATION", "val", " validation ") would silently break those
        consumers — this test pins the exact format.
        """
        # 'validation' branch: arrays present.
        rng = np.random.default_rng(20260426)
        n = 40
        y_proba_pos = rng.uniform(0.1, 0.9, n)
        y_validation = (y_proba_pos > 0.5).astype(int)
        y_validation_proba = np.column_stack([1.0 - y_proba_pos, y_proba_pos])
        _, source_validation = _select_threshold(y_validation, y_validation_proba)
        assert source_validation == "validation"
        # Pin the literal type and exact characters — defensive against
        # a regression to bytes / enum / capitalisation drift.
        assert isinstance(source_validation, str)
        assert source_validation == "validation" and len(source_validation) == 10

        # 'default' branch: arrays absent.
        _, source_none = _select_threshold(None, None)
        assert source_none == "default"
        assert isinstance(source_none, str)
        assert source_none == "default" and len(source_none) == 7

        # Mixed-absence variants must also fall back to default — the
        # contract says BOTH arrays must be present for the validation
        # branch.
        _, source_no_proba = _select_threshold(y_validation, None)
        assert source_no_proba == "default"
        _, source_no_labels = _select_threshold(None, y_validation_proba)
        assert source_no_labels == "default"


# ============================================================================
# Helper: _positive_class_proba (1A-M3 dedup)
# ============================================================================


class TestPositiveClassProba:
    """Locks the contract of the shared positive-class extraction helper.

    All callers in evaluator.py route through this helper, so the test
    here is the canonical guarantee that 1D and 2D proba arrays are both
    accepted and that the 2D path returns the column-1 view.
    """

    def test_returns_column_1_for_2d_array(self):
        proba = np.array([[0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])
        result = _positive_class_proba(proba)
        np.testing.assert_array_equal(result, np.array([0.9, 0.3, 0.6]))

    def test_returns_unchanged_for_1d_array(self):
        proba = np.array([0.9, 0.3, 0.6])
        result = _positive_class_proba(proba)
        # Same reference is fine — caller only reads it.
        np.testing.assert_array_equal(result, proba)


# ============================================================================
# 1A-M2: rebinarisation gate uses math.isclose, not raw float ==
# ============================================================================


class TestThresholdRebinarisationGuard:
    """When the validation-tuned threshold lands on (or vanishingly close to)
    0.5, the test-set rebinarisation should be skipped — there's no point
    re-applying ``proba >= 0.5`` if the model already does that. Direct
    ``!= 0.5`` would treat ``0.5 + 1e-15`` as different, triggering a
    no-op rebinarisation path. ``math.isclose`` collapses that gap.
    """

    def test_threshold_within_isclose_tolerance_skips_rebinarisation(self):
        """When the chosen threshold is essentially 0.5, test_metrics_at_05
        and test_metrics_at_optimal must be identical — the optimal-path
        binarisation never runs."""
        np.random.seed(RANDOM_STATE)
        n = 80
        # Validation set whose Youden's J optimum returns sklearn's `inf`
        # sentinel (random labels make every operating point equivalent),
        # which the upstream guard clamps back to 0.5. Test set then sees
        # threshold == 0.5 and must NOT re-binarise.
        y_val = np.random.randint(0, 2, n)
        y_val_proba = np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        y_val_pred = (y_val_proba[:, 1] >= 0.5).astype(int)

        y_test = np.random.randint(0, 2, n)
        y_test_proba = np.column_stack([np.random.rand(n), np.random.rand(n)])
        y_test_proba[:, 0] = 1 - y_test_proba[:, 1]
        y_test_pred = (y_test_proba[:, 1] >= 0.5).astype(int)

        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=y_val,
            y_validation_pred=y_val_pred,
            y_validation_proba=y_val_proba,
            y_test=y_test,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
        )

        # Threshold falls back to 0.5 (the all-0.5 proba degenerates Youden's
        # J), so the rebinarisation gate must skip.
        assert result["optimal_threshold"] == 0.5
        # Same y_pred → identical metrics in both at-0.5 and at-optimal blocks.
        assert result["test_metrics_at_05"] == result["test_metrics_at_optimal"]


# ============================================================================
# Block 5 (#10): business_utility metric driven by cost_matrix
# ============================================================================


class TestBusinessUtilityFromCostMatrix:
    """The evaluator must compute a cost-weighted business_utility metric
    from a caller-supplied ``cost_matrix`` and surface it in
    ``validation_metrics``, ``test_metrics``, and at the top-level result.

    Test design: build deterministic predictions where we KNOW the
    confusion matrix counts at the chosen threshold, supply a cost matrix
    with distinctive per-outcome values, and assert the multiplication
    matches the closed-form expectation.
    """

    @staticmethod
    def _make_split(rng, n, positive_score_mean, negative_score_mean, spread=0.05):
        """Reuse the same well-separated synthetic split builder as the
        threshold-tuning test fixture so test fixtures stay consistent."""
        n_pos = n // 2
        n_neg = n - n_pos
        pos_scores = np.clip(
            rng.normal(loc=positive_score_mean, scale=spread, size=n_pos),
            0.001,
            0.999,
        )
        neg_scores = np.clip(
            rng.normal(loc=negative_score_mean, scale=spread, size=n_neg),
            0.001,
            0.999,
        )
        y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
        y_proba_pos = np.concatenate([pos_scores, neg_scores])
        order = rng.permutation(len(y_true))
        y_true = y_true[order]
        y_proba_pos = y_proba_pos[order]
        y_proba = np.column_stack([1.0 - y_proba_pos, y_proba_pos])
        y_pred = (y_proba_pos >= 0.5).astype(int)
        return y_true, y_pred, y_proba

    def test_compute_business_utility_helper(self):
        """The standalone helper just multiplies counts by per-outcome dollars."""
        cost_matrix = {"tp": 100.0, "fp": -10.0, "fn": -50.0, "tn": 0.0}
        # 5 TP, 2 FP, 3 FN, 10 TN → 5*100 + 2*-10 + 3*-50 + 10*0 = 500 - 20 - 150 = 330
        utility = _compute_business_utility(tp=5, fp=2, fn=3, tn=10, cost_matrix=cost_matrix)
        assert utility == pytest.approx(330.0)

    def test_business_utility_in_validation_and_test_metrics(self):
        """When a cost_matrix is supplied, business_utility lands in BOTH
        validation_metrics and test_metrics, plus the top-level result."""
        rng = np.random.default_rng(20260427)
        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=100, positive_score_mean=0.70, negative_score_mean=0.30
        )
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.70, negative_score_mean=0.30
        )

        cost_matrix = {"tp": 100.0, "fp": -10.0, "fn": -50.0, "tn": 0.0}

        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=y_val,
            y_validation_pred=y_val_pred,
            y_validation_proba=y_val_proba,
            y_test=y_test,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
            cost_matrix=cost_matrix,
        )

        # Validation_metrics carries it.
        assert "business_utility" in result["validation_metrics"], (
            "business_utility must appear in validation_metrics so deployment "
            "tooling can rank candidates by business value at the chosen "
            "operating point."
        )
        # Test_metrics carries it (at the same chosen threshold).
        assert "business_utility" in result["test_metrics"]
        # Top-level mirror exists.
        assert "business_utility" in result

        # Sanity: dollars must be a float, not None.
        assert isinstance(result["test_metrics"]["business_utility"], float)
        assert isinstance(result["validation_metrics"]["business_utility"], float)
        assert isinstance(result["business_utility"], float)
        # Top-level mirrors the test_metrics value.
        assert result["business_utility"] == result["test_metrics"]["business_utility"]

    def test_business_utility_absent_when_no_cost_matrix(self):
        """When cost_matrix is not provided, the metric is NOT silently
        defaulted — it stays absent from validation_metrics/test_metrics
        and the top-level mirror is None. Callers can then distinguish
        'not configured' from 'configured but zero'."""
        rng = np.random.default_rng(20260427)
        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=80, positive_score_mean=0.70, negative_score_mean=0.30
        )
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=80, positive_score_mean=0.70, negative_score_mean=0.30
        )

        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=y_val,
            y_validation_pred=y_val_pred,
            y_validation_proba=y_val_proba,
            y_test=y_test,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
            cost_matrix=None,
        )

        # Top-level mirror is explicitly None when no cost_matrix.
        assert result["business_utility"] is None
        # Sub-metrics dicts must NOT have a stray placeholder.
        assert "business_utility" not in result["validation_metrics"]
        assert "business_utility" not in result["test_metrics"]

    def test_business_utility_uses_chosen_threshold_not_raw_predictions(self):
        """business_utility must be computed at the validation-tuned chosen
        threshold, NOT at the model's default 0.5. We verify by
        constructing a case where shifting the threshold flips the FN/TN
        counts and changes the utility number."""
        rng = np.random.default_rng(20260428)
        # Validation: positives at 0.40, negatives at 0.20 → opt around 0.30
        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=100, positive_score_mean=0.40, negative_score_mean=0.20
        )
        # Test: positives at 0.55, negatives at 0.45 → at 0.30 chosen
        # threshold many rows are predicted positive; at 0.5 default,
        # the prediction split is very different.
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.55, negative_score_mean=0.45
        )
        # Cost matrix that strongly differentiates predicted-positive from
        # predicted-negative outcomes.
        cost_matrix = {"tp": 1000.0, "fp": -1.0, "fn": -1000.0, "tn": 1.0}

        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=y_val,
            y_validation_pred=y_val_pred,
            y_validation_proba=y_val_proba,
            y_test=y_test,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
            cost_matrix=cost_matrix,
        )

        chosen = result["optimal_threshold"]
        # Recompute utility at the chosen threshold by reapplying it
        # ourselves and confirm it matches the evaluator's number.
        from sklearn.metrics import confusion_matrix as _cm

        y_test_pred_at_chosen = (y_test_proba[:, 1] >= chosen).astype(int)
        cm = _cm(y_test, y_test_pred_at_chosen)
        tn, fp, fn, tp = cm.ravel()
        expected = (
            tp * cost_matrix["tp"]
            + fp * cost_matrix["fp"]
            + fn * cost_matrix["fn"]
            + tn * cost_matrix["tn"]
        )
        assert result["test_metrics"]["business_utility"] == pytest.approx(expected)

        # Negative assertion: the utility computed at default 0.5 must
        # differ when chosen != 0.5 — proves the metric tracks the chosen
        # threshold, not the raw y_test_pred.
        if not np.isclose(chosen, 0.5):
            y_test_pred_at_05 = (y_test_proba[:, 1] >= 0.5).astype(int)
            cm_05 = _cm(y_test, y_test_pred_at_05)
            tn5, fp5, fn5, tp5 = cm_05.ravel()
            utility_at_05 = (
                tp5 * cost_matrix["tp"]
                + fp5 * cost_matrix["fp"]
                + fn5 * cost_matrix["fn"]
                + tn5 * cost_matrix["tn"]
            )
            assert utility_at_05 != pytest.approx(expected), (
                "If business_utility regresses to using y_test_pred (model's "
                "default 0.5), this assertion will trip — by construction the "
                "0.5-utility differs from the chosen-threshold utility."
            )
