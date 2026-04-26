"""Tests for model evaluator node.

Tests the evaluate_model function with various problem types and edge cases.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _check_success_criteria,
    _compute_classification_metrics,
    _compute_optimal_threshold,
    _compute_precision_at_k,
    evaluate_model,
)

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
    np.random.seed(42)
    model = MockBinaryClassifier()
    return {
        "trained_model": model,
        "problem_type": "binary_classification",
        "X_train_preprocessed": np.random.rand(100, 5),
        "X_validation_preprocessed": np.random.rand(30, 5),
        "X_test_preprocessed": np.random.rand(20, 5),
        "train_data": {"y": np.random.randint(0, 2, 100)},
        "validation_data": {"y": np.random.randint(0, 2, 30)},
        "test_data": {"y": np.random.randint(0, 2, 20)},
        "success_criteria": {},
    }


@pytest.fixture
def regression_state():
    """Create state for regression evaluation."""
    np.random.seed(42)
    model = MockRegressor()
    return {
        "trained_model": model,
        "problem_type": "regression",
        "X_train_preprocessed": np.random.rand(100, 5),
        "X_validation_preprocessed": np.random.rand(30, 5),
        "X_test_preprocessed": np.random.rand(20, 5),
        "train_data": {"y": np.random.rand(100)},
        "validation_data": {"y": np.random.rand(30)},
        "test_data": {"y": np.random.rand(20)},
        "success_criteria": {},
    }


@pytest.fixture
def real_classifier_state():
    """Create state with real trained classifier for accurate testing."""
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.rand(30, 5)
    y_val = np.random.randint(0, 2, 30)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
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
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_val = np.random.rand(30, 5)
    y_val = np.random.rand(30)
    X_test = np.random.rand(20, 5)
    y_test = np.random.rand(20)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
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
            "X_test_preprocessed": np.random.rand(20, 5),
            "test_data": {"y": np.random.randint(0, 2, 20)},
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
        np.random.seed(42)
        state = {
            "trained_model": MockClassifierNoProba(),
            "problem_type": "binary_classification",
            "X_test_preprocessed": np.random.rand(20, 5),
            "test_data": {"y": np.random.randint(0, 2, 20)},
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
