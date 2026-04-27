"""Shared fixtures for model_trainer evaluator tests.

Hosts the mock model classes and minimal-state fixtures that
``test_evaluator.py`` and its sibling test files
(``test_threshold_selection.py``, ``test_metrics_computation.py``,
``test_provenance.py``) all depend on. The split was performed in
1A-M-6; consolidating the fixtures here keeps every sibling file
importing the same shapes and keeps test names CI-stable.
"""

import numpy as np
import pytest

# ============================================================================
# Shared fixture sizing - keep splits tiny so every test fits in one pytest
# tick. Lifted from inline numerals so a single edit propagates everywhere.
# ============================================================================
N_TRAIN_SAMPLES = 100
N_VAL_SAMPLES = 30
N_TEST_SAMPLES = 20
N_FEATURES = 5
RANDOM_STATE = 42
RF_N_ESTIMATORS = 10


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
