"""Regression tests for apply_post_hoc_calibration.

These tests lock the 2026-05-04 fix for the FrozenEstimator regression
(`tier0_frozenestimator_regression.md`). The bug:

1. `LogisticRegression_Conformal` registry entry has
   `skip_post_hoc_calibration=True` but the flag was dropped when
   `run_tier0_test.py` rebuilt the candidate dict for Step 5b alt models.
2. The post-hoc calibration path then ran on a non-classifier wrapper
   (MapieConformalBinaryClassifier). `CalibratedClassifierCV.fit` does not
   validate the underlying estimator — sklearn only catches the mismatch
   in `predict_proba`, after the caller stored `calibration_applied=True`.
3. Downstream `evaluator.py:364` calls `calibrated_model.predict_proba(...)`
   under the `cal_info["calibration_applied"]` guard and crashed with:
   `ValueError: FrozenEstimator should either be a classifier ... Got a
   regressor with response_method=['decision_function', 'predict_proba']`.

The defense-in-depth fix adds an `is_classifier(model)` guard so that
non-classifier wrappers exit cleanly with `calibration_applied=False` and
a `skip_reason` field, preserving downstream contracts even when the
registry flag is missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression

from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
    apply_post_hoc_calibration,
)


@pytest.fixture
def binary_dataset():
    rng = np.random.default_rng(seed=42)
    n = 120
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_calibration_applied_for_sklearn_classifier(binary_dataset):
    """Happy path: a real classifier gets calibrated; cal_info reports True."""
    X, y = binary_dataset
    model = LogisticRegression(max_iter=200).fit(X, y)

    calibrated, cal_info = apply_post_hoc_calibration(model, X, y, method="isotonic")

    assert cal_info["calibration_applied"] is True
    assert cal_info["calibration_method"] == "isotonic"
    assert cal_info["calibration_fit_samples"] == len(y)
    proba = calibrated.predict_proba(X)
    assert proba.shape == (len(y), 2)


class _NotAClassifier(BaseEstimator, RegressorMixin):
    """Stand-in for the real failure mode (MapieConformalBinaryClassifier).

    Exposes ``predict_proba`` like the conformal wrapper does, but
    `is_classifier()` returns False because the class is RegressorMixin-tagged.
    sklearn's CalibratedClassifierCV.fit() accepts it silently and the crash
    only surfaces in predict_proba.
    """

    def fit(self, X: Any, y: Any) -> "_NotAClassifier":
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])


def test_skip_when_base_model_is_not_a_classifier(binary_dataset):
    """Defense-in-depth: non-classifier wrappers must exit cleanly.

    Without the guard, the (model, {calibration_applied: True}) pair would
    return successfully and downstream evaluator.py would crash on
    predict_proba inside sklearn._get_response_values. With the guard, the
    caller sees calibration_applied=False and a structured skip_reason.
    """
    X, y = binary_dataset
    not_a_classifier = _NotAClassifier().fit(X, y)

    returned, cal_info = apply_post_hoc_calibration(not_a_classifier, X, y, method="isotonic")

    assert returned is not_a_classifier, "must return the original model when skipping"
    assert cal_info["calibration_applied"] is False
    assert cal_info["skip_reason"] == "base_model_not_a_classifier"
    assert cal_info["calibration_method"] == "isotonic"
    assert "calibration_error" not in cal_info, (
        "skip path must not surface an error string — it's a clean skip"
    )


def test_evaluator_guard_compatible_with_skip_path(binary_dataset):
    """Evaluator at evaluator.py:363 reads cal_info.get("calibration_applied").

    Confirm both happy and skip paths produce the boolean shape the guard
    expects (not None, not missing key).
    """
    X, y = binary_dataset

    # Happy path
    classifier = LogisticRegression(max_iter=200).fit(X, y)
    _, happy_info = apply_post_hoc_calibration(classifier, X, y)
    assert happy_info.get("calibration_applied") is True

    # Skip path
    not_a_classifier = _NotAClassifier().fit(X, y)
    _, skip_info = apply_post_hoc_calibration(not_a_classifier, X, y)
    assert skip_info.get("calibration_applied") is False
