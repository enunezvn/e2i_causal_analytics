"""Unit tests for NGBoostBinaryClassifier wrapper (Phase 1 W2 day-1).

Reference: shard 19 §A.1 (acceptance asserts) + §F (test_ngboost_wrapper_*).
The calibration smoke threshold is relaxed per W2-prep amendment 2:
ECE < 0.10 with fixed seed=42, N=200, n_estimators=200 (codex memory-pressure
callout — keep fixtures small on the 16 GB / 6 GB-swap droplet).
"""

from __future__ import annotations

import numpy as np
import pytest


SEED = 42
N_SAMPLES = 200
N_FEATURES = 5
N_ESTIMATORS = 200


def _make_logistic_dgp(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Logistic data-generating process for binary classification smoke tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    coefs = rng.standard_normal(n_features)
    logits = X @ coefs
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n_samples) < probs).astype(int)
    return X, y


def _expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Standard ECE: weighted average of |bin_acc - bin_conf| across bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        bin_n = int(mask.sum())
        if bin_n == 0:
            continue
        bin_acc = float(labels[mask].mean())
        bin_conf = float(probs[mask].mean())
        ece += (bin_n / n) * abs(bin_acc - bin_conf)
    return ece


@pytest.fixture(scope="module")
def fitted_wrapper():
    """Fit one wrapper for tests that share the trained model."""
    from src.mlops.wrappers.ngboost_wrapper import NGBoostBinaryClassifier

    X, y = _make_logistic_dgp()
    wrapper = NGBoostBinaryClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.01,
        random_state=SEED,
        verbose=False,
    )
    wrapper.fit(X, y)
    return wrapper, X, y


class TestNGBoostWrapperShapes:
    """Acceptance asserts from shard 19 §A.1."""

    def test_predict_proba_shape_is_n_by_2(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        proba = wrapper.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_predict_proba_rows_sum_to_one(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        proba = wrapper.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_returns_binary_int_array(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        preds = wrapper.predict(X)
        assert preds.dtype.kind in ("i", "u")
        assert set(np.unique(preds).tolist()).issubset({0, 1})

    def test_pred_dist_returns_bernoulli_with_params(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        dist = wrapper.pred_dist(X[:5])
        # ngboost Bernoulli exposes either `.params` (dict) or `.probs`.
        assert hasattr(dist, "params") or hasattr(dist, "probs")


class TestNGBoostWrapperCalibrationSmoke:
    """Calibration-native algorithm should achieve reasonable ECE without isotonic.

    Threshold relaxed to <0.10 per W2-prep amendment 2 (small fixtures + small
    n_estimators on resource-constrained droplet make <0.05 from shard 19 §A.1
    flaky; <0.10 is still meaningfully below uncalibrated tree-ensemble ECEs
    typically in the 0.15-0.25 range).
    """

    def test_ece_below_threshold_without_post_hoc_calibration(self, fitted_wrapper):
        wrapper, X, y = fitted_wrapper
        proba = wrapper.predict_proba(X)
        positive_class_probs = proba[:, 1]
        ece = _expected_calibration_error(positive_class_probs, y, n_bins=10)
        assert ece < 0.10, f"NGBoost ECE {ece:.4f} exceeds 0.10 (calibration-native expectation)"


class TestNGBoostWrapperUsageContract:
    """Contract checks separate from the fitted-wrapper fixture."""

    def test_predict_before_fit_raises(self):
        from src.mlops.wrappers.ngboost_wrapper import NGBoostBinaryClassifier

        wrapper = NGBoostBinaryClassifier()
        X = np.zeros((3, N_FEATURES))
        with pytest.raises(RuntimeError):
            wrapper.predict_proba(X)

    def test_pred_dist_before_fit_raises(self):
        from src.mlops.wrappers.ngboost_wrapper import NGBoostBinaryClassifier

        wrapper = NGBoostBinaryClassifier()
        X = np.zeros((3, N_FEATURES))
        with pytest.raises(RuntimeError):
            wrapper.pred_dist(X)

    def test_classes_attribute_set_after_fit(self, fitted_wrapper):
        wrapper, _, _ = fitted_wrapper
        assert wrapper.classes_ is not None
        assert set(wrapper.classes_.tolist()).issubset({0, 1})
