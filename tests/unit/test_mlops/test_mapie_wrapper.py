"""Unit tests for MapieConformalBinaryClassifier wrapper (Phase 1 W2 day-3).

Reference: shard 19 §B.2 acceptance asserts.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

SEED = 42
N_SAMPLES = 200
N_FEATURES = 5


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


@pytest.fixture(scope="module")
def fitted_wrapper():
    """Fit one wrapper for tests that share the trained model."""
    from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

    X, y = _make_logistic_dgp()
    base = LogisticRegression(max_iter=1000, random_state=SEED)
    wrapper = MapieConformalBinaryClassifier(
        base_estimator=base,
        method="lac",
        cv=3,
        alpha=0.10,
        random_state=SEED,
    )
    wrapper.fit(X, y)
    return wrapper, X, y


class TestMapieWrapperShapes:
    """Acceptance asserts from shard 19 §B.2."""

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

    def test_classes_attribute_set_after_fit(self, fitted_wrapper):
        wrapper, _, _ = fitted_wrapper
        assert wrapper.classes_ is not None
        assert set(wrapper.classes_.tolist()).issubset({0, 1})


class TestMapieWrapperUsageContract:
    """Pre-fit guard rails."""

    def test_predict_proba_before_fit_raises(self):
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        wrapper = MapieConformalBinaryClassifier(base_estimator=LogisticRegression())
        X = np.zeros((3, N_FEATURES))
        with pytest.raises(RuntimeError):
            wrapper.predict_proba(X)

    def test_predict_sets_before_fit_raises(self):
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        wrapper = MapieConformalBinaryClassifier(base_estimator=LogisticRegression())
        X = np.zeros((3, N_FEATURES))
        with pytest.raises(RuntimeError):
            wrapper.predict_sets(X)


class TestMapieWrapperPredictSets:
    """Phase 1: prediction sets are logged for future-work only."""

    def test_predict_sets_returns_array_after_fit(self, fitted_wrapper):
        wrapper, X, _ = fitted_wrapper
        sets = wrapper.predict_sets(X[:5])
        # MAPIE returns prediction sets shape (n_samples, n_classes, n_alphas).
        # We check it's a numpy-like object with at least 2 dims.
        arr = np.asarray(sets)
        assert arr.ndim >= 2
