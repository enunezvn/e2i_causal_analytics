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


# --------------------------------------------------------------------------- #
# Plan v3 §4 T2.1 — honest split / cross-conformal mode selection             #
# --------------------------------------------------------------------------- #


def _make_imbalanced_logistic_dgp(
    n_samples: int,
    n_features: int = N_FEATURES,
    minority_count: int = 60,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Logistic DGP with a controlled minority count for mode-selection tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    coefs = rng.standard_normal(n_features)
    logits = X @ coefs
    # Sort by logit, take top minority_count as positives, rest as negatives.
    order = np.argsort(logits)
    y = np.zeros(n_samples, dtype=int)
    y[order[-minority_count:]] = 1
    return X, y


class TestMapieWrapperConformalModeSelection:
    """Plan v3 §4 T2.1 mode-selection contract.

    `fit()` chooses split-conformal vs cross-conformal based on the
    minority-class count of `y` and the `min_honest_split_n` threshold
    (default 50). The legacy training-set conformal mode is opt-in via
    `conformal_mode="prefit_legacy"`.
    """

    def test_min_honest_split_n_default_is_50(self):
        """Plan v3 §4 T2.1 / Vovk 2005: 50-positive floor for honest split."""
        from src.mlops.wrappers.mapie_wrapper import MIN_HONEST_SPLIT_N

        assert MIN_HONEST_SPLIT_N == 50

    def test_split_mode_selected_when_n_minority_above_threshold(self):
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        X, y = _make_imbalanced_logistic_dgp(n_samples=400, minority_count=60)
        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            random_state=SEED,
        )
        wrapper.fit(X, y)
        assert wrapper.fitted_conformal_mode_ == "split"
        assert wrapper.n_minority_ == 60
        # n_calib = round(0.20 * 400) = 80 (stratified split).
        assert wrapper.n_calib_ == 80

    def test_cross_mode_selected_when_n_minority_below_threshold(self):
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        X, y = _make_imbalanced_logistic_dgp(n_samples=300, minority_count=20)
        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            random_state=SEED,
        )
        wrapper.fit(X, y)
        assert wrapper.fitted_conformal_mode_ == "cross"
        assert wrapper.n_minority_ == 20
        # cross-mode calibrates over full data (MAPIE handles K-fold internally).
        assert wrapper.n_calib_ == 300

    def test_split_mode_selected_at_exact_threshold_boundary(self):
        """n_minority == MIN_HONEST_SPLIT_N → split (≥, not >)."""
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        X, y = _make_imbalanced_logistic_dgp(n_samples=200, minority_count=50)
        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            random_state=SEED,
        )
        wrapper.fit(X, y)
        assert wrapper.fitted_conformal_mode_ == "split"
        assert wrapper.n_minority_ == 50

    def test_cross_mode_selected_just_below_threshold_boundary(self):
        """n_minority == MIN_HONEST_SPLIT_N - 1 → cross."""
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        X, y = _make_imbalanced_logistic_dgp(n_samples=200, minority_count=49)
        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            random_state=SEED,
        )
        wrapper.fit(X, y)
        assert wrapper.fitted_conformal_mode_ == "cross"
        assert wrapper.n_minority_ == 49

    def test_explicit_prefit_legacy_mode_opt_in(self):
        """Callers can opt back into legacy training-set conformal."""
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        X, y = _make_imbalanced_logistic_dgp(n_samples=400, minority_count=60)
        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            conformal_mode="prefit_legacy",
            random_state=SEED,
        )
        wrapper.fit(X, y)
        assert wrapper.fitted_conformal_mode_ == "prefit_legacy"

    def test_explicit_split_mode_overrides_threshold(self):
        """conformal_mode='split' forces split even when n_minority is small.
        Caller responsibility — wrapper does NOT second-guess."""
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        X, y = _make_imbalanced_logistic_dgp(n_samples=200, minority_count=20)
        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            conformal_mode="split",
            random_state=SEED,
        )
        wrapper.fit(X, y)
        assert wrapper.fitted_conformal_mode_ == "split"

    def test_invalid_conformal_mode_raises_value_error(self):
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        with pytest.raises(ValueError, match="conformal_mode"):
            MapieConformalBinaryClassifier(
                base_estimator=LogisticRegression(),
                conformal_mode="aggressive",  # type: ignore[arg-type]
            )

    def test_invalid_calib_fraction_raises_value_error(self):
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        with pytest.raises(ValueError, match="calib_fraction"):
            MapieConformalBinaryClassifier(
                base_estimator=LogisticRegression(),
                calib_fraction=0.0,
            )
        with pytest.raises(ValueError, match="calib_fraction"):
            MapieConformalBinaryClassifier(
                base_estimator=LogisticRegression(),
                calib_fraction=1.0,
            )

    def test_invalid_min_honest_split_n_raises_value_error(self):
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        with pytest.raises(ValueError, match="min_honest_split_n"):
            MapieConformalBinaryClassifier(
                base_estimator=LogisticRegression(),
                min_honest_split_n=0,
            )


class TestMapieWrapperHonestCoverage:
    """Plan v3 §6 T2.1 acceptance: split-conformal coverage ∈ [0.85, 0.95]
    for α=0.10 on n_minority ≥ 50.

    Coverage is the fraction of held-out test instances whose true label
    is contained in the prediction set at the configured alpha. With
    α=0.10, marginal coverage should be ≥ 0.90 in expectation, with
    finite-sample fluctuation that the [0.85, 0.95] band absorbs.
    """

    # Shared coefs across train/test draws — exchangeability is the load-
    # bearing assumption for marginal-coverage, so train and test must be
    # drawn from the SAME logistic process. Generated once at class scope.
    _COVERAGE_COEFS: np.ndarray = np.random.default_rng(SEED).standard_normal(N_FEATURES)

    @classmethod
    def _stochastic_logistic_dgp(
        cls,
        n_samples: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stochastic logistic DGP — sized to guarantee n_minority > 50 and
        SHARING the class-level coefficients so train and test are i.i.d.
        from the same process. Per-call seed varies X and the y-noise draw
        but coefs stay fixed → exchangeability holds → α=0.10 coverage
        lands in [0.85, 0.95]."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, N_FEATURES))
        logits = X @ cls._COVERAGE_COEFS
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n_samples) < probs).astype(int)
        return X, y

    def _coverage_split(self, n_train: int = 800, n_test: int = 1000) -> float:
        from sklearn.linear_model import LogisticRegression

        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        X_tr, y_tr = self._stochastic_logistic_dgp(n_samples=n_train, seed=SEED)
        X_te, y_te = self._stochastic_logistic_dgp(n_samples=n_test, seed=SEED + 1)
        # Sanity: n_minority ≥ 50 → split mode fires.
        n_minority_train = int(min(np.sum(y_tr == 0), np.sum(y_tr == 1)))
        assert n_minority_train >= 50, f"DGP produced n_minority={n_minority_train} < 50"

        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            method="lac",
            alpha=0.10,
            random_state=SEED,
        )
        wrapper.fit(X_tr, y_tr)
        assert wrapper.fitted_conformal_mode_ == "split"

        sets = np.asarray(wrapper.predict_sets(X_te))
        # MAPIE prediction sets shape: (n_samples, n_classes, n_alphas).
        # For α=0.10 (single alpha) sets[:, :, 0] is the boolean mask of
        # included classes per sample. Coverage = fraction of samples whose
        # TRUE label class is in the predicted set.
        if sets.ndim == 3:
            mask = sets[:, :, 0]
        else:
            mask = sets
        covered = mask[np.arange(len(y_te)), y_te.astype(int)]
        return float(np.mean(covered))

    def test_split_conformal_coverage_in_acceptance_band_at_alpha_0_10(self):
        """Plan §6 T2.1 acceptance criterion."""
        coverage = self._coverage_split()
        assert 0.85 <= coverage <= 0.95, (
            f"Marginal coverage = {coverage:.3f} outside the plan §6 T2.1 "
            "acceptance band [0.85, 0.95] at α=0.10. The split-conformal "
            "guarantee should hold under stochastic-logistic DGP at "
            "n_minority ≥ 50."
        )


class TestMapieWrapperCrossConformalAcceptance:
    """Plan §6 T2.1 acceptance: cross-conformal switch tested at < 50."""

    def test_cross_conformal_fits_and_predicts_below_threshold(self):
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        X, y = _make_imbalanced_logistic_dgp(n_samples=200, minority_count=30)
        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
            method="lac",
            alpha=0.10,
            random_state=SEED,
        )
        wrapper.fit(X, y)
        # Cross-mode was selected at n_minority=30.
        assert wrapper.fitted_conformal_mode_ == "cross"
        # predict + predict_sets work end-to-end.
        proba = wrapper.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        sets = wrapper.predict_sets(X[:10])
        assert np.asarray(sets).ndim >= 2


class TestMapieWrapperStratifiedSplitEdgeCases:
    """MEDIUM-1: forced split mode with tiny minority should raise an actionable
    ValueError rather than letting a raw sklearn error propagate."""

    def test_forced_split_on_n_minority_1_raises_actionable_value_error(self):
        """n_minority=1 → sklearn stratified split fails; wrapper must re-raise
        with a message that names n_minority, calib_fraction, and remediation
        options."""
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, N_FEATURES))
        y = np.zeros(50, dtype=int)
        y[0] = 1  # n_minority = 1

        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=500, random_state=SEED),
            conformal_mode="split",
            calib_fraction=0.20,
            random_state=SEED,
        )
        with pytest.raises(ValueError, match="Stratified calibration split failed"):
            wrapper.fit(X, y)

    def test_forced_split_on_n_minority_2_succeeds(self):
        """n_minority=2 still has 2 class members → sklearn stratifies without
        error even though calib may contain 0 minority samples. The wrapper
        should complete fit (coverage guarantee is weak, but no crash)."""
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, N_FEATURES))
        y = np.zeros(100, dtype=int)
        y[:2] = 1  # n_minority = 2

        wrapper = MapieConformalBinaryClassifier(
            base_estimator=LogisticRegression(max_iter=500, random_state=SEED),
            conformal_mode="split",
            calib_fraction=0.20,
            random_state=SEED,
        )
        # Should not crash — sklearn can stratify with n_minority=2 (gets 0 in calib).
        wrapper.fit(X, y)
        assert wrapper.fitted_conformal_mode_ == "split"
        assert wrapper.n_minority_ == 2
