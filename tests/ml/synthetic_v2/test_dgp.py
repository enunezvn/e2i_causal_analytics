"""Tests for ``dgp.py`` base machinery (shard 02)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.synthetic_v2.dgp import (
    apply_block_correlation,
    sample_one_feature,
    solve_intercept,
    standardize_train_val_test,
)


class TestSampleOneFeature:
    def test_normal_distribution_roughly_matches_params(self) -> None:
        rng = np.random.default_rng(42)
        x = sample_one_feature(rng, n=10_000, distribution="normal", params={"loc": 5.0, "scale": 2.0})
        assert x.shape == (10_000,)
        # SE on mean ≈ 2.0/sqrt(10000) = 0.02; 4σ tolerance for stability
        assert abs(x.mean() - 5.0) < 0.08
        assert abs(x.std(ddof=0) - 2.0) < 0.08

    def test_uniform_distribution_within_bounds(self) -> None:
        rng = np.random.default_rng(42)
        x = sample_one_feature(rng, n=5_000, distribution="uniform", params={"low": -1.0, "high": 3.0})
        assert x.shape == (5_000,)
        assert x.min() >= -1.0
        assert x.max() <= 3.0
        # Mean ≈ 1.0; SE ≈ (4/sqrt(12)) / sqrt(5000) ≈ 0.016; 4σ
        assert abs(x.mean() - 1.0) < 0.07

    def test_bernoulli_distribution_returns_floats(self) -> None:
        rng = np.random.default_rng(42)
        x = sample_one_feature(rng, n=5_000, distribution="bernoulli", params={"p": 0.3})
        assert x.shape == (5_000,)
        assert x.dtype == np.float64
        assert set(np.unique(x)) <= {0.0, 1.0}
        # SE ≈ sqrt(0.21/5000) ≈ 0.0065; 4σ
        assert abs(x.mean() - 0.3) < 0.04

    def test_categorical_raises(self) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="categorical not directly samplable"):
            sample_one_feature(
                rng,
                n=10,
                distribution="categorical",
                params={"categories": ["a", "b"], "probabilities": [0.5, 0.5]},
            )

    def test_unknown_distribution_raises(self) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unknown distribution"):
            sample_one_feature(rng, n=10, distribution="poisson", params={"lam": 1.0})

    def test_invalid_n_raises(self) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="n must be positive"):
            sample_one_feature(rng, n=0, distribution="normal", params={"loc": 0, "scale": 1})

    def test_determinism_same_seed(self) -> None:
        x1 = sample_one_feature(
            np.random.default_rng(7),
            n=100,
            distribution="normal",
            params={"loc": 0.0, "scale": 1.0},
        )
        x2 = sample_one_feature(
            np.random.default_rng(7),
            n=100,
            distribution="normal",
            params={"loc": 0.0, "scale": 1.0},
        )
        np.testing.assert_array_equal(x1, x2)


class TestApplyBlockCorrelation:
    def _make_independent_features(self, rng: np.random.Generator, n: int, n_features: int) -> np.ndarray:
        return rng.normal(size=(n, n_features))

    def test_realizes_target_pearson_r_positive(self) -> None:
        rng = np.random.default_rng(123)
        X = self._make_independent_features(rng, n=20_000, n_features=4)
        out = apply_block_correlation(rng, X, blocks=[([0, 1, 2], 0.7)])
        # Pearson r in the block ≈ 0.7
        c = np.corrcoef(out[:, [0, 1, 2]], rowvar=False)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(c[i, j] - 0.7) < 0.04, f"corr[{i},{j}]={c[i, j]}"
        # Untouched column 3 stays roughly uncorrelated with [0,1,2]
        for i in range(3):
            assert abs(np.corrcoef(out[:, i], out[:, 3])[0, 1]) < 0.05

    def test_realizes_target_pearson_r_negative_pair(self) -> None:
        rng = np.random.default_rng(321)
        X = self._make_independent_features(rng, n=20_000, n_features=2)
        out = apply_block_correlation(rng, X, blocks=[([0, 1], -0.7)])
        c = np.corrcoef(out, rowvar=False)
        assert abs(c[0, 1] - (-0.7)) < 0.04

    def test_realizes_target_pearson_r_negative_three_col(self) -> None:
        """3-col block with PSD-valid negative r (-0.3 > -1/(3-1) = -0.5)."""
        rng = np.random.default_rng(2024)
        X = self._make_independent_features(rng, n=20_000, n_features=3)
        out = apply_block_correlation(rng, X, blocks=[([0, 1, 2], -0.3)])
        c = np.corrcoef(out, rowvar=False)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(c[i, j] - (-0.3)) < 0.04, f"corr[{i},{j}]={c[i, j]}"

    def test_preserves_marginal_mean_and_std(self) -> None:
        rng = np.random.default_rng(99)
        X = rng.normal(loc=5.0, scale=2.0, size=(20_000, 3))
        out = apply_block_correlation(rng, X, blocks=[([0, 1, 2], 0.5)])
        for i in range(3):
            assert abs(out[:, i].mean() - X[:, i].mean()) < 1e-9
            assert abs(out[:, i].std(ddof=0) - X[:, i].std(ddof=0)) < 1e-9

    def test_singleton_block_is_no_op(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 3))
        out = apply_block_correlation(rng, X, blocks=[([0], 0.7)])
        np.testing.assert_array_equal(out, X)

    def test_no_blocks_is_no_op(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 3))
        out = apply_block_correlation(rng, X, blocks=[])
        np.testing.assert_array_equal(out, X)

    def test_non_psd_block_raises(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 3))
        # 3 cols with r=-0.7 → eigenvalue 1+2(-0.7)=-0.4 < 0 → not PSD
        with pytest.raises(ValueError, match="positive semi-definite|requires r in"):
            apply_block_correlation(rng, X, blocks=[([0, 1, 2], -0.7)])

    def test_out_of_range_index_raises(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 3))
        with pytest.raises(ValueError, match="out-of-range column index"):
            apply_block_correlation(rng, X, blocks=[([0, 5], 0.5)])

    def test_duplicate_indices_raise(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 3))
        with pytest.raises(ValueError, match="duplicate column indices"):
            apply_block_correlation(rng, X, blocks=[([0, 0, 1], 0.5)])

    def test_overlapping_blocks_raise(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 4))
        with pytest.raises(ValueError, match="overlap"):
            apply_block_correlation(
                rng,
                X,
                blocks=[([0, 1], 0.5), ([1, 2], 0.5)],
            )

    def test_two_d_required(self) -> None:
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="must be 2-D"):
            apply_block_correlation(rng, np.zeros(5), blocks=[])

    def test_determinism_with_same_input(self) -> None:
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)  # rng is unused in current impl; outputs match
        X = np.random.default_rng(0).normal(size=(500, 3))
        out_a = apply_block_correlation(rng_a, X, blocks=[([0, 1, 2], 0.5)])
        out_b = apply_block_correlation(rng_b, X, blocks=[([0, 1, 2], 0.5)])
        np.testing.assert_array_equal(out_a, out_b)


class TestStandardizeTrainValTest:
    def test_train_z_score_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=5.0, scale=2.0, size=(1_000, 4))
        X_val = rng.normal(loc=5.0, scale=2.0, size=(200, 4))
        X_test = rng.normal(loc=5.0, scale=2.0, size=(200, 4))
        Xt, Xv, Xte, mean, std = standardize_train_val_test(X_train, X_val, X_test)
        np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Xt.std(axis=0, ddof=0), 1.0, atol=1e-12)
        # Val/test use train stats — their realized mean is NOT exactly 0
        assert np.any(np.abs(Xv.mean(axis=0)) > 1e-3) or X_val.shape[0] < 50
        # mean/std returned are train statistics
        np.testing.assert_array_equal(mean, X_train.mean(axis=0))

    def test_no_leakage_val_test_use_train_stats(self) -> None:
        rng = np.random.default_rng(0)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(1_000, 2))
        X_val = rng.normal(loc=10.0, scale=5.0, size=(200, 2))  # very different
        X_test = rng.normal(loc=10.0, scale=5.0, size=(200, 2))
        Xt, Xv, Xte, mean, std = standardize_train_val_test(X_train, X_val, X_test)
        # Using train mean ≈ 0, train std ≈ 1: val z-scores should be far from 0
        assert np.all(np.abs(Xv.mean(axis=0)) > 5.0)
        assert np.all(np.abs(Xte.mean(axis=0)) > 5.0)

    def test_zero_variance_column_uses_safe_std_internally(self) -> None:
        """Internal divide uses safe-substituted std → no NaN. Returned ``std``
        is the raw value (0.0 for degenerate columns) so callers can detect them.
        """
        X_train = np.zeros((100, 3))
        X_val = np.zeros((20, 3))
        X_test = np.zeros((20, 3))
        Xt, Xv, Xte, _, std = standardize_train_val_test(X_train, X_val, X_test)
        assert not np.any(np.isnan(Xt))
        assert not np.any(np.isnan(Xv))
        assert not np.any(np.isnan(Xte))
        # Raw std is 0.0 for degenerate columns (caller can detect via std == 0)
        np.testing.assert_array_equal(std, np.zeros(3))

    def test_returned_std_is_raw_train_std(self) -> None:
        """Returned ``std`` must be the raw train-set std for downstream
        de-standardization (``X_z * std + mean``). A non-degenerate column's
        std is returned exactly.
        """
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=5.0, scale=2.0, size=(1_000, 4))
        X_val = X_train[:200]
        X_test = X_train[200:400]
        _, _, _, mean, std = standardize_train_val_test(X_train, X_val, X_test)
        np.testing.assert_array_equal(std, X_train.std(axis=0, ddof=0))
        np.testing.assert_array_equal(mean, X_train.mean(axis=0))

    def test_ndim_validation(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            standardize_train_val_test(np.zeros(5), np.zeros((10, 2)), np.zeros((10, 2)))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must share the same n_features"):
            standardize_train_val_test(np.zeros((10, 3)), np.zeros((10, 2)), np.zeros((10, 3)))


class TestSolveIntercept:
    def test_solves_to_target_prevalence_within_tol(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5_000, 5))
        coefs = np.array([0.5, -0.3, 0.2, 0.0, 0.1])
        target = 0.20
        b = solve_intercept(X, coefs, target_prevalence=target, tol=1e-4)
        z = X @ coefs + b
        p = 1.0 / (1.0 + np.exp(-z))
        assert abs(p.mean() - target) < 1e-4

    @pytest.mark.parametrize("target", [0.05, 0.20, 0.40, 0.60, 0.85])
    def test_solves_across_target_prevalences(self, target: float) -> None:
        rng = np.random.default_rng(13)
        X = rng.normal(size=(3_000, 4))
        coefs = np.array([0.7, -0.4, 0.2, 0.1])
        b = solve_intercept(X, coefs, target_prevalence=target)
        p = 1.0 / (1.0 + np.exp(-(X @ coefs + b)))
        assert abs(p.mean() - target) < 1e-4

    def test_zero_coefs_collapses_to_logit(self) -> None:
        X = np.zeros((1_000, 3))
        coefs = np.zeros(3)
        target = 0.30
        b = solve_intercept(X, coefs, target_prevalence=target, tol=1e-4)
        # With all-zero coefs and X, sigmoid(b) == target → b == logit(target).
        # Bisection halts when |sigmoid(b) - target| < tol; the resulting
        # |b - logit(target)| is bounded by tol / sigmoid'(logit(target)).
        # For target=0.3, sigmoid'(logit(0.3)) = 0.3 * 0.7 = 0.21, so
        # |b - logit(target)| <= 1e-4 / 0.21 ≈ 4.8e-4. We assert 1e-3 to
        # leave a small safety margin against future bisection-bracket changes.
        expected = float(np.log(target / (1.0 - target)))
        assert abs(b - expected) < 1e-3

    def test_invalid_target_prevalence_raises(self) -> None:
        X = np.zeros((10, 2))
        coefs = np.zeros(2)
        with pytest.raises(ValueError, match="must be in"):
            solve_intercept(X, coefs, target_prevalence=0.0)
        with pytest.raises(ValueError, match="must be in"):
            solve_intercept(X, coefs, target_prevalence=1.0)

    def test_shape_mismatch_raises(self) -> None:
        X = np.zeros((10, 3))
        with pytest.raises(ValueError, match="coefficients shape must match"):
            solve_intercept(X, np.zeros(2), target_prevalence=0.3)

    def test_x_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="X must be 2-D"):
            solve_intercept(np.zeros(5), np.zeros(5), target_prevalence=0.3)

    def test_invalid_bracket_raises(self) -> None:
        X = np.zeros((10, 2))
        with pytest.raises(ValueError, match="bracket must be"):
            solve_intercept(X, np.zeros(2), target_prevalence=0.3, bracket=(5.0, 5.0))

    def test_too_narrow_bracket_raises_runtime_error(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 2))
        coefs = np.array([1.0, 1.0])
        with pytest.raises(RuntimeError, match="failed to converge"):
            solve_intercept(
                X,
                coefs,
                target_prevalence=0.05,
                bracket=(0.0, 0.0001),
                max_iter=10,
            )

    def test_determinism_same_inputs(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(500, 3))
        coefs = np.array([0.3, -0.2, 0.5])
        b1 = solve_intercept(X, coefs, target_prevalence=0.25)
        b2 = solve_intercept(X, coefs, target_prevalence=0.25)
        assert b1 == b2


class TestEndToEndPrimitiveDeterminism:
    """Byte-identical reruns when feeding identical seeds + inputs."""

    def test_sample_then_correlate_determinism(self) -> None:
        # Run the (sample → correlate) pipeline twice with the same seed
        def _run(seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            cols = []
            for _ in range(5):
                cols.append(
                    sample_one_feature(rng, n=500, distribution="normal", params={"loc": 0.0, "scale": 1.0})
                )
            X = np.stack(cols, axis=1)
            rng2 = np.random.default_rng(seed + 1000)
            return apply_block_correlation(rng2, X, blocks=[([0, 1, 2], 0.5), ([3, 4], -0.3)])

        out1 = _run(123)
        out2 = _run(123)
        np.testing.assert_array_equal(out1, out2)

    def test_intercept_solver_same_input_byte_identical(self) -> None:
        rng = np.random.default_rng(999)
        X = rng.normal(size=(200, 3))
        coefs = np.array([0.4, -0.2, 0.3])
        b1 = solve_intercept(X, coefs, target_prevalence=0.18)
        b2 = solve_intercept(X.copy(), coefs.copy(), target_prevalence=0.18)
        assert b1 == b2
