"""Tests for ``splits.py`` train/val/test stratified splitter (shard 02 §D)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.synthetic_v2.splits import stratified_train_val_test_split


def _make_balanced_dataset(rng: np.random.Generator, n: int = 1_000, prevalence: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n, 4))
    n_pos = int(round(n * prevalence))
    y = np.zeros(n, dtype=np.int64)
    pos_idx = rng.choice(n, size=n_pos, replace=False)
    y[pos_idx] = 1
    return X, y


class TestRatiosAndShape:
    def test_default_60_20_20_ratios_respected(self) -> None:
        rng = np.random.default_rng(42)
        X, y = _make_balanced_dataset(rng, n=1_000, prevalence=0.3)
        Xt, Xv, Xte, yt, yv, yte = stratified_train_val_test_split(
            X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        assert Xt.shape[0] == yt.shape[0]
        assert Xv.shape[0] == yv.shape[0]
        assert Xte.shape[0] == yte.shape[0]
        # Total preserved
        assert Xt.shape[0] + Xv.shape[0] + Xte.shape[0] == 1_000
        # Approximate 60/20/20 (sklearn rounds — exact depends on prevalence)
        assert abs(Xt.shape[0] - 600) <= 2
        assert abs(Xv.shape[0] - 200) <= 2
        assert abs(Xte.shape[0] - 200) <= 2
        # Same n_features
        assert Xt.shape[1] == Xv.shape[1] == Xte.shape[1] == 4

    def test_70_15_15_ratios(self) -> None:
        rng = np.random.default_rng(0)
        X, y = _make_balanced_dataset(rng, n=2_000, prevalence=0.4)
        Xt, Xv, Xte, *_ = stratified_train_val_test_split(
            X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=0
        )
        assert abs(Xt.shape[0] - 1400) <= 2
        assert abs(Xv.shape[0] - 300) <= 2
        assert abs(Xte.shape[0] - 300) <= 2


class TestStratificationPreserved:
    def test_class_balance_preserved_across_splits(self) -> None:
        rng = np.random.default_rng(42)
        X, y = _make_balanced_dataset(rng, n=2_000, prevalence=0.20)
        _, _, _, yt, yv, yte = stratified_train_val_test_split(
            X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        # Each split's prevalence within ±0.02 of the source
        assert abs(yt.mean() - 0.20) < 0.02
        assert abs(yv.mean() - 0.20) < 0.02
        assert abs(yte.mean() - 0.20) < 0.02

    def test_low_prevalence_stratification(self) -> None:
        rng = np.random.default_rng(7)
        X, y = _make_balanced_dataset(rng, n=10_000, prevalence=0.05)
        _, _, _, yt, yv, yte = stratified_train_val_test_split(
            X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=7
        )
        # Even at 5% prevalence, each split should have positives
        assert yt.sum() >= 1
        assert yv.sum() >= 1
        assert yte.sum() >= 1
        assert abs(yt.mean() - 0.05) < 0.01


class TestDeterminism:
    def test_same_seed_gives_same_splits(self) -> None:
        rng = np.random.default_rng(42)
        X, y = _make_balanced_dataset(rng, n=500, prevalence=0.3)
        out1 = stratified_train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=11)
        out2 = stratified_train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=11)
        for a, b in zip(out1, out2, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_splits(self) -> None:
        rng = np.random.default_rng(42)
        X, y = _make_balanced_dataset(rng, n=500, prevalence=0.3)
        out1 = stratified_train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=11)
        out2 = stratified_train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=12)
        # At least the y-arrays differ between splits
        assert not np.array_equal(out1[3], out2[3]) or not np.array_equal(out1[4], out2[4])


class TestValidation:
    def test_ratios_must_sum_to_one(self) -> None:
        rng = np.random.default_rng(42)
        X, y = _make_balanced_dataset(rng, n=200, prevalence=0.3)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            stratified_train_val_test_split(X, y, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2, seed=0)

    @pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5])
    def test_ratios_in_open_unit_interval(self, bad_ratio: float) -> None:
        rng = np.random.default_rng(42)
        X, y = _make_balanced_dataset(rng, n=200, prevalence=0.3)
        # Construct ratios where one is bad while sum approximately holds
        with pytest.raises(ValueError):
            stratified_train_val_test_split(
                X,
                y,
                train_ratio=bad_ratio,
                val_ratio=(1.0 - bad_ratio) / 2,
                test_ratio=(1.0 - bad_ratio) / 2,
                seed=0,
            )

    def test_X_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="X must be 2-D"):
            stratified_train_val_test_split(
                np.zeros(10), np.zeros(10), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0
            )

    def test_y_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="y must be 1-D with len"):
            stratified_train_val_test_split(
                np.zeros((10, 2)),
                np.zeros((5,)),
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=0,
            )
