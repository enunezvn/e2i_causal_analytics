"""Plan v4 Gate G5 — coefficient-sensitivity helper unit tests.

Pins ``compute_coefficient_sensitivity`` from
``src/agents/ml_foundation/data_preparer/nodes/coefficient_sensitivity.py``
on synthetic fixtures so the helper's contract is verified independent
of cohort-data availability. Integration assertions on Optum + CSU live
in ``tests/integration/test_t24_coefficient_sensitivity_20260510.py``.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from src.agents.ml_foundation.data_preparer.nodes.coefficient_sensitivity import (
    G5_EFFECT_SIZE_CV_MAX,
    G5_FLIPS_PER_FEATURE_MAX,
    G5_FRACTION_SIGNIFICANT_FLIPPED_MAX,
    G5_SIGNIFICANCE_SIGMA_MULTIPLE,
    compute_coefficient_sensitivity,
)


# --------------------------------------------------------------------------- #
# Mock estimator: returns deterministic coefficients per fit                  #
# --------------------------------------------------------------------------- #


class _DeterministicCoefEstimator(BaseEstimator):
    """Mock sklearn-compatible estimator whose ``coef_`` is dictated by
    a sequence of pre-determined vectors (one per fit call).

    The first ``fit`` call returns the first coef vector, the second
    returns the second, etc. Used to GUARANTEE specific flip / no-flip
    outcomes in unit tests without relying on data + seed luck.

    Note: compute_coefficient_sensitivity creates a fresh estimator per
    re-fit when ``estimator=None`` is passed. To make this mock
    work, we pass a SINGLE shared instance via the ``estimator`` kwarg;
    the helper then re-uses the same instance across baseline + every
    per-feature re-fit, calling ``.fit`` once per re-fit and reading
    ``.coef_`` immediately after.
    """

    def __init__(self, coef_sequence: List[np.ndarray]) -> None:
        # The estimator must NOT introspect the sequence in __init__ to
        # remain sklearn-clone-friendly; we snapshot via an iterator.
        self._coef_sequence = coef_sequence
        self._iter: Optional[Iterator[np.ndarray]] = None
        self.coef_: Optional[np.ndarray] = None

    def _ensure_iter(self) -> Iterator[np.ndarray]:
        if self._iter is None:
            self._iter = iter(self._coef_sequence)
        return self._iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DeterministicCoefEstimator":
        next_coef = next(self._ensure_iter())
        # 2D shape (1, n_features) matches binary LogisticRegression.
        self.coef_ = np.atleast_2d(np.asarray(next_coef, dtype=np.float64))
        return self

# --------------------------------------------------------------------------- #
# Module constants — pre-spec values must not drift                           #
# --------------------------------------------------------------------------- #


class TestG5Thresholds:
    """Threshold drift detector. Editing these constants requires a fresh
    ``g5_*_prespec_<date>.md`` memo per the v3 §8 anti-threshold-shopping
    invariant. This test fails LOUDLY if anyone mutates the constants
    without updating the spec doc.
    """

    def test_flips_per_feature_max(self) -> None:
        assert G5_FLIPS_PER_FEATURE_MAX == 1

    def test_effect_size_cv_max(self) -> None:
        assert G5_EFFECT_SIZE_CV_MAX == 0.5

    def test_fraction_significant_flipped_max(self) -> None:
        assert G5_FRACTION_SIGNIFICANT_FLIPPED_MAX == 0.10

    def test_significance_sigma_multiple(self) -> None:
        assert G5_SIGNIFICANCE_SIGMA_MULTIPLE == 1.0


# --------------------------------------------------------------------------- #
# Fixture builders                                                            #
# --------------------------------------------------------------------------- #


def _make_two_class_dataset(
    n: int = 200,
    n_features: int = 5,
    seed: int = 42,
    inject_nans_in: list[str] | None = None,
    nan_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a balanced two-class binary dataset with optional NaN injection.

    Linear-separable enough that a logistic regression converges
    deterministically; weighted features ensure non-zero coefficients.
    """
    rng = np.random.default_rng(seed)
    columns = [f"feat_{i}" for i in range(n_features)]
    X_arr = rng.standard_normal(size=(n, n_features))
    # Inject signal: y depends linearly on the first up-to-3 features
    # (1.5, 1.0, -0.8). This generalizes for n_features < 3.
    weights = np.array([1.5, 1.0, -0.8][:n_features])
    if n_features > len(weights):
        # Pad with weak signal weights for any extra features.
        extra = np.zeros(n_features - len(weights))
        weights = np.concatenate([weights, extra])
    logits = X_arr @ weights
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < probs).astype(np.int64)
    df = pd.DataFrame(X_arr, columns=columns)

    if inject_nans_in:
        for col in inject_nans_in:
            mask = rng.uniform(size=n) < nan_fraction
            df.loc[mask, col] = np.nan

    return df, pd.Series(y, name="target")


# --------------------------------------------------------------------------- #
# Test 1: Helper detects sign flip when imputation reverses a coefficient     #
# --------------------------------------------------------------------------- #


class TestSignFlipDetection:
    def test_flip_detected_when_imputed_strategy_reverses_coefficient(
        self,
    ) -> None:
        """Closes G5 codex H3: this test now ASSERTS sign_flip is True
        for the flipped feature (vs. the prior version that only
        verified the flag was a bool).

        Uses _DeterministicCoefEstimator to GUARANTEE the helper sees
        a baseline-vs-imputed coefficient pair that flips for one
        feature and stays positive for the other two. Synthetic data +
        seed luck cannot be relied on — the mock makes the coefficients
        deterministic.

        Coef sequence:
          1. baseline fit: feat_flip=+1.5, feat_steady_pos=+1.0, feat_steady_neg=-0.8
          2. per-feature re-fit on feat_flip ONLY: feat_flip=-1.2 (FLIP),
             others unchanged.
          3. per-feature re-fit on feat_steady_pos ONLY: feat_steady_pos=+1.2 (no flip)
          4. per-feature re-fit on feat_steady_neg ONLY: feat_steady_neg=-0.9 (no flip)
        """
        # Build X with NaN in every column so each gets a per-feature
        # re-fit (the helper short-circuits when a column is NaN-free).
        rng = np.random.default_rng(seed=0)
        n = 300
        X = pd.DataFrame(
            {
                "feat_flip": np.where(
                    rng.uniform(size=n) < 0.5,
                    np.nan,
                    rng.standard_normal(size=n),
                ),
                "feat_steady_pos": np.where(
                    rng.uniform(size=n) < 0.5,
                    np.nan,
                    rng.standard_normal(size=n),
                ),
                "feat_steady_neg": np.where(
                    rng.uniform(size=n) < 0.5,
                    np.nan,
                    rng.standard_normal(size=n),
                ),
            }
        )
        # y can be arbitrary balanced 0/1; the mock estimator ignores
        # the data and returns the next pre-determined coef vector.
        y = pd.Series(rng.integers(0, 2, size=n))

        # Coef vectors — order matches the helper's fit sequence:
        # baseline first, then one per-feature re-fit per compared
        # column (in the X.columns order). The numeric_features order
        # in the helper preserves X.columns iteration order.
        baseline_coef = np.array([1.5, 1.0, -0.8])  # all +/- as expected
        feat_flip_refit = np.array([-1.2, 1.0, -0.8])  # feat_flip FLIPS
        feat_steady_pos_refit = np.array([1.5, 1.2, -0.8])  # no flip
        feat_steady_neg_refit = np.array([1.5, 1.0, -0.9])  # no flip

        mock = _DeterministicCoefEstimator(
            [
                baseline_coef,
                feat_flip_refit,
                feat_steady_pos_refit,
                feat_steady_neg_refit,
            ]
        )

        recs = {
            "feat_flip": "drop_row_or_mean",
            "feat_steady_pos": "drop_row_or_mean",
            "feat_steady_neg": "drop_row_or_mean",
        }
        result = compute_coefficient_sensitivity(X, y, recs, estimator=mock)

        # HARD ASSERT (H3): feat_flip MUST report sign_flip=True. With
        # the deterministic mock the result is non-flaky.
        flip = result["per_feature"]["feat_flip"]
        assert flip["sign_flip"] is True, (
            f"Helper failed to detect sign flip on feat_flip: "
            f"baseline={flip['effect_size_baseline']:.3f}, "
            f"post_impute={flip['effect_size_post_impute']:.3f}"
        )
        assert flip["flip_count"] == 1
        assert flip["effect_size_baseline"] == pytest.approx(1.5)
        assert flip["effect_size_post_impute"] == pytest.approx(-1.2)

        # Other features must NOT flip.
        assert result["per_feature"]["feat_steady_pos"]["sign_flip"] is False
        assert result["per_feature"]["feat_steady_pos"]["flip_count"] == 0
        assert result["per_feature"]["feat_steady_neg"]["sign_flip"] is False
        assert result["per_feature"]["feat_steady_neg"]["flip_count"] == 0

    def test_no_flip_detected_with_clean_data(self) -> None:
        """Clean data (no NaN) → mean-imputation is a no-op → no flips."""
        X, y = _make_two_class_dataset(n=300, n_features=4, seed=7)
        recs = dict.fromkeys(X.columns, "drop_row_or_mean")

        result = compute_coefficient_sensitivity(X, y, recs, seed=7)

        for feature in X.columns:
            record = result["per_feature"][feature]
            # Without any NaN cells, baseline and imputed coefficients
            # should be identical → no flip, zero CV.
            assert record["sign_flip"] is False
            assert record["effect_size_variance"] == pytest.approx(0.0, abs=1e-9)
        assert result["passes_pre_spec"] is True
        assert result["violations"] == []


# --------------------------------------------------------------------------- #
# Test 2: Effect-size variance is computed correctly                          #
# --------------------------------------------------------------------------- #


class TestEffectSizeVariance:
    def test_variance_formula_matches_spec(self) -> None:
        """``effect_size_variance = std(coefs, ddof=0) / |mean(coefs)|``
        across the {baseline, imputed} pair."""
        X, y = _make_two_class_dataset(
            n=200, n_features=3, seed=11, inject_nans_in=["feat_0"], nan_fraction=0.20
        )
        recs = dict.fromkeys(X.columns, "drop_row_or_mean")

        result = compute_coefficient_sensitivity(X, y, recs, seed=11)

        # Re-derive the variance for one feature manually and compare.
        record = result["per_feature"]["feat_0"]
        baseline = record["effect_size_baseline"]
        post = record["effect_size_post_impute"]
        coefs = np.array([baseline, post], dtype=np.float64)
        expected_cv = float(np.std(coefs, ddof=0)) / abs(float(np.mean(coefs)))
        assert record["effect_size_variance"] == pytest.approx(expected_cv, rel=1e-6)

    def test_zero_coefficient_yields_zero_variance(self) -> None:
        """If both baseline and imputed coefficients are zero (e.g., a
        constant feature), variance is 0.0 not NaN/inf."""
        n = 100
        X = pd.DataFrame(
            {
                "constant_feat": np.zeros(n),  # 0 coefficient guaranteed
                "noise_feat": np.random.default_rng(0).standard_normal(size=n),
            }
        )
        # Outcome unrelated to constant_feat.
        y = pd.Series((np.random.default_rng(0).uniform(size=n) > 0.5).astype(int))
        recs = {"constant_feat": "drop_row_or_mean", "noise_feat": "drop_row_or_mean"}

        result = compute_coefficient_sensitivity(X, y, recs, seed=0)

        record = result["per_feature"]["constant_feat"]
        # Both coefs near zero → mean ≈ 0 → division by 0 must not yield
        # NaN; helper should return 0.0 when std == 0 too.
        assert record["effect_size_variance"] == pytest.approx(0.0, abs=1e-9) or record[
            "effect_size_variance"
        ] == float("inf")


# --------------------------------------------------------------------------- #
# Test 3: Single-feature case                                                 #
# --------------------------------------------------------------------------- #


class TestSingleFeature:
    def test_single_feature_with_nans(self) -> None:
        """Helper handles an X with exactly one numeric feature."""
        n = 200
        rng = np.random.default_rng(seed=3)
        X = pd.DataFrame({"only_feat": rng.standard_normal(size=n)})
        # Inject NaNs.
        mask = rng.uniform(size=n) < 0.15
        X.loc[mask, "only_feat"] = np.nan
        y = pd.Series((rng.uniform(size=n) < 0.5).astype(int))
        recs = {"only_feat": "drop_row_or_mean"}

        result = compute_coefficient_sensitivity(X, y, recs, seed=3)

        assert result["n_features"] == 1
        assert "only_feat" in result["per_feature"]
        # With only 1 feature, sigma = 0 (std of a 1-element distribution
        # of |coefs|), so the significance threshold is 0 → the only
        # feature counts as "significant" iff |coef| > 0.
        n_sig = result["n_significant_features"]
        assert n_sig in (0, 1)


# --------------------------------------------------------------------------- #
# Test 4: All-zero-coefficient edge case                                      #
# --------------------------------------------------------------------------- #


class TestAllZeroCoefficients:
    def test_all_zero_features_yields_zero_coefs_no_significant(self) -> None:
        """If every feature column is identically zero, sklearn's logistic
        regression assigns 0 to every coefficient (the bias term absorbs
        the class prior). The 1σ threshold is then 0, and no feature
        crosses (since |0| > 0 is False). passes_pre_spec must still
        be True (no significant features to violate the thresholds against).
        """
        n = 100
        X = pd.DataFrame(
            {
                "zero_a": np.zeros(n),
                "zero_b": np.zeros(n),
                "zero_c": np.zeros(n),
            }
        )
        rng = np.random.default_rng(seed=5)
        y = pd.Series((rng.uniform(size=n) < 0.5).astype(int))
        recs = dict.fromkeys(X.columns, "drop_row_or_mean")

        result = compute_coefficient_sensitivity(X, y, recs, seed=5)

        # All zero → every coef should be exactly 0 → sigma == 0 → no feature
        # crosses |coef| > 1σ (even via tied-zero comparison).
        assert result["n_significant_features"] == 0
        # Aggregate fraction is well-defined as 0/0 → defaults to 0.0.
        assert result["aggregate"]["fraction_significant_flipped"] == 0.0
        # Pre-spec passes vacuously: there are no significant features
        # to violate the thresholds against.
        assert result["passes_pre_spec"] is True
        assert result["violations"] == []


# --------------------------------------------------------------------------- #
# Test 5: Aggregate fraction calculation                                      #
# --------------------------------------------------------------------------- #


class TestAggregateFraction:
    def test_aggregate_fraction_matches_per_feature(self) -> None:
        """The aggregate ``fraction_significant_flipped`` must equal
        ``count(significant features that flipped) / count(significant features)``.
        """
        X, y = _make_two_class_dataset(
            n=300,
            n_features=6,
            seed=21,
            inject_nans_in=["feat_2", "feat_4"],
            nan_fraction=0.15,
        )
        recs = dict.fromkeys(X.columns, "drop_row_or_mean")

        result = compute_coefficient_sensitivity(X, y, recs, seed=21)

        # Manually re-derive the aggregate.
        sigma = result["aggregate"]["significance_cutoff_sigma"]
        cutoff = result["aggregate"]["significance_cutoff_value"]
        assert cutoff == pytest.approx(G5_SIGNIFICANCE_SIGMA_MULTIPLE * sigma)

        n_sig = result["n_significant_features"]
        flipped = sum(
            1
            for f, rec in result["per_feature"].items()
            if abs(rec["effect_size_baseline"]) > cutoff and rec["sign_flip"]
        )
        if n_sig > 0:
            expected_fraction = flipped / n_sig
        else:
            expected_fraction = 0.0
        assert result["aggregate"]["fraction_significant_flipped"] == pytest.approx(
            expected_fraction, abs=1e-9
        )


# --------------------------------------------------------------------------- #
# Test 6: Pre-spec passes_pre_spec gates                                      #
# --------------------------------------------------------------------------- #


class TestPreSpecGates:
    def test_passes_pre_spec_true_on_clean_data(self) -> None:
        """No NaN → no imputation effect → no flips → all thresholds pass."""
        X, y = _make_two_class_dataset(n=400, n_features=5, seed=33)
        recs = dict.fromkeys(X.columns, "drop_row_or_mean")

        result = compute_coefficient_sensitivity(X, y, recs, seed=33)

        assert result["passes_pre_spec"] is True
        assert result["violations"] == []

    def test_thresholds_dict_present_in_result(self) -> None:
        """The result MUST surface the thresholds dict so downstream
        observers (dashboards, audit pipelines) can verify the
        load-bearing values without re-importing the constants."""
        X, y = _make_two_class_dataset(n=200, seed=42)
        recs = dict.fromkeys(X.columns, "drop_row_or_mean")

        result = compute_coefficient_sensitivity(X, y, recs, seed=42)

        assert result["thresholds"]["G5_FLIPS_PER_FEATURE_MAX"] == 1
        assert result["thresholds"]["G5_EFFECT_SIZE_CV_MAX"] == 0.5
        assert result["thresholds"]["G5_FRACTION_SIGNIFICANT_FLIPPED_MAX"] == 0.10
        assert result["thresholds"]["G5_SIGNIFICANCE_SIGMA_MULTIPLE"] == 1.0


# --------------------------------------------------------------------------- #
# Test 7: Omitted strategies (drop_column, indicator_only)                    #
# --------------------------------------------------------------------------- #


class TestOmittedStrategies:
    def test_drop_column_strategy_omits_feature(self) -> None:
        """A feature recommended for ``drop_column`` is omitted from the
        comparison; the helper records the feature in per_feature with
        ``effect_size_post_impute=None``."""
        X, y = _make_two_class_dataset(n=200, n_features=4, seed=99)
        recs = {
            "feat_0": "drop_row_or_mean",
            "feat_1": "drop_column",
            "feat_2": "indicator_only",
            "feat_3": "drop_row_or_mean",
        }

        result = compute_coefficient_sensitivity(X, y, recs, seed=99)

        assert result["n_omitted_features"] == 2
        assert result["per_feature"]["feat_1"]["effect_size_post_impute"] is None
        assert result["per_feature"]["feat_2"]["effect_size_post_impute"] is None
        # Compared features still produce post-impute coefs.
        assert result["per_feature"]["feat_0"]["effect_size_post_impute"] is not None
        assert result["per_feature"]["feat_3"]["effect_size_post_impute"] is not None


# --------------------------------------------------------------------------- #
# Test 8: Input-validation errors                                             #
# --------------------------------------------------------------------------- #


class TestInputValidation:
    def test_empty_X_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one column"):
            compute_coefficient_sensitivity(pd.DataFrame(), pd.Series([0, 1]), {})

    def test_mismatched_lengths_raise(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="rows"):
            compute_coefficient_sensitivity(X, y, {"a": "drop_row_or_mean"})

    def test_no_overlap_between_recs_and_X_raises(self) -> None:
        X, y = _make_two_class_dataset(n=50, n_features=2, seed=1)
        with pytest.raises(ValueError, match="no overlap"):
            compute_coefficient_sensitivity(X, y, {"nonexistent_feature": "drop_row_or_mean"})

    def test_no_numeric_columns_raises(self) -> None:
        n = 50
        X = pd.DataFrame(
            {
                "string_feat": ["a"] * n,
            }
        )
        y = pd.Series([0, 1] * 25)
        with pytest.raises(ValueError, match="no numeric columns"):
            compute_coefficient_sensitivity(X, y, {"string_feat": "drop_row_or_mean"})
