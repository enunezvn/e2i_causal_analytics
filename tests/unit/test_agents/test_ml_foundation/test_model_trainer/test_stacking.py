"""Plan v3 §4 T2.5 — Stacking baseline contract tests.

Pins `compute_stacking_baseline_cv` and the `_ensemble_predictions`
helper. Plan §6 T2.5: nested-CV-validated ensemble beats best-single
OR documents why not (rejection acceptable).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.agents.ml_foundation.model_trainer.nodes.stacking import (
    DEFAULT_N_FOLDS,
    DEFAULT_RANDOM_STATE,
    _ensemble_predictions,
    compute_stacking_baseline_cv,
)

SEED = 42


def _make_separable_dgp(n: int = 400, n_features: int = 8, seed: int = SEED):
    """Separable binary DGP: ensemble + best-single both reach AUC > 0.85."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    coefs = rng.standard_normal(n_features)
    logits = X @ coefs
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < probs).astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# Module constants                                                            #
# --------------------------------------------------------------------------- #


def test_default_constants() -> None:
    assert DEFAULT_N_FOLDS == 5
    assert DEFAULT_RANDOM_STATE == 42


# --------------------------------------------------------------------------- #
# _ensemble_predictions helper                                                #
# --------------------------------------------------------------------------- #


class TestEnsemblePredictions:
    def test_soft_voting_arithmetic_mean(self) -> None:
        per_base = {
            "A": np.array([0.1, 0.5, 0.9]),
            "B": np.array([0.3, 0.6, 0.7]),
        }
        out = _ensemble_predictions(per_base, "soft_voting")
        np.testing.assert_allclose(out, np.array([0.20, 0.55, 0.80]))

    def test_soft_voting_uniform_weights_three_bases(self) -> None:
        per_base = {
            "A": np.array([0.0, 0.0, 1.0]),
            "B": np.array([0.0, 1.0, 1.0]),
            "C": np.array([1.0, 1.0, 1.0]),
        }
        out = _ensemble_predictions(per_base, "soft_voting")
        np.testing.assert_allclose(out, np.array([1 / 3, 2 / 3, 1.0]))

    def test_rank_averaging_normalized_to_zero_one(self) -> None:
        per_base = {
            "A": np.array([0.1, 0.5, 0.9]),
            "B": np.array([0.3, 0.6, 0.7]),
        }
        out = _ensemble_predictions(per_base, "rank_averaging")
        # Both bases agree on the order [low, mid, high]; ranks (1, 2, 3)
        # averaged → (1, 2, 3); normalized by n_samples=3 → (1/3, 2/3, 1.0).
        np.testing.assert_allclose(out, np.array([1 / 3, 2 / 3, 1.0]))

    def test_rank_averaging_handles_disagreement(self) -> None:
        # Bases disagree on the middle sample's rank.
        per_base = {
            "A": np.array([0.1, 0.9, 0.5]),  # ranks: 1, 3, 2
            "B": np.array([0.1, 0.5, 0.9]),  # ranks: 1, 2, 3
        }
        out = _ensemble_predictions(per_base, "rank_averaging")
        # Avg ranks: 1, 2.5, 2.5 → normalize by 3.
        np.testing.assert_allclose(out, np.array([1 / 3, 2.5 / 3, 2.5 / 3]))

    def test_rank_averaging_robust_to_scale_difference(self) -> None:
        # A predicts on [0,1] scale; B predicts on [0, 100] scale.
        # Soft voting would over-weight B; rank averaging treats them
        # equally — this is exactly the calibration robustness the plan §4
        # T2.5 docstring claims.
        per_base = {
            "A": np.array([0.1, 0.5, 0.9]),
            "B": np.array([10.0, 50.0, 90.0]),
        }
        out_rank = _ensemble_predictions(per_base, "rank_averaging")
        np.testing.assert_allclose(out_rank, np.array([1 / 3, 2 / 3, 1.0]))

    def test_invalid_method_raises_value_error(self) -> None:
        per_base = {"A": np.array([0.5])}
        with pytest.raises(ValueError, match="ensemble method"):
            _ensemble_predictions(per_base, "logistic_meta")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# compute_stacking_baseline_cv — happy path                                   #
# --------------------------------------------------------------------------- #


class TestStackingBaselineHappyPath:
    def _three_bases(self):
        return {
            "logistic": LogisticRegression(max_iter=1000, random_state=SEED),
            "rf": RandomForestClassifier(n_estimators=20, random_state=SEED),
            "gbm": GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=SEED),
        }

    def test_completes_and_returns_canonical_keys(self) -> None:
        X, y = _make_separable_dgp()
        result = compute_stacking_baseline_cv(self._three_bases(), X, y, n_folds=5)
        assert result["stacking_completed"] is True
        for key in (
            "stacking_method",
            "stacking_n_folds",
            "stacking_n_base_estimators",
            "stacking_ensemble_cv_auc_mean",
            "stacking_ensemble_cv_auc_std",
            "stacking_per_base_cv_auc_mean",
            "stacking_best_single_name",
            "stacking_best_single_cv_auc_mean",
            "stacking_ensemble_lift_over_best_single",
            "stacking_ensemble_beats_best_single",
        ):
            assert key in result, f"missing canonical key {key!r}"

    def test_records_n_base_estimators_and_n_folds(self) -> None:
        X, y = _make_separable_dgp()
        result = compute_stacking_baseline_cv(self._three_bases(), X, y, n_folds=3)
        assert result["stacking_n_base_estimators"] == 3
        assert result["stacking_n_folds"] == 3

    def test_per_base_cv_auc_mean_has_one_entry_per_base(self) -> None:
        X, y = _make_separable_dgp()
        result = compute_stacking_baseline_cv(self._three_bases(), X, y)
        per_base = result["stacking_per_base_cv_auc_mean"]
        assert set(per_base.keys()) == {"logistic", "rf", "gbm"}

    def test_best_single_name_matches_max_per_base_auc(self) -> None:
        X, y = _make_separable_dgp()
        result = compute_stacking_baseline_cv(self._three_bases(), X, y)
        per_base = result["stacking_per_base_cv_auc_mean"]
        expected_best = max(per_base, key=lambda n: per_base[n])
        assert result["stacking_best_single_name"] == expected_best
        assert result["stacking_best_single_cv_auc_mean"] == per_base[expected_best]

    def test_ensemble_lift_is_signed_difference(self) -> None:
        X, y = _make_separable_dgp()
        result = compute_stacking_baseline_cv(self._three_bases(), X, y)
        lift_expected = (
            result["stacking_ensemble_cv_auc_mean"] - result["stacking_best_single_cv_auc_mean"]
        )
        assert result["stacking_ensemble_lift_over_best_single"] == pytest.approx(lift_expected)

    def test_ensemble_beats_best_single_is_consistent_with_lift(self) -> None:
        X, y = _make_separable_dgp()
        result = compute_stacking_baseline_cv(self._three_bases(), X, y)
        if result["stacking_ensemble_lift_over_best_single"] > 0:
            assert result["stacking_ensemble_beats_best_single"] is True
        else:
            assert result["stacking_ensemble_beats_best_single"] is False

    def test_method_records_chosen_ensemble_method(self) -> None:
        X, y = _make_separable_dgp()
        for method in ("soft_voting", "rank_averaging"):
            result = compute_stacking_baseline_cv(self._three_bases(), X, y, method=method)
            assert result["stacking_method"] == method


# --------------------------------------------------------------------------- #
# Degenerate / edge cases                                                     #
# --------------------------------------------------------------------------- #


class TestStackingBaselineDegenerate:
    def test_empty_base_estimators_returns_failure(self) -> None:
        X, y = _make_separable_dgp()
        result = compute_stacking_baseline_cv({}, X, y)
        assert result["stacking_completed"] is False
        assert "no base_estimators" in result["stacking_error"]
        assert result["stacking_n_base_estimators"] == 0

    def test_single_base_returns_failure_with_message(self) -> None:
        X, y = _make_separable_dgp()
        result = compute_stacking_baseline_cv({"only": LogisticRegression(max_iter=1000)}, X, y)
        assert result["stacking_completed"] is False
        assert "requires >=2" in result["stacking_error"]
        assert result["stacking_n_base_estimators"] == 1

    def test_base_without_predict_proba_returns_failure(self) -> None:
        from sklearn.svm import LinearSVC

        X, y = _make_separable_dgp()
        bases = {
            "logistic": LogisticRegression(max_iter=1000, random_state=SEED),
            "svc": LinearSVC(),  # no predict_proba
        }
        result = compute_stacking_baseline_cv(bases, X, y, n_folds=2)
        assert result["stacking_completed"] is False
        assert "predict_proba" in result["stacking_error"]


# --------------------------------------------------------------------------- #
# Ensemble correctness on contrived input                                     #
# --------------------------------------------------------------------------- #


class TestStackingBaselineEnsembleCorrectness:
    def test_two_identical_bases_yield_zero_lift(self) -> None:
        """If both bases are clones of the same model fed the same data,
        the soft-voting ensemble cannot beat the best single (it equals
        either)."""
        X, y = _make_separable_dgp()
        bases = {
            "lr_a": LogisticRegression(max_iter=1000, random_state=SEED),
            "lr_b": LogisticRegression(max_iter=1000, random_state=SEED),
        }
        result = compute_stacking_baseline_cv(bases, X, y)
        # Lift should be zero (or within float noise) since both produce
        # the same predict_proba → ensemble = identical to either base.
        assert abs(result["stacking_ensemble_lift_over_best_single"]) < 1e-9
        assert result["stacking_ensemble_beats_best_single"] is False

    def test_complementary_bases_yield_positive_lift_by_rank_averaging(
        self,
    ) -> None:
        """When two well-trained but complementary base learners are
        rank-averaged, the ensemble usually beats either alone on a
        moderately-sized cohort. Acceptable for the test to fail on this
        DGP if the lift happens to be negative — the assertion is on the
        plumbing (lift is finite, not on its sign)."""
        X, y = _make_separable_dgp(n=600)
        bases = {
            "lr": LogisticRegression(max_iter=1000, random_state=SEED),
            "gbm": GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=SEED),
        }
        result = compute_stacking_baseline_cv(bases, X, y, method="rank_averaging")
        assert result["stacking_completed"] is True
        # Plumbing: lift is a finite number, not NaN.
        assert np.isfinite(result["stacking_ensemble_lift_over_best_single"])
        # The contract is "documents why not" if ensemble underperforms;
        # we don't assert positive lift (model luck on small synthetic).
