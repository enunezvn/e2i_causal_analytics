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


# ---------------------------------------------------------------------------
# v5 Gate B1 (2026-05-11): auto-policy regression tests.
# ---------------------------------------------------------------------------


class TestV5GateB1AutoPolicy:
    """Pins the v5 §2 B1 default auto-policy for calibration method.

    The policy: ``select_calibration_method`` returns ``"isotonic"`` when
    ``(y_val == 1).sum() > 100`` (= B1_AUTO_POLICY_N_POS_CROSSOVER), else
    ``"sigmoid"`` (Platt). This is the disease-agnostic default per the
    plan; tests pin the crossover constant + both branches + the audit-
    trail fields ``apply_post_hoc_calibration`` records.
    """

    def test_crossover_constant_pinned_at_100(self) -> None:
        """v5 §2 B1 declared the crossover at n_pos > 100."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            B1_AUTO_POLICY_N_POS_CROSSOVER,
        )

        assert B1_AUTO_POLICY_N_POS_CROSSOVER == 100, (
            "v5 B1 regression: crossover constant drifted from 100. "
            "Per plan §2 B1: 'isotonic for n_pos > 100; Platt for sparser'."
        )

    def test_select_isotonic_when_n_pos_above_crossover(self) -> None:
        """101+ positives → isotonic."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        y_val = np.concatenate([np.ones(150, dtype=np.int64), np.zeros(100, dtype=np.int64)])
        assert select_calibration_method(y_val) == "isotonic"

    def test_select_sigmoid_when_n_pos_below_crossover(self) -> None:
        """≤100 positives → sigmoid (Platt)."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        y_val = np.concatenate([np.ones(30, dtype=np.int64), np.zeros(200, dtype=np.int64)])
        assert select_calibration_method(y_val) == "sigmoid"

    def test_select_sigmoid_at_exact_crossover_boundary(self) -> None:
        """Exactly 100 positives → sigmoid (not isotonic — strict `>`)."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        y_val = np.concatenate([np.ones(100, dtype=np.int64), np.zeros(200, dtype=np.int64)])
        assert select_calibration_method(y_val) == "sigmoid"

    def test_select_isotonic_just_above_boundary(self) -> None:
        """101 positives → isotonic."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        y_val = np.concatenate([np.ones(101, dtype=np.int64), np.zeros(200, dtype=np.int64)])
        assert select_calibration_method(y_val) == "isotonic"

    def test_select_respects_custom_crossover(self) -> None:
        """Caller can override the crossover for cohort-specific policies."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        y_val = np.concatenate([np.ones(50, dtype=np.int64), np.zeros(200, dtype=np.int64)])
        assert select_calibration_method(y_val, n_pos_crossover=30) == "isotonic"
        assert select_calibration_method(y_val, n_pos_crossover=80) == "sigmoid"

    def test_select_degenerate_y_falls_back_to_sigmoid(self) -> None:
        """Non-binary / non-1D inputs → sigmoid (safer at low N)."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        # 2D input
        y_2d = np.ones((50, 2), dtype=np.int64)
        assert select_calibration_method(y_2d) == "sigmoid"
        # All zeros (no positives)
        y_zero = np.zeros(200, dtype=np.int64)
        assert select_calibration_method(y_zero) == "sigmoid"

    def test_auto_method_records_resolved_in_cal_info(self) -> None:
        """The audit-trail field ``calibration_method_resolved`` is
        always populated, even when the requested method is "auto"."""
        rng = np.random.default_rng(0)
        # 110 positives → policy resolves to isotonic
        n = 250
        y = np.concatenate([np.ones(110, dtype=np.int64), np.zeros(n - 110, dtype=np.int64)])
        rng.shuffle(y)
        X = rng.normal(size=(n, 3))
        model = LogisticRegression(max_iter=200).fit(X, y)
        _, info = apply_post_hoc_calibration(model, X, y, method="auto")
        assert info["calibration_applied"] is True
        assert info["calibration_method"] == "auto"
        assert info["calibration_method_resolved"] == "isotonic"
        assert info["calibration_fit_positives"] == 110
        assert info["calibration_auto_n_pos_crossover"] == 100

    def test_auto_method_resolves_to_sigmoid_at_low_n_pos(self) -> None:
        """50 positives → resolved to sigmoid under default crossover."""
        rng = np.random.default_rng(0)
        n = 200
        y = np.concatenate([np.ones(50, dtype=np.int64), np.zeros(n - 50, dtype=np.int64)])
        rng.shuffle(y)
        X = rng.normal(size=(n, 3))
        model = LogisticRegression(max_iter=200).fit(X, y)
        _, info = apply_post_hoc_calibration(model, X, y, method="auto")
        assert info["calibration_method"] == "auto"
        assert info["calibration_method_resolved"] == "sigmoid"
        assert info["calibration_fit_positives"] == 50

    def test_explicit_isotonic_bypasses_auto_policy(self) -> None:
        """Legacy callers that pass explicit "isotonic" keep that method
        even when the auto-policy would have chosen sigmoid."""
        rng = np.random.default_rng(0)
        n = 200
        y = np.concatenate([np.ones(30, dtype=np.int64), np.zeros(n - 30, dtype=np.int64)])
        rng.shuffle(y)
        X = rng.normal(size=(n, 3))
        model = LogisticRegression(max_iter=200).fit(X, y)
        _, info = apply_post_hoc_calibration(model, X, y, method="isotonic")
        assert info["calibration_method"] == "isotonic"
        assert info["calibration_method_resolved"] == "isotonic"
        # No auto-crossover field for non-auto calls
        assert info["calibration_auto_n_pos_crossover"] is None

    def test_explicit_sigmoid_bypasses_auto_policy(self) -> None:
        """Symmetric: explicit "sigmoid" keeps Platt even at high n_pos.

        Codex pass-1 MED-3: pin requested_method == "sigmoid" + the
        `calibration_auto_n_pos_crossover` field == None (matches the
        isotonic counterpart's pinning)."""
        rng = np.random.default_rng(0)
        n = 400
        y = np.concatenate([np.ones(150, dtype=np.int64), np.zeros(n - 150, dtype=np.int64)])
        rng.shuffle(y)
        X = rng.normal(size=(n, 3))
        model = LogisticRegression(max_iter=200).fit(X, y)
        _, info = apply_post_hoc_calibration(model, X, y, method="sigmoid")
        assert info["calibration_method"] == "sigmoid"
        assert info["calibration_method_resolved"] == "sigmoid"
        assert info["calibration_auto_n_pos_crossover"] is None

    def test_invalid_method_raises_value_error(self) -> None:
        """Unknown method names must fail loudly, not silently fall
        through. Protects future authors from typos like "platt"
        (correct: "sigmoid") or "isotonic_regression"."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(max_iter=200).fit(X, y)
        with pytest.raises(ValueError, match="is not one of"):
            apply_post_hoc_calibration(model, X, y, method="platt")

    def test_auto_records_resolved_even_when_skip_non_classifier(self) -> None:
        """When the base model isn't a classifier, the skip path still
        records the resolved method so audit consumers see what WOULD
        have been applied.

        Codex pass-1 MED-2: use a deterministically constructed y so
        the assertion can pin the expected resolution exactly (not
        the previous tautological ``in {"sigmoid", "isotonic"}``)."""
        rng = np.random.default_rng(0)
        n = 250
        # 50 positives → below crossover 100 → resolves to sigmoid.
        y = np.concatenate([np.ones(50, dtype=np.int64), np.zeros(n - 50, dtype=np.int64)])
        rng.shuffle(y)
        X = rng.normal(size=(n, 3))
        not_a_classifier = _NotAClassifier().fit(X, y)
        _, info = apply_post_hoc_calibration(not_a_classifier, X, y, method="auto")
        assert info["calibration_applied"] is False
        assert info["calibration_method"] == "auto"
        assert info["calibration_method_resolved"] == "sigmoid"
        assert info["skip_reason"] == "base_model_not_a_classifier"

    def test_select_degenerate_nan_y_falls_back_to_sigmoid(self) -> None:
        """Codex pass-1 MED-1: y_val with NaN values must fall back
        to sigmoid. Pre-fix, NaN survived int64 cast as the sentinel
        ``-9223372036854775808`` and was silently treated as a non-1
        label, potentially miscounting positives."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        # 150 finite "1" labels plus a NaN — pre-fix this might still
        # have resolved to isotonic; post-fix it falls back to sigmoid.
        y_nan = np.concatenate(
            [np.ones(150, dtype=np.float64), np.zeros(100, dtype=np.float64), np.array([np.nan])]
        )
        assert select_calibration_method(y_nan) == "sigmoid"

    def test_select_degenerate_non_binary_falls_back_to_sigmoid(self) -> None:
        """Codex pass-1 MED-1: multiclass / non-{0,1} labels fall back
        to sigmoid. The function's contract is binary classification."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        # 200 "1" + 100 "2" — both treated as positive under naive
        # ``arr == 1`` count? No — only "1" matches, so n_pos=200 > 100
        # → isotonic without the binary guard. With the guard, the
        # presence of "2" triggers the fallback to sigmoid.
        y_multi = np.concatenate(
            [np.ones(200, dtype=np.int64), np.full(100, 2, dtype=np.int64)]
        )
        assert select_calibration_method(y_multi) == "sigmoid"

    def test_select_empty_y_falls_back_to_sigmoid(self) -> None:
        """Codex pass-1 MED-1: empty arrays fall back to sigmoid."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        assert select_calibration_method(np.array([], dtype=np.int64)) == "sigmoid"

    def test_select_all_one_y_resolves_correctly(self) -> None:
        """All-one y_val: n_pos = n. If n > 100 → isotonic; if n ≤ 100
        → sigmoid. Pure positive-count logic with no binary-violation."""
        from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
            select_calibration_method,
        )

        y_all_one = np.ones(150, dtype=np.int64)
        assert select_calibration_method(y_all_one) == "isotonic"
        y_all_one_small = np.ones(50, dtype=np.int64)
        assert select_calibration_method(y_all_one_small) == "sigmoid"
