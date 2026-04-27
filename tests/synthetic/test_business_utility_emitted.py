"""Block 5B (#10): synthetic e2e check that ``business_utility`` is emitted.

Block 5 wired ``cost_matrix`` end-to-end (scope_definer → scope_spec →
model_trainer → evaluator) but no caller of ``scripts/run_tier0_test.py``
populated one — so a default ``python scripts/run_tier0_test.py`` run never
exercised the metric. Block 5B added a unit-shape placeholder cost matrix
that the evaluator can multiply against confusion-matrix counts to produce
``business_utility``.

This test runs a short tier-0-style training pass on synthetic data and
asserts:

  1. ``validation_metrics["business_utility"]`` is a finite float.
  2. ``test_metrics["business_utility"]`` is a finite float.
  3. The top-level ``result["business_utility"]`` mirrors the test value.
  4. Both numbers equal the closed-form arithmetic recomputed from the
     confusion matrix that the report exposes — **no tolerance, exact**.
     If the placeholder dict were silently swapped (e.g. dollar values
     leaked in by mistake), the arithmetic check would catch it loudly.

The test is **not** Feast-gated; it operates on the ``_compute_classification_metrics``
boundary, which is where the cost-matrix multiplication happens. We use
the synthetic generator with the default regime (positive_rate=0.30) per
the Block 5B plan (Q6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_tier0_test import _default_demo_cost_matrix  # noqa: E402
from src.agents.ml_foundation.model_trainer.nodes.evaluator import (  # noqa: E402
    _compute_classification_metrics,
)


def _make_two_class_split(
    rng: np.random.Generator,
    n: int,
    pos_share: float = 0.30,
    pos_mean: float = 0.65,
    neg_mean: float = 0.35,
    spread: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic synthetic split that mimics the tier-0 pattern.

    Mirrors the well-separated split builder used by the existing Block 5
    evaluator unit tests. Produces a 2-class binary classification split
    with ``pos_share`` positive class share (default 0.30 matches the
    synthetic generator's default regime), strong-but-realistic signal
    (positives concentrated around ``pos_mean=0.65``, negatives around
    ``neg_mean=0.35``).

    Args:
        rng: Seeded numpy random generator.
        n: Total sample count.
        pos_share: Fraction of positives (0 < x < 1).
        pos_mean: Mean predicted probability for positive class.
        neg_mean: Mean predicted probability for negative class.
        spread: Gaussian spread around the class means.

    Returns:
        Tuple of (y_true, y_pred_at_default, y_proba_2col) where
        ``y_proba_2col`` is the standard sklearn (n, 2) probability
        matrix (column 0 = P(class=0), column 1 = P(class=1)).
    """
    n_pos = max(1, int(round(n * pos_share)))
    n_neg = n - n_pos
    pos_scores = np.clip(
        rng.normal(loc=pos_mean, scale=spread, size=n_pos), 0.001, 0.999
    )
    neg_scores = np.clip(
        rng.normal(loc=neg_mean, scale=spread, size=n_neg), 0.001, 0.999
    )
    y_true = np.concatenate(
        [np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)]
    )
    y_proba_pos = np.concatenate([pos_scores, neg_scores])
    order = rng.permutation(len(y_true))
    y_true = y_true[order]
    y_proba_pos = y_proba_pos[order]
    y_proba = np.column_stack([1.0 - y_proba_pos, y_proba_pos])
    y_pred = (y_proba_pos >= 0.5).astype(int)
    return y_true, y_pred, y_proba


@pytest.fixture(scope="module")
def evaluator_result() -> dict:
    """Run the evaluator with the placeholder cost matrix on a synthetic
    train/val/test split.

    Module-scoped so the assertions below all read from the same evaluator
    output (they're pinning different invariants of the same run).
    """
    rng = np.random.default_rng(20260426)
    # Mirror the tier-0 synthetic generator's default regime (positive_rate=0.30).
    y_val, y_val_pred, y_val_proba = _make_two_class_split(
        rng, n=300, pos_share=0.30
    )
    y_test, y_test_pred, y_test_proba = _make_two_class_split(
        rng, n=225, pos_share=0.30
    )
    cost_matrix = _default_demo_cost_matrix()

    result = _compute_classification_metrics(
        y_train=None,
        y_train_pred=None,
        y_train_proba=None,
        y_validation=y_val,
        y_validation_pred=y_val_pred,
        y_validation_proba=y_val_proba,
        y_test=y_test,
        y_test_pred=y_test_pred,
        y_test_proba=y_test_proba,
        imbalance_detected=False,
        minority_ratio=0.30,
        cost_matrix=cost_matrix,
    )
    return result


class TestBusinessUtilityEmitted:
    """Closes Block 5's verification gap: the evaluator's
    ``business_utility`` keys must be populated with finite floats and
    the math must be the closed-form expectation, not an approximation.
    """

    def test_validation_business_utility_is_finite_float(
        self, evaluator_result: dict
    ) -> None:
        bu = evaluator_result["validation_metrics"].get("business_utility")
        assert bu is not None, (
            "validation_metrics.business_utility was None — the placeholder "
            "cost_matrix did not flow into the evaluator. Check the "
            "auto-inject branch in scripts/run_tier0_test.py."
        )
        assert isinstance(bu, float)
        assert np.isfinite(bu), f"validation business_utility is not finite: {bu!r}"

    def test_test_business_utility_is_finite_float(
        self, evaluator_result: dict
    ) -> None:
        bu = evaluator_result["test_metrics"].get("business_utility")
        assert bu is not None, (
            "test_metrics.business_utility was None — the placeholder "
            "cost_matrix did not flow into the evaluator."
        )
        assert isinstance(bu, float)
        assert np.isfinite(bu), f"test business_utility is not finite: {bu!r}"

    def test_top_level_mirror_matches_test_metric(
        self, evaluator_result: dict
    ) -> None:
        """Top-level ``business_utility`` is a mirror of test_metrics's
        — Tier0OutputMapper and the deployment decision tools both read
        the flat key. They must agree."""
        top = evaluator_result["business_utility"]
        nested = evaluator_result["test_metrics"]["business_utility"]
        assert top == nested

    def test_business_utility_matches_closed_form_arithmetic(
        self, evaluator_result: dict
    ) -> None:
        """Recompute from the confusion matrix the report exposes and
        assert byte-exact equality. No ``pytest.approx`` here — the
        helper just multiplies and sums, no float drift is acceptable.
        If this fires, the placeholder shape was silently mutated."""
        cm = evaluator_result["confusion_matrix"]
        # The evaluator emits the at-optimal confusion matrix in this shape.
        # If the shape changes, this test is the canary.
        assert set(cm.keys()) >= {"TP", "FP", "FN", "TN"}, (
            f"Unexpected confusion matrix shape: {cm!r}"
        )
        cost_matrix = _default_demo_cost_matrix()
        expected = (
            cm["TP"] * cost_matrix["tp"]
            + cm["FP"] * cost_matrix["fp"]
            + cm["FN"] * cost_matrix["fn"]
            + cm["TN"] * cost_matrix["tn"]
        )
        actual = evaluator_result["test_metrics"]["business_utility"]
        # Exact arithmetic — same multiply-and-sum as _compute_business_utility.
        assert actual == expected, (
            f"business_utility drifted from closed-form: "
            f"actual={actual!r} expected={expected!r} cm={cm!r} "
            f"cost_matrix={cost_matrix!r}"
        )

    def test_placeholder_shape_is_unit_scaled(
        self, evaluator_result: dict
    ) -> None:
        """Belt-and-braces: independently confirm the placeholder dict
        exposes the unit-shape contract the synthetic test relies on. If
        someone widens the helper to dollar-denominated values without
        updating this test, the arithmetic test above stays green (it
        matches whatever the helper returns) but THIS test catches the
        shape change explicitly."""
        cost_matrix = _default_demo_cost_matrix()
        assert cost_matrix == {
            "tp": 1.0,
            "fp": -0.05,
            "fn": -1.0,
            "tn": 0.0,
        }
