"""Tests for metric-computation helpers in the evaluator.

Covers:
* ``_compute_precision_at_k`` - the rank-truncated precision helper.
* ``_positive_class_proba`` - the 1D/2D probability-matrix coercer.
* ``_compute_business_utility`` (and its end-to-end surfacing through
  ``_compute_classification_metrics``) - the cost-matrix-driven Block 5
  metric.

Split from ``test_evaluator.py`` in 1A-M-6. Test names preserved
verbatim (CI history follows ``Class::method``).
"""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _compute_business_utility,
    _compute_classification_metrics,
    _compute_precision_at_k,
    _positive_class_proba,
)


class TestComputePrecisionAtK:
    """Test precision@k computation."""

    def test_computes_precision_at_k(self):
        """Should compute precision at various k values."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.column_stack(
            [
                1 - np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
                np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
            ]
        )

        result = _compute_precision_at_k(y_true, y_proba, k_values=[2, 4])

        assert 2 in result
        assert 4 in result
        assert 0.0 <= result[2] <= 1.0
        assert 0.0 <= result[4] <= 1.0

    def test_returns_empty_without_proba(self):
        """Should return empty dict without probabilities."""
        y_true = np.array([0, 0, 1, 1])

        result = _compute_precision_at_k(y_true, None, k_values=[2])

        assert result == {}

    def test_skips_k_larger_than_samples(self):
        """Should skip k values larger than sample size."""
        y_true = np.array([0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])

        result = _compute_precision_at_k(y_true, y_proba, k_values=[2, 100])

        assert 2 in result
        assert 100 not in result


# ============================================================================
# Helper: _positive_class_proba (1A-M3 dedup)
# ============================================================================


class TestPositiveClassProba:
    """Locks the contract of the shared positive-class extraction helper.

    All callers in evaluator.py route through this helper, so the test
    here is the canonical guarantee that 1D and 2D proba arrays are both
    accepted and that the 2D path returns the column-1 view.
    """

    def test_returns_column_1_for_2d_array(self):
        proba = np.array([[0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])
        result = _positive_class_proba(proba)
        np.testing.assert_array_equal(result, np.array([0.9, 0.3, 0.6]))

    def test_returns_unchanged_for_1d_array(self):
        proba = np.array([0.9, 0.3, 0.6])
        result = _positive_class_proba(proba)
        # Same reference is fine - caller only reads it.
        np.testing.assert_array_equal(result, proba)


# ============================================================================
# Block 5 (#10): business_utility metric driven by cost_matrix
# ============================================================================


class TestBusinessUtilityFromCostMatrix:
    """The evaluator must compute a cost-weighted business_utility metric
    from a caller-supplied ``cost_matrix`` and surface it in
    ``validation_metrics``, ``test_metrics``, and at the top-level result.

    Test design: build deterministic predictions where we KNOW the
    confusion matrix counts at the chosen threshold, supply a cost matrix
    with distinctive per-outcome values, and assert the multiplication
    matches the closed-form expectation.
    """

    @staticmethod
    def _make_split(rng, n, positive_score_mean, negative_score_mean, spread=0.05):
        """Reuse the same well-separated synthetic split builder as the
        threshold-tuning test fixture so test fixtures stay consistent."""
        n_pos = n // 2
        n_neg = n - n_pos
        pos_scores = np.clip(
            rng.normal(loc=positive_score_mean, scale=spread, size=n_pos),
            0.001,
            0.999,
        )
        neg_scores = np.clip(
            rng.normal(loc=negative_score_mean, scale=spread, size=n_neg),
            0.001,
            0.999,
        )
        y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
        y_proba_pos = np.concatenate([pos_scores, neg_scores])
        order = rng.permutation(len(y_true))
        y_true = y_true[order]
        y_proba_pos = y_proba_pos[order]
        y_proba = np.column_stack([1.0 - y_proba_pos, y_proba_pos])
        y_pred = (y_proba_pos >= 0.5).astype(int)
        return y_true, y_pred, y_proba

    def test_compute_business_utility_helper(self):
        """The standalone helper just multiplies counts by per-outcome dollars."""
        cost_matrix = {"tp": 100.0, "fp": -10.0, "fn": -50.0, "tn": 0.0}
        # 5 TP, 2 FP, 3 FN, 10 TN → 5*100 + 2*-10 + 3*-50 + 10*0 = 500 - 20 - 150 = 330
        utility = _compute_business_utility(tp=5, fp=2, fn=3, tn=10, cost_matrix=cost_matrix)
        assert utility == pytest.approx(330.0)

    def test_business_utility_in_validation_and_test_metrics(self):
        """When a cost_matrix is supplied, business_utility lands in BOTH
        validation_metrics and test_metrics, plus the top-level result."""
        rng = np.random.default_rng(20260427)
        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=100, positive_score_mean=0.70, negative_score_mean=0.30
        )
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.70, negative_score_mean=0.30
        )

        cost_matrix = {"tp": 100.0, "fp": -10.0, "fn": -50.0, "tn": 0.0}

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
            minority_ratio=0.5,
            cost_matrix=cost_matrix,
        )

        # Validation_metrics carries it.
        assert "business_utility" in result["validation_metrics"], (
            "business_utility must appear in validation_metrics so deployment "
            "tooling can rank candidates by business value at the chosen "
            "operating point."
        )
        # Test_metrics carries it (at the same chosen threshold).
        assert "business_utility" in result["test_metrics"]
        # Top-level mirror exists.
        assert "business_utility" in result

        # Sanity: dollars must be a float, not None.
        assert isinstance(result["test_metrics"]["business_utility"], float)
        assert isinstance(result["validation_metrics"]["business_utility"], float)
        assert isinstance(result["business_utility"], float)
        # Top-level mirrors the test_metrics value.
        assert result["business_utility"] == result["test_metrics"]["business_utility"]

    def test_business_utility_absent_when_no_cost_matrix(self):
        """When cost_matrix is not provided, the metric is NOT silently
        defaulted - it stays absent from validation_metrics/test_metrics
        and the top-level mirror is None. Callers can then distinguish
        'not configured' from 'configured but zero'."""
        rng = np.random.default_rng(20260427)
        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=80, positive_score_mean=0.70, negative_score_mean=0.30
        )
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=80, positive_score_mean=0.70, negative_score_mean=0.30
        )

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
            minority_ratio=0.5,
            cost_matrix=None,
        )

        # Top-level mirror is explicitly None when no cost_matrix.
        assert result["business_utility"] is None
        # Sub-metrics dicts must NOT have a stray placeholder.
        assert "business_utility" not in result["validation_metrics"]
        assert "business_utility" not in result["test_metrics"]

    def test_business_utility_uses_chosen_threshold_not_raw_predictions(self):
        """business_utility must be computed at the HEADLINE operating point,
        NOT at raw/default predictions when the chosen threshold IS the
        headline. In the IMBALANCED path the headline metrics are reported at
        the validation-tuned chosen threshold, so business_utility must track
        that chosen threshold (not the model's default 0.5).

        Findings #5 update: this test originally asserted the chosen-threshold
        behavior in the BALANCED path (``imbalance_detected=False``). But the
        balanced-path HEADLINE precision/recall/f1 are reported at 0.5, so the
        headline business_utility must be at 0.5 too (consistency with the
        headline metrics) — a chosen-threshold business_utility there is the
        very inconsistency Findings #5 fixes. The legitimate underlying intent
        of this guard — business_utility tracks the headline operating point
        and never silently regresses to raw predictions — is preserved by
        exercising it in the IMBALANCED path, where the headline genuinely IS
        the chosen threshold. (The balanced-path 0.5 contract is now covered
        by ``test_balanced_headline_confusion_matrix_matches_headline_precision``
        in ``test_evaluator.py``.)
        """
        rng = np.random.default_rng(20260428)
        # Validation: positives at 0.40, negatives at 0.20 → opt around 0.30
        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=100, positive_score_mean=0.40, negative_score_mean=0.20
        )
        # Test: positives at 0.55, negatives at 0.45 → at 0.30 chosen
        # threshold many rows are predicted positive; at 0.5 default,
        # the prediction split is very different.
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.55, negative_score_mean=0.45
        )
        # Cost matrix that strongly differentiates predicted-positive from
        # predicted-negative outcomes.
        cost_matrix = {"tp": 1000.0, "fp": -1.0, "fn": -1000.0, "tn": 1.0}

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
            # Findings #5: the headline business_utility tracks the chosen
            # threshold ONLY when the chosen threshold IS the headline operating
            # point, i.e. the imbalanced path.
            imbalance_detected=True,
            minority_ratio=0.5,
            cost_matrix=cost_matrix,
        )

        chosen = result["optimal_threshold"]
        # Recompute utility at the chosen threshold by reapplying it
        # ourselves and confirm it matches the evaluator's number.
        from sklearn.metrics import confusion_matrix as _cm

        y_test_pred_at_chosen = (y_test_proba[:, 1] >= chosen).astype(int)
        cm = _cm(y_test, y_test_pred_at_chosen)
        tn, fp, fn, tp = cm.ravel()
        expected = (
            tp * cost_matrix["tp"]
            + fp * cost_matrix["fp"]
            + fn * cost_matrix["fn"]
            + tn * cost_matrix["tn"]
        )
        assert result["test_metrics"]["business_utility"] == pytest.approx(expected)

        # Negative assertion: the utility computed at default 0.5 must
        # differ when chosen != 0.5 - proves the metric tracks the chosen
        # threshold, not the raw y_test_pred.
        if not np.isclose(chosen, 0.5):
            y_test_pred_at_05 = (y_test_proba[:, 1] >= 0.5).astype(int)
            cm_05 = _cm(y_test, y_test_pred_at_05)
            tn5, fp5, fn5, tp5 = cm_05.ravel()
            utility_at_05 = (
                tp5 * cost_matrix["tp"]
                + fp5 * cost_matrix["fp"]
                + fn5 * cost_matrix["fn"]
                + tn5 * cost_matrix["tn"]
            )
            assert utility_at_05 != pytest.approx(expected), (
                "If business_utility regresses to using y_test_pred (model's "
                "default 0.5), this assertion will trip - by construction the "
                "0.5-utility differs from the chosen-threshold utility."
            )
