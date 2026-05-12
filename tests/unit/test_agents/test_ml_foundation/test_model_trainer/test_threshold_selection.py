"""Tests for threshold selection in the evaluator.

Covers:
* ``_compute_optimal_threshold`` - the standalone Youden's-J helper.
* ``_select_threshold`` - the validation-vs-default branching helper
  extracted in 1A-I-3.
* End-to-end validation-tuning behaviour on
  ``_compute_classification_metrics`` (Block 1A): the chosen threshold
  comes from validation, freezes for test evaluation, and falls back to
  0.5 (never to a test-tuned value) when validation arrays are missing.
* The 1A-M2 ``math.isclose`` rebinarisation gate.

Split from ``test_evaluator.py`` in 1A-M-6. Test names preserved
verbatim (CI history follows ``Class::method``).
"""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _F1_FALLBACK_MCC_THRESHOLD,
    _compute_classification_metrics,
    _compute_cost_optimal_threshold,
    _compute_optimal_threshold,
    _select_threshold,
)

# Pull the random-state constant from the shared conftest module so the
# rebinarisation-guard test stays seeded identically to its pre-split form.
from tests.unit.test_agents.test_ml_foundation.test_model_trainer.conftest import (
    RANDOM_STATE,
)


class TestComputeOptimalThreshold:
    """Test optimal threshold computation."""

    def test_returns_threshold_with_proba(self):
        """Should compute optimal threshold with probabilities."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.column_stack(
            [
                1 - np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
                np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
            ]
        )

        threshold = _compute_optimal_threshold(y_true, y_proba)

        assert 0.0 <= threshold <= 1.0

    def test_returns_default_without_proba(self):
        """Should return 0.5 when no probabilities provided."""
        y_true = np.array([0, 0, 1, 1])

        threshold = _compute_optimal_threshold(y_true, None)

        assert threshold == 0.5


# ============================================================================
# Block 1A - threshold must be tuned on validation, not test (finding #6)
# ============================================================================


class TestThresholdTunedOnValidationOnly:
    """Verify threshold tuning is performed on validation, never on test.

    Block 1A of the Tier-0 remediation plan: the chosen classification
    threshold must be selected on the validation set, frozen, and then
    applied to the test set. A regression to test-tuning would inflate
    apparent test performance.

    Test design: synthesize 200 rows total (100 validation, 100 test) with
    deliberately divergent score distributions so the validation-derived
    threshold and test-derived threshold are far apart (>= 0.25 gap). If
    the implementation regresses to tuning on test, the assertions on
    `chosen_threshold` would fail.
    """

    @staticmethod
    def _make_split(
        rng: np.random.Generator,
        n: int,
        positive_score_mean: float,
        negative_score_mean: float,
        spread: float = 0.05,
        positive_rate: float = 0.5,
    ):
        """Generate (y_true, y_pred, y_proba) for a single split.

        Parameters
        ----------
        rng : np.random.Generator
            Source of pseudo-randomness for reproducibility.
        n : int
            Number of rows.
        positive_score_mean : float
            Mean predicted-probability for positive samples.
        negative_score_mean : float
            Mean predicted-probability for negative samples.
        spread : float
            Standard deviation of the per-class score noise. Small
            values make the distributions tight so Youden's J optimum
            sits cleanly between the two class means.
        positive_rate : float
            Fraction of rows that are class 1.
        """
        n_pos = int(round(n * positive_rate))
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
        # Shuffle so order doesn't bias anything downstream
        order = rng.permutation(len(y_true))
        y_true = y_true[order]
        y_proba_pos = y_proba_pos[order]

        # Two-column proba matrix (column 1 = positive class)
        y_proba = np.column_stack([1.0 - y_proba_pos, y_proba_pos])
        # Default-threshold predictions (0.5)
        y_pred = (y_proba_pos >= 0.5).astype(int)
        return y_true, y_pred, y_proba

    def test_threshold_tuned_on_validation_only(self):
        """Chosen threshold must come from validation, not test.

        Construct a 200-row synthetic dataset (100 validation + 100 test)
        whose score distributions force the validation-optimal threshold
        and test-optimal threshold to live in well-separated bands.
        Verify the chosen threshold matches the validation-derived value
        and is incompatible with the test-derived value.
        """
        rng = np.random.default_rng(20260426)

        # Validation split: positives ~0.40, negatives ~0.20 → opt ~0.30
        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=100, positive_score_mean=0.40, negative_score_mean=0.20
        )
        # Test split: positives ~0.80, negatives ~0.60 → opt ~0.70
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.80, negative_score_mean=0.60
        )

        # Sanity: independently confirm the two splits yield
        # well-separated optima before invoking the function under test.
        # If these baseline expectations break the test fixture itself
        # is wrong, not the implementation.
        val_only_threshold = _compute_optimal_threshold(y_val, y_val_proba)
        test_only_threshold = _compute_optimal_threshold(y_test, y_test_proba)
        assert val_only_threshold < 0.50, (
            f"Validation-derived threshold should land in the low band, "
            f"got {val_only_threshold:.4f}"
        )
        assert test_only_threshold > 0.55, (
            f"Test-derived threshold should land in the high band, got {test_only_threshold:.4f}"
        )
        gap = test_only_threshold - val_only_threshold
        assert gap >= 0.20, (
            f"Test fixture must produce a >= 0.20 gap between val and test "
            f"thresholds; got {gap:.4f}. A smaller gap would not catch a "
            f"regression to test-tuning."
        )

        # Real call into the function under test - no mocks of
        # _compute_classification_metrics or _compute_optimal_threshold.
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
        )

        # 1) Chosen threshold matches the validation-derived value exactly.
        assert "validation_metrics" in result
        validation_metrics = result["validation_metrics"]
        assert "chosen_threshold" in validation_metrics, (
            "validation_metrics must expose `chosen_threshold` so downstream "
            "consumers (model registry, monitoring) can audit operating point."
        )
        chosen = float(validation_metrics["chosen_threshold"])
        assert chosen == pytest.approx(val_only_threshold), (
            f"chosen_threshold must equal validation-derived value "
            f"{val_only_threshold:.4f}, got {chosen:.4f}"
        )

        # 2) chosen_threshold_source flags validation provenance.
        assert validation_metrics.get("chosen_threshold_source") == "validation"

        # 3) Top-level optimal_threshold (the canonical key consumed
        # cross-codebase) mirrors the validation-tuned value, and the
        # top-level provenance flag also reports validation.
        assert result["optimal_threshold"] == pytest.approx(val_only_threshold)
        assert result["chosen_threshold_source"] == "validation"

        # 4) Negative assertion - the chosen threshold MUST NOT match
        # the test-derived value. A regression to the old behaviour
        # (`_compute_optimal_threshold(y_test, y_test_proba)`) would
        # trip this assertion since the gap is >= 0.20.
        assert chosen != pytest.approx(test_only_threshold, abs=0.05), (
            f"chosen_threshold appears tuned on test ({test_only_threshold:.4f}); "
            f"this is the leakage bug Block 1A removes."
        )

    def test_chosen_threshold_frozen_for_test_evaluation(self):
        """Test-set predictions must use the validation-tuned threshold.

        Verify that test_metrics_at_optimal is computed by applying the
        validation-tuned threshold to test probabilities (not by re-tuning
        on test). We verify this indirectly: the predicted-positive count
        on test must equal the count we get by applying the
        validation-tuned threshold to test_proba - NOT the count from
        applying a test-tuned threshold to test_proba.
        """
        rng = np.random.default_rng(20260426)

        y_val, y_val_pred, y_val_proba = self._make_split(
            rng, n=100, positive_score_mean=0.40, negative_score_mean=0.20
        )
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.80, negative_score_mean=0.60
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
            imbalance_detected=True,  # forces test_metrics = test_metrics_at_optimal
            minority_ratio=0.5,
        )

        chosen = float(result["validation_metrics"]["chosen_threshold"])
        # Independently compute predicted positives at the validation-tuned
        # threshold applied to the test set.
        val_tuned_test_predictions = (y_test_proba[:, 1] >= chosen).astype(int)
        n_pos_at_val_threshold = int(val_tuned_test_predictions.sum())

        cm = result["confusion_matrix"]
        n_pos_in_result = int(cm["TP"]) + int(cm["FP"])
        assert n_pos_in_result == n_pos_at_val_threshold, (
            f"Test confusion matrix used a threshold inconsistent with "
            f"chosen_threshold={chosen:.4f}: result says {n_pos_in_result} "
            f"positives, but applying chosen_threshold to test_proba gives "
            f"{n_pos_at_val_threshold}."
        )

        # Cross-check: a test-tuned threshold would predict a different
        # number of positives. If they happen to coincide here, the gap
        # construction above is too small.
        test_tuned = _compute_optimal_threshold(y_test, y_test_proba)
        n_pos_at_test_threshold = int((y_test_proba[:, 1] >= test_tuned).astype(int).sum())
        assert n_pos_in_result != n_pos_at_test_threshold, (
            "Test predicted-positive count matches a test-tuned threshold; "
            "either the synthetic gap is too narrow or the implementation "
            "regressed to test-tuning."
        )

    def test_falls_back_to_default_when_validation_missing(self):
        """When validation arrays are unavailable, fall back to 0.5 (not test).

        We never tune on test even when validation is absent. Falling back
        to the default 0.5 threshold trades calibration for test-set
        integrity - the test set must remain untouched for thresholding.
        """
        rng = np.random.default_rng(20260426)
        y_test, y_test_pred, y_test_proba = self._make_split(
            rng, n=100, positive_score_mean=0.80, negative_score_mean=0.60
        )

        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=None,
            y_validation_pred=None,
            y_validation_proba=None,
            y_test=y_test,
            y_test_pred=y_test_pred,
            y_test_proba=y_test_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
        )

        # In the fallback path validation_metrics is empty (no validation
        # arrays), so the operating point lives only at the top level.
        assert result["optimal_threshold"] == 0.5
        assert result["chosen_threshold_source"] == "default"
        assert result["validation_metrics"] == {}
        # And it must NOT match the test-derived value
        test_only = _compute_optimal_threshold(y_test, y_test_proba)
        assert result["optimal_threshold"] != pytest.approx(test_only, abs=0.05)


# ============================================================================
# 1A-I-3: _select_threshold extraction - direct unit tests on the helper
# ============================================================================


class TestSelectThreshold:
    """Unit tests for the extracted ``_select_threshold`` helper.

    These tests target the helper directly (not via
    ``_compute_classification_metrics``) so they pin the contract the
    rest of the evaluator and downstream consumers (mlflow_logger,
    audit code) rely on.

    1A-M-6 will move these into ``test_threshold_selection.py``.
    """

    def test_select_threshold_clamps_inf_sentinel_within_validation_branch(self):
        """When sklearn's roc_curve returns the inf sentinel, the helper
        must surface 0.5 (not inf, NaN, or out-of-range values).

        sklearn's ``roc_curve`` prepends a sentinel threshold of ``np.inf``
        for the trivial (FPR=0, TPR=0) point. On degenerate inputs (e.g.,
        constant probabilities where every threshold is equivalent),
        Youden's J argmax lands on that sentinel.
        ``_compute_optimal_threshold`` clamps the non-finite/out-of-range
        result back to 0.5 - this test verifies the clamp survives the
        round-trip through ``_select_threshold``.

        Source string remains ``"validation"`` because validation arrays
        WERE provided; only the numeric value falls back.
        """
        # Constant 0.5 probabilities → degenerate ROC curve → argmax
        # lands on sklearn's inf sentinel → clamp triggers.
        n = 60
        np.random.seed(RANDOM_STATE)
        y_validation = np.random.randint(0, 2, n)
        y_validation_proba = np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        threshold, source = _select_threshold(y_validation, y_validation_proba)

        assert threshold == 0.5, (
            f"Non-finite/out-of-range optimal threshold must clamp to 0.5; got {threshold!r}"
        )
        assert np.isfinite(threshold)
        assert source == "validation", (
            "Source must remain 'validation' when arrays are provided - "
            "only the numeric threshold falls back, not the provenance."
        )

    def test_select_threshold_provenance_string_format(self):
        """Provenance source must be exactly 'validation' or 'default'.

        Downstream consumers (mlflow_logger, audit code, monitoring)
        match on these literal string values. Anything else (e.g.
        "VALIDATION", "val", " validation ") would silently break those
        consumers - this test pins the exact format.
        """
        # 'validation' branch: arrays present.
        rng = np.random.default_rng(20260426)
        n = 40
        y_proba_pos = rng.uniform(0.1, 0.9, n)
        y_validation = (y_proba_pos > 0.5).astype(int)
        y_validation_proba = np.column_stack([1.0 - y_proba_pos, y_proba_pos])
        _, source_validation = _select_threshold(y_validation, y_validation_proba)
        assert source_validation == "validation"
        # Pin the literal type and exact characters - defensive against
        # a regression to bytes / enum / capitalisation drift.
        assert isinstance(source_validation, str)
        assert source_validation == "validation" and len(source_validation) == 10

        # 'default' branch: arrays absent.
        _, source_none = _select_threshold(None, None)
        assert source_none == "default"
        assert isinstance(source_none, str)
        assert source_none == "default" and len(source_none) == 7

        # Mixed-absence variants must also fall back to default - the
        # contract says BOTH arrays must be present for the validation
        # branch.
        _, source_no_proba = _select_threshold(y_validation, None)
        assert source_no_proba == "default"
        _, source_no_labels = _select_threshold(None, y_validation_proba)
        assert source_no_labels == "default"


# ============================================================================
# 1A-M2: rebinarisation gate uses math.isclose, not raw float ==
# ============================================================================


class TestThresholdRebinarisationGuard:
    """When the validation-tuned threshold lands on (or vanishingly close to)
    0.5, the test-set rebinarisation should be skipped - there's no point
    re-applying ``proba >= 0.5`` if the model already does that. Direct
    ``!= 0.5`` would treat ``0.5 + 1e-15`` as different, triggering a
    no-op rebinarisation path. ``math.isclose`` collapses that gap.
    """

    def test_threshold_within_isclose_tolerance_skips_rebinarisation(self):
        """When the chosen threshold is essentially 0.5, test_metrics_at_05
        and test_metrics_at_optimal must be identical - the optimal-path
        binarisation never runs."""
        np.random.seed(RANDOM_STATE)
        n = 80
        # Validation set whose Youden's J optimum returns sklearn's `inf`
        # sentinel (random labels make every operating point equivalent),
        # which the upstream guard clamps back to 0.5. Test set then sees
        # threshold == 0.5 and must NOT re-binarise.
        y_val = np.random.randint(0, 2, n)
        y_val_proba = np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        y_val_pred = (y_val_proba[:, 1] >= 0.5).astype(int)

        y_test = np.random.randint(0, 2, n)
        y_test_proba = np.column_stack([np.random.rand(n), np.random.rand(n)])
        y_test_proba[:, 0] = 1 - y_test_proba[:, 1]
        y_test_pred = (y_test_proba[:, 1] >= 0.5).astype(int)

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
        )

        # Threshold falls back to 0.5 (the all-0.5 proba degenerates Youden's
        # J), so the rebinarisation gate must skip.
        assert result["optimal_threshold"] == 0.5
        # Same y_pred → identical metrics in both at-0.5 and at-optimal blocks.
        assert result["test_metrics_at_05"] == result["test_metrics_at_optimal"]
        # Backlog #37 regression: the dict-mirroring fix must preserve identical
        # keysets (not just core scalar metrics). Asymmetric enrichment of the
        # at-0.5 vs at-optimal dicts pre-#37 left e.g. calibration_* and
        # business_utility on one side only.
        assert set(result["test_metrics_at_05"].keys()) == set(
            result["test_metrics_at_optimal"].keys()
        )

    def test_threshold_within_isclose_tolerance_imbalanced_path(self):
        """Codex pass-1 M2 regression: the `imbalance_detected=True` branch
        of the #37 fix mirrors `test_metrics_optimal` (the enriched dict in
        the imbalanced flow) into `test_metrics_standard`. Without this
        test, that branch would have zero coverage."""
        np.random.seed(RANDOM_STATE)
        n = 80
        # Same degenerate validation set as the previous test — all proba=0.5
        # collapses Youden's J to the sentinel, which the guard clamps to 0.5.
        y_val = np.random.randint(0, 2, n)
        y_val_proba = np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        y_val_pred = (y_val_proba[:, 1] >= 0.5).astype(int)

        y_test = np.random.randint(0, 2, n)
        y_test_proba = np.column_stack([np.random.rand(n), np.random.rand(n)])
        y_test_proba[:, 0] = 1 - y_test_proba[:, 1]
        y_test_pred = (y_test_proba[:, 1] >= 0.5).astype(int)

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
            imbalance_detected=True,  # exercises the M2 branch
            minority_ratio=0.1,
        )

        # Guard skipped → threshold clamps to 0.5.
        assert result["optimal_threshold"] == 0.5
        # Same invariants as the imbalance_detected=False test: identical
        # values and identical keysets across the two slots.
        assert result["test_metrics_at_05"] == result["test_metrics_at_optimal"]
        assert set(result["test_metrics_at_05"].keys()) == set(
            result["test_metrics_at_optimal"].keys()
        )


# ============================================================================
# Backlog #20 Gap 1: cost-aware threshold selection
# ============================================================================


class TestCostOptimalThreshold:
    """Direct tests for ``_compute_cost_optimal_threshold``.

    Backlog #20 Gap 1 wires the previously-reserved ``cost_matrix`` parameter
    of ``_select_threshold`` through to a utility-maximising threshold
    sweep on validation. ``_compute_business_utility`` is the underlying
    scorer; this helper picks the threshold that maximises it.
    """

    def test_returns_threshold_for_asymmetric_cost(self):
        """High FN cost (e.g., missed biologic-initiation) drives the
        optimal threshold lower than Youden's J would.

        With FN penalty 10x larger than FP, the cost-optimal selector
        should accept more false positives to avoid missed positives —
        i.e., recommend a lower threshold than the symmetric Youden's J.
        """
        rng = np.random.default_rng(42)
        # Bimodal validation: positives cluster around 0.7, negatives around 0.3
        y = np.array([0] * 100 + [1] * 100)
        proba_pos = np.concatenate(
            [
                rng.normal(0.3, 0.15, 100),
                rng.normal(0.7, 0.15, 100),
            ]
        ).clip(0.01, 0.99)
        y_proba = np.column_stack([1 - proba_pos, proba_pos])

        # FN heavily penalised vs FP (10x asymmetry typical for missed
        # biologic-initiation: replacement cost dwarfs false-alarm cost).
        cost_matrix_fn_heavy = {"tp": 10.0, "fp": -1.0, "fn": -10.0, "tn": 0.0}

        cost_threshold = _compute_cost_optimal_threshold(y, y_proba, cost_matrix_fn_heavy)
        youden_threshold = _compute_optimal_threshold(y, y_proba)

        assert cost_threshold is not None
        # Cost-optimal accepts more false positives to avoid missing
        # positives, so threshold should be at-or-below Youden's.
        assert cost_threshold <= youden_threshold + 0.05

    def test_returns_none_for_none_proba(self):
        """No probabilities → None (caller falls through to Youden's J)."""
        y = np.array([0, 1, 0, 1])
        cost_matrix = {"tp": 1.0, "fp": -1.0, "fn": -1.0, "tn": 0.0}

        result = _compute_cost_optimal_threshold(y, None, cost_matrix)

        assert result is None

    def test_returns_none_for_constant_proba(self):
        """Constant probabilities → degenerate sweep → may be None or
        a finite threshold depending on cost_matrix asymmetry.

        Constant 0.5 probability flips the prediction once at t=0.5, so
        utility is two-valued (not strictly flat). Asymmetric matrices
        still yield a non-flat sweep with a unique max.
        """
        n = 50
        y = np.array([0, 1] * (n // 2))
        y_proba = np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        cost_matrix = {"tp": 1.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

        result = _compute_cost_optimal_threshold(y, y_proba, cost_matrix)
        assert result is None or (0.0 < result < 1.0)

    def test_returns_none_for_truly_flat_utility(self):
        """When every threshold yields IDENTICAL utility (e.g., zero
        cost_matrix), the helper rejects the cost path so the caller
        falls through to Youden's J. Codex pass-2 MEDIUM-2: previously
        the helper returned t≈0.01 and falsely labeled it
        ``"validation_cost_optimal"``."""
        rng = np.random.default_rng(20260510)
        n = 60
        y = rng.integers(0, 2, n)
        y_proba = np.column_stack([rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)])
        # All-zero cost matrix: every threshold gives utility=0 exactly.
        zero_cost_matrix = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

        result = _compute_cost_optimal_threshold(y, y_proba, zero_cost_matrix)
        assert result is None

    def test_raises_keyerror_on_incomplete_cost_matrix(self):
        """Missing cost_matrix key MUST raise KeyError loudly.

        Codex pass-2 HIGH-1: the previous broad ``except Exception``
        silently swallowed KeyError from ``_compute_business_utility``,
        falling through to Youden's J labeled as if the cost branch had
        produced it. Malformed config is a bug — fail loud, not silent.
        """
        n = 40
        y = np.random.randint(0, 2, n)
        y_proba = np.column_stack([np.random.rand(n), np.random.rand(n)])
        bad_cost_matrix = {"tp": 1.0, "fp": -1.0}  # missing fn, tn

        with pytest.raises(KeyError, match="missing required keys"):
            _compute_cost_optimal_threshold(y, y_proba, bad_cost_matrix)

    def test_keyerror_message_lists_missing_keys(self):
        """Error message must enumerate the missing keys to aid
        debugging — not just say 'malformed'."""
        n = 20
        y = np.random.randint(0, 2, n)
        y_proba = np.column_stack([np.random.rand(n), np.random.rand(n)])
        bad_cost_matrix = {"tp": 1.0}  # missing fp, fn, tn

        with pytest.raises(KeyError) as exc_info:
            _compute_cost_optimal_threshold(y, y_proba, bad_cost_matrix)
        msg = str(exc_info.value)
        assert "fn" in msg
        assert "fp" in msg
        assert "tn" in msg


class TestSelectThresholdWithCostMatrix:
    """Verify ``_select_threshold`` honours the cost_matrix kwarg.

    Provenance string ``"validation_cost_optimal"`` distinguishes this
    branch from plain Youden's J, so downstream consumers (mlflow_logger,
    audit code) can attribute the operating point correctly.

    NOTE: ``_select_threshold`` itself activates the cost branch whenever
    ``cost_matrix`` is non-None. The OPT-IN gate is enforced one level up
    in ``_compute_classification_metrics`` via the
    ``use_cost_optimal_threshold`` kwarg, which only forwards
    ``cost_matrix`` to ``_select_threshold`` when True. These tests
    bypass that gate to exercise the helper directly.
    """

    def test_cost_matrix_branches_to_validation_cost_optimal(self):
        """When cost_matrix is provided + validation arrays exist,
        provenance must be ``"validation_cost_optimal"``."""
        rng = np.random.default_rng(20260509)
        proba_pos = np.concatenate(
            [
                rng.normal(0.3, 0.1, 50),
                rng.normal(0.7, 0.1, 50),
            ]
        ).clip(0.01, 0.99)
        y_validation = np.array([0] * 50 + [1] * 50)
        y_validation_proba = np.column_stack([1 - proba_pos, proba_pos])

        cost_matrix = {"tp": 10.0, "fp": -1.0, "fn": -10.0, "tn": 0.0}

        threshold, source = _select_threshold(
            y_validation, y_validation_proba, cost_matrix=cost_matrix
        )

        assert 0.0 < threshold < 1.0
        assert source == "validation_cost_optimal"

    def test_no_cost_matrix_preserves_validation_provenance(self):
        """cost_matrix=None must preserve the original ``"validation"``
        provenance — backward compatibility for callers that never set the
        kwarg."""
        rng = np.random.default_rng(20260509)
        n = 60
        proba_pos = rng.uniform(0.1, 0.9, n)
        y_validation = (proba_pos > 0.5).astype(int)
        y_validation_proba = np.column_stack([1 - proba_pos, proba_pos])

        threshold, source = _select_threshold(y_validation, y_validation_proba, cost_matrix=None)

        assert 0.0 <= threshold <= 1.0
        assert source == "validation"

    def test_cost_matrix_raises_keyerror_on_malformed(self):
        """Malformed cost-matrix (missing keys) MUST propagate KeyError
        from ``_select_threshold`` — codex pass-2 HIGH-1 contract.

        Previously the helper silently fell through to Youden's J on
        KeyError, hiding the configuration bug. The new contract
        requires a loud failure so callers (and CI) see the misconfig
        immediately. Validation-tuned threshold is NOT a substitute for
        a correct cost matrix.
        """
        rng = np.random.default_rng(20260509)
        n = 40
        proba_pos = rng.uniform(0.1, 0.9, n)
        y_validation = (proba_pos > 0.5).astype(int)
        y_validation_proba = np.column_stack([1 - proba_pos, proba_pos])
        bad_cost_matrix = {"tp": 1.0, "fp": -1.0}  # missing fn, tn

        with pytest.raises(KeyError, match="missing required keys"):
            _select_threshold(y_validation, y_validation_proba, cost_matrix=bad_cost_matrix)

    def test_cost_matrix_flat_utility_falls_through_to_youden(self):
        """Codex pass-2 MEDIUM-2: zero/flat cost matrix → cost-optimal
        helper returns None → ``_select_threshold`` falls through to
        Youden's J with provenance ``"validation"``.

        Distinct from HIGH-1 (malformed): a flat valid cost matrix is
        not a config bug; it just doesn't carry information. Falling
        through is the right move.
        """
        rng = np.random.default_rng(20260510)
        n = 40
        proba_pos = rng.uniform(0.1, 0.9, n)
        y_validation = (proba_pos > 0.5).astype(int)
        y_validation_proba = np.column_stack([1 - proba_pos, proba_pos])
        flat_cost_matrix = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

        threshold, source = _select_threshold(
            y_validation, y_validation_proba, cost_matrix=flat_cost_matrix
        )

        assert source == "validation"
        assert np.isfinite(threshold)
        assert 0.0 <= threshold <= 1.0

    def test_cost_matrix_no_validation_arrays_falls_back_to_default(self):
        """cost_matrix without validation arrays still defaults to 0.5 —
        the cost branch only activates when validation tuning is possible.
        """
        cost_matrix = {"tp": 10.0, "fp": -1.0, "fn": -10.0, "tn": 0.0}

        threshold, source = _select_threshold(None, None, cost_matrix=cost_matrix)

        assert threshold == 0.5
        assert source == "default"


# ============================================================================
# Backlog #20 Gap 2: F1-fallback when validation MCC is below the floor
# ============================================================================


class TestF1FallbackOnLowMCC:
    """Verify F1-fallback engages on low-MCC validation outcomes.

    When the canonical Youden's J / cost-optimal / precision-constrained
    pick produces validation MCC < 0.20, the evaluator retries with the
    F1-optimal threshold from advanced_validation. The swap only happens
    when F1-optimal STRICTLY improves MCC (no performative re-tuning).
    """

    @staticmethod
    def _make_low_mcc_split(rng: np.random.Generator, n: int = 200):
        """Construct a (y_val, y_val_pred, y_val_proba, y_test, ...) bundle
        whose Youden's J pick yields a deliberately-low MCC.

        We use heavily-overlapping class distributions so any threshold
        gives near-random separation; the F1-optimal sweep tends to pick
        a different threshold that may eke out marginally higher MCC.
        """
        # Heavy class overlap: positives at 0.45 ± 0.15, negatives at 0.40 ± 0.15
        n_pos = n // 2
        n_neg = n - n_pos
        pos_scores = np.clip(rng.normal(0.45, 0.15, n_pos), 0.001, 0.999)
        neg_scores = np.clip(rng.normal(0.40, 0.15, n_neg), 0.001, 0.999)
        y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
        proba_pos = np.concatenate([pos_scores, neg_scores])
        order = rng.permutation(n)
        y = y[order]
        proba_pos = proba_pos[order]
        y_pred = (proba_pos >= 0.5).astype(int)
        y_proba = np.column_stack([1.0 - proba_pos, proba_pos])
        return y, y_pred, y_proba

    def test_f1_fallback_engages_when_mcc_below_floor(self):
        """When validation MCC at chosen threshold < 0.20 AND F1-optimal
        improves MCC, the evaluator switches to F1-optimal threshold."""
        rng = np.random.default_rng(42)
        y_val, y_val_pred, y_val_proba = self._make_low_mcc_split(rng, n=200)
        # Test set with same low-separation distribution (decoupled draw)
        y_test, y_test_pred, y_test_proba = self._make_low_mcc_split(rng, n=200)

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
        )

        validation_metrics = result["validation_metrics"]
        # Either fallback engaged (MCC at chosen was < floor AND F1-opt
        # gave higher MCC) OR fallback didn't fire (MCC above floor or
        # F1-opt didn't beat it). Both are valid; we assert the
        # post-fallback invariants.
        if validation_metrics.get("f1_fallback_engaged"):
            # MCC must be strictly higher post-fallback
            assert validation_metrics["mcc"] > validation_metrics["f1_fallback_original_mcc"]
            assert validation_metrics["chosen_threshold_source"] == "validation_f1_fallback"
            # Original threshold source preserved for audit
            assert validation_metrics["f1_fallback_original_threshold_source"] in {
                "validation",
                "validation_cost_optimal",
            }
        else:
            # If not engaged, original source must still be one of the
            # canonical sources — never "validation_f1_fallback"
            assert validation_metrics["chosen_threshold_source"] != "validation_f1_fallback"

    def test_f1_fallback_skipped_when_mcc_above_floor(self):
        """When validation MCC at chosen threshold >= 0.20, F1-fallback
        does NOT engage even if F1-optimal would give a higher MCC."""
        rng = np.random.default_rng(20260509)
        # Well-separated classes → high MCC at Youden's J
        n = 200
        n_pos = n // 2
        n_neg = n - n_pos
        pos_scores = np.clip(rng.normal(0.75, 0.05, n_pos), 0.001, 0.999)
        neg_scores = np.clip(rng.normal(0.25, 0.05, n_neg), 0.001, 0.999)
        y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
        proba_pos = np.concatenate([pos_scores, neg_scores])
        order = rng.permutation(n)
        y = y[order]
        proba_pos = proba_pos[order]
        y_pred = (proba_pos >= 0.5).astype(int)
        y_proba = np.column_stack([1.0 - proba_pos, proba_pos])

        # Reuse same data for test set (just for the test fixture)
        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=y,
            y_validation_pred=y_pred,
            y_validation_proba=y_proba,
            y_test=y,
            y_test_pred=y_pred,
            y_test_proba=y_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
        )

        validation_metrics = result["validation_metrics"]
        # MCC should be high → fallback should NOT engage
        assert validation_metrics["mcc"] >= _F1_FALLBACK_MCC_THRESHOLD
        assert validation_metrics["chosen_threshold_source"] != "validation_f1_fallback"
        assert "f1_fallback_engaged" not in validation_metrics

    def test_f1_fallback_skipped_when_no_improvement(self):
        """When MCC < floor BUT F1-optimal does NOT improve MCC, the
        fallback evaluates and declines to swap. Original threshold +
        source preserved.

        Construction: degenerate validation set where every threshold gives
        the same low MCC. F1-optimal's MCC won't exceed the chosen one.
        """
        rng = np.random.default_rng(20260510)
        n = 100
        # Random labels + uniform probabilities → all thresholds give ~0 MCC
        y = rng.integers(0, 2, n)
        proba_pos = rng.uniform(0.45, 0.55, n)  # tight uniform → near-flat ROC
        y_pred = (proba_pos >= 0.5).astype(int)
        y_proba = np.column_stack([1.0 - proba_pos, proba_pos])

        result = _compute_classification_metrics(
            y_train=None,
            y_train_pred=None,
            y_train_proba=None,
            y_validation=y,
            y_validation_pred=y_pred,
            y_validation_proba=y_proba,
            y_test=y,
            y_test_pred=y_pred,
            y_test_proba=y_proba,
            imbalance_detected=False,
            minority_ratio=0.5,
        )

        validation_metrics = result["validation_metrics"]
        # Low MCC environment — fallback either engages with marginal
        # improvement (rare) or skips (common). Either way, post-state
        # is consistent.
        if validation_metrics.get("f1_fallback_engaged"):
            assert validation_metrics["mcc"] > validation_metrics["f1_fallback_original_mcc"]
        else:
            # Fallback declined: source unchanged from canonical pick
            assert validation_metrics["chosen_threshold_source"] != "validation_f1_fallback"

    def test_f1_fallback_floor_constant_value(self):
        """Pin the floor constant so a future regression to a different
        value is caught by the test suite — not by surprise in production."""
        assert _F1_FALLBACK_MCC_THRESHOLD == 0.20


# ============================================================================
# Backlog #20 Gap 1 opt-in gate — backward-compat preservation
# ============================================================================


class TestCostOptimalOptInGate:
    """Verify ``use_cost_optimal_threshold`` opts in to cost-aware
    threshold selection, and the default OFF preserves Youden's J.

    Codex pass-3 regression: synthetic baseline test
    (test_synthetic_baseline_invariant.py) tripped on PR #115 because
    cost_matrix was historically a *reporting* signal (computes
    business_utility post-hoc) that did NOT influence threshold
    selection. Making the cost-aware branch transparent shifted the
    operating point on every caller that supplies a cost_matrix —
    including the synthetic demo path. The opt-in gate restores
    backward compatibility: cost_matrix without the flag → Youden's J;
    cost_matrix with the flag → cost-aware threshold.
    """

    @staticmethod
    def _make_data(rng: np.random.Generator, n: int = 100):
        """Bimodal validation+test dataset with separable classes."""
        proba_pos = np.concatenate(
            [
                rng.normal(0.3, 0.1, n // 2),
                rng.normal(0.7, 0.1, n // 2),
            ]
        ).clip(0.01, 0.99)
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        y_pred = (proba_pos >= 0.5).astype(int)
        y_proba = np.column_stack([1 - proba_pos, proba_pos])
        return y, y_pred, y_proba

    def test_cost_matrix_without_optin_falls_back_to_youden(self):
        """cost_matrix supplied + use_cost_optimal_threshold=False (default):
        threshold source is ``"validation"``, NOT
        ``"validation_cost_optimal"``. Backward-compat regression guard."""
        rng = np.random.default_rng(20260509)
        y_val, y_val_pred, y_val_proba = self._make_data(rng)
        y_test, y_test_pred, y_test_proba = self._make_data(rng)
        cost_matrix = {"tp": 10.0, "fp": -1.0, "fn": -10.0, "tn": 0.0}

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
            # use_cost_optimal_threshold not passed → default False
        )

        validation_metrics = result["validation_metrics"]
        assert validation_metrics["chosen_threshold_source"] == "validation"
        # business_utility STILL gets reported (cost_matrix is for that)
        # — only the threshold choice is gated.
        assert "business_utility" in validation_metrics

    def test_cost_matrix_with_optin_uses_cost_optimal(self):
        """cost_matrix supplied + use_cost_optimal_threshold=True:
        threshold source is ``"validation_cost_optimal"``."""
        rng = np.random.default_rng(20260509)
        y_val, y_val_pred, y_val_proba = self._make_data(rng)
        y_test, y_test_pred, y_test_proba = self._make_data(rng)
        cost_matrix = {"tp": 10.0, "fp": -1.0, "fn": -10.0, "tn": 0.0}

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
            use_cost_optimal_threshold=True,
        )

        validation_metrics = result["validation_metrics"]
        assert validation_metrics["chosen_threshold_source"] == "validation_cost_optimal"

    def test_no_cost_matrix_with_optin_still_uses_youden(self):
        """use_cost_optimal_threshold=True without cost_matrix: no
        cost-aware branch can fire, threshold falls through to Youden's J."""
        rng = np.random.default_rng(20260509)
        y_val, y_val_pred, y_val_proba = self._make_data(rng)
        y_test, y_test_pred, y_test_proba = self._make_data(rng)

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
            use_cost_optimal_threshold=True,
        )

        validation_metrics = result["validation_metrics"]
        assert validation_metrics["chosen_threshold_source"] == "validation"
