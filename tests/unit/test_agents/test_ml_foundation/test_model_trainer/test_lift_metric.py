"""Tests for ``minimum_lift_over_baseline`` (Section B unblocker).

The criterion is computed by ``_compute_baseline_test_metrics`` and
exposed in ``test_metrics`` as ``minimum_lift_over_baseline`` (absolute
AUC lift over a stratified-dummy baseline). The plan recommendation is
to enforce the criterion as a hard fail when the model AUC is missing
under "normal" conditions but to soft-skip the criterion specifically
when the dummy baseline cannot be computed (degenerate splits) — this
narrow exemption is gated on the criterion name and validated below.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _check_success_criteria,
    _compute_baseline_test_metrics,
)


def _split_indices(n: int, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * train_ratio)
    return idx[:cut], idx[cut:]


class TestComputeBaselineTestMetrics:
    """Direct unit tests for the baseline-AUC helper."""

    def test_returns_baseline_auc_for_signal_data(self) -> None:
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            weights=[0.85, 0.15],
            random_state=42,
        )
        train, test = _split_indices(len(X))
        result = _compute_baseline_test_metrics(
            y_train=y[train], y_test=y[test], problem_type="binary_classification"
        )
        # Stratified dummy on a slightly imbalanced split lands near 0.50.
        assert "baseline_test_auc" in result
        assert 0.30 <= result["baseline_test_auc"] <= 0.70

    def test_skipped_for_non_binary_problem(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2] * 10)
        result = _compute_baseline_test_metrics(
            y_train=y, y_test=y, problem_type="multiclass_classification"
        )
        assert result == {}

    def test_skipped_when_train_too_small(self) -> None:
        y_train = np.array([0, 1, 0])  # n=3 < 10
        y_test = np.array([0, 1] * 10)
        result = _compute_baseline_test_metrics(
            y_train=y_train, y_test=y_test, problem_type="binary_classification"
        )
        assert result == {}

    def test_skipped_when_test_too_small(self) -> None:
        y_train = np.array([0, 1] * 10)
        y_test = np.array([0, 1, 0, 1, 0])  # n=5 < 10
        result = _compute_baseline_test_metrics(
            y_train=y_train, y_test=y_test, problem_type="binary_classification"
        )
        assert result == {}

    def test_skipped_when_train_single_class(self) -> None:
        y_train = np.zeros(50, dtype=int)  # all zeros
        y_test = np.array([0, 1] * 10)
        result = _compute_baseline_test_metrics(
            y_train=y_train, y_test=y_test, problem_type="binary_classification"
        )
        assert result == {}

    def test_skipped_when_test_single_class(self) -> None:
        y_train = np.array([0, 1] * 25)
        y_test = np.zeros(20, dtype=int)
        result = _compute_baseline_test_metrics(
            y_train=y_train, y_test=y_test, problem_type="binary_classification"
        )
        assert result == {}


def _build_test_metrics(roc_auc: Optional[float], baseline_auc: Optional[float]) -> Dict[str, Any]:
    """Construct a minimal test_metrics dict for criterion-check tests."""
    metrics: Dict[str, Any] = {"accuracy": 0.80}
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    if baseline_auc is not None:
        metrics["baseline_test_auc"] = baseline_auc
        if roc_auc is not None:
            metrics["minimum_lift_over_baseline"] = float(roc_auc - baseline_auc)
    return metrics


class TestLiftCriterionAggregation:
    """``_check_success_criteria`` behavior on the new criterion."""

    def test_lift_passes_when_above_threshold(self) -> None:
        metrics = _build_test_metrics(roc_auc=0.78, baseline_auc=0.50)  # lift=0.28
        result = _check_success_criteria(
            test_metrics=metrics,
            success_criteria={"minimum_lift_over_baseline": 0.10},
            problem_type="binary_classification",
        )
        assert result["success_criteria_met"] is True
        assert result["success_criteria_results"]["minimum_lift_over_baseline"] is True

    def test_lift_fails_when_below_threshold(self) -> None:
        metrics = _build_test_metrics(roc_auc=0.55, baseline_auc=0.50)  # lift=0.05
        result = _check_success_criteria(
            test_metrics=metrics,
            success_criteria={"minimum_lift_over_baseline": 0.10},
            problem_type="binary_classification",
        )
        assert result["success_criteria_met"] is False
        assert result["success_criteria_results"]["minimum_lift_over_baseline"] is False

    def test_lift_soft_skipped_when_metric_missing(self) -> None:
        """Narrow exemption: missing lift metric → soft-skip, not hard-fail."""
        metrics = _build_test_metrics(roc_auc=0.80, baseline_auc=None)
        result = _check_success_criteria(
            test_metrics=metrics,
            success_criteria={"minimum_lift_over_baseline": 0.10},
            problem_type="binary_classification",
        )
        assert result["success_criteria_met"] is True  # soft-skip preserves all-met
        assert result["success_criteria_results"]["minimum_lift_over_baseline"] is None

    def test_other_criteria_still_hard_fail_when_missing(self) -> None:
        """Sanity check: only minimum_lift_over_baseline gets the exemption.

        Section B's narrow exemption must not drift to other default-injected
        criteria — silently passing on a missing metric would mask gaps.
        """
        metrics = {"accuracy": 0.80}  # no roc_auc / minimum_auc / lift
        result = _check_success_criteria(
            test_metrics=metrics,
            success_criteria={"minimum_auc": 0.75, "minimum_lift_over_baseline": 0.10},
            problem_type="binary_classification",
        )
        assert result["success_criteria_met"] is False
        assert result["success_criteria_results"]["minimum_auc"] is False
        assert result["success_criteria_results"]["minimum_lift_over_baseline"] is None

    def test_lift_combined_with_other_criteria(self) -> None:
        metrics = _build_test_metrics(roc_auc=0.85, baseline_auc=0.50)
        metrics["accuracy"] = 0.92
        result = _check_success_criteria(
            test_metrics=metrics,
            success_criteria={
                "minimum_auc": 0.80,
                "minimum_lift_over_baseline": 0.10,
                "accuracy": 0.90,
            },
            problem_type="binary_classification",
        )
        assert result["success_criteria_met"] is True
        # Each criterion participates separately.
        assert set(result["success_criteria_results"]) >= {
            "minimum_auc",
            "minimum_lift_over_baseline",
            "accuracy",
        }


class TestLiftEndToEndOnSignalData:
    """Lift is large for an actual classifier vs the stratified-dummy reference."""

    def test_lift_above_zero_when_model_has_signal(self) -> None:
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=4,
            weights=[0.7, 0.3],
            random_state=42,
        )
        train, test = _split_indices(len(X))
        model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X[train], y[train])
        from sklearn.metrics import roc_auc_score

        proba = model.predict_proba(X[test])[:, 1]
        model_auc = float(roc_auc_score(y[test], proba))
        baseline = _compute_baseline_test_metrics(
            y_train=y[train], y_test=y[test], problem_type="binary_classification"
        )
        assert "baseline_test_auc" in baseline
        lift = model_auc - baseline["baseline_test_auc"]
        # On a clearly-learnable distribution the lift comfortably clears
        # the 0.10 threshold; tolerate sklearn minor-version variance.
        assert lift > 0.10, (
            f"expected lift > 0.10; got model_auc={model_auc:.4f}, "
            f"baseline={baseline['baseline_test_auc']:.4f}, lift={lift:.4f}"
        )

    def test_lift_near_zero_when_labels_random(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 10))
        y = rng.integers(0, 2, size=300)
        train, test = _split_indices(len(X))
        model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X[train], y[train])
        from sklearn.metrics import roc_auc_score

        proba = model.predict_proba(X[test])[:, 1]
        model_auc = float(roc_auc_score(y[test], proba))
        baseline = _compute_baseline_test_metrics(
            y_train=y[train], y_test=y[test], problem_type="binary_classification"
        )
        lift = model_auc - baseline["baseline_test_auc"]
        # On pure-noise labels both should hover near 0.50 — lift well under
        # the 0.10 threshold proves the criterion meaningfully discriminates.
        assert abs(lift) < 0.10, f"expected |lift| < 0.10 on random labels; got {lift:.4f}"
