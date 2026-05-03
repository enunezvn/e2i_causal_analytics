"""Cycle-16 I-4 (Q2-B): bootstrap CI determinism under per-fold seeding.

Phase 1 W3-lite ships ``asyncio.gather`` with ``Semaphore(n_jobs)`` for fold
concurrency in ``ModelTrainerAgent._run_repeated_splits``. Per-fold model fit
seeds are threaded via ``state["fold_random_state"]`` (Day-3 work). Cycle-16
codex review flagged that ``evaluator._compute_bootstrap_ci`` uses
``np.random.choice`` against numpy's GLOBAL RNG, so two folds running
concurrently under ``n_jobs > 1`` would produce non-deterministic bootstrap CI
endpoints because they would interleave reads of the shared global RNG state.

This test file locks the contract that:
  1. ``_compute_bootstrap_ci`` accepts an optional ``random_state`` kwarg and
     produces bit-identical CI endpoints when called twice with the same seed.
  2. Different seeds produce different CI endpoints (no accidental no-op).
  3. The legacy ``random_state=None`` path is unchanged (uses global RNG —
     preserves byte-identity for callers that don't supply a fold seed,
     e.g., single-mode evaluation).
  4. ``evaluate_model`` threads ``state.get("fold_random_state")`` into the
     bootstrap CI helper so that repeated_k10 fold N's bootstrap CI is
     deterministic across runs regardless of execution order.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _compute_bootstrap_ci,
    evaluate_model,
)


def _make_classification_arrays(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(0.0, 1.0, size=(n, 2))
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    return y_true, y_pred, y_proba


class TestBootstrapCiDeterminismKwarg:
    """Direct unit tests on ``_compute_bootstrap_ci``."""

    def test_same_random_state_produces_identical_ci(self) -> None:
        y_true, y_pred, y_proba = _make_classification_arrays(seed=0)
        ci_a, n_a = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=200, random_state=42,
        )
        ci_b, n_b = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=200, random_state=42,
        )
        assert n_a == n_b == 200
        assert ci_a.keys() == ci_b.keys()
        for metric in ci_a:
            lo_a, hi_a = ci_a[metric]
            lo_b, hi_b = ci_b[metric]
            assert lo_a == lo_b, f"{metric}: lo {lo_a} != {lo_b}"
            assert hi_a == hi_b, f"{metric}: hi {hi_a} != {hi_b}"

    def test_different_random_state_produces_different_ci(self) -> None:
        y_true, y_pred, y_proba = _make_classification_arrays(seed=0)
        ci_42, _ = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=200, random_state=42,
        )
        ci_43, _ = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=200, random_state=43,
        )
        # At least one metric's CI endpoint must differ — bootstrap CIs are
        # noisy across seeds, so identical CIs would indicate the seed is
        # silently ignored
        any_diff = any(
            ci_42[m][0] != ci_43[m][0] or ci_42[m][1] != ci_43[m][1]
            for m in ci_42
        )
        assert any_diff, "random_state appears ignored — CI endpoints identical across seeds"

    def test_none_random_state_uses_global_rng_no_error(self) -> None:
        """Backward-compat: omitting ``random_state`` preserves legacy global-RNG path."""
        y_true, y_pred, y_proba = _make_classification_arrays(seed=0)
        np.random.seed(123)
        ci_a, _ = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=200,
        )
        # Re-seed global, repeat — should produce identical CIs (global RNG
        # is reproducible when seeded)
        np.random.seed(123)
        ci_b, _ = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=200,
        )
        assert ci_a.keys() == ci_b.keys()
        for metric in ci_a:
            assert ci_a[metric] == ci_b[metric], (
                f"global-RNG path not reproducible under explicit np.random.seed for {metric}"
            )

    def test_concurrency_simulation_per_fold_seed_isolates_results(self) -> None:
        """Simulate ``n_jobs=2`` interleaving: ensure per-fold seeds isolate results.

        Without explicit seed threading, two interleaved bootstrap CI calls
        share numpy's global RNG and produce different CIs depending on order
        of execution. With per-fold seeds, each fold's CI is fully determined
        by its seed regardless of what other folds are doing.
        """
        y_true, y_pred, y_proba = _make_classification_arrays(seed=0)
        # Fold 0 in isolation
        ci0_solo, _ = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=100, random_state=100,
        )
        # Now interleave fold 0 with a "fold 1" call in between
        _ = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=100, random_state=200,
        )
        ci0_after_interleave, _ = _compute_bootstrap_ci(
            y_true, y_pred, y_proba, problem_type="binary_classification",
            n_bootstrap=100, random_state=100,
        )
        # Fold 0's CI must be identical regardless of intervening fold 1 call
        for metric in ci0_solo:
            assert ci0_solo[metric] == ci0_after_interleave[metric], (
                f"per-fold seed not isolating: {metric} drifted under interleave"
            )


@pytest.mark.asyncio
class TestEvaluateModelThreadsFoldSeed:
    """Integration test: ``evaluate_model`` threads ``fold_random_state``."""

    async def _make_state(self, fold_random_state: Any) -> Dict[str, Any]:
        seed = 0
        rng = np.random.default_rng(seed)
        n = 100
        X_train = rng.standard_normal((n, 4))
        y_train = rng.integers(0, 2, size=n)
        X_val = rng.standard_normal((n, 4))
        y_val = rng.integers(0, 2, size=n)
        X_test = rng.standard_normal((n, 4))
        y_test = rng.integers(0, 2, size=n)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        state: Dict[str, Any] = {
            "trained_model": model,
            "problem_type": "binary_classification",
            "X_train_preprocessed": X_train,
            "X_validation_preprocessed": X_val,
            "X_test_preprocessed": X_test,
            "train_data": {"y": y_train},
            "validation_data": {"y": y_val},
            "test_data": {"y": y_test},
            "success_criteria": {},
        }
        if fold_random_state is not None:
            state["fold_random_state"] = fold_random_state
        return state

    async def test_evaluate_model_with_fold_random_state_is_deterministic(self) -> None:
        state_a = await self._make_state(fold_random_state=42)
        state_b = await self._make_state(fold_random_state=42)
        result_a = await evaluate_model(state_a)
        result_b = await evaluate_model(state_b)
        # Confidence intervals must be bit-identical when fold_random_state matches
        ci_a = result_a.get("confidence_interval", {})
        ci_b = result_b.get("confidence_interval", {})
        assert ci_a == ci_b, "fold_random_state did not produce deterministic CIs"

    async def test_evaluate_model_different_fold_random_state_differs(self) -> None:
        state_a = await self._make_state(fold_random_state=42)
        state_b = await self._make_state(fold_random_state=43)
        result_a = await evaluate_model(state_a)
        result_b = await evaluate_model(state_b)
        ci_a = result_a.get("confidence_interval", {})
        ci_b = result_b.get("confidence_interval", {})
        assert ci_a.keys() == ci_b.keys() and len(ci_a) > 0, (
            "no CIs computed — test setup invalid"
        )
        any_diff = any(ci_a[m] != ci_b[m] for m in ci_a)
        assert any_diff, "different fold_random_state produced identical CIs (seed not threaded)"
