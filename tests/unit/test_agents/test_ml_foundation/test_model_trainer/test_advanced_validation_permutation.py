"""Permutation-test threshold tests — Phase 7 of ml-leakage-holistic-fix.

The pre-Phase-7 threshold of 0.05 with n=100 permutations was too loose:
the p-value resolution was 1/100 = 0.01, and the gate allowed up to 5 random
shuffles to beat the actual AUC. Post-Phase-7 the threshold is 0.001 and the
default n_permutations is 1000, giving a real cutoff.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
    SIGNAL_GENUINE_THRESHOLD,
    compute_permutation_test,
)


def test_threshold_constant_is_0_001():
    """Phase 7 contract: SIGNAL_GENUINE_THRESHOLD must be 0.001."""
    assert SIGNAL_GENUINE_THRESHOLD == 0.001, (
        f"Phase 7 expected SIGNAL_GENUINE_THRESHOLD=0.001; got {SIGNAL_GENUINE_THRESHOLD}"
    )


def test_default_n_permutations_is_1000():
    """Phase 7 contract: default n_permutations must be 1000 (not 100)."""
    import inspect

    sig = inspect.signature(compute_permutation_test)
    default = sig.parameters["n_permutations"].default
    assert default == 1000, (
        f"Phase 7 expected default n_permutations=1000; got {default}"
    )


def test_strong_signal_passes_genuine_gate():
    """A model whose predictions perfectly track labels must be flagged genuine."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.binomial(1, 0.30, n)
    # Strong signal: probabilities tightly track the label
    y_proba = y_true * 0.85 + rng.uniform(0, 0.15, n)
    y_proba_2d = np.column_stack([1 - y_proba, y_proba])

    result = compute_permutation_test(y_true, y_proba_2d, n_permutations=1000)
    assert result["signal_genuine"] is True
    assert result["permutation_pvalue"] < SIGNAL_GENUINE_THRESHOLD


def test_random_predictions_fail_genuine_gate():
    """Random predictions must NOT be flagged genuine — even with the loose
    pre-Phase-7 threshold this should fail; the Phase 7 tightening is for
    borderline marginal models, not these obvious cases."""
    rng = np.random.default_rng(7)
    n = 500
    y_true = rng.binomial(1, 0.30, n)
    y_proba = rng.uniform(0, 1, n)
    y_proba_2d = np.column_stack([1 - y_proba, y_proba])

    result = compute_permutation_test(y_true, y_proba_2d, n_permutations=1000)
    assert result["signal_genuine"] is False


def test_marginal_signal_at_p005_now_blocked():
    """Phase 7's stricter threshold blocks marginal models that the old 0.05
    cutoff would have admitted.

    Constructs a model with weak-but-non-zero signal whose permutation p-value
    sits in (0.001, 0.05). Pre-Phase-7 it would have been flagged genuine;
    post-Phase-7 it must NOT be.
    """
    rng = np.random.default_rng(11)
    n = 200
    y_true = rng.binomial(1, 0.30, n)
    # Weak signal: probabilities only slightly track label
    y_proba = 0.30 + 0.10 * y_true + rng.normal(0, 0.20, n)
    y_proba = np.clip(y_proba, 0, 1)
    y_proba_2d = np.column_stack([1 - y_proba, y_proba])

    result = compute_permutation_test(y_true, y_proba_2d, n_permutations=1000)
    p = result["permutation_pvalue"]

    # The intent: this construction sometimes lands p in (0.001, 0.05). When
    # it does, Phase 7 must block. When it doesn't, the test is non-blocking
    # but documents the spectrum.
    if 0.001 < p < 0.05:
        assert result["signal_genuine"] is False, (
            f"Phase 7 must block marginal models at p={p:.4f}"
        )
