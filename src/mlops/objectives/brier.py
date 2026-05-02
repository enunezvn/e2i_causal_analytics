"""Brier-score custom objective for gradient-boosted binary classifiers.

LightGBM and XGBoost expose `objective=callable` hooks that compute
gradient + hessian per training example from the current raw-score
predictions (logits, NOT probabilities — the booster does not apply
sigmoid internally for custom objectives; per LightGBM 4.6 docs §Custom
Objective the callback receives `preds` as raw margins). This module
implements the chain-rule form of the Brier loss under a sigmoid link:

  L(z, y) = (sigmoid(z) - y)^2,   p := sigmoid(z),   y in {0, 1}

  dL/dz = 2 (p - y) · p · (1 - p)

  d²L/dz² (exact) = 2 · [p(1-p)]² + 2 (p - y)(1 - 2p) · p(1 - p)
  d²L/dz² ≈ 2 · [p(1-p)]²   (Newton positive-definite diagonal approx;
                              drops the sign-indeterminate (1-2p)(p-y)
                              term to keep curvature ≥ 0 for all z, y)

Reference: shard 17 Week 3 Day 1 of
`.claude/plans/adaptive_criteria_v3_followup/`. The shard's brief
specification ("gradient = 2(p - y), hessian = 2p(1-p)") is the
probability-scale derivative pair: `dL/dp = 2(p-y)` and the LOGLOSS
hessian `p(1-p)` (which is NOT the Brier hessian). Both forms apply
only if the callee passed `sigmoid(z)`, which LightGBM's custom
objective does NOT.

Amendments vs shard 17 Day 1 verbatim (codex cycle 12, 2026-05-02):
1. **Gradient**: code uses chain-rule logit-scale `2(p-y)·p(1-p)`
   instead of the shard's literal `2(p-y)`. Without the chain factor,
   training diverges (gradient in wrong space).
2. **Hessian**: code uses Newton-PD diagonal of the Brier loss on
   logit scale `2·[p(1-p)]²` (max 1/8 at p=0.5), instead of the
   shard's literal `2p(1-p)` (max 1/2 at p=0.5; this is the LOGLOSS
   hessian, not Brier). The shard formula overestimates Brier
   curvature by `1/(p(1-p))` — exactly 4× at p=0.5 — shrinking
   Newton leaf updates `−grad/hess` by 4× and starving convergence.
   Confirmed by pre-fix smoke test failure: Brier-trained
   `brier_loss = 0.0765` vs logloss baseline `0.0099` (7.7× worse on
   Brier's own loss). Both amendments propose corresponding shard-17
   text revisions (USER-DECISION REQUIRED — see cycle_12_verdict.md).
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid for both positive and negative inputs."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def brier_objective_lightgbm(
    arg1: Any,
    arg2: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """LightGBM custom-objective callback for Brier-score training.

    Supports BOTH LightGBM calling conventions:
      - Sklearn API (``LGBMClassifier(objective=callable)``): callee invokes
        as ``func(y_true, y_pred)`` — labels first, raw scores second.
      - Native API (``lgb.train(params={'objective': callable}, ...)``):
        callee invokes as ``func(y_pred, dataset)`` — raw scores first,
        Dataset object second (with ``.get_label()`` method).

    The dispatch is duck-typed: if ``arg2`` has ``.get_label``, native API;
    otherwise sklearn API.

    Returns:
        Tuple (gradient, hessian) each shape (n_samples,) and dtype float64.
        Gradient is dL/dz; hessian is the Newton-diagonal approximation
        (positive-definite for any p in [0, 1]).
    """
    if hasattr(arg2, "get_label"):
        # Native LightGBM: (y_pred, dataset)
        z = np.asarray(arg1, dtype=np.float64)
        y = np.asarray(arg2.get_label(), dtype=np.float64)
    else:
        # Sklearn API: (y_true, y_pred)
        y = np.asarray(arg1, dtype=np.float64)
        z = np.asarray(arg2, dtype=np.float64)
    p = _sigmoid(z)
    p1mp = p * (1.0 - p)
    grad = 2.0 * (p - y) * p1mp
    hess = 2.0 * p1mp * p1mp
    return grad, hess
