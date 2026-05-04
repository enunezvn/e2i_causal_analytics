"""Brier-score custom objective for gradient-boosted binary classifiers.

LightGBM and XGBoost expose `objective=callable` hooks that compute
gradient + hessian per training example from the current raw-score
predictions (logits, NOT probabilities — the booster does not apply
sigmoid internally for custom objectives; per LightGBM 4.6 docs §Custom
Objective and XGBoost 3.x §Customized Objective the callback receives
raw margins, not probabilities). This module implements the chain-rule
form of the Brier loss under a sigmoid link:

  L(z, y) = (sigmoid(z) - y)^2,   p := sigmoid(z),   y in {0, 1}

  dL/dz = 2 (p - y) · p · (1 - p)

  d²L/dz² (exact) = 2 · [p(1-p)]² + 2 (p - y)(1 - 2p) · p(1 - p)
  d²L/dz² ≈ 2 · [p(1-p)]²   (Newton positive-definite diagonal approx;
                              drops the sign-indeterminate (1-2p)(p-y)
                              term to keep curvature ≥ 0 for all z, y)

Reference: shard 17 Week 3 Days 1+2 of
`.claude/plans/adaptive_criteria_v3_followup/`. The shard's brief
specification ("gradient = 2(p - y), hessian = 2p(1-p)") is the
probability-scale derivative pair: `dL/dp = 2(p-y)` and the LOGLOSS
hessian `p(1-p)` (which is NOT the Brier hessian). Both forms apply
only if the callee passed `sigmoid(z)`, which neither LightGBM's nor
XGBoost's custom-objective callback does.

Amendments vs shard 17 Days 1+2 verbatim (codex cycle 12, 2026-05-02;
shard 17 footnote `[1]` documents both):
1. **Gradient**: code uses chain-rule logit-scale `2(p-y)·p(1-p)`
   instead of the shard's literal `2(p-y)`. Without the chain factor,
   training diverges (gradient in wrong space).
2. **Hessian**: code uses Newton-PD diagonal of the Brier loss on
   logit scale `2·[p(1-p)]²` (max 1/8 at p=0.5), instead of the
   shard's literal `2p(1-p)` (max 1/2 at p=0.5; this is the LOGLOSS
   hessian, not Brier). The shard formula overestimates Brier
   curvature by `1/(p(1-p))` — exactly 4× at p=0.5 — shrinking
   Newton leaf updates `−grad/hess` by 4× and starving convergence.
   Pre-fix LightGBM smoke evidence: `brier_loss = 0.0765` vs logloss
   baseline `0.0099` (7.7× worse on Brier's own loss). Post-fix
   passes within 0.005 of baseline.

Day-2 (XGBoost) reuses the same math verbatim per shard 17 footnote
`[1]`. Both `XGBClassifier(objective=callable)` (sklearn API,
`obj(y_true, y_pred)`) and `xgboost.train(..., obj=callable)` (native
API, `obj(y_pred, dtrain)`) follow the same arg-shape convention as
LightGBM, and the duck-typed dispatch on ``hasattr(arg2, 'get_label')``
correctly discriminates DMatrix (native) from ndarray (sklearn).
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


def _brier_grad_hess_from_raw_scores(
    arg1: Any,
    arg2: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared core for LightGBM + XGBoost custom Brier objective.

    Both libraries' `objective=callable` callbacks pass raw logits z (not
    probabilities) and labels y. The two libraries differ only in arg
    order across (sklearn vs native) APIs, both of which match each other:

      - Sklearn API (``LGBMClassifier``/``XGBClassifier(objective=fn)``):
        ``fn(y_true, y_pred)`` — labels first, raw margins second.
      - Native API (``lgb.train``, ``xgb.train(..., obj=fn)``):
        ``fn(y_pred, dataset_or_dmatrix)`` — raw margins first, dataset
        object second (with ``.get_label()``).

    Dispatch is duck-typed: ``hasattr(arg2, 'get_label')`` ⇒ native API.

    Returns:
        Tuple (gradient, hessian) each shape (n_samples,) and dtype
        float64. Gradient is dL/dz (chain-rule logit-scale). Hessian is
        the Newton-PD diagonal approximation `2·[p(1-p)]²` (always
        positive for p ∈ (0, 1)).
    """
    if hasattr(arg2, "get_label"):
        # Native API: (y_pred, dataset/dmatrix)
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


def brier_objective_lightgbm(
    arg1: Any,
    arg2: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """LightGBM custom-objective callback for Brier-score training.

    Thin wrapper over ``_brier_grad_hess_from_raw_scores``; see that
    helper's docstring for math + API conventions. Naming is
    framework-specific so registry lookups + framework-map entries can
    select the appropriate symbol per booster.
    """
    return _brier_grad_hess_from_raw_scores(arg1, arg2)


def brier_objective_xgboost(
    arg1: Any,
    arg2: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """XGBoost custom-objective callback for Brier-score training.

    Identical math + dispatch to ``brier_objective_lightgbm``; both
    boosters' custom-objective contracts (sklearn `obj(y_true, y_pred)`,
    native `obj(y_pred, dmatrix)`) follow the same arg conventions, and
    the chain-rule + Newton-PD-Brier-hessian derivation applies verbatim
    (shard 17 Days 1+2, footnote `[1]`).

    Convergence note: Brier-trained XGBoost reaches the same Brier-loss
    neighborhood ~4× more slowly than logloss-trained XGBoost (smoke
    tests use ``n_estimators=1000, learning_rate=0.10`` vs LightGBM
    smoke's ``n_estimators=300, learning_rate=0.05``). The dominant
    cause is **gradient-magnitude scaling**: the Brier gradient
    ``2(p-y)·p(1-p)`` has maximum magnitude 0.125 (at p=0.5), while
    the logloss gradient ``p-y`` has maximum magnitude 0.5 — a 4×
    deficit per leaf step that requires ~4× more iterations to
    accumulate equivalent cumulative leaf scores. ``reg_lambda=1.0``
    (XGBoost default vs LightGBM's 0) is a SECONDARY factor;
    empirical sweep 2026-05-02 (codex cycle 13) confirmed setting
    ``reg_lambda=0`` at ``n=300, lr=0.05`` does NOT close the gap
    (Brier delta = +0.0123 vs λ=1 delta = +0.0100; in fact slightly
    worse). At ``n=1000, lr=0.10`` with default ``reg_lambda=1.0``,
    Brier delta = +0.0046 (passes ≤+0.005 threshold). This is a
    hyperparameter-budget artifact of the gradient-magnitude
    asymmetry, not a math bug — the shared chain-rule +
    Newton-PD-Brier math is identical to the LightGBM path.

    ``predict_proba`` semantics: ``XGBClassifier`` with
    ``objective=callable`` returns sigmoid-applied 2D ``(n, 2)``
    probabilities — XGBoost applies the binary-link transform
    internally even when a custom objective is set. Downstream
    callers should consume ``predict_proba(X)[:, 1]`` directly
    WITHOUT post-processing sigmoid. This contrasts with
    ``LGBMClassifier``, which returns RAW logit scores from
    ``predict_proba`` when ``objective=callable`` (LightGBM warning:
    "Cannot compute class probabilities... Returning raw scores
    instead.") and requires manual ``_sigmoid`` application before
    Brier-loss / log-loss computation.
    """
    return _brier_grad_hess_from_raw_scores(arg1, arg2)
