"""Plan v3 §4 T2.5 — Stacking/ensemble baseline.

Constrained ensemble (soft voting or rank averaging) with nested CV that
provides a robustness check on the best-single-model selection. Per plan
§6 T2.5 acceptance:

    "Nested-CV-validated ensemble beats best-single-model OR documents
     why not (rejection acceptable)."

The helper does NOT replace the existing single-model selection; it
runs alongside as an audit. Codex flagged the absence (plan codex
critique #13) as "oddly missing" — the platform already trains 4
algorithms during model_selector benchmarking, so a calibrated
soft-voting or rank-averaging ensemble across them is a cheap
robustness check before any heavyweight Tier 3 modeling fires.

Plan §9 file: this module.

Two ensemble methods supported:

  * ``"soft_voting"``: arithmetic mean of base-estimator
    ``predict_proba[:, 1]`` per sample. Equivalent to sklearn
    ``VotingClassifier(voting="soft")`` with uniform weights.
  * ``"rank_averaging"``: per-sample average of within-base ranks of
    ``predict_proba[:, 1]``. Robust to scale/calibration differences
    across heterogeneous base learners (a tree model and a linear
    model produce probabilities on different scales; rank-averaging
    normalizes that). Standard in Kaggle-stack ensembling.

Both methods are CONSTRAINED: uniform weights, no meta-learner. A
meta-learner stack would need its own held-out fold (otherwise it
overfits to the OOF predictions); for the calibration phase we keep
the surface narrow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


EnsembleMethod = Literal["soft_voting", "rank_averaging"]

DEFAULT_N_FOLDS: int = 5
DEFAULT_RANDOM_STATE: int = 42


def _ensemble_predictions(
    per_base_proba: Dict[str, np.ndarray],
    method: EnsembleMethod,
) -> np.ndarray:
    """Combine per-base positive-class probabilities into an ensemble vector.

    Args:
        per_base_proba: Map ``{base_name → predict_proba_pos_array}``.
            Each value must be a 1-D array of length n_samples.
        method: ``"soft_voting"`` or ``"rank_averaging"``.

    Returns:
        1-D ensemble probability array of length n_samples. Values lie
        in [0, 1] for both methods (rank-averaging emits the per-base
        mean rank divided by n_samples).
    """
    proba_matrix = np.array(list(per_base_proba.values()))
    if method == "soft_voting":
        return np.asarray(np.mean(proba_matrix, axis=0))
    if method == "rank_averaging":
        # rank within each base estimator's predictions, then average
        # ranks across bases. Normalize by n_samples so the output is
        # in [0, 1] (rather than [1, n_samples]).
        from scipy.stats import rankdata

        n_samples = proba_matrix.shape[1]
        ranks = np.array([rankdata(row, method="average") for row in proba_matrix])
        return np.asarray(np.mean(ranks, axis=0) / float(n_samples))
    raise ValueError(
        f"Unknown ensemble method {method!r}; expected one of {{'soft_voting', 'rank_averaging'}}."
    )


def compute_stacking_baseline_cv(
    base_estimators: Dict[str, Any],
    X: Any,
    y: np.ndarray,
    n_folds: int = DEFAULT_N_FOLDS,
    method: EnsembleMethod = "soft_voting",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """Nested-CV stacking baseline (plan v3 §4 T2.5).

    For each of K stratified folds:

      1. Clone every base estimator and fit on K-1 folds.
      2. Predict positive-class probabilities on the held-out fold.
      3. Combine via ``method`` to produce ensemble probabilities.
      4. Compute fold AUC for each base AND for the ensemble.

    Then aggregate fold AUCs (mean ± std) and compare ensemble to the
    best single base. Per plan §6 T2.5 acceptance, the function returns
    metrics regardless of whether the ensemble beats the best single —
    "documents why not" is acceptable.

    Args:
        base_estimators: Dict mapping name → sklearn-compatible estimator
            (must have ``fit`` and ``predict_proba``). Cloned per fold to
            avoid mutating the user's references.
        X: Feature matrix (numpy array or DataFrame).
        y: Binary label array (0/1).
        n_folds: K for ``StratifiedKFold``.
        method: ``"soft_voting"`` (default) or ``"rank_averaging"``.
        random_state: Seed for ``StratifiedKFold(shuffle=True)``.

    Returns:
        Dict with keys (all on validation_metrics scope):

          * ``stacking_method``: the configured ensemble method.
          * ``stacking_n_folds``: K.
          * ``stacking_n_base_estimators``: count of base learners.
          * ``stacking_ensemble_cv_auc_mean`` / ``_std``: mean and std
            of fold AUCs for the ensemble.
          * ``stacking_per_base_cv_auc_mean``: dict {name → mean fold AUC}.
          * ``stacking_best_single_name``: which base had the highest
            mean fold AUC.
          * ``stacking_best_single_cv_auc_mean``: that AUC.
          * ``stacking_ensemble_lift_over_best_single``: signed lift
            (ensemble - best_single).
          * ``stacking_ensemble_beats_best_single``: bool.

        Returns the failure dict ``{"stacking_completed": False, "stacking_error": ...}``
        if a fold cannot run end-to-end (e.g., single-class fold,
        base estimator without ``predict_proba``).
    """
    if not base_estimators:
        return {
            "stacking_completed": False,
            "stacking_error": "no base_estimators provided",
            "stacking_method": method,
            "stacking_n_folds": n_folds,
            "stacking_n_base_estimators": 0,
        }
    if len(base_estimators) < 2:
        return {
            "stacking_completed": False,
            "stacking_error": (
                f"stacking requires >=2 base_estimators; got {len(base_estimators)}"
            ),
            "stacking_method": method,
            "stacking_n_folds": n_folds,
            "stacking_n_base_estimators": len(base_estimators),
        }

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_ensemble_aucs: List[float] = []
    per_base_fold_aucs: Dict[str, List[float]] = {n: [] for n in base_estimators}

    _is_df = hasattr(X, "iloc")

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if _is_df:
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        try:
            per_base_proba: Dict[str, np.ndarray] = {}
            for name, est in base_estimators.items():
                est_cloned = clone(est)
                est_cloned.fit(X_train, y_train)
                if not hasattr(est_cloned, "predict_proba"):
                    return {
                        "stacking_completed": False,
                        "stacking_error": (f"base estimator {name!r} lacks predict_proba"),
                        "stacking_method": method,
                        "stacking_n_folds": n_folds,
                        "stacking_n_base_estimators": len(base_estimators),
                    }
                proba = est_cloned.predict_proba(X_val)
                per_base_proba[name] = proba[:, 1] if proba.ndim == 2 else proba

            ensemble_proba = _ensemble_predictions(per_base_proba, method)
            fold_ensemble_aucs.append(float(roc_auc_score(y_val, ensemble_proba)))
            for name, p in per_base_proba.items():
                per_base_fold_aucs[name].append(float(roc_auc_score(y_val, p)))
        except Exception as e:
            logger.warning("Stacking CV fold %d failed: %s", fold_idx, e)
            continue

    if not fold_ensemble_aucs:
        return {
            "stacking_completed": False,
            "stacking_error": "all folds failed",
            "stacking_method": method,
            "stacking_n_folds": n_folds,
            "stacking_n_base_estimators": len(base_estimators),
        }

    per_base_means: Dict[str, float] = {
        n: float(np.mean(aucs)) if aucs else float("nan") for n, aucs in per_base_fold_aucs.items()
    }
    # Ignore NaN-only bases when picking best-single.
    valid_bases = {n: m for n, m in per_base_means.items() if not np.isnan(m)}
    if not valid_bases:
        return {
            "stacking_completed": False,
            "stacking_error": "no base produced valid AUCs",
            "stacking_method": method,
            "stacking_n_folds": n_folds,
            "stacking_n_base_estimators": len(base_estimators),
        }

    best_single_name = max(valid_bases, key=lambda n: valid_bases[n])
    best_single_auc = valid_bases[best_single_name]
    ensemble_mean = float(np.mean(fold_ensemble_aucs))
    ensemble_std = float(np.std(fold_ensemble_aucs))
    lift = ensemble_mean - best_single_auc

    result = {
        "stacking_completed": True,
        "stacking_method": method,
        "stacking_n_folds": n_folds,
        "stacking_n_base_estimators": len(base_estimators),
        "stacking_n_effective_folds": len(fold_ensemble_aucs),
        "stacking_ensemble_cv_auc_mean": ensemble_mean,
        "stacking_ensemble_cv_auc_std": ensemble_std,
        "stacking_per_base_cv_auc_mean": per_base_means,
        "stacking_best_single_name": best_single_name,
        "stacking_best_single_cv_auc_mean": best_single_auc,
        "stacking_ensemble_lift_over_best_single": float(lift),
        "stacking_ensemble_beats_best_single": ensemble_mean > best_single_auc,
    }

    if ensemble_mean > best_single_auc:
        logger.info(
            "T2.5 stacking: ensemble (%s) beats best single (%s): %.4f vs %.4f (lift=%+.4f)",
            method,
            best_single_name,
            ensemble_mean,
            best_single_auc,
            lift,
        )
    else:
        logger.info(
            "T2.5 stacking: ensemble (%s) does NOT beat best single (%s): "
            "%.4f vs %.4f (lift=%+.4f). Per plan §6 T2.5 this is acceptable; "
            "the audit reports the comparison and downstream selection "
            "remains best-single.",
            method,
            best_single_name,
            ensemble_mean,
            best_single_auc,
            lift,
        )

    return result
