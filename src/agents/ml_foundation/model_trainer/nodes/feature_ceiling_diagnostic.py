"""Feature-ceiling / class-separability diagnostic for model_trainer.

Why this node exists
--------------------
A low-performing classifier on imbalanced data is *usually misdiagnosed* as a
"class imbalance problem" when the real binding constraint is **class
separability** — whether the features can distinguish the classes at all.
Imbalance-handling techniques (SMOTE, cost-sensitive weights, under/over
sampling) shift the decision boundary but CANNOT manufacture separability:
when the feature set is the ceiling they leave PR-AUC flat or *degrade* it
while inflating balanced accuracy into a mirage of improvement (ULB
fraud-detection-handbook Ch.6; "Separability in Class Imbalance",
Chawla/Daily-Dose-of-Data-Science; and the empirical Optum-mart disproof,
docs/results/tier0_optum_mart_initiation_events_disproof_20260606.md).

This node runs a cheap, honest estimate of the native ceiling — a small
stratified-CV plain logistic regression (no class weighting, no resampling) on
the preprocessed training features — and reports:

* ``feature_ceiling_auc``      — mean CV ROC-AUC (bulk ranking quality)
* ``feature_ceiling_pr_auc``   — mean CV average-precision (the metric that
  matters at low prevalence)
* ``feature_ceiling_pr_auc_lift`` — PR-AUC ÷ prevalence (1.0 == no skill)
* ``feature_ceiling_label``    — feature_bound / intermediate / separable
* ``feature_ceiling_note``     — a plain-language recommendation

It is **advisory only** — it does not alter control flow or the imbalance
strategy. Downstream consumers (operator, report, MLflow, or a future gate)
can read ``feature_ceiling_label`` to decide whether throwing resampling at the
problem is worthwhile. Runs on the ORIGINAL preprocessed train (before
resampling) so it measures the native signal. Never raises — on any failure it
emits ``feature_ceiling_computed=False`` with a reason.

Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Subsample cap for the diagnostic fit — keeps the node well under ~2s even on
# large marts. The estimate is a ceiling check, not the production model, so a
# stratified subsample is plenty.
_DIAGNOSTIC_ROW_CAP = 20_000
_DIAGNOSTIC_MAX_FOLDS = 3
_DIAGNOSTIC_RANDOM_STATE = 42  # noqa: fixed seed — deterministic diagnostic

# Label thresholds (heuristics calibrated to the Optum-mart disproof, where a
# feature-bound ceiling sat at AUC~0.68 / PR-AUC lift~2, and the separable
# synthetic contrast at AUC~0.97 / lift~34).
_FEATURE_BOUND_AUC = 0.65
_FEATURE_BOUND_LIFT = 3.0
_SEPARABLE_AUC = 0.80


def _ensure_numpy(data: Any) -> Optional[np.ndarray]:
    """Coerce to a numpy array, or return None."""
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "values"):  # pandas DataFrame / Series
        return np.asarray(data.values)
    return np.asarray(data)


def _stratified_subsample(
    X: np.ndarray, y: np.ndarray, cap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to ``cap`` rows (preserves prevalence)."""
    if len(y) <= cap:
        return X, y
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=_DIAGNOSTIC_RANDOM_STATE)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def _classify(auc: float, pr_auc_lift: float) -> tuple[str, str]:
    """Map (AUC, PR-AUC lift) to a label + plain-language recommendation."""
    if auc < _FEATURE_BOUND_AUC and pr_auc_lift < _FEATURE_BOUND_LIFT:
        return (
            "feature_bound",
            "Weak, diffuse signal: PR-AUC is near the prevalence baseline. The "
            "ceiling is the FEATURE SET (separability), not class imbalance. "
            "Resampling / cost-sensitive weights will not raise PR-AUC and may "
            "degrade it — the lever is richer features, not rebalancing.",
        )
    if auc >= _SEPARABLE_AUC:
        return (
            "separable",
            "Classes are well separated in feature space: the model learns the "
            "boundary without rebalancing. Imbalance handling is optional and "
            "low-impact here; verify on PR-AUC, not balanced accuracy.",
        )
    return (
        "intermediate",
        "Moderate signal. Imbalance handling may shift the operating point but "
        "is unlikely to move the ceiling much; compare PR-AUC (NOT balanced "
        "accuracy) with and without it before committing.",
    )


def _compute_ceiling(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Cheap stratified-CV plain-LR estimate of the native ceiling."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_sub, y_sub = _stratified_subsample(X, y, _DIAGNOSTIC_ROW_CAP)
    minority_count = int(min(np.bincount(y_sub.astype(int))))
    n_splits = max(2, min(_DIAGNOSTIC_MAX_FOLDS, minority_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=_DIAGNOSTIC_RANDOM_STATE)

    # Plain LR — NO class_weight, NO resampling: this measures the native
    # separability of the feature set, which is exactly the ceiling that
    # imbalance handling cannot raise.
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    auc = float(
        np.mean(cross_val_score(clf, X_sub, y_sub, cv=cv, scoring="roc_auc"))
    )
    pr_auc = float(
        np.mean(cross_val_score(clf, X_sub, y_sub, cv=cv, scoring="average_precision"))
    )
    prevalence = float(y_sub.mean())
    lift = pr_auc / prevalence if prevalence > 0 else float("nan")
    label, note = _classify(auc, lift)
    return {
        "feature_ceiling_computed": True,
        "feature_ceiling_auc": auc,
        "feature_ceiling_pr_auc": pr_auc,
        "feature_ceiling_prevalence": prevalence,
        "feature_ceiling_pr_auc_lift": lift,
        "feature_ceiling_label": label,
        "feature_ceiling_note": note,
        "feature_ceiling_n_eval": int(len(y_sub)),
        "feature_ceiling_cv_folds": n_splits,
    }


def _skip(reason: str) -> Dict[str, Any]:
    """Advisory no-op result."""
    return {
        "feature_ceiling_computed": False,
        "feature_ceiling_label": "not_computed",
        "feature_ceiling_note": reason,
    }


async def feature_ceiling_diagnostic(state: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate the native feature-separability ceiling (advisory).

    Reads the ORIGINAL preprocessed training features (before resampling) and
    the training labels, runs a cheap stratified-CV plain logistic regression,
    and emits a separability label + recommendation. Advisory only: it never
    changes the imbalance strategy or raises.

    Args:
        state: ModelTrainerState with ``X_train_preprocessed`` and
            ``train_data['y']``.

    Returns:
        Dictionary with ``feature_ceiling_*`` keys (see module docstring).
    """
    problem_type = state.get("problem_type", "binary_classification")
    if problem_type not in ("binary_classification",):
        return _skip(f"Separability diagnostic only runs for binary_classification (got {problem_type!r}).")

    X = _ensure_numpy(state.get("X_train_preprocessed"))
    y = _ensure_numpy((state.get("train_data") or {}).get("y"))
    if X is None or y is None:
        return _skip("No preprocessed training features/labels available for the separability diagnostic.")
    y = y.flatten()

    classes = np.unique(y)
    if len(classes) < 2:
        return _skip(f"Only {len(classes)} class present — separability is undefined.")
    if int(min(np.bincount(y.astype(int)))) < 2:
        return _skip("Fewer than 2 minority samples — separability diagnostic skipped.")

    try:
        result = _compute_ceiling(X, y)
    except Exception as exc:  # never block training on a diagnostic
        logger.warning("Feature-ceiling diagnostic failed: %s", exc)
        return _skip(f"Separability diagnostic errored (non-fatal): {exc}")

    logger.info(
        "Feature-ceiling diagnostic: label=%s AUC=%.4f PR-AUC=%.4f (lift=%.2f over prevalence %.4f)",
        result["feature_ceiling_label"],
        result["feature_ceiling_auc"],
        result["feature_ceiling_pr_auc"],
        result["feature_ceiling_pr_auc_lift"],
        result["feature_ceiling_prevalence"],
    )
    if result["feature_ceiling_label"] == "feature_bound":
        logger.warning(
            "Feature-bound ceiling detected: resampling/cost-sensitive handling "
            "will not raise PR-AUC here — the lever is richer features. %s",
            result["feature_ceiling_note"],
        )
    return result
