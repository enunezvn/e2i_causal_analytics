"""Advanced model validation for imbalanced classification.

Addresses five validation gaps:
1. Permutation test — confirms signal is genuine vs leaked
2. Stratified k-fold CV — validates metric stability across folds
3. Calibration analysis — ECE + calibration curve
4. Imbalance-aware suspicion heuristic — PR-AUC + class ratio
5. Post-hoc calibration — Platt/isotonic without retraining
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# =============================================================================
# 1. PERMUTATION TEST
# =============================================================================


def compute_permutation_test(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_permutations: int = 100,
) -> Dict[str, Any]:
    """Permutation test: shuffle labels, recompute AUC, derive p-value.

    If the actual AUC is significantly higher than shuffled AUCs,
    the model has learned a genuine signal (not leakage artifact).

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (1D or 2D)
        n_permutations: Number of label shuffles

    Returns:
        Dictionary with p-value, shuffled AUC stats, and verdict
    """
    if y_proba is None:
        return {"permutation_pvalue": None, "signal_genuine": None}

    y_proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

    try:
        actual_auc = float(roc_auc_score(y_true, y_proba_pos))
    except ValueError:
        return {"permutation_pvalue": None, "signal_genuine": None}

    rng = np.random.default_rng(42)
    shuffled_aucs: List[float] = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_true)
        try:
            shuffled_aucs.append(float(roc_auc_score(y_shuffled, y_proba_pos)))
        except ValueError:
            continue

    if not shuffled_aucs:
        return {"permutation_pvalue": None, "signal_genuine": None}

    pvalue = float(np.mean([a >= actual_auc for a in shuffled_aucs]))

    return {
        "permutation_pvalue": pvalue,
        "permutation_auc_mean": float(np.mean(shuffled_aucs)),
        "permutation_auc_std": float(np.std(shuffled_aucs)),
        "actual_auc": actual_auc,
        "n_permutations": n_permutations,
        "signal_genuine": pvalue < 0.05,
    }


# =============================================================================
# 2. STRATIFIED K-FOLD CROSS-VALIDATION
# =============================================================================


def compute_stratified_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    *,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Stratified k-fold CV to validate metric stability.

    Clones the trained model (preserving hyperparameters), retrains
    on each fold, and reports metric distributions.

    Args:
        model: Trained sklearn-compatible model (will be cloned)
        X: Full feature matrix (all splits combined)
        y: Full label vector (all splits combined)
        n_folds: Number of CV folds
        random_state: Seed for the StratifiedKFold splitter (Day-3 W3-lite
            wiring per shard 21 §A audit; threaded by the orchestrator's
            ``resolve_fold_random_state`` helper so each repeated_k10 fold
            sees a distinct nested-CV draw rather than re-using the same
            historical 42).

    Returns:
        Dictionary with per-fold and aggregated metrics
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_metrics: Dict[str, List[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "f1": [],
        "mcc": [],
    }

    # DataFrame integer-position slicing requires .iloc; numpy uses []
    _is_df = hasattr(X, "iloc")

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if _is_df:
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        try:
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)

            y_pred = fold_model.predict(X_fold_val)
            y_proba = None
            if hasattr(fold_model, "predict_proba"):
                y_proba = fold_model.predict_proba(X_fold_val)[:, 1]

            fold_metrics["f1"].append(float(f1_score(y_fold_val, y_pred, zero_division=0)))
            fold_metrics["mcc"].append(float(matthews_corrcoef(y_fold_val, y_pred)))
            if y_proba is not None:
                fold_metrics["roc_auc"].append(float(roc_auc_score(y_fold_val, y_proba)))
                fold_metrics["pr_auc"].append(float(average_precision_score(y_fold_val, y_proba)))
        except Exception as e:
            logger.warning(f"CV fold {fold_idx} failed: {e}")
            continue

    if not fold_metrics["f1"]:
        return {"cv_completed": False, "cv_error": "All folds failed"}

    result: Dict[str, Any] = {"cv_completed": True, "n_folds": n_folds}
    for metric, values in fold_metrics.items():
        if values:
            result[f"cv_{metric}_mean"] = float(np.mean(values))
            result[f"cv_{metric}_std"] = float(np.std(values))
            result[f"cv_{metric}_folds"] = values

    return result


# =============================================================================
# 3. CALIBRATION ANALYSIS (ECE + curve)
# =============================================================================


def compute_calibration_analysis(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error and calibration curve data.

    ECE measures how well predicted probabilities match observed
    frequencies — critical for clinical decision-making.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (1D or 2D)
        n_bins: Number of calibration bins

    Returns:
        Dictionary with ECE, calibration curve points, and bin details
    """
    if y_proba is None:
        return {"calibration_ece": None}

    y_proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

    # Calibration curve for visualization
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba_pos, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        return {"calibration_ece": None}

    # ECE: weighted average of |accuracy - confidence| per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    bin_details: List[Dict[str, float]] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba_pos >= lo) & (y_proba_pos <= hi)
        else:
            mask = (y_proba_pos >= lo) & (y_proba_pos < hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        accuracy = float(y_true[mask].mean())
        confidence = float(y_proba_pos[mask].mean())
        gap = abs(accuracy - confidence)
        ece += (n_bin / n_total) * gap

        bin_details.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n_samples": n_bin,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": gap,
            }
        )

    result: Dict[str, Any] = {
        "calibration_ece": float(ece),
        "calibration_curve_true": fraction_of_positives.tolist(),
        "calibration_curve_pred": mean_predicted_value.tolist(),
        "calibration_bins": bin_details,
        "n_bins": n_bins,
    }
    # Brier decomposition (Murphy 1973 / Bröcker 2009) — shard 20 §C.3.
    # Defensively recompute the Brier score over the test sample so the
    # identity check is local to this function (the upstream
    # ``brier_score`` field comes from sklearn on the same arrays).
    brier_score_for_decomp = float(np.mean((y_proba_pos - y_true) ** 2))
    result.update(_compute_brier_decomposition(y_true, bin_details, brier_score_for_decomp))
    return result


def _compute_brier_decomposition(
    y_true: np.ndarray,
    bin_details: List[Dict[str, float]],
    brier_score: float,
) -> Dict[str, float]:
    """Murphy 1973 / Bröcker 2009 Brier decomposition (shard 20 §C.3).

    Reuses the per-bin (n, accuracy=o_k, confidence=f_k) tuples already
    computed by the ECE loop. Returns a dict with reliability, resolution,
    uncertainty, and a ``recombined`` value that should match
    ``brier_score`` within ~1e-6 for K ≥ 10 bins on tier0-scale data
    (sanity-check of the decomposition identity).

    Sign convention: Bröcker 2009 writes ``Brier = reliability − resolution
    + uncertainty`` (resolution enters with a minus sign because higher
    resolution reduces Brier). DeGroot 1983 writes the same identity
    differently — we keep Bröcker's printed-resolution sign.

    When ``bin_details`` is empty (e.g., calibration_curve raised) or the
    summed sample count is zero, returns NaN-valued fields so callers can
    detect the degenerate case.
    """
    nan_block: Dict[str, float] = {
        "brier_reliability": float("nan"),
        "brier_resolution": float("nan"),
        "brier_uncertainty": float("nan"),
        "brier_recombined": float("nan"),
        "brier_decomposition_residual": float("nan"),
    }
    if not bin_details:
        return nan_block
    n_total = float(sum(b["n_samples"] for b in bin_details))
    if n_total <= 0:
        return nan_block
    p_bar = float(np.mean(y_true == 1))
    reliability = (
        sum(b["n_samples"] * (b["confidence"] - b["accuracy"]) ** 2 for b in bin_details) / n_total
    )
    resolution = sum(b["n_samples"] * (b["accuracy"] - p_bar) ** 2 for b in bin_details) / n_total
    uncertainty = p_bar * (1.0 - p_bar)
    recombined = reliability - resolution + uncertainty
    return {
        "brier_reliability": float(reliability),
        "brier_resolution": float(resolution),
        "brier_uncertainty": float(uncertainty),
        "brier_recombined": float(recombined),
        "brier_decomposition_residual": float(abs(recombined - brier_score)),
    }


# =============================================================================
# 4. IMBALANCE-AWARE SUSPICION HEURISTIC
# =============================================================================


def check_imbalance_aware_suspicion(
    metrics_result: Dict[str, Any],
    class_distribution: Optional[Dict[str, int]],
    problem_type: str,
) -> Dict[str, Any]:
    """Imbalance-aware post-training suspicion check.

    Replaces the fixed AUC >= 0.99 heuristic with thresholds that
    adapt to the class ratio. Uses PR-AUC and MCC as primary signals
    since they are robust to imbalance.

    Args:
        metrics_result: Full metrics dict from evaluator
        class_distribution: {class_label: count} from training data
        problem_type: Problem type string

    Returns:
        Dictionary with leakage_suspected, suspicion_level, reasons
    """
    if problem_type not in ("binary_classification", "multiclass_classification"):
        # Delegate regression to original logic
        return _check_regression_suspicion(metrics_result)

    test_metrics = metrics_result.get("test_metrics", {})
    train_metrics = metrics_result.get("train_metrics", {})
    validation_metrics = metrics_result.get("validation_metrics", {})

    auc = test_metrics.get("roc_auc")
    pr_auc = test_metrics.get("pr_auc")
    precision = test_metrics.get("precision")
    recall = test_metrics.get("recall")
    mcc = test_metrics.get("mcc")
    brier = test_metrics.get("brier_score")

    # Determine minority ratio
    minority_ratio = 0.5
    if class_distribution:
        values = list(class_distribution.values())
        total = sum(values)
        if total > 0:
            minority_ratio = min(values) / total

    reasons: List[str] = []
    recommendations: List[str] = []

    # --- Adaptive AUC threshold ---
    # With severe imbalance (<5%), high AUC is expected — raise threshold
    if minority_ratio < 0.05:
        auc_threshold = 0.999
    elif minority_ratio < 0.20:
        auc_threshold = 0.995
    else:
        auc_threshold = 0.99

    if auc is not None and auc >= auc_threshold:
        reasons.append(
            f"AUC={auc:.4f} >= {auc_threshold} "
            f"(threshold adjusted for {minority_ratio:.1%} minority class)"
        )
        recommendations.append(
            "Check features for target leakage — "
            f"AUC threshold adjusted for class imbalance ({minority_ratio:.1%} minority)"
        )

    # --- PR-AUC check (robust to imbalance) ---
    # Baseline PR-AUC ≈ minority_ratio for random classifier
    # Suspicious if PR-AUC reaches 95%+ of the way from baseline to 1.0
    if pr_auc is not None and minority_ratio < 0.5:
        pr_auc_ceiling = 0.95 * (1.0 - minority_ratio) + minority_ratio
        if pr_auc >= pr_auc_ceiling:
            reasons.append(
                f"PR-AUC={pr_auc:.4f} >= {pr_auc_ceiling:.4f} "
                f"(95% of achievable range above {minority_ratio:.4f} baseline)"
            )
            recommendations.append(
                "PR-AUC reaching near theoretical maximum — verify feature independence from target"
            )

    # --- Perfect precision AND recall (always suspicious) ---
    if precision is not None and recall is not None:
        if precision >= 0.999 and recall >= 0.999:
            reasons.append(
                f"Perfect precision ({precision:.4f}) and recall ({recall:.4f}) "
                f"indicates tautological model"
            )
            recommendations.append(
                "Features likely encode the target directly — audit feature derivation pipeline"
            )

    # --- MCC check (balanced metric, ignores class ratio) ---
    if mcc is not None and mcc >= 0.95:
        reasons.append(f"MCC={mcc:.4f} >= 0.95 is implausible on real clinical data")
        recommendations.append(
            "MCC is robust to imbalance — high MCC with high AUC confirms suspicion"
        )

    # --- Brier score near zero ---
    if brier is not None and brier < 1e-6:
        reasons.append(f"Brier={brier:.2e} effectively zero — implausible calibration")
        recommendations.append("Zero calibration error means predicted probabilities are perfect")

    # --- All splits consistency check (imbalance-aware) ---
    # With severe imbalance, high ROC-AUC across splits is expected;
    # use PR-AUC for imbalanced data since it is the honest metric.
    if minority_ratio < 0.20:
        split_pr_aucs = []
        for m in [train_metrics, validation_metrics, test_metrics]:
            a = m.get("pr_auc")
            if a is not None:
                split_pr_aucs.append(a)
        if len(split_pr_aucs) >= 3 and all(a > 0.90 for a in split_pr_aucs):
            variance = float(np.var(split_pr_aucs))
            if variance < 0.001:
                reasons.append(
                    f"All splits PR-AUC > 0.90 (variance={variance:.6f}) — "
                    "no generalization gap (checked with imbalance-robust metric)"
                )
                recommendations.append(
                    "Consistently high PR-AUC across splits is unusual — "
                    "verify features are legitimately available at prediction time"
                )
    else:
        split_aucs = []
        for m in [train_metrics, validation_metrics, test_metrics]:
            a = m.get("roc_auc")
            if a is not None:
                split_aucs.append(a)
        if len(split_aucs) >= 3 and all(a > 0.98 for a in split_aucs):
            variance = float(np.var(split_aucs))
            if variance < 0.001:
                reasons.append(
                    f"All splits AUC > 0.98 (variance={variance:.6f}) — no generalization gap"
                )
                recommendations.append(
                    "Identical performance across splits suggests trivially recoverable signal"
                )

    if not reasons:
        return {
            "leakage_suspected": False,
            "suspicion_level": "none",
            "suspicion_reasons": [],
            "investigation_recommendations": [],
        }

    # Determine severity
    has_critical = False
    if auc is not None and auc >= 0.999:
        has_critical = True
    if precision is not None and recall is not None:
        if precision >= 0.999 and recall >= 0.999:
            has_critical = True
    if mcc is not None and mcc >= 0.99:
        has_critical = True

    return {
        "leakage_suspected": True,
        "suspicion_level": "critical" if has_critical else "high",
        "suspicion_reasons": reasons,
        "investigation_recommendations": recommendations,
    }


def _check_regression_suspicion(metrics_result: Dict[str, Any]) -> Dict[str, Any]:
    """Suspicion check for regression models."""
    test_metrics = metrics_result.get("test_metrics", {})
    reasons: List[str] = []
    recommendations: List[str] = []

    r2 = test_metrics.get("r2")
    rmse = test_metrics.get("rmse")

    if r2 is not None and r2 >= 0.999:
        reasons.append(f"R²={r2:.6f} >= 0.999 is implausible on real-world data")
        recommendations.append("Check features for target leakage")
    if rmse is not None and rmse < 1e-6:
        reasons.append(f"RMSE={rmse:.2e} is effectively zero")
        recommendations.append("Near-zero RMSE suggests features deterministically encode target")

    if not reasons:
        return {
            "leakage_suspected": False,
            "suspicion_level": "none",
            "suspicion_reasons": [],
            "investigation_recommendations": [],
        }

    has_critical = r2 is not None and r2 >= 0.999
    return {
        "leakage_suspected": True,
        "suspicion_level": "critical" if has_critical else "high",
        "suspicion_reasons": reasons,
        "investigation_recommendations": recommendations,
    }


# =============================================================================
# 5. POST-HOC CALIBRATION
# =============================================================================


def apply_post_hoc_calibration(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> Tuple[Any, Dict[str, Any]]:
    """Apply post-hoc calibration without retraining the base model.

    Uses CalibratedClassifierCV with cv="prefit" — fits a calibration
    mapping on validation data, preserving the base model's ranking.

    Args:
        model: Trained sklearn-compatible model
        X_val: Validation features (for fitting calibration)
        y_val: Validation labels

    Returns:
        Tuple of (calibrated_model, calibration_info_dict)
    """
    # Defense-in-depth: CalibratedClassifierCV.fit() does NOT validate that the
    # underlying estimator is a classifier — sklearn only catches the mismatch
    # later when predict_proba is called, by which time the caller has stored
    # `calibration_applied=True` and downstream code crashes inside
    # sklearn._get_response_values. Skip calibration cleanly when the base
    # model is not sklearn-classifier-compatible (covers conformal wrappers,
    # NGBoost-style distribution predictors, custom regressor wrappers).
    if not is_classifier(model):
        logger.info(
            "Skipping post-hoc calibration: base model is not a sklearn classifier "
            "(typically a conformal wrapper or distribution predictor). "
            "Set skip_post_hoc_calibration=True in the registry entry to silence this check."
        )
        return model, {
            "calibration_method": method,
            "calibration_applied": False,
            "skip_reason": "base_model_not_a_classifier",
        }
    try:
        # Use FrozenEstimator if available (sklearn >= 1.6), else cv="prefit"
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method=method)
        except ImportError:
            calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
        calibrated.fit(X_val, y_val)
        return calibrated, {
            "calibration_method": method,
            "calibration_applied": True,
            "calibration_fit_samples": len(y_val),
        }
    except Exception as e:
        logger.warning(f"Post-hoc calibration failed: {e}")
        return model, {
            "calibration_method": method,
            "calibration_applied": False,
            "calibration_error": str(e),
        }


# =============================================================================
# 6. THRESHOLD OPTIMIZATION (F1-based)
# =============================================================================


def optimize_threshold_f1(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Find the decision threshold that maximizes F1 score.

    Complements the existing Youden's J threshold (which maximizes
    sensitivity + specificity) with an F1-optimal threshold that
    balances precision and recall for the minority class.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (1D or 2D)

    Returns:
        Dictionary with optimal threshold and metrics at that threshold
    """
    if y_proba is None:
        return {"f1_optimal_threshold": 0.5}

    y_proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0.0
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_proba_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    y_pred_opt = (y_proba_pos >= best_threshold).astype(int)

    return {
        "f1_optimal_threshold": best_threshold,
        "f1_at_optimal": float(best_f1),
        "precision_at_f1_optimal": float(precision_score(y_true, y_pred_opt, zero_division=0)),
        "recall_at_f1_optimal": float(recall_score(y_true, y_pred_opt, zero_division=0)),
        "mcc_at_f1_optimal": float(matthews_corrcoef(y_true, y_pred_opt)),
    }


# =============================================================================
# 7. STRATIFIED SPLIT VALIDATION
# =============================================================================


def validate_stratified_splits(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Verify class distribution is preserved across data splits.

    Args:
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        tolerance: Max allowed deviation from train class ratio

    Returns:
        Dictionary with per-split ratios and stratification verdict
    """

    def _positive_ratio(y: np.ndarray) -> float:
        return float(np.mean(y))

    train_ratio = _positive_ratio(y_train)
    val_ratio = _positive_ratio(y_val)
    test_ratio = _positive_ratio(y_test)

    val_drift = abs(val_ratio - train_ratio)
    test_drift = abs(test_ratio - train_ratio)
    is_stratified = val_drift <= tolerance and test_drift <= tolerance

    return {
        "split_positive_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio,
        },
        "val_drift": float(val_drift),
        "test_drift": float(test_drift),
        "is_stratified": is_stratified,
        "stratification_warning": None
        if is_stratified
        else (
            f"Class ratio drift detected: "
            f"val={val_drift:.3f}, test={test_drift:.3f} (tolerance={tolerance}). "
            f"Consider using stratified splitting."
        ),
    }
