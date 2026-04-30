"""Model evaluation for model_trainer.

This module evaluates trained models on train/validation/test sets
using real sklearn metrics with bootstrap confidence intervals.

Version: 2.0.0
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .advanced_validation import (
    apply_post_hoc_calibration,
    check_imbalance_aware_suspicion,
    compute_calibration_analysis,
    compute_permutation_test,
    compute_stratified_cv,
    optimize_threshold_f1,
    validate_stratified_splits,
)

logger = logging.getLogger(__name__)


def _compute_baseline_test_metrics(
    y_train: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    problem_type: str,
) -> Dict[str, float]:
    """Stratified-dummy baseline AUC for the lift-over-baseline criterion.

    The baseline is a ``DummyClassifier(strategy="stratified")`` fit on
    ``y_train`` — its test-set AUC is the random-with-class-prior reference
    against which the trained model's lift is measured. ``"stratified"`` is
    the only strategy whose test predictions span both classes, so
    ``roc_auc_score`` is well-defined; ``"most_frequent"`` and ``"prior"``
    yield single-class predictions where AUC is undefined.

    Returns an empty dict when the problem is not binary classification, or
    when either split is too small / single-class to compute a meaningful
    baseline. The caller treats an empty return as "skip the criterion"
    (see narrow exemption at ``_check_success_criteria``); this is the only
    success criterion that is exemptable, and only under these guards.

    See ``.claude/plans/pre_phase2_unblockers.md`` Section B for the design.
    """
    if problem_type != "binary_classification":
        return {}
    if y_train is None or y_test is None:
        return {}
    if len(y_train) < 10 or len(y_test) < 10:
        return {}
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)
    if np.unique(y_train_arr).size < 2 or np.unique(y_test_arr).size < 2:
        return {}

    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(np.zeros((len(y_train_arr), 1)), y_train_arr)
    proba = dummy.predict_proba(np.zeros((len(y_test_arr), 1)))[:, 1]
    return {"baseline_test_auc": float(roc_auc_score(y_test_arr, proba))}


def _positive_class_proba(y_proba: np.ndarray) -> np.ndarray:
    """Return the positive-class probability column from a 1D or 2D proba array.

    sklearn's ``predict_proba`` returns a 2D ``(n_samples, n_classes)`` array
    where column 1 is P(class=1) for binary classifiers. Some callers
    (calibrators, custom wrappers) may already pass the 1D positive-class
    column. This helper accepts either shape and returns the 1D array.
    """
    if y_proba.ndim == 2:
        return y_proba[:, 1]
    return y_proba


async def evaluate_model(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate trained model on train/validation/test sets.

    CRITICAL EVALUATION PRINCIPLES:
    - Test set touched ONCE for final evaluation
    - Validation set already used for HPO
    - Training set evaluation for overfitting detection
    - Holdout set NOT evaluated (locked for post-deployment)

    Args:
        state: ModelTrainerState with trained_model, preprocessed data,
               problem_type, success_criteria

    Returns:
        Dictionary with train_metrics, validation_metrics, test_metrics,
        problem-specific metrics (auc_roc, precision, recall for classification;
        rmse, mae, r2 for regression), confidence_interval,
        success_criteria_met, success_criteria_results

    Raises:
        No exceptions - returns error in state if evaluation fails
    """
    # Extract trained model and data
    trained_model = state.get("trained_model")
    problem_type = state.get("problem_type", "binary_classification")
    success_criteria = state.get("success_criteria", {})
    # Block 5 (#10): optional dict mapping {tp,fp,fn,tn} → per-prediction
    # dollar value. None = skip business_utility computation.
    cost_matrix = state.get("cost_matrix")

    # Extract preprocessed data
    X_train_preprocessed = state.get("X_train_preprocessed")
    X_validation_preprocessed = state.get("X_validation_preprocessed")
    X_test_preprocessed = state.get("X_test_preprocessed")
    train_data = state.get("train_data", {})
    validation_data = state.get("validation_data", {})
    test_data = state.get("test_data", {})
    y_train = train_data.get("y")
    y_validation = validation_data.get("y")
    y_test = test_data.get("y")

    # Validate required inputs
    if trained_model is None:
        logger.error("No trained model available for evaluation")
        return {
            "error": "No trained model available for evaluation",
            "error_type": "missing_trained_model",
        }

    if X_test_preprocessed is None or y_test is None:
        logger.error("Missing test data for evaluation")
        return {
            "error": "Missing test data for evaluation",
            "error_type": "missing_test_data",
        }

    # The preprocessor (preprocessor.py:152) returns a numpy array, so
    # X_*_preprocessed is numpy. LightGBM 4.x stores feature_names_in_
    # at fit time even on numpy input ('Column_0..N'), and sklearn warns
    # on every predict where X has no feature names. Wrap X with the
    # preprocessor's post-encoding feature names so predict sees them
    # consistently. y_* stays numpy — metric helpers expect plain arrays.
    X_train_np = _wrap_with_feature_names(X_train_preprocessed, state)
    X_val_np = _wrap_with_feature_names(X_validation_preprocessed, state)
    X_test_np = _wrap_with_feature_names(X_test_preprocessed, state)
    y_train_np = _ensure_numpy(y_train)
    y_val_np = _ensure_numpy(y_validation)
    y_test_np = _ensure_numpy(y_test)

    logger.info(
        f"Evaluating model: problem_type={problem_type}, "
        f"X_test shape={X_test_np.shape if X_test_np is not None else 'None'}"
    )

    # Make predictions on all sets
    try:
        assert X_test_np is not None
        predictions = _make_predictions(
            model=trained_model,
            X_train=X_train_np,
            X_val=X_val_np,
            X_test=X_test_np,
            problem_type=problem_type,
        )
    except Exception as e:
        logger.error(f"Prediction failed during evaluation: {e}")
        return {
            "error": f"Prediction failed during evaluation: {str(e)}",
            "error_type": "prediction_failed",
        }

    # Get imbalance detection status
    imbalance_detected = state.get("imbalance_detected", False)
    minority_ratio = state.get("minority_ratio", 0.5)

    # y_test_np must be valid at this point (checked above)
    assert y_test_np is not None

    # Compute metrics based on problem type
    try:
        if problem_type in ["binary_classification"]:
            metrics_result = _compute_classification_metrics(
                y_train=y_train_np,
                y_train_pred=predictions["y_train_pred"],
                y_train_proba=predictions["y_train_proba"],
                y_validation=y_val_np,
                y_validation_pred=predictions["y_val_pred"],
                y_validation_proba=predictions["y_val_proba"],
                y_test=y_test_np,
                y_test_pred=predictions["y_test_pred"],
                y_test_proba=predictions["y_test_proba"],
                imbalance_detected=imbalance_detected,
                minority_ratio=minority_ratio,
                cost_matrix=cost_matrix,
            )
        elif problem_type == "multiclass_classification":
            metrics_result = _compute_multiclass_metrics(
                y_train=y_train_np,
                y_train_pred=predictions["y_train_pred"],
                y_train_proba=predictions["y_train_proba"],
                y_validation=y_val_np,
                y_validation_pred=predictions["y_val_pred"],
                y_validation_proba=predictions["y_val_proba"],
                y_test=y_test_np,
                y_test_pred=predictions["y_test_pred"],
                y_test_proba=predictions["y_test_proba"],
            )
        elif problem_type in ["regression", "continuous"]:
            metrics_result = _compute_regression_metrics(
                y_train=y_train_np,
                y_train_pred=predictions["y_train_pred"],
                y_validation=y_val_np,
                y_validation_pred=predictions["y_val_pred"],
                y_test=y_test_np,
                y_test_pred=predictions["y_test_pred"],
            )
        else:
            logger.error(f"Unsupported problem type: {problem_type}")
            return {
                "error": f"Unsupported problem type: {problem_type}",
                "error_type": "unsupported_problem_type",
            }
    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        return {
            "error": f"Metrics computation failed: {str(e)}",
            "error_type": "metrics_computation_failed",
        }

    # =========================================================================
    # ADVANCED VALIDATION (imbalance-aware)
    # =========================================================================
    if problem_type == "binary_classification":
        y_test_proba = predictions.get("y_test_proba")

        # 1. Permutation test — confirm signal is genuine
        logger.info("Running permutation test (100 shuffles)...")
        permutation_result = compute_permutation_test(y_test_np, y_test_proba, n_permutations=100)
        metrics_result["permutation_test"] = permutation_result
        if permutation_result.get("signal_genuine") is not None:
            logger.info(
                f"Permutation test: p={permutation_result['permutation_pvalue']:.4f}, "
                f"signal_genuine={permutation_result['signal_genuine']}"
            )

        # 2. Calibration analysis — ECE + calibration curve
        calibration_result = compute_calibration_analysis(y_test_np, y_test_proba)
        metrics_result["calibration_error"] = calibration_result.get("calibration_ece")
        metrics_result["calibration_analysis"] = calibration_result

        # 3. F1-optimal threshold (complements existing Youden's J)
        f1_threshold_result = optimize_threshold_f1(y_test_np, y_test_proba)
        metrics_result["f1_threshold_analysis"] = f1_threshold_result

        # 4. MCC from test metrics
        metrics_result["mcc"] = metrics_result.get("test_metrics", {}).get("mcc")

        # 5. Stratified k-fold CV — validate metric stability
        if X_train_np is not None and y_train_np is not None:
            logger.info("Running 5-fold stratified cross-validation...")
            # Concatenate while preserving feature names. If all inputs
            # are DataFrames (the post-#13 path), pd.concat keeps column
            # names so each cloned LGBM fold fit/predicts with names —
            # avoiding the 'X does not have valid feature names' warnings
            # that would re-appear if we np.vstack'd to a bare ndarray.
            try:
                import pandas as pd
            except ImportError:
                pd = None  # type: ignore[assignment]
            xs = [x for x in [X_train_np, X_val_np, X_test_np] if x is not None]
            arrays_y = [y for y in [y_train_np, y_val_np, y_test_np] if y is not None]
            if pd is not None and all(isinstance(x, pd.DataFrame) for x in xs):
                X_all = pd.concat(xs, axis=0, ignore_index=True)
            else:
                X_all = np.vstack([x.to_numpy() if hasattr(x, "to_numpy") else x for x in xs])
            y_all = np.concatenate(arrays_y)
            cv_result = compute_stratified_cv(trained_model, X_all, y_all, n_folds=5)
            metrics_result["cv_results"] = cv_result
            if cv_result.get("cv_completed"):
                logger.info(
                    f"CV results: AUC={cv_result.get('cv_roc_auc_mean', 0):.4f}"
                    f"±{cv_result.get('cv_roc_auc_std', 0):.4f}, "
                    f"PR-AUC={cv_result.get('cv_pr_auc_mean', 0):.4f}"
                    f"±{cv_result.get('cv_pr_auc_std', 0):.4f}"
                )

        # 6. Post-hoc calibration (isotonic) — better probability estimates
        if X_val_np is not None and y_val_np is not None:
            calibrated_model, cal_info = apply_post_hoc_calibration(
                trained_model, X_val_np, y_val_np, method="isotonic"
            )
            metrics_result["post_hoc_calibration"] = cal_info
            if cal_info.get("calibration_applied") and X_test_np is not None:
                cal_proba = calibrated_model.predict_proba(X_test_np)
                cal_proba_pos = _positive_class_proba(cal_proba)
                opt_thresh = metrics_result.get("optimal_threshold", 0.5)
                cal_pred = (cal_proba_pos >= opt_thresh).astype(int)
                cal_test_metrics = _compute_split_classification_metrics(
                    y_test_np, cal_pred, cal_proba
                )
                metrics_result["calibrated_test_metrics"] = cal_test_metrics
                # Compute ECE improvement
                cal_ece = compute_calibration_analysis(y_test_np, cal_proba)
                metrics_result["calibrated_ece"] = cal_ece.get("calibration_ece")
                # v3 B1 fix overlay: surface the post-isotonic ECE on the
                # inner ``test_metrics`` dict so the alias
                # ``maximum_calibration_error → calibrated_ece`` resolves
                # at criterion-check time. The outer ``metrics_result``
                # already carries the value at line above; this overlay
                # makes it reachable from ``_check_success_criteria``,
                # which only sees ``metrics_result["test_metrics"]``.
                inner_test_metrics = metrics_result.get("test_metrics")
                if isinstance(inner_test_metrics, dict):
                    inner_test_metrics["calibrated_ece"] = (
                        cal_ece.get("calibration_ece")
                        if cal_ece.get("calibration_ece") is not None
                        else float("nan")
                    )
                uncal_ece = metrics_result.get("calibration_error")
                if uncal_ece is not None and cal_ece.get("calibration_ece") is not None:
                    logger.info(
                        f"Calibration: ECE {uncal_ece:.4f} → "
                        f"{cal_ece['calibration_ece']:.4f} (isotonic)"
                    )

        # 7. Stratified split validation — check class ratio preservation
        if y_train_np is not None and y_val_np is not None:
            split_val = validate_stratified_splits(y_train_np, y_val_np, y_test_np)
            metrics_result["split_validation"] = split_val
            if split_val.get("stratification_warning"):
                logger.warning(split_val["stratification_warning"])

    # Check success criteria
    success_results = _check_success_criteria(
        metrics_result["test_metrics"],
        success_criteria,
        problem_type,
    )

    logger.info(
        f"Evaluation complete: success_criteria_met={success_results['success_criteria_met']}"
    )

    # Post-training leakage suspicion check (imbalance-aware)
    class_distribution = state.get("class_distribution")
    suspicion_result = check_imbalance_aware_suspicion(
        metrics_result, class_distribution, problem_type
    )

    if suspicion_result["leakage_suspected"]:
        logger.warning(
            f"LEAKAGE SUSPECTED: level={suspicion_result['suspicion_level']}, "
            f"reasons={suspicion_result['suspicion_reasons']}"
        )

    # Merge results
    return {
        **metrics_result,
        **success_results,
        **suspicion_result,
    }


def _is_causal_model(model: Any) -> bool:
    """Check if model is an EconML causal model.

    Args:
        model: Model instance

    Returns:
        True if model is a causal model (has .effect() but no .predict())
    """
    has_effect = hasattr(model, "effect") or hasattr(model, "const_marginal_effect")
    has_predict = hasattr(model, "predict")
    return has_effect and not has_predict


def _make_predictions(
    model: Any,
    X_train: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
    problem_type: str,
) -> Dict[str, Any]:
    """Make predictions on all data splits.

    Args:
        model: Trained model
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        problem_type: Problem type

    Returns:
        Dictionary with predictions and probabilities
    """
    is_classification = problem_type in [
        "binary_classification",
        "multiclass_classification",
    ]

    # Check if this is a causal model (EconML)
    if _is_causal_model(model):
        return _make_causal_predictions(model, X_train, X_val, X_test)

    has_proba = hasattr(model, "predict_proba")

    predictions = {}

    # Training set
    if X_train is not None:
        predictions["y_train_pred"] = model.predict(X_train)
        if is_classification and has_proba:
            predictions["y_train_proba"] = model.predict_proba(X_train)
        else:
            predictions["y_train_proba"] = None
    else:
        predictions["y_train_pred"] = None
        predictions["y_train_proba"] = None

    # Validation set
    if X_val is not None:
        predictions["y_val_pred"] = model.predict(X_val)
        if is_classification and has_proba:
            predictions["y_val_proba"] = model.predict_proba(X_val)
        else:
            predictions["y_val_proba"] = None
    else:
        predictions["y_val_pred"] = None
        predictions["y_val_proba"] = None

    # Test set (FINAL)
    predictions["y_test_pred"] = model.predict(X_test)
    if is_classification and has_proba:
        predictions["y_test_proba"] = model.predict_proba(X_test)
    else:
        predictions["y_test_proba"] = None

    return predictions


def _make_causal_predictions(
    model: Any,
    X_train: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
) -> Dict[str, Any]:
    """Make predictions for causal models (EconML).

    Causal models estimate treatment effects, not outcomes. For evaluation purposes,
    we use the CATE (Conditional Average Treatment Effect) as the "prediction".
    This allows downstream evaluation to compute metrics on effect heterogeneity.

    Args:
        model: EconML causal model
        X_train: Training features
        X_val: Validation features
        X_test: Test features

    Returns:
        Dictionary with CATE estimates as predictions (no probabilities)
    """
    logger.info("Using causal model prediction: effect() instead of predict()")

    predictions = {}

    # Determine which effect method to use
    if hasattr(model, "const_marginal_effect"):
        effect_fn = model.const_marginal_effect
    elif hasattr(model, "effect"):
        effect_fn = lambda X: model.effect(X)  # noqa: E731
    else:
        raise ValueError("Causal model has no effect() or const_marginal_effect() method")

    # Training set
    if X_train is not None:
        try:
            cate_train = effect_fn(X_train)
            # Convert CATE to binary predictions (positive effect = 1, negative = 0)
            predictions["y_train_pred"] = (cate_train.flatten() > 0).astype(int)
            # Use CATE values as "probabilities" (normalized to 0-1 for metrics)
            cate_norm = _normalize_cate(cate_train.flatten())
            predictions["y_train_proba"] = np.column_stack([1 - cate_norm, cate_norm])
        except Exception as e:
            logger.warning(f"Failed to compute CATE for training set: {e}")
            predictions["y_train_pred"] = None
            predictions["y_train_proba"] = None
    else:
        predictions["y_train_pred"] = None
        predictions["y_train_proba"] = None

    # Validation set
    if X_val is not None:
        try:
            cate_val = effect_fn(X_val)
            predictions["y_val_pred"] = (cate_val.flatten() > 0).astype(int)
            cate_norm = _normalize_cate(cate_val.flatten())
            predictions["y_val_proba"] = np.column_stack([1 - cate_norm, cate_norm])
        except Exception as e:
            logger.warning(f"Failed to compute CATE for validation set: {e}")
            predictions["y_val_pred"] = None
            predictions["y_val_proba"] = None
    else:
        predictions["y_val_pred"] = None
        predictions["y_val_proba"] = None

    # Test set (FINAL)
    try:
        cate_test = effect_fn(X_test)
        predictions["y_test_pred"] = (cate_test.flatten() > 0).astype(int)
        cate_norm = _normalize_cate(cate_test.flatten())
        predictions["y_test_proba"] = np.column_stack([1 - cate_norm, cate_norm])
    except Exception as e:
        logger.warning(f"Failed to compute CATE for test set: {e}")
        # Fall back to random predictions for evaluation to proceed
        predictions["y_test_pred"] = np.zeros(len(X_test), dtype=int)
        predictions["y_test_proba"] = np.column_stack(
            [np.ones(len(X_test)) * 0.5, np.ones(len(X_test)) * 0.5]
        )

    return predictions


def _normalize_cate(cate: np.ndarray) -> np.ndarray:
    """Normalize CATE values to 0-1 range for use as pseudo-probabilities.

    Args:
        cate: Raw CATE values

    Returns:
        Normalized values in [0, 1]
    """
    cate_min = cate.min()
    cate_max = cate.max()
    if cate_max - cate_min > 1e-8:
        return (cate - cate_min) / (cate_max - cate_min)  # type: ignore[no-any-return]
    else:
        return np.ones_like(cate) * 0.5


# v3 NB grid p_t values (Vickers 2019). The validator-set ``_adaptive_p_t``
# audit field selects the regime's p_t at criterion-check time; emitting a
# grid keeps the cost bounded (6 floats) and lets downstream tools plot
# decision-curve analyses at any p_t without re-evaluating.
_V3_NB_GRID_P_T_VALUES: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def _compute_calibration_slope_intercept(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Tuple[float, float]:
    """Logistic recalibration. Returns ``(slope, intercept)`` per van Calster 2019.

    Fits ``LogisticRegression(C=1e10)`` on the logit of predicted positive-
    class probabilities. Slope == 1 and intercept == 0 mean perfect
    calibration; slope < 1 means over-confident, slope > 1 under-confident.

    Stability guard: requires ``n_pos >= 30`` AND ``n_neg >= 30``. Returns
    ``(nan, nan)`` if either count is too low (the LR fit is unstable
    below that threshold). Adverse-regime synthetic runs (``n_pos = 18`` at
    ``N=900, prev=0.02``) hit this guard, and the evaluator's NaN-guard
    branch records ``met=None`` for the calibration_*_deviation criteria.
    """
    if y_proba.ndim != 1:
        raise ValueError(
            f"y_proba must be 1d (positive-class probabilities); got shape {y_proba.shape}"
        )
    eps = 1e-9
    y_proba_clipped = np.clip(y_proba, eps, 1.0 - eps)
    logits = np.log(y_proba_clipped / (1.0 - y_proba_clipped)).reshape(-1, 1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos < 30 or n_neg < 30:
        return (float("nan"), float("nan"))
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(logits, y_true.astype(int))
    return (float(lr.coef_[0, 0]), float(lr.intercept_[0]))


def _compute_net_benefit_at_p_t(
    y_true: np.ndarray, y_proba: np.ndarray, p_t: float
) -> float:
    """Vickers 2006 net benefit at threshold probability ``p_t``.

    ``NB = TP/n - (FP/n) * p_t / (1 - p_t)``. Operating point: predict
    positive when ``y_proba >= p_t``. Returns ``nan`` for ``p_t`` outside
    ``(0, 1)`` or empty inputs (NB is undefined there).

    The decision-curve gate at ``NB > 0`` is operationally equivalent to
    ``precision > p_t`` (Vickers 2006 algebra), but stating it as NB makes
    the cost-ratio assumption explicit.
    """
    if not 0.0 < p_t < 1.0:
        return float("nan")
    n = len(y_true)
    if n == 0:
        return float("nan")
    y_pred = (y_proba >= p_t).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    return (tp / n) - (fp / n) * p_t / (1.0 - p_t)


def _compute_classification_metrics(
    y_train: Optional[np.ndarray],
    y_train_pred: Optional[np.ndarray],
    y_train_proba: Optional[np.ndarray],
    y_validation: Optional[np.ndarray],
    y_validation_pred: Optional[np.ndarray],
    y_validation_proba: Optional[np.ndarray],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_proba: Optional[np.ndarray],
    imbalance_detected: bool = False,
    minority_ratio: float = 0.5,
    cost_matrix: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute binary classification metrics using sklearn.

    Threshold-selection policy (Block 1A — finding #6):
        The classification operating point (Youden's J optimum, and the
        precision-constrained alternative when ``minority_ratio < 0.05``)
        is selected on the VALIDATION arrays, frozen, and then applied to
        the test set without re-tuning. This prevents test-set leakage
        into the operating point. If validation arrays are not provided,
        the function falls back to the default 0.5 threshold rather than
        tuning on test — test-set integrity is preserved at the cost of
        a possibly off-by-default operating point.

        ``validation_metrics`` is recomputed AT the chosen threshold when
        validation probabilities are available, so its precision/recall/F1
        reflect performance at the same operating point used for test.

    Args:
        y_train: Training labels
        y_train_pred: Training predictions (at model's default threshold)
        y_train_proba: Training probabilities
        y_validation: Validation labels — used for threshold selection
        y_validation_pred: Validation predictions (at model's default)
        y_validation_proba: Validation probabilities — used for threshold
            selection. When None, threshold falls back to 0.5.
        y_test: Test labels
        y_test_pred: Test predictions (at model's default threshold)
        y_test_proba: Test probabilities. The chosen threshold is applied
            to these to produce the final reported test metrics.
        imbalance_detected: Whether class imbalance was detected. When
            True, ``test_metrics`` reports values at the chosen threshold;
            otherwise it reports values at the model's default 0.5.
        minority_ratio: Ratio of minority class (0-1). When below 0.05,
            the precision-constrained threshold is also evaluated on
            validation and may override the Youden's J optimum.
        cost_matrix: Optional Block-5 (#10) per-outcome dollar matrix —
            keys ``tp``/``fp``/``fn``/``tn`` mapped to float per-prediction
            value. When supplied, ``business_utility`` is computed at the
            chosen threshold for both validation and test and exposed via
            ``validation_metrics["business_utility"]`` /
            ``test_metrics["business_utility"]`` plus a top-level mirror.

    Returns:
        Dictionary with the standard top-level metrics keys
        (``train_metrics``, ``validation_metrics``, ``test_metrics``,
        ``test_metrics_at_05``, ``test_metrics_at_optimal``, ``auc_roc``,
        ``precision``, ``recall``, ``f1_score``, ``pr_auc``,
        ``brier_score``, ``confusion_matrix``, ``optimal_threshold``,
        ``precision_at_k``, ``confidence_interval``, ``bootstrap_samples``,
        ``precision_constrained``, ``calibration_error``, ``f1_macro``,
        ``f1_weighted``, plus minority metrics when imbalance is detected).

        Block 1A-specific keys:
            - ``optimal_threshold`` (top level): the canonical chosen
              threshold (validation-tuned, or 0.5 fallback). Existing
              cross-codebase consumers read this key.
            - ``chosen_threshold_source`` (top level): provenance flag,
              one of ``"validation"`` or ``"default"``.
            - ``validation_metrics["chosen_threshold"]``: same numeric
              value as ``optimal_threshold``, exposed at the validation
              metric level so model-registry / monitoring consumers can
              audit the operating point that produced the validation
              numbers.
            - ``validation_metrics["chosen_threshold_source"]``:
              provenance flag mirrored at the validation level.
    """
    # Training metrics
    train_metrics = {}
    if y_train is not None and y_train_pred is not None:
        train_metrics = _compute_split_classification_metrics(y_train, y_train_pred, y_train_proba)

    # =====================================================================
    # THRESHOLD TUNING — VALIDATION SET ONLY (Block 1A — finding #6)
    # =====================================================================
    # The optimal classification threshold (and any precision-constrained
    # alternative) MUST be selected on validation data, then frozen, then
    # applied to test. Tuning on test leaks test info into the operating
    # point and inflates apparent test performance.
    #
    # Step 1 (1A-I-3): pick the canonical validation-vs-default threshold
    # via `_select_threshold`. Step 2 (rare-event override) stays inline
    # because it needs `minority_ratio` and produces a `precision_constrained`
    # dict consumed by the result builder below.
    # =====================================================================
    optimal_threshold, threshold_source = _select_threshold(
        y_validation, y_validation_proba, cost_matrix=cost_matrix
    )

    # For rare-event prediction, apply precision-constrained threshold
    # tuned ON VALIDATION (not test). This may override the Youden's J
    # optimum returned by `_select_threshold` above.
    precision_constrained: Optional[Dict[str, Any]] = None
    if minority_ratio < 0.05 and y_validation is not None and y_validation_proba is not None:
        precision_constrained = _compute_precision_constrained_threshold(
            y_validation, y_validation_proba
        )
        if precision_constrained and precision_constrained.get("target_achieved"):
            optimal_threshold = precision_constrained["precision_constrained_threshold"]
            logger.info(
                "Using precision-constrained threshold "
                f"{optimal_threshold:.4f} tuned on validation "
                f"(precision={precision_constrained['precision_at_threshold']:.4f}, "
                f"recall={precision_constrained['recall_at_threshold']:.4f})"
            )

    # Validation metrics: recomputed AT THE CHOSEN THRESHOLD when probas
    # are available so validation_metrics reflects performance at the same
    # frozen operating point used for test. We also persist
    # `chosen_threshold` and `chosen_threshold_source` here so downstream
    # consumers can audit which split produced the threshold.
    validation_metrics: Dict[str, Any] = {}
    if y_validation is not None and y_validation_pred is not None:
        if y_validation_proba is not None:
            y_val_proba_pos = _positive_class_proba(y_validation_proba)
            y_validation_pred_at_chosen = (y_val_proba_pos >= optimal_threshold).astype(int)
            validation_metrics = cast(
                Dict[str, Any],
                _compute_split_classification_metrics(
                    y_validation, y_validation_pred_at_chosen, y_validation_proba
                ),
            )
        else:
            validation_metrics = cast(
                Dict[str, Any],
                _compute_split_classification_metrics(
                    y_validation, y_validation_pred, y_validation_proba
                ),
            )
        validation_metrics["chosen_threshold"] = float(optimal_threshold)
        validation_metrics["chosen_threshold_source"] = threshold_source

    # CRITICAL: For imbalanced data, apply the FROZEN threshold tuned on
    # validation to test predictions. No re-tuning on test.
    # math.isclose tolerates the tiny float drift _compute_optimal_threshold
    # can return when its sklearn input lands exactly on the default — we
    # only want the rebinarisation pass when the chosen threshold is
    # meaningfully different from 0.5.
    y_test_pred_optimal = y_test_pred  # Default to model predictions
    if y_test_proba is not None and not math.isclose(optimal_threshold, 0.5):
        y_proba_pos = _positive_class_proba(y_test_proba)
        y_test_pred_optimal = (y_proba_pos >= optimal_threshold).astype(int)
        logger.info(
            f"Applying frozen threshold {optimal_threshold:.4f} "
            f"(tuned on {threshold_source}) to test predictions (vs default 0.5)"
        )

    # Test metrics at 0.5 threshold (standard)
    test_metrics_standard = _compute_split_classification_metrics(y_test, y_test_pred, y_test_proba)

    # Test metrics at the FROZEN chosen threshold (no re-tuning on test)
    test_metrics_optimal = _compute_split_classification_metrics(
        y_test, y_test_pred_optimal, y_test_proba
    )

    # Use optimal threshold metrics as primary when imbalance detected
    # This ensures we report useful metrics, not misleading ones
    if imbalance_detected:
        test_metrics = test_metrics_optimal
        logger.info(
            f"Using optimal threshold metrics for imbalanced data: "
            f"recall={test_metrics.get('recall', 0):.4f}, "
            f"precision={test_metrics.get('precision', 0):.4f}"
        )
    else:
        test_metrics = test_metrics_standard

    # Lift-over-baseline (Section B of pre_phase2_unblockers plan): inject a
    # stratified-dummy baseline AUC and the absolute lift the trained model
    # achieves over it. The criterion is read by ``_check_success_criteria``
    # via metric_aliases below; threshold (default 0.10) lives in
    # ``criteria_validator._define_classification_criteria``. Absolute lift
    # (auc - baseline_auc) is preferred over relative because: (a) it
    # matches the natural reading of the criterion name; (b) it is not
    # deflated when the baseline drifts above 0.50 on small/skewed splits.
    baseline_metrics = _compute_baseline_test_metrics(
        y_train, y_test, "binary_classification"
    )
    if "baseline_test_auc" in baseline_metrics:
        baseline_auc = baseline_metrics["baseline_test_auc"]
        test_metrics["baseline_test_auc"] = baseline_auc
        test_auc = test_metrics.get("roc_auc")
        if test_auc is not None:
            test_metrics["minimum_lift_over_baseline"] = float(test_auc - baseline_auc)

    # v3 emits (task 05 of adaptive_success_criteria plan, Option C):
    # surface metrics the v3 active gates resolve against. All emits land
    # on the inner ``test_metrics`` (B4 fix) — the dict that reaches
    # ``_check_success_criteria`` via ``metrics_result["test_metrics"]``.
    #
    # 1. ``train_val_auc_delta`` (B2 fix): emit the gap between train and
    #    validation AUC so the ``maximum_train_val_delta`` adaptive gate
    #    has a metric. Without this, every adaptive run hard-fails the
    #    criterion via the missing-metric path. Use absolute value so the
    #    "max delta" gate fires regardless of direction.
    train_auc_value = (
        train_metrics.get("roc_auc") if isinstance(train_metrics, dict) else None
    )
    val_auc_value = (
        validation_metrics.get("roc_auc")
        if isinstance(validation_metrics, dict)
        else None
    )
    if train_auc_value is not None and val_auc_value is not None:
        try:
            gap = float(train_auc_value) - float(val_auc_value)
            test_metrics["train_val_auc_delta"] = abs(gap)
        except (TypeError, ValueError):
            logger.warning(
                "train_val_auc_delta could not be computed "
                f"(train roc_auc={train_auc_value!r}, val roc_auc={val_auc_value!r})"
            )

    # 2-4. Calibration slope/intercept (van Calster 2019, NEW v3) and
    #      net-benefit grid (Vickers 2006, NEW v3). Both require positive-
    #      class probabilities; skip the emit when y_test_proba is None.
    if y_test_proba is not None:
        y_test_proba_pos = _positive_class_proba(y_test_proba)
        slope, intercept = _compute_calibration_slope_intercept(
            np.asarray(y_test), y_test_proba_pos
        )
        test_metrics["calibration_slope"] = slope
        test_metrics["calibration_intercept"] = intercept
        test_metrics["calibration_slope_deviation"] = (
            abs(slope - 1.0) if not math.isnan(slope) else float("nan")
        )
        test_metrics["calibration_intercept_magnitude"] = (
            abs(intercept) if not math.isnan(intercept) else float("nan")
        )
        # NB grid keyed on ``p_t={p_t:.2f}`` strings; the evaluator's
        # ``_resolve_net_benefit_from_grid`` reads the regime's ``p_t``
        # from ``success_criteria['_adaptive_p_t']`` and looks up the
        # matching key. Emit the full grid for downstream DCA plotting.
        # Cast through Any to satisfy mypy — ``test_metrics`` is typed
        # ``Dict[str, Optional[float]]`` from ``_compute_split_classification_metrics``,
        # but the v3 emits add a dict-valued ``net_benefit_grid``.
        test_metrics_any: Dict[str, Any] = test_metrics  # type: ignore[assignment]
        test_metrics_any["net_benefit_grid"] = {
            f"p_t={p_t:.2f}": _compute_net_benefit_at_p_t(
                np.asarray(y_test), y_test_proba_pos, p_t
            )
            for p_t in _V3_NB_GRID_P_T_VALUES
        }

    # Extract primary metrics for state
    auc_roc = test_metrics.get("roc_auc")
    precision = test_metrics.get("precision")
    recall = test_metrics.get("recall")
    f1 = test_metrics.get("f1_score")
    pr_auc = test_metrics.get("pr_auc")
    brier = test_metrics.get("brier_score")

    # Confusion matrix at optimal threshold
    cm = confusion_matrix(y_test, y_test_pred_optimal)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        confusion_dict = {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    else:
        confusion_dict = {"matrix": cm.tolist()}

    # Block 5 (#10): business_utility from cost_matrix at the chosen
    # (validation-tuned) threshold. We compute it on BOTH validation and
    # test using the same frozen threshold so the metric reported in
    # validation_metrics matches the operating point that produced the
    # test number — a deployment decision tool needs both.
    test_business_utility: Optional[float] = None
    val_business_utility: Optional[float] = None
    if cost_matrix is not None and cm.shape == (2, 2):
        test_business_utility = _compute_business_utility(
            int(tp), int(fp), int(fn), int(tn), cost_matrix
        )
        if y_validation is not None and y_validation_proba is not None:
            y_val_proba_pos = _positive_class_proba(y_validation_proba)
            y_val_pred_at_chosen = (y_val_proba_pos >= optimal_threshold).astype(int)
            val_cm = confusion_matrix(y_validation, y_val_pred_at_chosen)
            if val_cm.shape == (2, 2):
                v_tn, v_fp, v_fn, v_tp = val_cm.ravel()
                val_business_utility = _compute_business_utility(
                    int(v_tp), int(v_fp), int(v_fn), int(v_tn), cost_matrix
                )
        if val_business_utility is not None:
            validation_metrics["business_utility"] = val_business_utility
        test_metrics["business_utility"] = test_business_utility
        logger.info(
            f"business_utility (chosen_threshold={optimal_threshold:.4f}): "
            f"validation={val_business_utility}, test={test_business_utility}"
        )

    # Precision at k
    precision_at_k = _compute_precision_at_k(y_test, y_test_proba, k_values=[100, 500, 1000])

    # Bootstrap confidence intervals
    confidence_interval, bootstrap_samples = _compute_bootstrap_ci(
        y_test, y_test_pred_optimal, y_test_proba, problem_type="binary_classification"
    )

    # Build result dictionary
    result = {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_05": test_metrics_standard,  # Keep standard for reference
        "test_metrics_at_optimal": test_metrics_optimal,  # Optimal threshold metrics
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "confusion_matrix": confusion_dict,
        "optimal_threshold": optimal_threshold,
        "chosen_threshold_source": threshold_source,
        "precision_at_k": precision_at_k,
        "confidence_interval": confidence_interval,
        "bootstrap_samples": bootstrap_samples,
        "precision_constrained": precision_constrained,
        "calibration_error": None,  # Could add ECE computation
        "rmse": None,
        "mae": None,
        "r2": None,
        # Weighted metrics for imbalanced classification
        "f1_macro": test_metrics.get("f1_macro"),
        "f1_weighted": test_metrics.get("f1_weighted"),
        # Block 5 (#10): top-level mirror of test-set business_utility for
        # downstream consumers (Tier0OutputMapper, deployment decision
        # tools) that read flat metrics off the result dict. None when no
        # cost_matrix was provided.
        "business_utility": test_business_utility,
    }

    # Add minority class metrics when imbalance is detected
    # These are the key metrics for evaluating imbalanced classification
    if imbalance_detected:
        result["minority_recall"] = test_metrics.get("recall_class_1", 0.0)
        result["minority_precision"] = test_metrics.get("precision_class_1", 0.0)
        # Also report what metrics would be at 0.5 threshold for comparison
        result["minority_recall_at_05"] = test_metrics_standard.get("recall_class_1", 0.0)
        logger.info(
            f"Imbalance metrics (optimal threshold): "
            f"minority_recall={result['minority_recall']:.4f}, "
            f"minority_precision={result['minority_precision']:.4f}"
        )
        if float(result["minority_recall_at_05"]) == 0 and float(result["minority_recall"]) > 0:  # type: ignore[arg-type]
            logger.warning(
                f"Model would predict ALL negatives at 0.5 threshold! "
                f"Optimal threshold {optimal_threshold:.4f} rescues recall."
            )

    return result


def _compute_split_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """Compute classification metrics for a single split.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        # Weighted metrics (critical for imbalanced data)
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # Per-class metrics (essential for understanding imbalanced performance)
        "precision_class_0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_class_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_class_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_class_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        # Matthews Correlation Coefficient — robust to class imbalance
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    # Probability-based metrics
    if y_proba is not None:
        y_proba_pos = _positive_class_proba(y_proba)

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_pos))
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba_pos))
        except ValueError:
            metrics["pr_auc"] = None

        try:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_proba_pos))
        except ValueError:
            metrics["brier_score"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["brier_score"] = None

    return metrics


def _compute_multiclass_metrics(
    y_train: Optional[np.ndarray],
    y_train_pred: Optional[np.ndarray],
    y_train_proba: Optional[np.ndarray],
    y_validation: Optional[np.ndarray],
    y_validation_pred: Optional[np.ndarray],
    y_validation_proba: Optional[np.ndarray],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_proba: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Compute multiclass classification metrics.

    Args:
        y_train: Training labels
        y_train_pred: Training predictions
        y_train_proba: Training probabilities
        y_validation: Validation labels
        y_validation_pred: Validation predictions
        y_validation_proba: Validation probabilities
        y_test: Test labels
        y_test_pred: Test predictions
        y_test_proba: Test probabilities

    Returns:
        Dictionary of metrics
    """
    # Training metrics
    train_metrics = {}
    if y_train is not None and y_train_pred is not None:
        train_metrics = {
            "accuracy": float(accuracy_score(y_train, y_train_pred)),
            "f1_macro": float(f1_score(y_train, y_train_pred, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
            ),
        }

    # Validation metrics
    validation_metrics = {}
    if y_validation is not None and y_validation_pred is not None:
        validation_metrics = {
            "accuracy": float(accuracy_score(y_validation, y_validation_pred)),
            "f1_macro": float(
                f1_score(y_validation, y_validation_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_validation, y_validation_pred, average="weighted", zero_division=0)
            ),
        }

    # Test metrics
    test_metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision_macro": float(
            precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0)),
    }

    # AUC for multiclass (OvR)
    if y_test_proba is not None:
        try:
            test_metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_test, y_test_proba, multi_class="ovr")
            )
        except ValueError:
            test_metrics["roc_auc_ovr"] = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    confusion_dict = {"matrix": cm.tolist()}

    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "auc_roc": test_metrics.get("roc_auc_ovr"),
        "precision": test_metrics.get("precision_macro"),
        "recall": test_metrics.get("recall_macro"),
        "f1_score": test_metrics.get("f1_macro"),
        "pr_auc": None,
        "brier_score": None,
        "confusion_matrix": confusion_dict,
        "optimal_threshold": None,
        "precision_at_k": None,
        "confidence_interval": {},
        "bootstrap_samples": 0,
        "calibration_error": None,
        "rmse": None,
        "mae": None,
        "r2": None,
    }


def _compute_regression_metrics(
    y_train: Optional[np.ndarray],
    y_train_pred: Optional[np.ndarray],
    y_validation: Optional[np.ndarray],
    y_validation_pred: Optional[np.ndarray],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute regression metrics using sklearn.

    Args:
        y_train: Training labels
        y_train_pred: Training predictions
        y_validation: Validation labels
        y_validation_pred: Validation predictions
        y_test: Test labels
        y_test_pred: Test predictions

    Returns:
        Dictionary of metrics
    """
    # Training metrics
    train_metrics = {}
    if y_train is not None and y_train_pred is not None:
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_metrics = {
            "mse": float(train_mse),
            "rmse": float(np.sqrt(train_mse)),
            "mae": float(mean_absolute_error(y_train, y_train_pred)),
            "r2": float(r2_score(y_train, y_train_pred)),
        }

    # Validation metrics
    validation_metrics = {}
    if y_validation is not None and y_validation_pred is not None:
        val_mse = mean_squared_error(y_validation, y_validation_pred)
        validation_metrics = {
            "mse": float(val_mse),
            "rmse": float(np.sqrt(val_mse)),
            "mae": float(mean_absolute_error(y_validation, y_validation_pred)),
            "r2": float(r2_score(y_validation, y_validation_pred)),
        }

    # Test metrics (FINAL)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_metrics = {
        "mse": float(test_mse),
        "rmse": float(np.sqrt(test_mse)),
        "mae": float(mean_absolute_error(y_test, y_test_pred)),
        "r2": float(r2_score(y_test, y_test_pred)),
    }

    # Bootstrap confidence intervals
    confidence_interval, bootstrap_samples = _compute_bootstrap_ci(
        y_test, y_test_pred, None, problem_type="regression"
    )

    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "r2": test_metrics["r2"],
        "confidence_interval": confidence_interval,
        "bootstrap_samples": bootstrap_samples,
        # Classification metrics not applicable
        "auc_roc": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "pr_auc": None,
        "brier_score": None,
        "confusion_matrix": None,
        "optimal_threshold": None,
        "precision_at_k": None,
        "calibration_error": None,
    }


def _compute_business_utility(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    cost_matrix: Dict[str, float],
) -> float:
    """Compute business_utility = sum(cost_matrix[outcome] * count[outcome]).

    Each confusion-matrix outcome is multiplied by its per-prediction
    monetary value from ``cost_matrix`` and summed. A "cost" matrix is
    typically signed: revenue from true positives is positive; the cost
    of a false positive is negative; the cost of a missed target (false
    negative) is also negative. The caller decides the sign convention —
    this helper just multiplies and sums.

    Args:
        tp/fp/fn/tn: Confusion-matrix counts at the chosen threshold.
        cost_matrix: Dict with keys ``tp``/``fp``/``fn``/``tn`` mapped to
            float dollar values per prediction.

    Returns:
        Total business utility (float).

    Raises:
        KeyError: If any of the four required keys is missing from
            ``cost_matrix``. The scope_definer's ``_validate_cost_matrix``
            normally guards against this, but the helper enforces it
            again so callers cannot silently drop a value.
    """
    return float(
        tp * cost_matrix["tp"]
        + fp * cost_matrix["fp"]
        + fn * cost_matrix["fn"]
        + tn * cost_matrix["tn"]
    )


def _select_threshold(
    y_validation: Optional[np.ndarray],
    y_validation_proba: Optional[np.ndarray],
    *,
    cost_matrix: Optional[Dict[str, float]] = None,
) -> Tuple[float, str]:
    """Pick the canonical classification threshold + provenance string.

    Block 1A — finding #6: the operating point MUST be selected on
    validation, then frozen for test. This helper encodes that policy:
    when validation labels and probabilities are available, return the
    Youden's J optimum from `_compute_optimal_threshold`; otherwise fall
    back to the default 0.5 threshold and log a warning so test-set
    integrity is preserved at the cost of an off-by-default operating
    point.

    Note: the precision-constrained override (rare-event minority class)
    is NOT handled here — it requires additional caller context
    (`minority_ratio`) and produces a `precision_constrained` dict
    consumed elsewhere by the parent. The caller invokes this helper
    first and may then override the returned threshold.

    Args:
        y_validation: Validation labels. When None, falls back to default.
        y_validation_proba: Validation probabilities. When None, falls
            back to default.
        cost_matrix: Reserved for future cost-aware threshold selection
            (Block 5 #10 follow-up). Currently unused — the helper always
            picks a single threshold from validation Youden's J.

    Returns:
        Tuple of ``(chosen_threshold, chosen_threshold_source)``. The
        source string is one of the literals ``"validation"`` (validation
        arrays present, threshold tuned on them) or ``"default"`` (validation
        arrays absent, threshold pinned to 0.5). Downstream consumers
        (mlflow_logger, audit code) rely on these exact literals.
    """
    if y_validation is not None and y_validation_proba is not None:
        return _compute_optimal_threshold(y_validation, y_validation_proba), "validation"

    logger.warning(
        "Validation arrays unavailable for threshold tuning; "
        "falling back to default 0.5 threshold (test integrity preserved)."
    )
    return 0.5, "default"


def _compute_optimal_threshold(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> float:
    """Compute optimal classification threshold using Youden's J statistic.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities

    Returns:
        Optimal threshold
    """
    if y_proba is None:
        return 0.5

    y_proba_pos = _positive_class_proba(y_proba)

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        candidate = float(thresholds[optimal_idx])
        # sklearn's roc_curve prepends a sentinel threshold of `np.inf`
        # corresponding to the (FPR=0, TPR=0) trivial point. When no
        # threshold yields a positive Youden's J (e.g., a model that is
        # worse than chance on a small held-out set) argmax returns this
        # sentinel. Treat any non-finite or out-of-range value as
        # "no useful threshold found" and fall back to 0.5.
        if not np.isfinite(candidate) or candidate < 0.0 or candidate > 1.0:
            return 0.5
        return candidate
    except Exception:
        return 0.5


def _compute_precision_constrained_threshold(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    target_precision: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Find the lowest threshold where precision >= target.

    For rare-event prediction, Youden's J often yields very low precision.
    This finds a threshold that guarantees a minimum precision level.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_precision: Minimum required precision (default 5%)

    Returns:
        Dict with threshold details, or None if probabilities unavailable
    """
    if y_proba is None:
        return None

    y_proba_pos = _positive_class_proba(y_proba)

    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba_pos)
        # precision_recall_curve returns n+1 precisions but n thresholds
        # The last precision is always 1.0 with recall 0.0 (no corresponding threshold)

        # Find lowest threshold where precision >= target
        best_idx = None
        for i in range(len(thresholds)):
            if precisions[i] >= target_precision:
                if best_idx is None or thresholds[i] < thresholds[best_idx]:
                    best_idx = i

        if best_idx is not None:
            threshold = float(thresholds[best_idx])
            prec = float(precisions[best_idx])
            rec = float(recalls[best_idx])
            # Compute F1 at this threshold
            f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return {
                "precision_constrained_threshold": threshold,
                "precision_at_threshold": prec,
                "recall_at_threshold": rec,
                "f1_at_threshold": f1_val,
                "target_precision": target_precision,
                "target_achieved": True,
                "fallback_used": False,
            }

        # Fallback: F1-optimal threshold
        f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
        f1_best_idx = int(np.argmax(f1_scores))
        threshold = float(thresholds[f1_best_idx])
        prec = float(precisions[f1_best_idx])
        rec = float(recalls[f1_best_idx])
        f1_val = float(f1_scores[f1_best_idx])
        return {
            "precision_constrained_threshold": threshold,
            "precision_at_threshold": prec,
            "recall_at_threshold": rec,
            "f1_at_threshold": f1_val,
            "target_precision": target_precision,
            "target_achieved": False,
            "fallback_used": True,
        }

    except Exception as e:
        logger.warning(f"Precision-constrained threshold computation failed: {e}")
        return None


def _compute_precision_at_k(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    k_values: List[int],
) -> Dict[int, float]:
    """Compute precision at k for different k values.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        k_values: List of k values

    Returns:
        Dictionary of {k: precision_at_k}
    """
    if y_proba is None:
        return {}

    y_proba_pos = _positive_class_proba(y_proba)

    n_samples = len(y_true)
    result = {}

    for k in k_values:
        if k > n_samples:
            continue

        # Get top k indices by probability
        top_k_indices = np.argsort(y_proba_pos)[-k:]

        # Compute precision at k
        precision_at_k = np.mean(y_true[top_k_indices])
        result[k] = float(precision_at_k)

    return result


def _compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    problem_type: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[Dict[str, Tuple[float, float]], int]:
    """Compute bootstrap confidence intervals for metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        problem_type: Problem type
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (confidence_intervals, n_bootstrap)
    """
    n_samples = len(y_true)
    alpha = (1 - confidence) / 2

    # Get positive class probabilities if available
    y_proba_pos = _positive_class_proba(y_proba) if y_proba is not None else None

    # Store bootstrap metrics
    bootstrap_metrics: Dict[str, List[float]] = {}

    for _ in range(n_bootstrap):
        # Bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        if problem_type == "binary_classification":
            # Accuracy
            if "accuracy" not in bootstrap_metrics:
                bootstrap_metrics["accuracy"] = []
            bootstrap_metrics["accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))

            # AUC
            if y_proba_pos is not None:
                y_proba_boot = y_proba_pos[indices]
                try:
                    if "auc" not in bootstrap_metrics:
                        bootstrap_metrics["auc"] = []
                    bootstrap_metrics["auc"].append(roc_auc_score(y_true_boot, y_proba_boot))
                except ValueError:
                    pass

            # Precision, Recall, F1
            if "precision" not in bootstrap_metrics:
                bootstrap_metrics["precision"] = []
            bootstrap_metrics["precision"].append(
                precision_score(y_true_boot, y_pred_boot, zero_division=0)
            )

            if "recall" not in bootstrap_metrics:
                bootstrap_metrics["recall"] = []
            bootstrap_metrics["recall"].append(
                recall_score(y_true_boot, y_pred_boot, zero_division=0)
            )

        elif problem_type == "regression":
            y_pred_boot_reg = y_pred[indices]

            # RMSE
            if "rmse" not in bootstrap_metrics:
                bootstrap_metrics["rmse"] = []
            mse = mean_squared_error(y_true_boot, y_pred_boot_reg)
            bootstrap_metrics["rmse"].append(np.sqrt(mse))

            # MAE
            if "mae" not in bootstrap_metrics:
                bootstrap_metrics["mae"] = []
            bootstrap_metrics["mae"].append(mean_absolute_error(y_true_boot, y_pred_boot_reg))

            # R2
            if "r2" not in bootstrap_metrics:
                bootstrap_metrics["r2"] = []
            try:
                bootstrap_metrics["r2"].append(r2_score(y_true_boot, y_pred_boot_reg))
            except ValueError:
                pass

    # Compute confidence intervals
    confidence_intervals = {}
    for metric_name, values in bootstrap_metrics.items():
        if len(values) > 0:
            lower = float(np.percentile(values, alpha * 100))
            upper = float(np.percentile(values, (1 - alpha) * 100))
            confidence_intervals[metric_name] = (lower, upper)

    return confidence_intervals, n_bootstrap


def _check_metric_suspicion(
    metrics_result: Dict[str, Any],
    problem_type: str,
) -> Dict[str, Any]:
    """Post-training safety net: check for implausibly perfect metrics.

    Runs after metric computation to catch models that may be tautological
    (e.g., AUC=1.0, perfect precision+recall) — a sign of data leakage
    that pre-training checks may have missed.

    Args:
        metrics_result: Full metrics dict from _compute_*_metrics
        problem_type: Problem type

    Returns:
        Dictionary with leakage_suspected, suspicion_level,
        suspicion_reasons, investigation_recommendations
    """
    reasons: List[str] = []
    recommendations: List[str] = []

    test_metrics = metrics_result.get("test_metrics", {})
    train_metrics = metrics_result.get("train_metrics", {})
    validation_metrics = metrics_result.get("validation_metrics", {})

    if problem_type in ["binary_classification", "multiclass_classification"]:
        auc = test_metrics.get("roc_auc")
        precision = test_metrics.get("precision")
        recall = test_metrics.get("recall")
        brier = test_metrics.get("brier_score")

        # Check 1: AUC >= 0.99
        if auc is not None and auc >= 0.99:
            reasons.append(f"AUC={auc:.4f} >= 0.99 is implausible on real-world data")
            recommendations.append(
                "Check features for target leakage — no real-world clinical model achieves AUC >= 0.99"
            )

        # Check 2: Perfect precision AND recall
        if precision is not None and recall is not None:
            if precision >= 0.999 and recall >= 0.999:
                reasons.append(
                    f"Perfect precision ({precision:.4f}) and recall ({recall:.4f}) "
                    f"indicates tautological model"
                )
                recommendations.append(
                    "Features likely encode the target directly — audit feature derivation pipeline"
                )

        # Check 3: All splits AUC > 0.98 with near-zero variance
        split_aucs = []
        for m in [train_metrics, validation_metrics, test_metrics]:
            a = m.get("roc_auc")
            if a is not None:
                split_aucs.append(a)
        if len(split_aucs) >= 3 and all(a > 0.98 for a in split_aucs):
            variance = float(np.var(split_aucs))
            if variance < 0.01:
                reasons.append(
                    f"All splits AUC > 0.98 (variance={variance:.6f}) — "
                    f"no generalization gap across splits"
                )
                recommendations.append(
                    "Identical performance across splits suggests the signal is trivially recoverable"
                )

        # Check 4: Brier score == 0
        if brier is not None and brier < 1e-6:
            reasons.append(f"Brier score={brier:.2e} is effectively zero — implausible calibration")
            recommendations.append(
                "Zero calibration error means predicted probabilities are perfect — "
                "this only happens with deterministic features"
            )

    elif problem_type in ["regression", "continuous"]:
        r2 = test_metrics.get("r2")
        rmse = test_metrics.get("rmse")

        if r2 is not None and r2 >= 0.999:
            reasons.append(f"R²={r2:.6f} >= 0.999 is implausible on real-world data")
            recommendations.append("Check features for target leakage")

        if rmse is not None and rmse < 1e-6:
            reasons.append(f"RMSE={rmse:.2e} is effectively zero")
            recommendations.append(
                "Near-zero RMSE suggests features deterministically encode target"
            )

    # Determine suspicion level
    if not reasons:
        return {
            "leakage_suspected": False,
            "suspicion_level": "none",
            "suspicion_reasons": [],
            "investigation_recommendations": [],
        }

    # Determine severity from the original metric values (not string matching)
    has_critical = False
    if problem_type in ["binary_classification", "multiclass_classification"]:
        auc = test_metrics.get("roc_auc")
        prec = test_metrics.get("precision")
        rec = test_metrics.get("recall")
        if auc is not None and auc >= 0.99:
            has_critical = True
        if prec is not None and rec is not None and prec >= 0.999 and rec >= 0.999:
            has_critical = True
    elif problem_type in ["regression", "continuous"]:
        r2_val = test_metrics.get("r2")
        if r2_val is not None and r2_val >= 0.999:
            has_critical = True
    suspicion_level = "critical" if has_critical else "high"

    return {
        "leakage_suspected": True,
        "suspicion_level": suspicion_level,
        "suspicion_reasons": reasons,
        "investigation_recommendations": recommendations,
    }


def _check_success_criteria(
    test_metrics: Dict[str, float],
    success_criteria: Dict[str, float],
    problem_type: str,
) -> Dict[str, Any]:
    """Check if model meets success criteria.

    Args:
        test_metrics: Test set metrics
        success_criteria: Success thresholds
        problem_type: Problem type

    Returns:
        Dictionary with success_criteria_met and success_criteria_results
    """
    if not success_criteria:
        return {
            "success_criteria_met": True,
            "success_criteria_results": {},
        }

    # v3 (task 05 of adaptive_success_criteria plan): apply the adaptive
    # overlay BEFORE iterating criteria. No-op when ``_adaptive_inputs``
    # or ``baseline_test_auc`` are absent, so fixed-mode runs are
    # unaffected. The overlay returns a possibly-rebuilt dict; we use
    # it as ``success_criteria`` for the rest of this function.
    success_criteria = _apply_adaptive_criteria_overlay(
        success_criteria, test_metrics
    )

    # Per-criterion outcome: True=met, False=not met, None=soft-skipped
    # (see narrow exemption below).
    results: Dict[str, Optional[bool]] = {}
    all_met = True

    # Map metric aliases (including scope_definer naming conventions)
    metric_aliases = {
        "auc": "roc_auc",
        "roc_auc": "roc_auc",
        "minimum_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "minimum_precision": "precision",
        "recall": "recall",
        "minimum_recall": "recall",
        "f1": "f1_score",
        "f1_score": "f1_score",
        "minimum_f1": "f1_score",
        "rmse": "rmse",
        "minimum_rmse": "rmse",
        "mae": "mae",
        "r2": "r2",
        "minimum_r2": "r2",
        "mape": "mape",
        "minimum_mape": "mape",
        # Section B (pre_phase2_unblockers): self-mapping documents that
        # the criterion shares its name with the test_metrics key. Soft-skip
        # behavior on missing values is gated on this exact criterion name
        # below — see _LIFT_OVER_BASELINE_CRITERION usage.
        "minimum_lift_over_baseline": "minimum_lift_over_baseline",
        # Adaptive criteria (task 04 of adaptive_success_criteria plan v3):
        # all five criteria below are emitted by adaptive_success_criteria()
        # when ADAPTIVE_CRITERIA is on; the corresponding metrics are surfaced
        # by _compute_classification_metrics in task 05. Lower-is-better is
        # set on the resolved metric names below. ``minimum_net_benefit_at_p_t``
        # is intentionally absent — it resolves via a special-case lookup
        # against ``net_benefit_grid`` keyed on the regime's p_t (see W3 fix).
        "maximum_calibration_error": "calibrated_ece",
        "maximum_train_val_delta": "train_val_auc_delta",
        "minimum_mcc": "mcc",
        "maximum_calibration_slope_deviation": "calibration_slope_deviation",
        "maximum_calibration_intercept_magnitude": "calibration_intercept_magnitude",
    }
    # Single source of truth for the narrow exemption below; any future
    # criterion that wants soft-skip behavior must go through the same
    # explicit allowlist (do NOT accept any criterion silently).
    _LIFT_OVER_BASELINE_CRITERION = "minimum_lift_over_baseline"
    # v3 NB > 0 gate: resolved against ``net_benefit_grid[p_t=...]`` instead
    # of a metric_aliases lookup. The audit field ``_adaptive_p_t`` (set by
    # the validator when ADAPTIVE_CRITERIA is on) carries the regime's
    # threshold probability; when absent, the criterion soft-skips.
    _NB_AT_P_T_CRITERION = "minimum_net_benefit_at_p_t"

    # Metrics where lower is better
    lower_is_better = {
        "rmse",
        "mae",
        "brier_score",
        "mse",
        "mape",
        # Adaptive criteria (task 04 v3): all four resolved metrics are
        # MAXIMUM tolerable values, so lower-is-better. ``mcc`` is
        # higher-is-better (default); ``net_benefit_at_p_t`` is
        # higher-is-better and resolved via the special-case path.
        "calibrated_ece",
        "train_val_auc_delta",
        "calibration_slope_deviation",
        "calibration_intercept_magnitude",
    }

    for criterion_name, threshold in success_criteria.items():
        # v3: skip audit fields (any key starting with underscore). The
        # validator emits ``_adaptive_skipped`` (list — also caught by the
        # skip-non-numeric branch below) and ``_adaptive_p_t`` (float —
        # would otherwise be evaluated as a numeric criterion against the
        # missing ``adaptive_p_t`` test metric).
        if isinstance(criterion_name, str) and criterion_name.startswith("_"):
            continue

        # Skip non-numeric thresholds (e.g. experiment_id, baseline_model, None placeholders)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            logger.debug(f"Skipping non-numeric success criterion: {criterion_name}={threshold}")
            continue

        # v3 W3 fix: NB > 0 gate resolves against ``net_benefit_grid`` keyed
        # on the regime's p_t, not against a single ``net_benefit_at_p_t``
        # metric. Soft-skip when ``_adaptive_p_t`` is missing or the grid
        # does not carry the requested key.
        if criterion_name == _NB_AT_P_T_CRITERION:
            actual_value = _resolve_net_benefit_from_grid(test_metrics, success_criteria)
            metric_name = "net_benefit_at_p_t"
        else:
            # Resolve metric name
            metric_name = metric_aliases.get(criterion_name, criterion_name)
            actual_value = test_metrics.get(metric_name)

        if actual_value is None:
            # Default policy: a missing metric is a hard fail of the success
            # contract — silently passing would mask a genuine gap.
            #
            # NARROW EXEMPTION (Section B of pre_phase2_unblockers plan):
            # ``minimum_lift_over_baseline`` is computed by
            # ``_compute_baseline_test_metrics`` only when the binary problem
            # has enough rows in both train and test, and both splits have
            # both classes. When those guards trip the baseline AUC is
            # legitimately undefined, so we soft-skip rather than hard-fail.
            # ``met=None`` excludes the criterion from the aggregation; the
            # log line still surfaces the skip so it isn't silent.
            if criterion_name == _LIFT_OVER_BASELINE_CRITERION:
                logger.warning(
                    "Success criterion soft-skipped (degenerate split): "
                    f"{criterion_name} (no baseline AUC available — "
                    "see _compute_baseline_test_metrics guards)"
                )
                results[criterion_name] = None
                continue
            # v3 W3 fix: the NB > 0 gate also soft-skips when the audit
            # ``_adaptive_p_t`` is unset (fixed mode) or the grid does not
            # carry the regime's p_t. The validator does not emit the gate
            # under fixed mode, but a stale config could still send it
            # through — soft-skip rather than hard-fail.
            if criterion_name == _NB_AT_P_T_CRITERION:
                logger.warning(
                    "Success criterion soft-skipped (NB grid unavailable "
                    "or _adaptive_p_t unset): %s",
                    criterion_name,
                )
                results[criterion_name] = None
                continue

            logger.warning(
                f"Success criterion metric not available: {criterion_name} "
                f"(resolved to '{metric_name}', missing from test_metrics)"
            )
            results[criterion_name] = False
            all_met = False
            continue

        # v3 B3 fix: NaN actual values record met=None instead of
        # comparing (Python's `nan <= 0.15` evaluates to False). Fires
        # for adverse-regime calibration metrics when n_pos < 30 or
        # n_neg < 30 (the LR fit is unstable and emits NaN).
        if isinstance(actual_value, float) and math.isnan(actual_value):
            logger.warning(
                "Success criterion soft-skipped (metric value is NaN): "
                "%s (resolved to '%s')",
                criterion_name,
                metric_name,
            )
            results[criterion_name] = None
            continue

        # Check if metric meets threshold
        if metric_name in lower_is_better:
            met = actual_value <= threshold
        else:
            met = actual_value >= threshold

        results[criterion_name] = met
        if not met:
            logger.info(
                f"Success criterion not met: {criterion_name}={actual_value:.4f} "
                f"(threshold={threshold})"
            )
            all_met = False
        else:
            logger.info(
                f"Success criterion met: {criterion_name}={actual_value:.4f} "
                f"(threshold={threshold})"
            )

    # v2/v3 explicit-skip set: criterion names listed in
    # ``success_criteria['_adaptive_skipped']`` are recorded as met=None
    # in results — the audit-trail recording for criteria the adaptive
    # scheme intentionally excluded from aggregation. Plain None thresholds
    # do NOT enter this path; they fall through the existing
    # skip-non-numeric branch above and the validator's S4 defense in
    # ``_define_classification_criteria`` warns + replaces them. See
    # ``.claude/plans/adaptive_success_criteria/01-design.md`` §"Skip
    # semantics" for the full contract.
    adaptive_skipped = success_criteria.get("_adaptive_skipped")
    if isinstance(adaptive_skipped, list):
        for skipped_name in adaptive_skipped:
            if isinstance(skipped_name, str):
                results[skipped_name] = None

    return {
        "success_criteria_met": all_met,
        "success_criteria_results": results,
    }


def _apply_adaptive_criteria_overlay(
    success_criteria: Dict[str, Any],
    test_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Recompute v3 adaptive thresholds with the live ``baseline_test_auc``.

    The scope_definer cannot compute ``baseline_auc`` — it runs before any
    model is trained. It therefore stashes the four pre-eval inputs as
    ``success_criteria['_adaptive_inputs']``; this helper reads that stash
    and the freshly-computed ``baseline_test_auc`` from ``test_metrics``,
    calls ``adaptive_success_criteria()``, and applies the resulting
    ``(thresholds, skipped)`` tuple to a SHALLOW COPY of
    ``success_criteria``:

    - Skipped criteria are REMOVED from the dict (v3 invariant).
    - The v3-deprecated fixed gates (precision / F1) are popped so only
      v3 active gates fire downstream.
    - Firing thresholds OVERWRITE the seeded fixed values.
    - The audit list is stored at ``success_criteria['_adaptive_skipped']``.
    - The regime-keyed threshold probability is stored at
      ``success_criteria['_adaptive_p_t']`` for the NB > 0 gate resolver.

    Returns the (possibly overlaid) dict. When ``_adaptive_inputs`` or
    ``baseline_test_auc`` are absent, the criteria dict is returned
    unchanged. See ``.claude/plans/adaptive_success_criteria/01-design.md``
    for the v3 design contract and ``05-data-shape-introspection.md`` for
    the two-phase hand-off rationale.
    """
    if "_adaptive_inputs" not in success_criteria:
        return success_criteria
    inputs = success_criteria.get("_adaptive_inputs")
    if not isinstance(inputs, dict):
        logger.warning(
            "_adaptive_inputs is not a dict (%s); skipping adaptive overlay",
            type(inputs).__name__,
        )
        return success_criteria
    baseline_auc = test_metrics.get("baseline_test_auc")
    if baseline_auc is None:
        return success_criteria

    # Local import avoids a top-level cross-agent dependency.
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        _V3_DEPRECATED_FIXED_KEYS,
        _V3_REGIME_P_T,
        adaptive_success_criteria,
    )

    try:
        thresholds, skipped = adaptive_success_criteria(
            n_samples=inputs["n_samples"],
            prevalence=float(inputs["prevalence"]),
            baseline_auc=float(baseline_auc),
            feature_count=inputs["feature_count"],
            regime=inputs.get("regime"),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "adaptive overlay refused inputs (%s); leaving success_criteria unchanged",
            exc,
        )
        return success_criteria

    overlaid: Dict[str, Any] = dict(success_criteria)
    # Remove keys that adaptive skipped (v3 invariant: skipped ⇒ ABSENT).
    for key in skipped:
        overlaid.pop(key, None)
    # Drop v3-deprecated fixed gates (precision / F1) so only v3 active
    # gates fire downstream.
    for key in _V3_DEPRECATED_FIXED_KEYS:
        overlaid.pop(key, None)
    # Apply firing thresholds.
    overlaid.update(thresholds)
    # Audit fields.
    overlaid["_adaptive_skipped"] = sorted(skipped)
    regime_in = inputs.get("regime")
    effective_regime = regime_in if regime_in in _V3_REGIME_P_T else "clean"
    overlaid["_adaptive_p_t"] = _V3_REGIME_P_T[effective_regime]
    return overlaid


def _resolve_net_benefit_from_grid(
    test_metrics: Dict[str, Any],
    success_criteria: Dict[str, Any],
) -> Optional[float]:
    """Resolve the v3 ``minimum_net_benefit_at_p_t`` gate via the NB grid.

    The validator emits ``_adaptive_p_t`` on ``success_criteria`` when
    ADAPTIVE_CRITERIA is on; ``_compute_classification_metrics`` (task 05)
    emits ``test_metrics['net_benefit_grid']`` keyed on canonical
    ``"p_t={p_t:.2f}"`` strings. Returns the NB value at the regime's p_t,
    or ``None`` when the grid is absent / does not carry the requested key.

    Returning ``None`` triggers the soft-skip path in the caller — the
    criterion is recorded as ``met=None`` rather than hard-failing on a
    missing metric, because the NB gate only makes sense with a paired
    ``_adaptive_p_t`` audit value.
    """
    p_t = success_criteria.get("_adaptive_p_t")
    if not isinstance(p_t, (int, float)) or isinstance(p_t, bool):
        return None
    grid = test_metrics.get("net_benefit_grid")
    if not isinstance(grid, dict):
        return None
    value = grid.get(f"p_t={float(p_t):.2f}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _ensure_numpy(data: Any) -> Optional[np.ndarray]:
    """Convert data to numpy array if needed.

    Args:
        data: Input data

    Returns:
        Numpy array or None
    """
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        return data

    # Try pandas conversion
    try:
        import pandas as pd

        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values  # type: ignore[no-any-return]
    except ImportError:
        pass

    # Try list/tuple conversion
    if isinstance(data, (list, tuple)):
        return np.array(data)

    return data  # type: ignore[no-any-return]


def _wrap_with_feature_names(data: Any, state: Dict[str, Any]) -> Any:
    """Return X as a DataFrame with the preprocessor's output feature names.

    Falls back to the original `data` when the preprocessor or names are
    unavailable or the column count does not match. See the comment on
    the call site for why this matters for LightGBM 4.x.
    """
    try:
        import pandas as pd
    except ImportError:
        return data
    if data is None or isinstance(data, pd.DataFrame):
        return data
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        return data
    preprocessor = state.get("preprocessor")
    names = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = None
    if names is None or len(names) != data.shape[1]:
        return data
    return pd.DataFrame(data, columns=list(names))
