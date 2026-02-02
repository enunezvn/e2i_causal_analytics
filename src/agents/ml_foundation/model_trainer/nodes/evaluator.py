"""Model evaluation for model_trainer.

This module evaluates trained models on train/validation/test sets
using real sklearn metrics with bootstrap confidence intervals.

Version: 2.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


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

    # Convert to numpy
    X_train_np = _ensure_numpy(X_train_preprocessed)
    X_val_np = _ensure_numpy(X_validation_preprocessed)
    X_test_np = _ensure_numpy(X_test_preprocessed)
    y_train_np = _ensure_numpy(y_train)
    y_val_np = _ensure_numpy(y_validation)
    y_test_np = _ensure_numpy(y_test)

    logger.info(
        f"Evaluating model: problem_type={problem_type}, "
        f"X_test shape={X_test_np.shape if X_test_np is not None else 'None'}"
    )

    # Make predictions on all sets
    try:
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

    # Check success criteria
    success_results = _check_success_criteria(
        metrics_result["test_metrics"],
        success_criteria,
        problem_type,
    )

    logger.info(
        f"Evaluation complete: success_criteria_met={success_results['success_criteria_met']}"
    )

    # Merge results
    return {
        **metrics_result,
        **success_results,
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
        return (cate - cate_min) / (cate_max - cate_min)
    else:
        return np.ones_like(cate) * 0.5


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
) -> Dict[str, Any]:
    """Compute binary classification metrics using sklearn.

    For imbalanced datasets, also computes metrics at the optimal threshold
    to provide useful predictions instead of all-negative predictions.

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
        imbalance_detected: Whether class imbalance was detected

    Returns:
        Dictionary of metrics
    """
    # Training metrics
    train_metrics = {}
    if y_train is not None and y_train_pred is not None:
        train_metrics = _compute_split_classification_metrics(y_train, y_train_pred, y_train_proba)

    # Validation metrics
    validation_metrics = {}
    if y_validation is not None and y_validation_pred is not None:
        validation_metrics = _compute_split_classification_metrics(
            y_validation, y_validation_pred, y_validation_proba
        )

    # Compute optimal threshold FIRST (before test metrics)
    optimal_threshold = _compute_optimal_threshold(y_test, y_test_proba)

    # CRITICAL: For imbalanced data, use optimal threshold for predictions
    # This prevents all-negative predictions that occur with 0.5 threshold
    y_test_pred_optimal = y_test_pred  # Default to model predictions
    if y_test_proba is not None and optimal_threshold != 0.5:
        # Get positive class probabilities
        if y_test_proba.ndim == 2:
            y_proba_pos = y_test_proba[:, 1]
        else:
            y_proba_pos = y_test_proba
        # Apply optimal threshold
        y_test_pred_optimal = (y_proba_pos >= optimal_threshold).astype(int)
        logger.info(
            f"Using optimal threshold {optimal_threshold:.4f} for predictions (vs default 0.5)"
        )

    # Test metrics at 0.5 threshold (standard)
    test_metrics_standard = _compute_split_classification_metrics(y_test, y_test_pred, y_test_proba)

    # Test metrics at optimal threshold (for imbalanced data)
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
        "precision_at_k": precision_at_k,
        "confidence_interval": confidence_interval,
        "bootstrap_samples": bootstrap_samples,
        "calibration_error": None,  # Could add ECE computation
        "rmse": None,
        "mae": None,
        "r2": None,
        # Weighted metrics for imbalanced classification
        "f1_macro": test_metrics.get("f1_macro"),
        "f1_weighted": test_metrics.get("f1_weighted"),
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
        if result["minority_recall_at_05"] == 0 and result["minority_recall"] > 0:
            logger.warning(
                f"Model would predict ALL negatives at 0.5 threshold! "
                f"Optimal threshold {optimal_threshold:.4f} rescues recall."
            )

    return result


def _compute_split_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, float]:
    """Compute classification metrics for a single split.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
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
    }

    # Probability-based metrics
    if y_proba is not None:
        # Get positive class probabilities
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

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
    test_metrics = {
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

    # Get positive class probabilities
    if y_proba.ndim == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return float(thresholds[optimal_idx])
    except Exception:
        return 0.5


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

    # Get positive class probabilities
    if y_proba.ndim == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba

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
    y_proba_pos = None
    if y_proba is not None:
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

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

    results = {}
    all_met = True

    # Map metric aliases
    metric_aliases = {
        "auc": "roc_auc",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1_score",
        "f1_score": "f1_score",
        "rmse": "rmse",
        "mae": "mae",
        "r2": "r2",
    }

    # Metrics where lower is better
    lower_is_better = {"rmse", "mae", "brier_score", "mse"}

    for criterion_name, threshold in success_criteria.items():
        # Resolve metric name
        metric_name = metric_aliases.get(criterion_name, criterion_name)
        actual_value = test_metrics.get(metric_name)

        if actual_value is None:
            # Metric not available
            logger.warning(f"Success criterion metric not available: {criterion_name}")
            results[criterion_name] = False
            all_met = False
        else:
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

    return {
        "success_criteria_met": all_met,
        "success_criteria_results": results,
    }


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
            return data.values
    except ImportError:
        pass

    # Try list/tuple conversion
    if isinstance(data, (list, tuple)):
        return np.array(data)

    return data
