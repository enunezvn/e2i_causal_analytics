"""Model evaluation for model_trainer.

This module evaluates trained models on train/validation/test sets.
"""

from typing import Any, Dict


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
        return {
            "error": "No trained model available for evaluation",
            "error_type": "missing_trained_model",
        }

    if X_test_preprocessed is None or y_test is None:
        return {
            "error": "Missing test data for evaluation",
            "error_type": "missing_test_data",
        }

    # Make predictions on all sets
    try:
        # Training set
        if X_train_preprocessed is not None and y_train is not None:
            y_train_pred = trained_model.predict(X_train_preprocessed)
            if problem_type in ["binary_classification", "multiclass_classification"]:
                if hasattr(trained_model, "predict_proba"):
                    y_train_proba = trained_model.predict_proba(X_train_preprocessed)
                else:
                    y_train_proba = None
            else:
                y_train_proba = None
        else:
            y_train_pred = None
            y_train_proba = None

        # Validation set
        if X_validation_preprocessed is not None and y_validation is not None:
            y_validation_pred = trained_model.predict(X_validation_preprocessed)
            if problem_type in ["binary_classification", "multiclass_classification"]:
                if hasattr(trained_model, "predict_proba"):
                    y_validation_proba = trained_model.predict_proba(X_validation_preprocessed)
                else:
                    y_validation_proba = None
            else:
                y_validation_proba = None
        else:
            y_validation_pred = None
            y_validation_proba = None

        # Test set (FINAL EVALUATION)
        y_test_pred = trained_model.predict(X_test_preprocessed)
        if problem_type in ["binary_classification", "multiclass_classification"]:
            if hasattr(trained_model, "predict_proba"):
                y_test_proba = trained_model.predict_proba(X_test_preprocessed)
            else:
                y_test_proba = None
        else:
            y_test_proba = None

    except Exception as e:
        return {
            "error": f"Prediction failed during evaluation: {str(e)}",
            "error_type": "prediction_failed",
        }

    # Compute metrics based on problem type
    if problem_type == "binary_classification":
        metrics_result = _compute_classification_metrics(
            y_train,
            y_train_pred,
            y_train_proba,
            y_validation,
            y_validation_pred,
            y_validation_proba,
            y_test,
            y_test_pred,
            y_test_proba,
        )
    elif problem_type in ["regression", "continuous"]:
        metrics_result = _compute_regression_metrics(
            y_train,
            y_train_pred,
            y_validation,
            y_validation_pred,
            y_test,
            y_test_pred,
        )
    else:
        return {
            "error": f"Unsupported problem type: {problem_type}",
            "error_type": "unsupported_problem_type",
        }

    # Check success criteria
    success_results = _check_success_criteria(metrics_result["test_metrics"], success_criteria)

    # Merge results
    return {
        **metrics_result,
        **success_results,
    }


def _compute_classification_metrics(
    y_train,
    y_train_pred,
    y_train_proba,
    y_validation,
    y_validation_pred,
    y_validation_proba,
    y_test,
    y_test_pred,
    y_test_proba,
) -> Dict[str, Any]:
    """Compute classification metrics."""
    # TODO: Implement real metrics using sklearn
    # from sklearn.metrics import (
    #     roc_auc_score, precision_score, recall_score, f1_score,
    #     average_precision_score, confusion_matrix, brier_score_loss
    # )

    # PLACEHOLDER: Mock metrics
    train_metrics = {"accuracy": 0.85, "loss": 0.35} if y_train_pred is not None else {}
    validation_metrics = {"accuracy": 0.82, "loss": 0.42} if y_validation_pred is not None else {}
    test_metrics = {"accuracy": 0.80, "loss": 0.45}

    # Problem-specific metrics for classification
    auc_roc = 0.78  # TODO: Compute from y_test_proba
    precision = 0.75  # TODO: Compute from y_test, y_test_pred
    recall = 0.72  # TODO: Compute from y_test, y_test_pred
    f1_score = 0.73  # TODO: Compute from precision, recall
    pr_auc = 0.76  # TODO: Compute from y_test_proba
    brier_score = 0.18  # TODO: Compute from y_test_proba

    # Confusion matrix
    confusion_matrix = {
        "TP": 145,
        "TN": 132,
        "FP": 28,
        "FN": 35,
    }  # TODO: Compute from y_test, y_test_pred

    # Optimal threshold
    optimal_threshold = 0.5  # TODO: Compute using ROC curve

    # Precision at k
    precision_at_k = {100: 0.85, 500: 0.72, 1000: 0.65}  # TODO: Compute

    # Confidence intervals via bootstrapping
    # TODO: Implement bootstrap resampling for CI
    confidence_interval = {
        "auc": (0.75, 0.81),
        "precision": (0.71, 0.79),
        "recall": (0.68, 0.76),
    }
    bootstrap_samples = 1000

    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "pr_auc": pr_auc,
        "brier_score": brier_score,
        "confusion_matrix": confusion_matrix,
        "optimal_threshold": optimal_threshold,
        "precision_at_k": precision_at_k,
        "confidence_interval": confidence_interval,
        "bootstrap_samples": bootstrap_samples,
        "calibration_error": None,  # TODO: Compute ECE
    }


def _compute_regression_metrics(
    y_train, y_train_pred, y_validation, y_validation_pred, y_test, y_test_pred
) -> Dict[str, Any]:
    """Compute regression metrics."""
    # TODO: Implement real metrics using sklearn
    # from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # PLACEHOLDER: Mock metrics
    train_metrics = {"mse": 0.12, "mae": 0.28, "r2": 0.78} if y_train_pred is not None else {}
    validation_metrics = (
        {"mse": 0.15, "mae": 0.31, "r2": 0.74} if y_validation_pred is not None else {}
    )
    test_metrics = {"mse": 0.16, "mae": 0.33, "r2": 0.72}

    # Problem-specific metrics for regression
    rmse = 0.40  # TODO: sqrt(mse)
    mae = 0.33  # TODO: Compute from y_test, y_test_pred
    r2 = 0.72  # TODO: Compute from y_test, y_test_pred

    # Confidence intervals
    confidence_interval = {
        "rmse": (0.37, 0.43),
        "mae": (0.30, 0.36),
        "r2": (0.68, 0.76),
    }
    bootstrap_samples = 1000

    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "confidence_interval": confidence_interval,
        "bootstrap_samples": bootstrap_samples,
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


def _check_success_criteria(
    test_metrics: Dict[str, float], success_criteria: Dict[str, float]
) -> Dict[str, Any]:
    """Check if model meets success criteria."""
    if not success_criteria:
        # No criteria defined
        return {
            "success_criteria_met": True,
            "success_criteria_results": {},
        }

    results = {}
    all_met = True

    for metric_name, threshold in success_criteria.items():
        actual_value = test_metrics.get(metric_name)
        if actual_value is None:
            # Metric not available
            results[metric_name] = False
            all_met = False
        else:
            # Check if metric meets threshold
            met = actual_value >= threshold
            results[metric_name] = met
            if not met:
                all_met = False

    return {
        "success_criteria_met": all_met,
        "success_criteria_results": results,
    }
