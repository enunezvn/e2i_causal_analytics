"""Model training for model_trainer.

This module trains ML models with the best hyperparameters.
Uses get_model_class from optuna_optimizer for dynamic model instantiation.

Version: 2.0.0
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)


async def train_model(state: Dict[str, Any]) -> Dict[str, Any]:
    """Train ML model with best hyperparameters.

    CRITICAL TRAINING PRINCIPLES:
    - Train ONLY on training set
    - Validation set used ONLY for early stopping (if enabled)
    - NEVER train on validation, test, or holdout
    - Test set touched ONCE for final evaluation
    - Holdout locked until post-deployment

    Args:
        state: ModelTrainerState with best_hyperparameters, preprocessed data,
               algorithm_name, problem_type, early_stopping config

    Returns:
        Dictionary with trained_model, training_duration_seconds,
        early_stopped, final_epoch, training_started_at, training_status

    Raises:
        No exceptions - returns error in state if training fails
    """
    # Extract training configuration
    algorithm_name = state.get("algorithm_name", "")
    problem_type = state.get("problem_type", "binary_classification")
    best_hyperparameters = state.get("best_hyperparameters", {})
    early_stopping = state.get("early_stopping", False)
    early_stopping_patience = state.get("early_stopping_patience", 10)

    # Check if resampling was applied - use resampled data if available
    resampling_applied = state.get("resampling_applied", False)

    if resampling_applied:
        # Use resampled training data (already preprocessed)
        X_train_preprocessed = state.get("X_train_resampled")
        y_train = state.get("y_train_resampled")
        logger.info(
            f"Using resampled training data: strategy={state.get('resampling_strategy', 'unknown')}, "
            f"original_shape={state.get('original_train_shape')}, "
            f"resampled_shape={state.get('resampled_train_shape')}"
        )
    else:
        # Use preprocessed training data (standard path)
        X_train_preprocessed = state.get("X_train_preprocessed")
        train_data = state.get("train_data", {})
        y_train = train_data.get("y")

    # Extract validation data (never resampled)
    X_validation_preprocessed = state.get("X_validation_preprocessed")
    validation_data = state.get("validation_data", {})
    y_validation = validation_data.get("y")

    # Extract feature columns for setting on model
    feature_columns = state.get("feature_columns")

    # Validate required data
    if X_train_preprocessed is None or y_train is None:
        logger.error("Missing training data for model training")
        return {
            "error": "Missing training data for model training",
            "error_type": "missing_training_data",
            "training_status": "failed",
        }

    if not algorithm_name:
        logger.error("algorithm_name not specified")
        return {
            "error": "algorithm_name not specified",
            "error_type": "missing_algorithm_name",
            "training_status": "failed",
        }

    # Record training start
    training_started_at = datetime.now(tz=timezone.utc).isoformat()
    start_time = time.time()

    logger.info(
        f"Starting model training: algorithm={algorithm_name}, "
        f"problem_type={problem_type}, "
        f"X_train shape={_get_shape(X_train_preprocessed)}"
    )

    # Get model class
    model_class = _get_model_class_dynamic(algorithm_name, problem_type)
    if model_class is None:
        logger.error(f"Could not get model class for {algorithm_name}")
        return {
            "error": f"Unsupported algorithm: {algorithm_name}",
            "error_type": "unsupported_algorithm",
            "training_status": "failed",
        }

    # Prepare hyperparameters - filter out incompatible params
    filtered_params = _filter_hyperparameters(
        algorithm_name, best_hyperparameters
    )

    # Instantiate model
    try:
        model = model_class(**filtered_params)
        logger.info(f"Instantiated {algorithm_name} with params: {list(filtered_params.keys())}")
    except Exception as e:
        logger.error(f"Model instantiation failed: {e}")
        return {
            "error": f"Model instantiation failed: {str(e)}",
            "error_type": "instantiation_failed",
            "training_status": "failed",
        }

    # Prepare fit parameters
    fit_params = _prepare_fit_params(
        algorithm_name=algorithm_name,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        X_validation=X_validation_preprocessed,
        y_validation=y_validation,
    )

    # Train the model on TRAIN ONLY
    early_stopped = False
    final_epoch = None

    try:
        # Convert to numpy if needed
        X_train_np = _ensure_numpy(X_train_preprocessed)
        y_train_np = _ensure_numpy(y_train)

        # Fit model
        model.fit(X_train_np, y_train_np, **fit_params)

        # Check if early stopping occurred (for XGBoost/LightGBM)
        if early_stopping:
            early_stopped, final_epoch = _check_early_stopping(model, algorithm_name)

        logger.info(
            f"Model training completed: early_stopped={early_stopped}, "
            f"final_epoch={final_epoch}"
        )

        # Set feature names on model for SHAP compatibility
        if feature_columns is not None and len(feature_columns) > 0:
            try:
                model.feature_names_in_ = np.array(feature_columns)
                logger.info(f"Set feature_names_in_ with {len(feature_columns)} features")
            except (AttributeError, TypeError) as e:
                # Model doesn't support this attribute - this is fine
                logger.debug(f"Could not set feature_names_in_: {e}")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {
            "error": f"Model training failed: {str(e)}",
            "error_type": "training_failed",
            "training_status": "failed",
        }

    # Record training completion
    training_duration = time.time() - start_time
    training_completed_at = datetime.now(tz=timezone.utc).isoformat()

    logger.info(f"Training completed in {training_duration:.2f}s")

    return {
        "trained_model": model,
        "training_duration_seconds": training_duration,
        "early_stopped": early_stopped,
        "final_epoch": final_epoch,
        "training_started_at": training_started_at,
        "training_completed_at": training_completed_at,
        "training_status": "completed",
        "algorithm_name": algorithm_name,
        "framework": _get_framework(algorithm_name),
    }


def _get_model_class_dynamic(
    algorithm_name: str,
    problem_type: str,
) -> Optional[Type]:
    """Get model class for algorithm and problem type.

    Uses get_model_class from optuna_optimizer if available,
    falls back to direct imports.

    Args:
        algorithm_name: Algorithm name (XGBoost, LightGBM, RandomForest, etc.)
        problem_type: Problem type (binary_classification, regression, etc.)

    Returns:
        Model class or None if not found
    """
    # Try to use optuna_optimizer's get_model_class
    try:
        from src.mlops.optuna_optimizer import get_model_class
        return get_model_class(algorithm_name, problem_type)
    except ImportError:
        pass

    # Fallback to direct imports
    is_classification = problem_type in [
        "binary_classification",
        "multiclass_classification",
    ]

    try:
        if algorithm_name == "XGBoost":
            import xgboost as xgb
            return xgb.XGBClassifier if is_classification else xgb.XGBRegressor

        elif algorithm_name == "LightGBM":
            import lightgbm as lgb
            return lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor

        elif algorithm_name == "RandomForest":
            from sklearn.ensemble import (
                RandomForestClassifier,
                RandomForestRegressor,
            )
            return RandomForestClassifier if is_classification else RandomForestRegressor

        elif algorithm_name == "ExtraTrees":
            from sklearn.ensemble import (
                ExtraTreesClassifier,
                ExtraTreesRegressor,
            )
            return ExtraTreesClassifier if is_classification else ExtraTreesRegressor

        elif algorithm_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression

        elif algorithm_name == "Ridge":
            from sklearn.linear_model import Ridge
            return Ridge

        elif algorithm_name == "Lasso":
            from sklearn.linear_model import Lasso
            return Lasso

        elif algorithm_name == "GradientBoosting":
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            )
            return GradientBoostingClassifier if is_classification else GradientBoostingRegressor

        elif algorithm_name == "SVM":
            from sklearn.svm import SVC, SVR
            return SVC if is_classification else SVR

        elif algorithm_name == "CausalForest":
            from econml.dml import CausalForestDML
            return CausalForestDML

        elif algorithm_name == "LinearDML":
            from econml.dml import LinearDML
            return LinearDML

        elif algorithm_name in ("DRLearner", "SLearner", "TLearner", "XLearner"):
            # Meta-learners share similar interface
            from econml import metalearners, dr
            mapping = {
                "DRLearner": dr.DRLearner,
                "SLearner": metalearners.SLearner,
                "TLearner": metalearners.TLearner,
                "XLearner": metalearners.XLearner,
            }
            return mapping[algorithm_name]

        else:
            logger.warning(f"Unknown algorithm: {algorithm_name}")
            return None

    except ImportError as e:
        logger.warning(f"Could not import model for {algorithm_name}: {e}")
        return None


def _filter_hyperparameters(
    algorithm_name: str,
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Filter hyperparameters to remove incompatible ones.

    Different models accept different parameters. This function
    filters out parameters that would cause errors.

    Args:
        algorithm_name: Algorithm name
        hyperparameters: Raw hyperparameters

    Returns:
        Filtered hyperparameters
    """
    # Base parameters that most sklearn models accept
    common_params = {
        "random_state": 42,
        "n_jobs": -1,
    }

    # Algorithm-specific allowed parameters
    allowed_params = {
        "XGBoost": {
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda",
            "gamma", "scale_pos_weight", "random_state", "n_jobs", "verbosity",
            "eval_metric", "early_stopping_rounds", "use_label_encoder",
        },
        "LightGBM": {
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_samples", "reg_alpha", "reg_lambda",
            "num_leaves", "random_state", "n_jobs", "verbose", "importance_type",
            "subsample_freq", "min_split_gain",
        },
        "RandomForest": {
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "max_features", "bootstrap", "random_state", "n_jobs", "class_weight",
            "max_leaf_nodes", "min_impurity_decrease", "oob_score",
        },
        "ExtraTrees": {
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "max_features", "bootstrap", "random_state", "n_jobs", "class_weight",
        },
        "LogisticRegression": {
            "C", "penalty", "solver", "max_iter", "random_state", "class_weight",
            "l1_ratio", "tol", "warm_start",
        },
        "Ridge": {
            "alpha", "fit_intercept", "solver", "random_state", "tol",
        },
        "Lasso": {
            "alpha", "fit_intercept", "max_iter", "random_state", "tol",
            "warm_start", "selection",
        },
        "GradientBoosting": {
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "min_samples_split", "min_samples_leaf", "max_features",
            "random_state", "validation_fraction", "n_iter_no_change", "tol",
        },
        "SVM": {
            "C", "kernel", "degree", "gamma", "coef0", "shrinking",
            "probability", "tol", "cache_size", "class_weight", "random_state",
        },
        "CausalForest": {
            "n_estimators", "max_depth", "min_samples_leaf", "min_samples_split",
            "max_features", "inference", "n_jobs", "random_state",
            "model_y", "model_t", "discrete_treatment", "cv",
        },
        "LinearDML": {
            "model_y", "model_t", "discrete_treatment", "cv", "mc_iters",
            "random_state", "linear_first_stages",
        },
        "DRLearner": {
            "model_propensity", "model_regression", "model_final",
            "cv", "mc_iters", "random_state", "n_jobs",
        },
        "SLearner": {"overall_model", "cv", "random_state"},
        "TLearner": {"models", "cv", "random_state"},
        "XLearner": {"models", "propensity_model", "cate_models", "cv", "random_state"},
    }

    # Get allowed params for this algorithm
    allowed = allowed_params.get(algorithm_name, set())

    # Filter hyperparameters
    filtered = {}
    for key, value in hyperparameters.items():
        if key in allowed:
            filtered[key] = value

    # Add common params if not already set
    for key, value in common_params.items():
        if key in allowed and key not in filtered:
            filtered[key] = value

    # Algorithm-specific defaults
    if algorithm_name == "XGBoost":
        if "verbosity" not in filtered:
            filtered["verbosity"] = 0
        if "use_label_encoder" not in filtered:
            filtered["use_label_encoder"] = False
    elif algorithm_name == "LightGBM":
        if "verbose" not in filtered:
            filtered["verbose"] = -1
    elif algorithm_name == "LogisticRegression":
        if "max_iter" not in filtered:
            filtered["max_iter"] = 1000

    return filtered


def _prepare_fit_params(
    algorithm_name: str,
    early_stopping: bool,
    early_stopping_patience: int,
    X_validation: Any,
    y_validation: Any,
) -> Dict[str, Any]:
    """Prepare fit parameters for training.

    Handles early stopping for XGBoost/LightGBM.

    Args:
        algorithm_name: Algorithm name
        early_stopping: Whether to use early stopping
        early_stopping_patience: Early stopping patience
        X_validation: Validation features
        y_validation: Validation labels

    Returns:
        Dictionary of fit parameters
    """
    fit_params = {}

    if not early_stopping:
        return fit_params

    if X_validation is None or y_validation is None:
        logger.warning("Early stopping enabled but no validation data available")
        return fit_params

    # Convert validation data to numpy
    X_val_np = _ensure_numpy(X_validation)
    y_val_np = _ensure_numpy(y_validation)

    if algorithm_name == "XGBoost":
        fit_params["eval_set"] = [(X_val_np, y_val_np)]
        fit_params["verbose"] = False
        # Note: early_stopping_rounds is set in model params for newer XGBoost

    elif algorithm_name == "LightGBM":
        fit_params["eval_set"] = [(X_val_np, y_val_np)]
        fit_params["callbacks"] = [
            _get_lgbm_early_stopping_callback(early_stopping_patience)
        ]

    elif algorithm_name == "GradientBoosting":
        # sklearn GradientBoosting uses validation_fraction and n_iter_no_change
        # These are set in model params, not fit params
        pass

    return fit_params


def _get_lgbm_early_stopping_callback(patience: int):
    """Get LightGBM early stopping callback.

    Args:
        patience: Number of rounds without improvement

    Returns:
        LightGBM callback
    """
    try:
        import lightgbm as lgb
        return lgb.early_stopping(stopping_rounds=patience, verbose=False)
    except (ImportError, AttributeError):
        return None


def _check_early_stopping(model: Any, algorithm_name: str) -> tuple:
    """Check if early stopping occurred.

    Args:
        model: Trained model
        algorithm_name: Algorithm name

    Returns:
        Tuple of (early_stopped, final_epoch)
    """
    early_stopped = False
    final_epoch = None

    if algorithm_name == "XGBoost":
        # XGBoost stores best iteration
        if hasattr(model, "best_iteration"):
            best_iter = model.best_iteration
            if best_iter is not None and best_iter > 0:
                n_estimators = getattr(model, "n_estimators", None)
                if n_estimators and best_iter < n_estimators - 1:
                    early_stopped = True
                    final_epoch = best_iter

    elif algorithm_name == "LightGBM":
        # LightGBM stores best iteration
        if hasattr(model, "best_iteration_"):
            best_iter = model.best_iteration_
            if best_iter is not None and best_iter > 0:
                n_estimators = getattr(model, "n_estimators", None)
                if n_estimators and best_iter < n_estimators - 1:
                    early_stopped = True
                    final_epoch = best_iter

    elif algorithm_name == "GradientBoosting":
        # sklearn stores n_iter_ for early stopping
        if hasattr(model, "n_iter_"):
            final_epoch = model.n_iter_
            n_estimators = getattr(model, "n_estimators", None)
            if n_estimators and final_epoch < n_estimators:
                early_stopped = True

    return early_stopped, final_epoch


def _ensure_numpy(data: Any) -> np.ndarray:
    """Convert data to numpy array if needed.

    Args:
        data: Input data

    Returns:
        Numpy array
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


def _get_shape(data: Any) -> str:
    """Get shape string for data.

    Args:
        data: Input data

    Returns:
        Shape string
    """
    if data is None:
        return "None"
    if hasattr(data, "shape"):
        return str(data.shape)
    if hasattr(data, "__len__"):
        return f"({len(data)},)"
    return "unknown"


def _get_framework(algorithm_name: str) -> str:
    """Get framework name for algorithm.

    Args:
        algorithm_name: Algorithm name

    Returns:
        Framework name
    """
    framework_map = {
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "RandomForest": "sklearn",
        "ExtraTrees": "sklearn",
        "LogisticRegression": "sklearn",
        "Ridge": "sklearn",
        "Lasso": "sklearn",
        "GradientBoosting": "sklearn",
        "SVM": "sklearn",
        "CausalForest": "econml",
        "LinearDML": "econml",
        "SLearner": "econml",
    }
    return framework_map.get(algorithm_name, "unknown")
