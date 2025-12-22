"""Hyperparameter tuning for model_trainer.

This module uses Optuna to optimize hyperparameters on the validation set.
Uses the OptunaOptimizer from src.mlops for unified HPO management.

Version: 2.0.0
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


async def tune_hyperparameters(state: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna on validation set.

    CRITICAL: Hyperparameter tuning uses VALIDATION set only
    - NEVER use test set for tuning
    - Test set is reserved for final evaluation
    - Validation set is for hyperparameter selection and early stopping

    Args:
        state: ModelTrainerState with enable_hpo, hpo_trials,
               hyperparameter_search_space, validation data

    Returns:
        Dictionary with hpo_completed, best_hyperparameters,
        hpo_best_trial, hpo_trials_run, hpo_duration_seconds

    Raises:
        No exceptions - returns error in state if HPO fails
    """
    # Check if HPO is enabled
    enable_hpo = state.get("enable_hpo", False)
    default_hyperparameters = state.get("default_hyperparameters", {})

    if not enable_hpo:
        # HPO disabled, return default hyperparameters
        logger.info("HPO disabled, using default hyperparameters")
        return {
            "hpo_completed": False,
            "hpo_best_trial": None,
            "best_hyperparameters": default_hyperparameters,
            "hpo_trials_run": 0,
            "hpo_duration_seconds": 0.0,
        }

    # Extract HPO configuration
    hpo_trials = state.get("hpo_trials", 50)
    hpo_timeout_hours = state.get("hpo_timeout_hours")
    hyperparameter_search_space = state.get("hyperparameter_search_space", {})
    algorithm_name = state.get("algorithm_name", "")
    problem_type = state.get("problem_type", "binary_classification")
    experiment_id = state.get("experiment_id", "unknown")

    # Extract preprocessed data
    X_train_preprocessed = state.get("X_train_preprocessed")
    X_validation_preprocessed = state.get("X_validation_preprocessed")
    train_data = state.get("train_data", {})
    validation_data = state.get("validation_data", {})
    y_train = train_data.get("y")
    y_validation = validation_data.get("y")

    # Validate required data
    if X_train_preprocessed is None or y_train is None:
        logger.error("Missing training data for HPO")
        return {
            "error": "Missing training data for HPO",
            "error_type": "missing_hpo_data",
            "hpo_completed": False,
            "best_hyperparameters": default_hyperparameters,
        }

    if X_validation_preprocessed is None or y_validation is None:
        logger.error("Missing validation data for HPO")
        return {
            "error": "Missing validation data for HPO",
            "error_type": "missing_hpo_data",
            "hpo_completed": False,
            "best_hyperparameters": default_hyperparameters,
        }

    if not hyperparameter_search_space:
        # No search space defined, return defaults
        logger.info("No search space defined, using default hyperparameters")
        return {
            "hpo_completed": False,
            "hpo_best_trial": None,
            "best_hyperparameters": default_hyperparameters,
            "hpo_trials_run": 0,
            "hpo_duration_seconds": 0.0,
        }

    # Convert to numpy arrays if needed
    X_train = _ensure_numpy(X_train_preprocessed)
    X_val = _ensure_numpy(X_validation_preprocessed)
    y_train_np = _ensure_numpy(y_train)
    y_val_np = _ensure_numpy(y_validation)

    # Determine optimization metric based on problem type
    metric = _get_default_metric(problem_type)

    # Calculate timeout in seconds
    timeout_seconds = int(hpo_timeout_hours * 3600) if hpo_timeout_hours else 3600

    try:
        # Import OptunaOptimizer
        from src.mlops.optuna_optimizer import (
            OptunaOptimizer,
            PrunerFactory,
            get_model_class,
        )

        logger.info(
            f"Starting HPO for {algorithm_name}: "
            f"{hpo_trials} trials, {timeout_seconds}s timeout"
        )

        # Get model class
        model_class = get_model_class(algorithm_name, problem_type)
        if model_class is None:
            logger.warning(f"Could not get model class for {algorithm_name}")
            return {
                "error": f"Unsupported algorithm for HPO: {algorithm_name}",
                "error_type": "unsupported_algorithm",
                "hpo_completed": False,
                "best_hyperparameters": default_hyperparameters,
            }

        # Create optimizer with MLflow tracking
        optimizer = OptunaOptimizer(
            experiment_id=experiment_id,
            mlflow_tracking=True,
        )

        # Create study with median pruner
        study = await optimizer.create_study(
            study_name=f"{algorithm_name}_hpo",
            direction="maximize",
            pruner=PrunerFactory.median_pruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ),
        )

        # Create validation-based objective function
        objective = optimizer.create_validation_objective(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train_np,
            X_val=X_val,
            y_val=y_val_np,
            search_space=hyperparameter_search_space,
            problem_type=problem_type,
            metric=metric,
            fixed_params=_get_fixed_params(algorithm_name),
        )

        # Run optimization
        results = await optimizer.optimize(
            study=study,
            objective=objective,
            n_trials=hpo_trials,
            timeout=timeout_seconds,
        )

        # Merge best params with defaults (for params not in search space)
        best_hyperparameters = {**default_hyperparameters, **results["best_params"]}

        logger.info(
            f"HPO completed: best_value={results['best_value']:.4f}, "
            f"trials={results['n_trials']} ({results['n_pruned']} pruned)"
        )

        return {
            "hpo_completed": True,
            "hpo_best_trial": results["best_trial_number"],
            "best_hyperparameters": best_hyperparameters,
            "hpo_trials_run": results["n_completed"],
            "hpo_trials_pruned": results["n_pruned"],
            "hpo_duration_seconds": results["duration_seconds"],
            "hpo_best_value": results["best_value"],
            "hpo_study_name": results["study_name"],
            "hpo_metric": metric,
        }

    except ImportError as e:
        logger.warning(f"Optuna not available, using default hyperparameters: {e}")
        return {
            "error": f"Optuna import failed: {e}",
            "error_type": "optuna_not_available",
            "hpo_completed": False,
            "best_hyperparameters": default_hyperparameters,
            "hpo_trials_run": 0,
            "hpo_duration_seconds": 0.0,
        }

    except Exception as e:
        logger.error(f"HPO failed: {e}")
        return {
            "error": f"HPO failed: {e}",
            "error_type": "hpo_error",
            "hpo_completed": False,
            "best_hyperparameters": default_hyperparameters,
            "hpo_trials_run": 0,
            "hpo_duration_seconds": 0.0,
        }


def _ensure_numpy(data: Any) -> np.ndarray:
    """Convert data to numpy array if needed.

    Args:
        data: Input data (numpy array, pandas DataFrame/Series, list)

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

    # Return as-is and hope for the best
    return data


def _get_default_metric(problem_type: str) -> str:
    """Get default optimization metric for problem type.

    Args:
        problem_type: Problem type string

    Returns:
        Metric name for optimization
    """
    if problem_type == "binary_classification":
        return "roc_auc"
    elif problem_type == "multiclass_classification":
        return "f1"
    elif problem_type == "regression":
        return "rmse"
    else:
        return "roc_auc"


def _get_fixed_params(algorithm_name: str) -> Dict[str, Any]:
    """Get fixed parameters for algorithm that shouldn't be tuned.

    Args:
        algorithm_name: Algorithm name

    Returns:
        Dictionary of fixed parameters
    """
    fixed_params = {}

    if algorithm_name == "XGBoost":
        fixed_params = {
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
    elif algorithm_name == "LightGBM":
        fixed_params = {
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    elif algorithm_name == "RandomForest":
        fixed_params = {
            "random_state": 42,
            "n_jobs": -1,
        }
    elif algorithm_name == "LogisticRegression":
        fixed_params = {
            "random_state": 42,
            "max_iter": 1000,
        }
    elif algorithm_name in ["Ridge", "Lasso"]:
        fixed_params = {
            "random_state": 42,
        }

    return fixed_params
