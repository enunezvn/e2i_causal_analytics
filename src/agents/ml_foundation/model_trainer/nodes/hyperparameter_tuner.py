"""Hyperparameter tuning for model_trainer.

This module uses Optuna to optimize hyperparameters on the validation set.
Uses the OptunaOptimizer from src.mlops for unified HPO management.

Features:
- Validation-based HPO (no test set leakage)
- Warm-starting from similar successful patterns
- Pattern storage for procedural memory
- Opik instrumentation for observability

Version: 2.3.0
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _get_opik_connector():
    """Lazy import of OpikConnector to avoid circular imports."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except ImportError:
        logger.debug("OpikConnector not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get OpikConnector: {e}")
        return None


def _get_hpo_pattern_memory():
    """Lazy import of HPO pattern memory module."""
    try:
        from src.mlops import hpo_pattern_memory

        return hpo_pattern_memory
    except ImportError:
        logger.debug("HPO pattern memory not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get HPO pattern memory: {e}")
        return None


# HPO output contract validation
HPO_REQUIRED_FIELDS = {
    "hpo_completed": bool,
    "best_hyperparameters": dict,
}

HPO_OPTIONAL_FIELDS = {
    "hpo_best_trial": (int, type(None)),
    "hpo_best_value": (float, int, type(None)),
    "hpo_trials_run": int,
    "hpo_trials_pruned": int,
    "hpo_duration_seconds": (float, int),
    "hpo_study_name": (str, type(None)),
    "hpo_metric": (str, type(None)),
    "hpo_pattern_id": (str, type(None)),  # Procedural memory pattern ID
}


def validate_hpo_output(output: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate HPO output against contract.

    This ensures the HPO output contains all required fields with correct types.

    Args:
        output: HPO output dictionary

    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []

    # Check required fields
    for field, expected_type in HPO_REQUIRED_FIELDS.items():
        if field not in output:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(output[field], expected_type):
            errors.append(
                f"Invalid type for {field}: expected {expected_type.__name__}, "
                f"got {type(output[field]).__name__}"
            )

    # Check optional fields if present
    for field, expected_types in HPO_OPTIONAL_FIELDS.items():
        if field in output and output[field] is not None:
            if not isinstance(output[field], expected_types):
                type_names = (
                    expected_types.__name__
                    if not isinstance(expected_types, tuple)
                    else "/".join(t.__name__ for t in expected_types)
                )
                errors.append(
                    f"Invalid type for {field}: expected {type_names}, "
                    f"got {type(output[field]).__name__}"
                )

    # Validate hpo_completed consistency
    if output.get("hpo_completed"):
        # When HPO completed, these fields should be present
        if output.get("hpo_trials_run", 0) == 0:
            errors.append("hpo_completed=True but hpo_trials_run=0")
        if "hpo_best_value" not in output:
            errors.append("hpo_completed=True but hpo_best_value missing")
        if "hpo_study_name" not in output:
            errors.append("hpo_completed=True but hpo_study_name missing")

    return (len(errors) == 0, errors)


def validate_hyperparameter_types(
    hyperparameters: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Validate hyperparameter types match search space definitions.

    Args:
        hyperparameters: Best hyperparameters from HPO
        search_space: Search space definitions from model_selector

    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []

    for param_name, param_def in search_space.items():
        if param_name not in hyperparameters:
            # This may be acceptable if using defaults
            continue

        value = hyperparameters[param_name]
        param_type = param_def.get("type", "float")

        # Validate type
        if param_type == "int":
            if not isinstance(value, (int, np.integer)):
                errors.append(
                    f"Parameter {param_name}: expected int, got {type(value).__name__}"
                )
        elif param_type == "float":
            if not isinstance(value, (int, float, np.floating)):
                errors.append(
                    f"Parameter {param_name}: expected float, got {type(value).__name__}"
                )
        elif param_type == "categorical":
            choices = param_def.get("choices", [])
            if choices and value not in choices:
                errors.append(
                    f"Parameter {param_name}: value {value} not in choices {choices}"
                )

        # Validate range for numeric types
        if param_type in ("int", "float") and value is not None:
            low = param_def.get("low")
            high = param_def.get("high")
            if low is not None and value < low:
                errors.append(
                    f"Parameter {param_name}: value {value} below minimum {low}"
                )
            if high is not None and value > high:
                errors.append(
                    f"Parameter {param_name}: value {value} above maximum {high}"
                )

    return (len(errors) == 0, errors)


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

        # Try to get warm-start hyperparameters from procedural memory
        warmstart_config = None
        hpo_memory = _get_hpo_pattern_memory()
        if hpo_memory:
            try:
                n_samples = len(X_train) if X_train is not None else None
                n_features = X_train.shape[1] if X_train is not None and len(X_train.shape) > 1 else None
                warmstart_config = await hpo_memory.get_warmstart_hyperparameters(
                    algorithm_name=algorithm_name,
                    problem_type=problem_type,
                    n_samples=n_samples,
                    n_features=n_features,
                    metric=metric,
                    min_similarity=0.6,
                )
                if warmstart_config:
                    logger.info(
                        f"Using warm-start from pattern {warmstart_config.pattern_id[:8]} "
                        f"(similarity={warmstart_config.similarity_score:.2f})"
                    )
            except Exception as e:
                logger.debug(f"Warm-start retrieval failed (non-critical): {e}")

        # Create study with median pruner
        study = await optimizer.create_study(
            study_name=f"{algorithm_name}_hpo",
            direction="maximize",
            pruner=PrunerFactory.median_pruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ),
        )

        # Enqueue warm-start hyperparameters as initial trial if available
        if warmstart_config:
            try:
                # Filter to only hyperparameters in the current search space
                warmstart_params = {
                    k: v
                    for k, v in warmstart_config.initial_hyperparameters.items()
                    if k in hyperparameter_search_space
                }
                if warmstart_params:
                    study.enqueue_trial(warmstart_params)
                    logger.info(f"Enqueued warm-start trial with {len(warmstart_params)} params")
            except Exception as e:
                logger.debug(f"Failed to enqueue warm-start trial: {e}")

        # Get fixed parameters (random_state, n_jobs, etc.)
        fixed_params = _get_fixed_params(algorithm_name)

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
            fixed_params=fixed_params,
        )

        # Run optimization with Opik tracing
        opik = _get_opik_connector()
        hpo_start_time = time.time()

        if opik and opik.is_enabled:
            async with opik.trace_agent(
                agent_name="model_trainer",
                operation="hyperparameter_optimization",
                metadata={
                    "algorithm_name": algorithm_name,
                    "problem_type": problem_type,
                    "n_trials": hpo_trials,
                    "timeout_seconds": timeout_seconds,
                    "metric": metric,
                    "search_space_params": list(hyperparameter_search_space.keys()),
                },
                tags=["hpo", "optuna", algorithm_name.lower()],
            ) as hpo_span:
                results = await optimizer.optimize(
                    study=study,
                    objective=objective,
                    n_trials=hpo_trials,
                    timeout=timeout_seconds,
                )
                # Log HPO metrics to Opik
                hpo_span.set_attribute("best_value", results["best_value"])
                hpo_span.set_attribute("n_trials_completed", results["n_completed"])
                hpo_span.set_attribute("n_trials_pruned", results["n_pruned"])
                hpo_span.set_attribute("duration_seconds", time.time() - hpo_start_time)
        else:
            # No Opik tracing available
            results = await optimizer.optimize(
                study=study,
                objective=objective,
                n_trials=hpo_trials,
                timeout=timeout_seconds,
            )

        # Merge: defaults < optuna best < fixed params (fixed params have highest priority)
        best_hyperparameters = {
            **default_hyperparameters,
            **results["best_params"],
            **fixed_params,
        }

        logger.info(
            f"HPO completed: best_value={results['best_value']:.4f}, "
            f"trials={results['n_trials']} ({results['n_pruned']} pruned)"
        )

        hpo_output = {
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

        # Validate output against contract
        is_valid, validation_errors = validate_hpo_output(hpo_output)
        if not is_valid:
            logger.warning(f"HPO output validation warnings: {validation_errors}")
            hpo_output["validation_warnings"] = validation_errors

        # Validate hyperparameter types
        if hyperparameter_search_space:
            hp_valid, hp_errors = validate_hyperparameter_types(
                best_hyperparameters, hyperparameter_search_space
            )
            if not hp_valid:
                logger.warning(f"Hyperparameter type warnings: {hp_errors}")
                hpo_output["hyperparameter_warnings"] = hp_errors

        # Store successful HPO pattern for future warm-starting
        if hpo_memory and results["best_value"] is not None:
            try:
                from src.mlops.hpo_pattern_memory import HPOPatternInput

                pattern = HPOPatternInput(
                    algorithm_name=algorithm_name,
                    problem_type=problem_type,
                    search_space=hyperparameter_search_space,
                    best_hyperparameters=results["best_params"],
                    best_value=results["best_value"],
                    optimization_metric=metric,
                    n_trials=results["n_trials"],
                    n_completed=results["n_completed"],
                    n_pruned=results["n_pruned"],
                    duration_seconds=results["duration_seconds"],
                    study_name=results["study_name"],
                    n_samples=len(X_train) if X_train is not None else None,
                    n_features=X_train.shape[1] if X_train is not None and len(X_train.shape) > 1 else None,
                    experiment_id=experiment_id,
                )

                pattern_id = await hpo_memory.store_hpo_pattern(pattern)
                if pattern_id:
                    hpo_output["hpo_pattern_id"] = pattern_id
                    logger.info(f"Stored HPO pattern: {pattern_id[:8]}")

                # Record warm-start outcome if we used one
                if warmstart_config:
                    await hpo_memory.record_warmstart_outcome(
                        pattern_id=warmstart_config.pattern_id,
                        new_best_value=results["best_value"],
                        original_best_value=warmstart_config.original_best_value,
                    )
            except Exception as e:
                logger.debug(f"Pattern storage failed (non-critical): {e}")

        return hpo_output

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
