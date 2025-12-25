"""MLflow experiment tracking for model_trainer.

This module logs training runs to MLflow, including:
- Hyperparameters and configuration
- Training and evaluation metrics
- Trained model artifacts
- Model registration

It also persists training runs to the database with HPO linkage
for complete traceability between Optuna studies and training runs.

Version: 1.1.0
"""

import logging
import tempfile
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


def _get_training_run_repository():
    """Lazy import of MLTrainingRunRepository to avoid circular imports."""
    try:
        from src.repositories.ml_experiment import MLTrainingRunRepository

        return MLTrainingRunRepository()
    except ImportError:
        logger.debug("MLTrainingRunRepository not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get MLTrainingRunRepository: {e}")
        return None


async def log_to_mlflow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Log training run to MLflow.

    This node logs the complete training run including:
    - Parameters: algorithm, hyperparameters, problem type
    - Metrics: all train/validation/test metrics
    - Model: trained model artifact
    - Tags: metadata for filtering runs

    Args:
        state: ModelTrainerState with trained_model, metrics,
               best_hyperparameters, algorithm_name, etc.

    Returns:
        Dictionary with mlflow_run_id, mlflow_experiment_id,
        mlflow_model_uri, mlflow_registered, mlflow_status
    """
    # Check if MLflow logging is enabled
    enable_mlflow = state.get("enable_mlflow", True)

    if not enable_mlflow:
        logger.info("MLflow logging disabled")
        return {
            "mlflow_status": "disabled",
            "mlflow_run_id": None,
            "mlflow_experiment_id": None,
        }

    # Check if we have a trained model
    trained_model = state.get("trained_model")
    if trained_model is None:
        logger.warning("No trained model to log to MLflow")
        return {
            "mlflow_status": "skipped",
            "mlflow_run_id": None,
            "error": "No trained model available",
        }

    # Extract state values
    experiment_id = state.get("experiment_id", "unknown")
    experiment_name = state.get("experiment_name", f"model_trainer_{experiment_id}")
    algorithm_name = state.get("algorithm_name", "unknown")
    problem_type = state.get("problem_type", "binary_classification")
    framework = state.get("framework", _get_framework(algorithm_name))
    best_hyperparameters = state.get("best_hyperparameters", {})

    # Training metadata
    training_duration = state.get("training_duration_seconds", 0)
    early_stopped = state.get("early_stopped", False)
    final_epoch = state.get("final_epoch")

    # HPO metadata
    hpo_completed = state.get("hpo_completed", False)
    hpo_best_value = state.get("hpo_best_value")
    hpo_trials_run = state.get("hpo_trials_run", 0)
    hpo_study_name = state.get("hpo_study_name")  # Optuna study name for linkage
    hpo_best_trial = state.get("hpo_best_trial")  # Best trial number

    # Evaluation metrics
    evaluation_metrics = state.get("evaluation_metrics", {})
    train_metrics = evaluation_metrics.get("train_metrics", {})
    validation_metrics = evaluation_metrics.get("validation_metrics", {})
    test_metrics = evaluation_metrics.get("test_metrics", {})
    holdout_metrics = evaluation_metrics.get("holdout_metrics", {})

    # Model registration config
    register_model = state.get("register_model", False)
    model_name = state.get("model_name", f"{algorithm_name.lower()}_model")
    model_description = state.get("model_description", "")
    model_tags = state.get("model_tags", {})

    # Training data metadata (for database persistence)
    training_samples = state.get("training_samples", 0)
    validation_samples = state.get("validation_samples")
    test_samples = state.get("test_samples")
    feature_names = state.get("feature_names", [])

    try:
        from src.mlops.mlflow_connector import get_mlflow_connector

        mlflow_conn = get_mlflow_connector()

        # Create/get experiment
        mlflow_experiment_id = await mlflow_conn.get_or_create_experiment(
            name=experiment_name,
            tags={
                "problem_type": problem_type,
                "framework": framework,
                "source": "model_trainer_agent",
            },
        )

        logger.info(f"MLflow experiment: {experiment_name} (ID: {mlflow_experiment_id})")

        # Generate run name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{algorithm_name}_{timestamp}"

        # Start MLflow run
        async with mlflow_conn.start_run(
            experiment_id=mlflow_experiment_id,
            run_name=run_name,
            tags={
                "algorithm": algorithm_name,
                "problem_type": problem_type,
                "framework": framework,
                "source": "model_trainer_agent",
                "hpo_enabled": str(hpo_completed),
            },
            description=f"Training run for {algorithm_name} on {problem_type}",
        ) as run:
            mlflow_run_id = run.run_id

            # Log hyperparameters
            await _log_hyperparameters(run, best_hyperparameters, algorithm_name)

            # Log training configuration
            await run.log_params({
                "algorithm_name": algorithm_name,
                "problem_type": problem_type,
                "framework": framework,
                "hpo_enabled": hpo_completed,
                "hpo_trials": hpo_trials_run,
                "early_stopping_enabled": early_stopped,
            })

            # Log training metrics
            await _log_training_metrics(run, state)

            # Log evaluation metrics for each split
            await _log_split_metrics(run, "train", train_metrics)
            await _log_split_metrics(run, "validation", validation_metrics)
            await _log_split_metrics(run, "test", test_metrics)
            if holdout_metrics:
                await _log_split_metrics(run, "holdout", holdout_metrics)

            # Log primary metric for easy comparison
            primary_metric = _get_primary_metric(
                test_metrics or validation_metrics, problem_type
            )
            if primary_metric:
                await run.log_metrics({"primary_metric": primary_metric})

            # Log model artifact
            model_uri = await _log_model_artifact(
                run, trained_model, algorithm_name, framework
            )

            # Log additional artifacts
            await _log_additional_artifacts(run, state)

            metric_str = f"{primary_metric:.4f}" if primary_metric else "N/A"
            logger.info(
                f"Logged run to MLflow: run_id={mlflow_run_id}, "
                f"primary_metric={metric_str}"
            )

        # Register model if requested
        model_version = None
        if register_model and model_uri:
            model_version = await mlflow_conn.register_model(
                run_id=mlflow_run_id,
                model_name=model_name,
                model_path="model",
                description=model_description or f"{algorithm_name} model for {problem_type}",
                tags={
                    **model_tags,
                    "algorithm": algorithm_name,
                    "problem_type": problem_type,
                },
            )
            if model_version:
                logger.info(
                    f"Registered model: {model_name} v{model_version.version}"
                )

        # Persist training run to database with HPO linkage
        db_run_id = await _persist_training_run(
            experiment_id=experiment_id,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            algorithm_name=algorithm_name,
            hyperparameters=best_hyperparameters,
            training_samples=training_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            feature_names=feature_names,
            hpo_study_name=hpo_study_name,
            hpo_best_trial=hpo_best_trial,
        )

        return {
            "mlflow_status": "success",
            "mlflow_run_id": mlflow_run_id,
            "mlflow_experiment_id": mlflow_experiment_id,
            "mlflow_model_uri": model_uri,
            "mlflow_registered": model_version is not None,
            "mlflow_model_version": model_version.version if model_version else None,
            "mlflow_model_name": model_name if model_version else None,
            "db_training_run_id": str(db_run_id) if db_run_id else None,
        }

    except ImportError as e:
        logger.warning(f"MLflow not available: {e}")
        return {
            "mlflow_status": "unavailable",
            "error": f"MLflow import failed: {e}",
            "mlflow_run_id": None,
        }

    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")
        return {
            "mlflow_status": "failed",
            "error": f"MLflow logging failed: {e}",
            "mlflow_run_id": None,
        }


async def _log_hyperparameters(
    run: Any,
    hyperparameters: Dict[str, Any],
    algorithm_name: str,
) -> None:
    """Log hyperparameters to MLflow run.

    Args:
        run: MLflowRun object
        hyperparameters: Hyperparameter dictionary
        algorithm_name: Algorithm name for prefixing
    """
    if not hyperparameters:
        return

    # Log with hp_ prefix for clarity
    params = {}
    for key, value in hyperparameters.items():
        # Skip internal parameters
        if key.startswith("_"):
            continue

        # MLflow requires string values
        if isinstance(value, (list, dict)):
            params[f"hp_{key}"] = json.dumps(value)
        else:
            params[f"hp_{key}"] = value

    if params:
        await run.log_params(params)


async def _log_training_metrics(run: Any, state: Dict[str, Any]) -> None:
    """Log training-related metrics.

    Args:
        run: MLflowRun object
        state: Training state dictionary
    """
    metrics = {}

    # Training duration
    if "training_duration_seconds" in state:
        metrics["training_duration_seconds"] = state["training_duration_seconds"]

    # Early stopping info
    if state.get("early_stopped"):
        metrics["early_stopped"] = 1.0
        if state.get("final_epoch"):
            metrics["final_epoch"] = float(state["final_epoch"])

    # HPO metrics
    if state.get("hpo_completed"):
        metrics["hpo_completed"] = 1.0
        if state.get("hpo_best_value"):
            metrics["hpo_best_value"] = state["hpo_best_value"]
        if state.get("hpo_trials_run"):
            metrics["hpo_trials_run"] = float(state["hpo_trials_run"])
        if state.get("hpo_duration_seconds"):
            metrics["hpo_duration_seconds"] = state["hpo_duration_seconds"]

    if metrics:
        await run.log_metrics(metrics)


async def _log_split_metrics(
    run: Any,
    split_name: str,
    metrics: Dict[str, Any],
) -> None:
    """Log metrics for a data split.

    Args:
        run: MLflowRun object
        split_name: Split name (train, validation, test, holdout)
        metrics: Metrics dictionary
    """
    if not metrics:
        return

    prefixed_metrics = {}
    for key, value in metrics.items():
        # Skip non-numeric values
        if not isinstance(value, (int, float)):
            continue

        # Skip None values
        if value is None:
            continue

        prefixed_metrics[f"{split_name}_{key}"] = float(value)

    if prefixed_metrics:
        await run.log_metrics(prefixed_metrics)


async def _log_model_artifact(
    run: Any,
    model: Any,
    algorithm_name: str,
    framework: str,
) -> Optional[str]:
    """Log trained model as MLflow artifact.

    Args:
        run: MLflowRun object
        model: Trained model object
        algorithm_name: Algorithm name
        framework: ML framework

    Returns:
        Model URI if successful
    """
    # Determine MLflow flavor based on framework/algorithm
    flavor = _get_mlflow_flavor(algorithm_name, framework)

    try:
        model_uri = await run.log_model(
            model=model,
            artifact_path="model",
            flavor=flavor,
        )
        return model_uri
    except Exception as e:
        logger.warning(f"Failed to log model with {flavor} flavor: {e}")
        # Fallback to sklearn flavor
        try:
            model_uri = await run.log_model(
                model=model,
                artifact_path="model",
                flavor="sklearn",
            )
            return model_uri
        except Exception as e2:
            logger.error(f"Failed to log model: {e2}")
            return None


async def _log_additional_artifacts(run: Any, state: Dict[str, Any]) -> None:
    """Log additional artifacts like feature importance, etc.

    Args:
        run: MLflowRun object
        state: Training state dictionary
    """
    # Log feature importance if available
    feature_importance = state.get("feature_importance")
    if feature_importance:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(feature_importance, f, indent=2)
                f.flush()
                await run.log_artifact(f.name, "feature_importance.json")
        except Exception as e:
            logger.warning(f"Failed to log feature importance: {e}")

    # Log confusion matrix if available
    confusion_matrix = state.get("confusion_matrix")
    if confusion_matrix:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(confusion_matrix, f, indent=2)
                f.flush()
                await run.log_artifact(f.name, "confusion_matrix.json")
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    # Log evaluation summary
    evaluation_metrics = state.get("evaluation_metrics", {})
    if evaluation_metrics:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(evaluation_metrics, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "evaluation_summary.json")
        except Exception as e:
            logger.warning(f"Failed to log evaluation summary: {e}")


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
    return framework_map.get(algorithm_name, "sklearn")


def _get_mlflow_flavor(algorithm_name: str, framework: str) -> str:
    """Get MLflow flavor for model logging.

    Args:
        algorithm_name: Algorithm name
        framework: Framework name

    Returns:
        MLflow flavor name
    """
    if algorithm_name == "XGBoost" or framework == "xgboost":
        return "xgboost"
    elif algorithm_name == "LightGBM" or framework == "lightgbm":
        return "lightgbm"
    else:
        return "sklearn"


def _get_primary_metric(
    metrics: Dict[str, Any],
    problem_type: str,
) -> Optional[float]:
    """Get primary metric for model comparison.

    Args:
        metrics: Metrics dictionary
        problem_type: Problem type

    Returns:
        Primary metric value or None
    """
    if not metrics:
        return None

    # Primary metric by problem type
    primary_metric_map = {
        "binary_classification": ["roc_auc", "auc", "f1", "accuracy"],
        "multiclass_classification": ["f1_weighted", "f1", "accuracy"],
        "regression": ["r2", "rmse", "mae"],
    }

    primary_candidates = primary_metric_map.get(
        problem_type, ["roc_auc", "f1", "r2"]
    )

    for metric_name in primary_candidates:
        if metric_name in metrics and metrics[metric_name] is not None:
            return float(metrics[metric_name])

    return None


async def _persist_training_run(
    experiment_id: str,
    run_name: str,
    mlflow_run_id: str,
    algorithm_name: str,
    hyperparameters: Dict[str, Any],
    training_samples: int,
    validation_samples: Optional[int],
    test_samples: Optional[int],
    feature_names: List[str],
    hpo_study_name: Optional[str],
    hpo_best_trial: Optional[int],
) -> Optional[UUID]:
    """Persist training run to database with HPO linkage.

    This creates a record in ml_training_runs table that links the
    training run to its Optuna HPO study for complete traceability.

    Args:
        experiment_id: ML experiment ID (may be string or UUID)
        run_name: Human-readable run name
        mlflow_run_id: MLflow run ID for cross-reference
        algorithm_name: Algorithm used
        hyperparameters: Best hyperparameters used
        training_samples: Number of training samples
        validation_samples: Number of validation samples
        test_samples: Number of test samples
        feature_names: List of feature names
        hpo_study_name: Optuna study name for HPO linkage
        hpo_best_trial: Best trial number from Optuna

    Returns:
        Database run ID if successful, None otherwise
    """
    repo = _get_training_run_repository()
    if not repo:
        logger.debug("Training run repository not available, skipping DB persistence")
        return None

    try:
        # Convert experiment_id to UUID if it's a valid UUID string
        try:
            exp_uuid = UUID(experiment_id) if experiment_id != "unknown" else None
        except (ValueError, TypeError):
            exp_uuid = None

        if not exp_uuid:
            logger.debug("No valid experiment_id for DB persistence")
            return None

        # Create training run with HPO linkage
        run = await repo.create_run_with_hpo(
            experiment_id=exp_uuid,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            algorithm=algorithm_name,
            hyperparameters=hyperparameters or {},
            training_samples=training_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            feature_names=feature_names or [],
            optuna_study_name=hpo_study_name,
            optuna_trial_number=hpo_best_trial,
            is_best_trial=hpo_best_trial is not None,
        )

        logger.info(
            f"Persisted training run to database: id={run.id}, "
            f"hpo_study={hpo_study_name or 'None'}"
        )
        return run.id

    except Exception as e:
        # Non-fatal: MLflow logging succeeded, DB persistence is secondary
        logger.warning(f"Failed to persist training run to database: {e}")
        return None
