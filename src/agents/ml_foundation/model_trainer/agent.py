"""Model Trainer Agent - ML Foundation Tier 0.

This agent trains ML models with strict split enforcement, hyperparameter
optimization, and MLflow logging.

Outputs:
- TrainedModel: Trained model with hyperparameters
- ValidationMetrics: Train/validation/test metrics
- MLflowInfo: MLflow run and artifact information

Integration:
- Upstream: model_selector (requires ModelCandidate + QC gate passed)
- Downstream: feature_analyzer (consumes trained model)
- Database: ml_training_runs table
- Memory: Procedural memory (successful training patterns)
- Observability: Opik tracing
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from .graph import create_model_trainer_graph
from .state import ModelTrainerState

logger = logging.getLogger(__name__)


async def _get_training_run_repository():
    """Get MLTrainingRunRepository with async client (lazy import to avoid circular deps)."""
    try:
        from src.memory.services.factories import get_async_supabase_client
        from src.repositories.ml_experiment import MLTrainingRunRepository

        client = await get_async_supabase_client()
        return MLTrainingRunRepository(supabase_client=client)
    except Exception as e:
        logger.warning(f"Could not get training run repository: {e}")
        return None


def _get_opik_connector():
    """Get OpikConnector (lazy import to avoid circular deps)."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except Exception as e:
        logger.warning(f"Could not get Opik connector: {e}")
        return None


def _get_procedural_memory():
    """Get procedural memory client (lazy import with graceful degradation)."""
    try:
        from src.memory.procedural_memory import get_procedural_memory_client

        return get_procedural_memory_client()
    except Exception as e:
        logger.debug(f"Procedural memory not available: {e}")
        return None


class ModelTrainerAgent:
    """Model Trainer: Train ML models with HPO and validation.

    Responsibilities:
    - QC gate enforcement (MANDATORY)
    - Strict split enforcement (60/20/15/5)
    - Preprocessing isolation (fit on train only)
    - Hyperparameter optimization (Optuna on validation)
    - Model training (train on train set)
    - Evaluation (train/val/test sets, test touched ONCE)
    - MLflow logging (experiments, parameters, metrics, artifacts)
    - Artifact versioning

    Critical Principles:
    - NEVER train without QC pass
    - NEVER fit preprocessing on validation/test/holdout
    - NEVER tune on test set
    - NEVER touch holdout until post-deployment
    - ALWAYS validate split ratios
    - ALWAYS log to MLflow
    """

    # Agent metadata
    tier = 0
    tier_name = "ml_foundation"
    agent_name = "model_trainer"
    agent_type = "standard"
    sla_seconds = None  # Variable (depends on model complexity)
    tools = ["optuna", "mlflow", "feast"]  # Optuna for HPO, MLflow for tracking, Feast for features

    def __init__(self):
        """Initialize ModelTrainerAgent."""
        self.graph = create_model_trainer_graph()

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training workflow.

        Args:
            input_data: Dictionary with:
                - model_candidate (Dict): From model_selector
                - qc_report (Dict): From data_preparer (MUST have qc_passed=True)
                - experiment_id (str): Experiment identifier
                - success_criteria (Dict[str, float]): Performance thresholds
                - enable_hpo (bool): Whether to run hyperparameter optimization
                - hpo_trials (int): Number of Optuna trials (default: 50)
                - hpo_timeout_hours (float, optional): HPO timeout
                - early_stopping (bool): Enable early stopping (default: False)
                - early_stopping_patience (int): Early stopping patience (default: 10)
                Optional (if splits already prepared):
                - train_data (Dict): Training split
                - validation_data (Dict): Validation split
                - test_data (Dict): Test split
                - holdout_data (Dict): Holdout split

        Returns:
            Dictionary with:
                - training_run_id (str): Unique training run ID
                - model_id (str): Model identifier
                - trained_model (Any): Trained model object
                - train_metrics (Dict): Training set metrics
                - validation_metrics (Dict): Validation set metrics
                - test_metrics (Dict): Test set metrics (FINAL)
                - auc_roc, precision, recall, f1_score (classification)
                - rmse, mae, r2 (regression)
                - success_criteria_met (bool): Whether criteria met
                - mlflow_run_id (str): MLflow run ID (TODO)
                - model_artifact_uri (str): Model artifact URI (TODO)

        Raises:
            ValueError: If required inputs missing or QC validation failed
        """
        # Validate required inputs
        required_fields = ["model_candidate", "qc_report", "experiment_id"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        model_candidate = input_data["model_candidate"]
        qc_report = input_data["qc_report"]
        experiment_id = input_data["experiment_id"]

        # Validate model_candidate structure
        required_candidate_fields = [
            "algorithm_name",
            "algorithm_class",
            "hyperparameter_search_space",
            "default_hyperparameters",
        ]
        for field in required_candidate_fields:
            if field not in model_candidate:
                raise ValueError(f"model_candidate missing required field: {field}")

        # Extract model configuration
        algorithm_name = model_candidate["algorithm_name"]
        algorithm_class = model_candidate["algorithm_class"]
        hyperparameter_search_space = model_candidate["hyperparameter_search_space"]
        default_hyperparameters = model_candidate["default_hyperparameters"]

        # Extract training configuration (with defaults)
        success_criteria = input_data.get("success_criteria", {})
        enable_hpo = input_data.get("enable_hpo", True)
        hpo_trials = input_data.get("hpo_trials", 50)
        hpo_timeout_hours = input_data.get("hpo_timeout_hours")
        early_stopping = input_data.get("early_stopping", False)
        early_stopping_patience = input_data.get("early_stopping_patience", 10)
        problem_type = input_data.get("problem_type", "binary_classification")

        # Generate training run ID
        training_run_id = f"train_{uuid.uuid4().hex[:12]}"
        model_id = f"model_{algorithm_name}_{uuid.uuid4().hex[:8]}"

        # Construct initial state
        initial_state: ModelTrainerState = {
            # Input fields
            "model_candidate": model_candidate,
            "algorithm_name": algorithm_name,
            "algorithm_class": algorithm_class,
            "hyperparameter_search_space": hyperparameter_search_space,
            "default_hyperparameters": default_hyperparameters,
            "qc_report": qc_report,
            "experiment_id": experiment_id,
            "success_criteria": success_criteria,
            "problem_type": problem_type,
            "enable_hpo": enable_hpo,
            "hpo_trials": hpo_trials,
            "hpo_timeout_hours": hpo_timeout_hours,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            # IDs
            "training_run_id": training_run_id,
            "model_id": model_id,
            # MLflow and checkpointing config
            "enable_mlflow": input_data.get("enable_mlflow", True),
            "enable_checkpointing": input_data.get("enable_checkpointing", True),
            # Optional: Pre-loaded splits
            "train_data": input_data.get("train_data"),
            "validation_data": input_data.get("validation_data"),
            "test_data": input_data.get("test_data"),
            "holdout_data": input_data.get("holdout_data"),
        }

        # Execute LangGraph workflow with optional Opik tracing
        start_time = datetime.now(timezone.utc)
        logger.info(
            f"Starting model training for experiment {experiment_id}, "
            f"algorithm={algorithm_name}, problem_type={problem_type}"
        )

        opik = _get_opik_connector()
        try:
            if opik and opik.is_enabled:
                async with opik.trace_agent(
                    agent_name=self.agent_name,
                    operation="train_model",
                    metadata={
                        "tier": self.tier,
                        "experiment_id": experiment_id,
                        "algorithm_name": algorithm_name,
                        "problem_type": problem_type,
                        "enable_hpo": enable_hpo,
                        "hpo_trials": hpo_trials,
                    },
                    tags=[self.agent_name, "tier_0", "model_training"],
                    input_data={
                        "experiment_id": experiment_id,
                        "algorithm_name": algorithm_name,
                        "problem_type": problem_type,
                    },
                ) as span:
                    final_state = await self.graph.ainvoke(initial_state)
                    # Set output on span
                    if span and not final_state.get("error"):
                        span.set_output(
                            {
                                "training_run_id": training_run_id,
                                "model_id": model_id,
                                "success_criteria_met": final_state.get("success_criteria_met"),
                                "hpo_trials_run": final_state.get("hpo_trials_run", 0),
                            }
                        )
            else:
                final_state = await self.graph.ainvoke(initial_state)
        except Exception as e:
            logger.exception(f"Model training failed: {e}")
            raise RuntimeError(f"Model training workflow failed: {str(e)}") from e

        # Check for errors in final state
        if final_state.get("error"):
            error_msg = final_state.get("error")
            error_type = final_state.get("error_type", "unknown_error")
            raise RuntimeError(f"Training error ({error_type}): {error_msg}")

        # Extract outputs from final state
        trained_model = final_state.get("trained_model")
        train_metrics = final_state.get("train_metrics", {})
        validation_metrics = final_state.get("validation_metrics", {})
        test_metrics = final_state.get("test_metrics", {})

        # Extract problem-specific metrics
        auc_roc = final_state.get("auc_roc")
        precision = final_state.get("precision")
        recall = final_state.get("recall")
        f1_score = final_state.get("f1_score")
        pr_auc = final_state.get("pr_auc")
        confusion_matrix = final_state.get("confusion_matrix")
        brier_score = final_state.get("brier_score")
        calibration_error = final_state.get("calibration_error")
        optimal_threshold = final_state.get("optimal_threshold")
        precision_at_k = final_state.get("precision_at_k")
        rmse = final_state.get("rmse")
        mae = final_state.get("mae")
        r2 = final_state.get("r2")
        confidence_interval = final_state.get("confidence_interval", {})

        # Success criteria
        success_criteria_met = final_state.get("success_criteria_met", False)
        success_criteria_results = final_state.get("success_criteria_results", {})

        # Preprocessing and HPO info
        preprocessing_statistics = final_state.get("preprocessing_statistics", {})
        fitted_preprocessor = final_state.get("preprocessor")  # For inference
        X_validation_preprocessed = final_state.get("X_validation_preprocessed")  # For analysis
        X_test_preprocessed = final_state.get("X_test_preprocessed")  # For analysis
        best_hyperparameters = final_state.get("best_hyperparameters", {})
        hpo_completed = final_state.get("hpo_completed", False)
        hpo_best_trial = final_state.get("hpo_best_trial")
        hpo_trials_run = final_state.get("hpo_trials_run", 0)

        # Training metadata
        training_duration_seconds = final_state.get("training_duration_seconds", 0.0)
        early_stopped = final_state.get("early_stopped", False)
        training_started_at = final_state.get("training_started_at")
        training_completed_at = final_state.get("training_completed_at")

        # Split information
        train_samples = final_state.get("train_samples", 0)
        validation_samples = final_state.get("validation_samples", 0)
        test_samples = final_state.get("test_samples", 0)
        total_samples = final_state.get("total_samples", 0)

        # Class imbalance information
        imbalance_detected = final_state.get("imbalance_detected", False)
        imbalance_ratio = final_state.get("imbalance_ratio", 1.0)
        minority_ratio = final_state.get("minority_ratio", 0.5)
        imbalance_severity = final_state.get("imbalance_severity", "none")
        class_distribution = final_state.get("class_distribution", {})
        recommended_strategy = final_state.get("recommended_strategy", "none")
        strategy_rationale = final_state.get("strategy_rationale", "")

        # Resampling information
        resampling_applied = final_state.get("resampling_applied", False)
        resampling_strategy = final_state.get("resampling_strategy")
        original_distribution = final_state.get("original_distribution", {})
        resampled_distribution = final_state.get("resampled_distribution", {})

        # Extract sample counts from shape tuples (shape is (n_samples, n_features))
        original_train_shape = final_state.get("original_train_shape")
        resampled_train_shape = final_state.get("resampled_train_shape")
        original_train_samples = original_train_shape[0] if original_train_shape else None
        resampled_train_samples = resampled_train_shape[0] if resampled_train_shape else None

        # MLflow Integration - Extract values from graph result
        # The mlflow_logger node logs to MLflow and returns these values in state
        mlflow_run_id = final_state.get("mlflow_run_id")
        mlflow_experiment_id = final_state.get("mlflow_experiment_id")
        model_artifact_uri = final_state.get("mlflow_model_uri")
        preprocessing_artifact_uri = final_state.get("preprocessing_artifact_uri")
        mlflow_status = final_state.get("mlflow_status", "not_logged")
        mlflow_model_version = final_state.get("mlflow_model_version")
        mlflow_model_name = final_state.get("mlflow_model_name")

        # Log warning if MLflow logging failed
        if mlflow_status != "success" and mlflow_run_id is None:
            logger.warning(
                f"MLflow logging not completed for training run {training_run_id}. "
                f"Status: {mlflow_status}"
            )

        # Construct output
        output = {
            # Core outputs
            "training_run_id": training_run_id,
            "model_id": model_id,
            "trained_model": trained_model,
            # Metrics
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            # Classification metrics
            "auc_roc": auc_roc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "pr_auc": pr_auc,
            "confusion_matrix": confusion_matrix,
            "brier_score": brier_score,
            "calibration_error": calibration_error,
            "optimal_threshold": optimal_threshold,
            "precision_at_k": precision_at_k,
            # Regression metrics
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            # Confidence intervals
            "confidence_interval": confidence_interval,
            # Success criteria
            "success_criteria_met": success_criteria_met,
            "success_criteria_results": success_criteria_results,
            # MLflow info (extracted from mlflow_logger node)
            "mlflow_run_id": mlflow_run_id,
            "mlflow_experiment_id": mlflow_experiment_id,
            "mlflow_status": mlflow_status,
            "mlflow_model_version": mlflow_model_version,
            "mlflow_model_name": mlflow_model_name,
            "model_artifact_uri": model_artifact_uri,
            "preprocessing_artifact_uri": preprocessing_artifact_uri,
            # Training metadata
            "algorithm_name": algorithm_name,
            "algorithm_class": algorithm_class,
            "best_hyperparameters": best_hyperparameters,
            "hpo_completed": hpo_completed,
            "hpo_best_trial": hpo_best_trial,
            "hpo_trials_run": hpo_trials_run,
            "preprocessing_statistics": preprocessing_statistics,
            "fitted_preprocessor": fitted_preprocessor,  # For inference on new data
            "X_validation_preprocessed": X_validation_preprocessed,  # For analysis
            "X_test_preprocessed": X_test_preprocessed,  # For analysis
            "training_duration_seconds": training_duration_seconds,
            "early_stopped": early_stopped,
            "training_started_at": training_started_at,
            "training_completed_at": training_completed_at,
            # Split info
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "total_samples": total_samples,
            # Class imbalance info
            "imbalance_detected": imbalance_detected,
            "imbalance_ratio": imbalance_ratio,
            "minority_ratio": minority_ratio,
            "imbalance_severity": imbalance_severity,
            "class_distribution": class_distribution,
            "recommended_strategy": recommended_strategy,
            "strategy_rationale": strategy_rationale,
            # Resampling info
            "resampling_applied": resampling_applied,
            "original_train_samples": original_train_samples,
            "resampled_train_samples": resampled_train_samples,
            "resampling_strategy": resampling_strategy,
            "original_distribution": original_distribution,
            "resampled_distribution": resampled_distribution,
            # Database (updated after persistence)
            "persisted_to_db": False,
            # Context
            "experiment_id": experiment_id,
            "problem_type": problem_type,
            # Status
            "training_status": "completed",
            "framework": self._detect_framework(algorithm_class),
            "trained_by": "model_trainer",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Persist training run to database
        persisted = await self._persist_training_run(output)
        output["persisted_to_db"] = persisted

        # Update procedural memory with successful training pattern
        await self._update_procedural_memory(output)

        # Log completion
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"Model training complete: {algorithm_name} "
            f"(success_criteria_met: {success_criteria_met}) in {duration:.2f}s"
        )

        return output

    async def _persist_training_run(self, output: Dict[str, Any]) -> bool:
        """Persist training run to ml_training_runs table.

        Graceful degradation: If repository is unavailable or the parent
        experiment doesn't exist, logs a message and continues without error.

        Args:
            output: Agent output containing training run details

        Returns:
            True if persisted successfully, False otherwise
        """
        from uuid import uuid4

        try:
            repo = await _get_training_run_repository()
            if repo is None:
                logger.debug("Skipping training run persistence (no repository)")
                return False

            # Look up the experiment by its mlflow_experiment_id to get the actual UUID
            experiment_id_str = output.get("experiment_id", "")
            experiment_uuid = None

            if experiment_id_str:
                try:
                    # Get the experiment repository to look up by mlflow_id
                    from src.memory.services.factories import get_async_supabase_client
                    from src.repositories.ml_experiment import MLExperimentRepository

                    client = await get_async_supabase_client()
                    exp_repo = MLExperimentRepository(supabase_client=client)
                    experiment = await exp_repo.get_by_mlflow_id(experiment_id_str)

                    if experiment and experiment.id:
                        experiment_uuid = experiment.id
                        logger.debug(
                            f"Found experiment {experiment_id_str} with UUID {experiment_uuid}"
                        )
                    else:
                        logger.debug(
                            f"Experiment {experiment_id_str} not found in database, "
                            "skipping training run persistence"
                        )
                        return False
                except Exception as lookup_err:
                    logger.debug(f"Could not look up experiment: {lookup_err}")
                    return False

            if not experiment_uuid:
                logger.debug("No valid experiment UUID, skipping training run persistence")
                return False

            # Create training run record using create_run_with_hpo
            # which accepts HPO-related parameters
            result = await repo.create_run_with_hpo(
                experiment_id=experiment_uuid,
                run_name=output.get("training_run_id", f"run_{uuid4().hex[:8]}"),
                mlflow_run_id=output.get("mlflow_run_id", ""),
                algorithm=output.get("algorithm_name", "unknown"),
                hyperparameters=output.get("best_hyperparameters", {}),
                training_samples=output.get("train_samples", 0),
                feature_names=output.get("feature_names", []),
                optuna_study_name=output.get("hpo_study_name"),
                optuna_trial_number=output.get("hpo_best_trial"),
                is_best_trial=output.get("hpo_completed", False),
                validation_samples=output.get("validation_samples", 0),
                test_samples=output.get("test_samples", 0),
            )

            if result and result.id:
                # Update with metrics using the returned run's UUID
                await repo.update_run_metrics(
                    run_id=result.id,
                    train_metrics=output.get("train_metrics", {}),
                    validation_metrics=output.get("validation_metrics", {}),
                    test_metrics=output.get("test_metrics", {}),
                )

                logger.info(
                    f"Persisted training run: {result.run_name} for experiment {experiment_uuid}"
                )
                return True

            logger.debug("Training run not persisted (no result returned)")
            return False

        except Exception as e:
            logger.warning(f"Failed to persist training run: {e}")
            return False

    async def _update_procedural_memory(self, output: Dict[str, Any]) -> None:
        """Update procedural memory with successful training pattern.

        Graceful degradation: If memory is unavailable,
        logs a debug message and continues without error.

        Args:
            output: Agent output containing training run details
        """
        try:
            memory = _get_procedural_memory()
            if memory is None:
                logger.debug("Procedural memory not available, skipping update")
                return

            # Store successful training pattern for future reference
            await memory.store_pattern(
                agent_name=self.agent_name,
                pattern_type="model_training",
                pattern_data={
                    "algorithm_name": output.get("algorithm_name"),
                    "algorithm_class": output.get("algorithm_class"),
                    "problem_type": output.get("problem_type"),
                    "success_criteria_met": output.get("success_criteria_met"),
                    "hpo_completed": output.get("hpo_completed"),
                    "hpo_trials_run": output.get("hpo_trials_run"),
                    "best_hyperparameters": output.get("best_hyperparameters"),
                    "training_duration_seconds": output.get("training_duration_seconds"),
                    "early_stopped": output.get("early_stopped"),
                    "train_samples": output.get("train_samples"),
                    "test_metrics": output.get("test_metrics"),
                    "experiment_id": output.get("experiment_id"),
                    "training_run_id": output.get("training_run_id"),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(
                f"Updated procedural memory for training run: {output.get('training_run_id')}"
            )

        except Exception as e:
            logger.debug(f"Failed to update procedural memory: {e}")

    def _detect_framework(self, algorithm_class: str | None) -> str:
        """Detect ML framework from algorithm class name.

        Args:
            algorithm_class: The algorithm class name (e.g., "sklearn.ensemble.RandomForestClassifier")

        Returns:
            Framework name: "sklearn", "xgboost", "lightgbm", "catboost", "statsmodels", or "unknown"
        """
        if not algorithm_class:
            return "unknown"

        algorithm_lower = algorithm_class.lower()

        # Framework detection patterns
        if "sklearn" in algorithm_lower or "scikit" in algorithm_lower:
            return "sklearn"
        elif "xgboost" in algorithm_lower or "xgb" in algorithm_lower:
            return "xgboost"
        elif "lightgbm" in algorithm_lower or "lgb" in algorithm_lower:
            return "lightgbm"
        elif "catboost" in algorithm_lower:
            return "catboost"
        elif "statsmodels" in algorithm_lower:
            return "statsmodels"
        elif "econml" in algorithm_lower:
            return "econml"
        elif "dowhy" in algorithm_lower:
            return "dowhy"
        else:
            return "sklearn"  # Default fallback for common sklearn-compatible models
