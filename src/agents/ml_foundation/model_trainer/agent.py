"""Model Trainer Agent - ML Foundation Tier 0.

This agent trains ML models with strict split enforcement, hyperparameter
optimization, and MLflow logging.
"""

import uuid
from datetime import datetime
from typing import Any, Dict

from .graph import create_model_trainer_graph
from .state import ModelTrainerState


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

    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = None  # Variable (depends on model complexity)

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

        # Execute LangGraph workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)
        except Exception as e:
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

        # TODO: MLflow Integration
        # import mlflow
        #
        # # Start MLflow run
        # with mlflow.start_run(run_name=training_run_id):
        #     # Log parameters
        #     mlflow.log_params(best_hyperparameters)
        #     mlflow.log_param("algorithm_name", algorithm_name)
        #     mlflow.log_param("enable_hpo", enable_hpo)
        #     mlflow.log_param("hpo_trials_run", hpo_trials_run)
        #
        #     # Log metrics
        #     for metric_name, value in test_metrics.items():
        #         mlflow.log_metric(f"test_{metric_name}", value)
        #     for metric_name, value in validation_metrics.items():
        #         mlflow.log_metric(f"val_{metric_name}", value)
        #
        #     # Log model artifact
        #     mlflow.sklearn.log_model(trained_model, "model")
        #
        #     # Get MLflow info
        #     mlflow_run_id = mlflow.active_run().info.run_id
        #     model_artifact_uri = mlflow.get_artifact_uri("model")

        # PLACEHOLDER: MLflow info (TODO)
        mlflow_run_id = None
        mlflow_experiment_id = None
        model_artifact_uri = "TODO://mlflow/artifacts/model"
        preprocessing_artifact_uri = "TODO://mlflow/artifacts/preprocessor"

        # TODO: Database Persistence
        # Save to ml_training_runs table
        # training_run_repo = MLTrainingRunRepository()
        # await training_run_repo.create({
        #     "training_run_id": training_run_id,
        #     "experiment_id": experiment_id,
        #     "algorithm_name": algorithm_name,
        #     "hyperparameters": best_hyperparameters,
        #     "test_metrics": test_metrics,
        #     "success_criteria_met": success_criteria_met,
        #     ...
        # })

        persisted_to_db = False  # TODO: Set to True after DB save

        # Construct output
        return {
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
            # MLflow info (TODO)
            "mlflow_run_id": mlflow_run_id,
            "mlflow_experiment_id": mlflow_experiment_id,
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
            "training_duration_seconds": training_duration_seconds,
            "early_stopped": early_stopped,
            "training_started_at": training_started_at,
            "training_completed_at": training_completed_at,
            # Split info
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "total_samples": total_samples,
            # Database
            "persisted_to_db": persisted_to_db,  # TODO
            # Status
            "training_status": "completed",
            "framework": "sklearn",  # TODO: Detect from algorithm_class
            "trained_by": "model_trainer",
            "created_at": datetime.now(tz=None).isoformat(),
        }
