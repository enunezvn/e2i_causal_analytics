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

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn

from .aggregation import (
    AggregateStat,
    aggregate_fold_metrics,
)
from .graph import create_model_trainer_graph
from .memory_hooks import ModelTrainerMemoryHooks
from .splitting import FoldSpec, RepeatedStratifiedSplitter
from .state import ModelTrainerState

logger = logging.getLogger(__name__)

# Phase 1 W3-lite Day 4 (shard 17 W3 row Day 4 + shard 21 §B): the
# `evaluation_mode` flag selects between the legacy single-graph path
# ("single", default — byte-identical to the pre-W4-day-4 baseline) and the
# k=10 repeated train/val/test orchestrator ("repeated_k10").
#
# Naming: shard 21 §B locks "single" / "repeated_k10"; shard 17 W3 row Day 4
# uses "single_split" / "repeated_kfold". Implementation follows shard 21 since
# the orchestrator + tests are written against those names. Naming divergence
# is flagged in `adaptive_v3_followup_state.md` for user decision on whether
# to amend shard 17 separately.
_VALID_EVALUATION_MODES: Tuple[str, ...] = ("single", "repeated_k10")


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
        # W3-lite Day 4 (shard 21 §B) — `evaluation_mode` dispatch.
        # The orchestrator (`_run_repeated_splits`) recursively calls this method
        # per fold with `repeated_mode_fold_invocation=True` set; that sentinel
        # short-circuits the dispatch so the inner call falls through to the
        # legacy single-graph path. The per-fold input still carries
        # `evaluation_mode="repeated_k10"` so downstream nodes (split_enforcer,
        # mlflow_logger, evaluator) can branch on the active mode.
        evaluation_mode = input_data.get("evaluation_mode", "single")
        if evaluation_mode not in _VALID_EVALUATION_MODES:
            raise ValueError(
                f"Unknown evaluation_mode={evaluation_mode!r}; valid: {_VALID_EVALUATION_MODES}"
            )
        if evaluation_mode == "repeated_k10" and not input_data.get(
            "repeated_mode_fold_invocation", False
        ):
            return await self._run_repeated_splits(input_data)

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
        # Block 5 (#10): optional cost matrix for business_utility metric.
        # Caller is responsible for validation; scope_definer typically does
        # this via _validate_cost_matrix and forwards the dict here.
        cost_matrix = input_data.get("cost_matrix")

        # Generate training run ID
        training_run_id = f"train_{uuid.uuid4().hex[:12]}"
        model_id = f"model_{algorithm_name}_{uuid.uuid4().hex[:8]}"

        # Construct initial state
        initial_state: ModelTrainerState = {
            # D1.2: thread caller-provided audit_workflow_id (see scope_definer
            # for the rationale). Backlog #1 (closed 2026-05-09) tightened the
            # State to required-no-default to fix the LangGraph channel-reducer
            # bug (default_factory firing on every Schema reconstruction).
            # Caller-provided UUID is preferred; absent that, generate one at
            # the agent boundary. Either way the UUID is set ONCE before
            # graph.ainvoke, so LangGraph's reducer pins it across nodes.
            **(
                {"audit_workflow_id": input_data["audit_workflow_id"]}
                if input_data.get("audit_workflow_id") is not None
                else {"audit_workflow_id": uuid4()}
            ),
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
            "cost_matrix": cost_matrix,
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
            "train_data": input_data.get("train_data") or {},
            "validation_data": input_data.get("validation_data") or {},
            "test_data": input_data.get("test_data") or {},
            "holdout_data": input_data.get("holdout_data") or {},
            # Opt-in synthetic augmentation: path to a reviewed Phase-3 preview
            # cohort (.npz). None → ``augment_training_data`` is a no-op.
            "augmentation_data_path": input_data.get("augmentation_data_path"),
            # Configurable minimum samples per split (consumed by split_enforcer)
            "min_samples_per_split": input_data.get("min_samples_per_split", 10),
            # Day-3 fold-iteration random_state plumbing (W3-lite Day 3 +
            # consumed by `resolve_fold_random_state` in split_loader /
            # hyperparameter_tuner / model_trainer_node). Populated only when
            # the caller is the `_run_repeated_splits` orchestrator (Day-4)
            # via `_build_fold_input`; legacy callers omit the field and
            # the helper falls back to `random_state` -> `42`.
            **(
                {"fold_random_state": int(input_data["fold_random_state"])}
                if "fold_random_state" in input_data
                else {}
            ),
            **({"fold_idx": int(input_data["fold_idx"])} if "fold_idx" in input_data else {}),
            # Day-4 active evaluation_mode flag — split_enforcer / mlflow_logger
            # / evaluator branch on this; legacy callers omit it and the helper
            # sites default to single-mode behavior.
            "evaluation_mode": evaluation_mode,
            # Day-5 (cycle-15 I-4): propagate the orchestrator sentinel into
            # graph state so per-fold nodes (notably ``mlflow_logger``) can
            # detect "I'm being called per-fold inside repeated_k10" and open
            # a NESTED MLflow run with fold tags. Single-mode callers do not
            # set this field, so it defaults to False.
            "repeated_mode_fold_invocation": bool(
                input_data.get("repeated_mode_fold_invocation", False)
            ),
            # PR #463 Phase 2: forward the scope_spec + always-run flag into
            # graph state so the post-training ``learning_curve`` node can
            # read ``scope_spec.success_criteria.min_auc`` / ``problem_type``
            # / ``sufficiency`` without reaching back into the pipeline.
            **(
                {"scope_spec": input_data["scope_spec"]}
                if input_data.get("scope_spec") is not None
                else {}
            ),
            **(
                {"always_run_learning_curve": bool(input_data["always_run_learning_curve"])}
                if "always_run_learning_curve" in input_data
                else {}
            ),
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

        # Extract outputs from final state.
        # #633: return the DEPLOYED model as ``trained_model`` — the calibrated
        # estimator when post-hoc calibration was applied (evaluator promotes
        # ``deployed_model``), else the raw model. Downstream consumers (tier0
        # runner, deployer) treat ``output["trained_model"]`` as the artifact to
        # ship, so it MUST be the calibrated model whose probabilities the v3
        # gates were judged on. Falling back to the raw model keeps non-binary /
        # calibration-skipped paths unchanged.
        trained_model = final_state.get("deployed_model")
        if trained_model is None:
            trained_model = final_state.get("trained_model")
        calibration_applied = bool(final_state.get("calibration_applied", False))
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

        # Success criteria — extract the (possibly-overlaid) criteria
        # dict alongside the two derived fields so the runner / pipeline
        # can persist it (hops 3 and 4 in the v3 propagation chain).
        success_criteria_met = final_state.get("success_criteria_met", False)
        success_criteria_results = final_state.get("success_criteria_results", {})
        success_criteria_out = final_state.get("success_criteria", {})

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

        # Post-training leakage suspicion (from evaluator)
        leakage_suspected = final_state.get("leakage_suspected", False)
        suspicion_level = final_state.get("suspicion_level", "none")
        suspicion_reasons = final_state.get("suspicion_reasons", [])
        investigation_recommendations = final_state.get("investigation_recommendations", [])

        # Advanced validation (from evaluator)
        permutation_test = final_state.get("permutation_test", {})
        cv_results = final_state.get("cv_results", {})
        mcc = final_state.get("mcc")
        f1_threshold_analysis = final_state.get("f1_threshold_analysis", {})
        split_validation = final_state.get("split_validation", {})
        calibrated_ece = final_state.get("calibrated_ece")
        calibration_analysis = final_state.get("calibration_analysis", {})
        calibrated_test_metrics = final_state.get("calibrated_test_metrics", {})
        post_hoc_calibration = final_state.get("post_hoc_calibration", {})

        # Imbalance-aware evaluation fields (from evaluator)
        precision_constrained = final_state.get("precision_constrained")
        minority_recall = final_state.get("minority_recall")
        minority_precision = final_state.get("minority_precision")
        test_metrics_at_optimal = final_state.get("test_metrics_at_optimal", {})
        test_metrics_at_05 = final_state.get("test_metrics_at_05", {})

        # PR #463 Phase 2 — post-training learning-curve diagnostic. The
        # ``learning_curve`` node populates this dict only when
        # ``success_criteria_met is False``; otherwise it returns ``{}`` and
        # the field remains None.
        sufficiency_report = final_state.get("sufficiency_report")

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
            # #633: audit flag — True iff ``trained_model`` above is the
            # post-hoc calibrated estimator (not the raw model).
            "calibration_applied": calibration_applied,
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
            # Imbalance-aware evaluation
            "precision_constrained": precision_constrained,
            "minority_recall": minority_recall,
            "minority_precision": minority_precision,
            "test_metrics_at_optimal": test_metrics_at_optimal,
            "test_metrics_at_05": test_metrics_at_05,
            # Regression metrics
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            # Confidence intervals
            "confidence_interval": confidence_interval,
            # Success criteria
            "success_criteria_met": success_criteria_met,
            "success_criteria_results": success_criteria_results,
            "success_criteria": success_criteria_out,
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
            # Synthetic augmentation audit (opt-in; Phase 3 preview consumption).
            # ``applied`` is False both when not requested and when refused on a
            # schema mismatch (see ``skip_reason``). Synthetic rows are added to
            # the training split only — never validation/test/holdout.
            "training_augmentation": {
                "applied": bool(final_state.get("augmentation_applied", False)),
                "n_original": final_state.get("augmentation_n_original"),
                "n_synthetic": final_state.get("augmentation_n_synthetic"),
                "source": final_state.get("augmentation_source"),
                "audit_fingerprint": final_state.get("augmentation_fingerprint"),
                "skip_reason": final_state.get("augmentation_skip_reason"),
            },
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
            # Post-training leakage suspicion
            "leakage_suspected": leakage_suspected,
            "suspicion_level": suspicion_level,
            "suspicion_reasons": suspicion_reasons,
            "investigation_recommendations": investigation_recommendations,
            # Advanced validation
            "permutation_test": permutation_test,
            "cv_results": cv_results,
            "mcc": mcc,
            "f1_threshold_analysis": f1_threshold_analysis,
            "split_validation": split_validation,
            "calibrated_ece": calibrated_ece,
            "calibration_analysis": calibration_analysis,
            "calibrated_test_metrics": calibrated_test_metrics,
            "post_hoc_calibration": post_hoc_calibration,
            # PR #463 Phase 2: post-training learning-curve diagnostic.
            # None when the model met success_criteria; populated dict shaped
            # like DataSufficiencyReport otherwise.
            "sufficiency_report": sufficiency_report,
        }

        # Persist training run to database
        persisted = await self._persist_training_run(output)
        output["persisted_to_db"] = persisted

        # Update procedural memory with successful training pattern
        await self._update_procedural_memory(output)

        # Populate the semantic knowledge graph (e2i_causal) with the trained model
        # so Tier 0 runs actually grow it and the read-hooks return real context
        # (#749 — store_model_pattern was defined but never called).
        await self._update_semantic_memory(output)

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

    async def _update_semantic_memory(self, output: Dict[str, Any]) -> None:
        """Populate the semantic knowledge graph (FalkorDB ``e2i_causal``) with the
        trained model (#749).

        Mirrors ``_update_procedural_memory`` — graceful degradation: if semantic
        memory is unavailable the write is skipped, never raised. The
        ``store_model_pattern`` hook (Model + Hyperparameters + TRAINED_WITH /
        BELONGS_TO / USED_BY) was defined but never invoked, so Tier 0 runs left
        ``e2i_causal`` unpopulated.

        Args:
            output: Agent output carrying training_run_id, experiment_id,
                algorithm_name, test_metrics, best_hyperparameters,
                success_criteria_met.
        """
        try:
            training_run_id = output.get("training_run_id")
            experiment_id = output.get("experiment_id")
            if not training_run_id or not experiment_id:
                logger.debug(
                    "Missing training_run_id/experiment_id; skipping semantic-graph update"
                )
                return

            hooks = ModelTrainerMemoryHooks()
            await hooks.store_model_pattern(
                experiment_id=str(experiment_id),
                training_run_id=str(training_run_id),
                algorithm_name=output.get("algorithm_name") or "unknown",
                test_metrics=output.get("test_metrics", {}) or {},
                best_hyperparameters=output.get("best_hyperparameters", {}) or {},
                success_criteria_met=bool(output.get("success_criteria_met")),
            )
            logger.info(f"Updated semantic graph (e2i_causal) for training run: {training_run_id}")

        except Exception as e:
            logger.debug(f"Failed to update semantic memory: {e}")

    async def _run_repeated_splits(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """W3-lite Day-5 orchestrator (shard 21 §B/§C/§D + cycle-15 I-2/I-3/I-4).

        Iterates k=10 stratified shuffle-split draws over ``input_data['full_data']``,
        invoking the existing single-graph path once per fold with the fold's
        train/val/test indices materialized into the legacy split-dict shape
        consumed by ``load_splits``. Per-fold ``fold_random_state`` is threaded
        via ``input_data['fold_random_state']`` so Day-3 ``resolve_fold_random_state``
        replaces the historical hardcoded ``random_state=42`` in
        ``split_loader`` / ``hyperparameter_tuner`` / ``model_trainer_node``.

        Day-5 scope additions over Day-4 MVP:
          1. Parent MLflow run wraps the fold loop (shard 21 §C); per-fold
             nested children are opened by the per-fold ``mlflow_logger`` node
             when it observes ``state["evaluation_mode"]=="repeated_k10"`` AND
             ``state["repeated_mode_fold_invocation"]`` (cycle-15 I-4).
          2. Try/except per fold — partial-fold contract per cycle-15 I-3.
             Each ``fold_metrics`` entry carries
             ``fold_status: Literal["ok","failed"]``; aggregator skips failed
             folds; ``aggregate_status`` on the output dict is
             ``"PARTIAL"`` if any fold failed, ``"COMPLETE"`` otherwise.
          3. NEP 19 per-fold params logged to MLflow per cycle-15 I-2:
             ``seed_base / fold_idx / derived_seed=spec.seed /
             numpy.__version__ / sklearn.__version__``.
          4. ``aggregate_fold_metrics`` (shard 21 §D) is called over the
             flattened per-fold scalar dicts; aggregate ``mean / std /
             percentile_ci / bca_ci / n_folds`` per metric is emitted on
             ``output["aggregate_metrics"]`` and logged to the parent run.
          5. ``n_jobs`` (default 1) controls fold concurrency via
             ``asyncio.gather`` + ``Semaphore`` — concurrent at the asyncio
             level rather than process-level joblib.Parallel. Determinism is
             preserved because each fold's seed is sourced from
             ``spec.seed`` (FoldSpec) regardless of execution order.

             DIVERGENCE FROM SHARD 21 §F (cycle-16 I-3, inlined here so it
             survives any cycle_16_brief.md cleanup): shard 21 §F specifies
             process-level joblib.Parallel(n_jobs=2) with a ~5-7 min wall-clock
             target; this implementation uses asyncio.gather instead because
             (a) ``self.run`` is async and joblib doesn't natively support
             coroutines; (b) process-level Parallel duplicates the agent's
             ~280 MB Python state per worker, risky on the 16 GB shared
             droplet; (c) overlap of MLflow/DB I/O is the primary win
             expected from concurrency anyway. CONSEQUENCE: per-fold
             bootstrap-CI compute (~30s/fold per shard 21 §F T10.5) does NOT
             parallelize — wall-clock at n_jobs=2 is roughly equal to
             n_jobs=1 minus the I/O-overlap savings, NOT halved as §F's
             projection. Process-level Parallel is deferred to shard 22
             multi-disease orchestration where per-disease isolation is the
             natural motivator.

        Required input fields:
          - ``full_data``: ``{"X": pd.DataFrame, "y": pd.Series}`` (the unsplit
            stratification target — orchestrator owns the splitter).
          - ``model_candidate`` / ``qc_report`` / ``experiment_id``: as for
            single-mode (validation reused).

        Optional input fields:
          - ``seed_base`` (default 42): root seed for the splitter; per-fold
            seeds derive deterministically from ``(fold_idx, seed_base)``.
          - ``repeated_splits_config``: ``{k, train_frac, val_frac, test_frac,
            strategy, n_jobs}`` overrides; defaults match shard 21 §A (k=10 /
            70/15/15 / shuffle_split / n_jobs=1).
        """
        full_data = input_data.get("full_data")
        if not isinstance(full_data, dict) or "X" not in full_data or "y" not in full_data:
            raise ValueError(
                "evaluation_mode='repeated_k10' requires input_data['full_data'] = "
                "{'X': pd.DataFrame, 'y': pd.Series}"
            )

        X = full_data["X"]
        y = full_data["y"]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X))
        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y), name="y")

        seed_base = int(input_data.get("seed_base", 42))
        cfg = input_data.get("repeated_splits_config", {}) or {}
        n_jobs = max(1, int(cfg.get("n_jobs", 1)))
        splitter = RepeatedStratifiedSplitter(
            k=int(cfg.get("k", 10)),
            seed_base=seed_base,
            train_frac=float(cfg.get("train_frac", 0.70)),
            val_frac=float(cfg.get("val_frac", 0.15)),
            test_frac=float(cfg.get("test_frac", 0.15)),
            strategy=str(cfg.get("strategy", "shuffle_split")),
        )

        fold_specs: List[FoldSpec] = list(splitter.split(X, y))

        logger.info(
            f"_run_repeated_splits: starting k={splitter.k} folds "
            f"(seed_base={seed_base}, strategy={splitter.strategy}, n_jobs={n_jobs})"
        )

        # Open parent MLflow run wrapping the fold loop. Best-effort: if the
        # connector is unavailable, fall back to None and continue without
        # parent-level aggregate logging (per-fold runs still emit when their
        # own connector path succeeds).
        mlflow_conn = self._get_mlflow_connector_or_none()
        parent_experiment_id: Optional[str] = None
        parent_tags = {
            "evaluation_mode": "repeated_k10",
            "k": str(splitter.k),
            "seed_base": str(seed_base),
            "splitter_strategy": splitter.strategy,
            "n_jobs": str(n_jobs),  # cycle-16 C-2
            "source": "model_trainer_agent",
        }
        parent_run_name = f"repeated_k10_seed{seed_base}"
        experiment_id = input_data.get("experiment_id", "model_trainer_repeated")
        experiment_name = input_data.get("experiment_name", f"model_trainer_{experiment_id}")

        if mlflow_conn is not None:
            try:
                parent_experiment_id = await mlflow_conn.get_or_create_experiment(
                    name=experiment_name,
                    tags={"source": "model_trainer_agent", "mode": "repeated_k10"},
                )
            except Exception as exc:  # noqa: BLE001 — best-effort MLflow
                logger.warning(
                    f"_run_repeated_splits: parent experiment creation failed: {exc!r}; "
                    "continuing without parent-level aggregate logging"
                )
                mlflow_conn = None

        fold_outputs: List[Dict[str, Any]] = [{} for _ in fold_specs]
        fold_metrics: List[Dict[str, Any]] = [{} for _ in fold_specs]

        async def _execute_one_fold(spec: FoldSpec) -> None:
            """Run one fold; populate fold_outputs[idx] + fold_metrics[idx]."""
            idx = spec.fold_idx
            fold_input = self._build_fold_input(input_data, X, y, spec)
            try:
                fold_output = await self.run(fold_input)
                fold_outputs[idx] = fold_output
                fold_metrics[idx] = {
                    "fold_idx": idx,
                    "fold_random_state": spec.seed,
                    "fold_status": "ok",
                    "test_metrics": fold_output.get("test_metrics", {}),
                    "validation_metrics": fold_output.get("validation_metrics", {}),
                    "train_metrics": fold_output.get("train_metrics", {}),
                    "auc_roc": fold_output.get("auc_roc"),
                    "brier_score": fold_output.get("brier_score"),
                    "mlflow_run_id": fold_output.get("mlflow_run_id"),
                }
            except Exception as exc:  # noqa: BLE001 — cycle-15 I-3 partial contract
                logger.warning(
                    f"_run_repeated_splits: fold {idx} (seed={spec.seed}) failed: {exc!r}"
                )
                fold_outputs[idx] = {}
                fold_metrics[idx] = {
                    "fold_idx": idx,
                    "fold_random_state": spec.seed,
                    "fold_status": "failed",
                    "exception_repr": repr(exc),
                }

        # NEP 19 per-fold MLflow params (cycle-15 I-2): logged to the parent
        # run since the per-fold child runs are opened deeper in the graph
        # (mlflow_logger node). We log a per-fold flat dict so the parent
        # carries the full provenance trace even when individual children
        # fail to open.
        nep19_params: Dict[str, Any] = {
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        }
        for spec in fold_specs:
            nep19_params[f"fold_{spec.fold_idx:02d}_seed_base"] = seed_base
            nep19_params[f"fold_{spec.fold_idx:02d}_derived_seed"] = spec.seed

        async def _log_aggregate_to_parent(run, agg: Dict[str, AggregateStat]) -> None:
            """Log aggregate metrics to the parent MLflow run.

            Cycle-16 I-1 (Q1-C): emits ``aggregate_<metric>_bca_unstable`` as
            1.0|0.0 per metric so MLflow UI consumers can distinguish reliable
            BCa CIs from degenerate fallbacks where ``bca_ci_lo/hi`` are None
            and the percentile_ci should be preferred for downstream gates.

            Cycle-17 COSMETIC-1/2: also emits a parent-level summary —
            ``aggregate_bca_unstable_metric_count`` and ``_fraction`` so MLflow
            UI consumers comparing many runs can rank by BCa-unstable density
            without scanning every per-metric flag, plus a ``has_bca_unstable``
            tag (string) so non-chart consumers (run-list filters, run-search
            queries) can branch on a single boolean rather than a numeric metric.
            """
            metrics_payload: Dict[str, float] = {}
            n_unstable = 0
            n_total = 0
            for metric_name, stat in agg.items():
                metrics_payload[f"aggregate_{metric_name}_mean"] = stat.mean
                metrics_payload[f"aggregate_{metric_name}_std"] = stat.std
                metrics_payload[f"aggregate_{metric_name}_n_folds"] = float(stat.n_folds)
                metrics_payload[f"aggregate_{metric_name}_percentile_lo"] = stat.percentile_ci_lo
                metrics_payload[f"aggregate_{metric_name}_percentile_hi"] = stat.percentile_ci_hi
                if stat.bca_ci_lo is not None:
                    metrics_payload[f"aggregate_{metric_name}_bca_lo"] = stat.bca_ci_lo
                if stat.bca_ci_hi is not None:
                    metrics_payload[f"aggregate_{metric_name}_bca_hi"] = stat.bca_ci_hi
                metrics_payload[f"aggregate_{metric_name}_bca_unstable"] = (
                    1.0 if stat.bca_unstable_warning else 0.0
                )
                n_total += 1
                if stat.bca_unstable_warning:
                    n_unstable += 1
            if n_total > 0:
                metrics_payload["aggregate_bca_unstable_metric_count"] = float(n_unstable)
                metrics_payload["aggregate_bca_unstable_metric_fraction"] = float(
                    n_unstable
                ) / float(n_total)
            # Cycle-18 IMPORTANT-3 (Q4.A): wrap log_metrics in try/except so a
            # connector failure here cannot propagate past _log_aggregate_to_parent
            # and silently skip the cycle-17 I-4 partial-failure observability
            # block (n_failed_folds metric + aggregate_status tag) at the call
            # site. Symmetric with the set_tags handler immediately below.
            if metrics_payload:
                try:
                    await run.log_metrics(metrics_payload)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"parent aggregate metrics logging failed: {exc!r}")
            # COSMETIC-1: mirror the unstable flag as a string tag so MLflow
            # run-search queries (which can filter by tag but not by metric
            # value) can quickly find runs with any unstable BCa CI.
            try:
                await run.set_tags({"has_bca_unstable": "true" if n_unstable > 0 else "false"})
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"parent has_bca_unstable tag logging failed: {exc!r}")

        async def _run_all_folds() -> None:
            if n_jobs == 1:
                for spec in fold_specs:
                    await _execute_one_fold(spec)
            else:
                semaphore = asyncio.Semaphore(n_jobs)

                async def _bounded(spec: FoldSpec) -> None:
                    async with semaphore:
                        await _execute_one_fold(spec)

                await asyncio.gather(*(_bounded(s) for s in fold_specs))

        if mlflow_conn is not None and parent_experiment_id is not None:
            try:
                async with mlflow_conn.start_run(
                    experiment_id=parent_experiment_id,
                    run_name=parent_run_name,
                    tags=parent_tags,
                    description=(
                        f"Repeated k={splitter.k} train/val/test splits "
                        f"(seed_base={seed_base}, strategy={splitter.strategy})"
                    ),
                ) as parent_run:
                    parent_run_id = parent_run.run_id
                    try:
                        await parent_run.log_params(nep19_params)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"NEP 19 parent param logging failed: {exc!r}")
                    await _run_all_folds()
                    aggregate = aggregate_fold_metrics(fold_metrics)
                    await _log_aggregate_to_parent(parent_run, aggregate)
                    # Cycle-17 IMPORTANT-4: surface partial-failure observability
                    # at the parent run BEFORE it closes, so MLflow UI consumers
                    # can filter by aggregate_status / n_failed_folds without
                    # opening every child run.
                    n_failed_folds = sum(
                        1 for fm in fold_metrics if fm.get("fold_status") == "failed"
                    )
                    aggregate_status = "PARTIAL" if n_failed_folds > 0 else "COMPLETE"
                    try:
                        await parent_run.log_metrics({"n_failed_folds": float(n_failed_folds)})
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"parent n_failed_folds metric logging failed: {exc!r}")
                    try:
                        await parent_run.set_tags(
                            {
                                "aggregate_status": aggregate_status,
                                "n_failed_folds": str(n_failed_folds),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Cycle-18 COSMETIC-2 (Q4.C): aggregate_status is the
                        # primary consumer-visible signal for partial-failure
                        # runs; a silent failure to emit it would leave
                        # operators without visibility. WARNING (not DEBUG) so
                        # default log levels surface the issue.
                        logger.warning(f"parent aggregate_status tag logging failed: {exc!r}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"_run_repeated_splits: parent MLflow run wrapper failed: {exc!r}; "
                    "fold loop ran outside the parent context"
                )
                parent_run_id = None
                if not any(fm for fm in fold_metrics):
                    await _run_all_folds()
                aggregate = aggregate_fold_metrics(fold_metrics)
        else:
            parent_run_id = None
            await _run_all_folds()
            aggregate = aggregate_fold_metrics(fold_metrics)

        n_failed = sum(1 for fm in fold_metrics if fm.get("fold_status") == "failed")
        aggregate_status = "PARTIAL" if n_failed > 0 else "COMPLETE"

        logger.info(
            f"_run_repeated_splits: completed {len(fold_metrics) - n_failed}/{len(fold_metrics)} "
            f"folds OK (seed_base={seed_base}, status={aggregate_status})"
        )

        # Compose top-level output: forward fold-0 fields for legacy consumers
        # (test_metrics shape preserved per shard 21 Q-W3-6 RESOLVED — Option D)
        # and append the new repeated-mode fields so downstream gate-promotion
        # logic can branch on `evaluation_mode` + consume `aggregate_metrics`.
        primary = next((fo for fo in fold_outputs if fo), {})
        output: Dict[str, Any] = dict(primary)
        output["evaluation_mode"] = "repeated_k10"
        output["fold_metrics"] = fold_metrics
        output["aggregate_metrics"] = aggregate
        output["aggregate_status"] = aggregate_status
        output["test_metrics_population_strategy"] = "fold_mean"
        output["evaluation_result_schema_version"] = "adaptive_criteria_v3.phase1.1"
        output["legacy_projection_warning"] = (
            "test_metrics in repeated_k10 mode is fold-0; downstream callers "
            "MUST consume aggregate_metrics for promotion-gate logic"
        )
        output["seed_base"] = seed_base
        output["k_folds"] = splitter.k
        output["splitter_strategy"] = splitter.strategy
        output["n_jobs"] = n_jobs
        output["parent_mlflow_run_id"] = parent_run_id
        return output

    @staticmethod
    def _get_mlflow_connector_or_none():
        """Lazy-import MLflow connector (avoids circular deps + test-friendly None)."""
        try:
            from src.mlops.mlflow_connector import get_mlflow_connector

            return get_mlflow_connector()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"MLflow connector unavailable: {exc!r}")
            return None

    def _build_fold_input(
        self,
        input_data: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        spec: FoldSpec,
    ) -> Dict[str, Any]:
        """Materialize a per-fold ``input_data`` for recursive ``self.run`` invocation.

        Slices the full ``(X, y)`` per ``spec.{train_idx, val_idx, test_idx}`` and
        injects them into the legacy ``train_data`` / ``validation_data`` /
        ``test_data`` / ``holdout_data`` dict shape consumed by ``load_splits``.
        Holdout is empty (placeholder satisfying the validator) — repeated_k10
        does not produce a held-out split per shard 21 §A; the holdout-locked
        contract is single-mode-only.

        Side-effects:
          - Preserves ``evaluation_mode = "repeated_k10"`` on the per-fold input
            so downstream nodes (split_enforcer, evaluator, mlflow_logger) apply
            repeated-mode logic. Recursion-termination is the
            ``repeated_mode_fold_invocation = True`` sentinel — it is what
            prevents the recursive ``self.run`` invocation from re-entering
            ``_run_repeated_splits``, NOT a switch to ``evaluation_mode="single"``.
          - Sets ``fold_random_state = spec.seed`` on the per-fold input so
            ``resolve_fold_random_state`` (Day-3) returns the fold's seed in
            ``split_loader`` / ``hyperparameter_tuner`` / ``model_trainer_node``.
          - Strips ``full_data`` from the per-fold input (no longer needed
            after the splitter materialized indices).
        """
        # Slice WITHOUT reset_index — the splitter's positional indices into the
        # full dataset are already pairwise-disjoint (test_splitter_index_disjointness
        # locks this), so preserving them lets `split_enforcer._check_duplicate_indices`
        # see the disjoint sets directly. We ALSO emit explicit `indices` lists so the
        # enforcer's `_get_indices` short-circuit picks them up unambiguously regardless
        # of upstream pandas-index gymnastics.
        X_train = X.iloc[spec.train_idx]
        y_train = y.iloc[spec.train_idx]
        X_val = X.iloc[spec.val_idx]
        y_val = y.iloc[spec.val_idx]
        X_test = X.iloc[spec.test_idx]
        y_test = y.iloc[spec.test_idx]

        empty_X = X.iloc[:0]
        empty_y = y.iloc[:0]

        per_fold = dict(input_data)
        per_fold.pop("full_data", None)
        # `evaluation_mode="repeated_k10"` is preserved on the per-fold input so
        # downstream nodes (split_enforcer, mlflow_logger) can branch on the
        # active mode; the orchestrator-level dispatch is skipped by the
        # `repeated_mode_fold_invocation=True` sentinel so the recursive
        # `self.run` call falls through to the legacy single-graph path
        # without re-entering `_run_repeated_splits`.
        per_fold["evaluation_mode"] = "repeated_k10"
        per_fold["repeated_mode_fold_invocation"] = True
        per_fold["fold_random_state"] = spec.seed
        per_fold["fold_idx"] = spec.fold_idx
        per_fold["train_data"] = {
            "X": X_train,
            "y": y_train,
            "row_count": len(X_train),
            "indices": spec.train_idx.tolist(),
        }
        per_fold["validation_data"] = {
            "X": X_val,
            "y": y_val,
            "row_count": len(X_val),
            "indices": spec.val_idx.tolist(),
        }
        per_fold["test_data"] = {
            "X": X_test,
            "y": y_test,
            "row_count": len(X_test),
            "indices": spec.test_idx.tolist(),
        }
        per_fold["holdout_data"] = {
            "X": empty_X,
            "y": empty_y,
            "row_count": 0,
            "indices": [],
        }
        return per_fold

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
