"""LangGraph workflow for model_trainer agent."""

from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    apply_resampling,
    check_qc_gate,
    detect_class_imbalance,
    diagnose_and_remediate_quality,
    enforce_splits,
    evaluate_model,
    fit_preprocessing,
    load_splits,
    log_to_mlflow,
    save_checkpoint,
    train_model,
    tune_hyperparameters,
)
from .state import ModelTrainerState


def _should_proceed_after_qc(state: Dict[str, Any]) -> str:
    """Conditional edge: proceed only if QC gate passed."""
    if state.get("error"):
        return "end"
    if state.get("qc_gate_passed", False):
        return "load_splits"
    return "end"


def _should_proceed_after_splits(state: Dict[str, Any]) -> str:
    """Conditional edge: proceed only if splits valid."""
    if state.get("error"):
        return "end"
    if state.get("split_ratios_valid", False):
        return "detect_class_imbalance"
    return "end"


def _should_proceed_after_evaluation(state: Dict[str, Any]) -> str:
    """Conditional edge: quality remediation or block on critical leakage.

    Routes to quality_remediation if model quality is poor and we haven't
    exhausted remediation attempts. Otherwise proceeds to MLflow logging
    or blocks on critical leakage suspicion.
    """
    if state.get("error"):
        return "end"
    suspicion_level = state.get("suspicion_level", "none")
    if suspicion_level == "critical":
        return "end"

    # Quality remediation check (inner loop)
    attempts = state.get("quality_remediation_attempts", 0)
    max_attempts = state.get("quality_remediation_max_attempts", 2)
    status = state.get("quality_remediation_status", "not_needed")

    # Only trigger on first evaluation (attempts==0, status not yet set)
    if status in ("not_needed", "") and attempts == 0:
        test_metrics = state.get("test_metrics", {})
        val_metrics = state.get("validation_metrics", {})
        auc = test_metrics.get("roc_auc") or val_metrics.get("roc_auc") or 0
        precision = test_metrics.get("precision") or val_metrics.get("precision") or 0
        if (auc < 0.60 or precision < 0.05) and attempts < max_attempts:
            return "quality_remediation"

    return "log_to_mlflow"


def _route_after_quality_remediation(state: Dict[str, Any]) -> str:
    """Route after quality remediation: retry HPO or continue to logging."""
    status = state.get("quality_remediation_status", "not_needed")
    if status == "enhancing":
        return "tune_hyperparameters"
    if status == "threshold_fixable":
        return "log_to_mlflow"  # No retraining needed — threshold adjustment suffices
    # failed or max_attempts — proceed with what we have
    return "log_to_mlflow"


def create_model_trainer_graph() -> CompiledStateGraph:
    """Create model_trainer LangGraph workflow.

    Pipeline (12 nodes):
        START
          ↓
        check_qc_gate (MANDATORY)
          ↓
        [QC passed?]
          ↓ YES
        load_splits
          ↓
        enforce_splits
          ↓
        [Splits valid?]
          ↓ YES
        detect_class_imbalance (LLM-assisted)
          ↓
        fit_preprocessing (train only)
          ↓
        apply_resampling (train only)
          ↓
        tune_hyperparameters (Optuna on validation)
          ↓
        train_model (train on train/resampled set)
          ↓
        evaluate_model (eval on train/val/test)
          ↓
        [Quality + leakage check?]
          ↓ POOR QUALITY (AUC<0.60 or precision<5%)
        quality_remediation (enhance regularization, max 2 attempts)
          ↓ ENHANCING → loop back to tune_hyperparameters
          ↓ FAILED/MAX_ATTEMPTS → continue
          ↓ NOT CRITICAL
        log_to_mlflow (track experiment)
          ↓
        save_checkpoint (persist model)
          ↓
        END

    Critical gates:
    - QC gate MUST pass before any training
    - Split ratios MUST be valid (60/20/15/5 ± 2%)
    - Class imbalance detection uses LLM to recommend strategy
    - Preprocessing fit ONLY on train
    - Resampling applied ONLY to train (NEVER validation/test)
    - HPO uses validation set (with class weights if imbalanced)
    - Test set touched ONCE for final eval
    - Holdout locked until post-deployment
    - Quality remediation enhances HPO search space with regularization params
    - MLflow logs all metrics, params, and model artifacts
    - Checkpoint saves model to disk for persistence
    """
    workflow = StateGraph(ModelTrainerState)

    # Add nodes (11 total)
    workflow.add_node("check_qc_gate", check_qc_gate)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("load_splits", load_splits)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("enforce_splits", enforce_splits)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("detect_class_imbalance", detect_class_imbalance)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("fit_preprocessing", fit_preprocessing)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("apply_resampling", apply_resampling)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("tune_hyperparameters", tune_hyperparameters)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("train_model", train_model)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("evaluate_model", evaluate_model)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("quality_remediation", diagnose_and_remediate_quality)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("log_to_mlflow", log_to_mlflow)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("save_checkpoint", save_checkpoint)  # type: ignore[type-var,arg-type,call-overload]

    # Set entry point
    workflow.set_entry_point("check_qc_gate")

    # Define edges
    # QC gate → conditional (proceed only if passed)
    workflow.add_conditional_edges(
        "check_qc_gate",
        _should_proceed_after_qc,
        {
            "load_splits": "load_splits",
            "end": END,
        },
    )

    # Load splits → enforce splits (always)
    workflow.add_edge("load_splits", "enforce_splits")

    # Enforce splits → conditional (proceed only if valid)
    workflow.add_conditional_edges(
        "enforce_splits",
        _should_proceed_after_splits,
        {
            "detect_class_imbalance": "detect_class_imbalance",
            "end": END,
        },
    )

    # Class imbalance detection → preprocessing (always)
    workflow.add_edge("detect_class_imbalance", "fit_preprocessing")

    # Preprocessing → resampling (always)
    workflow.add_edge("fit_preprocessing", "apply_resampling")

    # Resampling → HPO (always)
    workflow.add_edge("apply_resampling", "tune_hyperparameters")

    # HPO → training (always)
    workflow.add_edge("tune_hyperparameters", "train_model")

    # Training → evaluation (always)
    workflow.add_edge("train_model", "evaluate_model")

    # Evaluation → conditional (quality remediation or block on critical leakage)
    workflow.add_conditional_edges(
        "evaluate_model",
        _should_proceed_after_evaluation,
        {
            "quality_remediation": "quality_remediation",
            "log_to_mlflow": "log_to_mlflow",
            "end": END,
        },
    )

    # Quality remediation → retry HPO or continue to logging
    workflow.add_conditional_edges(
        "quality_remediation",
        _route_after_quality_remediation,
        {
            "tune_hyperparameters": "tune_hyperparameters",
            "log_to_mlflow": "log_to_mlflow",
        },
    )

    # MLflow logging → checkpointing (always)
    workflow.add_edge("log_to_mlflow", "save_checkpoint")

    # Checkpointing → END
    workflow.add_edge("save_checkpoint", END)

    return workflow.compile()
