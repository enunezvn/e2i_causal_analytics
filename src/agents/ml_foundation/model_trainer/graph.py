"""LangGraph workflow for model_trainer agent."""

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from .nodes import (
    apply_resampling,
    check_qc_gate,
    detect_class_imbalance,
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


def create_model_trainer_graph() -> StateGraph:
    """Create model_trainer LangGraph workflow.

    Pipeline (11 nodes):
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
    - MLflow logs all metrics, params, and model artifacts
    - Checkpoint saves model to disk for persistence
    """
    workflow = StateGraph(ModelTrainerState)

    # Add nodes (11 total)
    workflow.add_node("check_qc_gate", check_qc_gate)
    workflow.add_node("load_splits", load_splits)
    workflow.add_node("enforce_splits", enforce_splits)
    workflow.add_node("detect_class_imbalance", detect_class_imbalance)
    workflow.add_node("fit_preprocessing", fit_preprocessing)
    workflow.add_node("apply_resampling", apply_resampling)
    workflow.add_node("tune_hyperparameters", tune_hyperparameters)
    workflow.add_node("train_model", train_model)
    workflow.add_node("evaluate_model", evaluate_model)
    workflow.add_node("log_to_mlflow", log_to_mlflow)
    workflow.add_node("save_checkpoint", save_checkpoint)

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

    # Evaluation → MLflow logging (always)
    workflow.add_edge("evaluate_model", "log_to_mlflow")

    # MLflow logging → checkpointing (always)
    workflow.add_edge("log_to_mlflow", "save_checkpoint")

    # Checkpointing → END
    workflow.add_edge("save_checkpoint", END)

    return workflow.compile()
