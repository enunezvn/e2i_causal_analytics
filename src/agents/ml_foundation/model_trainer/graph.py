"""LangGraph workflow for model_trainer agent."""

from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    apply_resampling,
    augment_training_data,
    check_qc_gate,
    detect_class_imbalance,
    diagnose_and_remediate_quality,
    enforce_splits,
    evaluate_model,
    feature_ceiling_diagnostic,
    fit_preprocessing,
    learning_curve,
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
        # Route through the opt-in augmentation node first (no-op unless the
        # operator set augmentation_data_path). Ratio validation has already
        # run on the REAL splits, so augmenting train afterward is safe.
        return "augment_training_data"
    return "end"


def _should_run_learning_curve(state: Dict[str, Any]) -> str:
    """F2: skip the 180s learning_curve diagnostic when it cannot help.

    The diagnostic exists to answer "why didn't the model pass + how much
    more data would close the gap." It is irrelevant when:

    - ``state['error']`` is set (an upstream node failed; the downstream
      conditional will route to END regardless of the diagnostic), or
    - ``success_criteria_met is True`` AND no ``always_run_learning_curve``
      override (the model passed; no gap to close).

    Routing the no-op cases past ``learning_curve`` avoids the ~180s
    walltime burn the diagnostic can take on slow proxy fits.
    """
    if state.get("error"):
        return "post_evaluation"
    if state.get("success_criteria_met") is True and not state.get(
        "always_run_learning_curve", False
    ):
        return "post_evaluation"
    return "learning_curve"


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


def _post_evaluation_passthrough(state: Dict[str, Any]) -> Dict[str, Any]:
    """F2: no-op pass-through used when ``learning_curve`` is bypassed.

    LangGraph requires every routed-to label to map to a real node. When
    ``_should_run_learning_curve`` routes ``evaluate_model`` past the
    diagnostic (error state or no gap to close), we still need a landing
    node before the post-eval conditional. This node returns an empty patch
    so state is preserved unchanged.
    """
    return {}


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
        augment_training_data (opt-in synthetic augmentation; no-op unless
                               augmentation_data_path is set — train only)
          ↓
        detect_class_imbalance (LLM-assisted)
          ↓
        fit_preprocessing (train only)
          ↓
        feature_ceiling_diagnostic (advisory; native separability ceiling)
          ↓
        apply_resampling (train only)
          ↓
        tune_hyperparameters (Optuna on validation)
          ↓
        train_model (train on train/resampled set)
          ↓
        evaluate_model (eval on train/val/test)
          ↓
        learning_curve (PR #463 — Phase 2 data-sufficiency diagnostic;
                        no-op if success_criteria_met=True)
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

    # Add nodes
    workflow.add_node("check_qc_gate", check_qc_gate)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("load_splits", load_splits)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("enforce_splits", enforce_splits)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("augment_training_data", augment_training_data)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("detect_class_imbalance", detect_class_imbalance)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("fit_preprocessing", fit_preprocessing)  # type: ignore[type-var,arg-type,call-overload]
    # Advisory separability diagnostic — runs on the preprocessed train BEFORE
    # resampling so it measures the native feature ceiling (does not alter flow).
    workflow.add_node("feature_ceiling_diagnostic", feature_ceiling_diagnostic)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("apply_resampling", apply_resampling)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("tune_hyperparameters", tune_hyperparameters)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("train_model", train_model)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("evaluate_model", evaluate_model)  # type: ignore[type-var,arg-type,call-overload]
    # PR #463 Phase 2: post-training learning-curve diagnostic. The node
    # itself short-circuits when ``success_criteria_met`` is True so the
    # wiring is unconditional — keeping the graph topology simple.
    workflow.add_node("learning_curve", learning_curve)  # type: ignore[type-var,arg-type,call-overload]
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
            "augment_training_data": "augment_training_data",
            "end": END,
        },
    )

    # Opt-in synthetic augmentation (no-op unless augmentation_data_path set) →
    # class imbalance detection. Placed AFTER enforce_splits (ratios validated
    # on real data) and BEFORE preprocessing so any synthetic rows flow through
    # the same preprocessing/resampling as real data.
    workflow.add_edge("augment_training_data", "detect_class_imbalance")

    # Class imbalance detection → preprocessing (always)
    workflow.add_edge("detect_class_imbalance", "fit_preprocessing")

    # Preprocessing → separability diagnostic → resampling (always).
    # The diagnostic is advisory (emits feature_ceiling_* state) and sits here
    # so it sees the preprocessed, pre-resampling train — the native ceiling.
    workflow.add_edge("fit_preprocessing", "feature_ceiling_diagnostic")
    workflow.add_edge("feature_ceiling_diagnostic", "apply_resampling")

    # Resampling → HPO (always)
    workflow.add_edge("apply_resampling", "tune_hyperparameters")

    # HPO → training (always)
    workflow.add_edge("tune_hyperparameters", "train_model")

    # Training → evaluation (always)
    workflow.add_edge("train_model", "evaluate_model")

    # F2: Evaluation → conditional. Route to ``learning_curve`` only when
    # the diagnostic has something to compute (model failed AND no upstream
    # error). On error or pass-without-override, skip straight to the
    # post-evaluation conditional so we don't burn 180s on a doomed run.
    workflow.add_conditional_edges(
        "evaluate_model",
        _should_run_learning_curve,
        {
            "learning_curve": "learning_curve",
            # Synthetic node name: bypass routes directly to the same
            # post-evaluation decision so behavior matches what would have
            # happened had ``learning_curve`` returned ``{}``.
            "post_evaluation": "post_evaluation",
        },
    )

    # post_evaluation is a no-op pass-through node — it exists so the
    # bypass edge has a target that itself routes via the existing
    # ``_should_proceed_after_evaluation`` conditional. Defining it inline
    # as a lambda is not possible (LangGraph nodes must be hashable),
    # so we use a real function below.
    workflow.add_node("post_evaluation", _post_evaluation_passthrough)  # type: ignore[type-var,arg-type,call-overload]

    # Both learning_curve and the bypass node feed the post-eval conditional.
    workflow.add_conditional_edges(
        "learning_curve",
        _should_proceed_after_evaluation,
        {
            "quality_remediation": "quality_remediation",
            "log_to_mlflow": "log_to_mlflow",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "post_evaluation",
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
