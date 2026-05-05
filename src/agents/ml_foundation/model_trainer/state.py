"""State definition for model_trainer agent.

Migrated from ``TypedDict(total=False)`` to pydantic v2 ``BaseModel``
in Shard B of the migration tracked at
``.claude/plans/typeddict_to_pydantic_migration_plan_20260504.md``.

Inherits from ``BaseAgentSchema`` (extra="allow",
arbitrary_types_allowed=True, dict-like accessors) so the existing
``state["key"]`` / ``state.get("key", default)`` call sites in
``model_trainer/nodes/`` work unchanged. Per Decision 8a, every field
is ``Optional[T] = None`` except ``audit_workflow_id`` (UUID with
``Field(default_factory=uuid4)`` matching scope_definer + data_preparer).

The ``arbitrary_types_allowed`` setting from BaseAgentSchema is
load-bearing here — model_trainer state holds 15+ non-pydantic
runtime objects: ``trained_model: Any`` (sklearn/xgboost),
``preprocessor: Any`` (fitted Pipeline), ``X_train_resampled``,
``X_train_preprocessed`` etc. (numpy arrays / pandas DataFrames).

NOTE on ``_repeated_mode_fold_invocation``: pydantic v2 reserves
underscore-prefixed names for private attributes — the field cannot
be declared directly with a leading underscore. The agent code at
``model_trainer/agent.py:1096`` passes the key as a dict-style update
(`per_fold["_repeated_mode_fold_invocation"] = True`); under the
BaseAgentSchema ``extra="allow"`` config + dict-like ``__setitem__``,
this routes into ``model_extra`` and remains accessible via
``state.get("_repeated_mode_fold_invocation", False)``. Behavior is
preserved at runtime; type-level documentation of this sentinel is
deferred to a follow-up sub-shard that renames the call sites.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import Field

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
)


class ModelTrainerState(BaseAgentSchema):
    """State for model_trainer agent.

    The model_trainer executes the complete ML training pipeline with strict
    split enforcement, hyperparameter optimization, and MLflow logging.
    """

    # === INPUT FIELDS ===
    # From model_selector
    model_candidate: Optional[Dict[str, Any]] = None  # Complete ModelCandidate
    algorithm_name: Optional[str] = None
    algorithm_class: Optional[str] = None
    hyperparameter_search_space: Optional[Dict[str, Dict[str, Any]]] = None  # Optuna search space
    default_hyperparameters: Optional[Dict[str, Any]] = None

    # From data_preparer
    qc_report: Optional[Dict[str, Any]] = None  # QC validation report
    experiment_id: Optional[str] = None
    feast_fallback_used: Optional[bool] = None  # data_preparer Feast historical-features fallback

    # From scope_definer
    success_criteria: Optional[Dict[str, Any]] = None  # Performance thresholds to meet
    problem_type: Optional[str] = None  # binary_classification, regression, etc.
    # Block 5: optional business cost matrix (tp/fp/fn/tn dollar values).
    # When set, the evaluator computes business_utility from the confusion
    # matrix at the chosen (validation-tuned) threshold and adds it to
    # validation_metrics + test_metrics (#10).
    cost_matrix: Optional[Dict[str, Any]] = None

    # Training configuration
    enable_hpo: Optional[bool] = None
    hpo_trials: Optional[int] = None
    hpo_timeout_hours: Optional[float] = None
    early_stopping: Optional[bool] = None
    early_stopping_patience: Optional[int] = None
    enable_mlflow: Optional[bool] = None
    enable_checkpointing: Optional[bool] = None

    # W3-lite Day 3 + Day 4 (shard 17 W3 rows Day 3-4 + shard 21 §A/§B):
    # repeated train/val/test fold-iteration plumbing. See module docstring
    # for the ``_repeated_mode_fold_invocation`` sentinel handling.
    evaluation_mode: Optional[str] = None  # "single" (default) | "repeated_k10"
    fold_random_state: Optional[int] = None  # Per-fold seed
    fold_idx: Optional[int] = None  # 0..k-1 fold index

    # === INTERMEDIATE FIELDS ===
    # QC Gate
    qc_gate_passed: Optional[bool] = None
    qc_gate_message: Optional[str] = None

    # Data Splits
    train_data: Optional[Dict[str, Any]] = None  # (X, y, row_count)
    validation_data: Optional[Dict[str, Any]] = None
    test_data: Optional[Dict[str, Any]] = None
    holdout_data: Optional[Dict[str, Any]] = None  # LOCKED

    # Split Validation
    min_samples_per_split: Optional[int] = None  # Minimum viable samples per split
    split_ratios_valid: Optional[bool] = None
    train_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None
    holdout_samples: Optional[int] = None
    total_samples: Optional[int] = None
    train_ratio: Optional[float] = None  # Should be ~0.60
    validation_ratio: Optional[float] = None  # Should be ~0.20
    test_ratio: Optional[float] = None  # Should be ~0.15
    holdout_ratio: Optional[float] = None  # Should be ~0.05
    split_validation_message: Optional[str] = None
    split_ratio_checks: Optional[List[str]] = None
    leakage_warnings: Optional[List[str]] = None

    # Class Imbalance Detection
    imbalance_detected: Optional[bool] = None
    imbalance_ratio: Optional[float] = None  # Majority/minority ratio
    minority_ratio: Optional[float] = None  # Minority class percentage
    imbalance_severity: Optional[str] = None  # none, moderate, severe, extreme
    class_distribution: Optional[Dict[int, int]] = None  # {0: 800, 1: 77}
    recommended_strategy: Optional[str] = None  # smote, random_oversample, class_weight, etc.
    strategy_rationale: Optional[str] = None  # LLM explanation

    # Resampling Results
    X_train_resampled: Optional[Any] = None  # Resampled features (np.ndarray / DataFrame)
    y_train_resampled: Optional[Any] = None  # Resampled labels
    resampling_applied: Optional[bool] = None
    resampling_strategy: Optional[str] = None
    original_train_shape: Optional[Tuple[int, ...]] = None  # Shape before resampling
    resampled_train_shape: Optional[Tuple[int, ...]] = None  # Shape after resampling
    original_distribution: Optional[Dict[int, int]] = None  # Class counts before
    resampled_distribution: Optional[Dict[int, int]] = None  # Class counts after

    # Feature Names (preserved from data_preparer)
    feature_columns: Optional[List[str]] = None

    # Preprocessing
    preprocessor: Optional[Any] = None  # Fitted pipeline (fit on train only)
    X_train_preprocessed: Optional[Any] = None
    X_validation_preprocessed: Optional[Any] = None
    X_test_preprocessed: Optional[Any] = None
    preprocessing_statistics: Optional[Dict[str, Any]] = None  # Stats from train split

    # Hyperparameter Tuning
    hpo_completed: Optional[bool] = None
    hpo_best_trial: Optional[int] = None
    best_hyperparameters: Optional[Dict[str, Any]] = None
    hpo_trials_run: Optional[int] = None
    hpo_duration_seconds: Optional[float] = None

    # Model Training
    trained_model: Optional[Any] = None  # Trained model (sklearn / xgboost / etc.)
    training_duration_seconds: Optional[float] = None
    early_stopped: Optional[bool] = None
    final_epoch: Optional[int] = None

    # Model Evaluation
    # Metrics dicts hold mixed-type values at runtime (numeric metrics +
    # string metadata like ``chosen_threshold_source`` + nested dicts like
    # ``net_benefit_grid``). The TypedDict declaration ``Dict[str, float]``
    # was aspirational; pydantic strict-validates so we widen to ``Any``.
    train_metrics: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Dict[str, Any]] = None  # FINAL test-set metrics

    # Classification Metrics (problem-type specific)
    auc_roc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    pr_auc: Optional[float] = None
    confusion_matrix: Optional[Dict[str, int]] = None  # TP, TN, FP, FN

    # Regression Metrics (problem-type specific)
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

    # Calibration (classification only)
    brier_score: Optional[float] = None
    calibration_error: Optional[float] = None  # Expected Calibration Error (ECE)
    calibrated_ece: Optional[float] = None  # ECE after post-hoc calibration
    calibration_analysis: Optional[Dict[str, Any]] = None  # Full calibration curve data
    calibrated_test_metrics: Optional[Dict[str, Any]] = None  # Metrics after post-hoc cal
    post_hoc_calibration: Optional[Dict[str, Any]] = None  # Calibration method info

    # Imbalance-Robust Metrics
    mcc: Optional[float] = None  # Matthews Correlation Coefficient

    # Threshold Analysis
    optimal_threshold: Optional[float] = None  # Youden's J
    f1_threshold_analysis: Optional[Dict[str, Any]] = None  # F1-optimal threshold + metrics
    precision_at_k: Optional[Dict[int, float]] = None  # {100: 0.35, 500: 0.28}; numeric-only

    # Imbalance-Aware Evaluation (from evaluator, propagated through agent)
    precision_constrained: Optional[Dict[str, Any]] = None  # Precision-constrained threshold info
    minority_recall: Optional[float] = None  # Recall on minority class at optimal threshold
    minority_precision: Optional[float] = None  # Precision on minority class at optimal threshold
    test_metrics_at_optimal: Optional[Dict[str, Any]] = None  # At optimal threshold
    test_metrics_at_05: Optional[Dict[str, Any]] = None  # At standard 0.5 threshold

    # Permutation Test
    permutation_test: Optional[Dict[str, Any]] = None  # p-value, shuffled AUC stats, verdict

    # Cross-Validation
    cv_results: Optional[Dict[str, Any]] = None  # Stratified k-fold metrics

    # Split Stratification
    split_validation: Optional[Dict[str, Any]] = None  # Class ratio drift across splits

    # Confidence Intervals
    confidence_interval: Optional[Dict[str, Tuple[float, ...]]] = None  # {'auc': (0.78, 0.85)}
    bootstrap_samples: Optional[int] = None

    # Success Criteria Check
    success_criteria_met: Optional[bool] = None
    success_criteria_results: Optional[Dict[str, bool]] = None  # Metric -> passed/failed

    # === OUTPUT FIELDS ===
    # Trained Model
    training_run_id: Optional[str] = None
    model_id: Optional[str] = None

    # MLflow Integration (populated by log_to_mlflow node)
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    mlflow_status: Optional[str] = None  # success, disabled, skipped, failed
    mlflow_model_uri: Optional[str] = None  # runs:/<run_id>/model
    mlflow_registered: Optional[bool] = None  # Registered in MLflow registry
    mlflow_model_version: Optional[str] = None
    mlflow_model_name: Optional[str] = None
    db_training_run_id: Optional[str] = None  # Database training run ID (UUID-string)
    # Legacy fields (kept for compatibility)
    model_artifact_uri: Optional[str] = None  # Deprecated, use mlflow_model_uri
    preprocessing_artifact_uri: Optional[str] = None
    registered_model_name: Optional[str] = None  # Deprecated
    model_version: Optional[int] = None  # Deprecated
    model_stage: Optional[str] = None  # MLflow stage (Staging, Production)

    # Artifacts
    model_artifact_path: Optional[str] = None  # Local model artifact path
    preprocessing_artifact_path: Optional[str] = None

    # Timing
    training_started_at: Optional[str] = None  # ISO timestamp
    training_completed_at: Optional[str] = None
    total_training_duration_seconds: Optional[float] = None  # End-to-end

    # Status
    training_status: Optional[str] = None  # running, completed, failed
    training_error: Optional[str] = None

    # Metadata
    framework: Optional[str] = None  # ML framework (econml, xgboost, sklearn, etc.)
    trained_by: Optional[str] = None  # Agent name
    created_at: Optional[str] = None  # ISO timestamp

    # Database
    persisted_to_db: Optional[bool] = None  # Saved to ml_training_runs table

    # Leakage Suspicion (post-training)
    leakage_suspected: Optional[bool] = None  # Metrics suggest data leakage
    suspicion_level: Optional[str] = None  # "critical" / "high" / "none"
    suspicion_reasons: Optional[List[str]] = None
    investigation_recommendations: Optional[List[str]] = None

    # Quality Remediation (post-evaluation inner loop)
    quality_remediation_status: Optional[str] = None  # not_needed | enhancing | improved | failed
    quality_remediation_attempts: Optional[int] = None  # counter for inner loop
    quality_remediation_max_attempts: Optional[int] = None  # default 2
    quality_remediation_history: Optional[List[Dict[str, Any]]] = None
    enhanced_search_space: Optional[Dict[str, Any]] = None  # Search space + regularization

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Audit chain — Decision 7a: typed as UUID with str↔UUID coercion via the
    # validator factory. ``default_factory=uuid4`` matches scope_definer +
    # data_preparer convention so existing agent flows that construct
    # ModelTrainerState as a dict literal without audit_workflow_id keep
    # working. A future sub-shard tightens to "caller MUST provide".
    audit_workflow_id: UUID = Field(default_factory=uuid4)

    _validate_audit_id = audit_workflow_id_validator()
