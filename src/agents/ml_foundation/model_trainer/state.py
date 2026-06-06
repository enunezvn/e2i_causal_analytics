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

NOTE on ``repeated_mode_fold_invocation`` (sub-shard D4 closure): the
field is declared without a leading underscore so it can be a proper
LangGraph channel. The original underscore-prefixed convention was
incompatible with pydantic v2 (which reserves underscore-prefixed names
for private attributes) and with LangGraph 1.0 (which only propagates
declared fields, not ``model_extra``). See the field declaration below
for the rename rationale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from pydantic import AliasChoices, Field

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
)
from src.agents.ml_foundation.data_preparer.schemas import QCReportSchema
from src.agents.ml_foundation.model_trainer.schemas import MetricsSchema, OptunaDistribution
from src.agents.ml_foundation.scope_definer.schemas import SuccessCriteriaSchema


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
    # D2.1: typed encoding of Optuna search space entries. Each value
    # is a discriminated union by ``type`` field (int|float|categorical),
    # validated against ``OptunaDistribution`` from model_trainer/schemas.py.
    # Producer dict literals (e.g., ``{"type": "int", "low": 1, "high": 100}``)
    # validate cleanly into the right variant; consumer access via
    # ``config["low"]`` works through the dict-shim on _OptunaDistributionBase.
    hyperparameter_search_space: Optional[Dict[str, OptunaDistribution]] = None
    default_hyperparameters: Optional[Dict[str, Any]] = None

    # From data_preparer
    # D2.2: typed QC contract; consumer-contract fields qc_passed/qc_errors/
    # qc_warnings are now declared on QCReportSchema, removing the runner-side
    # normalization shim that previously patched them in.
    qc_report: Optional[QCReportSchema] = None  # QC validation report
    experiment_id: Optional[str] = None
    feast_fallback_used: Optional[bool] = None  # data_preparer Feast historical-features fallback

    # From scope_definer
    # D2.3: typed success criteria. Schema declares all 9 v3 adaptive gate
    # fields + 2 caller-injected consumer keys (clinical_threshold_range,
    # dataset_disease). Underscore-prefixed audit keys (_adaptive_skipped,
    # _adaptive_p_t, _adaptive_inputs) flow through model_extra per pydantic
    # v2 reserved-name rule (BaseAgentSchema's extra="allow" preserves them).
    success_criteria: Optional[SuccessCriteriaSchema] = None  # Performance thresholds to meet
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

    # v5 Gate B2 — survival modeling (Cox + RSF) target derivation.
    # ``enable_survival_modeling`` gates the survival_model_node; when
    # False (default), the node is a no-op and the binary pipeline is
    # unaffected. ``survival_time_days`` and ``survival_event`` are the
    # cohort-scoped target arrays derived in the node. ``np.ndarray``
    # is accepted under ``arbitrary_types_allowed`` from
    # BaseAgentSchema. Default ``bool = False`` for the gate matches
    # the B3 ``enable_feature_engineering`` codex L2 convention so the
    # ``if state["enable_survival_modeling"]:`` guard does not get None.
    # L5 codex pass-1: explicit np.ndarray (not Any) so editor tooling
    # catches list/scalar misuse.
    enable_survival_modeling: bool = False
    survival_time_days: Optional[np.ndarray] = None  # float days
    survival_event: Optional[np.ndarray] = None  # bool
    survival_manifest_source: Optional[str] = None  # echoes the manifest_source used
    survival_target_error: Optional[str] = None  # set if derivation raised

    # W3-lite Day 3 + Day 4 (shard 17 W3 rows Day 3-4 + shard 21 §A/§B):
    # repeated train/val/test fold-iteration plumbing.
    evaluation_mode: Optional[str] = None  # "single" (default) | "repeated_k10"
    fold_random_state: Optional[int] = None  # Per-fold seed
    fold_idx: Optional[int] = None  # 0..k-1 fold index

    # Day-5 (cycle-15 I-4): orchestrator sentinel — when ``_run_repeated_splits``
    # invokes ``self.run(per_fold)`` recursively, this is set to True on the
    # per-fold input. Per-fold nodes (notably ``mlflow_logger``) branch on it
    # to open NESTED MLflow runs under the parent run instead of new top-level
    # runs. Single-mode callers omit this field; default None acts as False.
    #
    # Sub-shard D4 (PR #53, post-2444fd8): renamed from
    # ``_repeated_mode_fold_invocation`` to drop the underscore prefix.
    # Pydantic v2 reserves underscore-prefixed names for private attributes,
    # so the original name could not be a declared model field. Without a
    # declared field, LangGraph 1.0 dropped the value from channel state on
    # every node coercion (``model_extra`` is NOT propagated through channels)
    # — ``mlflow_logger.py:192`` always read False in repeated_k10 mode,
    # breaking MLflow run nesting (codex review B2, 2026-05-05). Now that
    # this is a declared field, LangGraph treats it as a proper channel and
    # propagates the value through node invocations.
    #
    # Backward-compat alias (codex review N1, 2026-05-05): pre-PR-#53
    # checkpoints persisted with the original underscore-prefixed key still
    # need to deserialize cleanly. ``validation_alias=AliasChoices(...)``
    # accepts BOTH the canonical name (post-PR-#53) AND the legacy
    # underscore form (pre-PR-#53) at construction/validation time.
    #
    # ``populate_by_name=True`` (set on BaseAgentSchema model_config — see
    # ``_pydantic_utils.py``) is what enables construction by the python
    # field name when a ``validation_alias`` is also present. Without it,
    # only the alias forms would work at construction (codex review M1,
    # 2026-05-05; corrected from a prior comment that had the causal
    # direction inverted).
    #
    # Serialization uses the python field name
    # (``repeated_mode_fold_invocation``) — newly written checkpoints use
    # the canonical name, NOT the legacy alias. This is asymmetric on
    # purpose: read-old / write-new.
    #
    # Dual-key precedence (codex review I1, 2026-05-05): if a malformed
    # checkpoint contains BOTH the canonical key AND the legacy underscore
    # key, ``AliasChoices`` resolves to the FIRST alias in the declaration
    # order below — i.e., the canonical name ALWAYS wins regardless of
    # the order in the input dict. The runner-up key lands in
    # ``model_extra`` with its discarded value (because ``extra="allow"``
    # is set on BaseAgentSchema). The declaration order below is therefore
    # load-bearing for the precedence guarantee — keep canonical first.
    #
    # This scenario is extremely unlikely in practice (no real checkpoint
    # writer would emit both forms) but the residue in ``model_extra``
    # could confuse a future reader debugging a hand-crafted payload.
    # The unit test
    # ``test_model_trainer_state_dual_key_payload_canonical_wins``
    # pins the documented behavior so any future regression fires loud.
    repeated_mode_fold_invocation: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices(
            "repeated_mode_fold_invocation",  # canonical (post-PR-#53)
            "_repeated_mode_fold_invocation",  # legacy underscore (pre-PR-#53)
        ),
    )

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

    # Synthetic Augmentation (opt-in; Phase 3 preview consumption).
    # When ``augmentation_data_path`` is set, ``augment_training_data`` (runs
    # after enforce_splits, before fit_preprocessing) concatenates the reviewed
    # synthetic cohort into the TRAINING split only — never val/test/holdout.
    # The remaining fields are an audit trail (also surfaced on the agent output
    # and ``PipelineResult.training_augmentation``).
    augmentation_data_path: Optional[str] = None
    augmentation_applied: Optional[bool] = None
    augmentation_n_original: Optional[int] = None
    augmentation_n_synthetic: Optional[int] = None
    augmentation_source: Optional[str] = None
    augmentation_fingerprint: Optional[str] = None
    augmentation_skip_reason: Optional[str] = None

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

    # Feature-Separability Diagnostic (advisory; feature_ceiling_diagnostic node)
    # Distinguishes a feature/separability ceiling from a true imbalance problem.
    feature_ceiling_computed: Optional[bool] = None
    feature_ceiling_auc: Optional[float] = None  # native CV ROC-AUC (plain LR, no rebalancing)
    feature_ceiling_pr_auc: Optional[float] = None  # native CV average-precision
    feature_ceiling_prevalence: Optional[float] = None
    feature_ceiling_pr_auc_lift: Optional[float] = None  # pr_auc / prevalence (1.0 == no skill)
    feature_ceiling_label: Optional[str] = None  # feature_bound | intermediate | separable | not_computed
    feature_ceiling_note: Optional[str] = None  # plain-language recommendation
    feature_ceiling_n_eval: Optional[int] = None
    feature_ceiling_cv_folds: Optional[int] = None

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
    # #633: the DEPLOYED artifact. The evaluator builds a post-hoc
    # ``calibrated_model`` for diagnostics; when calibration is actually
    # applied this is the genuinely better-calibrated estimator that gets
    # MLflow-logged / checkpointed / returned (and whose probabilities the
    # v3 calibration gates are judged on). When calibration is skipped
    # (calibration-native algo, no val data, unapplied) the deployed model
    # stays the raw ``trained_model``. Declared here because LangGraph drops
    # undeclared state keys (extra="ignore"), which would silently revert
    # downstream nodes to the raw model.
    deployed_model: Optional[Any] = None
    calibration_applied: Optional[bool] = None  # True iff deployed_model is calibrated
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
    # D2.5b: typed metrics contract. MetricsSchema accepts both ``auc_roc``
    # (canonical/legacy) and ``roc_auc`` (modern producer) via AliasChoices,
    # and declares 14 extra fields beyond the original chore-PR set
    # (per-class precision/recall, mcc/pr_auc/brier_score, calibration
    # metrics, threshold metadata, lift/baseline) to match runtime producer
    # output. See PR #66 D2.5 precedent on model_deployer/state.py:50.
    test_metrics: Optional[MetricsSchema] = None  # FINAL test-set metrics

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

    # Phase 3.4 Layer-3 model-eval ablation hook (.claude/plans/
    # adaptive_temporal_validity_redesign.md line 245). All fields are
    # OPT-IN: ``model_trainer_layer3_ablation_enabled`` is the master gate
    # (default False); when False the other six tuning knobs are ignored.
    # Advisory mode mirrors Sec.4 T2.2 / T2.3 lifecycle pattern — emits
    # signals to ``validation_metrics`` but does NOT mutate
    # ``success_criteria_met`` or block the deployer.
    model_trainer_layer3_ablation_enabled: Optional[bool] = (
        None  # Default-OFF master gate; False = inert
    )
    model_trainer_ablation_n_permutations: Optional[int] = (
        None  # Column-shuffle ablation null perm count (default 30)
    )
    model_trainer_ablation_permutation_n_permutations: Optional[int] = (
        None  # Label-shuffle perm null count (default 200)
    )
    model_trainer_ablation_z_threshold: Optional[float] = (
        None  # HIGH-band z (default 5.0; mirrors Phase 3.3 HIGH_Z)
    )
    model_trainer_ablation_strong_effect_threshold: Optional[float] = (
        None  # |delta_AUC| strong-effect escape (default 0.30)
    )
    model_trainer_ablation_delta_auc_floor: Optional[float] = (
        None  # Issue #194 joint-check floor (default 0.10)
    )
    model_trainer_ablation_max_features: Optional[int] = (
        None  # O(n²) blowup guard cap (default 100)
    )
    model_trainer_ablation_seed: Optional[int] = None  # RNG seed (default 42)
    model_trainer_ablation_model_factory: Optional[Any] = (
        None  # Callable[[], sklearn-classifier]; None → LogisticRegression
    )
    model_eval_ablation: Optional[Dict[str, Any]] = (
        None  # Phase 3.4 result payload (ran flag, per_feature, flagged_features)
    )

    # Cross-Validation
    cv_results: Optional[Dict[str, Any]] = None  # Stratified k-fold metrics

    # Split Stratification
    split_validation: Optional[Dict[str, Any]] = None  # Class ratio drift across splits

    # Confidence Intervals
    confidence_interval: Optional[Dict[str, Tuple[float, ...]]] = None  # {'auc': (0.78, 0.85)}
    bootstrap_samples: Optional[int] = None

    # Success Criteria Check
    success_criteria_met: Optional[bool] = None
    # Values are Optional[bool]: the v3 adaptive evaluator records ``None`` for a
    # criterion that was skipped or whose metric was NaN (Option C audit
    # contract). A ``Dict[str, bool]`` rejected those, crashing the
    # LangGraph->Pydantic coercion with ValidationError (#617).
    success_criteria_results: Optional[Dict[str, Optional[bool]]] = (
        None  # Metric -> passed/failed/skipped
    )

    # PR #463 Phase 2 — post-training learning-curve diagnostic.
    # Populated by the ``learning_curve`` node when
    # ``success_criteria_met is False``; remains None when the model passed
    # (the diagnostic is a no-op). Shape matches
    # ``src.utils.sufficiency_schemas.DataSufficiencyReport`` (dict form so
    # it round-trips through LangGraph channels without pydantic-validation
    # cost on every node coercion).
    sufficiency_report: Optional[Dict[str, Any]] = None

    # PR #463 Phase 2 — forwarded from ScopeDefiner via the pipeline so the
    # ``learning_curve`` node can resolve target metrics
    # (``scope_spec.success_criteria.min_auc``) and sufficiency overrides
    # (``scope_spec.sufficiency.target_mde``) without round-tripping through
    # the pipeline result.
    scope_spec: Optional[Dict[str, Any]] = None

    # PR #463 Phase 2 — opt-in flag forcing the learning-curve diagnostic to
    # run even when ``success_criteria_met`` is True. Default None ≡ False so
    # the node short-circuits on pass as documented in its docstring.
    always_run_learning_curve: Optional[bool] = None

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
    # validator factory. Required: caller MUST provide ``audit_workflow_id``
    # (backlog #1 tightening landed 2026-05-09). All callers thread the
    # field explicitly per PRs #58 / #62 / #65.
    audit_workflow_id: UUID

    _validate_audit_id = audit_workflow_id_validator()
