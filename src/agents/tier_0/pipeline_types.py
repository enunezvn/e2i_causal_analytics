"""Data types of the ML Foundation pipeline: its stages, configuration and result.

Split out of ``pipeline.py`` (#2297), which had reached the module-size ratchet's limit.
These carry no behaviour; ``pipeline`` re-exports all three, so every existing import
path (``src.agents.tier_0.pipeline`` and ``src.agents.tier_0``) keeps working.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID


class PipelineStage(str, Enum):
    """Pipeline execution stages."""

    SCOPE_DEFINITION = "scope_definition"
    DATA_PREPARATION = "data_preparation"
    MODEL_SELECTION = "model_selection"
    MODEL_TRAINING = "model_training"
    FEATURE_ANALYSIS = "feature_analysis"
    MODEL_DEPLOYMENT = "model_deployment"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the ML Foundation Pipeline."""

    # Stage control
    skip_deployment: bool = False
    skip_feature_analysis: bool = False
    target_environment: str = "staging"

    # Training configuration
    enable_hpo: bool = True
    hpo_trials: int = 50
    hpo_timeout_hours: Optional[float] = None
    early_stopping: bool = False

    # PR #463 Phase 2 — opt-in flag forcing the post-training learning-curve
    # diagnostic to run EVEN WHEN the model passes ``success_criteria_met``.
    # Default False (cost control) — the diagnostic is a 7-bucket cheap proxy
    # fit + power-law extrapolation that runs only on failure cases by
    # default. Setting True flips the gate inside ``learning_curve`` so the
    # full report is produced unconditionally; useful for offline audits or
    # paper-replication runs.
    always_run_learning_curve: bool = False

    # Phase 3 — synthetic-data preview on insufficiency (opt-in; default off).
    # When the post-training learning curve recommends more data
    # (``recommended_additional_samples``), produce a PREVIEW synthetic cohort
    # sized to that recommendation so the operator can inspect it. The preview
    # is written to artifacts and surfaced on ``PipelineResult.synthetic_preview``
    # — it is NEVER auto-mixed into training. ``synthetic_preview_scenario`` is
    # the ``synthetic_v2`` scenario to generate (required for the preview to
    # fire; the operator chooses it — there is no auto-inference).
    synthetic_preview_on_insufficient: bool = False
    synthetic_preview_scenario: Optional[str] = None

    # Opt-in synthetic AUGMENTATION (Phase 3 consumption). When set to a
    # reviewed preview cohort (.npz, e.g. produced by synthetic_preview_*),
    # model_trainer concatenates those rows into the TRAINING split ONLY
    # (never validation/test/holdout), after split-ratio validation and before
    # preprocessing. Strict feature-schema match is enforced — a mismatch is
    # refused (advisory), never silently mixed. Default None = off.
    augmentation_data_path: Optional[str] = None

    # Model selection
    interpretability_required: bool = False
    skip_benchmarks: bool = True
    skip_mlflow: bool = False

    # Data preparation
    # Skips ONLY the legacy name-based detect_leakage node. The data-driven
    # adaptive validity / FDR layer always runs as the safety net and can still
    # escalate leakage findings regardless of this flag (#533, Option 2).
    skip_leakage_check: bool = False
    # Track-2B-v3: activate the deterministic structural causal-role decider
    # (Layer-4) for THIS run's cohort. Dark by default — the decider only
    # decides when this is True AND a feature carries a CausalStructureAttestation
    # (0 attested today). Cohort-scoped via this per-run config rather than a
    # global read-default at adaptive_validity_check.py, so activation is opt-in
    # per pipeline run, not a single un-scoped global flip.
    adaptive_structural_decider_enabled: bool = False
    use_sample_data: bool = False

    # Data-sufficiency pre-flight (Phase 1).
    # ``force_low_power_run`` downgrades blocking SOFT_FAIL verdicts to
    # warnings on the causal_inference path. Safe-by-default (False) to
    # avoid silently producing low-power effect estimates in pharma
    # regulatory contexts. ``sufficiency_strictness_preset`` toggles
    # ``conservative``/``moderate``/``strict`` multipliers on the
    # EPV/regression-ratio thresholds (see sufficiency_defaults.py).
    # When set, both flags propagate into ``scope_spec.sufficiency``
    # so the sufficiency_check node sees them via the resolver
    # hierarchy without a new state-injection path.
    force_low_power_run: bool = False
    sufficiency_strictness_preset: Optional[str] = None

    # Feast Feature Store
    enable_feast: bool = True
    feast_feature_refs: Optional[List[str]] = None  # Feature refs to use
    feast_freshness_check: bool = True  # Check feature freshness in QC gate
    feast_max_staleness_hours: float = 24.0  # Max allowed feature staleness
    feast_fallback_enabled: bool = True  # Fall back to custom store if Feast fails

    # Observability
    enable_observability: bool = True
    sample_rate: float = 1.0

    # Callbacks
    on_stage_complete: Optional[Callable[[PipelineStage, Dict[str, Any]], None]] = None
    on_error: Optional[Callable[[PipelineStage, Exception], None]] = None


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    pipeline_run_id: str
    status: str  # "completed", "failed", "partial"
    current_stage: PipelineStage
    experiment_id: Optional[str] = None

    # Audit chain integration
    audit_workflow_id: Optional[UUID] = None

    # Stage outputs
    scope_spec: Optional[Dict[str, Any]] = None
    success_criteria: Optional[Dict[str, Any]] = None
    qc_report: Optional[Dict[str, Any]] = None
    baseline_metrics: Optional[Dict[str, Any]] = None
    sufficiency_report: Optional[Dict[str, Any]] = None  # Phase 1 pre-flight verdict
    # Gate N1 (codex-rescue HIGH-3 / N1-H3): the regulatory_adaptation_entry
    # emitted by data_preparer's leakage_remediation node when it adaptively
    # dropped leaked features. Threaded into the deployer (nested under
    # scope_spec) so the deployer backstop can FAIL CLOSED on an un-ingested
    # adaptation. ``None`` when no leakage remediation occurred.
    regulatory_adaptation_entry: Optional[Any] = None
    model_candidate: Optional[Dict[str, Any]] = None
    # #2207: the target-bearing frames data_preparer produced (train / validation /
    # test / holdout), handed to the trainer when the caller pre-loaded no splits.
    prepared_frames: Optional[Dict[str, Any]] = None
    training_result: Optional[Dict[str, Any]] = None
    shap_analysis: Optional[Dict[str, Any]] = None
    deployment_result: Optional[Dict[str, Any]] = None

    # Feast feature store outputs
    feature_freshness: Optional[Dict[str, Any]] = None
    feature_refs_used: Optional[List[str]] = None
    feast_enabled: bool = False

    # PR #463 Phase 2 — post-training learning-curve diagnostic emitted by
    # ModelTrainer's ``learning_curve`` node. Shape:
    # ``src.utils.sufficiency_schemas.DataSufficiencyReport``-dict. None when
    # the model met success_criteria (the diagnostic is a no-op) AND the
    # caller didn't set ``PipelineConfig.always_run_learning_curve``.
    # Standalone field rather than a key on a unified ``sufficiency_report``
    # because PR #462 (pre-flight DataPreparer sufficiency check) has not
    # landed on main yet — when it does, this field will be merged into the
    # unified report dict downstream of the merge.
    training_sufficiency_report: Optional[Dict[str, Any]] = None

    # Phase 3 — synthetic-data preview metadata, populated only when
    # ``PipelineConfig.synthetic_preview_on_insufficient`` is set, a scenario
    # is specified, and the learning curve recommended more data. Contains the
    # preview cohort's artifact paths + audit metadata; the cohort itself is on
    # disk and is NOT mixed into training (``auto_mixed_into_training=False``).
    synthetic_preview: Optional[Dict[str, Any]] = None

    # Opt-in synthetic-augmentation audit (Phase 3 consumption). Populated from
    # the trainer's ``training_augmentation`` when augmentation_data_path was
    # set. ``applied`` distinguishes a real augmentation from a refusal (the
    # ``skip_reason`` says why); synthetic rows touch the training split only.
    training_augmentation: Optional[Dict[str, Any]] = None

    # Metadata
    stages_completed: List[str] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_duration_seconds: Optional[float] = None
