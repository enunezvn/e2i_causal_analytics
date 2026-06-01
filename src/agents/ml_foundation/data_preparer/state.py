"""State definition for data_preparer agent.

Migrated from ``TypedDict(total=False)`` to pydantic v2 ``BaseModel``
in Shard A of the migration tracked at
``.claude/plans/typeddict_to_pydantic_migration_plan_20260504.md``.

The state inherits from ``BaseAgentSchema`` which provides:

- ``extra="allow"`` for forward-compat during the multi-shard rollout.
- TypedDict-compat dict-like accessors (``__getitem__``, ``get``, etc.)
  so the existing ``state["key"]`` / ``state.get("key", default)`` call
  sites in ``data_preparer/nodes/`` (33 + 232 = 265 total) continue
  to work unchanged.
- ``audit_workflow_id_validator()`` factory for str↔UUID coercion at
  checkpoint-replay boundaries (Decision 7a).

Per Decision 8a, every field is ``Optional[T] = None`` except
``audit_workflow_id`` (required — it identifies the audit chain).

``qc_report`` is NOT a field on this state — it's the output the agent
constructs in ``data_preparer/agent.py::run`` after the graph runs,
NOT a piece of pipeline state. ``QCReportSchema`` exists in
``data_preparer/schemas.py`` for callers that want to typed-validate
the agent output dict; wiring is left to a follow-up sub-shard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
)
from src.agents.ml_foundation.scope_definer.schemas import ScopeSpecSchema


class DataPreparerState(BaseAgentSchema):
    """State for data_preparer agent.

    The data_preparer validates data quality, computes baseline metrics,
    and enforces a QC gate that blocks downstream training if quality fails.
    """

    # === INPUT FIELDS ===
    # From scope_definer
    experiment_id: Optional[str] = None
    # D2.4: typed scope_spec contract. Schema declares 23 producer fields
    # + 24 caller-injected consumer keys read across data_preparer nodes
    # (date_column, required_columns, data_source, table_name, etc.).
    scope_spec: Optional[ScopeSpecSchema] = None  # ScopeSpec from scope_definer

    # Data source configuration
    # May be either:
    #   - str: Supabase table/view name (default path)
    #   - dict: file ingestion descriptor. Two shapes:
    #       {"type": "file_dir", "path": "<dir>"} — read canonical
    #         e2i_ml_v3_* files from a directory
    #       {"type": "files", "paths": {"patient_journeys": "<path>", ...}}
    data_source: Optional[Union[str, Dict[str, Any]]] = None
    split_id: Optional[str] = None  # ML split ID (if using existing split)

    # Validation configuration
    validation_suite: Optional[str] = None  # Great Expectations suite name
    # Skips ONLY the legacy name-based detect_leakage node. The data-driven
    # adaptive validity / FDR layer (adaptive_validity_check) ALWAYS runs as the
    # safety net and can still escalate leakage_severity (#533, Option 2).
    skip_leakage_check: Optional[bool] = None  # (NOT RECOMMENDED)

    # === INTERMEDIATE FIELDS ===
    # Data loading
    train_df: Optional[Any] = None  # pandas DataFrame (train split)
    validation_df: Optional[Any] = None  # pandas DataFrame (validation split)
    test_df: Optional[Any] = None  # pandas DataFrame (test split)
    holdout_df: Optional[Any] = None  # pandas DataFrame (holdout split)

    # Sampling-frame audit (advisory; emitted by audit_sampling_frame node)
    # Compares train_df distribution to scope_spec["deployment_reference"].
    # Failures are non-blocking: status surfaces in the report, never in
    # blocking_issues. See nodes/sampling_frame_audit.py for the report
    # schema.
    sampling_frame_audit_report: Optional[Dict[str, Any]] = None

    # Schema validation (Pandera)
    schema_validation_status: Optional[Literal["passed", "failed", "skipped", "error"]] = None
    schema_validation_errors: Optional[List[Dict[str, Any]]] = None
    schema_splits_validated: Optional[int] = None
    schema_validation_time_ms: Optional[int] = None

    # Quality checks
    expectation_results: Optional[List[Dict[str, Any]]] = None  # Great Expectations results
    failed_expectations: Optional[List[str]] = None
    warnings: Optional[List[Dict[str, Any]]] = None

    # Dimension scores
    completeness_score: Optional[float] = None
    validity_score: Optional[float] = None
    consistency_score: Optional[float] = None
    uniqueness_score: Optional[float] = None
    timeliness_score: Optional[float] = None
    overall_score: Optional[float] = None

    # Leakage detection
    leakage_detected: Optional[bool] = None
    leakage_issues: Optional[List[str]] = None
    leakage_findings: Optional[List[Dict[str, Any]]] = None  # Structured LeakageFinding dicts
    leakage_severity: Optional[str] = None  # "critical" / "high" / "moderate" / "info" / "none"
    leaked_features: Optional[List[str]] = None  # Feature names flagged at CRITICAL or HIGH

    # Adaptive validity check (Layer 3 + Layer 4 audit trail). Augments
    # leakage_findings/leaked_features with permutation-baseline-derived verdicts.
    adaptive_verdicts: Optional[List[Dict[str, Any]]] = None  # one record per scored feature
    adaptive_flagged_features: Optional[List[str]] = None  # features at z > 5σ above null
    adaptive_n_permutations: Optional[int] = None  # override default permutation count
    adaptive_seed: Optional[int] = None  # override default RNG seed
    # Plan v4 Layer B / Phase 2 — dark-launch flag for the deterministic
    # structural role decider. DEFAULT None (read as False) → the structural
    # decision path is NEVER taken in production until an operator opts in.
    # Declared (not relied on via ``extra``) because BaseAgentSchema is
    # ``extra="ignore"``: an undeclared key set on the state MODEL would be
    # silently dropped, so a future Phase-3 ramp could not enable it. See
    # .claude/plans/layer4-phase2-layerB-20260528.md §2.5.
    adaptive_structural_decider_enabled: Optional[bool] = None
    # #594: per-run switch for the Layer-3 FDR confident-set firing driver (#538).
    # Default True = FDR ON (matches the node default at adaptive_validity_check
    # ``state.get("adaptive_fdr_enabled", True)``; validated on real cohorts). The
    # tier0 synthetic-test runner sets this False for synthetic FIXTURE regimes
    # (scenario + legacy clean/adverse/default), whose deliberately
    # outcome-correlated features the FDR driver false-positively auto-drops —
    # falling back to the static σ-band so genuine leaks (e.g. journey_status) are
    # still caught WITHOUT over-dropping legitimate signal. Declared (not relied on
    # via ``extra``) because BaseAgentSchema is ``extra="ignore"``. ``bool`` (not
    # Optional) so the model default is True, never None → no silent FDR-off.
    adaptive_fdr_enabled: bool = True

    # Phase 1 of causal-role propagation (Issue #237 reframe). Typed
    # list of ``RoleAttribution`` rows (see
    # ``src.data.role_attribution.RoleAttribution`` TypedDict): one row
    # per feature whose causal role has been attested by either the
    # manifest (``source="manifest"``) or the Layer-4 LLM classifier
    # (``source="llm"``). Computed in ``finalize_output`` from
    # ``adaptive_verdicts`` and the resolved manifest's feature
    # contracts. Persisted to the sidecar JSON as ``role_attributions``
    # under ``schema_version="1.1"``. The field is audit-only in
    # Phase 1; Phase 2 (collider/mediator exclusion policy) will be the
    # first consumer.
    role_attributions: Optional[List[Dict[str, Any]]] = None

    # Leakage remediation (LLM-assisted)
    leakage_remediation_status: Optional[
        Literal[
            "not_needed",
            "applied",
            "failed",
            "manual_required",
            "error",
            "max_attempts_reached",
        ]
    ] = None
    leakage_remediation_attempts: Optional[int] = None
    leakage_remediated_features: Optional[List[str]] = None  # Clean features after remediation
    leakage_dropped_features: Optional[List[str]] = None  # Features removed due to leakage
    leakage_added_features: Optional[List[str]] = None  # Alternative features added
    leakage_remediation_reasoning: Optional[str] = None  # LLM reasoning summary
    leakage_remediation_viable: Optional[bool] = None  # Whether a viable feature set was found
    requires_leakage_revalidation: Optional[bool] = None  # Trigger re-check loop

    # Gate N1 (plan v4 §2 codex-rescue HIGH-3): one adaptation entry per
    # remediation pass. Downstream the orchestrator aggregates these into
    # ``validation_metrics["regulatory_eligibility_audit"]["adaptation_history"]``
    # which the model_deployer reads to set ``regulatory_eligible``. The
    # field is per-invocation; cumulative aggregation is the orchestrator's
    # job (see model_deployer/nodes/registry_manager.py).
    regulatory_adaptation_entry: Optional[Dict[str, Any]] = None

    # v5 Gate B3: feature engineering on the clean pre-anchor surface.
    # ``enable_feature_engineering`` gates the engineer_features_node; when
    # False (default), the node is a no-op and existing production runs are
    # unaffected. ``engineered_features`` is the canonical (train-split)
    # list of engineered feature names that were materialized; downstream
    # nodes (adaptive_validity_check, leakage_remediation) audit these
    # alongside base features. ``engineered_dispatch_source`` echoes the
    # manifest_source used to dispatch the engineering family — useful for
    # audit-trail reconstruction when a run later asks "which transforms
    # ran?". Pre-spec: docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md.
    # L2 (codex): bool=False (not Optional[bool]=None) so the documented
    # "default False" matches the declared default. Callers that did
    # ``if state["enable_feature_engineering"]:`` previously got None
    # which evaluates to False but is type-confusing; bool=False is
    # type-honest.
    enable_feature_engineering: bool = False
    engineered_features: Optional[List[str]] = None
    engineered_dispatch_source: Optional[str] = None

    # Baseline computation
    feature_stats: Optional[Dict[str, Dict[str, Any]]] = None  # Per-feature statistics
    target_rate: Optional[float] = None  # For classification
    target_distribution: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None

    # Feast registration
    feast_registration_status: Optional[
        Literal["completed", "empty", "skipped", "error", "blocked_stale_features"]
    ] = None
    feast_features_registered: Optional[int] = None  # Count of features registered
    feast_freshness_check: Optional[Dict[str, Any]] = None  # Freshness check result
    feast_warnings: Optional[List[str]] = None  # Non-blocking warnings
    feast_registered_at: Optional[str] = None  # ISO timestamp
    feast_blocked: Optional[bool] = None  # True when stale features hard-block training
    feast_fallback_used: Optional[bool] = None  # True when the historical-features fallback fired

    # Recommendations
    remediation_steps: Optional[List[str]] = None
    blocking_issues: Optional[List[str]] = None  # If non-empty, blocks training

    # Sufficiency pre-flight (Phase 1 of data-sufficiency diagnostics).
    # ``sufficiency_report`` is the model_dump of a DataSufficiencyReport
    # (src/utils/sufficiency_schemas.py): verdict + resolved thresholds +
    # detectable MDE + sensitivity grid. ``power_warnings`` carries
    # non-blocking SOFT_FAIL messages for the operator (predictive paths).
    # The HARD_FAIL / blocking SOFT_FAIL path appends to ``blocking_issues``
    # so the existing QC gate at ``finalize_output`` picks it up.
    sufficiency_report: Optional[Dict[str, Any]] = None
    power_warnings: Optional[List[str]] = None

    # === OUTPUT FIELDS ===
    # QC Report
    report_id: Optional[str] = None
    qc_status: Optional[Literal["passed", "failed", "warning", "skipped"]] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    validated_at: Optional[str] = None  # ISO timestamp

    # Data Readiness
    total_samples: Optional[int] = None
    train_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None
    holdout_samples: Optional[int] = None
    available_features: Optional[List[str]] = None
    missing_required_features: Optional[List[str]] = None
    is_ready: Optional[bool] = None
    qc_passed: Optional[bool] = None
    qc_score: Optional[float] = None
    blockers: Optional[List[str]] = None

    # Gate decision
    gate_passed: Optional[bool] = None  # CRITICAL: blocks model_trainer if False

    # Metadata
    validation_duration_seconds: Optional[float] = None
    computed_at: Optional[str] = None  # ISO timestamp
    training_samples: Optional[int] = None  # For baseline metrics

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None

    # QC Remediation Loop
    remediation_status: Optional[
        Literal["not_needed", "applied", "failed", "manual_required", "exhausted", "error"]
    ] = None
    remediation_attempts: Optional[int] = None  # Count of remediation attempts
    remediation_actions_taken: Optional[List[str]] = None  # Actions applied during remediation
    remediation_error: Optional[str] = None  # Error message if remediation failed
    requires_revalidation: Optional[bool] = None  # Whether to re-run validation after remediation
    llm_analysis: Optional[str] = None  # LLM-generated root cause summary
    root_causes: Optional[List[str]] = None  # Identified root causes
    recommended_actions: Optional[List[str]] = None  # Recommended manual actions
    estimated_effort: Optional[str] = None  # Low/Medium/High effort estimate
    blocking_issues_analysis: Optional[List[Dict[str, Any]]] = None  # Detailed blocking analysis
    failure_summary: Optional[str] = None  # Summary when remediation exhausted

    # Audit chain — Decision 7a: typed as UUID (never None), with str↔UUID
    # coercion for checkpoint-replay JSON-restore compat via the validator
    # factory. Decision 8a explicit override: NOT Optional[UUID]=None.
    #
    # Required: caller MUST provide ``audit_workflow_id`` (backlog #1
    # tightening landed 2026-05-09). The previous ``default_factory=uuid4``
    # was a transition mechanism for dict-literal construction without
    # threading the field; all callers (orchestrator + per-agent + caller
    # fixtures) now thread it explicitly per PRs #58 / #62 / #65.
    audit_workflow_id: UUID

    _validate_audit_id = audit_workflow_id_validator()
