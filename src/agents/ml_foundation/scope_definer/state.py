"""State definition for scope_definer agent.

Migrated from ``TypedDict(total=False)`` to pydantic v2 ``BaseModel``
in Shard A of the migration tracked at
``.claude/plans/typeddict_to_pydantic_migration_plan_20260504.md``.

The state inherits from ``BaseAgentSchema`` which provides:

- ``extra="allow"`` for forward-compat during the multi-shard rollout.
- TypedDict-compat dict-like accessors (``__getitem__``, ``get``, etc.)
  so the existing ``state["key"]`` / ``state.get("key", default)`` call
  sites in ``scope_definer/nodes/`` and ``memory_hooks.py`` continue
  to work unchanged.
- ``audit_workflow_id_validator()`` factory for str↔UUID coercion at
  checkpoint-replay boundaries (Decision 7a).

Per Decision 8a, every field is ``Optional[T] = None`` except
``audit_workflow_id`` (required — it identifies the audit chain).

``scope_spec`` and ``success_criteria`` retain ``Optional[Dict[str, Any]]``
typing for now — ``ScopeSpecSchema``/``SuccessCriteriaSchema`` are
available in ``scope_definer/schemas.py`` for callers that want to
opt in, and a follow-up sub-shard will tighten these once cross-agent
consumer migration is complete.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
)


class ScopeDefinerState(BaseAgentSchema):
    """State for scope_definer agent.

    The scope_definer transforms business requirements into formal ML
    problem specifications with success criteria.
    """

    # === INPUT FIELDS ===
    # Business request
    problem_description: Optional[str] = None
    business_objective: Optional[str] = None
    target_outcome: Optional[str] = None

    # Problem type hint (optional)
    problem_type_hint: Optional[
        Literal[
            "binary_classification",
            "multiclass_classification",
            "regression",
            "causal_inference",
            "time_series",
        ]
    ] = None

    # Target variable (optional)
    target_variable: Optional[str] = None

    # Features (optional)
    candidate_features: Optional[List[str]] = None

    # Constraints (optional)
    time_budget_hours: Optional[float] = None
    performance_requirements: Optional[Dict[str, float]] = None

    # Context
    brand: Optional[str] = None
    region: Optional[str] = None
    use_case: Optional[str] = None

    # Layer 5 manifest opt-in. The pipeline resolves which cohort manifest
    # (csu/optum/synthetic) applies — from data_source or an explicit override
    # via src.data.manifests.resolution.resolve_manifest_source — and threads
    # it here so scope_builder copies it onto scope_spec. Unset → no manifest
    # (cross-cohort false-positive guard). See scope_spec["feature_manifest_source"].
    feature_manifest_source: Optional[str] = None

    # Temporal anchoring (Block 1B scaffolding; consumed in Block 4+).
    # Inference cutoff time the model predicts from — feeds scope_spec so
    # downstream agents can clip lookback windows and post-prediction filters.
    # Accepts datetime, str, or pd.Timestamp at the API boundary; the scope
    # builder normalises to ISO 8601 string for storage in scope_spec.
    #
    # Distinguish from ``prediction_horizon_days`` (read from state via
    # ``state.get("prediction_horizon_days", 30)`` in scope_builder): the
    # horizon is a *duration* in days, while this field is the *anchor*
    # timestamp. Together they define the prediction window.
    prediction_timestamp: Optional[Any] = None

    # Cost matrix for business-utility-driven evaluation (Block 5 — finding
    # #10). Keys ``tp``/``fp``/``fn``/``tn`` map confusion-matrix outcomes to
    # their per-prediction monetary value. Optional: when absent, the
    # evaluator skips ``business_utility``. Forwarded verbatim onto
    # ``scope_spec["cost_matrix"]`` so the model_trainer can read it without
    # threading through a separate parameter.
    cost_matrix: Optional[Dict[str, float]] = None

    # Adaptive success criteria pre-eval inputs (task 05 of
    # adaptive_success_criteria plan). When ``ADAPTIVE_CRITERIA=true`` and
    # all four are present, ``criteria_validator`` stashes them on
    # ``success_criteria['_adaptive_inputs']`` for the evaluator overlay
    # to pick up alongside the live ``baseline_test_auc``. Optional —
    # when any is missing, the validator falls back to fixed thresholds
    # with ``criteria_source="adaptive_fallback_to_fixed"``.
    # NOTE: ``baseline_auc`` is intentionally NOT here — it is computed at
    # eval time inside the evaluator, not at scope-definition time.
    n_samples: Optional[int] = None  # training-split row count
    prevalence: Optional[float] = None  # positive-class rate, in [0, 1]
    feature_count: Optional[int] = None  # post-preprocessing feature count
    regime: Optional[Literal["default", "clean", "adverse"]] = None

    # === INTERMEDIATE FIELDS ===
    # Problem classification
    inferred_problem_type: Optional[str] = None
    inferred_target_variable: Optional[str] = None

    # Feature requirements
    required_features: Optional[List[str]] = None
    excluded_features: Optional[List[str]] = None
    feature_categories: Optional[List[str]] = None

    # Population criteria
    target_population: Optional[str] = None
    inclusion_criteria: Optional[List[str]] = None
    exclusion_criteria: Optional[List[str]] = None
    minimum_samples: Optional[int] = None

    # Constraints
    regulatory_constraints: Optional[List[str]] = None
    ethical_constraints: Optional[List[str]] = None
    technical_constraints: Optional[List[str]] = None

    # Success criteria (per-problem-type subsets)
    minimum_auc: Optional[float] = None
    minimum_precision: Optional[float] = None
    minimum_recall: Optional[float] = None
    minimum_f1: Optional[float] = None
    minimum_rmse: Optional[float] = None
    minimum_r2: Optional[float] = None
    baseline_model: Optional[str] = None
    minimum_lift_over_baseline: Optional[float] = None

    # Validation
    validation_passed: Optional[bool] = None
    validation_warnings: Optional[List[str]] = None
    validation_errors: Optional[List[str]] = None

    # === OUTPUT FIELDS ===
    # Experiment identification
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None

    # ScopeSpec (complete specification). Typed as Dict[str, Any] for now;
    # ``scope_definer/schemas.py::ScopeSpecSchema`` is available for
    # callers that want pydantic-validated access. Tightening to
    # ``Optional[ScopeSpecSchema]`` is deferred to a follow-up sub-shard
    # once cross-agent consumer migration is complete.
    scope_spec: Optional[Dict[str, Any]] = None

    # SuccessCriteria (complete criteria). Same deferral as scope_spec.
    success_criteria: Optional[Dict[str, Any]] = None

    # Metadata
    created_at: Optional[str] = None
    created_by: Optional[str] = None

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Audit chain — Decision 7a: typed as UUID (never None), with str↔UUID
    # coercion for checkpoint-replay JSON-restore compat via the validator
    # factory. Decision 8a explicit override: NOT Optional[UUID]=None.
    #
    # Required: caller MUST provide ``audit_workflow_id`` (backlog #1
    # tightening landed 2026-05-09). The previous ``default_factory=uuid4``
    # was a transition mechanism — agent flows now thread the field
    # explicitly through the orchestrator (PR #58), per-agent input_data
    # (PR #62), and caller fixtures (PR #65). Pydantic enforces presence
    # at construction; missing the field raises ValidationError.
    audit_workflow_id: UUID

    _validate_audit_id = audit_workflow_id_validator()
