"""Pydantic v2 schemas for ``scope_definer`` agent outputs.

These schemas formalise the two ``Dict[str, Any]`` payloads that the
agent currently emits via ``state.py``:

- ``scope_spec`` — the ML problem specification consumed by every
  downstream agent. Fields enumerated from
  ``scope_definer/nodes/scope_builder.py::build_scope_spec`` (lines
  160-183) plus the optional sampling-frame fields read by
  ``data_preparer/nodes/sampling_frame_audit.py`` (lines 78-106).

- ``success_criteria`` — the per-problem-type acceptance thresholds
  consumed by ``model_trainer.check_qc_gate`` and
  ``model_deployer.check_promotion_criteria``. Fields enumerated from
  ``scope_definer/nodes/criteria_validator.py``.

This module is part of the chore(schemas) scaffolding PR. It does NOT
modify ``state.py`` — Shard A wires these schemas into
``ScopeDefinerState`` after this PR merges.

Per Decision 8a, every field is ``Optional[T] = None`` unless invalid
(e.g. ``baseline_model`` defaults to a non-empty string because every
problem type has one).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import ConfigDict

from src.agents.ml_foundation._pydantic_utils import BaseAgentSchema
from src.utils.sufficiency_schemas import SufficiencyConfig


class ScopeSpecSchema(BaseAgentSchema):
    """Complete ML experiment specification produced by scope_definer.

    Round-trips through JSON via ``model_dump_json`` and
    ``model_validate_json``. ``extra="allow"`` (inherited from
    ``BaseAgentSchema``) lets unknown keys pass through during the
    migration's transition window — particularly useful for fields
    added by future agents that consume scope_spec without round-trip
    schema discipline yet.
    """

    # Identification
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None

    # Problem definition
    problem_type: Optional[
        Literal[
            "binary_classification",
            "multiclass_classification",
            "regression",
            "causal_inference",
            "time_series",
        ]
    ] = None
    prediction_target: Optional[str] = None
    prediction_horizon_days: Optional[int] = None
    prediction_timestamp: Optional[str] = None  # ISO 8601 string per scope_builder

    # Business utility (Block 5 finding #10)
    cost_matrix: Optional[Dict[str, float]] = None  # tp/fp/fn/tn → dollar value

    # Population
    target_population: Optional[str] = None
    inclusion_criteria: Optional[List[str]] = None
    exclusion_criteria: Optional[List[str]] = None

    # Features
    required_features: Optional[List[str]] = None
    excluded_features: Optional[List[str]] = None
    feature_categories: Optional[List[str]] = None

    # Constraints
    regulatory_constraints: Optional[List[str]] = None
    ethical_constraints: Optional[List[str]] = None
    technical_constraints: Optional[List[str]] = None
    minimum_samples: Optional[int] = None

    # Context
    brand: Optional[str] = None
    region: Optional[str] = None
    use_case: Optional[str] = None

    # Metadata
    created_by: Optional[str] = None
    created_at: Optional[str] = None  # ISO 8601 string

    # Sampling-frame audit overrides (read by data_preparer/nodes/sampling_frame_audit.py)
    deployment_reference: Optional[Dict[str, Any]] = None
    sampling_frame_audit: Optional[Dict[str, Any]] = None  # per-metric threshold overrides
    sampling_frame_max_drift: Optional[float] = None

    # D2.4: 24 caller-injected consumer-side keys read by data_preparer
    # nodes (and a few by model_selector). Pre-D2.4 these flowed through
    # ``model_extra`` because the producer at ``scope_builder.py`` does
    # NOT emit them — they come from the runner / orchestrator merging
    # them into scope_spec before the data_preparer agent runs. Each
    # consumer reads via ``scope_spec.get("key", default)``.

    # Schema validation (data_preparer/nodes/schema_validator.py:64-70)
    data_source: Optional[str] = None
    table_name: Optional[str] = None

    # Layer 5 manifest opt-in (adaptive_validity_check.py).
    # Names which feature manifest the pipeline should consult (e.g. "csu",
    # "optum"). Unset → Layer 1 manifest pass is skipped, so synthetic /
    # research regimes don't get cross-cohort false positives when their
    # column names happen to overlap with a registered manifest's vocabulary.
    feature_manifest_source: Optional[str] = None

    # Phase 2.9 Stage 2 KG entity-mapping (Stage 2 PR-A scaffold; Stage 2 PR-D
    # wires the consumer in adaptive_validity_check.py). Cohort runner sets
    # these per cohort.
    #
    # ``target_entity_codes``: list of ``(CodeSystem, code)`` tuples
    # representing the prediction target's KG entities. Examples:
    #   - CSU bio_initiation target → [("RXNORM", "479158"),
    #     ("RXNORM", "1011295"), ...] (omalizumab + dupilumab + future biologics)
    #   - Optum bio_initiation target → similar RxCUIs
    #   - A future Dupixent-specific brand-prediction target →
    #     [("RXNORM", "1011295")] (dupilumab only — narrower target)
    # Empty list (or unset) means "no KG-mappable target representation"
    # — typical for synthetic regimes. The voter's classify_kg_signal uses
    # these IDs to filter which kg_edges are relevant.
    #
    # ``kg_cache_path``: path to the offline KG cache file built by
    # ``scripts/build_kg_cache.py`` (PR-C). Pipeline reads the cache at
    # data_preparer node entry; cache miss policy depends on ``kg_mode``
    # (PR-E adds the mode field).
    target_entity_codes: Optional[List[Tuple[str, str]]] = None
    kg_cache_path: Optional[str] = None

    # Phase 2.9 Stage 2 PR-E: shadow-mode promotion gate.
    #
    # ``kg_mode`` controls how Layer 2 KG verdicts influence the final
    # decision:
    #   - ``"off"`` (default): Stage 1 behavior. KG cache is not loaded;
    #     verdicts never carry a KG signal. Backward-compatible with
    #     pre-Stage-2 cohorts.
    #   - ``"shadow"``: KG cache IS loaded and ``decided_by="kg"`` /
    #     ``kg_signal`` are recorded for audit, but the verdict severity
    #     is capped to ``"info"`` so KG cannot drop a feature. Used to
    #     observe KG signal quality on a cohort before promotion.
    #   - ``"promoted"``: KG signal participates in voter precedence
    #     normally; KG can drive ``severity="high"`` and drop features.
    #
    # Promotion is operator-driven (see ``compute_promotion_eligibility``)
    # — there is no auto-promote.
    kg_mode: Optional[Literal["off", "shadow", "promoted"]] = None

    # Quality checker (data_preparer/nodes/quality_checker.py:54-59)
    date_column: Optional[str] = None
    required_columns: Optional[List[str]] = None
    expected_dtypes: Optional[Dict[str, str]] = None
    unique_columns: Optional[List[str]] = None
    max_staleness_days: Optional[float] = None

    # Data transformer (data_preparer/nodes/data_transformer.py:62-80)
    target_column: Optional[str] = None
    exclude_columns: Optional[List[str]] = None  # deprecated; use excluded_features
    scaling_method: Optional[str] = None
    # encoding_method: "onehot" (default, nominal categoricals) | "label"
    # (legacy integer encoding for ALL categoricals). #790: nominal categoricals
    # one-hot by default so the linear champion is not handed false-ordinal codes.
    encoding_method: Optional[str] = None
    # ordinal_features: categoricals that ARE genuinely ordered (e.g. risk
    # bands, stage I-IV) and should stay integer-encoded (order preserved) even
    # under the one-hot default. Ignored when encoding_method="label" (all
    # categoricals are integer-encoded then anyway). #790.
    ordinal_features: Optional[List[str]] = None
    imputation_strategy: Optional[str] = None
    extract_datetime_features: Optional[bool] = None

    # Data loader (data_preparer/nodes/data_loader.py:50-82)
    filters: Optional[Dict[str, Any]] = None
    entity_column: Optional[str] = None
    split_date: Optional[str] = None  # ISO 8601
    val_days: Optional[int] = None
    test_days: Optional[int] = None
    use_sample_data: Optional[bool] = None
    sample_size: Optional[int] = None

    # Leakage detector (data_preparer/nodes/leakage_detector.py:110-369)
    event_date_column: Optional[str] = None
    target_date_column: Optional[str] = None
    feature_date_columns: Optional[List[str]] = None

    # Feast registrar (data_preparer/nodes/feast_registrar.py:90-93)
    entity_key: Optional[str] = None

    # F1 (PR #462 hotfix): data-sufficiency overrides nested under
    # `scope_spec.sufficiency`. Read by `data_preparer/nodes/sufficiency_check`
    # and written by `pipeline.py` + `scope_builder`. Before this field was
    # declared, `BaseAgentSchema`'s inherited `extra="ignore"` silently dropped
    # the entire `sufficiency` payload at pydantic coercion time — every
    # user-supplied `target_mde` / `force_low_power_run` / `epv_floor` override
    # vanished without a trace whenever any code path ran a typed scope_spec
    # through pydantic validation. Declaring the field as `SufficiencyConfig`
    # routes the dict through the typed model (which carries field-level
    # validation: `target_mde` must be in (0, 1), `epv_floor >= 1`, etc.).
    sufficiency: Optional[SufficiencyConfig] = None


class SuccessCriteriaSchema(BaseAgentSchema):
    # D3 (2026-05-05): override BaseAgentSchema's ``extra="ignore"`` with
    # ``extra="allow"`` because ``criteria_validator.py:305-326`` writes
    # underscore-prefixed audit keys (``_adaptive_skipped``, ``_adaptive_p_t``,
    # ``_adaptive_inputs``) that pydantic v2 cannot declare as model fields
    # (reserved namespace). The consumer at ``model_trainer/nodes/evaluator.py``
    # reads them via ``success_criteria.get("_adaptive_inputs", default)`` —
    # under base ``extra="ignore"`` those keys would be dropped at
    # construction and the consumer reads return None. Per-class ``allow``
    # override preserves the underscore-key flow.
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        validate_assignment=True,
    )

    """Acceptance thresholds + adaptive overlay for a scope.

    All metric thresholds are ``Optional[float] = None`` because the
    relevant subset depends on ``problem_type`` (e.g. classification
    fills auc/f1; regression fills rmse/r2). The adaptive overlay
    (Phase 5 work) attaches three private fields under
    ``_adaptive_*`` that the criteria_validator populates when
    ``ADAPTIVE_CRITERIA=true``.

    Underscore-prefixed adaptive keys (per codex review M1, 2026-05-05):
    ``criteria_validator.py:305-326`` writes three keys into the produced
    dict that this schema CANNOT declare as pydantic fields because
    pydantic v2 reserves underscore-prefixed names for private attributes:

    - ``_adaptive_skipped: List[str]`` — names of adaptive criteria that
      fell back to fixed thresholds because their inputs were missing.
    - ``_adaptive_p_t: Dict[str, float]`` — per-regime adaptive
      probability thresholds (e.g. ``{"clean": 0.5, "default": 0.6}``).
    - ``_adaptive_inputs: Dict[str, Any]`` — the four pre-eval inputs
      (``n_samples``, ``prevalence``, ``feature_count``, ``regime``) that
      the evaluator overlay uses alongside ``baseline_test_auc``.

    When this schema is wired (sub-shard D2) and replaces the inline
    ``Dict[str, Any]`` annotation on ``ScopeDefinerState.success_criteria``,
    these three keys flow through to ``model_extra`` (via inherited
    ``extra="allow"``) and stay accessible via:

    .. code-block:: python

        criteria.model_extra["_adaptive_inputs"]
        # or via the BaseAgentSchema dict-like shim:
        criteria["_adaptive_inputs"]
    """

    # Identification
    experiment_id: Optional[str] = None

    # Baseline
    baseline_model: Optional[str] = None
    minimum_lift_over_baseline: Optional[float] = None

    # Classification thresholds
    minimum_auc: Optional[float] = None
    minimum_precision: Optional[float] = None
    minimum_recall: Optional[float] = None
    minimum_f1: Optional[float] = None

    # Regression thresholds
    minimum_rmse: Optional[float] = None
    minimum_r2: Optional[float] = None
    # D2.3: minimum_mape — emitted by ALL 4 problem-type branches in
    # ``criteria_validator.py`` (classification fills it as None; regression
    # fills it as a real value). Pre-D2.3 it flowed through ``model_extra``;
    # now it's a declared field so consumer reads catch typos.
    minimum_mape: Optional[float] = None

    # D2.3: v3 adaptive active gates (6 fields). Emitted by
    # ``adaptive_success_criteria()`` at criteria_validator.py:118-173 when
    # ``ADAPTIVE_CRITERIA=true``. The model_trainer's evaluator iterates over
    # criteria.items() and reads each key by name — pre-D2.3 these keys were
    # undeclared and only worked because the iteration was dict-driven, not
    # field-access-driven. Wiring them as declared fields catches drift.
    minimum_net_benefit_at_p_t: Optional[float] = None
    minimum_mcc: Optional[float] = None
    maximum_calibration_slope_deviation: Optional[float] = None
    maximum_calibration_intercept_magnitude: Optional[float] = None
    maximum_calibration_error: Optional[float] = None
    maximum_train_val_delta: Optional[float] = None

    # D2.3: caller-injected consumer keys read by
    # ``model_trainer/nodes/evaluator.py`` at lines 887, 944, 898, 954. These
    # are NOT emitted by scope_definer's criteria_validator — they flow
    # through scope_definer's caller (e.g. scripts/run_tier0_test.py) merging
    # them into success_criteria before model_trainer reads them.
    clinical_threshold_range: Optional[Dict[str, Any]] = None
    dataset_disease: Optional[str] = None

    # Adaptive overlay (Phase 5; populated by criteria_validator)
    # Aliased with leading underscore in the source dict; pydantic field names
    # cannot start with underscore so we use Field(alias=...) via populate_by_name.
    criteria_source: Optional[
        Literal[
            "fixed",
            "adaptive",
            "adaptive_fallback_to_fixed",
        ]
    ] = None
