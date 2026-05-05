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

from typing import Any, Dict, List, Literal, Optional

from src.agents.ml_foundation._pydantic_utils import BaseAgentSchema


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


class SuccessCriteriaSchema(BaseAgentSchema):
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
