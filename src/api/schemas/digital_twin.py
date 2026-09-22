"""Public response contracts shared by the Digital Twin routes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SubgroupAxisProvenanceResponse(BaseModel):
    """Evidence source, publication support, and scoring fallback for an axis."""

    basis: str
    source: str
    min_group_rows: int
    min_treated_rows: Optional[int] = None
    min_control_rows: Optional[int] = None
    fallback: str
    support_unit: str
    estimand: str
    suppressed_groups: Dict[str, str] = Field(default_factory=dict)


class EffectHeterogeneityResponse(BaseModel):
    """Heterogeneous effects across subgroups."""

    by_specialty: Dict[str, Dict[str, float]]
    by_decile: Dict[str, Dict[str, float]]
    by_region: Dict[str, Dict[str, float]]
    by_adoption_stage: Dict[str, Dict[str, float]]
    top_segments: List[Dict[str, Any]]
    axis_provenance: Dict[str, SubgroupAxisProvenanceResponse] = Field(
        description=(
            "Per-axis evidence contract: source, cohort/per-twin basis, publication support "
            "floors, fallback rule, and groups suppressed for insufficient support."
        ),
    )


def heterogeneity_response(value: Any) -> EffectHeterogeneityResponse:
    """Map a live model or stored JSON to the public heterogeneity contract."""
    data: Mapping[str, Any]
    if isinstance(value, BaseModel):
        data = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        data = value
    else:
        raise TypeError(
            "effect heterogeneity must be a Pydantic domain model or stored JSON mapping"
        )
    return EffectHeterogeneityResponse(
        by_specialty=data.get("by_specialty", {}),
        by_decile=data.get("by_decile", {}),
        by_region=data.get("by_region", {}),
        by_adoption_stage=data.get("by_adoption_stage", {}),
        top_segments=(data.get("top_segments") or [])[:5],
        axis_provenance=data.get("axis_provenance", {}),
    )


def live_subgroups_basis(
    data_provenance: Optional[str], *, calculated: bool = True
) -> Literal["cohort_rows", "per_twin", "twin_weighted_legacy", "unknown"]:
    """Describe the evidence basis only when subgroup effects were calculated."""
    from src.digital_twin.effect.estimate import (
        PROVENANCE_COHORT,
        PROVENANCE_RWD,
        PROVENANCE_SYNTHETIC,
    )

    if not calculated:
        return "unknown"
    if data_provenance == PROVENANCE_COHORT:
        return "cohort_rows"
    if data_provenance in {PROVENANCE_SYNTHETIC, PROVENANCE_RWD}:
        return "per_twin"
    return "unknown"


class DigitalTwinHealthResponse(BaseModel):
    """Health status for Digital Twin service."""

    status: str = Field(..., description="Service health status")
    service: str = Field(default="digital-twin", description="Service name")
    models_available: int = Field(..., description="Number of twin models available")
    brands_simulable: int = Field(
        0,
        description=(
            "Brands with an active twin model AND at least one intervention whose effect is "
            "identified in the connected cohort, i.e. brands /simulate can actually serve. "
            "models_available > 0 with brands_simulable == 0 means the models are present but the "
            "cohort's treatment data is not (status is then 'degraded')."
        ),
    )
    simulations_pending: int = Field(..., description="Number of pending simulations")
    last_simulation_at: Optional[datetime] = Field(None, description="Timestamp of last simulation")


class InterventionTypeItem(BaseModel):
    """A canonical, selectable intervention type for the simulation dropdown."""

    value: str = Field(..., description="Canonical intervention_type value")
    label: str = Field(..., description="Human-readable label")
    effect_basis: str = Field(
        ...,
        description=(
            "'cohort_causal' (effect is IDENTIFIED in the connected cohort and estimated "
            "by direct DML causal estimation) or 'unavailable' (not identified in the "
            "data — no fabricated effect is produced)"
        ),
    )
    available: bool = Field(
        ...,
        description=(
            "True if a trained twin model exists for the requested brand/twin_type "
            "(else /simulate would 503)."
        ),
    )
    available_for_effect: bool = Field(
        ...,
        description=(
            "True only if the intervention's effect is IDENTIFIED in the connected cohort "
            "(a causal estimate is possible). The frontend should expose only "
            "effect-available interventions; the rest are an honest 'no effect data' "
            "state rather than a fabricated uplift (and /simulate returns 422 for them)."
        ),
    )


class InterventionTypesResponse(BaseModel):
    """Brand-aware list of canonical intervention types for the dropdown."""

    interventions: List[InterventionTypeItem] = Field(default_factory=list)
    model_resolution: Literal["resolved", "unavailable", "not_requested"] = Field(
        "not_requested",
        description=(
            "'resolved': the brand's active models were looked up, so available=False means no "
            "trained model exists. 'unavailable': the lookup itself failed (repository or DB "
            "unreachable) — every flag is unknown, not an absence; retry. 'not_requested': no "
            "brand was given, the catalog alone was served."
        ),
    )
    effect_availability_status: Optional[Literal["measured", "unmeasured"]] = Field(
        None,
        description=(
            "Set only when model_resolution is 'resolved'. 'measured': every cohort probe ran, so "
            "available_for_effect=False means the cohort holds too few usable rows for that "
            "channel. 'unmeasured': the probes errored and nothing usable was found — the flags "
            "are unknown, not a finding; retry, do not restore data."
        ),
    )
    brand: Optional[str] = Field(None, description="Brand the availability was resolved for")
    twin_type: str = Field(..., description="Twin type the availability was resolved for")
    timestamp: datetime = Field(..., description="Response timestamp")


# =============================================================================
# Fidelity + model schemas (moved out of routes/digital_twin.py by the module-size
# ratchet; #2206 added the honesty fields on TwinModelSummary)
# =============================================================================


class FidelityStatusEnum(str, Enum):
    """Whether the model's fidelity has ever been measured (#2206).

    'unvalidated': digital_twin_models.fidelity_score is NULL — no experiment outcome
    has been compared against the model; a NULL never reads as "passed".
    'below_threshold': measured and below the engine's 0.70 gate.
    'validated': measured, at or above the gate.
    """

    UNVALIDATED = "unvalidated"
    BELOW_THRESHOLD = "below_threshold"
    VALIDATED = "validated"


class R2ScoreBasisEnum(str, Enum):
    """What the model's r2_score was scored against (#2206).

    'synthetic_target': the fit's target is self-generated by
    synthetic_training_frame (data_provenance 'synthetic') — the R² says how well the
    model reproduces its own synthetic generator, not real-world outcomes.
    'rwd_target': trained on a real-world data file.
    """

    SYNTHETIC_TARGET = "synthetic_target"
    RWD_TARGET = "rwd_target"
    UNKNOWN = "unknown"


class FidelityGradeEnum(str, Enum):
    """Fidelity grade values."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNVALIDATED = "unvalidated"


class FidelityRecordResponse(BaseModel):
    """Fidelity validation record."""

    tracking_id: str
    simulation_id: str
    experiment_id: Optional[str] = None
    simulated_ate: float
    simulated_ci_lower: Optional[float] = None
    simulated_ci_upper: Optional[float] = None
    actual_ate: Optional[float] = None
    actual_ci_lower: Optional[float] = None
    actual_ci_upper: Optional[float] = None
    actual_sample_size: Optional[int] = None
    prediction_error: Optional[float] = None
    absolute_error: Optional[float] = None
    ci_coverage: Optional[bool] = None
    fidelity_grade: FidelityGradeEnum
    validation_notes: Optional[str] = None
    confounding_factors: List[str] = []
    created_at: datetime
    validated_at: Optional[datetime] = None
    validated_by: Optional[str] = None


class TwinModelSummary(BaseModel):
    """Summary of a twin generator model."""

    model_id: str
    model_name: str
    twin_type: str
    brand: str
    algorithm: str
    r2_score: Optional[float] = None
    rmse: Optional[float] = None
    training_samples: int
    is_active: bool
    created_at: datetime
    # --- #2206 honest surfacing (all derived from the stored rows) ---
    fidelity_status: FidelityStatusEnum = Field(
        description=(
            "'unvalidated' while fidelity_score is NULL / fidelity_sample_count is 0 — "
            "no experiment outcome has been compared against this model; not a pass."
        ),
    )
    fidelity_score: Optional[float] = Field(
        default=None, description="Mean fidelity over the model's A/B comparisons; NULL = none."
    )
    fidelity_sample_count: int = Field(
        default=0, description="Number of experiment comparisons behind fidelity_score."
    )
    data_provenance: Optional[str] = Field(
        default=None,
        description="Training-frame provenance as recorded: 'synthetic' or 'rwd_file'.",
    )
    r2_score_basis: R2ScoreBasisEnum = Field(
        description=(
            "What r2_score was scored against. 'synthetic_target': the target is "
            "self-generated by the synthetic training frame, so R² measures how well the "
            "model reproduces its own generator — not real-world outcomes."
        ),
    )
    brand_is_feature: bool = Field(
        description="Whether 'brand' is one of the model's feature_columns (False: brand is routing metadata).",
    )
    training_fingerprint: str = Field(
        description=(
            "Content hash of the RECORDED fit: training_config (with the training_frame "
            "source/seed/rows when the trainer recorded it), feature/target columns, and "
            "every reported metric except wall-clock. Equal fingerprints = the same "
            "recorded fit under several labels; the artifact itself is not hashed."
        ),
    )
    shared_fit_model_count: int = Field(
        description=(
            "Distinct BRANDS whose active model of this twin_type has the same "
            "training_fingerprint, including this one (two active rows of one brand "
            "count once). >1 means brand is a label over ONE recorded fit."
        ),
    )
    shared_fit_with: List[str] = Field(
        default_factory=list,
        description="Other brands sharing this exact fit that the caller may read (brand-scoped).",
    )
    training_frame_recorded: bool = Field(
        description=(
            "Whether a CONTENT digest of the training frame "
            "(training_config.training_frame.content_sha256) was recorded for this row. "
            "False for rows trained before it was recorded: their fingerprint compares "
            "configuration, columns and metrics only — training-frame and artifact "
            "identity were not recorded."
        ),
    )


class TwinModelDetailResponse(TwinModelSummary):
    """Detailed twin model information."""

    model_description: Optional[str] = None
    feature_columns: List[str]
    target_column: str
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    feature_importances: Dict[str, float]
    top_features: List[str]
    training_duration_seconds: float
    config: Dict[str, Any]


class ModelListResponse(BaseModel):
    """Response for listing models."""

    total_count: int
    models: List[TwinModelSummary]


class FidelityHistoryResponse(BaseModel):
    """Fidelity history for a model."""

    model_id: str
    total_validations: int
    average_fidelity_score: Optional[float] = None
    grade_distribution: Dict[str, int]
    records: List[FidelityRecordResponse]


class FidelityReportResponse(BaseModel):
    """Aggregated fidelity report for a model."""

    model_id: str
    total_validations: int
    average_fidelity_score: float
    coverage_rate: float
    grade_distribution: Dict[str, int]
    trend: str
    is_degrading: bool
    degradation_rate: Optional[float] = None
    recommendation: str
    generated_at: datetime
