"""Public response contracts shared by the Digital Twin routes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
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
    effect_availability_status: Literal["measured", "unmeasured"] = Field(
        "measured",
        description=(
            "'measured' when every cohort probe ran, so available_for_effect=False means the "
            "cohort holds too few usable rows for that channel. 'unmeasured' when the probes "
            "errored and nothing usable was found: the flags are then unknown, not a finding — "
            "retry, do not restore data."
        ),
    )
    brand: Optional[str] = Field(None, description="Brand the availability was resolved for")
    twin_type: str = Field(..., description="Twin type the availability was resolved for")
    timestamp: datetime = Field(..., description="Response timestamp")
