"""Public response contracts shared by the Digital Twin routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional

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


def live_subgroups_basis(data_provenance: Optional[str], *, calculated: bool = True) -> str:
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
