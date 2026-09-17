"""#2104 item 3: the detail read says HOW a stored row's subgroups were computed.

Rows written before #2097 on the cohort path carry by_specialty / by_decile /
by_adoption_stage that averaged region CATEs over the GENERATED TWINS, and a
simulation_confidence scored on twin count. The owner ruled they are annotated, not
migrated: ``subgroups_basis`` on the detail response tells them apart from cohort-path
rows and the stored JSON is served unchanged.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.digital_twin.effect.estimate import PROVENANCE_COHORT, PROVENANCE_SYNTHETIC

ADMIN = {"app_metadata": {"role": "admin"}}

_LEGACY_SPECIALTY = {"oncology": {"mean": 0.05, "std": 0.01, "n": 12}}
_REGION = {"northeast": {"ate": 0.1, "std": 0.0, "n": 900}}


def _row(**overrides):
    row = {
        "simulation_id": str(uuid4()),
        "model_id": str(uuid4()),
        "intervention_type": "email_campaign",
        "intervention_config": {},
        "brand": "Kisqali",
        "twin_count": 600,
        "simulated_ate": 0.2585,
        "simulated_ci_lower": 0.1995,
        "simulated_ci_upper": 0.3174,
        "simulated_std_error": 0.03,
        "recommendation": "deploy",
        "recommendation_rationale": "r",
        "simulation_confidence": 0.8,
        "simulation_status": "completed",
        "execution_time_ms": 1,
        "created_at": datetime.now(timezone.utc),
        "population_filters": {},
        "effect_scope_regions": [],
        "effect_heterogeneity": {
            "by_specialty": {},
            "by_decile": {},
            "by_region": {},
            "by_adoption_stage": {},
            "top_segments": [],
        },
    }
    row.update(overrides)
    return row


def _read_detail(row):
    from src.api.routes import digital_twin as dt

    repo = SimpleNamespace(get_simulation=AsyncMock(return_value=row))
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        return asyncio.run(dt.get_simulation(row["simulation_id"], user=ADMIN))


@pytest.mark.unit
def test_a_pre_2097_cohort_row_is_annotated_twin_weighted_legacy_and_served_unchanged():
    eh = {
        "by_specialty": dict(_LEGACY_SPECIALTY),
        "by_decile": {},
        "by_region": dict(_REGION),
        "by_adoption_stage": {},
        "top_segments": [],
    }
    detail = _read_detail(_row(data_provenance=PROVENANCE_COHORT, effect_heterogeneity=eh))

    assert detail.subgroups_basis == "twin_weighted_legacy"
    # Annotated, not migrated: the stored JSON comes back verbatim, and the twin-scored
    # confidence is not recomputed (unchanged apart from the 3-dp rounding every row gets).
    assert detail.effect_heterogeneity.by_specialty == _LEGACY_SPECIALTY
    assert detail.effect_heterogeneity.by_region == _REGION
    assert detail.simulation_confidence == 0.8


@pytest.mark.unit
def test_a_cohort_row_with_region_only_is_cohort_rows():
    eh = {"by_specialty": {}, "by_decile": {}, "by_region": dict(_REGION), "by_adoption_stage": {}}
    detail = _read_detail(_row(data_provenance=PROVENANCE_COHORT, effect_heterogeneity=eh))

    assert detail.subgroups_basis == "cohort_rows"


@pytest.mark.unit
def test_a_new_specialty_axis_keeps_its_support_provenance_in_detail():
    provenance = {
        "basis": "cohort_rows",
        "source": "hcp_profiles.specialty",
        "min_group_rows": 100,
        "min_treated_rows": 20,
        "min_control_rows": 20,
        "fallback": "region_then_cohort",
        "support_unit": "cohort_rows",
        "estimand": "observed_region_mix_mean_cate",
        "suppressed_groups": {},
    }
    eh = {
        "by_specialty": {"oncology": {"ate": 0.05, "std": 0.0, "n": 300}},
        "by_decile": {},
        "by_region": dict(_REGION),
        "by_adoption_stage": {},
        "axis_provenance": {"specialty": provenance},
    }

    detail = _read_detail(_row(data_provenance=PROVENANCE_COHORT, effect_heterogeneity=eh))

    assert detail.subgroups_basis == "cohort_rows"
    assert detail.effect_heterogeneity.axis_provenance["specialty"].model_dump() == provenance


@pytest.mark.unit
def test_live_domain_heterogeneity_maps_to_the_same_specialty_contract():
    from src.api.routes import digital_twin as dt
    from src.digital_twin.models.simulation_models import (
        EffectHeterogeneity,
        SubgroupAxisProvenance,
    )

    live = EffectHeterogeneity(
        by_specialty={"oncology": {"ate": 0.05, "std": 0.0, "n": 300}},
        axis_provenance={
            "specialty": SubgroupAxisProvenance(
                basis="cohort_rows",
                source="hcp_profiles.specialty",
                min_group_rows=100,
                min_treated_rows=20,
                min_control_rows=20,
                fallback="region_then_cohort",
                support_unit="cohort_rows",
                estimand="observed_region_mix_mean_cate",
            )
        },
    )

    response = dt._heterogeneity_response(live)

    assert response.by_specialty["oncology"] == {"ate": 0.05, "std": 0.0, "n": 300.0}
    assert response.axis_provenance["specialty"].source == "hcp_profiles.specialty"


@pytest.mark.unit
def test_live_heterogeneity_mapper_rejects_untyped_runtime_objects():
    from src.api.routes import digital_twin as dt

    with pytest.raises(TypeError, match="Pydantic domain model or stored JSON mapping"):
        dt._heterogeneity_response(object())


@pytest.mark.unit
def test_a_synthetic_row_is_per_twin():
    detail = _read_detail(_row(data_provenance=PROVENANCE_SYNTHETIC))

    assert detail.subgroups_basis == "per_twin"


@pytest.mark.unit
def test_a_row_without_provenance_is_unknown():
    detail = _read_detail(_row(data_provenance=None))

    assert detail.subgroups_basis == "unknown"
