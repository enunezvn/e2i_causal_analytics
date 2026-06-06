"""R4b/H11 regression: the twin read GETs (no auth at all before) must require a
viewer-tier token and fail-closed brand scoping — a non-admin cannot read another
brand's simulations via ?brand=. Admin / ['all'] is unaffected (canonical
resolve_brand_for_read bypass)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

VIEWER_KISQALI = {"app_metadata": {"role": "viewer", "brands": ["Kisqali"]}}
ADMIN = {"app_metadata": {"role": "admin"}}


@pytest.mark.unit
def test_list_simulations_denies_out_of_grant_brand():
    from src.api.routes.digital_twin import BrandEnum, list_simulations

    with pytest.raises(HTTPException) as ei:
        asyncio.run(list_simulations(brand=BrandEnum.REMIBRUTINIB, user=VIEWER_KISQALI))
    assert ei.value.status_code == 403


@pytest.mark.unit
def test_list_simulations_admin_allowed():
    from src.api.routes import digital_twin as dt
    from src.api.routes.digital_twin import BrandEnum, list_simulations

    repo = SimpleNamespace(simulations=SimpleNamespace(list_simulations=AsyncMock(return_value=[])))
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        resp = asyncio.run(
            list_simulations(
                brand=BrandEnum.REMIBRUTINIB,
                model_id=None,
                status=None,
                page=1,
                page_size=20,
                user=ADMIN,
            )
        )
    assert resp.total_count == 0


@pytest.mark.unit
def _row(brand: str) -> dict:
    """The RAW twin_simulations row dict that repo.get_simulation actually returns."""
    from datetime import datetime, timezone

    return {
        "simulation_id": str(uuid4()),
        "model_id": str(uuid4()),
        "intervention_type": "email_campaign",
        "intervention_config": {"channel": "email"},
        "brand": brand,
        "twin_count": 1000,
        "simulated_ate": 0.1,
        "simulated_ci_lower": 0.05,
        "simulated_ci_upper": 0.15,
        "simulated_std_error": 0.02,
        "recommendation": "deploy",
        "recommendation_rationale": "ok",
        "simulation_confidence": 0.9,
        "simulation_status": "completed",
        "data_provenance": "synthetic_uplift_v1",
        "execution_time_ms": 10,
        "created_at": datetime.now(timezone.utc),
        "effect_heterogeneity": {},
    }


def test_get_simulation_out_of_grant_is_404():
    from src.api.routes import digital_twin as dt
    from src.api.routes.digital_twin import get_simulation

    # The REAL row shape is a dict (repo returns result.data[0]); brand Remibrutinib
    # is out of the Kisqali grant.
    repo = SimpleNamespace(get_simulation=AsyncMock(return_value=_row("Remibrutinib")))
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(get_simulation(str(uuid4()), user=VIEWER_KISQALI))
    # 404, not 403 — do not leak existence of another tenant's simulation.
    assert ei.value.status_code == 404


def test_get_simulation_in_grant_maps_dict_and_surfaces_provenance():
    """Non-admin reading their OWN brand gets a 200 with the dict correctly mapped
    (the handler must NOT object-access the dict row) + data_provenance surfaced."""
    from src.api.routes import digital_twin as dt
    from src.api.routes.digital_twin import get_simulation

    repo = SimpleNamespace(get_simulation=AsyncMock(return_value=_row("Kisqali")))
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        resp = asyncio.run(get_simulation(str(uuid4()), user=VIEWER_KISQALI))
    assert resp.brand == "Kisqali"
    assert resp.intervention_type == "email_campaign"
    assert resp.simulated_ate == 0.1
    assert resp.is_significant is True  # CI [0.05, 0.15] excludes 0
    assert resp.effect_direction == "positive"
    assert resp.data_provenance == "synthetic_uplift_v1"
