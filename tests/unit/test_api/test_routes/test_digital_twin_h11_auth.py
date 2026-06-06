"""R4b/H11 regression: the twin read GETs (no auth at all before) must require a
viewer-tier token and fail-closed brand scoping — a non-admin cannot read another
brand's simulations via ?brand=. Admin / ['all'] is unaffected (canonical
resolve_brand_for_read bypass)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
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
        resp = asyncio.run(list_simulations(brand=BrandEnum.REMIBRUTINIB, user=ADMIN))
    assert resp.total_count == 0


@pytest.mark.unit
def test_get_simulation_out_of_grant_is_404():
    from src.api.routes import digital_twin as dt
    from src.api.routes.digital_twin import get_simulation

    # A real-ish result whose brand is Remibrutinib (out of the Kisqali grant).
    result = MagicMock()
    result.intervention_config.extra_params = {"brand": "Remibrutinib", "twin_type": "hcp"}
    repo = SimpleNamespace(get_simulation=AsyncMock(return_value=result))
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(get_simulation(str(uuid4()), user=VIEWER_KISQALI))
    # 404, not 403 — do not leak existence of another tenant's simulation.
    assert ei.value.status_code == 404
