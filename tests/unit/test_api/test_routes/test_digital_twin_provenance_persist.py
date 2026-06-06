"""R4b/H5b regression: data_provenance must round-trip through the persisted row
to the GET reads (list + detail), not just the live POST response (R1). Migration
030 adds the column; save_simulation persists it; the read mappings surface it."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

ADMIN = {"app_metadata": {"role": "admin"}}


@pytest.mark.unit
def test_save_simulation_row_includes_data_provenance():
    """The persisted row must carry data_provenance (so reads can surface it)."""
    from src.digital_twin.twin_repository import SimulationRepository

    captured = {}

    class _Tbl:
        def insert(self, row):
            captured["row"] = row
            return self

        async def execute(self):
            return MagicMock(data=[captured["row"]])

    client = MagicMock()
    client.table = MagicMock(return_value=_Tbl())
    repo = SimulationRepository(supabase_client=client)

    result = MagicMock()
    result.simulation_id = uuid4()
    result.data_provenance = "synthetic_uplift_v1"
    result.population_filters.to_dict.return_value = {}
    result.effect_heterogeneity.model_dump.return_value = {}
    result.status.value = "completed"
    result.recommendation.value = "deploy"
    result.created_at = None
    result.completed_at = None

    asyncio.run(repo.save_simulation(result, "Kisqali"))
    assert captured["row"]["data_provenance"] == "synthetic_uplift_v1"


@pytest.mark.unit
def test_list_simulations_surfaces_data_provenance():
    from datetime import datetime, timezone

    from src.api.routes import digital_twin as dt
    from src.api.routes.digital_twin import list_simulations

    row = {
        "simulation_id": str(uuid4()),
        "intervention_type": "email_campaign",
        "brand": "Kisqali",
        "twin_type": "hcp",
        "twin_count": 1000,
        "simulated_ate": 0.1,
        "recommendation": "deploy",
        "simulation_status": "completed",
        "created_at": datetime.now(timezone.utc),
        "data_provenance": "synthetic_uplift_v1",
    }
    repo = SimpleNamespace(
        simulations=SimpleNamespace(list_simulations=AsyncMock(return_value=[row]))
    )
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        resp = asyncio.run(
            list_simulations(
                brand=None, model_id=None, status=None, page=1, page_size=20, user=ADMIN
            )
        )
    assert resp.simulations[0].data_provenance == "synthetic_uplift_v1"
