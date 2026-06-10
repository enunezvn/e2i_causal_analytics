"""Faithful real-DB regression for #857: get_models_for_target must not 414.

``csu_treatment_initiation`` owns ~253 experiments in the prod registry (the
synthetic-validation load). The pre-#857 implementation fetched every experiment
id and passed it to ``.in_("experiment_id", [...])``, overflowing the PostgREST
GET URL -> ``414 URI too long`` -> the live orchestrator resolved NO model. The
server-side FK-embedded join must resolve the deployable models without error.

Gated by ``E2I_DB_INTEGRATION=1``; queries the local docker Supabase (the prod
DB). Run with ``-n0``.
"""

from __future__ import annotations

import os

import pytest

from src.memory.services.factories import get_async_supabase_client
from src.repositories.ml_experiment import MLModelRegistryRepository

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful real-DB test; set E2I_DB_INTEGRATION=1",
)


@pytest.mark.asyncio
async def test_resolves_deployable_models_without_414_at_scale():
    client = await get_async_supabase_client()
    assert client is not None, "needs the live docker Supabase (check .env)"
    repo = MLModelRegistryRepository(supabase_client=client)

    # Must NOT raise (pre-#857 this raised APIError 414 'URI too long').
    names = await repo.get_models_for_target("csu_treatment_initiation", "hcp")

    assert isinstance(names, list)
    # The two activation-registered production models (#840) resolve.
    assert "csu_treatment_initiation_lr_full_v1" in names, names
    assert "csu_treatment_initiation_lr_balanced_v1" in names, names
