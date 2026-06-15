"""WS-BACKEND live: the SystemHealth probes report UP (were false
'database unhealthy' / 'vector_store degraded' alarms) and the model-status
resolver surfaces the real gold-standard models. Real DB, no mocks.

DROPLET-ONLY LIVE SMOKE: these assert prod-specific data state (the 12 activated
gold-standard models; the prod RLS posture on rag_document_chunks), which CI's
ephemeral integration Supabase does not have. No CI lane sets E2I_LIVE_SMOKE, so
they skip in CI and run only when opted in against the deployed stack on the
droplet (precedent: E2I_RUN_LIVE_RAG in tests/rag/). Verified live 2026-06-15.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_LIVE_SMOKE") != "1",
    reason="droplet-only live smoke; set E2I_LIVE_SMOKE=1 to run against the deployed stack",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_probes_report_up_live():
    from src.agents.health_score.health_client import SupabaseHealthClient

    client = SupabaseHealthClient()
    try:
        db = await client._check_database()
        vs = await client._check_vector_store()
    finally:
        await client.close()

    # database: was a false PGRST202 404 from rpc("version"); now httpx REST probe.
    assert db["ok"] is True, db
    # vector_store: was unconditionally degraded (VECTOR_STORE_URL unset); pgvector
    # lives in Postgres and is reachable.
    assert vs["ok"] is True and not vs.get("degraded"), vs


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_resolver_surfaces_goldstd_live():
    from src.api.routes.predictions import _resolve_production_model_names

    names = await _resolve_production_model_names()
    # The 12 gold-standard staging models were previously invisible (only 2 legacy
    # csu_* production models showed).
    assert any("goldstd" in n for n in names), names
    assert len(names) >= 12
