"""WS-ENGINE live: the chatbot kpi_calculate_tool computes a defined KPI from the
REAL engine (not the materialized business_metrics fixture).

Requires the live Supabase + the kpi_query_registry RPC. On a synthetic-gold
instance (E2I_KPI_INCLUDE_SYNTHETIC / E2I_INCLUDE_SYNTHETIC set) NBRx (WS3-BI-007)
computes from treatment_events; the value is asserted only as present/positive so
the test is not brittle to data regeneration (real-results, never mocked).

DROPLET-ONLY LIVE SMOKE: these compute against the deployed KPI engine + real
treatment_events, which CI's ephemeral integration Supabase does not carry. No CI
lane sets E2I_LIVE_SMOKE, so they skip in CI and run only when opted in against
the deployed stack on the droplet (precedent: E2I_RUN_LIVE_RAG in tests/rag/).
The unknown-KPI fail-closed path is also covered by the CI-portable unit test
tests/unit/test_api/test_chatbot_kpi_tool.py, so no CI coverage is lost.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_LIVE_SMOKE") != "1",
    reason="droplet-only live smoke; set E2I_LIVE_SMOKE=1 to run against the deployed stack",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kpi_calculate_tool_computes_nbrx_live():
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    resp = await kpi_calculate_tool.ainvoke({"kpi_name": "NBRx", "brand": "Kisqali"})

    # The whole point of WS-ENGINE: NBRx resolves + computes instead of count:0.
    assert resp["success"] is True, resp
    assert resp["kpi_id"] == "WS3-BI-007"
    assert resp["value"] is not None and resp["value"] > 0
    assert resp["data_source"] in {"synthetic", "database"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kpi_calculate_tool_unknown_kpi_fails_closed():
    from src.api.routes.chatbot_tools import kpi_calculate_tool

    resp = await kpi_calculate_tool.ainvoke({"kpi_name": "definitely-not-a-kpi-xyz"})
    assert resp["success"] is False
    assert "did not resolve" in resp["error"]
