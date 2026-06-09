"""Faithful (real-DB, NO mocks) regression tests for the Home KPI-summary
fabrication bug (H1).

The Home "Total TRx (MTD)" / "HCPs Reached" tiles are fed by
copilotkit.get_kpi_summary(). The buggy code built BusinessMetricRepository with
the WRONG kwarg (-> TypeError -> repo None -> data_source='fallback'), read the
SYNTHETIC `business_metrics` table even when fixed, and fell back to hardcoded
`_FALLBACK_KPIS` (Kisqali trx_volume=22100, hcp_reach=3200) -> fabricated values
shown as real on the landing page.

The honest fix reads the REAL allowlisted KPI queries (treatment_events / triggers
via the kpi_query RPC). Stale/empty source -> honest zeros with data_source='database'
(NOT fabricated); a hard query failure -> data_source='unavailable' (fail-closed).

Opt-in (real docker supabase-db required), skipped in CI by default:
    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_kpi_summary_realdb.py -p no:cacheprovider
"""

import os

import pytest

from src.api.routes.copilotkit import get_kpi_summary

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

# The exact hardcoded fabrication the tiles used to show (must never appear).
_FABRICATED_KISQALI_TRX = 22100
_FABRICATED_KISQALI_HCP_REACH = 3200


@pytest.mark.asyncio
async def test_kpi_summary_never_returns_fabricated_fallback():
    """get_kpi_summary must not return the hardcoded _FALLBACK_KPIS values nor
    data_source='fallback' for a valid brand."""
    result = await get_kpi_summary("Kisqali")

    assert result["data_source"] != "fallback", "tiles still serve the hardcoded fallback"
    assert result["data_source"] in {"database", "unavailable"}, result["data_source"]

    metrics = result["metrics"]
    assert metrics.get("trx_volume") != _FABRICATED_KISQALI_TRX, (
        "trx_volume is the fabricated value"
    )
    assert metrics.get("hcp_reach") != _FABRICATED_KISQALI_HCP_REACH, (
        "hcp_reach is the fabricated value"
    )


@pytest.mark.asyncio
async def test_kpi_summary_trx_matches_real_allowlist_query():
    """trx_volume must equal the REAL business_impact_trx allowlist query
    (treatment_events prescriptions, 30d), not a synthetic/fabricated number."""
    from src.api.dependencies.supabase_client import get_supabase

    client = get_supabase()
    rpc = client.rpc(
        "kpi_query", {"query_id": "business_impact_trx", "params": ["Kisqali"]}
    ).execute()
    real_trx = float(rpc.data[0]["trx"]) if rpc.data else 0.0

    result = await get_kpi_summary("Kisqali")
    assert result["data_source"] == "database"
    assert float(result["metrics"]["trx_volume"]) == real_trx


@pytest.mark.asyncio
async def test_kpi_summary_hcp_reach_is_integer_count_not_coverage_fraction():
    """hcp_reach must be a whole-number HCP COUNT (the FE renders it as a count),
    never the WS3-BI-004 coverage fraction (~27.3)."""
    result = await get_kpi_summary("Kisqali")
    hcp_reach = result["metrics"].get("hcp_reach")
    assert hcp_reach is not None
    assert float(hcp_reach) == int(hcp_reach), (
        f"hcp_reach must be an integer count, got {hcp_reach}"
    )


@pytest.mark.asyncio
async def test_kpi_summary_invalid_brand_is_conformant_not_raw_error_dict():
    """Invalid brand must still return the normal {brand, period, metrics, data_source}
    shape (honest 'unavailable'), not a bare {'error': ...} that breaks the FE contract."""
    result = await get_kpi_summary("NotARealBrand")
    assert set(result.keys()) >= {"brand", "metrics", "data_source"}
    assert result["data_source"] == "unavailable"


@pytest.mark.asyncio
async def test_kpi_summary_all_brand_market_share_is_none_not_misleading_zero():
    """market_share (TRx share) is inherently per-brand; for the aggregate "All"
    view it must be honest None, not the misleading 0 a NULL-brand share query
    would yield. Brand-agnostic metrics (e.g. trx_volume) ARE defined for "All"."""
    result = await get_kpi_summary("All")
    assert result["data_source"] == "database"
    assert result["metrics"]["market_share"] is None
    assert "trx_volume" in result["metrics"]
