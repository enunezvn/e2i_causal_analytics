"""Issue #931 — faithful real-DB proof that per-segment benchmark means read the
COMPLETE (brand, region) slice (no single-window truncation).

Run against the docker Supabase (gated by ``E2I_DB_INTEGRATION=1``) with ``-n0``.
The bug is LATENT on current data (max brand×region = 2326 rows, under the 5000 window),
so we reproduce the multi-page scenario FAITHFULLY by shrinking ``page_size`` below the
real slice size: ``Remibrutinib`` × ``south`` has > 500 rows, so a single ``.limit(500)``
window truncates while ``get_by_region_paged(page_size=500)`` pages to exhaustion and
returns the WHOLE slice — yielding the correct mean. No mocks.
"""

from __future__ import annotations

import os

import pytest

_RUN = os.environ.get("E2I_DB_INTEGRATION") == "1"

if _RUN:
    from dotenv import load_dotenv

    load_dotenv()

pytestmark = pytest.mark.skipif(
    not _RUN, reason="set E2I_DB_INTEGRATION=1 to run faithful real-DB tests"
)

_BRAND = "Remibrutinib"
_REGION = "south"
_PAGE = 500  # deliberately below the real slice size to force multi-page paging


@pytest.fixture(autouse=True)
def _reset_async_client_cache():
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


async def _full_slice_rows(client):
    """Ground truth: ALL rows for (brand, region) in one high-limit window (slice < 5000)."""
    res = (
        await client.table("business_metrics")
        .select("metric_name,value")
        .eq("brand", _BRAND)
        .eq("region", _REGION)
        .limit(100000)
        .execute()
    )
    return res.data or []


@pytest.mark.asyncio
async def test_paged_fetch_exhausts_multi_page_real_slice():
    """get_by_region_paged returns the WHOLE slice across pages; a single .limit(page)
    window truncates it — proving the fix on real data."""
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.business_metric import BusinessMetricRepository

    client = await get_async_supabase_client()
    repo = BusinessMetricRepository(client)

    full = await _full_slice_rows(client)
    full_count = len(full)
    assert full_count > _PAGE, (
        f"test precondition: {_BRAND}/{_REGION} must exceed {_PAGE} rows to span pages "
        f"(got {full_count})"
    )

    # Single un-paged window of page_size rows truncates (the #931 bug shape).
    single = (
        await client.table("business_metrics")
        .select("metric_name,value")
        .eq("brand", _BRAND)
        .eq("region", _REGION)
        .limit(_PAGE)
        .execute()
    )
    assert len(single.data or []) == _PAGE  # truncated

    # The paged fetch exhausts the slice.
    paged = await repo.get_by_region_paged(
        region=_REGION,
        brand=_BRAND,
        include_synthetic=True,
        columns="metric_name,value",
        page_size=_PAGE,
    )
    assert len(paged) == full_count, (
        f"paged fetch must return the whole {full_count}-row slice, got {len(paged)}"
    )


@pytest.mark.asyncio
async def test_paged_mean_matches_full_slice_mean():
    """The per-(region, metric) mean from the paged fetch equals the full-slice mean —
    a single-window read would bias it once the slice exceeds the window."""
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.business_metric import BusinessMetricRepository

    client = await get_async_supabase_client()
    repo = BusinessMetricRepository(client)

    full = await _full_slice_rows(client)
    # Pick the metric with the most rows in this slice.
    from collections import Counter

    counts = Counter(r["metric_name"] for r in full if r.get("metric_name"))
    metric, _ = counts.most_common(1)[0]
    truth_vals = [
        float(r["value"])
        for r in full
        if r.get("metric_name") == metric and r.get("value") is not None
    ]
    truth_mean = sum(truth_vals) / len(truth_vals)

    paged = await repo.get_by_region_paged(
        region=_REGION,
        brand=_BRAND,
        include_synthetic=True,
        columns="metric_name,value",
        page_size=_PAGE,
    )
    paged_vals = [
        float(r["value"])
        for r in paged
        if r.get("metric_name") == metric and r.get("value") is not None
    ]
    paged_mean = sum(paged_vals) / len(paged_vals)

    assert len(paged_vals) == len(truth_vals)
    assert paged_mean == pytest.approx(truth_mean)
