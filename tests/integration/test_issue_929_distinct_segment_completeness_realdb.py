"""Issue #929 — faithful real-DB proof that the benchmark frame no longer drops regions.

Run against the docker Supabase (gated by ``E2I_DB_INTEGRATION=1``) with ``-n0``
(xdist + a real async client do not mix). Remibrutinib has 4 regions in
``business_metrics`` (northeast 1697, south 2278, midwest 1714, west 1600 = 7289 rows).
Before the fix, ``get_distinct_values`` read a single ``.limit(5000)`` window with no
``ORDER BY`` and silently dropped ``west``; the cross-segment P75/P90 standard was then
computed over only 3 of 4 regions. These tests prove all 4 regions are now discovered
and that ``west`` appears in the peer-benchmark frame.

No mocks: this reads the live synthetic-gold substrate end-to-end.
"""

from __future__ import annotations

import os

import pytest

_RUN = os.environ.get("E2I_DB_INTEGRATION") == "1"

# The worktree has no local .env; let python-dotenv walk up to the main repo's .env
# so the faithful run can resolve the Supabase client. Gated so CI (skipped) never
# depends on it.
if _RUN:
    from dotenv import load_dotenv

    load_dotenv()

pytestmark = pytest.mark.skipif(
    not _RUN, reason="set E2I_DB_INTEGRATION=1 to run faithful real-DB tests"
)

_BRAND = "Remibrutinib"
_EXPECTED_REGIONS = {"northeast", "south", "midwest", "west"}


@pytest.fixture(autouse=True)
def _reset_async_client_cache():
    """Reset the loop-bound async client cache around each test (see #851)."""
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


@pytest.mark.asyncio
async def test_get_distinct_regions_returns_all_four_regions():
    """All 4 regions are discovered — the >5000-row brand no longer drops ``west``."""
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.business_metric import BusinessMetricRepository

    client = await get_async_supabase_client()
    repo = BusinessMetricRepository(client)

    regions = await repo.get_distinct_values("region", brand=_BRAND, include_synthetic=True)

    assert set(regions) == _EXPECTED_REGIONS, f"expected all 4 regions for {_BRAND}, got {regions}"


@pytest.mark.asyncio
async def test_peer_benchmark_frame_includes_west():
    """The peer-benchmark frame carries a row for EVERY region, including ``west``."""
    from src.agents.gap_analyzer.connectors.benchmark_store import BenchmarkStore

    store = BenchmarkStore(include_synthetic=True)
    frame = await store.get_peer_benchmarks(
        brand=_BRAND,
        metrics=["trx"],
        segments=["region"],
    )

    assert not frame.empty, "expected a non-empty peer-benchmark frame"
    assert "region" in frame.columns
    regions = set(frame["region"].tolist())
    assert _EXPECTED_REGIONS.issubset(regions), (
        f"peer-benchmark frame must include west; got regions {regions}"
    )
