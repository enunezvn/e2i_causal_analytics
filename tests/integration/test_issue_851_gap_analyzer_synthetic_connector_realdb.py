"""Issue #851: gap_analyzer production connector path must read the synthetic substrate.

Faithful real-DB integration tests (docker Supabase, gated by ``E2I_DB_INTEGRATION=1``).
Run with ``-n0`` (xdist + a real async client do not mix) and ``LOKY_MAX_CPU_COUNT=1``.

These reproduce the TRIPLE block described in #851 and prove the fix:
  1. ``get_data_connector`` returned a CLIENT-LESS connector (the #845 DI family) — no
     DB, every fetch returns [].
  2. ``include_synthetic`` was not plumbed through the connector → the Shard-07 default
     excluded synthetic rows even given a client.
  3. ``BenchmarkStore`` hardcoded title-case regions (``Northeast`` …) while the live
     enum is lowercase (``northeast`` …) → zero matches.

The PRODUCTION DEFAULT stays fail-closed: ``include_synthetic=False`` must NOT read
synthetic rows (real-mode isolation). Kisqali in the prod DB has ONLY synthetic rows,
so the default connector correctly returns no data for it.
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


@pytest.fixture(autouse=True)
def _reset_async_client_cache():
    """Reset the module-global async Supabase client cache around EACH test.

    ``get_async_supabase_client`` caches a client bound to the event loop that
    created it. pytest-asyncio gives each test its own loop, so a client cached by
    an earlier test points at a CLOSED loop ("Event loop is closed") when the next
    test reuses it. Clearing the cache before each test forces a fresh client bound
    to that test's own loop (the connector resolves it lazily). Cheap: one extra
    ``acreate_client`` per test.
    """
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


# A brand that exists ONLY as synthetic rows in the prod docker DB (Shard 05).
_SYNTH_BRAND = "Kisqali"
_METRICS = ["trx", "conversion_rate", "market_share", "nrx"]


@pytest.mark.asyncio
async def test_connector_reads_synthetic_via_production_path():
    """RED before fix: the production connector returns EMPTY for a synthetic-only brand.

    GREEN after fix: with the lazily-resolved client and ``include_synthetic=True``,
    ``fetch_performance_data`` returns real synthetic rows (lowercase regions matched).
    """
    from src.agents.gap_analyzer.connectors import get_data_connector

    connector = get_data_connector(include_synthetic=True)
    df = await connector.fetch_performance_data(
        brand=_SYNTH_BRAND,
        metrics=_METRICS,
        segments=["region"],
        time_period="2012-01-01_2026-12-31",
    )
    assert not df.empty, "connector returned no synthetic rows (triple block not fixed)"
    assert "region" in df.columns
    regions = set(df["region"].astype(str))
    # Live enum is lowercase — the fix must match these, not title-case.
    assert regions & {"northeast", "south", "midwest", "west"}, f"unexpected regions: {regions}"


@pytest.mark.asyncio
async def test_production_default_stays_fail_closed_for_synthetic_only_brand():
    """The prod default (include_synthetic=False) must NOT leak synthetic rows.

    Kisqali has ONLY synthetic rows in the prod DB, so real-mode returns empty —
    that is the CORRECT fail-closed isolation, not a regression.
    """
    from src.agents.gap_analyzer.connectors import get_data_connector

    connector = get_data_connector()  # default include_synthetic=False
    df = await connector.fetch_performance_data(
        brand=_SYNTH_BRAND,
        metrics=_METRICS,
        segments=["region"],
        time_period="2012-01-01_2026-12-31",
    )
    assert df.empty, "real-mode default leaked synthetic rows (provenance isolation broken)"


@pytest.mark.asyncio
async def test_benchmark_store_discovers_lowercase_regions():
    """RED before fix: peer benchmarks query hardcoded title-case regions → zero rows.

    GREEN after fix: regions discovered dynamically from the data (lowercase) → rows.
    """
    from src.agents.gap_analyzer.connectors import get_benchmark_store

    store = get_benchmark_store(include_synthetic=True)
    peers = await store.get_peer_benchmarks(
        brand=_SYNTH_BRAND, metrics=_METRICS, segments=["region"]
    )
    assert not peers.empty, "benchmark store found no peers (title-case region mismatch)"
    # Benchmark frames must be per-region wide (region column + metric columns) so the
    # gap math can align them with current_data — the shape the mock store establishes.
    assert "region" in peers.columns
    regions = set(peers["region"].astype(str))
    assert regions & {"northeast", "south", "midwest", "west"}, f"unexpected regions: {regions}"
    assert any(m in peers.columns for m in _METRICS), f"no metric columns in {list(peers.columns)}"


@pytest.mark.asyncio
async def test_gap_analyzer_returns_opportunities_via_production_connector():
    """END-TO-END: the agent recovers >0 opportunities from the synthetic substrate
    THROUGH the production connector path (NOT the tier0_data passthrough)."""
    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False, include_synthetic=True)
    out = await agent.run(
        {
            "query": "gaps in Kisqali by region",
            "metrics": _METRICS,
            "segments": ["region"],
            "brand": _SYNTH_BRAND,
            "gap_type": "all",
            "time_period": "2012-01-01_2026-12-31",
            # NOTE: deliberately NO tier0_data — this exercises the production connector.
        }
    )
    opps = out.get("prioritized_opportunities") or []
    assert len(opps) >= 3, f"expected >=3 opportunities via production connector, got {len(opps)}"
    assert (out.get("total_addressable_value") or 0) > 0, "TAV should be > 0"
