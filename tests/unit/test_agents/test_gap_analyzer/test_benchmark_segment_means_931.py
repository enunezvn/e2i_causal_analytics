"""Issue #931 — BenchmarkStore must aggregate the COMPLETE (brand, region) slice.

The per-segment mean must come from the paged, exhaustive fetch (``get_by_region_paged``)
— NOT the truncating single-window ``get_by_region(limit=5000)``. These CI-safe unit
tests pin (a) the wiring (paged method is used, the truncating one is not) and (b) the
mean is computed over ALL returned rows.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.gap_analyzer.connectors.benchmark_store import BenchmarkStore


@pytest.mark.unit
@pytest.mark.asyncio
async def test_benchmark_uses_paged_region_fetch_not_truncating_limit():
    """_fetch_segment_metric_frame must call the exhaustive paged fetch, never the
    single-window get_by_region(limit=5000) that silently truncates the mean (#931)."""
    store = BenchmarkStore(supabase_client=MagicMock())

    fake_repo = MagicMock()
    fake_repo.get_distinct_values = AsyncMock(return_value=["north", "south"])
    fake_repo.get_by_region_paged = AsyncMock(return_value=[{"metric_name": "trx", "value": 10.0}])
    # A truthy get_by_region that would FAIL the test if it were (wrongly) used.
    fake_repo.get_by_region = AsyncMock(
        side_effect=AssertionError("benchmark must not use the truncating get_by_region")
    )
    store._repository = fake_repo

    await store.get_peer_benchmarks(brand="Kisqali", metrics=["trx"], segments=["region"])

    assert fake_repo.get_by_region_paged.await_count >= 1
    fake_repo.get_by_region.assert_not_called()
    # The paged fetch is asked for region+brand with the provenance flag threaded.
    kwargs = fake_repo.get_by_region_paged.await_args.kwargs
    assert kwargs.get("brand") == "Kisqali"
    assert kwargs.get("include_synthetic") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_benchmark_mean_is_over_all_returned_rows():
    """The per-segment value is the mean of EVERY row the paged fetch returns, not the
    first window. get_targets returns the raw per-segment frame (no cross-segment broadcast),
    so we can assert the mean directly."""
    store = BenchmarkStore(supabase_client=MagicMock())

    fake_repo = MagicMock()
    fake_repo.get_distinct_values = AsyncMock(return_value=["south"])

    # 3 rows for south -> mean target = (10+20+30)/3 = 20.0 (not 10.0, the first row).
    async def _paged(region, brand, include_synthetic, columns="*", **_):
        return [
            {"metric_name": "trx", "target": 10.0},
            {"metric_name": "trx", "target": 20.0},
            {"metric_name": "trx", "target": 30.0},
        ]

    fake_repo.get_by_region_paged = AsyncMock(side_effect=_paged)
    store._repository = fake_repo

    frame = await store.get_targets(brand="Remibrutinib", metrics=["trx"], segments=["region"])

    assert not frame.empty
    south = frame[frame["region"] == "south"]
    assert float(south["trx"].iloc[0]) == pytest.approx(20.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_segment_frame_rejects_unknown_value_field():
    """value_field is interpolated into the PostgREST select list; only the real numeric
    columns 'value'/'target' are allowed (defense-in-depth against a future caller
    injecting an arbitrary column, #931 codex review)."""
    store = BenchmarkStore(supabase_client=MagicMock())
    fake_repo = MagicMock()
    fake_repo.get_distinct_values = AsyncMock(return_value=["south"])
    fake_repo.get_by_region_paged = AsyncMock(return_value=[])
    store._repository = fake_repo

    with pytest.raises(ValueError):
        await store._fetch_segment_metric_frame(
            brand="Remibrutinib",
            metrics=["trx"],
            segments=["region"],
            value_field="value,is_synthetic",
        )
