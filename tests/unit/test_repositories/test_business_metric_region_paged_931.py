"""Issue #931 — per-segment benchmark means must read the COMPLETE (brand, region) slice.

``BenchmarkStore._fetch_segment_metric_frame`` averaged the rows returned by
``get_by_region(limit=5000)`` — a single un-ordered ``.limit()`` window (via
``get_many``). Once a (brand, region) slice exceeds the window, the per-(region, metric)
mean is computed over a truncated, arbitrarily-ordered sample, biasing the P75/P90
benchmark. This is the per-VALUE sibling of #929 (which fixed the segment-NAME drop).

``get_by_region_paged`` pages PK-ordered ``.range()`` windows to exhaustion — the same
blessed idiom as the #929 ``get_distinct_values`` fix. These CI-safe unit tests use a
faithful in-memory table (honours ``.eq``/``.order``/``.range``/``.execute``) so the
REAL paging/accumulation/termination runs, not a mock call-chain.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.repositories.business_metric import BusinessMetricRepository


class _FakePagedTable:
    """In-memory PostgREST-like query honouring select/eq/order/range/execute."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self._select: list[str] | None = None
        self._filters: list[tuple[str, object]] = []
        self._order_by: tuple[str, bool] | None = None
        self._range: tuple[int, int] | None = None

    def select(self, cols: str):
        self._select = [c.strip() for c in cols.split(",")]
        return self

    def eq(self, col: str, val):
        self._filters.append((col, val))
        return self

    def order(self, col: str, desc: bool = False):
        self._order_by = (col, desc)
        return self

    def range(self, start: int, end: int):
        self._range = (start, end)
        return self

    async def execute(self):
        rows = list(self._rows)
        for col, val in self._filters:
            rows = [r for r in rows if r.get(col) == val]
        if self._order_by:
            col, desc = self._order_by
            rows.sort(key=lambda r: r.get(col), reverse=desc)
        if self._range:
            start, end = self._range
            rows = rows[start : end + 1]
        if self._select and self._select != ["*"]:
            rows = [{c: r.get(c) for c in self._select} for r in rows]
        res = MagicMock()
        res.data = rows
        return res


class _FakeClient:
    """Returns a FRESH table per ``.table()`` call (the impl rebuilds per page)."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def table(self, _name: str):
        return _FakePagedTable(self._rows)


def _slice_rows(n: int, region: str = "south", brand: str = "Remibrutinib") -> list[dict]:
    """n rows for (brand, region), metric trx, ascending value, PK-ordered."""
    return [
        {
            "metric_id": f"{i:05d}",
            "region": region,
            "brand": brand,
            "metric_name": "trx",
            "value": float(i),
            "is_synthetic": True,
        }
        for i in range(1, n + 1)
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_returns_all_rows_across_pages():
    """Every row in the slice is returned even when it spans multiple pages."""
    rows = _slice_rows(7)  # 7 rows, page_size=2 -> 4 pages
    repo = BusinessMetricRepository(_FakeClient(rows))

    out = await repo.get_by_region_paged(
        region="south", brand="Remibrutinib", include_synthetic=True, page_size=2
    )

    assert len(out) == 7
    # The full mean over ALL rows differs from a first-window (rows 1..2) mean.
    full_mean = sum(r["value"] for r in out) / len(out)
    assert full_mean == pytest.approx(sum(range(1, 8)) / 7)  # 4.0, not 1.5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_filters_brand_and_region_each_page():
    rows = _slice_rows(3, region="south", brand="Remibrutinib") + _slice_rows(
        3, region="west", brand="Kisqali"
    )
    repo = BusinessMetricRepository(_FakeClient(rows))

    out = await repo.get_by_region_paged(
        region="south", brand="Remibrutinib", include_synthetic=True, page_size=2
    )

    assert len(out) == 3
    assert all(r.get("region") in (None, "south") for r in out)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_orders_by_pk():
    captured: dict[str, object] = {}

    class _RecordingTable(_FakePagedTable):
        def order(self, col: str, desc: bool = False):
            captured["order_col"] = col
            return super().order(col, desc)

    class _RecordingClient(_FakeClient):
        def table(self, _name: str):
            return _RecordingTable(self._rows)

    repo = BusinessMetricRepository(_RecordingClient(_slice_rows(3)))
    await repo.get_by_region_paged(region="south", include_synthetic=True, page_size=2)

    assert captured.get("order_col") == repo.id_column == "metric_id"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_default_excludes_synthetic():
    """Default (real mode) applies the .eq('is_synthetic', False) provenance predicate."""
    captured: list[tuple[str, object]] = []

    class _RecordingTable(_FakePagedTable):
        def eq(self, col: str, val):
            captured.append((col, val))
            return super().eq(col, val)

    class _RecordingClient(_FakeClient):
        def table(self, _name: str):
            return _RecordingTable(self._rows)

    repo = BusinessMetricRepository(_RecordingClient(_slice_rows(1)))
    await repo.get_by_region_paged(region="south", brand="Remibrutinib", page_size=2)

    assert ("is_synthetic", False) in captured


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_opt_in_skips_provenance():
    captured: list[tuple[str, object]] = []

    class _RecordingTable(_FakePagedTable):
        def eq(self, col: str, val):
            captured.append((col, val))
            return super().eq(col, val)

    class _RecordingClient(_FakeClient):
        def table(self, _name: str):
            return _RecordingTable(self._rows)

    repo = BusinessMetricRepository(_RecordingClient(_slice_rows(1)))
    await repo.get_by_region_paged(
        region="south", brand="Remibrutinib", include_synthetic=True, page_size=2
    )

    assert ("is_synthetic", False) not in captured


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_warns_not_silent_on_bound(caplog):
    repo = BusinessMetricRepository(_FakeClient(_slice_rows(7)))

    with caplog.at_level(logging.WARNING):
        out = await repo.get_by_region_paged(
            region="south",
            brand="Remibrutinib",
            include_synthetic=True,
            page_size=2,
            max_pages=1,  # 1 page x 2 rows < 7 -> bound hit before exhaustion
        )

    assert len(out) == 2  # bounded, but...
    assert any(
        "south" in rec.getMessage() and "max_pages" in rec.getMessage() for rec in caplog.records
    ), f"expected a non-silent bound warning, got: {[r.getMessage() for r in caplog.records]}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_returns_empty_without_client():
    repo = BusinessMetricRepository(supabase_client=None)
    assert await repo.get_by_region_paged(region="south") == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_robust_to_server_row_cap_below_page_size():
    """Cap-agnostic paging: if PostgREST's db-max-rows caps a response BELOW the
    requested page_size (a DIFFERENT environment — CI's fresh DB, a future prod config),
    the fetch must STILL exhaust the slice, never stop early on the first short (capped)
    page. The loop advances by the rows actually returned and terminates only on an EMPTY
    page — so it is correct for ANY server cap, not just one >= page_size."""

    class _CappedTable(_FakePagedTable):
        SERVER_CAP = 3

        async def execute(self):
            res = await super().execute()
            res.data = res.data[: _CappedTable.SERVER_CAP]  # server returns <= cap rows
            return res

    class _CappedClient(_FakeClient):
        def table(self, _name: str):
            return _CappedTable(self._rows)

    # 7 rows, server caps EVERY response at 3, page_size requested = 10 (> cap).
    repo = BusinessMetricRepository(_CappedClient(_slice_rows(7)))

    out = await repo.get_by_region_paged(
        region="south", brand="Remibrutinib", include_synthetic=True, page_size=10
    )

    assert len(out) == 7  # all rows despite the server cap (3) being below page_size (10)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_region_paged_rejects_nonpositive_page_size():
    repo = BusinessMetricRepository(_FakeClient(_slice_rows(1)))
    with pytest.raises(ValueError):
        await repo.get_by_region_paged(region="south", page_size=0)
