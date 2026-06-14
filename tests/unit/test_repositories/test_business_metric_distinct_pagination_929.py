"""Issue #929 — ``get_distinct_values`` must page to exhaustion (no silent truncation).

The benchmark frame silently dropped segment values because ``get_distinct_values``
read a single ``.limit(5000)`` window with no ``ORDER BY``: any distinct value whose
rows fall outside that arbitrary window is never seen, so the cross-segment P75/P90
"standard" is computed over a pagination-truncated segment set (Remibrutinib's
``west`` region — 1600 rows — was omitted).

These CI-safe unit tests use a faithful in-memory table that honours ``.eq`` filters,
``.order`` and ``.range`` paging (the SAME contract supabase-py/PostgREST give), so we
exercise the REAL pagination/accumulation/termination logic rather than asserting a
mock call-chain. The fix mirrors the already-blessed dispatcher probe
(``_resolve_gap_inputs``, #874 R2): PK-ordered ``.range()`` windows paged until a short
page signals the end, bounded by ``max_pages`` with a WARN (never a silent drop).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.repositories.business_metric import BusinessMetricRepository


class _FakePagedTable:
    """In-memory PostgREST-like query honouring select/eq/order/range/execute.

    Faithfully simulates server-side range pagination: ``.range(start, end)`` returns
    the inclusive slice of the (filtered, ordered) backing rows projected to the
    selected columns. No mock-chain assertions — real paging behaviour.
    """

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
        if self._select:
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


def _rows_spanning_pages() -> list[dict]:
    """3 regions where ``west`` exists ONLY beyond the first page (metric_id order).

    With ``page_size=2`` the single-window read sees rows[0:2] = {northeast, south}
    and never reaches ``west`` — exactly the #929 silent drop, in miniature.
    """
    return [
        {"metric_id": "0001", "region": "northeast", "brand": "Remibrutinib"},
        {"metric_id": "0002", "region": "south", "brand": "Remibrutinib"},
        {"metric_id": "0003", "region": "west", "brand": "Remibrutinib"},
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_distinct_values_pages_to_exhaustion_no_silent_drop():
    """All distinct values are returned even when they span multiple pages."""
    repo = BusinessMetricRepository(_FakeClient(_rows_spanning_pages()))

    values = await repo.get_distinct_values(
        "region", brand="Remibrutinib", include_synthetic=True, page_size=2
    )

    # west lives only in the 2nd page; a single-window read would drop it.
    assert values == ["northeast", "south", "west"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_distinct_values_applies_brand_filter_across_pages():
    """The brand filter composes with pagination (other brands excluded every page)."""
    rows = _rows_spanning_pages() + [
        {"metric_id": "0004", "region": "atlantic", "brand": "Kisqali"},
        {"metric_id": "0005", "region": "pacific", "brand": "Kisqali"},
    ]
    repo = BusinessMetricRepository(_FakeClient(rows))

    values = await repo.get_distinct_values(
        "region", brand="Remibrutinib", include_synthetic=True, page_size=2
    )

    assert values == ["northeast", "south", "west"]
    assert "atlantic" not in values and "pacific" not in values


@pytest.mark.unit
@pytest.mark.asyncio
async def test_distinct_values_exact_multiple_of_page_size_terminates_cleanly():
    """A run whose total is an exact multiple of page_size still terminates and is
    complete — cap-agnostic paging (#938) ends on the trailing EMPTY page (one extra
    round trip), not on a short page."""
    rows = [
        {"metric_id": f"{i:04d}", "region": reg, "brand": "Remibrutinib"}
        for i, reg in enumerate(["northeast", "south", "midwest", "west"], start=1)
    ]
    repo = BusinessMetricRepository(_FakeClient(rows))

    values = await repo.get_distinct_values(
        "region", brand="Remibrutinib", include_synthetic=True, page_size=2
    )

    assert values == ["midwest", "northeast", "south", "west"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_distinct_values_warns_not_silent_when_page_bound_hit(caplog):
    """Hitting ``max_pages`` before exhausting the slice WARNS (truncation is never silent)."""
    repo = BusinessMetricRepository(_FakeClient(_rows_spanning_pages()))

    with caplog.at_level(logging.WARNING):
        values = await repo.get_distinct_values(
            "region",
            brand="Remibrutinib",
            include_synthetic=True,
            page_size=2,
            max_pages=1,  # 1 page x 2 rows < 3 rows -> bound hit before exhaustion
        )

    # Bounded result is still returned, but the truncation is announced.
    assert values == ["northeast", "south"]
    assert any(
        "region" in rec.getMessage()
        and "Remibrutinib" in rec.getMessage()
        and "max_pages" in rec.getMessage()
        for rec in caplog.records
    ), f"expected a non-silent bound warning, got: {[r.getMessage() for r in caplog.records]}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_distinct_values_orders_by_pk_for_deterministic_paging():
    """Paging must order by the PK so OFFSET windows never skip/duplicate rows."""

    captured: dict[str, object] = {}

    class _RecordingTable(_FakePagedTable):
        def order(self, col: str, desc: bool = False):
            captured["order_col"] = col
            return super().order(col, desc)

    class _RecordingClient(_FakeClient):
        def table(self, _name: str):
            return _RecordingTable(self._rows)

    repo = BusinessMetricRepository(_RecordingClient(_rows_spanning_pages()))
    await repo.get_distinct_values("region", include_synthetic=True, page_size=2)

    assert captured.get("order_col") == repo.id_column == "metric_id"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_distinct_values_reraises_42703_mid_scan_not_partial():
    """42703 (undefined column) is a schema-level error: it fires on the FIRST page
    (the column simply does not exist) and fails soft to []. If a 42703 ever surfaces
    AFTER a full page has succeeded, that is anomalous and must FAIL CLOSED (raise) —
    never return a silently-partial distinct set (the #845/#851 fail-OPEN family)."""
    from postgrest.exceptions import APIError

    class _MidScanClient(_FakeClient):
        def __init__(self, rows):
            super().__init__(rows)
            self._calls = 0

        def table(self, _name: str):
            client = self
            table = _FakePagedTable(self._rows)
            original = table.execute

            async def _execute():
                client._calls += 1
                if client._calls == 1:
                    return await original()  # a full first page -> loop continues
                raise APIError({"code": "42703", "message": "mid-scan"})

            table.execute = _execute  # type: ignore[method-assign]
            return table

    # Two rows == page_size -> page 0 is "full" -> a second page is fetched -> 42703.
    rows = [
        {"metric_id": "0001", "region": "northeast", "brand": "Remibrutinib"},
        {"metric_id": "0002", "region": "south", "brand": "Remibrutinib"},
    ]
    repo = BusinessMetricRepository(_MidScanClient(rows))

    with pytest.raises(APIError):
        await repo.get_distinct_values(
            "region", brand="Remibrutinib", include_synthetic=True, page_size=2
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_distinct_values_robust_to_server_row_cap_below_page_size():
    """Cap-agnostic paging (#938): if PostgREST caps a response BELOW page_size (a
    DIFFERENT environment — CI's fresh DB, a future prod config), distinct discovery must
    STILL exhaust the slice, never stop early on the first short (capped) page. The loop
    must advance by the rows ACTUALLY returned and terminate ONLY on an EMPTY page, so it
    is correct for ANY server cap — not just one >= page_size."""

    class _CappedTable(_FakePagedTable):
        SERVER_CAP = 1

        async def execute(self):
            res = await super().execute()
            res.data = res.data[: _CappedTable.SERVER_CAP]  # server returns <= cap rows
            return res

    class _CappedClient(_FakeClient):
        def table(self, _name: str):
            return _CappedTable(self._rows)

    # 3 distinct regions; server caps EVERY response at 1 row; page_size requested = 10.
    repo = BusinessMetricRepository(_CappedClient(_rows_spanning_pages()))

    values = await repo.get_distinct_values(
        "region", brand="Remibrutinib", include_synthetic=True, page_size=10
    )

    assert values == ["northeast", "south", "west"]  # all 3 despite cap(1) < page_size(10)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_distinct_values_rejects_nonpositive_page_size():
    repo = BusinessMetricRepository(_FakeClient(_rows_spanning_pages()))
    with pytest.raises(ValueError):
        await repo.get_distinct_values("region", page_size=0)
