"""Regression tests for consolidator pagination past PostgREST's row cap.

Finding H5 (memory-system-review-20260603): ``deduplicate_episodic`` and
``extract_procedural_templates`` SELECT episodic candidate rows with no
``.range()``/``.limit()``. PostgREST caps a single response at
``db-max-rows`` (default 1000) and the repo configures no override, so once a
brand's candidate set exceeds 1000 rows the server silently returns a subset.
Dedup groups / template clusters straddling the boundary are then undercounted,
corrupting ``SUM(dedup_counter)`` cluster sizing.

The stock ``FakeSupabase`` in ``test_consolidator.py`` returns every matching
row, so it cannot reproduce the truncation. These tests use a fake that
faithfully models the cap: an un-ranged ``execute()`` returns at most
``PAGE_CAP`` rows, while a ``.range(start, end)`` request returns that window.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.memory.lifecycle.consolidator import Consolidator

PAGE_CAP = 1000  # emulates PostgREST db-max-rows default


class _PagedQuery:
    def __init__(self, store: "PagedFakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._filters: Dict[str, Any] = {}
        self._is_null_cols: List[str] = []
        self._range: Optional[tuple] = None  # (start, end) inclusive, PostgREST-style

    def select(self, cols: str, count: Optional[str] = None) -> "_PagedQuery":
        return self

    def eq(self, col: str, val: Any) -> "_PagedQuery":
        self._filters[col] = val
        return self

    def is_(self, col: str, val: str) -> "_PagedQuery":
        if val == "null":
            self._is_null_cols.append(col)
        return self

    def range(self, start: int, end: int) -> "_PagedQuery":
        self._range = (start, end)
        return self

    def _match(self) -> List[Dict[str, Any]]:
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            # Model the schema default: is_synthetic is NOT NULL DEFAULT false
            # (migration 063), so a seeded row that omits it reads as False.
            if col == "is_synthetic":
                rows = [r for r in rows if r.get(col, False) == want]
            else:
                rows = [r for r in rows if r.get(col) == want]
        for col in self._is_null_cols:
            rows = [r for r in rows if r.get(col) is None]
        return rows

    def execute(self) -> MagicMock:
        rows = self._match()
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        else:
            # PostgREST silently caps an un-ranged response at db-max-rows.
            rows = rows[:PAGE_CAP]
        mock = MagicMock()
        mock.data = rows
        return mock


class PagedFakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {"episodic_memories": []}

    def table(self, name: str) -> _PagedQuery:
        return _PagedQuery(self, name)


@pytest.fixture
def paged_supabase() -> PagedFakeSupabase:
    return PagedFakeSupabase()


@pytest.fixture(autouse=True)
def patch_client(paged_supabase):
    with patch(
        "src.memory.lifecycle.consolidator.get_supabase_client",
        return_value=paged_supabase,
    ):
        yield


def test_select_all_rows_paginates_past_the_cap(paged_supabase: PagedFakeSupabase):
    """The pagination helper must walk .range() windows until exhaustion."""
    paged_supabase.rows["episodic_memories"] = [{"memory_id": f"m{i}"} for i in range(2500)]
    consolidator = Consolidator()

    fetched = consolidator._select_all_rows(
        lambda: paged_supabase.table("episodic_memories").select("memory_id")
    )

    assert len(fetched) == 2500


@pytest.mark.asyncio
async def test_deduplicate_episodic_examines_all_candidates_past_the_cap(
    paged_supabase: PagedFakeSupabase,
):
    """1500 NULL-signature candidates must all be examined, not truncated to 1000.

    Rows are intentionally minimal so ``_compute_dedup_signature`` returns None
    (they fall into the ``unkeyed`` bucket and skip group processing), isolating
    the SELECT/pagination behavior measured by ``episodic_dedup_examined``.
    """
    paged_supabase.rows["episodic_memories"] = [
        {"memory_id": f"m{i}", "brand": "Kisqali", "dedup_signature": None} for i in range(1500)
    ]

    result = await Consolidator().deduplicate_episodic(brand="Kisqali")

    assert result.episodic_dedup_examined == 1500
