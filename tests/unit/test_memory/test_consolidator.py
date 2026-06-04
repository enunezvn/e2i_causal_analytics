"""Unit tests for Consolidator (subsystem 2)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.memory.lifecycle.consolidator import Consolidator


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._select_cols: Optional[str] = None
        self._select_count_mode: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._is_null_cols: List[str] = []
        self._gte: Dict[str, Any] = {}
        self._update_payload: Dict[str, Any] = {}
        self._mode = None  # 'select' | 'update'
        self._range: Optional[tuple] = None  # (start, end) inclusive, PostgREST-style

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        self._select_cols = cols
        self._select_count_mode = count
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        self._gte[col] = val
        return self

    def is_(self, col: str, val: str) -> "_FakeQuery":
        if val == "null":
            self._is_null_cols.append(col)
        return self

    def range(self, start: int, end: int) -> "_FakeQuery":
        self._range = (start, end)
        return self

    def _match(self) -> List[Dict[str, Any]]:
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            rows = [r for r in rows if r.get(col) == want]
        for col, threshold in self._gte.items():
            rows = [r for r in rows if (r.get(col) or 0) >= threshold]
        for col in self._is_null_cols:
            rows = [r for r in rows if r.get(col) is None]
        return rows

    def execute(self) -> MagicMock:
        rows = self._match()
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        if self._mode == "update":
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        break
        mock = MagicMock()
        mock.data = rows
        mock.count = len(rows) if self._select_count_mode == "exact" else None
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "causal_paths": [],
            "episodic_memories": [],
            "procedural_memories": [],
        }

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()


@pytest.fixture(autouse=True)
def patch_client(fake_supabase):
    with patch("src.memory.lifecycle.consolidator.get_supabase_client", return_value=fake_supabase):
        yield


@pytest.mark.asyncio
async def test_promote_to_semantic_requires_min_confirmations(fake_supabase: FakeSupabase):
    fake_supabase.rows["causal_paths"].append(
        {
            "path_id": "cp1",
            "brand": "Kisqali",
            "validation_status": "confirmed",
            "confirmation_count": 1,
            "consolidated_at": None,
        }
    )
    # Only 2 episodic memories -- below the 3 default threshold.
    fake_supabase.rows["episodic_memories"].extend(
        [{"memory_id": "m1", "causal_path_id": "cp1"}, {"memory_id": "m2", "causal_path_id": "cp1"}]
    )
    result = await Consolidator().run()
    assert result.promoted_to_semantic == 0
    assert fake_supabase.rows["causal_paths"][0]["consolidated_at"] is None


@pytest.mark.asyncio
async def test_promote_to_semantic_succeeds_when_threshold_reached(fake_supabase: FakeSupabase):
    fake_supabase.rows["causal_paths"].append(
        {
            "path_id": "cp1",
            "brand": "Kisqali",
            "validation_status": "confirmed",
            "confirmation_count": 1,
            "consolidated_at": None,
        }
    )
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {"memory_id": f"m{i}", "causal_path_id": "cp1"}
        )
    result = await Consolidator().run()
    assert result.promoted_to_semantic == 1
    assert fake_supabase.rows["causal_paths"][0]["consolidated_at"] is not None
    assert fake_supabase.rows["causal_paths"][0]["confirmation_count"] == 3


@pytest.mark.asyncio
async def test_overturned_paths_never_promoted(fake_supabase: FakeSupabase):
    fake_supabase.rows["causal_paths"].append(
        {
            "path_id": "cp1",
            "brand": "Kisqali",
            "validation_status": "overturned",
            "confirmation_count": 10,
            "consolidated_at": None,
        }
    )
    for i in range(5):
        fake_supabase.rows["episodic_memories"].append(
            {"memory_id": f"m{i}", "causal_path_id": "cp1"}
        )
    result = await Consolidator().run()
    assert result.promoted_to_semantic == 0
    assert fake_supabase.rows["causal_paths"][0]["consolidated_at"] is None


@pytest.mark.asyncio
async def test_brand_scoped_run_isolates_brands(fake_supabase: FakeSupabase):
    """Calling Consolidator.run(brand=X) must only touch X-brand causal_paths."""
    fake_supabase.rows["causal_paths"].extend(
        [
            {
                "path_id": "cp1",
                "brand": "Kisqali",
                "validation_status": "confirmed",
                "confirmation_count": 5,
                "consolidated_at": None,
            },
            {
                "path_id": "cp2",
                "brand": "Fabhalta",
                "validation_status": "confirmed",
                "confirmation_count": 5,
                "consolidated_at": None,
            },
        ]
    )
    for cp in ("cp1", "cp2"):
        for i in range(3):
            fake_supabase.rows["episodic_memories"].append(
                {"memory_id": f"{cp}-m{i}", "causal_path_id": cp}
            )
    result = await Consolidator().run(brand="Kisqali")
    assert result.promoted_to_semantic == 1
    by_brand = result.by_brand
    assert "Kisqali" in by_brand and by_brand["Kisqali"]["semantic"] == 1
    # Fabhalta untouched.
    cp_fab = next(c for c in fake_supabase.rows["causal_paths"] if c["path_id"] == "cp2")
    assert cp_fab["consolidated_at"] is None


@pytest.mark.asyncio
async def test_promote_to_procedural_uses_usage_and_success_rate(fake_supabase: FakeSupabase):
    fake_supabase.rows["procedural_memories"].extend(
        [
            {
                "procedure_id": "p1",
                "procedure_name": "high-use",
                "applicable_brands": ["Kisqali"],
                "success_rate": 0.9,
                "usage_count": 12,
            },
            {
                "procedure_id": "p2",
                "procedure_name": "low-use",
                "applicable_brands": ["Kisqali"],
                "success_rate": 0.95,
                "usage_count": 2,  # below threshold
            },
        ]
    )
    result = await Consolidator().run(brand="Kisqali")
    assert result.promoted_to_procedural == 1
    p1 = next(p for p in fake_supabase.rows["procedural_memories"] if p["procedure_id"] == "p1")
    assert p1["procedure_name"].startswith("[PROC] ")
    p2 = next(p for p in fake_supabase.rows["procedural_memories"] if p["procedure_id"] == "p2")
    assert not p2["procedure_name"].startswith("[PROC] ")


@pytest.mark.asyncio
async def test_consolidator_is_idempotent(fake_supabase: FakeSupabase):
    """Running twice produces the same end-state and the second pass is a no-op."""
    fake_supabase.rows["causal_paths"].append(
        {
            "path_id": "cp1",
            "brand": "Kisqali",
            "validation_status": "confirmed",
            "confirmation_count": 1,
            "consolidated_at": None,
        }
    )
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {"memory_id": f"m{i}", "causal_path_id": "cp1"}
        )
    r1 = await Consolidator().run()
    r2 = await Consolidator().run()
    assert r1.promoted_to_semantic == 1
    # Second pass: candidates query filters consolidated_at IS NULL, so the path
    # is no longer a candidate and the count is 0.
    assert r2.promoted_to_semantic == 0
