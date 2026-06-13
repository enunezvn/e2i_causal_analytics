"""Unit tests for sentinel registry & dispatcher (subsystem 6)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.sentinels.registry import (
    dispatch_sentinels,
    evaluate_sentinel,
    register_sentinel,
)


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode = None
        self._filters: Dict[str, Any] = {}
        self._gt: Dict[str, Any] = {}
        self._gte: Dict[str, Any] = {}
        self._lt: Dict[str, Any] = {}
        self._lte: Dict[str, Any] = {}
        self._insert_payload: Any = None
        self._update_payload: Dict[str, Any] = {}

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        return self

    def insert(self, payload: Any) -> "_FakeQuery":
        self._mode = "insert"
        self._insert_payload = payload
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def delete(self) -> "_FakeQuery":
        self._mode = "delete"
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def gt(self, col: str, val: Any) -> "_FakeQuery":
        self._gt[col] = val
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        self._gte[col] = val
        return self

    def lt(self, col: str, val: Any) -> "_FakeQuery":
        self._lt[col] = val
        return self

    def lte(self, col: str, val: Any) -> "_FakeQuery":
        self._lte[col] = val
        return self

    def neq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = ("!=", val)
        return self

    def is_(self, col: str, val: str) -> "_FakeQuery":
        return self

    def order(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def limit(self, n: int) -> "_FakeQuery":
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        self._filters[col] = ("in", vals)
        return self

    def _match(self) -> List[Dict[str, Any]]:
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            if isinstance(want, tuple) and want[0] == "!=":
                rows = [r for r in rows if r.get(col) != want[1]]
            elif isinstance(want, tuple) and want[0] == "in":
                rows = [r for r in rows if r.get(col) in want[1]]
            else:
                rows = [r for r in rows if r.get(col) == want]
        for col, threshold in self._gt.items():
            rows = [r for r in rows if (r.get(col) or 0) > threshold]
        for col, threshold in self._gte.items():
            rows = [r for r in rows if (r.get(col) or 0) >= threshold]
        for col, threshold in self._lt.items():
            rows = [r for r in rows if (r.get(col) or 0) < threshold]
        for col, threshold in self._lte.items():
            rows = [r for r in rows if (r.get(col) or 0) <= threshold]
        return rows

    def execute(self) -> MagicMock:
        if self._mode == "insert":
            payload = self._insert_payload
            rows_to_insert = payload if isinstance(payload, list) else [payload]
            inserted = []
            for r in rows_to_insert:
                row = dict(r)
                if self.table_name == "sentinels":
                    row["sentinel_id"] = row.get("sentinel_id") or "fake-sent-id"
                self.store.rows.setdefault(self.table_name, []).append(row)
                inserted.append(row)
            mock = MagicMock()
            mock.data = inserted
            return mock

        rows = self._match()
        if self._mode == "update":
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        break
        mock = MagicMock()
        mock.data = rows
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "sentinels": [],
            "causal_paths": [],
            "triggers": [],
            "insight_edges": [],
        }

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()


@pytest.fixture(autouse=True)
def patch_clients(fake_supabase):
    fake_redis = AsyncMock()
    fake_redis.publish = AsyncMock(return_value=1)
    with (
        patch("src.memory.sentinels.registry.get_supabase_client", return_value=fake_supabase),
        patch("src.memory.lifecycle.invalidator.get_supabase_client", return_value=fake_supabase),
        patch("src.memory.lifecycle.invalidator.get_redis_client", return_value=fake_redis),
    ):
        yield fake_supabase


@pytest.mark.asyncio
async def test_register_rejects_empty_brand():
    with pytest.raises(ValueError):
        await register_sentinel(
            name="bad",
            pattern_type="threshold_breach",
            pattern_config={
                "table": "causal_paths",
                "column": "causal_effect_size",
                "op": "<",
                "value": 0.05,
            },
            action_type="invalidate",
            action_config={"source_type": "causal_path"},
            brand="",
        )


@pytest.mark.asyncio
async def test_register_rejects_unknown_pattern():
    with pytest.raises(ValueError):
        await register_sentinel(
            name="bad",
            pattern_type="completely_made_up",
            pattern_config={},
            action_type="notify",
            action_config={},
            brand="Kisqali",
        )


@pytest.mark.asyncio
async def test_register_persists(fake_supabase: FakeSupabase):
    sid = await register_sentinel(
        name="kisqali-effect-floor",
        pattern_type="threshold_breach",
        pattern_config={
            "table": "causal_paths",
            "column": "causal_effect_size",
            "op": "<",
            "value": 0.05,
        },
        action_type="invalidate",
        action_config={"source_type": "causal_path"},
        brand="Kisqali",
    )
    assert sid == "fake-sent-id"
    assert fake_supabase.rows["sentinels"][0]["brand"] == "Kisqali"
    assert fake_supabase.rows["sentinels"][0]["enabled"] is True


@pytest.mark.asyncio
async def test_threshold_breach_returns_matches(fake_supabase: FakeSupabase):
    fake_supabase.rows["causal_paths"].extend(
        [
            # is_synthetic is NOT NULL DEFAULT false on the live table (#894:
            # the evaluators now default-exclude synthetic rows)
            {
                "path_id": "p1",
                "brand": "Kisqali",
                "causal_effect_size": 0.02,
                "is_synthetic": False,
            },
            {
                "path_id": "p2",
                "brand": "Kisqali",
                "causal_effect_size": 0.10,
                "is_synthetic": False,
            },
            {
                "path_id": "p3",
                "brand": "Fabhalta",
                "causal_effect_size": 0.01,
                "is_synthetic": False,
            },
        ]
    )
    sentinel = {
        "pattern_type": "threshold_breach",
        "pattern_config": {
            "table": "causal_paths",
            "column": "causal_effect_size",
            "op": "<",
            "value": 0.05,
        },
        "brand": "Kisqali",
    }
    matches = await evaluate_sentinel(sentinel)
    # Only Kisqali p1 matches; Fabhalta p3 must NOT appear despite < 0.05.
    assert {m["row_id"] for m in matches} == {"p1"}


@pytest.mark.asyncio
async def test_dispatcher_fires_and_invalidates_only_in_brand(fake_supabase: FakeSupabase):
    """Two sentinels (Kisqali + Fabhalta), one matching row each.
    After dispatch, only the Kisqali downstream artifact is invalidated."""
    fake_supabase.rows["sentinels"].extend(
        [
            {
                "sentinel_id": "s-k",
                "name": "k-floor",
                "pattern_type": "threshold_breach",
                "pattern_config": {
                    "table": "causal_paths",
                    "column": "causal_effect_size",
                    "op": "<",
                    "value": 0.05,
                },
                "action_type": "invalidate",
                "action_config": {"source_type": "causal_path"},
                "brand": "Kisqali",
                "enabled": True,
                "fire_count": 0,
            },
            {
                "sentinel_id": "s-f",
                "name": "f-floor",
                "pattern_type": "threshold_breach",
                "pattern_config": {
                    "table": "causal_paths",
                    "column": "causal_effect_size",
                    "op": "<",
                    "value": 0.05,
                },
                "action_type": "invalidate",
                "action_config": {"source_type": "causal_path"},
                "brand": "Fabhalta",
                "enabled": True,
                "fire_count": 0,
            },
        ]
    )
    fake_supabase.rows["causal_paths"].extend(
        [
            {
                "path_id": "cp-k",
                "brand": "Kisqali",
                "causal_effect_size": 0.01,
                "is_synthetic": False,
            },
        ]
    )
    # A Fabhalta trigger that COULD be invalidated if brand were sloppy.
    fake_supabase.rows["insight_edges"].extend(
        [
            {
                "source_type": "causal_path",
                "source_id": "cp-k",
                "target_type": "trigger",
                "target_id": "tr-k",
                "brand": "Kisqali",
            },
            {
                "source_type": "causal_path",
                "source_id": "cp-k",
                "target_type": "trigger",
                "target_id": "tr-f",
                "brand": "Fabhalta",  # cross-brand edge -- must be skipped
            },
        ]
    )
    fake_supabase.rows["triggers"].extend(
        [
            {"trigger_id": "tr-k", "invalidated_at": None},
            {"trigger_id": "tr-f", "invalidated_at": None},
        ]
    )

    result = await dispatch_sentinels()
    assert result.examined == 2
    # Only Kisqali sentinel had a match; Fabhalta sentinel found nothing.
    assert result.fired == 1

    tr_k = next(t for t in fake_supabase.rows["triggers"] if t["trigger_id"] == "tr-k")
    tr_f = next(t for t in fake_supabase.rows["triggers"] if t["trigger_id"] == "tr-f")
    assert tr_k["invalidated_at"] is not None
    assert tr_f["invalidated_at"] is None  # cross-brand blocked

    # The Kisqali sentinel got its fire_count bumped.
    s_k = next(s for s in fake_supabase.rows["sentinels"] if s["sentinel_id"] == "s-k")
    assert s_k["fire_count"] == 1
