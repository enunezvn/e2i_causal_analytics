"""Tests for codex iter-0 M3 closure: cascade propagation depth
off-by-one fix.

The old code emitted ``depth=1`` for a cascade with NO downstream
edges (the BFS counter incremented after the root-only sweep). The
iter-1 fix emits ``max(0, depth - 1)`` — "hops past source" semantics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.lifecycle.invalidator import cascade_invalidate
from src.mlops import lifecycle_monitoring as lm

# ---------------------------------------------------------------------
# FakeSupabase — borrowed from tests/unit/test_memory/test_invalidator.py
# ---------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode = None
        self._select_cols: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._is_null_cols: List[str] = []
        self._update_payload: Dict[str, Any] = {}

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        self._select_cols = cols
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def is_(self, col: str, val: str) -> "_FakeQuery":
        if val == "null":
            self._is_null_cols.append(col)
        return self

    def execute(self) -> MagicMock:
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            rows = [r for r in rows if r.get(col) == want]
        for col in self._is_null_cols:
            rows = [r for r in rows if r.get(col) is None]

        if self._mode == "update":
            updated_rows = []
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        updated_rows.append(orig)
                        break
            mock = MagicMock()
            mock.data = updated_rows
            return mock

        mock = MagicMock()
        mock.data = rows
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "insight_edges": [],
            "triggers": [],
            "ml_predictions": [],
            "executive_insights": [],
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
        patch("src.memory.lifecycle.invalidator.get_supabase_client", return_value=fake_supabase),
        patch("src.memory.lifecycle.invalidator.get_redis_client", return_value=fake_redis),
    ):
        yield fake_redis


@pytest.fixture
def captured_traces(monkeypatch: pytest.MonkeyPatch):
    """Capture Opik trace payloads so we can assert the depth field."""
    captured: list[tuple[str, dict]] = []

    def _record(span_name: str, payload: dict) -> None:
        captured.append((span_name, dict(payload)))

    monkeypatch.setattr(lm, "_emit_opik_trace", _record)
    return captured


@pytest.fixture
def captured_metrics(monkeypatch: pytest.MonkeyPatch):
    captured: list[tuple[str, float, dict]] = []

    def _emit(metric_name: str, value: float, tags: dict | None = None) -> None:
        captured.append((metric_name, float(value), dict(tags or {})))

    monkeypatch.setattr(lm, "_emit_mlflow_metric", _emit)
    return captured


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_with_no_downstream_emits_propagation_depth_zero(
    fake_supabase, captured_traces, captured_metrics
) -> None:
    """A cascade rooted at a node with NO downstream edges MUST emit
    propagation_depth=0 (hops past source = 0), not the loop counter
    value of 1.

    Codex iter-0 M3 closure: the old code emitted depth=1 for this
    case because the BFS counter incremented after the root-only
    sweep.
    """
    # No edges seeded — cascade has nothing downstream to propagate to.
    await cascade_invalidate(
        source_type="causal_path",
        source_id="cp-leaf",
        reason="test",
        scope_brand="kisqali",
    )

    cascade_trace = next(t for t in captured_traces if t[0] == "e2i.staleness.cascade")
    assert cascade_trace[1]["depth"] == 0, (
        "cascade with no downstream edges must report propagation_depth=0 "
        "(hops past source), not 1 (loop-counter)"
    )

    depth_metric = next(m for m in captured_metrics if m[0] == "e2i.cascade.propagation_depth")
    assert depth_metric[1] == 0.0


@pytest.mark.asyncio
async def test_cascade_with_one_downstream_emits_propagation_depth_one(
    fake_supabase, captured_traces, captured_metrics
) -> None:
    """A cascade with exactly one downstream edge MUST emit
    propagation_depth=1 (one hop past source)."""
    fake_supabase.rows["insight_edges"].append(
        {
            "source_type": "causal_path",
            "source_id": "cp1",
            "target_type": "trigger",
            "target_id": "tr1",
            "brand": "kisqali",
        }
    )
    fake_supabase.rows["triggers"].append(
        {"trigger_id": "tr1", "brand": "kisqali", "invalidated_at": None}
    )

    await cascade_invalidate(
        source_type="causal_path",
        source_id="cp1",
        reason="test",
        scope_brand="kisqali",
    )

    cascade_trace = next(t for t in captured_traces if t[0] == "e2i.staleness.cascade")
    assert cascade_trace[1]["depth"] == 1, (
        "cascade with one downstream edge must report propagation_depth=1"
    )
    depth_metric = next(m for m in captured_metrics if m[0] == "e2i.cascade.propagation_depth")
    assert depth_metric[1] == 1.0


@pytest.mark.asyncio
async def test_cascade_emits_edges_visited_count(
    fake_supabase, captured_traces, captured_metrics
) -> None:
    """edges_visited counts every edge inspected including ones
    skipped by brand filter — that's the I/O-cost observable for
    dashboard panels."""
    fake_supabase.rows["insight_edges"].extend(
        [
            {
                "source_type": "causal_path",
                "source_id": "cp1",
                "target_type": "trigger",
                "target_id": "tr1",
                "brand": "kisqali",
            },
            {
                "source_type": "causal_path",
                "source_id": "cp1",
                "target_type": "trigger",
                "target_id": "tr2",
                "brand": "fabhalta",  # cross-brand — will be skipped
            },
        ]
    )
    fake_supabase.rows["triggers"].append(
        {"trigger_id": "tr1", "brand": "kisqali", "invalidated_at": None}
    )

    await cascade_invalidate(
        source_type="causal_path",
        source_id="cp1",
        reason="test",
        scope_brand="kisqali",
    )

    cascade_trace = next(t for t in captured_traces if t[0] == "e2i.staleness.cascade")
    # Both edges INSPECTED, one cross-brand-skipped.
    assert cascade_trace[1]["edges_visited"] == 2
