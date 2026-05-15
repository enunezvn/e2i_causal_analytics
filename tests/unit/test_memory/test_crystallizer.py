"""Unit tests for Crystallizer (subsystem 7) — brand strictness, edge wiring."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.memory.crystallization.crystallizer import Crystallizer


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode = None
        self._filters: Dict[str, Any] = {}
        self._in_filters: Dict[str, List[Any]] = {}
        self._gte: Dict[str, Any] = {}
        self._insert_payload: Any = None

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        return self

    def insert(self, payload: Any) -> "_FakeQuery":
        self._mode = "insert"
        self._insert_payload = payload
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        self._gte[col] = val
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        self._in_filters[col] = vals
        return self

    def order(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def limit(self, n: int) -> "_FakeQuery":
        return self

    def execute(self) -> MagicMock:
        if self._mode == "insert":
            payload = self._insert_payload
            rows_to_insert: List[Dict[str, Any]]
            if isinstance(payload, list):
                rows_to_insert = payload
            else:
                rows_to_insert = [payload]
            inserted = []
            for r in rows_to_insert:
                row = dict(r)
                if self.table_name == "executive_insights":
                    row["insight_id"] = row.get("insight_id") or str(uuid.uuid4())
                self.store.rows.setdefault(self.table_name, []).append(row)
                inserted.append(row)
            mock = MagicMock()
            mock.data = inserted
            return mock
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            rows = [r for r in rows if r.get(col) == want]
        for col, allowed in self._in_filters.items():
            rows = [r for r in rows if r.get(col) in allowed]
        for col, threshold in self._gte.items():
            rows = [r for r in rows if (r.get(col) or "") >= threshold]
        mock = MagicMock()
        mock.data = rows
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "episodic_memories": [],
            "executive_insights": [],
            "insight_edges": [],
        }

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()


@pytest.fixture(autouse=True)
def patch_client(fake_supabase):
    with patch(
        "src.memory.crystallization.crystallizer.get_supabase_client", return_value=fake_supabase
    ):
        yield


def _seed_episodic(
    db: FakeSupabase,
    *,
    brand: str,
    causal_path_id: str,
    agents: List[str],
    region: Optional[str] = "northeast",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for i, agent in enumerate(agents):
        db.rows["episodic_memories"].append(
            {
                "memory_id": f"{brand}-{causal_path_id}-{i}",
                "agent_name": agent,
                "brand": brand,
                "region": region,
                "causal_path_id": causal_path_id,
                "event_type": "agent_action",
                "description": f"{agent} on {causal_path_id}",
                "outcome_type": "success",
                "occurred_at": now,
                "raw_content": {},
            }
        )


@pytest.mark.asyncio
async def test_crystallize_requires_two_distinct_agents(fake_supabase: FakeSupabase):
    _seed_episodic(fake_supabase, brand="Kisqali", causal_path_id="cp1", agents=["causal_impact"])
    result = await Crystallizer(min_agents=2).run_for_brand("Kisqali")
    assert result.insights_created == 0


@pytest.mark.asyncio
async def test_crystallize_creates_insight_and_edges(fake_supabase: FakeSupabase):
    _seed_episodic(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer", "heterogeneous_optimizer"],
    )
    result = await Crystallizer().run_for_brand("Kisqali")
    assert result.insights_created == 1
    insights = fake_supabase.rows["executive_insights"]
    assert len(insights) == 1
    assert insights[0]["brand"] == "Kisqali"

    edges = fake_supabase.rows["insight_edges"]
    # 3 source episodic memories + 1 causal_path summarizes edge.
    assert len(edges) == 4
    # All edges are brand-tagged Kisqali.
    assert {e["brand"] for e in edges} == {"Kisqali"}


@pytest.mark.asyncio
async def test_crystallize_never_co_aggregates_across_brands(fake_supabase: FakeSupabase):
    """Add Kisqali AND Fabhalta memories with overlapping cycles. The two
    crystallization runs must each produce exactly one brand-pure insight."""
    _seed_episodic(
        fake_supabase,
        brand="Kisqali",
        causal_path_id="cp1",
        agents=["causal_impact", "gap_analyzer"],
    )
    _seed_episodic(
        fake_supabase,
        brand="Fabhalta",
        causal_path_id="cp2",
        agents=["causal_impact", "gap_analyzer"],
    )
    r_k = await Crystallizer().run_for_brand("Kisqali")
    r_f = await Crystallizer().run_for_brand("Fabhalta")
    assert r_k.insights_created == 1
    assert r_f.insights_created == 1

    insights = fake_supabase.rows["executive_insights"]
    assert len(insights) == 2
    brands = {i["brand"] for i in insights}
    assert brands == {"Kisqali", "Fabhalta"}

    # Every edge must carry the same brand as its target insight.
    edges_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for e in fake_supabase.rows["insight_edges"]:
        edges_by_target.setdefault(e["target_id"], []).append(e)
    for insight in insights:
        target_edges = edges_by_target.get(insight["insight_id"], [])
        for e in target_edges:
            assert e["brand"] == insight["brand"]


@pytest.mark.asyncio
async def test_crystallize_rejects_empty_brand():
    with pytest.raises(ValueError):
        await Crystallizer().run_for_brand("")
