"""
Brand-isolation invariants across all four subsystems.

These tests are the safety net for the tenancy model in the plan: brand
is the de facto tenant boundary, and the high-severity blast-radius
risks must be structurally impossible.

Scenarios:
- A Kisqali overturn does not invalidate Fabhalta dependents (cascade)
- A Fabhalta sentinel does not fire on a Kisqali threshold breach (sentinel)
- The crystallizer never co-aggregates cross-brand episodic memories
- Empty-brand inputs are rejected at every entry point
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.coordination.signals import InsightSignalBus
from src.memory.crystallization.crystallizer import Crystallizer
from src.memory.lifecycle.invalidator import cascade_invalidate
from src.memory.sentinels.registry import register_sentinel


# Reuse the small fake supabase from the per-subsystem tests by lifting just
# what's needed here -- a fresh, minimal implementation focused on cross-
# subsystem behaviors.
class _Q:
    def __init__(self, store, table):
        self.store = store
        self.table_name = table
        self.mode = None
        self.filters: Dict[str, Any] = {}
        self.in_filters: Dict[str, List] = {}
        self.gte_filters: Dict[str, Any] = {}
        self.lt_filters: Dict[str, Any] = {}
        self.update_payload: Dict[str, Any] = {}
        self.insert_payload: Any = None
        self.is_null_cols: List[str] = []

    def select(self, *a, **kw):
        self.mode = "select"
        return self

    def insert(self, payload):
        self.mode = "insert"
        self.insert_payload = payload
        return self

    def update(self, payload):
        self.mode = "update"
        self.update_payload = payload
        return self

    def delete(self):
        self.mode = "delete"
        return self

    def eq(self, c, v):
        self.filters[c] = v
        return self

    def neq(self, c, v):
        self.filters[c] = ("!=", v)
        return self

    def gte(self, c, v):
        self.gte_filters[c] = v
        return self

    def lt(self, c, v):
        self.lt_filters[c] = v
        return self

    def is_(self, c, v):
        if v == "null":
            self.is_null_cols.append(c)
        return self

    def in_(self, c, vs):
        self.in_filters[c] = vs
        return self

    def order(self, *a, **kw):
        return self

    def range(self, *a, **kw):
        # L7 (#694): crystallizer candidate SELECT now pages via .range(); no-op
        # here (seeded data < one page, so a single page returns everything).
        return self

    def limit(self, n):
        return self

    def execute(self):
        if self.mode == "insert":
            payload = self.insert_payload
            rows = payload if isinstance(payload, list) else [payload]
            inserted = []
            for r in rows:
                row = dict(r)
                if self.table_name == "executive_insights":
                    row.setdefault("insight_id", f"ei-{len(self.store.rows[self.table_name]) + 1}")
                if self.table_name == "sentinels":
                    row.setdefault("sentinel_id", f"s-{len(self.store.rows[self.table_name]) + 1}")
                self.store.rows.setdefault(self.table_name, []).append(row)
                inserted.append(row)
            m = MagicMock()
            m.data = inserted
            return m

        rows = list(self.store.rows.get(self.table_name, []))
        for c, want in self.filters.items():
            if isinstance(want, tuple) and want[0] == "!=":
                rows = [r for r in rows if r.get(c) != want[1]]
            else:
                rows = [r for r in rows if r.get(c) == want]
        for c, vs in self.in_filters.items():
            rows = [r for r in rows if r.get(c) in vs]
        for c, t in self.gte_filters.items():
            rows = [r for r in rows if (r.get(c) or "") >= t]
        for c, t in self.lt_filters.items():
            rows = [r for r in rows if (r.get(c) or 0) < t]
        for c in self.is_null_cols:
            rows = [r for r in rows if r.get(c) is None]
        if self.mode == "update":
            for r in rows:
                for o in self.store.rows[self.table_name]:
                    if o is r:
                        o.update(self.update_payload)
                        break
        m = MagicMock()
        m.data = rows
        return m


class FakeDB:
    def __init__(self):
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "insight_edges": [],
            "triggers": [],
            "ml_predictions": [],
            "executive_insights": [],
            "causal_paths": [],
            "episodic_memories": [],
            "sentinels": [],
        }

    def table(self, name):
        return _Q(self, name)


@pytest.fixture
def db():
    return FakeDB()


@pytest.fixture(autouse=True)
def patch_all(db):
    fake_redis = AsyncMock()
    fake_redis.publish = AsyncMock(return_value=1)
    fake_redis.set = AsyncMock(return_value=True)
    fake_redis.eval = AsyncMock(return_value=1)
    fake_redis.xadd = AsyncMock(return_value="1-0")
    with (
        patch("src.memory.lifecycle.invalidator.get_supabase_client", return_value=db),
        patch("src.memory.lifecycle.invalidator.get_redis_client", return_value=fake_redis),
        patch("src.memory.crystallization.crystallizer.get_supabase_client", return_value=db),
        patch("src.memory.sentinels.registry.get_supabase_client", return_value=db),
        patch("src.memory.coordination.signals.get_redis_client", return_value=fake_redis),
    ):
        yield db, fake_redis


# ----------------------------------------------------------------------------
# Scenario 1: cascade is brand-tight
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kisqali_overturn_does_not_invalidate_fabhalta_dependents(patch_all):
    db, _ = patch_all
    db.rows["insight_edges"].extend(
        [
            {
                "source_type": "causal_path",
                "source_id": "cp",
                "target_type": "trigger",
                "target_id": "tk",
                "brand": "Kisqali",
            },
            {
                "source_type": "causal_path",
                "source_id": "cp",
                "target_type": "trigger",
                "target_id": "tf",
                "brand": "Fabhalta",
            },
        ]
    )
    db.rows["triggers"].extend(
        [
            {"trigger_id": "tk", "invalidated_at": None},
            {"trigger_id": "tf", "invalidated_at": None},
        ]
    )
    await cascade_invalidate(
        source_type="causal_path",
        source_id="cp",
        reason="overturn",
        scope_brand="Kisqali",
        publish_signal=False,
    )
    tk = next(r for r in db.rows["triggers"] if r["trigger_id"] == "tk")
    tf = next(r for r in db.rows["triggers"] if r["trigger_id"] == "tf")
    assert tk["invalidated_at"] is not None
    assert tf["invalidated_at"] is None  # cross-brand never touched


# ----------------------------------------------------------------------------
# Scenario 2: crystallizer never co-aggregates
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crystallizer_separates_two_brand_runs(patch_all):
    db, _ = patch_all
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    for brand, cp in (("Kisqali", "cpK"), ("Fabhalta", "cpF")):
        for a in ("causal_impact", "gap_analyzer"):
            db.rows["episodic_memories"].append(
                {
                    "memory_id": f"{brand}-{a}",
                    "agent_name": a,
                    "brand": brand,
                    "region": "northeast",
                    "causal_path_id": cp,
                    "event_type": "agent_action",
                    "description": f"{a} on {cp}",
                    "outcome_type": "success",
                    "occurred_at": now,
                    "raw_content": {},
                }
            )
    rk = await Crystallizer().run_for_brand("Kisqali")
    rf = await Crystallizer().run_for_brand("Fabhalta")
    assert rk.insights_created == 1
    assert rf.insights_created == 1
    insights = db.rows["executive_insights"]
    assert len(insights) == 2
    # No insight has the wrong brand.
    for ins in insights:
        for edge in db.rows["insight_edges"]:
            if edge["target_id"] == ins["insight_id"]:
                assert edge["brand"] == ins["brand"]


# ----------------------------------------------------------------------------
# Scenario 3: empty brand rejected on every entry point
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_brand_rejected_everywhere():
    bus = InsightSignalBus()
    with pytest.raises(ValueError):
        await bus.publish(topic="x", brand="", payload={})

    with pytest.raises(ValueError):
        await cascade_invalidate(
            source_type="causal_path",
            source_id="x",
            reason="r",
            scope_brand="",
            publish_signal=False,
        )

    with pytest.raises(ValueError):
        await Crystallizer().run_for_brand("")

    with pytest.raises(ValueError):
        await register_sentinel(
            name="x",
            pattern_type="threshold_breach",
            pattern_config={
                "table": "causal_paths",
                "column": "causal_effect_size",
                "op": "<",
                "value": 0.05,
            },
            action_type="notify",
            action_config={},
            brand="",
        )


# ----------------------------------------------------------------------------
# Scenario 4: signal bus is brand-keyed at the stream level
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_streams_are_brand_namespaced(patch_all):
    _, fake_redis = patch_all
    bus = InsightSignalBus()
    await bus.publish(topic="cohort:rebuilt", brand="Kisqali", payload={})
    await bus.publish(topic="cohort:rebuilt", brand="Fabhalta", payload={})

    # Stream keys should be different.
    args_list = [c.args for c in fake_redis.xadd.await_args_list]
    keys = {a[0] for a in args_list}
    assert "insights:cohort:rebuilt:Kisqali" in keys
    assert "insights:cohort:rebuilt:Fabhalta" in keys
