"""Unit tests for cascade_invalidate (subsystem 3) — brand-isolation enforced on every hop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.lifecycle.invalidator import cascade_invalidate


class _FakeQuery:
    """Chainable fake matching the supabase-py builder we use."""

    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode = None  # 'select' | 'update' | 'delete'
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
                # Find the original index and update in-place to keep references.
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
    """In-memory fake of the subset of supabase-py used by cascade_invalidate."""

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
    """Patch both Supabase and Redis clients used by the invalidator."""
    fake_redis = AsyncMock()
    fake_redis.publish = AsyncMock(return_value=1)
    with (
        patch("src.memory.lifecycle.invalidator.get_supabase_client", return_value=fake_supabase),
        patch("src.memory.lifecycle.invalidator.get_redis_client", return_value=fake_redis),
    ):
        yield fake_redis


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _seed_kisqali_chain(db: FakeSupabase) -> None:
    """A typical chain: causal_path -> trigger -> ml_prediction -> executive_insight."""
    db.rows["insight_edges"].extend(
        [
            {
                "source_type": "causal_path",
                "source_id": "cp1",
                "target_type": "trigger",
                "target_id": "tr1",
                "brand": "Kisqali",
            },
            {
                "source_type": "trigger",
                "source_id": "tr1",
                "target_type": "ml_prediction",
                "target_id": "pred1",
                "brand": "Kisqali",
            },
            {
                "source_type": "ml_prediction",
                "source_id": "pred1",
                "target_type": "executive_insight",
                "target_id": "ei1",
                "brand": "Kisqali",
            },
        ]
    )
    db.rows["triggers"].append({"trigger_id": "tr1", "invalidated_at": None})
    db.rows["ml_predictions"].append({"prediction_id": "pred1", "invalidated_at": None})
    db.rows["executive_insights"].append({"insight_id": "ei1", "invalidated_at": None})


def _seed_fabhalta_chain(db: FakeSupabase) -> None:
    db.rows["insight_edges"].append(
        {
            "source_type": "causal_path",
            "source_id": "cp2",
            "target_type": "trigger",
            "target_id": "tr2",
            "brand": "Fabhalta",
        }
    )
    db.rows["triggers"].append({"trigger_id": "tr2", "invalidated_at": None})


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_marks_downstream_invalidated(fake_supabase: FakeSupabase):
    _seed_kisqali_chain(fake_supabase)
    result = await cascade_invalidate(
        source_type="causal_path",
        source_id="cp1",
        reason="overturned by refutation",
        scope_brand="Kisqali",
        publish_signal=False,
    )
    # Trigger, ml_prediction, executive_insight all invalidated.
    assert result.invalidated_by_type.get("trigger", 0) == 1
    assert result.invalidated_by_type.get("ml_prediction", 0) == 1
    assert result.invalidated_by_type.get("executive_insight", 0) == 1

    assert fake_supabase.rows["triggers"][0]["invalidated_at"] is not None
    assert fake_supabase.rows["ml_predictions"][0]["invalidated_at"] is not None
    assert fake_supabase.rows["executive_insights"][0]["invalidated_at"] is not None


@pytest.mark.asyncio
async def test_cascade_brand_isolation_kisqali_does_not_touch_fabhalta(
    fake_supabase: FakeSupabase,
):
    """Core safety test: a Kisqali overturn never affects a Fabhalta dependent."""
    _seed_kisqali_chain(fake_supabase)
    _seed_fabhalta_chain(fake_supabase)
    # Also add a malicious cross-brand-looking edge (Fabhalta-tagged) hanging
    # off Kisqali's causal_path. The cascade must refuse to traverse it.
    fake_supabase.rows["insight_edges"].append(
        {
            "source_type": "causal_path",
            "source_id": "cp1",
            "target_type": "trigger",
            "target_id": "tr2",  # Fabhalta trigger
            "brand": "Fabhalta",  # not 'Kisqali' or 'all'
        }
    )

    result = await cascade_invalidate(
        source_type="causal_path",
        source_id="cp1",
        reason="Kisqali overturn",
        scope_brand="Kisqali",
        publish_signal=False,
    )

    # Kisqali chain invalidated.
    kisq_trigger = next(t for t in fake_supabase.rows["triggers"] if t["trigger_id"] == "tr1")
    assert kisq_trigger["invalidated_at"] is not None
    # Fabhalta trigger NOT touched, even though an edge pointed at it.
    fab_trigger = next(t for t in fake_supabase.rows["triggers"] if t["trigger_id"] == "tr2")
    assert fab_trigger["invalidated_at"] is None
    # And the cascade recorded that it skipped a cross-brand edge.
    assert result.skipped_cross_brand >= 1


@pytest.mark.asyncio
async def test_cascade_traverses_brand_all_edges(fake_supabase: FakeSupabase):
    """Edges authored with brand='all' are traversed regardless of scope_brand."""
    fake_supabase.rows["insight_edges"].append(
        {
            "source_type": "causal_path",
            "source_id": "cpX",
            "target_type": "trigger",
            "target_id": "trX",
            "brand": "all",
        }
    )
    fake_supabase.rows["triggers"].append({"trigger_id": "trX", "invalidated_at": None})

    result = await cascade_invalidate(
        source_type="causal_path",
        source_id="cpX",
        reason="cross-brand finding",
        scope_brand="Kisqali",  # but the edge is 'all' so it traverses
        publish_signal=False,
    )
    assert result.invalidated_by_type.get("trigger", 0) == 1


@pytest.mark.asyncio
async def test_cascade_publishes_brand_scoped_signal(fake_supabase: FakeSupabase, patch_clients):
    _seed_kisqali_chain(fake_supabase)
    fake_redis = patch_clients
    await cascade_invalidate(
        source_type="causal_path",
        source_id="cp1",
        reason="r",
        scope_brand="Kisqali",
        publish_signal=True,
    )
    fake_redis.publish.assert_awaited()
    # The channel must be brand-namespaced.
    channel = fake_redis.publish.await_args.args[0]
    assert channel == "invalidation:e2i:Kisqali"


@pytest.mark.asyncio
async def test_cascade_already_invalidated_target_not_double_counted(
    fake_supabase: FakeSupabase,
):
    """L8 (#694): an already-invalidated target (UPDATE matches no rows due to
    the ``is_('invalidated_at', 'null')`` guard) must NOT increment the metric.

    record_hit() was previously called unconditionally, over-counting
    invalidated_by_type for targets that were already invalidated.
    """
    fake_supabase.rows["insight_edges"].append(
        {
            "source_type": "causal_path",
            "source_id": "cp1",
            "target_type": "trigger",
            "target_id": "tr1",
            "brand": "Kisqali",
        }
    )
    # Target trigger is ALREADY invalidated → UPDATE will match zero rows.
    fake_supabase.rows["triggers"].append(
        {"trigger_id": "tr1", "invalidated_at": "2026-01-01T00:00:00+00:00"}
    )

    result = await cascade_invalidate(
        source_type="causal_path",
        source_id="cp1",
        reason="overturned again",
        scope_brand="Kisqali",
        publish_signal=False,
    )

    assert result.invalidated_by_type.get("trigger", 0) == 0, (
        "already-invalidated target must not be counted as a fresh invalidation"
    )


@pytest.mark.asyncio
async def test_cascade_rejects_empty_brand():
    with pytest.raises(ValueError):
        await cascade_invalidate(
            source_type="causal_path",
            source_id="cp1",
            reason="r",
            scope_brand="",
            publish_signal=False,
        )
