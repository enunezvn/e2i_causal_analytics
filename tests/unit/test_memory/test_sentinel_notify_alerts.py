"""Unit test for sentinel ``notify`` → ``e2i:alerts`` wiring (#375 item 3).

The shipped ``notify`` action was logs-only. Plan §3.8 specs that all four
plan-actions publish to the Redis ``e2i:alerts`` channel for CopilotKit
real-time delivery. We extend the registry's existing ``notify`` action with
the same publication side-effect (logs-only stays as audit trail).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.sentinels.registry import dispatch_sentinels


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._lt: Dict[str, Any] = {}
        self._update_payload: Dict[str, Any] = {}

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def insert(self, payload: Any) -> "_FakeQuery":
        self._mode = "insert"
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def lt(self, col: str, val: Any) -> "_FakeQuery":
        self._lt[col] = val
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def gt(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def lte(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def neq(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def is_(self, col: str, val: str) -> "_FakeQuery":
        return self

    def order(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def limit(self, n: int) -> "_FakeQuery":
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        return self

    def execute(self) -> MagicMock:
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            # Model the live `is_synthetic NOT NULL DEFAULT false` column: a
            # planted row without the key is a REAL row (#894 sentinel reads
            # default-exclude synthetic via .eq('is_synthetic', False)).
            default = False if col == "is_synthetic" else None
            rows = [r for r in rows if r.get(col, default) == want]
        for col, threshold in self._lt.items():
            rows = [r for r in rows if (r.get(col) or 0) < threshold]
        if self._mode == "update":
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        break
        mock = MagicMock()
        mock.data = rows
        mock.count = None
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


@pytest.fixture
def fake_redis():
    r = AsyncMock()
    r.publish = AsyncMock(return_value=1)
    return r


@pytest.mark.asyncio
async def test_notify_action_publishes_to_e2i_alerts(fake_supabase: FakeSupabase, fake_redis):
    """A sentinel with action_type='notify' MUST publish to e2i:alerts."""
    fake_supabase.rows["sentinels"].append(
        {
            "sentinel_id": "s-notify",
            "name": "kisqali-floor-notify",
            "pattern_type": "threshold_breach",
            "pattern_config": {
                "table": "causal_paths",
                "column": "causal_effect_size",
                "op": "<",
                "value": 0.05,
            },
            "action_type": "notify",
            "action_config": {"channel": "ops"},
            "brand": "Kisqali",
            "enabled": True,
            "fire_count": 0,
            "last_fired_at": None,
            "cooldown_minutes": None,
        }
    )
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-k", "brand": "Kisqali", "causal_effect_size": 0.01}
    )
    with (
        patch("src.memory.sentinels.registry.get_supabase_client", return_value=fake_supabase),
        patch("src.memory.services.factories.get_redis_client", return_value=fake_redis),
    ):
        result = await dispatch_sentinels()
    assert result.fired == 1
    # At least one publish call to e2i:alerts.
    publish_calls = fake_redis.publish.await_args_list
    assert any(call.args and call.args[0] == "e2i:alerts" for call in publish_calls), (
        f"expected publish on 'e2i:alerts'; actual calls: {publish_calls}"
    )
