"""Unit tests for sentinel cooldown enforcement (#375 item 2).

The dispatcher must skip sentinels whose ``last_fired_at`` is within
``cooldown_minutes`` of ``now``. Sentinels with ``cooldown_minutes IS NULL``
have no cooldown gate (back-compat with PR #250 ship).

bool exclusion: ``cooldown_minutes=False`` (Python int subclass) MUST be
rejected at registration, not coerced silently to 0. ``cooldown_minutes=True``
same — would coerce to 1 and could silently pass through. (Load-bearing pattern
from PR #374 / max_staleness filter.)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.sentinels.registry import dispatch_sentinels, register_sentinel


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._lt: Dict[str, Any] = {}
        self._insert_payload: Any = None
        self._update_payload: Dict[str, Any] = {}
        self._select_count_mode: Optional[str] = None

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        self._select_count_mode = count
        return self

    def insert(self, payload: Any) -> "_FakeQuery":
        self._mode = "insert"
        self._insert_payload = payload
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
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
        mock = MagicMock()
        if self._mode == "insert":
            payload = self._insert_payload
            rows_to_insert = payload if isinstance(payload, list) else [payload]
            inserted = []
            for r in rows_to_insert:
                row = dict(r)
                if self.table_name == "sentinels":
                    row.setdefault(
                        "sentinel_id", f"fake-{len(self.store.rows[self.table_name]) + 1}"
                    )
                self.store.rows.setdefault(self.table_name, []).append(row)
                inserted.append(row)
            mock.data = inserted
            mock.count = None
            return mock
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
        mock.data = rows
        mock.count = len(rows) if self._select_count_mode == "exact" else None
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
        patch("src.memory.services.factories.get_redis_client", return_value=fake_redis),
    ):
        yield fake_supabase


# ---------------------------------------------------------------------------
# register_sentinel validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_rejects_bool_cooldown_minutes_false():
    """cooldown_minutes=False would coerce silently to 0 under naive int check.

    Python: isinstance(False, int) is True. We must reject bool explicitly so
    the cooldown gate is never silently disabled by a bad caller.
    """
    with pytest.raises(ValueError, match="cooldown_minutes"):
        await register_sentinel(
            name="k-floor",
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
            cooldown_minutes=False,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_register_rejects_bool_cooldown_minutes_true():
    with pytest.raises(ValueError, match="cooldown_minutes"):
        await register_sentinel(
            name="k-floor",
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
            cooldown_minutes=True,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_register_rejects_negative_cooldown_minutes():
    with pytest.raises(ValueError, match="cooldown_minutes"):
        await register_sentinel(
            name="k-floor",
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
            cooldown_minutes=-1,
        )


@pytest.mark.asyncio
async def test_register_persists_cooldown_minutes(fake_supabase: FakeSupabase):
    await register_sentinel(
        name="k-floor",
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
        cooldown_minutes=360,
    )
    assert fake_supabase.rows["sentinels"][0]["cooldown_minutes"] == 360


# ---------------------------------------------------------------------------
# Dispatcher cooldown enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_skips_sentinel_within_cooldown(fake_supabase: FakeSupabase):
    """Sentinel with last_fired_at 1h ago and cooldown=360min must NOT fire."""
    now = datetime.now(timezone.utc)
    one_hour_ago = (now - timedelta(hours=1)).isoformat()
    fake_supabase.rows["sentinels"].append(
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
            "fire_count": 1,
            "last_fired_at": one_hour_ago,
            "cooldown_minutes": 360,  # 6h cooldown — 1h ago is INSIDE cooldown
        }
    )
    # Matching row exists; only cooldown should suppress firing.
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-k", "brand": "Kisqali", "causal_effect_size": 0.01}
    )
    result = await dispatch_sentinels()
    assert result.examined == 1
    assert result.fired == 0, "sentinel inside cooldown must not fire"
    # fire_count unchanged
    s_k = fake_supabase.rows["sentinels"][0]
    assert s_k["fire_count"] == 1


@pytest.mark.asyncio
async def test_dispatcher_fires_after_cooldown_elapses(fake_supabase: FakeSupabase):
    """Sentinel with last_fired_at 10h ago and cooldown=360min (6h) MUST fire."""
    now = datetime.now(timezone.utc)
    ten_hours_ago = (now - timedelta(hours=10)).isoformat()
    fake_supabase.rows["sentinels"].append(
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
            "fire_count": 1,
            "last_fired_at": ten_hours_ago,
            "cooldown_minutes": 360,  # 6h — outside since last fire 10h ago
        }
    )
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-k", "brand": "Kisqali", "causal_effect_size": 0.01}
    )
    result = await dispatch_sentinels()
    assert result.examined == 1
    assert result.fired == 1


@pytest.mark.asyncio
async def test_dispatcher_null_cooldown_always_evaluates(fake_supabase: FakeSupabase):
    """cooldown_minutes IS NULL → no cooldown gate; sentinel re-evaluates."""
    now = datetime.now(timezone.utc)
    one_minute_ago = (now - timedelta(minutes=1)).isoformat()
    fake_supabase.rows["sentinels"].append(
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
            "fire_count": 1,
            "last_fired_at": one_minute_ago,
            "cooldown_minutes": None,  # explicit NULL
        }
    )
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-k", "brand": "Kisqali", "causal_effect_size": 0.01}
    )
    result = await dispatch_sentinels()
    assert result.fired == 1


@pytest.mark.asyncio
async def test_dispatcher_zero_cooldown_always_evaluates(fake_supabase: FakeSupabase):
    """cooldown_minutes=0 means "no cooldown" semantically; sentinel always
    re-evaluates as long as condition holds (cosmetically the same as NULL)."""
    now = datetime.now(timezone.utc)
    one_second_ago = (now - timedelta(seconds=1)).isoformat()
    fake_supabase.rows["sentinels"].append(
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
            "fire_count": 1,
            "last_fired_at": one_second_ago,
            "cooldown_minutes": 0,
        }
    )
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-k", "brand": "Kisqali", "causal_effect_size": 0.01}
    )
    result = await dispatch_sentinels()
    assert result.fired == 1


@pytest.mark.asyncio
async def test_dispatcher_never_fired_sentinel_evaluates(fake_supabase: FakeSupabase):
    """Brand-new sentinel (last_fired_at IS NULL) MUST evaluate regardless of cooldown."""
    fake_supabase.rows["sentinels"].append(
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
            "last_fired_at": None,
            "cooldown_minutes": 360,  # cooldown set but never fired before
        }
    )
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-k", "brand": "Kisqali", "causal_effect_size": 0.01}
    )
    result = await dispatch_sentinels()
    assert result.fired == 1


# ---------------------------------------------------------------------------
# Issue #375 plan-specified missing test names
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_drop_cooldown(fake_supabase: FakeSupabase):
    """Plan §3.12 ``test_data_drop_cooldown``.

    Fire a data_drop (freshness) sentinel, then immediately re-evaluate.
    Cooldown must suppress the second fire.
    """
    # Stale trigger (older than the freshness window)
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    fake_supabase.rows["triggers"].append(
        {"trigger_id": "tr-stale", "brand": "Kisqali", "updated_at": stale_ts}
    )
    fake_supabase.rows["sentinels"].append(
        {
            "sentinel_id": "s-fresh",
            "name": "kisqali-trigger-freshness",
            "pattern_type": "freshness",
            "pattern_config": {
                "table": "triggers",
                "ts_column": "updated_at",
                "max_age_hours": 24,
            },
            "action_type": "notify",
            "action_config": {},
            "brand": "Kisqali",
            "enabled": True,
            "fire_count": 0,
            "last_fired_at": None,
            "cooldown_minutes": 720,  # 12h
        }
    )
    # First dispatch: fires.
    r1 = await dispatch_sentinels()
    assert r1.fired == 1, f"expected first-pass fire, got {r1.fired}"
    # Second dispatch immediately: cooldown blocks.
    r2 = await dispatch_sentinels()
    assert r2.fired == 0, f"expected second-pass cooldown skip, got {r2.fired}"


# ---------------------------------------------------------------------------
# M10 (#694): a sentinel whose every action FAILED must NOT enter cooldown.
# Bumping last_fired_at/fire_count unconditionally would suppress retries of a
# sentinel that never actually performed any work.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_failed_actions_do_not_enter_cooldown(fake_supabase: FakeSupabase):
    """If all of a sentinel's actions raise, last_fired_at must stay None so the
    sentinel remains eligible to retry on the next dispatch pass."""
    fake_supabase.rows["sentinels"].append(
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
            "last_fired_at": None,
            "cooldown_minutes": 360,
        }
    )
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-k", "brand": "Kisqali", "causal_effect_size": 0.01}
    )

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("invalidate failed")

    with patch("src.memory.sentinels.registry.cascade_invalidate", side_effect=_boom):
        result = await dispatch_sentinels()

    # Pattern matched, so the sentinel was "fired" in the dispatcher sense...
    assert result.fired == 1
    # ...but every action raised, so no work was done.
    assert result.actions_taken == 0
    s_k = fake_supabase.rows["sentinels"][0]
    # Cooldown must NOT have been entered: last_fired_at untouched, fire_count not bumped.
    assert s_k["last_fired_at"] is None, "failed-only sentinel must not enter cooldown"
    assert s_k["fire_count"] == 0


@pytest.mark.asyncio
async def test_dispatcher_successful_action_enters_cooldown(fake_supabase: FakeSupabase):
    """Sanity: a successful action DOES bump last_fired_at/fire_count (cooldown engaged)."""
    fake_supabase.rows["sentinels"].append(
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
            "last_fired_at": None,
            "cooldown_minutes": 360,
        }
    )
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-k", "brand": "Kisqali", "causal_effect_size": 0.01}
    )

    async def _ok(*args: Any, **kwargs: Any) -> None:
        return None

    with patch("src.memory.sentinels.registry.cascade_invalidate", side_effect=_ok):
        result = await dispatch_sentinels()

    assert result.fired == 1
    assert result.actions_taken == 1
    s_k = fake_supabase.rows["sentinels"][0]
    assert s_k["last_fired_at"] is not None, "successful sentinel must enter cooldown"
    assert s_k["fire_count"] == 1
