"""Plan §3.12 missing test cases (#375 item 5).

Plan list of 7 missing tests:

1. ``test_promotion_score_calculation``  — DROPPED. The plan's weighted
   5-factor score (replication, consistency, confidence, recency,
   freshness) depends on Phase 1 graded staleness, which is DROPPED per
   Decision 3 = KEEP BINARY (see plan §"DECISIONS ADOPTED — 2026-05-19").
   The shipped score is count-based (``SEMANTIC_MIN_CONFIRMATIONS``);
   see ``test_consolidator.test_promote_to_semantic_requires_min_confirmations``
   which already covers the count-based promotion gate.

2. ``test_merge_duplicates``             — DROPPED. Per issue #375 explicit
   out-of-scope: "Episodic dedup / merging (find_merge_candidates, Jaccard,
   CONSOLIDATED_INTO) — separate larger item; defer." Tracked separately.

3. ``test_merge_averages_confidence``    — DROPPED. Same reason as #2.

4. ``test_staleness_blocks_promotion``   — DROPPED. Per Decision 3
   = KEEP BINARY, there is no graded ``staleness_score`` to compare against
   a "block threshold" — promotion is gated on confirmation count alone.
   Equivalent shipped behaviour: an OVERTURNED causal_path never promotes
   (see ``test_consolidator.test_overturned_paths_never_promoted``).

5. ``test_data_drop_cooldown``           — IMPLEMENTED in
   ``test_sentinel_cooldown.py`` (commit ``50934e77``).

6. ``test_cohort_drift_fires``           — IMPLEMENTED below.

7. ``test_schedule_sentinel``            — IMPLEMENTED below.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
        self._gte: Dict[str, Any] = {}
        self._update_payload: Dict[str, Any] = {}
        self._insert_payload: Any = None

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def insert(self, payload: Any) -> "_FakeQuery":
        self._mode = "insert"
        self._insert_payload = payload
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def lt(self, col: str, val: Any) -> "_FakeQuery":
        self._lt[col] = val
        return self

    def gt(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        self._gte[col] = val
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
                        "sentinel_id",
                        f"fake-{len(self.store.rows[self.table_name]) + 1}",
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
        for col, threshold in self._gte.items():
            rows = [r for r in rows if (r.get(col) or 0) >= threshold]
        if self._mode == "update":
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        break
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
# test_cohort_drift_fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cohort_drift_fires(fake_supabase: FakeSupabase):
    """Plan §3.12 ``test_cohort_drift_fires`` — proxied via the shipped
    ``drift_score`` pattern.

    The shipped registry's ``drift_score`` evaluator is a placeholder
    that returns ``[]`` (no drift_monitor table yet). To exercise the
    full firing path with a deterministic match, we monkey-patch the
    underlying evaluator to return a single match — proving the
    dispatcher's plumbing (drift_score → action) is sound end-to-end.

    Once the drift_monitor's persisted alerts table lands (separate
    PR), the evaluator can drop the patch and read live data; this
    test guarantees the upstream wiring is correct in the meantime.
    """

    async def fake_eval_drift(cfg: Dict[str, Any], brand: str) -> List[Dict[str, Any]]:
        return [{"row_id": "drift-1", "brand": brand, "drift_score": 0.42}]

    fake_supabase.rows["sentinels"].append(
        {
            "sentinel_id": "s-drift",
            "name": "pluvicto-cohort-drift",
            "pattern_type": "drift_score",
            "pattern_config": {"max_drift_score": 0.30},
            "action_type": "notify",
            "action_config": {},
            "brand": "Pluvicto",
            "enabled": True,
            "fire_count": 0,
            "last_fired_at": None,
            "cooldown_minutes": 2880,
        }
    )
    with patch("src.memory.sentinels.registry._eval_drift_score", new=fake_eval_drift):
        result = await dispatch_sentinels()
    assert result.fired == 1, f"expected drift_score sentinel to fire, got {result}"
    # fire_count bumped + last_fired_at stamped
    s = fake_supabase.rows["sentinels"][0]
    assert s["fire_count"] == 1
    assert s["last_fired_at"] is not None


# ---------------------------------------------------------------------------
# test_schedule_sentinel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_sentinel(fake_supabase: FakeSupabase, tmp_path):
    """Plan §3.12 ``test_schedule_sentinel`` — schedule-driven firing
    EXERCISED THROUGH THE NEW PLAN-VOCAB CONFIG LOADER.

    This test is the END-TO-END regression anchor for the new YAML loader:
    a YAML entry with ``trigger_type: schedule`` (plan vocab) loads via
    ``config_loader.load_sentinels_from_yaml`` and persists with
    ``pattern_type='new_causal_path'`` (shipped vocab). The dispatcher
    then evaluates against the ``new_causal_path`` pattern and fires when
    a causal_path was created since ``last_fired_at``.

    Without the config_loader (i.e. before #375), there was NO way to
    register a sentinel using the plan's ``schedule`` vocab — direct
    ``register_sentinel`` calls require the shipped pattern_type. So
    this test ROOTS the loader→dispatcher contract end-to-end.
    """
    from src.memory.sentinels.config_loader import load_sentinels_from_yaml

    yaml_path = tmp_path / "schedule_sentinel.yaml"
    yaml_path.write_text(
        """\
sentinels:
  - id: sentinel_test_schedule
    name: Test schedule sentinel
    trigger_type: schedule
    condition: {}
    action: run_full_consolidation
    brands: ["all"]
    active: true
    cooldown_minutes: 0
"""
    )
    # Also patch loader's get_supabase_client (same object).
    with patch(
        "src.memory.sentinels.config_loader.get_supabase_client",
        return_value=fake_supabase,
    ):
        registered = await load_sentinels_from_yaml(yaml_path)
    assert registered == 1
    # Verify the stored row uses the SHIPPED pattern type.
    s = fake_supabase.rows["sentinels"][0]
    assert s["pattern_type"] == "new_causal_path", (
        f"loader must translate plan 'schedule' → shipped 'new_causal_path'; "
        f"got {s['pattern_type']}"
    )
    # Stamp last_fired_at to 2 days ago so the next dispatch evaluates
    # since=that-stamp.
    now = datetime.now(timezone.utc)
    s["last_fired_at"] = (now - timedelta(days=2)).isoformat()
    s["fire_count"] = 1

    # New path created 1 day ago — within the (last_fired_at, now) window.
    fake_supabase.rows["causal_paths"].append(
        {
            "path_id": "cp-new",
            "brand": "Kisqali",
            "created_at": (now - timedelta(days=1)).isoformat(),
        }
    )
    # Old path from a week ago — outside the window.
    fake_supabase.rows["causal_paths"].append(
        {
            "path_id": "cp-old",
            "brand": "Kisqali",
            "created_at": (now - timedelta(days=7)).isoformat(),
        }
    )
    result = await dispatch_sentinels()
    assert result.fired == 1, (
        f"expected schedule sentinel (loaded via YAML plan-vocab) to fire "
        f"on a new path since last_fired_at; got {result}"
    )

    # Reset to "no new paths" by bumping last_fired_at to NOW.
    s["last_fired_at"] = now.isoformat()
    s["fire_count"] = 2
    result2 = await dispatch_sentinels()
    assert result2.fired == 0, (
        f"expected schedule sentinel to NOT fire when no new paths exist "
        f"since the most-recent firing; got {result2}"
    )
