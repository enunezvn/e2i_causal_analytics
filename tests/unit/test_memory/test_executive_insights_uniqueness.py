"""Duplicate-prevention contract for the crystallizer.

Two crystallizer runs for the same (brand, region, kpi, causal_path) within
the active (not-invalidated) window MUST NOT produce two
``executive_insights`` rows.

Backed by the partial-unique-index ``uix_executive_insights_active_causal_path``
added in migration 021. The crystallizer catches the unique-violation
exception from supabase-py and returns ``("", 0)`` as a skip-signal that
``run_for_brand`` observes via ``if not insight_id: continue``.

The test uses an in-memory FakeDB that enforces the unique constraint on
insert, mirroring Postgres semantics. This way we test the crystallizer's
exception handling and skip-signal without needing a live database.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.memory.crystallization.crystallizer import Crystallizer


class _UniqueViolationFakeDB:
    """Fake supabase-style client that enforces the partial-unique-index.

    Mirrors migration 021's ``uix_executive_insights_active_causal_path``:
    inserting a row with the same ``(brand, region, kpi, key_metrics->>'causal_path_id')``
    as an existing row whose ``invalidated_at IS NULL`` raises a UniqueViolation-
    flavoured exception. The crystallizer's exception handler matches on
    "unique" / "duplicate" / "uix_executive_insights" in the message.
    """

    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "executive_insights": [],
            "insight_edges": [],
            "episodic_memories": [],
        }

    def table(self, name: str) -> "_UniqueViolationQuery":
        return _UniqueViolationQuery(self, name)


class _UniqueViolationQuery:
    def __init__(self, store: _UniqueViolationFakeDB, table: str) -> None:
        self.store = store
        self.table_name = table
        self.mode: str | None = None
        self.filters: Dict[str, Any] = {}
        self.gte_filters: Dict[str, Any] = {}
        self.in_filters: Dict[str, List[Any]] = {}
        self.payload: Any = None

    def select(self, *_a: Any, **_kw: Any) -> "_UniqueViolationQuery":
        self.mode = "select"
        return self

    def insert(self, payload: Any) -> "_UniqueViolationQuery":
        self.mode = "insert"
        self.payload = payload
        return self

    def eq(self, c: str, v: Any) -> "_UniqueViolationQuery":
        self.filters[c] = v
        return self

    def gte(self, c: str, v: Any) -> "_UniqueViolationQuery":
        self.gte_filters[c] = v
        return self

    def in_(self, c: str, vs: List[Any]) -> "_UniqueViolationQuery":
        self.in_filters[c] = vs
        return self

    def order(self, *_a: Any, **_kw: Any) -> "_UniqueViolationQuery":
        # L7 (#694): crystallizer candidate SELECT now orders + pages; no-op here.
        return self

    def range(self, *_a: Any, **_kw: Any) -> "_UniqueViolationQuery":
        return self

    def execute(self) -> Any:
        if self.mode == "insert":
            rows = self.payload if isinstance(self.payload, list) else [self.payload]
            inserted = []
            for r in rows:
                if self.table_name == "executive_insights":
                    # Enforce partial-unique on (brand, region, kpi,
                    # key_metrics->>'causal_path_id') WHERE invalidated_at IS NULL
                    causal_path = (r.get("key_metrics") or {}).get("causal_path_id")
                    for existing in self.store.rows["executive_insights"]:
                        if (
                            existing.get("invalidated_at") is None
                            and existing.get("brand") == r.get("brand")
                            and existing.get("region") == r.get("region")
                            and existing.get("kpi") == r.get("kpi")
                            and (existing.get("key_metrics") or {}).get("causal_path_id")
                            == causal_path
                        ):
                            raise RuntimeError(
                                "duplicate key value violates unique constraint "
                                f'"uix_executive_insights_active_causal_path" '
                                f"(brand={r.get('brand')}, region={r.get('region')}, "
                                f"kpi={r.get('kpi')}, causal_path={causal_path})"
                            )
                row = dict(r)
                row.setdefault("insight_id", f"ei-{len(self.store.rows[self.table_name]) + 1}")
                row.setdefault("invalidated_at", None)
                self.store.rows.setdefault(self.table_name, []).append(row)
                inserted.append(row)
            m = MagicMock()
            m.data = inserted
            return m

        # select
        rows = list(self.store.rows.get(self.table_name, []))
        for c, want in self.filters.items():
            rows = [r for r in rows if r.get(c) == want]
        for c, t in self.gte_filters.items():
            rows = [r for r in rows if (r.get(c) or "") >= t]
        for c, vs in self.in_filters.items():
            rows = [r for r in rows if r.get(c) in vs]
        m = MagicMock()
        m.data = rows
        return m


@pytest.fixture
def fake_db() -> _UniqueViolationFakeDB:
    return _UniqueViolationFakeDB()


@pytest.fixture(autouse=True)
def patch_supabase(fake_db: _UniqueViolationFakeDB):
    with patch(
        "src.memory.crystallization.crystallizer.get_supabase_client",
        return_value=fake_db,
    ):
        yield


def _seed_episodic_memories_for_group(
    db: _UniqueViolationFakeDB,
    *,
    brand: str,
    region: str,
    causal_path_id: str,
    kpi: str = "discontinuation_rate",
) -> None:
    """Seed 2 episodic memories from distinct agents — meets min_agents=2."""
    now = datetime.now(timezone.utc).isoformat()
    for agent in ("causal_impact", "gap_analyzer"):
        db.rows["episodic_memories"].append(
            {
                "memory_id": f"em-{agent}-{causal_path_id}",
                "agent_name": agent,
                "brand": brand,
                "region": region,
                "causal_path_id": causal_path_id,
                "event_type": "agent_action",
                "description": f"{agent} on {causal_path_id}",
                "outcome_type": "success",
                "occurred_at": now,
                "raw_content": {"kpi": kpi},
            }
        )


@pytest.mark.asyncio
async def test_two_crystallizations_for_same_group_produce_one_row(
    fake_db: _UniqueViolationFakeDB,
) -> None:
    """Two runs on same (brand, region, kpi, causal_path) → exactly one row.

    First run inserts. Second run hits the partial-unique constraint;
    crystallizer catches the violation and returns the skip-signal.
    Final ``executive_insights`` count must be 1.
    """
    _seed_episodic_memories_for_group(
        fake_db, brand="Brand-X", region="northeast", causal_path_id="cp-1"
    )

    r1 = await Crystallizer().run_for_brand("Brand-X", region="northeast")
    assert r1.insights_created == 1, r1
    assert len(fake_db.rows["executive_insights"]) == 1

    r2 = await Crystallizer().run_for_brand("Brand-X", region="northeast")
    assert r2.insights_created == 0, (
        f"second run must skip (uix constraint), but created {r2.insights_created}; "
        f"errors={r2.errors}"
    )
    assert len(fake_db.rows["executive_insights"]) == 1, (
        f"executive_insights must have exactly 1 row after both runs, has "
        f"{len(fake_db.rows['executive_insights'])}"
    )


@pytest.mark.asyncio
async def test_crystallization_after_invalidation_allowed(
    fake_db: _UniqueViolationFakeDB,
) -> None:
    """Partial-index permits a NEW active row once the prior is invalidated.

    Falsifiability check: if the index were `UNIQUE` without the partial
    WHERE clause, this test would fail (the prior invalidated row would
    block the new insert).
    """
    _seed_episodic_memories_for_group(
        fake_db, brand="Brand-X", region="northeast", causal_path_id="cp-2"
    )

    r1 = await Crystallizer().run_for_brand("Brand-X", region="northeast")
    assert r1.insights_created == 1

    # Mark the first row invalidated (simulates JIT verifier finding an
    # overturned ancestor and stamping invalidated_at).
    fake_db.rows["executive_insights"][0]["invalidated_at"] = datetime.now(timezone.utc).isoformat()

    # Now a second crystallization should succeed.
    r2 = await Crystallizer().run_for_brand("Brand-X", region="northeast")
    assert r2.insights_created == 1, (
        f"recrystallize after invalidation must succeed, but got "
        f"{r2.insights_created}; errors={r2.errors}"
    )
    assert len(fake_db.rows["executive_insights"]) == 2, (
        f"executive_insights must have 2 rows after invalidate+recrystallize, "
        f"has {len(fake_db.rows['executive_insights'])}"
    )


@pytest.mark.asyncio
async def test_migration_declares_partial_unique_index() -> None:
    """Migration 021 must declare uix_executive_insights_active_causal_path.

    Pins the schema-level guard. Removing the index from the migration
    trips this test.
    """
    from pathlib import Path

    sql_path = (
        Path(__file__).parent.parent.parent.parent
        / "database"
        / "memory"
        / "021_insight_lifecycle.sql"
    )
    assert sql_path.exists(), f"migration file not found at {sql_path}"
    content = sql_path.read_text().lower()
    assert "uix_executive_insights_active_causal_path" in content, (
        "Migration 021 must declare the partial-unique-index "
        "uix_executive_insights_active_causal_path so the crystallizer "
        "dedup mechanism is structurally backed."
    )
    assert "where invalidated_at is null" in content, (
        "Index must be PARTIAL on invalidated_at IS NULL so recall + recrystallize is permitted."
    )
    # Ralph iter-0 HEDGE: pin the column expression. If the index is
    # mis-keyed (e.g. ``key_metrics->>'wrong_path_id'``), the FakeDB tests
    # still pass because they enforce the correct key independently of SQL.
    assert "key_metrics" in content, "Index expression must reference key_metrics JSONB column"
    assert "causal_path_id" in content, (
        "Index expression must extract causal_path_id from key_metrics"
    )
    # Codex iter-0 HIGH-2: COALESCE on nullable columns. Without it,
    # Postgres UNIQUE permits multiple NULLs (multiple region=NULL or
    # kpi=NULL rows would all coexist without triggering dedup).
    assert "coalesce" in content, (
        "Index must use COALESCE on nullable columns so NULL values "
        "don't bypass the dedup constraint."
    )
