"""Migration 023 default-cooldown behaviour (#375 iter-1 M2).

Codex iter-0 M2: ``database/memory/023_sentinel_cooldown.sql`` declared
``cooldown_minutes INTEGER`` with no DEFAULT and no backfill. Effect:
existing rows AND API-created sentinels stay at ``cooldown_minutes IS NULL``
unless callers explicitly set a value, preserving the original PR #250
"always fires" behaviour by accident rather than design.

iter-1 fix: migration now declares ``DEFAULT 0`` AND issues an explicit
backfill ``UPDATE sentinels SET cooldown_minutes = 0 WHERE cooldown_minutes
IS NULL``. We chose ``0`` (explicit opt-in to a non-zero cooldown) over
``60`` (60-minute default) because:

* PR #250 shipped a "no cooldown gate" semantics. Switching to a 60-minute
  default would silently change behaviour for sentinels that were registered
  before #375.
* The dispatcher already treats ``cooldown_minutes == 0`` identically to
  ``NULL`` (see ``_is_in_cooldown``: ``not (cooldown > 0)`` returns True
  for 0). So ``0`` and ``NULL`` are semantically equivalent, and choosing
  ``0`` for the default makes the column non-null and operator-explicit
  rather than relying on tri-state semantics.
* Operators who want a cooldown gate set it explicitly (in YAML or via
  POST /api/sentinels). The new default makes the absence-of-gate
  intention loud (``cooldown_minutes=0``) rather than tacit (``NULL``).

The migration test pattern below is text-only because we don't have a
running Postgres in the unit test environment; the actual DDL behaviour
is verified by:

1. ``test_023_declares_default_zero``       — text grep verifies the DDL
   says ``DEFAULT 0``
2. ``test_023_backfills_null_to_zero``      — text grep verifies the backfill
3. ``test_023_default_is_not_60_or_other``  — pin our choice; if a future
   change wants to switch the default, this test must be updated
   explicitly (and the rationale captured in the commit body).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MIGRATION_PATH = (
    Path(__file__).resolve().parents[3] / "database" / "memory" / "023_sentinel_cooldown.sql"
)


@pytest.fixture(scope="module")
def migration_text() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8")


def test_023_declares_default_zero(migration_text: str):
    """Migration 023 must declare DEFAULT 0 for cooldown_minutes so insertions
    without an explicit value materialise as 0 (== "no cooldown gate") rather
    than NULL.
    """
    # Allow whitespace + case variation, but the column declaration must
    # contain DEFAULT 0 (and we explicitly forbid DEFAULT NULL or omitted).
    pat = re.compile(
        r"cooldown_minutes\s+INTEGER\b[^,;]*DEFAULT\s+0\b",
        re.IGNORECASE | re.DOTALL,
    )
    assert pat.search(migration_text), (
        "Migration 023 must declare `cooldown_minutes INTEGER ... DEFAULT 0`; "
        "absence means new rows insert as NULL, which the dispatcher treats "
        "as 'no cooldown' but loses the operator-explicit signal."
    )


def test_023_backfills_null_to_zero(migration_text: str):
    """Existing rows from PR #250 (pre-023) MUST be backfilled to
    ``cooldown_minutes = 0`` to make the column non-null and matched against
    the new DEFAULT.
    """
    pat = re.compile(
        r"UPDATE\s+sentinels\s+SET\s+cooldown_minutes\s*=\s*0\b"
        r".*?WHERE\s+cooldown_minutes\s+IS\s+NULL",
        re.IGNORECASE | re.DOTALL,
    )
    assert pat.search(migration_text), (
        "Migration 023 must include a backfill: "
        "`UPDATE sentinels SET cooldown_minutes = 0 WHERE cooldown_minutes IS NULL;` "
        "otherwise existing sentinels stay NULL after migration."
    )


def test_023_default_is_not_60_or_other_nonzero(migration_text: str):
    """Pin the default at 0. If a future migration wants to change this,
    that change must be explicit (this test breaks; commit body must
    document the new semantics).

    Searches for `DEFAULT <nonzero-integer>` paired with `cooldown_minutes`
    and asserts no such pairing exists.
    """
    # Find every (cooldown_minutes ... DEFAULT N) pairing.
    findings = re.findall(
        r"cooldown_minutes\s+INTEGER\b[^,;]*DEFAULT\s+(\d+)\b",
        migration_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    nonzero = [v for v in findings if int(v) != 0]
    assert not nonzero, (
        f"Migration 023 must keep DEFAULT 0; found nonzero default(s) {nonzero}. "
        "If you intend to change the default, update this test AND document the "
        "behaviour change in the migration header."
    )


# ---------------------------------------------------------------------------
# Behavioural assertion against the FakeSupabase shape — when an insert omits
# cooldown_minutes, the column default fires on row insertion.
# ---------------------------------------------------------------------------


def test_023_default_fires_when_register_sentinel_omits_cooldown_minutes(
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end behavioural test against the FakeSupabase used by the
    dispatcher tests: ``register_sentinel`` called WITHOUT
    ``cooldown_minutes`` must result in a stored row with ``cooldown_minutes
    == 0`` (DB default fires).

    The test simulates the Postgres DEFAULT fire by having FakeSupabase
    inject ``cooldown_minutes=0`` on insert when the payload omits it,
    matching the migration's behaviour.
    """
    import asyncio
    from typing import Any, Dict, List, Optional
    from unittest.mock import AsyncMock, MagicMock

    from src.memory.sentinels.registry import register_sentinel

    class _Query:
        def __init__(self, store: "FakeSb", table: str) -> None:
            self.store = store
            self.table_name = table
            self._mode: Optional[str] = None
            self._insert_payload: Any = None

        def select(self, *a: Any, **kw: Any) -> "_Query":
            self._mode = "select"
            return self

        def insert(self, payload: Any) -> "_Query":
            self._mode = "insert"
            self._insert_payload = payload
            return self

        def eq(self, *a: Any, **kw: Any) -> "_Query":
            return self

        def execute(self) -> MagicMock:
            mock = MagicMock()
            if self._mode == "insert":
                row = dict(self._insert_payload)
                # Simulate the migration DEFAULT 0 firing if the caller
                # omitted cooldown_minutes from the payload.
                row.setdefault("cooldown_minutes", 0)
                row.setdefault("sentinel_id", "fake-defaulted")
                self.store.rows.setdefault(self.table_name, []).append(row)
                mock.data = [row]
            else:
                mock.data = []
            mock.count = None
            return mock

    class FakeSb:
        def __init__(self) -> None:
            self.rows: Dict[str, List[Dict[str, Any]]] = {"sentinels": []}

        def table(self, name: str) -> _Query:
            return _Query(self, name)

    fake_supabase = FakeSb()
    monkeypatch.setattr(
        "src.memory.sentinels.registry.get_supabase_client",
        lambda: fake_supabase,
    )
    # AsyncMock for any incidental Redis use during register_sentinel (the
    # registry doesn't currently publish but the cascade_invalidate path
    # might; safer to stub).
    monkeypatch.setattr(
        "src.memory.services.factories.get_redis_client",
        lambda: AsyncMock(),
    )

    asyncio.run(
        register_sentinel(
            name="default-fires-test",
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
            # cooldown_minutes deliberately omitted — DEFAULT 0 must fire.
        )
    )

    stored = fake_supabase.rows["sentinels"][0]
    assert "cooldown_minutes" in stored, (
        "Migration 023 DEFAULT 0 must surface as a stored value; FakeSupabase "
        "simulates the DB default by `setdefault('cooldown_minutes', 0)`."
    )
    assert stored["cooldown_minutes"] == 0, (
        f"DEFAULT 0 should land 0; got {stored['cooldown_minutes']!r}"
    )
