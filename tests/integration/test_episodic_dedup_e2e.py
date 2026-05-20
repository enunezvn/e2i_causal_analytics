"""E2E integration test for episodic-memory deduplication (issue #388).

PR for issue #388 ships:

* Migration ``database/memory/026_episodic_dedup.sql`` — adds
  ``dedup_signature TEXT`` + ``dedup_counter INT DEFAULT 1`` columns and
  a partial-unique-index ``uix_episodic_memories_dedup_signature`` on
  ``(COALESCE(brand, ''), dedup_signature) WHERE dedup_signature IS NOT NULL``.
* ``src/memory/lifecycle/consolidator.py::deduplicate_episodic`` —
  wired into ``Consolidator.run()`` before semantic promotion.

Unit tests at ``tests/unit/test_memory/test_consolidator_dedup.py`` pin
the dedup-collapse semantics with a ``FakeSupabase`` shim. Those tests
exercise the Python logic but neither the migration NOR the DB-level
unique-index enforcement.

This e2e test plugs that gap:

1. Bootstraps a minimal ``episodic_memories`` table (the columns the
   dedup migration touches — sliced from
   ``008_agentic_memory_schema.sql:111-216`` so this test has no
   upstream dependency on the full memory schema lineage that 008
   creates).
2. Applies migration 026 to the test DB so the new columns + index
   exist (idempotent — re-applies cleanly).
3. Seeds rows directly via psycopg.
4. Verifies the partial-unique-index enforces the brand+dedup_signature
   contract — a second INSERT with the same (brand, dedup_signature)
   raises ``UniqueViolation``.
5. Verifies the index permits multiple rows with NULL dedup_signature
   (un-examined rows pass through).
6. Verifies brand-boundary isolation at the DB level — same
   dedup_signature under different brand values does NOT collide.

Skip-gate: requires ``TEST_POSTGRES_URL`` env (matching the pattern in
``test_021_insight_lifecycle_migration.py`` +
``test_portfolio_summary_e2e.py``). Skips cleanly on dev laptops / CI
environments without a test DB provisioned.

Run manually pre-PR::

    set -a && source .env.test && set +a
    pytest tests/integration/test_episodic_dedup_e2e.py \\
        -v -m integration --no-header

Forced-isolation: every test uses a unique-suffix brand prefix +
deletes rows for that prefix pre/post so tests do not collide on a
shared dev DB. Same trade-off as test_portfolio_summary_e2e.py: parallel
xdist runs against the same DB will need ``--dist no``.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Iterator

import pytest

psycopg = pytest.importorskip("psycopg")

# ----------------------------------------------------------------------------
# Skip-gate
# ----------------------------------------------------------------------------

_TEST_DB_URL = os.environ.get("TEST_POSTGRES_URL")


def _have_psql() -> bool:
    return subprocess.run(["which", "psql"], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _TEST_DB_URL or not _have_psql(),
        reason="needs TEST_POSTGRES_URL env + local psql to exercise the real DB",
    ),
]

# ----------------------------------------------------------------------------
# Migration apply (episodic_memories slice + 026 dedup)
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MIGRATION_026 = REPO_ROOT / "database" / "memory" / "026_episodic_dedup.sql"

# Minimal bootstrap of the ``episodic_memories`` table. The full 008
# migration also creates pgvector + enum types + several other tables
# we do NOT need for this test's scope; this slice creates ONLY the
# columns migration 026 touches plus the bare minimum for INSERT to
# succeed. The pgvector embedding column is omitted (008 has it but the
# dedup feature doesn't touch embeddings).
_BOOTSTRAP_EPISODIC_MEMORIES_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    cycle_id UUID,
    event_type TEXT NOT NULL,
    event_subtype VARCHAR(100),
    description TEXT NOT NULL,
    raw_content JSONB DEFAULT '{}',
    causal_path_id VARCHAR(50),
    agent_name TEXT,
    brand VARCHAR(50),
    region VARCHAR(20),
    occurred_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    importance_score FLOAT DEFAULT 0.5
);
"""


def _exec_sql_via_psql(url: str, sql: str, *, label: str) -> None:
    """Pipe a SQL string to psql. Used for the bootstrap step."""
    result = subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1"],
        input=sql,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{label} failed:\nstderr={result.stderr}\nstdout={result.stdout}"
    )


def _apply_migration(url: str, migration_path: Path) -> None:
    """Apply a migration via psql. Idempotent migrations re-apply OK."""
    result = subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1", "-f", str(migration_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Migration apply ({migration_path.name}) failed:\n"
        f"stderr={result.stderr}\nstdout={result.stdout}"
    )


@pytest.fixture(scope="module")
def db_with_migrations() -> str:
    """Bootstrap ``episodic_memories`` (sliced from 008) then apply 026.

    Module-scoped so the bootstrap + migration apply runs once per test
    file (both are idempotent so safe but expensive to repeat).
    """
    assert _TEST_DB_URL is not None  # pytestmark guarantees
    _exec_sql_via_psql(
        _TEST_DB_URL,
        _BOOTSTRAP_EPISODIC_MEMORIES_SQL,
        label="episodic_memories bootstrap",
    )
    _apply_migration(_TEST_DB_URL, MIGRATION_026)
    # Apply again to prove idempotency.
    _apply_migration(_TEST_DB_URL, MIGRATION_026)
    return _TEST_DB_URL


# ----------------------------------------------------------------------------
# Per-test isolation: unique brand prefix + pre/post DELETE
# ----------------------------------------------------------------------------


@pytest.fixture()
def db_conn(db_with_migrations: str) -> Iterator["psycopg.Connection"]:
    """Open a psycopg connection per test. Closed on teardown."""
    conn = psycopg.connect(db_with_migrations, connect_timeout=10)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def brand_namespace(db_conn: "psycopg.Connection") -> Iterator[str]:
    """A unique brand prefix used to scope rows for this test.
    Pre-cleans + post-cleans rows matching the prefix so the test does
    not see leakage from previous runs or leak its own rows onto
    subsequent runs.
    """
    prefix = f"e2e-dedup-{uuid.uuid4().hex[:8]}-"
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM episodic_memories WHERE brand LIKE %s",
            (prefix + "%",),
        )
    db_conn.commit()
    yield prefix
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM episodic_memories WHERE brand LIKE %s",
            (prefix + "%",),
        )
    db_conn.commit()


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_migration_026_adds_dedup_columns(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """After migration 026 the dedup_signature + dedup_counter columns
    exist on episodic_memories and dedup_counter defaults to 1."""
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, column_default, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'episodic_memories'
              AND column_name IN ('dedup_signature', 'dedup_counter')
            ORDER BY column_name
            """
        )
        rows = cur.fetchall()
    col_by_name = {r[0]: r for r in rows}
    assert "dedup_signature" in col_by_name, "dedup_signature column missing"
    assert "dedup_counter" in col_by_name, "dedup_counter column missing"
    # dedup_counter is NOT NULL DEFAULT 1.
    _, _, default, nullable = col_by_name["dedup_counter"]
    assert nullable == "NO", "dedup_counter should be NOT NULL"
    assert "1" in (default or ""), f"dedup_counter default should be 1, got {default!r}"


def test_partial_unique_index_blocks_same_brand_same_signature(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """The partial-unique-index on (COALESCE(brand,''), dedup_signature)
    WHERE dedup_signature IS NOT NULL must raise UniqueViolation for a
    second INSERT with the same (brand, signature)."""
    brand = brand_namespace + "kis"
    sig = "v1:primary:" + ("a" * 64)
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO episodic_memories
                (event_type, description, brand, dedup_signature)
            VALUES (%s, %s, %s, %s)
            """,
            ("ANALYSIS_COMPLETED", "first", brand, sig),
        )
    db_conn.commit()

    # Second INSERT with the same (brand, signature) must raise.
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodic_memories
                    (event_type, description, brand, dedup_signature)
                VALUES (%s, %s, %s, %s)
                """,
                ("ANALYSIS_COMPLETED", "second", brand, sig),
            )
    db_conn.rollback()

    # Verify only one row survived.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM episodic_memories WHERE brand = %s",
            (brand,),
        )
        count = cur.fetchone()[0]
    assert count == 1


def test_partial_unique_index_permits_null_signature_duplicates(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """Rows with dedup_signature IS NULL are NOT subject to the
    constraint — the partial-unique-index filter ``WHERE dedup_signature
    IS NOT NULL`` excludes them. Pre-dedup rows must therefore be able
    to land freely until the consolidator stamps signatures."""
    brand = brand_namespace + "fab"
    with db_conn.cursor() as cur:
        for desc in ("first", "second", "third"):
            cur.execute(
                """
                INSERT INTO episodic_memories
                    (event_type, description, brand, dedup_signature)
                VALUES (%s, %s, %s, NULL)
                """,
                ("ANALYSIS_COMPLETED", desc, brand),
            )
    db_conn.commit()
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM episodic_memories WHERE brand = %s",
            (brand,),
        )
        count = cur.fetchone()[0]
    assert count == 3, "NULL-signature rows must NOT collide on the partial index"


def test_partial_unique_index_respects_brand_boundary(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """Same dedup_signature under DIFFERENT brand values must NOT
    collide. Brand is the leading column of the index, so distinct
    brands produce distinct index entries."""
    sig = "v1:primary:" + ("b" * 64)
    brand_a = brand_namespace + "kis"
    brand_b = brand_namespace + "fab"
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO episodic_memories
                (event_type, description, brand, dedup_signature)
            VALUES (%s, %s, %s, %s)
            """,
            ("ANALYSIS_COMPLETED", "kis-row", brand_a, sig),
        )
        cur.execute(
            """
            INSERT INTO episodic_memories
                (event_type, description, brand, dedup_signature)
            VALUES (%s, %s, %s, %s)
            """,
            ("ANALYSIS_COMPLETED", "fab-row", brand_b, sig),
        )
    db_conn.commit()
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM episodic_memories WHERE brand LIKE %s",
            (brand_namespace + "%",),
        )
        count = cur.fetchone()[0]
    assert count == 2, "Different brands with same signature must both land"


def test_dedup_counter_check_constraint_blocks_zero(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """CHECK constraint dedup_counter >= 1 must reject dedup_counter = 0."""
    brand = brand_namespace + "rem"
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodic_memories
                    (event_type, description, brand, dedup_counter)
                VALUES (%s, %s, %s, %s)
                """,
                ("ANALYSIS_COMPLETED", "zero-counter", brand, 0),
            )
    db_conn.rollback()


def test_consolidator_deduplicate_episodic_against_real_db(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """End-to-end: insert 3 identical-key rows directly, then run
    ``Consolidator.deduplicate_episodic`` against a psycopg-backed
    shim, assert 1 row remains with dedup_counter == 3.

    This is the integration-test counterpart to the unit test
    ``test_dedup_collapses_identical_episodic_rows`` — exercises the
    REAL DB-level partial-unique-index + REAL SQL semantics on the
    consolidator code path."""
    import asyncio
    from typing import Any, Dict, List, Optional, Tuple

    brand = brand_namespace + "kis"

    # Seed 3 identical-key rows.
    with db_conn.cursor() as cur:
        for i in range(3):
            cur.execute(
                """
                INSERT INTO episodic_memories
                    (event_type, event_subtype, description, brand,
                     causal_path_id, agent_name, occurred_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW() + (%s || ' seconds')::interval)
                """,
                (
                    "ANALYSIS_COMPLETED",
                    "ate_estimation",
                    f"row {i}",
                    brand,
                    "cp-merged-388",
                    "estimator",
                    str(i),
                ),
            )
    db_conn.commit()

    # Build a tiny psycopg-backed supabase shim that the consolidator can use.
    class _Shim:
        def __init__(self, conn: "psycopg.Connection", table: str) -> None:
            self._conn = conn
            self._table = table
            self._mode: Optional[str] = None
            self._select_cols = "*"
            self._eq_filters: List[Tuple[str, Any]] = []
            self._is_null_cols: List[str] = []
            self._update_payload: Dict[str, Any] = {}
            self._in_filter: Optional[Tuple[str, List[Any]]] = None

        def select(self, cols: str, count: Optional[str] = None) -> "_Shim":
            self._mode = "select"
            self._select_cols = cols
            return self

        def update(self, payload: Dict[str, Any]) -> "_Shim":
            self._mode = "update"
            self._update_payload = payload
            return self

        def delete(self) -> "_Shim":
            self._mode = "delete"
            return self

        def eq(self, col: str, val: Any) -> "_Shim":
            self._eq_filters.append((col, val))
            return self

        def is_(self, col: str, val: Any) -> "_Shim":
            self._is_null_cols.append(col)
            return self

        def in_(self, col: str, vals: List[Any]) -> "_Shim":
            self._in_filter = (col, list(vals))
            return self

        def execute(self) -> Any:
            where_parts: List[str] = []
            params: List[Any] = []
            for col, val in self._eq_filters:
                where_parts.append(f"{col} = %s")
                params.append(val)
            for col in self._is_null_cols:
                where_parts.append(f"{col} IS NULL")
            if self._in_filter is not None:
                col, vals = self._in_filter
                placeholders = ",".join(["%s"] * len(vals))
                where_parts.append(f"{col} IN ({placeholders})")
                params.extend(vals)
            where_sql = " WHERE " + " AND ".join(where_parts) if where_parts else ""

            class _Result:
                def __init__(self, data: List[Dict[str, Any]]) -> None:
                    self.data = data
                    self.count: Optional[int] = None

            if self._mode == "select":
                sql = f"SELECT {self._select_cols} FROM {self._table}{where_sql}"
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                    col_names = [d[0] for d in cur.description] if cur.description else []
                    rows = [dict(zip(col_names, r, strict=True)) for r in cur.fetchall()]
                # memory_id values come back as UUID; coerce to str for the
                # consolidator which round-trips them through .eq().
                for r in rows:
                    if "memory_id" in r and r["memory_id"] is not None:
                        r["memory_id"] = str(r["memory_id"])
                    if "occurred_at" in r and r["occurred_at"] is not None:
                        r["occurred_at"] = str(r["occurred_at"])
                return _Result(rows)
            if self._mode == "update":
                set_parts = ", ".join(f"{k} = %s" for k in self._update_payload.keys())
                set_params = list(self._update_payload.values())
                sql = f"UPDATE {self._table} SET {set_parts}{where_sql}"
                with self._conn.cursor() as cur:
                    cur.execute(sql, set_params + params)
                self._conn.commit()
                return _Result([])
            if self._mode == "delete":
                sql = f"DELETE FROM {self._table}{where_sql}"
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                self._conn.commit()
                return _Result([])
            return _Result([])

    class _Client:
        def __init__(self, conn: "psycopg.Connection") -> None:
            self._conn = conn

        def table(self, name: str) -> "_Shim":
            return _Shim(self._conn, name)

    # Patch the factory + run dedup.
    from unittest.mock import patch

    from src.memory.lifecycle.consolidator import Consolidator

    fake = _Client(db_conn)
    with patch(
        "src.memory.lifecycle.consolidator.get_supabase_client",
        return_value=fake,
    ):
        consolidator = Consolidator()
        # Explicit-loop pattern: tests/integration/ forbids bare
        # ``asyncio.run`` per the project-wide CI guard at
        # ``tests/integration/test_no_bare_asyncio_run_in_integration_tests.py``
        # (RAGAS nest_asyncio pollution chain — see GH #220 / #218 / #215).
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(consolidator.deduplicate_episodic(brand=brand, region=None))
        finally:
            loop.close()

    # Verify the DB state.
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT memory_id, dedup_signature, dedup_counter
            FROM episodic_memories
            WHERE brand = %s
            """,
            (brand,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1, f"expected 1 canonical row, got {len(rows)}"
    sig, counter = rows[0][1], rows[0][2]
    assert sig is not None and sig.startswith("v1:primary:")
    assert counter == 3


def test_consolidator_merges_late_arrival_against_real_db(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """Iter-1 H1 e2e: after a canonical is stamped, a late-arrival row
    with the same key fields MUST be merged into the existing canonical
    via the application's pre-check path — NOT stamped as a duplicate-
    canonical (which would hit the DB partial-unique-index UniqueViolation).

    This is the integration-test counterpart to the unit test
    ``test_dedup_collapses_late_inserted_duplicate_after_canonical_stamped``.
    Exercises the REAL partial-unique-index against the real Postgres
    table — proves that the application path successfully avoids the
    DB-level conflict by pre-checking and merging.
    """
    import asyncio
    from typing import Any, Dict, List, Optional, Tuple
    from unittest.mock import patch

    from src.memory.lifecycle.consolidator import Consolidator

    brand = brand_namespace + "kis-late"

    # Phase 1: seed + run dedup on 3 identical rows.
    with db_conn.cursor() as cur:
        for i in range(3):
            cur.execute(
                """
                INSERT INTO episodic_memories
                    (event_type, event_subtype, description, brand,
                     causal_path_id, agent_name, occurred_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW() + (%s || ' seconds')::interval)
                """,
                (
                    "ANALYSIS_COMPLETED",
                    "ate_estimation",
                    f"row {i}",
                    brand,
                    "cp-late-388",
                    "estimator",
                    str(i),
                ),
            )
    db_conn.commit()

    # Reuse the shim from the previous test — defined inline to keep the
    # tests self-contained (each test owns its own shim instance so
    # there's no cross-test state).
    class _Shim:
        def __init__(self, conn: "psycopg.Connection", table: str) -> None:
            self._conn = conn
            self._table = table
            self._mode: Optional[str] = None
            self._select_cols = "*"
            self._eq_filters: List[Tuple[str, Any]] = []
            self._is_null_cols: List[str] = []
            self._update_payload: Dict[str, Any] = {}
            self._in_filter: Optional[Tuple[str, List[Any]]] = None

        def select(self, cols: str, count: Optional[str] = None) -> "_Shim":
            self._mode = "select"
            self._select_cols = cols
            return self

        def update(self, payload: Dict[str, Any]) -> "_Shim":
            self._mode = "update"
            self._update_payload = payload
            return self

        def delete(self) -> "_Shim":
            self._mode = "delete"
            return self

        def eq(self, col: str, val: Any) -> "_Shim":
            self._eq_filters.append((col, val))
            return self

        def is_(self, col: str, val: Any) -> "_Shim":
            self._is_null_cols.append(col)
            return self

        def in_(self, col: str, vals: List[Any]) -> "_Shim":
            self._in_filter = (col, list(vals))
            return self

        def execute(self) -> Any:
            where_parts: List[str] = []
            params: List[Any] = []
            for col, val in self._eq_filters:
                where_parts.append(f"{col} = %s")
                params.append(val)
            for col in self._is_null_cols:
                where_parts.append(f"{col} IS NULL")
            if self._in_filter is not None:
                col, vals = self._in_filter
                placeholders = ",".join(["%s"] * len(vals))
                where_parts.append(f"{col} IN ({placeholders})")
                params.extend(vals)
            where_sql = " WHERE " + " AND ".join(where_parts) if where_parts else ""

            class _Result:
                def __init__(self, data: List[Dict[str, Any]]) -> None:
                    self.data = data
                    self.count: Optional[int] = None

            if self._mode == "select":
                sql = f"SELECT {self._select_cols} FROM {self._table}{where_sql}"
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                    col_names = [d[0] for d in cur.description] if cur.description else []
                    rows = [dict(zip(col_names, r, strict=True)) for r in cur.fetchall()]
                for r in rows:
                    if "memory_id" in r and r["memory_id"] is not None:
                        r["memory_id"] = str(r["memory_id"])
                    if "occurred_at" in r and r["occurred_at"] is not None:
                        r["occurred_at"] = str(r["occurred_at"])
                return _Result(rows)
            if self._mode == "update":
                set_parts = ", ".join(f"{k} = %s" for k in self._update_payload.keys())
                set_params = list(self._update_payload.values())
                sql = f"UPDATE {self._table} SET {set_parts}{where_sql}"
                with self._conn.cursor() as cur:
                    cur.execute(sql, set_params + params)
                self._conn.commit()
                return _Result([])
            if self._mode == "delete":
                sql = f"DELETE FROM {self._table}{where_sql}"
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                self._conn.commit()
                return _Result([])
            return _Result([])

    class _Client:
        def __init__(self, conn: "psycopg.Connection") -> None:
            self._conn = conn

        def table(self, name: str) -> "_Shim":
            return _Shim(self._conn, name)

    fake = _Client(db_conn)

    with patch(
        "src.memory.lifecycle.consolidator.get_supabase_client",
        return_value=fake,
    ):
        consolidator = Consolidator()
        # Explicit-loop pattern per integration-test asyncio.run guard.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(consolidator.deduplicate_episodic(brand=brand, region=None))
        finally:
            loop.close()

    # Phase-1 sanity: canonical stamped with counter=3.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT dedup_signature, dedup_counter FROM episodic_memories WHERE brand = %s",
            (brand,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] is not None
    assert rows[0][1] == 3
    canonical_sig_phase1 = rows[0][0]

    # Phase 2: insert a 4th identical row simulating a late arrival.
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO episodic_memories
                (event_type, event_subtype, description, brand,
                 causal_path_id, agent_name, occurred_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW() + INTERVAL '5 seconds')
            """,
            (
                "ANALYSIS_COMPLETED",
                "ate_estimation",
                "late row",
                brand,
                "cp-late-388",
                "estimator",
            ),
        )
    db_conn.commit()

    # Phase 3: re-run dedup. Must merge late row into existing canonical.
    with patch(
        "src.memory.lifecycle.consolidator.get_supabase_client",
        return_value=fake,
    ):
        consolidator = Consolidator()
        # Explicit-loop pattern per integration-test asyncio.run guard.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(consolidator.deduplicate_episodic(brand=brand, region=None))
        finally:
            loop.close()

    # Phase-2 verification: still 1 row, counter incremented to 4.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT dedup_signature, dedup_counter FROM episodic_memories WHERE brand = %s",
            (brand,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1, f"expected merge into existing canonical, got {len(rows)} rows"
    assert rows[0][0] == canonical_sig_phase1
    assert rows[0][1] == 4
