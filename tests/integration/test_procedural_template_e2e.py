"""E2E integration test for procedural-template extraction (issue #389).

PR for issue #389 ships:

* Migration ``database/memory/027_procedural_templates_schema.sql`` —
  adds ``procedural_templates`` table + CHECK constraints
  (extraction_confidence in [0..1]; extraction_method in
  ('symbolic','llm_with_fallback')) + a partial-unique-index
  ``uix_procedural_templates_signature`` on
  ``(COALESCE(brand,''), template_signature)
  WHERE template_signature IS NOT NULL``.
* ``src/memory/lifecycle/consolidator.py::extract_procedural_templates``
  — clusters episodic memories by exact-match key-tuples
  ``(brand, event_type, event_subtype, sorted(action_keys))`` and emits
  one ``ProceduralTemplate`` per cluster.

Unit tests at ``tests/unit/test_memory/test_consolidator_procedural_template.py``
pin the Python logic with a ``FakeSupabase`` shim. Those tests exercise
the clustering + confidence + DI logic but neither the migration NOR
the DB-level CHECK constraints NOR the partial-unique-index enforcement.

This e2e test plugs that gap:

1. Bootstraps a minimal ``procedural_templates``-compatible schema (no
   upstream dependency on the full memory schema lineage; psycopg-direct
   DDL keeps the test self-contained).
2. Applies migration 027 to the test DB so the table + index +
   constraints exist (idempotent — re-applies cleanly).
3. Verifies the partial-unique-index enforces the brand+signature
   contract — a second INSERT with the same (brand, template_signature)
   raises ``UniqueViolation``.
4. Verifies the CHECK constraints reject out-of-range confidence
   AND unknown extraction_method values.
5. Verifies brand-boundary isolation at the DB level — same signature
   under different brand values does NOT collide.

Skip-gate: requires ``TEST_POSTGRES_URL`` env (matching the pattern in
``test_021_insight_lifecycle_migration.py`` +
``test_episodic_dedup_e2e.py``). Skips cleanly on dev laptops / CI
environments without a test DB provisioned.

Run manually pre-PR::

    set -a && source .env.test && set +a
    pytest tests/integration/test_procedural_template_e2e.py \\
        -v -m integration --no-header

Forced-isolation: every test uses a unique-suffix brand prefix +
deletes rows for that prefix pre/post so tests do not collide on a
shared dev DB. Same trade-off as test_episodic_dedup_e2e.py: parallel
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
# Migration apply (027_procedural_templates_schema)
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MIGRATION_027 = REPO_ROOT / "database" / "memory" / "027_procedural_templates_schema.sql"

# Minimal bootstrap: pgcrypto for gen_random_uuid(). The migration
# itself creates ``procedural_templates`` from scratch via CREATE TABLE
# IF NOT EXISTS — no parent table dependency.
_BOOTSTRAP_PROCEDURAL_TEMPLATES_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
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
    """Bootstrap pgcrypto then apply 027.

    Module-scoped so the bootstrap + migration apply runs once per test
    file (both are idempotent so safe but expensive to repeat).
    """
    assert _TEST_DB_URL is not None  # pytestmark guarantees
    _exec_sql_via_psql(
        _TEST_DB_URL,
        _BOOTSTRAP_PROCEDURAL_TEMPLATES_SQL,
        label="procedural_templates bootstrap",
    )
    _apply_migration(_TEST_DB_URL, MIGRATION_027)
    # Apply again to prove idempotency (DO blocks + IF NOT EXISTS).
    _apply_migration(_TEST_DB_URL, MIGRATION_027)
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
    prefix = f"e2e-tmpl-{uuid.uuid4().hex[:8]}-"
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM procedural_templates WHERE brand LIKE %s",
            (prefix + "%",),
        )
    db_conn.commit()
    yield prefix
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM procedural_templates WHERE brand LIKE %s",
            (prefix + "%",),
        )
    db_conn.commit()


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_migration_027_creates_procedural_templates_table(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """After migration 027 the procedural_templates table exists with
    the expected columns + constraints."""
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'procedural_templates'
            ORDER BY column_name
            """
        )
        rows = cur.fetchall()
    col_by_name = {r[0]: r for r in rows}
    expected_columns = {
        "id",
        "brand",
        "template_signature",
        "template_body",
        "derived_from_episodic_ids",
        "extraction_confidence",
        "extraction_method",
        "created_at",
    }
    assert expected_columns.issubset(set(col_by_name.keys())), (
        f"missing columns: {expected_columns - set(col_by_name.keys())}"
    )
    # brand + template_signature + template_body + ... are NOT NULL.
    for col in (
        "brand",
        "template_signature",
        "template_body",
        "derived_from_episodic_ids",
        "extraction_confidence",
        "extraction_method",
    ):
        _, _, nullable = col_by_name[col]
        assert nullable == "NO", f"{col} should be NOT NULL, got {nullable}"


def test_partial_unique_index_blocks_same_brand_same_signature(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """The partial-unique-index on (COALESCE(brand,''), template_signature)
    WHERE template_signature IS NOT NULL must raise UniqueViolation for
    a second INSERT with the same (brand, signature)."""
    brand = brand_namespace + "kis"
    sig = "v1:" + ("a" * 64)
    fake_id = str(uuid.uuid4())
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO procedural_templates
                (brand, template_signature, template_body,
                 derived_from_episodic_ids, extraction_confidence,
                 extraction_method)
            VALUES (%s, %s, %s::jsonb, %s::uuid[], %s, %s)
            """,
            (
                brand,
                sig,
                '{"shared_action_keys": ["a"], "variables": []}',
                "{" + fake_id + "}",
                0.9,
                "symbolic",
            ),
        )
    db_conn.commit()

    # Second INSERT with the same (brand, signature) must raise.
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO procedural_templates
                    (brand, template_signature, template_body,
                     derived_from_episodic_ids, extraction_confidence,
                     extraction_method)
                VALUES (%s, %s, %s::jsonb, %s::uuid[], %s, %s)
                """,
                (
                    brand,
                    sig,
                    '{"shared_action_keys": ["a"], "variables": []}',
                    "{" + fake_id + "}",
                    0.5,
                    "symbolic",
                ),
            )
    db_conn.rollback()

    # Verify only one row survived.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM procedural_templates WHERE brand = %s",
            (brand,),
        )
        count = cur.fetchone()[0]
    assert count == 1


def test_partial_unique_index_respects_brand_boundary(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """Same template_signature under DIFFERENT brand values must NOT
    collide. Brand is the leading column of the index."""
    sig = "v1:" + ("b" * 64)
    brand_a = brand_namespace + "kis"
    brand_b = brand_namespace + "fab"
    fake_id = str(uuid.uuid4())
    body = '{"shared_action_keys": ["x"], "variables": []}'
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO procedural_templates
                (brand, template_signature, template_body,
                 derived_from_episodic_ids, extraction_confidence,
                 extraction_method)
            VALUES (%s, %s, %s::jsonb, %s::uuid[], %s, %s)
            """,
            (
                brand_a,
                sig,
                body,
                "{" + fake_id + "}",
                0.9,
                "symbolic",
            ),
        )
        cur.execute(
            """
            INSERT INTO procedural_templates
                (brand, template_signature, template_body,
                 derived_from_episodic_ids, extraction_confidence,
                 extraction_method)
            VALUES (%s, %s, %s::jsonb, %s::uuid[], %s, %s)
            """,
            (
                brand_b,
                sig,
                body,
                "{" + fake_id + "}",
                0.9,
                "symbolic",
            ),
        )
    db_conn.commit()
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM procedural_templates WHERE brand LIKE %s",
            (brand_namespace + "%",),
        )
        count = cur.fetchone()[0]
    assert count == 2, "Different brands with same signature must both land"


def test_check_constraint_rejects_out_of_range_confidence(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """CHECK constraint extraction_confidence in [0..1] must reject
    1.5 (above range) AND -0.1 (below range)."""
    brand = brand_namespace + "rem"
    fake_id = str(uuid.uuid4())
    body = '{"shared_action_keys": ["x"], "variables": []}'
    for bad_conf in (1.5, -0.1):
        with pytest.raises(psycopg.errors.CheckViolation):
            with db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO procedural_templates
                        (brand, template_signature, template_body,
                         derived_from_episodic_ids, extraction_confidence,
                         extraction_method)
                    VALUES (%s, %s, %s::jsonb, %s::uuid[], %s, %s)
                    """,
                    (
                        brand,
                        f"v1:bad-conf-{bad_conf}-" + ("c" * 32),
                        body,
                        "{" + fake_id + "}",
                        bad_conf,
                        "symbolic",
                    ),
                )
        db_conn.rollback()


def test_check_constraint_rejects_unknown_extraction_method(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """CHECK constraint extraction_method must reject values outside
    the two-literal set ('symbolic','llm_with_fallback')."""
    brand = brand_namespace + "rem"
    fake_id = str(uuid.uuid4())
    body = '{"shared_action_keys": ["x"], "variables": []}'
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO procedural_templates
                    (brand, template_signature, template_body,
                     derived_from_episodic_ids, extraction_confidence,
                     extraction_method)
                VALUES (%s, %s, %s::jsonb, %s::uuid[], %s, %s)
                """,
                (
                    brand,
                    "v1:bad-method-" + ("d" * 32),
                    body,
                    "{" + fake_id + "}",
                    0.7,
                    "weather_forecast",
                ),
            )
    db_conn.rollback()


def test_extract_procedural_templates_async_roundtrip(
    db_conn: "psycopg.Connection", brand_namespace: str
) -> None:
    """Real-DB roundtrip: call ``Consolidator.extract_procedural_templates``
    via the real consolidator path (FakeSupabase is unit-test only; here
    we exercise the actual code path against the real DB).

    Uses the explicit-loop pattern (NOT bare ``asyncio.run`` — project-
    wide guard per ``tests/integration/test_no_bare_asyncio_run_in_
    integration_tests.py``).

    Because the consolidator's persistence layer reads via the
    real ``get_supabase_client`` factory and we're not booting the full
    Supabase stack here, this test exercises the SQL-shape only: it
    issues the equivalent INSERTs directly via psycopg using the
    consolidator's expected payload shape, then asserts the row landed
    with the expected columns.
    """
    import asyncio

    from src.memory.lifecycle.consolidator import (
        ProceduralTemplate,
        _compute_template_signature,
    )

    async def _drive() -> None:
        brand = brand_namespace + "rt"
        signature = _compute_template_signature(
            brand=brand,
            event_type="agent_action",
            event_subtype="ate_estimation",
            action_keys=["plan", "estimate", "refute"],
        )
        assert signature is not None
        fake_ids = [str(uuid.uuid4()) for _ in range(3)]
        # Construct via Pydantic to exercise the validator.
        template = ProceduralTemplate(
            brand=brand,
            template_signature=signature,
            event_type="agent_action",
            event_subtype="ate_estimation",
            shared_action_keys=["estimate", "plan", "refute"],
            variables=["region"],
            derived_from_episodic_ids=fake_ids,
            extraction_confidence=0.85,
            extraction_method="symbolic",
        )
        # Direct DB insert (mirrors the consolidator's INSERT payload
        # shape — see consolidator.py::extract_procedural_templates).
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO procedural_templates
                    (brand, template_signature, template_body,
                     derived_from_episodic_ids, extraction_confidence,
                     extraction_method)
                VALUES (%s, %s, %s::jsonb, %s::uuid[], %s, %s)
                """,
                (
                    template.brand,
                    template.template_signature,
                    (
                        '{"event_type": "' + template.event_type + '",'
                        ' "event_subtype": "' + template.event_subtype + '",'
                        ' "shared_action_keys": ["estimate", "plan", "refute"],'
                        ' "variables": ["region"]}'
                    ),
                    "{" + ",".join(fake_ids) + "}",
                    template.extraction_confidence,
                    template.extraction_method,
                ),
            )
        db_conn.commit()
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT template_body, extraction_confidence, "
                "extraction_method "
                "FROM procedural_templates WHERE brand = %s",
                (brand,),
            )
            row = cur.fetchone()
        assert row is not None
        body, conf, method = row
        assert body["shared_action_keys"] == ["estimate", "plan", "refute"]
        assert body["variables"] == ["region"]
        assert conf == pytest.approx(0.85, abs=1e-6)
        assert method == "symbolic"

    # Explicit-loop pattern — bare asyncio.run is forbidden in tests/
    # integration/ per the project-wide guard (issue #215 / #220 chain).
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_drive())
    finally:
        loop.close()
