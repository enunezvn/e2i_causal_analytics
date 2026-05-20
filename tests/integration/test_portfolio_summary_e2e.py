"""E2E integration test for ``GET /api/executive-insights/portfolio-summary``
against a REAL Postgres DB (issue #376).

PR #384 (merged 2026-05-20 at ``c634de78``) shipped:

* Migration ``database/memory/025_crystaldigest_schema_completion.sql``
  — adds 15 ADD COLUMN IF NOT EXISTS to ``executive_insights`` plus
  two DO-block CHECK constraints (effect_direction enum,
  consolidation_tier enum).
* ``src/api/routes/executive_insights.py:198-270`` — the
  ``/portfolio-summary`` endpoint that aggregates by brand from the
  same table.

Unit tests at ``tests/unit/test_api/test_executive_insights.py`` use a
``_FakeSupabase`` that implements ``.table().select().eq().execute()``
in-memory. They pin the AGGREGATION LOGIC but exercise neither the
migration NOR the real query semantics (NULL handling, TEXT[] coercion,
the index from the migration).

This e2e test plugs that gap:

1. Bootstraps a minimal ``executive_insights`` table (the 13-column
   shape from ``021_insight_lifecycle.sql`` lines 180-199 — sliced
   out of the full 021 migration so this test has no upstream-table
   dependency on ``causal_paths`` / ``episodic_memories`` / etc that
   021 also ALTERs).
2. Applies migration 025 to the test DB so the 15 new columns exist
   (idempotent — re-applies cleanly on a DB that already has the
   migration).
3. Seeds rows into ``executive_insights`` via psycopg, with ALL 15
   new fields populated for at least one row.
4. Hits ``GET /api/executive-insights/portfolio-summary`` against the
   real FastAPI app with the supabase factory shimmed to a psycopg-
   backed adapter pointed at the SAME DB connection (NOT a mock —
   the shim issues real SQL against the real DB).
5. Hits ``GET /api/executive-insights/{insight_id}`` via the same
   real-DB path and confirms all 15 new fields round-trip through
   the Pydantic response model.
6. Pins the route-ordering contract: ``/portfolio-summary`` resolves
   to 200, NOT to 404-from-uuid-parse-fail (which would happen if
   the dynamic ``/{insight_id}`` route came first and tried to
   parse ``"portfolio-summary"`` as a UUID).

Skip-gate: requires ``TEST_POSTGRES_URL`` env (matching the established
pattern in ``test_021_insight_lifecycle_migration.py`` +
``test_audit_sidecar_supabase_mirror.py``). Skips cleanly on dev
laptops / CI environments without a test DB provisioned.

Run manually pre-PR::

    set -a && source .env.test && set +a  # provides TEST_POSTGRES_URL
    pytest tests/integration/test_portfolio_summary_e2e.py \\
        -v -m integration --no-header

Forced-isolation: every test uses a unique brand prefix and deletes
rows for that prefix pre/post so tests don't disturb each other on a
shared dev DB. The trade-off is the same as
``test_audit_sidecar_supabase_mirror.py``: parallel xdist runs against
the same DB will collide; gate with ``--dist no`` if running in
parallel.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

psycopg = pytest.importorskip("psycopg")

# ----------------------------------------------------------------------------
# Skip-gate
# ----------------------------------------------------------------------------

_TEST_DB_URL = os.environ.get("TEST_POSTGRES_URL")


def _have_psql() -> bool:
    return subprocess.run(["which", "psql"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(
    not _TEST_DB_URL or not _have_psql(),
    reason="needs TEST_POSTGRES_URL env + local psql to exercise the real DB",
)

# ----------------------------------------------------------------------------
# Migration apply (executive_insights base + 025 schema completion)
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MIGRATION_025 = REPO_ROOT / "database" / "memory" / "025_crystaldigest_schema_completion.sql"

# Minimal bootstrap of the ``executive_insights`` table. The full 021
# migration ALTERs upstream tables (causal_paths, episodic_memories,
# triggers, ml_predictions) we do NOT need for this test's scope; the
# portfolio-summary + get-one routes only touch executive_insights.
# Sliced from 021_insight_lifecycle.sql lines 180-199 (CREATE TABLE +
# the brand/recall/crystallized_at index). pgcrypto is enabled so
# ``gen_random_uuid()`` resolves on a fresh DB.
_BOOTSTRAP_EXECUTIVE_INSIGHTS_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS executive_insights (
    insight_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title                   VARCHAR(500) NOT NULL,
    narrative               TEXT NOT NULL,
    brand                   VARCHAR(50) NOT NULL,
    region                  VARCHAR(20),
    kpi                     VARCHAR(100),
    time_window_start       TIMESTAMPTZ,
    time_window_end         TIMESTAMPTZ,
    key_metrics             JSONB NOT NULL DEFAULT '{}',
    recall                  BOOLEAN NOT NULL DEFAULT FALSE,
    recall_reason           TEXT,
    recall_at               TIMESTAMPTZ,
    crystallized_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    crystallized_by_cycle_id UUID,
    crystallized_by_user_id VARCHAR(100),
    invalidated_at          TIMESTAMPTZ,
    invalidation_reason     TEXT,
    source_count            INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_executive_insights_brand_recall
    ON executive_insights(brand, recall, crystallized_at DESC);
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
    """Bootstrap ``executive_insights`` (sliced from 021) then apply 025.

    Module-scoped so the bootstrap + migration apply runs once per test
    file (both are idempotent so this is safe but expensive to repeat).
    """
    assert _TEST_DB_URL is not None  # pytestmark guarantees
    _exec_sql_via_psql(
        _TEST_DB_URL,
        _BOOTSTRAP_EXECUTIVE_INSIGHTS_SQL,
        label="executive_insights bootstrap",
    )
    _apply_migration(_TEST_DB_URL, MIGRATION_025)
    return _TEST_DB_URL


# ----------------------------------------------------------------------------
# Per-test isolation: unique brand prefix + pre/post DELETE
# ----------------------------------------------------------------------------


@pytest.fixture()
def db_conn(db_with_migrations: str) -> Iterator[psycopg.Connection]:
    """Open a psycopg connection per test. Closed on teardown."""
    conn = psycopg.connect(db_with_migrations, connect_timeout=10)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def brand_namespace(db_conn: psycopg.Connection) -> Iterator[str]:
    """A unique brand prefix used to scope rows for this test. The
    fixture pre-cleans + post-cleans rows matching the prefix so the
    test does not see leakage from previous runs or leak its own rows
    onto subsequent runs.
    """
    prefix = f"e2e-{uuid.uuid4().hex[:8]}-"
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM executive_insights WHERE brand LIKE %s",
            (prefix + "%",),
        )
    db_conn.commit()
    yield prefix
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM executive_insights WHERE brand LIKE %s",
            (prefix + "%",),
        )
    db_conn.commit()


# ----------------------------------------------------------------------------
# psycopg-backed supabase shim
# ----------------------------------------------------------------------------
#
# The portfolio-summary route at src/api/routes/executive_insights.py:215
# calls `get_supabase_client().table("executive_insights").select(...)
# .eq("recall", False).execute().data`. To exercise the REAL DB without
# spinning up a Supabase REST server, this shim implements the same
# fluent-builder API on top of a psycopg connection.
#
# This is NOT a mock of the database — the shim issues real SQL against
# the real connection. It IS a mock of the Supabase REST surface (the
# layer between the route and the DB driver). The decision is
# deliberate: testing through Supabase REST would require a running
# PostgREST instance, which is heavyweight for this test's scope. The
# shim catches schema regressions, NULL handling, query-shape
# regressions, and the migration apply itself — all the DB-side
# semantics the unit-test FakeSupabase cannot cover.


class _SupabaseQueryShim:
    """Translates Supabase's fluent API into a SQL SELECT statement."""

    def __init__(self, conn: psycopg.Connection, table: str) -> None:
        self._conn = conn
        self._table = table
        self._select_cols = "*"
        self._eq_filters: List[Tuple[str, Any]] = []
        self._order_col: Optional[str] = None
        self._order_desc: bool = False
        self._limit_n: Optional[int] = None

    def select(self, cols: str, *_args: Any, **_kwargs: Any) -> "_SupabaseQueryShim":
        # The route passes columns as a comma-separated string. We
        # forward as-is to the SQL SELECT clause — psycopg will reject
        # any non-existent column with an UndefinedColumn error, which
        # is the desired schema-pin behavior.
        self._select_cols = cols
        return self

    def eq(self, col: str, val: Any) -> "_SupabaseQueryShim":
        self._eq_filters.append((col, val))
        return self

    def order(self, col: str, *, desc: bool = False) -> "_SupabaseQueryShim":
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, n: int) -> "_SupabaseQueryShim":
        self._limit_n = n
        return self

    def execute(self) -> Any:
        sql_parts: List[str] = [f"SELECT {self._select_cols} FROM {self._table}"]
        params: List[Any] = []
        if self._eq_filters:
            wheres: List[str] = []
            for col, val in self._eq_filters:
                wheres.append(f"{col} = %s")
                params.append(val)
            sql_parts.append("WHERE " + " AND ".join(wheres))
        if self._order_col:
            sql_parts.append(f"ORDER BY {self._order_col} {'DESC' if self._order_desc else 'ASC'}")
        if self._limit_n is not None:
            sql_parts.append("LIMIT %s")
            params.append(self._limit_n)
        sql = " ".join(sql_parts)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            col_names = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(col_names, r, strict=True)) for r in cur.fetchall()]

        # The route reads `.data` off the execute() result. Mirror.
        class _Result:
            def __init__(self, data: List[Dict[str, Any]]) -> None:
                self.data = data

        return _Result(rows)


class _SupabaseShim:
    """The factory the route calls. Issues real SQL via psycopg."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def table(self, name: str) -> _SupabaseQueryShim:
        return _SupabaseQueryShim(self._conn, name)


# ----------------------------------------------------------------------------
# Row seeder
# ----------------------------------------------------------------------------


def _seed_full_row(
    conn: psycopg.Connection,
    *,
    insight_id: str,
    brand: str,
    region: str = "northeast",
    crystallized_at: Optional[datetime] = None,
    effect_size: Optional[float] = 0.42,
    recall: bool = False,
    causal_path_id: Optional[str] = None,
    invalidated_at: Optional[datetime] = None,
    invalidation_reason: Optional[str] = None,
) -> None:
    """INSERT one row with all 15 new fields populated. Returns no value;
    the caller pulls the row back via SELECT."""
    if crystallized_at is None:
        crystallized_at = datetime.now(timezone.utc)
    if causal_path_id is None:
        causal_path_id = f"cp-{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO executive_insights (
                insight_id, title, narrative, brand, region, kpi,
                crystallized_at, recall, source_count, key_metrics,
                invalidated_at, invalidation_reason,
                -- analytical 8
                effect_size, effect_ci_lower, effect_ci_upper, effect_direction,
                cohort_size, confounders_controlled,
                sensitivity_checks_passed, sensitivity_checks_failed,
                -- narrative prose 2
                limitations, recommended_next_analysis,
                -- lineage 5
                provenance_chain_id, provenance_depth, consolidation_tier,
                replication_count, data_version
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s
            )
            """,
            (
                insight_id,
                f"Insight for {brand}",
                "Synthetic e2e narrative.",
                brand,
                region,
                "trx",
                crystallized_at,
                recall,
                3,
                f'{{"causal_path_id": "{causal_path_id}"}}',
                invalidated_at,
                invalidation_reason,
                # analytical 8
                effect_size,
                0.30 if effect_size is not None else None,
                0.55 if effect_size is not None else None,
                ("positive" if (effect_size or 0) > 0 else "null"),
                1200,
                ["age", "prior_use"],
                ["placebo_treatment"],
                ["data_subset"],
                # narrative prose 2
                "Small pre-period sample.",
                "Replicate on Q3 cohort.",
                # lineage 5
                "chain-abc-123",
                2,
                "semantic",
                3,
                "2026-05-19-snapshot",
            ),
        )
    conn.commit()


# ----------------------------------------------------------------------------
# FastAPI TestClient w/ shim mounted
# ----------------------------------------------------------------------------


@pytest.fixture()
def test_client(db_conn: psycopg.Connection) -> Iterator[Any]:
    """Build a TestClient pointed at the real route with the supabase
    factory swapped for the psycopg-backed shim.

    The route imports ``get_supabase_client`` from the factories module
    at the top of ``src/api/routes/executive_insights.py``. We patch
    THAT binding (not the factories module itself) so the route's
    in-module reference resolves to our shim.
    """
    # Defer FastAPI import until inside the fixture to avoid module-load
    # cost when the test is skipped.
    from unittest.mock import patch

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes.executive_insights import router

    shim = _SupabaseShim(db_conn)

    app = FastAPI()
    app.include_router(router, prefix="/api")

    with patch(
        "src.api.routes.executive_insights.get_supabase_client",
        return_value=shim,
    ):
        with TestClient(app) as client:
            yield client


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


@pytest.mark.integration
def test_portfolio_summary_hits_real_db_and_aggregates_by_brand(
    db_conn: psycopg.Connection,
    brand_namespace: str,
    test_client: Any,
) -> None:
    """E2E: portfolio-summary aggregates real DB rows by brand.

    Seeds 3 rows into the real DB across 2 brands (one with effect_size
    NULL to exercise the NULL-skip code path at
    ``src/api/routes/executive_insights.py:240-250``), hits the route,
    asserts the aggregation is correct.

    What this catches that the unit test does NOT:
    1. Migration 025 actually applied — the SELECT mentions columns
       that only exist after the migration. A regression that drops a
       column from the SELECT (without dropping from the response
       model) would 500 here.
    2. Real Postgres semantics — TEXT[] columns coming back as Python
       lists, NULL effect_size handled by the route's
       ``if effect is not None`` guard against real NULL not a Python
       ``None`` sentinel.
    3. Brand-namespace isolation behaves as a WHERE clause (the route
       does NOT filter by brand here, but the shim emits real SQL so
       a regression that orders by a non-indexed column would surface).
    """
    brand_a = brand_namespace + "kisqali"
    brand_b = brand_namespace + "fabhalta"

    now = datetime.now(timezone.utc)
    _seed_full_row(
        db_conn,
        insight_id=str(uuid.uuid4()),
        brand=brand_a,
        crystallized_at=now - timedelta(hours=2),
        effect_size=0.40,
    )
    _seed_full_row(
        db_conn,
        insight_id=str(uuid.uuid4()),
        brand=brand_a,
        crystallized_at=now - timedelta(hours=1),
        effect_size=0.50,
    )
    _seed_full_row(
        db_conn,
        insight_id=str(uuid.uuid4()),
        brand=brand_b,
        crystallized_at=now,
        effect_size=None,  # legacy-style row; tests the NULL skip in mean
    )

    response = test_client.get("/api/executive-insights/portfolio-summary")
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. body={response.text}"
    )
    body = response.json()

    by_brand = {b["brand"]: b for b in body["by_brand"]}

    # Our two brands MUST be present (other brands from concurrent test
    # runs may also be — the route returns ALL non-recalled brands).
    assert brand_a in by_brand, f"brand_a={brand_a} not in by_brand keys={list(by_brand)}"
    assert brand_b in by_brand, f"brand_b={brand_b} not in by_brand keys={list(by_brand)}"

    # brand_a: 2 numeric rows, mean(0.40, 0.50) = 0.45
    a = by_brand[brand_a]
    assert a["insight_count"] == 2
    assert a["effect_size_sample_count"] == 2
    assert a["average_effect_size"] == pytest.approx(0.45)

    # brand_b: 1 row with effect_size=NULL — mean must be None,
    # sample_count must be 0 (the route excludes NULL effect_size rows).
    b = by_brand[brand_b]
    assert b["insight_count"] == 1
    assert b["effect_size_sample_count"] == 0
    assert b["average_effect_size"] is None


@pytest.mark.integration
def test_portfolio_summary_excludes_recalled_rows_from_real_db(
    db_conn: psycopg.Connection,
    brand_namespace: str,
    test_client: Any,
) -> None:
    """E2E: rows where ``recall=TRUE`` MUST NOT contribute to the
    portfolio summary. This is the recall-cascade contract from
    issue #376 §D. The unit test pins it against the fake supabase;
    this version pins it against a real Postgres BOOLEAN column.

    A regression that changes the route's WHERE clause from
    ``.eq("recall", False)`` to omit-the-filter would surface here as
    the recalled row's wildly-different effect_size polluting the mean.
    """
    brand = brand_namespace + "kisqali"
    now = datetime.now(timezone.utc)
    _seed_full_row(
        db_conn,
        insight_id=str(uuid.uuid4()),
        brand=brand,
        crystallized_at=now,
        effect_size=0.40,
        recall=False,
    )
    # Recalled row with an outlier effect that would visibly skew the
    # mean if the recall filter regressed. Force a different
    # causal_path_id so the partial-unique-index on (brand, region, kpi,
    # causal_path_id) WHERE invalidated_at IS NULL does not collide.
    _seed_full_row(
        db_conn,
        insight_id=str(uuid.uuid4()),
        brand=brand,
        crystallized_at=now,
        effect_size=999.0,
        recall=True,
        causal_path_id=f"cp-recalled-{uuid.uuid4().hex[:8]}",
    )

    response = test_client.get("/api/executive-insights/portfolio-summary")
    assert response.status_code == 200, response.text
    body = response.json()
    by_brand = {b["brand"]: b for b in body["by_brand"]}
    k = by_brand[brand]
    assert k["insight_count"] == 1, (
        f"Recalled row leaked into count: insight_count={k['insight_count']}"
    )
    assert k["average_effect_size"] == pytest.approx(0.40), (
        f"Recalled row leaked into mean: average_effect_size="
        f"{k['average_effect_size']!r} (expected 0.40 if recall filter "
        f"works; ~499 if it doesn't)"
    )


@pytest.mark.integration
def test_portfolio_summary_excludes_invalidated_rows(
    db_conn: psycopg.Connection,
    brand_namespace: str,
    test_client: Any,
) -> None:
    """``/portfolio-summary`` MUST exclude rows where ``invalidated_at``
    IS NOT NULL — even when ``recall = false`` (issue #385 fix).

    The route docstring at
    ``src/api/routes/executive_insights.py`` for
    ``get_portfolio_summary`` promises:

        "Aggregates across all non-recalled, non-invalidated crystals."

    Before the #385 fix the SELECT only applied ``.eq("recall", False)``
    so a row with ``recall=False`` AND ``invalidated_at IS NOT NULL``
    (the silent-cascade case from the ``InsightVerifierMiddleware``,
    migration ``021_insight_lifecycle.sql``) would:
      * INCREMENT ``insight_count``
      * POLLUTE ``average_effect_size`` (if numeric)
      * UPDATE ``latest_crystallized_at`` (if newer)

    This test seeds exactly that contamination case end-to-end against
    a real Postgres DB and asserts the invalidated row is excluded from
    count + mean + sample count.

    Historical note: this test previously pinned the LEAK behavior
    (see git history for ``test_portfolio_summary_invalidated_at_currently_leaks_pinned_for_issue_385``)
    so that a partial fix could not pass silently; on the #385 fix the
    assertions were flipped to assert the corrected contract.
    """
    brand = brand_namespace + "kisqali"
    now = datetime.now(timezone.utc)
    _seed_full_row(
        db_conn,
        insight_id=str(uuid.uuid4()),
        brand=brand,
        crystallized_at=now,
        effect_size=0.40,
        recall=False,
        invalidated_at=None,  # ACTIVE row — must contribute
    )
    # The contamination case: invalidated_at IS NOT NULL but recall=False.
    # An outlier effect_size that would be visibly wrong if it leaked
    # into the mean. Distinct causal_path so the partial-unique-index
    # on ``executive_insights`` does not block this insert (the index
    # is partial on invalidated_at IS NULL, so this row is exempt —
    # see 021_insight_lifecycle.sql:219-226).
    _seed_full_row(
        db_conn,
        insight_id=str(uuid.uuid4()),
        brand=brand,
        crystallized_at=now,
        effect_size=999.0,
        recall=False,
        invalidated_at=now - timedelta(hours=1),
        invalidation_reason="jit_verifier_ancestor_overturned",
        causal_path_id=f"cp-invalidated-{uuid.uuid4().hex[:8]}",
    )

    response = test_client.get("/api/executive-insights/portfolio-summary")
    assert response.status_code == 200, response.text
    body = response.json()
    by_brand = {b["brand"]: b for b in body["by_brand"]}
    k = by_brand[brand]

    assert k["insight_count"] == 1, (
        f"Invalidated row must NOT contribute to insight_count. Got "
        f"insight_count={k['insight_count']} (if 2, the #385 filter "
        f"regressed — invalidated row leaks into the portfolio summary)."
    )
    assert k["effect_size_sample_count"] == 1, (
        f"Invalidated row must NOT contribute to effect_size_sample_count. "
        f"Got effect_size_sample_count={k['effect_size_sample_count']}"
    )
    assert k["average_effect_size"] == pytest.approx(0.40), (
        f"Invalidated row must NOT pollute the mean. Expected 0.40 "
        f"(just the active row); got "
        f"average_effect_size={k['average_effect_size']!r} (if ~499, the "
        f"invalidated row's effect_size=999.0 is leaking through)."
    )


@pytest.mark.integration
def test_get_insight_round_trips_all_15_new_fields_from_real_db(
    db_conn: psycopg.Connection,
    brand_namespace: str,
    test_client: Any,
) -> None:
    """E2E: a row with all 15 new fields populated must round-trip
    through ``GET /api/executive-insights/{insight_id}`` and emerge with
    every field intact.

    This is the migration 025 acceptance test: if any column drops out
    of the SELECT * or fails to map back to the Pydantic response, this
    assertion catches it. The unit test pins the response shape; this
    test pins the migration + SELECT + Pydantic round-trip end-to-end.
    """
    brand = brand_namespace + "kisqali"
    insight_id = str(uuid.uuid4())
    _seed_full_row(
        db_conn,
        insight_id=insight_id,
        brand=brand,
        crystallized_at=datetime.now(timezone.utc),
        effect_size=0.42,
    )

    response = test_client.get(f"/api/executive-insights/{insight_id}")
    assert response.status_code == 200, response.text
    row = response.json()

    # Original 13 fields — sanity check the row mapped correctly.
    assert row["insight_id"] == insight_id
    assert row["brand"] == brand
    assert row["region"] == "northeast"
    assert row["kpi"] == "trx"

    # Analytical 8 — all populated.
    assert row["effect_size"] == pytest.approx(0.42)
    assert row["effect_ci_lower"] == pytest.approx(0.30)
    assert row["effect_ci_upper"] == pytest.approx(0.55)
    assert row["effect_direction"] == "positive"
    assert row["cohort_size"] == 1200
    assert row["confounders_controlled"] == ["age", "prior_use"]
    assert row["sensitivity_checks_passed"] == ["placebo_treatment"]
    assert row["sensitivity_checks_failed"] == ["data_subset"]

    # Narrative prose 2 — populated by _seed_full_row.
    assert row["limitations"] == "Small pre-period sample."
    assert row["recommended_next_analysis"] == "Replicate on Q3 cohort."

    # Lineage 5 — all populated.
    assert row["provenance_chain_id"] == "chain-abc-123"
    assert row["provenance_depth"] == 2
    assert row["consolidation_tier"] == "semantic"
    assert row["replication_count"] == 3
    assert row["data_version"] == "2026-05-19-snapshot"


@pytest.mark.integration
def test_portfolio_summary_route_resolves_before_dynamic_insight_id_route(
    db_conn: psycopg.Connection,
    brand_namespace: str,
    test_client: Any,
) -> None:
    """REGRESSION: ``/portfolio-summary`` MUST resolve to its handler,
    NOT fall through to ``GET /{insight_id}`` (which would try to parse
    ``"portfolio-summary"`` as a UUID and either 404 or 422).

    The route order at ``src/api/routes/executive_insights.py:198`` and
    ``:273`` is load-bearing — FastAPI matches in declaration order, so
    the literal ``/portfolio-summary`` route MUST be declared BEFORE the
    dynamic ``/{insight_id}`` route. The unit test
    ``test_portfolio_summary_route_does_not_shadow_insight_id_route``
    pins this against the fake supabase; THIS version pins it against
    the real DB to catch the case where the dynamic route would match
    + then 500 on a real DB query that the fake silently passes.

    Calls BOTH endpoints in the SAME test:
      * ``/portfolio-summary`` -> 200, list-of-brands payload shape.
      * ``/{real_insight_id}`` -> 200, single-row payload shape.
    Both must succeed; the contrast is what proves the route order is
    correct (if the literal route shadowed the dynamic one, the second
    call would 200 with a list — wrong shape).
    """
    insight_id = str(uuid.uuid4())
    _seed_full_row(
        db_conn,
        insight_id=insight_id,
        brand=brand_namespace + "kisqali",
        crystallized_at=datetime.now(timezone.utc),
        effect_size=0.40,
    )

    # The literal route MUST win.
    summary_resp = test_client.get("/api/executive-insights/portfolio-summary")
    assert summary_resp.status_code == 200, (
        f"Expected 200 from /portfolio-summary, got {summary_resp.status_code}. "
        f"body={summary_resp.text}. If the dynamic /{{insight_id}} route "
        f"shadowed this, we'd see a 404 or a 422 (UUID parse fail), since "
        f"'portfolio-summary' is not a valid UUID."
    )
    summary_body = summary_resp.json()
    # Shape pin: portfolio-summary returns by_brand list + total counters.
    assert "by_brand" in summary_body
    assert "total_brands" in summary_body
    assert "total_insights" in summary_body
    assert isinstance(summary_body["by_brand"], list)

    # The dynamic route on a real UUID still works.
    insight_resp = test_client.get(f"/api/executive-insights/{insight_id}")
    assert insight_resp.status_code == 200, (
        f"Expected 200 from /{{insight_id}}, got {insight_resp.status_code}. "
        f"body={insight_resp.text}"
    )
    insight_body = insight_resp.json()
    # Shape pin: get-one returns a single row dict with insight_id.
    assert insight_body["insight_id"] == insight_id
    assert "by_brand" not in insight_body, (
        "Dynamic /{insight_id} route returned a portfolio-summary shape — "
        "route ordering may have collapsed."
    )
