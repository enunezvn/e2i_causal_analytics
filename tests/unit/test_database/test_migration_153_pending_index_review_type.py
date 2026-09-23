"""Migration 153 (#2244): the pending-review uniqueness is per QUEUE, not per estimand.

Two producers write pending ``expert_reviews`` rows for the same estimand with
different meanings: the runtime ``ExpertReviewGate`` (``dag_approval``, plus the
``quarterly_audit`` renewals of #2090, which the gate adopts as its own consult)
and Lane B's structural author (``initial_dag``, whose loader refuses any other
type). Migration 140's ``uq_er_pending_estimand`` keyed on the estimand alone, so
the two collided: whichever inserted second hit a 23505 and the recovery adopted
the OTHER queue's row.

Two levels of proof live here:

* text locks on the migration and rollback files (always run);
* a real-Postgres rehearsal (opt-in, ``E2I_DB_INTEGRATION=1`` + docker, like
  ``tests/unit/test_database/learning_loop/``): a throwaway container of prod's
  own image, the ``expert_reviews`` table from 010's DDL, then the VERBATIM 140,
  152, 153 and rollback files. It shows the collision under 140 (the premise),
  its absence under 153, uniqueness kept inside each queue, and the rollback
  restoring 140's definition. Nothing here touches ``supabase-db``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pytest

# The container start alone outruns the suite-wide 30 s timeout (learning_loop sets 300).
pytestmark = pytest.mark.timeout(300)

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO / "database" / "migrations"
M140 = MIGRATIONS / "140_expert_reviews_estimand_key.sql"
M152 = MIGRATIONS / "152_expert_review_type_initial_dag.sql"
M134 = MIGRATIONS / "134_guarded_causal_path_promote.sql"
M153 = MIGRATIONS / "153_expert_reviews_pending_index_per_queue.sql"
R153 = MIGRATIONS / "rollback_153_expert_reviews_pending_index_per_queue.sql"
TABLE_DDL = REPO / "database" / "ml" / "010_causal_validation_tables.sql"

RUNTIME_INDEX = "uq_er_pending_estimand_runtime"
STRUCTURAL_INDEX = "uq_er_pending_estimand_structural"
OLD_INDEX = "uq_er_pending_estimand"

# The index definitions as Postgres reports them (pg_indexes.indexdef) once each file
# has run. Pinned verbatim so a rehearsal cannot pass on a "close enough" predicate.
RUNTIME_INDEXDEF = (
    f"CREATE UNIQUE INDEX {RUNTIME_INDEX} ON public.expert_reviews USING btree (estimand_key) "
    "WHERE (((approval_status)::text = 'pending'::text) AND "
    "(review_type <> 'initial_dag'::expert_review_type))"
)
STRUCTURAL_INDEXDEF = (
    f"CREATE UNIQUE INDEX {STRUCTURAL_INDEX} ON public.expert_reviews USING btree (estimand_key) "
    "WHERE (((approval_status)::text = 'pending'::text) AND "
    "(review_type = 'initial_dag'::expert_review_type))"
)
OLD_INDEXDEF = (
    f"CREATE UNIQUE INDEX {OLD_INDEX} ON public.expert_reviews USING btree (estimand_key) "
    "WHERE ((approval_status)::text = 'pending'::text)"
)


def _sql(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.strip().startswith("--")
    )


# --------------------------------------------------------------------------
# File-shape locks
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_migration_and_rollback_exist_and_the_number_is_unique():
    assert M153.is_file(), M153
    assert R153.is_file(), R153
    files = sorted(MIGRATIONS.glob("153_*.sql"))
    assert [f.name for f in files] == [M153.name]


@pytest.mark.unit
def test_migration_replaces_the_estimand_index_with_one_per_queue():
    s = _sql(M153)
    # The old index must be DROPPED, not left beside the new ones: it is the index
    # that forbids the two queues from coexisting.
    assert re.search(rf"DROP INDEX IF EXISTS (public\.)?{OLD_INDEX}\s*;", s)
    assert re.search(
        rf"CREATE UNIQUE INDEX IF NOT EXISTS {RUNTIME_INDEX}\s+ON public\.expert_reviews "
        r"\(estimand_key\)\s+WHERE approval_status = 'pending'\s+AND review_type <> 'initial_dag'",
        s,
    )
    assert re.search(
        rf"CREATE UNIQUE INDEX IF NOT EXISTS {STRUCTURAL_INDEX}\s+ON public\.expert_reviews "
        r"\(estimand_key\)\s+WHERE approval_status = 'pending'\s+AND review_type = 'initial_dag'",
        s,
    )
    # The drop precedes both creates: a new index with a NEW name can never no-op on
    # the old definition, and the old one must be gone before the queues are used.
    assert s.index(f"DROP INDEX IF EXISTS {OLD_INDEX}") < s.index(RUNTIME_INDEX)
    assert s.index(f"DROP INDEX IF EXISTS {OLD_INDEX}") < s.index(STRUCTURAL_INDEX)
    assert f"COMMENT ON INDEX {RUNTIME_INDEX} IS" in s
    assert f"COMMENT ON INDEX {STRUCTURAL_INDEX} IS" in s


@pytest.mark.unit
def test_migration_replaces_both_db_side_chronology_readers_with_the_runtime_queue():
    """codex r3: ``dag_structure_rejected`` (134, inside the promotion RPC) and the
    schema-owned ``is_dag_approved`` (010) ranked both queues."""
    s = _sql(M153)
    rej = s[s.index("CREATE OR REPLACE FUNCTION public.dag_structure_rejected") :]
    rej = rej[: rej.index("$$;") + 3]
    # BOTH scans (latest adjudication and the reopen check) carry the predicate.
    assert rej.count("review_type <> 'initial_dag'") == 2, rej
    assert "p_dag_version_hash text" in rej and "p_brand text DEFAULT NULL" in rej
    appr = s[s.index("CREATE OR REPLACE FUNCTION public.is_dag_approved") :]
    appr = appr[: appr.index("LANGUAGE plpgsql")]
    assert appr.count("review_type <> 'initial_dag'") == 1, appr
    # Same signatures as 134 / 010, so these REPLACE rather than overload.
    assert "p_dag_hash VARCHAR(64)" in appr and "p_brand VARCHAR(50) DEFAULT NULL" in appr
    # 134's grant posture is asserted, not assumed.
    assert "has_function_privilege('service_role'" in s
    assert "has_function_privilege('anon'" in s


@pytest.mark.unit
def test_rollback_restores_both_readers_to_their_pre_153_bodies():
    s = _sql(R153)
    rej = s[s.index("CREATE OR REPLACE FUNCTION public.dag_structure_rejected") :]
    rej = rej[: rej.index("$$;") + 3]
    assert "initial_dag" not in rej
    # byte-for-byte the 134 body (comments stripped on both sides)
    m134 = _sql(M134)
    body134 = m134[m134.index("CREATE OR REPLACE FUNCTION public.dag_structure_rejected") :]
    body134 = body134[: body134.index("$$;") + 3]
    assert rej == body134
    appr = s[s.index("CREATE OR REPLACE FUNCTION public.is_dag_approved") :]
    appr = appr[: appr.index("LANGUAGE plpgsql")]
    assert "initial_dag" not in appr


@pytest.mark.unit
def test_migration_is_wrapped_and_touches_no_rows():
    s = _sql(M153).upper()
    # Plain (non-CONCURRENTLY) index DDL and no ALTER TYPE: run_migrations.sh wraps the
    # file in its --single-transaction, so the drop and the two creates land together
    # or not at all -- there is no window with NO pending uniqueness.
    assert "CONCURRENTLY" not in s
    assert "ALTER TYPE" not in s
    assert "\nBEGIN;" not in s and "\nCOMMIT;" not in s  # plpgsql BEGIN/END blocks are fine
    assert "DELETE FROM" not in s and "UPDATE " not in s and "INSERT INTO" not in s
    assert "DROP TABLE" not in s and "DROP COLUMN" not in s


@pytest.mark.unit
def test_migration_does_not_edit_140_and_points_back_to_it():
    # Never edit an applied migration: 140's own text is untouched and 153 names it.
    assert re.search(
        rf"CREATE UNIQUE INDEX IF NOT EXISTS {OLD_INDEX}\s+ON public\.expert_reviews "
        r"\(estimand_key\)\s+WHERE approval_status = 'pending'",
        _sql(M140),
    )
    raw = M153.read_text()
    assert "140" in raw and "#2244" in raw and "initial_dag" in raw


@pytest.mark.unit
def test_rollback_restores_140s_index_and_deletes_the_ledger_row():
    s = _sql(R153)
    assert re.search(rf"DROP INDEX IF EXISTS (public\.)?{RUNTIME_INDEX}\s*;", s)
    assert re.search(rf"DROP INDEX IF EXISTS (public\.)?{STRUCTURAL_INDEX}\s*;", s)
    # 140's definition, byte-for-byte in the parts that matter.
    assert re.search(
        rf"CREATE UNIQUE INDEX IF NOT EXISTS {OLD_INDEX}\s+ON public\.expert_reviews "
        r"\(estimand_key\)\s+WHERE approval_status = 'pending'\s*;",
        s,
    )
    # The rollback fails loudly BEFORE dropping anything if two queues hold a pending
    # row on one estimand -- the restored index could not be created and the file
    # would otherwise fail after the drops, inside one transaction (so nothing is
    # lost) but with a less useful message.
    assert "RAISE EXCEPTION" in s
    assert s.index("RAISE EXCEPTION") < s.index(f"DROP INDEX IF EXISTS {RUNTIME_INDEX}")
    # run_migrations.sh skips any file already in the ledger, so a rollback that left
    # the row in place would make the next deploy silently skip re-applying 153.
    assert re.search(
        r"DELETE FROM public\.schema_migrations\s+WHERE filename = '153_expert_reviews_pending_index_per_queue\.sql'",
        s,
    )
    assert "DROP TABLE" not in s.upper() and "DROP COLUMN" not in s.upper()


@pytest.mark.unit
def test_rollback_is_not_a_forward_migration():
    # scripts/run_migrations.sh apply_dir() skips rollback_*.sql; the name is the guard.
    assert R153.name.startswith("rollback_")


# --------------------------------------------------------------------------
# Real Postgres (opt-in)
# --------------------------------------------------------------------------

_OPT_IN = "E2I_DB_INTEGRATION"


def _docker_image_of_prod() -> str | None:
    """The image tag prod's Postgres runs; ``docker inspect`` reads container metadata
    only (no connection to the database)."""
    if shutil.which("docker") is None:
        return None
    proc = subprocess.run(
        ["docker", "inspect", "supabase-db", "--format", "{{.Config.Image}}"],
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode().strip() or None


def _enum_and_table_ddl() -> str:
    """The ``expert_review_type`` enum and the ``expert_reviews`` table exactly as
    ``database/ml/010_causal_validation_tables.sql`` creates them, plus the nullable
    columns later migrations added (097, 136, 142 -- ADD COLUMN IF NOT EXISTS, no
    defaults, no constraints) so the verbatim 140 file has every column it reads."""
    text = TABLE_DDL.read_text()
    enum_start = text.index("CREATE TYPE expert_review_type AS ENUM")
    enum_end = text.index(");", enum_start) + 2
    table_start = text.index("CREATE TABLE IF NOT EXISTS expert_reviews")
    table_end = text.index("\n);", table_start) + 3
    return "\n".join(
        [
            text[enum_start:enum_end],
            text[table_start:table_end],
            "ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS dag_structure_json JSONB;",
            "ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS agent_assessment_json JSONB;",
            "ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;",
            "ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS adjustment_set_hash VARCHAR(64);",
            "CREATE TABLE IF NOT EXISTS public.schema_migrations (filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now());",
            # 010's schema-owned reader, verbatim (153 replaces it; the rollback restores it).
            _function_ddl_from_010(),
            # migration 134's promote RPC updates causal_paths (its own smoke test calls it).
            "CREATE TABLE IF NOT EXISTS public.causal_paths (path_id TEXT PRIMARY KEY, validation_status TEXT);",
        ]
    )


def _function_ddl_from_010() -> str:
    text = TABLE_DDL.read_text()
    start = text.index("CREATE OR REPLACE FUNCTION is_dag_approved(")
    end = text.index("$$ LANGUAGE plpgsql;", start) + len("$$ LANGUAGE plpgsql;")
    return text[start:end]


@pytest.fixture(scope="module")
def throwaway_pg() -> Iterator[object]:
    if os.environ.get(_OPT_IN) != "1":
        pytest.skip(
            f"real-DB rehearsal: opt-in with {_OPT_IN}=1 (docker, prod's Postgres image); "
            "a skipped real-DB test is not coverage"
        )
    image = _docker_image_of_prod()
    if image is None:
        pytest.skip("docker or the supabase-db container is not reachable")
    from tests.unit.test_database.learning_loop._pg import ThrowawayPg

    pg = ThrowawayPg(image=image)
    pg.start()
    try:
        yield pg
    finally:
        pg.stop()


@pytest.fixture
def db(throwaway_pg, request) -> Iterator[tuple[object, str]]:
    """A fresh database holding the expert_reviews table with 140 + 152 applied
    (prod's state before 153)."""
    from tests.unit.test_database.learning_loop._pg import PgConn, apply_migration

    name = "t153_" + re.sub(r"[^a-z0-9]", "_", request.node.name.lower())[:40]
    # OWNER postgres: in PG15 the public schema belongs to pg_database_owner, and the
    # migrations run as postgres (the role run_migrations.sh uses against prod).
    throwaway_pg.rows("postgres", f"CREATE DATABASE {name} OWNER postgres")
    conn = PgConn(throwaway_pg, name)
    # Owned by ``postgres``, the role run_migrations.sh applies with against prod (the
    # index swap needs table ownership).
    conn.execute(_enum_and_table_ddl(), user="postgres")
    assert apply_migration(conn, M134, record=M134.name) == "wrapped"
    assert apply_migration(conn, M140, record=M140.name) == "wrapped"
    assert apply_migration(conn, M152, record=M152.name) == "unwrapped"
    yield throwaway_pg, name


def _indexdefs(pg, db: str) -> dict[str, str]:
    rows = pg.rows(
        db,
        "SELECT indexname || '|' || indexdef FROM pg_indexes "
        "WHERE tablename = 'expert_reviews' AND indexname LIKE 'uq_er_pending%' ORDER BY 1",
    )
    return dict(line.split("|", 1) for line in rows)


def _insert(conn, review_type: str, status: str, dag_hash: str) -> str:
    """One row on the estimand (B, T, Y); returns the id. ``created_at`` is the
    insert time, so successive calls are strictly ordered for the chronology rule."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.expert_reviews (review_type, reviewer_id, approval_status, "
            "brand, treatment_variable, outcome_variable, dag_version_hash, created_at) "
            "VALUES (%s, 'q', %s, 'B', 'T', 'Y', %s, clock_timestamp()) RETURNING review_id",
            (review_type, status, dag_hash),
        )
        return str(cur.fetchone()[0])


def _insert_pending(conn, review_type: str) -> str:
    return _insert(conn, review_type, "pending", "h_" + review_type)


def _rejected(conn, dag_hash: str, brand: str = "B") -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT public.dag_structure_rejected(%s, %s)", (dag_hash, brand))
        return bool(cur.fetchone()[0])


def _approved(conn, dag_hash: str, brand: str = "B") -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT public.is_dag_approved(%s, %s)", (dag_hash, brand))
        return bool(cur.fetchone()[0])


def _unique_violation(conn, review_type: str) -> str | None:
    """The index a second pending row of ``review_type`` violates, or None if it inserts."""
    import psycopg

    try:
        with conn.transaction():
            _insert_pending(conn, review_type)
    except psycopg.errors.UniqueViolation as exc:
        return exc.diag.constraint_name
    return None


@pytest.mark.unit
def test_under_140_a_structural_author_review_collides_with_a_gate_consult(db):
    """The premise (#2244): one pending row per estimand REGARDLESS of review_type.
    This is the cheap disproof the fix rests on; it runs against the verbatim 140."""
    pg, name = db
    assert _indexdefs(pg, name) == {OLD_INDEX: OLD_INDEXDEF}
    with pg.connect(name) as conn:
        _insert_pending(conn, "initial_dag")
        conn.commit()
        assert _unique_violation(conn, "dag_approval") == OLD_INDEX
        assert _unique_violation(conn, "quarterly_audit") == OLD_INDEX


@pytest.mark.unit
def test_153_lets_the_two_queues_coexist_and_keeps_each_unique(db):
    from tests.unit.test_database.learning_loop._pg import apply_migration

    pg, name = db
    with pg.connect(name) as conn:
        # Pre-existing pending rows survive the swap (the new keys partition the old one).
        _insert_pending(conn, "dag_approval")
        conn.commit()

    assert apply_migration(_conn(pg, name), M153, record=M153.name) == "wrapped"
    assert _indexdefs(pg, name) == {
        RUNTIME_INDEX: RUNTIME_INDEXDEF,
        STRUCTURAL_INDEX: STRUCTURAL_INDEXDEF,
    }
    assert pg.rows(name, "SELECT filename FROM public.schema_migrations ORDER BY 1") == [
        M134.name,
        M140.name,
        M152.name,
        M153.name,
    ]

    with pg.connect(name) as conn:
        # The structural-author review now has its own slot beside the gate consult...
        assert _unique_violation(conn, "initial_dag") is None
        # ...and each queue is still one-pending-per-estimand.
        assert _unique_violation(conn, "initial_dag") == STRUCTURAL_INDEX
        assert _unique_violation(conn, "dag_approval") == RUNTIME_INDEX
        # A renewal (#2090) is the RUNTIME queue: it shares the gate consult's slot.
        assert _unique_violation(conn, "quarterly_audit") == RUNTIME_INDEX
        # Resolving a row frees its queue's slot and no other.
        conn.execute(
            "UPDATE public.expert_reviews SET approval_status = 'approved' "
            "WHERE review_type = 'dag_approval'"
        )
        conn.commit()
        assert _unique_violation(conn, "dag_approval") is None
        assert _unique_violation(conn, "initial_dag") == STRUCTURAL_INDEX


@pytest.mark.unit
def test_rollback_153_restores_140_and_refuses_while_both_queues_are_pending(db):
    from tests.unit.test_database.learning_loop._pg import apply_migration

    pg, name = db
    conn_ = _conn(pg, name)
    apply_migration(conn_, M153, record=M153.name)
    with pg.connect(name) as conn:
        _insert_pending(conn, "initial_dag")
        _insert_pending(conn, "dag_approval")
        conn.commit()

    # Two queues pending on one estimand: the rollback must refuse BEFORE dropping.
    proc = pg.run_script(name, R153.read_bytes(), single_transaction=True, user="postgres")
    assert proc.returncode != 0
    assert "rollback 153" in proc.stderr.decode()
    assert _indexdefs(pg, name) == {
        RUNTIME_INDEX: RUNTIME_INDEXDEF,
        STRUCTURAL_INDEX: STRUCTURAL_INDEXDEF,
    }
    assert M153.name in pg.rows(name, "SELECT filename FROM public.schema_migrations")

    # Resolve one; the rollback then restores 140's exact index and un-ledgers 153.
    pg.rows(
        name,
        "UPDATE public.expert_reviews SET approval_status = 'rejected' "
        "WHERE review_type = 'initial_dag'",
    )
    proc = pg.run_script(name, R153.read_bytes(), single_transaction=True, user="postgres")
    assert proc.returncode == 0, proc.stderr.decode()
    assert _indexdefs(pg, name) == {OLD_INDEX: OLD_INDEXDEF}
    assert M153.name not in pg.rows(name, "SELECT filename FROM public.schema_migrations")
    # ...and 153 re-applies cleanly afterwards (the ledger row is gone).
    assert apply_migration(conn_, M153, record=M153.name) == "wrapped"
    assert _indexdefs(pg, name) == {
        RUNTIME_INDEX: RUNTIME_INDEXDEF,
        STRUCTURAL_INDEX: STRUCTURAL_INDEXDEF,
    }


@pytest.mark.unit
def test_153_db_side_readers_ignore_the_structural_queue_and_the_rollback_restores_them(db):
    """codex r3: the promotion RPC's ``dag_structure_rejected`` and the schema's
    ``is_dag_approved`` ranked both queues. Premise first (134 + 010 as applied),
    then 153, then the rollback."""
    from tests.unit.test_database.learning_loop._pg import apply_migration

    pg, name = db
    with pg.connect(name) as conn:
        _insert(conn, "initial_dag", "rejected", "h-rej")
        _insert(conn, "initial_dag", "approved", "h-appr")
        conn.commit()
        # Premise under 134/010: an authored verdict decides for the runtime gate.
        assert _rejected(conn, "h-rej") is True
        assert _approved(conn, "h-appr") is True

    apply_migration(_conn(pg, name), M153, record=M153.name)
    with pg.connect(name) as conn:
        # The authored verdicts no longer decide...
        assert _rejected(conn, "h-rej") is False
        assert _approved(conn, "h-appr") is False
        # ...the runtime queue's still do, with 134's chronology intact:
        _insert(conn, "dag_approval", "rejected", "h-rej")
        conn.commit()
        assert _rejected(conn, "h-rej") is True
        # a NEWER authored approval of the same hash does not mask that rejection
        _insert(conn, "initial_dag", "approved", "h-rej")
        conn.commit()
        assert _rejected(conn, "h-rej") is True
        assert _approved(conn, "h-rej") is False
        # a newer pending RUNTIME row reopens it; a pending authored row does not
        _insert(conn, "initial_dag", "pending", "h-rej")
        conn.commit()
        assert _rejected(conn, "h-rej") is True
        _insert(conn, "dag_approval", "pending", "h-rej")
        conn.commit()
        assert _rejected(conn, "h-rej") is False
        _insert(conn, "dag_approval", "approved", "h-appr")
        conn.commit()
        assert _approved(conn, "h-appr") is True
        # grants survived the replace (134's posture)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT has_function_privilege('service_role', "
                "'public.dag_structure_rejected(text, text)', 'EXECUTE'), "
                "has_function_privilege('anon', "
                "'public.dag_structure_rejected(text, text)', 'EXECUTE')"
            )
            assert cur.fetchone() == (True, False)
        # leave one pending row per estimand so the rollback's precondition passes
        conn.execute(
            "UPDATE public.expert_reviews SET approval_status = 'superseded' "
            "WHERE approval_status = 'pending' AND review_type = 'initial_dag'"
        )
        conn.commit()

    proc = pg.run_script(name, R153.read_bytes(), single_transaction=True, user="postgres")
    assert proc.returncode == 0, proc.stderr.decode()
    with pg.connect(name) as conn:
        # Both queues again: h-rej's newest non-pending row is the authored
        # approval, so 134's rule reads "not rejected"; h-appr's authored
        # approval counts once more.
        assert _rejected(conn, "h-rej") is False
        assert _approved(conn, "h-appr") is True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT has_function_privilege('service_role', "
                "'public.dag_structure_rejected(text, text)', 'EXECUTE')"
            )
            assert cur.fetchone() == (True,)


def _conn(pg, name: str):
    from tests.unit.test_database.learning_loop._pg import PgConn

    return PgConn(pg, name)
