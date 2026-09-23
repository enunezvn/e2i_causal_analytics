"""Migration 157 (#2260): the verdicts mirror keys on the RUN, not only on the second.

``scripts/mirror_audit_sidecar_to_supabase.py`` upserts every sidecar verdict into
``adaptive_validity_verdicts`` on ``(experiment_id, feature, written_at)``, and
``written_at`` has second resolution. Since #2257, runs of one scope share the scope's
experiment id, so two runs of one scope in the same second produced the same key and
the mirror kept one run's verdicts and dropped the other's. The sidecar now carries
the pipeline's ``audit_workflow_id`` and the key includes it.

Real Postgres, opt-in (``E2I_DB_INTEGRATION=1`` + docker, like
``tests/unit/test_database/learning_loop/``): a throwaway container of prod's own
image, the VERBATIM 040-043 migrations that build the table, then every
``157_*.sql``. The sidecars are written by the real producer and read by the real
mirror entry point. Nothing here touches ``supabase-db``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

# The container start alone outruns the suite-wide 30 s timeout (learning_loop sets 300).
pytestmark = pytest.mark.timeout(300)

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO / "database" / "migrations"
TABLE_MIGRATIONS = [
    MIGRATIONS / "040_adaptive_validity_verdicts.sql",
    MIGRATIONS / "041_role_attributions.sql",
    MIGRATIONS / "042_audit_evaluator_shadow_columns.sql",
    MIGRATIONS / "043_audit_evaluator_soft_gate_columns.sql",
]

EXPERIMENT_ID = "exp_remi_al_20260610180110_119813"
FROZEN_SECOND = (2026, 9, 23, 7, 0, 0)


def _m157() -> list[Path]:
    return sorted(MIGRATIONS.glob("157_*.sql"))


@pytest.mark.unit
def test_migration_157_exists_and_the_number_is_unique():
    assert len(_m157()) == 1, _m157()


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
def db(throwaway_pg, request, monkeypatch) -> Iterator[str]:
    """A fresh database holding ``adaptive_validity_verdicts`` as prod builds it
    (040-043), plus 157. Yields a DSN for the mirror's ``--database-url``."""
    from tests.unit.test_database.learning_loop._pg import PgConn, apply_migration

    name = "t157_" + re.sub(r"[^a-z0-9]", "_", request.node.name.lower())[:40]
    # OWNER postgres: in PG15 the public schema belongs to pg_database_owner, and the
    # migrations run as postgres (the role run_migrations.sh uses against prod).
    throwaway_pg.rows("postgres", f"CREATE DATABASE {name} OWNER postgres")
    conn = PgConn(throwaway_pg, name)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS public.schema_migrations "
        "(filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now());",
        user="postgres",
    )
    for path in TABLE_MIGRATIONS + _m157():
        apply_migration(conn, path, record=path.name)
    # The password travels in the environment, never in the DSN (libpq reads PGPASSWORD).
    monkeypatch.setenv("PGPASSWORD", throwaway_pg.client_env()["PGPASSWORD"])
    yield throwaway_pg.dsn(name)


def _write_run_sidecar(artifacts_dir: Path, monkeypatch, *, run_id, feature_severity: str):
    """One data-preparer run's sidecar, written by the real producer at a frozen second."""
    from src.agents.ml_foundation.data_preparer import graph

    class _FrozenDatetime(graph.datetime):  # type: ignore[misc,name-defined]
        @classmethod
        def now(cls, tz=None):
            return graph.datetime(*FROZEN_SECOND, tzinfo=tz)

    monkeypatch.setattr(graph, "datetime", _FrozenDatetime)
    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(artifacts_dir))
    state = {
        "experiment_id": EXPERIMENT_ID,
        "adaptive_verdicts": [
            {"feature": "disease_severity", "layer": "3", "severity": feature_severity}
        ],
    }
    if run_id is not None:
        state["audit_workflow_id"] = run_id
    path = graph.write_adaptive_verdicts_sidecar(state)
    assert path is not None
    return path


def _mirror(artifacts_dir: Path, dsn: str) -> None:
    from scripts.mirror_audit_sidecar_to_supabase import main as mirror_main

    rc = mirror_main(
        [
            "--artifacts-dir",
            str(artifacts_dir),
            "--since",
            "2025-01-01T00:00:00Z",
            "--database-url",
            dsn,
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0


def _count(dsn: str) -> int:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM adaptive_validity_verdicts")
        return int(cur.fetchone()[0])


def _rows(dsn: str) -> list[tuple]:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT experiment_id, feature, written_at, verdict->>'severity', "
            "audit_workflow_id::text, imported_at FROM adaptive_validity_verdicts "
            "ORDER BY verdict->>'severity'"
        )
        return list(cur.fetchall())


def test_two_runs_of_one_scope_in_one_second_both_survive_the_mirror(db, tmp_path, monkeypatch):
    run_a, run_b = uuid4(), uuid4()
    _write_run_sidecar(tmp_path, monkeypatch, run_id=run_a, feature_severity="high")
    _write_run_sidecar(tmp_path, monkeypatch, run_id=run_b, feature_severity="info")
    _mirror(tmp_path, db)

    assert _count(db) == 2, "one run's verdicts were dropped on the shared key"
    rows = _rows(db)
    # Same experiment, feature and second: only the run tells them apart.
    assert {(r[0], r[1], r[2]) for r in rows} == {(rows[0][0], "disease_severity", rows[0][2])}
    assert [(r[3], r[4]) for r in rows] == [("high", str(run_a)), ("info", str(run_b))]


def test_re_mirroring_the_same_runs_adds_no_rows_and_rewrites_nothing(db, tmp_path, monkeypatch):
    _write_run_sidecar(tmp_path, monkeypatch, run_id=uuid4(), feature_severity="high")
    _write_run_sidecar(tmp_path, monkeypatch, run_id=uuid4(), feature_severity="info")
    _mirror(tmp_path, db)
    first = _rows(db)
    _mirror(tmp_path, db)
    second = _rows(db)
    assert len(first) == 2
    # Byte-identical re-import is a no-op: same rows, and imported_at never advanced.
    assert second == first


def test_legacy_sidecars_without_a_run_id_keep_the_old_dedup(db, tmp_path, monkeypatch):
    """A sidecar written before #2260 carries no run id. Two of them on one key still
    fold into one row (NULL run id folds to one sentinel), exactly as under 040."""
    _write_run_sidecar(tmp_path, monkeypatch, run_id=None, feature_severity="high")
    _write_run_sidecar(tmp_path, monkeypatch, run_id=None, feature_severity="info")
    _mirror(tmp_path, db)
    assert _count(db) == 1
    rows = _rows(db)
    assert rows[0][4] is None
