"""Migrations 159 + 160 (#2310, #2311, #2308): retrain candidates get their own stage, a
lineage link to the row they retrain, the exact MLflow version, and register-only deploys a
non-active status. The backfill moves only the three audited rows a retraining-history row
links.

Real Postgres, opt-in (``E2I_DB_INTEGRATION=1`` + docker). It lives in the learning_loop
suite so the deploy's real-DB gate (``scripts/deploy/realdb_suite_gate.sh``), which runs
that directory, runs it too (codex r4). A throwaway container of prod's own image,
the tables built by the repo's VERBATIM DDL (``database/ml/mlops_tables.sql`` +
``017_model_monitoring_tables.sql``), a copy of the live rows the backfill is written for
(ids, names, versions, stages, run ids and statuses as measured read-only on 2026-09-28),
then 159 + 160 exactly as the runner applies them. Nothing here touches ``supabase-db``
beyond ``docker inspect`` for the image tag.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pytest

pytestmark = pytest.mark.timeout(300)

REPO = Path(__file__).resolve().parents[4]
MIGRATIONS = REPO / "database" / "migrations"
M159 = MIGRATIONS / "159_registry_candidate_stage_and_lineage_enums.sql"
M160 = MIGRATIONS / "160_registry_candidate_stage_and_lineage.sql"
R159 = MIGRATIONS / "rollback_159_registry_candidate_stage_and_lineage_enums.sql"
R160 = MIGRATIONS / "rollback_160_registry_candidate_stage_and_lineage.sql"

PARENT = "4ec55d13-46c8-4df4-9ec8-7723fad67fb3"
FAILED_ROW = "c524db0f-1df1-4f21-806f-3b857c5245b9"
CAND_A = "cff4f2b5-a87e-4947-aa2a-243a7fb0ee45"
CAND_B = "faf4ed1d-15cf-4ae2-88c7-e927e64ec53f"
DEP_A = "53d7e7d7-19af-44b5-9f1c-2d149142448a"
DEP_B = "b3ffe870-c58d-4dc9-8f02-f8eb11d92760"
JOB_FAILED = "073c38eb-586b-4f83-adb3-4a7ec0d1d20b"
JOB_A = "836578bf-0456-433d-9593-dbd373581579"
JOB_B = "36cd579b-102f-41ec-9003-9261d315f785"
NAME = "initiation_kisqali_goldstd_lr_v1"
# Rows the backfill must never touch.
SIBLING = "8d1244df-7c38-435e-820f-1f201e51af24"  # another goldstd v1.0 reference row
DECOY = "11111111-2222-3333-4444-555555555555"  # same name, a version no history row names
OTHER_DEP = "99999999-8888-7777-6666-555555555555"  # an 'active' no-endpoint deploy of SIBLING
# codex r1: a retrain completed AFTER the audit -- the join matches it, the pin must not.
LATE = "22222222-3333-4444-5555-666666666666"
JOB_LATE = "33333333-4444-5555-6666-777777777777"


def _code(path: Path) -> str:
    return "\n".join(re.sub(r"--.*$", "", line) for line in path.read_text().splitlines())


@pytest.mark.unit
def test_the_runner_applies_159_unwrapped_and_160_wrapped():
    """Postgres refuses a new enum value inside the transaction that added it: 159 must run
    un-wrapped (each statement commits) and 160, which uses the values, wrapped and atomic."""
    from tests.unit.test_database.learning_loop._pg import runner_unwraps

    assert runner_unwraps(M159.read_text()) is True
    assert runner_unwraps(M160.read_text()) is False
    assert "ADD VALUE" not in _code(M160).upper()
    for p in (M159, M160):
        up = _code(p).upper()
        assert not re.search(r"^\s*(BEGIN|COMMIT|ROLLBACK)\s*;", up, re.M), p.name


@pytest.mark.unit
def test_159_and_160_numbers_are_unique_and_rollbacks_are_skipped_by_the_runner():
    assert [p.name for p in MIGRATIONS.glob("159_*.sql")] == [M159.name]
    assert [p.name for p in MIGRATIONS.glob("160_*.sql")] == [M160.name]
    assert R159.name.startswith("rollback_") and R160.name.startswith("rollback_")
    assert f"filename = '{M160.name}'" in _code(R160)


# --------------------------------------------------------------------------
# Real Postgres (opt-in)
# --------------------------------------------------------------------------

_OPT_IN = "E2I_DB_INTEGRATION"


def _docker_image_of_prod() -> str | None:
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


# The live rows (read-only, 2026-09-28) plus two rows the backfill must leave alone.
_SEED = f"""
INSERT INTO ml_model_registry (id, model_name, model_version, algorithm, stage, is_champion,
                               mlflow_run_id, registered_at) VALUES
 ('{PARENT}', '{NAME}', '1.0', 'logistic_regression_calibrated', 'staging', false, NULL,
  '2026-09-28 03:03:46+00'),
 ('{FAILED_ROW}', '{NAME}', '1.0_retrained_20260928_1617_d18ac8', 'logistic_regression_calibrated',
  'development', false, 'fef4480afb82435d894629decb4d73d5', '2026-09-28 16:19:35+00'),
 ('{CAND_A}', '{NAME}', '1.0_retrained_20260928_1934_725a94', 'logistic_regression_calibrated',
  'staging', false, '830aef651bba4bf5bb7202a96e614e7a', '2026-09-28 19:36:07+00'),
 ('{CAND_B}', '{NAME}', '1.0_retrained_20260928_2036_6fbd91', 'logistic_regression_calibrated',
  'staging', false, 'df85634d4315423d97bdf115ae02d4af', '2026-09-28 20:38:44+00'),
 ('{SIBLING}', 'hcp_adoption_kisqali_goldstd_lr_v1', '1.0', 'logistic_regression_calibrated',
  'staging', false, NULL, '2026-09-28 03:06:38+00'),
 ('{DECOY}', '{NAME}', '1.0_manual_copy', 'logistic_regression_calibrated',
  'staging', false, 'deadbeef', '2026-09-28 21:00:00+00'),
 ('{LATE}', '{NAME}', '1.0_retrained_20260929_0101_aaaaaa', 'logistic_regression_calibrated',
  'staging', false, 'late-run', '2026-09-29 01:03:00+00');
INSERT INTO ml_deployments (id, model_registry_id, deployment_name, environment, status) VALUES
 ('{DEP_A}', '{CAND_A}', 'd_a', 'staging', 'active'),
 ('{DEP_B}', '{CAND_B}', 'd_b', 'staging', 'active'),
 ('{OTHER_DEP}', '{SIBLING}', 'd_sib', 'staging', 'active');
INSERT INTO ml_retraining_history (id, trigger_type, trigger_reason, model_id, old_model_version,
                                   new_model_version, status, triggered_at) VALUES
 ('c4252b47-4978-41ff-91df-2ff6b6d4bcbf', 'manual', 'r', '{PARENT}', '{PARENT}',
  '{PARENT}_retrained_20260923_0542', 'failed', '2026-09-23 05:42:45+00'),
 ('{JOB_FAILED}', 'manual', 'r', '{PARENT}', '{PARENT}',
  '1.0_retrained_20260928_1617_d18ac8', 'failed', '2026-09-28 16:17:42+00'),
 ('{JOB_A}', 'manual', 'r', '{PARENT}', '{PARENT}',
  '1.0_retrained_20260928_1934_725a94', 'completed', '2026-09-28 19:34:09+00'),
 ('{JOB_B}', 'manual', 'r', '{PARENT}', '{PARENT}',
  '1.0_retrained_20260928_2036_6fbd91', 'completed', '2026-09-28 20:36:59+00'),
 ('{JOB_LATE}', 'manual', 'r', '{PARENT}', '{PARENT}',
  '1.0_retrained_20260929_0101_aaaaaa', 'completed', '2026-09-29 01:01:00+00');
"""


@pytest.fixture
def conn(throwaway_pg, request):
    from tests.unit.test_database.learning_loop._pg import PgConn, apply_migration

    name = "t160_" + re.sub(r"[^a-z0-9]", "_", request.node.name.lower())[:40]
    throwaway_pg.rows("postgres", f"CREATE DATABASE {name} OWNER postgres")
    c = PgConn(throwaway_pg, name)
    c.execute(
        "CREATE TABLE IF NOT EXISTS public.schema_migrations "
        "(filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now());"
        # The types mlops_tables.sql takes from earlier core files (labels irrelevant here).
        "CREATE TYPE data_split_type AS ENUM "
        "('train', 'validation', 'test', 'holdout', 'unassigned');"
        "CREATE TYPE agent_name_enum AS ENUM ('orchestrator');"
        "CREATE TYPE brand_type AS ENUM ('Kisqali');"
        "CREATE TYPE region_type AS ENUM ('northeast');",
        user="postgres",
    )
    for base in ("database/ml/mlops_tables.sql", "database/ml/017_model_monitoring_tables.sql"):
        apply_migration(c, REPO / base)
    assert c.rows(
        "select count(*) from pg_tables where schemaname = 'public' and tablename in "
        "('ml_model_registry', 'ml_deployments', 'ml_retraining_history')"
    ) == ["3"]
    assert c.rows("select count(*) from pg_trigger where tgname = 'tr_single_champion'") == ["1"]
    c.execute(_SEED, user="postgres")
    return c


def _apply_159_160(c) -> None:
    from tests.unit.test_database.learning_loop._pg import apply_migration

    assert apply_migration(c, M159, record=M159.name) == "unwrapped"
    assert apply_migration(c, M160, record=M160.name) == "wrapped"


def _registry(c) -> dict[str, tuple[str, str, str]]:
    rows = c.rows(
        "select id, stage, coalesce(retrain_of_id::text, '-'), "
        "coalesce(mlflow_model_version::text, '-') from ml_model_registry"
    )
    return {r.split("|")[0]: tuple(r.split("|")[1:]) for r in rows}


def _status(c) -> dict[str, str]:
    return dict(r.split("|") for r in c.rows("select id, status from ml_deployments"))


def _history_dep(c) -> dict[str, str]:
    rows = c.rows("select id, coalesce(deployment_id::text, '-') from ml_retraining_history")
    return dict(r.split("|") for r in rows)


def test_backfill_links_retrains_and_moves_only_their_rows(conn):
    _apply_159_160(conn)
    reg = _registry(conn)
    assert reg[CAND_A] == ("candidate", PARENT, "4")
    assert reg[CAND_B] == ("candidate", PARENT, "5")
    assert reg[FAILED_ROW] == ("archived", PARENT, "3")
    # Untouched: the parent, a sibling reference row, and a same-name row no job names.
    assert reg[PARENT] == ("staging", "-", "-")
    assert reg[SIBLING] == ("staging", "-", "-")
    assert reg[DECOY] == ("staging", "-", "-")
    assert reg[LATE] == ("staging", "-", "-")  # matched by the join, outside the pin
    assert _status(conn) == {DEP_A: "registered", DEP_B: "registered", OTHER_DEP: "active"}
    hist = _history_dep(conn)
    assert hist[JOB_A] == DEP_A and hist[JOB_B] == DEP_B and hist[JOB_FAILED] == "-"
    assert hist[JOB_LATE] == "-"


def test_second_application_changes_nothing(conn):
    from tests.unit.test_database.learning_loop._pg import apply_migration

    _apply_159_160(conn)
    before = (_registry(conn), _status(conn), _history_dep(conn))
    apply_migration(conn, M159)
    apply_migration(conn, M160)
    assert (_registry(conn), _status(conn), _history_dep(conn)) == before


def test_lineage_is_immutable_once_set_and_never_self(conn):
    from tests.unit.test_database.learning_loop._pg import DbFixtureError

    _apply_159_160(conn)
    with pytest.raises(DbFixtureError, match="immutable"):
        conn.execute(
            f"UPDATE ml_model_registry SET retrain_of_id = '{CAND_B}' WHERE id = '{CAND_A}'",
            user="postgres",
        )
    with pytest.raises(DbFixtureError, match="immutable"):
        conn.execute(
            f"UPDATE ml_model_registry SET retrain_of_id = NULL WHERE id = '{CAND_A}'",
            user="postgres",
        )
    with pytest.raises(DbFixtureError, match="retrain_of_not_self"):
        conn.execute(
            f"UPDATE ml_model_registry SET retrain_of_id = id WHERE id = '{DECOY}'",
            user="postgres",
        )
    # NULL -> parent stays allowed (the writer sets it once, at insert or on a fresh row).
    conn.execute(
        f"UPDATE ml_model_registry SET retrain_of_id = '{PARENT}' WHERE id = '{DECOY}'",
        user="postgres",
    )


def test_rollbacks_restore_the_pre_160_rows_and_160_reapplies(conn):
    from tests.unit.test_database.learning_loop._pg import apply_migration

    def rollback(path: Path):
        return conn.pg.run_script(
            conn.db, path.read_bytes(), single_transaction=True, user="postgres"
        )

    _apply_159_160(conn)
    # codex r1: a post-160 candidate no retraining-history row links (a standalone deploy)
    # must be restored too, not stranded at 'candidate' once the columns are dropped.
    conn.execute(
        f"UPDATE ml_model_registry SET retrain_of_id = '{PARENT}', stage = 'candidate' "
        f"WHERE id = '{DECOY}'",
        user="postgres",
    )
    refused = rollback(R159)
    assert refused.returncode != 0 and b"rollback 159 refused" in refused.stderr

    assert rollback(R160).returncode == 0
    stages = dict(r.split("|") for r in conn.rows("select id, stage from ml_model_registry"))
    assert stages[CAND_A] == stages[CAND_B] == stages[DECOY] == "staging"
    assert stages[FAILED_ROW] == "development"
    assert _status(conn) == {DEP_A: "active", DEP_B: "active", OTHER_DEP: "active"}
    assert set(_history_dep(conn).values()) == {"-"}
    cols = conn.rows(
        "select column_name from information_schema.columns where table_name = "
        "'ml_model_registry' and column_name in ('retrain_of_id', 'mlflow_model_version')"
    )
    assert cols == []
    assert conn.rows(f"select count(*) from schema_migrations where filename = '{M160.name}'") == [
        "0"
    ]
    assert rollback(R160).returncode == 0  # a second run changes nothing
    assert rollback(R159).returncode == 0

    apply_migration(conn, M160, record=M160.name)
    assert _registry(conn)[CAND_A] == ("candidate", PARENT, "4")


# #2319 item 5: rollback_160 reverted EVERY 'registered' deployment to 'active', including
# register-only records the new writer (model_deployer, #2308) produces after deploy. For
# those, 'active' is the plausible-wrong status #2308 removed. Only the two deployments 160
# itself moved (DEP_A / DEP_B, the pinned audited rows) go back.
LATE_DEP = "44444444-5555-6666-7777-888888888888"  # a post-160 retrain's register-only record


@pytest.mark.unit
def test_rollback_160_deployment_revert_names_the_pinned_rows():
    """CI ratchet for the opt-in rehearsal below: the deployment revert is scoped to the rows
    160 changed, never an unscoped ``status = 'registered'`` sweep."""
    code = _code(R160)
    update = re.search(r"UPDATE ml_deployments d\s+SET status = 'active'(.*?);", code, re.S)
    assert update is not None
    for pinned in (CAND_A, CAND_B, FAILED_ROW):
        assert pinned in update.group(1)
    assert "endpoint_url IS NULL" in update.group(1)


def test_rollback_160_leaves_register_only_deployments_it_did_not_create(conn):
    def rollback(path: Path):
        return conn.pg.run_script(
            conn.db, path.read_bytes(), single_transaction=True, user="postgres"
        )

    _apply_159_160(conn)
    # After deploy, the new writer records a promote-only (no endpoint) deploy of a
    # non-retrain model as 'registered' ...
    conn.execute(
        f"UPDATE ml_deployments SET status = 'registered' WHERE id = '{OTHER_DEP}'",
        user="postgres",
    )
    # ... and a post-160 retrain registers a linked candidate with a 'registered' record
    # (prod: 9376e88d / 3e14671d, 2026-09-29).
    conn.execute(
        f"UPDATE ml_model_registry SET retrain_of_id = '{PARENT}', stage = 'candidate' "
        f"WHERE id = '{DECOY}';"
        f"INSERT INTO ml_deployments (id, model_registry_id, deployment_name, environment, "
        f"status) VALUES ('{LATE_DEP}', '{DECOY}', 'd_late', 'staging', 'registered')",
        user="postgres",
    )

    assert rollback(R160).returncode == 0
    assert _status(conn) == {
        DEP_A: "active",  # 160 moved these two: restored
        DEP_B: "active",
        OTHER_DEP: "registered",  # not 160's: untouched (nothing serves it)
        LATE_DEP: "registered",  # not 160's: untouched
    }
    stages = dict(r.split("|") for r in conn.rows("select id, stage from ml_model_registry"))
    assert stages[DECOY] == "staging"  # the lineage-row restore is unchanged
    assert rollback(R160).returncode == 0  # a second run changes nothing
    assert _status(conn)[OTHER_DEP] == "registered"
    # rollback 159 still refuses while any row uses a 159 value (its documented contract).
    refused = rollback(R159)
    assert refused.returncode != 0 and b"rollback 159 refused" in refused.stderr
