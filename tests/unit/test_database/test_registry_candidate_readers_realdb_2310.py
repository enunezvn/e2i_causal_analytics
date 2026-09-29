"""#2310 R3: the stage-agnostic readers of ``ml_model_registry`` never pick a retrain candidate.

A retrain registers its row at stage ``candidate`` (migration 159) under the model_name of the
row it retrains, with ``retrain_of_id`` pointing at it (migration 160). The canonical row for a
name is the one whose stage is NOT ``candidate`` / ``archived`` / ``deprecated``. Lineage
(``retrain_of_id``) is never used to exclude a row: a promoted retrain becomes canonical by a
stage change alone.

Real Postgres + real PostgREST, opt-in (``E2I_DB_INTEGRATION=1`` + docker): a throwaway
Postgres container of prod's own image, the tables built from the repo's VERBATIM DDL
(``database/ml/mlops_tables.sql`` + ``017_model_monitoring_tables.sql``) plus migrations 159,
160 and 161 as the runner applies them, and a throwaway PostgREST of prod's own image in front of it.
The readers run unmodified through real ``postgrest`` clients. Nothing here reads or writes
``supabase-db`` / ``supabase-rest`` beyond ``docker inspect`` for the image tags.

The seed mirrors the live rows migration 160 was written for (ids, names, versions, stages as
Lane A measured them 2026-09-28). The parent row is UPDATEd after its retrain rows are written
(the backfill script's ``auc`` refresh does exactly that), so its live tuple sits after them.
``test_the_fixture_reproduces_the_unordered_hazard`` sends the pre-#2310 query shape (an
unordered ``.eq(model_name).limit(1)``) through the same PostgREST and asserts it answers with a
NON-canonical row (the archived failed retrain or a candidate), so a green here is the fix at
work and not a lucky heap order.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

pytestmark = pytest.mark.timeout(300)

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO / "database" / "migrations"
M159 = MIGRATIONS / "159_registry_candidate_stage_and_lineage_enums.sql"
M160 = MIGRATIONS / "160_registry_candidate_stage_and_lineage.sql"
M161 = MIGRATIONS / "161_get_latest_model_canonical.sql"
R161 = MIGRATIONS / "rollback_161_get_latest_model_canonical.sql"

EXP_INIT = "0e0e0e0e-0000-4000-8000-000000000001"
EXP_HCP = "0e0e0e0e-0000-4000-8000-000000000002"
EXP_CHAMP = "0e0e0e0e-0000-4000-8000-000000000003"
EXP_DUAL = "0e0e0e0e-0000-4000-8000-000000000004"

NAME = "initiation_kisqali_goldstd_lr_v1"
PARENT = "4ec55d13-46c8-4df4-9ec8-7723fad67fb3"
FAILED_ROW = "c524db0f-1df1-4f21-806f-3b857c5245b9"
CAND_A = "cff4f2b5-a87e-4947-aa2a-243a7fb0ee45"
CAND_B = "faf4ed1d-15cf-4ae2-88c7-e927e64ec53f"

HCP_NAME = "hcp_adoption_kisqali_goldstd_lr_v1"
HCP_REF = "8d1244df-7c38-435e-820f-1f201e51af24"
HCP_CAND = "8d1244df-0000-4000-8000-00000000000c"

CHAMP_CAND = "c0c0c0c0-0000-4000-8000-000000000001"

DUAL_NAME = "lane2310_dual_canonical"
DUAL_OLD = "d0d0d0d0-0000-4000-8000-000000000001"
DUAL_NEW = "d0d0d0d0-0000-4000-8000-000000000002"

NULL_NAME = "lane2310_null_stage"
NULL_STAGE = "e0e0e0e0-0000-4000-8000-000000000001"
NULL_CAND = "e0e0e0e0-0000-4000-8000-000000000002"
DEPRECATED = "e0e0e0e0-0000-4000-8000-000000000003"

EXP_EDGE = "0e0e0e0e-0000-4000-8000-000000000005"
UNDATED_NAME = "lane2310_undated"
UNDATED = "f0f0f0f0-0000-4000-8000-000000000001"  # canonical, registered_at NULL
DATED = "f0f0f0f0-0000-4000-8000-000000000002"  # canonical, dated
ARCHIVED_CHAMP = "f0f0f0f0-0000-4000-8000-000000000003"  # archived but still is_champion

_OPT_IN = "E2I_DB_INTEGRATION"


def _docker_image(container: str) -> str | None:
    if shutil.which("docker") is None:
        return None
    proc = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.Config.Image}}"],
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode().strip() or None


@pytest.fixture(scope="module")
def throwaway_pg() -> Iterator[Any]:
    if os.environ.get(_OPT_IN) != "1":
        pytest.skip(
            f"real-DB rehearsal: opt-in with {_OPT_IN}=1 (docker, prod's Postgres image); "
            "a skipped real-DB test is not coverage"
        )
    image = _docker_image("supabase-db")
    if image is None or _docker_image("supabase-rest") is None:
        pytest.skip("docker or the supabase-db / supabase-rest containers are not reachable")
    from tests.unit.test_database.learning_loop._pg import ThrowawayPg

    pg = ThrowawayPg(image=image)
    pg.start()
    try:
        yield pg
    finally:
        pg.stop()


_SEED = f"""
INSERT INTO ml_experiments (id, experiment_name, prediction_target) VALUES
 ('{EXP_INIT}', 'initiation_kisqali_goldstd_eval_v1', 'initiation_kisqali'),
 ('{EXP_HCP}', 'hcp_adoption_kisqali_goldstd_eval_v1', 'hcp_adoption_kisqali'),
 ('{EXP_CHAMP}', 'lane2310_champion_exp', 'lane2310_target'),
 ('{EXP_DUAL}', 'lane2310_dual_exp', 'lane2310_dual_target'),
 ('{EXP_EDGE}', 'lane2310_edge_exp', 'lane2310_edge_target');
INSERT INTO ml_model_registry (id, experiment_id, model_name, model_version, algorithm, stage,
                               is_champion, mlflow_run_id, registered_at) VALUES
 ('{PARENT}', '{EXP_INIT}', '{NAME}', '1.0', 'logistic_regression_calibrated', 'staging', false,
  NULL, '2026-09-28 03:03:46+00');
INSERT INTO ml_model_registry (id, experiment_id, model_name, model_version, algorithm, stage,
                               is_champion, mlflow_run_id, registered_at, retrain_of_id,
                               mlflow_model_version) VALUES
 ('{FAILED_ROW}', '{EXP_INIT}', '{NAME}', '1.0_retrained_20260928_1617_d18ac8',
  'logistic_regression_calibrated', 'archived', false, 'fef4480afb82435d894629decb4d73d5',
  '2026-09-28 16:19:35+00', '{PARENT}', 3),
 ('{CAND_A}', '{EXP_INIT}', '{NAME}', '1.0_retrained_20260928_1934_725a94',
  'logistic_regression_calibrated', 'candidate', false, '830aef651bba4bf5bb7202a96e614e7a',
  '2026-09-28 19:36:07+00', '{PARENT}', 4),
 ('{CAND_B}', '{EXP_INIT}', '{NAME}', '1.0_retrained_20260928_2036_6fbd91',
  'logistic_regression_calibrated', 'candidate', false, 'df85634d4315423d97bdf115ae02d4af',
  '2026-09-28 20:38:44+00', '{PARENT}', 5);
-- The backfill script's auc refresh: the parent's live tuple now sits after its candidates.
UPDATE ml_model_registry SET auc = 0.7123 WHERE id = '{PARENT}';

INSERT INTO ml_model_registry (id, experiment_id, model_name, model_version, algorithm, stage,
                               is_champion, registered_at, artifact_path) VALUES
 ('{HCP_REF}', '{EXP_HCP}', '{HCP_NAME}', '1.0', 'logistic_regression_calibrated', 'staging',
  false, '2026-09-28 03:06:38+00', '/app/data/ml_artifacts/hcp/model.pkl');
INSERT INTO ml_model_registry (id, experiment_id, model_name, model_version, algorithm, stage,
                               is_champion, registered_at, retrain_of_id) VALUES
 ('{HCP_CAND}', '{EXP_HCP}', '{HCP_NAME}', '1.0_retrained_20260929_0100_aaaaaa',
  'logistic_regression_calibrated', 'candidate', false, '2026-09-29 01:00:00+00', '{HCP_REF}'),
 ('{CHAMP_CAND}', '{EXP_CHAMP}', 'lane2310_champion_model', '2.0_retrained_x',
  'logistic_regression', 'candidate', true, '2026-09-29 02:00:00+00', NULL);

INSERT INTO ml_model_registry (id, experiment_id, model_name, model_version, algorithm, stage,
                               is_champion, registered_at) VALUES
 ('{DUAL_OLD}', '{EXP_DUAL}', '{DUAL_NAME}', '1.0', 'logistic_regression', 'development', false,
  '2026-09-01 00:00:00+00'),
 ('{DUAL_NEW}', '{EXP_DUAL}', '{DUAL_NAME}', '2.0', 'logistic_regression', 'staging', false,
  '2026-09-02 00:00:00+00'),
 ('{NULL_STAGE}', '{EXP_DUAL}', '{NULL_NAME}', '1.0', 'logistic_regression', NULL, false,
  '2026-09-01 00:00:00+00'),
 ('{NULL_CAND}', '{EXP_DUAL}', '{NULL_NAME}', '1.0_retrained_y', 'logistic_regression',
  'candidate', false, '2026-09-03 00:00:00+00'),
 ('{DEPRECATED}', '{EXP_DUAL}', '{NULL_NAME}', '0.9', 'logistic_regression', 'deprecated', false,
  '2026-09-04 00:00:00+00');

-- registered_at is nullable: an undated canonical row must not beat a dated one. And an
-- archived row that kept is_champion (transition_stage does not clear it on a direct archive).
INSERT INTO ml_model_registry (id, experiment_id, model_name, model_version, algorithm, stage,
                               is_champion, is_synthetic, registered_at) VALUES
 ('{UNDATED}', '{EXP_EDGE}', '{UNDATED_NAME}', '0.1', 'logistic_regression', 'staging', false,
  false, NULL),
 ('{DATED}', '{EXP_EDGE}', '{UNDATED_NAME}', '0.2', 'logistic_regression', 'staging', false,
  false, '2026-09-05 00:00:00+00'),
 ('{ARCHIVED_CHAMP}', '{EXP_EDGE}', 'lane2310_archived_champion', '1.0', 'logistic_regression',
  'archived', true, false, '2026-09-06 00:00:00+00');

-- A holdout metric the candidate already carries: a name-handle re-record must not delete it.
INSERT INTO ml_performance_metrics (model_id, metric_name, metric_value, source, measured_at)
VALUES ('{CAND_A}', 'auc_roc', 0.6100, 'holdout', '2026-09-28 19:40:00+00');
"""


@pytest.fixture(scope="module")
def conn(throwaway_pg):
    from tests.unit.test_database.learning_loop._pg import PgConn, apply_migration

    name = "t2310_readers_" + uuid.uuid4().hex[:8]
    throwaway_pg.rows("postgres", f"CREATE DATABASE {name} OWNER postgres")
    c = PgConn(throwaway_pg, name)
    c.execute(
        "CREATE TABLE IF NOT EXISTS public.schema_migrations "
        "(filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now());"
        "CREATE TYPE data_split_type AS ENUM "
        "('train', 'validation', 'test', 'holdout', 'unassigned');"
        "CREATE TYPE agent_name_enum AS ENUM ('orchestrator');"
        "CREATE TYPE brand_type AS ENUM ('Kisqali');"
        "CREATE TYPE region_type AS ENUM ('northeast');",
        user="postgres",
    )
    for base in ("database/ml/mlops_tables.sql", "database/ml/017_model_monitoring_tables.sql"):
        apply_migration(c, REPO / base)
    # Columns later migrations add that the readers under test select (069 is_synthetic,
    # 083 training_provenance); the verbatim base DDL predates them.
    c.execute(
        "ALTER TABLE ml_model_registry ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL "
        "DEFAULT false; ALTER TABLE ml_model_registry ADD COLUMN IF NOT EXISTS "
        "training_provenance TEXT;",
        user="postgres",
    )
    assert apply_migration(c, M159, record=M159.name) == "unwrapped"
    assert apply_migration(c, M160, record=M160.name) == "wrapped"
    assert apply_migration(c, M161, record=M161.name) == "wrapped"
    c.execute(_SEED, user="postgres")
    # Statistics as autovacuum keeps them on a live table: the planner then reads this small
    # table in physical order, where the updated parent sits after its candidates.
    c.execute("ANALYZE ml_model_registry;", user="postgres")
    c.execute(
        "GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;"
        "GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;"
        "GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;",
        user="postgres",
    )
    return c


@pytest.fixture(scope="module")
def rest(conn):
    from tests.unit.test_database.learning_loop.test_ml_registry_promotion_gate_realdb import (
        ThrowawayRest,
    )

    server = ThrowawayRest(conn)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def aclient(rest):
    return rest.service_role_client()


def _one(conn, sql: str) -> str:
    (value,) = conn.rows(sql)
    return value


# --------------------------------------------------------------------------------------------
# The fixture reproduces the hazard (so a green below is the fix, not the heap order)
# --------------------------------------------------------------------------------------------


async def test_the_fixture_reproduces_the_unordered_hazard(conn, aclient):
    """The pre-#2310 query shape, through the same PostgREST and planner the readers use."""
    res = await (
        aclient.table("ml_model_registry").select("id").eq("model_name", NAME).limit(1).execute()
    )
    first = res.data[0]["id"]
    assert first in {FAILED_ROW, CAND_A, CAND_B}, (
        "the unordered name lookup must answer with a non-canonical row"
    )
    newest = _one(
        conn,
        f"select id from ml_model_registry where model_name = '{NAME}' "
        "order by registered_at desc limit 1",
    )
    assert newest == CAND_B


# --------------------------------------------------------------------------------------------
# 1. drift_monitoring._resolve_model_id: canonical for names, exact for uuids
# --------------------------------------------------------------------------------------------


async def test_name_handle_resolves_to_the_canonical_row(aclient):
    from src.repositories.drift_monitoring import _resolve_model_id

    assert await _resolve_model_id(aclient, NAME) == PARENT


async def test_uuid_handle_passes_through_even_for_a_candidate(aclient):
    """The exact-id path (provenance, completion, lineage) is unchanged."""
    from src.repositories.drift_monitoring import _resolve_model_id

    assert await _resolve_model_id(aclient, CAND_A) == CAND_A


async def test_version_handle_of_a_candidate_is_not_resolved_by_the_name_resolver(aclient):
    """A candidate's unique version label is still a name-style handle: the canonical resolver
    does not return a candidate for it. Callers that mean the candidate pass its uuid."""
    from src.repositories.drift_monitoring import _resolve_model_id

    assert await _resolve_model_id(aclient, "1.0_retrained_20260928_1934_725a94") is None


async def test_more_than_one_canonical_row_is_deterministic_and_logged(aclient, caplog):
    from src.repositories.drift_monitoring import _resolve_model_id

    with caplog.at_level(logging.WARNING):
        assert await _resolve_model_id(aclient, DUAL_NAME) == DUAL_NEW
    assert any(DUAL_NAME in r.getMessage() for r in caplog.records)


async def test_a_null_stage_row_is_canonical_and_deprecated_is_not(aclient):
    """stage has a default but no NOT NULL: a NULL-stage row is not a candidate. The newer
    candidate and the newest (deprecated) row both lose to it."""
    from src.repositories.drift_monitoring import _resolve_model_id

    assert await _resolve_model_id(aclient, NULL_NAME) == NULL_STAGE


async def test_an_undated_canonical_row_never_beats_a_dated_one(aclient, patched_async_factory):
    """registered_at is nullable and DESC puts NULL first in Postgres: NULLS LAST is required."""
    from src.api.routes import explain
    from src.repositories.drift_monitoring import _resolve_model_id

    assert await _resolve_model_id(aclient, UNDATED_NAME) == DATED
    assert await explain._resolve_model_registry_id(UNDATED_NAME) == DATED


async def test_exact_name_version_resolver(aclient):
    from src.repositories.model_registry_roles import resolve_model_id_by_name_version

    assert await resolve_model_id_by_name_version(aclient, NAME, "1.0") == PARENT
    assert (
        await resolve_model_id_by_name_version(aclient, NAME, "1.0_retrained_20260928_1934_725a94")
        == CAND_A
    )
    assert await resolve_model_id_by_name_version(aclient, NAME, "9.9") is None


async def test_metric_recorder_by_name_rewrites_the_canonical_row_only(conn, aclient):
    """record_run DELETEs the prior rows of the resolved id: by name that must be the parent."""
    from src.mlops.gold_standard_eval.recorder import MetricRecorder
    from src.repositories.drift_monitoring import PerformanceMetricRepository

    recorder = MetricRecorder(PerformanceMetricRepository(aclient))
    ts = datetime(2026, 9, 1, tzinfo=timezone.utc)
    await recorder.record_run(NAME, [(ts, {"auc_roc": 0.71}, 500)], source="holdout")

    by_model = dict(
        r.split("|")
        for r in conn.rows(
            "select model_id, count(*) from ml_performance_metrics where source = 'holdout' "
            "group by model_id"
        )
    )
    assert by_model.get(CAND_A) == "1", "the candidate's own holdout row was deleted"
    assert by_model.get(PARENT) == "1"


# --------------------------------------------------------------------------------------------
# 2. explain: the registry id and latest version by name exclude candidates
# --------------------------------------------------------------------------------------------


@pytest.fixture
def patched_async_factory(monkeypatch, rest):
    from src.memory.services import factories

    client = rest.service_role_client()

    async def _async_client() -> Any:
        return client

    monkeypatch.setattr(factories, "get_async_supabase_client", _async_client)
    return client


async def test_explain_registry_id_is_the_canonical_row(patched_async_factory):
    from src.api.routes import explain

    assert await explain._resolve_model_registry_id(NAME) == PARENT


async def test_explain_latest_version_ignores_candidates(patched_async_factory):
    from src.api.routes import explain

    versions = await explain._get_latest_versions_by_model_type()
    assert versions["initiation"] == "1.0"
    assert versions["hcp_adoption"] == "1.0"


# --------------------------------------------------------------------------------------------
# 3. backfill script: the registry auc refresh updates exactly one row, by id
# --------------------------------------------------------------------------------------------


def _script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_lane2310_{name}", REPO / "scripts" / name)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def test_backfill_auc_refresh_touches_only_the_canonical_row(conn, aclient):
    backfill = _script("backfill_goldstd_holdout_metrics.py")
    before = dict(
        r.split("|")
        for r in conn.rows(
            f"select id, coalesce(auc::text, '-') from ml_model_registry where model_name = '{NAME}'"
        )
    )
    target = await backfill._registry_target(aclient, NAME)
    assert target == PARENT
    await backfill._update_registry_auc(aclient, target, 0.6543)
    after = dict(
        r.split("|")
        for r in conn.rows(
            f"select id, coalesce(auc::text, '-') from ml_model_registry where model_name = '{NAME}'"
        )
    )
    assert after[PARENT] == "0.6543"
    assert {k: v for k, v in after.items() if k != PARENT} == {
        k: v for k, v in before.items() if k != PARENT
    }


# --------------------------------------------------------------------------------------------
# 4. promote_hcp_adoption_champions: a candidate does not block the owner's promotion
# --------------------------------------------------------------------------------------------


async def test_promotion_row_fetch_ignores_the_candidate(aclient):
    promote = _script("promote_hcp_adoption_champions.py")
    row = await promote._fetch_registry_row(aclient, HCP_NAME)
    assert row is not None and row["id"] == HCP_REF


async def test_promotion_row_fetch_still_refuses_two_canonical_rows(aclient):
    promote = _script("promote_hcp_adoption_champions.py")
    with pytest.raises(RuntimeError, match="ambiguous"):
        await promote._fetch_registry_row(aclient, DUAL_NAME)


# --------------------------------------------------------------------------------------------
# 6. repository: candidates appear only when asked for
# --------------------------------------------------------------------------------------------


async def test_champion_lookup_skips_a_candidate_unless_asked(aclient):
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo = MLModelRegistryRepository(supabase_client=aclient)
    assert await repo.get_champion_model(experiment_id=uuid.UUID(EXP_CHAMP)) is None
    asked = await repo.get_champion_model(
        experiment_id=uuid.UUID(EXP_CHAMP), include_candidates=True
    )
    assert asked is not None and str(asked.id) == CHAMP_CAND


async def test_an_archived_champion_is_never_the_champion(aclient):
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo = MLModelRegistryRepository(supabase_client=aclient)
    for include in (False, True):
        got = await repo.get_champion_model(
            experiment_id=uuid.UUID(EXP_EDGE), include_candidates=include
        )
        assert got is None, (include, got)


async def test_models_by_stage_returns_candidates_only_for_the_candidate_stage(aclient):
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo = MLModelRegistryRepository(supabase_client=aclient)
    staging = {str(m.id) for m in await repo.get_models_by_stage("staging")}
    assert PARENT in staging and not staging & {CAND_A, CAND_B, HCP_CAND}
    candidates = {str(m.id) for m in await repo.get_models_by_stage("candidate")}
    assert {CAND_A, CAND_B, HCP_CAND, CHAMP_CAND} <= candidates


# --------------------------------------------------------------------------------------------
# 7. already stage-filtered readers: pinned, no code change
# --------------------------------------------------------------------------------------------


def test_goldstd_kpi_selector_excludes_candidates(rest):
    from src.kpi.goldstd_model_perf import _registry_query

    ids = {r["id"] for r in _registry_query(rest.sync_service_role_client()).execute().data}
    assert PARENT in ids and HCP_REF in ids
    assert not ids & {CAND_A, CAND_B, HCP_CAND, FAILED_ROW}


async def test_drift_and_retraining_sweep_models_exclude_candidates(rest):
    from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector

    connector = SupabaseDataConnector()
    connector._client, connector._initialized = rest.sync_service_role_client(), True
    ids = {r["id"] for r in await connector.get_available_models(stages=["production", "staging"])}
    assert PARENT in ids and not ids & {CAND_A, CAND_B, HCP_CAND}


async def test_models_status_names_exclude_candidates_only_rows(monkeypatch, rest):
    """/models/status lists names: a name that exists ONLY as a candidate must not appear."""
    from src.api.routes import predictions

    names = await _with_async_factory(
        monkeypatch, rest, predictions._resolve_production_model_names
    )
    assert "lane2310_champion_model" not in names
    assert NAME in names


def test_health_facts_exclude_candidates(monkeypatch, rest):
    from src.api.routes import health_score

    monkeypatch.setattr(health_score, "_health_source_client", rest.sync_service_role_client)
    facts = health_score._fetch_model_registry_facts()
    assert PARENT in facts and not set(facts) & {CAND_A, CAND_B, HCP_CAND}


async def _with_async_factory(monkeypatch, rest, fn):
    import src.repositories as repositories
    from src.memory.services import factories

    aclient = rest.service_role_client()

    async def _async_client() -> Any:
        return aclient

    monkeypatch.setattr(factories, "get_async_supabase_client", _async_client)
    monkeypatch.setattr(repositories, "get_supabase_client", rest.sync_service_role_client)
    return await fn()


# --------------------------------------------------------------------------------------------
# SQL: get_latest_model (migration 161)
# --------------------------------------------------------------------------------------------


async def test_get_latest_model_rpc_returns_the_canonical_row(aclient):
    res = await aclient.rpc(
        "get_latest_model", {"p_experiment_name": "initiation_kisqali_goldstd_eval_v1"}
    ).execute()
    assert [r["model_id"] for r in res.data] == [PARENT]
    edge = await aclient.rpc(
        "get_latest_model", {"p_experiment_name": "lane2310_edge_exp"}
    ).execute()
    assert [r["model_id"] for r in edge.data] == [DATED]


def test_161_rollback_restores_the_old_body_and_reapplies(conn):
    from tests.unit.test_database.learning_loop._pg import apply_migration

    q = "select model_id from get_latest_model('initiation_kisqali_goldstd_eval_v1')"
    proc = conn.pg.run_script(conn.db, R161.read_bytes(), single_transaction=True, user="postgres")
    assert proc.returncode == 0, proc.stderr
    assert conn.rows(q) == [CAND_B], "the pre-161 body answers with the newest candidate"
    assert conn.rows(f"select count(*) from schema_migrations where filename = '{M161.name}'") == [
        "0"
    ]
    assert apply_migration(conn, M161, record=M161.name) == "wrapped"
    assert conn.rows(q) == [PARENT]


def test_seed_names_are_unique_per_version(conn):
    """(model_name, model_version) is unique: the exact resolver can never be ambiguous."""
    dupes = conn.rows(
        "select model_name, model_version from ml_model_registry group by 1, 2 having count(*) > 1"
    )
    assert dupes == []
    assert re.fullmatch(r"\d+", _one(conn, "select count(*) from ml_model_registry"))
