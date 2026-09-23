"""The #968 promotion gate on ``ml_model_registry``, through real PostgREST into real Postgres (#2259).

Opt-in: ``E2I_DB_INTEGRATION=1``. The repository under test is the production
``MLModelRegistryRepository``, driven through a real ``AsyncPostgrestClient`` against a throwaway
PostgREST container of prod's own image (``docker inspect supabase-rest``), authenticated with a
``service_role`` JWT the way the backend is, over a clone of the post-deploy schema. Nothing is
faked: the enum that rejects ``'Production'``, the ``tr_single_champion`` trigger and the grants
are the ones prod has.

Why a real server matters here: the deploy agent passes MLflow-cased stages
(``"Production"``), which the ``model_stage_enum`` rejects. A mocked client accepts any string,
so it could never show that the gate was dead and no promotion had ever persisted.
"""

from __future__ import annotations

import atexit
import os
import secrets
import socket
import subprocess
import time
import urllib.request
import uuid
from typing import Any, Dict, Iterator, Optional

import pytest

from src.repositories.ml_experiment import MLModelRegistryRepository
from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(not _pg.db_integration_enabled(), reason=_pg.OPT_IN_SKIP_REASON),
    pytest.mark.timeout(300),
]

PROD_REST_CONTAINER = "supabase-rest"
MIGRATION_158 = (
    _pg.REPO_ROOT / "database" / "migrations" / "158_backfill_provable_training_provenance.sql"
)
# The two rows rule D of the #2259 investigation proves (prediction_synthesizer_deploy, 2026-06-10),
# as (id, model_name) on prod.
PROVABLE = (
    ("5fd7826b-28d7-491b-b9b1-8b5494dbe1ff", "csu_treatment_initiation_lr_full_v1"),
    ("d765b451-12df-46df-955f-63359b506b52", "csu_treatment_initiation_lr_balanced_v1"),
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ThrowawayRest:
    """PostgREST (prod's image) in front of one throwaway database, reaped like ThrowawayPg."""

    def __init__(self, conn: _pg.PgConn):
        self.conn = conn
        self.name = _pg.CONTAINER_PREFIX + "rest-" + secrets.token_hex(4)
        self.port = _free_port()
        self.jwt_secret = secrets.token_hex(32)

    def start(self, ready_timeout_s: int = 60) -> None:
        # Registered BEFORE the container exists; any failure removes it before re-raising
        # (the ThrowawayPg contract). The name prefix also lets reap_orphans() find a leak.
        atexit.register(self.stop)
        try:
            self._start(ready_timeout_s)
        except BaseException:
            self.stop()
            raise

    def _start(self, ready_timeout_s: int) -> None:
        image = (
            subprocess.run(
                ["docker", "inspect", PROD_REST_CONTAINER, "--format", "{{.Config.Image}}"],
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        pg = self.conn.pg
        db_uri = f"postgres://postgres:{pg._password}@127.0.0.1:{pg.host_port}/{self.conn.db}"
        env = {
            **os.environ,
            "PGRST_DB_URI": db_uri,
            "PGRST_DB_SCHEMAS": "public",
            "PGRST_DB_ANON_ROLE": "anon",
            "PGRST_JWT_SECRET": self.jwt_secret,
            "PGRST_DB_USE_LEGACY_GUCS": "false",
            "PGRST_SERVER_HOST": "127.0.0.1",
            "PGRST_SERVER_PORT": str(self.port),
        }
        passed = [a for k in env if k.startswith("PGRST_") for a in ("-e", k)]
        proc = subprocess.run(
            ["docker", "run", "-d", "--name", self.name, "--network", "host",
             "--label", f"{_pg.OWNER_LABEL}={_pg.owner_identity()}",
             "--memory", "256m", "--memory-swap", "256m", *passed, image],
            env=env,
            capture_output=True,
        )  # fmt: skip
        if proc.returncode != 0:
            raise _pg.DbFixtureError(f"postgrest docker run failed: {proc.stderr.decode()}")
        # Listening is not ready: until the schema cache has loaded every request answers 503
        # PGRST002. A table read only succeeds once it has.
        probe = f"http://127.0.0.1:{self.port}/ml_model_registry?limit=0"
        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(probe, headers={"Authorization": self._bearer()}),
                    timeout=2,
                ) as resp:
                    if resp.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise _pg.DbFixtureError(f"{self.name} schema cache not ready after {ready_timeout_s}s")

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        if subprocess.run(["docker", "inspect", self.name], capture_output=True).returncode == 0:
            raise _pg.DbFixtureError(f"could not remove throwaway container {self.name}")

    def _bearer(self) -> str:
        import jwt

        token = jwt.encode(
            {"role": "service_role", "exp": int(time.time()) + 3600},
            self.jwt_secret,
            algorithm="HS256",
        )
        return f"Bearer {token}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": self._bearer(),
        }

    def service_role_client(self) -> Any:
        from postgrest import AsyncPostgrestClient

        return AsyncPostgrestClient(f"http://127.0.0.1:{self.port}", headers=self._headers())

    def sync_service_role_client(self) -> Any:
        """The sync flavour, for the readers that use the sync Supabase client."""
        from postgrest import SyncPostgrestClient

        return SyncPostgrestClient(f"http://127.0.0.1:{self.port}", headers=self._headers())


@pytest.fixture
def registry_db(clone_db) -> _pg.PgConn:
    return clone_db("registry_gate")


@pytest.fixture
def rest(registry_db: _pg.PgConn) -> Iterator[ThrowawayRest]:
    server = ThrowawayRest(registry_db)
    try:
        server.start()
        yield server
    finally:
        server.stop()


def _q(value: Optional[str]) -> str:
    return "NULL" if value is None else "'" + value.replace("'", "''") + "'"


def _experiment(conn: _pg.PgConn, name: str) -> str:
    exp_id = str(uuid.uuid4())
    conn.execute(
        "insert into ml_experiments (id, experiment_name, prediction_target) "
        f"values ('{exp_id}', {_q(name)}, 'lane_2259_target')",
        user="postgres",
    )
    return exp_id


def _model(
    conn: _pg.PgConn,
    experiment_id: str,
    name: str,
    *,
    stage: str,
    provenance: Optional[str],
    version: str = "1.0",
    champion: bool = False,
    is_synthetic: bool = False,
    model_id: Optional[str] = None,
    artifact_path: Optional[str] = "/app/data/ml_artifacts/lane_2259/model.pkl",
) -> str:
    model_id = model_id or str(uuid.uuid4())
    conn.execute(
        "insert into ml_model_registry (id, experiment_id, model_name, model_version, algorithm, "
        "stage, is_champion, is_synthetic, training_provenance, artifact_path) values ("
        f"'{model_id}', '{experiment_id}', {_q(name)}, {_q(version)}, 'logistic_regression', "
        f"'{stage}', {str(champion).lower()}, {str(is_synthetic).lower()}, {_q(provenance)}, "
        f"{_q(artifact_path)})",
        user="postgres",
    )
    return model_id


def _row(conn: _pg.PgConn, model_id: str) -> Dict[str, str]:
    (line,) = conn.rows(
        "select stage || '|' || is_champion || '|' || coalesce(training_provenance, '<null>') "
        f"|| '|' || (promoted_at is not null) from ml_model_registry where id = '{model_id}'"
    )
    stage, champion, provenance, promoted = line.split("|")
    return {"stage": stage, "is_champion": champion, "provenance": provenance, "promoted": promoted}


# ---------------------------------------------------------------------------
# transition_stage: the deploy agent's MLflow-cased stage reaches the DB enum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mlflow_stage, db_stage",
    [("Production", "production"), ("Staging", "staging"), ("Shadow", "shadow"),
     ("Archived", "archived")],
)  # fmt: skip
async def test_mlflow_cased_stage_persists_as_the_enum_value(
    registry_db: _pg.PgConn, rest: ThrowawayRest, mlflow_stage: str, db_stage: str
) -> None:
    """model_deployer passes ``state["current_stage"]`` (MLflow casing) straight through."""
    exp = _experiment(registry_db, "lane_2259_casing")
    model_id = _model(registry_db, exp, "lane_2259_casing_model", stage="development",
                      provenance="real")  # fmt: skip
    repo = MLModelRegistryRepository(supabase_client=rest.service_role_client())

    assert await repo.transition_stage(uuid.UUID(model_id), mlflow_stage) is True

    assert _row(registry_db, model_id)["stage"] == db_stage


async def test_unknown_stage_is_refused_before_any_write(
    registry_db: _pg.PgConn, rest: ThrowawayRest
) -> None:
    exp = _experiment(registry_db, "lane_2259_unknown")
    model_id = _model(registry_db, exp, "lane_2259_unknown_model", stage="staging",
                      provenance="real")  # fmt: skip
    repo = MLModelRegistryRepository(supabase_client=rest.service_role_client())

    with pytest.raises(ValueError, match="stage"):
        await repo.transition_stage(uuid.UUID(model_id), "candidate")

    assert _row(registry_db, model_id)["stage"] == "staging"


# ---------------------------------------------------------------------------
# archive_existing: only the promoted model's own earlier production versions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage_spelling", ["production", "Production"])
async def test_promotion_archives_only_the_same_model_name(
    registry_db: _pg.PgConn, rest: ThrowawayRest, stage_spelling: str
) -> None:
    """The first real promotion must not archive another model's serving champion."""
    other_exp = _experiment(registry_db, "hcp_adoption_kisqali_goldstd_eval_v1")
    other = _model(registry_db, other_exp, "hcp_adoption_kisqali_goldstd_lr_v1",
                   stage="production", provenance="synthetic_gold", champion=True)  # fmt: skip
    exp = _experiment(registry_db, "lane_2259_scope")
    old = _model(registry_db, exp, "lane_2259_scope_model", version="1.0",
                 stage="production", provenance="real", champion=True)  # fmt: skip
    new = _model(registry_db, exp, "lane_2259_scope_model", version="2.0",
                 stage="staging", provenance="real")  # fmt: skip
    repo = MLModelRegistryRepository(supabase_client=rest.service_role_client())

    assert await repo.transition_stage(uuid.UUID(new), stage_spelling) is True

    assert _row(registry_db, new)["stage"] == "production"
    assert _row(registry_db, new)["is_champion"] == "true"
    assert _row(registry_db, old)["stage"] == "archived"
    assert _row(registry_db, other) == {
        "stage": "production",
        "is_champion": "true",
        "provenance": "synthetic_gold",
        "promoted": "false",
    }


# ---------------------------------------------------------------------------
# The gate itself: fail closed on unknown provenance, keep refusing synthetic_gold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage_spelling", ["production", "Production"])
@pytest.mark.parametrize("provenance", [None, "synthetic_gold"])
async def test_production_is_refused_without_provable_real_provenance(
    registry_db: _pg.PgConn, rest: ThrowawayRest, stage_spelling: str, provenance: Optional[str]
) -> None:
    exp = _experiment(registry_db, "lane_2259_gate")
    serving = _model(registry_db, exp, "lane_2259_gate_model", version="1.0",
                     stage="production", provenance="real", champion=True)  # fmt: skip
    candidate = _model(registry_db, exp, "lane_2259_gate_model", version="2.0",
                       stage="staging", provenance=provenance)  # fmt: skip
    repo = MLModelRegistryRepository(supabase_client=rest.service_role_client())

    with pytest.raises(ValueError, match="training_provenance"):
        await repo.transition_stage(uuid.UUID(candidate), stage_spelling)

    # Refused before ANY write: the candidate did not move and nothing was archived.
    assert _row(registry_db, candidate)["stage"] == "staging"
    assert _row(registry_db, serving)["stage"] == "production"


@pytest.mark.parametrize("provenance", [None, "synthetic_gold"])
async def test_non_production_stages_are_not_gated(
    registry_db: _pg.PgConn, rest: ThrowawayRest, provenance: Optional[str]
) -> None:
    """Staging is where unproven models are supposed to live; only production is gated."""
    exp = _experiment(registry_db, "lane_2259_staging")
    model_id = _model(registry_db, exp, "lane_2259_staging_model", stage="development",
                      provenance=provenance)  # fmt: skip
    repo = MLModelRegistryRepository(supabase_client=rest.service_role_client())

    assert await repo.transition_stage(uuid.UUID(model_id), "Staging") is True
    assert _row(registry_db, model_id)["stage"] == "staging"


# ---------------------------------------------------------------------------
# The deploy agent: the ONLY caller of transition_stage, end to end
# ---------------------------------------------------------------------------


async def _store(rest: ThrowawayRest, monkeypatch, model_id: str) -> Dict[str, Any]:
    """``ModelDeployerAgent._store_to_database`` with its client factory pointed at the server.

    Only the transport is swapped: the factory would build a Supabase client for prod, and this
    returns a PostgREST client for the throwaway database. Everything after it is the real code.
    """
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent
    from src.memory.services import factories

    client = rest.service_role_client()

    async def _client() -> Any:
        return client

    monkeypatch.setattr(factories, "get_async_supabase_client", _client)
    output: Dict[str, Any] = {"deployment_successful": True, "status": "completed"}
    state = {
        "model_registry_id": model_id,
        "deployment_name": "lane_2259_deploy",
        "target_environment": "production",
        "promotion_successful": True,
        # What promote_stage leaves behind: the MLflow-cased promotion target.
        "current_stage": "Production",
        "deployment_action": "promote",
    }
    await ModelDeployerAgent()._store_to_database(output, state)
    return output


async def test_deploy_agent_promotion_reaches_the_registry(
    registry_db: _pg.PgConn, rest: ThrowawayRest, monkeypatch
) -> None:
    """A real-provenance promotion through the agent persists (it never had, #2259)."""
    exp = _experiment(registry_db, "lane_2259_agent_ok")
    model_id = _model(registry_db, exp, "lane_2259_agent_ok_model", stage="staging",
                      provenance="real")  # fmt: skip

    output = await _store(rest, monkeypatch, model_id)

    assert output.get("db_persist_skipped_reason") is None
    assert output["db_persisted"] is True
    assert "promotion_refused_reason" not in output
    assert output["deployment_successful"] is True
    assert _row(registry_db, model_id)["stage"] == "production"
    assert registry_db.rows(
        f"select status from ml_deployments where model_registry_id = '{model_id}'"
    ) == ["active"]


async def test_deploy_agent_surfaces_a_refused_promotion(
    registry_db: _pg.PgConn, rest: ThrowawayRest, monkeypatch, caplog
) -> None:
    """A refusal is a first-class outcome of the deploy, not a buried persistence failure."""
    exp = _experiment(registry_db, "lane_2259_agent_refused")
    model_id = _model(registry_db, exp, "lane_2259_agent_refused_model", stage="staging",
                      provenance=None)  # fmt: skip

    with caplog.at_level("ERROR"):
        output = await _store(rest, monkeypatch, model_id)

    assert "training_provenance" in output["promotion_refused_reason"]
    # The ml_deployments row WAS written and confirmed; only the promotion was refused.
    assert output["db_persisted"] is True
    assert output.get("db_persist_skipped_reason") is None
    assert _row(registry_db, model_id)["stage"] == "staging"
    # ...and the deploy does not report (or record) a success it did not have.
    assert output["deployment_successful"] is False
    assert output["status"] == "failed"
    assert registry_db.rows(
        f"select status from ml_deployments where model_registry_id = '{model_id}'"
    ) == ["pending"]
    assert any(
        r.levelname == "ERROR" and "training_provenance" in r.getMessage() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# The direct production writers that bypass transition_stage (P3/P4/P5)
# ---------------------------------------------------------------------------


async def test_serving_deploy_registers_its_rows_as_synthetic_gold(
    registry_db: _pg.PgConn, rest: ThrowawayRest, tmp_path
) -> None:
    """prediction_synthesizer_deploy trains only on synthetic_v2 scenario C; its rows say so."""
    from src.mlops.prediction_synthesizer_deploy import TrainedModel, register_deployed_models

    artifact = tmp_path / "lane_2259_csu.pkl"
    artifact.write_bytes(b"artifact")
    model = TrainedModel(model_name="lane_2259_csu_lr", model=None, feature_names=["x"],
                         algorithm="logistic_regression", auc=0.7, n_features=1,
                         training_samples=10)  # fmt: skip

    await register_deployed_models(
        rest.service_role_client(), [model], {model.model_name: str(artifact)},
        experiment_name="lane_2259_csu_live",
    )  # fmt: skip

    (line,) = registry_db.rows(
        "select stage || '|' || coalesce(training_provenance, '<null>') from ml_model_registry "
        "where model_name = 'lane_2259_csu_lr'"
    )
    assert line == "production|synthetic_gold"


async def test_register_model_row_refuses_production_without_provenance(
    registry_db: _pg.PgConn, rest: ThrowawayRest, tmp_path
) -> None:
    from src.mlops.prediction_synthesizer_deploy import register_model_row

    artifact = tmp_path / "lane_2259_null.pkl"
    artifact.write_bytes(b"artifact")
    exp = _experiment(registry_db, "lane_2259_register_row")

    with pytest.raises(ValueError, match="training_provenance"):
        await register_model_row(
            rest.service_role_client(), experiment_id=exp, model_name="lane_2259_null_lr",
            model_version="1.0", algorithm="logistic_regression", artifact_path=str(artifact),
            auc=0.7, feature_count=1, stage="production",
        )  # fmt: skip

    assert registry_db.rows(
        "select count(*) from ml_model_registry where model_name = 'lane_2259_null_lr'"
    ) == ["0"]


async def test_champion_promotion_holds_a_null_provenance_row(
    registry_db: _pg.PgConn, rest: ThrowawayRest, monkeypatch, capsys
) -> None:
    """The weekly hcp_adoption promotion reads provenance from the row and holds a NULL one.

    Scoring is replaced (it loads the artifacts and the live cohort, neither of which exists in a
    throwaway database); the registry read, the hold decision and the write are the real ones.
    """
    import scripts.promote_hcp_adoption_champions as promo

    ids = {}
    for brand in promo.BRANDS:
        exp = _experiment(registry_db, f"hcp_adoption_{brand}_goldstd_eval_v1")
        ids[brand] = _model(
            registry_db, exp, f"hcp_adoption_{brand}_goldstd_lr_v1", stage="staging",
            provenance=None if brand == "kisqali" else "synthetic_gold",
        )  # fmt: skip

    async def scored(client: Any, brand: str, artifact_path: str) -> Dict[str, Any]:
        metrics = {"auc_roc": 0.8, "accuracy": 0.75, "pr_auc": 0.7, "brier_score": 0.2,
                   "calibration_slope": 1.0}  # fmt: skip
        return {"metrics": metrics, "intercept": 0.0, "n": 1000, "prevalence": 0.4}

    async def stored(client: Any, model_id: str) -> Dict[str, Any]:
        return {"auc_roc": 0.8, "accuracy": 0.75}

    monkeypatch.setattr(promo, "_score_artifact", scored)
    monkeypatch.setattr(promo, "_stored_holdout", stored)

    assert await promo.run(rest.service_role_client(), execute=True, only_brand=None) == 0

    assert _row(registry_db, ids["kisqali"])["stage"] == "staging"
    assert _row(registry_db, ids["fabhalta"])["stage"] == "production"
    assert _row(registry_db, ids["remibrutinib"])["stage"] == "production"
    out = capsys.readouterr().out
    assert "HOLD hcp_adoption_kisqali_goldstd_lr_v1" in out and "training_provenance" in out


# ---------------------------------------------------------------------------
# Migration 158: the provable-only backfill
# ---------------------------------------------------------------------------


def _csu_row(
    conn: _pg.PgConn, exp: str, model_id: str, name: str, provenance: Optional[str]
) -> str:
    return _model(conn, exp, name, stage="archived", provenance=provenance, model_id=model_id,
                  artifact_path=f"/app/data/ml_artifacts/csu_treatment_initiation/{name}.pkl")  # fmt: skip


def _provenance(conn: _pg.PgConn, model_id: str) -> str:
    return _row(conn, model_id)["provenance"]


def test_migration_158_backfills_only_the_two_provable_rows(registry_db: _pg.PgConn) -> None:
    exp = _experiment(registry_db, "csu_treatment_initiation_live_v1")
    full = _csu_row(registry_db, exp, *PROVABLE[0], None)
    balanced = _csu_row(registry_db, exp, *PROVABLE[1], None)
    # Same writer and experiment, not one of the two ids: the id list is the scope.
    lookalike = _csu_row(registry_db, exp, str(uuid.uuid4()), "csu_treatment_initiation_lr_x_v1",
                         None)  # fmt: skip
    # Rule E: fabricated synthetic-generator metadata has no training to describe.
    gen_exp = _experiment(registry_db, "synth_remibrutinib_exp_0001")
    fabricated = _model(registry_db, gen_exp, "synth_remibrutinib_exp_0001_model_0",
                        stage="production", provenance=None, is_synthetic=True,
                        artifact_path=None)  # fmt: skip

    _pg.apply_migration(registry_db, MIGRATION_158)

    assert _provenance(registry_db, full) == "synthetic_gold"
    assert _provenance(registry_db, balanced) == "synthetic_gold"
    assert _provenance(registry_db, lookalike) == "<null>"
    assert _provenance(registry_db, fabricated) == "<null>"
    # Idempotent: a second application changes nothing.
    _pg.apply_migration(registry_db, MIGRATION_158)
    assert _provenance(registry_db, full) == "synthetic_gold"


def test_migration_158_never_overwrites_a_provenance_already_set(registry_db: _pg.PgConn) -> None:
    exp = _experiment(registry_db, "csu_treatment_initiation_live_v1")
    healed = _csu_row(registry_db, exp, *PROVABLE[0], "real")

    _pg.apply_migration(registry_db, MIGRATION_158)

    assert _provenance(registry_db, healed) == "real"


def test_migration_158_skips_a_row_that_no_longer_matches_the_proof(
    registry_db: _pg.PgConn,
) -> None:
    """The ids alone are not the proof: the writer's experiment and artifact location are."""
    other = _experiment(registry_db, "some_other_experiment")
    moved = _csu_row(registry_db, other, *PROVABLE[0], None)

    _pg.apply_migration(registry_db, MIGRATION_158)

    assert _provenance(registry_db, moved) == "<null>"


async def test_the_database_write_enforces_the_gate_not_only_the_read(
    registry_db: _pg.PgConn, rest: ThrowawayRest
) -> None:
    """A stale read (provenance changed after get_by_id) must not promote, nor archive anything."""
    from src.repositories.ml_experiment import MLModelRegistry

    exp = _experiment(registry_db, "lane_2259_stale")
    serving = _model(registry_db, exp, "lane_2259_stale_model", version="1.0",
                     stage="production", provenance="real", champion=True)  # fmt: skip
    candidate = _model(registry_db, exp, "lane_2259_stale_model", version="2.0",
                       stage="staging", provenance=None)  # fmt: skip
    repo = MLModelRegistryRepository(supabase_client=rest.service_role_client())

    async def stale(model_id: str, **_: Any) -> MLModelRegistry:
        return MLModelRegistry(model_name="lane_2259_stale_model", training_provenance="real")

    repo.get_by_id = stale  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="training_provenance"):
        await repo.transition_stage(uuid.UUID(candidate), "Production")

    assert _row(registry_db, candidate)["stage"] == "staging"
    assert _row(registry_db, serving)["stage"] == "production"


def test_migration_158_rollback_undoes_it_and_lets_it_reapply(registry_db: _pg.PgConn) -> None:
    exp = _experiment(registry_db, "csu_treatment_initiation_live_v1")
    full = _csu_row(registry_db, exp, *PROVABLE[0], None)
    key = MIGRATION_158.name
    _pg.apply_migration(registry_db, MIGRATION_158, record=key)
    assert _provenance(registry_db, full) == "synthetic_gold"

    rollback = MIGRATION_158.with_name("rollback_" + MIGRATION_158.name)
    proc = registry_db.pg.run_script(
        registry_db.db, rollback.read_bytes(), single_transaction=True, user="postgres"
    )
    assert proc.returncode == 0, proc.stderr.decode()

    assert _provenance(registry_db, full) == "<null>"
    # The ledger row goes too, so the next deploy re-applies 158 instead of skipping it.
    assert registry_db.rows(f"select count(*) from schema_migrations where filename = '{key}'") == [
        "0"
    ]


# ---------------------------------------------------------------------------
# The generator exemption is safe only if no serving reader can surface its rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_synthetic", [True, False])
async def test_serving_readers_never_surface_a_synthetic_production_champion(
    registry_db: _pg.PgConn, rest: ThrowawayRest, monkeypatch, is_synthetic: bool
) -> None:
    """Each serving reader, run as prod runs it, against a row only is_synthetic keeps out.

    Prod sets E2I_INCLUDE_SYNTHETIC=true (measured on e2i_api 2026-09-23), which turns the shared
    ``apply_provenance_filter`` into a no-op, so the test sets it too. The row is a production
    champion with an artifact and a real-looking name, so every other predicate admits it. The
    ``is_synthetic=False`` case is the control: each reader DOES surface that row, so the absence
    in the synthetic case is the synthetic exclusion at work, not a missed match.
    """
    import src.repositories as repositories
    from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector
    from src.agents.orchestrator.nodes import dispatcher
    from src.api.routes import explain, health_score, predictions
    from src.memory.services import factories
    from src.services.hcp_segment_likelihood import (
        ChampionNotPromotedError,
        resolve_hcp_adoption_champion,
    )

    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    exp = _experiment(registry_db, "hcp_adoption_kisqali_goldstd_eval_v1")
    name = "hcp_adoption_kisqali_goldstd_lr_v1"
    model_id = _model(registry_db, exp, name, stage="production", provenance=None,
                      champion=True, is_synthetic=is_synthetic)  # fmt: skip

    aclient, sclient = rest.service_role_client(), rest.sync_service_role_client()

    async def _async_client() -> Any:
        return aclient

    monkeypatch.setattr(factories, "get_async_supabase_client", _async_client)
    monkeypatch.setattr(repositories, "get_supabase_client", lambda: sclient)
    monkeypatch.setattr(health_score, "_health_source_client", lambda: sclient)
    repo = MLModelRegistryRepository(supabase_client=aclient)
    connector = SupabaseDataConnector()
    connector._client, connector._initialized = sclient, True

    try:
        hcp = (await resolve_hcp_adoption_champion("Kisqali", db=aclient))[0]
    except ChampionNotPromotedError:
        hcp = None
    champion = await repo.get_champion_model(experiment_id=uuid.UUID(exp))
    surfaced = {
        "get_models_for_target": name in await repo.get_models_for_target("lane_2259_target"),
        "get_model_performance_for_target": name
        in await repo.get_model_performance_for_target("lane_2259_target"),
        "get_champion_model": champion is not None and champion.model_name == name,
        "resolve_hcp_adoption_champion": hcp == name,
        "_probe_prediction_champions": any(
            n == name for n, _ in dispatcher._probe_prediction_champions()
        ),
        "_fetch_model_registry_facts": model_id in health_score._fetch_model_registry_facts(),
        "_resolve_production_model_names": name
        in await predictions._resolve_production_model_names(),
        "get_available_models": any(
            r.get("model_name") == name
            for r in await connector.get_available_models(stages=["production"])
        ),
        "explain._resolve_model_registry_id": await explain._resolve_model_registry_id(name)
        == model_id,
    }

    assert surfaced == dict.fromkeys(surfaced, not is_synthetic)


# ---------------------------------------------------------------------------
# promote_stage: the gate runs BEFORE MLflow moves (real MLflow, sqlite store)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provenance, mlflow_stage",
    [(None, "None"), ("synthetic_gold", "None"), ("real", "Production")],
)
async def test_promote_stage_refuses_before_mlflow_moves(
    registry_db: _pg.PgConn,
    rest: ThrowawayRest,
    monkeypatch,
    tmp_path,
    provenance: Optional[str],
    mlflow_stage: str,
) -> None:
    """A refused model must not reach MLflow's Production stage either.

    MLflow's stage is read on its own (e.g. src/kpi/calculators/model_performance.py reads
    get_latest_versions(stages=["Production", ...])), so a DB-only refusal left a refused model
    surfacing through MLflow. MLflow here is real, on a sqlite store in tmp_path: never the
    tracking server.
    """
    from mlflow.tracking import MlflowClient

    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.memory.services import factories
    from src.mlops.mlflow_connector import MLflowConnector

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)
    monkeypatch.setattr(MLflowConnector, "_instance", None)
    mlflow_client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    name = f"lane_2259_mlflow_{provenance}"
    mlflow_client.create_registered_model(name)
    version = mlflow_client.create_model_version(name, source=str(tmp_path / "artifact")).version

    exp = _experiment(registry_db, "lane_2259_mlflow")
    model_id = _model(registry_db, exp, name, stage="staging", provenance=provenance)
    client = rest.service_role_client()

    async def _client() -> Any:
        return client

    monkeypatch.setattr(factories, "get_async_supabase_client", _client)

    result = await registry_manager.promote_stage(
        {
            "registered_model_name": name,
            "model_version": int(version),
            "promotion_target_stage": "Production",
            "current_stage": "Shadow",
            "model_registry_id": model_id,
        }
    )

    assert mlflow_client.get_model_version(name, version).current_stage == mlflow_stage
    if mlflow_stage == "Production":
        assert result["promotion_successful"] is True
        assert "promotion_refused_reason" not in result
    else:
        assert result["promotion_successful"] is False
        assert result["current_stage"] == "Shadow"
        assert "training_provenance" in result["promotion_refused_reason"]
    # The DB row is only ever moved by the deploy agent's transition_stage, never here.
    assert _row(registry_db, model_id)["stage"] == "staging"
