"""#2310 / #2311 / #2308: the retrain candidate write path, through real PostgREST into real Postgres.

Opt-in: ``E2I_DB_INTEGRATION=1``. The fixtures of ``test_ml_registry_promotion_gate_realdb``:
a throwaway Postgres of prod's image holding prod's schema plus every pending migration (so
159/160 -- the 'candidate' / 'registered' enum values, ``retrain_of_id`` with its FK and
immutability trigger, ``mlflow_model_version``), and a throwaway PostgREST of prod's image in
front of it, authenticated as ``service_role``. The registry writer, the deploy record and the
repository are the production code; only the MLflow registration call is replaced (it would
need a trained run artifact; the MLflow tag/stage behaviour is proven on a real sqlite MLflow
in ``test_retrain_candidate_write_2310.py``).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, patch

import pytest

from tests.unit.test_database.learning_loop import _pg
from tests.unit.test_database.learning_loop.test_ml_registry_promotion_gate_realdb import (  # noqa: F401
    ThrowawayRest,
    _q,
    registry_db,
    rest,
)

pytestmark = [
    pytest.mark.skipif(not _pg.db_integration_enabled(), reason=_pg.OPT_IN_SKIP_REASON),
    pytest.mark.timeout(300),
]

NAME = "lane_2310_initiation_goldstd_lr_v1"
VERSION = "1.0_retrained_20260928_1934_725a94"
MLFLOW_EXP = "exp_lane_2310_725a94"


def _seed(conn: _pg.PgConn, run_id: str = "run-2310") -> Dict[str, str]:
    exp, parent = str(uuid.uuid4()), str(uuid.uuid4())
    conn.execute(
        "insert into ml_experiments (id, experiment_name, prediction_target, "
        f"mlflow_experiment_id) values ('{exp}', 'lane_2310_eval_v1', 'lane_2310_target', "
        f"'{MLFLOW_EXP}');"
        "insert into ml_model_registry (id, experiment_id, model_name, model_version, "
        f"algorithm, stage) values ('{parent}', '{exp}', '{NAME}', '1.0', 'logistic_regression', "
        "'staging');"
        "insert into ml_training_runs (experiment_id, run_name, mlflow_run_id, algorithm, "
        f"training_samples, status) values ('{exp}', 'train', '{run_id}', 'LogisticRegression', "
        "100, 'completed');",
        user="postgres",
    )
    return {"exp": exp, "parent": parent}


def _retrain_of(ids: Dict[str, str]) -> Dict[str, Any]:
    return {
        "model_id": ids["parent"],
        "model_name": NAME,
        "model_version": "1.0",
        "experiment_id": ids["exp"],
        "experiment_name": "lane_2310_eval_v1",
        "new_model_version": VERSION,
    }


def _rows(conn: _pg.PgConn) -> Dict[str, str]:
    lines = conn.rows(
        "select id || '|' || stage || '|' || coalesce(retrain_of_id::text, '-') || '|' || "
        "coalesce(mlflow_model_version::text, '-') from ml_model_registry "
        f"where model_name = '{NAME}' and model_version = '{VERSION}'"
    )
    return {ln.split("|")[0]: "|".join(ln.split("|")[1:]) for ln in lines}


async def _register(rest: ThrowawayRest, retrain_of: Optional[Dict[str, Any]], version: int):
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    client = rest.service_role_client()
    state = ModelDeployerState(
        audit_workflow_id=uuid.uuid4(),
        model_uri="runs:/run-2310/model",
        experiment_id=MLFLOW_EXP,
        deployment_name=f"{MLFLOW_EXP}_deployment",
        validation_metrics={"roc_auc": 0.836},
        success_criteria_met=True,
        retrain_of=retrain_of,
    )
    mlflow = AsyncMock(side_effect=lambda uri, name: (name, version, "None"))
    with (
        patch.object(registry_manager, "_register_model_mlflow", mlflow),
        # #2311: MLflow is replaced here; it answers "no earlier version from this run".
        patch.object(registry_manager, "_model_versions_for_run", AsyncMock(return_value=[])),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=client)
        ),
    ):
        out = await registry_manager.register_model(state)
    return out, mlflow


async def test_a_retrain_lands_as_a_linked_candidate_in_the_real_enum_and_fk(
    registry_db: _pg.PgConn, rest: ThrowawayRest
) -> None:
    ids = _seed(registry_db)
    out, _ = await _register(rest, _retrain_of(ids), version=4)
    assert out["registration_successful"] is True
    assert _rows(registry_db) == {out["model_registry_id"]: f"candidate|{ids['parent']}|4"}
    # The parent reference row is untouched.
    assert registry_db.rows(
        f"select stage from ml_model_registry where id = '{ids['parent']}'"
    ) == ["staging"]


async def test_a_retry_reuses_the_row_and_its_mlflow_version(
    registry_db: _pg.PgConn, rest: ThrowawayRest
) -> None:
    ids = _seed(registry_db)
    first, _ = await _register(rest, _retrain_of(ids), version=4)
    again, mlflow = await _register(rest, _retrain_of(ids), version=5)
    mlflow.assert_not_awaited()
    assert again["model_registry_id"] == first["model_registry_id"]
    assert again["model_version"] == 4
    assert len(_rows(registry_db)) == 1


@pytest.mark.parametrize("squatter_parent", [None, "other"])
async def test_a_row_with_other_lineage_is_refused_and_never_healed(
    registry_db: _pg.PgConn, rest: ThrowawayRest, squatter_parent: Optional[str]
) -> None:
    ids = _seed(registry_db)
    parent = None
    if squatter_parent == "other":
        parent = str(uuid.uuid4())
        registry_db.execute(
            "insert into ml_model_registry (id, experiment_id, model_name, model_version, "
            f"algorithm, stage) values ('{parent}', '{ids['exp']}', '{NAME}', '0.9', 'lr', "
            "'archived')",
            user="postgres",
        )
    squatter = str(uuid.uuid4())
    registry_db.execute(
        "insert into ml_model_registry (id, experiment_id, model_name, model_version, algorithm, "
        f"stage, mlflow_run_id, retrain_of_id) values ('{squatter}', '{ids['exp']}', '{NAME}', "
        f"'{VERSION}', 'lr', 'staging', 'run-2310', {_q(parent)})",
        user="postgres",
    )
    out, mlflow = await _register(rest, _retrain_of(ids), version=4)
    assert out["registration_successful"] is False
    mlflow.assert_not_awaited()
    assert _rows(registry_db) == {squatter: f"staging|{parent or '-'}|-"}


@pytest.mark.parametrize(
    "raced_parent, raced_version, reused",
    [("parent", 4, True), ("parent", 3, False), (None, 4, False)],
)
async def test_the_unique_race_reuse_applies_the_same_rules(
    registry_db: _pg.PgConn,
    rest: ThrowawayRest,
    raced_parent: Optional[str],
    raced_version: int,
    reused: bool,
) -> None:
    """A concurrent delivery writes the row between the pre-check and the insert: the real
    23505 sends the writer to its re-read, which must apply lineage + MLflow-version rules."""
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
        _persist_model_registry_row,
    )
    from src.repositories.ml_experiment import MLModelRegistryRepository

    ids = _seed(registry_db)
    raced = str(uuid.uuid4())
    parent = ids["parent"] if raced_parent else None
    registry_db.execute(
        "insert into ml_model_registry (id, experiment_id, model_name, model_version, algorithm, "
        f"stage, mlflow_run_id, retrain_of_id, mlflow_model_version) values ('{raced}', "
        f"'{ids['exp']}', '{NAME}', '{VERSION}', 'lr', 'candidate', 'run-2310', {_q(parent)}, "
        f"{raced_version})",
        user="postgres",
    )
    real_lookup = MLModelRegistryRepository.get_by_name_version
    calls = {"n": 0}

    async def _missed_first(self, *a: Any, **k: Any):
        calls["n"] += 1
        return None if calls["n"] == 1 else await real_lookup(self, *a, **k)

    with patch.object(MLModelRegistryRepository, "get_by_name_version", _missed_first):
        rid = await _persist_model_registry_row(
            rest.service_role_client(),
            experiment_id_str=MLFLOW_EXP,
            model_uri="runs:/run-2310/model",
            registered_model_name=NAME,
            model_version=4,
            validation_metrics=None,
            version_label=VERSION,
            expected_experiment_id=ids["exp"],
            retrain_of_id=ids["parent"],
        )
    assert calls["n"] == 2  # the pre-check missed; the insert hit the real unique key
    assert rid == (raced if reused else None)


async def test_lineage_cannot_be_rewritten_through_the_api(
    registry_db: _pg.PgConn, rest: ThrowawayRest
) -> None:
    ids = _seed(registry_db)
    out, _ = await _register(rest, _retrain_of(ids), version=4)
    client = rest.service_role_client()
    with pytest.raises(Exception, match="immutable"):
        await (
            client.table("ml_model_registry")
            .update({"retrain_of_id": None})
            .eq("id", out["model_registry_id"])
            .execute()
        )
    assert _rows(registry_db)[out["model_registry_id"]] == f"candidate|{ids['parent']}|4"


async def test_the_candidate_deploy_record_is_registered_and_the_row_stays_a_candidate(
    registry_db: _pg.PgConn, rest: ThrowawayRest, monkeypatch
) -> None:
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent
    from src.memory.services import factories

    ids = _seed(registry_db)
    out, _ = await _register(rest, _retrain_of(ids), version=4)
    client = rest.service_role_client()

    async def _client() -> Any:
        return client

    monkeypatch.setattr(factories, "get_async_supabase_client", _client)
    output: Dict[str, Any] = {"deployment_successful": True, "status": "completed"}
    await ModelDeployerAgent()._store_to_database(
        output,
        {
            "model_registry_id": out["model_registry_id"],
            "deployment_name": f"{MLFLOW_EXP}_deployment",
            "target_environment": "candidate",
            "promotion_successful": True,
            "current_stage": "Candidate",
            "deployment_action": "promote",
        },
    )
    assert output["db_persisted"] is True
    assert output["deployment_successful"] is True
    assert registry_db.rows(
        "select id || '|' || status || '|' || environment from ml_deployments "
        f"where model_registry_id = '{out['model_registry_id']}'"
    ) == [f"{output['deployment_record_id']}|registered|candidate"]
    assert _rows(registry_db)[out["model_registry_id"]].startswith("candidate|")
