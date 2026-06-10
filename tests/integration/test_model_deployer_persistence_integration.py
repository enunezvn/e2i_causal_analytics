"""Faithful real-DB integration tests for model_deployer persistence (#829).

Proves, against the REAL Supabase, that the F4 follow-up wiring actually writes
rows:

* ``_persist_model_registry_row`` writes a real ``ml_model_registry`` row whose
  ``experiment_id`` FK + ``algorithm`` + ``hyperparameters`` are sourced from
  real seeded ``ml_experiments`` / ``ml_training_runs`` substrate (no
  fabrication), is idempotent on re-run (UNIQUE(model_name, model_version)), and
  FAILS CLOSED (returns ``None``, writes nothing) when the experiment cannot be
  resolved;
* ``ModelDeployerAgent._store_to_database`` writes a real ``ml_deployments`` row
  FK-linked to that registry row and flips ``db_persisted=True`` ONLY when the
  row is confirmed — and, with a real client but NO ``model_registry_id``,
  writes nothing and keeps ``db_persisted=False`` (the #830 honesty contract,
  now proven the client's mere presence does not fabricate a row).

Run gate
--------
``E2I_DB_INTEGRATION=1`` plus a reachable async Supabase client (``SUPABASE_URL``
+ key in the process env). Mirrors the ``E2I_DB_INTEGRATION`` opt-in used by the
other ``tests/integration`` suites so unit-only CI lanes stay green and never
touch prod. NO mocks; every row is isolated by a unique tag and torn down
deterministically in a ``finally`` block.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason=(
        "E2I_DB_INTEGRATION!=1; integration test requires a real Supabase client "
        "and explicit opt-in. Run with E2I_DB_INTEGRATION=1 and SUPABASE_URL set."
    ),
)

from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent  # noqa: E402
from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (  # noqa: E402
    _persist_model_registry_row,
)
from src.agents.ml_foundation.model_trainer.schemas import MetricsSchema  # noqa: E402
from src.memory.services.factories import get_async_supabase_client  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    """Reset the cached async client so each test builds a fresh one on its OWN
    event loop. The global cache binds ``httpx.AsyncClient`` to the creating
    loop; pytest-asyncio's per-test loops would otherwise reuse a client from a
    closed loop and raise ``RuntimeError: Event loop is closed``. Test-only
    isolation (prod has one long-lived loop). Mirrors the fixture in
    ``test_async_supabase_client_realdb.py``.
    """
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


async def _seed_exp_and_run(client, *, mlflow_exp: str, run_id: str, algorithm: str):
    """Seed a real ml_experiments + ml_training_runs pair; return (exp, run) rows."""
    exp = (
        await client.table("ml_experiments")
        .insert(
            {
                "experiment_name": f"f4_829_test_{mlflow_exp}",
                "prediction_target": "treatment_initiated",
                "mlflow_experiment_id": mlflow_exp,
                "created_by": "f4_829_integration_test",
            }
        )
        .execute()
    ).data[0]
    run = (
        await client.table("ml_training_runs")
        .insert(
            {
                "experiment_id": exp["id"],
                "run_name": f"run_{run_id}",
                "mlflow_run_id": run_id,
                "algorithm": algorithm,
                "hyperparameters": {"max_depth": 5, "n_estimators": 200},
                "training_samples": 1234,
                "status": "finished",
                "is_best_trial": True,
                "test_metrics": {"auc": 0.8},
            }
        )
        .execute()
    ).data[0]
    return exp, run


async def _cleanup(
    client,
    *,
    deployment_names=(),
    registry_ids=(),
    run_ids=(),
    exp_ids=(),
):
    """Delete child-first to respect FKs. Best-effort; never raises."""
    for name in deployment_names:
        try:
            await client.table("ml_deployments").delete().eq("deployment_name", name).execute()
        except Exception:
            pass
    for rid in registry_ids:
        try:
            await client.table("ml_model_registry").delete().eq("id", rid).execute()
        except Exception:
            pass
    for rid in run_ids:
        try:
            await client.table("ml_training_runs").delete().eq("id", rid).execute()
        except Exception:
            pass
    for eid in exp_ids:
        try:
            await client.table("ml_experiments").delete().eq("id", eid).execute()
        except Exception:
            pass


async def test_persist_model_registry_row_writes_real_row_and_is_idempotent():
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    mlflow_exp = f"exp_f4test_{tag}"
    run_id = f"run_f4test_{tag}"
    model_name = f"f4_829_model_{tag}"
    registry_ids: list[str] = []
    exp = run = None
    try:
        exp, run = await _seed_exp_and_run(
            client, mlflow_exp=mlflow_exp, run_id=run_id, algorithm="xgboost"
        )

        rid = await _persist_model_registry_row(
            client,
            experiment_id_str=mlflow_exp,
            model_uri=f"runs:/{run_id}/model",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=MetricsSchema(
                roc_auc=0.81, pr_auc=0.42, brier_score=0.15, calibration_slope=0.97
            ),
        )
        assert rid is not None, "registry row must be written and confirmed"
        registry_ids.append(rid)

        # The row really exists with the experiment FK + the seeded algorithm
        # (sourced from ml_training_runs, NOT fabricated) + mapped metrics.
        rows = (
            await client.table("ml_model_registry").select("*").eq("id", rid).limit(1).execute()
        ).data
        assert rows, "registry row must be readable from the DB"
        row = rows[0]
        assert row["experiment_id"] == exp["id"]
        assert row["model_name"] == model_name
        assert row["algorithm"] == "xgboost"
        assert row["hyperparameters"] == {"max_depth": 5, "n_estimators": 200}
        assert float(row["auc"]) == pytest.approx(0.81)
        assert float(row["pr_auc"]) == pytest.approx(0.42)

        # Idempotent re-run (UNIQUE(model_name, model_version)): same id, one row.
        rid2 = await _persist_model_registry_row(
            client,
            experiment_id_str=mlflow_exp,
            model_uri=f"runs:/{run_id}/model",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid2 == rid
        cnt = (
            await client.table("ml_model_registry")
            .select("id", count="exact")
            .eq("model_name", model_name)
            .execute()
        ).count
        assert cnt == 1
    finally:
        await _cleanup(
            client,
            registry_ids=registry_ids,
            run_ids=[run["id"]] if run else [],
            exp_ids=[exp["id"]] if exp else [],
        )


async def test_persist_model_registry_row_fails_closed_on_unresolvable_experiment():
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    model_name = f"f4_829_noexp_{tag}"
    rid = await _persist_model_registry_row(
        client,
        experiment_id_str=f"exp_does_not_exist_{tag}",
        model_uri="runs:/whatever/model",
        registered_model_name=model_name,
        model_version=1,
        validation_metrics=None,
    )
    assert rid is None, "unresolvable experiment must fail closed (no fabricated FK)"
    # And nothing was written under that model name.
    cnt = (
        await client.table("ml_model_registry")
        .select("id", count="exact")
        .eq("model_name", model_name)
        .execute()
    ).count
    assert cnt == 0


async def test_persist_fails_closed_when_run_belongs_to_a_different_experiment():
    """Provenance guard (codex HIGH): a model_uri run id that belongs to a
    DIFFERENT experiment than the resolved one must NOT source algorithm /
    hyperparameters from that foreign run — fail closed, write nothing."""
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    model_name = f"f4_829_xexp_{tag}"
    expA = runA = expB = runB = None
    try:
        # Experiment A (the resolved experiment) + Experiment B (owns the run).
        expA, runA = await _seed_exp_and_run(
            client, mlflow_exp=f"expA_{tag}", run_id=f"runA_{tag}", algorithm="xgboost"
        )
        expB, runB = await _seed_exp_and_run(
            client, mlflow_exp=f"expB_{tag}", run_id=f"runB_{tag}", algorithm="lightgbm"
        )

        rid = await _persist_model_registry_row(
            client,
            experiment_id_str=f"expA_{tag}",  # resolves to experiment A
            model_uri=f"runs:/runB_{tag}/model",  # but the run belongs to B
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid is None, "cross-experiment run provenance must fail closed"
        cnt = (
            await client.table("ml_model_registry")
            .select("id", count="exact")
            .eq("model_name", model_name)
            .execute()
        ).count
        assert cnt == 0
    finally:
        await _cleanup(
            client,
            run_ids=[r["id"] for r in (runA, runB) if r],
            exp_ids=[e["id"] for e in (expA, expB) if e],
        )


async def test_persist_fails_closed_on_name_version_collision_from_foreign_experiment():
    """Idempotency provenance guard (codex HIGH): an existing
    (model_name, model_version) row that belongs to a DIFFERENT experiment is a
    real collision, NOT our row — fail closed instead of mis-linking the
    deployment to foreign provenance. The pre-existing row is untouched."""
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    model_name = f"f4_829_coll_{tag}"
    expA = runA = expB = runB = None
    registry_ids: list[str] = []
    try:
        expA, runA = await _seed_exp_and_run(
            client, mlflow_exp=f"cexpA_{tag}", run_id=f"crunA_{tag}", algorithm="xgboost"
        )
        # First write the row under experiment A.
        rid_a = await _persist_model_registry_row(
            client,
            experiment_id_str=f"cexpA_{tag}",
            model_uri=f"runs:/crunA_{tag}/model",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid_a is not None
        registry_ids.append(rid_a)

        # Now experiment B tries to claim the SAME (model_name, version).
        expB, runB = await _seed_exp_and_run(
            client, mlflow_exp=f"cexpB_{tag}", run_id=f"crunB_{tag}", algorithm="lightgbm"
        )
        rid_b = await _persist_model_registry_row(
            client,
            experiment_id_str=f"cexpB_{tag}",
            model_uri=f"runs:/crunB_{tag}/model",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid_b is None, "foreign-experiment name+version collision must fail closed"
        # Exactly one row remains, and it is experiment A's original row.
        rows = (
            await client.table("ml_model_registry")
            .select("id,experiment_id")
            .eq("model_name", model_name)
            .execute()
        ).data
        assert len(rows) == 1
        assert rows[0]["id"] == rid_a
        assert rows[0]["experiment_id"] == expA["id"]
    finally:
        await _cleanup(
            client,
            registry_ids=registry_ids,
            run_ids=[r["id"] for r in (runA, runB) if r],
            exp_ids=[e["id"] for e in (expA, expB) if e],
        )


async def test_store_to_database_writes_deployment_and_flips_db_persisted():
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    mlflow_exp = f"exp_f4dep_{tag}"
    run_id = f"run_f4dep_{tag}"
    model_name = f"f4_829_depmodel_{tag}"
    deployment_name = f"f4_829_deploy_{tag}"
    registry_ids: list[str] = []
    exp = run = None
    try:
        exp, run = await _seed_exp_and_run(
            client, mlflow_exp=mlflow_exp, run_id=run_id, algorithm="lightgbm"
        )
        rid = await _persist_model_registry_row(
            client,
            experiment_id_str=mlflow_exp,
            model_uri=f"runs:/{run_id}/model",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid is not None
        registry_ids.append(rid)

        agent = ModelDeployerAgent()
        output: dict = {"deployment_successful": True}
        state = {
            "model_registry_id": rid,
            "deployment_name": deployment_name,
            "target_environment": "staging",
            "deployed_by": "f4_829_integration_test",
            "resources": {"cpu": "2", "memory": "4Gi"},
        }
        await agent._store_to_database(output, state)

        assert output.get("db_persisted") is True, (
            "db_persisted must flip True only after a confirmed ml_deployments write; "
            f"reason={output.get('db_persist_skipped_reason')!r}"
        )
        dep_rows = (
            await client.table("ml_deployments")
            .select("*")
            .eq("deployment_name", deployment_name)
            .execute()
        ).data
        assert len(dep_rows) == 1
        assert dep_rows[0]["model_registry_id"] == rid
    finally:
        await _cleanup(
            client,
            deployment_names=[deployment_name],
            registry_ids=registry_ids,
            run_ids=[run["id"]] if run else [],
            exp_ids=[exp["id"]] if exp else [],
        )


def _mlflow_server_reachable(tracking_uri: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


async def test_register_model_node_end_to_end_real_mlflow_writes_registry():
    """GOLD-STANDARD faithful path: log a REAL model to MLflow, drive the REAL
    ``register_model`` node, and prove it writes a real ``ml_model_registry``
    row (experiment FK + algorithm sourced from the seeded training run, metrics
    mapped from validation_metrics) and surfaces ``model_registry_id``.

    Skipped when the local MLflow server is unreachable (environmental); the
    other tests still cover the DB-persistence path without MLflow.
    """
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("mlflow.sklearn")
    pytest.importorskip("sklearn")
    np = pytest.importorskip("numpy")
    from sklearn.linear_model import LogisticRegression

    tracking_uri = "http://localhost:5000"
    if not _mlflow_server_reachable(tracking_uri):
        pytest.skip(f"MLflow server not reachable at {tracking_uri}")

    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import register_model

    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    mlflow_exp = f"exp_node_e2e_{tag}"
    model_name = f"f4_829_nodee2e_{tag}"
    registry_ids: list[str] = []
    exp = run = None
    try:
        # 1) Log a real model to MLflow -> runs:/<run_id>/model
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"f4_829_probe_{tag}")
        rng = np.random.RandomState(0)
        X = rng.rand(40, 3)
        y = (X[:, 0] > 0.5).astype(int)
        with mlflow.start_run() as mlrun:
            model = LogisticRegression().fit(X, y)
            mlflow.sklearn.log_model(model, name="model")
            run_id = mlrun.info.run_id
        model_uri = f"runs:/{run_id}/model"

        # 2) Seed the experiment + the training run referenced by run_id
        exp, run = await _seed_exp_and_run(
            client, mlflow_exp=mlflow_exp, run_id=run_id, algorithm="logistic_regression"
        )

        # 3) Drive the REAL node (real MLflow registration)
        result = await register_model(
            {
                "model_uri": model_uri,
                "experiment_id": mlflow_exp,
                "deployment_name": model_name,
                "validation_metrics": {"roc_auc": 0.77, "pr_auc": 0.30},
            }
        )

        assert result.get("registration_successful") is True, (
            "real MLflow registration must succeed"
        )
        rid = result.get("model_registry_id")
        assert rid, "node must surface a real model_registry_id after a real registration"
        registry_ids.append(rid)

        row = (
            await client.table("ml_model_registry").select("*").eq("id", rid).limit(1).execute()
        ).data[0]
        assert row["experiment_id"] == exp["id"]
        assert row["algorithm"] == "logistic_regression"
        assert float(row["auc"]) == pytest.approx(0.77)
    finally:
        await _cleanup(
            client,
            registry_ids=registry_ids,
            run_ids=[run["id"]] if run else [],
            exp_ids=[exp["id"]] if exp else [],
        )
        try:
            from mlflow.tracking import MlflowClient

            MlflowClient(tracking_uri=tracking_uri).delete_registered_model(model_name)
        except Exception:
            pass


async def test_persist_fails_closed_when_uri_run_id_absent_from_training_runs():
    """Provenance fabrication guard (codex F1): when ``model_uri`` pins an EXACT
    run (``runs:/<run_id>/...``) that is ABSENT from ``ml_training_runs``, the
    row must NOT be written by substituting ``get_best_run()`` — that would stamp
    the row with this run_id + URI while sourcing ``algorithm`` /
    ``hyperparameters`` from a DIFFERENT run. Fail closed, write nothing.

    The experiment is seeded WITH a best run so ``get_best_run()`` *would* return
    a run if the buggy fallback were taken — proving the guard, not an empty DB.
    """
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    mlflow_exp = f"exp_f4absent_{tag}"
    seeded_run_id = f"run_f4absent_{tag}"
    model_name = f"f4_829_absentrun_{tag}"
    exp = run = None
    try:
        exp, run = await _seed_exp_and_run(
            client, mlflow_exp=mlflow_exp, run_id=seeded_run_id, algorithm="xgboost"
        )
        rid = await _persist_model_registry_row(
            client,
            experiment_id_str=mlflow_exp,
            model_uri=f"runs:/missing_{tag}/model",  # a run id NOT in ml_training_runs
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid is None, (
            "an absent exact run id must fail closed, not fall back to get_best_run "
            "(which fabricates provenance)"
        )
        cnt = (
            await client.table("ml_model_registry")
            .select("id", count="exact")
            .eq("model_name", model_name)
            .execute()
        ).count
        assert cnt == 0
    finally:
        await _cleanup(
            client,
            run_ids=[run["id"]] if run else [],
            exp_ids=[exp["id"]] if exp else [],
        )


async def test_persist_reuse_fails_closed_on_run_id_mismatch_same_experiment():
    """Idempotency provenance guard (codex F2): an existing
    ``(model_name, model_version)`` row in the SAME experiment but sourced from a
    DIFFERENT run than the one this deployment references must NOT be silently
    reused (mis-linking the deployment to the wrong model artifact). Fail closed;
    the original row is untouched."""
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    mlflow_exp = f"exp_f4mism_{tag}"
    run_id_1 = f"run1_f4mism_{tag}"
    run_id_2 = f"run2_f4mism_{tag}"
    model_name = f"f4_829_mism_{tag}"
    exp = run1 = None
    run2_id = None
    registry_ids: list[str] = []
    try:
        exp, run1 = await _seed_exp_and_run(
            client, mlflow_exp=mlflow_exp, run_id=run_id_1, algorithm="xgboost"
        )
        # A SECOND real run in the SAME experiment.
        run2 = (
            await client.table("ml_training_runs")
            .insert(
                {
                    "experiment_id": exp["id"],
                    "run_name": f"run_{run_id_2}",
                    "mlflow_run_id": run_id_2,
                    "algorithm": "lightgbm",
                    "hyperparameters": {"num_leaves": 31},
                    "training_samples": 1234,
                    "status": "finished",
                    "is_best_trial": False,
                    "test_metrics": {"auc": 0.7},
                }
            )
            .execute()
        ).data[0]
        run2_id = run2["id"]

        rid1 = await _persist_model_registry_row(
            client,
            experiment_id_str=mlflow_exp,
            model_uri=f"runs:/{run_id_1}/model",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid1 is not None
        registry_ids.append(rid1)

        # Re-deploy same name+version+experiment but referencing run 2.
        rid2 = await _persist_model_registry_row(
            client,
            experiment_id_str=mlflow_exp,
            model_uri=f"runs:/{run_id_2}/model",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid2 is None, "same name+version+experiment but different run must fail closed"
        rows = (
            await client.table("ml_model_registry")
            .select("id,mlflow_run_id")
            .eq("model_name", model_name)
            .execute()
        ).data
        assert len(rows) == 1
        assert rows[0]["id"] == rid1
        assert rows[0]["mlflow_run_id"] == run_id_1
    finally:
        await _cleanup(
            client,
            registry_ids=registry_ids,
            run_ids=[r for r in ((run1["id"] if run1 else None), run2_id) if r],
            exp_ids=[exp["id"]] if exp else [],
        )


async def test_persist_reuse_ok_when_redeploy_uri_carries_no_run_id():
    """No false fail-close (codex F2 fix must not over-constrain): a re-deploy
    whose ``model_uri`` carries NO run id (``models:/`` form) must still REUSE the
    existing same name+version+experiment row — we fail closed only on a DEFINITE
    run-id conflict, never on a missing run id."""
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    mlflow_exp = f"exp_f4reuse_{tag}"
    run_id = f"run_f4reuse_{tag}"
    model_name = f"f4_829_reuse_{tag}"
    exp = run = None
    registry_ids: list[str] = []
    try:
        exp, run = await _seed_exp_and_run(
            client, mlflow_exp=mlflow_exp, run_id=run_id, algorithm="xgboost"
        )
        rid1 = await _persist_model_registry_row(
            client,
            experiment_id_str=mlflow_exp,
            model_uri=f"runs:/{run_id}/model",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid1 is not None
        registry_ids.append(rid1)

        # Re-deploy with a models:/ URI (no embedded run id) — must reuse, not fail.
        rid2 = await _persist_model_registry_row(
            client,
            experiment_id_str=mlflow_exp,
            model_uri=f"models:/{model_name}/1",
            registered_model_name=model_name,
            model_version=1,
            validation_metrics=None,
        )
        assert rid2 == rid1, "a missing run id must not block idempotent reuse"
    finally:
        await _cleanup(
            client,
            registry_ids=registry_ids,
            run_ids=[run["id"]] if run else [],
            exp_ids=[exp["id"]] if exp else [],
        )


async def test_store_to_database_without_registry_id_is_fail_closed_with_real_client():
    """A real client present but NO model_registry_id must write nothing and keep
    db_persisted=False — the client's presence alone never fabricates a row."""
    client = await get_async_supabase_client()
    tag = uuid.uuid4().hex[:10]
    deployment_name = f"f4_829_noreg_{tag}"
    try:
        agent = ModelDeployerAgent()
        output: dict = {"deployment_successful": True}
        state = {
            "deployment_name": deployment_name,
            "target_environment": "staging",
        }
        await agent._store_to_database(output, state)

        assert output.get("db_persisted") is False
        assert output.get("db_persist_skipped_reason")
        cnt = (
            await client.table("ml_deployments")
            .select("id", count="exact")
            .eq("deployment_name", deployment_name)
            .execute()
        ).count
        assert cnt == 0
    finally:
        await _cleanup(client, deployment_names=[deployment_name])
