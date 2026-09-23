"""#2242: a retrain carries the LOGICAL identity of the registry row it retrains.

Before: ``_cohort_input_from_training_config`` handed ``MLFoundationPipeline.run`` only
the physical label column, so the scope was named ``<brand> - treatment_initiated``,
a second run of that name took scope_definer's refresh path (which never persists the
freshly minted experiment id, so the trainer and the registry writer could not resolve
it), and the candidate was registered as ``<generated experiment id>_deployment`` —
NOT attached to the ``ml_model_registry`` row being retrained (real job c4252b47,
2026-09-23: ``exp_kisq_al_20260923054245_ad8158`` for ``initiation_kisqali_goldstd_lr_v1``).

After: the trigger resolves the retrained row's identity (model name / version /
experiment) and records it as ``training_config["retrain_of"]`` with the candidate's
version; the physical ``target_outcome`` stays the frame column the loader needs; the
scope attaches to the retrained model's own ``ml_experiments`` row; the candidate is
registered as a new version of the same registered model; and
``ml_retraining_history`` (model_id, new_model_version) names that exact row.

Real code over the in-memory async supabase fake (``tests/unit/_fakes/async_supabase``);
patched only: drift history / perf tracker / Celery ``.delay`` (trigger side channels),
the MLflow registration call (no MLflow here), and scope_definer's memory/Opik writers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.services.retraining_trigger import RetrainingTriggerService, TriggerReason
from src.tasks.drift_monitoring_tasks import _cohort_input_from_training_config
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

pytestmark = pytest.mark.unit

# Literals of the migration-151 Kisqali / Remibrutinib initiation contracts (abbreviated
# column list — the identity under test does not depend on it).
_COLUMNS = ["disease_severity", "age_at_diagnosis", "treatment_initiated"]


def _contract(brand: str) -> str:
    import json

    return json.dumps(
        {
            "type": "table",
            "table": "patient_journeys",
            "filters": {"brand": brand, "is_synthetic": True},
            "columns": _COLUMNS,
        },
        sort_keys=True,
    )


def _goldstd_db() -> tuple[FakeAsyncSupabase, Dict[str, Dict[str, str]]]:
    """Two goldstd initiation models + their eval experiments, shaped like prod.

    ``mlflow_experiment_id`` is NULL and ``brand`` is 'Remibrutinib' on BOTH experiment
    rows — exactly the live rows (the brand is a known pre-existing data defect).
    """
    ids: Dict[str, Dict[str, str]] = {}
    experiments = []
    registry = []
    for brand in ("Kisqali", "Remibrutinib"):
        key = f"initiation_{brand.lower()}"
        exp_id, model_id = str(uuid4()), str(uuid4())
        ids[brand] = {"exp": exp_id, "model": model_id, "name": f"{key}_goldstd_lr_v1"}
        experiments.append(
            {
                "id": exp_id,
                "experiment_name": f"{key}_goldstd_eval_v1",
                "mlflow_experiment_id": None,
                "prediction_target": key,
                "brand": "Remibrutinib",
                "description": f"Gold-standard eval pipeline for the {key} cohort.",
                "is_synthetic": False,
            }
        )
        registry.append(
            {
                "id": model_id,
                "experiment_id": exp_id,
                "model_name": f"{key}_goldstd_lr_v1",
                "model_version": "1.0",
                "stage": "staging",
                "is_synthetic": False,
                "cohort_data_source": _contract(brand),
                "cohort_target_outcome": "treatment_initiated",
                "cohort_feature_manifest_source": "synthetic_csu",
            }
        )
    db = FakeAsyncSupabase(
        {
            "ml_experiments": experiments,
            "ml_model_registry": registry,
            "ml_retraining_history": [],
            "ml_training_runs": [],
        }
    )
    return db, ids


async def _trigger(db: FakeAsyncSupabase, handle: str) -> Dict[str, Any]:
    service = RetrainingTriggerService()
    drift_repo = MagicMock()
    drift_repo.get_latest_drift_status = AsyncMock(return_value=[])
    tracker = MagicMock()
    tracker.get_performance_trend = AsyncMock(side_effect=Exception("no perf"))
    with (
        patch(
            "src.repositories.drift_monitoring.get_drift_monitoring_client",
            AsyncMock(return_value=db),
        ),
        patch("src.repositories.drift_monitoring.DriftHistoryRepository", return_value=drift_repo),
        patch("src.services.performance_tracking.get_performance_tracker", return_value=tracker),
        patch("src.tasks.drift_monitoring_tasks.execute_model_retraining") as mock_task,
    ):
        mock_task.delay = MagicMock(return_value=MagicMock(id="task-1"))
        job = await service.trigger_retraining(
            model_version=handle, reason=TriggerReason.MANUAL, cohort=None
        )
        queued = mock_task.delay.call_args.kwargs
    return {"job": job, **queued}


# --------------------------------------------------------------------------- identity


@pytest.mark.asyncio
async def test_identity_is_read_from_the_registry_row_and_its_experiment():
    from src.services.cohort_contract import load_registry_model_identity

    db, ids = _goldstd_db()
    identity = await load_registry_model_identity(db, ids["Kisqali"]["model"])
    assert identity == {
        "model_id": ids["Kisqali"]["model"],
        "model_name": "initiation_kisqali_goldstd_lr_v1",
        "model_version": "1.0",
        "experiment_id": ids["Kisqali"]["exp"],
        "experiment_name": "initiation_kisqali_goldstd_eval_v1",
        "prediction_target": "initiation_kisqali",
    }
    # the experiment's brand column is NOT part of the identity: it reads
    # 'Remibrutinib' on every goldstd experiment (pre-existing data defect)
    assert "brand" not in identity


@pytest.mark.asyncio
async def test_identity_is_empty_for_an_unregistered_or_experimentless_row():
    from src.services.cohort_contract import load_registry_model_identity

    db, ids = _goldstd_db()
    assert await load_registry_model_identity(db, str(uuid4())) == {}
    assert await load_registry_model_identity(None, ids["Kisqali"]["model"]) == {}
    db.rows("ml_model_registry")[0]["experiment_id"] = None
    assert await load_registry_model_identity(db, ids["Kisqali"]["model"]) == {}


@pytest.mark.asyncio
async def test_trigger_records_the_retrained_identity_and_a_registry_valid_version():
    db, ids = _goldstd_db()
    out = await _trigger(db, ids["Kisqali"]["model"])  # the sweep passes the row uuid
    tc = out["training_config"]
    new_version = out["new_version"]
    # the candidate version derives from the ROW's version, not the uuid handle (a
    # "<uuid>_retrained_<ts>" string is 60 chars > ml_model_registry.model_version(50))
    assert new_version.startswith("1.0_retrained_")
    assert len(new_version) <= 50
    assert out["job"].new_model_version == new_version
    assert tc["retrain_of"] == {
        "model_id": ids["Kisqali"]["model"],
        "model_name": "initiation_kisqali_goldstd_lr_v1",
        "model_version": "1.0",
        "experiment_id": ids["Kisqali"]["exp"],
        "experiment_name": "initiation_kisqali_goldstd_eval_v1",
        "prediction_target": "initiation_kisqali",
        "new_model_version": new_version,
    }
    (history,) = db.rows("ml_retraining_history")
    assert history["model_id"] == ids["Kisqali"]["model"]
    assert history["new_model_version"] == new_version
    assert history["config"]["retrain_of"]["model_name"] == "initiation_kisqali_goldstd_lr_v1"


@pytest.mark.asyncio
async def test_retraining_a_retrained_candidate_keeps_the_version_bounded():
    db, ids = _goldstd_db()
    db.rows("ml_model_registry")[0]["model_version"] = "1.0_retrained_20260923_0542"
    out = await _trigger(db, ids["Kisqali"]["model"])
    assert out["new_version"].startswith("1.0_retrained_")
    assert out["new_version"].count("_retrained_") == 1


@pytest.mark.asyncio
async def test_unregistered_handle_keeps_the_legacy_version_and_no_identity():
    db, _ = _goldstd_db()
    out = await _trigger(db, "propensity_v2.1.0")
    assert out["new_version"].startswith("propensity_retrained_")
    assert "retrain_of" not in out["training_config"]


def test_cohort_input_keeps_the_physical_label_and_carries_the_logical_identity():
    retrain_of = {
        "model_id": "m",
        "model_name": "initiation_kisqali_goldstd_lr_v1",
        "model_version": "1.0",
        "experiment_id": "e",
        "experiment_name": "initiation_kisqali_goldstd_eval_v1",
        "prediction_target": "initiation_kisqali",
        "new_model_version": "1.0_retrained_20260923_0542",
    }
    import json

    pipeline_input = _cohort_input_from_training_config(
        {
            "data_source": json.loads(_contract("Kisqali")),
            "target_outcome": "treatment_initiated",
            "retrain_of": retrain_of,
        }
    )
    assert pipeline_input["target_outcome"] == "treatment_initiated"  # the frame column
    assert pipeline_input["retrain_of"] == retrain_of
    assert pipeline_input["brand"] == "Kisqali"  # #2241: from the contract filters


# ------------------------------------------------------------------ end-to-end chain


async def _scope(db: FakeAsyncSupabase, pipeline_input: Dict[str, Any]):
    """Run the REAL pipeline scope stage (real ScopeDefinerAgent graph) over ``db``."""
    from src.agents.tier_0.pipeline import (
        MLFoundationPipeline,
        PipelineConfig,
        PipelineResult,
        PipelineStage,
    )
    from src.repositories.ml_experiment import MLExperimentRepository

    pipeline = MLFoundationPipeline(PipelineConfig())
    result = PipelineResult(
        pipeline_run_id="run-2242",
        status="in_progress",
        current_stage=PipelineStage.SCOPE_DEFINITION,
        audit_workflow_id=uuid4(),
    )
    hooks = MagicMock()
    hooks.return_value.store_experiment_pattern = AsyncMock(return_value=True)
    hooks.return_value.store_scope_definition = AsyncMock(return_value=True)
    with (
        patch.object(MLFoundationPipeline, "_get_audit_service", return_value=None),
        patch(
            "src.agents.ml_foundation.scope_definer.agent._get_experiment_repository",
            AsyncMock(return_value=MLExperimentRepository(supabase_client=db)),
        ),
        patch(
            "src.agents.ml_foundation.scope_definer.agent._get_opik_connector",
            return_value=None,
        ),
        patch(
            "src.agents.ml_foundation.scope_definer.agent._get_procedural_memory",
            return_value=None,
        ),
        patch("src.agents.ml_foundation.scope_definer.agent.ScopeDefinerMemoryHooks", hooks),
    ):
        await pipeline._run_scope_definition(pipeline_input, result, None)
    return result


async def _register(
    db: FakeAsyncSupabase,
    experiment_id: str,
    retrain_of: Optional[Dict[str, Any]],
    run_id: str,
    mlflow_version: int = 1,
):
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    exp_uuid = next(
        r["id"] for r in db.rows("ml_experiments") if r["mlflow_experiment_id"] == experiment_id
    )
    db.rows("ml_training_runs").append(
        {
            "id": str(uuid4()),
            "experiment_id": exp_uuid,
            "run_name": "train_x",
            "mlflow_run_id": run_id,
            "algorithm": "LogisticRegression",
            "hyperparameters": {},
            "status": "finished",
            "is_synthetic": False,
        }
    )
    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri=f"runs:/{run_id}/model",
        experiment_id=experiment_id,
        deployment_name=f"{experiment_id}_deployment",
        validation_metrics={"roc_auc": 0.8375},
        success_criteria_met=True,
        retrain_of=retrain_of,
    )
    mlflow = AsyncMock(
        side_effect=lambda uri, name: (name, mlflow_version, "None"),
    )
    with (
        patch.object(registry_manager, "_register_model_mlflow", mlflow),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        out = await registry_manager.register_model(state)
    return out, mlflow


@pytest.mark.asyncio
async def test_the_candidate_lands_as_a_new_version_of_the_retrained_model():
    """trigger -> training_config -> pipeline input -> scope -> registration -> history."""
    db, ids = _goldstd_db()
    queued = await _trigger(db, ids["Kisqali"]["model"])
    pipeline_input = _cohort_input_from_training_config(queued["training_config"])

    result = await _scope(db, pipeline_input)

    # (a) the scope is the retrained model's experiment, named from its logical target;
    #     the physical label still drives data preparation
    assert result.scope_spec["experiment_name"] == "initiation_kisqali_goldstd_eval_v1"
    assert result.scope_spec["prediction_target"] == "treatment_initiated"
    assert result.scope_spec["brand"] == "Kisqali"
    parent_exp = next(r for r in db.rows("ml_experiments") if r["id"] == ids["Kisqali"]["exp"])
    assert parent_exp["mlflow_experiment_id"] == result.experiment_id  # resolvable now
    assert len(db.rows("ml_experiments")) == 2  # no new scope row
    # the retrained model's experiment is attached, never refreshed/clobbered
    assert parent_exp["prediction_target"] == "initiation_kisqali"
    assert parent_exp["brand"] == "Remibrutinib"  # pre-existing defect left untouched
    assert parent_exp["description"].startswith("Gold-standard eval pipeline")

    # (b) the candidate is registered as a new version of the retrained model
    out, mlflow = await _register(
        db, result.experiment_id, pipeline_input["retrain_of"], run_id="run-a"
    )
    assert out["registration_successful"] is True
    assert mlflow.await_args.args[1] == "initiation_kisqali_goldstd_lr_v1"
    candidate = next(r for r in db.rows("ml_model_registry") if r["id"] == out["model_registry_id"])
    assert candidate["model_name"] == "initiation_kisqali_goldstd_lr_v1"
    assert candidate["experiment_id"] == ids["Kisqali"]["exp"]

    # (c) ml_retraining_history names that exact row: model_id -> the parent's name,
    #     new_model_version -> the candidate's version (UNIQUE(model_name, model_version))
    (history,) = db.rows("ml_retraining_history")
    parent = next(r for r in db.rows("ml_model_registry") if r["id"] == history["model_id"])
    linked = [
        r
        for r in db.rows("ml_model_registry")
        if r["model_name"] == parent["model_name"]
        and r["model_version"] == history["new_model_version"]
    ]
    assert [r["id"] for r in linked] == [out["model_registry_id"]]


@pytest.mark.asyncio
async def test_a_second_retrain_reuses_the_same_experiment_and_adds_a_second_version():
    db, ids = _goldstd_db()
    first = await _scope(
        db,
        _cohort_input_from_training_config(
            (await _trigger(db, ids["Kisqali"]["model"]))["training_config"]
        ),
    )
    await _register(db, first.experiment_id, _retrain_of(db, 0), run_id="run-a")
    # a later retrain (a different minute -> a different candidate version)
    db.rows("ml_retraining_history").clear()
    second_input = _cohort_input_from_training_config(
        (await _trigger(db, ids["Kisqali"]["model"]))["training_config"]
    )
    second_input["retrain_of"]["new_model_version"] = "1.0_retrained_20990101_0000"
    second = await _scope(db, second_input)
    assert second.experiment_id == first.experiment_id  # resolvable, not a fresh orphan id
    out, _ = await _register(
        db, second.experiment_id, second_input["retrain_of"], run_id="run-b", mlflow_version=2
    )
    names = sorted(
        r["model_version"]
        for r in db.rows("ml_model_registry")
        if r["model_name"] == "initiation_kisqali_goldstd_lr_v1"
    )
    assert len(names) == 3 and names[0] == "1.0"
    assert out["model_registry_id"] is not None


def _retrain_of(db: FakeAsyncSupabase, idx: int) -> Dict[str, Any]:
    return dict(db.rows("ml_retraining_history")[idx]["config"]["retrain_of"])


@pytest.mark.asyncio
async def test_kisqali_and_remibrutinib_initiation_retrains_never_collide():
    db, ids = _goldstd_db()
    scopes = {}
    for brand in ("Kisqali", "Remibrutinib"):
        pipeline_input = _cohort_input_from_training_config(
            (await _trigger(db, ids[brand]["model"]))["training_config"]
        )
        scopes[brand] = await _scope(db, pipeline_input)
    k, r = scopes["Kisqali"], scopes["Remibrutinib"]
    assert k.experiment_id != r.experiment_id
    assert k.scope_spec["experiment_name"] == "initiation_kisqali_goldstd_eval_v1"
    assert r.scope_spec["experiment_name"] == "initiation_remibrutinib_goldstd_eval_v1"
    by_id = {row["id"]: row for row in db.rows("ml_experiments")}
    assert by_id[ids["Kisqali"]["exp"]]["mlflow_experiment_id"] == k.experiment_id
    assert by_id[ids["Remibrutinib"]["exp"]]["mlflow_experiment_id"] == r.experiment_id
    assert len(by_id) == 2


@pytest.mark.asyncio
async def test_a_retrain_whose_experiment_is_gone_fails_closed_at_scope():
    db, ids = _goldstd_db()
    pipeline_input = _cohort_input_from_training_config(
        (await _trigger(db, ids["Kisqali"]["model"]))["training_config"]
    )
    db.rows("ml_experiments").clear()
    with pytest.raises(RuntimeError, match="retrain"):
        await _scope(db, pipeline_input)
    assert db.rows("ml_experiments") == []  # nothing created in its place


@pytest.mark.asyncio
async def test_registration_refuses_a_candidate_outside_the_retrained_models_experiment():
    """The registry writer's backstop: the resolved experiment must be the parent's."""
    db, ids = _goldstd_db()
    queued = await _trigger(db, ids["Kisqali"]["model"])
    pipeline_input = _cohort_input_from_training_config(queued["training_config"])
    scoped = await _scope(db, pipeline_input)
    wrong = dict(pipeline_input["retrain_of"], experiment_id=ids["Remibrutinib"]["exp"])
    out, _ = await _register(db, scoped.experiment_id, wrong, run_id="run-a")
    assert out["model_registry_id"] is None
    assert len(db.rows("ml_model_registry")) == 2  # only the two parents


@pytest.mark.asyncio
async def test_a_non_retrain_run_registers_exactly_as_before():
    db, _ = _goldstd_db()
    exp_id = str(uuid4())
    db.rows("ml_experiments").append(
        {
            "id": exp_id,
            "experiment_name": "Kisqali - treatment_initiated",
            "mlflow_experiment_id": "exp_kisq_al_20260923054245_ad8158",
            "prediction_target": "treatment_initiated",
            "is_synthetic": False,
        }
    )
    out, mlflow = await _register(db, "exp_kisq_al_20260923054245_ad8158", None, run_id="run-z")
    assert mlflow.await_args.args[1] == "exp_kisq_al_20260923054245_ad8158_deployment"
    row = next(r for r in db.rows("ml_model_registry") if r["id"] == out["model_registry_id"])
    assert row["model_name"] == "exp_kisq_al_20260923054245_ad8158_deployment"
    assert row["model_version"] == "1"
