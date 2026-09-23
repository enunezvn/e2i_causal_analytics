"""#2255: a pipeline candidate's ``training_provenance`` says what it was trained on.

Before: the pipeline's registry writer never set ``ml_model_registry.training_provenance``,
so a retrain candidate trained on its ``synthetic_gold`` parent's cohort (migration-151
initiation contracts: ``filters.is_synthetic = true``) landed with NULL, and the #968
promotion gate (``MLModelRegistryRepository.transition_stage`` refuses
``synthetic_gold -> production``) could not see it.

After: the provenance is DERIVED from what was trained on — a table cohort contract
whose ``filters`` pin ``is_synthetic`` selects only synthetic rows (``synthetic_gold``,
the only synthetic value migration 083 allows) or only real rows (``real``), and a real
load the trainer augmented with synthetic rows is ``mixed``. An unpinned load stays NULL
(unknown), never guessed — not even from the parent (codex r1: an unpinned load is real
rows in strict mode and both in showcase mode, whatever the parent was trained on). A
retrain of a ``synthetic_gold`` parent on a REAL cohort (the remedy #968 prescribes) is
therefore ``real`` and promotable.

Real code over the in-memory async supabase fake (the #2242 harness); patched only:
trigger side channels, scope_definer memory/Opik writers, and the MLflow call.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.tasks.drift_monitoring_tasks import _cohort_input_from_training_config
from tests.unit._fakes.async_supabase import FakeAsyncSupabase
from tests.unit.test_services.test_retrain_registry_linkage_2242 import (
    _goldstd_db,
    _scope,
    _trigger,
)

pytestmark = pytest.mark.unit

_COLUMNS = ["disease_severity", "age_at_diagnosis", "treatment_initiated"]


def _table(**filters: Any) -> Dict[str, Any]:
    return {
        "type": "table",
        "table": "patient_journeys",
        "filters": filters,
        "columns": _COLUMNS,
    }


# ------------------------------------------------------------------- derivation


@pytest.mark.parametrize(
    "data_source, expected",
    [
        (_table(brand="Kisqali", is_synthetic=True), "synthetic_gold"),
        (_table(brand="Kisqali", is_synthetic="true"), "synthetic_gold"),
        (_table(brand="Kisqali", is_synthetic=False), "real"),
        (_table(brand="Kisqali", is_synthetic="false"), "real"),  # never bool("false")
        (json.dumps(_table(brand="Kisqali", is_synthetic=True)), "synthetic_gold"),
        (_table(brand="Kisqali"), None),  # not pinned: env-dependent row set
        (_table(brand="Kisqali", is_synthetic="maybe"), None),
        ("patient_journeys", None),  # a bare table name pins nothing
        ({"type": "file_dir", "path": "/data/cohort"}, None),
        (None, None),
    ],
)
def test_provenance_is_derived_only_from_a_pinned_is_synthetic_filter(data_source, expected):
    from src.services.cohort_contract import training_provenance_from_contract

    assert training_provenance_from_contract(data_source) == expected


# ------------------------------------------------------------------ registration


async def _register(
    db: FakeAsyncSupabase,
    experiment_id: str,
    retrain_of: Optional[Dict[str, Any]],
    data_source: Any,
    run_id: str = "run-a",
) -> Dict[str, Any]:
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
        data_source=data_source,
        target_outcome="treatment_initiated",
        retrain_of=retrain_of,
    )
    with (
        patch.object(
            registry_manager,
            "_register_model_mlflow",
            AsyncMock(side_effect=lambda uri, name: (name, 1, "None")),
        ),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        return await registry_manager.register_model(state)


def _row(db: FakeAsyncSupabase, row_id: str) -> Dict[str, Any]:
    return next(r for r in db.rows("ml_model_registry") if r["id"] == row_id)


async def _goldstd_retrain(cohort_override: Optional[Dict[str, Any]] = None):
    """A synthetic_gold parent (as on prod) retrained through the real chain."""
    db, ids = _goldstd_db()
    for row in db.rows("ml_model_registry"):
        row["training_provenance"] = "synthetic_gold"
    queued = await _trigger(db, ids["Kisqali"]["model"])
    training_config = dict(queued["training_config"], **(cohort_override or {}))
    pipeline_input = _cohort_input_from_training_config(training_config)
    scoped = await _scope(db, pipeline_input)
    out = await _register(
        db, scoped.experiment_id, pipeline_input["retrain_of"], pipeline_input["data_source"]
    )
    assert out["registration_successful"] is True
    return db, ids, pipeline_input, out


@pytest.mark.asyncio
async def test_a_retrain_on_the_synthetic_gold_contract_is_labelled_and_refused_production():
    from src.repositories.ml_experiment import MLModelRegistryRepository

    db, _, pipeline_input, out = await _goldstd_retrain()
    assert pipeline_input["data_source"]["filters"]["is_synthetic"] is True  # mig-151 shape
    candidate = _row(db, out["model_registry_id"])
    assert candidate["training_provenance"] == "synthetic_gold"

    repo = MLModelRegistryRepository(supabase_client=db)
    with pytest.raises(ValueError, match="synthetic_gold"):  # the #968 gate
        await repo.transition_stage(candidate["id"], "production")
    assert candidate.get("stage") != "production"
    assert await repo.transition_stage(candidate["id"], "staging") is True  # control


@pytest.mark.asyncio
async def test_a_retrain_whose_load_is_not_pinned_does_not_inherit_the_parents_label():
    """codex r1 HIGH: the parent's label says nothing about an unpinned load's rows."""
    unpinned = _table(brand="Kisqali")
    db, _, _, out = await _goldstd_retrain({"data_source": unpinned})
    assert _row(db, out["model_registry_id"])["training_provenance"] is None


@pytest.mark.asyncio
async def test_a_real_data_retrain_of_a_synthetic_gold_parent_is_real_and_promotable():
    """#968's prescribed remedy: the gate must not follow the parent onto real data."""
    from src.repositories.ml_experiment import MLModelRegistryRepository

    real = _table(brand="Kisqali", is_synthetic=False)
    db, _, _, out = await _goldstd_retrain({"data_source": real})
    candidate = _row(db, out["model_registry_id"])
    assert candidate["training_provenance"] == "real"
    repo = MLModelRegistryRepository(supabase_client=db)
    assert await repo.transition_stage(candidate["id"], "production") is True


def _plain_db() -> FakeAsyncSupabase:
    return FakeAsyncSupabase(
        {
            "ml_experiments": [
                {
                    "id": str(uuid4()),
                    "experiment_name": "Kisqali - treatment_initiated",
                    "mlflow_experiment_id": "exp_kisq_al_1",
                    "prediction_target": "treatment_initiated",
                    "is_synthetic": False,
                }
            ],
            "ml_model_registry": [],
            "ml_training_runs": [],
        }
    )


@pytest.mark.asyncio
async def test_a_non_retrain_run_on_a_pinned_synthetic_table_is_synthetic_gold():
    db = _plain_db()
    out = await _register(db, "exp_kisq_al_1", None, _table(brand="Kisqali", is_synthetic=True))
    assert _row(db, out["model_registry_id"])["training_provenance"] == "synthetic_gold"


@pytest.mark.asyncio
async def test_a_non_retrain_run_with_an_unpinned_source_stays_unknown():
    db = _plain_db()
    out = await _register(db, "exp_kisq_al_1", None, "patient_journeys")
    assert _row(db, out["model_registry_id"])["training_provenance"] is None


@pytest.mark.asyncio
async def test_a_reused_row_heals_a_null_provenance_but_never_overwrites_one():
    db = _plain_db()
    pinned = _table(brand="Kisqali", is_synthetic=True)
    first = await _register(db, "exp_kisq_al_1", None, pinned, run_id="run-a")
    row = _row(db, first["model_registry_id"])
    row["training_provenance"] = None  # a row written before #2255
    # the same run re-deployed (the reuse path) heals the NULL column
    again = await _register(db, "exp_kisq_al_1", None, pinned, run_id="run-a")
    assert again["model_registry_id"] == row["id"]
    assert row["training_provenance"] == "synthetic_gold"
    real = _table(brand="Kisqali", is_synthetic=False)
    await _register(db, "exp_kisq_al_1", None, real, run_id="run-a")
    assert row["training_provenance"] == "synthetic_gold"  # heal is NULL-only


# ------------------------------------------------------- codex r1 (2026-09-23) findings


@pytest.mark.parametrize(
    "pinned, augmented, expected",
    [
        (False, True, "mixed"),  # real rows + synthetic augmentation rows
        (True, True, "synthetic_gold"),  # every row is synthetic either way
        (False, False, "real"),
        (None, True, None),  # unknown load stays unknown
    ],
)
def test_synthetic_augmentation_is_part_of_the_provenance(pinned, augmented, expected):
    from src.agents.ml_foundation.model_deployer.nodes.training_provenance import (
        candidate_training_provenance,
    )

    source = _table(brand="Kisqali") if pinned is None else _table(is_synthetic=pinned)
    state = {"data_source": source, "training_augmentation_applied": augmented}
    assert candidate_training_provenance(state) == expected


@pytest.mark.asyncio
async def test_the_augmentation_flag_reaches_the_deployer_state():
    """pipeline -> deployer input -> ModelDeployerAgent.run -> declared state field."""
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState
    from src.agents.tier_0.pipeline import (
        MLFoundationPipeline,
        PipelineConfig,
        PipelineResult,
        PipelineStage,
    )

    assert ModelDeployerState(audit_workflow_id=uuid4()).training_augmentation_applied is False
    deployer_input: Dict[str, Any] = {}

    class _Deployer:
        async def run(self, payload):
            deployer_input.update(payload)
            return {"deployment_successful": False}

    pipeline = MLFoundationPipeline(PipelineConfig())
    result = PipelineResult(
        pipeline_run_id="run-1", status="running", current_stage=PipelineStage.MODEL_DEPLOYMENT
    )
    result.experiment_id = "exp_x"
    result.scope_spec = {}
    result.training_result = {"model_artifact_uri": "runs:/x/model", "validation_metrics": {}}
    result.training_augmentation = {"applied": True, "rows_added": 40}
    with (
        patch.object(MLFoundationPipeline, "_get_agent", return_value=_Deployer()),
        patch.object(MLFoundationPipeline, "_get_audit_service", return_value=None),
    ):
        await pipeline._run_model_deployment(
            {"target_outcome": "treatment_initiated", "data_source": "patient_journeys"},
            result,
            None,
        )
    assert deployer_input["training_augmentation_applied"] is True

    captured: Dict[str, Any] = {}

    class _Graph:
        async def ainvoke(self, initial_state, *_a, **_k):
            captured.update(dict(initial_state))
            return {**initial_state, "deployment_successful": False}

    agent = ModelDeployerAgent.__new__(ModelDeployerAgent)
    agent.agent_name = "model_deployer"
    agent.tier = 0
    agent.graph = _Graph()
    with patch(
        "src.agents.ml_foundation.model_deployer.agent._get_opik_connector", return_value=None
    ):
        try:
            await agent.run(
                {
                    "model_uri": "runs:/x/model",
                    "experiment_id": "exp_x",
                    "validation_metrics": {},
                    "success_criteria_met": True,
                    "deployment_name": "d",
                    "training_augmentation_applied": True,
                }
            )
        except Exception:
            pass  # the stub graph's output need not satisfy the agent's post-processing
    assert captured["training_augmentation_applied"] is True


@pytest.mark.asyncio
async def test_reuse_heals_only_a_provenance_the_stored_cohort_agrees_with():
    """codex r1 HIGH: a redeploy claiming a different cohort than the artifact's stored
    one must not relabel it (a stored synthetic cohort healed 'real' would pass #968)."""
    db = _plain_db()
    synthetic = _table(brand="Kisqali", is_synthetic=True)
    first = await _register(db, "exp_kisq_al_1", None, synthetic, run_id="run-a")
    row = _row(db, first["model_registry_id"])
    row["training_provenance"] = None  # a row written before #2255
    real = _table(brand="Kisqali", is_synthetic=False)
    await _register(db, "exp_kisq_al_1", None, real, run_id="run-a")
    assert row["training_provenance"] is None  # the claim contradicts the stored cohort
    await _register(db, "exp_kisq_al_1", None, synthetic, run_id="run-a")
    assert row["training_provenance"] == "synthetic_gold"  # agreeing claim heals
