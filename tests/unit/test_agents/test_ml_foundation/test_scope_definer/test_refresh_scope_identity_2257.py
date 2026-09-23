"""#2257: a non-retrain run whose scope name already exists runs under THAT row's id.

Before: ``_persist_scope_spec``'s get-or-refresh-by-name path (added 2026-07-11, PR
#1197, to stop every Tier-0 run inserting a duplicate scope row) refreshed the existing
row's columns but returned without persisting the freshly minted experiment id. The
pipeline then ran under that minted id; the trainer's and the registry writer's
``get_by_mlflow_id`` resolved nothing, so no ``ml_training_runs`` row and no
``ml_model_registry`` candidate were written (every same-name duplicate on prod has
zero training runs). ``get_by_name`` was an unordered ``limit(1)``, so with several rows
of one name (8 x ``Remibrutinib - treatment_initiated`` on prod) the refreshed row was
whichever the database returned first.

After: the refresh binds the run to the row it refreshes — the row's
``mlflow_experiment_id`` becomes the run's experiment id (compare-and-set of the minted
id into a NULL column, the #2242 mechanism) — and the row is chosen deterministically
(oldest ``created_at``, then ``id``).

Real code over the in-memory async supabase fake; patched only: scope_definer's
memory / Opik writers, the pipeline audit service, and the MLflow registration call.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from tests.unit._fakes.async_supabase import FakeAsyncSupabase

pytestmark = pytest.mark.unit

_NAME = "Remibrutinib - treatment_initiated"


def _input() -> Dict[str, Any]:
    """A NON-retrain pipeline input whose scope name is ``_NAME``."""
    return {
        "problem_description": "Predict treatment initiation",
        "business_objective": "Target likely initiators",
        "target_outcome": "treatment_initiated",
        "brand": "Remibrutinib",
    }


def _row(mlflow_id: Any, created_at: str, **extra: Any) -> Dict[str, Any]:
    return {
        "id": str(uuid4()),
        "experiment_name": _NAME,
        "mlflow_experiment_id": mlflow_id,
        "prediction_target": "",
        "brand": "Remibrutinib",
        "created_at": created_at,
        "is_synthetic": False,
        **extra,
    }


async def _scope(db: FakeAsyncSupabase, pipeline_input: Dict[str, Any]):
    from src.agents.tier_0.pipeline import (
        MLFoundationPipeline,
        PipelineConfig,
        PipelineResult,
        PipelineStage,
    )
    from src.repositories.ml_experiment import MLExperimentRepository

    pipeline = MLFoundationPipeline(PipelineConfig())
    result = PipelineResult(
        pipeline_run_id="run-2257",
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


def _resolved(db: FakeAsyncSupabase, experiment_id: str) -> List[Dict[str, Any]]:
    return [r for r in db.rows("ml_experiments") if r["mlflow_experiment_id"] == experiment_id]


@pytest.mark.asyncio
async def test_a_refreshed_scope_runs_under_the_existing_rows_experiment_id():
    db = FakeAsyncSupabase({"ml_experiments": [_row("exp_remi_al_20260610180738_69f516", "t1")]})

    result = await _scope(db, _input())

    assert result.experiment_id == "exp_remi_al_20260610180738_69f516"
    assert [r["experiment_name"] for r in _resolved(db, result.experiment_id)] == [_NAME]
    # one identity across the scope's outputs (criteria_validator stamps the minted id)
    assert result.scope_spec["experiment_id"] == result.experiment_id
    assert result.success_criteria["experiment_id"] == result.experiment_id
    assert len(db.rows("ml_experiments")) == 1  # refreshed, not duplicated (#1197)
    assert db.rows("ml_experiments")[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_duplicate_names_resolve_to_the_oldest_row_on_every_run():
    """8 same-name rows live on prod: the refresh must pick one deterministically."""
    newer = _row("exp_remi_newer", "2026-06-11T00:22:32+00:00")
    oldest = _row("exp_remi_oldest", "2026-06-10T18:01:10+00:00")
    middle = _row("exp_remi_middle", "2026-06-10T18:07:38+00:00")
    db = FakeAsyncSupabase({"ml_experiments": [newer, oldest, middle]})  # scrambled

    first = await _scope(db, _input())
    second = await _scope(db, _input())

    assert first.experiment_id == second.experiment_id == "exp_remi_oldest"
    by_id = {r["id"]: r for r in db.rows("ml_experiments")}
    assert by_id[oldest["id"]]["status"] == "completed"
    # the other duplicates are left exactly as they were
    assert "status" not in by_id[newer["id"]] and "status" not in by_id[middle["id"]]
    assert len(by_id) == 3


@pytest.mark.asyncio
async def test_a_row_without_an_experiment_id_takes_the_minted_one():
    row = _row(None, "t1")
    db = FakeAsyncSupabase({"ml_experiments": [row]})

    result = await _scope(db, _input())

    assert result.experiment_id.startswith("exp_remi_")
    assert row["mlflow_experiment_id"] == result.experiment_id
    assert [r["id"] for r in _resolved(db, result.experiment_id)] == [row["id"]]


@pytest.mark.asyncio
async def test_a_set_experiment_id_is_never_overwritten():
    """compare-and-set: only a NULL column takes the minted id (older runs' ids stay
    resolvable)."""
    row = _row("exp_remi_existing", "t1")
    db = FakeAsyncSupabase({"ml_experiments": [row]})
    await _scope(db, _input())
    await _scope(db, _input())
    assert row["mlflow_experiment_id"] == "exp_remi_existing"


@pytest.mark.asyncio
async def test_a_first_run_still_creates_the_row_under_its_minted_id():
    db = FakeAsyncSupabase({"ml_experiments": []})
    result = await _scope(db, _input())
    (row,) = db.rows("ml_experiments")
    assert row["experiment_name"] == _NAME
    assert row["mlflow_experiment_id"] == result.experiment_id


@pytest.mark.asyncio
async def test_a_refreshed_scopes_candidate_is_registered():
    """The harm in #2257: the registry writer resolved nothing and wrote no row."""
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    db = FakeAsyncSupabase(
        {
            "ml_experiments": [_row("exp_remi_al_20260610180738_69f516", "t1")],
            "ml_model_registry": [],
            "ml_training_runs": [],
        }
    )
    result = await _scope(db, _input())
    (exp,) = _resolved(db, result.experiment_id)  # the trainer persists only under this
    db.rows("ml_training_runs").append(
        {
            "id": str(uuid4()),
            "experiment_id": exp["id"],
            "run_name": "train_x",
            "mlflow_run_id": "run-2257",
            "algorithm": "LogisticRegression",
            "hyperparameters": {},
            "status": "finished",
            "is_synthetic": False,
        }
    )
    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri="runs:/run-2257/model",
        experiment_id=result.experiment_id,
        deployment_name=f"{result.experiment_id}_deployment",
        validation_metrics={"roc_auc": 0.81},
        success_criteria_met=True,
    )
    mlflow = AsyncMock(side_effect=lambda uri, name: (name, 1, "None"))
    with (
        patch.object(registry_manager, "_register_model_mlflow", mlflow),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        out = await registry_manager.register_model(state)

    (candidate,) = db.rows("ml_model_registry")
    assert out["model_registry_id"] == candidate["id"]
    assert candidate["experiment_id"] == exp["id"]


@pytest.mark.asyncio
async def test_a_refresh_whose_id_cannot_be_bound_fails_closed():
    """codex r3: once an existing row is chosen, running on under the minted id would
    recreate the unresolvable identity silently — the scope stage must fail instead."""
    row = _row(None, "t1")
    db = FakeAsyncSupabase({"ml_experiments": [row]})
    with patch(
        "src.agents.ml_foundation.scope_definer.agent._claim_experiment_id",
        AsyncMock(side_effect=RuntimeError("db went away")),
    ):
        with pytest.raises(RuntimeError, match="could not be bound"):
            await _scope(db, _input())
    assert len(db.rows("ml_experiments")) == 1  # nothing created in its place


@pytest.mark.asyncio
async def test_a_failure_before_any_row_is_chosen_still_degrades_gracefully():
    """The pre-#2257 contract: no usable repository -> the run continues on its minted id."""
    from src.repositories.ml_experiment import MLExperimentRepository

    repo = MLExperimentRepository(supabase_client=FakeAsyncSupabase({}))
    repo.get_by_name = AsyncMock(side_effect=RuntimeError("db down"))  # type: ignore[method-assign]
    with patch(
        "src.agents.ml_foundation.scope_definer.agent._get_experiment_repository",
        AsyncMock(return_value=repo),
    ):
        from src.agents.ml_foundation.scope_definer.agent import ScopeDefinerAgent

        output = {"experiment_id": "exp_minted", "experiment_name": _NAME}
        assert await ScopeDefinerAgent()._persist_scope_spec(output) is None
