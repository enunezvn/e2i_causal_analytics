"""#2248 option (a), codex r1 HIGH: a retrained model records the calibration it deployed.

The registry row a pipeline deploy writes copies the training run's hyperparameters
(``registry_manager``: ``hyperparameters=run.hyperparameters``). The trainer persists the
run, so the post-hoc method actually applied to the deployed estimator is merged there —
otherwise the next retrain of a sigmoid-calibrated retrain would revert to the auto policy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ml_foundation.model_trainer.agent import (
    ModelTrainerAgent,
    hyperparameters_of_record,
)
from src.services.cohort_contract import contract_from_registry_row

_HP = {"C": 1.7757, "solver": "saga", "penalty": "l2"}


@pytest.mark.unit
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_the_applied_method_is_merged_into_the_runs_hyperparameters(method):
    output = {
        "best_hyperparameters": dict(_HP),
        "calibration_applied": True,
        "post_hoc_calibration": {
            "calibration_method": "auto",
            "calibration_method_resolved": method,
            "calibration_applied": True,
        },
    }
    assert hyperparameters_of_record(output) == {**_HP, "calibration_method": method}
    # and the next retrain's contract reads it back off the registry row
    row = {"hyperparameters": hyperparameters_of_record(output)}
    assert contract_from_registry_row(row)["calibration_method"] == method


@pytest.mark.unit
@pytest.mark.parametrize(
    "output",
    [
        {
            "calibration_applied": False,
            "post_hoc_calibration": {
                "calibration_method_resolved": "isotonic",
                "calibration_applied": False,
            },
        },
        {
            "calibration_applied": True,
            "post_hoc_calibration": {
                "calibration_applied": False,
                "skip_reason": "skip_post_hoc_calibration_flag",
            },
        },
        {
            "calibration_applied": True,
            "post_hoc_calibration": {
                "calibration_method_resolved": "platt",
                "calibration_applied": True,
            },
        },
        {},
    ],
)
def test_nothing_is_recorded_when_no_known_method_was_deployed(output):
    out = hyperparameters_of_record({"best_hyperparameters": dict(_HP), **output})
    assert out == _HP


@pytest.mark.unit
def test_the_trainers_own_hyperparameters_are_not_mutated():
    hp = dict(_HP)
    hyperparameters_of_record(
        {
            "best_hyperparameters": hp,
            "calibration_applied": True,
            "post_hoc_calibration": {
                "calibration_method_resolved": "sigmoid",
                "calibration_applied": True,
            },
        }
    )
    assert hp == _HP


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_persisted_training_run_carries_the_method():
    repo = MagicMock()
    repo.create_run_with_hpo = AsyncMock(return_value=SimpleNamespace(id="r1", run_name="t"))
    repo.update_run_metrics = AsyncMock(return_value=True)  # #2296: the run is finalised
    repo.complete_run = AsyncMock(return_value=True)
    exp_repo = MagicMock()
    exp_repo.get_by_mlflow_id = AsyncMock(
        return_value=SimpleNamespace(id="11111111-1111-1111-1111-111111111111")
    )
    output: Dict[str, Any] = {
        "experiment_id": "exp-2248",
        "algorithm_name": "LogisticRegression",
        "best_hyperparameters": dict(_HP),
        "calibration_applied": True,
        "post_hoc_calibration": {
            "calibration_method_resolved": "sigmoid",
            "calibration_applied": True,
        },
    }
    with (
        patch(
            "src.agents.ml_foundation.model_trainer.agent._get_training_run_repository",
            AsyncMock(return_value=repo),
        ),
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            AsyncMock(return_value=object()),
        ),
        patch("src.repositories.ml_experiment.MLExperimentRepository", return_value=exp_repo),
    ):
        await ModelTrainerAgent()._persist_training_run(output)
    assert repo.create_run_with_hpo.await_args.kwargs["hyperparameters"] == {
        **_HP,
        "calibration_method": "sigmoid",
    }
