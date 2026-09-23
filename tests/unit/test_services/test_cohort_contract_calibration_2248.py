"""#2248 option (a), owner decision 2026-09-23: a retrain uses the calibration method of
the model it retrains.

The parent's method of record is ``ml_model_registry.hyperparameters.calibration_method``
(the ``method`` hyperparameter of the registered ``CalibratedClassifierCV``). The
cohort contract carries it to the retrain; a row without one keeps today's auto policy
and the retrain says so. A value outside the two methods the evaluator can apply is
never guessed into one.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.services.cohort_contract import (
    contract_from_registry_row,
    contract_from_training_config,
    heal_registry_cohort_contract,
    load_registry_cohort_contract,
)
from src.tasks.drift_monitoring_tasks import (
    _cohort_input_from_training_config,
    _execute_real_retraining,
)
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

_ROW = {
    "id": "x",
    "cohort_data_source": "patient_journeys",
    "cohort_target_outcome": "treatment_initiated",
    "cohort_feature_manifest_source": "synthetic_csu",
}


@pytest.mark.unit
@pytest.mark.parametrize(
    "hyperparameters",
    [{"calibration_method": "sigmoid"}, json.dumps({"calibration_method": "sigmoid", "C": 1})],
)
def test_the_registered_calibration_method_joins_the_contract(hyperparameters):
    contract = contract_from_registry_row({**_ROW, "hyperparameters": hyperparameters})
    assert contract["calibration_method"] == "sigmoid"


@pytest.mark.unit
@pytest.mark.parametrize(
    "hyperparameters",
    [
        None,
        {},
        {"C": 1.0},
        {"calibration_method": "platt"},  # not a method the evaluator applies: never mapped
        {"calibration_method": "auto"},  # a policy, not the parent's recorded method
        {"calibration_method": None},
        "not json",
        json.dumps(["sigmoid"]),
    ],
)
def test_no_recorded_method_means_no_contract_key(hyperparameters):
    contract = contract_from_registry_row({**_ROW, "hyperparameters": hyperparameters})
    assert "calibration_method" not in contract
    assert contract["target_outcome"] == "treatment_initiated"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_registry_load_reads_the_hyperparameters_column():
    rid = str(uuid4())
    selected: list = []

    class _Recording(FakeAsyncSupabase):
        def table(self, name):
            q = super().table(name)
            real_select = q.select

            def _select(*cols, **kw):
                selected.append(cols)
                return real_select(*cols, **kw)

            q.select = _select  # type: ignore[method-assign]
            return q

    db = _Recording(
        {
            "ml_model_registry": [
                {**_ROW, "id": rid, "hyperparameters": {"calibration_method": "sigmoid"}}
            ]
        }
    )
    _, contract = await load_registry_cohort_contract(db, rid)
    assert contract["calibration_method"] == "sigmoid"
    assert any("hyperparameters" in str(c) for c in selected)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_heal_never_writes_the_calibration_method_as_a_column():
    """It is not a cohort_* column; healing a contract that carries it writes only the
    NULL contract columns."""
    rid = str(uuid4())
    db = FakeAsyncSupabase(
        {
            "ml_model_registry": [
                {
                    "id": rid,
                    "cohort_data_source": None,
                    "cohort_target_outcome": None,
                    "cohort_feature_manifest_source": None,
                }
            ]
        }
    )
    written = await heal_registry_cohort_contract(
        db, rid, {"data_source": "t", "target_outcome": "y", "calibration_method": "sigmoid"}
    )
    assert set(written) == {"cohort_data_source", "cohort_target_outcome"}
    assert "calibration_method" not in contract_from_training_config(
        {"data_source": "t", "target_outcome": "y", "calibration_method": "sigmoid"}
    )


# ---------------------------------------------------------------------------
# the retrain input
# ---------------------------------------------------------------------------

_CONTRACT = {"data_source": "patient_journeys", "target_outcome": "treatment_initiated"}


@pytest.mark.unit
def test_the_retrain_hands_the_pipeline_the_parents_method():
    pipeline_input = _cohort_input_from_training_config(
        {**_CONTRACT, "calibration_method": "sigmoid"}
    )
    assert pipeline_input["calibration_method"] == "sigmoid"


@pytest.mark.unit
def test_without_a_recorded_method_the_retrain_keeps_auto_and_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger="src.tasks.drift_monitoring_tasks"):
        pipeline_input = _cohort_input_from_training_config(dict(_CONTRACT))
    assert "calibration_method" not in pipeline_input
    assert any("calibration method" in r.getMessage() for r in caplog.records)


def _patched(result: Any):
    pipe = MagicMock()
    pipe.run = AsyncMock(return_value=result)
    repo = MagicMock()
    repo.update = AsyncMock()
    repo.mark_failed = AsyncMock()
    repo.get_by_id = AsyncMock(return_value=None)
    service = MagicMock()
    service.complete_retraining = AsyncMock()
    patches = (
        patch("src.agents.tier_0.pipeline.MLFoundationPipeline", MagicMock(return_value=pipe)),
        patch("src.repositories.drift_monitoring.RetrainingHistoryRepository", return_value=repo),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
        ),
    )
    return patches, repo


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra, expected",
    [
        ({"calibration_method": "sigmoid"}, "calibration_method=sigmoid"),
        ({}, "calibration_method=auto (the parent's method is not recorded)"),
    ],
)
async def test_a_refused_retrain_records_which_calibration_it_ran(extra, expected):
    from types import SimpleNamespace

    result = SimpleNamespace(
        status="completed",
        deployment_result=None,
        training_result={"validation_metrics": {"roc_auc": 0.8}, "success_criteria_met": False},
    )
    patches, repo = _patched(result)
    with patches[0], patches[1], patches[2]:
        out = await _execute_real_retraining("rt", "v1", "v2", {**_CONTRACT, **extra})
    assert out["status"] == "failed"
    assert expected in repo.mark_failed.await_args.args[1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_completed_retrain_reports_the_calibration_it_ran():
    from types import SimpleNamespace

    result = SimpleNamespace(
        status="completed",
        deployment_result={"model_version": "v2"},
        training_result={"validation_metrics": {"roc_auc": 0.8}, "success_criteria_met": True},
    )
    patches, _ = _patched(result)
    with patches[0], patches[1], patches[2]:
        out = await _execute_real_retraining(
            "rt", "v1", "v2", {**_CONTRACT, "calibration_method": "sigmoid"}
        )
    assert out["status"] == "completed"
    assert out["calibration_method"] == "sigmoid"
