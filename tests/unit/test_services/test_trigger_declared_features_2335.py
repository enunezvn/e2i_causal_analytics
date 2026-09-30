"""#2335: the manual retrain trigger refuses a job whose requirement is undeclared.

``POST /api/monitoring/retraining/trigger/{id}`` accepted a bare table name or a
column-less / file_dir ``data_source`` and had no ``candidate_features`` field, so such a
job was recorded and enqueued, then (since the scope stage now fails closed on an
undeclared requirement) could only fail. The trigger now:

- accepts an optional ``candidate_features`` and threads it through ``cohort_contract()``
  into the retrain's ``training_config``;
- refuses with 422 — nothing recorded, nothing enqueued — when the EFFECTIVE contract
  (request merged over the registry row) declares no source.

Real service code over the in-memory async supabase fake; patched only: drift history,
the perf tracker and Celery ``.delay`` (trigger side channels).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.cohort_contract import UNDECLARED_REQUIRED_FEATURES
from src.services.retraining_trigger import (
    RetrainingTriggerService,
    RetrainRefusedError,
    TriggerReason,
)
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

pytestmark = pytest.mark.unit

COLUMNS = ["disease_severity", "age_at_diagnosis", "treatment_initiated"]


def _table_contract(columns: Optional[list] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": "table",
        "table": "patient_journeys",
        "filters": {"brand": "Kisqali", "is_synthetic": True},
    }
    if columns is not None:
        out["columns"] = columns
    return out


def _db(cohort_data_source: Optional[str]) -> tuple[FakeAsyncSupabase, str]:
    exp_id, model_id = str(uuid4()), str(uuid4())
    db = FakeAsyncSupabase(
        {
            "ml_experiments": [
                {
                    "id": exp_id,
                    "experiment_name": "initiation_kisqali_goldstd_eval_v1",
                    "prediction_target": "initiation_kisqali",
                }
            ],
            "ml_model_registry": [
                {
                    "id": model_id,
                    "experiment_id": exp_id,
                    "model_name": "initiation_kisqali_goldstd_lr_v1",
                    "model_version": "1.0",
                    "stage": "staging",
                    "cohort_data_source": cohort_data_source,
                    "cohort_target_outcome": "treatment_initiated",
                    "cohort_feature_manifest_source": None,
                }
            ],
            "ml_retraining_history": [],
            "ml_training_runs": [],
        }
    )
    return db, model_id


async def _trigger(
    db: FakeAsyncSupabase, handle: str, cohort: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
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
        try:
            job = await service.trigger_retraining(
                model_version=handle, reason=TriggerReason.MANUAL, cohort=cohort
            )
        finally:
            delay_calls = mock_task.delay.call_count
    return {"job": job, "delay_calls": delay_calls}


# --------------------------------------------------------------------------- service


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registry_source",
    [
        "patient_journeys",  # bare table name
        json.dumps({"type": "file_dir", "path": "data/rwd/optum/initiation"}),
        json.dumps(_table_contract()),  # table dict without columns
        json.dumps(_table_contract(["treatment_initiated"])),  # columns = only the label
    ],
)
async def test_an_undeclared_requirement_is_refused_with_422_and_nothing_recorded(
    registry_source: str,
) -> None:
    db, model_id = _db(registry_source)
    with pytest.raises(RetrainRefusedError) as exc:
        await _trigger(db, model_id, cohort=None)

    assert exc.value.reason == "undeclared_required_features"
    assert exc.value.http_status == 422
    assert UNDECLARED_REQUIRED_FEATURES in str(exc.value)
    assert db.rows("ml_retraining_history") == []


@pytest.mark.asyncio
async def test_a_request_without_any_data_source_is_refused_too() -> None:
    db, model_id = _db(None)
    with pytest.raises(RetrainRefusedError) as exc:
        await _trigger(db, model_id, cohort={"target_outcome": "treatment_initiated"})
    assert exc.value.http_status == 422
    assert db.rows("ml_retraining_history") == []


@pytest.mark.asyncio
async def test_request_candidate_features_declare_the_requirement_and_reach_training_config() -> (
    None
):
    db, model_id = _db("patient_journeys")
    out = await _trigger(
        db, model_id, cohort={"candidate_features": ["disease_severity", "age_at_diagnosis"]}
    )

    assert out["delay_calls"] == 1
    cfg = out["job"].training_config
    assert cfg["candidate_features"] == ["disease_severity", "age_at_diagnosis"]
    assert cfg["data_source"] == "patient_journeys"
    assert len(db.rows("ml_retraining_history")) == 1


@pytest.mark.asyncio
async def test_a_registry_contract_with_columns_still_enqueues() -> None:
    db, model_id = _db(json.dumps(_table_contract(COLUMNS), sort_keys=True))
    out = await _trigger(db, model_id, cohort=None)
    assert out["delay_calls"] == 1
    assert "candidate_features" not in out["job"].training_config


@pytest.mark.asyncio
async def test_request_columns_merged_over_a_column_less_row_declare_the_requirement() -> None:
    db, model_id = _db("patient_journeys")
    out = await _trigger(db, model_id, cohort={"data_source": _table_contract(COLUMNS)})
    assert out["delay_calls"] == 1


# --------------------------------------------------------------------------- API model + route


def test_the_request_model_threads_candidate_features_into_the_cohort_contract() -> None:
    from src.api.routes.monitoring import TriggerRetrainingRequest

    req = TriggerRetrainingRequest(
        reason="manual", data_source="patient_journeys", candidate_features=["a", "b"]
    )
    assert req.cohort_contract()["candidate_features"] == ["a", "b"]
    assert "candidate_features" not in TriggerRetrainingRequest(reason="manual").cohort_contract()


@pytest.fixture
def client() -> TestClient:
    from src.api.dependencies.auth import require_admin
    from src.api.routes.monitoring import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: {"role": "admin"}
    return TestClient(app)


def test_the_route_returns_422_on_an_undeclared_requirement(client: TestClient) -> None:
    with patch("src.services.retraining_trigger.get_retraining_trigger_service") as get_service:
        service = MagicMock()
        service.trigger_retraining = AsyncMock(
            side_effect=RetrainRefusedError(
                "m1", "undeclared_required_features", UNDECLARED_REQUIRED_FEATURES
            )
        )
        get_service.return_value = service
        response = client.post(
            "/monitoring/retraining/trigger/m1",
            json={"reason": "manual", "data_source": "patient_journeys"},
        )

    assert response.status_code == 422, response.text
    assert UNDECLARED_REQUIRED_FEATURES in response.json()["detail"]


def test_the_route_threads_request_candidates_to_the_service(client: TestClient) -> None:
    with patch("src.services.retraining_trigger.get_retraining_trigger_service") as get_service:
        service = MagicMock()
        service.trigger_retraining = AsyncMock(side_effect=RuntimeError("stop here"))
        get_service.return_value = service
        client.post(
            "/monitoring/retraining/trigger/m1",
            json={"reason": "manual", "candidate_features": ["a"]},
        )

    assert service.trigger_retraining.await_args.kwargs["cohort"]["candidate_features"] == ["a"]
