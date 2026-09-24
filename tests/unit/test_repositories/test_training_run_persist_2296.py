"""#2296: the trainer's training-run row never finalised, so no candidate could be registered.

Measured on the real Kisqali retrain (2026-09-24): ``test_metrics`` carries three NaN
values — ``net_benefit_at_primary_tau`` / ``..._treat_all`` / ``..._relative_to_treat_all``
— because no disease-specific ``primary_tau`` is configured for the cohort, so the
evaluator emits its documented "not computed" block (``evaluator.py`` ``nan_block``).
supabase-py's strict JSON encoder rejects NaN before the request is sent, the trainer
swallowed that at WARNING, and every ``ml_training_runs`` row stayed ``running`` with
empty metrics. Separately, nothing ever finalised a run, and the registry writer's
fallback (``get_best_run``) looked for ``status='finished'``, a value no writer uses.

Tests run the REAL repository against the in-memory async-supabase fake, which accepts
whatever payload reaches it; strict-JSON compliance is asserted explicitly.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.repositories.ml_experiment import MLTrainingRunRepository
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

# The shape the evaluator emits for this cohort (values from the live run).
TEST_METRICS: Dict[str, Any] = {
    "roc_auc": 0.8352780405760538,
    "precision": 0.6267029972752044,
    "primary_tau": None,
    "net_benefit_at_primary_tau": float("nan"),
    "net_benefit_at_primary_tau_treat_all": float("nan"),
    "net_benefit_at_primary_tau_relative_to_treat_all": float("nan"),
    "nb_anchor_passes": None,
    "net_benefit_grid": {"p_t=0.30": 0.1971, "p_t=0.50": float("inf")},
}


def _strict(payload: Any) -> None:
    json.dumps(payload, allow_nan=False)


async def _new_run(db: FakeAsyncSupabase, experiment_id, **kw):
    repo = MLTrainingRunRepository(supabase_client=db)
    run = await repo.create_run_with_hpo(
        experiment_id=experiment_id,
        run_name="train_2296",
        mlflow_run_id=kw.get("mlflow_run_id", "mlf-2296"),
        algorithm="LogisticRegression",
        hyperparameters=kw.get("hyperparameters", {"C": 1.36, "tol": float("nan")}),
        training_samples=6888,
        feature_names=["a"],
    )
    return repo, run


# ---------------------------------------------------------------------------
# NaN / Inf -> JSON null at the persistence boundary (never 0, never invented)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_finite_metrics_are_stored_as_null_and_finite_ones_untouched():
    db = FakeAsyncSupabase({"ml_training_runs": []})
    repo, run = await _new_run(db, uuid4())
    assert await repo.update_run_metrics(
        run_id=run.id, train_metrics={"roc_auc": 0.85}, test_metrics=TEST_METRICS
    )
    (row,) = db.rows("ml_training_runs")
    _strict(row["test_metrics"])
    _strict(row["hyperparameters"])
    tm = row["test_metrics"]
    assert tm["net_benefit_at_primary_tau"] is None
    assert tm["net_benefit_at_primary_tau_treat_all"] is None
    assert tm["net_benefit_at_primary_tau_relative_to_treat_all"] is None
    assert tm["net_benefit_grid"] == {"p_t=0.30": 0.1971, "p_t=0.50": None}
    assert tm["roc_auc"] == 0.8352780405760538 and tm["precision"] == 0.6267029972752044
    assert row["hyperparameters"] == {"C": 1.36, "tol": None}
    assert row["train_metrics"] == {"roc_auc": 0.85}


# ---------------------------------------------------------------------------
# a finalised run is found by the registry writer's fallback lookup
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_completed_run_is_the_experiments_best_run():
    exp = uuid4()
    db = FakeAsyncSupabase({"ml_training_runs": []})
    repo, run = await _new_run(db, exp)
    await repo.update_run_metrics(run_id=run.id, test_metrics=TEST_METRICS)
    assert await repo.get_best_run(exp) is None  # still running: not a candidate
    assert await repo.complete_run(run.id)
    (row,) = db.rows("ml_training_runs")
    assert row["status"] == "completed"  # the value every reader of the table uses
    best = await repo.get_best_run(exp)
    assert best is not None and str(best.id) == str(run.id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_best_run_ranks_by_the_auc_the_evaluator_actually_emits():
    exp = uuid4()
    db = FakeAsyncSupabase({"ml_training_runs": []})
    repo = MLTrainingRunRepository(supabase_client=db)
    ids = {}
    for name, auc in (("low", 0.70), ("high", 0.84)):
        _, run = await _new_run(db, exp, mlflow_run_id=f"mlf-{name}")
        await repo.update_run_metrics(run_id=run.id, test_metrics={"roc_auc": auc})
        await repo.complete_run(run.id)
        ids[name] = str(run.id)
    best = await repo.get_best_run(exp)
    assert str(best.id) == ids["high"]


# ---------------------------------------------------------------------------
# the trainer finalises its run, and a failed write is loud
# ---------------------------------------------------------------------------


def _output(experiment_id: str) -> Dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "training_run_id": "train_2296",
        "mlflow_run_id": "mlf-2296",
        "algorithm_name": "LogisticRegression",
        "best_hyperparameters": {"C": 1.36},
        "train_metrics": {"roc_auc": 0.85},
        "validation_metrics": {"roc_auc": 0.8375},
        "test_metrics": dict(TEST_METRICS),
    }


def _agent_patches(db, exp_uuid):
    from types import SimpleNamespace

    exp_repo = AsyncMock()
    exp_repo.get_by_mlflow_id = AsyncMock(return_value=SimpleNamespace(id=exp_uuid))
    return (
        patch(
            "src.agents.ml_foundation.model_trainer.agent._get_training_run_repository",
            AsyncMock(return_value=MLTrainingRunRepository(supabase_client=db)),
        ),
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            AsyncMock(return_value=db),
        ),
        patch("src.repositories.ml_experiment.MLExperimentRepository", return_value=exp_repo),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_trainer_persists_and_completes_its_run():
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    exp = uuid4()
    db = FakeAsyncSupabase({"ml_training_runs": []})
    p1, p2, p3 = _agent_patches(db, exp)
    with p1, p2, p3:
        assert await ModelTrainerAgent()._persist_training_run(_output("exp_kisq_al_x")) is True
    (row,) = db.rows("ml_training_runs")
    assert row["status"] == "completed"
    assert row["test_metrics"]["roc_auc"] == 0.8352780405760538
    assert row["test_metrics"]["net_benefit_at_primary_tau"] is None
    best = await MLTrainingRunRepository(supabase_client=db).get_best_run(exp)
    assert best is not None and best.mlflow_run_id == "mlf-2296"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_failed_training_run_write_is_loud(caplog):
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    class _Refusing(FakeAsyncSupabase):
        def table(self, name):
            q = super().table(name)
            real = q.execute

            async def _execute():
                if q._op == "update":
                    raise ValueError("Out of range float values are not JSON compliant: nan")
                return await real()

            q.execute = _execute  # type: ignore[method-assign]
            return q

    db = _Refusing({"ml_training_runs": []})
    p1, p2, p3 = _agent_patches(db, uuid4())
    with p1, p2, p3, caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="training run"):
            await ModelTrainerAgent()._persist_training_run(_output("exp_kisq_al_x"))
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# codex r1 MED: zero-row writes are failures, and a half-written run is marked failed
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_updates_that_match_no_row_report_failure():
    db = FakeAsyncSupabase({"ml_training_runs": []})
    repo = MLTrainingRunRepository(supabase_client=db)
    ghost = uuid4()
    assert await repo.update_run_metrics(run_id=ghost, test_metrics={"roc_auc": 0.8}) is False
    assert await repo.complete_run(ghost) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_insert_that_returns_no_row_raises():
    class _NoRepresentation(FakeAsyncSupabase):
        def table(self, name):
            q = super().table(name)
            real = q.execute

            async def _execute():
                out = await real()
                if q._op == "insert":
                    out.data = []
                return out

            q.execute = _execute  # type: ignore[method-assign]
            return q

    with pytest.raises(RuntimeError, match="no row"):
        await _new_run(_NoRepresentation({"ml_training_runs": []}), uuid4())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_run_whose_metrics_write_fails_is_marked_failed_not_left_running():
    from src.agents.ml_foundation.model_trainer.agent import (
        ModelTrainerAgent,
        TrainingRunPersistenceError,
    )

    class _MetricsRefused(FakeAsyncSupabase):
        def table(self, name):
            q = super().table(name)
            real = q.execute

            async def _execute():
                if q._op == "update" and "test_metrics" in (q._payload or {}):
                    raise ValueError("metrics write refused")
                return await real()

            q.execute = _execute  # type: ignore[method-assign]
            return q

    db = _MetricsRefused({"ml_training_runs": []})
    p1, p2, p3 = _agent_patches(db, uuid4())
    with p1, p2, p3:
        with pytest.raises(TrainingRunPersistenceError):
            await ModelTrainerAgent()._persist_training_run(_output("exp_kisq_al_x"))
    (row,) = db.rows("ml_training_runs")
    assert row["status"] == "failed"
    assert "metrics write refused" in row["error_message"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repeated_split_folds_do_not_swallow_a_persistence_failure():
    """codex r1 MED: the fold handler turned every exception into a 'failed fold'."""
    import numpy as np
    import pandas as pd

    from src.agents.ml_foundation.model_trainer.agent import (
        ModelTrainerAgent,
        TrainingRunPersistenceError,
    )

    agent = ModelTrainerAgent()

    async def _run(_fold_input):
        raise TrainingRunPersistenceError("training run persistence failed: boom")

    X = pd.DataFrame({"a": np.arange(200, dtype=float), "b": np.arange(200, dtype=float) % 7})
    y = pd.Series(np.arange(200) % 2)
    with (
        patch.object(agent, "run", AsyncMock(side_effect=_run)),
        patch.object(agent, "_get_mlflow_connector_or_none", lambda: None),
    ):
        with pytest.raises(TrainingRunPersistenceError):
            await agent._run_repeated_splits(
                {
                    "model_candidate": {
                        "algorithm_name": "LogisticRegression",
                        "algorithm_class": "sklearn.linear_model.LogisticRegression",
                        "hyperparameter_search_space": {},
                        "default_hyperparameters": {},
                    },
                    "qc_report": {"qc_passed": True},
                    "experiment_id": "exp_repeated",
                    "evaluation_mode": "repeated_k10",
                    "enable_mlflow": False,
                    "full_data": {"X": X, "y": y},
                }
            )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_parent_mlflow_wrapper_does_not_rerun_folds_on_a_persistence_failure():
    import contextlib

    import numpy as np
    import pandas as pd

    from src.agents.ml_foundation.model_trainer.agent import (
        ModelTrainerAgent,
        TrainingRunPersistenceError,
    )

    agent = ModelTrainerAgent()
    run_mock = AsyncMock(side_effect=TrainingRunPersistenceError("persist failed"))

    class _Run:
        run_id = "parent"

        async def log_params(self, *_a, **_k):
            return None

        async def log_metrics(self, *_a, **_k):
            return None

        async def set_tags(self, *_a, **_k):
            return None

    class _Conn:
        async def get_or_create_experiment(self, **_k):
            return "exp-parent"

        @contextlib.asynccontextmanager
        async def start_run(self, **_k):
            yield _Run()

    X = pd.DataFrame({"a": np.arange(200, dtype=float), "b": np.arange(200, dtype=float) % 7})
    y = pd.Series(np.arange(200) % 2)
    with (
        patch.object(agent, "run", run_mock),
        patch.object(agent, "_get_mlflow_connector_or_none", lambda: _Conn()),
    ):
        with pytest.raises(TrainingRunPersistenceError):
            await agent._run_repeated_splits(
                {
                    "model_candidate": {
                        "algorithm_name": "LogisticRegression",
                        "algorithm_class": "sklearn.linear_model.LogisticRegression",
                        "hyperparameter_search_space": {},
                        "default_hyperparameters": {},
                    },
                    "qc_report": {"qc_passed": True},
                    "experiment_id": "exp_repeated",
                    "evaluation_mode": "repeated_k10",
                    "full_data": {"X": X, "y": y},
                }
            )
    assert run_mock.await_count == 1  # not re-run outside the wrapper


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_legacy_finished_run_is_still_a_best_run_candidate():
    """codex r2 MED: 'finished' is documented as a successful legacy status."""
    exp = uuid4()
    db = FakeAsyncSupabase({"ml_training_runs": []})
    repo, run = await _new_run(db, exp)
    await repo.update_run_metrics(run_id=run.id, test_metrics={"roc_auc": 0.8})
    await repo.complete_run(run.id, status="finished")
    best = await repo.get_best_run(exp)
    assert best is not None and str(best.id) == str(run.id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_mlflow_logger_never_claims_a_database_row_it_did_not_write(caplog):
    """codex r2 HIGH: its repository has no client, so the 'persisted' id was an
    in-memory uuid4 — a db_training_run_id no table holds."""
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import (
        _persist_training_run,
    )

    with caplog.at_level(logging.DEBUG):
        out = await _persist_training_run(
            experiment_id=str(uuid4()),
            run_name="r",
            mlflow_run_id="m",
            algorithm_name="LogisticRegression",
            hyperparameters={},
            training_samples=1,
            validation_samples=1,
            test_samples=1,
            feature_names=[],
            hpo_study_name=None,
            hpo_best_trial=None,
        )
    assert out is None
    assert not any("Persisted training run to database" in r.getMessage() for r in caplog.records)
