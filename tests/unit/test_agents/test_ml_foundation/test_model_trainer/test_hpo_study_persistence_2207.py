"""#2207: the live HPO path persists its Optuna study + trials.

Census finding (2026-09-22): ``ml_hpo_studies`` / ``ml_hpo_trials`` (migration ml/016,
356b57963) had a writer — ``OptunaOptimizer.save_to_database`` — with zero call sites,
while the tuner that actually runs (``hyperparameter_tuner.tune_hyperparameters``, the
model_trainer node inside MLFoundationPipeline) wrote 943 ``ml_hpo_patterns`` rows.
Owner decision (a): wire the study/trial persistence into THAT path.

Two things the writer could not do as written, measured against the live schema:

* ``ml_hpo_studies.experiment_id`` is ``UUID REFERENCES ml_experiments(id)``, but the
  tuner's ``experiment_id`` is the pipeline's label (``tier0_e2e_dd343e1e``,
  ``exp_kisq_us_...``) — the value ``ml_experiments.mlflow_experiment_id`` holds, not the
  row's UUID. Inserting the label raises ``invalid input syntax for type uuid``. The
  saver now resolves the label through ``MLExperimentRepository.get_by_mlflow_id`` (the
  same lookup ``model_trainer/agent.py`` uses for ``ml_training_runs``) and writes NULL
  when there is no row — never a fabricated id.
* ``study_name`` is UNIQUE, so a re-run of the same experiment label must upsert, not
  fail on the second insert.

And the contract every writer in this lane shares: persistence never fails the parent.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import numpy as np
import optuna
import pytest

from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
    tune_hyperparameters,
)
from src.mlops.optuna_optimizer import OptunaOptimizer

# ---------------------------------------------------------------------------
# async fake of the supabase client ``save_to_database`` drives
# ---------------------------------------------------------------------------


class _AsyncQuery:
    def __init__(self, store: Dict[str, List[Dict[str, Any]]], table: str):
        self._store, self._table = store, table
        self._op, self._payload, self._on_conflict = "select", None, None

    def insert(self, data):
        self._op, self._payload = "insert", data
        return self

    def upsert(self, data, on_conflict=None, **_k):
        self._op, self._payload, self._on_conflict = "upsert", data, on_conflict
        return self

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a):
        return self

    def limit(self, *_a):
        return self

    async def execute(self):
        rows = self._store.setdefault(self._table, [])
        payload = self._payload if isinstance(self._payload, list) else [self._payload]
        out = []
        for p in payload:
            row = dict(p)
            row.setdefault("id", str(uuid.uuid4()))
            if self._op == "upsert" and self._on_conflict:
                keys = [k.strip() for k in self._on_conflict.split(",")]
                existing = next(
                    (r for r in rows if all(r.get(k) == row.get(k) for k in keys)), None
                )
                if existing is not None:
                    row["id"] = existing["id"]
                    existing.update(row)
                    out.append(dict(existing))
                    continue
            rows.append(row)
            out.append(dict(row))
        return SimpleNamespace(data=out)


class FakeAsyncSupabase:
    def __init__(self):
        self.store: Dict[str, List[Dict[str, Any]]] = {}

    def table(self, name):
        return _AsyncQuery(self.store, name)


def _study_with_trials(n: int = 3) -> optuna.Study:
    study = optuna.create_study(study_name=f"e2i_lbl_{uuid.uuid4().hex[:6]}_rf_hpo")
    study.optimize(lambda t: t.suggest_int("n_estimators", 10, 20) / 20.0, n_trials=n)
    return study


def _results(study: optuna.Study) -> Dict[str, Any]:
    return {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "n_completed": len(study.trials),
        "n_pruned": 0,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "duration_seconds": 1.25,
    }


def _tuner_state(experiment_id: str = "tier0_e2e_2207test") -> Dict[str, Any]:
    rng = np.random.default_rng(2207)
    return {
        "enable_hpo": True,
        "hpo_trials": 2,
        "algorithm_name": "RandomForest",
        "problem_type": "binary_classification",
        "experiment_id": experiment_id,
        "default_hyperparameters": {"n_estimators": 10},
        "hyperparameter_search_space": {"n_estimators": {"type": "int", "low": 5, "high": 12}},
        "X_train_preprocessed": rng.random((60, 4)),
        "X_validation_preprocessed": rng.random((30, 4)),
        "train_data": {"y": rng.integers(0, 2, 60)},
        "validation_data": {"y": rng.integers(0, 2, 30)},
    }


# ---------------------------------------------------------------------------
# call-site wiring: the tuner that actually runs persists its study
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tune_hyperparameters_persists_the_study_it_ran():
    with (
        patch.object(
            OptunaOptimizer,
            "save_to_database",
            new=AsyncMock(return_value={"success": True, "study_id": "s-1", "trials_saved": 2}),
        ) as save,
        patch(
            "src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner."
            "_get_hpo_pattern_memory",
            return_value=None,
        ),
    ):
        out = await tune_hyperparameters(_tuner_state())

    assert out["hpo_completed"] is True
    save.assert_awaited_once()
    kwargs = save.await_args.kwargs
    assert isinstance(kwargs["study"], optuna.Study)
    assert kwargs["optimization_results"]["study_name"] == kwargs["study"].study_name
    assert kwargs["algorithm_name"] == "RandomForest"
    assert kwargs["problem_type"] == "binary_classification"
    assert kwargs["metric"]  # the objective the study optimised, not a default
    assert kwargs["search_space"] == {"n_estimators": {"type": "int", "low": 5, "high": 12}}
    assert out["hpo_study_id"] == "s-1"


@pytest.mark.asyncio
async def test_persistence_failure_never_fails_hpo():
    with (
        patch.object(
            OptunaOptimizer, "save_to_database", new=AsyncMock(side_effect=RuntimeError("db down"))
        ),
        patch(
            "src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner."
            "_get_hpo_pattern_memory",
            return_value=None,
        ),
    ):
        out = await tune_hyperparameters(_tuner_state())
    assert out["hpo_completed"] is True
    assert "best_hyperparameters" in out
    assert "hpo_study_id" not in out


# ---------------------------------------------------------------------------
# the saver against the live schema's constraints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_experiment_label_is_resolved_to_the_ml_experiments_uuid():
    db = FakeAsyncSupabase()
    exp_uuid = str(uuid.uuid4())
    study = _study_with_trials(3)
    opt = OptunaOptimizer(experiment_id="tier0_e2e_dd343e1e", mlflow_tracking=False)
    with (
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=db),
        ),
        patch(
            "src.repositories.ml_experiment.MLExperimentRepository.get_by_mlflow_id",
            new=AsyncMock(return_value=SimpleNamespace(id=exp_uuid)),
        ) as lookup,
    ):
        out = await opt.save_to_database(
            study, _results(study), algorithm_name="RandomForest", metric="roc_auc"
        )

    assert out["success"] is True
    lookup.assert_awaited_once()
    assert lookup.await_args.args[0] == "tier0_e2e_dd343e1e"
    (row,) = db.store["ml_hpo_studies"]
    assert row["experiment_id"] == exp_uuid
    assert row["study_name"] == study.study_name
    assert row["status"] == "completed"
    trials = db.store["ml_hpo_trials"]
    assert len(trials) == 3 == out["trials_saved"]
    assert {t["study_id"] for t in trials} == {row["id"]}
    assert sorted(t["trial_number"] for t in trials) == [0, 1, 2]


@pytest.mark.asyncio
async def test_unknown_experiment_label_writes_null_not_a_fabricated_uuid():
    db = FakeAsyncSupabase()
    study = _study_with_trials(1)
    opt = OptunaOptimizer(experiment_id="unknown", mlflow_tracking=False)
    with (
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=db),
        ),
        patch(
            "src.repositories.ml_experiment.MLExperimentRepository.get_by_mlflow_id",
            new=AsyncMock(return_value=None),
        ),
    ):
        out = await opt.save_to_database(study, _results(study))
    assert out["success"] is True
    (row,) = db.store["ml_hpo_studies"]
    assert row.get("experiment_id") is None


@pytest.mark.asyncio
async def test_a_real_uuid_experiment_id_is_passed_through_without_lookup():
    db = FakeAsyncSupabase()
    study = _study_with_trials(1)
    exp_uuid = str(uuid.uuid4())
    opt = OptunaOptimizer(experiment_id=exp_uuid, mlflow_tracking=False)
    with (
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=db),
        ),
        patch(
            "src.repositories.ml_experiment.MLExperimentRepository.get_by_mlflow_id",
            new=AsyncMock(side_effect=AssertionError("must not look up a uuid")),
        ),
    ):
        await opt.save_to_database(study, _results(study))
    assert db.store["ml_hpo_studies"][0]["experiment_id"] == exp_uuid


@pytest.mark.asyncio
async def test_rerunning_the_same_study_name_upserts_instead_of_failing_unique():
    db = FakeAsyncSupabase()
    study = _study_with_trials(2)
    opt = OptunaOptimizer(experiment_id="unknown", mlflow_tracking=False)
    with (
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=db),
        ),
        patch(
            "src.repositories.ml_experiment.MLExperimentRepository.get_by_mlflow_id",
            new=AsyncMock(return_value=None),
        ),
    ):
        first = await opt.save_to_database(study, _results(study))
        second = await opt.save_to_database(study, _results(study))
    assert first["success"] and second["success"]
    assert len(db.store["ml_hpo_studies"]) == 1
    assert first["study_id"] == second["study_id"]
    assert len(db.store["ml_hpo_trials"]) == 2  # (study_id, trial_number) unique
