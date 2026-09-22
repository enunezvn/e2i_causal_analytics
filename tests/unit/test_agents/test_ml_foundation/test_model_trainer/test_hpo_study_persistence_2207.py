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

import copy
import json
import math
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
        self._filters: List[tuple] = []

    def insert(self, data):
        self._op, self._payload = "insert", data
        return self

    def upsert(self, data, on_conflict=None, **_k):
        self._op, self._payload, self._on_conflict = "upsert", data, on_conflict
        return self

    def select(self, *_a, **_k):
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def gt(self, col, val):
        self._filters.append(("gt", col, val))
        return self

    def limit(self, *_a):
        return self

    def _match(self, row):
        for kind, col, val in self._filters:
            if kind == "eq" and str(row.get(col)) != str(val):
                return False
            if kind == "gt" and not (row.get(col) is not None and row[col] > val):
                return False
        return True

    async def execute(self):
        rows = self._store.setdefault(self._table, [])
        if self._op == "delete":
            gone = [r for r in rows if self._match(r)]
            rows[:] = [r for r in rows if not self._match(r)]
            return SimpleNamespace(data=gone)
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


class _AsyncRpc:
    """In-memory stand-in for the persist_hpo_study SQL function (migration ml/045):
    upsert the study on study_name and REPLACE its trial set — atomically, i.e. on
    any error nothing changes. The real atomicity is Postgres's; this fake keeps the
    same all-or-nothing contract so a partial-failure test means what it says."""

    def __init__(self, db, name, params):
        self._db, self._name, self._params = db, name, params

    async def execute(self):
        assert self._name == "persist_hpo_study", self._name
        # PostgREST serialises the params with the stdlib encoder (codex r7): a
        # Pydantic model, a numpy scalar or a non-finite float in the payload fails
        # the call before the SQL function ever runs. Enforce the same boundary.
        encoded = json.dumps(self._params, allow_nan=False)
        self._params = json.loads(encoded)
        study, trials = self._params["p_study"], self._params["p_trials"]
        # Postgres rejects a non-finite numeric the way json does: the whole call fails.
        for v in [study.get("best_value"), *(t.get("value") for t in trials)]:
            if v is not None and not math.isfinite(v):
                raise RuntimeError(f"invalid input syntax for type numeric: {v}")
        # Stage every mutation on a COPY (the transaction's private view) ...
        staged = copy.deepcopy(self._db.store)
        studies = staged.setdefault("ml_hpo_studies", [])
        existing = next((r for r in studies if r["study_name"] == study["study_name"]), None)
        if existing is None:
            existing = {"id": str(uuid.uuid4())}
            studies.append(existing)
        existing.update(study)
        sid = existing["id"]
        rows = staged.setdefault("ml_hpo_trials", [])
        rows[:] = [r for r in rows if r["study_id"] != sid]
        rows.extend({"id": str(uuid.uuid4()), "study_id": sid, **t} for t in trials)
        if len({t["trial_number"] for t in trials}) != len(trials):
            raise RuntimeError(
                'duplicate key value violates unique constraint "unique_trial_in_study"'
            )
        # ... and a failure injected AFTER the parent + trial mutations, before commit,
        # discards the staged copy: the committed store is untouched.
        if self._db.fail_before_commit is not None:
            raise self._db.fail_before_commit
        self._db.store.clear()
        self._db.store.update(staged)
        self._db.rpc_calls.append((self._name, self._params))
        return SimpleNamespace(data=sid)


class FakeAsyncSupabase:
    def __init__(self):
        self.store: Dict[str, List[Dict[str, Any]]] = {}
        self.rpc_calls: List[tuple] = []
        self.fail_before_commit: Any = None

    def table(self, name):
        return _AsyncQuery(self.store, name)

    def rpc(self, name, params):
        return _AsyncRpc(self, name, params)


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


@pytest.mark.asyncio
async def test_a_shorter_rerun_removes_the_previous_runs_trailing_trials():
    """Codex r4: in-memory Optuna reruns reuse the study name; the parent row is
    replaced but trial rows beyond the new run's count used to linger, so n_trials
    disagreed with the child rows."""
    db = FakeAsyncSupabase()
    name = f"e2i_rerun_{uuid.uuid4().hex[:6]}_rf_hpo"
    longer = optuna.create_study(study_name=name)
    longer.optimize(lambda t: t.suggest_int("n_estimators", 10, 20) / 20.0, n_trials=4)
    shorter = optuna.create_study(study_name=name)
    shorter.optimize(lambda t: t.suggest_int("n_estimators", 10, 20) / 20.0, n_trials=2)
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
        first = await opt.save_to_database(longer, _results(longer))
        second = await opt.save_to_database(shorter, _results(shorter))
    assert first["study_id"] == second["study_id"]
    (row,) = db.store["ml_hpo_studies"]
    assert row["n_trials"] == 2
    trials = [t for t in db.store["ml_hpo_trials"] if t["study_id"] == row["id"]]
    assert sorted(t["trial_number"] for t in trials) == [0, 1]
    assert second["trials_saved"] == 2


@pytest.mark.asyncio
async def test_a_failure_mid_way_leaves_the_previous_run_intact_and_reports_failure():
    """Codex r5/r6: the study + trial set must be one transaction — a failure AFTER
    the parent and the new trials have been written (before commit) leaves the
    committed parent AND children exactly as they were. The fake stages every
    mutation on a copy and injects the failure after it, like Postgres does."""
    db = FakeAsyncSupabase()
    name = f"e2i_atomic_{uuid.uuid4().hex[:6]}_rf_hpo"
    first = optuna.create_study(study_name=name)
    first.optimize(lambda t: t.suggest_int("n_estimators", 10, 20) / 20.0, n_trials=3)
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
        ok = await opt.save_to_database(first, _results(first))
        before = {k: [dict(r) for r in v] for k, v in db.store.items()}
        second = optuna.create_study(study_name=name)
        second.optimize(lambda t: t.suggest_int("n_estimators", 10, 20) / 20.0, n_trials=1)
        db.fail_before_commit = RuntimeError("statement timeout")
        failed = await opt.save_to_database(second, _results(second))

    assert ok["success"] is True and failed["success"] is False
    assert db.store == before  # parent unchanged, trial set unchanged: no mixed state
    assert len(db.store["ml_hpo_trials"]) == 3
    # and every write went through the single RPC — no table-level inserts/upserts
    assert len(db.rpc_calls) == 1
    assert "ml_hpo_studies" in db.store and all(
        r["study_name"] == name for r in db.store["ml_hpo_studies"]
    )


@pytest.mark.asyncio
async def test_a_failed_trial_with_a_non_finite_value_does_not_lose_the_study():
    """Codex r6: the live objective returns -inf on a caught trial failure
    (OptunaOptimizer.create_objective) — one such trial made the payload invalid
    (json / numeric(10,6)) and lost the whole study. Non-finite values go to NULL."""
    db = FakeAsyncSupabase()
    study = optuna.create_study(study_name=f"e2i_inf_{uuid.uuid4().hex[:6]}_rf_hpo")

    def objective(t):
        n = t.suggest_int("n_estimators", 10, 20)
        t.report(float("nan"), step=0)
        return float("-inf") if t.number == 1 else n / 20.0

    study.optimize(objective, n_trials=3)
    assert any(t.value == float("-inf") for t in study.trials)
    results = _results(study)
    results["best_value"] = float("-inf")  # the worst case: the run itself was degenerate
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
        out = await opt.save_to_database(study, results)

    assert out["success"] is True, out
    (_, params) = db.rpc_calls[0]
    assert params["p_study"]["best_value"] is None
    values = {t["trial_number"]: t["value"] for t in params["p_trials"]}
    assert values[1] is None
    assert all(v is not None and math.isfinite(v) for k, v in values.items() if k != 1)
    assert all(
        v is None or math.isfinite(v)
        for t in params["p_trials"]
        for v in t["intermediate_values"].values()
    )
    assert len(db.store["ml_hpo_trials"]) == 3


def _typed_search_space():
    """The search space the way the LIVE graph holds it: StateGraph(ModelTrainerState)
    validates the dict literals into Pydantic Optuna*Distribution objects (state.py)."""
    from uuid import uuid4

    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(
        audit_workflow_id=uuid4(),
        hyperparameter_search_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 0.3, "log": True},
            "objective": {"type": "categorical", "choices": ["binary:logistic", "binary:hinge"]},
        },
    )
    space = state.hyperparameter_search_space
    assert space is not None and not isinstance(space["n_estimators"], dict)  # Pydantic, not dict
    return space


@pytest.mark.asyncio
async def test_the_payload_built_from_the_live_typed_search_space_is_json_native():
    """Codex r7 (HIGH): the live graph hands the saver Pydantic distribution objects
    and numpy scalars can ride in params/attrs; PostgREST's encoder rejects both."""
    import numpy as np

    db = FakeAsyncSupabase()
    study = optuna.create_study(study_name=f"e2i_typed_{uuid.uuid4().hex[:6]}_rf_hpo")

    def objective(t):
        # numpy scalars ride into params/attrs on the real path (sklearn metrics)
        t.set_user_attr("np_flag", np.bool_(True))
        t.set_user_attr("np_score", np.float64(0.5))
        return t.suggest_int("n_estimators", 10, 20) / 20.0

    study.optimize(objective, n_trials=2)
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
        out = await opt.save_to_database(study, _results(study), search_space=_typed_search_space())
    assert out["success"] is True, out
    (_, params) = db.rpc_calls[0]
    json.dumps(params, allow_nan=False)  # the exact payload survives the stdlib encoder
    space = params["p_study"]["search_space"]
    # model_dump(mode="json") of the typed distribution: the dict literal's keys plus
    # the variant's defaulted ones (log=None on int) — plain JSON either way.
    assert space["n_estimators"] == {"type": "int", "low": 50, "high": 500, "step": 50, "log": None}
    assert space["learning_rate"]["log"] is True and space["learning_rate"]["low"] == 1e-4
    assert space["objective"]["choices"] == ["binary:logistic", "binary:hinge"]
    t0 = next(t for t in params["p_trials"] if t["trial_number"] == 0)
    assert t0["user_attrs"] == {"np_flag": True, "np_score": 0.5}


def test_build_persist_payload_is_the_single_serialiser_and_never_carries_pydantic():
    study = _study_with_trials(1)
    opt = OptunaOptimizer(experiment_id="unknown", mlflow_tracking=False)
    study_record, trial_records = opt.build_persist_payload(
        study,
        _results(study),
        algorithm_name="RandomForest",
        problem_type="binary_classification",
        metric="roc_auc",
        search_space=_typed_search_space(),
        experiment_uuid=None,
    )
    json.dumps({"p_study": study_record, "p_trials": trial_records}, allow_nan=False)
    assert study_record["search_space"]["objective"]["type"] == "categorical"


@pytest.mark.asyncio
async def test_a_large_regression_objective_persists_as_its_value():
    """Codex r7 (MEDIUM): numeric(10,6) capped objectives at 9999.999999 — a 1e6 rmse
    aborted the atomic RPC. ml/045 widens best_value / trial value to double precision,
    and the payload keeps the real number."""
    db = FakeAsyncSupabase()
    study = optuna.create_study(study_name=f"e2i_big_{uuid.uuid4().hex[:6]}_lr_hpo")
    study.optimize(lambda t: 1e6 + t.suggest_int("k", 1, 3), n_trials=2)
    results = _results(study)
    assert results["best_value"] >= 1e6
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
        out = await opt.save_to_database(study, results)
    assert out["success"] is True
    (_, params) = db.rpc_calls[0]
    assert params["p_study"]["best_value"] == results["best_value"]
    assert all(t["value"] >= 1e6 for t in params["p_trials"])
