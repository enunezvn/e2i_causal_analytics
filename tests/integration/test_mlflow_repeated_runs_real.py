"""W3-lite Day-5+1 — cycle-16 deferred I-5/I-6/I-7 real-MLflow integration smoke.

Spec: shard 21 §C (parent + nested children topology) + §H item 9 (real-MLflow
acceptance smoke). Cycle-16 codex deferred items:

- I-5 (Q3-A): mid-fold exception → parent stays open + failed child FAILED +
  subsequent fold child correctly attached to parent.
- I-7 (Q3-D): real-MLflow end-to-end smoke → 1 parent + k children visible
  via ``MlflowClient.search_runs``.

The earlier ``test_mlflow_repeated_runs.py`` exercises orchestrator-level
contracts using an in-memory ``_FakeConnector``. THIS file uses the REAL
``MLflowConnector`` against a temporary ``file://`` tracking store so that
MLflow's actual thread-local active-run state, run-status transitions, and
``mlflow.search_runs`` parent-attribution are exercised end-to-end.

Both tests are ``@pytest.mark.slow`` because they create temp dirs, spawn
real MLflow tracking-store I/O, and run the LangGraph fold loop. Default
fixture k=4 to keep wall-clock <60s (override per-test where the spec
demands k=10).
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Any, Dict, Iterator
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

SEED = 42
N = 200
N_FEATURES = 4


def _make_full_data(prevalence: float = 0.30) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(
        rng.standard_normal((N, N_FEATURES)),
        columns=[f"x{i}" for i in range(N_FEATURES)],
    )
    n_positive = int(round(N * prevalence))
    y_arr = np.zeros(N, dtype=int)
    positive_idx = rng.choice(N, size=n_positive, replace=False)
    y_arr[positive_idx] = 1
    return {"X": X, "y": pd.Series(y_arr, name="y")}


@pytest.fixture
def real_mlflow_tempdir(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Provide a temp file:// MLflow tracking URI + reset connector singleton.

    The MLflowConnector is a singleton; reset its class-level ``_instance``
    so the env-var override is picked up by a fresh ``__init__`` call. Also
    explicitly resets the MLflow global tracking URI + ends any leftover
    active run so prior tests' state doesn't leak into the new tempdir.
    """
    import mlflow

    from src.mlops.mlflow_connector import MLflowConnector

    tempdir = tempfile.mkdtemp(prefix="mlflow_repeated_smoke_")
    tracking_uri = f"file://{tempdir}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    # Reset singleton + clear MLflow's process-global active-run + tracking
    # URI so prior tests don't leak state into this fixture.
    MLflowConnector._instance = None  # type: ignore[assignment]
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(tracking_uri)

    yield tracking_uri

    while mlflow.active_run() is not None:
        mlflow.end_run()
    MLflowConnector._instance = None  # type: ignore[assignment]
    shutil.rmtree(tempdir, ignore_errors=True)


REAL_SMOKE_EXPERIMENT_NAME = "test_real_mlflow_smoke_unique"


def _build_minimal_graph_recorder(force_fold_failure_idx: int | None = None):
    """Build a graph-mock that produces the minimum state log_to_mlflow needs.

    The mock returns a state dict with placeholder ``trained_model`` (a
    minimally-fitted sklearn DummyClassifier) + ``framework`` + ``algorithm_name``
    + minimal metrics so ``log_to_mlflow`` can run end-to-end without a
    real training pipeline. The graph mock then INVOKES the real
    ``log_to_mlflow`` node with the populated state — that's what produces
    the actual nested run against real MLflow.

    When ``force_fold_failure_idx`` is set, the matching fold's call to
    ``log_to_mlflow`` will raise ``RuntimeError`` AFTER the nested run is
    opened (simulating mid-fold failure for I-5 verification).
    """
    from sklearn.dummy import DummyClassifier

    async def fake_ainvoke(state: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(state.get("fold_idx", 0))
        is_repeated_fold = state.get("evaluation_mode") == "repeated_k10" and bool(
            state.get("_repeated_mode_fold_invocation", False)
        )

        # Minimal fitted model (DummyClassifier needs only sample arrays)
        model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
        enriched_state: Dict[str, Any] = {
            **state,
            "trained_model": model,
            "framework": "sklearn",
            "algorithm_name": "Dummy",
            # State coming from the recursive `agent.run` call does NOT carry
            # `experiment_name` (the agent's `initial_state` only forwards
            # `experiment_id`); force the unique smoke name so all per-fold
            # mlflow_logger invocations land on the same experiment we'll
            # search after the orchestrator returns.
            "experiment_name": REAL_SMOKE_EXPERIMENT_NAME,
            "evaluation_metrics": {
                "train_metrics": {"auroc": 0.70 + 0.005 * idx},
                "validation_metrics": {"auroc": 0.68 + 0.005 * idx},
                "test_metrics": {"auroc": 0.65 + 0.005 * idx},
            },
            "best_hyperparameters": {},
            "training_samples": 100,
            "feature_names": ["x0", "x1"],
            "test_metrics": {"auroc": 0.65 + 0.005 * idx},
            "auc_roc": 0.65 + 0.005 * idx,
            "brier_score": 0.20 - 0.001 * idx,
        }

        if is_repeated_fold:
            from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import (
                log_to_mlflow,
            )

            # Patch _log_model_artifact to a no-op so we don't pull MLflow
            # model-saving infrastructure during the smoke.
            async def _noop_artifact(*_args: Any, **_kwargs: Any) -> str:
                return "runs:/dummy_model_uri/model"

            with patch(
                "src.agents.ml_foundation.model_trainer.nodes.mlflow_logger._log_model_artifact",
                side_effect=_noop_artifact,
            ):
                if force_fold_failure_idx is not None and idx == force_fold_failure_idx:
                    # Simulate mid-fold failure AFTER mlflow_logger opened the
                    # nested run: patch the per-split metric logger to raise.
                    async def _raise_mid_run(*_args: Any, **_kwargs: Any) -> None:
                        raise RuntimeError(f"injected mid-fold failure at fold {idx}")

                    with patch(
                        "src.agents.ml_foundation.model_trainer.nodes.mlflow_logger._log_split_metrics",
                        side_effect=_raise_mid_run,
                    ):
                        await log_to_mlflow(enriched_state)
                else:
                    await log_to_mlflow(enriched_state)

        return enriched_state

    return AsyncMock(side_effect=fake_ainvoke)


def _make_input(k: int = 4) -> Dict[str, Any]:
    return {
        "model_candidate": {
            "algorithm_name": "Dummy",
            "algorithm_class": "sklearn.dummy.DummyClassifier",
            "hyperparameter_search_space": {},
            "default_hyperparameters": {},
        },
        "qc_report": {"qc_passed": True},
        "experiment_id": "test_real_mlflow_smoke",
        # Match the recorder's hardcoded experiment_name so parent + children
        # land in the SAME experiment for cross-attribution searches.
        "experiment_name": REAL_SMOKE_EXPERIMENT_NAME,
        "success_criteria": {},
        "enable_hpo": False,
        "enable_mlflow": True,
        "enable_checkpointing": False,
        "evaluation_mode": "repeated_k10",
        "repeated_splits_config": {"k": k},
        "full_data": _make_full_data(),
    }


# ---------------------------------------------------------------------------
# I-7 — Real-MLflow end-to-end smoke for shard 21 §H item 9 acceptance
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_real_mlflow_parent_plus_k_children_visible_via_search_runs(
    real_mlflow_tempdir: str,
) -> None:
    """Cycle-16 I-7 (Q3-D) + shard 21 §H item 9: 1 parent + k children visible.

    Drives the orchestrator end-to-end against a real ``file://`` MLflow
    tracking store and verifies the resulting topology via the public
    ``MlflowClient.search_runs`` API. Uses k=10 per the verdict text.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    mock_graph = _build_minimal_graph_recorder()
    with patch.object(agent.graph, "ainvoke", mock_graph):
        output = await agent.run(_make_input(k=10))

    # Sanity: orchestrator returned a parent_mlflow_run_id + 10 fold_metrics
    assert output["evaluation_mode"] == "repeated_k10"
    assert output["aggregate_status"] == "COMPLETE"
    assert len(output["fold_metrics"]) == 10
    parent_run_id = output.get("parent_mlflow_run_id")
    assert parent_run_id is not None, "Orchestrator did not return parent_mlflow_run_id"

    # Real MLflow query: parent + 10 nested children visible
    mlflow.set_tracking_uri(real_mlflow_tempdir)
    client = MlflowClient(tracking_uri=real_mlflow_tempdir)
    experiments = client.search_experiments()
    exp = next((e for e in experiments if REAL_SMOKE_EXPERIMENT_NAME in e.name), None)
    assert exp is not None, (
        f"Experiment not created in tracking store: {[e.name for e in experiments]}"
    )

    # Children are runs whose mlflow.parentRunId tag equals the parent run id
    children = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    assert len(children) == 10, (
        f"Expected 10 nested child runs attached to parent {parent_run_id}; "
        f"got {len(children)}: {[r.info.run_name for r in children]}"
    )

    # Parent run itself should be findable + FINISHED
    parent_run = client.get_run(parent_run_id)
    assert parent_run is not None
    assert parent_run.info.status == "FINISHED"

    # All 10 children should be FINISHED (no orphans)
    for child in children:
        assert child.info.status == "FINISHED", (
            f"Child {child.info.run_id} status={child.info.status}, expected FINISHED"
        )


# ---------------------------------------------------------------------------
# I-5 — Real-MLflow mid-fold-exception orphan + recovery semantics
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_real_mlflow_mid_fold_exception_failed_child_subsequent_attached(
    real_mlflow_tempdir: str,
) -> None:
    """Cycle-16 I-5 (Q3-A): mid-fold exception → failed child FAILED, parent open, next fold attached.

    Simulates fold 1's mlflow_logger raising AFTER the nested child run was
    opened (via patched ``_log_split_metrics``). The exception propagates out
    of ``start_run`` so the connector marks the child FAILED, but
    ``log_to_mlflow``'s outer try/except swallows the exception and returns
    ``mlflow_status="failed"`` — orchestrator does NOT see the failure.

    What this verifies (the MLflow lifecycle, NOT the orchestrator status):
      * Parent run remains open through the failure (subsequent folds still
        produce children attached to this parent — no LIFO state corruption).
      * Failed fold's child run carries status FAILED in the tracking store.
      * Folds 2-3 produce children with status FINISHED, all attached to the
        SAME parent run id.

    Uses k=4 (3 OK + 1 FAIL) to keep wall-clock <60s while still exercising
    the "next fold after failure" path.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    # Inject failure at fold 1; folds 0, 2, 3 succeed
    mock_graph = _build_minimal_graph_recorder(force_fold_failure_idx=1)
    with patch.object(agent.graph, "ainvoke", mock_graph):
        output = await agent.run(_make_input(k=4))

    # Orchestrator-level: fold completed because log_to_mlflow swallowed the
    # exception. We only verify the MLflow tracking-store lifecycle here.
    assert output["evaluation_mode"] == "repeated_k10"
    assert len(output["fold_metrics"]) == 4

    parent_run_id = output.get("parent_mlflow_run_id")
    assert parent_run_id is not None

    # Real MLflow query
    mlflow.set_tracking_uri(real_mlflow_tempdir)
    client = MlflowClient(tracking_uri=real_mlflow_tempdir)
    experiments = client.search_experiments()
    exp = next((e for e in experiments if REAL_SMOKE_EXPERIMENT_NAME in e.name), None)
    assert exp is not None

    children = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    # All 4 fold-children should be present (open-then-fail still leaves a run)
    assert len(children) == 4, (
        f"Expected 4 nested children (3 OK + 1 FAIL); got {len(children)}: "
        f"{[(r.info.run_name, r.info.status) for r in children]}"
    )

    statuses = {r.info.run_name: r.info.status for r in children}
    # Fold 1 failed mid-run — should be FAILED
    assert statuses.get("fold_01") == "FAILED", (
        f"Failed fold's child run not in FAILED status: {statuses}"
    )
    # Folds 0, 2, 3 succeeded — should be FINISHED
    for ok_idx in (0, 2, 3):
        assert statuses.get(f"fold_{ok_idx:02d}") == "FINISHED", (
            f"OK fold_{ok_idx:02d} not FINISHED: {statuses}"
        )

    # Parent stays FINISHED — failure didn't poison the wrapping context
    parent_run = client.get_run(parent_run_id)
    assert parent_run.info.status == "FINISHED"


# ---------------------------------------------------------------------------
# I-6 — Real-MLflow concurrent n_jobs=2 lock verification (smoke)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_real_mlflow_n_jobs_2_lock_preserves_topology(
    real_mlflow_tempdir: str,
) -> None:
    """Cycle-16 I-6 (Q3-C) verification: n_jobs=2 + lock preserves parent ↔ child.

    Even with the mlflow_logger lock in place, run k=4 folds with
    ``n_jobs=2`` against real MLflow and assert all 4 children are correctly
    attached to the single parent run (no cross-attachment from concurrent
    nested-run opens).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    mock_graph = _build_minimal_graph_recorder()
    input_data = _make_input(k=4)
    input_data["repeated_splits_config"]["n_jobs"] = 2  # exercise the lock

    with patch.object(agent.graph, "ainvoke", mock_graph):
        output = await agent.run(input_data)

    parent_run_id = output.get("parent_mlflow_run_id")
    assert parent_run_id is not None

    mlflow.set_tracking_uri(real_mlflow_tempdir)
    client = MlflowClient(tracking_uri=real_mlflow_tempdir)
    experiments = client.search_experiments()
    exp = next((e for e in experiments if REAL_SMOKE_EXPERIMENT_NAME in e.name), None)
    assert exp is not None

    children = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    assert len(children) == 4, (
        f"n_jobs=2 produced wrong topology — expected 4 children attached to "
        f"parent {parent_run_id}, got {len(children)}: "
        f"{[(r.info.run_name, r.data.tags.get('mlflow.parentRunId')) for r in children]}"
    )
    for child in children:
        # Cross-check tag value matches the parent run id (not some sibling)
        assert child.data.tags.get("mlflow.parentRunId") == parent_run_id, (
            f"Child {child.info.run_name} has wrong parentRunId tag: "
            f"{child.data.tags.get('mlflow.parentRunId')} vs expected {parent_run_id}"
        )
