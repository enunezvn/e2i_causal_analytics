"""W3-lite Day-5 G.4 — per-fold MLflow nested-run topology + aggregate logging.

Spec: shard 21 §C (parent + 10 children per fold, with fold tags), shard 21 §G.4.
Cycle-15 deferred items I-2 (NEP 19 version params) + I-4 (per-fold tags).

These tests stub ``MLflowConnector`` end-to-end so we can capture (a) parent run
with aggregate-level metrics, (b) ``k`` nested child runs with fold tags + per-fold
metrics, (c) ``fold_idx`` / ``fold_seed`` / ``evaluation_mode`` tags on each child.
The actual MLflow tracking server is NOT touched — we patch
``get_mlflow_connector`` (consumed by ``mlflow_logger``) and
``ModelTrainerAgent._get_mlflow_connector_or_none`` (consumed by the
orchestrator) to return a shared fake connector instance that records every
call into in-memory lists.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

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


@dataclass
class _FakeRun:
    run_id: str
    run_name: str
    tags: Dict[str, str]
    is_nested: bool
    parent_run_id: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    async def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.metrics.update(metrics)

    async def log_params(self, params: Dict[str, Any]) -> None:
        self.params.update({k: str(v) for k, v in params.items()})

    async def set_tags(self, tags: Dict[str, str]) -> None:
        self.tags.update({k: str(v) for k, v in tags.items()})


class _FakeConnector:
    """In-memory MLflow connector substitute for orchestrator + node tests.

    Maintains an LIFO stack of currently-open runs so that ``nested=True``
    correctly attaches a child to the most-recent open run as parent.
    """

    def __init__(self) -> None:
        self.runs: List[_FakeRun] = []
        self._stack: List[_FakeRun] = []
        self._counter = 0
        self._enabled = True

    async def get_or_create_experiment(
        self, name: str, tags: Optional[Dict[str, str]] = None
    ) -> str:
        return f"exp:{name}"

    @asynccontextmanager
    async def start_run(
        self,
        experiment_id: str,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        nested: bool = False,
    ):
        self._counter += 1
        parent_run_id = self._stack[-1].run_id if (nested and self._stack) else None
        run = _FakeRun(
            run_id=f"run-{self._counter:04d}",
            run_name=run_name,
            tags=dict(tags or {}),
            is_nested=nested,
            parent_run_id=parent_run_id,
        )
        self.runs.append(run)
        self._stack.append(run)
        try:
            yield run
        finally:
            self._stack.pop()


def _make_input(evaluation_mode: str = "repeated_k10", k: int = 10) -> Dict[str, Any]:
    full_data = _make_full_data()
    return {
        "model_candidate": {
            "algorithm_name": "LightGBM",
            "algorithm_class": "lightgbm.LGBMClassifier",
            "hyperparameter_search_space": {},
            "default_hyperparameters": {"n_estimators": 5, "verbose": -1},
        },
        "qc_report": {"qc_passed": True},
        "experiment_id": "test_g4_mlflow_repeated",
        "experiment_name": "test_g4_mlflow_repeated_exp",
        "success_criteria": {},
        "enable_hpo": False,
        "enable_mlflow": True,  # so mlflow_logger node also opens nested children
        "enable_checkpointing": False,
        "evaluation_mode": evaluation_mode,
        "repeated_splits_config": {"k": k},
        "full_data": full_data,
    }


def _fold_invocation_recorder_with_logging(connector: _FakeConnector):
    """Build a graph mock that ALSO opens a nested run (mirrors mlflow_logger node).

    The real ``log_to_mlflow`` node uses its own
    ``from src.mlops.mlflow_connector import get_mlflow_connector`` import. We
    replicate that path here in the mock so the per-fold child run is opened
    against the SAME ``_FakeConnector`` instance that the orchestrator uses
    for the parent run — matching the production flow where parent + nested
    children share the connector singleton.
    """
    from unittest.mock import AsyncMock

    async def fake_ainvoke(state: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(state.get("fold_idx", 0))
        is_repeated_fold = state.get("evaluation_mode") == "repeated_k10" and bool(
            state.get("repeated_mode_fold_invocation", False)
        )
        if is_repeated_fold:
            fold_seed = int(state.get("fold_random_state", 0))
            tags = {
                "fold_idx": str(idx),
                "evaluation_mode": "repeated_k10",
                "fold_seed": str(fold_seed),
                "algorithm": str(state.get("model_candidate", {}).get("algorithm_name", "unknown")),
                "framework": "lightgbm",
                "source": "model_trainer_agent",
            }
            async with connector.start_run(
                experiment_id="exp:test_g4_mlflow_repeated_exp",
                run_name=f"fold_{idx:02d}",
                tags=tags,
                nested=True,
            ) as child_run:
                await child_run.log_metrics(
                    {
                        "train_auroc": 0.80 + 0.005 * idx,
                        "validation_auroc": 0.78 + 0.005 * idx,
                        "test_auroc": 0.77 + 0.005 * idx,
                    }
                )
                await child_run.log_params(
                    {
                        "fold_idx": idx,
                        "fold_seed": fold_seed,
                    }
                )
        return {
            **state,
            "trained_model": object(),
            "train_metrics": {"auroc": 0.80 + 0.005 * idx},
            "validation_metrics": {"auroc": 0.78 + 0.005 * idx},
            "test_metrics": {"auroc": 0.77 + 0.005 * idx, "accuracy": 0.85 + 0.003 * idx},
            "auc_roc": 0.77 + 0.005 * idx,
            "brier_score": 0.18 - 0.002 * idx,
            "framework": "lightgbm",
        }

    return AsyncMock(side_effect=fake_ainvoke)


# ---------------------------------------------------------------------------
# G.4 test fixtures
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parent_run_has_aggregate_metrics() -> None:
    """Parent run must log ``aggregate_<metric>_mean / _std / _bca_lo / _bca_hi / _n_folds``."""
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    fake = _FakeConnector()
    agent = ModelTrainerAgent()
    mock_graph = _fold_invocation_recorder_with_logging(fake)
    with (
        patch.object(
            ModelTrainerAgent,
            "_get_mlflow_connector_or_none",
            staticmethod(lambda: fake),
        ),
        patch.object(agent.graph, "ainvoke", mock_graph),
    ):
        output = await agent.run(_make_input(k=10))

    parent_runs = [r for r in fake.runs if not r.is_nested]
    assert len(parent_runs) == 1, (
        f"Expected exactly 1 parent run, got {len(parent_runs)}: "
        f"{[r.run_name for r in parent_runs]}"
    )
    parent = parent_runs[0]
    assert parent.run_name == "repeated_k10_seed42"
    assert parent.tags.get("evaluation_mode") == "repeated_k10"
    assert parent.tags.get("k") == "10"
    assert parent.tags.get("seed_base") == "42"
    # aggregate_<metric>_mean / _std / _n_folds always emit; bca only when CI is finite
    assert "aggregate_auc_roc_mean" in parent.metrics
    assert "aggregate_auc_roc_std" in parent.metrics
    assert "aggregate_auc_roc_n_folds" in parent.metrics
    assert "aggregate_auc_roc_percentile_lo" in parent.metrics
    assert "aggregate_auc_roc_percentile_hi" in parent.metrics
    # BCa endpoints emit only when finite (k=10 with smooth values should produce finite)
    assert "aggregate_auc_roc_bca_lo" in parent.metrics
    assert "aggregate_auc_roc_bca_hi" in parent.metrics
    # Sanity: aggregate_status COMPLETE
    assert output["aggregate_status"] == "COMPLETE"
    # Cycle-17 COSMETIC-1/2: when all metrics are BCa-stable the summary
    # count is 0 and the tag mirrors that for non-chart consumers.
    assert parent.metrics.get("aggregate_bca_unstable_metric_count") == 0.0
    assert parent.metrics.get("aggregate_bca_unstable_metric_fraction") == 0.0
    assert parent.tags.get("has_bca_unstable") == "false"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_10_nested_child_runs_logged() -> None:
    """Parent run must have exactly 10 nested child runs at k=10 (shard 21 §C)."""
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    fake = _FakeConnector()
    agent = ModelTrainerAgent()
    mock_graph = _fold_invocation_recorder_with_logging(fake)
    with (
        patch.object(
            ModelTrainerAgent,
            "_get_mlflow_connector_or_none",
            staticmethod(lambda: fake),
        ),
        patch.object(agent.graph, "ainvoke", mock_graph),
    ):
        await agent.run(_make_input(k=10))

    parent_runs = [r for r in fake.runs if not r.is_nested]
    assert len(parent_runs) == 1
    parent_run_id = parent_runs[0].run_id
    children = [r for r in fake.runs if r.parent_run_id == parent_run_id]
    assert len(children) == 10, (
        f"Expected 10 nested children attached to parent {parent_run_id}, "
        f"got {len(children)}: {[r.run_name for r in children]}"
    )
    # Names should be fold_00..fold_09
    child_names = sorted(r.run_name for r in children)
    assert child_names == [f"fold_{i:02d}" for i in range(10)]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_child_run_logs_per_fold_metrics() -> None:
    """Each child run must carry per-fold split metrics (train_/validation_/test_)."""
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    fake = _FakeConnector()
    agent = ModelTrainerAgent()
    mock_graph = _fold_invocation_recorder_with_logging(fake)
    with (
        patch.object(
            ModelTrainerAgent,
            "_get_mlflow_connector_or_none",
            staticmethod(lambda: fake),
        ),
        patch.object(agent.graph, "ainvoke", mock_graph),
    ):
        await agent.run(_make_input(k=10))

    children = [r for r in fake.runs if r.is_nested]
    assert len(children) == 10
    for child in children:
        assert "train_auroc" in child.metrics, (
            f"{child.run_name} missing train_auroc metric: {sorted(child.metrics.keys())}"
        )
        assert "validation_auroc" in child.metrics
        assert "test_auroc" in child.metrics


@pytest.mark.integration
@pytest.mark.asyncio
async def test_child_run_logs_fold_seed_param_and_tags() -> None:
    """Each child run must have ``fold_idx`` + ``fold_seed`` params, all distinct fold_seed values
    (cycle-15 I-4 + Q1 I-2 NEP 19 verification)."""
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    fake = _FakeConnector()
    agent = ModelTrainerAgent()
    mock_graph = _fold_invocation_recorder_with_logging(fake)
    with (
        patch.object(
            ModelTrainerAgent,
            "_get_mlflow_connector_or_none",
            staticmethod(lambda: fake),
        ),
        patch.object(agent.graph, "ainvoke", mock_graph),
    ):
        await agent.run(_make_input(k=10))

    children = [r for r in fake.runs if r.is_nested]
    assert len(children) == 10
    fold_seeds: List[str] = []
    fold_indices: List[str] = []
    for child in children:
        # tag-level check (cycle-15 I-4)
        assert child.tags.get("evaluation_mode") == "repeated_k10"
        assert "fold_idx" in child.tags
        assert "fold_seed" in child.tags
        # param-level check (cycle-15 I-2 NEP 19; orchestrator emits both
        # at the parent + child level via the per-fold ainvoke mock)
        assert "fold_idx" in child.params
        assert "fold_seed" in child.params
        fold_seeds.append(child.tags["fold_seed"])
        fold_indices.append(child.tags["fold_idx"])
    # Distinct fold_seed values across folds (collisions would break selection-bias correction)
    assert len(set(fold_seeds)) == 10, f"fold_seed collisions: {fold_seeds}"
    # All fold indices 0..9 present
    assert sorted(fold_indices) == [str(i) for i in range(10)]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parent_run_has_nep19_version_params() -> None:
    """Parent run must log ``numpy_version`` + ``sklearn_version`` (cycle-15 I-2)."""
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    fake = _FakeConnector()
    agent = ModelTrainerAgent()
    mock_graph = _fold_invocation_recorder_with_logging(fake)
    with (
        patch.object(
            ModelTrainerAgent,
            "_get_mlflow_connector_or_none",
            staticmethod(lambda: fake),
        ),
        patch.object(agent.graph, "ainvoke", mock_graph),
    ):
        await agent.run(_make_input(k=10))

    parent_runs = [r for r in fake.runs if not r.is_nested]
    assert len(parent_runs) == 1
    parent = parent_runs[0]
    assert "numpy_version" in parent.params
    assert "sklearn_version" in parent.params
    # Per-fold NEP 19 derived seeds at the parent level (one per fold)
    for i in range(10):
        assert f"fold_{i:02d}_seed_base" in parent.params
        assert f"fold_{i:02d}_derived_seed" in parent.params


def _fold_invocation_recorder_constant_metric(connector: _FakeConnector):
    """Variant of the recorder that emits a CONSTANT metric across folds.

    Used to drive ``bca_unstable_warning=True`` for I-1 verification: when
    all per-fold values for a metric are equal, the jackknife denominator
    collapses and BCa is degenerate.
    """
    from unittest.mock import AsyncMock

    async def fake_ainvoke(state: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(state.get("fold_idx", 0))
        is_repeated_fold = state.get("evaluation_mode") == "repeated_k10" and bool(
            state.get("repeated_mode_fold_invocation", False)
        )
        if is_repeated_fold:
            tags = {
                "fold_idx": str(idx),
                "evaluation_mode": "repeated_k10",
                "fold_seed": str(int(state.get("fold_random_state", 0))),
            }
            async with connector.start_run(
                experiment_id="exp:test_g4_mlflow_repeated_exp",
                run_name=f"fold_{idx:02d}",
                tags=tags,
                nested=True,
            ):
                pass
        return {
            **state,
            "trained_model": object(),
            # Smooth metric — BCa stable
            "auc_roc": 0.77 + 0.005 * idx,
            # Constant metric — BCa unstable (degenerate jackknife)
            "brier_score": 0.20,
            "framework": "lightgbm",
        }

    return AsyncMock(side_effect=fake_ainvoke)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_nested_run_lock_serializes_concurrent_repeated_folds() -> None:
    """Cycle-16 I-6 (Q3-C): asyncio.Lock prevents concurrent nested-run opens.

    Under ``n_jobs > 1`` two folds running in parallel via asyncio.gather
    share the event loop's thread-local state. MLflow's
    ``mlflow.start_run(nested=True)`` consults a thread-local active-run
    stack — concurrent unprotected opens would mis-attach children.

    This test verifies the module-level lock at the mlflow_logger boundary
    serializes the nested-run lifecycle. With the lock acquired, even if
    a fold yields control mid-run (simulated via ``asyncio.sleep``), no
    other fold can open its nested run until the first releases.
    """
    import asyncio as _asyncio

    from src.agents.ml_foundation.model_trainer.nodes import mlflow_logger as mlf

    # Reset the module-level lock so we're testing the lazy-init path
    # within this loop's lifetime.
    mlf._nested_run_lock = None

    lock = mlf._get_nested_run_lock()
    assert isinstance(lock, _asyncio.Lock)
    # Same call returns the same lock — no per-call reallocation
    assert mlf._get_nested_run_lock() is lock

    # Acquired-state observation: when held, second acquirer blocks
    async with mlf._maybe_serialize_nested_run(serialize=True):
        # While held, attempting to acquire elsewhere must fail-fast under
        # asyncio.wait_for with a tiny timeout (proves serialization)
        async def _try_acquire() -> bool:
            try:
                await _asyncio.wait_for(lock.acquire(), timeout=0.05)
                lock.release()
                return True
            except _asyncio.TimeoutError:
                return False

        acquired = await _try_acquire()
        assert acquired is False, "Lock did not actually serialize — second acquirer succeeded"

    # After release, lock acquires immediately
    async with mlf._maybe_serialize_nested_run(serialize=True):
        pass

    # Single-mode path skips the lock entirely (no overhead)
    async with mlf._maybe_serialize_nested_run(serialize=False):
        # Lock must be acquirable concurrently — no serialization
        try:
            await _asyncio.wait_for(lock.acquire(), timeout=0.05)
            lock.release()
            unblocked = True
        except _asyncio.TimeoutError:
            unblocked = False
        assert unblocked, "single-mode incorrectly held the nested-run lock"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parent_run_logs_bca_unstable_warning_per_metric() -> None:
    """Parent run must log ``aggregate_<metric>_bca_unstable: 1.0|0.0`` per metric (cycle-16 I-1).

    Asserts presence + correct boolean-as-float encoding for both stable
    (varying smooth values) and unstable (constant values) cases. MLflow UI
    consumers need this flag to distinguish reliable BCa CIs from degenerate
    fallbacks where ``bca_ci_lo/hi`` are None and percentile_ci should be
    preferred.
    """
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    fake = _FakeConnector()
    agent = ModelTrainerAgent()
    mock_graph = _fold_invocation_recorder_constant_metric(fake)
    with (
        patch.object(
            ModelTrainerAgent,
            "_get_mlflow_connector_or_none",
            staticmethod(lambda: fake),
        ),
        patch.object(agent.graph, "ainvoke", mock_graph),
    ):
        output = await agent.run(_make_input(k=10))

    parent_runs = [r for r in fake.runs if not r.is_nested]
    assert len(parent_runs) == 1
    parent = parent_runs[0]
    # Stable metric: bca_unstable=0.0 + finite bca endpoints
    assert "aggregate_auc_roc_bca_unstable" in parent.metrics, (
        f"Missing aggregate_auc_roc_bca_unstable in parent metrics: {sorted(parent.metrics.keys())}"
    )
    assert parent.metrics["aggregate_auc_roc_bca_unstable"] == 0.0
    assert "aggregate_auc_roc_bca_lo" in parent.metrics
    assert "aggregate_auc_roc_bca_hi" in parent.metrics
    # Unstable metric: bca_unstable=1.0 + bca endpoints OMITTED (because they
    # are None when the warning fires) — percentile_ci is the fallback
    assert "aggregate_brier_score_bca_unstable" in parent.metrics
    assert parent.metrics["aggregate_brier_score_bca_unstable"] == 1.0
    assert "aggregate_brier_score_bca_lo" not in parent.metrics
    assert "aggregate_brier_score_bca_hi" not in parent.metrics
    # In-process aggregate dict still carries the flag for direct callers
    aggregate = output["aggregate_metrics"]
    assert aggregate["auc_roc"].bca_unstable_warning is False
    assert aggregate["brier_score"].bca_unstable_warning is True

    # Cycle-17 COSMETIC-1/2: parent-level summary metrics + tag for
    # multi-run comparison ergonomics. The constant-metric recorder above
    # produces N=2 metrics (auc_roc stable, brier_score unstable), so the
    # summary should be count=1, fraction=0.5, tag="true".
    assert "aggregate_bca_unstable_metric_count" in parent.metrics
    assert parent.metrics["aggregate_bca_unstable_metric_count"] == 1.0
    assert "aggregate_bca_unstable_metric_fraction" in parent.metrics
    assert parent.metrics["aggregate_bca_unstable_metric_fraction"] == pytest.approx(0.5, abs=1e-12)
    assert parent.tags.get("has_bca_unstable") == "true"


def _fold_invocation_recorder_with_failure(connector: _FakeConnector, failed_fold_idx: int):
    """Recorder variant where one fold raises mid-execution.

    Used for cycle-17 IMPORTANT-4 verification: when a fold raises,
    `_run_repeated_splits` records `fold_status="failed"` and the
    parent run's lifecycle code MUST emit `n_failed_folds` metric +
    `aggregate_status` tag BEFORE the parent run closes.
    """
    from unittest.mock import AsyncMock

    async def fake_ainvoke(state: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(state.get("fold_idx", 0))
        is_repeated_fold = state.get("evaluation_mode") == "repeated_k10" and bool(
            state.get("repeated_mode_fold_invocation", False)
        )
        if is_repeated_fold and idx == failed_fold_idx:
            # Simulate a downstream node failure inside the fold's subgraph
            raise RuntimeError(f"simulated fold {idx} failure")
        if is_repeated_fold:
            tags = {
                "fold_idx": str(idx),
                "evaluation_mode": "repeated_k10",
                "fold_seed": str(int(state.get("fold_random_state", 0))),
            }
            async with connector.start_run(
                experiment_id="exp:test_g4_mlflow_repeated_exp",
                run_name=f"fold_{idx:02d}",
                tags=tags,
                nested=True,
            ):
                pass
        return {
            **state,
            "trained_model": object(),
            "auc_roc": 0.77 + 0.005 * idx,
            "brier_score": 0.18 - 0.002 * idx,
            "framework": "lightgbm",
        }

    return AsyncMock(side_effect=fake_ainvoke)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parent_run_records_partial_failure_observability() -> None:
    """Cycle-17 IMPORTANT-4: parent run carries `aggregate_status` + `n_failed_folds`.

    Per the cycle-17 verdict, fold-level exceptions are caught inside
    `_execute_one_fold` and converted to `fold_status="failed"`, so the parent
    MLflow context never sees the exception and would otherwise close as
    FINISHED with no partial-failure signal at the run level. The cycle-17
    fix logs `n_failed_folds` as a parent metric and `aggregate_status` as a
    parent tag BEFORE the parent context closes. This test exercises that path.
    """
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    fake = _FakeConnector()
    agent = ModelTrainerAgent()
    failed_idx = 2
    mock_graph = _fold_invocation_recorder_with_failure(fake, failed_fold_idx=failed_idx)
    with (
        patch.object(
            ModelTrainerAgent,
            "_get_mlflow_connector_or_none",
            staticmethod(lambda: fake),
        ),
        patch.object(agent.graph, "ainvoke", mock_graph),
    ):
        output = await agent.run(_make_input(k=5))

    # Output dict status (already covered by other tests; reaffirm here)
    assert output.get("aggregate_status") == "PARTIAL"
    assert sum(1 for fm in output["fold_metrics"] if fm.get("fold_status") == "failed") == 1

    # Parent run observability: n_failed_folds metric + aggregate_status tag
    parent_runs = [r for r in fake.runs if not r.is_nested]
    assert len(parent_runs) == 1
    parent = parent_runs[0]
    assert "n_failed_folds" in parent.metrics, (
        f"Parent run is missing n_failed_folds metric: {sorted(parent.metrics.keys())}"
    )
    assert parent.metrics["n_failed_folds"] == 1.0
    assert parent.tags.get("aggregate_status") == "PARTIAL", (
        f"Parent run aggregate_status tag wrong: {parent.tags!r}"
    )
    assert parent.tags.get("n_failed_folds") == "1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parent_run_records_complete_status_when_all_ok() -> None:
    """Cycle-17 IMPORTANT-4 complement: zero-failure path emits COMPLETE + n_failed_folds=0.

    Mirrors `test_parent_run_records_partial_failure_observability` for the
    success path so we have explicit coverage of both branches of the new
    parent-status logic.
    """
    from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent

    fake = _FakeConnector()
    agent = ModelTrainerAgent()
    mock_graph = _fold_invocation_recorder_with_logging(fake)
    with (
        patch.object(
            ModelTrainerAgent,
            "_get_mlflow_connector_or_none",
            staticmethod(lambda: fake),
        ),
        patch.object(agent.graph, "ainvoke", mock_graph),
    ):
        output = await agent.run(_make_input(k=5))

    assert output.get("aggregate_status") == "COMPLETE"
    parent_runs = [r for r in fake.runs if not r.is_nested]
    assert len(parent_runs) == 1
    parent = parent_runs[0]
    assert parent.metrics.get("n_failed_folds") == 0.0
    assert parent.tags.get("aggregate_status") == "COMPLETE"
    assert parent.tags.get("n_failed_folds") == "0"
