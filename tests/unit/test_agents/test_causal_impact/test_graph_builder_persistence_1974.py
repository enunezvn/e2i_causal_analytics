"""graph_builder persists every discovery run to ``public.discovered_dags`` (#1974).

Owner requirement: the write must never be silently best-effort. These tests
pin the visible contract of the persistence step wired into
``GraphBuilderNode.execute``:

* success  -> ``discovered_dag_id`` in the returned state, no warning;
* failure  -> ``discovered_dag_persist_error`` + an entry in the ``warnings``
  accumulator + an ERROR log carrying the dag hash and session id, and the
  run itself continues (the DAG is still built; persistence never crashes an
  analysis);
* a missing Supabase (``ServiceConnectionError``) degrades the same way but at
  WARNING level — mirrors ``refutation._build_expert_review_gate``'s
  config-missing branch; unlike that precedent, an UNEXPECTED error is also
  caught (never re-raised into the node's outer failure path), because a lost
  audit row must not turn minutes of causal compute into ``status=failed`` —
  it must be VISIBLE instead;
* the manual / skipped-discovery paths do not touch the repository at all.

The unit-root conftest stubs ``_persist_discovered_dag`` for every other unit
test (the #788 precedent); this module opts out by nodeid and exercises the
real step against a fake repository.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import src.agents.causal_impact.nodes.graph_builder as graph_builder_mod
from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.agents.causal_impact.state import CausalImpactState
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
)
from src.memory.services.factories import ServiceConnectionError
from src.repositories.discovered_dag import DiscoveredDagPersistError

SESSION = "7c9e6679-7425-40de-944b-e07fc1f90ae7"


class _StubRunner:
    """Canned successful discovery: c->t (stability 0.95) and t->y (1.0)."""

    async def discover_dag(self, data, config, session_id=None) -> DiscoveryResult:
        dag = nx.DiGraph()
        dag.add_edge("c", "t")
        dag.add_edge("t", "y")
        edges = [
            DiscoveredEdge("c", "t", confidence=0.95, algorithms=["pc"], bootstrap_stability=0.95),
            DiscoveredEdge("t", "y", confidence=1.0, algorithms=["pc"], bootstrap_stability=1.0),
        ]
        return DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=[
                AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.PC,
                    adjacency_matrix=np.zeros((3, 3), dtype=int),
                    edge_list=[("c", "t"), ("t", "y")],
                    runtime_seconds=0.02,
                    converged=True,
                )
            ],
            session_id=session_id,
        )


class _FakeRepository:
    def __init__(self, *, dag_id: str = "dag-0001", error: Optional[Exception] = None) -> None:
        self.dag_id = dag_id
        self.error = error
        self.payloads: List[Dict[str, Any]] = []

    async def record(self, payload: Dict[str, Any]) -> str:
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return self.dag_id


class _Factory:
    """Async factory standing in for ``_build_discovered_dag_repository``."""

    def __init__(
        self, repository: Optional[_FakeRepository] = None, raises: Optional[Exception] = None
    ):
        self.repository = repository
        self.raises = raises
        self.calls = 0

    async def __call__(self) -> Any:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.repository


def _state(**overrides: Any) -> CausalImpactState:
    state: Dict[str, Any] = {
        "query": "What is the causal effect of t on y?",
        "query_id": "analysis-1974",
        "session_id": SESSION,
        "treatment_var": "t",
        "outcome_var": "y",
        "confounders": ["c"],
        "modeled_confounders": ["c"],
        "data_source": "database",
        "data_cache": {
            "estimation_data": pd.DataFrame(
                {"t": [0.0, 1.0] * 20, "y": [0.0, 1.0] * 20, "c": [0.5] * 40}
            )
        },
        "auto_discover": True,
        "discovery_guided": True,
        "warnings": [],
    }
    state.update(overrides)
    return cast(CausalImpactState, state)


def _wire(monkeypatch: pytest.MonkeyPatch, factory: _Factory) -> GraphBuilderNode:
    monkeypatch.setattr(graph_builder_mod, "_build_discovered_dag_repository", factory)
    node = GraphBuilderNode()
    node._discovery_runner = _StubRunner()  # type: ignore[assignment]
    return node


def _persist_warnings(result: Dict[str, Any]) -> List[str]:
    return [w for w in result.get("warnings", []) if "persist" in w.lower()]


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_sets_discovered_dag_id_and_sends_the_run(monkeypatch):
    repo = _FakeRepository(dag_id="dag-abc")
    factory = _Factory(repo)
    node = _wire(monkeypatch, factory)

    result = await node.execute(_state())

    assert result.get("status") != "failed"
    assert result["discovered_dag_id"] == "dag-abc"
    assert "discovered_dag_persist_error" not in result
    assert _persist_warnings(result) == []
    assert factory.calls == 1
    assert len(repo.payloads) == 1
    payload = repo.payloads[0]
    # Keyed by the run and carrying the shipped DAG's hash + the estimand.
    assert payload["dag_version_hash"] == result["dag_version_hash"]
    assert payload["query_id"] == "analysis-1974"
    assert payload["session_id"] == SESSION
    assert payload["treatment_variable"] == "t"
    assert payload["outcome_variable"] == "y"
    assert payload["gate_decision"] == result["causal_graph"]["discovery_gate_decision"]
    assert payload["algorithms_used"] == ["pc"]
    assert payload["n_edges"] == 2
    # The stub runner sets no metadata: n_samples / features come from the
    # frame discovery ran on, never a fabricated 0.
    assert payload["n_samples"] == 40
    assert set(payload["feature_names"]) == {"t", "y", "c"}
    # The shipped DAG (what was hashed) rides in metadata with provenance.
    shipped = payload["metadata"]["shipped_dag"]
    assert shipped["edges"] == [list(e) for e in result["causal_graph"]["edges"]]
    assert shipped["edge_provenance"] == result["causal_graph"]["edge_provenance"]
    assert shipped["adjustment_sets"] == result["causal_graph"]["adjustment_sets"]
    assert payload["metadata"]["discovery_latency_ms"] == result["discovery_latency_ms"]
    # Unit tree pins real mode (no E2I_INCLUDE_SYNTHETIC, data_source=database).
    assert payload["is_synthetic"] is False


@pytest.mark.asyncio
async def test_declared_synthetic_data_source_is_persisted_as_synthetic(monkeypatch):
    repo = _FakeRepository()
    node = _wire(monkeypatch, _Factory(repo))

    await node.execute(_state(data_source="synthetic"))

    assert repo.payloads[0]["is_synthetic"] is True


@pytest.mark.asyncio
async def test_showcase_deployment_flag_is_persisted_as_synthetic(monkeypatch):
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    repo = _FakeRepository()
    node = _wire(monkeypatch, _Factory(repo))

    await node.execute(_state())

    assert repo.payloads[0]["is_synthetic"] is True


# ---------------------------------------------------------------------------
# Failure: visible, never a crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repository_failure_sets_error_key_warning_and_error_log(monkeypatch, caplog):
    repo = _FakeRepository(error=DiscoveredDagPersistError("record_discovered_dag failed: boom"))
    node = _wire(monkeypatch, _Factory(repo))

    with caplog.at_level(logging.ERROR, logger=graph_builder_mod.__name__):
        result = await node.execute(_state())

    # The analysis continues: DAG built, phase advances, no failure status.
    assert "causal_graph" in result
    assert result.get("status") != "failed"
    assert result["current_phase"] == "estimating"
    # ...but the loss is visible in three places.
    assert "discovered_dag_id" not in result
    error = result["discovered_dag_persist_error"]
    assert "boom" in error
    assert result["dag_version_hash"] in error
    assert SESSION in error
    warnings = _persist_warnings(result)
    assert len(warnings) == 1 and "boom" in warnings[0]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "no ERROR log for the failed persistence"
    assert result["dag_version_hash"] in errors[0].getMessage()
    assert SESSION in errors[0].getMessage()


@pytest.mark.asyncio
async def test_missing_supabase_degrades_with_warning_level(monkeypatch, caplog):
    factory = _Factory(raises=ServiceConnectionError("Supabase", "SUPABASE_URL is not set"))
    node = _wire(monkeypatch, factory)

    with caplog.at_level(logging.WARNING, logger=graph_builder_mod.__name__):
        result = await node.execute(_state())

    assert result.get("status") != "failed"
    assert "discovered_dag_id" not in result
    assert "unavailable" in result["discovered_dag_persist_error"]
    assert "SUPABASE_URL" in result["discovered_dag_persist_error"]
    assert len(_persist_warnings(result)) == 1
    levels = {r.levelno for r in caplog.records if "persist" in r.getMessage().lower()}
    assert logging.WARNING in levels
    assert logging.ERROR not in levels  # config-missing is degrade, not a bug


@pytest.mark.asyncio
async def test_unexpected_error_is_caught_visibly_not_reraised(monkeypatch, caplog):
    """A non-connection error (a bug, a transport failure such as the
    httpx.ConnectError measured under the dead-Supabase pin) must not reach
    the node's outer ``except`` and fail the whole analysis — but it must be
    logged at ERROR and surfaced in state + warnings."""
    factory = _Factory(raises=RuntimeError("All connection attempts failed"))
    node = _wire(monkeypatch, factory)

    with caplog.at_level(logging.ERROR, logger=graph_builder_mod.__name__):
        result = await node.execute(_state())

    assert result.get("status") != "failed"
    assert "graph_builder_error" not in result
    assert "All connection attempts failed" in result["discovered_dag_persist_error"]
    assert len(_persist_warnings(result)) == 1
    assert any(r.levelno == logging.ERROR for r in caplog.records)


@pytest.mark.asyncio
async def test_persist_step_returns_only_new_warnings(monkeypatch):
    """warnings is an operator.add accumulator: the node must return ONLY its
    new entries, never re-submit the pre-existing ones (state.py discipline)."""
    repo = _FakeRepository(error=DiscoveredDagPersistError("boom"))
    node = _wire(monkeypatch, _Factory(repo))

    result = await node.execute(_state(warnings=["pre-existing warning"]))

    assert "pre-existing warning" not in result["warnings"]
    assert len(result["warnings"]) == 1


@pytest.mark.asyncio
async def test_unconvertible_frame_is_a_persist_error_not_a_raise(monkeypatch):
    """codex iter-1 LOW: the helper's never-raise guarantee must cover its own
    frame preparation. A cache value pandas cannot turn into a DataFrame is
    reported as a persist error, never propagated to the node's outer except."""
    factory = _Factory(_FakeRepository())
    monkeypatch.setattr(graph_builder_mod, "_build_discovered_dag_repository", factory)
    state = _state(data_cache={"estimation_data": object()})
    result = await _StubRunner().discover_dag(None, DiscoveryConfig(), None)

    delta = await graph_builder_mod._persist_discovered_dag(
        state=state,
        discovery_result=result,
        gate_evaluation={"decision": "reject", "confidence": 0.0, "reasons": []},
        causal_graph={"dag_version_hash": "h" * 64, "nodes": [], "edges": []},
        treatment="t",
        outcome="y",
        discovery_latency_ms=1.0,
    )

    assert "discovered_dag_id" not in delta
    assert "DataFrame" in delta["discovered_dag_persist_error"]
    assert len(delta["warnings"]) == 1
    assert factory.calls == 0  # positive control: never reached the repository


# ---------------------------------------------------------------------------
# No discovery -> no persistence (positive-controlled)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_dag_path_never_touches_the_repository(monkeypatch):
    factory = _Factory(_FakeRepository())
    node = _wire(monkeypatch, factory)

    result = await node.execute(_state(auto_discover=False))

    assert "causal_graph" in result  # positive control: the node ran
    assert factory.calls == 0
    assert "discovered_dag_id" not in result
    assert "discovered_dag_persist_error" not in result


@pytest.mark.asyncio
async def test_skipped_discovery_never_touches_the_repository(monkeypatch):
    factory = _Factory(_FakeRepository())
    node = _wire(monkeypatch, factory)

    result = await node.execute(_state(data_cache={}))

    # Positive control: the skip itself is surfaced (existing M-gb1 contract).
    assert result.get("discovery_skip_reason")
    assert factory.calls == 0
    assert "discovered_dag_id" not in result
    assert "discovered_dag_persist_error" not in result


# ---------------------------------------------------------------------------
# State declaration: LangGraph drops undeclared channels
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_persistence_keys_survive_langgraph_channel_filter():
    from langgraph.graph import END, StateGraph

    def probe(state):
        return {
            "discovered_dag_id": "dag-1",
            "discovered_dag_persist_error": "err",
            "undeclared_sentinel_1974": "dropped?",
        }

    g = StateGraph(CausalImpactState)
    g.add_node("probe", probe)
    g.set_entry_point("probe")
    g.add_edge("probe", END)
    out = g.compile().invoke(
        {
            "query": "q",
            "query_id": "t1",
            "treatment_var": "t",
            "outcome_var": "y",
            "confounders": [],
            "data_source": "synthetic",
        }
    )
    # Positive control: the filter is active — an undeclared key is dropped.
    assert "undeclared_sentinel_1974" not in out
    assert out["discovered_dag_id"] == "dag-1"
    assert out["discovered_dag_persist_error"] == "err"


# ---------------------------------------------------------------------------
# The unit-tree containment stub exists and is bypassed here
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unit_tree_stub_repository_is_as_strict_as_the_transport():
    """codex iter-2 MED: the conftest stand-in must reject what httpx rejects
    (allow_nan=False), or a NaN-bearing payload passes the unit tree while
    production reports a persist failure."""
    from tests.unit.conftest import _UnitStubDiscoveredDagRepository

    stub = _UnitStubDiscoveredDagRepository()
    with pytest.raises(ValueError):
        await stub.record({"score": float("nan")})
    with pytest.raises(ValueError):
        await stub.record({"score": float("inf")})
    # Positive control: a finite payload is accepted and recorded.
    assert await stub.record({"score": 0.5}) == "unit-stub-discovered-dag-id"
    assert stub.payloads == [{"score": 0.5}]


@pytest.mark.asyncio
async def test_this_module_exercises_the_real_persist_step(monkeypatch):
    """Guard against the conftest stub silently covering these tests: the
    real step must be the one running (the fake factory gets called)."""
    factory = _Factory(_FakeRepository())
    node = _wire(monkeypatch, factory)
    await node.execute(_state())
    assert factory.calls == 1
    assert not hasattr(graph_builder_mod._persist_discovered_dag, "assert_awaited")
