"""DiscoveredDagRepository — the writer for ``public.discovered_dags`` (#1974).

Pins the contract the migration ml/036 RPC ``record_discovered_dag(jsonb)``
expects (one payload, one atomic call, a receipt the caller verifies) and the
payload builder that maps a ``DiscoveryResult`` + gate evaluation + the shipped
``CausalGraph`` onto the 026 columns. The faithful SQL-shape proof is the
BEGIN/ROLLBACK rehearsal that feeds the builder's exact payload through the RPC.

Every failure path here is VISIBLE by design (owner requirement on #1974: the
table's emptiness must never again mean nothing): no client, an RPC error, or a
receipt that does not match what was sent all raise ``DiscoveredDagPersistError``
— the repository never returns a silent None.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.base import (
    AlgorithmResult,
    CausalPriorKnowledge,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    EdgeType,
    GateDecision,
)
from src.repositories.discovered_dag import (
    DiscoveredDagPersistError,
    DiscoveredDagRepository,
    build_discovered_dag_payload,
    resolve_frame_provenance,
)

HASH = "a" * 64
SESSION = "2f1a5c7e-0b3d-4e8f-9a6b-1c2d3e4f5a6b"


# ---------------------------------------------------------------------------
# Fixtures: a realistic DiscoveryResult / gate evaluation / shipped graph
# ---------------------------------------------------------------------------


def _config() -> DiscoveryConfig:
    return DiscoveryConfig(
        algorithms=[DiscoveryAlgorithmType.PC],
        alpha=0.05,
        ensemble_threshold=0.5,
        bootstrap_resamples=20,
        latent_diagnostic=True,
        prior_knowledge=CausalPriorKnowledge(
            tiers=[["c"], ["t"], ["y"]], required_edges=[("t", "y")]
        ),
    )


def _result(*, success: bool = True, with_metadata: bool = True) -> DiscoveryResult:
    config = _config()
    if not success:
        return DiscoveryResult(success=False, config=config, metadata={"error": "no converge"})
    dag = nx.DiGraph()
    dag.add_edge("c", "t")
    dag.add_edge("t", "y")
    edges = [
        DiscoveredEdge(
            source="c",
            target="t",
            edge_type=EdgeType.DIRECTED,
            confidence=0.9,
            algorithm_votes=1,
            algorithms=["pc"],
            bootstrap_stability=0.9,
        ),
        DiscoveredEdge(
            source="t",
            target="y",
            edge_type=EdgeType.DIRECTED,
            confidence=1.0,
            algorithm_votes=1,
            algorithms=["pc"],
            bootstrap_stability=1.0,
        ),
    ]
    runs = [
        AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.PC,
            adjacency_matrix=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int),
            edge_list=[("c", "t"), ("t", "y")],
            runtime_seconds=1.25,
            converged=True,
            score=None,
            metadata={"n_ci_tests": np.int64(17)},
        )
    ]
    metadata: Dict[str, Any] = {}
    if with_metadata:
        metadata = {
            "total_runtime_seconds": 1.5,
            "node_names": ["c", "t", "y"],
            "n_samples": 40,
            "bootstrap": {"n_resamples": 20, "n_succeeded": 20},
            "latent_diagnostic": {
                "ran": True,
                "converged": True,
                "bidirected_edges": [("t", "y")],
                "treatment": "t",
                "outcome": "y",
                "flag": True,
            },
        }
    return DiscoveryResult(
        success=True,
        config=config,
        ensemble_dag=dag,
        edges=edges,
        algorithm_results=runs,
        gate_decision=GateDecision.ACCEPT,
        gate_confidence=0.91,
        metadata=metadata,
    )


def _gate_evaluation() -> Dict[str, Any]:
    return {
        "decision": "accept",
        "confidence": 0.91,
        "reasons": ["Corroboration (bootstrap): 95.00%", "High confidence"],
        "n_high_confidence_edges": 1,
        "high_confidence_edges": [{"source": "c", "target": "t", "confidence": 0.9}],
        "n_rejected_edges": 0,
        "warnings": [],
        "metadata": {"latent_diagnostic": {"flag": True}},
    }


def _causal_graph() -> Dict[str, Any]:
    return {
        "nodes": ["c", "t", "y"],
        "edges": [("c", "t"), ("t", "y")],
        "treatment_nodes": ["t"],
        "outcome_nodes": ["y"],
        "adjustment_sets": [["c"]],
        "dag_dot": "digraph {}",
        "confidence": 0.91,
        "dag_version_hash": HASH,
        "discovery_enabled": True,
        "discovery_gate_decision": "accept",
        "discovery_algorithms_used": ["pc"],
        "discovery_confidence": 0.91,
        "discovery_n_edges": 2,
        "augmented_edges": [],
        "discovery_dag_overridden": False,
        "edge_provenance": [
            {"source": "c", "target": "t", "provenance": "discovered"},
            {"source": "t", "target": "y", "provenance": "required_prior"},
        ],
        "latent_diagnostic": {"flag": True, "ran": True},
    }


def _payload(**overrides: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "discovery_result": _result(),
        "gate_evaluation": _gate_evaluation(),
        "causal_graph": _causal_graph(),
        "treatment": "t",
        "outcome": "y",
        "query_id": "analysis-123",
        "session_id": SESSION,
        "is_synthetic": True,
        "discovery_latency_ms": 1500.0,
    }
    kwargs.update(overrides)
    return build_discovered_dag_payload(**kwargs)


# ---------------------------------------------------------------------------
# Recording supabase-style client
# ---------------------------------------------------------------------------


class _Query:
    def __init__(self, data: List[Dict[str, Any]] | None = None) -> None:
        self.calls: List[tuple] = []
        self._data = data or []

    def _rec(self, name: str, *args: Any) -> "_Query":
        self.calls.append((name, args))
        return self

    def select(self, *a: Any) -> "_Query":
        return self._rec("select", *a)

    def eq(self, *a: Any) -> "_Query":
        return self._rec("eq", *a)

    def order(self, *a: Any, **kw: Any) -> "_Query":
        return self._rec("order", *a)

    def limit(self, *a: Any) -> "_Query":
        return self._rec("limit", *a)

    def offset(self, *a: Any) -> "_Query":
        return self._rec("offset", *a)

    async def execute(self) -> Any:
        return MagicMock(data=self._data)


class _Client:
    def __init__(self, receipt: Any = None, rpc_error: Exception | None = None) -> None:
        self.rpc_calls: List[tuple] = []
        self.query = _Query()
        builder = MagicMock()
        if rpc_error is not None:
            builder.execute = AsyncMock(side_effect=rpc_error)
        else:
            builder.execute = AsyncMock(return_value=MagicMock(data=receipt))
        self._builder = builder

    def rpc(self, name: str, params: Dict[str, Any]) -> Any:
        self.rpc_calls.append((name, params))
        return self._builder

    def table(self, name: str) -> _Query:
        self.query.calls.append(("table", (name,)))
        return self.query


# ---------------------------------------------------------------------------
# Repository identity
# ---------------------------------------------------------------------------


def test_repository_identity():
    assert DiscoveredDagRepository.table_name == "discovered_dags"
    assert DiscoveredDagRepository.RPC_NAME == "record_discovered_dag"
    assert DiscoveredDagRepository.HAS_PROVENANCE is True


# ---------------------------------------------------------------------------
# record(): one RPC call, receipt verified, failures visible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_calls_rpc_with_payload_and_returns_dag_id():
    payload = _payload()
    client = _Client(receipt={"dag_id": "dag-1", "n_algorithm_runs": 1, "n_edges": 2})
    repo = DiscoveredDagRepository(supabase_client=client)

    dag_id = await repo.record(payload)

    assert dag_id == "dag-1"
    assert client.rpc_calls == [("record_discovered_dag", {"p_payload": payload})]


@pytest.mark.asyncio
async def test_record_raises_when_receipt_counts_do_not_match_payload():
    payload = _payload()
    client = _Client(receipt={"dag_id": "dag-1", "n_algorithm_runs": 1, "n_edges": 1})
    repo = DiscoveredDagRepository(supabase_client=client)

    with pytest.raises(DiscoveredDagPersistError) as info:
        await repo.record(payload)
    assert "n_edges" in str(info.value)
    assert "dag-1" in str(info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [None, [], {}, {"n_edges": 2}, "dag-1"])
async def test_record_raises_on_malformed_receipt(receipt: Any):
    client = _Client(receipt=receipt)
    repo = DiscoveredDagRepository(supabase_client=client)
    with pytest.raises(DiscoveredDagPersistError):
        await repo.record(_payload())


@pytest.mark.asyncio
async def test_record_wraps_rpc_failure_with_cause():
    boom = RuntimeError("All connection attempts failed")
    client = _Client(rpc_error=boom)
    repo = DiscoveredDagRepository(supabase_client=client)
    with pytest.raises(DiscoveredDagPersistError) as info:
        await repo.record(_payload())
    assert info.value.__cause__ is boom
    assert "record_discovered_dag" in str(info.value)


@pytest.mark.asyncio
async def test_record_without_client_raises_not_silently_returns():
    """BaseRepository.create returns the entity untouched when no client is
    configured; the writer must NOT inherit that silence."""
    repo = DiscoveredDagRepository(supabase_client=None)
    with pytest.raises(DiscoveredDagPersistError):
        await repo.record(_payload())


@pytest.mark.asyncio
async def test_record_refuses_payload_without_provenance():
    """The RPC rejects a payload without is_synthetic; the repository refuses
    it before the round-trip so the failure is local and explicit."""
    payload = _payload()
    payload.pop("is_synthetic")
    client = _Client(receipt={"dag_id": "dag-1", "n_algorithm_runs": 1, "n_edges": 2})
    repo = DiscoveredDagRepository(supabase_client=client)
    with pytest.raises(DiscoveredDagPersistError, match="is_synthetic"):
        await repo.record(payload)
    assert client.rpc_calls == []


# ---------------------------------------------------------------------------
# Payload builder: every 026 column, from what the node holds
# ---------------------------------------------------------------------------


def test_payload_maps_dag_row_columns():
    payload = _payload()

    assert payload["session_id"] == SESSION
    assert payload["query_id"] == "analysis-123"
    assert payload["dag_version_hash"] == HASH
    assert payload["treatment_variable"] == "t"
    assert payload["outcome_variable"] == "y"
    assert payload["n_samples"] == 40
    assert payload["n_features"] == 3
    assert payload["feature_names"] == ["c", "t", "y"]
    assert payload["config"] == _config().to_dict()
    assert payload["algorithms_used"] == ["pc"]
    assert payload["ensemble_threshold"] == 0.5
    assert payload["alpha"] == 0.05
    assert payload["n_edges"] == 2
    assert payload["n_nodes"] == 3
    assert [(e["source"], e["target"]) for e in payload["edge_list"]] == [("c", "t"), ("t", "y")]
    assert payload["confidence_scores"] == {"c->t": 0.9, "t->y": 1.0}
    # Adjacency in feature_names order (c, t, y): c->t, t->y.
    assert payload["adjacency_matrix"] == [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    assert payload["gate_decision"] == "accept"
    assert payload["gate_confidence"] == 0.91
    assert payload["gate_reasons"] == _gate_evaluation()["reasons"]
    assert payload["total_runtime_seconds"] == 1.5
    assert payload["is_synthetic"] is True
    assert payload["discovery_timestamp"].endswith("+00:00")


def test_payload_metadata_carries_everything_the_node_holds():
    """What 026 has no column for rides in ``metadata`` under stable keys —
    the shipped DAG (what was hashed), the gate's full evaluation, the
    discovery runner's metadata (bootstrap / latent diagnostic), and the
    node-level facts."""
    meta = _payload()["metadata"]

    shipped = meta["shipped_dag"]
    assert shipped["nodes"] == ["c", "t", "y"]
    assert shipped["edges"] == [["c", "t"], ["t", "y"]]
    assert shipped["edge_provenance"] == _causal_graph()["edge_provenance"]
    assert shipped["adjustment_sets"] == [["c"]]
    assert shipped["augmented_edges"] == []
    assert shipped["discovery_dag_overridden"] is False
    assert shipped["confidence"] == 0.91
    assert meta["gate_evaluation"] == _gate_evaluation()
    assert meta["discovery"]["bootstrap"] == {"n_resamples": 20, "n_succeeded": 20}
    assert meta["discovery"]["latent_diagnostic"]["flag"] is True
    assert meta["discovery_latency_ms"] == 1500.0
    assert meta["success"] is True
    assert meta["algorithm_agreement"] == pytest.approx(1.0)


def test_payload_algorithm_runs_one_per_algorithm_result():
    runs = _payload()["algorithm_runs"]
    assert len(runs) == 1
    run = runs[0]
    assert run["algorithm"] == "pc"
    assert run["runtime_seconds"] == 1.25
    assert run["converged"] is True
    assert run["n_edges"] == 2
    assert run["edge_list"] == [["c", "t"], ["t", "y"]]
    assert run["adjacency_matrix"] == [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    assert run["score"] is None
    assert run["parameters"] == _config().to_dict()
    assert run["metadata"] == {"n_ci_tests": 17}


def test_payload_edges_carry_confidence_type_votes_algorithms_and_stability():
    edges = _payload()["edges"]
    assert [(e["source_node"], e["target_node"]) for e in edges] == [("c", "t"), ("t", "y")]
    assert {e["edge_type"] for e in edges} == {"directed"}
    assert [e["confidence"] for e in edges] == [0.9, 1.0]
    assert [e["algorithm_votes"] for e in edges] == [1, 1]
    assert [e["algorithms"] for e in edges] == [["pc"], ["pc"]]
    assert [e["metadata"]["bootstrap_stability"] for e in edges] == [0.9, 1.0]


def test_payload_enum_values_are_the_db_labels():
    """The RPC casts these strings into the moved enums; they must be the
    Python enum VALUES, never the member names."""
    payload = _payload()
    assert payload["gate_decision"] in {m.value for m in GateDecision}
    assert payload["algorithm_runs"][0]["algorithm"] in {m.value for m in DiscoveryAlgorithmType}
    assert all(e["edge_type"] in {m.value for m in EdgeType} for e in payload["edges"])
    assert payload["algorithms_used"] == [a.value for a in _config().algorithms]


def test_payload_is_plain_json_even_with_numpy_inputs():
    """numpy scalars/arrays from the algorithm wrappers and tuples from the
    latent diagnostic must not reach httpx — the builder normalises to plain
    JSON types."""
    payload = _payload()
    text = json.dumps(payload)  # raises on numpy types
    round_trip = json.loads(text)
    assert round_trip == payload
    assert isinstance(payload["algorithm_runs"][0]["metadata"]["n_ci_tests"], int)
    assert payload["metadata"]["discovery"]["latent_diagnostic"]["bidirected_edges"] == [["t", "y"]]


def test_payload_maps_non_finite_floats_to_null():
    """codex iter-2 MED: httpx encodes JSON with allow_nan=False, so a NaN or
    Infinity anywhere in the payload would fail the whole write at the
    transport. JSON has no NaN; the honest representation of "no finite
    value" is null. Positive control: finite values survive unchanged."""
    result = _result()
    result.algorithm_results[0].score = float("nan")
    result.algorithm_results[0].metadata["objective"] = float("inf")
    result.metadata["bootstrap"]["mean_stability"] = float("-inf")
    result.metadata["bootstrap"]["finite_control"] = 0.25
    payload = _payload(discovery_result=result)

    run = payload["algorithm_runs"][0]
    assert run["score"] is None
    assert run["metadata"]["objective"] is None
    assert payload["metadata"]["discovery"]["bootstrap"]["mean_stability"] is None
    assert payload["metadata"]["discovery"]["bootstrap"]["finite_control"] == 0.25
    json.dumps(payload, allow_nan=False)  # the transport's exact requirement


def test_payload_falls_back_to_the_frame_when_result_metadata_is_missing():
    """A DiscoveryResult without runner metadata (a failed run, a stub runner)
    still yields honest n_samples / feature_names from the frame it ran on —
    never a fabricated 0 / []."""
    frame = pd.DataFrame({"t": [0, 1, 0, 1], "y": [0, 1, 1, 0], "c": [1, 1, 0, 0]})
    payload = _payload(
        discovery_result=_result(success=False),
        gate_evaluation={"decision": "reject", "confidence": 0.0, "reasons": ["Discovery failed"]},
        frame=frame,
    )
    assert payload["n_samples"] == 4
    assert payload["feature_names"] == ["t", "y", "c"]
    assert payload["n_features"] == 3
    assert payload["n_edges"] == 0
    assert payload["n_nodes"] == 0
    assert payload["edge_list"] == []
    assert payload["adjacency_matrix"] is None
    assert payload["edges"] == []
    assert payload["algorithm_runs"] == []
    assert payload["gate_decision"] == "reject"
    assert payload["metadata"]["success"] is False
    assert payload["metadata"]["discovery"]["error"] == "no converge"


def test_payload_requires_n_samples_from_somewhere():
    """No runner metadata AND no frame: the builder refuses rather than
    inventing a sample count (the column is NOT NULL for a reason)."""
    with pytest.raises(ValueError, match="n_samples"):
        _payload(
            discovery_result=_result(success=False),
            gate_evaluation={"decision": "reject", "confidence": 0.0, "reasons": []},
        )


def test_payload_requires_feature_names_from_somewhere():
    """codex iter-1 MED: n_samples present in runner metadata but NO node_names,
    no frame and no ensemble graph must raise, not persist feature_names=[] /
    n_features=0 as if the run had seen no variables."""
    result = _result(success=False)
    result.metadata["n_samples"] = 40
    with pytest.raises(ValueError, match="feature_names"):
        _payload(
            discovery_result=result,
            gate_evaluation={"decision": "reject", "confidence": 0.0, "reasons": []},
        )


def test_payload_keeps_an_explicitly_empty_node_list():
    """Positive control for the test above: an EXPLICIT empty node list from
    the runner is data, not absence, and is kept as-is."""
    result = _result(success=False)
    result.metadata["n_samples"] = 40
    result.metadata["node_names"] = []
    payload = _payload(
        discovery_result=result,
        gate_evaluation={"decision": "reject", "confidence": 0.0, "reasons": []},
    )
    assert payload["feature_names"] == []
    assert payload["n_features"] == 0


def test_payload_requires_dag_version_hash():
    graph = _causal_graph()
    graph.pop("dag_version_hash")
    with pytest.raises(ValueError, match="dag_version_hash"):
        _payload(causal_graph=graph)


def test_payload_keeps_a_non_uuid_session_id_visible_not_dropped():
    """discovered_dags.session_id is UUID; an orchestrator session id that is
    not a UUID cannot go in the column, but it is not thrown away either."""
    payload = _payload(session_id="orchestrator-session-7")
    assert payload["session_id"] is None
    assert payload["metadata"]["session_id_raw"] == "orchestrator-session-7"
    # Positive control: a real UUID lands in the column, no raw copy.
    assert _payload()["session_id"] == SESSION
    assert "session_id_raw" not in _payload()["metadata"]


# ---------------------------------------------------------------------------
# Provenance resolution (ADR-017): never a silent False
# ---------------------------------------------------------------------------


def test_frame_column_wins_when_present(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    synthetic = pd.DataFrame({"t": [0, 1], "y": [1, 0], "is_synthetic": [False, True]})
    real = pd.DataFrame({"t": [0, 1], "y": [1, 0], "is_synthetic": [False, False]})
    assert resolve_frame_provenance(synthetic, {"data_source": "database"}) is True
    # The data's own column overrides a deployment-level flag.
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    assert resolve_frame_provenance(real, {"data_source": "synthetic"}) is False


def test_declared_synthetic_data_source_is_synthetic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    frame = pd.DataFrame({"t": [0, 1], "y": [1, 0]})
    assert resolve_frame_provenance(frame, {"data_source": "synthetic"}) is True


def test_showcase_deployment_flag_marks_unfiltered_reads_synthetic(
    monkeypatch: pytest.MonkeyPatch,
):
    """With E2I_INCLUDE_SYNTHETIC set, apply_provenance_filter is skipped on
    every read, so a frame without the column may carry synthetic rows — the
    same fact the agent-analyze API already reports as data_source='synthetic'."""
    frame = pd.DataFrame({"t": [0, 1], "y": [1, 0]})
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    assert resolve_frame_provenance(frame, {"data_source": "database"}) is True
    # Positive control: same inputs, flag off -> real.
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert resolve_frame_provenance(frame, {"data_source": "database"}) is False


def test_strict_deployment_real_read_is_real(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    frame = pd.DataFrame({"t": [0, 1], "y": [1, 0]})
    assert resolve_frame_provenance(frame, {"data_source": "patient_journeys"}) is False
    assert resolve_frame_provenance(None, {}) is False


# ---------------------------------------------------------------------------
# Readers: provenance-filtered by default (HAS_PROVENANCE)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_by_dag_version_hash_excludes_synthetic_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    client = _Client()
    repo = DiscoveredDagRepository(supabase_client=client)

    await repo.find_by_dag_version_hash(HASH)

    calls = client.query.calls
    assert ("table", ("discovered_dags",)) in calls
    assert ("eq", ("dag_version_hash", HASH)) in calls
    assert ("eq", ("is_synthetic", False)) in calls


@pytest.mark.asyncio
async def test_find_by_dag_version_hash_opt_in_includes_synthetic(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    client = _Client()
    repo = DiscoveredDagRepository(supabase_client=client)

    await repo.find_by_dag_version_hash(HASH, include_synthetic=True)

    calls = client.query.calls
    assert ("eq", ("dag_version_hash", HASH)) in calls  # positive control: eq recorded
    assert ("eq", ("is_synthetic", False)) not in calls


@pytest.mark.asyncio
async def test_find_by_query_id_filters_on_query_id():
    client = _Client()
    repo = DiscoveredDagRepository(supabase_client=client)
    await repo.find_by_query_id("analysis-123", include_synthetic=True)
    assert ("eq", ("query_id", "analysis-123")) in client.query.calls
