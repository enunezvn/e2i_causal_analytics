"""Discovered-DAG repository — the writer for ``public.discovered_dags`` (#1974).

WHY THIS EXISTS: migration ml/026 (2025-12-30) created the causal-discovery
tables for exactly this record — "discovered DAG structures with confidence
scores, algorithm run metadata, gate evaluation decisions" — but nothing under
``src/`` ever wrote to them. Two blockers, both measured on the live prod
Supabase (issue #1974): the tables lived in schema ``ml``, which PostgREST does
not expose (``PGRST106``), and ``service_role`` held no grant on them.
Migration ml/036 moves them into ``public`` with explicit grants and adds the
atomic RPC ``record_discovered_dag(jsonb)``; this module is the client side.

CONTRACT (owner requirement on #1974): the write is NOT silently best-effort.
Every failure raises :class:`DiscoveredDagPersistError` — no client, an RPC
error, a receipt that does not match what was sent. The graph_builder node
turns that into an error log + a declared state key + a response warning, so
an empty table can never again mean nothing.

ATOMICITY: one RPC call inserts the ``discovered_dags`` row, one
``discovery_algorithm_runs`` row per algorithm and one ``discovered_edges`` row
per ensemble edge in a single transaction, and returns a receipt
``{dag_id, n_algorithm_runs, n_edges}`` that :meth:`DiscoveredDagRepository.record`
verifies against the payload it sent. Sequential inserts would leave partial
rows on a mid-way failure with no way to tell.

PROVENANCE (ADR-017): ``is_synthetic`` is resolved by
:func:`resolve_frame_provenance` from the strongest evidence available at the
write site and is always STATED in the payload — the RPC rejects a payload
that omits it, and :meth:`record` refuses to send one.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, cast
from uuid import UUID

import numpy as np
import pandas as pd

from src.causal_engine.discovery.base import DiscoveryResult
from src.repositories.base import BaseRepository
from src.repositories.provenance import (
    PROVENANCE_COLUMN,
    apply_provenance_filter,
    deployment_includes_synthetic,
)
from src.utils.type_helpers import parse_supabase_rows

logger = logging.getLogger(__name__)


class DiscoveredDagPersistError(RuntimeError):
    """A discovered DAG could not be persisted (or the receipt disagreed).

    Raised for every failure path of :meth:`DiscoveredDagRepository.record`
    so callers have ONE visible signal; the original exception (if any) is
    chained as ``__cause__``.
    """


# ---------------------------------------------------------------------------
# JSON normalisation — algorithm wrappers hand back numpy scalars/arrays and
# the latent diagnostic carries tuples; httpx must see plain JSON types.
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        # Enum members (GateDecision / EdgeType / DiscoveryAlgorithmType).
        return value.value
    return str(value)


def _to_plain_json(value: Any) -> Any:
    """Round-trip through JSON so every leaf is a plain JSON type."""
    return json.loads(json.dumps(value, default=_json_default))


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


# ---------------------------------------------------------------------------
# Provenance resolution
# ---------------------------------------------------------------------------


def resolve_frame_provenance(frame: Optional[pd.DataFrame], state: Mapping[str, Any]) -> bool:
    """Decide ``is_synthetic`` for a DAG discovered from ``frame``.

    Evidence, strongest first — the answer is never a silent default:

    1. The frame's own ``is_synthetic`` column (a caller that passes tagged
       rows): synthetic iff ANY row is synthetic. A DAG learned on a mixed
       frame cannot be attributed to real data.
    2. The caller DECLARED ``data_source='synthetic'`` (the agent's fixture
       path, and what the agent-analyze API labels a showcase run).
    3. The deployment runs with ``E2I_INCLUDE_SYNTHETIC`` (see
       :func:`deployment_includes_synthetic`): every PostgREST read then SKIPS
       the ``is_synthetic=false`` predicate, so a frame without the column may
       carry the synthetic showcase substrate. This is exactly the fact the
       agent-analyze API already reports as ``data_source='synthetic'`` for the
       same run, so the persisted row agrees with the response.
    4. Otherwise the strict real-mode gate filtered every read: real.
    """
    if frame is not None and PROVENANCE_COLUMN in frame.columns:
        return bool(frame[PROVENANCE_COLUMN].astype(bool).any())
    if str(state.get("data_source") or "").strip().lower() == "synthetic":
        return True
    return deployment_includes_synthetic()


# ---------------------------------------------------------------------------
# Payload builder — every 026 column, from what the node holds
# ---------------------------------------------------------------------------


def _as_uuid_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        return str(UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


def _matrix_or_none(matrix: Any) -> Optional[List[List[int]]]:
    if matrix is None:
        return None
    array = np.asarray(matrix)
    if array.size == 0:
        return None
    return cast(List[List[int]], array.tolist())


def build_discovered_dag_payload(
    *,
    discovery_result: DiscoveryResult,
    gate_evaluation: Mapping[str, Any],
    causal_graph: Mapping[str, Any],
    treatment: Optional[str],
    outcome: Optional[str],
    query_id: Optional[str],
    session_id: Optional[str],
    is_synthetic: bool,
    frame: Optional[pd.DataFrame] = None,
    discovery_latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Map a discovery run onto the ``record_discovered_dag`` payload.

    Column mapping (``public.discovered_dags``):

    * ``edge_list`` / ``edges`` / ``confidence_scores`` / ``adjacency_matrix``
      are the ENSEMBLE edges discovery drew (``DiscoveryResult.edges``);
    * ``metadata.shipped_dag`` is the DAG that actually shipped after the gate
      (manual fallback on REVIEW/REJECT, manual+extras on AUGMENT), i.e. the
      graph ``dag_version_hash`` was computed from, with its per-edge
      provenance, adjustment sets, augmented edges and the override flag;
    * ``metadata.gate_evaluation`` is the gate's full ``to_dict()`` (reasons,
      warnings, high-confidence edges, its metadata);
    * ``metadata.discovery`` is the runner's metadata (bootstrap summary,
      latent diagnostic, node names, runtime, or the error of a failed run).

    ``n_samples`` / ``feature_names`` come from the runner's metadata, else from
    ``frame`` (the frame discovery ran on); with neither the builder raises —
    the columns are NOT NULL and a fabricated 0 would be a plausible-wrong
    value. ``dag_version_hash`` is required for the same reason: it is the key
    the expert-review workflow joins on.
    """
    dag_version_hash = causal_graph.get("dag_version_hash")
    if not dag_version_hash:
        raise ValueError("build_discovered_dag_payload: causal_graph has no dag_version_hash")

    result_meta: Dict[str, Any] = dict(discovery_result.metadata or {})
    config = discovery_result.config
    config_dict = config.to_dict() if config is not None else {}

    n_samples = result_meta.get("n_samples")
    if n_samples is None and frame is not None:
        n_samples = int(len(frame))
    if n_samples is None:
        raise ValueError(
            "build_discovered_dag_payload: n_samples cannot be determined "
            "(no runner metadata and no frame)"
        )

    feature_names = result_meta.get("node_names")
    if feature_names is None and frame is not None:
        feature_names = [str(c) for c in frame.columns]
    if feature_names is None and discovery_result.ensemble_dag is not None:
        feature_names = sorted(str(n) for n in discovery_result.ensemble_dag.nodes())
    feature_names = [str(name) for name in (feature_names or [])]

    edges = list(discovery_result.edges or [])
    adjacency = None
    if discovery_result.ensemble_dag is not None and feature_names:
        adjacency = _matrix_or_none(discovery_result.to_adjacency_matrix(node_order=feature_names))

    session_uuid = _as_uuid_str(session_id)

    shipped_dag = {
        "nodes": list(causal_graph.get("nodes") or []),
        "edges": [list(edge) for edge in (causal_graph.get("edges") or [])],
        "treatment_nodes": list(causal_graph.get("treatment_nodes") or []),
        "outcome_nodes": list(causal_graph.get("outcome_nodes") or []),
        "edge_provenance": list(causal_graph.get("edge_provenance") or []),
        "adjustment_sets": [list(s) for s in (causal_graph.get("adjustment_sets") or [])],
        "augmented_edges": [list(edge) for edge in (causal_graph.get("augmented_edges") or [])],
        "discovery_dag_overridden": bool(causal_graph.get("discovery_dag_overridden", False)),
        "confidence": causal_graph.get("confidence"),
    }

    metadata: Dict[str, Any] = {
        "shipped_dag": shipped_dag,
        "gate_evaluation": dict(gate_evaluation),
        "discovery": result_meta,
        "discovery_latency_ms": discovery_latency_ms,
        "success": bool(discovery_result.success),
        "algorithm_agreement": discovery_result.algorithm_agreement,
    }
    if session_id and session_uuid is None:
        metadata["session_id_raw"] = str(session_id)

    algorithm_runs = [
        {
            "algorithm": _enum_value(run.algorithm),
            "runtime_seconds": run.runtime_seconds,
            "converged": bool(run.converged),
            "n_edges": len(run.edge_list or []),
            "edge_list": [list(edge) for edge in (run.edge_list or [])],
            "adjacency_matrix": _matrix_or_none(run.adjacency_matrix),
            "score": run.score,
            "parameters": config_dict,
            "metadata": dict(run.metadata or {}),
        }
        for run in (discovery_result.algorithm_results or [])
    ]

    edge_rows = [
        {
            "source_node": edge.source,
            "target_node": edge.target,
            "edge_type": _enum_value(edge.edge_type),
            "confidence": edge.confidence,
            "algorithm_votes": edge.algorithm_votes,
            "algorithms": list(edge.algorithms or []),
            "metadata": {"bootstrap_stability": edge.bootstrap_stability},
        }
        for edge in edges
    ]

    payload: Dict[str, Any] = {
        "session_id": session_uuid,
        "query_id": query_id,
        "dag_version_hash": str(dag_version_hash),
        "treatment_variable": treatment,
        "outcome_variable": outcome,
        "discovery_timestamp": discovery_result.created_at.isoformat(),
        "n_samples": int(n_samples),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "config": config_dict,
        "algorithms_used": [_enum_value(a) for a in (config.algorithms if config else [])],
        "ensemble_threshold": config.ensemble_threshold if config else None,
        "alpha": config.alpha if config else None,
        "n_edges": len(edges),
        "n_nodes": discovery_result.n_nodes,
        "edge_list": [edge.to_dict() for edge in edges],
        "confidence_scores": {f"{edge.source}->{edge.target}": edge.confidence for edge in edges},
        "adjacency_matrix": adjacency,
        "gate_decision": _enum_value(gate_evaluation.get("decision")),
        "gate_confidence": gate_evaluation.get("confidence"),
        "gate_reasons": list(gate_evaluation.get("reasons") or []),
        "total_runtime_seconds": result_meta.get("total_runtime_seconds"),
        "metadata": metadata,
        "is_synthetic": bool(is_synthetic),
        "algorithm_runs": algorithm_runs,
        "edges": edge_rows,
    }
    return cast(Dict[str, Any], _to_plain_json(payload))


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class DiscoveredDagRepository(BaseRepository[Dict[str, Any]]):
    """Writer + provenance-aware readers for ``public.discovered_dags``.

    Requires an ASYNC supabase client (``get_async_supabase_client``):
    :class:`BaseRepository` awaits ``execute()``, and so does :meth:`record`.
    """

    table_name = "discovered_dags"
    model_class = None
    HAS_PROVENANCE = True
    RPC_NAME = "record_discovered_dag"

    async def record(self, payload: Mapping[str, Any]) -> str:
        """Persist one discovery run atomically; return the new ``dag_id``.

        Raises :class:`DiscoveredDagPersistError` on every failure path:
        no client configured, a payload that does not state ``is_synthetic``,
        an RPC/transport error (chained as ``__cause__``), a malformed
        receipt, or a receipt whose row counts disagree with the payload.
        """
        if payload.get("is_synthetic") is None:
            raise DiscoveredDagPersistError(
                f"{self.RPC_NAME}: payload does not state is_synthetic (ADR-017 provenance)"
            )
        if not self.client:
            raise DiscoveredDagPersistError(
                f"{self.RPC_NAME}: no Supabase client configured — nothing was persisted"
            )
        try:
            result = await self.client.rpc(self.RPC_NAME, {"p_payload": dict(payload)}).execute()
        except Exception as exc:
            raise DiscoveredDagPersistError(f"{self.RPC_NAME} failed: {exc}") from exc

        receipt = getattr(result, "data", None)
        if not isinstance(receipt, dict) or not receipt.get("dag_id"):
            raise DiscoveredDagPersistError(
                f"{self.RPC_NAME} returned no dag_id (receipt={receipt!r})"
            )
        dag_id = str(receipt["dag_id"])
        expected_runs = len(payload.get("algorithm_runs") or [])
        expected_edges = len(payload.get("edges") or [])
        got_runs = receipt.get("n_algorithm_runs")
        got_edges = receipt.get("n_edges")
        if got_runs != expected_runs or got_edges != expected_edges:
            raise DiscoveredDagPersistError(
                f"{self.RPC_NAME} receipt for dag {dag_id} disagrees with the payload: "
                f"n_algorithm_runs {got_runs}/{expected_runs}, n_edges {got_edges}/{expected_edges}"
            )
        return dag_id

    async def find_by_dag_version_hash(
        self,
        dag_version_hash: str,
        *,
        include_synthetic: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Rows for a shipped-DAG hash (the expert-review key), newest first."""
        if not self.client:
            return []
        query = (
            self.client.table(self.table_name).select("*").eq("dag_version_hash", dag_version_hash)
        )
        query = apply_provenance_filter(query, include_synthetic=include_synthetic)
        result = await query.order("created_at", desc=True).limit(limit).execute()
        return parse_supabase_rows(result.data or [])

    async def find_by_query_id(
        self,
        query_id: str,
        *,
        include_synthetic: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Rows for an agent run id (agent-analyze analysis_id / query_id)."""
        if not self.client:
            return []
        query = self.client.table(self.table_name).select("*").eq("query_id", query_id)
        query = apply_provenance_filter(query, include_synthetic=include_synthetic)
        result = await query.order("created_at", desc=True).limit(limit).execute()
        return parse_supabase_rows(result.data or [])
