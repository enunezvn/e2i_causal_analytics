#!/usr/bin/env python3
"""Issue #169: persist the converter's HCP influence graph into FalkorDB.

Follow-up to PR #168 (issue #156 item 2). PR #168 builds an in-memory
``networkx.Graph`` from per-cohort shared-patient cliques and scores it
to populate ``influence_network_size`` + ``peer_influence_score`` on the
HCP parquet artifact. The semantic-memory layer at
``src/memory/semantic_memory.py`` exposes ``get_hcp_influence_network``
+ ``count_hcp_influence_network`` Cypher helpers that read from an
EXISTING FalkorDB graph that nobody populates. This script populates
that backend.

Operational decoupling: the converter does NOT take a hard FalkorDB
dependency — that's intentional. This script reads the converter's
input parquet drop (medication.parquet + procedure.parquet + the
already-built patient_journeys.parquet for index dates and cohort
membership) and rebuilds the influence graph via the shared
``build_hcp_influence_graph`` helper so the FalkorDB ingest is
byte-for-byte parity with the Parquet artifact.

Cypher schema (aligned with ``semantic_memory.py``):

* Node label: ``HCP``
* Node identity: ``id`` property (matches the existing
  ``MATCH (h:HCP {id: $hcp_id})`` query); also populated with
  ``npi`` (identical to ``id`` for influence-graph nodes), ``cohort_id``,
  ``ingested_at``.
* Edge type: ``SHARED_PATIENTS``
* Edge properties: ``weight`` (int, count of shared patients),
  ``cohort_id``, ``ingested_at``.

Cohort tagging keeps CSU and Optum graphs queryable independently —
see :func:`src.memory.semantic_memory.FalkorDBSemanticMemory.get_hcp_influence_network`
which accepts an optional ``cohort_id`` filter after issue #169.

Idempotency: every write uses ``MERGE``, so a rerun is a no-op. The
``--replace`` flag explicitly wipes the prior rows tagged with
``cohort_id`` before reloading — it does NOT wipe the whole graph
(other cohorts are safe).

Usage::

    python scripts/persist_hcp_influence_to_falkordb.py \\
        --parquet-dir data/rwd/Optum_Parquet \\
        --cohort-dir data/rwd/optum/initiation \\
        --cohort-id optum_initiation_v3

    # Replace the cohort_id rows in the graph before reload:
    python scripts/persist_hcp_influence_to_falkordb.py ... --replace

    # Dry run (build graph + log counts, no FalkorDB writes):
    python scripts/persist_hcp_influence_to_falkordb.py ... --dry-run

The exit code is 0 on success, 1 on missing inputs / FalkorDB
unreachable, 2 on CLI misuse.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root on path so this script runs both via `python -m scripts...`
# and direct `python scripts/...` invocation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from scripts.convert_optum_rwd import (  # noqa: E402
    LOOKBACK_DAYS,
    PEER_INFLUENCE_SCALE,
    build_hcp_influence_graph,
    score_hcp_influence_graph,
)

logger = logging.getLogger("persist_hcp_influence")

# FalkorDB MERGE chunk size — Cypher statements per round trip. Tuned
# empirically; FalkorDB accepts large UNWIND lists but the JSON
# serialisation cost climbs > ~5000.
DEFAULT_BATCH_SIZE = 1000


# --------------------------------------------------------------------------- #
# Graph build from raw parquet                                                #
# --------------------------------------------------------------------------- #


def load_cohort_inputs(
    parquet_dir: Path,
    cohort_dir: Path,
) -> tuple[set[int], dict[int, pd.Timestamp], pd.DataFrame, pd.DataFrame]:
    """Load the four inputs needed to rebuild the influence graph.

    Args:
        parquet_dir: directory holding the raw Optum drop. Must contain
            ``medication.parquet`` and ``procedure.parquet``.
        cohort_dir: directory holding the converter's per-cohort output.
            Must contain ``e2i_ml_v3_patient_journeys.parquet`` with at
            least ``patient_id`` (or ``patid``) + ``index_date`` columns.

    Returns:
        ``(kept_patids, idx_by_patid, med, proc)`` — the exact arguments
        accepted by :func:`build_hcp_influence_graph`.

    Raises:
        FileNotFoundError: if any required parquet is missing.
    """
    med_path = parquet_dir / "medication.parquet"
    proc_path = parquet_dir / "procedure.parquet"
    journeys_path = cohort_dir / "e2i_ml_v3_patient_journeys.parquet"

    for p in (med_path, proc_path, journeys_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    med = pd.read_parquet(med_path)
    proc = pd.read_parquet(proc_path)
    journeys = pd.read_parquet(journeys_path)

    # The converter emits `patient_id`; the raw Optum drop keys on
    # `patid`. Accept either to keep this script useful for synthetic
    # fixtures that follow a single schema.
    if "patient_id" in journeys.columns:
        pid_col = "patient_id"
    elif "patid" in journeys.columns:
        pid_col = "patid"
    else:
        raise ValueError(
            f"{journeys_path} has neither 'patient_id' nor 'patid' column — "
            "cannot derive cohort membership"
        )

    if "index_date" not in journeys.columns:
        raise ValueError(
            f"{journeys_path} missing 'index_date' column — required for the "
            "per-patient temporal gate"
        )

    kept_patids: set[int] = set()
    idx_by_patid: dict[int, pd.Timestamp] = {}
    for _, r in journeys.iterrows():
        pid = r.get(pid_col)
        if pd.isna(pid):
            continue
        pid_int = int(pid)
        kept_patids.add(pid_int)
        idx = r.get("index_date")
        if not pd.isna(idx):
            idx_by_patid[pid_int] = pd.Timestamp(idx)

    return kept_patids, idx_by_patid, med, proc


# --------------------------------------------------------------------------- #
# FalkorDB ingest                                                             #
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    """ISO-8601 UTC timestamp (deterministic format used as edge / node prop)."""
    return datetime.now(timezone.utc).isoformat()


def delete_cohort_rows(graph: Any, cohort_id: str) -> tuple[int, int]:
    """Delete every HCP node + SHARED_PATIENTS edge tagged with ``cohort_id``.

    Used by ``--replace`` to wipe-and-reload without touching other cohorts.

    Returns:
        ``(nodes_deleted, relationships_deleted)``. Best-effort counters —
        FalkorDB's ``QueryResult`` exposes ``nodes_deleted`` /
        ``relationships_deleted`` attributes; defaults to 0 if missing.
    """
    # DETACH DELETE on the nodes drops both the cohort's HCP rows AND
    # any edges incident to them, so we don't need a separate edge-delete
    # pass.
    result = graph.query(
        "MATCH (h:HCP {cohort_id: $cohort_id}) DETACH DELETE h",
        {"cohort_id": cohort_id},
    )
    nodes_deleted = int(getattr(result, "nodes_deleted", 0) or 0)
    rels_deleted = int(getattr(result, "relationships_deleted", 0) or 0)
    logger.info(
        "Deleted cohort '%s' from FalkorDB: %d nodes, %d edges",
        cohort_id,
        nodes_deleted,
        rels_deleted,
    )
    return nodes_deleted, rels_deleted


def persist_graph_to_falkordb(
    graph: Any,
    falkordb_graph: Any,
    cohort_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ingested_at: str | None = None,
) -> tuple[int, int]:
    """Persist a networkx graph into FalkorDB as ``HCP`` + ``SHARED_PATIENTS``.

    Args:
        graph: ``networkx.Graph`` from :func:`build_hcp_influence_graph`.
        falkordb_graph: a FalkorDB ``Graph`` handle (i.e.
            ``client.select_graph(get_config().semantic.graph_name)`` — the
            graph the semantic-memory readers use, #890). Tests inject a mock.
        cohort_id: e.g. ``"optum_initiation_v3"``. Set on every node + edge
            so cohorts stay queryable independently.
        batch_size: number of rows per ``UNWIND`` batch. Defaults to
            :data:`DEFAULT_BATCH_SIZE`.
        ingested_at: ISO-8601 timestamp string. Defaults to ``now(UTC)``.

    Returns:
        ``(nodes_written, edges_written)`` — counts of the input graph
        (NOT the FalkorDB-reported counters, which are post-MERGE so a
        rerun would report 0 on idempotent calls).

    Notes:
        * Uses ``MERGE`` (idempotent). Rerun is a no-op on identical
          inputs; on changed weights, the edge's ``weight`` property is
          overwritten on match (``ON MATCH SET``).
        * Edges with ``weight >= 1`` only. Per issue #169 spec
          ``weight >= 1`` is automatically true for any edge that
          appears in the networkx output (a clique of size n yields
          weight=1 per pair from a single patient and accumulates).
        * Both endpoints are tagged with the same ``cohort_id``;
          downstream cohort-scoped traversal predicates rely on this
          symmetry.
    """
    if graph is None:
        logger.warning("persist_graph_to_falkordb: graph is None — nothing to write")
        return 0, 0
    if ingested_at is None:
        ingested_at = _now_iso()

    nodes = list(graph.nodes())
    # Codex pass-2 MEDIUM: networkx.Graph.edges() iteration order on an
    # UNDIRECTED graph is implementation-defined and can swap (a, b) on
    # a rerun (different process / hash randomization). The FalkorDB
    # MERGE below is DIRECTED, so without canonicalisation a rerun
    # could create `(B)->(A)` alongside a prior `(A)->(B)` instead of
    # updating in place — breaking idempotency. Sort each endpoint pair
    # lexically so the persisted direction is deterministic across runs.
    # Aggregate by canonical key to defend against the (admittedly
    # impossible-for-nx.Graph) case where both orders appear in one
    # iteration.
    canonical_weights: dict[tuple[str, str], int] = {}
    for a, b, d in graph.edges(data=True):
        weight = int(d.get("weight", 0))
        if weight < 1:
            continue
        src, dst = sorted((str(a), str(b)))
        canonical_weights[(src, dst)] = weight
    edges = [(src, dst, w) for (src, dst), w in canonical_weights.items()]

    if not nodes:
        logger.info("persist_graph_to_falkordb: cohort '%s' has 0 HCPs", cohort_id)
        return 0, 0

    # Node MERGE in batches.
    node_query = """
    UNWIND $rows AS row
    MERGE (h:HCP {id: row.id, cohort_id: $cohort_id})
    ON CREATE SET h.npi = row.id, h.ingested_at = $ingested_at, h.e2i_entity_type = 'hcp'
    ON MATCH SET h.npi = row.id, h.ingested_at = $ingested_at, h.e2i_entity_type = 'hcp'
    """
    for i in range(0, len(nodes), batch_size):
        chunk = [{"id": str(n)} for n in nodes[i : i + batch_size]]
        falkordb_graph.query(
            node_query,
            {"rows": chunk, "cohort_id": cohort_id, "ingested_at": ingested_at},
        )

    # Edge MERGE in batches.
    if edges:
        edge_query = """
        UNWIND $rows AS row
        MATCH (a:HCP {id: row.src, cohort_id: $cohort_id})
        MATCH (b:HCP {id: row.dst, cohort_id: $cohort_id})
        MERGE (a)-[r:SHARED_PATIENTS {cohort_id: $cohort_id}]->(b)
        ON CREATE SET r.weight = row.weight, r.ingested_at = $ingested_at
        ON MATCH SET r.weight = row.weight, r.ingested_at = $ingested_at
        """
        for i in range(0, len(edges), batch_size):
            rows = [
                {"src": a, "dst": b, "weight": int(w)} for (a, b, w) in edges[i : i + batch_size]
            ]
            falkordb_graph.query(
                edge_query,
                {"rows": rows, "cohort_id": cohort_id, "ingested_at": ingested_at},
            )

    logger.info(
        "Persisted cohort '%s' to FalkorDB: %d HCP nodes, %d SHARED_PATIENTS edges",
        cohort_id,
        len(nodes),
        len(edges),
    )
    return len(nodes), len(edges)


# --------------------------------------------------------------------------- #
# Round-trip readback (used by parity test)                                   #
# --------------------------------------------------------------------------- #


def read_influence_from_falkordb(
    falkordb_graph: Any,
    cohort_id: str,
    scale: float = PEER_INFLUENCE_SCALE,
) -> tuple[dict[str, int], dict[str, float]]:
    """Re-derive ``influence_network_size`` + ``peer_influence_score`` from FalkorDB.

    Reads every node + edge tagged with ``cohort_id``, rebuilds a
    ``networkx.Graph``, and feeds it through :func:`score_hcp_influence_graph`
    so the readback path exercises the same scoring code as the converter
    — modulo float rounding the values must match the Parquet artifact.

    Used by the round-trip parity test required by issue #169 acceptance.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available — cannot rebuild for readback")
        return {}, {}

    nodes_result = falkordb_graph.query(
        "MATCH (h:HCP {cohort_id: $cohort_id}) RETURN h.id AS id",
        {"cohort_id": cohort_id},
    )
    # NOTE: persistence canonicalises endpoint order to `sorted((a, b))`
    # so every undirected pair is stored once. We still issue an
    # undirected match here (no arrow) so the readback is robust to any
    # future change in the persistence direction convention.
    #
    # Codex pass-3 MEDIUM: aggregate by canonical endpoints to dedupe
    # legacy duplicate rows (e.g., if both `(A)->(B)` and `(B)->(A)` were
    # persisted by a pre-canonicalisation run). `max(r.weight)` is the
    # conservative aggregation choice — for clean data both rows carry
    # the same weight so the result is unchanged; for divergent legacy
    # data the larger count of shared patients wins. `WITH` enforces
    # the grouping in the FalkorDB / openCypher dialect.
    edges_result = falkordb_graph.query(
        "MATCH (a:HCP {cohort_id: $cohort_id})-[r:SHARED_PATIENTS {cohort_id: $cohort_id}]-"
        "(b:HCP {cohort_id: $cohort_id}) "
        "WHERE a.id < b.id "
        "WITH a.id AS src, b.id AS dst, max(r.weight) AS weight "
        "RETURN src, dst, weight",
        {"cohort_id": cohort_id},
    )

    graph: Any = nx.Graph()
    for row in nodes_result.result_set or []:
        graph.add_node(str(row[0]))
    for row in edges_result.result_set or []:
        graph.add_edge(str(row[0]), str(row[1]), weight=int(row[2]))

    return score_hcp_influence_graph(graph, scale=scale)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Issue #169: persist the converter's HCP influence graph into FalkorDB"),
    )
    p.add_argument(
        "--parquet-dir",
        required=True,
        type=Path,
        help="Directory containing medication.parquet + procedure.parquet",
    )
    p.add_argument(
        "--cohort-dir",
        required=True,
        type=Path,
        help="Directory containing e2i_ml_v3_patient_journeys.parquet",
    )
    p.add_argument(
        "--cohort-id",
        required=True,
        help="Cohort tag stored on every node + edge (e.g. optum_initiation_v3)",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS,
        help=f"Per-patient temporal gate (default: {LOOKBACK_DAYS} — PR #168 contract)",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Wipe the cohort's prior nodes/edges before reload",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the graph and log counts but do NOT write to FalkorDB",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Cypher UNWIND batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _resolve_target_graph_name() -> str:
    """Resolve the FalkorDB graph the semantic-memory READERS use.

    #169's intent is that ``FalkorDBSemanticMemory.get_hcp_influence_network``
    / ``count_hcp_influence_network`` return the persisted data. Those readers
    resolve ``get_config().semantic.graph_name`` (deployed: ``e2i_causal``,
    see #749), so the writer must target the same graph. The previous
    hardcoded ``"e2i_semantic"`` wrote into an orphan graph nothing reads
    (issue #890).

    Raises if the memory config cannot be loaded — failing closed beats
    silently writing to the wrong graph.
    """
    from src.memory.services.config import get_config

    return str(get_config().semantic.graph_name)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    try:
        kept_patids, idx_by_patid, med, proc = load_cohort_inputs(args.parquet_dir, args.cohort_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input error: %s", exc)
        return 1

    logger.info(
        "Loaded %d kept patients (%d with index_date) from %s",
        len(kept_patids),
        len(idx_by_patid),
        args.cohort_dir,
    )

    graph = build_hcp_influence_graph(
        kept_patids=kept_patids,
        med=med,
        proc=proc,
        idx_by_patid=idx_by_patid,
        lookback_days=args.lookback_days,
    )
    if graph is None:
        logger.error("networkx not available — cannot proceed")
        return 1

    logger.info(
        "Built influence graph: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    if args.dry_run:
        logger.info("Dry run — skipping FalkorDB writes")
        return 0

    try:
        from src.memory.services.factories import get_falkordb_client
    except ImportError as exc:
        logger.error("Cannot import FalkorDB client factory: %s", exc)
        return 1

    try:
        client = get_falkordb_client()
        target_graph = _resolve_target_graph_name()
        logger.info("Persisting into semantic graph: %s", target_graph)
        falkordb_graph = client.select_graph(target_graph)
    except Exception as exc:  # noqa: BLE001 — surface any connection error
        logger.error("FalkorDB unreachable: %s", exc)
        return 1

    if args.replace:
        delete_cohort_rows(falkordb_graph, args.cohort_id)

    nodes_written, edges_written = persist_graph_to_falkordb(
        graph,
        falkordb_graph,
        cohort_id=args.cohort_id,
        batch_size=args.batch_size,
    )

    logger.info(
        "Done. Cohort='%s' nodes=%d edges=%d",
        args.cohort_id,
        nodes_written,
        edges_written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
