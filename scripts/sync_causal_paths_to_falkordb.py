#!/usr/bin/env python3
"""Sync validated ``causal_paths`` (Supabase SSOT) into the FalkorDB knowledge graph.

Closes the long-standing gap where the discovered causal chains lived only in the
``causal_paths`` table and never appeared in the FalkorDB graph that the
Knowledge-Graph page and graph-stats read. The dashboard "Primary Causal Value
Chains" section reads ``causal_paths`` directly (see
``GET /api/causal/value-chains``) and does NOT depend on this sync — this only
brings the graph surface into line with the SSOT.

Each chain becomes a ``(:Variable)-[:CAUSES]->(:Variable)`` path. ``confidence``
is set on EVERY edge (so the path clears the graph's ``confidence >= min`` gate),
the chain-level ``ate_estimate`` rides the terminal edge, and
brand/region/method/validation_status/confirmation_count/discovery_date are
stamped on each edge. All writes are idempotent ``MERGE``s.

Usage::

    # dry-run — counts only, no writes (DEFAULT, safe):
    python scripts/sync_causal_paths_to_falkordb.py

    # apply to FalkorDB:
    python scripts/sync_causal_paths_to_falkordb.py --execute

Env (read from the process; use ``set -a; source .env``): SUPABASE_URL,
SUPABASE_SERVICE_KEY, FALKORDB_URL (preferred, e.g.
``redis://:pw@falkordb:6379/0``) or the discrete FALKORDB_HOST / FALKORDB_PORT /
FALKORDB_PASSWORD, FALKORDB_GRAPH_NAME (default ``e2i_causal``).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx


def _node_sequence(row: Dict[str, Any]) -> List[str]:
    chain = row.get("causal_chain")
    if isinstance(chain, dict):
        nodes = chain.get("nodes")
        if isinstance(nodes, list) and len(nodes) >= 2 and all(isinstance(n, str) for n in nodes):
            return list(nodes)
    seq: List[str] = []
    if isinstance(row.get("start_node"), str) and row["start_node"]:
        seq.append(row["start_node"])
    inter = row.get("intermediate_nodes")
    if isinstance(inter, list):
        seq.extend(n for n in inter if isinstance(n, str) and n)
    if isinstance(row.get("end_node"), str) and row["end_node"]:
        seq.append(row["end_node"])
    return seq


def _mediation_edges(start: str, mediators: List[str], end: str) -> List[Tuple[str, str, bool]]:
    """Edges for one chain as PARALLEL mediation, returned as ``(cause, effect,
    is_terminal)``.

    A chain ``treatment_arm -> [m1, m2, ...] -> outcome`` does NOT mean the
    mediators form a serial chain among themselves — they are parallel mediators
    that each independently sit between treatment and outcome. The previous
    consecutive-pair encoding (``start->m1->m2->outcome``) invented spurious
    ``m1->m2`` edges, and because the generator orders mediators RANDOMLY per
    row, those mediator->mediator edges accumulated in BOTH directions —
    reciprocal cycles (``adherence<->prior_therapy`` etc.) that turned the
    Knowledge-Graph variable layer into a hairball.

    Emitting ``start -> mediator`` (non-terminal) and ``mediator -> outcome``
    (terminal, so the chain effect rides it) for each mediator removes every
    mediator->mediator edge while preserving the real mediation structure. With
    no mediators it degrades to a single direct ``start -> outcome`` terminal
    edge. Self/degenerate pairs (a mediator equal to start or end) are dropped.
    """
    edges: List[Tuple[str, str, bool]] = []
    real_mediators = [m for m in mediators if m and m != start and m != end]
    if not real_mediators:
        if start != end:
            edges.append((start, end, True))
        return edges
    for m in real_mediators:
        edges.append((start, m, False))
        edges.append((m, end, True))
    return edges


def _variable_roles(edges: List[Tuple[str, str, bool]]) -> Dict[str, str]:
    """Per-variable role from position across ALL chain edges (SSOT topology).

    ``driver``  — only ever a cause (pure source: interventions and exogenous
    drivers such as ``treatment_arm`` or ``competitor_activity``);
    ``outcome`` — only ever an effect (pure sink);
    ``mediator`` — both (transmits effects somewhere in the DAG).

    The AI-Insights graph colors nodes by this property. It is deliberately
    topology-derived — no hand-curated lever ontology — so it can never
    contradict the validated causal_paths rows it is computed from.
    """
    causes = {c for c, _, _ in edges}
    effects = {e for _, e, _ in edges}
    roles: Dict[str, str] = {}
    for name in causes | effects:
        if name in causes and name not in effects:
            roles[name] = "driver"
        elif name in effects and name not in causes:
            roles[name] = "outcome"
        else:
            roles[name] = "mediator"
    return roles


# Terminal patient-journey outcomes -> the commercial KPI they drive, matched by
# KPI *name* (live KPI nodes carry no ``id``). These bridge edges connect the
# variable layer to the KPI layer so the gold standard renders as ONE connected
# graph rather than two disjoint islands. Kept brand/region-agnostic (a single
# shared edge each), so the page's parallel-edge dedupe leaves them as one edge.
_VARIABLE_KPI_BRIDGE: Dict[str, str] = {
    "treatment_initiated": "NRx",
    "persistent_180d": "Patient_Retention",
    "discontinued_180d": "Patient_Retention",
    # Commercial grain (2026-07-07): bridge only to KPI nodes verified live in
    # the graph (MATCH (k:KPI) — TRx/NRx/Market_Share exist; NBRx/ROI do not,
    # so nbrx_volume/roi/intent_to_prescribe stay unbridged rather than
    # pointing at nodes that would silently never match).
    "trx_volume": "TRx",
    "nrx_volume": "NRx",
    "trx_market_share": "Market_Share",
}


def fetch_validated_paths() -> List[Dict[str, Any]]:
    """Page through every validated causal_paths row via the Supabase REST API."""
    url = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/causal_paths"
    key = os.environ["SUPABASE_SERVICE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    select = (
        "path_id,start_node,end_node,intermediate_nodes,causal_chain,"
        "causal_effect_size,confidence_level,method_used,validation_status,"
        "confirmation_count,discovery_date,brand,region"
    )
    rows: List[Dict[str, Any]] = []
    offset, page = 0, 1000
    with httpx.Client(timeout=30) as client:
        while True:
            resp = client.get(
                url,
                headers={
                    **headers,
                    "Range-Unit": "items",
                    "Range": f"{offset}-{offset + page - 1}",
                },
                params={
                    "select": select,
                    "validation_status": "eq.validated",
                    "order": "causal_effect_size.desc",
                },
            )
            resp.raise_for_status()
            batch = resp.json()
            rows.extend(batch)
            if len(batch) < page:
                break
            offset += page
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="Write to FalkorDB (default: dry-run)")
    args = ap.parse_args()

    rows = fetch_validated_paths()
    chains = [(r, _node_sequence(r)) for r in rows]
    chains = [(r, seq) for r, seq in chains if len(seq) >= 2]

    # Parallel-mediation edges per chain (start->mediator, mediator->outcome);
    # NEVER mediator->mediator (see _mediation_edges for the de-cycle rationale).
    chain_edges = [(row, _mediation_edges(seq[0], seq[1:-1], seq[-1])) for row, seq in chains]
    distinct_nodes = {n for _, seq in chains for n in seq}
    edge_count = sum(len(edges) for _, edges in chain_edges)
    roles = _variable_roles([e for _, edges in chain_edges for e in edges])
    role_counts = {
        r: sum(1 for v in roles.values() if v == r) for r in ("driver", "mediator", "outcome")
    }
    print(f"causal_paths (validated): {len(rows)}")
    print(f"chains with >=2 nodes:    {len(chains)}")
    print(f"distinct variable nodes:  {len(distinct_nodes)}")
    print(f"CAUSES edges to MERGE:     {edge_count} (parallel mediation, no mediator->mediator)")
    print(
        f"variable->KPI bridges:     {len(_VARIABLE_KPI_BRIDGE)} (matched by KPI name if present)"
    )
    print(
        f"variable roles to stamp:   {role_counts} (topology-derived, colors the AI-Insights graph)"
    )

    if not args.execute:
        print("\nDRY-RUN — no writes. Re-run with --execute to apply.")
        return 0

    from falkordb import FalkorDB  # imported lazily so dry-run needs no driver

    # Prefer FALKORDB_URL — the single connection var the platform/compose actually
    # sets (e.g. redis://:pw@falkordb:6379/0). Without this the discrete-var
    # fallback defaults to localhost:6379 and refuses the connection inside the
    # container. Mirrors the app client (falkordb_client.py::_parse_falkordb_config)
    # and sibling scripts (cleanup_falkordb_shells.py::_connect).
    falkordb_url = os.environ.get("FALKORDB_URL")
    if falkordb_url:
        parsed = urlparse(falkordb_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        password = parsed.password
    else:
        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6379"))
        password = os.environ.get("FALKORDB_PASSWORD") or None
    graph_name = os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")
    db = FalkorDB(host=host, port=port, password=password)
    g = db.select_graph(graph_name)

    written = 0
    for row, edges in chain_edges:
        conf = row.get("confidence_level")
        eff = row.get("causal_effect_size")
        params = {
            "conf": float(conf) if conf is not None else None,
            "eff": float(eff) if eff is not None else None,
            "method": row.get("method_used"),
            "brand": row.get("brand"),
            "region": row.get("region"),
            "vstatus": row.get("validation_status"),
            "cc": row.get("confirmation_count"),
            "ddate": str(row.get("discovery_date")) if row.get("discovery_date") else None,
        }
        for src_name, tgt_name, is_terminal in edges:
            params["sid"] = f"var:{src_name}"
            params["sname"] = src_name
            params["tid"] = f"var:{tgt_name}"
            params["tname"] = tgt_name
            params["is_terminal"] = is_terminal
            g.query(
                """
                MERGE (a:Variable {id: $sid}) SET a.name = $sname
                MERGE (b:Variable {id: $tid}) SET b.name = $tname
                MERGE (a)-[r:CAUSES {brand: $brand, region: $region}]->(b)
                SET r.confidence = $conf,
                    r.method = $method,
                    r.validation_status = $vstatus,
                    r.confirmation_count = $cc,
                    r.discovery_date = $ddate,
                    r.ate_estimate = CASE WHEN $is_terminal THEN $eff ELSE r.ate_estimate END
                """,
                params=params,
            )
            written += 1

    # Stamp the topology-derived role on every variable node just written —
    # the AI-Insights "Active Causal Chains" graph colors nodes by it. MATCH
    # (never MERGE) so a role can only land on nodes this sync owns; SET is
    # idempotent and self-corrects when new chains change a node's position.
    stamped = 0
    for var_name, role in roles.items():
        res = g.query(
            "MATCH (v:Variable {id: $vid}) SET v.role = $role RETURN count(v)",
            params={"vid": f"var:{var_name}", "role": role},
        )
        touched = res.result_set[0][0] if getattr(res, "result_set", None) else 0
        stamped += int(touched or 0)

    # Bridge the terminal variable outcomes into the KPI layer so the variable
    # graph and the KPI graph are one connected component. MATCH both endpoints
    # (never CREATE a bare KPI or orphan variable); MERGE a single, idempotent,
    # brand/region-agnostic CAUSES edge tagged is_bridge.
    bridged = 0
    for outcome, kpi_name in _VARIABLE_KPI_BRIDGE.items():
        res = g.query(
            """
            MATCH (v:Variable {id: $vid})
            MATCH (k:KPI {name: $kpi})
            MERGE (v)-[r:CAUSES]->(k)
            SET r.is_bridge = true,
                r.confidence = $conf,
                r.validation_status = 'validated'
            RETURN count(r)
            """,
            params={"vid": f"var:{outcome}", "kpi": kpi_name, "conf": 0.7},
        )
        touched = res.result_set[0][0] if getattr(res, "result_set", None) else 0
        bridged += int(touched or 0)

    print(
        f"\nEXECUTED — MERGEd {written} causal edges across {len(chains)} chains, "
        f"stamped roles on {stamped} variable node(s) "
        f"+ {bridged} variable->KPI bridge(s) into '{graph_name}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
