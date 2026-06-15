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
SUPABASE_SERVICE_KEY, FALKORDB_HOST, FALKORDB_PORT, FALKORDB_PASSWORD,
FALKORDB_GRAPH_NAME (default ``e2i_causal``).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

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

    distinct_nodes = {n for _, seq in chains for n in seq}
    edge_count = sum(len(seq) - 1 for _, seq in chains)
    print(f"causal_paths (validated): {len(rows)}")
    print(f"chains with >=2 nodes:    {len(chains)}")
    print(f"distinct variable nodes:  {len(distinct_nodes)}")
    print(f"CAUSES edges to MERGE:     {edge_count}")

    if not args.execute:
        print("\nDRY-RUN — no writes. Re-run with --execute to apply.")
        return 0

    from falkordb import FalkorDB  # imported lazily so dry-run needs no driver

    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6379"))
    password = os.environ.get("FALKORDB_PASSWORD") or None
    graph_name = os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")
    db = FalkorDB(host=host, port=port, password=password)
    g = db.select_graph(graph_name)

    written = 0
    for row, seq in chains:
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
        for i in range(len(seq) - 1):
            params["sid"] = f"var:{seq[i]}"
            params["sname"] = seq[i]
            params["tid"] = f"var:{seq[i + 1]}"
            params["tname"] = seq[i + 1]
            params["is_terminal"] = i == len(seq) - 2
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

    print(f"\nEXECUTED — MERGEd {written} edges across {len(chains)} chains into '{graph_name}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
