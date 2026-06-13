#!/usr/bin/env python3
"""Issue #890: guarded cleanup of empty FalkorDB graph shells.

Live verification on 2026-06-12 found 11 graphs in FalkorDB, of which 10 are
empty shells:

* ``e2i_semantic`` / ``e2i_knowledge`` — bare keys (no labels, no indexes)
  created on READ by pre-remap config defaults (memory subsystem before
  #749 / commit 3dc677e5; RAG graph backend before commit 922491ef).
  FalkorDB creates a graph key on any GRAPH.QUERY, including a read.
* 8 UUID-named graphs — fully-indexed but node-less Graphiti schemas
  (Entity/Episodic/Community/Saga labels). graphiti-core's FalkorDriver
  maps group_id -> database name and builds indices in its constructor, so
  every session UUID passed as group_id created its own empty graph (fixed
  in src/memory/graphiti_service.py, same PR as this script).

This script deletes ONLY those shells, with belt-and-braces guards:

* DRY-RUN BY DEFAULT — pass ``--execute`` to actually delete.
* Hardcoded allowlist — only the 10 shells named in #890 can ever be
  dropped; anything else (including any future graph) is ignored.
* Live emptiness re-check — a graph is deleted only if it has 0 nodes at
  execution time. A shell that gained data since #890 is skipped loudly.
* ``e2i_causal`` (the populated production graph) is not on the allowlist
  and is therefore structurally untouchable.

Usage:
    # Dry run (default): report what would be deleted
    python scripts/cleanup_falkordb_shells.py

    # Actually delete (batch-phase only, with user sign-off)
    python scripts/cleanup_falkordb_shells.py --execute

Environment:
    FALKORDB_URL        redis://:<password>@<host>:<port>/0  (preferred), or
    FALKORDB_HOST       default localhost
    FALKORDB_PORT       default 6381 (host-side port of e2i_falkordb_dev)
    FALKORDB_PASSWORD   required unless embedded in FALKORDB_URL
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cleanup-falkordb-shells")

# The 10 empty shells verified in issue #890 (2026-06-12). e2i_causal is
# deliberately NOT listed — this allowlist is the only set of graphs this
# script can ever touch.
SHELL_GRAPH_ALLOWLIST: frozenset[str] = frozenset(
    {
        "e2i_semantic",
        "e2i_knowledge",
        "f789fbc0-9779-4ae2-9fb6-4d962f7f3da1",
        "9aed4469-faea-4ad5-8aa8-9061c6546b83",
        "9e294651-9507-4a68-aea8-fbcbb9c5689c",
        "634af254-8b0b-45f5-88d0-83e78a9c7a63",
        "378a69b4-e00d-44a0-b3ea-5fd978dd2864",
        "fb027e43-fd41-4bc8-b386-3e5a50b412ca",
        "1e73cce5-c8e7-4cc2-92f4-aaba710fb6e7",
        "ba42bd26-d4dc-4cf6-82bc-159cc9542894",
    }
)


@dataclass
class CleanupResult:
    """Outcome of a cleanup pass."""

    would_delete: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped_non_empty: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)


def _count_nodes(client: Any, graph_name: str) -> int:
    """Count nodes in an EXISTING graph.

    Only call for graphs already present in ``list_graphs()`` — querying a
    nonexistent name would itself create a shell (the #890 mechanism).
    """
    graph = client.select_graph(graph_name)
    result = graph.query("MATCH (n) RETURN count(n)")
    return int(result.result_set[0][0])


def cleanup_shells(client: Any, execute: bool = False) -> CleanupResult:
    """Delete (or report) empty allowlisted shell graphs.

    Args:
        client: a FalkorDB client exposing ``list_graphs()`` and
            ``select_graph(name)`` (graphs expose ``query`` and ``delete``).
        execute: when False (default), report only — delete nothing.

    Returns:
        CleanupResult with per-graph dispositions.
    """
    result = CleanupResult()
    existing = set(client.list_graphs())

    for name in sorted(SHELL_GRAPH_ALLOWLIST):
        if name not in existing:
            result.absent.append(name)
            continue

        node_count = _count_nodes(client, name)
        if node_count > 0:
            logger.warning(
                "REFUSING to delete %s: %d nodes present (was an empty shell in #890)",
                name,
                node_count,
            )
            result.skipped_non_empty.append(name)
            continue

        if execute:
            client.select_graph(name).delete()
            logger.info("Deleted empty shell graph: %s", name)
            result.deleted.append(name)
        else:
            logger.info("[dry-run] Would delete empty shell graph: %s", name)
            result.would_delete.append(name)

    return result


def _connect() -> Any:
    """Create a FalkorDB client from FALKORDB_URL or discrete env vars."""
    from falkordb import FalkorDB

    falkordb_url = os.environ.get("FALKORDB_URL")
    if falkordb_url:
        from urllib.parse import urlparse

        parsed = urlparse(falkordb_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        password = parsed.password
    else:
        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6381"))
        password = os.environ.get("FALKORDB_PASSWORD")

    if not password:
        raise SystemExit("FALKORDB_PASSWORD (or a password in FALKORDB_URL) is required")

    logger.info("Connecting to FalkorDB at %s:%d", host, port)
    return FalkorDB(host=host, port=port, password=password)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the shells (default: dry-run report only)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = _connect()
    result = cleanup_shells(client, execute=args.execute)

    print()
    print("=== FalkorDB shell cleanup (issue #890) ===")
    print(f"Mode:               {'EXECUTE' if args.execute else 'dry-run'}")
    print(f"Deleted:            {result.deleted or '-'}")
    print(f"Would delete:       {result.would_delete or '-'}")
    print(f"Skipped non-empty:  {result.skipped_non_empty or '-'}")
    print(f"Already absent:     {sorted(result.absent) or '-'}")
    if result.skipped_non_empty:
        print(
            "\nWARNING: some allowlisted shells now contain data and were NOT "
            "deleted. Investigate before re-running."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
