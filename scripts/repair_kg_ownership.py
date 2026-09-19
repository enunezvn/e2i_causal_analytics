#!/usr/bin/env python3
"""One-off repair of knowledge-graph damage written before #2174 / #2175.

A node in ``e2i_causal`` is *curated* when it has no ``agent`` property
(``src/tasks/graph_reseed_tasks.py``). The fixed writers keep that convention
for new writes; this script repairs what the old writers already left behind:

1. **Empty target Variable** (#2175) — ``(:Variable {id: 'var:'})``, the
   nameless node every untargeted experiment was linked to. Deleted with its
   edges; the Experiments themselves are kept.
2. **Unowned ProblemType** (#2174) — agent output written without ``agent``,
   so it counted as curated. Stamped ``agent = 'scope_definer'`` (the only
   writer of ProblemType).
3. **Stamped synced Variable** (#2174) — a Variable on a validated causal-path
   sync edge (``CAUSES`` with ``validation_status`` and no ``agent``) that an old
   agent hook stamped with ``agent``. The ``agent`` property is removed; the
   sync's topology role is restored by the next
   ``sync_causal_paths_to_falkordb.py --execute``.

The duplicate Brand from #2176 is handled by ``scripts/dedup_falkordb_brands.py``.

DRY-RUN BY DEFAULT — prints the counts. Pass ``--execute`` to write. Idempotent.

Env: FALKORDB_URL (preferred) or FALKORDB_HOST / FALKORDB_PORT /
FALKORDB_PASSWORD, and FALKORDB_GRAPH_NAME (default ``e2i_causal``).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict
from urllib.parse import urlparse

_EMPTY_TARGET = "MATCH (v:Variable) WHERE v.id = 'var:'"
_UNOWNED_PROBLEM_TYPE = "MATCH (p:ProblemType) WHERE p.agent IS NULL"
_STAMPED_SYNCED = (
    "MATCH (v:Variable)-[r:CAUSES]-() "
    "WHERE v.agent IS NOT NULL AND r.agent IS NULL AND r.validation_status IS NOT NULL"
)


def _count(graph: Any, match: str, var: str) -> int:
    rows = graph.query(f"{match} RETURN count(DISTINCT {var})").result_set
    return int(rows[0][0]) if rows else 0


def repair(graph: Any, execute: bool = False) -> Dict[str, int]:
    """Return the number of damaged nodes found; fix them when ``execute``."""
    report = {
        "empty_target_variables": _count(graph, _EMPTY_TARGET, "v"),
        "unowned_problem_types": _count(graph, _UNOWNED_PROBLEM_TYPE, "p"),
        "stamped_synced_variables": _count(graph, _STAMPED_SYNCED, "v"),
    }
    if execute:
        graph.query(f"{_EMPTY_TARGET} DETACH DELETE v")
        graph.query(f"{_UNOWNED_PROBLEM_TYPE} SET p.agent = 'scope_definer'")
        graph.query(f"{_STAMPED_SYNCED} WITH DISTINCT v SET v.agent = NULL")
    return report


def _connect() -> Any:
    from falkordb import FalkorDB

    url = os.environ.get("FALKORDB_URL")
    if url:
        parsed = urlparse(url)
        host, port, password = parsed.hostname or "localhost", parsed.port or 6379, parsed.password
    else:
        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6379"))
        password = os.environ.get("FALKORDB_PASSWORD") or None
    graph_name = os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")
    return FalkorDB(host=host, port=port, password=password).select_graph(graph_name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="Write the repair (default: dry-run)")
    args = ap.parse_args()

    graph = _connect()
    report = repair(graph, execute=args.execute)
    for key, value in report.items():
        print(f"{key}: {value}")
    if args.execute:
        after = repair(graph, execute=False)
        print("EXECUTED — remaining after repair:", after)
        return 0 if not any(after.values()) else 1
    print("DRY-RUN — no writes. Re-run with --execute to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
