"""Beat-scheduled emptiness sentinel + self-heal reseed for the knowledge graph (#1761).

Why this exists
---------------
In #1758 a container recreation wiped the FalkorDB ``e2i_causal`` graph and
``/knowledge-graph`` rendered empty for four days, until a human noticed,
root-caused it, and manually re-ran the two seed scripts. #1759 removed the known
wipe vector (the data dir now lives on the ``/data`` volume), but an empty graph
can still recur — volume loss, host rebuild, RDB corruption, a future image quirk —
and nothing in the platform heals it. This task is that missing half.

What "empty" means here
-----------------------
Emptiness is measured on the CURATED core, not on the total node count:

    MATCH (n) WHERE n.agent IS NULL RETURN count(n)

Agent runtime writes (``causal_impact.store_causal_path`` and friends) stamp an
``agent`` property; curated seed/sync nodes carry none. That is exactly the
predicate ``semantic_memory.list_nodes(curated_only=True)`` appends, and
``KnowledgeGraph.tsx`` always sends ``curated_only: true`` — so this counts what
the page renders. It matters: after the #1758 wipe, agents repopulated runtime
nodes within hours, so a TOTAL count would have read non-empty for the whole
outage. (Live prod 2026-08-21: 100 total nodes, 89 curated, 11 agent-written.)

A failed probe is UNKNOWN, never zero. Reading a FalkorDB outage as "empty" would
fire a CREATE-based reseed on every tick and duplicate the graph the moment the
database came back.

Memory budget
-------------
This module is imported at worker boot (``src/tasks/__init__``) and the reseed runs
inside ``worker_light``, which sits under a 1.5 GiB cgroup limit (measured 1.03 GiB
resident, ~476 MiB headroom on 2026-08-21). So:

* nothing heavy is imported at module scope — ``falkordb`` and ``redis`` are
  imported lazily inside their helpers (~15 MiB each when they do load);
* the connection resolver is a local 12-line copy of
  ``src/api/dependencies/falkordb_client.py::_parse_falkordb_config`` rather than an
  import of it — measured, that import costs 92.5 MiB (tenacity + circuit breaker)
  for a function that needs none of it;
* the seed scripts run as SUBPROCESSES, never imports. Two reasons: the deployed
  image ships ``scripts/`` but a top-level ``import scripts.*`` in a task module
  crash-looped every worker in the 2026-05-26 incident, and a subprocess's RSS is
  reclaimed the moment it exits. #1761 shed ``seed_falkordb.py``'s
  ``from src.rag.config import FalkorDBConfig`` for the same budget: that one line
  cost 721.7 MiB of child RSS, which does not fit in the headroom above.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple
from urllib.parse import urlparse

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# The page's ``curated_only`` predicate — see the module docstring.
CURATED_COUNT_QUERY: Final[str] = "MATCH (n) WHERE n.agent IS NULL RETURN count(n) as count"

# Double-fire guard. ``seed_falkordb.py`` builds the structural layer with CREATE,
# so two concurrent reseeds produce a duplicated graph rather than an idempotent
# one. worker_light runs two replicas x concurrency 2, and the beat tick is
# delivered to exactly one of them — but a retry, a manual dispatch or an operator
# running the scripts by hand can overlap, so the lock is real work, not ceremony.
RESEED_LOCK_KEY: Final[str] = "e2i:graph:reseed:lock"
RESEED_LOCK_TTL_SECONDS: Final[int] = 1800

# Per-script wall clock. The structural seed is ~100 nodes / ~280 edges and the
# causal sync ~109 chains; both finish in seconds against a healthy FalkorDB. The
# generous ceiling exists so a hung script releases the lock via task failure
# rather than sitting on it until the TTL.
SCRIPT_TIMEOUT_SECONDS: Final[int] = 600

# Compare-and-delete: a bare DEL would drop a lock another holder acquired after
# ours expired.
_RELEASE_LOCK_LUA: Final[str] = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end return 0"
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR: Final[Path] = _REPO_ROOT / "scripts"

# Order matters: the structural layer first (Brands/Regions/KPIs/Agents and the
# relationships between them), then the causal_paths MERGE sync that hangs the 109
# validated Variable CAUSES chains off it. NEVER ``--clear-first`` — a self-heal
# that wipes the graph it is healing is #1758 with extra steps.
RESEED_STEPS: Final[Tuple[Tuple[str, Tuple[str, ...]], ...]] = (
    ("seed_falkordb.py", ()),
    ("sync_causal_paths_to_falkordb.py", ("--execute",)),
)


class GraphReseedError(RuntimeError):
    """Raised when the self-heal ran and the curated core is still empty."""


def _falkordb_conn() -> Tuple[str, int, Optional[str]]:
    """Derive host/port/password from FALKORDB_URL if set, else the discrete vars.

    Local copy of ``src/api/dependencies/falkordb_client.py::_parse_falkordb_config``
    — see the module docstring for why it is copied rather than imported.
    """
    url = os.environ.get("FALKORDB_URL")
    if url:
        parsed = urlparse(url)
        return parsed.hostname or "localhost", parsed.port or 6379, parsed.password
    return (
        os.environ.get("FALKORDB_HOST", "localhost"),
        int(os.environ.get("FALKORDB_PORT", "6379")),
        os.environ.get("FALKORDB_PASSWORD"),
    )


def _graph_name() -> str:
    return os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")


def _open_graph(graph_name: str) -> Any:
    """Return a FalkorDB graph handle. Raises if FalkorDB is unreachable."""
    from falkordb import FalkorDB  # lazy: ~15 MiB, and only on the sentinel path

    host, port, password = _falkordb_conn()
    return FalkorDB(host=host, port=port, password=password).select_graph(graph_name)


def _curated_node_count(graph: Any) -> int:
    """Count curated (non-agent-written) nodes. Raises on any query failure."""
    result = graph.query(CURATED_COUNT_QUERY)
    if not result.result_set:
        # An empty result_set is not a zero count — COUNT always returns a row.
        raise GraphReseedError("curated-count query returned no rows")
    return int(result.result_set[0][0])


def _redis_client() -> Any:
    """Sync Redis client for the reseed lock (the task itself is sync)."""
    import redis  # lazy: ~15 MiB

    url = os.environ.get("REDIS_URL", "redis://localhost:6382")
    return redis.Redis.from_url(url, decode_responses=True)


def _missing_scripts() -> List[str]:
    return [name for name, _ in RESEED_STEPS if not (_SCRIPTS_DIR / name).is_file()]


def _run_step(script: str, extra_args: Tuple[str, ...], graph_name: str) -> Dict[str, Any]:
    """Run one seed script as a subprocess against ``graph_name``."""
    cmd = [sys.executable, str(_SCRIPTS_DIR / script), *extra_args]
    if script == "seed_falkordb.py":
        # Explicit beats implicit: the script also reads FALKORDB_GRAPH_NAME, but
        # passing it pins the target even if that default ever drifts.
        cmd += ["--graph-name", graph_name]

    # FALKORDB_GRAPH_NAME is forwarded so BOTH scripts write the graph this task
    # probed. sync_causal_paths_to_falkordb.py has no --graph-name flag; the env
    # var is its only handle.
    env = {**os.environ, "FALKORDB_GRAPH_NAME": graph_name}

    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        check=False,
    )
    step: Dict[str, Any] = {"script": script, "returncode": proc.returncode}
    if proc.returncode == 0:
        logger.info("graph reseed step OK: %s (graph=%s)", script, graph_name)
    else:
        logger.error(
            "graph reseed step FAILED: %s rc=%s (graph=%s)\nstdout tail:\n%s\nstderr tail:\n%s",
            script,
            proc.returncode,
            graph_name,
            (proc.stdout or "")[-2000:],
            (proc.stderr or "")[-2000:],
        )
        step["stderr_tail"] = (proc.stderr or "")[-2000:]
    return step


@celery_app.task(
    name="src.tasks.graph_emptiness_sentinel",
    # No retries: the beat tick is the retry, 30 minutes later. An auto-retry would
    # contend with the reseed lock it just released and turn one failed heal into a
    # burst of CREATE-based reseeds.
    max_retries=0,
)
def graph_emptiness_sentinel() -> Dict[str, Any]:
    """Probe the curated core; reseed it under a lock when it is empty.

    Returns a status dict on every non-fatal outcome. Raises :class:`GraphReseedError`
    only for the one case an operator must see: the reseed ran and the graph is still
    empty.
    """
    graph_name = _graph_name()

    try:
        graph = _open_graph(graph_name)
        curated = _curated_node_count(graph)
    except Exception as exc:  # noqa: BLE001 — an unknown probe is not an empty graph
        logger.error(
            "graph emptiness probe FAILED for %s: %s — treating as UNKNOWN, not empty "
            "(a reseed on an unreachable graph would duplicate it on recovery)",
            graph_name,
            exc,
        )
        return {"status": "probe_failed", "graph": graph_name, "error": str(exc)}

    if curated > 0:
        logger.info(
            "graph emptiness sentinel: %s healthy (%d curated nodes)", graph_name, curated
        )
        return {"status": "ok", "graph": graph_name, "curated_node_count": curated}

    logger.critical(
        "KNOWLEDGE GRAPH EMPTY: %s has 0 curated nodes — /knowledge-graph renders "
        "nothing. Attempting self-heal reseed (#1758/#1761).",
        graph_name,
    )

    missing = _missing_scripts()
    if missing:
        logger.critical(
            "cannot self-heal %s: seed scripts missing from %s: %s",
            graph_name,
            _SCRIPTS_DIR,
            missing,
        )
        return {"status": "scripts_missing", "graph": graph_name, "missing": missing}

    try:
        redis_client = _redis_client()
        token = uuid.uuid4().hex
        acquired = redis_client.set(
            RESEED_LOCK_KEY, token, nx=True, ex=RESEED_LOCK_TTL_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 — fail closed: no lock, no CREATE-based reseed
        logger.error(
            "graph reseed lock unavailable (%s) — skipping reseed of %s rather than "
            "risking a concurrent duplicate seed",
            exc,
            graph_name,
        )
        return {"status": "lock_error", "graph": graph_name, "error": str(exc)}

    if not acquired:
        logger.warning(
            "graph reseed already in progress (lock %s held) — skipping this tick",
            RESEED_LOCK_KEY,
        )
        return {"status": "lock_unavailable", "graph": graph_name}

    steps: List[Dict[str, Any]] = []
    try:
        # Re-check inside the lock: another worker may have healed the graph between
        # our probe and our acquisition.
        recheck = _curated_node_count(graph)
        if recheck > 0:
            logger.info(
                "graph %s recovered before reseed (%d curated nodes) — nothing to do",
                graph_name,
                recheck,
            )
            return {
                "status": "recovered_before_reseed",
                "graph": graph_name,
                "curated_node_count": recheck,
            }

        for script, extra_args in RESEED_STEPS:
            # A failed structural seed must NOT abort the causal sync: the sync's
            # MERGEs restore the Variable layer on their own, and partial recovery
            # beats none. The post-verify below is the real gate.
            steps.append(_run_step(script, extra_args, graph_name))

        post_count = _curated_node_count(graph)
    finally:
        try:
            redis_client.eval(_RELEASE_LOCK_LUA, 1, RESEED_LOCK_KEY, token)
        except Exception as exc:  # noqa: BLE001 — TTL still bounds the lock
            logger.warning("could not release graph reseed lock: %s", exc)

    failed = [step for step in steps if step["returncode"] != 0]

    if post_count <= 0:
        raise GraphReseedError(
            f"self-heal reseed of {graph_name} completed but the curated core is "
            f"still empty (0 nodes); step results: {steps}"
        )

    if failed:
        logger.critical(
            "graph %s reseeded to %d curated nodes, but %d step(s) failed: %s — the "
            "graph may be incomplete",
            graph_name,
            post_count,
            len(failed),
            [step["script"] for step in failed],
        )
        return {
            "status": "reseeded_with_errors",
            "graph": graph_name,
            "curated_node_count": post_count,
            "steps": steps,
        }

    logger.critical(
        "graph %s self-healed: %d curated nodes restored (#1761)", graph_name, post_count
    )
    return {
        "status": "reseeded",
        "graph": graph_name,
        "curated_node_count": post_count,
        "steps": steps,
    }
