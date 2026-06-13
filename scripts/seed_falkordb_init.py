#!/usr/bin/env python3
"""
E2I Causal Analytics - Docker Init Seeder for FalkorDB.

Auto-seeds the FalkorDB e2i_causal graph on container startup.
Designed to run as a one-shot init container in Docker Compose.

Only seeds when the graph is empty — safe to run on every `docker compose up`.

The former e2i_semantic seeding step was retired in #890: the deployed
semantic graph is e2i_causal (#749) and nothing reads e2i_semantic.

Environment variables:
    FALKORDB_HOST       FalkorDB hostname (default: falkordb)
    FALKORDB_PORT       FalkorDB port (default: 6379)
    FALKORDB_PASSWORD   FalkorDB password (required)
"""

import logging
import os
import re
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("falkordb-init")


def _validate_host(value: str) -> str:
    """Reject anything that's not a plausible DNS hostname or IP literal.

    Locks the host string down to characters legal in DNS labels
    (letters, digits, dot, hyphen) plus square brackets and colons for
    IPv6 literals. This neutralizes the taint-flow concern even though
    subprocess is invoked with a list (no shell), making OS command
    injection structurally impossible.
    """
    if not re.fullmatch(r"[A-Za-z0-9.\-:\[\]]+", value):
        raise ValueError(f"Refusing to use suspicious FALKORDB_HOST: {value!r}")
    return value


def _validate_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"FALKORDB_PORT out of range: {port}")
    return port


FALKORDB_HOST = _validate_host(os.getenv("FALKORDB_HOST", "falkordb"))
FALKORDB_PORT = _validate_port(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD", "")

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)


def count_nodes(graph_name: str) -> int:
    """Count nodes in a FalkorDB graph. Returns 0 if graph doesn't exist."""
    try:
        from falkordb import FalkorDB

        db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD)
        g = db.select_graph(graph_name)
        result = g.query("MATCH (n) RETURN count(n)")
        return int(result.result_set[0][0])
    except Exception as e:
        logger.debug("Could not count nodes in %s: %s", graph_name, e)
        return 0


def seed_causal_graph() -> bool:
    """Seed e2i_causal graph using seed_falkordb.py."""
    count = count_nodes("e2i_causal")
    if count > 0:
        logger.info("e2i_causal: already has %d nodes -- skipping", count)
        return True

    logger.info("e2i_causal: empty -- seeding...")
    # FALKORDB_HOST/PORT come from docker-compose internal env, not user input;
    # invocation uses list args (no shell=True) so injection is not possible.
    result = subprocess.run(  # nosemgrep
        [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "seed_falkordb.py"),
            "--host",
            FALKORDB_HOST,
            "--port",
            str(FALKORDB_PORT),
            "--clear-first",
        ],
        cwd=PROJECT_DIR,
        env={**os.environ, "FALKORDB_PASSWORD": FALKORDB_PASSWORD},
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("e2i_causal seeding failed (exit code %d)", result.returncode)
        return False

    final = count_nodes("e2i_causal")
    logger.info("e2i_causal: seeded -- %d nodes", final)
    return final > 0


# NOTE (#890): the e2i_semantic seeding step was retired. The deployed
# semantic graph is e2i_causal (config/005_memory_config.yaml, #749) — no
# runtime reader uses e2i_semantic, and even this seeder's read-only count
# probe re-created the empty e2i_semantic graph shell (FalkorDB creates a
# graph key on any GRAPH.QUERY). scripts/seed_semantic_graph.py remains
# available for explicit manual runs against the legacy graph.


def main() -> int:
    logger.info("=== FalkorDB Init Seeder ===")
    logger.info("Host: %s:%d", FALKORDB_HOST, FALKORDB_PORT)

    if not FALKORDB_PASSWORD:
        logger.error("FALKORDB_PASSWORD is not set")
        return 1

    success = seed_causal_graph()

    if success:
        logger.info("=== Init seeding complete ===")
        return 0
    else:
        logger.error("=== Init seeding had failures ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
