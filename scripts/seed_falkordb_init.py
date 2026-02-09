#!/usr/bin/env python3
"""
E2I Causal Analytics - Docker Init Seeder for FalkorDB.

Auto-seeds both FalkorDB graphs (e2i_causal + e2i_semantic) on container startup.
Designed to run as a one-shot init container in Docker Compose.

Only seeds graphs that are empty — safe to run on every `docker compose up`.

Environment variables:
    FALKORDB_HOST       FalkorDB hostname (default: falkordb)
    FALKORDB_PORT       FalkorDB port (default: 6379)
    FALKORDB_PASSWORD   FalkorDB password (required)
"""

import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("falkordb-init")

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "falkordb")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
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
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "seed_falkordb.py"),
            "--host", FALKORDB_HOST,
            "--port", str(FALKORDB_PORT),
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


def seed_semantic_graph() -> bool:
    """Seed e2i_semantic graph using seed_semantic_graph.py."""
    count = count_nodes("e2i_semantic")
    if count > 0:
        logger.info("e2i_semantic: already has %d nodes -- skipping", count)
        return True

    logger.info("e2i_semantic: empty -- seeding...")
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "seed_semantic_graph.py"),
            "--clear-first",
        ],
        cwd=PROJECT_DIR,
        env={
            **os.environ,
            "FALKORDB_HOST": FALKORDB_HOST,
            "FALKORDB_PORT": str(FALKORDB_PORT),
            "FALKORDB_PASSWORD": FALKORDB_PASSWORD,
        },
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("e2i_semantic seeding failed (exit code %d)", result.returncode)
        return False

    final = count_nodes("e2i_semantic")
    logger.info("e2i_semantic: seeded -- %d nodes", final)
    return final > 0


def main() -> int:
    logger.info("=== FalkorDB Init Seeder ===")
    logger.info("Host: %s:%d", FALKORDB_HOST, FALKORDB_PORT)

    if not FALKORDB_PASSWORD:
        logger.error("FALKORDB_PASSWORD is not set")
        return 1

    success = True
    success = seed_causal_graph() and success
    success = seed_semantic_graph() and success

    if success:
        logger.info("=== Init seeding complete ===")
        return 0
    else:
        logger.error("=== Init seeding had failures ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
