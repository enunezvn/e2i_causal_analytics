#!/usr/bin/env python3
"""`feast serve` with the shared registry lock around the materialize endpoints (#2207).

Feast's file registry is written IN PLACE (``FileRegistryStore._write_registry``: open
"wb" + write, no rename) and every ``materialize`` / ``materialize_incremental`` call
writes it (``Registry.apply_materialization`` records the interval). The e2i_feast
materializer loop therefore takes ``flock /feast/data/.registry.lock`` around every
``feast materialize`` (docker/feast/materializer-entrypoint.sh, #556 H2), and the
serve container takes it around ``feast apply``. The feature server's own
``POST /materialize`` / ``/materialize-incremental`` handlers take no lock — so once
the worker's beats drive materialization over HTTP (src/feature_store/
feast_remote_materialize.py) two writers could race on the registry (codex r1 HIGH-3).

This entrypoint builds the same FeatureStore the CLI would, wraps the two store
methods the handlers call with an exclusive ``fcntl.flock`` on the SAME lock file, and
starts the same HTTP server (``FeatureStore.serve`` -> ``feature_server.start_server``).
The handlers look the methods up on the instance at call time, so the wrapped bound
attributes are what run. Sync ``def`` routes run in Starlette's threadpool, so blocking
on the lock is fine. Everything else about the server is unchanged.

Run as ``python3 /serve_locked.py`` from the serve entrypoint; on a startup failure the
entrypoint falls back to plain ``feast serve`` (unlocked but serving) so a wrapper
defect can never take online serving down.
"""

from __future__ import annotations

import fcntl
import functools
import logging
import os
import sys
from typing import Any, Callable

LOCK_PATH = os.environ.get("FEAST_REGISTRY_LOCK", "/feast/data/.registry.lock")
REPO_PATH = os.environ.get("FEAST_REPO_PATH", "/feast")
HOST = os.environ.get("FEAST_SERVE_HOST", "0.0.0.0")
PORT = int(os.environ.get("FEAST_SERVE_PORT", "6566"))

logger = logging.getLogger("serve_locked")


def locked(fn: Callable[..., Any], lock_path: str = LOCK_PATH) -> Callable[..., Any]:
    """``fn`` under an exclusive advisory lock on ``lock_path`` (created if absent)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with open(lock_path, "a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return fn(*args, **kwargs)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return wrapper


def lock_store_materialization(store: Any, lock_path: str = LOCK_PATH) -> Any:
    """Wrap ``store.materialize`` and ``store.materialize_incremental`` (in place)."""
    store.materialize = locked(store.materialize, lock_path)
    store.materialize_incremental = locked(store.materialize_incremental, lock_path)
    return store


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from feast import FeatureStore  # only importable in the feast sidecar image

    os.chdir(REPO_PATH)  # feast's repo parsing is cwd-relative, like `feast --chdir`
    store = lock_store_materialization(FeatureStore(repo_path="."), LOCK_PATH)
    logger.info(
        "starting feast serve on %s:%s with materialize endpoints locked on %s",
        HOST,
        PORT,
        LOCK_PATH,
    )
    store.serve(host=HOST, port=PORT, type_="http", no_access_log=False, registry_ttl_sec=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
