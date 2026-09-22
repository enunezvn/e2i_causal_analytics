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
starts the same HTTP server (``FeatureStore.serve`` -> ``feature_server.start_server``,
CLI defaults: access log on, keep-alive 5 s, registry TTL 5 s). The handlers look the
methods up on the instance at call time, so the wrapped bound attributes are what run.
The two handlers are sync ``def`` routes (Starlette threadpool); the online read path
(``/get-online-features``) is ``async`` and does not share that pool. The lock wait is
BOUNDED (``FEAST_REGISTRY_LOCK_WAIT_SECONDS``, default 600 s — longer than any loop
cycle): a caller that cannot get the lock in time gets an error (HTTP 500), which the
worker records as a failed job and fails loud on, instead of a thread parked forever.
The worker's HTTP timeout for these calls (``FeastConfig.materialize_timeout_seconds``,
900 s) exceeds this wait plus a materialize run, so the worker never records a failure
for a call the server later completes (codex r3 MED-4).

There is deliberately NO fallback to an unlocked server: if this cannot start, the
container fails and the deploy's feast recreate step rolls back and fails loud
(.github/workflows/deploy.yml) rather than leaving an unlocked writer in production.
The startup path (FeatureStore -> wrap -> get_app) was exercised on feast 0.43.0 on the
dev host against a temp repo before this shipped.
"""

from __future__ import annotations

import fcntl
import functools
import logging
import os
import sys
import time
from typing import Any, Callable

LOCK_PATH = os.environ.get("FEAST_REGISTRY_LOCK", "/feast/data/.registry.lock")
LOCK_WAIT_SECONDS = float(os.environ.get("FEAST_REGISTRY_LOCK_WAIT_SECONDS", "600"))
REPO_PATH = os.environ.get("FEAST_REPO_PATH", "/feast")
HOST = os.environ.get("FEAST_SERVE_HOST", "0.0.0.0")
PORT = int(os.environ.get("FEAST_SERVE_PORT", "6566"))
_POLL_SECONDS = 0.5

logger = logging.getLogger("serve_locked")


class RegistryLockTimeout(RuntimeError):
    """The registry lock stayed busy for longer than the bounded wait."""


def _acquire(lock_file: Any, wait_seconds: float) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise RegistryLockTimeout(
                    f"registry lock {lock_file.name} busy for more than {wait_seconds:.0f}s"
                ) from None
            time.sleep(_POLL_SECONDS)


def locked(
    fn: Callable[..., Any], lock_path: str = LOCK_PATH, wait_seconds: float = LOCK_WAIT_SECONDS
) -> Callable[..., Any]:
    """``fn`` under an exclusive advisory lock on ``lock_path`` (created if absent),
    waiting at most ``wait_seconds`` for it."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with open(lock_path, "a") as lock_file:
            _acquire(lock_file, wait_seconds)
            try:
                return fn(*args, **kwargs)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return wrapper


def lock_store_materialization(
    store: Any, lock_path: str = LOCK_PATH, wait_seconds: float = LOCK_WAIT_SECONDS
) -> Any:
    """Wrap ``store.materialize`` and ``store.materialize_incremental`` (in place)."""
    store.materialize = locked(store.materialize, lock_path, wait_seconds)
    store.materialize_incremental = locked(store.materialize_incremental, lock_path, wait_seconds)
    return store


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from feast import FeatureStore  # only importable in the feast sidecar image

    os.chdir(REPO_PATH)  # feast's repo parsing is cwd-relative, like `feast --chdir`
    store = lock_store_materialization(FeatureStore(repo_path="."), LOCK_PATH, LOCK_WAIT_SECONDS)
    logger.info(
        "starting feast serve on %s:%s with materialize endpoints registry-locked on %s "
        "(bounded wait %.0fs)",
        HOST,
        PORT,
        LOCK_PATH,
        LOCK_WAIT_SECONDS,
    )
    store.serve(
        host=HOST,
        port=PORT,
        type_="http",
        no_access_log=False,
        keep_alive_timeout=5,
        registry_ttl_sec=5,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
