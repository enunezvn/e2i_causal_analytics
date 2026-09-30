"""Container healthcheck for one Celery worker node (#2341).

Usage (docker-compose ``CMD-SHELL``, so ``$HOSTNAME`` expands)::

    python -m src.workers.healthcheck worker_light@$HOSTNAME

Exit 0 when THAT node answers ``ping`` with ``pong`` inside ``--timeout`` seconds,
1 otherwise.

Why not ``celery inspect ping``
-------------------------------
It leaks a Redis key per late reply, and on a bare broadcast it reports the wrong
node's health.

* Every CLI run is a new process, so kombu gives it a new reply mailbox
  ``<oid>.reply.celery.pidbox`` (the oid is derived from pid/thread/object id).
  ``inspect ping`` collects replies until ``--timeout`` (default 1 s) passes with
  no new one, then deletes the mailbox and its binding.
* On the Redis transport a worker's reply is two round trips: ``SMEMBERS`` of the
  binding set to route it, then ``LPUSH`` into the mailbox list
  (``kombu/transport/virtual/base.py::_lookup`` then ``redis.py::_put``). A worker
  that is stalled between the two (swap, a busy main loop) routes the reply while
  the binding still exists and pushes it after the client deleted the mailbox. The
  ``LPUSH`` recreates the list, with no binding, no reader and no TTL.
* ``control_queue_expires`` / ``control_queue_ttl`` do not help: the Redis
  channel's ``_new_queue`` ignores ``expires`` and ``_put`` is a plain ``LPUSH``
  (read in the installed kombu 5.6.1).
* A bare broadcast asks every worker on the broker, so each container's check
  multiplies the replies by the number of workers, and exits 0 if ANY node answers.
  A dead node can look healthy while its peers are up.

This probe addresses ONE node (``destination``) and stops at its first reply
(``limit=1``). A reply that arrives inside the budget is consumed, so no mailbox
is left behind. A reply can still leak only when the node is slower than
``--timeout``, and then the check is failing anyway. Keep ``--timeout`` well
under the compose ``healthcheck.timeout``. If Docker kills the probe
mid-wait, the mailbox and its binding are both left behind.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional, Sequence

DEFAULT_TIMEOUT_SECONDS = 5.0


def node_answers(app: Any, node: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """True when ``node`` itself replies ``pong``; replies from other nodes do not count."""
    replies = app.control.ping(destination=[node], timeout=timeout, limit=1) or []
    return any((reply.get(node) or {}).get("ok") == "pong" for reply in replies)


def main(argv: Optional[Sequence[str]] = None, app: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Healthcheck for one Celery worker node.")
    parser.add_argument("node", help="full node name, e.g. worker_light@$HOSTNAME")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="seconds to wait for the node's reply (default %(default)s)",
    )
    args = parser.parse_args(argv)

    if app is None:
        from src.workers.celery_app import celery_app as app

    try:
        ok = node_answers(app, args.node, args.timeout)
    except Exception as exc:  # noqa: BLE001 - any failure means "not healthy"
        print(f"unhealthy: {args.node}: {exc!r}", file=sys.stderr)
        return 1
    if not ok:
        print(f"unhealthy: {args.node} did not answer ping in {args.timeout}s", file=sys.stderr)
        return 1
    print(f"pong: {args.node}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
