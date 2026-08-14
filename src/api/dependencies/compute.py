"""Per-worker bounded heavy-compute limiter (Priority 1 OOM fix).

Why this exists
---------------
A few API routes run genuinely heavy in-process compute:

* ``POST /api/digital-twin/simulate`` builds a twin population and runs a
  simulation (default 10k twins, up to 100k) — a single request peaks at
  roughly ~1.3 GiB.
* ``POST /api/explain/predict`` and ``/api/explain/predict/batch`` run SHAP.

The ``e2i_api`` container is capped at a 5G cgroup. With gunicorn running 4
workers, 4 concurrent heavy requests peaked at ~5.2 GiB and OOM-killed the
container. This module bounds the number of *in-flight* heavy operations per
worker so concurrent heavy requests can no longer exceed the cap.

Design
------
* **Reject fast, do not queue.** When the per-worker slot budget is exhausted we
  raise :class:`HeavyComputeSaturated` immediately (the caller maps it to a 503
  with ``Retry-After``). Queueing would just rebuild the memory/latency we are
  trying to bound — a saturated worker should shed load, not absorb it.
* **Default concurrency = 1.** Combined with gunicorn cut from 4 to 2 workers,
  this caps the worst case at 2 workers x 1 in-flight heavy op ~= 2.6 GiB, which
  sits safely under the 5G cgroup. Override with ``HEAVY_COMPUTE_MAX_CONCURRENCY``
  (and the executor pool with ``HEAVY_COMPUTE_EXECUTOR_WORKERS``).
* **Single-threaded loop => a plain counter is correct.** asyncio runs one
  coroutine at a time and ``asyncio.Semaphore`` has no non-blocking ``acquire``,
  so we use a check-then-increment counter with no ``await`` between the check
  and the increment. That is atomic on the event loop and gives us the
  reject-fast semantics a semaphore cannot.
* **Lazy, per-loop instantiation.** Under gunicorn each uvicorn worker runs its
  own event loop. The limiter is created lazily, keyed by the running loop, so a
  primitive is never bound to the wrong loop (the classic import-time-binding
  bug). The executor is plain threads and is process-global.
* **A bounded executor moves blocking sync compute off the event loop** so it
  cannot stall the worker's other requests / health checks.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENCY = 1

# Agent-graph compute pool (#1601). Deliberately SEPARATE from the shared
# heavy-compute pool above — see ``run_in_agent_compute_executor`` for the
# measurements that force the split.
#
# PER GUNICORN WORKER PROCESS, like everything else in this module. The api
# container runs ``--workers 2`` inside ``cpus: '2'``
# (docker/docker-compose.yml), so the CONTAINER-wide budget is 2 processes x 1
# thread = 2 concurrent heavy causal computations, which is exactly the CPU
# quota. A per-process 2 would have meant 4 on 2 CPUs: the runs would only
# thrash each other, and a suite that legitimately needs ~223s inside a 240s
# cooperative budget would start missing it under concurrency.
_DEFAULT_AGENT_COMPUTE_WORKERS = 1

T = TypeVar("T")


class HeavyComputeSaturated(Exception):
    """Raised when the per-worker heavy-compute slot budget is exhausted.

    Callers should translate this into an HTTP 503 with a ``Retry-After`` header
    (see ``src/api/main.py``) — the work was not started and the client should
    retry shortly.
    """


def _max_concurrency_from_env() -> int:
    raw = os.environ.get("HEAVY_COMPUTE_MAX_CONCURRENCY")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid HEAVY_COMPUTE_MAX_CONCURRENCY=%r; falling back to %d",
            raw,
            _DEFAULT_MAX_CONCURRENCY,
        )
        return _DEFAULT_MAX_CONCURRENCY
    if value < 1:
        logger.warning("HEAVY_COMPUTE_MAX_CONCURRENCY=%d < 1; clamping to 1", value)
        return 1
    return value


def _executor_workers_from_env(default: int) -> int:
    raw = os.environ.get("HEAVY_COMPUTE_EXECUTOR_WORKERS")
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid HEAVY_COMPUTE_EXECUTOR_WORKERS=%r; falling back to %d",
            raw,
            default,
        )
        return default
    return max(1, value)


class HeavyComputeLimiter:
    """Counts in-flight heavy ops and rejects fast once the budget is reached.

    Correct on a single-threaded event loop: :meth:`acquire` reads and mutates
    ``_in_flight`` with no ``await`` in between, so no other coroutine can
    interleave between the capacity check and the increment.
    """

    def __init__(self, max_concurrency: int) -> None:
        self.max_concurrency = max_concurrency
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def acquire(self) -> None:
        if self._in_flight >= self.max_concurrency:
            raise HeavyComputeSaturated(
                "heavy compute capacity reached "
                f"({self._in_flight}/{self.max_concurrency} in flight)"
            )
        self._in_flight += 1

    def release(self) -> None:
        if self._in_flight > 0:
            self._in_flight -= 1


# --------------------------------------------------------------------------- #
# Lazy, per-loop limiter + process-global bounded executor
# --------------------------------------------------------------------------- #
# Keyed by the running event loop so each uvicorn worker gets its own limiter
# and a primitive is never created on (and bound to) the wrong loop.
_limiters: dict[asyncio.AbstractEventLoop, HeavyComputeLimiter] = {}

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()

# #1601: the agent-graph pool. Process-global like ``_executor`` (plain threads,
# never bound to a loop) but sized and consumed independently.
_agent_compute_executor: ThreadPoolExecutor | None = None
_agent_compute_lock = threading.Lock()

# True while THIS asyncio context already holds a heavy slot. Lets a batch
# endpoint hold one slot for its whole fan-out while the inner per-item calls
# reuse it instead of each contending for (and self-rejecting on) a new slot.
_slot_held: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "heavy_compute_slot_held", default=False
)


def get_heavy_compute_limiter() -> HeavyComputeLimiter:
    """Return the limiter for the currently-running event loop (lazily created)."""
    loop = asyncio.get_running_loop()
    limiter = _limiters.get(loop)
    if limiter is None:
        limiter = HeavyComputeLimiter(_max_concurrency_from_env())
        _limiters[loop] = limiter
    return limiter


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="heavy-compute",
                )
    return _executor


def _agent_compute_workers_from_env() -> int:
    """Worker count for the agent-graph pool (``AGENT_COMPUTE_EXECUTOR_WORKERS``)."""
    raw = os.environ.get("AGENT_COMPUTE_EXECUTOR_WORKERS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_AGENT_COMPUTE_WORKERS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid AGENT_COMPUTE_EXECUTOR_WORKERS=%r; falling back to %d",
            raw,
            _DEFAULT_AGENT_COMPUTE_WORKERS,
        )
        return _DEFAULT_AGENT_COMPUTE_WORKERS
    return max(1, value)


def _get_agent_compute_executor(max_workers: int) -> ThreadPoolExecutor:
    global _agent_compute_executor
    if _agent_compute_executor is None:
        with _agent_compute_lock:
            if _agent_compute_executor is None:
                _agent_compute_executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="agent-compute",
                )
    return _agent_compute_executor


def _reset_limiter_cache_for_tests() -> None:
    """Test hook: drop cached limiters/executors so env overrides re-apply.

    Not part of the production API; used by the unit tests to start each case
    from a clean limiter bound to the test's event loop.
    """
    global _executor, _agent_compute_executor
    _limiters.clear()
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
    if _agent_compute_executor is not None:
        _agent_compute_executor.shutdown(wait=False, cancel_futures=True)
        _agent_compute_executor = None
    _slot_held.set(False)


@asynccontextmanager
async def heavy_compute_slot(*, reuse_if_held: bool = False) -> AsyncIterator[None]:
    """Acquire one heavy-compute slot for the duration of the block.

    Raises :class:`HeavyComputeSaturated` on enter if the per-worker budget is
    exhausted (reject fast — nothing is queued).

    :param reuse_if_held: when True and this asyncio context already holds a
        slot, become a no-op (do not acquire or release a second slot). Used by
        per-item work invoked inside a batch that already holds one slot, so the
        batch's single slot is shared rather than contended item-by-item.
    """
    if reuse_if_held and _slot_held.get():
        # The enclosing scope already owns a slot; ride on it.
        yield
        return

    limiter = get_heavy_compute_limiter()
    limiter.acquire()
    token = _slot_held.set(True)
    try:
        yield
    finally:
        _slot_held.reset(token)
        limiter.release()


async def run_in_bounded_executor(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking callable in the shared bounded thread pool, off the loop.

    Does NOT itself acquire a heavy-compute slot — compose it with
    :func:`heavy_compute_slot` to bound concurrency. The pool size defaults to
    the heavy-compute concurrency limit (override with
    ``HEAVY_COMPUTE_EXECUTOR_WORKERS``) so the executor never outpaces the slot
    budget.
    """
    loop = asyncio.get_running_loop()
    max_workers = _executor_workers_from_env(_max_concurrency_from_env())
    executor = _get_executor(max_workers)
    call = functools.partial(func, *args, **kwargs)
    # run_in_executor returns Future[Any]; the callable's return type is T.
    result: T = await loop.run_in_executor(executor, call)
    return result


async def run_in_agent_compute_executor(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run LONG-RUNNING agent-graph compute on a bounded pool, off the loop (#1601).

    Why a SECOND pool instead of :func:`run_in_bounded_executor`
    -----------------------------------------------------------
    The causal_impact agent graph off-loaded four heavy sync calls with
    ``asyncio.to_thread``, i.e. onto the loop's DEFAULT executor
    (``min(32, cpu+4)`` = 12 threads here), outside every in-process bound the
    container has. Measured on this box (2026-08-14, real callables, real frame
    shapes):

    ==============================================  ========  ===============
    call                                                wall   peak RSS delta
    ==============================================  ========  ===============
    ``_select_estimator_with_energy_score``           124.9 s        177.4 MB
      (37,515x12, the largest frame observed in prod)
    ``_reconstruct_dowhy_artifacts`` (5k subsample)    10.2 s          0.3 MB
    ``refute_estimate(num_simulations=1)``              9.7 s          0.5 MB
    ==============================================  ========  ===============

    So estimation is the MEMORY term — 177 MB x 12 concurrent copies is ~2.1 GiB
    against a 5G cgroup already carrying ~549 MB/worker of imports — and the
    refutation calls are a pure CPU-TIME term. Bounding both is right; putting
    them on the SHARED pool is not. That pool is ONE thread (default
    ``HEAVY_COMPUTE_*`` are unset repo-wide) serving reject-fast API endpoints
    plus #1598's composer sync tools under a 120 s envelope, whereas one causal
    run occupies a thread for ~350 s (125 s estimation + a suite designed to run
    ~223 s inside its 240 s cooperative budget, see
    ``orchestrator/nodes/router.py``). Sharing would make every composer sync
    step blow its envelope and stall ``rank_drivers`` SHAP (#1590) and
    ``POST /api/digital-twin/simulate`` for minutes — trading an unlikely OOM for
    a certain cross-feature failure.

    The cost of the split is the extra in-flight memory the two pools can hold
    at once: measured 177 MB, negligible against the 5G cgroup, and far cheaper
    than the head-of-line blocking it avoids.

    Sizing. Default 1 PER WORKER PROCESS. The container runs ``--workers 2``
    inside ``cpus: '2'``, so container-wide that is 2 concurrent heavy causal
    computations — exactly the CPU quota, and ~354 MB of in-flight estimation
    memory. Two concurrent chat turns still proceed in parallel when gunicorn
    lands them on different workers; two on the SAME worker serialize, which is
    the intended trade (4 threads on 2 CPUs would only thrash, and the ~223 s
    suite would start missing its 240 s budget). Override with
    ``AGENT_COMPUTE_EXECUTOR_WORKERS``.

    No ``heavy_compute_slot`` is taken: the slot is reject-fast (503 +
    ``Retry-After``) for API entry points that can shed load, and a chat turn's
    analysis should queue briefly rather than fail outright — the same call
    #1590 made for ``rank_drivers`` and #1598 for composer sync tools.

    NOT a timeout. A thread cannot be cancelled, so wrapping this in
    ``asyncio.wait_for`` would abandon a still-running call that keeps holding a
    bounded slot. Callers bound the WORK cooperatively instead, via the
    ``compute_deadline`` in ``CausalImpactState``, which lets the thread return.
    Because this pool QUEUES, that budget must be re-checked on the worker
    thread rather than only before submitting — see
    ``agents/causal_impact/nodes/_compute_budget.run_bounded_with_budget``,
    which is how the causal_impact nodes call in here.

    Context is copied exactly as ``asyncio.to_thread`` does, so replacing a
    ``to_thread`` call with this helper preserves contextvar propagation.
    """
    loop = asyncio.get_running_loop()
    executor = _get_agent_compute_executor(_agent_compute_workers_from_env())
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, func, *args, **kwargs)
    result: T = await loop.run_in_executor(executor, call)
    return result


# --------------------------------------------------------------------------- #
# Priority 2 — heavy-compute offload to worker_heavy (DARK by default)
# --------------------------------------------------------------------------- #
# When HEAVY_OFFLOAD_ENABLED is set, the two heavy API paths (digital-twin
# simulate, SHAP explain) enqueue the work as a Celery task on worker_heavy
# instead of running it inline. Default OFF: behavior is byte-identical to the
# P1 inline path (bounded executor + heavy_compute_slot). worker_heavy still
# ships at replicas: 0, so enabling this also requires scaling worker_heavy up
# (box headroom permitting). Read at call time so ops/tests can toggle it.

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def heavy_offload_enabled() -> bool:
    """Whether heavy API compute should be offloaded to ``worker_heavy`` (P2).

    DARK by default: unless ``HEAVY_OFFLOAD_ENABLED`` is set to a truthy value
    (``1``/``true``/``yes``/``on``, case-insensitive) the API keeps the P1 inline
    path. Evaluated per-request (not cached at import) so it can be flipped via
    env without a code change.
    """
    return os.environ.get("HEAVY_OFFLOAD_ENABLED", "false").strip().lower() in _TRUTHY


async def await_celery_result(
    async_result: Any,
    *,
    timeout: float,
    poll_interval: float = 0.25,
) -> Any:
    """Await a Celery ``AsyncResult`` WITHOUT blocking the event loop.

    Polls ``async_result.ready()`` with a small ``asyncio.sleep`` (never
    ``.get()``, which blocks the loop) so the API worker stays responsive while a
    task runs on ``worker_heavy``. Preserves the synchronous HTTP contract:

    * on timeout -> raises :class:`TimeoutError` (the route maps it to HTTP 408);
    * on task failure -> re-raises the worker exception (the route's existing
      ``except Exception`` maps it to HTTP 500), matching the inline path.

    :param async_result: a Celery ``AsyncResult`` (or any object exposing
        ``ready()``, ``successful()``, ``result`` and ``get()``).
    :param timeout: max seconds to wait before raising ``TimeoutError``.
    :param poll_interval: seconds between readiness polls.
    :returns: the task's return value (a JSON-safe dict for the P2 tasks).
    """
    waited = 0.0
    while not async_result.ready():
        if waited >= timeout:
            raise TimeoutError(f"heavy offload task did not complete within {timeout:.0f}s")
        await asyncio.sleep(poll_interval)
        waited += poll_interval

    if not async_result.successful():
        # Re-raise the worker-side exception so the route's existing handlers
        # produce the same error response the inline path would.
        raise (
            async_result.result
            if isinstance(async_result.result, BaseException)
            else RuntimeError(str(async_result.result))
        )
    return async_result.result
