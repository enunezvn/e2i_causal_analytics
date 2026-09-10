"""Cross-worker in-flight lock for build-once endpoints (#1993).

``POST /expert-reviews/{review_id}/assessment`` builds an LLM assessment when
the row has none cached. Two concurrent uncached requests for one review both
built it (measured on main: builds=2, both ``cached=False``, last write wins):
two bills for one value. The API runs ``gunicorn --workers 2`` (measured on the
live container), so a process-local ``asyncio.Lock`` covers only the requests
that land on the same worker; the cross-worker lock lives in Redis.

How ``hold`` works (two stages, ONE wait budget of TTL + one poll):

1. Always take the per-id, refcounted, process-local ``asyncio.Lock`` FIRST,
   whatever the Redis state. Same-worker requests queue here, so at most one
   request per worker goes on to Redis for a given id, and a Redis recovery
   can never bypass an active local holder (codex round-4: A acquired locally
   during an outage and was still building when the cooldown lifted; B on the
   same worker found no key and built concurrently).
2. The local holder then attempts ``SET NX`` for cross-worker exclusion.

``Lease.mode``: ``"redis"`` when the key was obtained, ``"local"`` when Redis
is unavailable (same-worker exclusion only), ``"none"`` when the budget was
exhausted in either stage (no usable lock; the caller decides).

Redis path (the app's already-initialised ``redis.asyncio`` client):

- acquire: ``SET <prefix>:<id> <uuid-token> NX PX <ttl_ms>``. The TTL tracks
  the client-visible request cap: nginx ``proxy_read_timeout 120s`` for
  ``location /api/`` (docker/nginx/nginx.secure.conf:227), after which the
  caller gets a 504 while the backend keeps building. gunicorn ``--timeout
  120`` does NOT cap a build: with ``UvicornWorker`` the worker heartbeat comes
  from the event loop and the build runs in ``to_thread``, so gunicorn never
  kills it. If a build outlives the TTL the key expires and one waiter may
  build once more (the pre-#1993 behaviour, bounded to one extra build); the
  compare-and-delete release keeps that new holder's key safe from the overrun
  holder. No PEXPIRE heartbeat by design: the route logs each build's elapsed
  seconds so the TTL can be revisited on data.
- wait: a loser polls ``GET`` every ``poll_seconds`` until the key is gone
  (released or expired) and then retries the SET. The wait is bounded by the
  TTL; on exhaustion ``hold`` yields ``mode == "none"`` (NO lock held) and the
  CALLER decides. The assessment route answers 409 rather than building
  unlocked: an unlocked build could overlap a legitimate holder, and a client
  that waited the full TTL has already received nginx's 504, so it would serve
  nobody. Budget accounting (codex round-5): every stage-2 command (client
  getter, SET, GET) and every poll sleep is trimmed to the remaining budget;
  no SET is attempted once the budget is gone; and budget exhaustion is a
  distinct outcome from a Redis failure (it never enters the degrade
  cooldown). A SET that succeeds inside the last trimmed command is KEPT, not
  released to answer 409: the key is held so no overlap is possible and the
  build is persisted for the next caller, while discarding a won key serves
  nobody (nginx has already dropped a client that waited the full TTL). The
  release EVAL runs after the build, outside the wait budget, and keeps the
  plain per-command cap. The guarantee, precisely: Redis being DOWN never fails a request
  (local fallback below); exhausting the wait bound DOES, in EITHER mode,
  because that client has already been 504'd by nginx, so a LOCAL holder
  exceeding the bound also yields ``mode == "none"``.
- release: compare-and-delete Lua (``GET == token -> DEL``) in ``finally``, so
  a lease that outlived its TTL can never delete the NEXT holder's key. A
  failed release is logged and left to the TTL; it is not an outage signal.

Worker shutdown (bounded limitation): a holder cancelled by its worker's loop
shutdown (recycle, deploy) releases in ``finally`` where it still can; if the
process dies first the key clears at its TTL. The in-flight build is lost like
any in-flight request on that worker; nothing is corrupted. Draining in-flight
builds is the lifespan's job (main.py follow-up, outside this module).

Cleanup ordering: the release EVAL is the only await on a cleanup path (the
stage-1/stage-2 exception handlers and the budget-exhaustion path are
synchronous), and the local mutex release/unref sits in a nested ``finally``
below it, so a cancellation arriving during the EVAL can never strand the
per-id local entry; the Redis key then clears at its TTL.

A leaked key (a worker dying mid-hold, a lost release) only delays the requests
that would BUILD -- ``force=true``, or a retry after a failed persist -- by at
most the remaining TTL: the cached fast path runs before the lock, so a review
with a stored assessment is never held up.

Degraded path (Redis unavailable, erroring, or hanging): the stage-1 local
mutex is the only exclusion (same-worker requests still serialise;
cross-worker duplicates are possible: the pre-#1993 behaviour, degraded by
design), warn ONCE per process, and stop probing Redis for
``degrade_cooldown_seconds`` so a dead Redis costs one bounded attempt, not a
timeout per poll. The cooldown and the one-time warning are per process (each
gunicorn worker holds its own lock instance). Only transport failures
(``durable_job_store._REDIS_DEGRADE_ERRORS``) count as an outage; any other
exception still degrades (never fail the request) but is logged as an ERROR
with its traceback so a bug is not disguised as an outage. Every Redis command
is capped by ``op_timeout_seconds``. The
local table holds only in-flight ids (an entry is dropped when its last
holder/waiter leaves), so it is bounded by concurrency, and a held entry is
never evicted.

Deliberate deviation from ``durable_job_store``: the default client getter reads
the client the lifespan initialised (``redis_client._redis_client``) instead of
calling ``get_redis()``, because ``get_redis()`` runs ``init_redis()``, a
5-attempt exponential backoff (2..30 s), when the client is None, and a request
must not wait on that.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Literal, Optional

from src.api.dependencies.durable_job_store import _REDIS_DEGRADE_ERRORS

logger = logging.getLogger(__name__)

# nginx proxy_read_timeout 120s for location /api/ (docker/nginx/nginx.secure.conf:227):
# the longest a caller waits on a build; gunicorn --timeout does not cap it (see above).
DEFAULT_TTL_MS = 120_000
DEFAULT_POLL_SECONDS = 0.2
DEFAULT_OP_TIMEOUT_SECONDS = 1.0  # cap per Redis command; never the 3 s socket wait per poll
DEFAULT_DEGRADE_COOLDOWN_SECONDS = 30.0  # after a failure, skip Redis this long (no storm)

# Compare-and-delete: only the holder whose token is still stored may delete.
_RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end return 0"
)

# Zero-arg async factory yielding a Redis client (injectable for tests).
RedisFactory = Callable[[], Awaitable[Any]]


class _BudgetExhausted(Exception):
    """The shared wait budget ran out during stage 2 -- never a Redis failure,
    never a degrade event; the caller yields ``mode == "none"``."""


async def _default_redis_factory() -> Any:
    """The app's already-initialised async Redis client, WITHOUT ``get_redis()``:
    that would run ``init_redis()``'s multi-attempt backoff on the request path
    when startup ran in Redis-degraded mode. Not initialised -> degrade."""
    from src.api.dependencies import redis_client

    client = redis_client._redis_client
    if client is None:
        raise RuntimeError("Redis client not initialised (startup degraded mode)")
    return client


LockMode = Literal["redis", "local", "none"]


@dataclass
class Lease:
    """What ``InflightLock.hold`` yields.

    ``mode``: ``"redis"`` (cross-worker), ``"local"`` (this process only) or
    ``"none"`` (the bounded wait was exhausted; NO lock is held and the caller
    must not build).
    ``waited``: another request held this id when we arrived, so the caller
    should re-read whatever that request may have produced before building.
    """

    key: str
    token: str
    mode: LockMode = "redis"
    waited: bool = False


@dataclass
class _LocalEntry:
    lock: asyncio.Lock
    refs: int = 0  # holders + waiters; the entry lives while refs > 0


class InflightLock:
    """Per-id mutual exclusion across gunicorn workers, degrading gracefully.

    Construct one per logical operation (``prefix`` namespaces the Redis keys);
    use ``async with lock.hold(id) as lease:`` around the build.
    """

    def __init__(
        self,
        prefix: str,
        *,
        ttl_ms: int = DEFAULT_TTL_MS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        op_timeout_seconds: float = DEFAULT_OP_TIMEOUT_SECONDS,
        degrade_cooldown_seconds: float = DEFAULT_DEGRADE_COOLDOWN_SECONDS,
        redis_factory: Optional[RedisFactory] = None,
    ) -> None:
        self.prefix = prefix
        self.ttl_ms = ttl_ms
        self.poll_seconds = poll_seconds
        self.op_timeout_seconds = op_timeout_seconds
        self.degrade_cooldown_seconds = degrade_cooldown_seconds
        self._redis_factory: RedisFactory = redis_factory or _default_redis_factory
        self._local: Dict[str, _LocalEntry] = {}
        self._degraded_until = 0.0
        self._warned = False

    def key(self, key_id: str) -> str:
        return f"{self.prefix}:{key_id}"

    @property
    def _wait_seconds(self) -> float:
        # One poll past the TTL so a dead holder's key has expired by the bound.
        return self.ttl_ms / 1000.0 + self.poll_seconds

    # -- Redis side --------------------------------------------------------

    async def _redis(self, deadline: Optional[float] = None) -> Optional[Any]:
        """The client, or None to run in local mode. Raises _BudgetExhausted when
        the getter itself cannot complete inside the remaining budget."""
        if time.monotonic() < self._degraded_until:
            return None
        try:
            return await self._op(self._redis_factory, deadline)
        except _BudgetExhausted:
            raise
        except _REDIS_DEGRADE_ERRORS as e:
            self._degrade(e)
            return None
        except Exception as e:  # never fail the request; but do not call a bug an outage
            self._degrade(e, unexpected=True)
            return None

    def _degrade(self, exc: BaseException, *, unexpected: bool = False) -> None:
        # Per process: each gunicorn worker has its own instance, hence its own
        # cooldown and its own one-time warning. Reserved for factory/acquire
        # failures; a failed RELEASE never enters the cooldown (see _release_redis)
        # and neither does budget exhaustion (_BudgetExhausted).
        self._degraded_until = time.monotonic() + self.degrade_cooldown_seconds
        if unexpected:
            logger.error(
                f"{self.prefix}: unexpected {type(exc).__name__} from the Redis in-flight "
                f"lock ({exc!r}), not a transport failure (likely a bug); using the "
                f"process-local fallback for {self.degrade_cooldown_seconds:g}s",
                exc_info=True,
            )
            return
        msg = (
            f"{self.prefix}: Redis in-flight lock unavailable ({exc!r}); using the "
            f"process-local fallback for {self.degrade_cooldown_seconds:g}s (same-worker "
            "requests still serialise; cross-worker duplicate builds are possible)"
        )
        if not self._warned:
            self._warned = True
            logger.warning(msg)
        else:
            logger.debug(msg)

    async def _op(
        self, command: Callable[[], Awaitable[Any]], deadline: Optional[float] = None
    ) -> Any:
        """Run one Redis command capped by ``op_timeout_seconds`` and, when a
        ``deadline`` is given, by the remaining wait budget. A timeout that
        coincides with the deadline is budget exhaustion, not a Redis failure."""
        if deadline is None:
            return await asyncio.wait_for(command(), timeout=self.op_timeout_seconds)
        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _BudgetExhausted()
        try:
            return await asyncio.wait_for(
                command(), timeout=min(self.op_timeout_seconds, remaining)
            )
        except TimeoutError:
            if loop.time() >= deadline:
                raise _BudgetExhausted() from None
            raise

    async def _acquire_redis(self, client: Any, lease: Lease, deadline: float) -> bool:
        """True = acquired (a SET that lands inside the last trimmed command is
        kept; see the module docstring). Raises _BudgetExhausted when the
        shared budget runs out -- no SET is attempted past the deadline -- and
        a Redis error otherwise (the caller degrades to local mode). Only the
        holder of the per-id local mutex calls this, so at most one waiter per
        worker polls Redis for a given id."""
        loop = asyncio.get_running_loop()
        while True:
            if loop.time() >= deadline:
                raise _BudgetExhausted()
            if await self._op(
                lambda: client.set(lease.key, lease.token, nx=True, px=self.ttl_ms), deadline
            ):
                return True
            lease.waited = True  # held by ANOTHER worker
            # Poll for the holder's release (or its TTL expiry), then retry the SET.
            while await self._op(lambda: client.get(lease.key), deadline) is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise _BudgetExhausted()
                await asyncio.sleep(min(self.poll_seconds, remaining))

    async def _release_redis(self, client: Any, lease: Lease) -> None:
        # A lost release is NOT an outage signal: the TTL bounds the leaked key
        # and the next acquire judges Redis for itself. Entering the cooldown
        # here would send this whole worker to the local lock for 30 s on one
        # lost DEL. Never fail the response. Runs after the build, outside the
        # wait budget: only the per-command cap applies.
        try:
            await self._op(lambda: client.eval(_RELEASE_LUA, 1, lease.key, lease.token))
        except _REDIS_DEGRADE_ERRORS as e:
            logger.warning(
                f"{lease.key}: in-flight lock release failed ({e!r}); the key expires "
                f"with its TTL ({self.ttl_ms} ms)"
            )
        except Exception as e:
            logger.error(
                f"{lease.key}: unexpected {type(e).__name__} releasing the in-flight lock "
                f"({e!r}); the key expires with its TTL ({self.ttl_ms} ms)",
                exc_info=True,
            )

    # -- process-local fallback --------------------------------------------

    def _local_ref(self, key_id: str) -> _LocalEntry:
        entry = self._local.get(key_id)
        if entry is None:
            entry = self._local[key_id] = _LocalEntry(asyncio.Lock())
        entry.refs += 1
        return entry

    def _local_unref(self, key_id: str) -> None:
        entry = self._local.get(key_id)
        if entry is None:
            return
        entry.refs -= 1
        if entry.refs <= 0:
            del self._local[key_id]

    # -- public API ----------------------------------------------------------

    @asynccontextmanager
    async def hold(self, key_id: str) -> AsyncIterator[Lease]:
        lease = Lease(key=self.key(key_id), token=uuid.uuid4().hex)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._wait_seconds  # ONE budget for both stages

        # Stage 1 (both modes): the per-id local mutex. Same-worker requests
        # queue here, so at most one of them goes on to Redis for this id and a
        # Redis recovery can never bypass an active local holder.
        entry = self._local_ref(key_id)
        held_local = held_redis = False
        client: Any = None
        try:
            lease.waited = entry.lock.locked()
            try:
                await asyncio.wait_for(
                    entry.lock.acquire(), timeout=max(0.0, deadline - loop.time())
                )
                held_local = True
            except TimeoutError:
                lease.mode = "none"
                logger.warning(
                    f"{lease.key}: in-flight lock wait budget exhausted after "
                    f"{self._wait_seconds:g}s behind a local holder; no lock held, "
                    "the caller decides"
                )
            # Stage 2 (the local holder only): cross-worker exclusion via SET NX,
            # every command and sleep trimmed to the remaining budget.
            if held_local:
                try:
                    client = await self._redis(deadline)
                    if client is not None:
                        held_redis = await self._acquire_redis(client, lease, deadline)
                except _BudgetExhausted:
                    # Distinct from a Redis failure: Redis answered, the budget
                    # simply ran out behind another worker's holder. No degrade.
                    lease.mode = "none"
                    logger.warning(
                        f"{lease.key}: in-flight lock wait budget exhausted after "
                        f"{self._wait_seconds:g}s behind another worker's holder; no "
                        "lock held, the caller decides"
                    )
                except _REDIS_DEGRADE_ERRORS as e:
                    self._degrade(e)
                    client = None
                except Exception as e:  # never fail the request; but do not call a bug an outage
                    self._degrade(e, unexpected=True)
                    client = None
                if not held_redis and lease.mode != "none":
                    lease.mode = "local"
        except BaseException:
            if held_local:
                entry.lock.release()
            self._local_unref(key_id)
            raise

        try:
            yield lease
        finally:
            try:
                if held_redis:
                    await self._release_redis(client, lease)
            finally:
                # ALWAYS runs, even when the release EVAL is cancelled mid-await
                # (codex round-6): a locked local entry would otherwise outlive
                # the request and 409 every later same-worker request for this
                # id until the worker recycled -- the Redis TTL cannot clear a
                # process-local lock. The cancellation still propagates; the
                # Redis key, if the EVAL never landed, clears at its TTL.
                if held_local:
                    entry.lock.release()
                self._local_unref(key_id)
