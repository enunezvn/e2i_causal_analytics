"""Cross-worker in-flight lock for build-once endpoints (#1993).

``POST /expert-reviews/{review_id}/assessment`` builds an LLM assessment when
the row has none cached. Two concurrent uncached requests for one review both
built it (measured on main: builds=2, both ``cached=False``, last write wins):
two bills for one value. The API runs ``gunicorn --workers 2`` (measured on the
live container), so a process-local ``asyncio.Lock`` covers only the requests
that land on the same worker; the cross-worker lock lives in Redis.

Redis path (the app's already-initialised ``redis.asyncio`` client):

- acquire: ``SET <prefix>:<id> <uuid-token> NX PX <ttl_ms>``. The TTL tracks
  gunicorn ``--timeout 120``: no holder outlives its own request, so a dead
  worker's key clears on its own.
- wait: a loser polls ``GET`` every ``poll_seconds`` until the key is gone
  (released or expired) and then retries the SET. The wait is bounded by the
  TTL, after which the request proceeds UNLOCKED (logged): the lock is a cost
  optimisation and must never fail or hang a request.
- release: compare-and-delete Lua (``GET == token -> DEL``) in ``finally``, so
  a lease that outlived its TTL can never delete the NEXT holder's key.

Degraded path (Redis unavailable, erroring, or hanging): fall back to a
process-local ``asyncio.Lock`` per id (same-worker requests still serialise;
cross-worker duplicates are possible: the pre-#1993 behaviour, degraded by
design), warn ONCE per process, and stop probing Redis for
``degrade_cooldown_seconds`` so a dead Redis costs one bounded attempt, not a
timeout per poll. Every Redis command is capped by ``op_timeout_seconds``. The
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
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_MS = 120_000  # gunicorn --timeout 120 (measured on the live container)
DEFAULT_POLL_SECONDS = 0.2
DEFAULT_OP_TIMEOUT_SECONDS = 1.0  # cap per Redis command; never the 3 s socket wait per poll
DEFAULT_DEGRADE_COOLDOWN_SECONDS = 30.0  # after a failure, skip Redis this long (no storm)

# Compare-and-delete: only the holder whose token is still stored may delete.
_RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end return 0"
)

# Zero-arg async factory yielding a Redis client (injectable for tests).
RedisFactory = Callable[[], Awaitable[Any]]


async def _default_redis_factory() -> Any:
    """The app's already-initialised async Redis client, WITHOUT ``get_redis()``:
    that would run ``init_redis()``'s multi-attempt backoff on the request path
    when startup ran in Redis-degraded mode. Not initialised -> degrade."""
    from src.api.dependencies import redis_client

    client = redis_client._redis_client
    if client is None:
        raise RuntimeError("Redis client not initialised (startup degraded mode)")
    return client


@dataclass
class Lease:
    """What ``InflightLock.hold`` yields.

    ``mode``: ``"redis"`` (cross-worker), ``"local"`` (this process only) or
    ``"none"`` (the bounded wait was exhausted; running unlocked).
    ``waited``: another request held this id when we arrived, so the caller
    should re-read whatever that request may have produced before building.
    """

    key: str
    token: str
    mode: str = "redis"
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

    async def _redis(self) -> Optional[Any]:
        if time.monotonic() < self._degraded_until:
            return None
        try:
            return await asyncio.wait_for(self._redis_factory(), timeout=self.op_timeout_seconds)
        except Exception as e:
            self._degrade(e)
            return None

    def _degrade(self, exc: BaseException) -> None:
        self._degraded_until = time.monotonic() + self.degrade_cooldown_seconds
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

    async def _op(self, awaitable: Awaitable[Any]) -> Any:
        return await asyncio.wait_for(awaitable, timeout=self.op_timeout_seconds)

    async def _acquire_redis(self, client: Any, lease: Lease) -> bool:
        """True = acquired. False = the bounded wait was exhausted. Raises on a
        Redis error (the caller degrades to the local lock)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._wait_seconds
        while True:
            if await self._op(client.set(lease.key, lease.token, nx=True, px=self.ttl_ms)):
                return True
            lease.waited = True
            # Poll for the holder's release (or its TTL expiry), then retry the SET.
            while await self._op(client.get(lease.key)) is not None:
                if loop.time() >= deadline:
                    return False
                await asyncio.sleep(self.poll_seconds)
            if loop.time() >= deadline:
                return False

    async def _release_redis(self, client: Any, lease: Lease) -> None:
        try:
            await self._op(client.eval(_RELEASE_LUA, 1, lease.key, lease.token))
        except Exception as e:  # the TTL bounds the leaked key; never fail the response
            self._degrade(e)

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
        client = await self._redis()
        held_redis = False
        if client is not None:
            try:
                held_redis = await self._acquire_redis(client, lease)
                if not held_redis:
                    lease.mode = "none"
                    logger.warning(
                        f"{lease.key}: in-flight lock wait exceeded {self._wait_seconds:g}s; "
                        "proceeding unlocked"
                    )
            except Exception as e:
                self._degrade(e)
                client = None

        entry: Optional[_LocalEntry] = None
        held_local = False
        if not held_redis and lease.mode != "none":
            lease.mode = "local"
            entry = self._local_ref(key_id)
            try:
                lease.waited = lease.waited or entry.lock.locked()
                await asyncio.wait_for(entry.lock.acquire(), timeout=self._wait_seconds)
                held_local = True
            except TimeoutError:
                lease.mode = "none"
                logger.warning(
                    f"{lease.key}: local in-flight lock wait exceeded {self._wait_seconds:g}s; "
                    "proceeding unlocked"
                )
            except BaseException:
                self._local_unref(key_id)
                raise

        try:
            yield lease
        finally:
            if held_redis:
                await self._release_redis(client, lease)
            if held_local and entry is not None:
                entry.lock.release()
            if entry is not None:
                self._local_unref(key_id)
