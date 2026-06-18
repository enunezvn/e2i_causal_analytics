"""Generic Redis-backed, cross-worker job store for async submit->poll endpoints.

The API runs multiple gunicorn workers (``--workers 2``). A module-level ``dict``
job cache lives in ONE worker's process, so a POST (submit) handled by worker A
and a GET (poll) handled by worker B hit different caches -> the poll spuriously
404s even though the job is running fine. This store persists each job as JSON in
Redis (shared across workers) under ``"<prefix>:<id>"`` with a TTL, so any worker
can read any job.

When Redis is unavailable (tests, local, a transient outage) it transparently
falls back to a bounded in-process dict so a single worker still works and a
request degrades gracefully instead of 500-ing. Writes are mirrored to the
fallback so a same-worker read still succeeds if Redis is only intermittently
reachable. The fallback re-creates the per-worker cross-worker gap by design (it
is a single-process degraded mode), which ``is_durable()`` surfaces for health.

Mirrors the resilience design of ``segments._DurableAnalysesStore`` but is
generic over the stored Pydantic model and needs no enumeration index.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Errors that should DEGRADE to the in-memory fallback rather than surface a 500.
# RuntimeError covers "redis not initialised" raised by get_redis().
try:  # pragma: no cover - import shape varies by redis version
    from redis.exceptions import RedisError as _RedisError

    _REDIS_DEGRADE_ERRORS: tuple = (
        _RedisError,
        ConnectionError,
        TimeoutError,
        OSError,
        RuntimeError,
    )
except Exception:  # pragma: no cover - redis not installed
    _REDIS_DEGRADE_ERRORS = (ConnectionError, TimeoutError, OSError, RuntimeError)

# A stored record can be undecodable independently of Redis transport: a truncated
# write, manual tampering, or — most realistically — SCHEMA EVOLUTION during a
# rolling deploy (an old-format record written by the previous image, read by the
# new code with a changed model). Treat these as a miss/degrade, never a 500.
# (Non-finite floats are NOT a concern here: model_dump_json serializes NaN/inf to
# null and the Optional float fields round-trip back to None.)
_RECORD_DECODE_ERRORS: tuple = (ValidationError, json.JSONDecodeError, ValueError)

DEFAULT_TTL_SECONDS = 3600  # 1h: ample for a multi-minute job + the FE poll/view window
DEFAULT_MAX_FALLBACK = 256

# Zero-arg async factory yielding a Redis client (defaults to the app's canonical
# ``get_redis``; injectable for tests).
RedisFactory = Callable[[], Awaitable[Any]]


async def _default_redis_factory() -> Any:
    """Return the app's canonical async Redis client (lazy import avoids a hard
    Redis dependency at import time and import cycles)."""
    from src.api.dependencies.redis_client import get_redis

    return await get_redis()


class DurableJobStore(Generic[T]):
    """Cross-worker job store: Redis-backed with a bounded in-memory fallback.

    Stores a single Pydantic model per job id as JSON. Construct one per logical
    job type (its ``prefix`` namespaces the Redis keys).
    """

    def __init__(
        self,
        prefix: str,
        model_cls: Type[T],
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_fallback: int = DEFAULT_MAX_FALLBACK,
        redis_factory: Optional[RedisFactory] = None,
    ) -> None:
        self.prefix = prefix
        self.model_cls = model_cls
        self.ttl_seconds = ttl_seconds
        self.max_fallback = max_fallback
        self._redis_factory: RedisFactory = redis_factory or _default_redis_factory
        # In-process fallback (FIFO-bounded) used when Redis is unavailable.
        self._memory: "OrderedDict[str, str]" = OrderedDict()
        # Last-observed storage mode, so /health can surface silent per-worker
        # degradation (a process serving from the in-memory fallback re-creates
        # the cross-worker 404 this store exists to fix). None until first probe.
        self._last_durable: Optional[bool] = None

    def _key(self, job_id: str) -> str:
        return f"{self.prefix}:{job_id}"

    async def _redis(self) -> Optional[Any]:
        try:
            client = await self._redis_factory()
            self._last_durable = True
            return client
        except _REDIS_DEGRADE_ERRORS as e:
            self._last_durable = False
            logger.warning(f"{self.prefix}: Redis unavailable, using in-memory fallback: {e}")
            return None
        except Exception as e:  # pragma: no cover - defensive
            self._last_durable = False
            logger.warning(f"{self.prefix}: unexpected Redis factory error, degrading: {e}")
            return None

    def _mem_set(self, job_id: str, raw: str) -> None:
        self._memory[job_id] = raw
        self._memory.move_to_end(job_id)
        while len(self._memory) > self.max_fallback:
            self._memory.popitem(last=False)

    async def set(self, job_id: str, model: T) -> None:
        """Persist ``model`` under ``job_id`` (Redis + memory mirror)."""
        try:
            raw = model.model_dump_json()
        except Exception as e:  # noqa: BLE001 - never let a serialization edge case 500/crash a task
            logger.error(f"{self.prefix}: failed to serialize job {job_id}, not caching: {e}")
            return
        client = await self._redis()
        if client is not None:
            try:
                await client.set(self._key(job_id), raw, ex=self.ttl_seconds)
            except _REDIS_DEGRADE_ERRORS as e:
                self._last_durable = False
                logger.warning(f"{self.prefix}: Redis SET failed, mirroring to memory only: {e}")
        # Always mirror so a same-worker read succeeds even if Redis is flaky.
        self._mem_set(job_id, raw)

    async def get(self, job_id: str) -> Optional[T]:
        """Return the model for ``job_id`` (Redis first, then memory), or None."""
        client = await self._redis()
        if client is not None:
            try:
                raw = await client.get(self._key(job_id))
                if raw is not None:
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode("utf-8")
                    return self.model_cls.model_validate_json(raw)
            except _REDIS_DEGRADE_ERRORS as e:
                self._last_durable = False
                logger.warning(f"{self.prefix}: Redis GET failed, trying memory: {e}")
            except _RECORD_DECODE_ERRORS as e:
                # Undecodable stored record (corruption / schema drift): treat as a
                # miss and fall through to memory rather than 500.
                logger.warning(
                    f"{self.prefix}: undecodable Redis record {job_id}, treating as miss: {e}"
                )
        raw_mem = self._memory.get(job_id)
        if raw_mem is not None:
            try:
                return self.model_cls.model_validate_json(raw_mem)
            except _RECORD_DECODE_ERRORS as e:
                # Poison record in the fallback: drop it and report a miss.
                logger.warning(f"{self.prefix}: undecodable memory record {job_id}, dropping: {e}")
                self._memory.pop(job_id, None)
        return None

    async def is_durable(self) -> bool:
        """True if this worker can currently reach Redis (probes the factory)."""
        return (await self._redis()) is not None
