"""
AgentLease — atomic Redis-based locks for shared E2I resources.

Two agents must not race on the same cohort, causal_path, model_version,
or dataset. AgentLease uses Redis SET NX PX (atomic create-if-not-exists
with millisecond TTL) — the standard single-node lock primitive — to
serialize access without distributed-consensus overhead.

Resource namespace: ``lease:{resource_type}:{resource_id}``

Holder identity is a UUID minted on acquire(), so renew() and release()
refuse to operate on a lease the caller does not own (anti-fencing).

Not a distributed lock across redis replicas; if Redis fails over to a
replica before the SET propagates, two acquirers could both succeed.
For E2I (single Redis container), this is acceptable. If multi-replica
Redis lands later, swap to Redlock.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from src.memory.services.factories import get_redis_client

logger = logging.getLogger(__name__)


class LeaseAcquisitionError(RuntimeError):
    """Raised when acquire() can't obtain the lease within the timeout."""


class AgentLease:
    """
    Async context manager and low-level API for resource leases.

    Usage:

        async with AgentLease("cohort", cohort_id, ttl_seconds=300) as lease:
            ...  # exclusive section
            # renew if work runs long:
            await lease.renew(ttl_seconds=300)

        # Or low-level:
        lease = AgentLease("cohort", cohort_id)
        if await lease.acquire(ttl_seconds=300, wait_seconds=10):
            try:
                ...
            finally:
                await lease.release()
    """

    KEY_PREFIX = "lease"

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        ttl_seconds: int = 300,
        holder_id: Optional[str] = None,
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.ttl_seconds = ttl_seconds
        # Holder identity: random per AgentLease instance unless caller provides one.
        # Used as the Redis value so release() can verify ownership.
        self.holder_id = holder_id or str(uuid.uuid4())
        self._held = False

    @property
    def key(self) -> str:
        return f"{self.KEY_PREFIX}:{self.resource_type}:{self.resource_id}"

    async def acquire(
        self,
        ttl_seconds: Optional[int] = None,
        wait_seconds: float = 0.0,
        poll_interval: float = 0.1,
    ) -> bool:
        """
        Try to acquire the lease. Returns True on success.

        If ``wait_seconds > 0``, polls until acquired or timeout. Otherwise
        returns immediately with the result of a single SET NX attempt.
        """
        ttl = ttl_seconds or self.ttl_seconds
        ttl_ms = ttl * 1000
        redis = get_redis_client()
        deadline = asyncio.get_event_loop().time() + wait_seconds

        while True:
            # SET NX PX is the atomic single-roundtrip lock primitive.
            ok = await redis.set(self.key, self.holder_id, nx=True, px=ttl_ms)
            if ok:
                self._held = True
                logger.debug(f"acquired lease {self.key} (holder={self.holder_id[:8]}, ttl={ttl}s)")
                return True
            if asyncio.get_event_loop().time() >= deadline:
                return False
            await asyncio.sleep(poll_interval)

    async def renew(self, ttl_seconds: Optional[int] = None) -> bool:
        """
        Extend the lease TTL. Returns True if we still own it and renewal succeeded.

        Atomic check-and-set via Lua to prevent renewing someone else's lease
        (anti-fencing: TTL expired, another holder took over, original holder
        wakes up and tries to renew — must fail).
        """
        if not self._held:
            return False
        ttl = ttl_seconds or self.ttl_seconds
        ttl_ms = ttl * 1000
        redis = get_redis_client()
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "  return redis.call('pexpire', KEYS[1], ARGV[2]) "
            "else return 0 end"
        )
        result = await redis.eval(script, 1, self.key, self.holder_id, ttl_ms)
        renewed = bool(result)
        if not renewed:
            self._held = False
            logger.warning(f"lease {self.key} lost during renew (holder={self.holder_id[:8]})")
        return renewed

    async def release(self) -> bool:
        """
        Release the lease. Returns True if we owned it AND release succeeded.

        Atomic ownership-check + delete via Lua.
        """
        if not self._held:
            return False
        redis = get_redis_client()
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "  return redis.call('del', KEYS[1]) "
            "else return 0 end"
        )
        result = await redis.eval(script, 1, self.key, self.holder_id)
        released = bool(result)
        self._held = False
        if released:
            logger.debug(f"released lease {self.key}")
        else:
            logger.warning(f"lease {self.key} not released (holder mismatch or expired)")
        return released

    async def is_held_by(self, holder_id: str) -> bool:
        """Whether the lease key currently maps to the given holder."""
        redis = get_redis_client()
        current = await redis.get(self.key)
        return current == holder_id

    async def current_holder(self) -> Optional[str]:
        """Return the holder_id currently owning the lease, or None."""
        redis = get_redis_client()
        return await redis.get(self.key)

    # Context manager — raises LeaseAcquisitionError on failure (no silent skip).
    async def __aenter__(self) -> "AgentLease":
        ok = await self.acquire(ttl_seconds=self.ttl_seconds)
        if not ok:
            raise LeaseAcquisitionError(
                f"could not acquire lease {self.key} (current holder unknown or held by another)"
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()
