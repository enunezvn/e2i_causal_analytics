"""Health Client for real system health checks.

Provides a real implementation of the HealthClient protocol that checks
actual system components via Supabase and other services.

This replaces the mock fallback behavior in ComponentHealthNode when
no health_client is injected, ensuring tests use real system checks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SupabaseHealthClient:
    """Real health client that checks actual system components.

    Checks health of:
    - Database (Supabase PostgreSQL)
    - Cache (Redis)
    - Vector store (if available)
    - API gateway (FastAPI health endpoint)
    - Message queue (if available)

    Usage:
        client = SupabaseHealthClient()
        result = await client.check("/health/db")
        print(result)  # {"ok": True, "latency_ms": 15}

        # Or for testing, use the factory function:
        client = get_health_client_for_testing()
    """

    def __init__(
        self,
        supabase_url: str | None = None,
        redis_url: str | None = None,
        api_base_url: str | None = None,
        timeout_seconds: float = 5.0,
    ):
        """Initialize health client.

        Args:
            supabase_url: Supabase project URL (or from SUPABASE_URL env var)
            redis_url: Redis connection URL (or from REDIS_URL env var)
            api_base_url: Base URL for API health checks (default: localhost:8000)
            timeout_seconds: Timeout for health checks
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL", "")
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.timeout = timeout_seconds
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def check(self, endpoint: str) -> dict[str, Any]:
        """Check health of an endpoint.

        Args:
            endpoint: Health endpoint path (e.g., "/health/db")

        Returns:
            Dict with 'ok' (bool), 'latency_ms' (int), and optional 'error' (str)
        """
        start_time = time.time()

        try:
            if endpoint == "/health/db":
                return await self._check_database()
            elif endpoint == "/health/cache":
                return await self._check_cache()
            elif endpoint == "/health/vectors":
                return await self._check_vector_store()
            elif endpoint == "/health/api":
                return await self._check_api()
            elif endpoint == "/health/queue":
                return await self._check_message_queue()
            else:
                # Generic API health check
                return await self._check_api_endpoint(endpoint)

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Health check failed for {endpoint}: {e}")
            return {
                "ok": False,
                "latency_ms": latency_ms,
                "error": str(e),
            }

    async def _check_database(self) -> dict[str, Any]:
        """Check Supabase database health.

        The previous probe called ``rpc("version")`` — which resolves to a
        non-existent ``public.version()`` (PGRST202 404) — AND ``await``-ed the
        Supabase client (the sync ``Client`` is not awaitable). Both falsely
        reported the DB unhealthy though Postgres is fully up. Reuse the robust
        httpx REST probe instead: a 200/401 from ``/rest/v1/`` means the service
        (and the Postgres behind it) is reachable, independent of the sync/async
        client surface.
        """
        return await self._check_supabase_rest_health()

    async def _check_supabase_rest_health(self) -> dict[str, Any]:
        """Check Supabase REST API health directly."""
        start_time = time.time()

        if not self.supabase_url:
            return {
                "ok": False,
                "latency_ms": 0,
                "error": "SUPABASE_URL not configured",
            }

        try:
            client = await self._get_http_client()
            # Hit the REST API health endpoint
            url = f"{self.supabase_url}/rest/v1/"
            response = await client.get(url)

            latency_ms = int((time.time() - start_time) * 1000)

            # 401 is expected without auth, but it means the service is up
            if response.status_code in (200, 401):
                return {
                    "ok": True,
                    "latency_ms": latency_ms,
                }
            else:
                return {
                    "ok": False,
                    "latency_ms": latency_ms,
                    "error": f"HTTP {response.status_code}",
                }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "ok": False,
                "latency_ms": latency_ms,
                "error": str(e),
            }

    async def _check_cache(self) -> dict[str, Any]:
        """Check Redis cache health."""
        start_time = time.time()

        try:
            import redis.asyncio as redis

            # Parse Redis URL
            client = redis.from_url(self.redis_url)

            # Ping Redis
            pong = await asyncio.wait_for(client.ping(), timeout=self.timeout)

            latency_ms = int((time.time() - start_time) * 1000)
            await client.close()

            return {
                "ok": bool(pong),
                "latency_ms": latency_ms,
            }

        except ImportError:
            return {
                "ok": False,
                "latency_ms": 0,
                "error": "redis package not installed",
            }
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "ok": False,
                "degraded": True,  # Cache is optional
                "latency_ms": latency_ms,
                "error": str(e),
            }

    async def _check_vector_store(self) -> dict[str, Any]:
        """Check vector store health (pgvector in Postgres).

        The vector store is pgvector INSIDE Postgres (extension ``vector`` backing
        ``rag_document_chunks``) — there is NO separate vector service, so the old
        ``VECTOR_STORE_URL`` probe was structurally unset and degraded unconditionally
        (the false alarm). Probe the pgvector-backed table over PostgREST with httpx
        (robust to the sync/async Supabase client). ``200/206`` = the route resolves
        (healthy); ``404`` = the table is absent (real failure). A ``401/403`` is
        ambiguous — it is returned for a restricted-but-present table AND for an
        invalid/missing JWT that rejects everything — so it is disambiguated with a
        second probe of a guaranteed-nonexistent table: if THAT returns ``404`` the
        token is valid and the real table exists (healthy); if it ALSO ``401/403``s
        the auth is broken and the table is unverifiable (degraded, not masked).
        """
        start_time = time.time()

        if not self.supabase_url:
            return {
                "ok": False,
                "degraded": True,
                "latency_ms": 0,
                "error": "SUPABASE_URL not configured",
            }

        try:
            client = await self._get_http_client()
            apikey = os.getenv("SUPABASE_ANON_KEY", "") or os.getenv("SUPABASE_KEY", "")
            headers = {"apikey": apikey} if apikey else None
            url = f"{self.supabase_url}/rest/v1/rag_document_chunks?limit=1"
            response = await client.get(url, headers=headers)
            code = response.status_code

            # Definitive signals (measured against the live PostgREST, 2026-06-15):
            #   200/206 -> the route resolves and is readable -> healthy.
            #   404     -> PGRST205, the table is absent -> real failure.
            if code in (200, 206):
                latency_ms = int((time.time() - start_time) * 1000)
                return {"ok": True, "latency_ms": latency_ms}

            # 401/403 is AMBIGUOUS: it is returned both for a restricted-but-present
            # table (RLS/grant denies the probe role) AND for an invalid/missing JWT
            # that rejects every request before table resolution. Naively treating
            # 401 as healthy would mask an absent/inaccessible table. Disambiguate
            # with a guaranteed-nonexistent table: PostgREST resolves the table name
            # AFTER JWT validation, so with a VALID token an absent table returns 404
            # while a present-but-restricted table returns 401/403.
            if code in (401, 403):
                bogus_url = f"{self.supabase_url}/rest/v1/__vector_health_probe_absent__?limit=1"
                bogus = await client.get(bogus_url, headers=headers)
                latency_ms = int((time.time() - start_time) * 1000)
                if bogus.status_code == 404:
                    # JWT is valid (the bogus name resolved to 404), so the original
                    # 401/403 proves rag_document_chunks EXISTS and is merely
                    # restricted to the probe role -> healthy.
                    return {"ok": True, "latency_ms": latency_ms}
                # The bogus probe also 401/403'd: the token itself is invalid or
                # missing, so the table's status is unverifiable. Surface it as
                # degraded rather than masking a possibly-absent table.
                return {
                    "ok": False,
                    "degraded": True,
                    "latency_ms": latency_ms,
                    "error": (
                        f"vector store unverifiable (table HTTP {code}, "
                        f"auth probe HTTP {bogus.status_code})"
                    ),
                }

            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "ok": False,
                "degraded": True,
                "latency_ms": latency_ms,
                "error": f"HTTP {code}",
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "ok": False,
                "degraded": True,
                "latency_ms": latency_ms,
                "error": str(e),
            }

    async def _check_api(self) -> dict[str, Any]:
        """Check API gateway health."""
        return await self._check_api_endpoint("/health")

    async def _check_api_endpoint(self, endpoint: str) -> dict[str, Any]:
        """Check a specific API endpoint."""
        start_time = time.time()

        try:
            client = await self._get_http_client()
            url = f"{self.api_base_url}{endpoint}"
            response = await client.get(url)

            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "ok": response.status_code == 200,
                "latency_ms": latency_ms,
                "status_code": response.status_code,
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "ok": False,
                "latency_ms": latency_ms,
                "error": str(e),
            }

    async def _check_message_queue(self) -> dict[str, Any]:
        """Check message queue health."""
        start_time = time.time()

        # Message queue is optional, return degraded if not configured
        queue_url = os.getenv("MESSAGE_QUEUE_URL") or os.getenv("CELERY_BROKER_URL")

        if not queue_url:
            return {
                "ok": False,
                "degraded": True,
                "latency_ms": 0,
                "error": "MESSAGE_QUEUE_URL not configured",
            }

        # For Redis-based Celery, use the cache check
        if "redis" in queue_url.lower():
            result = await self._check_cache()
            result["source"] = "redis (message queue)"
            return result

        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "ok": False,
            "degraded": True,
            "latency_ms": latency_ms,
            "error": "Unsupported message queue type",
        }


def get_health_client_for_testing() -> SupabaseHealthClient:
    """Get a health client configured for testing.

    This factory function creates a SupabaseHealthClient with
    appropriate defaults for the test environment.

    Returns:
        SupabaseHealthClient instance
    """
    return SupabaseHealthClient(
        timeout_seconds=10.0,  # Longer timeout for testing
    )


class SimpleHealthClient:
    """Simplified health client for minimal testing scenarios.

    This client performs basic connectivity checks without requiring
    full Supabase/Redis infrastructure.
    """

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self._http_client: httpx.AsyncClient | None = None

    async def check(self, endpoint: str) -> dict[str, Any]:
        """Perform a simple health check."""
        start_time = time.time()

        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=5.0)

            # All endpoints go to the local API
            url = f"{self.api_base_url}{endpoint}"
            response = await self._http_client.get(url)

            latency_ms = int((time.time() - start_time) * 1000)

            # Introduce variance - never return exactly the same values
            # This helps distinguish from mock data
            return {
                "ok": response.status_code == 200,
                "latency_ms": latency_ms,
                "status_code": response.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "ok": False,
                "latency_ms": latency_ms,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
