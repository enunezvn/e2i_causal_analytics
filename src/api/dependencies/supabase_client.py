"""Supabase client dependency for FastAPI.

Provides database connection for:
- PostgreSQL data access
- Authentication
- Row-level security
- Realtime subscriptions

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)

# Configuration from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "") or os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# Global client reference
_supabase_client: Optional[Any] = None

# Circuit breaker for health checks
_health_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(failure_threshold=3, reset_timeout_seconds=30.0)
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before=before_log(logger, logging.WARNING),
    reraise=True,
)
def init_supabase() -> Optional[Any]:
    """
    Initialize Supabase client.

    Returns:
        Supabase client instance or None if not configured

    Raises:
        ConnectionError: If Supabase connection fails with valid credentials
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        # #471: surface per-var truthiness so operators can see WHICH
        # of SUPABASE_URL / SUPABASE_KEY / SUPABASE_SERVICE_KEY is
        # missing (was just "not configured" — collapsed all 3 cases).
        from src.utils.env_diagnostics import env_state

        logger.warning(
            "Supabase credentials not configured - database features unavailable. "
            "Diagnostic: %s; %s; %s. If .env contains these, ensure "
            "load_dotenv() ran before module import.",
            env_state("SUPABASE_URL"),
            env_state("SUPABASE_KEY"),
            env_state("SUPABASE_SERVICE_KEY"),
        )
        return None

    logger.info(f"Initializing Supabase connection to {SUPABASE_URL[:50]}...")

    try:
        import httpx
        from supabase import create_client
        from supabase.client import ClientOptions

        # Use service key if available for admin operations, otherwise use anon key
        key = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_KEY

        # Pass an explicit ClientOptions to avoid supabase-py's default
        # `timeout=<int>` / legacy verify path-string forwarding to httpx,
        # which now emits DeprecationWarnings on httpx >= 0.27.
        options = ClientOptions(
            postgrest_client_timeout=httpx.Timeout(30.0, connect=10.0),
            storage_client_timeout=30,
            function_client_timeout=30,
            schema="public",
        )
        _supabase_client = create_client(SUPABASE_URL, key, options=options)

        # Verify connection by checking auth
        logger.info("Supabase client initialized successfully")

        return _supabase_client

    except ImportError:
        logger.warning("supabase package not installed - database features unavailable")
        return None

    except Exception as e:
        _supabase_client = None
        logger.error(f"Failed to connect to Supabase: {e}")
        raise ConnectionError(f"Supabase connection failed: {e}") from e


def get_supabase() -> Optional[Any]:
    """
    Get Supabase client instance.

    Returns:
        Supabase client or None if unavailable
    """
    global _supabase_client

    if _supabase_client is None:
        try:
            _supabase_client = init_supabase()
        except Exception:
            return None

    return _supabase_client


def close_supabase() -> None:
    """Close Supabase client."""
    global _supabase_client

    if _supabase_client is not None:
        logger.info("Closing Supabase connection")
        # Supabase client doesn't require explicit closing
        _supabase_client = None
        logger.info("Supabase connection closed")


# Transport-level failures mean the PostgREST/Postgres round-trip could NOT be
# made (server down, DNS, refused, timed out) -> genuinely unhealthy. Any other
# exception (a PostgREST/API error such as "function not found" or "relation
# does not exist") means the HTTP round-trip SUCCEEDED -> the service is
# reachable -> healthy. This is the distinction the old fail-open code lacked.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _probe_postgrest_reachable(client: Any) -> None:
    """Blocking PostgREST reachability probe (run via asyncio.to_thread).

    Returns normally if the service is reachable. Raises a transport error only
    when EVERY probe fails at the transport layer (server unreachable).

    Two probes are attempted because the empty-name RPC may legitimately return
    a PostgREST API error even when the server is up; a second select probe then
    confirms reachability. An API-level error from either probe is treated as
    "reachable" (the round-trip completed) and swallowed.
    """
    try:
        import httpx

        transport_errors: tuple[type[BaseException], ...] = (
            *_TRANSPORT_ERRORS,
            httpx.TransportError,
        )
    except Exception:  # httpx should always be present, but degrade gracefully
        transport_errors = _TRANSPORT_ERRORS

    try:
        client.rpc("", {}).execute()
        return  # Round-trip completed.
    except transport_errors:
        # Transport-level failure on probe 1 — try the second probe before
        # concluding the service is unreachable.
        client.table("_health_check_noop").select("*").limit(0).execute()
        return  # If probe 2's round-trip completes, the service is reachable.
    except Exception:
        # API-level error (e.g. PGRST function/relation not found): the HTTP
        # round-trip succeeded, so PostgREST is reachable -> healthy.
        return


async def supabase_health_check() -> Dict[str, Any]:
    """
    Check Supabase health status via a real PostgREST connectivity probe.

    Fails CLOSED: if the round-trip to PostgREST/Postgres cannot be made, the
    status is ``unhealthy`` (the previous implementation incorrectly reported
    ``healthy`` even when the server was unreachable). The blocking supabase-py
    client call is offloaded to a worker thread so it does not block the event
    loop.

    Returns:
        Dict with status and connection info
    """
    import time

    if not _health_circuit_breaker.allow_request():
        return {"status": "circuit_open"}

    try:
        client = get_supabase()

        if client is None:
            return {
                "status": "unavailable",
                "error": "Supabase not configured",
            }

        start = time.time()

        # Run the synchronous PostgREST probe off the event loop. A transport
        # error propagates here -> caught below -> unhealthy (fail closed).
        await asyncio.to_thread(_probe_postgrest_reachable, client)

        latency_ms = (time.time() - start) * 1000

        _health_circuit_breaker.record_success()

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "connected": True,
        }

    except Exception as e:
        _health_circuit_breaker.record_failure()
        return {
            "status": "unhealthy",
            "error": str(e),
        }
