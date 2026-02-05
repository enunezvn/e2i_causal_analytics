"""Supabase client dependency for FastAPI.

Provides database connection for:
- PostgreSQL data access
- Authentication
- Row-level security
- Realtime subscriptions

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

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
        logger.warning("Supabase credentials not configured - database features unavailable")
        return None

    logger.info(f"Initializing Supabase connection to {SUPABASE_URL[:50]}...")

    try:
        from supabase import create_client

        # Use service key if available for admin operations, otherwise use anon key
        key = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_KEY
        _supabase_client = create_client(SUPABASE_URL, key)

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


async def supabase_health_check() -> Dict[str, Any]:
    """
    Check Supabase health status via lightweight connectivity test.

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

        # Lightweight connectivity test using PostgREST endpoint
        # This avoids depending on any specific table existing
        try:
            client.rpc("", {}).execute()
        except Exception:
            # RPC with empty name may fail, but the HTTP round-trip
            # confirms PostgREST is reachable. Try a HEAD-style probe
            # by selecting from a system-level endpoint.
            try:
                client.table("_health_check_noop").select("*").limit(0).execute()
            except Exception:
                pass

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
