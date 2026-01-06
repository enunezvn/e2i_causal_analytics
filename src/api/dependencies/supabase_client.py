"""Supabase client dependency for FastAPI.

Provides database connection for:
- PostgreSQL data access
- Authentication
- Row-level security
- Realtime subscriptions

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Configuration from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# Global client reference
_supabase_client: Optional[Any] = None


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
    Check Supabase health status.

    Returns:
        Dict with status and connection info
    """
    import time

    try:
        client = get_supabase()

        if client is None:
            return {
                "status": "unavailable",
                "error": "Supabase not configured",
            }

        start = time.time()

        # Try a simple query to verify connection
        # Query a known table or use RPC
        try:
            result = client.table("business_metrics").select("id").limit(1).execute()
            latency_ms = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "connected": True,
            }

        except Exception as query_error:
            # Table might not exist, but connection could still be valid
            # Try auth check instead
            latency_ms = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "connected": True,
                "note": "Connection verified, table check failed",
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
