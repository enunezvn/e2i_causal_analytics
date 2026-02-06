"""FalkorDB client dependency for FastAPI.

Provides graph database connection for:
- Knowledge graph storage
- Causal path queries
- Entity relationship traversal

Note: FalkorDB uses Redis protocol on a different port.

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


def _parse_falkordb_config() -> tuple[str, int, str | None]:
    """Derive host/port/password from FALKORDB_URL if set, else fall back to individual env vars."""
    url = os.environ.get("FALKORDB_URL")
    if url:
        parsed = urlparse(url)
        return parsed.hostname or "localhost", parsed.port or 6379, parsed.password
    return (
        os.environ.get("FALKORDB_HOST", "localhost"),
        int(os.environ.get("FALKORDB_PORT", "6379")),
        os.environ.get("FALKORDB_PASSWORD"),
    )


# Configuration from environment
FALKORDB_HOST, FALKORDB_PORT, FALKORDB_PASSWORD = _parse_falkordb_config()
FALKORDB_GRAPH_NAME = os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")

# Global client reference
_falkordb_client: Optional[Any] = None
_graph: Optional[Any] = None

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
async def init_falkordb() -> Any:
    """
    Initialize FalkorDB connection.

    Returns:
        FalkorDB client instance

    Raises:
        ConnectionError: If FalkorDB connection fails after retries
    """
    global _falkordb_client, _graph

    if _falkordb_client is not None:
        try:
            _falkordb_client.list_graphs()
            return _falkordb_client
        except Exception:
            _falkordb_client = None
            _graph = None

    # Read env vars at call time to support runtime configuration
    host, port, password = _parse_falkordb_config()
    graph_name = os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")

    logger.info(f"Initializing FalkorDB connection to {host}:{port}")

    try:
        from falkordb import FalkorDB

        _falkordb_client = FalkorDB(host=host, port=port, password=password)
        _graph = _falkordb_client.select_graph(graph_name)

        # Verify connection by listing graphs
        graphs = _falkordb_client.list_graphs()
        logger.info(f"FalkorDB connected. Available graphs: {graphs}")

        return _falkordb_client

    except ImportError:
        logger.warning("falkordb package not installed - graph features unavailable")
        return None

    except Exception as e:
        _falkordb_client = None
        _graph = None
        logger.error(f"Failed to connect to FalkorDB: {e}")
        raise ConnectionError(f"FalkorDB connection failed: {e}") from e


async def get_falkordb() -> Optional[Any]:
    """
    Get FalkorDB client instance.

    Returns:
        FalkorDB client or None if unavailable
    """
    global _falkordb_client

    if _falkordb_client is None:
        try:
            _falkordb_client = await init_falkordb()
        except Exception:
            return None

    return _falkordb_client


async def get_graph() -> Optional[Any]:
    """
    Get FalkorDB graph instance.

    Returns:
        FalkorDB graph or None if unavailable
    """
    global _graph

    if _graph is None:
        client = await get_falkordb()
        if client:
            _graph = client.select_graph(FALKORDB_GRAPH_NAME)

    return _graph


async def close_falkordb() -> None:
    """Close FalkorDB connection."""
    global _falkordb_client, _graph

    if _falkordb_client is not None:
        logger.info("Closing FalkorDB connection")
        # FalkorDB uses Redis connection under the hood
        # Close is handled by connection pool
        _falkordb_client = None
        _graph = None
        logger.info("FalkorDB connection closed")


async def falkordb_health_check() -> Dict[str, Any]:
    """
    Check FalkorDB health status.

    Returns:
        Dict with status and graph info
    """
    import time

    if not _health_circuit_breaker.allow_request():
        return {"status": "circuit_open"}

    try:
        client = await get_falkordb()

        if client is None:
            return {
                "status": "unavailable",
                "error": "FalkorDB not configured",
            }

        start = time.time()
        graphs = client.list_graphs()
        latency_ms = (time.time() - start) * 1000

        graph = await get_graph()
        node_count = 0
        edge_count = 0

        if graph:
            try:
                result = graph.query("MATCH (n) RETURN count(n) as count")
                if result.result_set:
                    node_count = result.result_set[0][0]

                result = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
                if result.result_set:
                    edge_count = result.result_set[0][0]
            except Exception:
                pass  # Graph may be empty

        _health_circuit_breaker.record_success()

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "graphs": graphs,
            "current_graph": FALKORDB_GRAPH_NAME,
            "node_count": node_count,
            "edge_count": edge_count,
        }

    except Exception as e:
        _health_circuit_breaker.record_failure()
        return {
            "status": "unhealthy",
            "error": str(e),
        }
