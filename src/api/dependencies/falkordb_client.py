"""FalkorDB client dependency for FastAPI.

Provides graph database connection for:
- Knowledge graph storage
- Causal path queries
- Entity relationship traversal

Note: FalkorDB uses Redis protocol on a different port.

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Configuration from environment
FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT", "6381"))
FALKORDB_GRAPH_NAME = os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")

# Global client reference
_falkordb_client: Optional[Any] = None
_graph: Optional[Any] = None


async def init_falkordb() -> Any:
    """
    Initialize FalkorDB connection.

    Returns:
        FalkorDB client instance

    Raises:
        ConnectionError: If FalkorDB connection fails
    """
    global _falkordb_client, _graph

    if _falkordb_client is not None:
        return _falkordb_client

    # Read env vars at call time to support runtime configuration
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6381"))
    graph_name = os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")

    logger.info(f"Initializing FalkorDB connection to {host}:{port}")

    try:
        from falkordb import FalkorDB

        _falkordb_client = FalkorDB(host=host, port=port)
        _graph = _falkordb_client.select_graph(graph_name)

        # Verify connection by listing graphs
        graphs = _falkordb_client.list_graphs()
        logger.info(f"FalkorDB connected. Available graphs: {graphs}")

        return _falkordb_client

    except ImportError:
        logger.warning("falkordb package not installed - graph features unavailable")
        return None

    except Exception as e:
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

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "graphs": graphs,
            "current_graph": FALKORDB_GRAPH_NAME,
            "node_count": node_count,
            "edge_count": edge_count,
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
