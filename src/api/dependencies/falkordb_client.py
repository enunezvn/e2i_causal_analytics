"""FalkorDB client dependency for FastAPI.

Provides graph database connection for:
- Knowledge graph storage
- Causal path queries
- Entity relationship traversal

Note: FalkorDB uses Redis protocol on a different port.

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import logging
import os
import time
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

# Cache for the (expensive) node/edge count diagnostics. Full-graph
# ``count(n)`` / ``count(r)`` scans are O(graph) and must NOT run on the hot
# readiness path (it's polled frequently by orchestrators). They live in
# ``falkordb_diagnostics()`` instead, behind this short TTL cache.
_DIAGNOSTICS_TTL_SECONDS = 60.0
_diagnostics_cache: Optional[Dict[str, Any]] = None
_diagnostics_cached_at: float = 0.0
# Singleflight for cache population (#1762 codex MED): /api/graph/health is
# public, so without this, concurrent requests hitting a cold/expired cache
# would EACH run the three O(graph) count scans. Per-process only (each
# uvicorn worker scans at most once per TTL window), which is the same
# blast-radius bound the TTL itself provides.
_diagnostics_scan_lock = asyncio.Lock()


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
    Check FalkorDB readiness.

    This is the hot readiness/liveness probe path (``/ready``). It proves
    connectivity with a single cheap ``list_graphs()`` call and reports latency.

    It deliberately does NOT run ``count(n)`` / ``count(r)`` full-graph scans:
    those are O(graph) and would block the event loop on every probe. Node/edge
    counts are diagnostics, not a readiness signal — they live in
    ``falkordb_diagnostics()`` (cached, off the hot path) instead.

    Returns:
        Dict with status and graph info
    """
    if not _health_circuit_breaker.allow_request():
        return {"status": "circuit_open"}

    try:
        client = await get_falkordb()

        if client is None:
            return {
                "status": "unavailable",
                "error": "FalkorDB not configured",
            }

        # Reachability check only — no full-graph scans on the readiness path.
        # ``list_graphs()`` is sync (Redis round-trip), so run it off the loop.
        start = time.time()
        graphs = await asyncio.to_thread(client.list_graphs)
        latency_ms = (time.time() - start) * 1000

        _health_circuit_breaker.record_success()

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "graphs": graphs,
            "current_graph": FALKORDB_GRAPH_NAME,
        }

    except Exception as e:
        _health_circuit_breaker.record_failure()
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def falkordb_diagnostics(*, use_cache: bool = True) -> Dict[str, Any]:
    """
    Return FalkorDB node/edge counts for diagnostics dashboards.

    This runs the expensive full-graph ``count(n)`` / ``count(r)`` scans that
    were removed from the readiness path. It is cached for
    ``_DIAGNOSTICS_TTL_SECONDS`` and the sync ``graph.query`` calls are run via
    ``asyncio.to_thread(...)`` so a diagnostics caller never blocks the event
    loop. This is NOT a readiness signal — orchestrators should poll
    ``/ready`` (``falkordb_health_check``) instead.

    Args:
        use_cache: When True (default), return the cached counts if they are
            still within the TTL window; otherwise force a fresh scan.

    Returns:
        Dict with ``status`` and, when healthy, ``node_count`` / ``edge_count``.
    """
    global _diagnostics_cache, _diagnostics_cached_at

    def _fresh_cache() -> Optional[Dict[str, Any]]:
        if (
            use_cache
            and _diagnostics_cache is not None
            and (time.time() - _diagnostics_cached_at) < _DIAGNOSTICS_TTL_SECONDS
        ):
            return {**_diagnostics_cache, "cached": True}
        return None

    cached = _fresh_cache()
    if cached is not None:
        return cached

    # Singleflight (#1762): the public health endpoint can see concurrent
    # requests on a cold/expired cache; serialize population and re-check
    # inside the lock so followers serve the leader's scan instead of
    # re-running the O(graph) counts themselves.
    async with _diagnostics_scan_lock:
        cached = _fresh_cache()
        if cached is not None:
            return cached
        return await _scan_diagnostics()


async def _scan_diagnostics() -> Dict[str, Any]:
    """Run the count scans and populate the cache. Callers hold the scan lock."""
    global _diagnostics_cache, _diagnostics_cached_at

    now = time.time()
    graph = await get_graph()
    if graph is None:
        return {"status": "unavailable", "error": "FalkorDB not configured"}

    def _scan_counts() -> tuple[int, int, int]:
        nodes = 0
        edges = 0
        curated = 0
        result = graph.query("MATCH (n) RETURN count(n) as count")
        if result.result_set:
            nodes = result.result_set[0][0]
        result = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
        if result.result_set:
            edges = result.result_set[0][0]
        # Curated gold-standard layer: seed/sync nodes carry no ``agent``
        # property, agent-written runtime nodes do (the same predicate as the
        # page's ``curated_only``). This is the #1760 emptiness tripwire —
        # after the #1758 wipe, agents repopulated runtime nodes within hours,
        # so a TOTAL count reads non-empty while everything the
        # /knowledge-graph page renders is gone.
        result = graph.query("MATCH (n) WHERE n.agent IS NULL RETURN count(n) as count")
        if result.result_set:
            curated = result.result_set[0][0]
        return nodes, edges, curated

    try:
        node_count, edge_count, curated_node_count = await asyncio.to_thread(_scan_counts)
    except Exception as e:
        # A failed scan is UNKNOWN, never zero: silent node_count=0 is a
        # plausible-wrong value that reads exactly like the #1758 wipe to the
        # graph-content sentinel (#1760). Not cached, so the next call within
        # the TTL window rescans instead of replaying the failure.
        return {
            "status": "unknown",
            "current_graph": FALKORDB_GRAPH_NAME,
            "error": str(e),
        }

    payload: Dict[str, Any] = {
        "status": "healthy",
        "current_graph": FALKORDB_GRAPH_NAME,
        "node_count": node_count,
        "edge_count": edge_count,
        "curated_node_count": curated_node_count,
    }
    _diagnostics_cache = payload
    _diagnostics_cached_at = now
    return {**payload, "cached": False}
