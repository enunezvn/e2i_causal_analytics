"""
E2I Hybrid RAG - Search Logger

Logs search queries and statistics to the database for auditing and debugging.
Integrates with SearchStats dataclass from the hybrid retriever.

Part of Phase 1, Checkpoint 1.4.
"""

import asyncio
import hashlib
import logging
from typing import Any, Dict, Optional
from uuid import UUID

from src.rag.types import SearchStats

logger = logging.getLogger(__name__)


class SearchLogger:
    """
    Logs search queries and statistics to the database.

    Uses the log_rag_search PostgreSQL function for efficient
    insertion with proper indexing.

    Example:
        ```python
        from supabase import create_client
        from src.rag.search_logger import SearchLogger
        from src.rag.types import SearchStats

        supabase = create_client(url, key)
        logger = SearchLogger(supabase)

        # After a search
        stats = SearchStats(
            query="TRx conversion Remibrutinib",
            total_latency_ms=150.0,
            vector_count=5,
            fulltext_count=3,
            graph_count=2,
            fused_count=8,
            sources_used={"vector": True, "fulltext": True, "graph": True}
        )

        log_id = await logger.log_search(stats)
        ```
    """

    def __init__(
        self,
        supabase_client: Any,
        enabled: bool = True
    ):
        """
        Initialize the search logger.

        Args:
            supabase_client: Supabase client instance
            enabled: Whether logging is enabled
        """
        self.client = supabase_client
        self.enabled = enabled
        self._log_queue: list[SearchStats] = []
        self._batch_size = 10

    async def log_search(
        self,
        stats: SearchStats,
        session_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        extracted_entities: Optional[Dict[str, Any]] = None
    ) -> Optional[UUID]:
        """
        Log a search query and its statistics.

        Args:
            stats: SearchStats from the hybrid retriever
            session_id: Optional session identifier
            user_id: Optional user identifier
            config: Optional search configuration used
            extracted_entities: Optional entities extracted from query

        Returns:
            UUID of the log entry, or None if logging disabled/failed
        """
        if not self.enabled:
            return None

        try:
            # Use the RPC function for efficient logging
            response = await asyncio.to_thread(
                self._execute_log_rpc,
                stats,
                session_id,
                user_id,
                config or {},
                extracted_entities or {}
            )

            if response.data:
                log_id = UUID(response.data)
                logger.debug(f"Logged search: {log_id}")
                return log_id

            return None

        except Exception as e:
            # Don't fail the search just because logging failed
            logger.warning(f"Failed to log search: {e}")
            return None

    def _execute_log_rpc(
        self,
        stats: SearchStats,
        session_id: Optional[UUID],
        user_id: Optional[str],
        config: Dict[str, Any],
        extracted_entities: Dict[str, Any]
    ) -> Any:
        """
        Execute the log_rag_search RPC call synchronously.
        """
        return self.client.rpc(
            "log_rag_search",
            {
                "p_query": stats.query,
                "p_session_id": str(session_id) if session_id else None,
                "p_user_id": user_id,
                "p_vector_count": stats.vector_count,
                "p_fulltext_count": stats.fulltext_count,
                "p_graph_count": stats.graph_count,
                "p_fused_count": stats.fused_count,
                "p_total_latency_ms": stats.total_latency_ms,
                "p_vector_latency_ms": stats.vector_latency_ms,
                "p_fulltext_latency_ms": stats.fulltext_latency_ms,
                "p_graph_latency_ms": stats.graph_latency_ms,
                "p_fusion_latency_ms": stats.fusion_latency_ms,
                "p_sources_used": stats.sources_used,
                "p_errors": stats.errors,
                "p_config": config,
                "p_extracted_entities": extracted_entities
            }
        ).execute()

    async def log_search_batch(
        self,
        stats_list: list[SearchStats],
        session_id: Optional[UUID] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        Log multiple search queries in batch.

        Args:
            stats_list: List of SearchStats to log
            session_id: Optional session identifier for all
            user_id: Optional user identifier for all

        Returns:
            Number of successfully logged entries
        """
        if not self.enabled:
            return 0

        success_count = 0
        for stats in stats_list:
            log_id = await self.log_search(stats, session_id, user_id)
            if log_id:
                success_count += 1

        return success_count

    async def get_slow_queries(
        self,
        limit: int = 10,
        threshold_ms: float = 1000.0
    ) -> list[Dict[str, Any]]:
        """
        Get recent slow queries for debugging.

        Args:
            limit: Maximum number of queries to return
            threshold_ms: Latency threshold in milliseconds

        Returns:
            List of slow query records
        """
        try:
            response = await asyncio.to_thread(
                lambda: self.client.from_("rag_search_logs")
                .select("*")
                .gt("total_latency_ms", threshold_ms)
                .order("total_latency_ms", desc=True)
                .limit(limit)
                .execute()
            )

            return response.data or []

        except Exception as e:
            logger.warning(f"Failed to get slow queries: {e}")
            return []

    async def get_search_stats_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get search statistics summary for the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Summary statistics
        """
        try:
            # Use the rag_search_stats view
            response = await asyncio.to_thread(
                lambda: self.client.from_("rag_search_stats")
                .select("*")
                .order("hour", desc=True)
                .limit(hours)
                .execute()
            )

            if not response.data:
                return {
                    "total_queries": 0,
                    "avg_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "error_rate": 0
                }

            total_queries = sum(row.get("query_count", 0) for row in response.data)
            total_errors = sum(row.get("error_count", 0) for row in response.data)

            latencies = [row.get("avg_latency_ms", 0) for row in response.data if row.get("avg_latency_ms")]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            p95_values = [row.get("p95_latency_ms", 0) for row in response.data if row.get("p95_latency_ms")]
            p95_latency = max(p95_values) if p95_values else 0

            return {
                "total_queries": total_queries,
                "avg_latency_ms": round(avg_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "error_rate": round(total_errors / total_queries * 100, 2) if total_queries > 0 else 0,
                "hours_analyzed": len(response.data)
            }

        except Exception as e:
            logger.warning(f"Failed to get search stats: {e}")
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "error_rate": 0,
                "error": str(e)
            }

    def __repr__(self) -> str:
        return f"SearchLogger(enabled={self.enabled})"
