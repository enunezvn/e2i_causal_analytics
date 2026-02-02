"""
E2I Hybrid RAG - Full-Text Search Backend Client

PostgreSQL full-text search backend using ts_rank.
Good for:
- Exact term matching (KPI names, brand names)
- Acronym matching (TRx, HCP, ROI)
- Pattern matching with trigrams

Part of Phase 1, Checkpoint 1.3.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.rag.config import HybridSearchConfig
from src.rag.exceptions import FulltextSearchError
from src.rag.types import RetrievalResult, RetrievalSource

logger = logging.getLogger(__name__)


class FulltextBackend:
    """
    PostgreSQL full-text search backend.

    Uses websearch_to_tsquery and ts_rank for relevance scoring.
    Searches across:
    - causal_paths (causal relationships)
    - agent_activities (agent analysis outputs)
    - triggers (trigger explanations)

    Example:
        ```python
        from supabase import create_client
        from src.rag.backends import FulltextBackend
        from src.rag.config import HybridSearchConfig

        supabase = create_client(url, key)
        backend = FulltextBackend(supabase, HybridSearchConfig())

        results = await backend.search(
            query="TRx conversion Remibrutinib",
            filters={"brand": "Remibrutinib"}
        )
        ```
    """

    def __init__(self, supabase_client: Any, config: Optional[HybridSearchConfig] = None):
        """
        Initialize the full-text backend.

        Args:
            supabase_client: Supabase client instance
            config: Search configuration
        """
        self.client = supabase_client
        self.config = config or HybridSearchConfig()
        self._last_latency_ms: float = 0.0

    async def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Execute full-text search.

        Args:
            query: Natural language search query
            filters: Optional filters (brand, region, etc.)
            top_k: Override default top_k from config

        Returns:
            List of RetrievalResult ordered by ts_rank (descending)

        Raises:
            FulltextSearchError: If search fails or times out
        """
        start_time = time.time()
        top_k = top_k or self.config.fulltext_top_k
        timeout_seconds = self.config.fulltext_timeout_ms / 1000

        # Skip empty queries
        if not query or not query.strip():
            logger.debug("Skipping full-text search for empty query")
            return []

        try:
            # Execute RPC call with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(self._execute_rpc, query.strip(), top_k, filters or {}),
                timeout=timeout_seconds,
            )

            self._last_latency_ms = (time.time() - start_time) * 1000

            # Parse results
            results = []
            for row in response.data:
                # Filter by minimum rank threshold
                rank = float(row.get("rank", 0))
                if rank < self.config.fulltext_min_rank:
                    continue

                results.append(
                    RetrievalResult(
                        id=str(row["id"]),
                        content=row.get("content", ""),
                        source=RetrievalSource.FULLTEXT,
                        score=rank,
                        metadata={
                            **row.get("metadata", {}),
                            "source_table": row.get("source_table", "unknown"),
                        },
                        query_latency_ms=self._last_latency_ms,
                        raw_score=rank,
                    )
                )

            logger.debug(
                f"Full-text search returned {len(results)} results in {self._last_latency_ms:.1f}ms"
            )

            return results

        except asyncio.TimeoutError as e:
            self._last_latency_ms = self.config.fulltext_timeout_ms
            logger.warning(f"Full-text search timeout after {self.config.fulltext_timeout_ms}ms")
            raise FulltextSearchError(
                message=f"Full-text search timeout after {self.config.fulltext_timeout_ms}ms",
                backend="supabase_fulltext",
                details={"timeout_ms": self.config.fulltext_timeout_ms},
            ) from e

        except Exception as e:
            self._last_latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Full-text search error: {e}")
            raise FulltextSearchError(
                message=f"Full-text search failed: {e}",
                backend="supabase_fulltext",
                original_error=e,
            ) from e

    def _execute_rpc(self, search_query: str, match_count: int, filters: Dict[str, Any]) -> Any:
        """
        Execute the Supabase RPC call synchronously.

        This is called in a thread to allow async timeout.
        """
        return self.client.rpc(
            "rag_fulltext_search",
            {"search_query": search_query, "match_count": match_count, "filters": filters},
        ).execute()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the full-text backend is healthy.

        Returns:
            Dict with status, latency_ms, and any error message
        """
        try:
            start_time = time.time()

            await asyncio.wait_for(
                asyncio.to_thread(self._execute_rpc, "health_check_query", 1, {}), timeout=3.0
            )

            latency_ms = (time.time() - start_time) * 1000

            return {"status": "healthy", "latency_ms": latency_ms, "error": None}

        except asyncio.TimeoutError:
            return {"status": "unhealthy", "latency_ms": 3000, "error": "Health check timeout"}

        except Exception as e:
            return {"status": "unhealthy", "latency_ms": 0, "error": str(e)}

    @property
    def last_latency_ms(self) -> float:
        """Get latency from last query."""
        return self._last_latency_ms

    def __repr__(self) -> str:
        return (
            f"FulltextBackend("
            f"top_k={self.config.fulltext_top_k}, "
            f"timeout_ms={self.config.fulltext_timeout_ms}, "
            f"min_rank={self.config.fulltext_min_rank})"
        )
