"""
E2I Hybrid RAG - Vector Backend Client

Supabase/pgvector semantic search backend.
Uses cosine similarity for vector search across:
- insight_embeddings (causal insights, agent outputs)
- episodic_memories (conversation history)
- procedural_memories (successful patterns)

Part of Phase 1, Checkpoint 1.3.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.rag.config import HybridSearchConfig
from src.rag.exceptions import VectorSearchError
from src.rag.types import RetrievalResult, RetrievalSource

logger = logging.getLogger(__name__)


class VectorBackend:
    """
    Supabase/pgvector semantic search backend.

    Performs cosine similarity search using the hybrid_vector_search
    PostgreSQL function.

    Example:
        ```python
        from supabase import create_client
        from src.rag.backends import VectorBackend
        from src.rag.config import HybridSearchConfig

        supabase = create_client(url, key)
        backend = VectorBackend(supabase, HybridSearchConfig())

        results = await backend.search(
            embedding=[0.1, 0.2, ...],  # 1536-dim vector
            filters={"brand": "Remibrutinib"}
        )
        ```
    """

    def __init__(self, supabase_client: Any, config: Optional[HybridSearchConfig] = None):
        """
        Initialize the vector backend.

        Args:
            supabase_client: Supabase client instance
            config: Search configuration
        """
        self.client = supabase_client
        self.config = config or HybridSearchConfig()
        self._last_latency_ms: float = 0.0

    async def search(
        self,
        embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Execute vector similarity search.

        Args:
            embedding: Query embedding vector (1536 dimensions for OpenAI)
            filters: Optional filters (brand, region, date_range, etc.)
            top_k: Override default top_k from config

        Returns:
            List of RetrievalResult ordered by similarity (descending)

        Raises:
            VectorSearchError: If search fails or times out
        """
        start_time = time.time()
        top_k = top_k or self.config.vector_top_k
        timeout_seconds = self.config.vector_timeout_ms / 1000

        try:
            # Execute RPC call with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(self._execute_rpc, embedding, top_k, filters or {}),
                timeout=timeout_seconds,
            )

            self._last_latency_ms = (time.time() - start_time) * 1000

            # Parse results
            results = []
            for row in response.data:
                # Filter by minimum similarity threshold
                similarity = float(row.get("similarity", 0))
                if similarity < self.config.vector_min_similarity:
                    continue

                results.append(
                    RetrievalResult(
                        id=str(row["id"]),
                        content=row.get("content", ""),
                        source=RetrievalSource.VECTOR,
                        score=similarity,
                        metadata=row.get("metadata", {}),
                        query_latency_ms=self._last_latency_ms,
                        raw_score=similarity,
                    )
                )

            logger.debug(
                f"Vector search returned {len(results)} results "
                f"in {self._last_latency_ms:.1f}ms"
            )

            return results

        except asyncio.TimeoutError:
            self._last_latency_ms = self.config.vector_timeout_ms
            logger.warning(f"Vector search timeout after {self.config.vector_timeout_ms}ms")
            raise VectorSearchError(
                message=f"Vector search timeout after {self.config.vector_timeout_ms}ms",
                backend="supabase_vector",
                details={"timeout_ms": self.config.vector_timeout_ms},
            )

        except Exception as e:
            self._last_latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Vector search error: {e}")
            raise VectorSearchError(
                message=f"Vector search failed: {e}", backend="supabase_vector", original_error=e
            )

    def _execute_rpc(
        self, embedding: List[float], match_count: int, filters: Dict[str, Any]
    ) -> Any:
        """
        Execute the Supabase RPC call synchronously.

        This is called in a thread to allow async timeout.
        """
        return self.client.rpc(
            "rag_vector_search",
            {"query_embedding": embedding, "match_count": match_count, "filters": filters},
        ).execute()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the vector backend is healthy.

        Returns:
            Dict with status, latency_ms, and any error message
        """
        try:
            # Use a zero vector for health check (should be fast)
            start_time = time.time()

            await asyncio.wait_for(
                asyncio.to_thread(self._execute_rpc, [0.0] * 1536, 1, {}),  # Zero vector
                timeout=5.0,
            )

            latency_ms = (time.time() - start_time) * 1000

            return {"status": "healthy", "latency_ms": latency_ms, "error": None}

        except asyncio.TimeoutError:
            return {"status": "unhealthy", "latency_ms": 5000, "error": "Health check timeout"}

        except Exception as e:
            return {"status": "unhealthy", "latency_ms": 0, "error": str(e)}

    @property
    def last_latency_ms(self) -> float:
        """Get latency from last query."""
        return self._last_latency_ms

    def __repr__(self) -> str:
        return (
            f"VectorBackend("
            f"top_k={self.config.vector_top_k}, "
            f"timeout_ms={self.config.vector_timeout_ms}, "
            f"min_similarity={self.config.vector_min_similarity})"
        )
