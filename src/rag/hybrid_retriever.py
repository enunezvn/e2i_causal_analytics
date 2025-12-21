"""
E2I Hybrid RAG - Hybrid Retriever Orchestrator

Main orchestrator for hybrid search across:
- Vector (Supabase/pgvector) - semantic similarity
- Full-text (PostgreSQL) - exact matching
- Graph (FalkorDB/Graphiti) - causal relationships

Implements:
- Parallel backend dispatch with timeouts
- Reciprocal Rank Fusion (RRF) with k=60
- Graph boost (1.3x) for causally-connected results
- Graceful degradation when backends fail

Part of Phase 1, Checkpoint 1.3.
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.rag.backends import FulltextBackend, GraphBackend, VectorBackend
from src.rag.config import (
    RAGConfig,
)
from src.rag.types import (
    BackendHealth,
    BackendStatus,
    ExtractedEntities,
    RetrievalResult,
    RetrievalSource,
    SearchStats,
)

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that orchestrates vector, full-text, and graph search.

    Architecture:
        1. Extract entities from query (brands, KPIs, regions)
        2. Dispatch parallel searches to all backends
        3. Apply Reciprocal Rank Fusion (RRF) to combine results
        4. Boost graph results by 1.3x for causal connectivity
        5. Return top_k fused results with source attribution

    Example:
        ```python
        from supabase import create_client
        from falkordb import FalkorDB
        from src.rag import HybridRetriever
        from src.rag.config import RAGConfig

        supabase = create_client(url, key)
        falkordb = FalkorDB(host="localhost", port=6381)

        retriever = HybridRetriever(
            supabase_client=supabase,
            falkordb_client=falkordb,
            config=RAGConfig()
        )

        # Search with embedding
        results = await retriever.search(
            query="Why is TRx declining for Remibrutinib in West region?",
            embedding=[0.1, 0.2, ...],  # Pre-computed embedding
            entities=ExtractedEntities(
                brands=["Remibrutinib"],
                regions=["West"],
                kpis=["TRx"]
            )
        )
        ```
    """

    # RRF constant - balances emphasis on top-ranked vs. lower-ranked items
    RRF_K = 60

    # Graph boost multiplier for causally-connected results
    GRAPH_BOOST = 1.3

    def __init__(
        self,
        supabase_client: Any,
        falkordb_client: Any,
        config: Optional[RAGConfig] = None,
        embedding_service: Optional[Any] = None,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            supabase_client: Supabase client for vector and fulltext search
            falkordb_client: FalkorDB client for graph search
            config: RAG configuration
            embedding_service: Optional embedding service for on-demand embedding
        """
        self.config = config or RAGConfig()
        self.embedding_service = embedding_service

        # Initialize backends
        self.vector_backend = VectorBackend(
            supabase_client=supabase_client, config=self.config.search
        )
        self.fulltext_backend = FulltextBackend(
            supabase_client=supabase_client, config=self.config.search
        )
        self.graph_backend = GraphBackend(
            falkordb_client=falkordb_client,
            falkordb_config=self.config.falkordb,
            search_config=self.config.search,
        )

        # Track backend health
        self._backend_health: Dict[str, BackendHealth] = {}
        self._last_search_stats: Optional[SearchStats] = None

    async def search(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        entities: Optional[ExtractedEntities] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid search across all backends.

        Args:
            query: Natural language query text
            embedding: Pre-computed query embedding (1536 dimensions)
            entities: Extracted entities for graph search
            filters: Optional filters (brand, region, date_range)
            top_k: Override default top_k from config

        Returns:
            List of RetrievalResult ordered by fused score (descending)

        Raises:
            FusionError: If all backends fail
            RetrieverError: If a required backend fails
        """
        start_time = time.time()
        top_k = top_k or self.config.search.final_top_k
        filters = filters or {}

        # Generate embedding if not provided
        if embedding is None and self.embedding_service:
            try:
                embedding = await self.embedding_service.embed(query)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
                embedding = None

        # Extract entities if not provided
        if entities is None:
            entities = ExtractedEntities()  # Empty entities

        # Dispatch parallel searches
        backend_results = await self._dispatch_parallel_searches(
            query=query, embedding=embedding, entities=entities, filters=filters
        )

        # Check if all backends failed
        total_results = sum(len(r) for r in backend_results.values())
        if total_results == 0:
            logger.warning("All backends returned zero results")
            # Return empty list rather than raising error
            self._last_search_stats = SearchStats(
                query=query,
                total_latency_ms=(time.time() - start_time) * 1000,
                vector_count=0,
                fulltext_count=0,
                graph_count=0,
                fused_count=0,
                sources_used={
                    "vector": embedding is not None,
                    "fulltext": bool(query and query.strip()),
                    "graph": True,
                },
                vector_latency_ms=self.vector_backend.last_latency_ms,
                fulltext_latency_ms=self.fulltext_backend.last_latency_ms,
                graph_latency_ms=self.graph_backend.last_latency_ms,
            )
            return []

        # Apply RRF fusion
        fused_results = self._apply_rrf_fusion(backend_results, top_k)

        # Apply graph boost
        boosted_results = self._apply_graph_boost(fused_results)

        # Sort by final score and limit
        final_results = sorted(boosted_results, key=lambda x: x.score, reverse=True)[:top_k]

        # Update search stats
        total_latency_ms = (time.time() - start_time) * 1000
        vector_count = len(backend_results.get(RetrievalSource.VECTOR, []))
        fulltext_count = len(backend_results.get(RetrievalSource.FULLTEXT, []))
        graph_count = len(backend_results.get(RetrievalSource.GRAPH, []))

        self._last_search_stats = SearchStats(
            query=query,
            total_latency_ms=total_latency_ms,
            vector_count=vector_count,
            fulltext_count=fulltext_count,
            graph_count=graph_count,
            fused_count=len(final_results),
            sources_used={
                "vector": embedding is not None,
                "fulltext": bool(query and query.strip()),
                "graph": True,
            },
            vector_latency_ms=self.vector_backend.last_latency_ms,
            fulltext_latency_ms=self.fulltext_backend.last_latency_ms,
            graph_latency_ms=self.graph_backend.last_latency_ms,
        )

        logger.info(
            f"Hybrid search completed: {len(final_results)} results "
            f"in {total_latency_ms:.1f}ms "
            f"(v:{len(backend_results.get(RetrievalSource.VECTOR, []))} "
            f"f:{len(backend_results.get(RetrievalSource.FULLTEXT, []))} "
            f"g:{len(backend_results.get(RetrievalSource.GRAPH, []))})"
        )

        return final_results

    async def _dispatch_parallel_searches(
        self,
        query: str,
        embedding: Optional[List[float]],
        entities: ExtractedEntities,
        filters: Dict[str, Any],
    ) -> Dict[RetrievalSource, List[RetrievalResult]]:
        """
        Dispatch searches to all backends in parallel.

        Returns dict of source -> results, with empty list for failed backends.
        """
        results: Dict[RetrievalSource, List[RetrievalResult]] = {
            RetrievalSource.VECTOR: [],
            RetrievalSource.FULLTEXT: [],
            RetrievalSource.GRAPH: [],
        }

        # Build task list
        tasks = []
        task_sources = []

        # Vector search (requires embedding)
        if embedding is not None:
            tasks.append(self._safe_vector_search(embedding, filters))
            task_sources.append(RetrievalSource.VECTOR)
        else:
            logger.debug("Skipping vector search - no embedding provided")

        # Fulltext search (requires query text)
        if query and query.strip():
            tasks.append(self._safe_fulltext_search(query, filters))
            task_sources.append(RetrievalSource.FULLTEXT)

        # Graph search (can work with or without entities)
        tasks.append(self._safe_graph_search(entities, query, filters))
        task_sources.append(RetrievalSource.GRAPH)

        # Execute all tasks in parallel
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for source, result in zip(task_sources, completed, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"{source.value} search failed: {result}")
                    results[source] = []
                else:
                    results[source] = result

        return results

    async def _safe_vector_search(
        self, embedding: List[float], filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Execute vector search with error handling."""
        try:
            return await self.vector_backend.search(embedding=embedding, filters=filters)
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def _safe_fulltext_search(
        self, query: str, filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Execute fulltext search with error handling."""
        try:
            return await self.fulltext_backend.search(query=query, filters=filters)
        except Exception as e:
            logger.error(f"Fulltext search error: {e}")
            return []

    async def _safe_graph_search(
        self, entities: ExtractedEntities, query: str, filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Execute graph search with error handling."""
        try:
            return await self.graph_backend.search(entities=entities, query=query, filters=filters)
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return []

    def _apply_rrf_fusion(
        self, backend_results: Dict[RetrievalSource, List[RetrievalResult]], top_k: int
    ) -> List[RetrievalResult]:
        """
        Apply Reciprocal Rank Fusion to combine results from multiple backends.

        RRF Score = sum(1 / (k + rank_i)) for each backend i

        Where:
        - k is a constant (60) to prevent over-emphasis on top results
        - rank_i is the 1-based rank in backend i

        Args:
            backend_results: Dict of source -> ranked results
            top_k: Maximum results to return

        Returns:
            List of results with RRF scores
        """
        # Collect all unique result IDs with their ranks per backend
        id_ranks: Dict[str, Dict[RetrievalSource, int]] = defaultdict(dict)
        id_to_result: Dict[str, RetrievalResult] = {}

        for source, results in backend_results.items():
            for rank, result in enumerate(results, start=1):
                id_ranks[result.id][source] = rank
                # Keep the result with most metadata, preserving graph_context
                if result.id not in id_to_result:
                    id_to_result[result.id] = result
                else:
                    existing = id_to_result[result.id]
                    # Prefer result with graph_context, or more metadata
                    if result.graph_context and not existing.graph_context:
                        id_to_result[result.id] = result
                    elif len(result.metadata) > len(existing.metadata):
                        # Also preserve graph_context from existing if new doesn't have it
                        if existing.graph_context and not result.graph_context:
                            result.graph_context = existing.graph_context
                        id_to_result[result.id] = result
                    elif existing.graph_context is None and result.graph_context:
                        # Merge graph_context into existing
                        existing.graph_context = result.graph_context

        # Calculate RRF scores with weights
        weights = self.config.search.fusion_weights
        rrf_scores: Dict[str, float] = {}

        for result_id, source_ranks in id_ranks.items():
            score = 0.0
            sources_found = []

            for source, rank in source_ranks.items():
                weight = weights.get(source.value, 0.33)
                score += weight * (1.0 / (self.RRF_K + rank))
                sources_found.append(source.value)

            rrf_scores[result_id] = score

            # Update result with RRF score and source attribution
            result = id_to_result[result_id]
            result.score = score
            result.metadata["rrf_sources"] = sources_found
            result.metadata["rrf_score"] = score

        # Sort by RRF score and return top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [id_to_result[rid] for rid in sorted_ids[:top_k]]

    def _apply_graph_boost(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Apply graph boost to results that have causal graph context.

        Results with graph_context (causal connections) get a 1.3x score boost.

        Args:
            results: List of fused results

        Returns:
            Results with graph boost applied
        """
        for result in results:
            if result.graph_context:
                # Has causal connections - apply boost
                original_score = result.score
                result.score *= self.GRAPH_BOOST
                result.metadata["graph_boosted"] = True
                result.metadata["pre_boost_score"] = original_score

                logger.debug(
                    f"Applied graph boost to {result.id}: "
                    f"{original_score:.4f} -> {result.score:.4f}"
                )

        return results

    async def health_check(self) -> Dict[str, BackendHealth]:
        """
        Check health of all backends.

        Returns:
            Dict of backend name -> BackendHealth
        """
        health_tasks = [
            ("vector", self.vector_backend.health_check()),
            ("fulltext", self.fulltext_backend.health_check()),
            ("graph", self.graph_backend.health_check()),
        ]

        results = await asyncio.gather(*[task for _, task in health_tasks], return_exceptions=True)

        health = {}
        for (name, _), result in zip(health_tasks, results, strict=False):
            if isinstance(result, Exception):
                health[name] = BackendHealth(
                    status=BackendStatus.UNHEALTHY,
                    latency_ms=0.0,
                    last_check=datetime.now(timezone.utc),
                    error_message=str(result),
                )
            else:
                # Convert string status to BackendStatus enum
                status_str = result.get("status", "unknown")
                try:
                    status = BackendStatus(status_str)
                except ValueError:
                    status = BackendStatus.UNKNOWN

                health[name] = BackendHealth(
                    status=status,
                    latency_ms=float(result.get("latency_ms", 0)),
                    last_check=datetime.now(timezone.utc),
                    error_message=result.get("error"),
                )

        self._backend_health = health
        return health

    async def get_causal_subgraph(
        self,
        center_node_id: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get a subgraph for visualization from FalkorDB/Graphiti.

        Delegates to the graph backend for actual query execution.

        Args:
            center_node_id: ID of center node for ego graph
            node_types: Filter by node types
            relationship_types: Filter by relationship types
            max_depth: Max traversal depth
            limit: Max nodes to return

        Returns:
            Dict with 'nodes', 'edges', and 'metadata' for visualization
        """
        return await self.graph_backend.get_causal_subgraph(
            center_node_id=center_node_id,
            node_types=node_types,
            relationship_types=relationship_types,
            max_depth=max_depth,
            limit=limit,
        )

    async def get_causal_path(
        self, source_id: str, target_id: str, max_length: int = 4
    ) -> List[Any]:
        """
        Find causal paths between two nodes.

        Delegates to the graph backend for actual query execution.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_length: Max path length

        Returns:
            List of GraphPath objects
        """
        return await self.graph_backend.get_causal_path(
            source_id=source_id, target_id=target_id, max_length=max_length
        )

    @property
    def last_search_stats(self) -> Optional[SearchStats]:
        """Get stats from the last search operation."""
        return self._last_search_stats

    @property
    def backend_health(self) -> Dict[str, BackendHealth]:
        """Get cached backend health status."""
        return self._backend_health

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"vector={self.vector_backend!r}, "
            f"fulltext={self.fulltext_backend!r}, "
            f"graph={self.graph_backend!r})"
        )
