"""
Hybrid retrieval implementation for CausalRAG.

Combines multiple retrieval strategies:
- Dense: Vector similarity search via Supabase pgvector
- Sparse: Full-text search via PostgreSQL ts_rank
- Graph: FalkorDB semantic graph traversal

Uses memory backends for retrieval:
- Episodic Memory: Conversation history, agent actions
- Procedural Memory: Successful patterns, tool sequences
- Semantic Memory: Entity relationships, causal chains
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.rag.memory_connector import get_memory_connector
from src.rag.models.retrieval_models import RetrievalResult

logger = logging.getLogger(__name__)


# Hybrid retrieval weights (can be tuned by Feedback Learner)
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.3
GRAPH_WEIGHT = 0.2


class DenseRetriever:
    """
    Dense vector retrieval using memory backends.

    Backend: Supabase pgvector via episodic/procedural memories
    Dimension: 1536 (OpenAI text-embedding-3-small)
    """

    def __init__(self):
        self.embedding_dim = 1536

    async def search(
        self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search vector store for semantically similar content.

        Uses memory connector to search across:
        - episodic_memories (conversations, agent actions)
        - procedural_memories (successful patterns)

        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional filters (brand, region, agent_name)

        Returns:
            List of RetrievalResult with dense retrieval method
        """
        connector = get_memory_connector()

        try:
            results = await connector.vector_search_by_text(
                query_text=query, k=k, filters=filters, min_similarity=0.5
            )
            logger.debug(f"Dense retrieval returned {len(results)} results for: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []


class BM25Retriever:
    """
    Sparse retrieval using PostgreSQL full-text search.

    Uses hybrid_fulltext_search function for BM25-like ranking.
    """

    async def search(
        self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search using BM25-like sparse retrieval.

        Uses memory connector to search across:
        - causal_paths (causal relationships)
        - agent_activities (agent actions)
        - triggers (alerts and recommended actions)

        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional filters (brand, agent_name)

        Returns:
            List of RetrievalResult with sparse retrieval method
        """
        connector = get_memory_connector()

        try:
            results = await connector.fulltext_search(query_text=query, k=k, filters=filters)
            logger.debug(f"Sparse retrieval returned {len(results)} results for: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []


class GraphRetriever:
    """
    Graph-based retrieval using FalkorDB semantic memory.

    Traverses semantic graph to find related entities and causal paths.
    """

    def traverse(
        self,
        entities: List[str],
        relationship: str = "causal_path",
        max_depth: int = 3,
    ) -> List[RetrievalResult]:
        """
        Traverse semantic graph to find related content.

        Uses memory connector to traverse:
        - Causal chains (cause → effect relationships)
        - Patient networks (patient → HCP → treatment)
        - HCP influence networks (HCP → colleagues → patients)

        Args:
            entities: Starting entity IDs for traversal
            relationship: Type of relationship to follow
                - "causal_path": Follow causal chains
                - "patient_network": Expand patient connections
                - "hcp_network": Expand HCP influence
            max_depth: Maximum traversal depth

        Returns:
            List of RetrievalResult with graph retrieval method
        """
        connector = get_memory_connector()
        all_results = []

        for entity_id in entities:
            try:
                results = connector.graph_traverse(
                    entity_id=entity_id, relationship=relationship, max_depth=max_depth
                )
                all_results.extend(results)

            except Exception as e:
                logger.error(f"Graph traversal failed for {entity_id}: {e}")

        # Deduplicate by source_id
        seen = set()
        unique_results = []
        for result in all_results:
            if result.source_id not in seen:
                seen.add(result.source_id)
                unique_results.append(result)

        logger.debug(f"Graph retrieval returned {len(unique_results)} results")
        return unique_results

    def traverse_kpi(self, kpi_name: str, min_confidence: float = 0.5) -> List[RetrievalResult]:
        """
        Find causal paths impacting a specific KPI.

        Args:
            kpi_name: Name of the KPI (e.g., "TRx", "NRx", "conversion_rate")
            min_confidence: Minimum confidence threshold

        Returns:
            List of RetrievalResult with graph retrieval method
        """
        connector = get_memory_connector()

        try:
            results = connector.graph_traverse_kpi(kpi_name=kpi_name, min_confidence=min_confidence)
            logger.debug(f"KPI graph retrieval returned {len(results)} results for: {kpi_name}")
            return results

        except Exception as e:
            logger.error(f"KPI graph traversal failed: {e}")
            return []


class HybridRetriever:
    """
    Combines multiple retrieval strategies with Reciprocal Rank Fusion.

    Retrieval methods:
    - Dense: Vector similarity via episodic/procedural memories
    - Sparse: Full-text search via PostgreSQL ts_rank
    - Graph: FalkorDB semantic graph traversal
    """

    def __init__(self):
        self.dense = DenseRetriever()
        self.sparse = BM25Retriever()
        self.graph = GraphRetriever()

    async def search(
        self,
        query: str,
        weights: Optional[Dict[str, float]] = None,
        k: int = 10,
        entities: Optional[List[str]] = None,
        kpi_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid search with configurable weights.

        Args:
            query: Search query text
            weights: Override default retrieval weights
                - "dense": Vector similarity weight (default 0.5)
                - "sparse": Full-text search weight (default 0.3)
                - "graph": Graph traversal weight (default 0.2)
            k: Number of results to return
            entities: Entity IDs for graph traversal
            kpi_name: KPI name for targeted graph traversal
            filters: Filters for dense/sparse search

        Returns:
            Fused results from all retrieval methods
        """
        weights = weights or {
            "dense": DENSE_WEIGHT,
            "sparse": SPARSE_WEIGHT,
            "graph": GRAPH_WEIGHT,
        }

        # Run dense and sparse searches concurrently
        dense_task = asyncio.create_task(self.dense.search(query, k=k * 2, filters=filters))
        sparse_task = asyncio.create_task(self.sparse.search(query, k=k * 2, filters=filters))

        # Get graph results (synchronous)
        graph_results = []
        if entities:
            graph_results = self.graph.traverse(entities, max_depth=3)
        elif kpi_name:
            graph_results = self.graph.traverse_kpi(kpi_name)

        # Wait for async results
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # Apply Reciprocal Rank Fusion
        result_lists = [dense_results, sparse_results, graph_results]
        weight_list = [weights["dense"], weights["sparse"], weights["graph"]]

        fused = self._reciprocal_rank_fusion(
            result_lists=result_lists,
            weights=weight_list,
        )

        return fused[:k]

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        weights: List[float],
        k: int = 60,
    ) -> List[RetrievalResult]:
        """
        Fuse multiple ranked lists using RRF algorithm.

        RRF score = sum(weight_i / (k + rank_i))

        Args:
            result_lists: List of ranked result lists
            weights: Weight for each list
            k: RRF constant (default 60)

        Returns:
            Fused and reranked results
        """
        scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        for results, weight in zip(result_lists, weights, strict=False):
            for rank, result in enumerate(results, start=1):
                # Use source_id for deduplication - same document from different sources should be boosted
                key = result.source_id
                rrf_score = weight / (k + rank)
                scores[key] = scores.get(key, 0) + rrf_score
                result_map[key] = result

        # Sort by fused score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Update result scores with fused scores
        fused_results = []
        for key in sorted_keys:
            if key in result_map:
                result = result_map[key]
                # Create new result with fused score
                fused_result = RetrievalResult(
                    source_id=result.source_id,
                    content=result.content,
                    source=result.source,
                    score=scores[key],
                    retrieval_method=result.retrieval_method,
                    metadata={
                        **result.metadata,
                        "rrf_score": scores[key],
                        "original_score": result.score,
                    },
                )
                fused_results.append(fused_result)

        return fused_results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def hybrid_search(
    query: str,
    k: int = 10,
    entities: Optional[List[str]] = None,
    kpi_name: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[RetrievalResult]:
    """
    Execute hybrid search using default retriever.

    Args:
        query: Search query text
        k: Number of results
        entities: Optional entity IDs for graph traversal
        kpi_name: Optional KPI name for targeted traversal
        filters: Optional filters

    Returns:
        Fused retrieval results
    """
    retriever = HybridRetriever()
    return await retriever.search(
        query=query, k=k, entities=entities, kpi_name=kpi_name, filters=filters
    )
