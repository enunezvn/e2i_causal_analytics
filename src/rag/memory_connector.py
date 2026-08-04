"""
Memory Connector for RAG Integration.

Bridges the memory systems (Episodic, Procedural, Semantic) to the RAG retriever layer.

Usage:
    from src.rag.memory_connector import MemoryConnector

    connector = MemoryConnector()
    results = await connector.vector_search(query_embedding, k=10)
    results = await connector.fulltext_search(query_text, k=10)
    results = await connector.graph_traverse(entity_id, max_depth=3)
"""

import logging
from typing import Any, Dict, List, Optional

from src.memory.semantic_memory import get_semantic_memory

# #1475: the search methods await the ASYNC Supabase client. The previous sync
# get_supabase_client() + sync .rpc().execute() ran the whole RPC round-trip on
# the event loop inside async defs, stalling every other in-flight request for
# ~2s per warm RAG leg (issue #1475 measurement).
from src.memory.services.factories import get_async_supabase_client, get_embedding_service
from src.rag.models.retrieval_models import RetrievalResult
from src.rag.types import RetrievalSource

logger = logging.getLogger(__name__)


def _is_invalidated_under_max_staleness(
    metadata: Optional[Dict[str, Any]], max_staleness: Optional[float]
) -> bool:
    """Return True iff the row should be dropped by the max_staleness filter.

    Under Decision 3 = KEEP BINARY (plan §"DECISIONS ADOPTED" 2026-05-19), the
    boolean-degraded semantics are:
      - max_staleness is None: never drop (no filter active)
      - max_staleness NOT < 1.0 (incl. NaN, +inf, exactly 1.0): never drop
      - max_staleness < 1.0 (incl. 0.0, negative): drop iff
        metadata['invalidated_at'] is truthy

    Public-API contract (M1 of #381): the four public retriever surfaces
    (``HybridRetriever.search``, ``hybrid_search`` convenience function,
    ``DenseRetriever.search``, ``BM25Retriever.search``) all accept a
    ``max_staleness: Optional[float]`` parameter whose contract is THIS
    binary-degradation predicate. Fractional values (e.g. 0.5) are accepted
    for forward compatibility with a possible future graded-staleness
    reinstatement (see the plan's Decision-3 reinstatement checklist) but
    currently behave identically to 0.0. The implementation is intentionally
    centralised here so any future reinstatement has one mutation point.

    Type guard (codex iter-1 LOW): non-(int|float) inputs (including bool — a
    bool is technically an int in Python but ``False == 0.0`` would
    spuriously activate the filter — and any string/None/Any) degrade to
    "no filter" rather than raising. This matches the SQL RPC's parse-error
    no-op contract at database/memory/022_hybrid_search_max_staleness.sql:55-60.

    NaN handling mirrors the SQL RPC (`v_max_staleness < 1.0` evaluates to
    False for NaN in PostgreSQL, so NaN does not activate the filter in
    either layer).

    Phase 2 finishing (issue #373). When/if Decision 3 is revisited and graded
    staleness is introduced, this helper is the single mutation point.
    """
    # Type guard — accept only None or real numeric values. Bool is excluded
    # despite being a subclass of int (False would otherwise coerce to 0.0
    # and activate the filter).
    if max_staleness is None:
        return False
    if isinstance(max_staleness, bool) or not isinstance(max_staleness, (int, float)):
        return False
    # `not (x < 1.0)` correctly returns True for NaN, +inf, and exactly 1.0,
    # giving the same "no filter" branch as max_staleness=None.
    if not (max_staleness < 1.0):
        return False
    if not isinstance(metadata, dict):
        return False
    invalidated_at = metadata.get("invalidated_at")
    # Falsy values (None, "", 0) all mean "not invalidated"
    return bool(invalidated_at)


class MemoryConnector:
    """
    Connects RAG retriever to memory backends.

    Provides unified interface for:
    - Vector search via episodic/procedural memories (Supabase pgvector)
    - Full-text search via hybrid_fulltext_search SQL function
    - Graph traversal via semantic memory (FalkorDB)
    """

    def __init__(self):
        """Initialize memory connector."""
        self._embedding_service = None

    async def get_embedding_service(self):
        """Lazy embedding service initialization."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    # ========================================================================
    # VECTOR SEARCH (Dense Retrieval)
    # ========================================================================

    async def vector_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.5,
        max_staleness: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Search memories by vector similarity.

        Uses Supabase hybrid_vector_search RPC function to search across:
        - episodic_memories (conversations, agent actions)
        - procedural_memories (successful patterns)

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional filters (brand, region, agent_name, date_from, date_to)
            min_similarity: Minimum similarity threshold (default 0.5)
            max_staleness: Optional staleness ceiling. Under Decision 3 = KEEP BINARY
                (plan §"DECISIONS ADOPTED" 2026-05-19), semantics are:
                - None (default): no filter, include all rows (backward-compatible)
                - >= 1.0: include all rows
                - < 1.0 (incl 0.0): exclude rows whose metadata carries ``invalidated_at``
                Forwarded into the RPC filters dict AND applied as a Python post-filter
                (belt-and-suspenders).

        Returns:
            List of RetrievalResult with dense retrieval method
        """
        client = await get_async_supabase_client()

        # Build RPC filters dict, including max_staleness when provided
        rpc_filters: Dict[str, Any] = dict(filters or {})
        if max_staleness is not None:
            rpc_filters["max_staleness"] = max_staleness

        try:
            # Call Supabase RPC function (awaited — the RPC round-trip must
            # not hold the event loop, #1475)
            response = await client.rpc(
                "hybrid_vector_search",
                {"query_embedding": query_embedding, "match_count": k, "filters": rpc_filters},
            ).execute()

            results = []
            for row in response.data or []:
                # Filter by similarity threshold
                similarity = row.get("similarity", 0)
                if similarity < min_similarity:
                    continue
                # Python-side post-filter for max_staleness (belt-and-suspenders;
                # primary filter is at the SQL RPC layer when migration 022 is applied)
                if _is_invalidated_under_max_staleness(row.get("metadata"), max_staleness):
                    continue
                result_metadata = row.get("metadata", {}).copy() if row.get("metadata") else {}
                result_metadata["source_name"] = row.get("source_table", "unknown")
                results.append(
                    RetrievalResult(
                        source_id=row.get("id", ""),
                        content=row.get("content", ""),
                        source=RetrievalSource.VECTOR.value,
                        score=float(similarity),
                        retrieval_method="dense",
                        metadata=result_metadata,
                    )
                )

            logger.debug(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def vector_search_by_text(
        self,
        query_text: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.5,
        max_staleness: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Search memories by text (auto-generates embedding).

        Args:
            query_text: Text query
            k: Number of results to return
            filters: Optional filters
            min_similarity: Minimum similarity threshold
            max_staleness: Optional staleness ceiling (see vector_search docstring).

        Returns:
            List of RetrievalResult with dense retrieval method
        """
        embedding_service = await self.get_embedding_service()
        embedding = await embedding_service.embed(query_text)

        return await self.vector_search(
            query_embedding=embedding,
            k=k,
            filters=filters,
            min_similarity=min_similarity,
            max_staleness=max_staleness,
        )

    # ========================================================================
    # FULL-TEXT SEARCH (Sparse Retrieval)
    # ========================================================================

    async def fulltext_search(
        self,
        query_text: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        max_staleness: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Search using PostgreSQL full-text search (BM25-like ranking).

        Uses Supabase hybrid_fulltext_search RPC function to search across:
        - causal_paths (causal relationships)
        - agent_activities (agent actions)
        - triggers (alerts and actions)

        Args:
            query_text: Search query text
            k: Number of results to return
            filters: Optional filters (brand, agent_name)
            max_staleness: Optional staleness ceiling (see vector_search docstring).

        Returns:
            List of RetrievalResult with sparse retrieval method
        """
        client = await get_async_supabase_client()

        # Build RPC filters dict, including max_staleness when provided
        rpc_filters: Dict[str, Any] = dict(filters or {})
        if max_staleness is not None:
            rpc_filters["max_staleness"] = max_staleness

        try:
            # Call Supabase RPC function (awaited — the RPC round-trip must
            # not hold the event loop, #1475)
            response = await client.rpc(
                "hybrid_fulltext_search",
                {"search_query": query_text, "match_count": k, "filters": rpc_filters},
            ).execute()

            # Python-side post-filter for max_staleness (belt-and-suspenders).
            # The SQL RPC (migration 022) is the authoritative filter for
            # tables carrying invalidated_at (currently `triggers`); this
            # post-filter catches any row whose metadata leaks through with
            # invalidated_at set when max_staleness < 1.0.
            filtered_rows = [
                row
                for row in (response.data or [])
                if not _is_invalidated_under_max_staleness(row.get("metadata"), max_staleness)
            ]

            results = []
            max_rank = 0.0

            # First pass: find max rank for normalization (over filtered rows)
            for row in filtered_rows:
                rank = float(row.get("rank", 0))
                if rank > max_rank:
                    max_rank = rank

            # Second pass: normalize and create results
            for row in filtered_rows:
                rank = float(row.get("rank", 0))
                # Normalize rank to 0-1 score
                normalized_score = rank / max_rank if max_rank > 0 else 0.0

                result_metadata = row.get("metadata", {}).copy() if row.get("metadata") else {}
                result_metadata["source_name"] = row.get("source_table", "unknown")
                results.append(
                    RetrievalResult(
                        source_id=row.get("id", ""),
                        content=row.get("content", ""),
                        source=RetrievalSource.FULLTEXT.value,
                        score=normalized_score,
                        retrieval_method="sparse",
                        metadata=result_metadata,
                    )
                )

            logger.debug(f"Fulltext search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Fulltext search failed: {e}")
            return []

    # ========================================================================
    # GRAPH TRAVERSAL (Graph Retrieval)
    # ========================================================================

    def graph_traverse(
        self, entity_id: str, relationship: str = "causal_path", max_depth: int = 3
    ) -> List[RetrievalResult]:
        """
        Traverse semantic graph to find related entities.

        Uses FalkorDB semantic memory for graph operations.

        Args:
            entity_id: Starting entity ID
            relationship: Type of relationship to follow
            max_depth: Maximum traversal depth

        Returns:
            List of RetrievalResult with graph retrieval method
        """
        semantic = get_semantic_memory()

        try:
            if relationship == "causal_path":
                # Traverse causal chains
                chains = semantic.traverse_causal_chain(entity_id, max_depth=max_depth)
                return self._chains_to_results(chains)

            elif relationship == "patient_network":
                # Get patient network
                network = semantic.get_patient_network(entity_id, max_depth=max_depth)
                return self._network_to_results(network, "patient")

            elif relationship == "hcp_network":
                # Get HCP influence network
                network = semantic.get_hcp_influence_network(entity_id, max_depth=max_depth)
                return self._network_to_results(network, "hcp")

            else:
                # Generic traversal using find_common_paths
                paths = semantic.find_common_paths(entity_id, entity_id, max_depth=max_depth)
                return self._paths_to_results(paths)

        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return []

    def graph_traverse_kpi(
        self, kpi_name: str, min_confidence: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Find causal paths impacting a specific KPI.

        Args:
            kpi_name: Name of the KPI (e.g., "TRx", "NRx", "conversion_rate")
            min_confidence: Minimum confidence threshold

        Returns:
            List of RetrievalResult with graph retrieval method
        """
        semantic = get_semantic_memory()

        try:
            paths = semantic.find_causal_paths_for_kpi(kpi_name, min_confidence=min_confidence)
            return self._paths_to_results(paths)

        except Exception as e:
            logger.error(f"KPI path traversal failed: {e}")
            return []

    # ========================================================================
    # RESULT CONVERSION HELPERS
    # ========================================================================

    def _chains_to_results(self, chains: List[Dict]) -> List[RetrievalResult]:
        """Convert causal chain results to RetrievalResult format."""
        results = []
        for i, chain in enumerate(chains):
            # Build content from chain path
            path_nodes = chain.get("path", [])
            content = " → ".join(str(node) for node in path_nodes)

            results.append(
                RetrievalResult(
                    source_id=chain.get("start_entity_id", f"chain_{i}"),
                    content=content or f"Causal chain {i + 1}",
                    source=RetrievalSource.GRAPH.value,
                    score=chain.get("confidence", 0.8),
                    retrieval_method="graph",
                    metadata={
                        "source_name": "semantic_graph",
                        "path_length": chain.get("path_length", len(path_nodes)),
                        "relationships": chain.get("relationships", []),
                        "effect_sizes": chain.get("effect_sizes", []),
                    },
                )
            )

        return results

    def _network_to_results(self, network: Dict, network_type: str) -> List[RetrievalResult]:
        """Convert network result to RetrievalResult format."""
        results = []

        # Add center node
        center = network.get("center_node", {})
        if center:
            results.append(
                RetrievalResult(
                    source_id=center.get("id", ""),
                    content=f"{network_type.upper()} center: {center.get('id', 'unknown')}",
                    source=RetrievalSource.GRAPH.value,
                    score=1.0,  # Center node has highest relevance
                    retrieval_method="graph",
                    metadata={
                        "source_name": "semantic_graph",
                        "node_type": "center",
                        "properties": center.get("properties", {}),
                    },
                )
            )

        # Add connected nodes
        connections = network.get("connections", [])
        for i, conn in enumerate(connections):
            # Score decreases with distance
            depth = conn.get("depth", 1)
            score = max(0.5, 1.0 - (depth * 0.2))

            results.append(
                RetrievalResult(
                    source_id=conn.get("node_id", f"conn_{i}"),
                    content=f"Connected {conn.get('node_type', 'entity')}: {conn.get('node_id', '')}",
                    source=RetrievalSource.GRAPH.value,
                    score=score,
                    retrieval_method="graph",
                    metadata={
                        "source_name": "semantic_graph",
                        "node_type": conn.get("node_type"),
                        "relationship": conn.get("relationship"),
                        "depth": depth,
                        "properties": conn.get("properties", {}),
                    },
                )
            )

        return results

    def _paths_to_results(self, paths: List[Dict]) -> List[RetrievalResult]:
        """Convert path results to RetrievalResult format."""
        results = []
        for i, path in enumerate(paths):
            # Build content from path
            nodes = path.get("nodes", [])
            content = " → ".join(str(n) for n in nodes)

            results.append(
                RetrievalResult(
                    source_id=path.get("path_id", f"path_{i}"),
                    content=content or f"Path {i + 1}",
                    source=RetrievalSource.GRAPH.value,
                    score=path.get("confidence", 0.7),
                    retrieval_method="graph",
                    metadata={
                        "source_name": "semantic_graph",
                        "path_length": path.get("length", len(nodes)),
                        "relationships": path.get("relationships", []),
                        "kpi_impact": path.get("kpi_impact"),
                    },
                )
            )

        return results


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_memory_connector: Optional[MemoryConnector] = None


def get_memory_connector() -> MemoryConnector:
    """
    Get or create memory connector singleton.

    Returns:
        MemoryConnector: Singleton instance
    """
    global _memory_connector
    if _memory_connector is None:
        _memory_connector = MemoryConnector()
    return _memory_connector


def reset_memory_connector() -> None:
    """Reset the memory connector singleton (for testing)."""
    global _memory_connector
    _memory_connector = None
