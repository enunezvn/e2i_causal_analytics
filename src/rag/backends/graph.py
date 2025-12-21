"""
E2I Hybrid RAG - Graph Backend Client

FalkorDB/Graphiti knowledge graph backend for:
- Causal path discovery
- Entity relationship traversal
- Semantic memory retrieval
- Graph visualization data

Part of Phase 1, Checkpoint 1.3.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.rag.config import FalkorDBConfig, HybridSearchConfig
from src.rag.types import (
    ExtractedEntities,
    GraphPath,
    RetrievalResult,
    RetrievalSource,
)
from src.rag.exceptions import GraphSearchError

logger = logging.getLogger(__name__)


class GraphBackend:
    """
    FalkorDB/Graphiti graph search backend.

    Performs Cypher queries to find:
    - Causal paths between entities
    - Related concepts (brands -> KPIs -> regions)
    - Historical patterns and relationships
    - Semantic memory nodes from Graphiti

    Example:
        ```python
        from falkordb import FalkorDB
        from src.rag.backends import GraphBackend
        from src.rag.config import FalkorDBConfig, HybridSearchConfig

        falkordb = FalkorDB(host="localhost", port=6381)
        backend = GraphBackend(
            falkordb_client=falkordb,
            falkordb_config=FalkorDBConfig(),
            search_config=HybridSearchConfig()
        )

        # Search with extracted entities
        entities = ExtractedEntities(
            brands=["Remibrutinib"],
            kpis=["conversion_rate"]
        )
        results = await backend.search(entities)

        # Get graph for visualization
        graph_data = await backend.get_causal_subgraph("node-123", max_depth=2)
        ```
    """

    def __init__(
        self,
        falkordb_client: Any,
        falkordb_config: Optional[FalkorDBConfig] = None,
        search_config: Optional[HybridSearchConfig] = None
    ):
        """
        Initialize the graph backend.

        Args:
            falkordb_client: FalkorDB client instance
            falkordb_config: FalkorDB connection configuration
            search_config: Search configuration
        """
        self.client = falkordb_client
        self.falkordb_config = falkordb_config or FalkorDBConfig()
        self.search_config = search_config or HybridSearchConfig()
        self._last_latency_ms: float = 0.0
        self._graph = None

    def _get_graph(self) -> Any:
        """Get or create graph connection."""
        if self._graph is None:
            self._graph = self.client.select_graph(self.falkordb_config.graph_name)
        return self._graph

    async def search(
        self,
        entities: ExtractedEntities,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Execute graph search using extracted entities.

        Args:
            entities: Extracted entities from query (brands, KPIs, etc.)
            query: Original query text (for context)
            filters: Optional filters
            top_k: Override default top_k from config

        Returns:
            List of RetrievalResult with graph_context

        Raises:
            GraphSearchError: If search fails or times out
        """
        start_time = time.time()
        top_k = top_k or self.search_config.graph_top_k
        timeout_seconds = self.search_config.graph_timeout_ms / 1000

        try:
            # Build Cypher query from entities
            cypher_query = self._build_entity_query(entities, top_k)

            # Execute query with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._execute_query,
                    cypher_query
                ),
                timeout=timeout_seconds
            )

            self._last_latency_ms = (time.time() - start_time) * 1000

            # Parse results
            results = self._parse_query_results(result)

            logger.debug(
                f"Graph search returned {len(results)} results "
                f"in {self._last_latency_ms:.1f}ms"
            )

            return results

        except asyncio.TimeoutError:
            self._last_latency_ms = self.search_config.graph_timeout_ms
            logger.warning(
                f"Graph search timeout after {self.search_config.graph_timeout_ms}ms"
            )
            raise GraphSearchError(
                message=f"Graph search timeout after {self.search_config.graph_timeout_ms}ms",
                backend="falkordb_graph",
                details={"timeout_ms": self.search_config.graph_timeout_ms}
            )

        except Exception as e:
            self._last_latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Graph search error: {e}")
            raise GraphSearchError(
                message=f"Graph search failed: {e}",
                backend="falkordb_graph",
                original_error=e
            )

    def _execute_query(self, cypher_query: str) -> Any:
        """Execute Cypher query synchronously."""
        graph = self._get_graph()
        return graph.query(cypher_query)

    def _build_entity_query(
        self,
        entities: ExtractedEntities,
        limit: int
    ) -> str:
        """
        Build Cypher query from extracted entities.

        If no specific entities found, returns a general causal path query.
        """
        match_clauses = []
        where_clauses = []
        return_items = []

        # Build entity-specific matches
        if entities.brands:
            brand_list = ", ".join(f"'{b}'" for b in entities.brands)
            match_clauses.append("(b:Brand)")
            where_clauses.append(f"b.name IN [{brand_list}]")
            return_items.append("b")

        if entities.regions:
            region_list = ", ".join(f"'{r}'" for r in entities.regions)
            match_clauses.append("(r:Region)")
            where_clauses.append(f"r.name IN [{region_list}]")
            return_items.append("r")

        if entities.kpis:
            kpi_list = ", ".join(f"'{k}'" for k in entities.kpis)
            match_clauses.append("(k:KPI)")
            where_clauses.append(f"k.name IN [{kpi_list}]")
            return_items.append("k")

        if entities.agents:
            agent_list = ", ".join(f"'{a}'" for a in entities.agents)
            match_clauses.append("(a:Agent)")
            where_clauses.append(f"a.name IN [{agent_list}]")
            return_items.append("a")

        # Default: find recent causal paths
        if not match_clauses:
            return f"""
                MATCH path = (n)-[rel:CAUSES|AFFECTS|CORRELATES*1..{self.falkordb_config.max_path_length}]->(m)
                RETURN n, rel, m, length(path) as path_length
                ORDER BY path_length ASC
                LIMIT {limit}
            """

        # Build query with entity matches and causal path traversal
        match_str = "MATCH " + ", ".join(match_clauses)
        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Find causal paths from matched entities
        return f"""
            {match_str}
            {where_str}
            OPTIONAL MATCH path = ({return_items[0]})-[rel:CAUSES|AFFECTS*1..{self.falkordb_config.max_path_length}]->(target)
            RETURN {', '.join(return_items)},
                   path,
                   nodes(path) as path_nodes,
                   relationships(path) as path_rels,
                   length(path) as path_length
            ORDER BY path_length ASC
            LIMIT {limit}
        """

    def _parse_query_results(self, result: Any) -> List[RetrievalResult]:
        """Parse FalkorDB query results into RetrievalResult objects."""
        results = []
        seen_ids = set()

        for record in result.result_set:
            parsed = self._parse_record(record)
            if parsed and parsed["id"] not in seen_ids:
                seen_ids.add(parsed["id"])

                # Calculate relevance based on path length
                path_length = parsed.get("path_length", 0)
                relevance = max(0.2, 1.0 - (path_length * 0.15))

                if relevance >= self.search_config.graph_min_relevance:
                    results.append(RetrievalResult(
                        id=parsed["id"],
                        content=parsed.get("description", ""),
                        source=RetrievalSource.GRAPH,
                        score=relevance,
                        metadata=parsed.get("properties", {}),
                        graph_context={
                            "connected_nodes": parsed.get("neighbors", []),
                            "path_length": path_length,
                            "relationship_types": parsed.get("rel_types", []),
                            "node_type": parsed.get("node_type", "unknown")
                        },
                        query_latency_ms=self._last_latency_ms,
                        raw_score=relevance
                    ))

        return results

    def _parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """Parse a single FalkorDB record."""
        try:
            # Handle different record structures
            if not record or len(record) == 0:
                return None

            # First item is usually the main node
            node = record[0]
            if not hasattr(node, 'id'):
                return None

            properties = dict(node.properties) if hasattr(node, 'properties') else {}

            result = {
                "id": str(node.id),
                "description": properties.get(
                    "description",
                    properties.get("name", f"Node-{node.id}")
                ),
                "properties": properties,
                "node_type": node.labels[0] if hasattr(node, 'labels') and node.labels else "unknown",
                "neighbors": [],
                "rel_types": [],
                "path_length": 0
            }

            # Extract path information if available
            for item in record[1:]:
                if isinstance(item, int):
                    result["path_length"] = item
                elif isinstance(item, list):
                    # Could be nodes or relationships
                    for sub_item in item:
                        if hasattr(sub_item, 'id'):
                            result["neighbors"].append(str(sub_item.id))
                        elif hasattr(sub_item, 'type'):
                            result["rel_types"].append(sub_item.type)

            return result

        except Exception as e:
            logger.debug(f"Error parsing graph record: {e}")
            return None

    async def get_causal_subgraph(
        self,
        center_node_id: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get a subgraph for visualization.

        Args:
            center_node_id: ID of center node for ego graph
            node_types: Filter by node types
            relationship_types: Filter by relationship types
            max_depth: Max traversal depth
            limit: Max nodes to return

        Returns:
            Dict with 'nodes', 'edges', and 'metadata' for visualization
        """
        timeout_seconds = self.search_config.graph_timeout_ms / 1000

        try:
            if center_node_id:
                # Ego graph around center node
                query = f"""
                    MATCH path = (center)-[*1..{max_depth}]-(neighbor)
                    WHERE id(center) = {center_node_id}
                    UNWIND relationships(path) as rel
                    UNWIND nodes(path) as node
                    RETURN DISTINCT node, rel
                    LIMIT {limit}
                """
            else:
                # Full graph query
                rel_filter = f"[:{':'.join(relationship_types)}]" if relationship_types else "[r]"
                type_filter = ""
                if node_types:
                    type_list = " OR ".join(f"n:{t}" for t in node_types)
                    type_filter = f"WHERE ({type_list})"

                query = f"""
                    MATCH (n)-{rel_filter}->(m)
                    {type_filter}
                    RETURN n, r, m
                    LIMIT {limit}
                """

            result = await asyncio.wait_for(
                asyncio.to_thread(self._execute_query, query),
                timeout=timeout_seconds
            )

            return self._parse_graph_for_visualization(result, center_node_id, max_depth)

        except asyncio.TimeoutError:
            raise GraphSearchError(
                message="Graph subgraph query timeout",
                backend="falkordb_graph"
            )

        except Exception as e:
            raise GraphSearchError(
                message=f"Graph subgraph query failed: {e}",
                backend="falkordb_graph",
                original_error=e
            )

    def _parse_graph_for_visualization(
        self,
        result: Any,
        center_node_id: Optional[str],
        max_depth: int
    ) -> Dict[str, Any]:
        """Parse graph results for visualization format."""
        nodes_map = {}
        edges = []

        for record in result.result_set:
            for item in record:
                if hasattr(item, 'id') and hasattr(item, 'properties'):
                    # It's a node
                    node_id = str(item.id)
                    if node_id not in nodes_map:
                        props = dict(item.properties) if hasattr(item, 'properties') else {}
                        nodes_map[node_id] = {
                            "id": node_id,
                            "label": props.get("name", f"Node-{node_id}"),
                            "type": item.labels[0] if hasattr(item, 'labels') and item.labels else "Unknown",
                            "properties": props
                        }
                elif hasattr(item, 'type') and hasattr(item, 'src_node'):
                    # It's a relationship
                    edges.append({
                        "source": str(item.src_node),
                        "target": str(item.dest_node),
                        "type": item.type,
                        "properties": dict(item.properties) if hasattr(item, 'properties') else {}
                    })

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes_map),
                "total_edges": len(edges),
                "query_depth": max_depth,
                "center_node": center_node_id
            }
        }

    async def get_causal_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 4
    ) -> List[GraphPath]:
        """
        Find causal paths between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_length: Max path length

        Returns:
            List of GraphPath objects
        """
        timeout_seconds = self.search_config.graph_timeout_ms / 1000

        query = f"""
            MATCH path = (s)-[:CAUSES|AFFECTS*1..{max_length}]->(t)
            WHERE id(s) = {source_id} AND id(t) = {target_id}
            RETURN nodes(path) as path_nodes,
                   relationships(path) as path_rels,
                   length(path) as path_length
            ORDER BY path_length ASC
            LIMIT 10
        """

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._execute_query, query),
                timeout=timeout_seconds
            )

            paths = []
            for record in result.result_set:
                nodes = record[0] if len(record) > 0 else []
                rels = record[1] if len(record) > 1 else []
                path_length = record[2] if len(record) > 2 else 0

                if nodes:
                    paths.append(GraphPath(
                        source_node=str(nodes[0].id) if nodes else source_id,
                        target_node=str(nodes[-1].id) if nodes else target_id,
                        relationship_types=[r.type for r in rels if hasattr(r, 'type')],
                        path_length=path_length,
                        nodes=[
                            {
                                "id": str(n.id),
                                "label": n.properties.get("name", str(n.id)),
                                "type": n.labels[0] if n.labels else "Unknown"
                            }
                            for n in nodes if hasattr(n, 'id')
                        ]
                    ))

            return paths

        except Exception as e:
            raise GraphSearchError(
                message=f"Causal path query failed: {e}",
                backend="falkordb_graph",
                original_error=e
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the graph backend is healthy.

        Returns:
            Dict with status, latency_ms, and any error message
        """
        try:
            start_time = time.time()

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._execute_query,
                    "MATCH (n) RETURN count(n) LIMIT 1"
                ),
                timeout=5.0
            )

            latency_ms = (time.time() - start_time) * 1000
            node_count = result.result_set[0][0] if result.result_set else 0

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "node_count": node_count,
                "error": None
            }

        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "latency_ms": 5000,
                "error": "Health check timeout"
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "latency_ms": 0,
                "error": str(e)
            }

    @property
    def last_latency_ms(self) -> float:
        """Get latency from last query."""
        return self._last_latency_ms

    def __repr__(self) -> str:
        return (
            f"GraphBackend("
            f"graph={self.falkordb_config.graph_name}, "
            f"top_k={self.search_config.graph_top_k}, "
            f"timeout_ms={self.search_config.graph_timeout_ms})"
        )
