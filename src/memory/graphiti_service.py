"""
E2I Graphiti Service
Main Graphiti service wrapper for automatic entity/relationship extraction.

Usage:
    from src.memory.graphiti_service import E2IGraphitiService, get_graphiti_service

    # Get service singleton
    service = await get_graphiti_service()

    # Add an episode (extracts entities/relationships automatically)
    result = await service.add_episode(
        content="Dr. Smith prescribed Remibrutinib to patient P123 for CSU treatment.",
        source="orchestrator",
        session_id="session-abc",
        metadata={"agent_tier": 1}
    )

    # Search the knowledge graph
    results = await service.search("What treatments does Dr. Smith prescribe?")

    # Get entity subgraph
    subgraph = await service.get_entity_subgraph("HCP:smith-001", max_depth=2)
"""

import os
import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from uuid import uuid4

from .graphiti_config import (
    E2IEntityType,
    E2IRelationshipType,
    get_graphiti_config,
    GraphitiConfig,
)
from .services.factories import get_falkordb_client, ServiceConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from an episode."""
    entity_id: str
    entity_type: E2IEntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ExtractedRelationship:
    """A relationship extracted from an episode."""
    source_id: str
    source_type: E2IEntityType
    target_id: str
    target_type: E2IEntityType
    relationship_type: E2IRelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class EpisodeResult:
    """Result of adding an episode to the graph."""
    episode_id: str
    entities_extracted: List[ExtractedEntity]
    relationships_extracted: List[ExtractedRelationship]
    success: bool
    error: Optional[str] = None


@dataclass
class SearchResult:
    """Result from a graph search."""
    entity_id: str
    entity_type: str
    name: str
    score: float
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SubgraphResult:
    """Result of a subgraph query."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    center_entity_id: str
    depth: int


class E2IGraphitiService:
    """
    Graphiti integration for automatic knowledge extraction.

    This service wraps the Graphiti framework to provide:
    - Automatic entity and relationship extraction from text
    - Temporal episode tracking
    - Graph search and traversal
    - Integration with existing FalkorDB semantic memory
    """

    def __init__(self, config: Optional[GraphitiConfig] = None):
        """
        Initialize the Graphiti service.

        Args:
            config: Optional GraphitiConfig. If not provided, loads from YAML.
        """
        self.config = config or get_graphiti_config()
        self._graphiti = None
        self._falkordb = None
        self._initialized = False
        self._llm_client = None

    async def initialize(self) -> None:
        """
        Initialize the Graphiti client and FalkorDB connection.

        This method must be called before using the service.
        """
        if self._initialized:
            return

        logger.info("Initializing E2I Graphiti Service...")

        try:
            # Initialize FalkorDB connection
            self._falkordb = get_falkordb_client()
            self._graph = self._falkordb.select_graph(self.config.graph_name)

            # Initialize Graphiti client
            await self._init_graphiti()

            self._initialized = True
            logger.info(f"Graphiti service initialized with graph: {self.config.graph_name}")

        except ImportError as e:
            logger.error(f"Graphiti package not installed: {e}")
            raise ServiceConnectionError(
                "Graphiti",
                "graphiti-core package is not installed. Run: pip install graphiti-core[falkordb,anthropic]"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti service: {e}")
            raise ServiceConnectionError("Graphiti", f"Failed to initialize: {e}", e)

    async def _init_graphiti(self) -> None:
        """Initialize the Graphiti client with FalkorDB backend."""
        try:
            from graphiti_core import Graphiti
            from graphiti_core.llm_client.anthropic_client import AnthropicClient
            from graphiti_core.llm_client import LLMConfig
            from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            # Initialize LLM client for entity extraction
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ServiceConnectionError(
                    "Graphiti",
                    "ANTHROPIC_API_KEY environment variable is not set"
                )

            # Create LLM config and client
            # Claude 3.5 Haiku max output tokens is 8192
            llm_config = LLMConfig(
                api_key=api_key,
                model=self.config.model,
                max_tokens=8192,
            )
            llm_client = AnthropicClient(config=llm_config)

            # Initialize embedder (required for Graphiti search)
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                embedder_config = OpenAIEmbedderConfig(api_key=openai_key)
                embedder = OpenAIEmbedder(config=embedder_config)
            else:
                logger.warning("OPENAI_API_KEY not set, using default embedder")
                embedder = None

            # Create FalkorDB driver (uses Redis protocol, not Neo4j Bolt)
            falkor_driver = FalkorDriver(
                host=self.config.falkordb_host,
                port=self.config.falkordb_port,
                database=self.config.graph_name,
            )

            # Create Graphiti instance with FalkorDB driver
            self._graphiti = Graphiti(
                graph_driver=falkor_driver,
                llm_client=llm_client,
                embedder=embedder,
            )

            # Build indices and constraints for Graphiti
            await self._graphiti.build_indices_and_constraints()

            self._llm_client = llm_client
            logger.info("Graphiti client initialized successfully with FalkorDB driver")

        except ImportError as e:
            logger.warning(f"Graphiti import failed: {e}, using fallback mode")
            self._graphiti = None
        except Exception as e:
            logger.warning(f"Full Graphiti initialization failed: {e}, using fallback mode")
            self._graphiti = None

    async def add_episode(
        self,
        content: str,
        source: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        reference_time: Optional[datetime] = None,
    ) -> EpisodeResult:
        """
        Add an episode to the knowledge graph.

        This method:
        1. Stores the episode content
        2. Extracts entities and relationships using the LLM
        3. Updates the graph with extracted knowledge

        Args:
            content: The text content of the episode
            source: Source of the episode (agent_name or "user")
            session_id: Session identifier for grouping episodes
            metadata: Optional metadata to attach to the episode
            reference_time: Optional timestamp (defaults to now)

        Returns:
            EpisodeResult with extracted entities and relationships
        """
        if not self._initialized:
            await self.initialize()

        episode_id = str(uuid4())
        ref_time = reference_time or datetime.now(timezone.utc)
        metadata = metadata or {}

        try:
            if self._graphiti is not None:
                # Use Graphiti for extraction
                result = await self._graphiti.add_episode(
                    name=f"episode_{episode_id[:8]}",
                    episode_body=content,
                    source_description=source,
                    reference_time=ref_time,
                    group_id=session_id,
                )

                # Convert Graphiti result to our format
                entities = self._convert_graphiti_entities(result)
                relationships = self._convert_graphiti_relationships(result)

                logger.info(
                    f"Episode {episode_id[:8]} added: {len(entities)} entities, "
                    f"{len(relationships)} relationships"
                )

                return EpisodeResult(
                    episode_id=episode_id,
                    entities_extracted=entities,
                    relationships_extracted=relationships,
                    success=True,
                )
            else:
                # Fallback: Store episode directly without extraction
                return await self._add_episode_fallback(
                    episode_id, content, source, session_id, ref_time, metadata
                )

        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            return EpisodeResult(
                episode_id=episode_id,
                entities_extracted=[],
                relationships_extracted=[],
                success=False,
                error=str(e),
            )

    async def _add_episode_fallback(
        self,
        episode_id: str,
        content: str,
        source: str,
        session_id: str,
        reference_time: datetime,
        metadata: Dict[str, Any],
    ) -> EpisodeResult:
        """
        Fallback episode storage without Graphiti extraction.

        Stores the episode node directly in FalkorDB.
        """
        try:
            # Create episode node
            query = """
            CREATE (e:Episode {
                episode_id: $episode_id,
                content: $content,
                source: $source,
                session_id: $session_id,
                valid_at: $valid_at,
                created_at: $created_at,
                metadata: $metadata
            })
            RETURN e
            """
            self._graph.query(
                query,
                params={
                    "episode_id": episode_id,
                    "content": content[:10000],  # Limit content size
                    "source": source,
                    "session_id": session_id,
                    "valid_at": reference_time.isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": str(metadata),
                }
            )

            logger.info(f"Episode {episode_id[:8]} stored in fallback mode")
            return EpisodeResult(
                episode_id=episode_id,
                entities_extracted=[],
                relationships_extracted=[],
                success=True,
            )

        except Exception as e:
            logger.error(f"Fallback episode storage failed: {e}")
            return EpisodeResult(
                episode_id=episode_id,
                entities_extracted=[],
                relationships_extracted=[],
                success=False,
                error=str(e),
            )

    def _convert_graphiti_entities(self, graphiti_result: Any) -> List[ExtractedEntity]:
        """Convert Graphiti extraction result to ExtractedEntity list."""
        entities = []

        # AddEpisodeResults uses 'nodes' for EntityNode list
        nodes = getattr(graphiti_result, 'nodes', [])
        for entity in nodes:
            try:
                # EntityNode uses 'labels' (list) not 'label', and 'attributes' not 'properties'
                labels = getattr(entity, 'labels', [])
                first_label = labels[0] if labels else 'Entity'
                entity_type = self._map_entity_type(first_label)
                entities.append(ExtractedEntity(
                    entity_id=getattr(entity, 'uuid', ''),
                    entity_type=entity_type,
                    name=getattr(entity, 'name', ''),
                    properties=getattr(entity, 'attributes', {}),
                    confidence=1.0,  # Graphiti doesn't expose confidence on nodes
                ))
            except Exception as e:
                logger.warning(f"Failed to convert entity: {e}")

        return entities

    def _convert_graphiti_relationships(self, graphiti_result: Any) -> List[ExtractedRelationship]:
        """Convert Graphiti extraction result to ExtractedRelationship list."""
        relationships = []

        # AddEpisodeResults uses 'edges' for EntityEdge list
        edges = getattr(graphiti_result, 'edges', [])
        for edge in edges:
            try:
                # EntityEdge uses 'name' not 'label', and source_node_uuid/target_node_uuid
                edge_name = getattr(edge, 'name', 'RELATES_TO')
                rel_type = self._map_relationship_type(edge_name)
                relationships.append(ExtractedRelationship(
                    source_id=getattr(edge, 'source_node_uuid', ''),
                    source_type=E2IEntityType.AGENT,  # Default - node types not on edge
                    target_id=getattr(edge, 'target_node_uuid', ''),
                    target_type=E2IEntityType.AGENT,  # Default - node types not on edge
                    relationship_type=rel_type,
                    properties=getattr(edge, 'attributes', {}),
                    confidence=1.0,  # Graphiti doesn't expose confidence on edges
                ))
            except Exception as e:
                logger.warning(f"Failed to convert relationship: {e}")

        return relationships

    def _map_entity_type(self, label: str) -> E2IEntityType:
        """Map a string label to E2IEntityType."""
        label_upper = label.upper()
        for entity_type in E2IEntityType:
            if entity_type.value.upper() == label_upper:
                return entity_type
        # Default to Agent for unknown types
        return E2IEntityType.AGENT

    def _map_relationship_type(self, label: str) -> E2IRelationshipType:
        """Map a string label to E2IRelationshipType."""
        label_upper = label.upper().replace(" ", "_")
        for rel_type in E2IRelationshipType:
            if rel_type.value == label_upper:
                return rel_type
        # Default to RELATES_TO for unknown types
        return E2IRelationshipType.RELATES_TO

    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        entity_types: Optional[List[E2IEntityType]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Search the knowledge graph.

        Args:
            query: Natural language search query
            session_id: Optional session to scope the search
            entity_types: Optional list of entity types to filter
            limit: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self._graphiti is not None:
                # Use Graphiti search
                results = await self._graphiti.search(
                    query=query,
                    group_ids=[session_id] if session_id else None,
                    num_results=limit,
                )
                return self._convert_search_results(results, entity_types)
            else:
                # Fallback to direct FalkorDB search
                return await self._search_fallback(query, entity_types, limit)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _search_fallback(
        self,
        query: str,
        entity_types: Optional[List[E2IEntityType]],
        limit: int,
    ) -> List[SearchResult]:
        """Fallback search using direct FalkorDB queries."""
        try:
            # Build entity type filter
            type_filter = ""
            if entity_types:
                labels = [et.value for et in entity_types]
                type_filter = f"WHERE any(l in labels(n) WHERE l IN {labels})"

            # Search by name/content containing query terms
            cypher = f"""
            MATCH (n)
            {type_filter}
            WHERE n.name CONTAINS $query OR n.content CONTAINS $query
            RETURN n, labels(n) as types
            LIMIT $limit
            """

            result = self._graph.query(
                cypher,
                params={"query": query, "limit": limit}
            )

            search_results = []
            for row in result.result_set:
                node = row[0]
                types = row[1]

                search_results.append(SearchResult(
                    entity_id=node.properties.get("id", str(uuid4())),
                    entity_type=types[0] if types else "Unknown",
                    name=node.properties.get("name", ""),
                    score=1.0,  # No scoring in fallback mode
                    properties=dict(node.properties),
                ))

            return search_results

        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    def _convert_search_results(
        self,
        graphiti_results: Any,
        entity_types: Optional[List[E2IEntityType]],
    ) -> List[SearchResult]:
        """Convert Graphiti search results to SearchResult list."""
        results = []

        for item in graphiti_results:
            try:
                entity_type = item.label if hasattr(item, 'label') else "Unknown"

                # Filter by entity types if specified
                if entity_types:
                    try:
                        mapped_type = self._map_entity_type(entity_type)
                        if mapped_type not in entity_types:
                            continue
                    except ValueError:
                        continue

                results.append(SearchResult(
                    entity_id=item.uuid if hasattr(item, 'uuid') else str(uuid4()),
                    entity_type=entity_type,
                    name=item.name if hasattr(item, 'name') else "",
                    score=item.score if hasattr(item, 'score') else 1.0,
                    properties=item.properties if hasattr(item, 'properties') else {},
                ))

            except Exception as e:
                logger.warning(f"Failed to convert search result: {e}")

        return results

    async def get_entity_subgraph(
        self,
        entity_id: str,
        max_depth: int = 2,
    ) -> SubgraphResult:
        """
        Get the subgraph around an entity.

        Args:
            entity_id: The ID of the center entity
            max_depth: Maximum traversal depth

        Returns:
            SubgraphResult with nodes and edges
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Query subgraph using Cypher
            cypher = f"""
            MATCH path = (center {{id: $entity_id}})-[*0..{max_depth}]-(related)
            WITH nodes(path) as ns, relationships(path) as rs
            UNWIND ns as n
            UNWIND rs as r
            RETURN DISTINCT n, r
            """

            result = self._graph.query(cypher, params={"entity_id": entity_id})

            nodes = []
            edges = []
            seen_nodes = set()
            seen_edges = set()

            for row in result.result_set:
                node = row[0]
                rel = row[1] if len(row) > 1 else None

                # Add node
                node_id = node.properties.get("id", str(id(node)))
                if node_id not in seen_nodes:
                    nodes.append({
                        "id": node_id,
                        "labels": list(node.labels) if hasattr(node, 'labels') else [],
                        "properties": dict(node.properties),
                    })
                    seen_nodes.add(node_id)

                # Add edge
                if rel:
                    edge_key = f"{rel.src_node}-{rel.relation}-{rel.dest_node}"
                    if edge_key not in seen_edges:
                        edges.append({
                            "source": str(rel.src_node),
                            "target": str(rel.dest_node),
                            "type": rel.relation,
                            "properties": dict(rel.properties) if hasattr(rel, 'properties') else {},
                        })
                        seen_edges.add(edge_key)

            return SubgraphResult(
                nodes=nodes,
                edges=edges,
                center_entity_id=entity_id,
                depth=max_depth,
            )

        except Exception as e:
            logger.error(f"Subgraph query failed: {e}")
            return SubgraphResult(
                nodes=[],
                edges=[],
                center_entity_id=entity_id,
                depth=max_depth,
            )

    async def get_causal_chains(
        self,
        start_entity_id: Optional[str] = None,
        kpi_name: Optional[str] = None,
        min_confidence: float = 0.5,
        max_depth: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query causal chains from the graph.

        Args:
            start_entity_id: Optional starting entity
            kpi_name: Optional KPI to find causes for
            min_confidence: Minimum confidence threshold
            max_depth: Maximum chain length

        Returns:
            List of causal chain dictionaries
        """
        if not self._initialized:
            await self.initialize()

        try:
            if kpi_name:
                # Find causes of a KPI
                cypher = f"""
                MATCH path = (cause)-[:CAUSES|IMPACTS*1..{max_depth}]->(kpi:KPI {{name: $kpi_name}})
                WHERE all(r in relationships(path) WHERE r.confidence >= $min_confidence)
                RETURN path
                LIMIT 20
                """
                params = {"kpi_name": kpi_name, "min_confidence": min_confidence}
            elif start_entity_id:
                # Find effects from an entity
                cypher = f"""
                MATCH path = (start {{id: $entity_id}})-[:CAUSES|IMPACTS*1..{max_depth}]->(effect)
                WHERE all(r in relationships(path) WHERE r.confidence >= $min_confidence)
                RETURN path
                LIMIT 20
                """
                params = {"entity_id": start_entity_id, "min_confidence": min_confidence}
            else:
                # Get all causal chains
                cypher = f"""
                MATCH path = (a)-[:CAUSES|IMPACTS*1..{max_depth}]->(b)
                WHERE all(r in relationships(path) WHERE r.confidence >= $min_confidence)
                RETURN path
                LIMIT 50
                """
                params = {"min_confidence": min_confidence}

            result = self._graph.query(cypher, params=params)

            chains = []
            for row in result.result_set:
                path = row[0]
                chains.append(self._path_to_chain(path))

            return chains

        except Exception as e:
            logger.error(f"Causal chain query failed: {e}")
            return []

    def _path_to_chain(self, path: Any) -> Dict[str, Any]:
        """Convert a graph path to a chain dictionary."""
        nodes = []
        edges = []

        if hasattr(path, 'nodes'):
            for node in path.nodes():
                nodes.append({
                    "id": node.properties.get("id", ""),
                    "type": list(node.labels)[0] if node.labels else "Unknown",
                    "name": node.properties.get("name", ""),
                })

        if hasattr(path, 'relationships'):
            for rel in path.relationships():
                edges.append({
                    "type": rel.relation,
                    "confidence": rel.properties.get("confidence", 1.0),
                    "effect_size": rel.properties.get("effect_size"),
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "length": len(nodes) - 1,
        }

    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with node counts, edge counts, etc.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Count nodes by type
            node_counts = {}
            for entity_type in E2IEntityType:
                cypher = f"MATCH (n:{entity_type.value}) RETURN count(n) as count"
                result = self._graph.query(cypher)
                count = result.result_set[0][0] if result.result_set else 0
                node_counts[entity_type.value] = count

            # Count relationships by type
            edge_counts = {}
            for rel_type in E2IRelationshipType:
                cypher = f"MATCH ()-[r:{rel_type.value}]->() RETURN count(r) as count"
                result = self._graph.query(cypher)
                count = result.result_set[0][0] if result.result_set else 0
                edge_counts[rel_type.value] = count

            # Total counts
            total_nodes = sum(node_counts.values())
            total_edges = sum(edge_counts.values())

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "nodes_by_type": node_counts,
                "edges_by_type": edge_counts,
                "graph_name": self.config.graph_name,
            }

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "nodes_by_type": {},
                "edges_by_type": {},
                "graph_name": self.config.graph_name,
                "error": str(e),
            }

    async def close(self) -> None:
        """Close the Graphiti service and release resources."""
        if self._graphiti is not None:
            await self._graphiti.close()
            self._graphiti = None

        self._initialized = False
        logger.info("Graphiti service closed")


# Singleton instance
_graphiti_service: Optional[E2IGraphitiService] = None


async def get_graphiti_service() -> E2IGraphitiService:
    """
    Get the Graphiti service singleton.

    Returns:
        E2IGraphitiService instance (initialized)
    """
    global _graphiti_service
    if _graphiti_service is None:
        _graphiti_service = E2IGraphitiService()
        await _graphiti_service.initialize()
    return _graphiti_service


def reset_graphiti_service() -> None:
    """Reset the Graphiti service singleton (for testing)."""
    global _graphiti_service
    if _graphiti_service is not None:
        # Try to close properly, handling both sync and async contexts
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - schedule close as a task
            loop.create_task(_graphiti_service.close())
        except RuntimeError:
            # No running loop - run close synchronously
            try:
                asyncio.run(_graphiti_service.close())
            except Exception as e:
                logger.warning(f"Failed to close graphiti service: {e}")
    _graphiti_service = None
    logger.info("Graphiti service reset")


async def reset_graphiti_service_async() -> None:
    """Async version of reset for use in async contexts."""
    global _graphiti_service
    if _graphiti_service is not None:
        await _graphiti_service.close()
    _graphiti_service = None
    logger.info("Graphiti service reset")
