"""
E2I Agentic Memory - Semantic Memory (FalkorDB Graph)
Graph-based semantic memory for entity relationships and causal chains.

Technology: FalkorDB (Redis-compatible graph database)

Features:
- E2I entity graph (Patient, HCP, Trigger, CausalPath, etc.)
- Relationship management with confidence scores
- Patient and HCP network traversal
- Causal chain discovery
- KPI impact path finding

Usage:
    from src.memory.semantic_memory import (
        get_semantic_memory,
        query_semantic_graph,
        sync_to_semantic_graph
    )

    # Get semantic memory instance
    semantic = get_semantic_memory()

    # Add entities and relationships
    semantic.add_e2i_entity(E2IEntityType.PATIENT, "pat_123", {"name": "John"})
    semantic.add_e2i_relationship(
        E2IEntityType.PATIENT, "pat_123",
        E2IEntityType.HCP, "hcp_456",
        "TREATED_BY"
    )

    # Query networks
    network = semantic.get_patient_network("pat_123", max_depth=2)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.memory.episodic_memory import E2IEntityType
from src.memory.services.config import get_config
from src.memory.services.factories import get_falkordb_client, get_supabase_client

logger = logging.getLogger(__name__)


# ============================================================================
# LABEL MAPPINGS
# ============================================================================

# Map E2I entity types to graph node labels
E2I_TO_LABEL = {
    E2IEntityType.PATIENT: "Patient",
    E2IEntityType.HCP: "HCP",
    E2IEntityType.TRIGGER: "Trigger",
    E2IEntityType.CAUSAL_PATH: "CausalPath",
    E2IEntityType.PREDICTION: "Prediction",
    E2IEntityType.TREATMENT: "Treatment",
    E2IEntityType.EXPERIMENT: "Experiment",
    E2IEntityType.AGENT_ACTIVITY: "AgentActivity",
}

# Reverse mapping for lookups
LABEL_TO_E2I = {v: k for k, v in E2I_TO_LABEL.items()}


# ============================================================================
# SEMANTIC MEMORY CLASS
# ============================================================================


class FalkorDBSemanticMemory:
    """
    FalkorDB-based semantic memory with E2I entity support.

    Provides graph-based storage for:
    - E2I entities (patients, HCPs, triggers, causal paths)
    - Relationships between entities
    - Network traversal and discovery
    - Causal chain analysis
    """

    def __init__(self):
        """Initialize semantic memory with configuration."""
        self._config = get_config()
        self._client = None
        self._graph = None

    @property
    def client(self):
        """Lazy FalkorDB client initialization."""
        if self._client is None:
            self._client = get_falkordb_client()
        return self._client

    @property
    def graph(self):
        """Get or create the semantic graph."""
        if self._graph is None:
            graph_name = self._config.semantic.graph_name
            self._graph = self.client.select_graph(graph_name)
            logger.info(f"Selected FalkorDB graph: {graph_name}")
        return self._graph

    # ========================================================================
    # ENTITY MANAGEMENT
    # ========================================================================

    def add_e2i_entity(
        self,
        entity_type: E2IEntityType,
        entity_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add an E2I entity to the semantic graph.

        Uses MERGE to create or update the entity.

        Args:
            entity_type: E2I entity type
            entity_id: Unique entity identifier
            properties: Additional properties to store

        Returns:
            True if successful
        """
        label = E2I_TO_LABEL.get(entity_type, "Entity")
        props = properties.copy() if properties else {}
        props["e2i_entity_type"] = entity_type.value
        props["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Build property string for Cypher
        prop_items = [f"{k}: ${k}" for k in props.keys()]
        prop_string = ", ".join(prop_items)

        query = f"""
        MERGE (e:{label} {{id: $entity_id}})
        ON CREATE SET e += {{{prop_string}}}
        ON MATCH SET e += {{{prop_string}}}
        RETURN e
        """

        params = {"entity_id": entity_id, **props}
        self.graph.query(query, params)

        logger.debug(f"Added/updated {label} entity: {entity_id}")
        return True

    def get_entity(self, entity_type: E2IEntityType, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by type and ID.

        Args:
            entity_type: E2I entity type
            entity_id: Entity identifier

        Returns:
            Entity properties dict or None if not found
        """
        label = E2I_TO_LABEL.get(entity_type, "Entity")

        query = f"""
        MATCH (e:{label} {{id: $entity_id}})
        RETURN e
        """

        result = self.graph.query(query, {"entity_id": entity_id})

        if result.result_set and len(result.result_set) > 0:
            node = result.result_set[0][0]
            return dict(node.properties)

        return None

    def delete_entity(self, entity_type: E2IEntityType, entity_id: str) -> bool:
        """
        Delete an entity and its relationships.

        Args:
            entity_type: E2I entity type
            entity_id: Entity identifier

        Returns:
            True if deleted, False if not found
        """
        label = E2I_TO_LABEL.get(entity_type, "Entity")

        query = f"""
        MATCH (e:{label} {{id: $entity_id}})
        DETACH DELETE e
        """

        result = self.graph.query(query, {"entity_id": entity_id})
        deleted = result.nodes_deleted > 0

        if deleted:
            logger.debug(f"Deleted {label} entity: {entity_id}")
        return deleted

    # ========================================================================
    # RELATIONSHIP MANAGEMENT
    # ========================================================================

    def add_e2i_relationship(
        self,
        source_type: E2IEntityType,
        source_id: str,
        target_type: E2IEntityType,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a relationship between E2I entities.

        Ensures both entities exist before creating the relationship.

        Common relationship types:
        - TREATED_BY: Patient → HCP
        - PRESCRIBED: Patient → Brand
        - PRESCRIBES: HCP → Brand
        - GENERATED: Prediction → Trigger
        - CAUSES: CausalPath relationship
        - IMPACTS: CausalPath → KPI

        Args:
            source_type: Source entity type
            source_id: Source entity ID
            target_type: Target entity type
            target_id: Target entity ID
            rel_type: Relationship type (e.g., "TREATED_BY")
            properties: Relationship properties (e.g., confidence, weight)

        Returns:
            True if successful
        """
        # Ensure both entities exist
        self.add_e2i_entity(source_type, source_id)
        self.add_e2i_entity(target_type, target_id)

        source_label = E2I_TO_LABEL.get(source_type, "Entity")
        target_label = E2I_TO_LABEL.get(target_type, "Entity")

        props = properties.copy() if properties else {}
        props["updated_at"] = datetime.now(timezone.utc).isoformat()

        prop_items = [f"{k}: ${k}" for k in props.keys()]
        prop_string = ", ".join(prop_items) if prop_items else ""

        if prop_string:
            query = f"""
            MATCH (s:{source_label} {{id: $source_id}})
            MATCH (t:{target_label} {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r += {{{prop_string}}}
            RETURN r
            """
        else:
            query = f"""
            MATCH (s:{source_label} {{id: $source_id}})
            MATCH (t:{target_label} {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            RETURN r
            """

        params = {"source_id": source_id, "target_id": target_id, **props}
        self.graph.query(query, params)

        logger.debug(f"Added relationship: {source_id} -[{rel_type}]-> {target_id}")
        return True

    def get_relationships(
        self, entity_type: E2IEntityType, entity_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity.

        Args:
            entity_type: Entity type
            entity_id: Entity ID
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationship dicts with source, target, type, properties
        """
        label = E2I_TO_LABEL.get(entity_type, "Entity")

        if direction == "outgoing":
            query = f"""
            MATCH (e:{label} {{id: $entity_id}})-[r]->(t)
            RETURN e.id as source, type(r) as rel_type, t.id as target, properties(r) as props
            """
        elif direction == "incoming":
            query = f"""
            MATCH (s)-[r]->(e:{label} {{id: $entity_id}})
            RETURN s.id as source, type(r) as rel_type, e.id as target, properties(r) as props
            """
        else:
            query = f"""
            MATCH (e:{label} {{id: $entity_id}})-[r]-(connected)
            RETURN
                CASE WHEN startNode(r).id = e.id THEN e.id ELSE connected.id END as source,
                type(r) as rel_type,
                CASE WHEN endNode(r).id = e.id THEN e.id ELSE connected.id END as target,
                properties(r) as props
            """

        result = self.graph.query(query, {"entity_id": entity_id})

        relationships = []
        for record in result.result_set:
            relationships.append(
                {
                    "source": record[0],
                    "rel_type": record[1],
                    "target": record[2],
                    "properties": dict(record[3]) if record[3] else {},
                }
            )

        return relationships

    # ========================================================================
    # NETWORK TRAVERSAL
    # ========================================================================

    def get_patient_network(self, patient_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get the relationship network around a patient.

        Args:
            patient_id: Patient entity ID
            max_depth: Maximum traversal depth (1-5, clamped for safety)

        Returns:
            Dict with patient_id, hcps, treatments, triggers, causal_paths
        """
        # Sanitize max_depth to prevent injection and limit traversal
        safe_depth = max(1, min(5, int(max_depth)))

        # FalkorDB doesn't support parameterized variable-length bounds,
        # so we use string formatting with the sanitized value
        query = f"""
        MATCH (p:Patient {{id: $patient_id}})-[*1..{safe_depth}]-(connected)
        RETURN DISTINCT connected
        """

        result = self.graph.query(query, {"patient_id": patient_id})

        network = {
            "patient_id": patient_id,
            "hcps": [],
            "treatments": [],
            "triggers": [],
            "causal_paths": [],
            "brands": [],
        }

        for record in result.result_set:
            connected = record[0]  # Now first element since we only return connected
            labels = connected.labels if hasattr(connected, "labels") else []

            node_data = {
                "id": connected.properties.get("id"),
                "properties": dict(connected.properties),
            }

            if "HCP" in labels:
                network["hcps"].append(node_data)
            elif "Treatment" in labels:
                network["treatments"].append(node_data)
            elif "Trigger" in labels:
                network["triggers"].append(node_data)
            elif "CausalPath" in labels:
                network["causal_paths"].append(node_data)
            elif "Brand" in labels:
                network["brands"].append(node_data)

        return network

    def get_hcp_influence_network(self, hcp_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get the influence network around an HCP.

        Args:
            hcp_id: HCP entity ID
            max_depth: Maximum traversal depth (1-5, clamped for safety)

        Returns:
            Dict with hcp_id, influenced_hcps, patients, brands_prescribed
        """
        # Sanitize max_depth to prevent injection and limit traversal
        safe_depth = max(1, min(5, int(max_depth)))

        # FalkorDB doesn't support parameterized variable-length bounds
        query = f"""
        MATCH (h:HCP {{id: $hcp_id}})-[*1..{safe_depth}]-(connected)
        RETURN DISTINCT connected
        """

        result = self.graph.query(query, {"hcp_id": hcp_id})

        network = {"hcp_id": hcp_id, "influenced_hcps": [], "patients": [], "brands_prescribed": []}

        for record in result.result_set:
            connected = record[0]  # First element since we only return connected
            labels = connected.labels if hasattr(connected, "labels") else []

            node_data = {
                "id": connected.properties.get("id"),
                "properties": dict(connected.properties),
            }

            if "HCP" in labels:
                network["influenced_hcps"].append(node_data)
            elif "Patient" in labels:
                network["patients"].append(node_data)
            elif "Brand" in labels:
                network["brands_prescribed"].append(node_data)

        return network

    # ========================================================================
    # CAUSAL CHAIN ANALYSIS
    # ========================================================================

    def traverse_causal_chain(
        self, start_entity_id: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Traverse causal relationships from a starting entity.

        Follows CAUSES and IMPACTS relationships.

        Args:
            start_entity_id: Starting entity ID
            max_depth: Maximum chain length (1-5, clamped for safety)

        Returns:
            List of causal chains with nodes and relationships
        """
        # Sanitize max_depth to prevent injection and limit traversal
        safe_depth = max(1, min(5, int(max_depth)))

        # FalkorDB doesn't support parameterized variable-length bounds
        query = f"""
        MATCH path = (s {{id: $start_id}})-[:CAUSES|IMPACTS*1..{safe_depth}]->(t)
        RETURN
            [n IN nodes(path) | {{id: n.id, type: labels(n)[0]}}] as nodes,
            [r IN relationships(path) | {{type: type(r), conf: r.confidence}}] as rels
        """

        result = self.graph.query(query, {"start_id": start_entity_id})

        chains = []
        for record in result.result_set:
            chains.append(
                {"nodes": record[0], "relationships": record[1], "path_length": len(record[1])}
            )

        return chains

    def find_causal_paths_for_kpi(
        self, kpi_name: str, min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find all causal paths that impact a specific KPI.

        Useful for understanding what drives KPI changes.

        Args:
            kpi_name: Name of the KPI (e.g., "TRx", "NRx")
            min_confidence: Minimum confidence threshold

        Returns:
            List of causal paths with effect sizes and confidence
        """
        query = """
        MATCH (cp:CausalPath)-[r:IMPACTS]->(k:KPI {name: $kpi_name})
        WHERE r.confidence >= $min_confidence
        RETURN cp.id as path_id, cp.effect_size as effect_size,
               r.confidence as confidence, cp.method_used as method
        ORDER BY r.confidence DESC
        """

        result = self.graph.query(query, {"kpi_name": kpi_name, "min_confidence": min_confidence})

        return [
            {
                "path_id": record[0],
                "effect_size": record[1],
                "confidence": record[2],
                "method": record[3],
            }
            for record in result.result_set
        ]

    def find_common_paths(
        self, entity1_id: str, entity2_id: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find paths connecting two entities.

        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID
            max_depth: Maximum path length

        Returns:
            List of connecting paths
        """
        query = """
        MATCH path = (e1 {id: $entity1_id})-[r*1..$max_depth]-(e2 {id: $entity2_id})
        RETURN
            [n IN nodes(path) | {id: n.id, type: labels(n)[0]}] as nodes,
            [rel IN relationships(path) | type(rel)] as rel_types,
            length(path) as path_length
        ORDER BY path_length
        LIMIT 10
        """

        result = self.graph.query(
            query, {"entity1_id": entity1_id, "entity2_id": entity2_id, "max_depth": max_depth}
        )

        return [
            {"nodes": record[0], "relationship_types": record[1], "path_length": record[2]}
            for record in result.result_set
        ]

    # ========================================================================
    # GRAPH STATISTICS
    # ========================================================================

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the semantic graph.

        Returns:
            Dict with node and relationship counts by type
        """
        # Count nodes by label
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        """
        node_result = self.graph.query(node_query)

        node_counts = {}
        total_nodes = 0
        for record in node_result.result_set:
            label = record[0] or "Unknown"
            count = record[1]
            node_counts[label] = count
            total_nodes += count

        # Count relationships by type
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        """
        rel_result = self.graph.query(rel_query)

        rel_counts = {}
        total_rels = 0
        for record in rel_result.result_set:
            rel_type = record[0]
            count = record[1]
            rel_counts[rel_type] = count
            total_rels += count

        return {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "nodes_by_type": node_counts,
            "relationships_by_type": rel_counts,
        }

    # ========================================================================
    # GRAPH API METHODS
    # ========================================================================

    def list_nodes(
        self,
        entity_types: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List nodes with filtering and pagination.

        Args:
            entity_types: Filter by node labels
            search: Text search in node properties
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of node dictionaries
        """
        # Build WHERE clauses
        where_parts = []
        params = {}

        if entity_types:
            labels_match = " OR ".join([f"'{t}' IN labels(n)" for t in entity_types])
            where_parts.append(f"({labels_match})")

        if search:
            where_parts.append("(n.name CONTAINS $search OR n.id CONTAINS $search)")
            params["search"] = search

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        query = f"""
        MATCH (n)
        {where_clause}
        RETURN n, labels(n)[0] as type
        SKIP {offset}
        LIMIT {limit}
        """

        result = self.graph.query(query, params)

        nodes = []
        for record in result.result_set:
            node = record[0]
            node_type = record[1]
            node_dict = dict(node.properties)
            node_dict["type"] = node_type
            node_dict["id"] = node_dict.get("id", str(node.id))
            nodes.append(node_dict)

        return nodes

    def count_nodes(
        self, entity_types: Optional[List[str]] = None, search: Optional[str] = None
    ) -> int:
        """Count nodes matching filters."""
        where_parts = []
        params = {}

        if entity_types:
            labels_match = " OR ".join([f"'{t}' IN labels(n)" for t in entity_types])
            where_parts.append(f"({labels_match})")

        if search:
            where_parts.append("(n.name CONTAINS $search OR n.id CONTAINS $search)")
            params["search"] = search

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        query = f"""
        MATCH (n)
        {where_clause}
        RETURN count(n) as count
        """

        result = self.graph.query(query, params)
        return result.result_set[0][0] if result.result_set else 0

    def list_relationships(
        self,
        relationship_types: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List relationships with filtering and pagination.

        Args:
            relationship_types: Filter by relationship types
            source_id: Filter by source node ID
            target_id: Filter by target node ID
            min_confidence: Minimum confidence threshold
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of relationship dictionaries
        """
        where_parts = []
        params = {}

        if source_id:
            where_parts.append("s.id = $source_id")
            params["source_id"] = source_id

        if target_id:
            where_parts.append("t.id = $target_id")
            params["target_id"] = target_id

        if min_confidence is not None:
            where_parts.append("r.confidence >= $min_confidence")
            params["min_confidence"] = min_confidence

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        # Build relationship type pattern
        if relationship_types:
            rel_pattern = "|".join(relationship_types)
            rel_match = f"-[r:{rel_pattern}]->"
        else:
            rel_match = "-[r]->"

        query = f"""
        MATCH (s){rel_match}(t)
        {where_clause}
        RETURN r, s.id as source_id, t.id as target_id, type(r) as rel_type
        SKIP {offset}
        LIMIT {limit}
        """

        result = self.graph.query(query, params)

        relationships = []
        for record in result.result_set:
            rel = record[0]
            rel_dict = dict(rel.properties) if rel.properties else {}
            rel_dict["id"] = str(rel.id)
            rel_dict["source_id"] = record[1]
            rel_dict["target_id"] = record[2]
            rel_dict["type"] = record[3]
            relationships.append(rel_dict)

        return relationships

    def count_relationships(
        self,
        relationship_types: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> int:
        """Count relationships matching filters."""
        where_parts = []
        params = {}

        if source_id:
            where_parts.append("s.id = $source_id")
            params["source_id"] = source_id

        if target_id:
            where_parts.append("t.id = $target_id")
            params["target_id"] = target_id

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        if relationship_types:
            rel_pattern = "|".join(relationship_types)
            rel_match = f"-[r:{rel_pattern}]->"
        else:
            rel_match = "-[r]->"

        query = f"""
        MATCH (s){rel_match}(t)
        {where_clause}
        RETURN count(r) as count
        """

        result = self.graph.query(query, params)
        return result.result_set[0][0] if result.result_set else 0

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID."""
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n)[0] as type
        """
        result = self.graph.query(query, {"node_id": node_id})

        if result.result_set and len(result.result_set) > 0:
            node = result.result_set[0][0]
            node_type = result.result_set[0][1]
            node_dict = dict(node.properties)
            node_dict["type"] = node_type
            node_dict["id"] = node_dict.get("id", str(node.id))
            return node_dict

        return None


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_semantic_memory: Optional[FalkorDBSemanticMemory] = None


def get_semantic_memory() -> FalkorDBSemanticMemory:
    """
    Get or create semantic memory singleton.

    Returns:
        FalkorDBSemanticMemory: Singleton instance
    """
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = FalkorDBSemanticMemory()
    return _semantic_memory


def reset_semantic_memory() -> None:
    """Reset the semantic memory singleton (for testing)."""
    global _semantic_memory
    _semantic_memory = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


async def query_semantic_graph(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Query the semantic graph (used by investigator node).

    Args:
        query: Query configuration with:
            - start_nodes: List of entity IDs to start from
            - entity_type: Type of query ("patient", "hcp", etc.)
            - follow_causal: Whether to follow causal relationships
            - max_depth: Maximum traversal depth

    Returns:
        List of query results
    """
    semantic = get_semantic_memory()

    start_nodes = query.get("start_nodes", [])
    max_depth = query.get("max_depth", 2)

    results = []
    for node_id in start_nodes:
        if query.get("entity_type") == "patient":
            network = semantic.get_patient_network(node_id, max_depth)
            results.append({"type": "patient_network", "data": network})
        elif query.get("entity_type") == "hcp":
            network = semantic.get_hcp_influence_network(node_id, max_depth)
            results.append({"type": "hcp_network", "data": network})
        elif query.get("follow_causal"):
            chains = semantic.traverse_causal_chain(node_id, max_depth)
            results.extend([{"type": "causal_chain", "data": chain} for chain in chains])

    return results


async def sync_to_semantic_graph(triplet: Dict[str, Any]) -> bool:
    """
    Add a triplet (subject-predicate-object) to the semantic graph.

    Args:
        triplet: Dict with:
            - subject: Subject entity ID
            - subject_type: Subject entity type label
            - predicate: Relationship type
            - object: Object entity ID
            - object_type: Object entity type label
            - confidence: Optional confidence score

    Returns:
        True if successful
    """
    semantic = get_semantic_memory()

    subject_type = triplet.get("subject_type", "Entity")
    object_type = triplet.get("object_type", "Entity")

    # Map type labels to E2I entity types
    source_e2i_type = LABEL_TO_E2I.get(subject_type)
    target_e2i_type = LABEL_TO_E2I.get(object_type)

    if source_e2i_type and target_e2i_type:
        return semantic.add_e2i_relationship(
            source_type=source_e2i_type,
            source_id=triplet["subject"],
            target_type=target_e2i_type,
            target_id=triplet["object"],
            rel_type=triplet["predicate"],
            properties={"confidence": triplet.get("confidence", 0.8)},
        )

    # Fall back to generic entity handling
    semantic.add_e2i_entity(E2IEntityType.PATIENT, triplet["subject"])
    semantic.add_e2i_entity(E2IEntityType.PATIENT, triplet["object"])

    return True


async def sync_data_layer_to_semantic_cache() -> Dict[str, Any]:
    """
    Sync E2I data layer relationships to Supabase semantic cache.

    Calls the sync_hcp_patient_relationships_to_cache database function.

    Returns:
        Result from the sync operation
    """
    client = get_supabase_client()

    result = client.rpc("sync_hcp_patient_relationships_to_cache", {}).execute()

    return result.data
