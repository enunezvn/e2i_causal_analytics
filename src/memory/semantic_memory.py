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
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from src.memory.episodic_memory import E2IEntityType
from src.memory.services.config import get_config
from src.memory.services.factories import get_falkordb_client

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
# CYPHER IDENTIFIER WHITELISTS (injection hardening)
# ============================================================================
#
# Node labels and relationship types CANNOT be parameterized in openCypher —
# they are part of the query structure, not values — so any user-supplied
# label/type that reaches the WHERE clause or relationship pattern must be
# validated against a known-good allowlist BEFORE it touches the query string.
# These sets mirror the EntityType / RelationshipType enums in
# ``src/api/models/graph.py`` (kept here to avoid an api->memory import cycle)
# plus the seed-data labels actually present in the FalkorDB graph.

KNOWN_NODE_LABELS: frozenset[str] = frozenset(
    {
        # E2I_TO_LABEL values
        "Patient",
        "HCP",
        "Trigger",
        "CausalPath",
        "Prediction",
        "Treatment",
        "Experiment",
        "AgentActivity",
        # Additional graph / seed-data labels (mirror api EntityType enum)
        "Brand",
        "Region",
        "KPI",
        "Agent",
        "Episode",
        "Community",
        "HCPSpecialty",
        "JourneyStage",
    }
)

KNOWN_RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    {
        "TREATED_BY",
        "PRESCRIBED",
        "PRESCRIBES",
        "CAUSES",
        "IMPACTS",
        "INFLUENCES",
        "DISCOVERED",
        "GENERATED",
        "MENTIONS",
        "MEMBER_OF",
        "RELATES_TO",
        "RECEIVED",
        "RECEIVES",
        "LOCATED_IN",
        "PRACTICES_IN",
        "MEASURED_IN",
        "LEADS_TO",
        "TRACKS",
        "AFFECTS",
        "EXPLAINS",
        "ANALYZES",
        "PREDICTS",
        "MONITORS",
        "USES",
        # Relationship types used in graph traversal but not in the api enum
        "SHARED_PATIENTS",
    }
)


def _validate_node_labels(entity_types: List[str]) -> List[str]:
    """
    Validate user-supplied node labels against the known-label allowlist.

    Node labels are structural in Cypher and cannot be parameterized, so the
    only safe options are (a) reject unknown values or (b) interpolate. We
    choose (a): any label not in :data:`KNOWN_NODE_LABELS` raises ``ValueError``
    so a poisoned value (e.g. ``Patient' OR '1'='1``) can never be interpolated
    into the query.

    Args:
        entity_types: Raw, user-supplied label strings.

    Returns:
        The validated labels (unchanged, but guaranteed safe to interpolate).

    Raises:
        ValueError: If any label is not a recognized node label.
    """
    invalid = [t for t in entity_types if t not in KNOWN_NODE_LABELS]
    if invalid:
        raise ValueError(
            f"Unknown node label(s): {invalid}. Allowed labels: {sorted(KNOWN_NODE_LABELS)}"
        )
    return entity_types


def _validate_relationship_types(relationship_types: List[str]) -> List[str]:
    """
    Validate user-supplied relationship types against the allowlist.

    Relationship types are structural in Cypher and cannot be parameterized.
    Any type not in :data:`KNOWN_RELATIONSHIP_TYPES` raises ``ValueError`` so a
    poisoned value can never reach the ``-[r:...]->`` pattern.

    Args:
        relationship_types: Raw, user-supplied relationship-type strings.

    Returns:
        The validated relationship types (guaranteed safe to interpolate).

    Raises:
        ValueError: If any type is not a recognized relationship type.
    """
    invalid = [t for t in relationship_types if t not in KNOWN_RELATIONSHIP_TYPES]
    if invalid:
        raise ValueError(
            f"Unknown relationship type(s): {invalid}. "
            f"Allowed types: {sorted(KNOWN_RELATIONSHIP_TYPES)}"
        )
    return relationship_types


# A Cypher identifier: a leading letter/underscore then letters/digits/
# underscores. Node labels, relationship types and property KEYS are structural
# tokens that cannot be parameterized, so they are spliced into the query string;
# constraining them to this shape makes injection (``]``, ``(``, ``{``, quotes,
# whitespace, comment markers) impossible.
_SAFE_CYPHER_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_cypher_identifier(value: str, kind: str) -> str:
    """Validate a structural Cypher token (label / relationship type) on WRITE.

    Reason-before-rules: the WRITE methods accept OPEN-ENDED labels/relationship
    types from agents and the LLM cognitive path (e.g. ``"SegmentEffect"``,
    ``"Variable"``) that are deliberately NOT in the read-side closed allowlists
    (:func:`_validate_node_labels` / :func:`_validate_relationship_types`). An
    allowlist here would silently drop those legitimate writes while adding no
    extra injection safety, so writes use a strict-identifier regex instead —
    it blocks every injection payload yet preserves dynamic-but-safe tokens.

    Raises:
        ValueError: if ``value`` is not a valid Cypher identifier.
    """
    if not isinstance(value, str) or not _SAFE_CYPHER_IDENT.match(value):
        raise ValueError(
            f"Invalid {kind} {value!r}: must be a Cypher identifier "
            f"(letters/digits/underscore, not starting with a digit)."
        )
    return value


def _validate_property_keys(properties: Optional[Dict[str, Any]]) -> None:
    """Validate that every property KEY is a safe Cypher identifier on WRITE.

    Property *values* are passed as bound parameters (``$key``) and are safe;
    the *keys*, however, are interpolated into the ``SET`` clause, so a poisoned
    key (e.g. ``"x} ) DETACH DELETE n //"``) would be an injection vector.

    Raises:
        ValueError: if any key is not a valid Cypher identifier.
    """
    if not properties:
        return
    for key in properties:
        _validate_cypher_identifier(str(key), "property key")


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
        entity_type: Union[E2IEntityType, str],
        entity_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add an E2I entity to the semantic graph.

        Uses MERGE to create or update the entity.

        Args:
            entity_type: E2I entity type (enum or string label)
            entity_id: Unique entity identifier
            properties: Additional properties to store

        Returns:
            True if successful
        """
        if isinstance(entity_type, E2IEntityType):
            label = E2I_TO_LABEL.get(entity_type, "Entity")
            type_value = entity_type.value
        else:
            label = entity_type
            type_value = entity_type

        # H2: validate structural tokens before they are spliced into Cypher.
        # ``label`` may be a caller-supplied string on the str branch; property
        # keys are interpolated into the SET clause. Values are bound params.
        _validate_cypher_identifier(label, "node label")
        _validate_property_keys(properties)

        props = properties.copy() if properties else {}
        # The two keys added below are hardcoded identifier literals (not
        # caller-controlled), so they need no further validation before being
        # interpolated into the SET clause alongside the validated caller keys.
        props["e2i_entity_type"] = type_value
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
        deleted: bool = result.nodes_deleted > 0

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
        # H2: validate structural tokens BEFORE any graph write (including the
        # entity-ensure calls below) so a poisoned rel_type or property key can
        # never reach graph.query. ``rel_type`` is the live LLM-fed vector
        # (rag/cognitive_backends passes ``relationship_type.upper()``).
        _validate_cypher_identifier(rel_type, "relationship type")
        _validate_property_keys(properties)

        # Ensure both entities exist
        self.add_e2i_entity(source_type, source_id)
        self.add_e2i_entity(target_type, target_id)

        source_label = E2I_TO_LABEL.get(source_type, "Entity")
        target_label = E2I_TO_LABEL.get(target_type, "Entity")

        props = properties.copy() if properties else {}
        # ``updated_at`` is a hardcoded identifier literal (not caller-controlled),
        # so it needs no validation before interpolation alongside the validated
        # caller keys; validation of the caller's keys ran above, before any write.
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

    def get_patient_network(
        self,
        patient_id: str,
        max_depth: int = 2,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get the relationship network around a patient with pagination.

        Args:
            patient_id: Patient entity ID
            max_depth: Maximum traversal depth (1-5, clamped for safety)
            limit: Maximum nodes to return (default 100, max 500)
            offset: Pagination offset for results

        Returns:
            Dict with patient_id, hcps, treatments, triggers, causal_paths, pagination
        """
        # Sanitize inputs to prevent injection and limit resource usage
        safe_depth = max(1, min(5, int(max_depth)))
        safe_limit = max(1, min(500, int(limit)))
        safe_offset = max(0, int(offset))

        # FalkorDB doesn't support parameterized variable-length bounds,
        # so we use string formatting with the sanitized value
        query = f"""
        MATCH (p:Patient {{id: $patient_id}})-[*1..{safe_depth}]-(connected)
        RETURN DISTINCT connected
        SKIP {safe_offset}
        LIMIT {safe_limit}
        """

        result = self.graph.query(query, {"patient_id": patient_id})

        network: Dict[str, Any] = {
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

        # Add pagination metadata
        result_count = len(result.result_set)
        network["pagination"] = {
            "offset": safe_offset,
            "limit": safe_limit,
            "returned": result_count,
            "has_more": result_count == safe_limit,
        }

        return network

    def get_hcp_influence_network(
        self,
        hcp_id: str,
        max_depth: int = 2,
        limit: int = 100,
        offset: int = 0,
        cohort_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get the influence network around an HCP with pagination.

        Args:
            hcp_id: HCP entity ID
            max_depth: Maximum traversal depth (1-5, clamped for safety)
            limit: Maximum nodes to return (default 100, max 500)
            offset: Pagination offset for results
            cohort_id: optional cohort tag (e.g., ``"optum_initiation_v3"``).
                When supplied, restricts traversal to nodes AND relationships
                carrying the same ``cohort_id`` property as the start HCP —
                required to keep CSU and Optum influence graphs queryable
                independently after issue #169 ingestion populates the
                backend. **Strongly recommended whenever influence data
                has been ingested under any cohort tag**: an unscoped
                call (``cohort_id=None``) will MATCH every ``HCP`` node
                that shares the requested ``id`` — and the issue #169
                schema deliberately creates one node per ``(id, cohort_id)``
                pair, so a same-NPI provider that appears in both CSU
                and Optum cohorts would have its neighborhoods merged
                in the result.

        Returns:
            Dict with hcp_id, influenced_hcps, patients, brands_prescribed, pagination
        """
        # Sanitize inputs to prevent injection and limit resource usage
        safe_depth = max(1, min(5, int(max_depth)))
        safe_limit = max(1, min(500, int(limit)))
        safe_offset = max(0, int(offset))

        # FalkorDB doesn't support parameterized variable-length bounds.
        # Issue #169 codex pass-1 MEDIUM-1: cohort traversal must constrain
        # EVERY relationship in the path (not just the terminal node) so
        # cross-cohort or non-SHARED_PATIENTS edges can't smuggle a node
        # whose `cohort_id` happens to match. Bind the path and enforce
        # the predicate over `relationships(path)` plus the edge's own
        # `cohort_id` property — both are populated by the issue #169
        # persistence script on every SHARED_PATIENTS edge.
        if cohort_id is not None:
            query = f"""
            MATCH path = (h:HCP {{id: $hcp_id, cohort_id: $cohort_id}})
                -[:SHARED_PATIENTS*1..{safe_depth}]-(connected)
            WHERE connected.cohort_id = $cohort_id
              AND all(r IN relationships(path) WHERE r.cohort_id = $cohort_id)
            RETURN DISTINCT connected
            SKIP {safe_offset}
            LIMIT {safe_limit}
            """
            params: Dict[str, Any] = {"hcp_id": hcp_id, "cohort_id": cohort_id}
        else:
            query = f"""
            MATCH (h:HCP {{id: $hcp_id}})-[*1..{safe_depth}]-(connected)
            RETURN DISTINCT connected
            SKIP {safe_offset}
            LIMIT {safe_limit}
            """
            params = {"hcp_id": hcp_id}

        result = self.graph.query(query, params)

        network: Dict[str, Any] = {
            "hcp_id": hcp_id,
            "influenced_hcps": [],
            "patients": [],
            "brands_prescribed": [],
        }

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

        # Add pagination metadata
        result_count = len(result.result_set)
        network["pagination"] = {
            "offset": safe_offset,
            "limit": safe_limit,
            "returned": result_count,
            "has_more": result_count == safe_limit,
        }

        return network

    # ========================================================================
    # CAUSAL CHAIN ANALYSIS
    # ========================================================================

    def traverse_causal_chain(
        self,
        start_entity_id: str,
        max_depth: int = 3,
        limit: int = 50,
        brands: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Traverse causal relationships from a starting entity with result limit.

        Follows CAUSES and IMPACTS relationships.

        Args:
            start_entity_id: Starting entity ID
            max_depth: Maximum chain length (1-5, clamped for safety)
            limit: Maximum number of chains to return (default 50, max 200)
            brands: Optional brand-access scope (H1 / #694). When provided, only
                paths whose EVERY relationship (the causal finding) carries one
                of these brands are returned; unbranded findings are excluded
                (fail-closed for a scoped caller). Pass ``None`` (admin) to
                disable filtering and traverse the whole graph.

        Returns:
            List of causal chains with nodes, relationships, and metadata
        """
        # Sanitize inputs to prevent injection and limit resource usage
        safe_depth = max(1, min(5, int(max_depth)))
        safe_limit = max(1, min(200, int(limit)))

        params: Dict[str, Any] = {"start_id": start_entity_id}
        # H1 (#694): brand-scope on the FINDING (the CAUSES/IMPACTS relationship),
        # not the shared Variable nodes (e.g. var:TRx spans brands). r.brand IS
        # NULL (unbranded) fails `IN $brands`, so scoped callers never see
        # unbranded or cross-brand findings.
        brand_filter = ""
        if brands is not None:
            brand_filter = "WHERE ALL(r IN relationships(path) WHERE r.brand IN $brands)"
            params["brands"] = list(brands)

        # FalkorDB doesn't support parameterized variable-length bounds
        query = f"""
        MATCH path = (s {{id: $start_id}})-[:CAUSES|IMPACTS*1..{safe_depth}]->(t)
        {brand_filter}
        RETURN
            [n IN nodes(path) | {{id: n.id, type: labels(n)[0]}}] as nodes,
            [r IN relationships(path) | {{type: type(r), conf: r.confidence}}] as rels
        LIMIT {safe_limit}
        """

        result = self.graph.query(query, params)

        chains = []
        for record in result.result_set:
            chains.append(
                {"nodes": record[0], "relationships": record[1], "path_length": len(record[1])}
            )

        return chains

    # ========================================================================
    # NETWORK COUNT METHODS (for pagination)
    # ========================================================================

    def count_patient_network(self, patient_id: str, max_depth: int = 2) -> int:
        """
        Count total nodes in patient network.

        Used for pagination to determine total pages.

        Args:
            patient_id: Patient entity ID
            max_depth: Maximum traversal depth (1-5)

        Returns:
            Total count of connected nodes
        """
        safe_depth = max(1, min(5, int(max_depth)))

        query = f"""
        MATCH (p:Patient {{id: $patient_id}})-[*1..{safe_depth}]-(connected)
        RETURN count(DISTINCT connected) as total
        """

        result = self.graph.query(query, {"patient_id": patient_id})
        return result.result_set[0][0] if result.result_set else 0

    def count_hcp_influence_network(
        self,
        hcp_id: str,
        max_depth: int = 2,
        cohort_id: Optional[str] = None,
    ) -> int:
        """
        Count total nodes in HCP influence network.

        Used for pagination to determine total pages.

        Args:
            hcp_id: HCP entity ID
            max_depth: Maximum traversal depth (1-5)
            cohort_id: optional cohort tag — see :meth:`get_hcp_influence_network`
                for the rationale (issue #169 cross-cohort isolation).

        Returns:
            Total count of connected nodes
        """
        safe_depth = max(1, min(5, int(max_depth)))

        if cohort_id is not None:
            # Issue #169 codex pass-1 MEDIUM-1: constrain every relationship
            # in the path to the same cohort + SHARED_PATIENTS type so the
            # count cannot leak across non-influence edges.
            query = f"""
            MATCH path = (h:HCP {{id: $hcp_id, cohort_id: $cohort_id}})
                -[:SHARED_PATIENTS*1..{safe_depth}]-(connected)
            WHERE connected.cohort_id = $cohort_id
              AND all(r IN relationships(path) WHERE r.cohort_id = $cohort_id)
            RETURN count(DISTINCT connected) as total
            """
            params: Dict[str, Any] = {"hcp_id": hcp_id, "cohort_id": cohort_id}
        else:
            query = f"""
            MATCH (h:HCP {{id: $hcp_id}})-[*1..{safe_depth}]-(connected)
            RETURN count(DISTINCT connected) as total
            """
            params = {"hcp_id": hcp_id}

        result = self.graph.query(query, params)
        return result.result_set[0][0] if result.result_set else 0

    def count_hcp_influence_degree(
        self,
        hcp_id: str,
        cohort_id: str,
    ) -> int:
        """
        Issue #169 codex pass-1 MEDIUM-2: 1-hop count for Parquet parity.

        The Parquet artifact's ``influence_network_size`` is exactly
        ``graph.degree(npi)`` — the count of distinct neighbors via the
        ``SHARED_PATIENTS`` relationship within the same cohort. The
        depth-N traversal in :meth:`count_hcp_influence_network`
        intentionally counts transitive neighborhoods, so it will NOT
        match ``influence_network_size`` for ``max_depth > 1`` on any
        non-trivial graph. This helper is the dedicated entry point
        for "give me the same number the converter wrote to Parquet".

        Args:
            hcp_id: HCP entity id (the obfuscated/real NPI string).
            cohort_id: cohort tag the FalkorDB rows are keyed under.
                Required — there is no "global degree" view across cohorts.

        Returns:
            Distinct 1-hop SHARED_PATIENTS neighbor count.
        """
        query = """
        MATCH (h:HCP {id: $hcp_id, cohort_id: $cohort_id})
            -[:SHARED_PATIENTS]-(neighbor:HCP {cohort_id: $cohort_id})
        RETURN count(DISTINCT neighbor) as total
        """
        result = self.graph.query(query, {"hcp_id": hcp_id, "cohort_id": cohort_id})
        # Codex pass-2 LOW: defensive null-coerce in case FalkorDB ever
        # returns `[[None]]` for the count row. Standard count() returns
        # 0 on no matches, so this is belt-and-suspenders.
        if not result.result_set:
            return 0
        value = result.result_set[0][0]
        return int(value) if value is not None else 0

    def count_causal_chains(self, start_entity_id: str, max_depth: int = 3) -> int:
        """
        Count total causal chains from a starting entity.

        Used to understand the size of the causal graph before traversal.

        Args:
            start_entity_id: Starting entity ID
            max_depth: Maximum chain length (1-5)

        Returns:
            Total count of causal chains
        """
        safe_depth = max(1, min(5, int(max_depth)))

        query = f"""
        MATCH path = (s {{id: $start_id}})-[:CAUSES|IMPACTS*1..{safe_depth}]->(t)
        RETURN count(path) as total
        """

        result = self.graph.query(query, {"start_id": start_entity_id})
        return result.result_set[0][0] if result.result_set else 0

    def find_causal_paths_for_kpi(
        self,
        kpi_name: str,
        min_confidence: float = 0.5,
        brands: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find all causal paths that impact a specific KPI.

        Useful for understanding what drives KPI changes.

        Args:
            kpi_name: Name of the KPI (e.g., "TRx", "NRx")
            min_confidence: Minimum confidence threshold
            brands: Optional brand-access scope (H1 / #694). When provided, only
                CausalPath nodes carrying one of these brands are returned;
                unbranded paths are excluded (fail-closed for a scoped caller).
                Pass ``None`` (admin) to disable filtering.

        Returns:
            List of causal paths with effect sizes and confidence
        """
        params: Dict[str, Any] = {"kpi_name": kpi_name, "min_confidence": min_confidence}
        # H1 (#694): brand-scope on the CausalPath node (cp.brand IS NULL fails
        # `IN $brands`, so scoped callers never see unbranded/cross-brand paths).
        brand_filter = ""
        if brands is not None:
            brand_filter = "AND cp.brand IN $brands"
            params["brands"] = list(brands)

        query = f"""
        MATCH (cp:CausalPath)-[r:IMPACTS]->(k:KPI {{name: $kpi_name}})
        WHERE r.confidence >= $min_confidence
        {brand_filter}
        RETURN cp.id as path_id, cp.effect_size as effect_size,
               r.confidence as confidence, cp.method_used as method
        ORDER BY r.confidence DESC
        """

        result = self.graph.query(query, params)

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
        params: Dict[str, Any] = {}

        if entity_types:
            # Validate against the known-label allowlist; rejects injection
            # attempts (e.g. ``Patient' OR '1'='1``) before any interpolation.
            _validate_node_labels(entity_types)
            # Pass labels as a parameter and filter with a membership predicate
            # — defense in depth so the validated values are never spliced as
            # literals into the query string.
            where_parts.append("any(lbl IN labels(n) WHERE lbl IN $entity_types)")
            params["entity_types"] = entity_types

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
            # Use 'id' property if exists, otherwise use 'name' as identifier
            node_dict["id"] = node_dict.get("id") or node_dict.get("name") or str(node.id)
            nodes.append(node_dict)

        return nodes

    def count_nodes(
        self, entity_types: Optional[List[str]] = None, search: Optional[str] = None
    ) -> int:
        """Count nodes matching filters."""
        where_parts = []
        params: Dict[str, Any] = {}

        if entity_types:
            # Same allowlist validation + parameterized membership as list_nodes.
            _validate_node_labels(entity_types)
            where_parts.append("any(lbl IN labels(n) WHERE lbl IN $entity_types)")
            params["entity_types"] = entity_types

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
            params["min_confidence"] = min_confidence  # type: ignore[assignment]

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        # Build relationship type pattern. Relationship types are structural in
        # Cypher and cannot be parameterized, so we validate each against the
        # allowlist first — rejecting injection attempts before they can reach
        # the ``-[r:...]->`` pattern. Validated values are exact allowlist
        # members (no quotes/spaces/brackets), so splicing them is safe.
        if relationship_types:
            _validate_relationship_types(relationship_types)
            rel_pattern = "|".join(relationship_types)
            rel_match = f"-[r:{rel_pattern}]->"
        else:
            rel_match = "-[r]->"

        query = f"""
        MATCH (s){rel_match}(t)
        {where_clause}
        RETURN r, coalesce(s.name, s.id, toString(id(s))) as source_name, coalesce(t.name, t.id, toString(id(t))) as target_name, type(r) as rel_type
        SKIP {offset}
        LIMIT {limit}
        """

        result = self.graph.query(query, params)

        relationships = []
        for record in result.result_set:
            rel = record[0]
            rel_dict = dict(rel.properties) if rel.properties else {}
            rel_dict["id"] = str(rel.id)
            # Use node's name property as the identifier
            rel_dict["source_id"] = (
                str(record[1]) if record[1] is not None else str(rel.id) + "_src"
            )
            rel_dict["target_id"] = (
                str(record[2]) if record[2] is not None else str(rel.id) + "_tgt"
            )
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
            # Allowlist-validate before splicing the structural rel-type pattern.
            _validate_relationship_types(relationship_types)
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
        """Get a single node by ID or name."""
        # Try matching by 'id' property first, then by 'name'
        query = """
        MATCH (n)
        WHERE n.id = $node_id OR n.name = $node_id
        RETURN n, labels(n)[0] as type
        LIMIT 1
        """
        result = self.graph.query(query, {"node_id": node_id})

        if result.result_set and len(result.result_set) > 0:
            node = result.result_set[0][0]
            node_type = result.result_set[0][1]
            node_dict = dict(node.properties)
            node_dict["type"] = node_type
            # Use 'id' property if exists, otherwise use 'name' as identifier
            node_dict["id"] = node_dict.get("id") or node_dict.get("name") or str(node.id)
            return node_dict

        return None

    def _node_to_dict(self, node: Any) -> Dict[str, Any]:
        """Normalize a FalkorDB node object into a route-friendly dict."""
        props = dict(node.properties) if getattr(node, "properties", None) else {}
        labels = node.labels if hasattr(node, "labels") else []
        node_type = labels[0] if labels else props.get("type")
        node_dict = dict(props)
        if node_type:
            node_dict["type"] = node_type
        node_dict["id"] = props.get("id") or props.get("name") or str(getattr(node, "id", ""))
        return node_dict

    def traverse_from_node(
        self,
        start_node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "outgoing",
        max_depth: int = 2,
        min_confidence: Optional[float] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """
        Generic variable-depth traversal from a start node.

        Returns the connected subgraph as ``{"nodes": [...], "relationships":
        [...], "max_depth_reached": int}`` — the shape the ``/graph/traverse``
        and ``/graph/nodes/{id}/network`` (generic branch) endpoints expect.

        Args:
            start_node_id: ID (or name) of the node to traverse from.
            relationship_types: Optional allowlist-validated rel-type filter.
            direction: ``outgoing``, ``incoming`` or ``both`` (default outgoing).
            max_depth: Traversal depth, clamped to 1..5 for safety.
            min_confidence: Optional minimum edge ``confidence`` filter.
            limit: Maximum number of paths to materialize (clamped to 1..500).

        Returns:
            Dict with deduplicated ``nodes`` and ``relationships``.

        Raises:
            ValueError: If any relationship type is not in the allowlist.
        """
        safe_depth = max(1, min(5, int(max_depth)))
        safe_limit = max(1, min(500, int(limit)))

        # Relationship types are structural in Cypher; validate before splice.
        rel_filter = ""
        if relationship_types:
            _validate_relationship_types(relationship_types)
            rel_filter = ":" + "|".join(relationship_types)

        # Direction determines arrow orientation; both => undirected match.
        if direction == "incoming":
            pattern = (
                f"path = (start {{id: $start_id}})<-[{rel_filter}*1..{safe_depth}]-(connected)"
            )
        elif direction == "both":
            pattern = f"path = (start {{id: $start_id}})-[{rel_filter}*1..{safe_depth}]-(connected)"
        else:  # outgoing (default)
            pattern = (
                f"path = (start {{id: $start_id}})-[{rel_filter}*1..{safe_depth}]->(connected)"
            )

        params: Dict[str, Any] = {"start_id": start_node_id}
        where = ""
        if min_confidence is not None:
            where = "WHERE all(r IN relationships(path) WHERE r.confidence >= $min_confidence)"
            params["min_confidence"] = min_confidence

        query = f"""
        MATCH {pattern}
        {where}
        RETURN nodes(path) as path_nodes, relationships(path) as path_rels
        LIMIT {safe_limit}
        """

        result = self.graph.query(query, params)

        nodes_by_id: Dict[str, Dict[str, Any]] = {}
        rels_by_id: Dict[str, Dict[str, Any]] = {}

        for record in result.result_set:
            path_nodes = record[0] or []
            path_rels = record[1] or []

            for raw_node in path_nodes:
                node_dict = self._node_to_dict(raw_node)
                nodes_by_id[node_dict["id"]] = node_dict

            for raw_rel in path_rels:
                rel_props = dict(raw_rel.properties) if getattr(raw_rel, "properties", None) else {}
                rel_id = str(getattr(raw_rel, "id", len(rels_by_id)))
                rel_dict: Dict[str, Any] = dict(rel_props)
                rel_dict["id"] = rel_id
                rel_type = getattr(raw_rel, "relation", None) or rel_props.get("type")
                if rel_type:
                    rel_dict["type"] = rel_type
                # Populate endpoint ids from the FalkorDB edge nodes when present
                # so the route's relationship converter can wire source/target.
                src_node = getattr(raw_rel, "src_node", None)
                dest_node = getattr(raw_rel, "dest_node", None)
                if src_node is not None:
                    src_dict = self._node_to_dict(src_node)
                    rel_dict.setdefault("source_id", src_dict["id"])
                if dest_node is not None:
                    dest_dict = self._node_to_dict(dest_node)
                    rel_dict.setdefault("target_id", dest_dict["id"])
                rels_by_id[rel_id] = rel_dict

        return {
            "nodes": list(nodes_by_id.values()),
            "relationships": list(rels_by_id.values()),
            "max_depth_reached": safe_depth,
        }

    def find_causal_chains(
        self,
        kpi_name: Optional[str] = None,
        source_entity_id: Optional[str] = None,
        target_entity_id: Optional[str] = None,
        max_length: int = 4,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Find causal chains (CAUSES/IMPACTS paths) in the graph.

        The ``/graph/causal-chains`` endpoint can scope a query by a target
        KPI, a source entity, a target entity, or any combination. Each chain
        is returned as ``{"nodes": [...], "relationships": [...],
        "confidence": float}``.

        Args:
            kpi_name: Optional KPI name the chain must terminate at.
            source_entity_id: Optional id the chain must start from.
            target_entity_id: Optional id the chain must terminate at.
            max_length: Max chain length, clamped to 1..10.
            min_confidence: Minimum mean edge confidence to keep a chain.

        Returns:
            List of causal-chain dicts.
        """
        safe_length = max(1, min(10, int(max_length)))

        start_match = "(s {id: $source_id})" if source_entity_id else "(s)"
        if kpi_name:
            end_match = "(t:KPI {name: $kpi_name})"
        elif target_entity_id:
            end_match = "(t {id: $target_id})"
        else:
            end_match = "(t)"

        query = f"""
        MATCH path = {start_match}-[:CAUSES|IMPACTS*1..{safe_length}]->{end_match}
        WHERE all(r IN relationships(path) WHERE coalesce(r.confidence, 1.0) >= $min_confidence)
        RETURN
            [n IN nodes(path) | {{id: n.id, type: labels(n)[0]}}] as nodes,
            [r IN relationships(path) | {{type: type(r), confidence: r.confidence}}] as rels,
            reduce(c = 1.0, r IN relationships(path) | c * coalesce(r.confidence, 1.0)) as conf
        LIMIT 200
        """

        params: Dict[str, Any] = {"min_confidence": min_confidence}
        if source_entity_id:
            params["source_id"] = source_entity_id
        if kpi_name:
            params["kpi_name"] = kpi_name
        elif target_entity_id:
            params["target_id"] = target_entity_id

        result = self.graph.query(query, params)

        chains = []
        for record in result.result_set:
            chains.append(
                {
                    "nodes": record[0],
                    "relationships": record[1],
                    "confidence": record[2] if len(record) > 2 else None,
                }
            )

        return chains

    def semantic_search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Property-based search across graph nodes (FalkorDB fallback for
        ``/graph/search`` when the Graphiti semantic service is unavailable).

        This is a CONTAINS match over node ``name``/``id`` — the search term is
        passed as a parameter (never interpolated). It is intentionally simple:
        the rich NL/vector search lives in the Graphiti service; this fallback
        keeps the endpoint functional rather than 500-ing.

        Args:
            query: Free-text search term (parameterized).
            entity_types: Optional allowlist-validated label filter.
            k: Maximum number of results, clamped to 1..50.

        Returns:
            List of result dicts (id, name, type, score, properties).

        Raises:
            ValueError: If any entity type is not in the allowlist.
        """
        safe_k = max(1, min(50, int(k)))

        where_parts = ["(n.name CONTAINS $search OR n.id CONTAINS $search)"]
        params: Dict[str, Any] = {"search": query}

        if entity_types:
            _validate_node_labels(entity_types)
            where_parts.append("any(lbl IN labels(n) WHERE lbl IN $entity_types)")
            params["entity_types"] = entity_types

        where_clause = " AND ".join(where_parts)

        cypher = f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN n, labels(n)[0] as type
        LIMIT {safe_k}
        """

        result = self.graph.query(cypher, params)

        results = []
        for record in result.result_set:
            node = record[0]
            node_type = record[1]
            props = dict(node.properties) if getattr(node, "properties", None) else {}
            node_id = props.get("id") or props.get("name") or str(getattr(node, "id", ""))
            results.append(
                {
                    "id": node_id,
                    "name": props.get("name") or node_id,
                    "type": node_type,
                    "score": 1.0,  # CONTAINS match is binary; no relevance score
                    "properties": props,
                    "relationships": [],
                }
            )

        return results


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
