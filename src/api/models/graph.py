"""
E2I Graph API Models
=====================

Pydantic models for knowledge graph API operations.

Endpoints:
- Graph node listing and retrieval
- Relationship queries
- Subgraph traversal
- Causal chain queries
- openCypher query execution
- Real-time graph updates

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# ENUMS
# =============================================================================


class EntityType(str, Enum):
    """Entity types in the E2I knowledge graph."""

    PATIENT = "Patient"
    HCP = "HCP"
    BRAND = "Brand"
    REGION = "Region"
    KPI = "KPI"
    CAUSAL_PATH = "CausalPath"
    TRIGGER = "Trigger"
    AGENT = "Agent"
    EPISODE = "Episode"
    COMMUNITY = "Community"
    # Additional E2I types from semantic memory
    TREATMENT = "Treatment"
    PREDICTION = "Prediction"
    EXPERIMENT = "Experiment"
    AGENT_ACTIVITY = "AgentActivity"
    # Domain-specific types from seed data
    HCP_SPECIALTY = "HCPSpecialty"
    JOURNEY_STAGE = "JourneyStage"


class RelationshipType(str, Enum):
    """Relationship types in the E2I knowledge graph."""

    TREATED_BY = "TREATED_BY"
    PRESCRIBED = "PRESCRIBED"
    PRESCRIBES = "PRESCRIBES"
    CAUSES = "CAUSES"
    IMPACTS = "IMPACTS"
    INFLUENCES = "INFLUENCES"
    DISCOVERED = "DISCOVERED"
    GENERATED = "GENERATED"
    MENTIONS = "MENTIONS"
    MEMBER_OF = "MEMBER_OF"
    RELATES_TO = "RELATES_TO"
    # Additional E2I relationship types
    RECEIVED = "RECEIVED"
    LOCATED_IN = "LOCATED_IN"
    PRACTICES_IN = "PRACTICES_IN"
    MEASURED_IN = "MEASURED_IN"
    LEADS_TO = "LEADS_TO"


class SortOrder(str, Enum):
    """Sort order for list queries."""

    ASC = "asc"
    DESC = "desc"


class NodeSortField(str, Enum):
    """Fields to sort nodes by."""

    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    NAME = "name"
    TYPE = "type"


# =============================================================================
# BASE MODELS
# =============================================================================


class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(..., description="Unique node identifier")
    type: EntityType = Field(..., description="Node entity type")
    name: str = Field(..., description="Node display name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "hcp_12345",
                "type": "HCP",
                "name": "Dr. Smith",
                "properties": {"specialty": "Oncology", "region": "Northeast", "tier": "A"},
                "created_at": "2025-01-15T10:30:00Z",
            }
        }
    )


class GraphRelationship(BaseModel):
    """A relationship (edge) in the knowledge graph."""

    id: str = Field(..., description="Unique relationship identifier")
    type: RelationshipType = Field(..., description="Relationship type")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "rel_001",
                "type": "PRESCRIBES",
                "source_id": "hcp_12345",
                "target_id": "brand_kisqali",
                "properties": {"frequency": "high", "preference_score": 0.85},
                "confidence": 0.92,
            }
        }
    )


class GraphPath(BaseModel):
    """A path through the graph (sequence of nodes and relationships)."""

    nodes: List[GraphNode] = Field(..., description="Nodes in the path")
    relationships: List[GraphRelationship] = Field(
        ..., description="Relationships connecting nodes"
    )
    total_confidence: Optional[float] = Field(None, description="Combined path confidence")
    path_length: int = Field(..., description="Number of hops in the path")


# =============================================================================
# REQUEST MODELS
# =============================================================================


class ListNodesRequest(BaseModel):
    """Request for listing graph nodes."""

    entity_types: Optional[List[EntityType]] = Field(None, description="Filter by entity types")
    search: Optional[str] = Field(
        None, max_length=500, description="Text search in node names/properties"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description='Property filters (e.g., {"region": "Northeast"})'
    )
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    sort_by: NodeSortField = Field(default=NodeSortField.CREATED_AT, description="Sort field")
    sort_order: SortOrder = Field(default=SortOrder.DESC, description="Sort order")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_types": ["HCP", "Brand"],
                "filters": {"region": "Northeast"},
                "limit": 20,
            }
        }
    )


class ListRelationshipsRequest(BaseModel):
    """Request for listing graph relationships."""

    relationship_types: Optional[List[RelationshipType]] = Field(
        None, description="Filter by relationship types"
    )
    source_id: Optional[str] = Field(None, description="Filter by source node")
    target_id: Optional[str] = Field(None, description="Filter by target node")
    min_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "relationship_types": ["CAUSES", "IMPACTS"],
                "min_confidence": 0.6,
                "limit": 30,
            }
        }
    )


class TraverseRequest(BaseModel):
    """Request for graph traversal from a starting node."""

    start_node_id: str = Field(..., description="Starting node ID")
    relationship_types: Optional[List[RelationshipType]] = Field(
        None, description="Relationship types to follow"
    )
    direction: str = Field(
        default="outgoing", description="Traversal direction: outgoing, incoming, both"
    )
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum traversal depth")
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum edge confidence"
    )
    include_properties: bool = Field(default=True, description="Include node/edge properties")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_node_id": "kpi_trx",
                "relationship_types": ["CAUSES", "IMPACTS"],
                "direction": "incoming",
                "max_depth": 3,
                "min_confidence": 0.5,
            }
        }
    )


class CausalChainRequest(BaseModel):
    """Request for causal chain queries."""

    kpi_name: Optional[str] = Field(None, description="Target KPI name")
    source_entity_id: Optional[str] = Field(None, description="Source entity for chain")
    target_entity_id: Optional[str] = Field(None, description="Target entity for chain")
    min_effect_size: Optional[float] = Field(None, description="Minimum causal effect size")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence")
    max_chain_length: int = Field(default=4, ge=1, le=10, description="Maximum chain length")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"kpi_name": "TRx", "min_confidence": 0.6, "max_chain_length": 3}
        }
    )


class CypherQueryRequest(BaseModel):
    """Request for executing openCypher queries."""

    query: str = Field(..., min_length=1, max_length=5000, description="openCypher query string")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    timeout_seconds: int = Field(default=30, ge=1, le=120, description="Query timeout")
    read_only: bool = Field(default=True, description="Enforce read-only query")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "MATCH (h:HCP)-[:PRESCRIBES]->(b:Brand) WHERE b.name = $brand RETURN h LIMIT 10",
                "parameters": {"brand": "Kisqali"},
                "read_only": True,
            }
        }
    )


class AddEpisodeRequest(BaseModel):
    """Request for adding a knowledge episode."""

    content: str = Field(..., min_length=1, max_length=10000, description="Episode content text")
    source: str = Field(..., description="Source of the episode (agent name or 'user')")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    extract_entities: bool = Field(
        default=True, description="Automatically extract entities/relationships"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Dr. Smith prescribed Kisqali for the patient with HR+/HER2- breast cancer, resulting in improved TRx in the Northeast region.",
                "source": "orchestrator",
                "session_id": "sess_abc123",
                "metadata": {"query_type": "prescription_analysis"},
            }
        }
    )


class SearchGraphRequest(BaseModel):
    """Request for natural language graph search."""

    query: str = Field(
        ..., min_length=1, max_length=1000, description="Natural language search query"
    )
    entity_types: Optional[List[EntityType]] = Field(
        None, description="Filter results by entity types"
    )
    session_id: Optional[str] = Field(None, description="Session ID for context")
    k: int = Field(default=10, ge=1, le=50, description="Number of results")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What factors impact TRx in the Northeast?",
                "entity_types": ["HCP", "Brand", "KPI"],
                "k": 10,
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class ListNodesResponse(BaseModel):
    """Response for listing graph nodes."""

    nodes: List[GraphNode] = Field(..., description="List of nodes")
    total_count: int = Field(..., description="Total matching nodes")
    limit: int = Field(..., description="Applied limit")
    offset: int = Field(..., description="Applied offset")
    has_more: bool = Field(..., description="Whether more results exist")
    query_latency_ms: float = Field(..., description="Query latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ListRelationshipsResponse(BaseModel):
    """Response for listing graph relationships."""

    relationships: List[GraphRelationship] = Field(..., description="List of relationships")
    total_count: int = Field(..., description="Total matching relationships")
    limit: int = Field(..., description="Applied limit")
    offset: int = Field(..., description="Applied offset")
    has_more: bool = Field(..., description="Whether more results exist")
    query_latency_ms: float = Field(..., description="Query latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TraverseResponse(BaseModel):
    """Response for graph traversal."""

    subgraph: Dict[str, Any] = Field(..., description="Subgraph with nodes and edges")
    nodes: List[GraphNode] = Field(..., description="All traversed nodes")
    relationships: List[GraphRelationship] = Field(..., description="All traversed relationships")
    paths: List[GraphPath] = Field(default_factory=list, description="Discovered paths")
    max_depth_reached: int = Field(..., description="Actual depth reached")
    query_latency_ms: float = Field(..., description="Query latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CausalChainResponse(BaseModel):
    """Response for causal chain queries."""

    chains: List[GraphPath] = Field(..., description="Discovered causal chains")
    total_chains: int = Field(..., description="Number of chains found")
    strongest_chain: Optional[GraphPath] = Field(None, description="Highest confidence chain")
    aggregate_effect: Optional[float] = Field(None, description="Aggregate causal effect")
    query_latency_ms: float = Field(..., description="Query latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chains": [],
                "total_chains": 3,
                "aggregate_effect": 0.72,
                "query_latency_ms": 45.2,
            }
        }
    )


class CypherQueryResponse(BaseModel):
    """Response for openCypher query execution."""

    results: List[Dict[str, Any]] = Field(..., description="Query results")
    columns: List[str] = Field(..., description="Result column names")
    row_count: int = Field(..., description="Number of result rows")
    query_latency_ms: float = Field(..., description="Query latency in milliseconds")
    read_only: bool = Field(..., description="Whether query was read-only")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AddEpisodeResponse(BaseModel):
    """Response for adding a knowledge episode."""

    episode_id: str = Field(..., description="Created episode ID")
    extracted_entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Entities extracted from content"
    )
    extracted_relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Relationships extracted from content"
    )
    content_summary: Optional[str] = Field(None, description="Brief content summary")
    processing_latency_ms: float = Field(..., description="Processing latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SearchGraphResponse(BaseModel):
    """Response for natural language graph search."""

    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Number of results")
    query: str = Field(..., description="Original query")
    query_latency_ms: float = Field(..., description="Query latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphStatsResponse(BaseModel):
    """Response for graph statistics."""

    total_nodes: int = Field(..., description="Total node count")
    total_relationships: int = Field(..., description="Total relationship count")
    nodes_by_type: Dict[str, int] = Field(..., description="Node counts by entity type")
    relationships_by_type: Dict[str, int] = Field(..., description="Relationship counts by type")
    total_episodes: int = Field(default=0, description="Total episodes in graph")
    total_communities: int = Field(default=0, description="Total communities")
    last_updated: Optional[datetime] = Field(None, description="Last graph update time")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NodeNetworkResponse(BaseModel):
    """Response for node network queries."""

    node_id: str = Field(..., description="Central node ID")
    node_type: EntityType = Field(..., description="Central node type")
    connected_nodes: Dict[str, List[Dict[str, Any]]] = Field(
        ..., description="Connected nodes grouped by type"
    )
    total_connections: int = Field(..., description="Total connected nodes")
    max_depth: int = Field(..., description="Traversal depth used")
    query_latency_ms: float = Field(..., description="Query latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node_id": "patient_001",
                "node_type": "Patient",
                "connected_nodes": {
                    "hcps": [{"id": "hcp_001", "properties": {"name": "Dr. Smith"}}],
                    "treatments": [{"id": "tx_001", "properties": {"name": "Kisqali"}}],
                    "triggers": [],
                },
                "total_connections": 2,
                "max_depth": 2,
                "query_latency_ms": 15.3,
            }
        }
    )


# =============================================================================
# WEBSOCKET MODELS
# =============================================================================


class GraphStreamMessage(BaseModel):
    """Message format for WebSocket graph streaming."""

    event_type: str = Field(..., description="Event type: node_added, edge_added, update, etc.")
    payload: Dict[str, Any] = Field(..., description="Event payload data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = Field(None, description="Associated session ID")


class GraphSubscription(BaseModel):
    """Subscription configuration for graph streaming."""

    entity_types: Optional[List[EntityType]] = Field(
        None, description="Entity types to subscribe to"
    )
    relationship_types: Optional[List[RelationshipType]] = Field(
        None, description="Relationship types to subscribe to"
    )
    session_ids: Optional[List[str]] = Field(None, description="Session IDs to filter events")
    include_properties: bool = Field(default=True, description="Include full properties")
