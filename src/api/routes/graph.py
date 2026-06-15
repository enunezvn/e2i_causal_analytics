"""
E2I Knowledge Graph API
========================

FastAPI endpoints for knowledge graph operations:
- Node listing and retrieval
- Relationship queries
- Graph traversal
- Causal chain analysis
- openCypher query execution
- Real-time streaming updates

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from src.api.dependencies.auth import (
    WS_BEARER_SUBPROTOCOL,
    authenticate_websocket,
    get_current_user,
    get_user_brands,
    is_auth_enabled,
)
from src.api.models.graph import (
    AddEpisodeRequest,
    AddEpisodeResponse,
    CausalChainRequest,
    CausalChainResponse,
    CypherQueryRequest,
    CypherQueryResponse,
    # Enums
    EntityType,
    # Base models
    GraphNode,
    GraphPath,
    GraphRelationship,
    GraphStatsResponse,
    GraphStreamMessage,
    GraphSubscription,
    # Request models
    ListNodesResponse,
    ListRelationshipsResponse,
    NodeNetworkResponse,
    NodeSortField,
    RelationshipType,
    SearchGraphRequest,
    SearchGraphResponse,
    SortOrder,
    TraverseRequest,
    TraverseResponse,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/graph",
    tags=["Knowledge Graph"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# WebSocket connection manager for real-time updates
class ConnectionManager:
    """Manages WebSocket connections for graph streaming."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, GraphSubscription] = {}
        # Brand grants of the AUTHENTICATED recipient on each connection, cached
        # at connect() time so broadcast() can enforce publisher-grant authZ
        # without re-reading the JWT. ``None`` means "unknown/unauthenticated
        # recipient" (auth disabled at connect, or no user) -> treated as
        # fail-closed against any SCOPED message (see _brands_allow).
        self.connection_brands: Dict[str, Optional[List[str]]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        subprotocol: Optional[str] = None,
        user_brands: Optional[List[str]] = None,
    ):
        # Echo ONLY a subprotocol the client actually offered. Echoing one it
        # did not offer fails the RFC 6455 handshake; passing None accepts with
        # no negotiated subprotocol (the fail-open path).
        if subprotocol is not None:
            await websocket.accept(subprotocol=subprotocol)
        else:
            await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_brands[client_id] = user_brands
        logger.info(f"Graph WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        self.connection_brands.pop(client_id, None)
        logger.info(f"Graph WebSocket disconnected: {client_id}")

    def set_subscription(self, client_id: str, subscription: GraphSubscription):
        self.subscriptions[client_id] = subscription

    async def broadcast(
        self, message: GraphStreamMessage, visible_brands: Optional[List[str]] = None
    ):
        """Broadcast a message to subscribed connections, scoped by brand grant.

        Two INDEPENDENT filters compose (a connection must pass BOTH):

        1. The recipient's self-chosen subscription (``_matches_subscription``).
        2. Publisher-grant authorization (``_brands_allow``): ``visible_brands``
           is the brand grant of the user who PRODUCED this event. A recipient
           receives it only if their cached grants intersect it.

        ``visible_brands=None`` means the message is UNSCOPED (auth disabled at
        the producing route -> no user) and is delivered to everyone — the
        fail-OPEN *authentication* posture. A SCOPED message (``visible_brands``
        is a list, i.e. auth on) is fail-CLOSED: a recipient with no matching
        grant — including one whose own grants are unknown (``None``) — gets
        nothing. The authN fail-open posture must NOT leak into authZ.

        ``visible_brands`` is a parameter only; it is never serialized into the
        payload sent to the client (no model / frontend-schema change).
        """
        # Snapshot the connections: send_json awaits below can yield control and
        # let a concurrent disconnect() mutate active_connections, which would
        # raise "dictionary changed size during iteration" on the live iterator
        # (outside the per-send try/except). Iterating a copy is safe.
        for client_id, websocket in list(self.active_connections.items()):
            try:
                # Filter 1: subscription.
                sub = self.subscriptions.get(client_id)
                if sub and not self._matches_subscription(message, sub):
                    continue
                # Filter 2: publisher-grant authorization.
                if not self._brands_allow(self.connection_brands.get(client_id), visible_brands):
                    continue
                await websocket.send_json(message.model_dump(mode="json"))
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")

    @staticmethod
    def _brands_allow(
        recipient_brands: Optional[List[str]], visible_brands: Optional[List[str]]
    ) -> bool:
        """Decide whether a recipient may receive a (possibly scoped) message.

        See ``broadcast`` for the fail-open-authN / fail-closed-authZ rationale.
        """
        # Unscoped message (auth disabled / dev) -> deliver to all.
        if visible_brands is None:
            return True
        # Authed recipient with unknown brands -> fail CLOSED.
        if recipient_brands is None:
            return False
        # 'all' grant on either side matches everything (admin).
        if "all" in recipient_brands or "all" in visible_brands:
            return True
        return bool(set(recipient_brands) & set(visible_brands))

    def _matches_subscription(self, message: GraphStreamMessage, sub: GraphSubscription) -> bool:
        """Check if message matches subscription filters."""
        payload = message.payload

        # Check entity type filter
        if sub.entity_types and "type" in payload:
            if payload["type"] not in [t.value for t in sub.entity_types]:
                return False

        # Check session ID filter
        if sub.session_ids and message.session_id:
            if message.session_id not in sub.session_ids:
                return False

        return True


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _get_graphiti_service():
    """Get the Graphiti service with proper error handling."""
    try:
        from src.memory.graphiti_service import get_graphiti_service

        return await get_graphiti_service()
    except ImportError:
        logger.warning("Graphiti service not available, using fallback")
        return None
    except Exception as e:
        logger.error(f"Failed to get Graphiti service: {e}")
        return None


async def _get_semantic_memory():
    """Get FalkorDB semantic memory."""
    try:
        from src.memory.semantic_memory import get_semantic_memory

        return get_semantic_memory()
    except ImportError:
        logger.warning("Semantic memory not available")
        return None
    except Exception as e:
        logger.error(f"Failed to get semantic memory: {e}")
        return None


def _convert_to_graph_node(data: Dict[str, Any]) -> GraphNode:
    """Convert raw node data to GraphNode model."""
    # Try e2i_entity_type first (set by causal agents), then type/entity_type
    type_str = data.get("e2i_entity_type") or data.get("type") or data.get("entity_type") or "Agent"
    try:
        entity_type = EntityType(type_str)
    except ValueError:
        # Map common causal-engine types to the closest enum member
        _TYPE_MAP = {
            "SegmentEffect": EntityType.PREDICTION,
            "CausalEffect": EntityType.CAUSAL_PATH,
            "TreatmentEffect": EntityType.TREATMENT,
            "ExperimentResult": EntityType.EXPERIMENT,
        }
        entity_type = _TYPE_MAP.get(type_str, EntityType.AGENT)
        data = {**data, "original_type": type_str}

    # Derive a display name when name/label are missing
    name = data.get("name") or data.get("label") or ""
    if not name:
        # Try common property fields that make good display names
        name = data.get("segment_id") or data.get("entity_name") or data.get("title") or ""
    if not name:
        # Fall back to a human-readable version of the node ID
        node_id = data.get("id", data.get("node_id", ""))
        if ":" in node_id:
            # e.g. "segment_effect:hcp_specialty_Oncology:..." → "hcp_specialty_Oncology"
            parts = node_id.split(":")
            name = parts[1] if len(parts) > 1 else parts[0]
        else:
            name = node_id

    return GraphNode(
        id=data.get("id", data.get("node_id", "")),
        type=entity_type,
        name=name,
        properties={
            k: v
            for k, v in data.items()
            if k
            not in [
                "id",
                "node_id",
                "type",
                "entity_type",
                "e2i_entity_type",
                "name",
                "label",
                "created_at",
                "updated_at",
            ]
        },
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def _convert_to_graph_relationship(data: Dict[str, Any]) -> GraphRelationship:
    """Convert raw relationship data to GraphRelationship model."""
    # Get type with fallback for unknown types
    type_str = data.get("type", data.get("relationship_type", "RELATES_TO"))
    try:
        rel_type = RelationshipType(type_str)
    except ValueError:
        # Unknown type - use RELATES_TO as fallback but preserve in properties
        rel_type = RelationshipType.RELATES_TO
        data = {**data, "original_type": type_str}

    return GraphRelationship(
        id=data.get("id", data.get("rel_id", "")),
        type=rel_type,
        source_id=data.get("source_id", data.get("from_id", "")),
        target_id=data.get("target_id", data.get("to_id", "")),
        properties={
            k: v
            for k, v in data.items()
            if k
            not in [
                "id",
                "rel_id",
                "type",
                "relationship_type",
                "source_id",
                "from_id",
                "target_id",
                "to_id",
                "confidence",
                "created_at",
            ]
        },
        confidence=data.get("confidence"),
        created_at=data.get("created_at"),
    )


# =============================================================================
# NODE ENDPOINTS
# =============================================================================


@router.get(
    "/nodes",
    response_model=ListNodesResponse,
    summary="List graph nodes",
    operation_id="list_graph_nodes",
)
async def list_nodes(
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types"),
    search: Optional[str] = Query(None, max_length=500, description="Text search"),
    # Cap raised to 2000 so the Knowledge-Graph page can pull the FULL graph in
    # one window (~640 nodes today) and compute connected components client-side
    # for the "Show all types" view. The default stays small for ad-hoc callers.
    limit: int = Query(50, ge=1, le=2000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    sort_by: NodeSortField = Query(NodeSortField.CREATED_AT, description="Sort field"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="Sort order"),
) -> ListNodesResponse:
    """
    List nodes in the knowledge graph with filtering and pagination.

    Supports filtering by:
    - Entity types (HCP, Brand, Patient, etc.)
    - Text search across node names and properties
    """
    start_time = time.time()

    try:
        semantic = await _get_semantic_memory()
        if not semantic:
            # Return empty result if service unavailable
            return ListNodesResponse(
                nodes=[],
                total_count=0,
                limit=limit,
                offset=offset,
                has_more=False,
                query_latency_ms=0.0,
            )

        # Parse entity types
        types_filter = None
        if entity_types:
            types_filter = [t.strip() for t in entity_types.split(",")]

        # Build and execute query
        nodes_data = semantic.list_nodes(
            entity_types=types_filter, search=search, limit=limit, offset=offset
        )

        # Convert to GraphNode objects
        nodes = [_convert_to_graph_node(n) for n in nodes_data]

        # Get total count
        total_count = semantic.count_nodes(entity_types=types_filter, search=search)

        latency_ms = (time.time() - start_time) * 1000

        return ListNodesResponse(
            nodes=nodes,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + len(nodes)) < total_count,
            query_latency_ms=latency_ms,
        )

    except Exception as e:
        logger.exception("Failed to list nodes")
        raise HTTPException(status_code=500, detail="Failed to list nodes") from e


@router.get(
    "/nodes/{node_id}",
    response_model=GraphNode,
    summary="Get node by ID",
    operation_id="get_graph_node",
)
async def get_node(node_id: str) -> GraphNode:
    """
    Get a specific node by ID.

    Returns full node details including all properties.
    """
    try:
        semantic = await _get_semantic_memory()
        if not semantic:
            raise HTTPException(status_code=503, detail="Graph service unavailable")

        node_data = semantic.get_node(node_id)
        if not node_data:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        return _convert_to_graph_node(node_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get node %s", node_id)
        raise HTTPException(status_code=500, detail="Failed to get node") from e


@router.get(
    "/nodes/{node_id}/network",
    response_model=NodeNetworkResponse,
    summary="Get node relationship network",
    operation_id="get_node_network",
)
async def get_node_network(
    node_id: str, max_depth: int = Query(2, ge=1, le=5, description="Maximum traversal depth")
) -> NodeNetworkResponse:
    """
    Get the relationship network around a node.

    Returns all connected nodes within max_depth hops, grouped by type.
    Supports Patient and HCP nodes with specialized network discovery.
    """
    start_time = time.time()

    try:
        semantic = await _get_semantic_memory()
        if not semantic:
            raise HTTPException(status_code=503, detail="Graph service unavailable")

        # Get the node to determine its type
        node_data = semantic.get_node(node_id)
        if not node_data:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        node_type_str = node_data.get("type", node_data.get("entity_type", ""))

        # Map to EntityType enum
        try:
            node_type = EntityType(node_type_str)
        except ValueError:
            node_type = EntityType.AGENT  # fallback

        # Call appropriate network method based on node type
        if node_type == EntityType.PATIENT:
            network = semantic.get_patient_network(node_id, max_depth=max_depth)
            connected_nodes = {
                "hcps": network.get("hcps", []),
                "treatments": network.get("treatments", []),
                "triggers": network.get("triggers", []),
                "causal_paths": network.get("causal_paths", []),
                "brands": network.get("brands", []),
            }
        elif node_type == EntityType.HCP:
            network = semantic.get_hcp_influence_network(node_id, max_depth=max_depth)
            connected_nodes = {
                "influenced_hcps": network.get("influenced_hcps", []),
                "patients": network.get("patients", []),
                "brands_prescribed": network.get("brands_prescribed", []),
            }
        else:
            # Generic traversal for other node types
            subgraph = semantic.traverse_from_node(start_node_id=node_id, max_depth=max_depth)
            # Group nodes by type from traversal results
            connected_nodes = {}
            for node in subgraph.get("nodes", []):
                n_type = node.get("type", node.get("entity_type", "other"))
                if n_type not in connected_nodes:
                    connected_nodes[n_type] = []
                connected_nodes[n_type].append(
                    {
                        "id": node.get("id"),
                        "properties": {
                            k: v for k, v in node.items() if k not in ["id", "type", "entity_type"]
                        },
                    }
                )

        # Calculate total connections
        total_connections = sum(len(nodes) for nodes in connected_nodes.values())

        latency_ms = (time.time() - start_time) * 1000

        return NodeNetworkResponse(
            node_id=node_id,
            node_type=node_type,
            connected_nodes=connected_nodes,
            total_connections=total_connections,
            max_depth=max_depth,
            query_latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get network for node %s", node_id)
        raise HTTPException(status_code=500, detail="Failed to get node network") from e


# =============================================================================
# RELATIONSHIP ENDPOINTS
# =============================================================================


@router.get(
    "/relationships",
    response_model=ListRelationshipsResponse,
    summary="List graph relationships",
    operation_id="list_graph_relationships",
)
async def list_relationships(
    relationship_types: Optional[str] = Query(
        None, description="Comma-separated relationship types"
    ),
    source_id: Optional[str] = Query(None, description="Filter by source node"),
    target_id: Optional[str] = Query(None, description="Filter by target node"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence"),
    # Cap raised to 2000 to match /nodes — the Knowledge-Graph page pulls the
    # full edge set (~610 today) so it can compute connected components and the
    # rendered stats client-side. Default stays small for ad-hoc callers.
    limit: int = Query(50, ge=1, le=2000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> ListRelationshipsResponse:
    """
    List relationships in the knowledge graph with filtering.

    Supports filtering by:
    - Relationship types (CAUSES, IMPACTS, PRESCRIBES, etc.)
    - Source/target node IDs
    - Minimum confidence threshold
    """
    start_time = time.time()

    try:
        semantic = await _get_semantic_memory()
        if not semantic:
            return ListRelationshipsResponse(
                relationships=[],
                total_count=0,
                limit=limit,
                offset=offset,
                has_more=False,
                query_latency_ms=0.0,
            )

        # Parse relationship types
        types_filter = None
        if relationship_types:
            types_filter = [t.strip() for t in relationship_types.split(",")]

        # Execute query
        rels_data = semantic.list_relationships(
            relationship_types=types_filter,
            source_id=source_id,
            target_id=target_id,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )

        relationships = [_convert_to_graph_relationship(r) for r in rels_data]
        total_count = semantic.count_relationships(
            relationship_types=types_filter, source_id=source_id, target_id=target_id
        )

        latency_ms = (time.time() - start_time) * 1000

        return ListRelationshipsResponse(
            relationships=relationships,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + len(relationships)) < total_count,
            query_latency_ms=latency_ms,
        )

    except Exception as e:
        logger.exception("Failed to list relationships")
        raise HTTPException(status_code=500, detail="Failed to list relationships") from e


# =============================================================================
# TRAVERSAL ENDPOINTS
# =============================================================================


@router.post(
    "/traverse",
    response_model=TraverseResponse,
    summary="Traverse graph from node",
    operation_id="traverse_graph",
)
async def traverse_graph(request: TraverseRequest) -> TraverseResponse:
    """
    Traverse the graph from a starting node.

    Returns a subgraph containing:
    - All nodes within max_depth hops
    - All relationships connecting them
    - Paths from start to discovered nodes
    """
    start_time = time.time()

    try:
        graphiti = await _get_graphiti_service()
        if graphiti:
            # Use Graphiti service for traversal
            result = await graphiti.get_entity_subgraph(
                entity_id=request.start_node_id, max_depth=request.max_depth
            )

            nodes = [_convert_to_graph_node(n) for n in result.nodes]
            relationships = [
                _convert_to_graph_relationship(r) for r in result.edges
            ]  # edges, not relationships

            latency_ms = (time.time() - start_time) * 1000

            return TraverseResponse(
                subgraph={
                    "nodes": [n.model_dump() for n in nodes],
                    "relationships": [r.model_dump() for r in relationships],
                },
                nodes=nodes,
                relationships=relationships,
                paths=[],
                max_depth_reached=result.depth,
                query_latency_ms=latency_ms,
            )

        # Fallback to semantic memory
        semantic = await _get_semantic_memory()
        if not semantic:
            raise HTTPException(status_code=503, detail="Graph service unavailable")

        subgraph_data = semantic.traverse_from_node(
            start_node_id=request.start_node_id,
            relationship_types=(
                [r.value for r in request.relationship_types]
                if request.relationship_types
                else None
            ),
            direction=request.direction,
            max_depth=request.max_depth,
            min_confidence=request.min_confidence,
        )

        nodes = [_convert_to_graph_node(n) for n in subgraph_data.get("nodes", [])]
        relationships = [
            _convert_to_graph_relationship(r) for r in subgraph_data.get("relationships", [])
        ]

        latency_ms = (time.time() - start_time) * 1000

        return TraverseResponse(
            subgraph=subgraph_data,
            nodes=nodes,
            relationships=relationships,
            paths=[],
            max_depth_reached=subgraph_data.get("max_depth_reached", request.max_depth),
            query_latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Graph traversal failed")
        raise HTTPException(status_code=500, detail="Graph traversal failed") from e


# =============================================================================
# CAUSAL CHAIN ENDPOINTS
# =============================================================================


@router.post(
    "/causal-chains",
    response_model=CausalChainResponse,
    summary="Query causal chains",
    operation_id="query_causal_chains",
)
async def query_causal_chains(request: CausalChainRequest) -> CausalChainResponse:
    """
    Query causal chains in the knowledge graph.

    Finds chains of CAUSES/IMPACTS relationships connecting:
    - Entities to KPIs
    - Source to target entities
    - Upstream drivers to downstream outcomes
    """
    start_time = time.time()

    try:
        graphiti = await _get_graphiti_service()
        if graphiti:
            # Use Graphiti service for causal chain queries
            # Returns List[Dict] with nodes/edges per chain
            chains_data = await graphiti.get_causal_chains(
                kpi_name=request.kpi_name,
                max_depth=request.max_chain_length,
                min_confidence=request.min_confidence,
            )

            # Convert to GraphPath objects
            chains = []
            for chain in chains_data:
                nodes = [_convert_to_graph_node(n) for n in chain.get("nodes", [])]
                rels = [_convert_to_graph_relationship(r) for r in chain.get("edges", [])]
                chains.append(
                    GraphPath(
                        nodes=nodes,
                        relationships=rels,
                        total_confidence=chain.get("confidence"),
                        path_length=chain.get("length", len(rels)),
                    )
                )

            strongest = None
            if chains:
                strongest = max(chains, key=lambda c: c.total_confidence or 0)

            latency_ms = (time.time() - start_time) * 1000

            return CausalChainResponse(
                chains=chains,
                total_chains=len(chains),
                strongest_chain=strongest,
                aggregate_effect=None,  # Not provided by graphiti service
                query_latency_ms=latency_ms,
            )

        # Fallback to semantic memory
        semantic = await _get_semantic_memory()
        if not semantic:
            raise HTTPException(status_code=503, detail="Graph service unavailable")

        chains_data = semantic.find_causal_chains(
            kpi_name=request.kpi_name,
            source_entity_id=request.source_entity_id,
            target_entity_id=request.target_entity_id,
            max_length=request.max_chain_length,
            min_confidence=request.min_confidence,
        )

        chains = []
        for chain in chains_data:
            nodes = [_convert_to_graph_node(n) for n in chain.get("nodes", [])]
            rels = [_convert_to_graph_relationship(r) for r in chain.get("relationships", [])]
            chains.append(
                GraphPath(
                    nodes=nodes,
                    relationships=rels,
                    total_confidence=chain.get("confidence"),
                    path_length=len(rels),
                )
            )

        strongest = None
        if chains:
            strongest = max(chains, key=lambda c: c.total_confidence or 0)

        latency_ms = (time.time() - start_time) * 1000

        return CausalChainResponse(
            chains=chains,
            total_chains=len(chains),
            strongest_chain=strongest,
            aggregate_effect=None,
            query_latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Causal chain query failed")
        raise HTTPException(status_code=500, detail="Causal chain query failed") from e


# =============================================================================
# CYPHER QUERY ENDPOINT
# =============================================================================


@router.post(
    "/query",
    response_model=CypherQueryResponse,
    summary="Execute openCypher query (DISABLED)",
    operation_id="execute_cypher_query",
)
async def execute_cypher_query(request: CypherQueryRequest) -> CypherQueryResponse:
    """
    Raw openCypher query passthrough — **disabled by design**.

    This endpoint previously called an arbitrary-Cypher executor against the
    knowledge graph, which holds Patient/HCP PHI/PII. Executing user-supplied
    Cypher cannot be made safe with a keyword-based "read-only" filter:
    substring checks for ``CREATE``/``DELETE`` are trivially bypassed (e.g. a
    property literally named ``createdAt``, ``CALL`` procedures, or comment
    evasion), and even a perfectly read-only query is an unbounded PHI
    exfiltration primitive.

    Rather than implement an arbitrary-query executor, this endpoint refuses
    all requests with ``501 Not Implemented``. Use the purpose-built,
    parameterized endpoints instead — ``/graph/nodes``, ``/graph/relationships``,
    ``/graph/traverse``, ``/graph/causal-chains``, ``/graph/search`` — each of
    which validates and parameterizes its inputs. For ad-hoc interactive
    exploration, use the FalkorDB Browser in a trusted/admin context.
    """
    logger.warning(
        "Rejected raw Cypher query request (endpoint disabled); read_only=%s, query_length=%d",
        request.read_only,
        len(request.query),
    )
    raise HTTPException(
        status_code=501,
        detail=(
            "Raw Cypher query execution is disabled. Use the structured graph "
            "endpoints (/graph/nodes, /graph/relationships, /graph/traverse, "
            "/graph/causal-chains, /graph/search) instead."
        ),
    )


# =============================================================================
# EPISODE ENDPOINTS
# =============================================================================


@router.post(
    "/episodes",
    response_model=AddEpisodeResponse,
    summary="Add knowledge episode",
    operation_id="add_graph_episode",
)
async def add_episode(
    request: AddEpisodeRequest,
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> AddEpisodeResponse:
    """
    Add a knowledge episode to the graph.

    Graphiti automatically:
    - Extracts entities (HCP, Brand, Patient, etc.)
    - Discovers relationships
    - Links to existing graph nodes
    - Tracks temporal validity

    The episode-added broadcast is scoped to the PUBLISHER's brand grants
    (``get_user_brands(user)``) so only recipients sharing a brand (or admins
    with the ``all`` grant) receive it. ``get_current_user`` is the OPTIONAL
    auth dependency — it returns ``None`` (never 401s) when auth is disabled or
    no credential is presented, which yields an UNSCOPED broadcast
    (``visible_brands=None`` -> deliver to all), preserving dev/test fail-open
    behaviour. (The route itself is already token-protected by the global auth
    middleware when auth is enabled; this dependency only reads the user.)
    """
    start_time = time.time()
    # Brand grants of the publisher; None when unauthenticated (dev fail-open).
    visible_brands: Optional[List[str]] = get_user_brands(user) if user else None

    try:
        graphiti = await _get_graphiti_service()
        if not graphiti:
            raise HTTPException(status_code=503, detail="Graphiti service unavailable")

        result = await graphiti.add_episode(
            content=request.content,
            source=request.source,
            session_id=request.session_id,
            metadata=request.metadata,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Broadcast to WebSocket subscribers, scoped to the publisher's brands.
        await manager.broadcast(
            GraphStreamMessage(
                event_type="episode_added",
                payload={
                    "episode_id": result.episode_id,
                    "source": request.source,
                    "entities_count": len(result.entities_extracted),
                    "relationships_count": len(result.relationships_extracted),
                },
                session_id=request.session_id,
            ),
            visible_brands=visible_brands,
        )

        return AddEpisodeResponse(
            episode_id=result.episode_id,
            extracted_entities=[
                {
                    "type": (
                        e.entity_type.value
                        if hasattr(e.entity_type, "value")
                        else str(e.entity_type)
                    ),
                    "name": e.name,
                    "confidence": e.confidence,
                }
                for e in result.entities_extracted
            ],
            extracted_relationships=[
                {
                    "type": (
                        r.relationship_type.value
                        if hasattr(r.relationship_type, "value")
                        else str(r.relationship_type)
                    ),
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "confidence": r.confidence,
                }
                for r in result.relationships_extracted
            ],
            content_summary=(
                request.content[:200] + "..." if len(request.content) > 200 else request.content
            ),
            processing_latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to add episode")
        raise HTTPException(status_code=500, detail="Failed to add episode") from e


# =============================================================================
# SEARCH ENDPOINTS
# =============================================================================


@router.post(
    "/search",
    response_model=SearchGraphResponse,
    summary="Search knowledge graph",
    operation_id="search_graph",
)
async def search_graph(request: SearchGraphRequest) -> SearchGraphResponse:
    """
    Natural language search across the knowledge graph.

    Uses Graphiti's semantic search to find relevant:
    - Entities matching the query
    - Related facts and relationships
    - Historical episodes
    """
    start_time = time.time()

    try:
        graphiti = await _get_graphiti_service()
        if graphiti:
            results = await graphiti.search(
                query=request.query,
                session_id=request.session_id,
                entity_types=None,  # Service expects E2IEntityType, not str
                limit=request.k,  # 'limit' not 'k'
            )

            # Filter by min_score
            if request.min_score > 0:
                results = [r for r in results if r.score >= request.min_score]

            latency_ms = (time.time() - start_time) * 1000

            return SearchGraphResponse(
                results=[
                    {
                        "id": r.entity_id,
                        "name": r.name,
                        "type": r.entity_type,
                        "score": r.score,
                        "properties": r.properties,
                        "relationships": r.relationships,
                    }
                    for r in results
                ],
                total_results=len(results),
                query=request.query,
                query_latency_ms=latency_ms,
            )

        # Fallback to semantic memory search
        semantic = await _get_semantic_memory()
        if not semantic:
            raise HTTPException(status_code=503, detail="Graph service unavailable")

        results = semantic.semantic_search(
            query=request.query,
            entity_types=[t.value for t in request.entity_types] if request.entity_types else None,
            k=request.k,
        )

        latency_ms = (time.time() - start_time) * 1000

        return SearchGraphResponse(
            results=results,
            total_results=len(results),
            query=request.query,
            query_latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Graph search failed")
        raise HTTPException(status_code=500, detail="Graph search failed") from e


# =============================================================================
# STATS ENDPOINT
# =============================================================================


@router.get(
    "/stats",
    response_model=GraphStatsResponse,
    summary="Get graph statistics",
    operation_id="get_graph_stats",
)
async def get_graph_stats() -> GraphStatsResponse:
    """
    Get knowledge graph statistics.

    Returns counts for:
    - Total nodes and relationships
    - Breakdown by entity/relationship type
    - Episode and community counts
    """
    try:
        graphiti = await _get_graphiti_service()
        if graphiti:
            stats = await graphiti.get_graph_stats()
            # Handle dict return from graphiti service
            return GraphStatsResponse(
                total_nodes=stats.get("total_nodes", 0),
                total_relationships=stats.get("total_edges", 0),  # edges -> relationships
                nodes_by_type=stats.get("nodes_by_type", {}),
                relationships_by_type=stats.get("edges_by_type", {}),  # edges -> relationships
                total_episodes=stats.get("total_episodes", 0),
                total_communities=stats.get("total_communities", 0),
                last_updated=stats.get("last_updated"),
            )

        # Fallback to semantic memory
        semantic = await _get_semantic_memory()
        if not semantic:
            return GraphStatsResponse(
                total_nodes=0,
                total_relationships=0,
                nodes_by_type={},
                relationships_by_type={},
                total_episodes=0,
                total_communities=0,
                last_updated=None,
            )

        stats = semantic.get_graph_stats()
        return GraphStatsResponse(
            total_nodes=stats.get("total_nodes", 0),
            total_relationships=stats.get("total_relationships", 0),
            nodes_by_type=stats.get("nodes_by_type", {}),
            relationships_by_type=stats.get("relationships_by_type", {}),
            total_episodes=0,
            total_communities=0,
            last_updated=stats.get("last_updated"),
        )

    except Exception as e:
        logger.exception("Failed to get graph stats")
        raise HTTPException(status_code=500, detail="Failed to get graph statistics") from e


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================


@router.websocket("/stream")
async def graph_stream(websocket: WebSocket, client_id: Optional[str] = None):
    """
    WebSocket endpoint for real-time graph updates.

    Clients can subscribe to:
    - New nodes/relationships
    - Episode additions
    - Graph updates

    Send a JSON subscription message to filter events:
    {
        "entity_types": ["HCP", "Brand"],
        "session_ids": ["sess_123"]
    }

    Authentication
    --------------
    The graph stream broadcasts node/relationship payloads that may contain
    Patient/HCP PHI/PII, so it must NOT be readable by anonymous clients — this
    is the WebSocket-channel parity for the HTTP exposure PR #657 closed (graph
    data endpoints removed from the public allowlist). The HTTP-scoped auth
    middleware does NOT run on a WebSocket handshake, so the handshake is
    authenticated here.

    Browsers cannot set WS headers, so the JWT is carried in
    ``Sec-WebSocket-Protocol`` as ``['bearer', base64url(jwt)]`` (PR #679; see
    ``frontend/src/hooks/use-websocket.ts``).

    The project's auth posture is INTENTIONALLY fail-open when unconfigured:

    * ``is_auth_enabled()`` True  -> a valid token is REQUIRED; an anonymous or
      invalid handshake is rejected with close code 1008 BEFORE accept.
    * ``is_auth_enabled()`` False -> accept WITHOUT a token (testing mode /
      Supabase unset), preserving dev + e2e behaviour.

    Authorization: publisher-grant brand filtering IS enforced (PR #684). The
    authenticated recipient's brand grants are cached on the connection here
    (``user_brands`` -> ``ConnectionManager.connect``) and evaluated in
    ``ConnectionManager.broadcast`` via ``_brands_allow`` — a recipient receives
    an ``episode_added`` event only if their grants intersect the PUBLISHER's
    (or either side holds the ``all`` admin grant). A stricter PER-EPISODE brand
    (vs. the publisher's grant set) is a deliberately-deferred follow-up.
    """
    client_id = client_id or f"client_{id(websocket)}"

    accept_subprotocol: Optional[str] = None
    # Recipient brand grants, cached on the connection for publisher-grant authZ
    # in ConnectionManager.broadcast. ``None`` when auth is disabled (no user) ->
    # messages are also unscoped in that mode, so deliver-all is preserved
    # (consistent fail-open authN); when auth is on, an authed recipient with no
    # matching brand receives nothing (fail-closed authZ).
    user_brands: Optional[List[str]] = None
    if is_auth_enabled():
        user = await authenticate_websocket(websocket)
        if user is None:
            # Reject anonymous/invalid handshakes with a policy-violation close
            # WITHOUT accepting (do not leak any broadcast data).
            await websocket.close(code=1008)
            return
        user_brands = get_user_brands(user) if user else None
        # Echo the bearer subprotocol only when the client offered it.
        subprotocols = websocket.scope.get("subprotocols") or []
        if subprotocols and subprotocols[0] == WS_BEARER_SUBPROTOCOL:
            accept_subprotocol = WS_BEARER_SUBPROTOCOL

    await manager.connect(
        websocket, client_id, subprotocol=accept_subprotocol, user_brands=user_brands
    )

    try:
        while True:
            # Receive subscription updates
            data = await websocket.receive_text()
            try:
                sub_data = json.loads(data)
                subscription = GraphSubscription(**sub_data)
                manager.set_subscription(client_id, subscription)
                await websocket.send_json(
                    {"type": "subscription_updated", "message": "Subscription filters applied"}
                )
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================


@router.get("/health", summary="Graph service health check", operation_id="graph_health_check")
async def graph_health() -> Dict[str, Any]:
    """
    Health check for graph services.

    Returns status of:
    - Graphiti service
    - FalkorDB connection
    - WebSocket connections
    """
    graphiti_status = "unavailable"
    falkordb_status = "unavailable"

    try:
        graphiti = await _get_graphiti_service()
        if graphiti:
            graphiti_status = "connected"
    except Exception:
        pass

    try:
        semantic = await _get_semantic_memory()
        if semantic:
            falkordb_status = "connected"
    except Exception:
        pass

    return {
        "status": (
            "healthy"
            if graphiti_status == "connected" or falkordb_status == "connected"
            else "degraded"
        ),
        "graphiti": graphiti_status,
        "falkordb": falkordb_status,
        "websocket_connections": len(manager.active_connections),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
