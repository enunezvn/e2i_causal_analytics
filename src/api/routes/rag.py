"""
E2I Hybrid RAG API Endpoints
============================

FastAPI endpoints for hybrid retrieval-augmented generation.

Provides:
- Hybrid search across vector, fulltext, and graph backends
- Causal subgraph retrieval for visualization
- Causal path finding between entities
- Backend health monitoring

Integration Points:
- HybridRetriever (src/rag/hybrid_retriever.py)
- EntityExtractor (src/rag/entity_extractor.py)
- HealthMonitor (src/rag/health_monitor.py)
- SearchLogger (src/rag/search_logger.py)

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.rag import (
    HybridRetriever,
    EntityExtractor,
    HealthMonitor,
    SearchLogger,
    RAGConfig,
    RetrievalResult,
    RetrievalSource,
    ExtractedEntities,
    BackendStatus,
)
from src.rag.exceptions import (
    RAGError,
    CircuitBreakerOpenError,
    BackendTimeoutError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["Hybrid RAG"])


# =============================================================================
# ENUMS & REQUEST/RESPONSE MODELS
# =============================================================================


class SearchMode(str, Enum):
    """Search mode options."""
    HYBRID = "hybrid"       # Use all backends with RRF fusion
    VECTOR_ONLY = "vector"  # Semantic search only
    FULLTEXT_ONLY = "fulltext"  # Keyword search only
    GRAPH_ONLY = "graph"    # Graph traversal only


class ResultFormat(str, Enum):
    """Result format options."""
    FULL = "full"           # All metadata and scores
    COMPACT = "compact"     # Essential fields only
    IDS_ONLY = "ids"        # Just document IDs


class SearchResultItem(BaseModel):
    """Single search result item."""
    document_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content or snippet")
    score: float = Field(..., description="Relevance score (0-1)")
    source: str = Field(..., description="Source backend (vector/fulltext/graph)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "causal_path_001",
                "content": "Kisqali TRx decline in Q3 2024 was caused by...",
                "score": 0.89,
                "source": "vector",
                "metadata": {
                    "brand": "Kisqali",
                    "region": "northeast",
                    "time_period": "Q3_2024"
                }
            }
        }


class ExtractedEntitiesResponse(BaseModel):
    """Extracted entities from query."""
    brands: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)
    kpis: List[str] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=list)
    journey_stages: List[str] = Field(default_factory=list)
    time_references: List[str] = Field(default_factory=list)
    hcp_segments: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """Hybrid search request payload."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Search mode")
    top_k: int = Field(default=10, ge=1, le=50, description="Maximum results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score")
    require_all_sources: bool = Field(default=False, description="Require results from all backends")
    include_graph_boost: bool = Field(default=True, description="Apply graph context boost")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")
    format: ResultFormat = Field(default=ResultFormat.FULL, description="Result format")
    session_id: Optional[str] = Field(None, description="Session ID for logging")
    user_id: Optional[str] = Field(None, description="User ID for logging")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Why did Kisqali TRx drop in the West during Q3?",
                "mode": "hybrid",
                "top_k": 10,
                "min_score": 0.5,
                "include_graph_boost": True
            }
        }


class SearchResponse(BaseModel):
    """Hybrid search response payload."""
    search_id: str = Field(..., description="Unique search identifier")
    query: str = Field(..., description="Original query")
    timestamp: datetime = Field(..., description="Search timestamp")

    # Results
    results: List[SearchResultItem] = Field(..., description="Ranked search results")
    total_results: int = Field(..., description="Total number of results")

    # Extracted entities
    entities: ExtractedEntitiesResponse = Field(..., description="Extracted domain entities")

    # Search stats
    stats: Dict[str, Any] = Field(..., description="Search statistics per backend")

    # Performance
    latency_ms: float = Field(..., description="Total search latency in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "search_id": "SEARCH-20241220-abc123",
                "query": "Why did Kisqali TRx drop in the West during Q3?",
                "timestamp": "2024-12-20T10:30:00Z",
                "results": [],
                "total_results": 8,
                "entities": {
                    "brands": ["Kisqali"],
                    "regions": ["west"],
                    "kpis": ["trx"],
                    "time_references": ["Q3"]
                },
                "stats": {
                    "vector_count": 5,
                    "fulltext_count": 3,
                    "graph_count": 2
                },
                "latency_ms": 234.5
            }
        }


class GraphNode(BaseModel):
    """Node in the causal graph."""
    id: str = Field(..., description="Node identifier")
    label: str = Field(..., description="Display label")
    type: str = Field(..., description="Node type (brand/kpi/region/agent)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")


class GraphEdge(BaseModel):
    """Edge in the causal graph."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relationship: str = Field(..., description="Relationship type")
    weight: float = Field(default=1.0, description="Edge weight")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge properties")


class CausalSubgraphResponse(BaseModel):
    """Causal subgraph for visualization."""
    entity: str = Field(..., description="Center entity of the subgraph")
    nodes: List[GraphNode] = Field(..., description="Graph nodes")
    edges: List[GraphEdge] = Field(..., description="Graph edges")
    depth: int = Field(..., description="Traversal depth")
    node_count: int = Field(..., description="Total node count")
    edge_count: int = Field(..., description="Total edge count")
    query_time_ms: float = Field(..., description="Query time in milliseconds")


class CausalPathResponse(BaseModel):
    """Causal path between two entities."""
    source: str = Field(..., description="Source entity")
    target: str = Field(..., description="Target entity")
    paths: List[List[str]] = Field(..., description="List of paths (node sequences)")
    shortest_path_length: int = Field(..., description="Length of shortest path")
    total_paths: int = Field(..., description="Number of paths found")
    query_time_ms: float = Field(..., description="Query time in milliseconds")


class BackendHealthStatus(BaseModel):
    """Health status for a single backend."""
    status: str = Field(..., description="healthy/degraded/unhealthy/unknown")
    latency_ms: float = Field(..., description="Last check latency")
    last_check: datetime = Field(..., description="Last health check time")
    consecutive_failures: int = Field(default=0, description="Consecutive failure count")
    circuit_breaker_state: Optional[str] = Field(None, description="Circuit breaker state")
    error: Optional[str] = Field(None, description="Last error message")


class HealthResponse(BaseModel):
    """Overall RAG health response."""
    status: str = Field(..., description="Overall status: healthy/degraded/unhealthy")
    timestamp: datetime = Field(..., description="Health check timestamp")
    backends: Dict[str, BackendHealthStatus] = Field(..., description="Per-backend health")
    monitoring_enabled: bool = Field(..., description="Whether background monitoring is active")


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


class RAGService:
    """
    Service layer for RAG operations.

    Manages:
    - HybridRetriever for search
    - EntityExtractor for query understanding
    - HealthMonitor for backend health
    - SearchLogger for audit trail
    """

    _instance: Optional["RAGService"] = None
    _initialized: bool = False

    def __init__(self):
        if not RAGService._initialized:
            self._initialize()
            RAGService._initialized = True

    def _initialize(self):
        """Initialize RAG components."""
        try:
            # Load config from environment
            self.config = RAGConfig.from_env()

            # Initialize components
            self.entity_extractor = EntityExtractor()

            # Note: HybridRetriever and HealthMonitor require external connections
            # In production, these would be properly initialized
            # For now, we'll initialize them lazily when connections are available
            self._retriever: Optional[HybridRetriever] = None
            self._health_monitor: Optional[HealthMonitor] = None
            self._search_logger: Optional[SearchLogger] = None

            logger.info("RAG service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise

    @classmethod
    def get_instance(cls) -> "RAGService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def retriever(self) -> HybridRetriever:
        """Get or create HybridRetriever."""
        if self._retriever is None:
            self._retriever = HybridRetriever(self.config)
        return self._retriever

    @property
    def health_monitor(self) -> HealthMonitor:
        """Get or create HealthMonitor."""
        if self._health_monitor is None:
            from src.rag.config import HealthMonitorConfig
            self._health_monitor = HealthMonitor(config=HealthMonitorConfig())
        return self._health_monitor

    @property
    def search_logger(self) -> SearchLogger:
        """Get or create SearchLogger."""
        if self._search_logger is None:
            self._search_logger = SearchLogger(supabase_client=None)  # Will be configured
        return self._search_logger

    def extract_entities(self, query: str) -> ExtractedEntities:
        """Extract entities from query."""
        return self.entity_extractor.extract(query)

    async def search(
        self,
        query: str,
        mode: SearchMode,
        top_k: int,
        min_score: float,
        include_graph_boost: bool,
        filters: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Execute search based on mode.

        Returns:
            Tuple of (results, stats)
        """
        # Extract entities for graph queries
        entities = self.extract_entities(query)

        # Execute search via retriever
        results = await self.retriever.search(
            query=query,
            top_k=top_k,
            entities=entities,
            filters=filters,
        )

        # Apply minimum score filter
        results = [r for r in results if r.score >= min_score]

        # Get stats
        stats = self.retriever.get_last_query_stats()

        return results, stats.__dict__ if stats else {}

    async def get_causal_subgraph(
        self,
        entity: str,
        depth: int = 2,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get causal subgraph around an entity."""
        return await self.retriever.get_causal_subgraph(
            entity=entity,
            depth=depth,
            limit=limit,
        )

    async def get_causal_path(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """Find causal paths between two entities."""
        return await self.retriever.get_causal_path(
            source=source,
            target=target,
            max_depth=max_depth,
        )

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all backends."""
        return await self.health_monitor.get_health_status()


def get_rag_service() -> RAGService:
    """Dependency injection for RAG service."""
    return RAGService.get_instance()


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Execute hybrid search",
    description="""
    Perform hybrid search across vector, fulltext, and graph backends.

    **Search Modes:**
    - `hybrid`: Use all backends with RRF fusion (recommended)
    - `vector`: Semantic similarity search only
    - `fulltext`: Keyword-based search only
    - `graph`: Graph traversal only

    **Features:**
    - Automatic entity extraction from natural language
    - Reciprocal Rank Fusion for result combination
    - Graph boost for results with causal connections
    - Graceful degradation if backends fail

    **Example queries:**
    - "Why did Kisqali TRx drop in the West during Q3?"
    - "Compare Remibrutinib conversion rates across regions"
    - "What caused the adoption decline for Fabhalta?"
    """
)
async def search(
    request: SearchRequest,
    service: RAGService = Depends(get_rag_service),
) -> SearchResponse:
    """Execute hybrid search."""
    start_time = time.time()
    search_id = f"SEARCH-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

    try:
        # Extract entities
        entities = service.extract_entities(request.query)

        # Execute search
        results, stats = await service.search(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            min_score=request.min_score,
            include_graph_boost=request.include_graph_boost,
            filters=request.filters,
        )

        # Convert to response format
        result_items = []
        for r in results:
            if request.format == ResultFormat.IDS_ONLY:
                result_items.append(SearchResultItem(
                    document_id=r.id,
                    content="",
                    score=r.score,
                    source=r.source.value,
                    metadata={},
                ))
            elif request.format == ResultFormat.COMPACT:
                result_items.append(SearchResultItem(
                    document_id=r.id,
                    content=r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    score=r.score,
                    source=r.source.value,
                    metadata={},
                ))
            else:  # FULL
                result_items.append(SearchResultItem(
                    document_id=r.id,
                    content=r.content,
                    score=r.score,
                    source=r.source.value,
                    metadata=r.metadata or {},
                ))

        latency_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            search_id=search_id,
            query=request.query,
            timestamp=datetime.now(timezone.utc),
            results=result_items,
            total_results=len(result_items),
            entities=ExtractedEntitiesResponse(
                brands=entities.brands,
                regions=entities.regions,
                kpis=entities.kpis,
                agents=entities.agents,
                journey_stages=entities.journey_stages,
                time_references=entities.time_references,
                hcp_segments=entities.hcp_segments,
            ),
            stats=asdict(stats),
            latency_ms=round(latency_ms, 2),
        )

    except CircuitBreakerOpenError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Backend temporarily unavailable: {e.backend}. Retry after {e.reset_time_seconds:.0f}s"
        )
    except BackendTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Search timed out: {str(e)}"
        )
    except RAGError as e:
        logger.error(f"RAG error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get(
    "/graph/{entity}",
    response_model=CausalSubgraphResponse,
    summary="Get causal subgraph",
    description="""
    Retrieve the causal subgraph centered on a specific entity.

    Returns nodes and edges for visualization with Cytoscape.js.

    **Entity types:**
    - Brands: Kisqali, Remibrutinib, Fabhalta
    - KPIs: trx, nrx, conversion_rate, market_share
    - Regions: northeast, south, midwest, west
    - Agents: causal_impact, drift_monitor, gap_analyzer
    """
)
async def get_causal_subgraph(
    entity: str,
    depth: int = Query(default=2, ge=1, le=5, description="Traversal depth"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum nodes"),
    service: RAGService = Depends(get_rag_service),
) -> CausalSubgraphResponse:
    """Get causal subgraph for an entity."""
    start_time = time.time()

    try:
        result = await service.get_causal_subgraph(
            entity=entity,
            depth=depth,
            limit=limit,
        )

        nodes = [
            GraphNode(
                id=n.get("id", ""),
                label=n.get("label", n.get("id", "")),
                type=n.get("type", "unknown"),
                properties=n.get("properties", {}),
            )
            for n in result.get("nodes", [])
        ]

        edges = [
            GraphEdge(
                source=e.get("source", ""),
                target=e.get("target", ""),
                relationship=e.get("relationship", "relates_to"),
                weight=e.get("weight", 1.0),
                properties=e.get("properties", {}),
            )
            for e in result.get("edges", [])
        ]

        query_time_ms = (time.time() - start_time) * 1000

        return CausalSubgraphResponse(
            entity=entity,
            nodes=nodes,
            edges=edges,
            depth=depth,
            node_count=len(nodes),
            edge_count=len(edges),
            query_time_ms=round(query_time_ms, 2),
        )

    except RAGError as e:
        logger.error(f"Graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting subgraph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")


@router.get(
    "/causal-path",
    response_model=CausalPathResponse,
    summary="Find causal paths",
    description="""
    Find causal paths between two entities in the knowledge graph.

    Useful for understanding causal chains like:
    - Why did a KPI drop?
    - What connects a brand to a region's performance?
    - How are different agents related?
    """
)
async def get_causal_path(
    source: str = Query(..., description="Source entity"),
    target: str = Query(..., description="Target entity"),
    max_depth: int = Query(default=5, ge=1, le=10, description="Maximum path length"),
    service: RAGService = Depends(get_rag_service),
) -> CausalPathResponse:
    """Find causal paths between entities."""
    start_time = time.time()

    try:
        result = await service.get_causal_path(
            source=source,
            target=target,
            max_depth=max_depth,
        )

        paths = result.get("paths", [])
        shortest = min(len(p) for p in paths) if paths else 0

        query_time_ms = (time.time() - start_time) * 1000

        return CausalPathResponse(
            source=source,
            target=target,
            paths=paths,
            shortest_path_length=shortest,
            total_paths=len(paths),
            query_time_ms=round(query_time_ms, 2),
        )

    except RAGError as e:
        logger.error(f"Path finding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error finding path: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Path query failed: {str(e)}")


@router.get(
    "/entities",
    response_model=ExtractedEntitiesResponse,
    summary="Extract entities from query",
    description="""
    Extract domain-specific entities from a natural language query.

    Useful for:
    - Query understanding before search
    - Building graph queries
    - Debugging entity extraction
    """
)
async def extract_entities(
    query: str = Query(..., min_length=1, max_length=1000, description="Query to analyze"),
    service: RAGService = Depends(get_rag_service),
) -> ExtractedEntitiesResponse:
    """Extract entities from a query."""
    try:
        entities = service.extract_entities(query)

        return ExtractedEntitiesResponse(
            brands=entities.brands,
            regions=entities.regions,
            kpis=entities.kpis,
            agents=entities.agents,
            journey_stages=entities.journey_stages,
            time_references=entities.time_references,
            hcp_segments=entities.hcp_segments,
        )

    except Exception as e:
        logger.error(f"Entity extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="RAG backend health",
    description="""
    Get health status of all RAG backends.

    Shows:
    - Overall system health
    - Per-backend status (vector, fulltext, graph)
    - Circuit breaker states
    - Recent errors
    """
)
async def get_health(
    service: RAGService = Depends(get_rag_service),
) -> HealthResponse:
    """Get RAG backend health status."""
    try:
        health = await service.get_health_status()

        backends = {}
        for name, status in health.get("backends", {}).items():
            backends[name] = BackendHealthStatus(
                status=status.get("status", "unknown"),
                latency_ms=status.get("latency_ms", 0.0),
                last_check=datetime.fromisoformat(status["last_check"]) if status.get("last_check") else datetime.now(timezone.utc),
                consecutive_failures=status.get("consecutive_failures", 0),
                circuit_breaker_state=status.get("circuit_breaker", {}).get("state") if status.get("circuit_breaker") else None,
                error=status.get("error"),
            )

        return HealthResponse(
            status=health.get("status", "unknown"),
            timestamp=datetime.now(timezone.utc),
            backends=backends,
            monitoring_enabled=health.get("monitoring_enabled", False),
        )

    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        # Return degraded status on error
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now(timezone.utc),
            backends={},
            monitoring_enabled=False,
        )


@router.get(
    "/stats",
    summary="RAG usage statistics",
    description="Get usage statistics for the RAG system."
)
async def get_stats(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to look back"),
    service: RAGService = Depends(get_rag_service),
) -> Dict[str, Any]:
    """Get RAG usage statistics."""
    # In production, this would query the search_logs table
    return {
        "period_hours": hours,
        "total_searches": 0,
        "avg_latency_ms": 0,
        "top_queries": [],
        "backend_usage": {
            "vector": 0,
            "fulltext": 0,
            "graph": 0,
        },
        "error_rate": 0.0,
        "message": "Statistics will be populated once search logging is configured"
    }
