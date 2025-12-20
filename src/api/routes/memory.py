"""
E2I Memory System API
=====================

FastAPI endpoints for memory operations:
- Hybrid search across memory types
- Episodic memory insertion/retrieval
- Procedural memory feedback
- Semantic graph path queries

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import logging

from src.rag.retriever import hybrid_search
from src.rag.models.retrieval_models import RetrievalResult
from src.memory.episodic_memory import (
    insert_episodic_memory_with_text,
    get_memory_by_id,
    EpisodicMemoryInput as EpisodicInput,
    E2IEntityReferences
)
from src.memory.procedural_memory import (
    update_procedure_outcome,
    get_procedure_by_id,
    record_learning_signal,
    LearningSignalInput
)
from src.memory.semantic_memory import get_semantic_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["Memory System"])


# =============================================================================
# ENUMS & MODELS
# =============================================================================

class MemoryType(str, Enum):
    """Memory types for filtering search."""
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"
    ALL = "all"


class RetrievalMethod(str, Enum):
    """Retrieval methods for hybrid search."""
    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH = "graph"
    HYBRID = "hybrid"


# -----------------------------------------------------------------------------
# Search Models
# -----------------------------------------------------------------------------

class MemorySearchRequest(BaseModel):
    """Request for hybrid memory search."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    memory_types: Optional[List[MemoryType]] = Field(
        default=None,
        description="Memory types to search (all if not specified)"
    )
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HYBRID,
        description="Retrieval method"
    )
    entities: Optional[List[str]] = Field(
        default=None,
        description="Entity IDs for graph traversal"
    )
    kpi_name: Optional[str] = Field(
        default=None,
        description="KPI name for targeted graph traversal"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional filters (brand, region, agent_name)"
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom weights for hybrid retrieval"
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "Why did TRx drop in northeast region?",
            "k": 10,
            "retrieval_method": "hybrid",
            "kpi_name": "TRx",
            "filters": {"brand": "Kisqali", "region": "northeast"}
        }
    })


class MemorySearchResult(BaseModel):
    """Single search result."""
    content: str = Field(..., description="Retrieved content")
    source: str = Field(..., description="Source memory type or table")
    source_id: str = Field(..., description="Source record ID")
    score: float = Field(..., description="Relevance score (0-1)")
    retrieval_method: str = Field(..., description="Method used to retrieve")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MemorySearchResponse(BaseModel):
    """Response for memory search."""
    query: str = Field(..., description="Original query")
    results: List[MemorySearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Number of results returned")
    retrieval_method: str = Field(..., description="Method used")
    search_latency_ms: float = Field(..., description="Search latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# -----------------------------------------------------------------------------
# Episodic Memory Models
# -----------------------------------------------------------------------------

class EpisodicMemoryInput(BaseModel):
    """Input for creating episodic memory."""
    content: str = Field(..., min_length=1, max_length=10000, description="Memory content")
    event_type: str = Field(..., description="Type of event (query, response, action)")
    session_id: Optional[str] = Field(None, description="Session ID")
    agent_name: Optional[str] = Field(None, description="Agent that created this memory")
    brand: Optional[str] = Field(None, description="Brand context")
    region: Optional[str] = Field(None, description="Region context")
    hcp_id: Optional[str] = Field(None, description="Associated HCP ID")
    patient_id: Optional[str] = Field(None, description="Associated patient ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "content": "User asked about TRx trends in northeast region for Kisqali",
            "event_type": "query",
            "session_id": "sess_abc123",
            "agent_name": "orchestrator",
            "brand": "Kisqali",
            "region": "northeast"
        }
    })


class EpisodicMemoryResponse(BaseModel):
    """Response for episodic memory operations."""
    id: str = Field(..., description="Memory ID")
    content: str = Field(..., description="Memory content")
    event_type: str = Field(..., description="Event type")
    session_id: Optional[str] = None
    agent_name: Optional[str] = None
    brand: Optional[str] = None
    region: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Procedural Memory Models
# -----------------------------------------------------------------------------

class ProceduralFeedbackRequest(BaseModel):
    """Request to record procedural outcome feedback."""
    procedure_id: str = Field(..., description="ID of the procedure")
    outcome: str = Field(..., description="Outcome: success, partial, failure")
    score: float = Field(..., ge=0.0, le=1.0, description="Outcome score (0-1)")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    session_id: Optional[str] = Field(None, description="Session context")
    agent_name: Optional[str] = Field(None, description="Agent providing feedback")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "procedure_id": "proc_causal_analysis_001",
            "outcome": "success",
            "score": 0.85,
            "feedback_text": "Causal analysis correctly identified HCP engagement drop",
            "agent_name": "feedback_learner"
        }
    })


class ProceduralFeedbackResponse(BaseModel):
    """Response for procedural feedback recording."""
    procedure_id: str
    feedback_recorded: bool
    new_success_rate: Optional[float] = None
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# -----------------------------------------------------------------------------
# Semantic Memory Models
# -----------------------------------------------------------------------------

class SemanticPathRequest(BaseModel):
    """Request for semantic graph path queries."""
    start_entity_id: Optional[str] = Field(None, description="Starting entity ID")
    end_entity_id: Optional[str] = Field(None, description="Ending entity ID")
    kpi_name: Optional[str] = Field(None, description="KPI name for causal paths")
    relationship_type: Optional[str] = Field(
        default="causal_path",
        description="Type of relationship to follow"
    )
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum traversal depth")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "kpi_name": "TRx",
            "relationship_type": "causal_path",
            "max_depth": 3,
            "min_confidence": 0.6
        }
    })


class SemanticPathResponse(BaseModel):
    """Response for semantic path queries."""
    paths: List[Dict[str, Any]] = Field(..., description="Found paths")
    total_paths: int = Field(..., description="Number of paths found")
    max_depth_searched: int = Field(..., description="Depth searched")
    query_latency_ms: float = Field(..., description="Query latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/search", response_model=MemorySearchResponse)
async def search_memory(request: MemorySearchRequest) -> MemorySearchResponse:
    """
    Execute hybrid search across memory systems.

    Combines:
    - Dense retrieval (vector similarity via episodic/procedural)
    - Sparse retrieval (BM25-like full-text search)
    - Graph retrieval (FalkorDB semantic traversal)

    Uses Reciprocal Rank Fusion for result combining.
    """
    import time
    start_time = time.time()

    try:
        # Execute hybrid search
        results = await hybrid_search(
            query=request.query,
            k=request.k,
            entities=request.entities,
            kpi_name=request.kpi_name,
            filters=request.filters
        )

        # Filter by minimum score
        if request.min_score > 0:
            results = [r for r in results if r.score >= request.min_score]

        # Convert to response format
        search_results = [
            MemorySearchResult(
                content=r.content,
                source=r.source,
                source_id=r.source_id,
                score=r.score,
                retrieval_method=r.retrieval_method,
                metadata=r.metadata
            )
            for r in results
        ]

        latency_ms = (time.time() - start_time) * 1000

        return MemorySearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            retrieval_method=request.retrieval_method.value,
            search_latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")


@router.post("/episodic", response_model=EpisodicMemoryResponse)
async def create_episodic_memory(
    memory: EpisodicMemoryInput,
    background_tasks: BackgroundTasks
) -> EpisodicMemoryResponse:
    """
    Insert a new episodic memory.

    Creates an embedding and stores in Supabase with pgvector.
    Used to capture significant interactions for future retrieval.
    """
    try:
        # Build E2I entity references
        e2i_refs = E2IEntityReferences(
            patient_id=memory.patient_id,
            hcp_id=memory.hcp_id,
            brand=memory.brand,
            region=memory.region
        )

        # Build episodic memory input
        episodic_input = EpisodicInput(
            event_type=memory.event_type,
            description=memory.content,
            agent_name=memory.agent_name,
            e2i_refs=e2i_refs,
            raw_content=memory.metadata
        )

        # Insert memory with auto-generated embedding
        memory_id = await insert_episodic_memory_with_text(
            memory=episodic_input,
            text_to_embed=memory.content,
            session_id=memory.session_id
        )

        return EpisodicMemoryResponse(
            id=memory_id,
            content=memory.content,
            event_type=memory.event_type,
            session_id=memory.session_id,
            agent_name=memory.agent_name,
            brand=memory.brand,
            region=memory.region,
            metadata=memory.metadata or {}
        )

    except Exception as e:
        logger.error(f"Episodic memory insertion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create episodic memory: {str(e)}")


@router.get("/episodic/{memory_id}", response_model=EpisodicMemoryResponse)
async def get_episodic_memory_endpoint(memory_id: str) -> EpisodicMemoryResponse:
    """
    Retrieve a specific episodic memory by ID.
    """
    try:
        result = await get_memory_by_id(memory_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Episodic memory {memory_id} not found")

        return EpisodicMemoryResponse(
            id=result.get("memory_id", memory_id),
            content=result.get("description", ""),
            event_type=result.get("event_type", "unknown"),
            session_id=result.get("session_id"),
            agent_name=result.get("agent_name"),
            brand=result.get("brand"),
            region=result.get("region"),
            created_at=result.get("occurred_at", datetime.now(timezone.utc)),
            metadata=result.get("raw_content", {})
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve episodic memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memory: {str(e)}")


@router.post("/procedural/feedback", response_model=ProceduralFeedbackResponse)
async def record_procedural_feedback(request: ProceduralFeedbackRequest) -> ProceduralFeedbackResponse:
    """
    Record outcome feedback for a procedural memory.

    Updates the procedure's success rate based on the outcome.
    Used by Feedback Learner for continuous improvement.
    """
    try:
        # Determine success based on outcome
        success = request.outcome == "success"

        # Update procedure outcome (success/failure counts)
        await update_procedure_outcome(
            procedure_id=request.procedure_id,
            success=success
        )

        # Also record a learning signal for DSPy training
        signal = LearningSignalInput(
            signal_type="rating" if request.score else "thumbs_up" if success else "thumbs_down",
            signal_value=request.score,
            applies_to_type="procedure",
            applies_to_id=request.procedure_id,
            rated_agent=request.agent_name,
            signal_details={"feedback_text": request.feedback_text} if request.feedback_text else None
        )
        await record_learning_signal(
            signal=signal,
            session_id=request.session_id
        )

        # Get updated procedure to return new success rate
        procedure = await get_procedure_by_id(request.procedure_id)
        new_success_rate = None
        if procedure:
            usage = procedure.get("usage_count", 1)
            successes = procedure.get("success_count", 0)
            new_success_rate = successes / usage if usage > 0 else None

        return ProceduralFeedbackResponse(
            procedure_id=request.procedure_id,
            feedback_recorded=True,
            new_success_rate=new_success_rate,
            message="Feedback recorded successfully"
        )

    except Exception as e:
        logger.error(f"Failed to record procedural feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@router.get("/semantic/paths", response_model=SemanticPathResponse)
async def query_semantic_paths(
    kpi_name: Optional[str] = None,
    start_entity_id: Optional[str] = None,
    relationship_type: str = "causal_path",
    max_depth: int = 3,
    min_confidence: float = 0.5
) -> SemanticPathResponse:
    """
    Query semantic graph for causal paths.

    Traverses FalkorDB semantic memory to find:
    - Causal chains impacting KPIs
    - Entity relationships
    - Influence networks
    """
    import time
    start_time = time.time()

    try:
        semantic = get_semantic_memory()

        paths = []
        if kpi_name:
            # Find paths impacting the KPI
            paths = semantic.find_causal_paths_for_kpi(
                kpi_name=kpi_name,
                min_confidence=min_confidence
            )
        elif start_entity_id:
            # Traverse from entity
            paths = semantic.traverse_causal_chain(
                start_entity_id=start_entity_id,
                max_depth=max_depth
            )

        latency_ms = (time.time() - start_time) * 1000

        return SemanticPathResponse(
            paths=paths,
            total_paths=len(paths),
            max_depth_searched=max_depth,
            query_latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Semantic path query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Path query failed: {str(e)}")


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/stats")
async def get_memory_stats() -> Dict[str, Any]:
    """
    Get statistics about the memory systems.

    Returns counts and metrics for each memory type.
    """
    try:
        # Placeholder - will be implemented with actual stats queries
        return {
            "episodic": {
                "total_memories": 0,
                "recent_24h": 0
            },
            "procedural": {
                "total_procedures": 0,
                "average_success_rate": 0.0
            },
            "semantic": {
                "total_entities": 0,
                "total_relationships": 0
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
