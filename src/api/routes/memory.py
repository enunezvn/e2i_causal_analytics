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

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import (
    get_user_brands,
    is_cross_brand_admin,
    require_viewer,
    resolve_brand_for_read,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.memory.episodic_memory import (
    E2IEntityReferences,
    count_memories_by_type,
    get_memory_by_id,
    get_recent_memories,
    insert_episodic_memory_with_text,
)
from src.memory.episodic_memory import (
    EpisodicMemoryInput as EpisodicInput,
)
from src.memory.procedural_memory import (
    LearningSignalInput,
    get_memory_statistics,
    get_procedure_by_id,
    record_learning_signal,
    update_procedure_outcome,
)
from src.memory.semantic_memory import get_semantic_memory
from src.rag.retriever import hybrid_search

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/memory",
    tags=["Memory System"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# AUTHORIZATION HELPERS (security review finding #4)
# =============================================================================
#
# REASON-BEFORE-RULES note: ``episodic_memories`` is the agents' tri-memory
# store. It has NO per-end-user ownership column — the tenant boundary that
# already exists in the data is ``brand`` (with ``region``/``agent_name`` as
# secondary scopes). So we scope episodic reads/writes by the caller's brand
# grants, mirroring the per-tenant pattern in ``routes/sentinels.py``, rather
# than inventing a user_id column that the schema doesn't have. PHI/PII
# references (patient_id, hcp_id) are accepted on write but never echoed in
# responses (``EpisodicMemoryResponse`` omits them).


def _is_admin(user: Dict[str, Any]) -> bool:
    """Cross-brand admin — delegates to the canonical ``auth`` helper so the
    admin-detection policy cannot silently drift between modules (codex MED-1)."""
    return is_cross_brand_admin(user)


def _brand_allowed(user: Dict[str, Any], brand: Optional[str]) -> bool:
    """True if the caller may read/write memories for ``brand``."""
    if _is_admin(user):
        return True
    if brand is None:
        # Unbranded memories are not tenant-scoped; allow for any auth'd user.
        return True
    return brand in set(get_user_brands(user))


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
        default=None, description="Memory types to search (all if not specified)"
    )
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HYBRID, description="Retrieval method"
    )
    entities: Optional[List[str]] = Field(
        default=None, description="Entity IDs for graph traversal"
    )
    kpi_name: Optional[str] = Field(
        default=None, description="KPI name for targeted graph traversal"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional filters (brand, region, agent_name)"
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None, description="Custom weights for hybrid retrieval"
    )
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Why did TRx drop in northeast region?",
                "k": 10,
                "retrieval_method": "hybrid",
                "kpi_name": "TRx",
                "filters": {"brand": "Kisqali", "region": "northeast"},
            }
        }
    )


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
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "User asked about TRx trends in northeast region for Kisqali",
                "event_type": "query",
                "session_id": "sess_abc123",
                "agent_name": "orchestrator",
                "brand": "Kisqali",
                "region": "northeast",
            }
        }
    )


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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "procedure_id": "proc_causal_analysis_001",
                "outcome": "success",
                "score": 0.85,
                "feedback_text": "Causal analysis correctly identified HCP engagement drop",
                "agent_name": "feedback_learner",
            }
        }
    )


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
        default="causal_path", description="Type of relationship to follow"
    )
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum traversal depth")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_name": "TRx",
                "relationship_type": "causal_path",
                "max_depth": 3,
                "min_confidence": 0.6,
            }
        }
    )


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


@router.post(
    "/search",
    response_model=MemorySearchResponse,
    summary="Search memory systems",
    operation_id="search_memory",
)
async def search_memory(
    request: MemorySearchRequest,
    user: Dict[str, Any] = Depends(require_viewer),
) -> MemorySearchResponse:
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

    # H1 (cross-tenant PHI reads): scope the search to the caller's brand grant.
    # The episodic RPC treats a missing brand as "all brands", and the FalkorDB
    # causal graph carries no brand property (so its traversal cannot be tenant-
    # scoped) — therefore a non-admin gets (a) the episodic filter pinned to an
    # in-grant brand and (b) the graph-traversal params dropped entirely. A
    # cross-brand admin is unrestricted.
    filters = dict(request.filters or {})
    entities = request.entities
    kpi_name = request.kpi_name
    if not is_cross_brand_admin(user):
        allowed, scoped_brand = resolve_brand_for_read(user, filters.get("brand"))
        if not allowed:
            # Defensive empty — never leak out-of-grant existence or content.
            return MemorySearchResponse(
                query=request.query,
                results=[],
                total_results=0,
                retrieval_method=request.retrieval_method.value,
                search_latency_ms=(time.time() - start_time) * 1000,
            )
        filters["brand"] = scoped_brand
        # Graph traversal is not brand-scopable (no brand on causal-graph nodes);
        # restrict it to cross-brand admins. Non-admins get episodic results only.
        entities = None
        kpi_name = None

    try:
        # Execute hybrid search
        results = await hybrid_search(
            query=request.query,
            k=request.k,
            entities=entities,
            kpi_name=kpi_name,
            filters=filters,
        )

        # Filter by minimum score
        if request.min_score > 0:
            results = [r for r in results if r.score >= request.min_score]

        # Convert to response format
        search_results = [
            MemorySearchResult(
                content=r.content,
                source=r.source.value if hasattr(r.source, "value") else str(r.source),
                source_id=r.source_id,
                score=r.score,
                retrieval_method=r.metadata.get("retrieval_method", "unknown"),
                metadata=r.metadata,
            )
            for r in results
        ]

        latency_ms = (time.time() - start_time) * 1000

        return MemorySearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            retrieval_method=request.retrieval_method.value,
            search_latency_ms=latency_ms,
        )

    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Memory search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Memory search failed") from e


@router.post(
    "/episodic",
    response_model=EpisodicMemoryResponse,
    summary="Create episodic memory",
    operation_id="create_episodic_memory",
)
async def create_episodic_memory(
    memory: EpisodicMemoryInput,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_viewer),
) -> EpisodicMemoryResponse:
    """
    Insert a new episodic memory.

    Creates an embedding and stores in Supabase with pgvector.
    Used to capture significant interactions for future retrieval.

    Authorization (finding #4): a non-admin caller may only write into a
    brand they hold a grant for. PHI/PII references (patient_id, hcp_id) are
    accepted but never echoed in the response.
    """
    # FINDING #4: enforce brand grant BEFORE writing.
    if not _brand_allowed(user, memory.brand):
        raise HTTPException(
            status_code=403,
            detail=f"no grant for brand={memory.brand!r}",
        )
    try:
        # Build E2I entity references
        e2i_refs = E2IEntityReferences(
            patient_id=memory.patient_id,
            hcp_id=memory.hcp_id,
            brand=memory.brand,
            region=memory.region,
        )

        # Build episodic memory input
        episodic_input = EpisodicInput(
            event_type=memory.event_type,
            description=memory.content,
            agent_name=memory.agent_name,
            e2i_refs=e2i_refs,
            raw_content=memory.metadata,
        )

        # Insert memory with auto-generated embedding
        memory_id = await insert_episodic_memory_with_text(
            memory=episodic_input, text_to_embed=memory.content, session_id=memory.session_id
        )

        return EpisodicMemoryResponse(
            id=memory_id,
            content=memory.content,
            event_type=memory.event_type,
            session_id=memory.session_id,
            agent_name=memory.agent_name,
            brand=memory.brand,
            region=memory.region,
            metadata=memory.metadata or {},
        )

    except HTTPException:
        raise
    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Episodic memory insertion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create episodic memory") from e


@router.get(
    "/episodic",
    response_model=List[EpisodicMemoryResponse],
    summary="List recent episodic memories",
    operation_id="list_episodic_memories",
)
async def list_episodic_memories(
    limit: int = 20,
    event_type: Optional[str] = None,
    agent_name: Optional[str] = None,
    brand: Optional[str] = None,
    session_id: Optional[str] = None,
    user: Dict[str, Any] = Depends(require_viewer),
) -> List[EpisodicMemoryResponse]:
    """
    List recent episodic memories ordered by occurrence time (most recent first).

    Supports filtering by event type, agent, brand, and session.

    Authorization (finding #4): a non-admin caller's results are scoped to
    their brand grants. Requesting an out-of-grant ``?brand`` returns an
    empty list (existence not disclosed); omitting ``?brand`` restricts to
    the caller's first granted brand. Admins may query any brand.
    """
    try:
        # FINDING #4: scope the brand filter to the caller's grants.
        if not _is_admin(user):
            allowed_brands = get_user_brands(user)
            if brand is not None:
                if brand not in set(allowed_brands):
                    # Defensive empty list — don't leak other-brand existence.
                    return []
            elif allowed_brands:
                # No brand param: restrict to the caller's grant. The data
                # layer filters by a single brand, so pin to the first grant
                # rather than returning unscoped cross-brand rows.
                brand = allowed_brands[0]
            # No grants + no brand param: nothing tenant-scoped is visible,
            # but unbranded agent memories remain readable (brand stays None).

        rows = await get_recent_memories(
            limit=limit,
            event_types=[event_type] if event_type else None,
            agent_name=agent_name,
            brand=brand,
        )
        if session_id:
            rows = [r for r in rows if r.get("session_id") == session_id]

        return [
            EpisodicMemoryResponse(
                id=r.get("memory_id") or r.get("id", ""),
                content=r.get("description", ""),
                event_type=r.get("event_type", "unknown"),
                session_id=r.get("session_id"),
                agent_name=r.get("agent_name"),
                brand=r.get("brand"),
                region=r.get("region"),
                created_at=r.get("occurred_at", datetime.now(timezone.utc)),
                metadata=r.get("raw_content") or {},
            )
            for r in rows
        ]

    except HTTPException:
        raise
    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to list episodic memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list memories") from e


@router.get(
    "/episodic/{memory_id}",
    response_model=EpisodicMemoryResponse,
    summary="Get episodic memory",
    operation_id="get_episodic_memory",
)
async def get_episodic_memory_endpoint(
    memory_id: str,
    user: Dict[str, Any] = Depends(require_viewer),
) -> EpisodicMemoryResponse:
    """
    Retrieve a specific episodic memory by ID.

    Authorization (finding #4): a non-admin caller can only read a memory
    whose ``brand`` they are granted; otherwise 404 (existence not disclosed).
    """
    try:
        result = await get_memory_by_id(memory_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Episodic memory {memory_id} not found")

        # FINDING #4: brand-membership check. Return 404 (not 403) so the
        # response doesn't confirm a memory exists to an unauthorized caller.
        if not _brand_allowed(user, result.get("brand")):
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
            metadata=result.get("raw_content", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to retrieve episodic memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve memory") from e


@router.post(
    "/procedural/feedback",
    response_model=ProceduralFeedbackResponse,
    summary="Record procedural feedback",
    operation_id="record_procedural_feedback",
)
async def record_procedural_feedback(
    request: ProceduralFeedbackRequest,
    user: Dict[str, Any] = Depends(require_viewer),
) -> ProceduralFeedbackResponse:
    """
    Record outcome feedback for a procedural memory.

    Updates the procedure's success rate based on the outcome.
    Used by Feedback Learner for continuous improvement.
    """
    try:
        # Determine success based on outcome
        success = request.outcome == "success"

        # Update procedure outcome (success/failure counts)
        await update_procedure_outcome(procedure_id=request.procedure_id, success=success)

        # Also record a learning signal for DSPy training
        signal = LearningSignalInput(
            signal_type="rating" if request.score else "thumbs_up" if success else "thumbs_down",
            signal_value=request.score,
            applies_to_type="procedure",
            applies_to_id=request.procedure_id,
            rated_agent=request.agent_name,
            signal_details=(
                {"feedback_text": request.feedback_text} if request.feedback_text else None
            ),
        )
        await record_learning_signal(signal=signal, session_id=request.session_id)

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
            message="Feedback recorded successfully",
        )

    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to record procedural feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to record feedback") from e


@router.get(
    "/semantic/paths",
    response_model=SemanticPathResponse,
    summary="Query semantic paths",
    operation_id="query_semantic_paths",
)
async def query_semantic_paths(
    kpi_name: Optional[str] = None,
    start_entity_id: Optional[str] = None,
    relationship_type: str = "causal_path",
    max_depth: int = 3,
    min_confidence: float = 0.5,
    user: Dict[str, Any] = Depends(require_viewer),
) -> SemanticPathResponse:
    """
    Query semantic graph for causal paths.

    Traverses FalkorDB semantic memory to find:
    - Causal chains impacting KPIs
    - Entity relationships
    - Influence networks
    """
    # H1 (BOLA on the causal graph): the FalkorDB causal graph carries no brand
    # property on its nodes (verified on the live ``e2i_causal`` graph), so a
    # traversal cannot be tenant-scoped. Until ``brand`` is added to the causal-
    # graph nodes, restrict this endpoint to cross-brand admins (fail-closed)
    # rather than leak another tenant's causal chains to a scoped viewer.
    if not is_cross_brand_admin(user):
        raise HTTPException(
            status_code=403,
            detail="cross-brand admin grant required for semantic path traversal",
        )

    import time

    start_time = time.time()

    try:
        semantic = get_semantic_memory()

        paths = []
        if kpi_name:
            # Find paths impacting the KPI
            paths = semantic.find_causal_paths_for_kpi(
                kpi_name=kpi_name, min_confidence=min_confidence
            )
        elif start_entity_id:
            # Traverse from entity
            paths = semantic.traverse_causal_chain(
                start_entity_id=start_entity_id, max_depth=max_depth
            )

        latency_ms = (time.time() - start_time) * 1000

        return SemanticPathResponse(
            paths=paths,
            total_paths=len(paths),
            max_depth_searched=max_depth,
            query_latency_ms=latency_ms,
        )

    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Semantic path query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Path query failed") from e


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================


@router.get(
    "/stats",
    summary="Get memory statistics",
    operation_id="get_memory_stats",
)
async def get_memory_stats(
    user: Dict[str, Any] = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    Get statistics about the memory systems.

    Returns counts and metrics for each memory type:
    - Episodic: total memories, recent 24h count
    - Procedural: total procedures, average success rate
    - Semantic: total entities (nodes), total relationships
    """
    try:
        # Get episodic memory stats
        total_episodic = await count_memories_by_type(days_back=365 * 10)  # All time
        recent_episodic = await count_memories_by_type(days_back=1)  # Last 24h

        # Get procedural memory stats
        proc_stats = await get_memory_statistics(days_back=30)
        proc_totals = proc_stats.get("totals_by_type", {})
        total_procedures = sum(proc_totals.values()) if proc_totals else 0

        # Calculate average success rate from daily breakdown
        daily_stats = proc_stats.get("daily_breakdown", [])
        success_rates = [s.get("success_rate", 0.0) for s in daily_stats if s.get("success_rate")]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0

        # Get semantic memory stats from FalkorDB
        semantic = get_semantic_memory()
        try:
            graph_stats = semantic.get_graph_stats()
            total_entities = graph_stats.get("total_nodes", 0)
            total_relationships = graph_stats.get("total_relationships", 0)
        except Exception as e:
            logger.warning(f"Failed to get semantic graph stats: {e}")
            total_entities = 0
            total_relationships = 0

        return {
            "episodic": {
                "total_memories": total_episodic,
                "recent_24h": recent_episodic,
            },
            "procedural": {
                "total_procedures": total_procedures,
                "average_success_rate": round(avg_success_rate, 3),
            },
            "semantic": {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to get memory stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get memory stats") from e
