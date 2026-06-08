"""
E2I Cognitive Workflow API
==========================

FastAPI endpoints for the cognitive workflow cycle:
- Full cognitive query processing
- Session state management
- Memory-aware agent orchestration

The cognitive workflow integrates:
1. Summarizer - Compress context from working memory
2. Investigator - Retrieve relevant memories via hybrid search
3. Agent - Route to appropriate tier agents
4. Reflector - Store outcomes and record learning signals

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import (
    UserRole,
    has_role,
    is_cross_brand_admin,
    require_viewer,
    resolve_brand_for_read,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.memory.working_memory import get_working_memory
from src.rag.retriever import hybrid_search

logger = logging.getLogger(__name__)


def _caller_id(user: Dict[str, Any]) -> Optional[str]:
    """Return the authenticated principal's stable id.

    Supabase tokens carry the user id as ``sub``; our verified-user dict
    mirrors it as ``id`` (see ``verify_supabase_token``). Prefer ``sub`` to
    stay forward-compatible with raw-claim consumers, fall back to ``id``.
    """
    return user.get("sub") or user.get("id")


def _is_admin(user: Dict[str, Any]) -> bool:
    """True if the caller holds ADMIN role (cross-user access)."""
    return has_role(user, UserRole.ADMIN)


def _assert_session_owner(session: Dict[str, Any], user: Dict[str, Any], session_id: str) -> None:
    """Authorize the caller against a session's owner.

    Sessions are user-private (they persist ``user_id``). A non-owner,
    non-admin caller must not learn the session even exists, so we raise
    404 (not 403) to avoid leaking existence — mirroring the per-tenant
    pattern in ``routes/sentinels.py``.
    """
    if _is_admin(user):
        return
    owner = session.get("user_id")
    if owner != _caller_id(user):
        # 404, not 403: do not confirm the session exists to a non-owner.
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# Orchestrator singleton for agent routing
_orchestrator_instance = None


def get_orchestrator():
    """Get or create OrchestratorAgent singleton.

    Wires the orchestrator with a registry of real Tier 0-5 agents via
    ``create_agent_registry``. Without a registry the dispatcher silently
    falls back to canned mock narratives (``dispatcher._mock_agent_execution``),
    so every "cognitive" API response would be fabricated — that mode is now
    only used when the factory itself fails to instantiate any agents.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        try:
            from src.agents.factory import create_agent_registry
            from src.agents.orchestrator import OrchestratorAgent

            # Exclude orchestrator from its own dispatch registry — it
            # routes to OTHER agents, not to itself, and including it
            # would also cause the factory to instantiate a second
            # OrchestratorAgent during registry construction.
            registry = create_agent_registry(exclude_agents=["orchestrator"])
            _orchestrator_instance = OrchestratorAgent(agent_registry=registry)
            logger.info(
                f"OrchestratorAgent initialized for cognitive workflow with "
                f"{len(registry)} real agents: {sorted(registry.keys())}"
            )
        except Exception as e:
            logger.warning(f"OrchestratorAgent initialization failed: {e}")
            return None
    return _orchestrator_instance


router = APIRouter(
    prefix="/cognitive",
    tags=["Cognitive Workflow"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS & MODELS
# =============================================================================


class QueryType(str, Enum):
    """Types of cognitive queries."""

    CAUSAL = "causal"  # Causal inference questions
    PREDICTION = "prediction"  # ML prediction requests
    OPTIMIZATION = "optimization"  # Resource optimization
    MONITORING = "monitoring"  # Health/drift monitoring
    EXPLANATION = "explanation"  # Explainability requests
    GENERAL = "general"  # General analytics


class SessionState(str, Enum):
    """Session states."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CognitivePhase(str, Enum):
    """Phases of the cognitive workflow."""

    SUMMARIZE = "summarize"
    INVESTIGATE = "investigate"
    EXECUTE = "execute"
    REFLECT = "reflect"
    COMPLETE = "complete"


# -----------------------------------------------------------------------------
# Query Models
# -----------------------------------------------------------------------------


class CognitiveQueryRequest(BaseModel):
    """Request for full cognitive query processing."""

    query: str = Field(..., min_length=1, max_length=5000, description="User query")
    session_id: Optional[str] = Field(None, description="Existing session ID to continue")
    user_id: Optional[str] = Field(None, description="User identifier")
    brand: Optional[str] = Field(
        None, description="Brand context (Kisqali, Fabhalta, Remibrutinib)"
    )
    region: Optional[str] = Field(None, description="Region context")
    query_type: Optional[QueryType] = Field(
        None, description="Type of query (auto-detected if not specified)"
    )
    include_evidence: bool = Field(default=True, description="Include evidence trail in response")
    max_memory_results: int = Field(
        default=10, ge=1, le=50, description="Max memory results to retrieve"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Why did TRx drop 15% in northeast region last quarter?",
                "brand": "Kisqali",
                "region": "northeast",
                "query_type": "causal",
                "include_evidence": True,
            }
        }
    )


class EvidenceItem(BaseModel):
    """Single piece of evidence from memory retrieval."""

    content: str = Field(..., description="Evidence content")
    source: str = Field(..., description="Memory source")
    relevance_score: float = Field(..., description="Relevance score")
    retrieval_method: str = Field(..., description="How it was retrieved")


class CognitiveQueryResponse(BaseModel):
    """Response from cognitive query processing."""

    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    query_type: QueryType = Field(..., description="Detected or specified query type")
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Response confidence. None on degraded/placeholder paths where no "
            "real orchestrator analysis ran — never a fabricated default."
        ),
    )
    agent_used: str = Field(..., description="Primary agent that handled the query")
    evidence: Optional[List[EvidenceItem]] = Field(None, description="Evidence trail")
    phases_completed: List[CognitivePhase] = Field(..., description="Workflow phases completed")
    processing_time_ms: float = Field(..., description="Total processing time in ms")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Session Models
# -----------------------------------------------------------------------------


class SessionContext(BaseModel):
    """Current session context."""

    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = None
    brand: Optional[str] = None
    region: Optional[str] = None
    state: SessionState = Field(default=SessionState.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = Field(default=0)
    current_phase: Optional[CognitivePhase] = None


class SessionMessage(BaseModel):
    """Message in session history."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Full session state response."""

    context: SessionContext = Field(..., description="Session context")
    messages: List[SessionMessage] = Field(..., description="Message history")
    evidence_trail: List[EvidenceItem] = Field(
        default_factory=list, description="Accumulated evidence"
    )
    memory_stats: Dict[str, Any] = Field(default_factory=dict, description="Memory retrieval stats")


class CreateSessionRequest(BaseModel):
    """Request to create a new cognitive session."""

    user_id: Optional[str] = Field(None, description="User identifier")
    brand: Optional[str] = Field(None, description="Brand context")
    region: Optional[str] = Field(None, description="Region context")
    initial_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Initial context"
    )


class CreateSessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str = Field(..., description="Created session ID")
    state: SessionState = Field(default=SessionState.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(..., description="Session expiration time")


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/query",
    response_model=CognitiveQueryResponse,
    summary="Process cognitive query",
    operation_id="process_cognitive_query",
)
async def process_cognitive_query(
    request: CognitiveQueryRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_viewer),
) -> CognitiveQueryResponse:
    """
    Process a query through the full cognitive workflow.

    Workflow phases:
    1. **Summarize**: Compress working memory context
    2. **Investigate**: Retrieve relevant memories via hybrid search
    3. **Execute**: Route to appropriate agent and process
    4. **Reflect**: Store outcomes and record learning signals

    Returns response with evidence trail and confidence score.

    Authorization: the owning ``user_id`` is derived from the authenticated
    token, never from the request body (which is ignored to prevent
    impersonation). Continuing an existing ``session_id`` you do not own is
    rejected with 404.
    """
    import time

    start_time = time.time()

    # FINDING #1: derive owner from the token; the body's user_id is ignored.
    caller_id = _caller_id(user)

    # H1 (cross-tenant PHI reads): scope memory retrieval to the caller's brand
    # grant before it reaches hybrid_search. The episodic RPC treats a missing
    # brand as "all brands", and the FalkorDB causal graph cannot be tenant-
    # scoped (no brand on its nodes), so a non-admin gets the episodic filter
    # pinned to an in-grant brand and the graph-traversal kpi dropped. An
    # *explicit* out-of-grant brand is a deliberate cross-tenant attempt -> 403.
    # A grant-less caller with no brand requested cannot be scoped, so we skip
    # memory retrieval entirely (never fall through to an unscoped all-brand
    # search) while still letting session ownership / orchestration run.
    effective_brand = request.brand
    graph_kpi = _extract_kpi_from_query(request.query)
    skip_memory_retrieval = False
    if not is_cross_brand_admin(user):
        allowed, scoped_brand = resolve_brand_for_read(user, request.brand)
        if not allowed:
            if request.brand is not None:
                raise HTTPException(status_code=403, detail="no grant for the requested brand")
            skip_memory_retrieval = True
        else:
            effective_brand = scoped_brand
        graph_kpi = None

    try:
        working_memory = get_working_memory()
        phases_completed = []

        # Create or retrieve session
        session_id = request.session_id or str(uuid.uuid4())

        if not request.session_id:
            # New session — owned by the authenticated caller.
            await working_memory.create_session(
                session_id=session_id,
                user_id=caller_id,
                initial_context={
                    "brand": request.brand,
                    "region": request.region,
                    **(request.metadata or {}),
                },
            )
        else:
            # FINDING #1 [CRITICAL IDOR]: continuing an existing session must
            # verify ownership BEFORE reading/appending to it. Otherwise any
            # authenticated user could inject messages into — and read the
            # evidence trail of — another user's session.
            existing = await working_memory.get_session(session_id)
            if not existing:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            _assert_session_owner(existing, user, session_id)

        # Phase 1: Summarize - Get compressed context
        phases_completed.append(CognitivePhase.SUMMARIZE)
        await working_memory.get_session(session_id)

        # Store user message
        await working_memory.add_message(
            session_id=session_id,
            role="user",
            content=request.query,
            metadata={"query_type": request.query_type.value if request.query_type else "auto"},
        )

        # Phase 2: Investigate - Retrieve relevant memories
        phases_completed.append(CognitivePhase.INVESTIGATE)
        if skip_memory_retrieval:
            # Grant-less caller: cannot tenant-scope, so retrieve nothing rather
            # than run an unscoped all-brand search (H1).
            memory_results = []
        else:
            memory_results = await hybrid_search(
                query=request.query,
                k=request.max_memory_results,
                kpi_name=graph_kpi,
                filters=_build_filters(effective_brand, request.region),
            )

        # Build evidence items
        evidence = (
            [
                EvidenceItem(
                    content=r.content[:500],  # Truncate for response
                    source=r.source.value if hasattr(r.source, "value") else str(r.source),
                    relevance_score=r.score,
                    retrieval_method=r.metadata.get("retrieval_method", "unknown"),
                )
                for r in memory_results[:5]  # Top 5 for response
            ]
            if request.include_evidence
            else None
        )

        # Store evidence in working memory
        for result in memory_results:
            await working_memory.append_evidence(
                session_id=session_id,
                evidence={
                    "content": result.content,
                    "source": result.source,
                    "score": result.score,
                },
            )

        # Phase 3: Execute - Route to agent
        phases_completed.append(CognitivePhase.EXECUTE)
        query_type = request.query_type or _detect_query_type(request.query)
        agent_name = _route_to_agent(query_type)

        # Execute via OrchestratorAgent (with fallback to placeholder)
        orchestrator = get_orchestrator()
        # FINDING #2: confidence starts UNKNOWN (None). It is only set to a
        # real value when the orchestrator actually produces one. The
        # degraded/placeholder paths leave it None so the UI cannot present a
        # fabricated 0.85 as a genuine model confidence.
        response_confidence: Optional[float] = None

        if orchestrator:
            try:
                orchestrator_result = await orchestrator.run(
                    {
                        "query": request.query,
                        "session_id": session_id,
                        "user_id": caller_id,
                        "user_context": {
                            "brand": request.brand,
                            "region": request.region,
                            "evidence": [e.content for e in evidence] if evidence else [],
                        },
                    }
                )

                response_text = orchestrator_result.get("response_text", "")
                agents_dispatched = orchestrator_result.get("agents_dispatched", [])

                if agents_dispatched:
                    agent_name = agents_dispatched[0]  # Primary agent used
                    # Only a real dispatch yields a real confidence. Pass
                    # through whatever the orchestrator reported (may be None).
                    response_confidence = orchestrator_result.get("response_confidence")
                else:
                    # Issue #251 F1+F2: the orchestrator ran but produced no
                    # dispatch. Surface that as a degraded marker — DO NOT
                    # fall through to the query_type-derived default, which
                    # silently mislabels the response as 'orchestrator' (for
                    # QueryType.GENERAL) or 'health_score' (for
                    # QueryType.MONITORING) and hides the real failure.
                    # Falsifiability-verified 2026-05-16: removing this else
                    # branch trips test_cognitive_degraded_marker.py with
                    # agent_used='orchestrator' (F1 leak).
                    agent_name = "orchestrator_degraded"
                    # FINDING #2: no real analysis ran -> leave confidence None.
                    response_confidence = None

                logger.info(
                    "Orchestrator processed query: agents=%s, confidence=%s",
                    agents_dispatched,
                    response_confidence,
                )

            except Exception as e:
                logger.warning(f"Orchestrator execution failed, using fallback: {e}")
                # Issue #251 F1 (live Docker path): when orchestrator.run()
                # raises, agent_name would otherwise stay at the
                # _route_to_agent(query_type) default — leaking 'orchestrator'
                # for GENERAL queries or 'health_score' for MONITORING. Surface
                # the degraded marker so operators see the real failure mode.
                agent_name = "orchestrator_degraded"
                # FINDING #2: fallback path produced no real analysis.
                response_confidence = None
                response_text = _generate_placeholder_response(
                    query=request.query,
                    query_type=query_type,
                    evidence=evidence,
                    brand=request.brand,
                )
        else:
            # Fallback to placeholder if orchestrator not available.
            # FINDING #2: placeholder path -> confidence stays None.
            response_text = _generate_placeholder_response(
                query=request.query, query_type=query_type, evidence=evidence, brand=request.brand
            )

        # Phase 4: Reflect - Store response and learn
        phases_completed.append(CognitivePhase.REFLECT)
        await working_memory.add_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            metadata={
                "agent_name": agent_name,
                "query_type": query_type.value,
                "evidence_count": len(memory_results),
            },
        )

        phases_completed.append(CognitivePhase.COMPLETE)
        processing_time_ms = (time.time() - start_time) * 1000

        return CognitiveQueryResponse(
            session_id=session_id,
            query=request.query,
            response=response_text,
            query_type=query_type,
            confidence=response_confidence,
            agent_used=agent_name,
            evidence=evidence,
            phases_completed=phases_completed,
            processing_time_ms=processing_time_ms,
            metadata={
                "brand": request.brand,
                "region": request.region,
                "memory_results_count": len(memory_results),
                "orchestrator_used": orchestrator is not None,
            },
        )

    except HTTPException:
        # Authorization / not-found decisions (e.g. cross-user 404) must
        # propagate unchanged — not be swallowed into a generic 500.
        raise
    except Exception as e:
        # FINDING #3: log the detail server-side, return a generic message
        # so internal error text (paths, stack hints) isn't disclosed.
        logger.error(f"Cognitive query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Query processing failed") from e


@router.get(
    "/session/{session_id}",
    response_model=SessionResponse,
    summary="Get cognitive session",
    operation_id="get_cognitive_session",
)
async def get_session(
    session_id: str,
    user: Dict[str, Any] = Depends(require_viewer),
) -> SessionResponse:
    """
    Get the current state of a cognitive session.

    Returns:
    - Session context (user, brand, state)
    - Message history
    - Accumulated evidence trail
    - Memory retrieval statistics

    Authorization: only the session owner (or an admin) may read a session.
    A non-owner gets 404 (existence is not disclosed).
    """
    try:
        working_memory = get_working_memory()

        # Get session context
        session = await working_memory.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # FINDING #1 [CRITICAL IDOR]: enforce ownership before returning the
        # session's messages and evidence_trail to the caller.
        _assert_session_owner(session, user, session_id)

        # Get messages
        messages_data = await working_memory.get_messages(session_id, limit=50)
        messages = [
            SessionMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", datetime.now(timezone.utc)),
                agent_name=m.get("metadata", {}).get("agent_name"),
                metadata=m.get("metadata", {}),
            )
            for m in messages_data
        ]

        # Get evidence trail
        evidence_data = await working_memory.get_evidence_trail(session_id)
        evidence = [
            EvidenceItem(
                content=e.get("content", ""),
                source=e.get("source", "unknown"),
                relevance_score=e.get("score", 0.0),
                retrieval_method=e.get("retrieval_method", "unknown"),
            )
            for e in evidence_data
        ]

        context = SessionContext(
            session_id=session_id,
            user_id=session.get("user_id"),
            brand=session.get("context", {}).get("brand"),
            region=session.get("context", {}).get("region"),
            state=SessionState(session.get("state", "active")),
            created_at=session.get("created_at", datetime.now(timezone.utc)),
            last_activity=session.get("last_activity", datetime.now(timezone.utc)),
            message_count=len(messages),
        )

        return SessionResponse(
            context=context,
            messages=messages,
            evidence_trail=evidence,
            memory_stats={"total_evidence": len(evidence), "message_count": len(messages)},
        )

    except HTTPException:
        raise
    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to retrieve session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve session") from e


@router.post(
    "/session",
    response_model=CreateSessionResponse,
    summary="Create cognitive session",
    operation_id="create_cognitive_session",
)
async def create_session(
    request: CreateSessionRequest,
    user: Dict[str, Any] = Depends(require_viewer),
) -> CreateSessionResponse:
    """
    Create a new cognitive session.

    Sessions maintain:
    - Working memory context
    - Message history
    - Evidence accumulation
    - Learning signals

    Sessions expire after 1 hour of inactivity.

    Authorization: the new session is owned by the authenticated caller; the
    request body's ``user_id`` is ignored to prevent impersonation.
    """
    try:
        working_memory = get_working_memory()
        session_id = str(uuid.uuid4())

        await working_memory.create_session(
            session_id=session_id,
            # FINDING #1: owner = authenticated caller, NOT request.user_id.
            user_id=_caller_id(user),
            initial_context={
                "brand": request.brand,
                "region": request.region,
                **(request.initial_context or {}),
            },
        )

        # Session expires in 1 hour
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        return CreateSessionResponse(
            session_id=session_id, state=SessionState.ACTIVE, expires_at=expires_at
        )

    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create session") from e


@router.delete(
    "/session/{session_id}",
    summary="Delete cognitive session",
    operation_id="delete_cognitive_session",
)
async def delete_session(
    session_id: str,
    user: Dict[str, Any] = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    Delete a cognitive session and its associated data.

    Authorization: only the session owner (or an admin) may delete it. A
    non-owner gets 404 (existence is not disclosed) and the session is left
    untouched.
    """
    try:
        working_memory = get_working_memory()

        # FINDING #1 [CRITICAL IDOR]: verify ownership BEFORE deleting.
        session = await working_memory.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        _assert_session_owner(session, user, session_id)

        await working_memory.delete_session(session_id)

        return {
            "session_id": session_id,
            "deleted": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to delete session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session") from e


@router.get(
    "/sessions",
    summary="List cognitive sessions",
    operation_id="list_cognitive_sessions",
)
async def list_sessions(
    user_id: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200, description="Max sessions to return"),
    user: Dict[str, Any] = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    List active cognitive sessions, ordered by most recent activity.

    Returns a lightweight summary per session (context only, no messages).

    Authorization: a non-admin caller's results are ALWAYS scoped to their
    own ``user_id``; the client-supplied ``?user_id`` is ignored (no
    cross-user enumeration). Admins may pass ``?user_id`` to inspect any user.
    """
    try:
        working_memory = get_working_memory()

        # FINDING #1 [CRITICAL IDOR]: scope to the caller unless admin.
        if _is_admin(user):
            effective_user_id = user_id
        else:
            effective_user_id = _caller_id(user)

        rows = await working_memory.list_sessions(user_id=effective_user_id, limit=limit)

        sessions: List[Dict[str, Any]] = []
        for row in rows:
            sessions.append(
                {
                    "session_id": row.get("session_id"),
                    "user_id": row.get("user_id"),
                    "brand": row.get("active_brand"),
                    "region": row.get("active_region"),
                    "state": row.get("current_phase", "active"),
                    "created_at": row.get("created_at"),
                    "last_activity": row.get("last_activity_at"),
                    "message_count": row.get("message_count", 0),
                }
            )

        return {"sessions": sessions, "total": len(sessions)}

    except HTTPException:
        raise
    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list sessions") from e


@router.get(
    "/status",
    summary="Get cognitive service status",
    operation_id="get_cognitive_status",
)
async def get_cognitive_status(
    user: Dict[str, Any] = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    Get current cognitive service status: configured agents and dependency health.
    """
    try:
        orchestrator = None
        agents: List[str] = []
        try:
            orchestrator = get_orchestrator()
            registry = getattr(orchestrator, "agent_registry", None) or {}
            agents = sorted(registry.keys()) if isinstance(registry, dict) else []
        except Exception:
            pass

        return {
            "status": "healthy" if orchestrator is not None else "degraded",
            "version": "4.2.0",
            "agents": agents,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Failed to get cognitive status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get status") from e


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _detect_query_type(query: str) -> QueryType:
    """Detect the type of query based on content."""
    query_lower = query.lower()

    if any(kw in query_lower for kw in ["why", "cause", "effect", "impact", "drove", "lead to"]):
        return QueryType.CAUSAL
    elif any(kw in query_lower for kw in ["predict", "forecast", "will", "expect", "likely"]):
        return QueryType.PREDICTION
    elif any(kw in query_lower for kw in ["optimize", "improve", "best", "recommend", "allocate"]):
        return QueryType.OPTIMIZATION
    elif any(kw in query_lower for kw in ["drift", "health", "status", "monitor", "alert"]):
        return QueryType.MONITORING
    elif any(kw in query_lower for kw in ["explain", "interpret", "understand", "how does"]):
        return QueryType.EXPLANATION
    else:
        return QueryType.GENERAL


def _route_to_agent(query_type: QueryType) -> str:
    """Route query type to appropriate agent."""
    routing = {
        QueryType.CAUSAL: "causal_impact",
        QueryType.PREDICTION: "prediction_synthesizer",
        QueryType.OPTIMIZATION: "resource_optimizer",
        QueryType.MONITORING: "health_score",
        QueryType.EXPLANATION: "explainer",
        QueryType.GENERAL: "orchestrator",
    }
    return routing.get(query_type, "orchestrator")


def _extract_kpi_from_query(query: str) -> Optional[str]:
    """Extract KPI name from query if present."""
    kpi_keywords = {
        "trx": "TRx",
        "nrx": "NRx",
        "conversion": "conversion_rate",
        "market share": "market_share",
        "adherence": "adherence_rate",
        "churn": "churn_rate",
    }

    query_lower = query.lower()
    for keyword, kpi_name in kpi_keywords.items():
        if keyword in query_lower:
            return kpi_name

    return None


def _build_filters(brand: Optional[str], region: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build filter dictionary from brand and region."""
    filters = {}
    if brand:
        filters["brand"] = brand
    if region:
        filters["region"] = region
    return filters if filters else None


def _generate_placeholder_response(
    query: str, query_type: QueryType, evidence: Optional[List[EvidenceItem]], brand: Optional[str]
) -> str:
    """Generate placeholder response (will be replaced by actual agent processing)."""
    evidence_summary = ""
    if evidence:
        evidence_summary = f"\n\nBased on {len(evidence)} relevant memory items retrieved."

    brand_context = f" for {brand}" if brand else ""

    responses = {
        QueryType.CAUSAL: f"Analyzing causal factors{brand_context}. This query involves causal inference analysis.{evidence_summary}",
        QueryType.PREDICTION: f"Generating predictions{brand_context}. This query involves ML prediction synthesis.{evidence_summary}",
        QueryType.OPTIMIZATION: f"Optimizing resources{brand_context}. This query involves resource optimization.{evidence_summary}",
        QueryType.MONITORING: f"Checking system health{brand_context}. This query involves monitoring and alerts.{evidence_summary}",
        QueryType.EXPLANATION: f"Generating explanation{brand_context}. This query involves model interpretability.{evidence_summary}",
        QueryType.GENERAL: f"Processing query{brand_context}. This is a general analytics query.{evidence_summary}",
    }

    return responses.get(query_type, responses[QueryType.GENERAL])


# =============================================================================
# DSPy-ENHANCED RAG ENDPOINT
# =============================================================================


class CognitiveRAGRequest(BaseModel):
    """Request for DSPy-enhanced cognitive RAG search."""

    query: str = Field(..., min_length=1, max_length=5000, description="Natural language query")
    conversation_id: Optional[str] = Field(
        None, description="Conversation/session ID for context continuity"
    )
    conversation_history: Optional[str] = Field(None, description="Compressed conversation history")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Why did Kisqali adoption increase in the Northeast last quarter?",
                "conversation_id": "session-abc-123",
            }
        }
    )


class CognitiveRAGResponse(BaseModel):
    """Response from DSPy-enhanced cognitive RAG search."""

    response: str = Field(..., description="Synthesized natural language response")
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list, description="Evidence pieces gathered"
    )
    hop_count: int = Field(default=0, description="Number of retrieval hops performed")
    visualization_config: Dict[str, Any] = Field(
        default_factory=dict, description="Chart configuration if applicable"
    )
    routed_agents: List[str] = Field(
        default_factory=list, description="Agents recommended for further processing"
    )
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    intent: str = Field(default="", description="Detected query intent")
    rewritten_query: str = Field(default="", description="DSPy-optimized query rewrite")
    dspy_signals: List[Dict[str, Any]] = Field(
        default_factory=list, description="Training signals for optimization"
    )
    worth_remembering: bool = Field(
        default=False, description="Whether this exchange should be stored in long-term memory"
    )
    latency_ms: float = Field(..., description="Total processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if processing failed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "Kisqali adoption increased 15% in the Northeast due to increased oncologist engagement and successful speaker programs.",
                "evidence": [{"content": "Northeast TRx up 15%...", "source": "agent_activities"}],
                "hop_count": 2,
                "entities": ["Kisqali", "Northeast"],
                "intent": "causal",
                "latency_ms": 16500.0,
            }
        }
    )


@router.post(
    "/rag",
    response_model=CognitiveRAGResponse,
    summary="Cognitive RAG search",
    operation_id="cognitive_rag_search",
)
async def cognitive_rag_search(
    request: CognitiveRAGRequest,
    user: Dict[str, Any] = Depends(require_viewer),
) -> CognitiveRAGResponse:
    """
    Execute DSPy-enhanced 4-phase cognitive RAG workflow.

    This endpoint provides LLM-powered multi-hop reasoning through:

    **Phase 1 - Summarizer**:
    - Query rewriting for optimal retrieval
    - Entity extraction (brands, regions, KPIs)
    - Intent classification

    **Phase 2 - Investigator**:
    - Multi-hop evidence gathering
    - Adaptive retrieval across episodic, semantic, procedural memory
    - Evidence relevance scoring

    **Phase 3 - Agent**:
    - Evidence synthesis into coherent response
    - Agent routing for specialized processing
    - Visualization configuration

    **Phase 4 - Reflector**:
    - Memory worthiness assessment
    - Fact extraction for long-term storage
    - DSPy training signal collection

    **Performance**: This 4-phase path runs sequential per-hop LLM decisions and
    per-item LLM relevance scoring; typical latency is ~15-18s for a multi-hop
    query (NOT sub-second). See docs/reports/rag-hybrid-search-audit-20260608.md.

    **Requirements**: ANTHROPIC_API_KEY environment variable must be set.

    Returns:
        CognitiveRAGResponse with synthesized response, evidence trail,
        and optimization signals.
    """
    try:
        from src.rag.causal_rag import CausalRAG

        # Create CausalRAG instance
        rag = CausalRAG()

        # Execute cognitive search
        result = await rag.cognitive_search(
            query=request.query,
            conversation_id=request.conversation_id,
            conversation_history=request.conversation_history,
        )

        return CognitiveRAGResponse(
            response=result.get("response", ""),
            evidence=result.get("evidence", []),
            hop_count=result.get("hop_count", 0),
            visualization_config=result.get("visualization_config", {}),
            routed_agents=result.get("routed_agents", []),
            entities=result.get("entities", []),
            intent=result.get("intent", ""),
            rewritten_query=result.get("rewritten_query", request.query),
            dspy_signals=result.get("dspy_signals", []),
            worth_remembering=result.get("worth_remembering", False),
            latency_ms=result.get("latency_ms", 0.0),
            error=result.get("error"),
        )

    except ImportError as e:
        # FINDING #3: log the offending module server-side; the client just
        # needs to know the feature is unavailable, not the import internals.
        logger.error(f"Cognitive RAG import error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, detail="Cognitive RAG dependencies not available"
        ) from e
    except ValueError as e:
        # FINDING #3: ValueError here is typically a misconfiguration (e.g. a
        # missing API key); the message can leak config details, so keep the
        # client response generic and log the specifics server-side.
        logger.error(f"Cognitive RAG configuration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=400, detail="Cognitive RAG request could not be processed"
        ) from e
    except Exception as e:
        # FINDING #3: generic client message; full detail logged server-side.
        logger.error(f"Cognitive RAG search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Cognitive RAG search failed") from e
