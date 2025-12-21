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

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import logging
import uuid

from src.memory.working_memory import get_working_memory
from src.rag.retriever import hybrid_search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cognitive", tags=["Cognitive Workflow"])


# =============================================================================
# ENUMS & MODELS
# =============================================================================

class QueryType(str, Enum):
    """Types of cognitive queries."""
    CAUSAL = "causal"           # Causal inference questions
    PREDICTION = "prediction"   # ML prediction requests
    OPTIMIZATION = "optimization"  # Resource optimization
    MONITORING = "monitoring"   # Health/drift monitoring
    EXPLANATION = "explanation" # Explainability requests
    GENERAL = "general"         # General analytics


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
    brand: Optional[str] = Field(None, description="Brand context (Kisqali, Fabhalta, Remibrutinib)")
    region: Optional[str] = Field(None, description="Region context")
    query_type: Optional[QueryType] = Field(None, description="Type of query (auto-detected if not specified)")
    include_evidence: bool = Field(default=True, description="Include evidence trail in response")
    max_memory_results: int = Field(default=10, ge=1, le=50, description="Max memory results to retrieve")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "Why did TRx drop 15% in northeast region last quarter?",
            "brand": "Kisqali",
            "region": "northeast",
            "query_type": "causal",
            "include_evidence": True
        }
    })


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
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
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
    evidence_trail: List[EvidenceItem] = Field(default_factory=list, description="Accumulated evidence")
    memory_stats: Dict[str, Any] = Field(default_factory=dict, description="Memory retrieval stats")


class CreateSessionRequest(BaseModel):
    """Request to create a new cognitive session."""
    user_id: Optional[str] = Field(None, description="User identifier")
    brand: Optional[str] = Field(None, description="Brand context")
    region: Optional[str] = Field(None, description="Region context")
    initial_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Initial context")


class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: str = Field(..., description="Created session ID")
    state: SessionState = Field(default=SessionState.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(..., description="Session expiration time")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/query", response_model=CognitiveQueryResponse)
async def process_cognitive_query(
    request: CognitiveQueryRequest,
    background_tasks: BackgroundTasks
) -> CognitiveQueryResponse:
    """
    Process a query through the full cognitive workflow.

    Workflow phases:
    1. **Summarize**: Compress working memory context
    2. **Investigate**: Retrieve relevant memories via hybrid search
    3. **Execute**: Route to appropriate agent and process
    4. **Reflect**: Store outcomes and record learning signals

    Returns response with evidence trail and confidence score.
    """
    import time
    start_time = time.time()

    try:
        working_memory = get_working_memory()
        phases_completed = []

        # Create or retrieve session
        session_id = request.session_id or str(uuid.uuid4())

        if not request.session_id:
            # New session
            await working_memory.create_session(
                session_id=session_id,
                user_id=request.user_id,
                initial_context={
                    "brand": request.brand,
                    "region": request.region,
                    **(request.metadata or {})
                }
            )

        # Phase 1: Summarize - Get compressed context
        phases_completed.append(CognitivePhase.SUMMARIZE)
        context = await working_memory.get_session(session_id)

        # Store user message
        await working_memory.add_message(
            session_id=session_id,
            role="user",
            content=request.query,
            metadata={"query_type": request.query_type.value if request.query_type else "auto"}
        )

        # Phase 2: Investigate - Retrieve relevant memories
        phases_completed.append(CognitivePhase.INVESTIGATE)
        memory_results = await hybrid_search(
            query=request.query,
            k=request.max_memory_results,
            kpi_name=_extract_kpi_from_query(request.query),
            filters=_build_filters(request.brand, request.region)
        )

        # Build evidence items
        evidence = [
            EvidenceItem(
                content=r.content[:500],  # Truncate for response
                source=r.source.value if hasattr(r.source, 'value') else str(r.source),
                relevance_score=r.score,
                retrieval_method=r.metadata.get("retrieval_method", "unknown")
            )
            for r in memory_results[:5]  # Top 5 for response
        ] if request.include_evidence else None

        # Store evidence in working memory
        for result in memory_results:
            await working_memory.append_evidence(
                session_id=session_id,
                evidence={
                    "content": result.content,
                    "source": result.source,
                    "score": result.score
                }
            )

        # Phase 3: Execute - Route to agent
        phases_completed.append(CognitivePhase.EXECUTE)
        query_type = request.query_type or _detect_query_type(request.query)
        agent_name = _route_to_agent(query_type)

        # For now, generate a placeholder response
        # In production, this would route to the actual agent
        response_text = _generate_placeholder_response(
            query=request.query,
            query_type=query_type,
            evidence=evidence,
            brand=request.brand
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
                "evidence_count": len(memory_results)
            }
        )

        phases_completed.append(CognitivePhase.COMPLETE)
        processing_time_ms = (time.time() - start_time) * 1000

        return CognitiveQueryResponse(
            session_id=session_id,
            query=request.query,
            response=response_text,
            query_type=query_type,
            confidence=0.85,  # Placeholder
            agent_used=agent_name,
            evidence=evidence,
            phases_completed=phases_completed,
            processing_time_ms=processing_time_ms,
            metadata={
                "brand": request.brand,
                "region": request.region,
                "memory_results_count": len(memory_results)
            }
        )

    except Exception as e:
        logger.error(f"Cognitive query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """
    Get the current state of a cognitive session.

    Returns:
    - Session context (user, brand, state)
    - Message history
    - Accumulated evidence trail
    - Memory retrieval statistics
    """
    try:
        working_memory = get_working_memory()

        # Get session context
        session = await working_memory.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Get messages
        messages_data = await working_memory.get_messages(session_id, limit=50)
        messages = [
            SessionMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", datetime.now(timezone.utc)),
                agent_name=m.get("metadata", {}).get("agent_name"),
                metadata=m.get("metadata", {})
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
                retrieval_method=e.get("retrieval_method", "unknown")
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
            message_count=len(messages)
        )

        return SessionResponse(
            context=context,
            messages=messages,
            evidence_trail=evidence,
            memory_stats={
                "total_evidence": len(evidence),
                "message_count": len(messages)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")


@router.post("/session", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """
    Create a new cognitive session.

    Sessions maintain:
    - Working memory context
    - Message history
    - Evidence accumulation
    - Learning signals

    Sessions expire after 1 hour of inactivity.
    """
    try:
        working_memory = get_working_memory()
        session_id = str(uuid.uuid4())

        await working_memory.create_session(
            session_id=session_id,
            user_id=request.user_id,
            initial_context={
                "brand": request.brand,
                "region": request.region,
                **(request.initial_context or {})
            }
        )

        # Session expires in 1 hour
        expires_at = datetime.now(timezone.utc).replace(hour=datetime.now(timezone.utc).hour + 1)

        return CreateSessionResponse(
            session_id=session_id,
            state=SessionState.ACTIVE,
            expires_at=expires_at
        )

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.delete("/session/{session_id}")
async def delete_session(session_id: str) -> Dict[str, Any]:
    """
    Delete a cognitive session and its associated data.
    """
    try:
        working_memory = get_working_memory()
        await working_memory.delete_session(session_id)

        return {
            "session_id": session_id,
            "deleted": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


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
        QueryType.GENERAL: "orchestrator"
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
        "churn": "churn_rate"
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
    query: str,
    query_type: QueryType,
    evidence: Optional[List[EvidenceItem]],
    brand: Optional[str]
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
        QueryType.GENERAL: f"Processing query{brand_context}. This is a general analytics query.{evidence_summary}"
    }

    return responses.get(query_type, responses[QueryType.GENERAL])


# =============================================================================
# DSPy-ENHANCED RAG ENDPOINT
# =============================================================================

class CognitiveRAGRequest(BaseModel):
    """Request for DSPy-enhanced cognitive RAG search."""
    query: str = Field(..., min_length=1, max_length=5000, description="Natural language query")
    conversation_id: Optional[str] = Field(None, description="Conversation/session ID for context continuity")
    conversation_history: Optional[str] = Field(None, description="Compressed conversation history")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "Why did Kisqali adoption increase in the Northeast last quarter?",
            "conversation_id": "session-abc-123"
        }
    })


class CognitiveRAGResponse(BaseModel):
    """Response from DSPy-enhanced cognitive RAG search."""
    response: str = Field(..., description="Synthesized natural language response")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Evidence pieces gathered")
    hop_count: int = Field(default=0, description="Number of retrieval hops performed")
    visualization_config: Dict[str, Any] = Field(default_factory=dict, description="Chart configuration if applicable")
    routed_agents: List[str] = Field(default_factory=list, description="Agents recommended for further processing")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    intent: str = Field(default="", description="Detected query intent")
    rewritten_query: str = Field(default="", description="DSPy-optimized query rewrite")
    dspy_signals: List[Dict[str, Any]] = Field(default_factory=list, description="Training signals for optimization")
    worth_remembering: bool = Field(default=False, description="Whether this exchange should be stored in long-term memory")
    latency_ms: float = Field(..., description="Total processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if processing failed")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "response": "Kisqali adoption increased 15% in the Northeast due to increased oncologist engagement and successful speaker programs.",
            "evidence": [{"content": "Northeast TRx up 15%...", "source": "agent_activities"}],
            "hop_count": 2,
            "entities": ["Kisqali", "Northeast"],
            "intent": "causal",
            "latency_ms": 1250.5
        }
    })


@router.post("/rag", response_model=CognitiveRAGResponse)
async def cognitive_rag_search(request: CognitiveRAGRequest) -> CognitiveRAGResponse:
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

    **Performance**: Typical latency < 2s for simple queries.

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
            conversation_history=request.conversation_history
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
            error=result.get("error")
        )

    except ImportError as e:
        logger.error(f"Cognitive RAG import error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cognitive RAG dependencies not available: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Cognitive RAG configuration error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Cognitive RAG search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cognitive RAG search failed: {str(e)[:200]}"
        )
