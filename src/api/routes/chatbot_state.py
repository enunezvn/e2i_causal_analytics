"""
E2I Chatbot State Definition for LangGraph.

Defines the TypedDict state schema for the E2I chatbot agent workflow.
Follows the pattern from E2IAgentState but adds:
- Streaming support fields
- User/session context
- Tool results tracking
- RAG context storage
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ChatbotState(TypedDict, total=False):
    """
    State for the E2I Chatbot LangGraph workflow.

    This state tracks the full conversation lifecycle including:
    - User context (user_id, session_id)
    - Messages (with LangGraph's add operator for accumulation)
    - Tool execution results
    - RAG retrieval context
    - Streaming chunks for SSE responses

    Attributes:
        user_id: User UUID from authentication
        session_id: Conversation session ID (format: user_id~uuid)
        request_id: Unique request identifier for tracing
        query: Original user query text
        messages: Conversation message history (accumulates via operator.add)
        intent: Classified intent from the query
        brand_context: Brand filter for this conversation
        region_context: Region filter for this conversation
        tool_results: Results from tool executions
        rag_context: Retrieved documents from RAG
        rag_sources: Source identifiers for retrieved documents
        response_text: Final generated response text
        response_chunks: Streaming chunks for SSE (when streaming)
        streaming_complete: Whether streaming is complete
        conversation_title: Auto-generated or user-set title
        agent_name: Current agent handling the request
        agent_tier: Tier of the current agent
        error: Error message if any
        metadata: Additional metadata
    """

    # User and session context
    user_id: str
    session_id: str
    request_id: str
    trace_id: Optional[str]  # Opik trace ID for observability

    # Query processing
    query: str
    intent: Optional[str]
    intent_confidence: Optional[float]  # DSPy classification confidence (0.0-1.0)
    intent_reasoning: Optional[str]  # DSPy classification reasoning
    intent_classification_method: Optional[str]  # "dspy" or "hardcoded"

    # Agent routing
    routed_agent: Optional[str]  # Primary agent routed for this query
    secondary_agents: List[str]  # Additional agents that may assist
    routing_confidence: Optional[float]  # Routing confidence (0.0-1.0)
    routing_rationale: Optional[str]  # Explanation for routing decision

    # Orchestrator integration (multi-agent dispatch)
    orchestrator_used: bool  # Whether orchestrator processed this query
    agents_dispatched: List[str]  # Agents that orchestrator dispatched to
    response_confidence: Optional[float]  # Orchestrator response confidence (0.0-1.0)

    # E2I context filters
    brand_context: Optional[str]
    region_context: Optional[str]

    # Message history (accumulates via operator.add)
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Tool execution
    tool_results: List[Dict[str, Any]]

    # RAG context
    rag_context: List[Dict[str, Any]]
    rag_sources: List[str]
    rag_rewritten_query: Optional[str]  # DSPy cognitive RAG rewritten query
    rag_retrieval_method: Optional[str]  # "cognitive" or "basic"

    # Response generation
    response_text: str
    response_chunks: List[str]
    streaming_complete: bool

    # Evidence synthesis (Phase 6 DSPy)
    confidence_statement: Optional[str]  # DSPy synthesis confidence statement
    evidence_citations: List[str]  # Source IDs cited in response
    synthesis_method: Optional[str]  # "dspy" or "hardcoded"
    follow_up_suggestions: List[str]  # Suggested follow-up questions

    # Conversation metadata
    conversation_title: Optional[str]
    agent_name: Optional[str]
    agent_tier: Optional[int]

    # Error handling
    error: Optional[str]

    # Extensible metadata
    metadata: Dict[str, Any]

    # Execution progress tracking (Phase 4: Stream Execution Progress)
    # These fields are synced to frontend via copilotkit_emit_state()
    agent_status: Optional[str]  # 'idle' | 'processing' | 'waiting' | 'complete' | 'error'
    progress_percent: int  # 0-100 progress indicator
    progress_steps: List[str]  # Completed step descriptions
    tools_executing: List[str]  # Currently executing tool names
    current_node: Optional[str]  # Current LangGraph node name


class AgentRequest(TypedDict):
    """
    Input schema for chatbot API requests.

    Used by both streaming and non-streaming endpoints.
    """

    query: str
    user_id: str
    request_id: str
    session_id: Optional[str]
    brand_context: Optional[str]
    region_context: Optional[str]


class StreamChunk(TypedDict):
    """
    Schema for streaming SSE response chunks.

    Types:
    - text: Response text chunk
    - session_id: New session ID (sent once at start)
    - conversation_title: Auto-generated title (sent once)
    - tool_call: Tool invocation notification
    - tool_result: Tool execution result
    - done: Stream completion signal
    - error: Error message
    """

    type: str  # text, session_id, conversation_title, tool_call, tool_result, done, error
    data: str


# Intent classification categories
class IntentType:
    """Known intent types for E2I chatbot queries."""

    KPI_QUERY = "kpi_query"
    CAUSAL_ANALYSIS = "causal_analysis"
    AGENT_STATUS = "agent_status"
    RECOMMENDATION = "recommendation"
    SEARCH = "search"
    GENERAL = "general"
    GREETING = "greeting"
    HELP = "help"
    MULTI_FACETED = "multi_faceted"  # Complex queries requiring Tool Composer
    COHORT_DEFINITION = "cohort_definition"  # Patient/HCP cohort construction queries


# Default initial state
def create_initial_state(
    user_id: str,
    query: str,
    request_id: str,
    session_id: Optional[str] = None,
    brand_context: Optional[str] = None,
    region_context: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> ChatbotState:
    """
    Create an initial ChatbotState for a new request.

    Args:
        user_id: User UUID
        query: User's query text
        request_id: Request identifier for tracing
        session_id: Optional existing session ID (new one generated if None)
        brand_context: Optional brand filter
        region_context: Optional region filter
        trace_id: Optional Opik trace ID for observability

    Returns:
        Initialized ChatbotState
    """
    import uuid

    # Generate session_id if not provided
    if not session_id:
        session_id = f"{user_id}~{uuid.uuid4()}"

    return ChatbotState(
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        trace_id=trace_id,
        query=query,
        intent=None,
        intent_confidence=None,
        intent_reasoning=None,
        intent_classification_method=None,
        routed_agent=None,
        secondary_agents=[],
        routing_confidence=None,
        routing_rationale=None,
        orchestrator_used=False,
        agents_dispatched=[],
        response_confidence=None,
        brand_context=brand_context,
        region_context=region_context,
        messages=[],
        tool_results=[],
        rag_context=[],
        rag_sources=[],
        rag_rewritten_query=None,
        rag_retrieval_method=None,
        response_text="",
        response_chunks=[],
        streaming_complete=False,
        confidence_statement=None,
        evidence_citations=[],
        synthesis_method=None,
        follow_up_suggestions=[],
        conversation_title=None,
        agent_name=None,
        agent_tier=None,
        error=None,
        metadata={},
        # Execution progress tracking (Phase 4)
        agent_status="idle",
        progress_percent=0,
        progress_steps=[],
        tools_executing=[],
        current_node=None,
    )
