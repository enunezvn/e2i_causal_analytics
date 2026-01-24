"""
E2I Chatbot LangGraph Workflow.

Builds a LangGraph agent workflow for the E2I chatbot with:
- Multi-node processing pipeline
- E2I-specific tool integration
- RAG retrieval for context
- Intent classification
- Orchestrator integration for multi-agent dispatch
- Streaming response support

Workflow (with orchestrator integration):
    init → load_context → classify_intent → retrieve_rag → orchestrator → generate → [tools] → finalize

The orchestrator node routes complex queries (causal_analysis, kpi_query,
recommendation, search, multi_faceted) through the 20-agent orchestrator
for specialized processing. Simple queries (greeting, help, agent_status,
general) skip orchestrator and generate responses directly.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.api.routes.chatbot_state import ChatbotState, IntentType, create_initial_state
from src.api.routes.chatbot_tools import E2I_CHATBOT_TOOLS
from src.utils.llm_factory import get_chat_llm, get_llm_provider
from src.memory.services.factories import get_async_supabase_client
from src.memory.working_memory import get_langgraph_checkpointer
from src.rag.retriever import hybrid_search
from src.repositories.chatbot_conversation import (
    ChatbotConversationRepository,
    get_chatbot_conversation_repository,
)
from src.repositories.chatbot_message import (
    ChatbotMessageRepository,
    get_chatbot_message_repository,
)
from src.memory.episodic_memory import (
    EpisodicMemoryInput,
    E2IEntityReferences,
    insert_episodic_memory_with_text,
)
from src.api.routes.chatbot_tracer import (
    get_chatbot_tracer,
    ChatbotTraceContext,
    CHATBOT_OPIK_TRACING_ENABLED,
)
from src.api.routes.chatbot_dspy import (
    classify_intent_dspy,
    CHATBOT_DSPY_INTENT_ENABLED,
    cognitive_rag_retrieve,
    CHATBOT_COGNITIVE_RAG_ENABLED,
    synthesize_response_dspy,
    CHATBOT_DSPY_SYNTHESIS_ENABLED,
    get_chatbot_signal_collector,
    route_agent_hardcoded,
)
from src.mlops.mlflow_connector import MLflowConnector
from src.api.routes.cognitive import get_orchestrator

# MLflow metrics feature flag
CHATBOT_MLFLOW_METRICS_ENABLED = os.getenv("CHATBOT_MLFLOW_METRICS", "true").lower() == "true"

# MLflow experiment name
CHATBOT_MLFLOW_EXPERIMENT = "chatbot_interactions"

# Phase 7: Training signal collection feature flag
CHATBOT_SIGNAL_COLLECTION_ENABLED = os.getenv("CHATBOT_SIGNAL_COLLECTION", "true").lower() == "true"

# Orchestrator integration feature flag - routes complex queries through orchestrator/tool_composer
CHATBOT_ORCHESTRATOR_ENABLED = os.getenv("CHATBOT_ORCHESTRATOR", "true").lower() == "true"

# Intents that should be routed through the orchestrator for agent dispatch
ORCHESTRATOR_ROUTED_INTENTS = {
    IntentType.CAUSAL_ANALYSIS,
    IntentType.KPI_QUERY,
    IntentType.RECOMMENDATION,
    IntentType.SEARCH,
    IntentType.MULTI_FACETED,
    IntentType.COHORT_DEFINITION,  # Route cohort queries through orchestrator to CohortConstructor
}

logger = logging.getLogger(__name__)

# Context variable for active trace context (accessible by nodes)
import contextvars
_active_trace_context: contextvars.ContextVar[Optional[ChatbotTraceContext]] = contextvars.ContextVar(
    "chatbot_trace_context", default=None
)


# =============================================================================
# PROGRESS TRACKING (Phase 4: Stream Execution Progress)
# =============================================================================

# Progress definitions for each workflow node
# Format: (percent, step_description, status)
WORKFLOW_PROGRESS = {
    "init": (5, "Initializing conversation...", "processing"),
    "load_context": (15, "Loading conversation context...", "processing"),
    "classify_intent": (25, "Analyzing intent...", "processing"),
    "retrieve_rag": (40, "Retrieving relevant knowledge...", "processing"),
    "orchestrator": (60, "Routing to specialized agents...", "processing"),
    "generate": (80, "Generating response...", "processing"),
    "tools": (70, "Executing tools...", "processing"),
    "finalize": (100, "Complete", "complete"),
}


def get_progress_update(
    node_name: str,
    current_steps: List[str] = None,
    tools_executing: List[str] = None,
    custom_step: str = None,
) -> Dict[str, Any]:
    """
    Generate progress state update for a workflow node.

    Args:
        node_name: Current node in the workflow
        current_steps: Accumulated step descriptions
        tools_executing: List of currently executing tool names
        custom_step: Custom step description (overrides default)

    Returns:
        Dict with progress fields to merge into state
    """
    progress_info = WORKFLOW_PROGRESS.get(node_name, (50, f"Processing {node_name}...", "processing"))
    percent, default_step, status = progress_info

    # Build step description
    step_description = custom_step if custom_step else default_step

    # Accumulate steps (add new step if not already present)
    steps = list(current_steps) if current_steps else []
    if step_description and step_description not in steps:
        steps.append(step_description)

    return {
        "agent_status": status,
        "progress_percent": percent,
        "progress_steps": steps,
        "tools_executing": tools_executing or [],
        "current_node": node_name,
    }


# MLflow connector singleton (lazy initialization)
_mlflow_connector: Optional[MLflowConnector] = None
_mlflow_experiment_id: Optional[str] = None


def _get_mlflow_connector() -> Optional[MLflowConnector]:
    """Get the MLflow connector singleton."""
    global _mlflow_connector
    if not CHATBOT_MLFLOW_METRICS_ENABLED:
        return None
    if _mlflow_connector is None:
        _mlflow_connector = MLflowConnector()
    return _mlflow_connector


async def _get_or_create_chatbot_experiment() -> Optional[str]:
    """Get or create the chatbot interactions MLflow experiment."""
    global _mlflow_experiment_id
    if not CHATBOT_MLFLOW_METRICS_ENABLED:
        return None
    if _mlflow_experiment_id is not None:
        return _mlflow_experiment_id

    mlflow_conn = _get_mlflow_connector()
    if mlflow_conn is None:
        return None

    try:
        _mlflow_experiment_id = await mlflow_conn.get_or_create_experiment(
            name=CHATBOT_MLFLOW_EXPERIMENT,
            tags={
                "platform": "e2i_causal_analytics",
                "component": "chatbot",
                "framework": "langgraph",
            },
        )
        logger.info(f"MLflow experiment '{CHATBOT_MLFLOW_EXPERIMENT}' ready: {_mlflow_experiment_id}")
        return _mlflow_experiment_id
    except Exception as e:
        logger.warning(f"Failed to get/create MLflow experiment: {e}")
        return None


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

E2I_CHATBOT_SYSTEM_PROMPT = """You are the E2I Analytics Assistant, an intelligent AI specialized in pharmaceutical commercial analytics for Novartis brands.

## Your Expertise

You help users with:
1. **KPI Analysis** - TRx, NRx, market share, conversion rates, patient starts
2. **Causal Analysis** - Understanding WHY metrics change and what drives performance
3. **Agent System** - Information about the 18-agent tiered architecture
4. **Recommendations** - AI-powered suggestions for HCP targeting and market access
5. **Insights Search** - Finding trends, causal paths, and historical patterns

## Brands You Support

- **Kisqali** - HR+/HER2- breast cancer
- **Fabhalta** - PNH (Paroxysmal Nocturnal Hemoglobinuria)
- **Remibrutinib** - CSU (Chronic Spontaneous Urticaria)

## Guidelines

1. **Data-Driven Responses**: Always use the available tools to fetch real data before answering
2. **Source Attribution**: Cite the data source when presenting metrics or insights
3. **Commercial Focus**: This is pharmaceutical COMMERCIAL analytics (sales, marketing, market access) - NOT clinical or medical advice
4. **Causal Clarity**: When discussing causation, be clear about confidence levels and methodology
5. **Actionable Insights**: Provide recommendations that can drive business decisions

## Tool Usage

Use tools proactively:
- Use `e2i_data_query_tool` for KPI metrics, causal chains, agent analyses
- Use `causal_analysis_tool` for understanding metric drivers
- Use `document_retrieval_tool` for searching the knowledge base
- Use `conversation_memory_tool` to reference previous conversation context

## Response Format

- Be concise but comprehensive
- Use bullet points for lists
- Highlight key metrics with **bold**
- Include confidence scores for causal claims
- Suggest follow-up questions when appropriate

{context}
"""


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================


def _matches_pattern(query_lower: str, patterns: list[str]) -> bool:
    """
    Check if query matches any pattern using word boundaries.

    For multi-word patterns (e.g., "good morning"), checks substring match.
    For single-word patterns, checks word boundary match to avoid false positives
    like "hi" matching "this".
    """
    import re

    for pattern in patterns:
        if " " in pattern:
            # Multi-word pattern: use substring match
            if pattern in query_lower:
                return True
        else:
            # Single-word pattern: use word boundary regex
            if re.search(rf"\b{re.escape(pattern)}\b", query_lower):
                return True
    return False


def _is_multi_faceted_query(query: str) -> bool:
    """
    Detect if query needs Tool Composer for multi-faceted processing.

    Multi-faceted queries require aggregating results from multiple agents
    or analyzing multiple aspects of the same question. Examples:
    - "Compare TRx trends across all brands and explain the causal factors"
    - "Show me the health score and recommendations for Kisqali"

    Args:
        query: User's query text

    Returns:
        True if query is multi-faceted and should use Tool Composer
    """
    import re

    query_lower = query.lower()

    # Score different facets of complexity
    facets = {
        # Query contains comparative/conjunction keywords suggesting multiple questions
        "conjunction_keywords": any(
            w in query_lower for w in ["compare", "trends", "explain", "also", "and then", "both"]
        ),
        # Query mentions multiple KPIs
        "multiple_kpis": len(
            re.findall(r"(trx|nrx|market share|conversion|volume|patient starts)", query_lower)
        )
        > 1,
        # Query spans cross-agent capabilities
        "cross_agent": any(
            w in query_lower for w in ["drift", "health", "causal", "experiment", "prediction"]
        ),
        # Query mentions multiple brands
        "multiple_brands": len(
            re.findall(r"(kisqali|fabhalta|remibrutinib|all brands)", query_lower)
        )
        > 1,
        # Query asks for both analysis AND recommendations
        "analysis_and_recommendation": ("why" in query_lower or "what caused" in query_lower)
        and any(w in query_lower for w in ["recommend", "suggest", "should"]),
    }

    # Need at least 2 facets to qualify as multi-faceted
    return sum(facets.values()) >= 2


def classify_intent(query: str) -> str:
    """
    Classify the user's query intent.

    Args:
        query: User's query text

    Returns:
        Intent classification string
    """
    query_lower = query.lower()

    # Greeting patterns (check first for quick responses)
    if _matches_pattern(query_lower, ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return IntentType.GREETING

    # Help patterns (check early for guidance requests)
    if _matches_pattern(query_lower, ["help", "what can you", "how do i", "guide me"]):
        return IntentType.HELP

    # Multi-faceted check - BEFORE individual intents
    # Complex queries needing Tool Composer should be detected early
    if _is_multi_faceted_query(query):
        return IntentType.MULTI_FACETED

    # KPI patterns
    if _matches_pattern(query_lower, ["kpi", "trx", "nrx", "market share", "conversion", "metric", "volume"]):
        return IntentType.KPI_QUERY

    # Causal patterns (including past tense variations)
    if _matches_pattern(query_lower, ["why", "cause", "caused", "impact", "effect", "driver", "causal", "because"]):
        return IntentType.CAUSAL_ANALYSIS

    # Agent patterns
    if _matches_pattern(query_lower, ["agent", "tier", "orchestrator", "status", "system"]):
        return IntentType.AGENT_STATUS

    # Recommendation patterns
    if _matches_pattern(query_lower, ["recommend", "suggest", "improve", "optimize", "strategy"]):
        return IntentType.RECOMMENDATION

    # Search patterns
    if _matches_pattern(query_lower, ["search", "find", "look for", "show me", "trend"]):
        return IntentType.SEARCH

    # Cohort definition patterns (patient/HCP cohort construction)
    if _matches_pattern(query_lower, [
        "cohort", "build a cohort", "create a cohort", "define a cohort",
        "patient population", "patient set", "eligibility", "eligible patient",
        "inclusion criteria", "exclusion criteria", "high-value hcp", "high value hcp",
        "hcp cohort", "physician cohort", "filter patient"
    ]):
        return IntentType.COHORT_DEFINITION

    return IntentType.GENERAL


# =============================================================================
# WORKFLOW NODES
# =============================================================================


async def init_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Initialize the conversation state.

    Sets up initial context, creates conversation if new session,
    and prepares for processing.
    """
    session_id = state.get("session_id")
    user_id = state.get("user_id")
    query = state.get("query", "")
    brand = state.get("brand_context")
    region = state.get("region_context")

    logger.info(f"Init node: session={session_id}, query={query[:50]}...")

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    is_new_conversation = False

    # Wrap node execution with tracing if available
    async def _execute_init():
        nonlocal is_new_conversation
        # Try to create or verify conversation exists in database
        try:
            client = await get_async_supabase_client()
            if client:
                conv_repo = get_chatbot_conversation_repository(client)

                # Check if conversation exists
                existing_conv = await conv_repo.get_by_session_id(session_id)

                if not existing_conv:
                    # Create new conversation
                    await conv_repo.create_conversation(
                        user_id=user_id,
                        session_id=session_id,
                        brand_context=brand,
                        region_context=region,
                        query_type="general",
                    )
                    is_new_conversation = True
                    logger.debug(f"Created new conversation: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to create/verify conversation: {e}")

    # Execute with tracing if available
    if trace_ctx:
        async with trace_ctx.trace_node("init") as node_span:
            await _execute_init()
            node_span.log_init(
                is_new_conversation=is_new_conversation,
                session_id=session_id,
                user_id=user_id,
            )
    else:
        await _execute_init()

    # Add human message to state
    human_msg = HumanMessage(content=query)

    # Include progress update in return
    progress = get_progress_update("init")

    return {
        "messages": [human_msg],
        "metadata": {
            "init_timestamp": str(datetime.now(timezone.utc)),
            "is_new_conversation": is_new_conversation,
        },
        **progress,
    }


async def load_context_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Load conversation context from previous messages.

    Retrieves conversation history from database if session exists.
    """
    session_id = state.get("session_id")
    brand = state.get("brand_context")
    region = state.get("region_context")

    logger.debug(f"Loading context for session: {session_id}")

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    # Load previous messages from database
    previous_messages: List[Dict[str, Any]] = []
    conversation_title = None

    async def _execute_load_context():
        nonlocal previous_messages, conversation_title, brand, region
        try:
            client = await get_async_supabase_client()
            if client:
                msg_repo = get_chatbot_message_repository(client)
                conv_repo = get_chatbot_conversation_repository(client)

                # Get conversation metadata
                conv = await conv_repo.get_by_session_id(session_id)
                if conv:
                    conversation_title = conv.get("title")
                    # Use conversation's brand/region if not provided in request
                    if not brand:
                        brand = conv.get("brand_context")
                    if not region:
                        region = conv.get("region_context")

                # Get recent messages for context
                recent_msgs = await msg_repo.get_recent_messages(session_id, count=10)
                previous_messages = recent_msgs
                logger.debug(f"Loaded {len(previous_messages)} previous messages")

        except Exception as e:
            logger.warning(f"Failed to load conversation context: {e}")

    # Execute with tracing if available
    if trace_ctx:
        async with trace_ctx.trace_node("load_context") as node_span:
            await _execute_load_context()
            node_span.log_context_load(
                previous_message_count=len(previous_messages),
                brand_context=brand,
                region_context=region,
            )
    else:
        await _execute_load_context()

    # Convert previous messages to LangChain message format
    history_messages = []
    for msg in previous_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    # Include progress update
    progress = get_progress_update(
        "load_context",
        current_steps=state.get("progress_steps", []),
    )

    return {
        "messages": history_messages,  # Prepend history
        "conversation_title": conversation_title,
        "brand_context": brand,
        "region_context": region,
        "metadata": {
            **state.get("metadata", {}),
            "context_loaded": True,
            "previous_message_count": len(previous_messages),
        },
        **progress,
    }


async def retrieve_rag_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Retrieve relevant context using cognitive RAG with DSPy query rewriting.

    Phase 5 DSPy integration: Uses cognitive RAG pipeline with:
    - Query rewriting for E2I domain optimization
    - Evidence relevance scoring
    - Training signal collection for optimization

    Falls back to basic hybrid_search if cognitive RAG is disabled or fails.
    """
    query = state.get("query", "")
    brand = state.get("brand_context") or ""
    intent = state.get("intent") or ""
    messages = state.get("messages", [])

    logger.debug(f"RAG retrieval: query={query[:50]}..., intent={intent}, cognitive_enabled={CHATBOT_COGNITIVE_RAG_ENABLED}")

    # Build conversation context from recent messages (last 3)
    conversation_context = ""
    if messages:
        recent_msgs = messages[-3:] if len(messages) > 3 else messages
        context_parts = []
        for msg in recent_msgs:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = getattr(msg, "content", str(msg))
            if content:
                context_parts.append(f"{role}: {content[:100]}...")
        conversation_context = "\n".join(context_parts)

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    rag_context = []
    rag_sources = []
    relevance_scores = []
    kpi_name = None
    error = None
    retrieval_method = "basic"
    rewritten_query = None
    search_keywords = []
    graph_entities = []
    avg_relevance = 0.0

    async def _execute_cognitive_rag():
        """Execute cognitive RAG pipeline with DSPy query rewriting."""
        nonlocal rag_context, rag_sources, relevance_scores, retrieval_method
        nonlocal rewritten_query, search_keywords, graph_entities, avg_relevance, error

        try:
            result = await cognitive_rag_retrieve(
                query=query,
                conversation_context=conversation_context,
                brand_context=brand,
                intent=str(intent) if intent else "",
                k=5,
                enable_multi_hop=False,  # Start with single-hop
                collect_signal=True,  # Collect training signals
            )

            # Extract results from cognitive RAG
            rewritten_query = result.rewritten_query
            search_keywords = result.search_keywords
            graph_entities = result.graph_entities
            retrieval_method = result.retrieval_method
            avg_relevance = result.avg_relevance_score

            # Convert evidence to rag_context format
            rag_context = [
                {
                    "source_id": e.get("source_id", "unknown"),
                    "content": e.get("content", "")[:500],
                    "score": e.get("relevance_score", e.get("score", 0.0)),
                    "source": e.get("source", "rag"),
                    "key_insight": e.get("key_insight", ""),
                }
                for e in result.evidence
            ]

            rag_sources = [e.get("source_id", "unknown") for e in result.evidence]
            relevance_scores = [e.get("relevance_score", e.get("score", 0.0)) for e in result.evidence]

            logger.debug(
                f"Cognitive RAG complete: method={retrieval_method}, results={len(rag_context)}, "
                f"avg_relevance={avg_relevance:.2f}, keywords={search_keywords[:3]}"
            )

        except Exception as e:
            logger.warning(f"Cognitive RAG failed, falling back to basic: {e}")
            # Don't set error - we'll fall back to basic RAG
            raise  # Re-raise to trigger fallback

    async def _execute_basic_rag():
        """Execute basic hybrid RAG (fallback)."""
        nonlocal rag_context, rag_sources, relevance_scores, kpi_name, error, retrieval_method

        try:
            retrieval_method = "basic"

            # Adjust retrieval based on intent
            if intent == IntentType.KPI_QUERY:
                # Extract KPI name from query if present
                for kpi in ["trx", "nrx", "market share", "conversion"]:
                    if kpi in query.lower():
                        kpi_name = kpi
                        break

            results = await hybrid_search(
                query=query,
                k=5,
                kpi_name=kpi_name,
                filters={"brand": brand} if brand else None,
            )

            rag_context = [
                {
                    "source_id": r.source_id,
                    "content": r.content[:500],  # Truncate for context window
                    "score": r.score,
                    "source": r.source,
                }
                for r in results
            ]

            rag_sources = [r.source_id for r in results]
            relevance_scores = [r.score for r in results]

        except Exception as e:
            logger.error(f"Basic RAG retrieval failed: {e}")
            error = f"RAG retrieval error: {str(e)}"

    async def _execute_rag():
        """Execute RAG with cognitive pipeline or fallback."""
        if CHATBOT_COGNITIVE_RAG_ENABLED:
            try:
                await _execute_cognitive_rag()
            except Exception:
                # Cognitive RAG failed, fall back to basic
                await _execute_basic_rag()
        else:
            await _execute_basic_rag()

    # Execute with tracing if available
    if trace_ctx:
        async with trace_ctx.trace_node("retrieve_rag") as node_span:
            await _execute_rag()
            # Log enhanced metrics for cognitive RAG
            node_span.log_rag_retrieval(
                result_count=len(rag_context),
                relevance_scores=relevance_scores,
                kpi_filter=kpi_name,
                brand_filter=brand,
                retrieval_method=retrieval_method,
            )
            # Log cognitive RAG specific metrics
            if retrieval_method == "cognitive":
                node_span.log_metadata({
                    "rewritten_query": rewritten_query[:100] if rewritten_query else None,
                    "search_keywords": search_keywords[:5],
                    "graph_entities": graph_entities[:5],
                    "avg_relevance_score": avg_relevance,
                })
    else:
        await _execute_rag()

    # Build progress update
    custom_step = f"Retrieved {len(rag_context)} documents" if rag_context else "Retrieving knowledge..."
    progress = get_progress_update(
        "retrieve_rag",
        current_steps=state.get("progress_steps", []),
        custom_step=custom_step,
    )

    if error:
        # Set error status on failure
        progress["agent_status"] = "error"
        return {
            "rag_context": [],
            "rag_sources": [],
            "error": error,
            **progress,
        }

    return {
        "rag_context": rag_context,
        "rag_sources": rag_sources,
        "rag_rewritten_query": rewritten_query,
        "rag_retrieval_method": retrieval_method,
        **progress,
    }


async def classify_intent_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Classify the user's query intent and route to specialized agent.

    Phase 3 DSPy integration: Uses ChatbotIntentClassifier for ML-based
    classification with confidence scores and training signal collection.

    Also routes to a specialized agent based on query keywords and intent.
    """
    query = state.get("query", "")
    brand_context = state.get("brand_context", "") or ""
    messages = state.get("messages", [])

    # Build conversation context from recent messages (last 3)
    conversation_context = ""
    if messages:
        recent_msgs = messages[-3:] if len(messages) > 3 else messages
        context_parts = []
        for msg in recent_msgs:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = getattr(msg, "content", str(msg))
            if content:
                context_parts.append(f"{role}: {content[:100]}...")
        conversation_context = "\n".join(context_parts)

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    intent = None
    confidence = 0.0
    reasoning = ""
    classification_method = "unknown"

    # Agent routing variables
    routed_agent = "chatbot"
    secondary_agents = []
    routing_confidence = 0.0
    routing_rationale = ""

    async def _execute_classify():
        nonlocal intent, confidence, reasoning, classification_method
        nonlocal routed_agent, secondary_agents, routing_confidence, routing_rationale

        # Use DSPy classifier with fallback to hardcoded
        intent, confidence, reasoning, classification_method = await classify_intent_dspy(
            query=query,
            conversation_context=conversation_context,
            brand_context=brand_context,
            collect_signal=True,  # Collect training signals for optimization
        )
        logger.debug(
            f"Intent classified: {intent} (confidence={confidence:.2f}, method={classification_method})"
        )

        # Route to specialized agent based on query and intent
        routed_agent, secondary_agents, routing_confidence, routing_rationale = route_agent_hardcoded(
            query=query,
            intent=intent,
        )
        logger.debug(
            f"Routed to agent: {routed_agent} (confidence={routing_confidence:.2f}, "
            f"secondary={secondary_agents}, rationale={routing_rationale})"
        )

    # Execute with tracing if available
    if trace_ctx:
        async with trace_ctx.trace_node("classify_intent") as node_span:
            await _execute_classify()
            node_span.log_intent_classification(
                intent=intent,
                confidence=confidence,
                classification_method=classification_method,
            )
    else:
        await _execute_classify()

    # Build progress update with custom step showing detected intent
    custom_step = f"Intent: {intent}" if intent else None
    progress = get_progress_update(
        "classify_intent",
        current_steps=state.get("progress_steps", []),
        custom_step=custom_step,
    )

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "intent_reasoning": reasoning,
        "intent_classification_method": classification_method,
        "routed_agent": routed_agent,
        "secondary_agents": secondary_agents,
        "routing_confidence": routing_confidence,
        "routing_rationale": routing_rationale,
        **progress,
    }


def _build_partial_failure_warning(
    failed_agents: List[str],
    failure_details: List[Dict[str, Any]],
) -> str:
    """Build a user-friendly warning message about partial failures.

    Phase 3: Partial failure handling enhancement.

    Args:
        failed_agents: List of agent names that failed
        failure_details: List of failure detail dicts with agent_name and error

    Returns:
        Warning message string, or empty string if no failures
    """
    if not failed_agents:
        return ""

    # Build failure summary
    failure_lines = []
    for detail in failure_details:
        agent = detail.get("agent_name", "Unknown agent")
        error = detail.get("error", "Unknown error")
        # Simplify error messages for users
        if "timed out" in error.lower():
            failure_lines.append(f"• {agent}: Operation took too long")
        elif "connection" in error.lower() or "unavailable" in error.lower():
            failure_lines.append(f"• {agent}: Service temporarily unavailable")
        else:
            # Truncate long error messages
            error_summary = error[:100] + "..." if len(error) > 100 else error
            failure_lines.append(f"• {agent}: {error_summary}")

    # Build warning message
    if len(failed_agents) == 1:
        warning = (
            "⚠️ **Note**: One analysis component did not complete successfully:\n"
            + "\n".join(failure_lines)
            + "\n\nThe results above are based on the analyses that completed successfully."
        )
    else:
        warning = (
            f"⚠️ **Note**: {len(failed_agents)} analysis components did not complete successfully:\n"
            + "\n".join(failure_lines)
            + "\n\nThe results above are based on the analyses that completed successfully."
        )

    return warning


async def orchestrator_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Route complex queries through the orchestrator for agent dispatch.

    The orchestrator provides:
    - Multi-agent coordination (20 agents across 6 tiers)
    - Tool composer for multi-faceted queries
    - Specialized agent dispatch (causal_impact, experiment_designer, etc.)

    For simple intents (greeting, help, agent_status), this node is skipped
    and the chatbot generates responses directly via generate_node.
    """
    intent = state.get("intent")
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    rag_context = state.get("rag_context", [])
    brand_context = state.get("brand_context")
    region_context = state.get("region_context")

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    # Initialize result with pass-through behavior (no changes)
    result = {}
    orchestrator_used = False
    agents_dispatched = []
    response_text = ""
    response_confidence = 0.0

    async def _execute_orchestrator():
        """Execute query through orchestrator with partial failure handling.

        Phase 3 Enhancement: Handles partial failures gracefully.
        If some agents succeed and others fail, returns the successful results
        with warnings about the failed agents.
        """
        nonlocal result, orchestrator_used, agents_dispatched, response_text, response_confidence

        orchestrator = get_orchestrator()
        if not orchestrator:
            logger.warning("Orchestrator not available, falling back to direct generation")
            return

        try:
            # Build evidence context from RAG results
            evidence = [ctx.get("content", "") for ctx in rag_context[:5]] if rag_context else []

            # Call orchestrator
            orchestrator_result = await orchestrator.run({
                "query": query,
                "session_id": session_id,
                "user_id": user_id,
                "user_context": {
                    "brand": brand_context,
                    "region": region_context,
                    "evidence": evidence,
                },
            })

            orchestrator_used = True
            response_text = orchestrator_result.get("response_text", "")
            response_confidence = orchestrator_result.get("response_confidence", 0.0)
            agents_dispatched = orchestrator_result.get("agents_dispatched", [])

            # Phase 3: Extract partial failure information
            has_partial_failure = orchestrator_result.get("has_partial_failure", False)
            successful_agents = orchestrator_result.get("successful_agents", [])
            failed_agents = orchestrator_result.get("failed_agents", [])
            failure_details = orchestrator_result.get("failure_details", [])
            status = orchestrator_result.get("status", "completed")

            # Determine primary agent from successful agents
            primary_agent = "orchestrator"
            if successful_agents:
                primary_agent = successful_agents[0]
            elif agents_dispatched:
                primary_agent = agents_dispatched[0]

            # Log based on success/failure status
            if has_partial_failure:
                logger.warning(
                    f"Orchestrator partial success: succeeded={successful_agents}, "
                    f"failed={failed_agents}, confidence={response_confidence:.2f}"
                )
                # Add partial failure warning to response text
                failure_warning = _build_partial_failure_warning(failed_agents, failure_details)
                if response_text and failure_warning:
                    response_text = f"{response_text}\n\n{failure_warning}"
            elif status == "failed":
                logger.error(
                    f"Orchestrator complete failure: all agents failed - {failed_agents}"
                )
            else:
                logger.info(
                    f"Orchestrator processed query: agents={agents_dispatched}, "
                    f"confidence={response_confidence:.2f}"
                )

            # Create response message and update state
            if response_text or has_partial_failure:
                response_msg = AIMessage(content=response_text or "Analysis could not be completed.")
                result = {
                    "messages": [response_msg],
                    "response_text": response_text,
                    "agent_name": primary_agent,
                    "routed_agent": primary_agent,
                    "agents_dispatched": agents_dispatched,
                    "orchestrator_used": True,
                    "response_confidence": response_confidence,
                    "metadata": {
                        **(state.get("metadata") or {}),
                        "orchestrator_used": True,
                        "agents_dispatched": agents_dispatched,
                        "orchestrator_confidence": response_confidence,
                        # Phase 3: Add partial failure metadata
                        "has_partial_failure": has_partial_failure,
                        "successful_agents": successful_agents,
                        "failed_agents": failed_agents,
                        "failure_details": failure_details,
                        "orchestrator_status": status,
                    },
                }

        except Exception as e:
            logger.error(f"Orchestrator execution failed with exception: {e}")
            # Fall through to generate_node by returning empty result

    # Only route through orchestrator for complex intents
    should_use_orchestrator = (
        CHATBOT_ORCHESTRATOR_ENABLED
        and intent in ORCHESTRATOR_ROUTED_INTENTS
    )

    if should_use_orchestrator:
        if trace_ctx:
            async with trace_ctx.trace_node("orchestrator") as node_span:
                await _execute_orchestrator()
                node_span.log_metadata({
                    "orchestrator_used": orchestrator_used,
                    "agents_dispatched": agents_dispatched,
                    "response_confidence": response_confidence,
                    "response_length": len(response_text),
                })
        else:
            await _execute_orchestrator()
    else:
        logger.debug(
            f"Skipping orchestrator: intent={intent} not in ORCHESTRATOR_ROUTED_INTENTS "
            f"or feature disabled (enabled={CHATBOT_ORCHESTRATOR_ENABLED})"
        )

    # Build progress update with dispatched agents info
    custom_step = None
    if orchestrator_used and agents_dispatched:
        custom_step = f"Dispatched to {', '.join(agents_dispatched[:2])}{'...' if len(agents_dispatched) > 2 else ''}"
    progress = get_progress_update(
        "orchestrator",
        current_steps=state.get("progress_steps", []),
        custom_step=custom_step,
    )

    # Merge progress into result
    result.update(progress)
    return result


async def generate_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Generate response using Claude with tools or DSPy synthesis.

    Phase 6 DSPy integration: When we have good RAG evidence, uses DSPy
    synthesis for structured responses with confidence statements and citations.
    Falls back to LLM with tools for complex queries requiring tool execution.

    NOTE: If orchestrator_node already produced a response (orchestrator_used=True),
    this node returns the existing response without re-generating.
    """
    # Check if orchestrator already handled this query
    if state.get("orchestrator_used") and state.get("response_text"):
        logger.debug("Skipping generate_node: orchestrator already produced response")
        # Pass through - orchestrator already set progress, just move forward
        progress = get_progress_update(
            "generate",
            current_steps=state.get("progress_steps", []),
            custom_step="Using orchestrator response...",
        )
        return {**progress}

    messages = list(state.get("messages", []))
    rag_context = state.get("rag_context", [])
    brand = state.get("brand_context")
    region = state.get("region_context")
    intent = state.get("intent", "general")
    query = state.get("query", "")
    # Get routed agent from classify_intent_node (defaults to "chatbot" if not set)
    routed_agent = state.get("routed_agent") or "chatbot"

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    # Build conversation context from recent messages
    conversation_context = ""
    if messages:
        recent_msgs = messages[-3:] if len(messages) > 3 else messages
        context_parts = []
        for msg in recent_msgs:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = getattr(msg, "content", str(msg))
            if content:
                context_parts.append(f"{role}: {content[:100]}...")
        conversation_context = "\n".join(context_parts)

    # Track generation metrics
    result = None
    provider = None
    model_name = None
    tool_calls_count = 0
    input_tokens = 0
    output_tokens = 0
    is_fallback = False
    synthesis_used = False
    synthesis_method = None
    confidence_statement = None
    evidence_citations = []
    follow_up_suggestions = []

    # Determine if we should use DSPy synthesis
    # Use synthesis when:
    # 1. Synthesis is enabled
    # 2. We have RAG evidence with good relevance scores
    # 3. Intent doesn't require tool execution (greetings, help don't need RAG)
    should_use_synthesis = (
        CHATBOT_DSPY_SYNTHESIS_ENABLED
        and rag_context
        and len(rag_context) >= 1
        and intent not in (IntentType.GREETING, IntentType.HELP, IntentType.AGENT_STATUS)
    )

    # Calculate average evidence relevance
    avg_evidence_score = 0.0
    if rag_context:
        scores = [ctx.get("relevance_score", ctx.get("score", 0.0)) for ctx in rag_context]
        avg_evidence_score = sum(scores) / len(scores) if scores else 0.0

    # Only use synthesis if evidence quality is sufficient
    if should_use_synthesis and avg_evidence_score < 0.3:
        should_use_synthesis = False
        logger.debug(f"Skipping synthesis: avg evidence score too low ({avg_evidence_score:.2f})")

    async def _execute_synthesis():
        """Execute DSPy evidence synthesis."""
        nonlocal result, synthesis_used, synthesis_method, confidence_statement
        nonlocal evidence_citations, follow_up_suggestions

        synthesis_result = await synthesize_response_dspy(
            query=query,
            intent=str(intent) if intent else "general",
            evidence=rag_context,
            brand_context=brand or "",
            conversation_context=conversation_context,
            collect_signal=True,
        )

        synthesis_used = True
        synthesis_method = synthesis_result.synthesis_method
        confidence_statement = synthesis_result.confidence_statement
        evidence_citations = synthesis_result.evidence_citations
        follow_up_suggestions = synthesis_result.follow_up_suggestions

        # Create AI message with synthesized response
        response_msg = AIMessage(content=synthesis_result.response)

        result = {
            "messages": [response_msg],
            "agent_name": routed_agent,
            "agent_tier": 1,
            "response_text": synthesis_result.response,
            "confidence_statement": confidence_statement,
            "evidence_citations": evidence_citations,
            "synthesis_method": synthesis_method,
            "follow_up_suggestions": follow_up_suggestions,
        }

        logger.info(
            f"DSPy synthesis complete: {len(synthesis_result.response)} chars, "
            f"confidence={synthesis_result.confidence_level}, "
            f"citations={len(evidence_citations)}, method={synthesis_method}"
        )

    async def _execute_llm_generate():
        """Execute LLM generation with tools."""
        nonlocal result, provider, model_name, tool_calls_count, input_tokens, output_tokens, is_fallback

        # Build context string for system prompt
        context_parts = []
        if brand:
            context_parts.append(f"Current Brand Filter: {brand}")
        if region:
            context_parts.append(f"Current Region Filter: {region}")
        if rag_context:
            context_parts.append("\n## Retrieved Context\n")
            for ctx in rag_context[:3]:  # Top 3 for context window
                context_parts.append(f"- [{ctx.get('source', 'unknown')}] {ctx.get('content', '')[:200]}...")

        context_str = "\n".join(context_parts) if context_parts else ""

        # Create system message
        system_prompt = E2I_CHATBOT_SYSTEM_PROMPT.format(context=context_str)
        system_msg = SystemMessage(content=system_prompt)

        # Prepare messages for LLM
        llm_messages = [system_msg] + messages

        try:
            llm = get_chat_llm(
                model_tier="standard",
                max_tokens=1024,
                temperature=0.3,
            )
            provider = get_llm_provider()
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            logger.info(f"Using {provider} LLM for chatbot")

            # Bind tools to the model
            llm_with_tools = llm.bind_tools(E2I_CHATBOT_TOOLS)

            # Generate response
            response = await llm_with_tools.ainvoke(llm_messages)

            # Extract token usage if available (varies by provider)
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
            elif hasattr(response, "response_metadata"):
                meta = response.response_metadata
                if "usage" in meta:
                    input_tokens = meta["usage"].get("input_tokens", 0)
                    output_tokens = meta["usage"].get("output_tokens", 0)

            # Count tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls_count = len(response.tool_calls)

            result = {
                "messages": [response],
                "agent_name": routed_agent,
                "agent_tier": 1,
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            is_fallback = True
            result = _generate_fallback_response(state)

    async def _execute_generate():
        """Execute generation (synthesis or LLM)."""
        if should_use_synthesis:
            try:
                await _execute_synthesis()
            except Exception as e:
                logger.warning(f"Synthesis failed, falling back to LLM: {e}")
                await _execute_llm_generate()
        else:
            await _execute_llm_generate()

    # Execute with tracing if available
    if trace_ctx:
        async with trace_ctx.trace_node("generate") as node_span:
            await _execute_generate()
            node_span.log_generate(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name or "dspy_synthesis" if synthesis_used else "unknown",
                provider=provider or "dspy" if synthesis_used else "unknown",
                tool_calls_count=tool_calls_count,
                temperature=0.3,
                is_fallback=is_fallback,
            )
            # Log synthesis-specific metrics
            if synthesis_used:
                node_span.log_metadata({
                    "synthesis_method": synthesis_method,
                    "confidence_statement": confidence_statement[:100] if confidence_statement else None,
                    "citations_count": len(evidence_citations),
                    "follow_up_count": len(follow_up_suggestions),
                    "avg_evidence_score": avg_evidence_score,
                })
    else:
        await _execute_generate()

    # Add progress tracking to result
    progress = get_progress_update(
        "generate",
        current_steps=state.get("progress_steps", []),
        custom_step="Generated response",
    )
    if result:
        result.update(progress)
    else:
        result = progress

    return result


def _generate_fallback_response(state: ChatbotState) -> Dict[str, Any]:
    """Generate a fallback response when LLM is unavailable."""
    intent = state.get("intent", IntentType.GENERAL)
    query = state.get("query", "")
    routed_agent = state.get("routed_agent") or "chatbot"

    responses = {
        IntentType.GREETING: "Hello! I'm the E2I Analytics Assistant. I can help you with KPI analysis, causal inference, and insights for pharmaceutical brands. What would you like to know?",
        IntentType.HELP: "I can help you with:\n\n1. **KPI Analysis** - Get metrics like TRx, NRx, market share\n2. **Causal Analysis** - Understand why metrics change\n3. **Agent Status** - Check the 18-agent system\n4. **Recommendations** - Get AI-powered suggestions\n5. **Search** - Find trends and insights\n\nTry asking about a specific brand (Kisqali, Fabhalta, Remibrutinib) or metric!",
        IntentType.KPI_QUERY: "I can help with KPI analysis! I track metrics like TRx volume, NRx volume, market share, conversion rates, HCP reach, and patient starts. Which brand and metric would you like to explore?",
        IntentType.CAUSAL_ANALYSIS: "For causal analysis, I use DoWhy/EconML to identify factors driving your metrics. Tell me which KPI you'd like to analyze and I'll find the key drivers.",
        IntentType.AGENT_STATUS: "The E2I platform uses an 18-agent tiered architecture across 6 tiers. I can show you agent status and recent analyses. Which agent or tier interests you?",
        IntentType.RECOMMENDATION: "I can provide AI-powered recommendations for HCP targeting, patient journey optimization, and market access strategies. Which brand would you like recommendations for?",
        IntentType.SEARCH: "I can search the E2I knowledge base for insights, causal paths, and trends. What would you like me to find?",
    }

    response_text = responses.get(
        intent,
        "I'm the E2I Analytics Assistant. I can help with KPI analysis, causal inference, and pharmaceutical commercial analytics. What would you like to explore?",
    )

    return {
        "messages": [AIMessage(content=response_text)],
        "response_text": response_text,
        "agent_name": routed_agent,
    }


async def tool_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Execute tools based on LLM tool calls.

    This is handled by LangGraph's ToolNode.
    """
    # ToolNode handles this automatically
    pass


# =============================================================================
# EPISODIC MEMORY BRIDGE
# =============================================================================

# Significance thresholds for episodic memory storage
SIGNIFICANCE_THRESHOLD = 0.6  # Minimum score to save to episodic memory

# Intents that indicate significant interactions worth preserving
SIGNIFICANT_INTENTS = {
    IntentType.CAUSAL_ANALYSIS,
    IntentType.KPI_QUERY,
    IntentType.RECOMMENDATION,
    IntentType.SEARCH,
    IntentType.MULTI_FACETED,  # Complex queries via Tool Composer are highly significant
}

# Tool names that indicate actionable queries
SIGNIFICANT_TOOLS = {
    "e2i_data_query_tool",
    "causal_analysis_tool",
    "agent_routing_tool",
    "document_retrieval_tool",
    "orchestrator_tool",
    "tool_composer_tool",
}

# Event type mapping from intent/tool to episodic memory event types
INTENT_TO_EVENT_TYPE = {
    IntentType.CAUSAL_ANALYSIS: "causal_discovery",
    IntentType.KPI_QUERY: "user_query",
    IntentType.RECOMMENDATION: "agent_action",
    IntentType.SEARCH: "user_query",
    IntentType.AGENT_STATUS: "agent_action",
    IntentType.MULTI_FACETED: "multi_agent_analysis",  # Tool Composer orchestrated queries
}


def _calculate_significance_score(state: ChatbotState) -> float:
    """
    Calculate a significance score for the interaction.

    Significant interactions are worth preserving in episodic memory
    for cross-session learning and platform knowledge building.

    Scoring factors:
    - Tool usage (+0.3 each, max 0.6)
    - Significant intent (+0.25)
    - Brand/KPI specificity (+0.15)
    - RAG context retrieved (+0.1)
    - Response length indicator (+0.1)

    Returns:
        Float between 0.0 and 1.0
    """
    score = 0.0

    # Factor 1: Tool usage (strong signal of actionable query)
    tool_results = state.get("tool_results", [])
    if tool_results:
        # Each tool call adds significance, capped at 0.6
        tool_score = min(len(tool_results) * 0.3, 0.6)
        score += tool_score

    # Factor 2: Significant intent type
    intent = state.get("intent")
    if intent in SIGNIFICANT_INTENTS:
        score += 0.25

    # Factor 3: Brand/KPI specificity (indicates focused query)
    brand_context = state.get("brand_context")
    metadata = state.get("metadata", {})
    kpi_mentioned = metadata.get("kpi_name") or metadata.get("kpis")
    if brand_context or kpi_mentioned:
        score += 0.15

    # Factor 4: RAG context was retrieved (indicates knowledge-seeking)
    rag_context = state.get("rag_context", [])
    if rag_context:
        score += 0.1

    # Factor 5: Response substance (longer responses often contain insights)
    response_text = state.get("response_text", "")
    if len(response_text) > 500:
        score += 0.1

    return min(score, 1.0)


async def _save_to_episodic_memory(
    state: ChatbotState,
    response_text: str,
    tool_calls: List[Dict[str, Any]],
    significance_score: float,
) -> Optional[str]:
    """
    Save a significant chatbot interaction to episodic memory.

    This bridges chatbot conversations to the platform's long-term
    episodic memory system, enabling cross-session learning.

    Args:
        state: Current chatbot state
        response_text: Final assistant response
        tool_calls: List of tools that were called
        significance_score: Calculated significance (0.0-1.0)

    Returns:
        Memory ID if saved, None if skipped or failed
    """
    try:
        query = state.get("query", "")
        session_id = state.get("session_id")
        intent = state.get("intent")
        brand_context = state.get("brand_context")
        region_context = state.get("region_context")
        tool_results = state.get("tool_results", [])

        # Determine event type based on intent or tools used
        event_type = "user_query"  # default
        event_subtype = None

        # Handle intent - can be IntentType enum or string
        intent_value = None
        if intent:
            if hasattr(intent, "value"):
                # It's an IntentType enum
                intent_value = intent.value
                if intent in INTENT_TO_EVENT_TYPE:
                    event_type = INTENT_TO_EVENT_TYPE[intent]
            else:
                # It's already a string
                intent_value = intent
                # Try to map string intent to event type
                # Note: IntentType is not an enum, its values are string constants
                for intent_key, evt_type in INTENT_TO_EVENT_TYPE.items():
                    if intent_key == intent:
                        event_type = evt_type
                        break

        # Override based on tool usage
        tool_names = [tc.get("tool_name", "") for tc in tool_calls]
        if "causal_analysis_tool" in tool_names:
            event_type = "causal_discovery"
            event_subtype = "chatbot_causal_query"
        elif "e2i_data_query_tool" in tool_names:
            event_subtype = "chatbot_data_query"
        elif "agent_routing_tool" in tool_names:
            event_type = "agent_action"
            event_subtype = "chatbot_agent_routing"
        elif "document_retrieval_tool" in tool_names:
            event_subtype = "chatbot_document_search"

        # Build description for the memory
        description = f"User asked: {query[:200]}"
        if len(query) > 200:
            description += "..."

        # Extract entities mentioned in the interaction
        entities = {}
        if brand_context:
            entities["brands"] = [brand_context] if isinstance(brand_context, str) else brand_context
        if region_context:
            entities["regions"] = [region_context] if isinstance(region_context, str) else region_context

        # Extract KPIs from tool results or metadata
        metadata = state.get("metadata", {})
        kpi_name = metadata.get("kpi_name")
        if kpi_name:
            entities["kpis"] = [kpi_name]

        # Build E2I entity references
        e2i_refs = E2IEntityReferences(
            brand=brand_context if isinstance(brand_context, str) else None,
            region=region_context if isinstance(region_context, str) else None,
        )

        # Create episodic memory input
        memory_input = EpisodicMemoryInput(
            event_type=event_type,
            event_subtype=event_subtype,
            description=description,
            raw_content={
                "query": query,
                "response_preview": response_text[:500] if response_text else "",
                "tools_used": tool_names,
                "tool_results_count": len(tool_results),
                "intent": intent_value,
                "session_id": session_id,
            },
            entities=entities if entities else None,
            outcome_type="success" if response_text else "partial_success",
            agent_name="orchestrator",  # Chatbot acts as orchestrator proxy
            importance_score=significance_score,
            e2i_refs=e2i_refs,
        )

        # Text to embed combines query and response for semantic search
        text_to_embed = f"Query: {query}\n\nResponse: {response_text[:1000]}"

        # Insert into episodic memory
        memory_id = await insert_episodic_memory_with_text(
            memory=memory_input,
            text_to_embed=text_to_embed,
            session_id=session_id,
        )

        logger.info(
            f"Saved chatbot interaction to episodic memory: {memory_id} "
            f"(significance={significance_score:.2f}, event_type={event_type})"
        )
        return memory_id

    except Exception as e:
        logger.warning(f"Failed to save to episodic memory: {e}")
        return None


async def finalize_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Finalize the response and prepare for output.

    Persists both user and assistant messages to the database.
    """
    messages = state.get("messages", [])
    session_id = state.get("session_id")
    query = state.get("query", "")
    agent_name = state.get("agent_name")
    agent_tier = state.get("agent_tier")
    tool_results = state.get("tool_results", [])
    rag_context = state.get("rag_context", [])
    rag_sources = state.get("rag_sources", [])

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    # Get the last AI message as response
    response_text = ""
    tool_calls = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            response_text = msg.content
            # Extract tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = [
                    {"tool_name": tc.get("name", ""), "args": tc.get("args", {})}
                    for tc in msg.tool_calls
                ]
            break

    # Track finalization metrics
    messages_persisted = 0
    episodic_memory_saved = False
    significance_score = 0.0

    async def _execute_finalize():
        nonlocal messages_persisted, episodic_memory_saved, significance_score

        # Persist messages to database
        try:
            client = await get_async_supabase_client()
            if client:
                msg_repo = get_chatbot_message_repository(client)

                # Save user message
                await msg_repo.add_message(
                    session_id=session_id,
                    role="user",
                    content=query,
                    metadata={"request_id": state.get("request_id")},
                )
                messages_persisted += 1

                # Save assistant message with full context
                model_used = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
                await msg_repo.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=response_text,
                    agent_name=agent_name,
                    agent_tier=agent_tier,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    rag_context=rag_context,
                    rag_sources=rag_sources,
                    model_used=model_used,
                    metadata={
                        "request_id": state.get("request_id"),
                        "intent": state.get("intent"),
                        "brand_context": state.get("brand_context"),
                        "region_context": state.get("region_context"),
                    },
                )
                messages_persisted += 1

                logger.debug(f"Persisted messages for session: {session_id}")

        except Exception as e:
            logger.warning(f"Failed to persist messages: {e}")

        # =================================================================
        # EPISODIC MEMORY BRIDGE
        # Save significant interactions to episodic memory for cross-session
        # learning and platform knowledge building
        # =================================================================
        try:
            significance_score = _calculate_significance_score(state)

            if significance_score >= SIGNIFICANCE_THRESHOLD:
                memory_id = await _save_to_episodic_memory(
                    state=state,
                    response_text=response_text,
                    tool_calls=tool_calls,
                    significance_score=significance_score,
                )
                if memory_id:
                    episodic_memory_saved = True
                    logger.debug(
                        f"Episodic memory saved: {memory_id} "
                        f"(score={significance_score:.2f})"
                    )
            else:
                logger.debug(
                    f"Skipped episodic memory (score={significance_score:.2f} < "
                    f"threshold={SIGNIFICANCE_THRESHOLD})"
                )
        except Exception as e:
            # Don't fail the response if episodic memory save fails
            logger.warning(f"Episodic memory bridge error: {e}")

        # =================================================================
        # PHASE 7: UNIFIED TRAINING SIGNAL COLLECTION
        # Collect signals from all DSPy phases for feedback_learner optimization
        # =================================================================
        if CHATBOT_SIGNAL_COLLECTION_ENABLED:
            try:
                signal_collector = get_chatbot_signal_collector()

                # Create session signal with all available data
                signal = signal_collector.start_session(
                    session_id=session_id or "",
                    thread_id=state.get("request_id", ""),
                    query=query,
                    user_id=state.get("user_id"),
                    brand_context=state.get("brand_context", "") or "",
                    region_context=state.get("region_context", "") or "",
                )

                # Update intent signal (Phase 3)
                if state.get("intent"):
                    signal_collector.update_intent(
                        session_id=session_id or "",
                        intent=state.get("intent", ""),
                        confidence=state.get("intent_confidence", 0.0) or 0.0,
                        method=state.get("intent_classification_method", "") or "",
                        reasoning=state.get("intent_reasoning", "") or "",
                    )

                # Update routing signal (Phase 4) - using agent_name/tier from state
                if state.get("agent_name"):
                    signal_collector.update_routing(
                        session_id=session_id or "",
                        agent=state.get("agent_name", "") or "",
                        secondary_agents=[],  # Not tracked yet
                        confidence=0.8 if state.get("agent_name") else 0.0,
                        method="workflow",  # Implicit routing via workflow
                        rationale="",
                    )

                # Update RAG signal (Phase 5)
                if rag_context:
                    avg_relevance = 0.0
                    if rag_context:
                        scores = [
                            c.get("relevance_score", c.get("score", 0.5))
                            for c in rag_context
                        ]
                        avg_relevance = sum(scores) / len(scores) if scores else 0.0

                    signal_collector.update_rag(
                        session_id=session_id or "",
                        rewritten_query=state.get("rag_rewritten_query", "") or "",
                        keywords=[],  # Not tracked in state yet
                        entities=[],  # Not tracked in state yet
                        evidence_count=len(rag_context),
                        hop_count=1,  # Default single hop
                        avg_relevance=avg_relevance,
                        method=state.get("rag_retrieval_method", "basic") or "basic",
                    )

                # Update synthesis signal (Phase 6)
                signal_collector.update_synthesis(
                    session_id=session_id or "",
                    response_length=len(response_text),
                    confidence=_get_confidence_level(state.get("confidence_statement", "")),
                    citations_count=len(state.get("evidence_citations", []) or []),
                    method=state.get("synthesis_method", "") or "",
                    follow_up_count=len(state.get("follow_up_suggestions", []) or []),
                )

                # Finalize the session signal
                finalized_signal = signal_collector.finalize_session(
                    session_id=session_id or ""
                )

                # Persist to database if signal was finalized
                if finalized_signal:
                    db_id = await signal_collector.persist_signal_to_database(
                        finalized_signal
                    )
                    if db_id:
                        logger.debug(f"Training signal persisted to DB: id={db_id}")

                logger.debug(
                    f"Collected training signals for session: {session_id}, "
                    f"collector size: {len(signal_collector)}"
                )

            except Exception as e:
                # Don't fail the response if signal collection fails
                logger.warning(f"Training signal collection error: {e}")

    # Execute with tracing if available
    if trace_ctx:
        async with trace_ctx.trace_node("finalize") as node_span:
            await _execute_finalize()
            node_span.log_finalize(
                response_length=len(response_text),
                messages_persisted=messages_persisted,
                episodic_memory_saved=episodic_memory_saved,
                significance_score=significance_score,
            )
    else:
        await _execute_finalize()

    # Build final progress update (complete)
    progress = get_progress_update(
        "finalize",
        current_steps=state.get("progress_steps", []),
        custom_step="Complete",
    )

    return {
        "response_text": response_text,
        "streaming_complete": True,
        **progress,
    }


def _get_confidence_level(confidence_statement: str) -> str:
    """Extract confidence level from confidence statement."""
    if not confidence_statement:
        return "low"
    lower = confidence_statement.lower()
    if "high" in lower:
        return "high"
    elif "moderate" in lower or "medium" in lower:
        return "moderate"
    return "low"


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================


def should_use_tools(state: ChatbotState) -> Literal["tools", "finalize"]:
    """
    Determine if tool execution is needed based on LLM response.
    """
    messages = state.get("messages", [])

    if not messages:
        return "finalize"

    last_message = messages[-1]

    # Check if the last message has tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "finalize"


def after_tools(state: ChatbotState) -> Literal["generate", "finalize"]:
    """
    Determine next step after tool execution.
    """
    messages = state.get("messages", [])

    # If we have tool results, go back to generate for final response
    # Otherwise, finalize
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "tool":
            return "generate"

    return "finalize"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def create_e2i_chatbot_graph() -> StateGraph:
    """
    Create the E2I chatbot LangGraph workflow.

    Workflow (with orchestrator integration):
        init → load_context → classify_intent → retrieve_rag → orchestrator → generate
            ↓                                                                     ↓
            ↓                                                          [tools] ←→ ↑
            ↓                                                                     ↓
            └───────────────────────────────────────────────────────→ finalize → END

    The orchestrator node routes complex queries (causal_analysis, kpi_query,
    recommendation, search, multi_faceted) through the 20-agent orchestrator.
    Simple queries (greeting, help, agent_status, general) skip orchestrator
    and generate responses directly.

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create the graph
    workflow = StateGraph(ChatbotState)

    # Add nodes
    workflow.add_node("init", init_node)
    workflow.add_node("load_context", load_context_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_rag", retrieve_rag_node)
    workflow.add_node("orchestrator", orchestrator_node)  # Routes complex queries to specialized agents
    workflow.add_node("generate", generate_node)
    workflow.add_node("tools", ToolNode(E2I_CHATBOT_TOOLS))
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("init")

    # Add edges
    workflow.add_edge("init", "load_context")
    workflow.add_edge("load_context", "classify_intent")
    workflow.add_edge("classify_intent", "retrieve_rag")
    workflow.add_edge("retrieve_rag", "orchestrator")  # Route through orchestrator for potential agent dispatch
    workflow.add_edge("orchestrator", "generate")  # generate_node skips if orchestrator handled query

    # Conditional edge: generate → tools or finalize
    workflow.add_conditional_edges(
        "generate",
        should_use_tools,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )

    # After tools, go back to generate for response synthesis
    workflow.add_conditional_edges(
        "tools",
        after_tools,
        {
            "generate": "generate",
            "finalize": "finalize",
        },
    )

    # Finalize ends the workflow
    workflow.add_edge("finalize", END)

    # Get Redis checkpointer for cross-request memory persistence
    checkpointer = get_langgraph_checkpointer()

    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# EXPORTS
# =============================================================================


# Create the compiled graph
e2i_chatbot_graph = create_e2i_chatbot_graph()


async def run_chatbot(
    query: str,
    user_id: str,
    request_id: str,
    session_id: str = None,
    brand_context: str = None,
    region_context: str = None,
) -> Dict[str, Any]:
    """
    Run the E2I chatbot workflow.

    Args:
        query: User's query text
        user_id: User UUID
        request_id: Request identifier
        session_id: Optional session ID (generated if not provided)
        brand_context: Optional brand filter
        region_context: Optional region filter

    Returns:
        Final state with response
    """
    # Start timing for latency metrics
    start_time = time.time()

    # Get the tracer (singleton)
    tracer = get_chatbot_tracer()

    # Wrap workflow execution with Opik trace
    async with tracer.trace_workflow(
        query=query,
        session_id=session_id,
        user_id=user_id,
        brand_context=brand_context,
        region_context=region_context,
        metadata={"request_id": request_id},
    ) as trace_ctx:
        # Store trace context for node access
        _active_trace_context.set(trace_ctx)

        # Track for MLflow metrics
        result = None
        error_occurred = False
        error_type = None

        try:
            initial_state = create_initial_state(
                user_id=user_id,
                query=query,
                request_id=request_id,
                session_id=session_id,
                brand_context=brand_context,
                region_context=region_context,
                trace_id=trace_ctx.trace_id,
            )

            # Pass thread_id config for checkpointer (uses session_id for conversation tracking)
            config = {"configurable": {"thread_id": initial_state["session_id"]}}
            result = await e2i_chatbot_graph.ainvoke(initial_state, config=config)

            # Log workflow completion metrics
            trace_ctx.log_workflow_complete(
                status="success" if not result.get("error") else "error",
                success=not result.get("error"),
                intent=result.get("intent"),
                total_tokens=result.get("metadata", {}).get("total_tokens", 0),
                tool_calls_count=len(result.get("tool_results", [])),
                rag_result_count=len(result.get("rag_context", [])),
                response_length=len(result.get("response_text", "")),
            )

            return result

        except Exception as e:
            # Log failure
            error_occurred = True
            error_type = type(e).__name__
            trace_ctx.log_workflow_complete(
                status="failed",
                success=False,
                errors=[str(e)],
            )
            raise

        finally:
            # Clear trace context
            _active_trace_context.set(None)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # =====================================================================
            # MLFLOW SESSION METRICS (Phase 2)
            # Log chatbot session metrics to MLflow for experiment tracking
            # =====================================================================
            if CHATBOT_MLFLOW_METRICS_ENABLED:
                try:
                    experiment_id = await _get_or_create_chatbot_experiment()
                    mlflow_conn = _get_mlflow_connector()

                    if experiment_id and mlflow_conn:
                        # Generate run name from session and request
                        run_name = f"chat_{session_id[:8] if session_id else 'anon'}_{request_id[:8]}"

                        async with mlflow_conn.start_run(
                            experiment_id=experiment_id,
                            run_name=run_name,
                            tags={
                                "session_id": session_id or "none",
                                "request_id": request_id,
                                "trace_id": trace_ctx.trace_id if trace_ctx else "none",
                            },
                        ) as mlflow_run:
                            # Task 2.3: Log session params
                            await mlflow_run.log_params({
                                "user_id": user_id[:8] if user_id else "anon",  # Truncated for privacy
                                "brand_context": brand_context or "none",
                                "region_context": region_context or "none",
                                "query_length": len(query),
                                "is_new_session": str(session_id is None),
                            })

                            # Task 2.4: Log per-request metrics (latency, token usage)
                            metrics = {
                                "latency_ms": latency_ms,
                            }

                            if result:
                                # Token usage from metadata
                                metadata = result.get("metadata", {})
                                total_tokens = metadata.get("total_tokens", 0)
                                if total_tokens:
                                    metrics["total_tokens"] = total_tokens

                                # Response metrics
                                response_text = result.get("response_text", "")
                                metrics["response_length"] = len(response_text)

                                # Task 2.5: Log intent distribution metrics
                                intent = result.get("intent")
                                if intent:
                                    # Convert intent to numeric for MLflow (1 for each type)
                                    intent_str = intent.value if hasattr(intent, "value") else str(intent)
                                    metrics[f"intent_{intent_str}"] = 1

                                # Tool usage metrics
                                tool_results = result.get("tool_results", [])
                                metrics["tool_calls_count"] = len(tool_results)

                                # Task 2.6: Log RAG quality metrics
                                rag_context = result.get("rag_context", [])
                                metrics["rag_result_count"] = len(rag_context)
                                if rag_context:
                                    # Calculate average relevance score
                                    scores = [ctx.get("score", 0) for ctx in rag_context if "score" in ctx]
                                    if scores:
                                        metrics["rag_avg_relevance"] = sum(scores) / len(scores)
                                        metrics["rag_max_relevance"] = max(scores)
                                        metrics["rag_min_relevance"] = min(scores)

                            # Task 2.7: Log error tracking metrics
                            metrics["is_error"] = 1 if error_occurred else 0
                            if result and result.get("error"):
                                metrics["has_workflow_error"] = 1

                            await mlflow_run.log_metrics(metrics)

                            logger.debug(
                                f"MLflow metrics logged: run={run_name}, latency={latency_ms:.0f}ms, "
                                f"error={error_occurred}"
                            )

                except Exception as mlflow_error:
                    # MLflow logging should never break the chatbot
                    logger.warning(f"MLflow metrics logging failed: {mlflow_error}")


async def stream_chatbot(
    query: str,
    user_id: str,
    request_id: str,
    session_id: str = None,
    brand_context: str = None,
    region_context: str = None,
):
    """
    Stream the E2I chatbot workflow.

    Yields state updates as they occur.

    Args:
        query: User's query text
        user_id: User UUID
        request_id: Request identifier
        session_id: Optional session ID
        brand_context: Optional brand filter
        region_context: Optional region filter

    Yields:
        State updates from each node
    """
    initial_state = create_initial_state(
        user_id=user_id,
        query=query,
        request_id=request_id,
        session_id=session_id,
        brand_context=brand_context,
        region_context=region_context,
    )

    # Pass thread_id config for checkpointer (uses session_id for conversation tracking)
    config = {"configurable": {"thread_id": initial_state["session_id"]}}
    async for state_update in e2i_chatbot_graph.astream(initial_state, config=config):
        yield state_update
