"""
E2I Chatbot LangGraph Workflow.

Builds a LangGraph agent workflow for the E2I chatbot with:
- Multi-node processing pipeline
- E2I-specific tool integration
- RAG retrieval for context
- Intent classification
- Streaming response support

Workflow:
    init → load_context → retrieve_rag → classify_intent → generate → [tools] → finalize
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
)
from src.mlops.mlflow_connector import MLflowConnector

# MLflow metrics feature flag
CHATBOT_MLFLOW_METRICS_ENABLED = os.getenv("CHATBOT_MLFLOW_METRICS", "true").lower() == "true"

# MLflow experiment name
CHATBOT_MLFLOW_EXPERIMENT = "chatbot_interactions"

logger = logging.getLogger(__name__)

# Context variable for active trace context (accessible by nodes)
import contextvars
_active_trace_context: contextvars.ContextVar[Optional[ChatbotTraceContext]] = contextvars.ContextVar(
    "chatbot_trace_context", default=None
)

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

    return {
        "messages": [human_msg],
        "metadata": {
            "init_timestamp": str(datetime.now(timezone.utc)),
            "is_new_conversation": is_new_conversation,
        },
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
    }


async def retrieve_rag_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Retrieve relevant context using hybrid RAG.

    Performs semantic + sparse + graph retrieval.
    """
    query = state.get("query", "")
    brand = state.get("brand_context")
    intent = state.get("intent")

    logger.debug(f"RAG retrieval: query={query[:50]}..., intent={intent}")

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    rag_context = []
    rag_sources = []
    relevance_scores = []
    kpi_name = None
    error = None

    async def _execute_rag():
        nonlocal rag_context, rag_sources, relevance_scores, kpi_name, error
        try:
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
            logger.error(f"RAG retrieval failed: {e}")
            error = f"RAG retrieval error: {str(e)}"

    # Execute with tracing if available
    if trace_ctx:
        async with trace_ctx.trace_node("retrieve_rag") as node_span:
            await _execute_rag()
            node_span.log_rag_retrieval(
                result_count=len(rag_context),
                relevance_scores=relevance_scores,
                kpi_filter=kpi_name,
                brand_filter=brand,
                retrieval_method="hybrid",  # hybrid_search uses semantic + sparse
            )
    else:
        await _execute_rag()

    if error:
        return {
            "rag_context": [],
            "rag_sources": [],
            "error": error,
        }

    return {
        "rag_context": rag_context,
        "rag_sources": rag_sources,
    }


async def classify_intent_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Classify the user's query intent using DSPy (with hardcoded fallback).

    Phase 3 DSPy integration: Uses ChatbotIntentClassifier for ML-based
    classification with confidence scores and training signal collection.
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

    async def _execute_classify():
        nonlocal intent, confidence, reasoning, classification_method
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

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "intent_reasoning": reasoning,
        "intent_classification_method": classification_method,
    }


async def generate_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Generate response using Claude with tools.

    This is the main generation node that can invoke tools.
    """
    messages = list(state.get("messages", []))
    rag_context = state.get("rag_context", [])
    brand = state.get("brand_context")
    region = state.get("region_context")

    # Get active trace context for observability
    trace_ctx = _active_trace_context.get()

    # Build context string for system prompt
    context_parts = []
    if brand:
        context_parts.append(f"Current Brand Filter: {brand}")
    if region:
        context_parts.append(f"Current Region Filter: {region}")
    if rag_context:
        context_parts.append("\n## Retrieved Context\n")
        for ctx in rag_context[:3]:  # Top 3 for context window
            context_parts.append(f"- [{ctx['source']}] {ctx['content'][:200]}...")

    context_str = "\n".join(context_parts) if context_parts else ""

    # Create system message
    system_prompt = E2I_CHATBOT_SYSTEM_PROMPT.format(context=context_str)
    system_msg = SystemMessage(content=system_prompt)

    # Prepare messages for LLM
    llm_messages = [system_msg] + messages

    # Track generation metrics
    result = None
    provider = None
    model_name = None
    tool_calls_count = 0
    input_tokens = 0
    output_tokens = 0
    is_fallback = False

    async def _execute_generate():
        nonlocal result, provider, model_name, tool_calls_count, input_tokens, output_tokens, is_fallback
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
                "agent_name": "chatbot",
                "agent_tier": 1,
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            is_fallback = True
            result = _generate_fallback_response(state)

    # Execute with tracing if available
    if trace_ctx:
        async with trace_ctx.trace_node("generate") as node_span:
            await _execute_generate()
            node_span.log_generate(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name or "unknown",
                provider=provider or "unknown",
                tool_calls_count=tool_calls_count,
                temperature=0.3,
                is_fallback=is_fallback,
            )
    else:
        await _execute_generate()

    return result


def _generate_fallback_response(state: ChatbotState) -> Dict[str, Any]:
    """Generate a fallback response when LLM is unavailable."""
    intent = state.get("intent", IntentType.GENERAL)
    query = state.get("query", "")

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
        "agent_name": "chatbot_fallback",
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

    return {
        "response_text": response_text,
        "streaming_complete": True,
    }


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

    Workflow:
        init → load_context → classify_intent → retrieve_rag → generate
            ↓                                                      ↓
            ↓                                            [tools] ←→ ↑
            ↓                                                      ↓
            └──────────────────────────────────────────→ finalize → END

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
    workflow.add_node("generate", generate_node)
    workflow.add_node("tools", ToolNode(E2I_CHATBOT_TOOLS))
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("init")

    # Add edges
    workflow.add_edge("init", "load_context")
    workflow.add_edge("load_context", "classify_intent")
    workflow.add_edge("classify_intent", "retrieve_rag")
    workflow.add_edge("retrieve_rag", "generate")

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
