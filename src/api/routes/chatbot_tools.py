"""
E2I Chatbot Tools for LangGraph Integration.

Provides LangGraph-compatible tools for the E2I chatbot agent:
- e2i_data_query_tool: Unified access to ALL E2I analytics data
- causal_analysis_tool: Run causal analysis via RAG retrieval
- agent_routing_tool: Route to specific tier agents (keyword-based)
- conversation_memory_tool: Retrieve chat history
- document_retrieval_tool: Hybrid RAG search
- orchestrator_tool: Execute queries through the full 18-agent orchestrator system

Adapted from Pydantic AI patterns to LangGraph @tool decorators.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.rag.retriever import hybrid_search
from src.repositories import (
    AgentActivityRepository,
    BusinessMetricRepository,
    CausalPathRepository,
    TriggerRepository,
)
from src.memory.services.factories import get_async_supabase_client
from src.repositories.chatbot_conversation import (
    ChatbotConversationRepository,
    get_chatbot_conversation_repository,
)
from src.repositories.chatbot_message import (
    ChatbotMessageRepository,
    get_chatbot_message_repository,
)
from src.api.routes.cognitive import get_orchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND MODELS
# =============================================================================


class E2IQueryType(str, Enum):
    """Supported query types for E2I data queries."""

    KPI = "kpi"
    CAUSAL_CHAIN = "causal_chain"
    AGENT_ANALYSIS = "agent_analysis"
    TRIGGERS = "triggers"
    EXPERIMENTS = "experiments"
    PREDICTIONS = "predictions"
    RECOMMENDATIONS = "recommendations"
    DRIFT_REPORTS = "drift_reports"


class TimeRange(str, Enum):
    """Time range options for queries."""

    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_YEAR = "last_year"
    ALL_TIME = "all_time"


class E2IDataQueryInput(BaseModel):
    """Input schema for e2i_data_query_tool."""

    query_type: E2IQueryType = Field(
        description="Type of E2I data to query: kpi, causal_chain, agent_analysis, triggers, experiments, predictions, recommendations, drift_reports"
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter: Kisqali, Fabhalta, or Remibrutinib",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region filter (e.g., US, EU, APAC)",
    )
    kpi_name: Optional[str] = Field(
        default=None,
        description="Specific KPI name for KPI queries (e.g., TRx, NRx, conversion_rate)",
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Agent name filter for agent_analysis queries",
    )
    time_range: TimeRange = Field(
        default=TimeRange.LAST_30_DAYS,
        description="Time range for the query",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional filters as key-value pairs",
    )


class CausalAnalysisInput(BaseModel):
    """Input schema for causal_analysis_tool."""

    kpi_name: str = Field(
        description="KPI to analyze (e.g., TRx, NRx, conversion_rate, market_share)"
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter: Kisqali, Fabhalta, or Remibrutinib",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region filter (e.g., US, EU, APAC)",
    )
    time_period: Optional[str] = Field(
        default="last_30_days",
        description="Time period for analysis",
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for causal relationships",
    )


class AgentRoutingInput(BaseModel):
    """Input schema for agent_routing_tool."""

    query: str = Field(description="The user's query to route")
    target_agent: Optional[str] = Field(
        default=None,
        description="Specific agent to route to (if known)",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for routing decision",
    )


class ConversationMemoryInput(BaseModel):
    """Input schema for conversation_memory_tool."""

    session_id: str = Field(description="Session ID to retrieve history for")
    message_count: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recent messages to retrieve",
    )
    include_tool_calls: bool = Field(
        default=True,
        description="Whether to include tool call details",
    )


class DocumentRetrievalInput(BaseModel):
    """Input schema for document_retrieval_tool."""

    query: str = Field(description="Search query for document retrieval")
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter for documents",
    )
    kpi_name: Optional[str] = Field(
        default=None,
        description="KPI name for targeted retrieval",
    )


class OrchestratorToolInput(BaseModel):
    """Input schema for orchestrator_tool."""

    query: str = Field(
        description="The query to process through the E2I orchestrator and 18-agent system"
    )
    target_agent: Optional[str] = Field(
        default=None,
        description="Specific agent to route to (e.g., causal_impact, experiment_designer, drift_monitor)",
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand context for the query (Kisqali, Fabhalta, Remibrutinib)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region context for the query (US, EU, APAC)",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context continuity",
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_time_filter(time_range: TimeRange) -> datetime:
    """Convert time range enum to datetime filter."""
    now = datetime.utcnow()
    if time_range == TimeRange.LAST_7_DAYS:
        return now - timedelta(days=7)
    elif time_range == TimeRange.LAST_30_DAYS:
        return now - timedelta(days=30)
    elif time_range == TimeRange.LAST_90_DAYS:
        return now - timedelta(days=90)
    elif time_range == TimeRange.LAST_YEAR:
        return now - timedelta(days=365)
    else:  # ALL_TIME
        return datetime(2020, 1, 1)


async def _query_kpis(
    brand: Optional[str],
    region: Optional[str],
    kpi_name: Optional[str],
    since: datetime,
    limit: int,
) -> Dict[str, Any]:
    """Query KPI metrics from business_metrics table."""
    try:
        client = await get_async_supabase_client()
        repo = BusinessMetricRepository(client)

        filters = {}
        if brand:
            filters["brand"] = brand
        if region:
            filters["region"] = region
        if kpi_name:
            filters["kpi_name"] = kpi_name

        # Get metrics with filters
        metrics = await repo.get_many(filters=filters, limit=limit)

        return {
            "success": True,
            "query_type": "kpi",
            "count": len(metrics),
            "data": metrics,
            "filters_applied": filters,
        }
    except Exception as e:
        logger.error(f"KPI query failed: {e}")
        return {"success": False, "error": str(e), "query_type": "kpi"}


async def _query_causal_chains(
    brand: Optional[str],
    kpi_name: Optional[str],
    since: datetime,
    limit: int,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    """Query causal relationships from causal_paths table."""
    try:
        client = await get_async_supabase_client()
        repo = CausalPathRepository(client)

        filters = {}
        if brand:
            filters["brand"] = brand

        # Use RAG retriever for semantic search if kpi_name provided
        if kpi_name:
            results = await hybrid_search(
                query=f"causal paths affecting {kpi_name}",
                k=limit,
                kpi_name=kpi_name,
                filters={"brand": brand} if brand else None,
            )
            return {
                "success": True,
                "query_type": "causal_chain",
                "count": len(results),
                "data": [
                    {
                        "source_id": r.source_id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
                "kpi_analyzed": kpi_name,
            }

        paths = await repo.get_many(filters=filters, limit=limit)
        return {
            "success": True,
            "query_type": "causal_chain",
            "count": len(paths),
            "data": paths,
        }
    except Exception as e:
        logger.error(f"Causal chain query failed: {e}")
        return {"success": False, "error": str(e), "query_type": "causal_chain"}


async def _query_agent_analysis(
    agent_name: Optional[str],
    brand: Optional[str],
    since: datetime,
    limit: int,
) -> Dict[str, Any]:
    """Query agent analysis outputs from agent_activities table."""
    try:
        client = await get_async_supabase_client()
        repo = AgentActivityRepository(client)

        filters = {}
        if agent_name:
            filters["agent_name"] = agent_name
        if brand:
            filters["brand"] = brand

        activities = await repo.get_many(filters=filters, limit=limit)

        return {
            "success": True,
            "query_type": "agent_analysis",
            "count": len(activities),
            "data": activities,
            "agent_filter": agent_name,
        }
    except Exception as e:
        logger.error(f"Agent analysis query failed: {e}")
        return {"success": False, "error": str(e), "query_type": "agent_analysis"}


async def _query_triggers(
    brand: Optional[str],
    region: Optional[str],
    since: datetime,
    limit: int,
) -> Dict[str, Any]:
    """Query triggers/alerts from triggers table."""
    try:
        client = await get_async_supabase_client()
        repo = TriggerRepository(client)

        filters = {}
        if brand:
            filters["brand"] = brand
        if region:
            filters["region"] = region

        triggers = await repo.get_many(filters=filters, limit=limit)

        return {
            "success": True,
            "query_type": "triggers",
            "count": len(triggers),
            "data": triggers,
        }
    except Exception as e:
        logger.error(f"Triggers query failed: {e}")
        return {"success": False, "error": str(e), "query_type": "triggers"}


async def _query_via_rag(
    query_type: str,
    query: str,
    filters: Optional[Dict[str, Any]],
    limit: int,
) -> Dict[str, Any]:
    """Fallback RAG query for experiments, predictions, recommendations, drift_reports."""
    try:
        results = await hybrid_search(
            query=f"{query_type}: {query}",
            k=limit,
            filters=filters,
        )

        return {
            "success": True,
            "query_type": query_type,
            "count": len(results),
            "data": [
                {
                    "source_id": r.source_id,
                    "content": r.content,
                    "score": r.score,
                    "source": r.source,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "retrieval_method": "hybrid_rag",
        }
    except Exception as e:
        logger.error(f"RAG query for {query_type} failed: {e}")
        return {"success": False, "error": str(e), "query_type": query_type}


# =============================================================================
# LANGGRAPH TOOLS
# =============================================================================


@tool(args_schema=E2IDataQueryInput)
async def e2i_data_query_tool(
    query_type: E2IQueryType,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    kpi_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    time_range: TimeRange = TimeRange.LAST_30_DAYS,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query E2I analytics data across multiple data types.

    This tool provides unified access to ALL E2I analytics data including:
    - KPIs: TRx, NRx, market share, conversion rates
    - Causal chains: Discovered cause-effect relationships
    - Agent analyses: Outputs from the 18-agent system
    - Triggers: Alerts and explanations for metric changes
    - Experiments: A/B test designs and results
    - Predictions: ML model predictions
    - Recommendations: Generated recommendations
    - Drift reports: Data/model drift detection results

    Use this tool when users ask about E2I business metrics, causal relationships,
    agent outputs, or any analytical data from the platform.

    Args:
        query_type: Type of data to query (kpi, causal_chain, agent_analysis, etc.)
        brand: Optional brand filter (Kisqali, Fabhalta, Remibrutinib)
        region: Optional region filter (US, EU, APAC)
        kpi_name: Specific KPI name for KPI/causal queries
        agent_name: Agent name filter for agent_analysis queries
        time_range: Time range for the query (last_7_days, last_30_days, etc.)
        limit: Maximum results (1-100)
        filters: Additional key-value filters

    Returns:
        Dict with success status, query results, and metadata
    """
    logger.info(f"E2I data query: type={query_type}, brand={brand}, kpi={kpi_name}")

    since = _get_time_filter(time_range)

    if query_type == E2IQueryType.KPI:
        return await _query_kpis(brand, region, kpi_name, since, limit)

    elif query_type == E2IQueryType.CAUSAL_CHAIN:
        return await _query_causal_chains(brand, kpi_name, since, limit)

    elif query_type == E2IQueryType.AGENT_ANALYSIS:
        return await _query_agent_analysis(agent_name, brand, since, limit)

    elif query_type == E2IQueryType.TRIGGERS:
        return await _query_triggers(brand, region, since, limit)

    else:
        # Use RAG for experiments, predictions, recommendations, drift_reports
        query_str = f"{brand or ''} {region or ''} {kpi_name or ''}".strip() or "recent"
        combined_filters = filters or {}
        if brand:
            combined_filters["brand"] = brand
        return await _query_via_rag(query_type.value, query_str, combined_filters, limit)


@tool(args_schema=CausalAnalysisInput)
async def causal_analysis_tool(
    kpi_name: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    time_period: Optional[str] = "last_30_days",
    min_confidence: float = 0.7,
) -> Dict[str, Any]:
    """
    Run causal analysis to identify factors affecting a KPI.

    This tool performs causal inference analysis to find:
    - Direct causes of KPI changes
    - Indirect causal chains
    - Estimated effect magnitudes
    - Confidence scores for relationships

    Use this tool when users want to understand WHY a metric changed
    or what factors are driving performance.

    Args:
        kpi_name: KPI to analyze (TRx, NRx, conversion_rate, market_share)
        brand: Brand filter (Kisqali, Fabhalta, Remibrutinib)
        region: Region filter (US, EU, APAC)
        time_period: Time period for analysis
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        Dict with causal analysis results including chains and effects
    """
    logger.info(f"Causal analysis: kpi={kpi_name}, brand={brand}, confidence>={min_confidence}")

    try:
        # Use hybrid search with KPI-focused retrieval
        results = await hybrid_search(
            query=f"causal analysis of {kpi_name} drivers and effects",
            k=15,
            kpi_name=kpi_name,
            filters={"brand": brand} if brand else None,
        )

        # Filter by confidence if metadata available
        filtered_results = []
        for r in results:
            confidence = r.metadata.get("confidence", r.score)
            if confidence >= min_confidence:
                filtered_results.append(
                    {
                        "source_id": r.source_id,
                        "content": r.content,
                        "confidence": confidence,
                        "effect_magnitude": r.metadata.get("effect_magnitude"),
                        "causal_direction": r.metadata.get("causal_direction"),
                        "metadata": r.metadata,
                    }
                )

        return {
            "success": True,
            "kpi_analyzed": kpi_name,
            "brand": brand,
            "region": region,
            "causal_chains_found": len(filtered_results),
            "min_confidence_applied": min_confidence,
            "results": filtered_results,
            "analysis_type": "hybrid_causal_retrieval",
        }

    except Exception as e:
        logger.error(f"Causal analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "kpi_analyzed": kpi_name,
        }


@tool(args_schema=AgentRoutingInput)
async def agent_routing_tool(
    query: str,
    target_agent: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Route a query to the appropriate E2I agent tier.

    The E2I system has 18 agents organized in 6 tiers:
    - Tier 0: ML Foundation (scope_definer, data_preparer, feature_analyzer, etc.)
    - Tier 1: Orchestration (orchestrator, tool_composer)
    - Tier 2: Causal Analytics (causal_impact, gap_analyzer, heterogeneous_optimizer)
    - Tier 3: Monitoring (drift_monitor, experiment_designer, health_score)
    - Tier 4: Predictions (prediction_synthesizer, resource_optimizer)
    - Tier 5: Learning (explainer, feedback_learner)

    Use this tool when a query needs specialized agent processing.

    Args:
        query: The user's query to route
        target_agent: Specific agent to route to (if known)
        context: Additional context for routing decision

    Returns:
        Dict with routing decision and agent recommendation
    """
    logger.info(f"Agent routing: query={query[:50]}..., target={target_agent}")

    # Agent capability mapping
    agent_capabilities = {
        "causal_impact": ["why", "cause", "effect", "impact", "driver", "factor"],
        "gap_analyzer": ["gap", "opportunity", "roi", "underperforming", "potential"],
        "drift_monitor": ["drift", "change", "shift", "anomaly", "deviation"],
        "experiment_designer": ["test", "experiment", "a/b", "trial", "hypothesis"],
        "prediction_synthesizer": ["predict", "forecast", "future", "trend", "projection"],
        "explainer": ["explain", "why", "understand", "summarize", "interpret"],
        "health_score": ["health", "status", "score", "performance", "metric"],
    }

    # If target agent specified, validate and return
    if target_agent:
        if target_agent in agent_capabilities:
            return {
                "success": True,
                "routed_to": target_agent,
                "reason": "Explicit agent selection",
                "capabilities": agent_capabilities.get(target_agent, []),
            }
        else:
            return {
                "success": False,
                "error": f"Unknown agent: {target_agent}",
                "available_agents": list(agent_capabilities.keys()),
            }

    # Auto-route based on query keywords
    query_lower = query.lower()
    scores = {}

    for agent, keywords in agent_capabilities.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[agent] = score

    if scores:
        best_agent = max(scores, key=scores.get)
        return {
            "success": True,
            "routed_to": best_agent,
            "reason": f"Keyword match (score: {scores[best_agent]})",
            "all_scores": scores,
            "query_analyzed": query[:100],
        }

    # Default to explainer for general queries
    return {
        "success": True,
        "routed_to": "explainer",
        "reason": "Default routing (no specific keywords matched)",
        "query_analyzed": query[:100],
    }


@tool(args_schema=ConversationMemoryInput)
async def conversation_memory_tool(
    session_id: str,
    message_count: int = 10,
    include_tool_calls: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve conversation history from a chat session.

    This tool provides access to:
    - Recent messages in a conversation
    - Tool calls and results
    - Agent attributions
    - RAG context used

    Use this tool to provide context-aware responses based on
    previous conversation turns.

    Args:
        session_id: Session ID to retrieve history for
        message_count: Number of recent messages (1-50)
        include_tool_calls: Whether to include tool call details

    Returns:
        Dict with conversation history and metadata
    """
    logger.info(f"Conversation memory: session={session_id}, count={message_count}")

    try:
        client = await get_async_supabase_client()
        msg_repo = get_chatbot_message_repository(client)
        conv_repo = get_chatbot_conversation_repository(client)

        # Get conversation metadata
        conversation = await conv_repo.get_by_session_id(session_id)
        if not conversation:
            return {
                "success": False,
                "error": "Conversation not found",
                "session_id": session_id,
            }

        # Get recent messages
        messages = await msg_repo.get_recent_messages(session_id, count=message_count)

        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted = {
                "role": msg.get("role"),
                "content": msg.get("content"),
                "created_at": msg.get("created_at"),
                "agent_name": msg.get("agent_name"),
            }
            if include_tool_calls:
                formatted["tool_calls"] = msg.get("tool_calls", [])
                formatted["tool_results"] = msg.get("tool_results", [])
            formatted_messages.append(formatted)

        return {
            "success": True,
            "session_id": session_id,
            "conversation_title": conversation.get("title"),
            "brand_context": conversation.get("brand_context"),
            "region_context": conversation.get("region_context"),
            "message_count": len(formatted_messages),
            "messages": formatted_messages,
        }

    except Exception as e:
        logger.error(f"Conversation memory retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id,
        }


@tool(args_schema=DocumentRetrievalInput)
async def document_retrieval_tool(
    query: str,
    k: int = 5,
    brand: Optional[str] = None,
    kpi_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve relevant documents using hybrid RAG search.

    This tool performs hybrid retrieval combining:
    - Dense vector search (semantic similarity)
    - Sparse BM25 search (keyword matching)
    - Graph traversal (causal relationships)

    Use this tool when users need information from the knowledge base
    about E2I analytics, procedures, or historical data.

    Args:
        query: Search query for document retrieval
        k: Number of documents to retrieve (1-20)
        brand: Optional brand filter
        kpi_name: Optional KPI name for targeted retrieval

    Returns:
        Dict with retrieved documents and relevance scores
    """
    logger.info(f"Document retrieval: query={query[:50]}..., k={k}, brand={brand}")

    try:
        filters = {}
        if brand:
            filters["brand"] = brand

        results = await hybrid_search(
            query=query,
            k=k,
            kpi_name=kpi_name,
            filters=filters if filters else None,
        )

        documents = [
            {
                "source_id": r.source_id,
                "content": r.content,
                "relevance_score": r.score,
                "source": r.source,
                "retrieval_method": r.retrieval_method,
                "metadata": r.metadata,
            }
            for r in results
        ]

        return {
            "success": True,
            "query": query,
            "document_count": len(documents),
            "documents": documents,
            "filters_applied": {"brand": brand, "kpi_name": kpi_name},
        }

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }


@tool(args_schema=OrchestratorToolInput)
async def orchestrator_tool(
    query: str,
    target_agent: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a query through the E2I orchestrator and 18-agent system.

    This tool provides access to the full E2I multi-agent architecture:
    - Tier 0: ML Foundation (data prep, feature analysis, model training)
    - Tier 1: Orchestration (orchestrator, tool_composer)
    - Tier 2: Causal Analytics (causal_impact, gap_analyzer, heterogeneous_optimizer)
    - Tier 3: Monitoring (drift_monitor, experiment_designer, health_score)
    - Tier 4: Predictions (prediction_synthesizer, resource_optimizer)
    - Tier 5: Learning (explainer, feedback_learner)

    Use this tool for:
    - Complex causal analysis requiring the causal_impact agent
    - Experiment design through experiment_designer agent
    - Drift detection and model health checks
    - Multi-agent orchestrated queries
    - Any query that benefits from the full agent pipeline

    This tool routes through the real orchestrator, NOT just keyword matching.

    Args:
        query: The query to process through the orchestrator
        target_agent: Optional specific agent to route to
        brand: Brand context (Kisqali, Fabhalta, Remibrutinib)
        region: Region context (US, EU, APAC)
        session_id: Session ID for context continuity

    Returns:
        Dict with orchestrator response, agents dispatched, and confidence
    """
    logger.info(f"Orchestrator tool: query={query[:50]}..., target_agent={target_agent}")

    try:
        orchestrator = get_orchestrator()

        if orchestrator is None:
            logger.warning("Orchestrator unavailable, using fallback")
            # Fallback to hybrid search when orchestrator unavailable
            fallback_results = await hybrid_search(
                query=query,
                k=10,
                filters={"brand": brand} if brand else None,
            )
            return {
                "success": True,
                "fallback": True,
                "reason": "Orchestrator unavailable - using RAG fallback",
                "query": query,
                "result_count": len(fallback_results),
                "results": [
                    {
                        "content": r.content,
                        "score": r.score,
                        "source": r.source,
                    }
                    for r in fallback_results[:5]
                ],
            }

        # Build user context for orchestrator
        user_context = {}
        if brand:
            user_context["brand"] = brand
        if region:
            user_context["region"] = region
        if target_agent:
            user_context["target_agent"] = target_agent

        # Generate session_id if not provided
        effective_session_id = session_id or f"chatbot-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Call the orchestrator
        orchestrator_result = await orchestrator.run({
            "query": query,
            "session_id": effective_session_id,
            "user_context": user_context,
        })

        # Extract key fields from orchestrator response
        response_text = orchestrator_result.get("response_text", "")
        response_confidence = orchestrator_result.get("response_confidence", 0.85)
        agents_dispatched = orchestrator_result.get("agents_dispatched", [])
        analysis_results = orchestrator_result.get("analysis_results", {})

        return {
            "success": True,
            "fallback": False,
            "query": query,
            "response": response_text,
            "confidence": response_confidence,
            "agents_dispatched": agents_dispatched,
            "analysis_results": analysis_results,
            "target_agent_requested": target_agent,
            "context": {
                "brand": brand,
                "region": region,
                "session_id": effective_session_id,
            },
        }

    except Exception as e:
        logger.error(f"Orchestrator tool failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "fallback": True,
        }


# =============================================================================
# TOOL EXPORTS
# =============================================================================


# List of all E2I chatbot tools for use in LangGraph ToolNode
E2I_CHATBOT_TOOLS = [
    e2i_data_query_tool,
    causal_analysis_tool,
    agent_routing_tool,
    conversation_memory_tool,
    document_retrieval_tool,
    orchestrator_tool,
]

# Tool name to function mapping
E2I_TOOL_MAP = {
    "e2i_data_query_tool": e2i_data_query_tool,
    "causal_analysis_tool": causal_analysis_tool,
    "agent_routing_tool": agent_routing_tool,
    "conversation_memory_tool": conversation_memory_tool,
    "document_retrieval_tool": document_retrieval_tool,
    "orchestrator_tool": orchestrator_tool,
}


def get_e2i_chatbot_tools() -> List:
    """Get list of all E2I chatbot tools for LangGraph integration."""
    return E2I_CHATBOT_TOOLS


def get_tool_by_name(name: str):
    """Get a specific tool by name."""
    return E2I_TOOL_MAP.get(name)
