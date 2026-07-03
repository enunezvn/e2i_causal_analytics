"""
E2I Chatbot Tools for LangGraph Integration.

Provides LangGraph-compatible tools for the E2I chatbot agent:
- e2i_data_query_tool: Unified access to ALL E2I analytics data
- causal_analysis_tool: Run causal analysis via hybrid RAG search
- agent_routing_tool: Route to specific tier agents (keyword-based)
- conversation_memory_tool: Retrieve chat history
- document_retrieval_tool: Hybrid RAG search
- orchestrator_tool: Execute queries through the full 21-agent orchestrator system
- tool_composer_tool: Process multi-faceted queries via Tool Composer pipeline

Adapted from Pydantic AI patterns to LangGraph @tool decorators.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from src.agents.tool_composer import compose_query
from src.api.routes.chatbot_dspy import (
    CHATBOT_DSPY_ROUTING_ENABLED,
    VALID_AGENTS,
    route_agent_dspy,
    route_agent_hardcoded,
)
from src.api.routes.cognitive import get_orchestrator
from src.memory.services.factories import get_async_supabase_client
from src.rag.retriever import hybrid_search
from src.repositories import (
    AgentActivityRepository,
    BusinessMetricRepository,
    CausalPathRepository,
    TriggerRepository,
)
from src.repositories.chatbot_conversation import (
    get_chatbot_conversation_repository,
)
from src.repositories.chatbot_message import (
    get_chatbot_message_repository,
)
from src.services import cohort_resolution, kpi_resolution
from src.utils.llm_factory import get_chat_llm

logger = logging.getLogger(__name__)

# Try to import Opik for tracing
try:
    from src.mlops.opik_connector import OpikConnector

    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    logger.debug("Opik not available for agent routing tracing")


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
        description="Brand filter, resolved case-insensitively against the actual data values (e.g., Kisqali)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region filter, resolved case-insensitively against the actual data values (e.g., Northeast)",
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_type": "kpi",
                "brand": "Kisqali",
                "region": "Northeast",
                "kpi_name": "TRx",
                "time_range": "last_30_days",
                "limit": 10,
            }
        }
    )


class CausalAnalysisInput(BaseModel):
    """Input schema for causal_analysis_tool."""

    kpi_name: str = Field(
        description="KPI to analyze (e.g., TRx, NRx, conversion_rate, market_share)"
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter, resolved case-insensitively against the actual data values (e.g., Kisqali)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region filter, resolved case-insensitively against the actual data values (e.g., Northeast)",
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_name": "TRx",
                "brand": "Kisqali",
                "region": "Northeast",
                "time_period": "last_30_days",
                "min_confidence": 0.7,
            }
        }
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Why did TRx drop for Kisqali in Q3?",
                "target_agent": "causal_impact",
                "context": {"intent": "causal_analysis", "brand": "Kisqali"},
            }
        }
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "sess_abc123def456",
                "message_count": 10,
                "include_tool_calls": True,
            }
        }
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Kisqali conversion rate trends",
                "k": 5,
                "brand": "Kisqali",
                "kpi_name": "conversion_rate",
            }
        }
    )


class OrchestratorToolInput(BaseModel):
    """Input schema for orchestrator_tool."""

    query: str = Field(
        description="The query to process through the E2I orchestrator and 21-agent system"
    )
    target_agent: Optional[str] = Field(
        default=None,
        description="Specific agent to route to (e.g., causal_impact, experiment_designer, drift_monitor)",
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand context for the query, resolved case-insensitively against the actual data values (e.g., Kisqali)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region context for the query, resolved case-insensitively against the actual data values (e.g., Northeast)",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context continuity",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Analyze the impact of rep visits on Kisqali TRx in the Northeast",
                "target_agent": "causal_impact",
                "brand": "Kisqali",
                "region": "Northeast",
                "session_id": "sess_abc123",
            }
        }
    )


class ToolComposerToolInput(BaseModel):
    """Input schema for tool_composer_tool."""

    query: str = Field(
        description="Multi-faceted query requiring decomposition and multi-agent processing"
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand context for the query, resolved case-insensitively against the actual data values (e.g., Kisqali)",
    )
    region: Optional[str] = Field(
        default=None,
        description=(
            "Region context for the query, resolved case-insensitively against the "
            "actual geographic_region values in the data (US census regions)."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context continuity",
    )
    data_source: Optional[str] = Field(
        default=None,
        description=(
            "Optional cohort data source (table name or parquet/s3 path) used "
            "to resolve a real DataFrame for (brand, region) and supply it to "
            "the tools as estimation_data. If omitted or unresolvable, the "
            "composable tools fail-closed honestly."
        ),
    )
    max_parallel: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of parallel tool executions",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Compare TRx trends across Kisqali, Fabhalta, and Remibrutinib, then explain causal factors",
                "brand": None,
                "region": "Northeast",
                "session_id": "sess_abc123",
                "max_parallel": 3,
            }
        }
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_time_filter(time_range: TimeRange) -> datetime:
    """Convert time range enum to datetime filter."""
    now = datetime.now(timezone.utc)
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
            # business_metrics uses 'metric_name' column, not 'kpi_name'
            filters["metric_name"] = kpi_name

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
    include_synthetic: bool = False,
) -> Dict[str, Any]:
    """Query causal relationships from causal_paths table.

    Chat is an end-user real-mode surface: ``include_synthetic`` defaults to
    False so synthetic causal paths (planted ground-truth validation data,
    migration 063 provenance) never surface as real insight (#893). The opt-in
    exists for agent-context/validation callers only and is deliberately NOT
    exposed in the LLM tool schema. On an all-synthetic substrate the honest
    real-mode answer is empty (same fail-closed semantics as #872).
    """
    try:
        client = await get_async_supabase_client()
        repo = CausalPathRepository(client)

        # Note: causal_paths table does not have 'brand' column
        # Brand filtering is only available via hybrid_search (RAG index)
        filters: dict[str, str] = {}

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

        paths = await repo.get_many(
            filters=filters, limit=limit, include_synthetic=include_synthetic
        )
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

        # Note: agent_activities table does not have 'brand' column
        # Brand parameter kept for API compatibility but not used in direct queries
        filters = {}
        if agent_name:
            filters["agent_name"] = agent_name

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

        # Note: triggers table does not have 'brand' or 'region' columns
        # Parameters kept for API compatibility but not used in direct queries
        filters: dict[str, str] = {}

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
    - KPIs: STORED snapshot rows from business_metrics. For a COMPUTED KPI VALUE
      for a brand (e.g. "what is the NBRx/NRx/TRx/market share for Kisqali?"),
      prefer ``kpi_calculate_tool`` — it resolves the KPI definition and calculates
      from the real substrate, whereas this returns the raw stored rows (and 0 for
      a derived KPI like NBRx that is not materialized here).
    - Causal chains: Discovered cause-effect relationships
    - Agent analyses: Outputs from the 21-agent system
    - Triggers: Alerts and explanations for metric changes
    - Experiments: A/B test designs and results
    - Predictions: ML model predictions
    - Recommendations: Generated recommendations
    - Drift reports: Data/model drift detection results

    Use this tool when users ask about E2I business metrics, causal relationships,
    agent outputs, or any analytical data from the platform.

    Args:
        query_type: Type of data to query (kpi, causal_chain, agent_analysis, etc.)
        brand: Optional brand filter, resolved case-insensitively against the actual data values
        region: Optional region filter, resolved case-insensitively against the actual data values
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
        brand: Brand filter, resolved case-insensitively against the actual data values
        region: Region filter, resolved case-insensitively against the actual data values
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
    Route a query to the appropriate E2I agent tier using DSPy.

    The E2I system has 21 agents organized in 6 tiers:
    - Tier 0: ML Foundation (scope_definer, data_preparer, feature_analyzer, etc.)
    - Tier 1: Orchestration (orchestrator, tool_composer)
    - Tier 2: Causal Analytics (causal_impact, gap_analyzer, heterogeneous_optimizer)
    - Tier 3: Monitoring (drift_monitor, experiment_designer, experiment_monitor, health_score)
    - Tier 4: Predictions (prediction_synthesizer, resource_optimizer)
    - Tier 5: Learning (explainer, feedback_learner)

    Uses DSPy-based routing with intelligent agent selection and fallback
    to keyword matching when DSPy is unavailable.

    Args:
        query: The user's query to route
        target_agent: Specific agent to route to (if known)
        context: Additional context for routing decision (intent, brand_context)

    Returns:
        Dict with routing decision, confidence, rationale, and agent recommendation
    """
    logger.info(f"Agent routing: query={query[:50]}..., target={target_agent}")

    # Initialize Opik tracing if available
    opik_span = None
    if OPIK_AVAILABLE:
        try:
            opik = OpikConnector()
            opik_span = opik.start_span(  # type: ignore[attr-defined]
                name="agent_routing",
                metadata={"query_preview": query[:100], "target_agent": target_agent},
            )
        except Exception as e:
            logger.debug(f"Failed to start Opik span: {e}")

    try:
        # If target agent specified, validate and return
        if target_agent:
            if target_agent in VALID_AGENTS:
                result = {
                    "success": True,
                    "routed_to": target_agent,
                    "secondary_agents": [],
                    "routing_confidence": 1.0,
                    "rationale": "Explicit agent selection",
                    "routing_method": "explicit",
                    "query_analyzed": query[:100],
                }
                # Log to Opik
                if opik_span:
                    opik_span.set_metadata(
                        {
                            "routed_to": target_agent,
                            "routing_confidence": 1.0,
                            "routing_method": "explicit",
                        }
                    )
                return result
            else:
                return {
                    "success": False,
                    "error": f"Unknown agent: {target_agent}",
                    "available_agents": list(VALID_AGENTS),
                }

        # Extract context values
        intent = ""
        brand_context = ""
        if context:
            intent = context.get("intent", "")
            brand_context = context.get("brand_context", "") or context.get("brand", "")

        # Use DSPy routing with fallback to hardcoded
        (
            primary_agent,
            secondary_agents,
            confidence,
            rationale,
            routing_method,
        ) = await route_agent_dspy(
            query=query,
            intent=intent,
            brand_context=brand_context,
            collect_signal=True,
        )

        result = {
            "success": True,
            "routed_to": primary_agent,
            "secondary_agents": secondary_agents,
            "routing_confidence": confidence,
            "rationale": rationale,
            "routing_method": routing_method,
            "dspy_enabled": CHATBOT_DSPY_ROUTING_ENABLED,
            "query_analyzed": query[:100],
        }

        # Log routing decision to Opik
        if opik_span:
            opik_span.set_metadata(
                {
                    "routed_to": primary_agent,
                    "secondary_agents": secondary_agents,
                    "routing_confidence": confidence,
                    "routing_method": routing_method,
                    "intent": intent,
                    "brand_context": brand_context,
                }
            )

        return result

    except Exception as e:
        logger.error(f"Agent routing failed: {e}")
        # Fallback to hardcoded routing on error
        try:
            primary_agent, secondary_agents, confidence, rationale = route_agent_hardcoded(
                query, intent="" if not context else context.get("intent", "")
            )
            return {
                "success": True,
                "routed_to": primary_agent,
                "secondary_agents": secondary_agents,
                "routing_confidence": confidence,
                "rationale": rationale,
                "routing_method": "hardcoded_fallback",
                "fallback_reason": str(e),
                "query_analyzed": query[:100],
            }
        except Exception as fallback_error:
            return {
                "success": False,
                "error": str(e),
                "fallback_error": str(fallback_error),
                "query_analyzed": query[:100],
            }
    finally:
        # End Opik span
        if opik_span:
            try:
                opik_span.end()
            except Exception:
                pass


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
    Execute a query through the E2I orchestrator and 21-agent system.

    This tool provides access to the full E2I multi-agent architecture:
    - Tier 0: ML Foundation (data prep, feature analysis, model training)
    - Tier 1: Orchestration (orchestrator, tool_composer)
    - Tier 2: Causal Analytics (causal_impact, gap_analyzer, heterogeneous_optimizer)
    - Tier 3: Monitoring (drift_monitor, experiment_designer, experiment_monitor, health_score)
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
        brand: Brand context, resolved case-insensitively against the actual data values
        region: Region context, resolved case-insensitively against the actual data values
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
        orchestrator_result = await orchestrator.run(
            {
                "query": query,
                "session_id": effective_session_id,
                "user_context": user_context,
            }
        )

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


def _resolve_cohort_frame(
    brand: Optional[str],
    region: Optional[str],
    data_source: Optional[str],
) -> Optional["pd.DataFrame"]:
    """Resolve a real cohort DataFrame for (brand, region).

    Delegates to the shared :func:`cohort_resolution.resolve_cohort_frame`
    service (issue #779), which:

    * with an explicit ``data_source`` -> uses the tier0
      ``CohortConstructorAgent`` loader (preserves the original R4 behavior);
    * WITHOUT a ``data_source`` -> resolves the canonical ``patient_journeys``
      table filtered by brand + ``geographic_region``.

    Returns the cohort frame on success, or ``None`` when nothing resolves.
    RAISES on genuine loader/infra failure so the caller can log-and-proceed
    (the composable tools then fail closed honestly -- never substituting
    fabricated data).
    """
    return cohort_resolution.resolve_cohort_frame(brand, region, data_source=data_source)


@tool(args_schema=ToolComposerToolInput)
async def tool_composer_tool(
    query: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    session_id: Optional[str] = None,
    max_parallel: int = 3,
    data_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process multi-faceted queries through the E2I Tool Composer.

    The Tool Composer handles complex queries that require:
    - Multiple sub-questions answered by different agents
    - Cross-agent analysis (e.g., causal + drift + predictions)
    - Comparison across brands, KPIs, or time periods
    - Analysis combined with recommendations

    Pipeline:
    1. DECOMPOSE: Break query into atomic sub-questions
    2. PLAN: Map sub-questions to tools and create execution DAG
    3. EXECUTE: Run tools in dependency order with parallel execution
    4. SYNTHESIZE: Aggregate results into coherent response

    Example queries:
    - "Compare TRx trends across all brands and explain the causal factors"
    - "Show me the health score and recommendations for Kisqali"
    - "What caused the NRx drop and should we run an experiment?"

    Args:
        query: Multi-faceted query requiring decomposition
        brand: Brand context, resolved case-insensitively against the actual data values
        region: Region context, resolved case-insensitively against the actual data values
        session_id: Session ID for context continuity
        max_parallel: Maximum parallel tool executions (1-5)

    Returns:
        Dict with synthesized response from multiple agent outputs
    """
    logger.info(f"Tool composer: query={query[:50]}..., brand={brand}")

    try:
        # Build context for Tool Composer
        context: Dict[str, Any] = {
            "brand": brand,
            "region": region,
            "session_id": session_id or f"composer-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "max_parallel": max_parallel,
        }

        # Issue #810: KPI-aware data resolution. When the query targets a defined
        # KPI (e.g. "what drove <brand> conversion ..."), resolve the KPI's REAL
        # substrate (e.g. triggers ⋈ treatment_events for Conversion Rate) with
        # the KPI outcome materialized, rather than the patient-clinical cohort.
        # This takes precedence over the cohort frame for KPI-outcome questions.
        kpi_resolved = False
        try:
            kpi = kpi_resolution.recognize_kpi(query)
            if kpi is not None:
                kpi_frame = kpi_resolution.resolve_kpi_frame(kpi, brand, region)
                if kpi_frame is not None:
                    context["estimation_data"] = kpi_frame.frame
                    context["kpi_outcome"] = kpi_frame.outcome_column
                    context["kpi_name"] = kpi_frame.kpi_name
                    context["kpi_driver_columns"] = kpi_frame.driver_columns
                    kpi_resolved = True
                    logger.info(
                        "Tool composer: resolved KPI '%s' substrate (%d rows, "
                        "outcome=%s, truncated=%s) for brand=%s region=%s",
                        kpi_frame.kpi_name,
                        len(kpi_frame.frame),
                        kpi_frame.outcome_column,
                        kpi_frame.is_truncated,
                        brand,
                        region,
                    )
        except Exception as kpi_err:
            logger.warning(
                f"Tool composer: KPI resolution failed, falling back to cohort: {kpi_err}"
            )

        # Rec#1a: resolve a REAL cohort DataFrame for (brand, region) and thread
        # it into the composer context under the canonical "estimation_data" key.
        # The executor auto-injects it into composable tool kwargs. Best-effort:
        # if the loader is unavailable or raises, log and proceed -- the tools
        # then fail-closed honestly rather than fabricating values. Skipped when a
        # KPI substrate already resolved (the KPI frame is the right outcome data).
        if not kpi_resolved:
            try:
                estimation_frame = _resolve_cohort_frame(brand, region, data_source)
                if estimation_frame is not None:
                    context["estimation_data"] = estimation_frame
                    logger.info(
                        "Tool composer: resolved estimation_data frame "
                        f"({len(estimation_frame)} rows) for brand={brand} region={region}"
                    )
            except Exception as resolve_err:
                logger.warning(
                    f"Tool composer: cohort resolution failed, proceeding without "
                    f"estimation_data: {resolve_err}"
                )

        # Get LLM client for Tool Composer (use reasoning tier for complex queries)
        llm_client = get_chat_llm(model_tier="reasoning", max_tokens=4096)

        # Use the compose_query convenience function
        result = await compose_query(query=query, llm_client=llm_client, context=context)

        # Extract composition results from Pydantic CompositionResult model
        # result.decomposition contains sub_questions, result.plan has execution info,
        # result.execution has tool outputs, result.response has synthesized answer
        return {
            # F6 fail-closed: surface the REAL composition outcome. A total tool
            # failure (0/N succeeded) yields result.success=False / status=FAILED;
            # do NOT re-promote it to a hardcoded success envelope. The honest
            # synthesized_response + confidence (0.0) already flow through below.
            "success": result.success,
            "status": result.status.value,
            "query": query,
            "sub_questions": [
                {"id": sq.id, "question": sq.question, "intent": sq.intent}
                for sq in result.decomposition.sub_questions
            ],
            "tools_executed": result.execution.tools_executed,
            "execution_order": result.plan.get_execution_order(),
            "parallel_groups": result.plan.parallel_groups,
            "synthesized_response": result.response.answer,
            "confidence": result.response.confidence,
            "agent_outputs": result.execution.get_all_outputs(),
            "context": {
                "brand": brand,
                "region": region,
                "session_id": context["session_id"],
            },
        }

    except Exception as e:
        logger.error(f"Tool composer failed: {e}")
        # Fallback to orchestrator for simpler processing
        try:
            logger.info("Tool composer fallback: attempting orchestrator")
            orchestrator = get_orchestrator()
            if orchestrator:
                fallback_result = await orchestrator.run(
                    {
                        "query": query,
                        "session_id": session_id
                        or f"fallback-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "user_context": {"brand": brand, "region": region},
                    }
                )
                return {
                    "success": True,
                    "fallback": True,
                    "fallback_reason": f"Tool composer error: {str(e)}",
                    "query": query,
                    "response": fallback_result.get("response_text", ""),
                    "confidence": fallback_result.get("response_confidence", 0.7),
                }
        except Exception as fallback_error:
            logger.error(f"Orchestrator fallback also failed: {fallback_error}")

        return {
            "success": False,
            "error": str(e),
            "query": query,
        }


# =============================================================================
# TOOL EXPORTS
# =============================================================================


# =============================================================================
# KPI ENGINE TOOL — compute a DEFINED KPI on demand (the registry's calculable KPIs)
# =============================================================================


class KpiCalculateInput(BaseModel):
    """Input schema for kpi_calculate_tool."""

    kpi_name: str = Field(
        description=(
            "The KPI to compute, e.g. 'NBRx' (new-to-brand Rx), 'TRx', 'NRx', "
            "'market share', 'conversion rate', 'ROI', 'HCP coverage'. Resolved "
            "against the defined KPI registry."
        )
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter (e.g. Remibrutinib, Fabhalta, Kisqali), case-insensitive.",
    )
    region: Optional[str] = Field(default=None, description="Optional region/territory filter.")
    window: Optional[str] = Field(
        default=None,
        description=(
            "Time window, e.g. 'last 3 months', 'Q1 2025', or "
            "'2025-01-01 to 2025-03-31'. Omit for the engine's default window "
            "(the most recent 30 days of available data)."
        ),
    )


# Reporting window per KPI id, MIRRORED from the vetted SQL in the kpi_query
# allowlist registry (database/migrations/044/066, re-anchored by 089).
# The WS3 KPIs count over a 30-day window that ends at the DATA FRONTIER
# (`MAX(<domain ts>)` -- migration 089), NOT wall-clock NOW(): the synthetic
# gold-standard substrate is calendar-fixed by design, so a NOW()-anchored
# window silently decays to an empty set (the 2026-07-03 "NBRx = 0.0"
# incident). The engine surfaces the concrete as-of date per answer as
# `data_through`; this static note names the window SEMANTICS and the domain
# the frontier belongs to. Surfacing the real window lets the chatbot CITE it
# instead of presenting the figure as "the last 30 calendar days". Only KPIs
# whose window we have verified against the registry are listed; for any other
# KPI the field is omitted (honest absence over a guessed period -- ROI stays
# out: its two source probes' frontiers diverge). KEEP IN SYNC with those
# migrations.
KPI_REPORTING_WINDOWS = {
    "WS3-BI-005": "most recent 30 days of prescription data",  # TRx
    "WS3-BI-006": "most recent 30 days of prescription data",  # NRx
    "WS3-BI-007": "most recent 30 days of prescription data",  # NBRx
    "WS3-BI-008": "most recent 30 days of prescription data",  # TRx share
    "WS3-BI-009": "most recent 30 days of trigger data",  # Conversion rate
}


def _kpi_result_to_response(
    kpi: Any, result: Any, *, brand: Optional[str] = None, region: Optional[str] = None
) -> Dict[str, Any]:
    """Map a ``KPIResult`` onto the chatbot tool response (pure; unit-tested, no DB).

    Surfaces ``data_source='synthetic'`` when the engine answered from the
    synthetic-gold substrate so the chatbot/FE badges the figure honestly rather
    than passing it off as real-world data.

    Echoes the ``brand``/``region`` the figure was computed for so the synthesizer
    can name them instead of re-asking. Copies the engine's window provenance
    (``window_requested``/``window_applied``/``window_status``) so the chatbot can
    state exactly which period the figure covers. The static ``reporting_window``
    note is included ONLY when ``window_status == 'default'`` (no custom window was
    applied); when a custom window was honored or was not applicable, the stale
    default-window note would contradict the real answer.

    ``data_through`` (migration 089): the frontier-anchored default windows end
    at the domain's latest data date, not wall-clock now -- the substrate is
    calendar-fixed by design. When the engine surfaced that as-of date (stashed
    into metadata context by the calculator), copy it up so the chatbot cites
    e.g. "30 days ending 2025-04-23" instead of implying recency. Honest
    absence when the engine did not report one.
    """
    if getattr(result, "error", None):
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "kpi_id": kpi.id,
            "kpi_name": kpi.name,
            "error": result.error,
        }
    metadata = getattr(result, "metadata", None) or {}
    include_synthetic = bool(metadata.get("include_synthetic"))
    window_status = getattr(result, "window_status", "default")
    response: Dict[str, Any] = {
        "success": True,
        "query_type": "kpi_calculate",
        "kpi_id": kpi.id,
        "kpi_name": kpi.name,
        "value": result.value,
        "status": result.status,
        "data_source": "synthetic" if include_synthetic else "database",
        "brand": brand,
        "region": region,
        "window_requested": getattr(result, "window_requested", None),
        "window_applied": getattr(result, "window_applied", None),
        "window_status": window_status,
    }
    data_through = (metadata.get("context") or {}).get("data_through")
    if data_through is not None:
        response["data_through"] = data_through
    if window_status == "default":
        window = KPI_REPORTING_WINDOWS.get(kpi.id)
        if window:
            response["reporting_window"] = window
    return response


@tool(args_schema=KpiCalculateInput)
async def kpi_calculate_tool(
    kpi_name: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    window: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a DEFINED KPI on demand via the KPI engine (the registry's calculable KPIs).

    Use this for "what is the <KPI> for <brand>?" questions — NBRx (new-to-brand
    Rx), TRx, NRx, market share, conversion rate, ROI, HCP coverage, etc. Unlike
    ``e2i_data_query_tool`` (which reads the materialized ``business_metrics``
    fixture and returns 0 for a KPI that isn't stored there), this RESOLVES the KPI
    name to its definition and CALCULATES it from the real substrate (e.g. NBRx =
    count of each patient's first-brand prescription over ``treatment_events``).

    TIME WINDOW: pass ``window`` (e.g. "last 3 months", "Q1 2025",
    "2025-01-01 to 2025-03-31") to compute the volume KPIs (TRx/NRx/NBRx) over
    that period. The engine reports back ``window_status`` ("applied" when the
    requested window was honored, "not_applicable" when the KPI has no time
    dimension, "default" when no window was requested), plus ``window_requested``
    and ``window_applied``. When no custom window applies, the engine's default
    window is disclosed via ``reporting_window`` — it is FRONTIER-ANCHORED
    (migration 089): "the most recent 30 days of data", ending at the domain's
    latest data date (``data_through``), NOT the last 30 calendar days. State
    the brand and the period your answer actually covers — cite ``data_through``
    when present, and do NOT imply a figure covers a period it does not.

    Args:
        kpi_name: the KPI to compute (resolved against the defined KPI registry).
        brand: optional brand filter (case-insensitive).
        region: optional region/territory filter.
        window: optional time window (rolling or absolute); omit for the
            engine's default window (most recent 30 days of data).

    Returns:
        Dict with success, kpi_id, kpi_name, value, status, data_source, brand,
        region, the window provenance fields, data_through (the as-of date the
        default window ends at, when the engine reports one), and (when no
        custom window applies) reporting_window.

    STATUS SEMANTICS: "good"/"warning"/"critical" = value vs the KPI's defined
    thresholds; "informational" = the KPI has NO target BY DESIGN (volume
    metrics like TRx/NRx/NBRx and causal effect sizes are tracked for
    trend/context, not scored) — the value is real, do not call it a problem;
    "unknown" = the status could not be evaluated (missing data or calculation
    error) — do not speculate about causes beyond any error field present.
    """
    kpi = kpi_resolution.recognize_kpi(kpi_name)
    if kpi is None:
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "error": f"'{kpi_name}' did not resolve to a defined KPI.",
            "hint": "Try a defined KPI like NBRx, TRx, NRx, market share, conversion rate, or ROI.",
        }

    # Parse the requested window BEFORE touching the calculator: an unparseable
    # window is a user-input error, not a calculation error, so fail fast with a
    # helpful hint and never hit the DB.
    from src.services.time_window import WindowParseError, parse_window

    try:
        parsed = parse_window(window)
    except WindowParseError as e:
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "error": str(e),
            "hint": "Try 'last 3 months', 'Q1 2025', or '2025-01-01 to 2025-03-31'.",
        }

    context: Dict[str, Any] = {}
    if brand:
        context["brand"] = brand
    if region:
        # KPI calculators read ``context.get("region")`` (business_impact /
        # trigger_performance / data_quality); "territory" was a dead key, so a
        # region filter silently dropped -> region-agnostic windowed query while
        # the response still echoed the region. Pass the key the engine reads.
        context["region"] = region
    if parsed is not None:
        context["window"] = parsed.as_dict()

    try:
        # Local import avoids a chatbot_tools <-> kpi route import cycle at load.
        from src.api.routes.kpi import get_kpi_calculator

        calculator = get_kpi_calculator()
        # calculate() is synchronous (a DB RPC) -> off-load to a worker thread so
        # the chatbot event loop is never blocked (mirrors the cognitive-RAG fix).
        result = await asyncio.to_thread(calculator.calculate, kpi.id, context=context)
    except Exception as exc:  # noqa: BLE001 - surface as a tool error, never fabricate
        logger.error("kpi_calculate_tool: calculation failed for %s: %s", kpi.id, exc)
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "kpi_id": kpi.id,
            "kpi_name": kpi.name,
            "error": str(exc),
        }

    return _kpi_result_to_response(kpi, result, brand=brand, region=region)


# List of all E2I chatbot tools for use in LangGraph ToolNode
E2I_CHATBOT_TOOLS = [
    e2i_data_query_tool,
    kpi_calculate_tool,
    causal_analysis_tool,
    agent_routing_tool,
    conversation_memory_tool,
    document_retrieval_tool,
    orchestrator_tool,
    tool_composer_tool,
]

# Tool name to function mapping
E2I_TOOL_MAP = {
    "e2i_data_query_tool": e2i_data_query_tool,
    "kpi_calculate_tool": kpi_calculate_tool,
    "causal_analysis_tool": causal_analysis_tool,
    "agent_routing_tool": agent_routing_tool,
    "conversation_memory_tool": conversation_memory_tool,
    "document_retrieval_tool": document_retrieval_tool,
    "orchestrator_tool": orchestrator_tool,
    "tool_composer_tool": tool_composer_tool,
}


def get_e2i_chatbot_tools() -> List:
    """Get list of all E2I chatbot tools for LangGraph integration."""
    return E2I_CHATBOT_TOOLS


def get_tool_by_name(name: str):
    """Get a specific tool by name."""
    return E2I_TOOL_MAP.get(name)
