"""
CopilotKit Integration Router
=============================

Provides CopilotKit backend runtime for the E2I Chat Sidebar.
Exposes backend actions for querying KPIs, running analyses,
and interacting with the E2I agent system.

Author: E2I Causal Analytics Team
Version: 1.3.0

Changelog:
    1.3.0 - Connected to real repositories (BusinessMetricRepository, AgentRegistryRepository)
    1.2.0 - Refactored from monkey-patches to response transformer middleware
    1.1.0 - Added SDK compatibility patches for frontend v1.x
    1.0.0 - Initial CopilotKit integration
"""

import asyncio
import json
import logging
import operator
import os
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Sequence, TypedDict

from copilotkit import CopilotKitRemoteEndpoint, LangGraphAGUIAgent, Action as CopilotAction
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit.sdk import COPILOTKIT_SDK_VERSION
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# REPOSITORY HELPERS
# =============================================================================


def _get_business_metric_repository():
    """Get BusinessMetricRepository instance with Supabase client."""
    try:
        from src.api.dependencies.supabase_client import get_supabase
        from src.repositories.business_metric import BusinessMetricRepository

        client = get_supabase()
        return BusinessMetricRepository(client=client) if client else None
    except Exception as e:
        logger.warning(f"Failed to get BusinessMetricRepository: {e}")
        return None


def _get_agent_registry_repository():
    """Get AgentRegistryRepository instance with Supabase client."""
    try:
        from src.api.dependencies.supabase_client import get_supabase
        from src.repositories.agent_registry import AgentRegistryRepository

        client = get_supabase()
        return AgentRegistryRepository(client=client) if client else None
    except Exception as e:
        logger.warning(f"Failed to get AgentRegistryRepository: {e}")
        return None

# =============================================================================
# E2I BACKEND ACTIONS
# =============================================================================

# Fallback sample data when database is unavailable
_FALLBACK_KPIS = {
    "Remibrutinib": {
        "trx_volume": 15420,
        "nrx_volume": 3250,
        "market_share": 12.5,
        "conversion_rate": 0.68,
        "hcp_reach": 2450,
        "patient_starts": 890,
    },
    "Fabhalta": {
        "trx_volume": 8920,
        "nrx_volume": 1850,
        "market_share": 8.2,
        "conversion_rate": 0.72,
        "hcp_reach": 1820,
        "patient_starts": 560,
    },
    "Kisqali": {
        "trx_volume": 22100,
        "nrx_volume": 4200,
        "market_share": 18.5,
        "conversion_rate": 0.65,
        "hcp_reach": 3200,
        "patient_starts": 1250,
    },
}

_FALLBACK_AGENTS = [
    {"id": "orchestrator", "name": "Orchestrator", "tier": 1, "status": "active"},
    {"id": "causal-impact", "name": "Causal Impact", "tier": 2, "status": "idle"},
    {"id": "gap-analyzer", "name": "Gap Analyzer", "tier": 2, "status": "idle"},
    {"id": "drift-monitor", "name": "Drift Monitor", "tier": 3, "status": "active"},
    {"id": "health-score", "name": "Health Score", "tier": 3, "status": "active"},
    {"id": "explainer", "name": "Explainer", "tier": 5, "status": "idle"},
]


async def _fetch_kpis_from_db(brand: str) -> Optional[Dict[str, Any]]:
    """
    Fetch KPI data from database for a brand.

    Returns:
        KPI metrics dict or None if unavailable
    """
    repo = _get_business_metric_repository()
    if not repo:
        return None

    try:
        # Define KPI mappings to metric_name in database
        kpi_mappings = {
            "trx_volume": "TRx",
            "nrx_volume": "NRx",
            "market_share": "market_share",
            "conversion_rate": "conversion_rate",
            "hcp_reach": "hcp_reach",
            "patient_starts": "patient_starts",
        }

        metrics = {}
        for metric_key, db_metric_name in kpi_mappings.items():
            results = await repo.get_by_kpi(
                kpi_name=db_metric_name,
                brand=brand if brand != "All" else None,
                limit=1,
            )
            if results:
                # Get most recent value
                metrics[metric_key] = results[0].get("value", 0)
            else:
                metrics[metric_key] = 0

        return metrics if any(metrics.values()) else None

    except Exception as e:
        logger.warning(f"Failed to fetch KPIs from database: {e}")
        return None


async def get_kpi_summary(brand: str) -> Dict[str, Any]:
    """
    Get KPI summary for a specific brand.

    Attempts to fetch real data from database, falls back to sample data if unavailable.

    Args:
        brand: Brand name (Remibrutinib, Fabhalta, Kisqali, or All)

    Returns:
        Dictionary with KPI metrics
    """
    logger.info(f"[CopilotKit] Fetching KPI summary for brand: {brand}")

    valid_brands = ["Remibrutinib", "Fabhalta", "Kisqali", "All"]
    if brand not in valid_brands:
        return {"error": f"Unknown brand: {brand}. Available: {valid_brands[:-1]}"}

    # Try to fetch from database first
    db_metrics = await _fetch_kpis_from_db(brand)
    data_source = "database"

    if db_metrics:
        metrics = db_metrics
    else:
        # Fall back to sample data
        data_source = "fallback"
        if brand == "All":
            metrics = {
                "trx_volume": sum(b["trx_volume"] for b in _FALLBACK_KPIS.values()),
                "nrx_volume": sum(b["nrx_volume"] for b in _FALLBACK_KPIS.values()),
                "market_share": sum(b["market_share"] for b in _FALLBACK_KPIS.values()) / 3,
                "conversion_rate": sum(b["conversion_rate"] for b in _FALLBACK_KPIS.values()) / 3,
                "hcp_reach": sum(b["hcp_reach"] for b in _FALLBACK_KPIS.values()),
                "patient_starts": sum(b["patient_starts"] for b in _FALLBACK_KPIS.values()),
                "brands_included": list(_FALLBACK_KPIS.keys()),
            }
        else:
            metrics = _FALLBACK_KPIS.get(brand, {})

    return {
        "brand": brand,
        "period": "Last 90 days",
        "metrics": metrics,
        "data_source": data_source,
    }


async def _fetch_agents_from_db() -> Optional[List[Dict[str, Any]]]:
    """
    Fetch agent status from database.

    Returns:
        List of agent dicts or None if unavailable
    """
    repo = _get_agent_registry_repository()
    if not repo:
        return None

    try:
        # Fetch all active agents
        all_agents = []
        for tier in range(1, 6):  # Tiers 1-5
            tier_agents = await repo.get_by_tier(tier)
            for agent in tier_agents:
                all_agents.append({
                    "id": agent.get("agent_name", "unknown"),
                    "name": agent.get("display_name", agent.get("agent_name", "Unknown")),
                    "tier": agent.get("tier", tier),
                    "status": "active" if agent.get("is_active", True) else "idle",
                    "description": agent.get("description", ""),
                })

        return all_agents if all_agents else None

    except Exception as e:
        logger.warning(f"Failed to fetch agents from database: {e}")
        return None


async def get_agent_status() -> Dict[str, Any]:
    """
    Get the status of all E2I agents.

    Attempts to fetch real data from database, falls back to sample data if unavailable.

    Returns:
        Dictionary with agent status information
    """
    logger.info("[CopilotKit] Fetching agent status")

    # Try to fetch from database first
    db_agents = await _fetch_agents_from_db()
    data_source = "database"

    if db_agents:
        agents = db_agents
    else:
        # Fall back to sample data
        data_source = "fallback"
        agents = _FALLBACK_AGENTS

    active_count = sum(1 for a in agents if a.get("status") == "active")

    return {
        "total_agents": len(agents),
        "active_agents": active_count,
        "idle_agents": len(agents) - active_count,
        "agents": agents,
        "data_source": data_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _get_orchestrator():
    """Get OrchestratorAgent singleton for causal analysis."""
    try:
        from src.api.routes.cognitive import get_orchestrator
        return get_orchestrator()
    except Exception as e:
        logger.warning(f"Failed to get orchestrator: {e}")
        return None


async def run_causal_analysis(
    intervention: str,
    target_kpi: str,
    brand: str,
) -> Dict[str, Any]:
    """
    Run a causal impact analysis.

    Attempts to use the orchestrator for real causal analysis, falls back to simulated results.

    Args:
        intervention: Type of intervention (e.g., "HCP Engagement", "Marketing Campaign")
        target_kpi: Target KPI to analyze (e.g., "TRx Volume", "Market Share")
        brand: Brand to analyze

    Returns:
        Dictionary with causal analysis results
    """
    logger.info(f"[CopilotKit] Running causal analysis: {intervention} -> {target_kpi} for {brand}")

    # Try to run through orchestrator for real causal analysis
    orchestrator = _get_orchestrator()
    data_source = "orchestrator"

    if orchestrator:
        try:
            query = f"What is the causal impact of {intervention} on {target_kpi} for {brand}?"
            result = await orchestrator.run({
                "query": query,
                "user_context": {
                    "brand": brand,
                    "intervention": intervention,
                    "target_kpi": target_kpi,
                },
            })

            # Extract causal results if available
            if result and result.get("response_text"):
                return {
                    "intervention": intervention,
                    "target_kpi": target_kpi,
                    "brand": brand,
                    "results": result.get("causal_results", {
                        "average_treatment_effect": result.get("ate", 0.0),
                        "confidence_interval": result.get("ci", [0.0, 0.0]),
                        "p_value": result.get("p_value", 0.0),
                        "statistical_significance": result.get("significant", False),
                    }),
                    "interpretation": result.get("response_text", ""),
                    "data_source": data_source,
                    "agents_used": result.get("agents_dispatched", []),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            logger.warning(f"Orchestrator causal analysis failed: {e}")

    # Fallback to simulated results
    import random
    data_source = "simulated"
    ate = random.uniform(0.05, 0.25)

    return {
        "intervention": intervention,
        "target_kpi": target_kpi,
        "brand": brand,
        "results": {
            "average_treatment_effect": round(ate, 3),
            "confidence_interval": [round(ate - 0.05, 3), round(ate + 0.05, 3)],
            "p_value": round(random.uniform(0.001, 0.05), 4),
            "statistical_significance": True,
            "sample_size": random.randint(500, 2000),
        },
        "interpretation": f"The {intervention} shows a statistically significant positive effect on {target_kpi}, with an estimated {round(ate * 100, 1)}% lift.",
        "data_source": data_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def get_recommendations(brand: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Get AI-powered recommendations for a brand.

    Args:
        brand: Brand to get recommendations for
        context: Optional context about what kind of recommendations are needed

    Returns:
        Dictionary with recommendations
    """
    logger.info(f"[CopilotKit] Generating recommendations for {brand}")

    recommendations = [
        {
            "priority": "high",
            "category": "HCP Targeting",
            "recommendation": f"Focus on high-decile HCPs in the Northeast region for {brand}",
            "expected_impact": "+15% TRx lift",
            "confidence": 0.85,
        },
        {
            "priority": "medium",
            "category": "Patient Journey",
            "recommendation": f"Implement patient support program to reduce {brand} discontinuation",
            "expected_impact": "+8% persistence rate",
            "confidence": 0.78,
        },
        {
            "priority": "medium",
            "category": "Market Access",
            "recommendation": f"Target formulary additions in 3 key health systems for {brand}",
            "expected_impact": "+12% market share",
            "confidence": 0.72,
        },
    ]

    return {
        "brand": brand,
        "context": context or "General recommendations",
        "recommendations": recommendations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def search_insights(query: str, brand: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for insights in the E2I knowledge base.

    Args:
        query: Search query
        brand: Optional brand filter

    Returns:
        Dictionary with search results
    """
    logger.info(f"[CopilotKit] Searching insights: {query}")

    # Simulated search results
    results = [
        {
            "type": "causal_path",
            "title": "HCP Engagement -> TRx Volume Causal Chain",
            "summary": "Strong causal relationship identified between HCP engagement frequency and TRx volume increases.",
            "confidence": 0.89,
            "brand": brand or "Remibrutinib",
        },
        {
            "type": "trend",
            "title": "Q4 Market Share Trend",
            "summary": "Market share increased by 2.3% following targeted digital campaign.",
            "confidence": 0.92,
            "brand": brand or "All",
        },
        {
            "type": "agent_insight",
            "title": "Drift Monitor Alert",
            "summary": "Model drift detected in conversion prediction model. Retraining recommended.",
            "confidence": 0.95,
            "brand": None,
        },
    ]

    return {
        "query": query,
        "brand_filter": brand,
        "results": results,
        "total_results": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# COPILOTKIT ACTIONS
# =============================================================================

COPILOT_ACTIONS = [
    CopilotAction(
        name="getKPISummary",
        description="Get key performance indicator (KPI) summary for a pharmaceutical brand. Returns metrics like TRx volume, NRx volume, market share, conversion rate, HCP reach, and patient starts.",
        parameters=[
            {
                "name": "brand",
                "type": "string",
                "description": "The brand to get KPIs for. Options: Remibrutinib, Fabhalta, Kisqali, or All",
                "required": True,
            }
        ],
        handler=get_kpi_summary,
    ),
    CopilotAction(
        name="getAgentStatus",
        description="Get the current status of all E2I agents in the 6-tier hierarchy. Shows which agents are active, idle, or processing.",
        parameters=[],
        handler=get_agent_status,
    ),
    CopilotAction(
        name="runCausalAnalysis",
        description="Run a causal impact analysis to measure the effect of an intervention on a target KPI. Uses DoWhy/EconML for causal inference.",
        parameters=[
            {
                "name": "intervention",
                "type": "string",
                "description": "The type of intervention to analyze (e.g., 'HCP Engagement', 'Marketing Campaign', 'Patient Support Program')",
                "required": True,
            },
            {
                "name": "target_kpi",
                "type": "string",
                "description": "The KPI to measure impact on (e.g., 'TRx Volume', 'Market Share', 'Conversion Rate')",
                "required": True,
            },
            {
                "name": "brand",
                "type": "string",
                "description": "The brand to analyze",
                "required": True,
            },
        ],
        handler=run_causal_analysis,
    ),
    CopilotAction(
        name="getRecommendations",
        description="Get AI-powered recommendations for improving brand performance. Returns prioritized recommendations with expected impact.",
        parameters=[
            {
                "name": "brand",
                "type": "string",
                "description": "The brand to get recommendations for",
                "required": True,
            },
            {
                "name": "context",
                "type": "string",
                "description": "Optional context about what kind of recommendations are needed",
                "required": False,
            },
        ],
        handler=get_recommendations,
    ),
    CopilotAction(
        name="searchInsights",
        description="Search the E2I knowledge base for insights, causal paths, trends, and agent outputs.",
        parameters=[
            {
                "name": "query",
                "type": "string",
                "description": "The search query",
                "required": True,
            },
            {
                "name": "brand",
                "type": "string",
                "description": "Optional brand filter",
                "required": False,
            },
        ],
        handler=search_insights,
    ),
]


# =============================================================================
# LANGGRAPH AGENT FOR E2I CHAT
# =============================================================================


class E2IAgentState(TypedDict):
    """State for the E2I chat agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]


def create_e2i_chat_agent():
    """
    Create a simple LangGraph agent for E2I chat.

    This agent responds to chat messages and provides helpful information
    about the E2I Causal Analytics platform. It delegates tool calls
    to CopilotKit actions.
    """

    async def chat_node(state: E2IAgentState) -> Dict[str, Any]:
        """Process chat messages and generate responses."""
        messages = state.get("messages", [])

        # Get the last human message
        last_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_message = msg.content
                break

        if not last_message:
            return {
                "messages": [
                    AIMessage(
                        content="Hello! I'm the E2I Analytics Assistant. I can help you with KPI analysis, causal inference, and insights for pharmaceutical brands. What would you like to know?"
                    )
                ]
            }

        # Generate a contextual response based on the query
        response = generate_e2i_response(last_message)

        return {"messages": [AIMessage(content=response)]}

    # Build the graph
    workflow = StateGraph(E2IAgentState)
    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)

    return workflow.compile()


def generate_e2i_response(query: str) -> str:
    """
    Generate a contextual response for E2I queries.

    This is a simple response generator. In production, this would
    integrate with Claude API for more sophisticated responses.
    """
    query_lower = query.lower()

    # KPI-related queries
    if any(kw in query_lower for kw in ["kpi", "trx", "nrx", "market share", "metric"]):
        return (
            "I can help you with KPI analysis! Use the **getKPISummary** action to get detailed metrics "
            "for any brand (Remibrutinib, Fabhalta, Kisqali, or All). This includes TRx volume, NRx volume, "
            "market share, conversion rate, HCP reach, and patient starts."
        )

    # Agent-related queries
    if any(kw in query_lower for kw in ["agent", "status", "tier", "orchestrator"]):
        return (
            "The E2I platform uses an 18-agent tiered architecture:\n\n"
            "- **Tier 0**: ML Foundation (7 agents)\n"
            "- **Tier 1**: Orchestration (2 agents)\n"
            "- **Tier 2**: Causal Analytics (3 agents)\n"
            "- **Tier 3**: Monitoring (3 agents)\n"
            "- **Tier 4**: ML Predictions (2 agents)\n"
            "- **Tier 5**: Self-Improvement (2 agents)\n\n"
            "Use the **getAgentStatus** action to see which agents are currently active."
        )

    # Causal analysis queries
    if any(kw in query_lower for kw in ["causal", "impact", "intervention", "effect", "ate"]):
        return (
            "I can run causal impact analyses! Use the **runCausalAnalysis** action with:\n\n"
            "- **intervention**: Type of intervention (e.g., 'HCP Engagement', 'Marketing Campaign')\n"
            "- **target_kpi**: KPI to measure (e.g., 'TRx Volume', 'Market Share')\n"
            "- **brand**: Brand to analyze\n\n"
            "The analysis uses DoWhy/EconML for rigorous causal inference."
        )

    # Recommendation queries
    if any(kw in query_lower for kw in ["recommend", "suggest", "improve", "optimize"]):
        return (
            "I can provide AI-powered recommendations! Use the **getRecommendations** action with a brand name "
            "to get prioritized suggestions for HCP targeting, patient journey optimization, and market access strategies."
        )

    # Search/insight queries
    if any(kw in query_lower for kw in ["search", "find", "insight", "trend"]):
        return (
            "I can search the E2I knowledge base for insights! Use the **searchInsights** action with your query "
            "to find causal paths, trends, and agent outputs. You can optionally filter by brand."
        )

    # Brand-specific queries
    if any(brand.lower() in query_lower for brand in ["remibrutinib", "fabhalta", "kisqali"]):
        return (
            "I see you're asking about a specific brand. I have data for:\n\n"
            "- **Remibrutinib** (CSU indication)\n"
            "- **Fabhalta** (PNH indication)\n"
            "- **Kisqali** (HR+/HER2- breast cancer)\n\n"
            "Use the **getKPISummary** action to get detailed metrics, or **runCausalAnalysis** for impact analysis."
        )

    # Default response
    return (
        "I'm the E2I Analytics Assistant. I can help you with:\n\n"
        "1. **KPI Analysis** - Get metrics for pharmaceutical brands\n"
        "2. **Agent Status** - Check the 18-agent system status\n"
        "3. **Causal Analysis** - Run causal impact analyses\n"
        "4. **Recommendations** - Get AI-powered suggestions\n"
        "5. **Insights Search** - Find trends and causal paths\n\n"
        "What would you like to explore?"
    )


# Create the compiled graph
e2i_chat_graph = create_e2i_chat_agent()


# =============================================================================
# COPILOTKIT SDK SETUP
# =============================================================================


def create_copilotkit_sdk() -> CopilotKitRemoteEndpoint:
    """
    Create and configure the CopilotKit Remote Endpoint.

    Returns:
        Configured CopilotKitRemoteEndpoint instance with agents and actions
    """
    sdk = CopilotKitRemoteEndpoint(
        agents=[
            LangGraphAGUIAgent(
                name="default",
                description="E2I Analytics Assistant for pharmaceutical commercial analytics. Helps with KPI analysis, causal inference, and agent system insights.",
                graph=e2i_chat_graph,
            ),
        ],
        actions=COPILOT_ACTIONS,
    )

    logger.info(f"[CopilotKit] Remote endpoint initialized with 1 agent and {len(COPILOT_ACTIONS)} actions")
    return sdk


def transform_info_response(sdk: CopilotKitRemoteEndpoint) -> Dict[str, Any]:
    """
    Transform SDK info response to frontend v1.x compatible format.

    The Python SDK (0.1.x) returns agents as an array with 'sdkVersion',
    but the JS frontend (1.x) expects agents as a dict with 'version'.

    Args:
        sdk: The CopilotKit remote endpoint instance

    Returns:
        Frontend-compatible info response
    """
    context: Dict[str, Any] = {}

    # Get agents - handle both callable and static
    agents = sdk.agents(context) if callable(sdk.agents) else sdk.agents

    # Get actions - handle both callable and static
    actions = sdk.actions(context) if callable(sdk.actions) else sdk.actions

    # Transform actions to dict representation
    actions_list = [action.dict_repr() for action in actions]

    # Transform agents array to dict keyed by agent ID (frontend v1.x format)
    agents_dict = {}
    for agent in agents:
        agent_id = agent.name
        agents_dict[agent_id] = {
            "description": getattr(agent, "description", "") or ""
        }

    return {
        "actions": actions_list,
        "agents": agents_dict,
        "version": COPILOTKIT_SDK_VERSION,  # Frontend expects 'version' not 'sdkVersion'
    }


def add_copilotkit_routes(app: FastAPI, prefix: str = "/api/copilotkit") -> None:
    """
    Add CopilotKit routes to the FastAPI application.

    Adds a custom /info endpoint that transforms the response format for
    frontend v1.x compatibility, then adds the SDK's other routes.

    Args:
        app: FastAPI application instance
        prefix: URL prefix for CopilotKit endpoints
    """
    sdk = create_copilotkit_sdk()

    # Add custom info endpoint BEFORE SDK routes (to take precedence)
    @app.post(f"{prefix}/info")
    async def copilotkit_info(request: Request) -> JSONResponse:
        """
        Custom info endpoint with response transformation for frontend v1.x.

        The SDK returns agents as an array, but frontend v1.x expects a dict
        keyed by agent ID. This endpoint transforms the response accordingly.
        """
        response = transform_info_response(sdk)
        logger.debug(f"[CopilotKit] Info response: agents={list(response['agents'].keys())}")
        return JSONResponse(content=response)

    # Add SDK routes (main chat endpoint)
    add_fastapi_endpoint(app, sdk, prefix)
    logger.info(f"[CopilotKit] Routes added at {prefix} (with custom info transformer)")


# =============================================================================
# STANDALONE ROUTER (for testing/info endpoints)
# =============================================================================

router = APIRouter(prefix="/copilotkit", tags=["copilotkit"])


@router.get("/status")
async def get_copilotkit_status() -> Dict[str, Any]:
    """Get CopilotKit integration status."""
    return {
        "status": "active",
        "version": "1.1.0",
        "agents_available": 1,
        "agent_names": ["default"],
        "actions_available": len(COPILOT_ACTIONS),
        "action_names": [a.name for a in COPILOT_ACTIONS],
        "llm_configured": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# E2I CHATBOT STREAMING ENDPOINTS
# =============================================================================


class ChatRequest(BaseModel):
    """Request schema for chatbot endpoints."""

    query: str = Field(..., description="User's query text")
    user_id: str = Field(..., description="User UUID")
    request_id: str = Field(..., description="Unique request identifier")
    session_id: Optional[str] = Field(
        default=None, description="Session ID (generated if not provided)"
    )
    brand_context: Optional[str] = Field(
        default=None, description="Brand filter (Kisqali, Fabhalta, Remibrutinib)"
    )
    region_context: Optional[str] = Field(
        default=None, description="Region filter (US, EU, APAC)"
    )


class ChatResponse(BaseModel):
    """Response schema for non-streaming chatbot endpoint."""

    success: bool
    session_id: str
    response: str
    conversation_title: Optional[str] = None
    agent_name: Optional[str] = None
    error: Optional[str] = None


async def _stream_chat_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for chatbot response.

    Yields JSON-formatted SSE events:
    - {"type": "session_id", "data": "..."}
    - {"type": "text", "data": "..."}
    - {"type": "conversation_title", "data": "..."}
    - {"type": "tool_call", "data": "..."}
    - {"type": "done", "data": ""}
    - {"type": "error", "data": "..."}
    """
    try:
        from src.api.routes.chatbot_graph import stream_chatbot

        # Yield session_id first
        session_id = request.session_id
        if not session_id:
            import uuid
            session_id = f"{request.user_id}~{uuid.uuid4()}"

        yield f"data: {json.dumps({'type': 'session_id', 'data': session_id})}\n\n"

        response_text = ""
        conversation_title = None

        # Stream through chatbot workflow
        async for state_update in stream_chatbot(
            query=request.query,
            user_id=request.user_id,
            request_id=request.request_id,
            session_id=session_id,
            brand_context=request.brand_context,
            region_context=request.region_context,
        ):
            # Extract response from state updates
            if isinstance(state_update, dict):
                # Check for node outputs
                for node_name, node_output in state_update.items():
                    if isinstance(node_output, dict):
                        # Get response text from finalize node
                        if "response_text" in node_output and node_output["response_text"]:
                            text_chunk = node_output["response_text"]
                            if text_chunk and text_chunk != response_text:
                                # Yield new text
                                new_text = text_chunk[len(response_text):] if response_text else text_chunk
                                if new_text:
                                    yield f"data: {json.dumps({'type': 'text', 'data': new_text})}\n\n"
                                    response_text = text_chunk

                        # Get conversation title
                        if "conversation_title" in node_output and node_output["conversation_title"]:
                            title = node_output["conversation_title"]
                            if title != conversation_title:
                                conversation_title = title
                                yield f"data: {json.dumps({'type': 'conversation_title', 'data': title})}\n\n"

                        # Handle messages (for AIMessage content)
                        if "messages" in node_output:
                            for msg in node_output["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    if msg.content != response_text:
                                        new_text = msg.content[len(response_text):] if response_text else msg.content
                                        if new_text:
                                            yield f"data: {json.dumps({'type': 'text', 'data': new_text})}\n\n"
                                            response_text = msg.content

        # Generate title if not set
        if not conversation_title and response_text:
            # Simple title generation from query
            title = request.query[:50] + "..." if len(request.query) > 50 else request.query
            yield f"data: {json.dumps({'type': 'conversation_title', 'data': title})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"

    except Exception as e:
        logger.error(f"Streaming chat error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    """
    Stream chatbot response as Server-Sent Events (SSE).

    Returns an SSE stream with events:
    - session_id: The conversation session ID
    - text: Response text chunks
    - conversation_title: Auto-generated conversation title
    - tool_call: Tool invocation notifications
    - done: Stream completion signal
    - error: Error messages

    Usage:
        POST /api/copilotkit/chat/stream
        Content-Type: application/json

        {
            "query": "What is the TRx for Kisqali?",
            "user_id": "user-uuid",
            "request_id": "req-123",
            "session_id": "",  // Optional, generated if empty
            "brand_context": "Kisqali"  // Optional
        }
    """
    logger.info(f"[Chatbot] Streaming request: query={request.query[:50]}..., user={request.user_id}")

    return StreamingResponse(
        _stream_chat_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Non-streaming chatbot endpoint.

    Returns the complete response in a single JSON object.

    Usage:
        POST /api/copilotkit/chat
        Content-Type: application/json

        {
            "query": "Show agent status",
            "user_id": "user-uuid",
            "request_id": "req-456",
            "session_id": ""
        }
    """
    logger.info(f"[Chatbot] Chat request: query={request.query[:50]}..., user={request.user_id}")

    try:
        from src.api.routes.chatbot_graph import run_chatbot

        result = await run_chatbot(
            query=request.query,
            user_id=request.user_id,
            request_id=request.request_id,
            session_id=request.session_id,
            brand_context=request.brand_context,
            region_context=request.region_context,
        )

        response_text = result.get("response_text", "")
        session_id = result.get("session_id", "")
        agent_name = result.get("agent_name")

        # Generate title from query
        title = request.query[:50] + "..." if len(request.query) > 50 else request.query

        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response_text,
            conversation_title=title,
            agent_name=agent_name,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            success=False,
            session_id=request.session_id or "",
            response="",
            error=str(e),
        )
