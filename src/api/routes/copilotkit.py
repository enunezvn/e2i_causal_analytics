"""
CopilotKit Integration Router
=============================

Provides CopilotKit backend runtime for the E2I Chat Sidebar.
Exposes backend actions for querying KPIs, running analyses,
and interacting with the E2I agent system.

Author: E2I Causal Analytics Team
Version: 1.1.0
"""

import logging
import operator
import os
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from copilotkit import CopilotKitRemoteEndpoint, LangGraphAGUIAgent, Action as CopilotAction
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from fastapi import APIRouter, FastAPI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# =============================================================================
# MONKEY-PATCH: Fix missing dict_repr on LangGraphAGUIAgent
# This is a workaround for a bug in copilotkit where LangGraphAGUIAgent
# doesn't inherit from copilotkit.Agent and lacks dict_repr() method.
# =============================================================================
if not hasattr(LangGraphAGUIAgent, 'dict_repr'):
    def _dict_repr(self):
        """Dict representation of the agent for CopilotKit info endpoint."""
        return {
            'id': self.name,  # Frontend v1.x expects 'id' field
            'name': self.name,
            'description': self.description or ''
        }
    LangGraphAGUIAgent.dict_repr = _dict_repr
    logger.info("[CopilotKit] Applied dict_repr monkey-patch to LangGraphAGUIAgent")

# =============================================================================
# MONKEY-PATCH: Fix info() to return agents as dict (not list)
# Frontend v1.x expects agents as {agentId: {description: "..."}} not [{id, name, description}]
# =============================================================================
from copilotkit.sdk import CopilotKitRemoteEndpoint, COPILOTKIT_SDK_VERSION
_original_info = CopilotKitRemoteEndpoint.info

def _patched_info(self, *, context):
    """Patched info method that returns agents as a dict keyed by agent ID."""
    actions = self.actions(context) if callable(self.actions) else self.actions
    agents = self.agents(context) if callable(self.agents) else self.agents

    actions_list = [action.dict_repr() for action in actions]

    # Convert agents list to dict keyed by agent ID (for frontend v1.x compatibility)
    agents_dict = {}
    for agent in agents:
        agent_repr = agent.dict_repr()
        agent_id = agent_repr.get('id') or agent_repr.get('name')
        agents_dict[agent_id] = {'description': agent_repr.get('description', '')}

    return {
        "actions": actions_list,
        "agents": agents_dict,
        "version": COPILOTKIT_SDK_VERSION  # Frontend expects 'version' not 'sdkVersion'
    }

CopilotKitRemoteEndpoint.info = _patched_info
logger.info("[CopilotKit] Applied info() monkey-patch for frontend v1.x compatibility")

# =============================================================================
# E2I BACKEND ACTIONS
# =============================================================================

# Sample KPI data for demonstration (would connect to real DB in production)
SAMPLE_KPIS = {
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

SAMPLE_AGENTS = [
    {"id": "orchestrator", "name": "Orchestrator", "tier": 1, "status": "active"},
    {"id": "causal-impact", "name": "Causal Impact", "tier": 2, "status": "idle"},
    {"id": "gap-analyzer", "name": "Gap Analyzer", "tier": 2, "status": "idle"},
    {"id": "drift-monitor", "name": "Drift Monitor", "tier": 3, "status": "active"},
    {"id": "health-score", "name": "Health Score", "tier": 3, "status": "active"},
    {"id": "explainer", "name": "Explainer", "tier": 5, "status": "idle"},
]


async def get_kpi_summary(brand: str) -> Dict[str, Any]:
    """
    Get KPI summary for a specific brand.

    Args:
        brand: Brand name (Remibrutinib, Fabhalta, Kisqali, or All)

    Returns:
        Dictionary with KPI metrics
    """
    logger.info(f"[CopilotKit] Fetching KPI summary for brand: {brand}")

    if brand == "All":
        # Aggregate all brands
        total = {
            "trx_volume": sum(b["trx_volume"] for b in SAMPLE_KPIS.values()),
            "nrx_volume": sum(b["nrx_volume"] for b in SAMPLE_KPIS.values()),
            "market_share": sum(b["market_share"] for b in SAMPLE_KPIS.values()) / 3,
            "conversion_rate": sum(b["conversion_rate"] for b in SAMPLE_KPIS.values()) / 3,
            "hcp_reach": sum(b["hcp_reach"] for b in SAMPLE_KPIS.values()),
            "patient_starts": sum(b["patient_starts"] for b in SAMPLE_KPIS.values()),
            "brands_included": list(SAMPLE_KPIS.keys()),
        }
        return {"brand": "All", "period": "Last 90 days", "metrics": total}

    if brand not in SAMPLE_KPIS:
        return {"error": f"Unknown brand: {brand}. Available: {list(SAMPLE_KPIS.keys())}"}

    return {
        "brand": brand,
        "period": "Last 90 days",
        "metrics": SAMPLE_KPIS[brand],
    }


async def get_agent_status() -> Dict[str, Any]:
    """
    Get the status of all E2I agents.

    Returns:
        Dictionary with agent status information
    """
    logger.info("[CopilotKit] Fetching agent status")

    active_count = sum(1 for a in SAMPLE_AGENTS if a["status"] == "active")

    return {
        "total_agents": len(SAMPLE_AGENTS),
        "active_agents": active_count,
        "idle_agents": len(SAMPLE_AGENTS) - active_count,
        "agents": SAMPLE_AGENTS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def run_causal_analysis(
    intervention: str,
    target_kpi: str,
    brand: str,
) -> Dict[str, Any]:
    """
    Run a causal impact analysis.

    Args:
        intervention: Type of intervention (e.g., "HCP Engagement", "Marketing Campaign")
        target_kpi: Target KPI to analyze (e.g., "TRx Volume", "Market Share")
        brand: Brand to analyze

    Returns:
        Dictionary with causal analysis results
    """
    logger.info(f"[CopilotKit] Running causal analysis: {intervention} -> {target_kpi} for {brand}")

    # Simulated causal analysis results
    import random
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


def add_copilotkit_routes(app: FastAPI, prefix: str = "/api/copilotkit") -> None:
    """
    Add CopilotKit routes to the FastAPI application.

    Args:
        app: FastAPI application instance
        prefix: URL prefix for CopilotKit endpoints
    """
    sdk = create_copilotkit_sdk()
    add_fastapi_endpoint(app, sdk, prefix)
    logger.info(f"[CopilotKit] Routes added at {prefix}")


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
