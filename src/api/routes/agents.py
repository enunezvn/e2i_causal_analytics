"""
E2I Agent Orchestration API
============================

FastAPI endpoints for agent status monitoring and orchestration.

Endpoints:
- GET /agents/status: Get status of all 18 agents in the tier hierarchy

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agent Orchestration"])


# =============================================================================
# ENUMS
# =============================================================================


class AgentStatusEnum(str, Enum):
    """Agent status values."""

    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"


class AgentTierEnum(int, Enum):
    """Agent tier levels (0-5)."""

    ML_FOUNDATION = 0
    ORCHESTRATION = 1
    CAUSAL_ANALYTICS = 2
    MONITORING = 3
    ML_PREDICTIONS = 4
    SELF_IMPROVEMENT = 5


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class AgentInfo(BaseModel):
    """Information about a single agent."""

    id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    tier: int = Field(..., ge=0, le=5, description="Agent tier (0-5)")
    status: AgentStatusEnum = Field(..., description="Current agent status")
    last_activity: Optional[str] = Field(None, description="ISO timestamp of last activity")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")


class AgentStatusResponse(BaseModel):
    """Response containing status of all agents."""

    agents: List[AgentInfo] = Field(..., description="List of all agents with their status")
    total_agents: int = Field(..., description="Total number of agents")
    active_count: int = Field(..., description="Number of active agents")
    processing_count: int = Field(..., description="Number of processing agents")
    error_count: int = Field(..., description="Number of agents in error state")
    timestamp: datetime = Field(..., description="Response timestamp")


# =============================================================================
# SAMPLE DATA
# =============================================================================

# Default agent configuration matching the 18-agent tier hierarchy
AGENT_REGISTRY = [
    # Tier 0 - ML Foundation (7 agents)
    AgentInfo(
        id="scope-definer",
        name="Scope Definer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        capabilities=["problem_scoping", "requirement_analysis"],
    ),
    AgentInfo(
        id="data-preparer",
        name="Data Preparer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        capabilities=["data_validation", "preprocessing"],
    ),
    AgentInfo(
        id="feature-analyzer",
        name="Feature Analyzer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        capabilities=["feature_engineering", "selection"],
    ),
    AgentInfo(
        id="model-selector",
        name="Model Selector",
        tier=0,
        status=AgentStatusEnum.IDLE,
        capabilities=["model_comparison", "benchmarking"],
    ),
    AgentInfo(
        id="model-trainer",
        name="Model Trainer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        capabilities=["training", "hyperparameter_tuning"],
    ),
    AgentInfo(
        id="model-deployer",
        name="Model Deployer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        capabilities=["deployment", "versioning"],
    ),
    AgentInfo(
        id="observability-connector",
        name="Observability Connector",
        tier=0,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["mlflow", "opik", "monitoring"],
    ),
    # Tier 1 - Orchestration (2 agents)
    AgentInfo(
        id="orchestrator",
        name="Orchestrator",
        tier=1,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["routing", "coordination", "agent_dispatch"],
    ),
    AgentInfo(
        id="tool-composer",
        name="Tool Composer",
        tier=1,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["tool_orchestration", "query_decomposition"],
    ),
    # Tier 2 - Causal Analytics (3 agents)
    AgentInfo(
        id="causal-impact",
        name="Causal Impact",
        tier=2,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["causal_tracing", "effect_estimation", "dowhy"],
    ),
    AgentInfo(
        id="gap-analyzer",
        name="Gap Analyzer",
        tier=2,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["opportunity_detection", "roi_analysis"],
    ),
    AgentInfo(
        id="heterogeneous-optimizer",
        name="Heterogeneous Optimizer",
        tier=2,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["cate_analysis", "segment_optimization", "econml"],
    ),
    # Tier 3 - Monitoring (3 agents)
    AgentInfo(
        id="drift-monitor",
        name="Drift Monitor",
        tier=3,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["data_drift", "model_drift", "alerting"],
    ),
    AgentInfo(
        id="experiment-designer",
        name="Experiment Designer",
        tier=3,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["ab_testing", "sample_size", "power_analysis"],
    ),
    AgentInfo(
        id="health-score",
        name="Health Score",
        tier=3,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["system_health", "performance_metrics"],
    ),
    # Tier 4 - ML Predictions (2 agents)
    AgentInfo(
        id="prediction-synthesizer",
        name="Prediction Synthesizer",
        tier=4,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["prediction_aggregation", "ensemble"],
    ),
    AgentInfo(
        id="resource-optimizer",
        name="Resource Optimizer",
        tier=4,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["resource_allocation", "optimization"],
    ),
    # Tier 5 - Self-Improvement (2 agents)
    AgentInfo(
        id="explainer",
        name="Explainer",
        tier=5,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["nl_generation", "insight_explanation"],
    ),
    AgentInfo(
        id="feedback-learner",
        name="Feedback Learner",
        tier=5,
        status=AgentStatusEnum.ACTIVE,
        capabilities=["feedback_integration", "self_improvement"],
    ),
]


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status() -> AgentStatusResponse:
    """
    Get status of all agents in the orchestration system.

    Returns the current status of all 18 agents across 6 tiers,
    including their capabilities and activity timestamps.
    """
    agents = AGENT_REGISTRY

    # Calculate counts
    active_count = sum(1 for a in agents if a.status == AgentStatusEnum.ACTIVE)
    processing_count = sum(1 for a in agents if a.status == AgentStatusEnum.PROCESSING)
    error_count = sum(1 for a in agents if a.status == AgentStatusEnum.ERROR)

    return AgentStatusResponse(
        agents=agents,
        total_agents=len(agents),
        active_count=active_count,
        processing_count=processing_count,
        error_count=error_count,
        timestamp=datetime.now(timezone.utc),
    )
