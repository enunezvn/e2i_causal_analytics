"""E2I Causal Analytics Agents.

This module provides the 21-agent tiered architecture (incl. cohort_constructor):
- Tier 0: ML Foundation (7 agents) + cohort_constructor
- Tier 1: Coordination (2 agents: orchestrator, tool_composer)
- Tier 2: Causal Analytics (3 agents)
- Tier 3: Monitoring (4 agents: drift_monitor, experiment_designer, experiment_monitor, health_score)
- Tier 4: ML Predictions (2 agents)
- Tier 5: Self-Improvement (2 agents)

Example:
    from src.agents import create_agent_registry, OrchestratorAgent

    # Create all agents
    registry = create_agent_registry()

    # Create orchestrator with agents
    orchestrator = OrchestratorAgent(agent_registry=registry)
"""

from src.agents.factory import (
    AGENT_REGISTRY_CONFIG,
    REQUIRE_FULL_REGISTRY_ENV,
    PartialAgentRegistryError,
    assert_full_agent_registry,
    create_agent_registry,
    get_agent_config,
    get_all_analytics_agents,
    get_tier2_agents,
    list_available_agents,
)

__all__ = [
    # Factory functions
    "create_agent_registry",
    "get_agent_config",
    "list_available_agents",
    "get_tier2_agents",
    "get_all_analytics_agents",
    "AGENT_REGISTRY_CONFIG",
    # #1448 registry-completeness gate
    "assert_full_agent_registry",
    "PartialAgentRegistryError",
    "REQUIRE_FULL_REGISTRY_ENV",
]
