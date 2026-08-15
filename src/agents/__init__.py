"""E2I Causal Analytics Agents.

This module provides the 22-agent tiered architecture. The roster itself is
SSOT in ``factory.AGENT_REGISTRY_CONFIG``; this breakdown is a summary and is
pinned against it by tests/unit/test_agents/test_agent_roster_ssot_1638.py.
- Tier 0: ML Foundation (9 agents, incl. cohort_constructor + cohort_profiler)
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
