"""
Agent Registry Repository.

Handles the 11 V3 agents with tier assignments.
"""

from typing import List, Optional
from src.repositories.base import BaseRepository


class AgentRegistryRepository(BaseRepository):
    """
    Repository for agent_registry table.

    V3 agents (11 total):
    - Tier 1: orchestrator
    - Tier 2: causal_impact, gap_analyzer, heterogeneous_optimizer
    - Tier 3: experiment_designer, drift_monitor, health_score
    - Tier 4: prediction_synthesizer, resource_optimizer
    - Tier 5: explainer, feedback_learner

    Supports:
    - Agent configuration queries
    - Intent routing
    - Tier-based retrieval
    """

    table_name = "agent_registry"
    model_class = None  # Set to AgentRegistry model when available

    async def get_by_name(self, agent_name: str) -> Optional[dict]:
        """
        Get agent configuration by name.

        Args:
            agent_name: Agent name (e.g., 'orchestrator')

        Returns:
            AgentRegistry record or None
        """
        results = await self.get_many(
            filters={"agent_name": agent_name},
            limit=1,
        )
        return results[0] if results else None

    async def get_by_tier(self, tier: int) -> List:
        """
        Get all agents in a specific tier.

        Args:
            tier: Agent tier (1-5)

        Returns:
            List of AgentRegistry records
        """
        return await self.get_many(
            filters={"tier": tier, "is_active": True},
        )

    async def get_by_intent(self, intent: str) -> List:
        """
        Find agents that handle a specific intent.

        Args:
            intent: Intent type (e.g., 'CAUSAL', 'WHAT_IF')

        Returns:
            List of AgentRegistry records that handle this intent
        """
        # TODO: Implement array contains query
        # intents column is TEXT[]
        return await self.get_many(filters={"is_active": True})

    async def route_intent_to_agent(self, intent: str) -> Optional[dict]:
        """
        Route an intent to the appropriate agent.

        Priority: Lower tier number = higher priority

        Args:
            intent: Intent type

        Returns:
            Best matching agent or None
        """
        agents = await self.get_by_intent(intent)
        if not agents:
            return None

        # Sort by tier (lower is higher priority)
        sorted_agents = sorted(agents, key=lambda a: a.get("tier", 99))
        return sorted_agents[0]

    async def get_active_agents(self) -> List:
        """
        Get all active agents.

        Returns:
            List of active AgentRegistry records
        """
        return await self.get_many(filters={"is_active": True})
