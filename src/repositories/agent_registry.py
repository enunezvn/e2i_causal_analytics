"""
Agent Registry Repository.

Handles the agent_registry table (21-agent roster; source of truth = src/agents/factory.py AGENT_REGISTRY_CONFIG).
"""

from typing import Any, Dict, List, Optional, cast

from src.repositories.base import BaseRepository

# Canonical mapping between the numeric agent tier (0-5) and the real
# ``agent_registry.agent_tier`` text-category enum (``agent_tier_type``). The DB
# has NO int ``tier`` column — ``agent_tier`` holds these text categories — so a
# numeric tier must be translated to its category before filtering (issue #825;
# the phantom ``{"tier": int}`` filter raised 42703). Numbering matches
# ``src/agents/factory.py`` AGENT_REGISTRY_CONFIG (the roster source of truth).
TIER_CATEGORY_BY_NUMBER: Dict[int, str] = {
    0: "ml_foundation",
    1: "coordination",
    2: "causal_analytics",
    3: "monitoring",
    4: "ml_predictions",
    5: "self_improvement",
}
TIER_NUMBER_BY_CATEGORY: Dict[str, int] = {v: k for k, v in TIER_CATEGORY_BY_NUMBER.items()}


def tier_number_for_category(category: Optional[str]) -> int:
    """Numeric tier (0-5) for an ``agent_tier`` text category; 99 if unknown
    (sorts unknown agents last)."""
    return TIER_NUMBER_BY_CATEGORY.get(category or "", 99)


class AgentRegistryRepository(BaseRepository):
    """
    Repository for agent_registry table.

    Roster: 22 agents (source of truth = src/agents/factory.py AGENT_REGISTRY_CONFIG):
    - Tier 0: scope_definer, cohort_constructor, cohort_profiler, data_preparer, feature_analyzer, model_selector, model_trainer, model_deployer, observability_connector
    - Tier 1: orchestrator, tool_composer
    - Tier 2: causal_impact, gap_analyzer, heterogeneous_optimizer
    - Tier 3: drift_monitor, experiment_designer, experiment_monitor, health_score
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
        Get all active agents in a specific tier.

        The real ``agent_registry`` schema stores the tier as the
        ``agent_tier`` text-category enum (NOT an int ``tier`` column), so the
        numeric tier is translated to its category before filtering (issue
        #825). An out-of-range tier yields no category and returns ``[]``.

        Args:
            tier: Agent tier (0-5)

        Returns:
            List of AgentRegistry records
        """
        category = TIER_CATEGORY_BY_NUMBER.get(tier)
        if category is None:
            return []
        return await self.get_many(
            filters={"agent_tier": category, "is_active": True},
        )

    async def get_by_intent(self, intent: str) -> List:
        """
        Find agents that handle a specific intent.

        Uses PostgreSQL JSONB containment (@>) to filter by routes_from_intents array.
        The routes_from_intents column is JSONB, e.g., '["CAUSAL", "IMPACT", "WHY"]'.

        Args:
            intent: Intent type (e.g., 'CAUSAL', 'WHAT_IF')

        Returns:
            List of AgentRegistry records that handle this intent
        """
        if not self.client:
            return []

        # Use Supabase contains() for JSONB array column filtering
        # This translates to PostgreSQL: routes_from_intents @> '["CAUSAL"]'::jsonb
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("is_active", True)
            .contains("routes_from_intents", [intent.upper()])
            .execute()
        )

        return [self._to_model(row) for row in result.data] if result.data else []

    async def get_by_capability(self, capability: str) -> List:
        """
        Find agents that have a specific capability.

        Uses PostgreSQL JSONB containment (@>) to filter by capabilities array.
        The capabilities column is JSONB, e.g., '["ate_estimation", "cate_calculation"]'.

        Args:
            capability: Capability name (e.g., 'ate_estimation', 'gap_identification')

        Returns:
            List of AgentRegistry records that have this capability
        """
        if not self.client:
            return []

        # Use Supabase contains() for JSONB array column filtering
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("is_active", True)
            .contains("capabilities", [capability.lower()])
            .execute()
        )

        return [self._to_model(row) for row in result.data] if result.data else []

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

        # Sort by tier (lower is higher priority). The real schema stores the
        # tier as the ``agent_tier`` text category, so derive the numeric tier
        # from it rather than the non-existent ``tier`` column (issue #825).
        sorted_agents = sorted(agents, key=lambda a: tier_number_for_category(a.get("agent_tier")))
        return cast(Dict[Any, Any], sorted_agents[0])

    async def get_active_agents(self) -> List:
        """
        Get all active agents.

        Returns:
            List of active AgentRegistry records
        """
        return await self.get_many(filters={"is_active": True})
