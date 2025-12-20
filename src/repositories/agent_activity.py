"""
Agent Activity Repository.

Handles agent analysis outputs and activity tracking.
"""

from typing import List, Optional
from src.repositories.base import BaseRepository


class AgentActivityRepository(BaseRepository):
    """
    Repository for agent_activities table.

    Supports:
    - Agent output queries
    - Analysis result retrieval
    - Activity tracking
    """

    table_name = "agent_activities"
    model_class = None  # Set to AgentActivity model when available

    async def get_by_agent(
        self,
        agent_type: str,
        limit: int = 100,
    ) -> List:
        """
        Get activities for a specific agent type.

        Args:
            agent_type: Agent type enum value
            limit: Maximum records

        Returns:
            List of AgentActivity records
        """
        return await self.get_many(
            filters={"agent_type": agent_type},
            limit=limit,
        )

    async def get_by_tier(
        self,
        tier: int,
        limit: int = 100,
    ) -> List:
        """
        Get activities for all agents in a tier.

        Args:
            tier: Agent tier (1-5)
            limit: Maximum records

        Returns:
            List of AgentActivity records
        """
        return await self.get_many(
            filters={"tier": tier},
            limit=limit,
        )

    async def get_analysis_results(
        self,
        agent_type: str,
        limit: int = 50,
    ) -> List:
        """
        Get analysis results from a specific agent.

        Used by RAG for indexing agent outputs.

        Args:
            agent_type: Agent type enum value
            limit: Maximum records

        Returns:
            List of AgentActivity records with analysis_results
        """
        # TODO: Filter for non-null analysis_results
        return await self.get_many(
            filters={"agent_type": agent_type},
            limit=limit,
        )

    async def get_recent_activities(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List:
        """
        Get recent agent activities.

        Args:
            hours: Number of hours to look back
            limit: Maximum records

        Returns:
            Recent activities
        """
        # TODO: Implement with time filter
        return await self.get_many(filters={}, limit=limit)
