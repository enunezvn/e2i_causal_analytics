"""
Trigger Repository.

Handles HCP triggers with change tracking.
"""

from typing import List, Optional
from src.repositories.base import BaseRepository


class TriggerRepository(BaseRepository):
    """
    Repository for triggers table.

    Supports:
    - HCP trigger queries
    - Change tracking
    - Reason filtering
    """

    table_name = "triggers"
    model_class = None  # Set to Trigger model when available

    async def get_by_hcp(
        self,
        hcp_id: str,
        limit: int = 100,
    ) -> List:
        """
        Get triggers for a specific HCP.

        Args:
            hcp_id: Healthcare provider identifier
            limit: Maximum records

        Returns:
            List of Trigger records
        """
        return await self.get_many(
            filters={"hcp_id": hcp_id},
            limit=limit,
        )

    async def get_by_brand(
        self,
        brand: str,
        limit: int = 1000,
    ) -> List:
        """
        Get triggers for a specific brand.

        Args:
            brand: Brand name
            limit: Maximum records

        Returns:
            List of Trigger records
        """
        return await self.get_many(
            filters={"brand": brand},
            limit=limit,
        )

    async def get_recent_triggers(
        self,
        days: int = 7,
        limit: int = 100,
    ) -> List:
        """
        Get triggers from the last N days.

        Args:
            days: Number of days to look back
            limit: Maximum records

        Returns:
            Recent triggers
        """
        # TODO: Implement with date filter
        return await self.get_many(filters={}, limit=limit)
