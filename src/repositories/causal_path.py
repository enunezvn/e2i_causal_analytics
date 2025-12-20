"""
Causal Path Repository.

Handles discovered causal relationships.
"""

from typing import List, Optional
from src.repositories.base import BaseRepository


class CausalPathRepository(BaseRepository):
    """
    Repository for causal_paths table.

    Supports:
    - Causal relationship queries
    - Path traversal
    - Effect estimation retrieval
    """

    table_name = "causal_paths"
    model_class = None  # Set to CausalPath model when available

    async def get_paths_for_cause(
        self,
        cause: str,
        limit: int = 100,
    ) -> List:
        """
        Get all causal paths originating from a cause.

        Args:
            cause: Cause entity name
            limit: Maximum records

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"cause": cause},
            limit=limit,
        )

    async def get_paths_for_effect(
        self,
        effect: str,
        limit: int = 100,
    ) -> List:
        """
        Get all causal paths leading to an effect.

        Args:
            effect: Effect entity name
            limit: Maximum records

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"effect": effect},
            limit=limit,
        )

    async def get_path_between(
        self,
        cause: str,
        effect: str,
    ) -> Optional[List]:
        """
        Get the causal path between two entities.

        Args:
            cause: Starting entity
            effect: Ending entity

        Returns:
            CausalPath if exists, None otherwise
        """
        results = await self.get_many(
            filters={"cause": cause, "effect": effect},
            limit=1,
        )
        return results[0] if results else None

    async def get_by_brand(
        self,
        brand: str,
        limit: int = 100,
    ) -> List:
        """
        Get causal paths related to a brand.

        Args:
            brand: Brand name
            limit: Maximum records

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"brand": brand},
            limit=limit,
        )
