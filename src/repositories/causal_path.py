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

    Provenance (#893): ``causal_paths`` carries the ``is_synthetic`` column
    (migration 063) and the synthetic loader stamps every loaded row, so all
    real-mode reads default-exclude synthetic paths. Validation/agent-context
    callers opt in per read with ``include_synthetic=True``.
    """

    table_name = "causal_paths"
    model_class = None  # Set to CausalPath model when available
    HAS_PROVENANCE = True  # causal_paths carries is_synthetic (migration 063, #893)

    async def get_paths_for_cause(
        self,
        cause: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get all causal paths originating from a cause.

        Args:
            cause: Cause entity name
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"cause": cause},
            limit=limit,
            include_synthetic=include_synthetic,
        )

    async def get_paths_for_effect(
        self,
        effect: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get all causal paths leading to an effect.

        Args:
            effect: Effect entity name
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"effect": effect},
            limit=limit,
            include_synthetic=include_synthetic,
        )

    async def get_path_between(
        self,
        cause: str,
        effect: str,
        include_synthetic: bool = False,
    ) -> Optional[List]:
        """
        Get the causal path between two entities.

        Args:
            cause: Starting entity
            effect: Ending entity
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            CausalPath if exists, None otherwise
        """
        results = await self.get_many(
            filters={"cause": cause, "effect": effect},
            limit=1,
            include_synthetic=include_synthetic,
        )
        return results[0] if results else None

    async def get_by_brand(
        self,
        brand: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get causal paths related to a brand.

        Args:
            brand: Brand name
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"brand": brand},
            limit=limit,
            include_synthetic=include_synthetic,
        )
