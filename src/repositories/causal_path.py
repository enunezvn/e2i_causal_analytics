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
    # The live schema keys on a varchar ``path_id`` (there is no ``id`` column);
    # base get_by_id/update/delete were latent 42703s until #894.
    id_column = "path_id"

    async def get_paths_for_cause(
        self,
        cause: str,
        limit: int = 100,
        include_synthetic: bool = False,
    ) -> List:
        """
        Get all causal paths originating from a cause.

        The live ``causal_paths`` schema stores the cause/effect pair as
        ``start_node``/``end_node`` (there are no ``cause``/``effect`` columns
        — filtering on them was a latent 42703 until #894). The argument names
        keep the caller-facing causal vocabulary.

        Args:
            cause: Cause entity name (matched against ``start_node``)
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"start_node": cause},
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
            effect: Effect entity name (matched against ``end_node`` — see
                :meth:`get_paths_for_cause` for the schema mapping)
            limit: Maximum records
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            List of CausalPath records
        """
        return await self.get_many(
            filters={"end_node": effect},
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
            cause: Starting entity (matched against ``start_node``)
            effect: Ending entity (matched against ``end_node``)
            include_synthetic: When True, do not exclude synthetic rows (opt-in).

        Returns:
            CausalPath if exists, None otherwise
        """
        results = await self.get_many(
            filters={"start_node": cause, "end_node": effect},
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

    async def get_distinct_questions(
        self,
        *,
        brand: Optional[str] = None,
        limit: int = 2000,
        include_synthetic: bool = True,
    ) -> List[dict]:
        """Distinct (treatment, outcome, brand) causal questions from the SSOT,
        each carrying its modeled backdoor set (``confounders_controlled``).

        Source of truth for the discovery leaderboard's questions (replaces the
        hand-curated cross-product). ``include_synthetic`` defaults True because
        the gold-standard substrate is synthetic.
        """
        # Empty dict (not None) = no brand filter; get_many iterates
        # filters.items(), which would raise AttributeError on None.
        filters = {"brand": brand} if brand else {}
        rows = await self.get_many(
            filters=filters, limit=limit, include_synthetic=include_synthetic
        )
        seen: dict = {}
        for r in rows:
            key = (r.get("start_node"), r.get("end_node"), r.get("brand"))
            if key[0] is None or key[1] is None or key in seen:
                continue
            seen[key] = {
                "treatment": r["start_node"],
                "outcome": r["end_node"],
                "brand": r.get("brand"),
                "confounders": list(r.get("confounders_controlled") or []),
            }
        return list(seen.values())
