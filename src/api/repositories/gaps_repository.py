"""Supabase persistence for the gaps route store (M2).

Replaces the process-local ``_analyses_store`` dict in ``src/api/routes/gaps.py``
so reads succeed across the 2 gunicorn worker processes. Follows the canonical
route-persistence pattern from ``src/api/routes/sentinels.py``: service-role
client via ``get_supabase_client``, sync ``.execute()`` offloaded to a worker
thread, full pydantic JSON stored in the ``payload`` JSONB column.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from src.api.routes.gaps import AnalysisStatus, GapAnalysisResponse

logger = logging.getLogger(__name__)

_TABLE = "gap_analyses"


def _escape_like(value: str) -> str:
    """Escape PostgREST/SQL ``LIKE``/``ILIKE`` metacharacters in ``value``.

    The brand filter uses a case-insensitive ``.ilike`` match (see
    ``list_completed``). ``.ilike`` interprets its argument as a *pattern*, so a
    caller-supplied brand containing ``%`` or ``_`` (or a literal backslash)
    would otherwise broaden the match — e.g. ``brand="%"`` matches every brand.
    Escaping these metacharacters with the default ``\\`` escape character makes
    the pattern a literal, whole-string, case-insensitive match. Backslash is
    escaped first so the subsequent ``%``/``_`` escapes are not double-escaped.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class GapsRepository:
    """Thin async repository over the ``gap_analyses`` table."""

    def __init__(self, client: Any = None) -> None:
        if client is None:
            from src.memory.services.factories import get_supabase_client

            client = get_supabase_client()
        self._client = client

    async def upsert(self, response: GapAnalysisResponse) -> None:
        """Insert or update one analysis (keyed by analysis_id)."""
        payload = response.model_dump(mode="json")
        row = {
            "analysis_id": response.analysis_id,
            "brand": response.brand,
            "status": response.status.value,
            "payload": payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        query = self._client.table(_TABLE).upsert(row, on_conflict="analysis_id")
        await asyncio.to_thread(query.execute)

    async def get(self, analysis_id: str) -> Optional[GapAnalysisResponse]:
        query = self._client.table(_TABLE).select("payload").eq("analysis_id", analysis_id).limit(1)
        result = await asyncio.to_thread(query.execute)
        rows = (result.data) or []
        if not rows:
            return None
        return GapAnalysisResponse.model_validate(rows[0]["payload"])

    async def list_completed(self, brand: Optional[str] = None) -> List[GapAnalysisResponse]:
        """Completed analyses (optionally brand-filtered), for list_opportunities."""
        query = (
            self._client.table(_TABLE)
            .select("payload")
            .eq("status", AnalysisStatus.COMPLETED.value)
        )
        if brand:
            # Case-insensitive brand match. The canonical brand casing across the
            # system is CAPITALIZED ("Kisqali", "Fabhalta", "Remibrutinib") — it
            # matches the Supabase ``brand_type`` ENUM and the synthetic ``Brand``
            # enum. ``gap_analyses.brand`` is a plain TEXT column (no ENUM
            # constraint), so historical rows were written with whatever casing
            # the caller sent (the frontend previously sent lowercase). ``.ilike``
            # keeps the grounded, capitalized analyses reachable regardless of the
            # request's casing so the GapAnalysis page never silently empties on a
            # casing mismatch again. The brand value is escaped first so its
            # ``LIKE`` metacharacters (``%``/``_``) are treated literally and the
            # filter is an exact, whole-string, case-insensitive match (a bare
            # ``.ilike`` would let ``brand="%"`` match every brand).
            query = query.ilike("brand", _escape_like(brand))
        result = await asyncio.to_thread(query.execute)
        rows = (result.data) or []
        return [GapAnalysisResponse.model_validate(r["payload"]) for r in rows]

    async def list_all(self) -> List[GapAnalysisResponse]:
        """All analyses (for get_gap_health 24h count + last timestamp)."""
        query = self._client.table(_TABLE).select("payload")
        result = await asyncio.to_thread(query.execute)
        rows = (result.data) or []
        return [GapAnalysisResponse.model_validate(r["payload"]) for r in rows]
