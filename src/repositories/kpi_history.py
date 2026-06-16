"""
KPI History Repository.

Read/write the ``kpi_history`` table (migration 079) — materialized monthly KPI
points for the Time-Series "KPI history" view. Written by the walk-forward
backfill (``src/kpi/history_backfill.py``); read by the
``GET /api/kpis/{kpi_id}/history`` endpoint.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional

from .base import BaseRepository

logger = logging.getLogger(__name__)


class KPIHistoryRepository(BaseRepository):
    """Repository for the ``kpi_history`` table."""

    table_name = "kpi_history"
    model_class = None  # dict rows

    async def upsert_points(self, points: List[Dict[str, Any]]) -> int:
        """Idempotently upsert KPI history points.

        Conflict key is ``(kpi_id, brand, region, metric_date)`` so re-running the
        backfill overwrites a month's value rather than duplicating it.

        Returns the number of rows written (best-effort).
        """
        if not self.client or not points:
            return 0
        try:
            result_or_coro = (
                self.client.table(self.table_name)
                .upsert(points, on_conflict="kpi_id,brand,region,metric_date")
                .execute()
            )
            result = await result_or_coro if inspect.isawaitable(result_or_coro) else result_or_coro
            return len(result.data) if getattr(result, "data", None) else len(points)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to upsert {len(points)} kpi_history points: {e}", exc_info=True)
            return 0

    async def get_history(
        self,
        kpi_id: str,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        """Return date-ordered (asc) KPI history points for one KPI.

        ``brand``/``region`` None means "global" (the '' rows). A provided
        brand/region filters to that scope.
        """
        if not self.client:
            return []
        try:
            query = self.client.table(self.table_name).select("*").eq("kpi_id", kpi_id)
            # '' is the canonical "global" scope value (see migration 079).
            query = query.eq("brand", brand if brand is not None else "")
            query = query.eq("region", region if region is not None else "")
            if start_date:
                query = query.gte("metric_date", start_date)
            if end_date:
                query = query.lte("metric_date", end_date)
            result_or_coro = query.order("metric_date", desc=False).limit(limit).execute()
            result = await result_or_coro if inspect.isawaitable(result_or_coro) else result_or_coro
            return result.data if getattr(result, "data", None) else []
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to read kpi_history for {kpi_id}: {e}", exc_info=True)
            return []


_kpi_history_repository: Optional[KPIHistoryRepository] = None


async def get_kpi_history_repository() -> KPIHistoryRepository:
    """Singleton ``KPIHistoryRepository`` with an async Supabase client."""
    global _kpi_history_repository
    if _kpi_history_repository is None:
        try:
            from src.memory.services.factories import get_async_supabase_client

            client = await get_async_supabase_client()
            _kpi_history_repository = KPIHistoryRepository(supabase_client=client)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not initialize KPIHistoryRepository with client: {e}")
            _kpi_history_repository = KPIHistoryRepository(supabase_client=None)
    return _kpi_history_repository
