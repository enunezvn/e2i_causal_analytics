"""
KPI history walk-forward backfill.
=================================

Materializes REAL monthly KPI points into ``kpi_history`` (migration 079) for the
Time-Series "KPI history" view. Two honest mechanisms, dispatched per KPI:

1. **Direct monthly source** — the KPI's value already lives as a monthly series
   (e.g. ``WS3-BI-010`` ROI <- ``business_metrics.roi``). Read it, group by month.
2. **As-of recompute** — recompute the KPI "as of" each month from a dated source
   table (e.g. ``treatment_events.event_date``). (Batch 2 — add handlers below.)

Anti-fabrication: a KPI gets a handler ONLY when its history can be produced from
real, time-dimensioned data. KPIs with no honest temporal source are intentionally
NOT registered (the UI shows an empty-state, never a synthesized flat line).

Run:  python -m src.kpi.history_backfill           # all registered KPIs
      python -m src.kpi.history_backfill WS3-BI-010 # one KPI
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.kpi.models import KPIStatus

logger = logging.getLogger(__name__)

# A handler returns a list of point dicts ready for kpi_history upsert.
Handler = Callable[[Any, Any], Awaitable[List[Dict[str, Any]]]]


def _status_for(kpi_meta: Any, value: float, lower_is_better: bool = False) -> Optional[str]:
    """Evaluate a value against the KPI threshold; None when unavailable.

    ROI is higher-is-better (the default). Add per-KPI direction when batch-2
    handlers cover lower-is-better KPIs (e.g. data lag).
    """
    threshold = getattr(kpi_meta, "threshold", None)
    if threshold is None:
        # No threshold by design -> the point is tracked for trend/context only.
        return str(KPIStatus.INFORMATIONAL.value)
    try:
        # `.value` on the status enum is typed Any → coerce to str so the
        # declared Optional[str] return is honoured (mypy no-any-return).
        return str(threshold.evaluate(value, lower_is_better).value)
    except Exception:  # noqa: BLE001
        return None


async def _backfill_roi(client: Any, kpi_meta: Any) -> List[Dict[str, Any]]:
    """WS3-BI-010 ROI <- business_metrics.roi (already a monthly series).

    Produces a global (brand='', region='') monthly series = mean(roi) per month,
    plus a per-brand (brand=X, region='') series = mean(roi) per (brand, month).
    Mean is over the real business_metrics rows for the group — no synthesis.
    """
    result = await (
        client.table("business_metrics")
        .select("metric_date,brand,roi")
        .not_.is_("roi", "null")
        .order("metric_date")
        .limit(20000)
        .execute()
    )
    rows = result.data or []
    # (scope_brand, metric_date) -> [roi, ...]
    global_acc: Dict[str, List[float]] = defaultdict(list)
    brand_acc: Dict[tuple, List[float]] = defaultdict(list)
    for r in rows:
        d = r.get("metric_date")
        roi = r.get("roi")
        if d is None or roi is None:
            continue
        roi = float(roi)
        global_acc[d].append(roi)
        b = r.get("brand")
        if b:
            brand_acc[(b, d)].append(roi)

    points: List[Dict[str, Any]] = []

    def _point(brand: str, date: str, vals: List[float]) -> Dict[str, Any]:
        value = sum(vals) / len(vals)
        return {
            "kpi_id": kpi_meta.id,
            "brand": brand,
            "region": "",
            "metric_date": date,
            "value": value,
            "status": _status_for(kpi_meta, value),
            "source": "business_metrics.roi",
            "is_synthetic": True,
        }

    for date, vals in global_acc.items():
        points.append(_point("", date, vals))
    for (brand, date), vals in brand_acc.items():
        points.append(_point(brand, date, vals))
    return points


# KPI_ID -> handler. Only honestly-backfillable KPIs are registered (vertical
# slice: ROI. Batch 2 adds as-of handlers for the event-derived KPIs).
HANDLERS: Dict[str, Handler] = {
    "WS3-BI-010": _backfill_roi,
}


async def run_backfill(kpi_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute + upsert history points for the registered (or requested) KPIs."""
    from src.kpi.registry import KPIRegistry
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.kpi_history import get_kpi_history_repository

    client = await get_async_supabase_client()
    if client is None:
        raise RuntimeError("No Supabase client for KPI history backfill")
    repo = await get_kpi_history_repository()
    registry = KPIRegistry()

    targets = kpi_ids or list(HANDLERS.keys())
    summary: Dict[str, Any] = {"written": {}, "skipped": [], "errors": {}}
    for kpi_id in targets:
        handler = HANDLERS.get(kpi_id)
        if handler is None:
            summary["skipped"].append(kpi_id)
            continue
        kpi_meta = registry.get(kpi_id)
        if kpi_meta is None:
            summary["errors"][kpi_id] = "not in registry"
            continue
        try:
            points = await handler(client, kpi_meta)
            written = await repo.upsert_points(points)
            summary["written"][kpi_id] = written
            logger.info("KPI history backfill %s: %d points", kpi_id, written)
        except Exception as e:  # noqa: BLE001
            summary["errors"][kpi_id] = str(e)
            logger.error("KPI history backfill failed for %s: %s", kpi_id, e, exc_info=True)
    return summary


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO)
    kpi_ids = sys.argv[1:] or None
    result = asyncio.run(run_backfill(kpi_ids))
    print("KPI history backfill summary:", result)


if __name__ == "__main__":
    main()
