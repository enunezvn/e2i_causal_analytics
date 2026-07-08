"""
KPI history weekly capture (present-state KPIs).
================================================

Going-FORWARD companion to ``src.kpi.history_backfill``. The backfill covers
KPIs whose history can be honestly *recomputed* from dated source rows. The
KPIs below cannot — they read a present-state universe (undated
``coverage_status``, as-of-now eligibility views) or use windows that don't
recast onto calendar months — so a backdated series would be a fabrication.

What CAN be done honestly is to record the live reading at capture time, the
way any real telemetry series accrues. This module computes each KPI through
the exact same calculator path as ``GET /api/kpis/{kpi_id}`` (registered
per-workstream calculators, synthetic-twin queries, fail-loud on missing data)
and upserts ONE point per KPI at today's date with ``source='weekly_capture'``.

Semantics (deliberately different from the backfill):

- **Append, never replace.** A capture is an as-of-that-day observation; it can
  never be recomputed later, so ``delete_source`` replace semantics would
  destroy real history. Re-running on the same day upserts (idempotent).
- **Frontier-append safe.** The weekly cron grows the substrate without
  rewriting history (``reseed_synthetic.sh`` default), so week-over-week
  captures track a genuinely evolving universe.
- **``--full`` reseed invalidates captures.** A destructive
  ``--anchor-to-now`` reseed rewrites the substrate the captured readings
  described; ``reseed_synthetic.sh --full`` therefore calls ``--purge`` before
  reseeding so stale observations don't masquerade as history of the new seed.

Run: ``python -m src.kpi.history_capture`` (wired into reseed_synthetic.sh
after the backfill step). ``--purge`` deletes all weekly_capture rows instead.
"""

import asyncio
import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CAPTURE_SOURCE = "weekly_capture"

#: Present-state / non-recastable KPIs — everything in the registry that has no
#: honest backfill handler and isn't served elsewhere (WS1-MP-* trends live in
#: ml_performance_metrics on /model-performance; CM-* are per-analysis
#: estimates, not platform series). Reasons mirror history_backfill's docstring.
CAPTURE_KPI_IDS: tuple = (
    # Coverage / eligibility vs a present-state universe:
    "WS1-DQ-001",
    "WS1-DQ-002",
    "WS1-DQ-003",
    "WS1-DQ-004",
    "WS1-DQ-005",
    "WS1-DQ-006",
    "WS1-DQ-007",
    "WS1-DQ-009",
    "WS3-BI-003",
    "WS3-BI-004",
    "BR-005",
    # Windowed / all-time readings that don't recast onto calendar months:
    "WS2-TR-002",
    "WS2-TR-003",
)


def _status_str(result: Any) -> Optional[str]:
    status = getattr(result, "status", None)
    value = getattr(status, "value", None)
    return str(value) if value is not None else (str(status) if status else None)


async def run_capture(kpi_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Capture today's live reading for each present-state KPI.

    A KPI that fails to compute (e.g. WS3-BI-004 while its denominator data is
    missing) is recorded in ``errors`` and skipped — no point is written, and
    prior captures are never touched.
    """
    # Local import: reuses the API layer's calculator factory so a captured
    # value can never diverge from what GET /api/kpis/{id} would have served.
    from src.api.routes.kpi import get_kpi_calculator
    from src.repositories.kpi_history import get_kpi_history_repository

    calculator = get_kpi_calculator()
    repo = await get_kpi_history_repository()
    today = date.today().isoformat()

    targets = list(kpi_ids or CAPTURE_KPI_IDS)
    summary: Dict[str, Any] = {"written": {}, "errors": {}, "date": today}
    points: List[Dict[str, Any]] = []
    for kpi_id in targets:
        try:
            result = await asyncio.to_thread(
                calculator.calculate, kpi_id, use_cache=False, force_refresh=True
            )
            error = getattr(result, "error", None)
            value = getattr(result, "value", None)
            if error or value is None:
                summary["errors"][kpi_id] = str(error or "no value")
                continue
            points.append(
                {
                    "kpi_id": kpi_id,
                    "brand": "",
                    "region": "",
                    "metric_date": today,
                    "value": float(value),
                    "status": _status_str(result),
                    "source": CAPTURE_SOURCE,
                    "is_synthetic": True,
                }
            )
        except Exception as e:  # noqa: BLE001
            summary["errors"][kpi_id] = str(e)
            logger.error("KPI capture failed for %s: %s", kpi_id, e, exc_info=True)

    if points:
        written = await repo.upsert_points(points)
        for p in points:
            summary["written"][p["kpi_id"]] = p["value"]
        logger.info("KPI weekly capture: %d points written for %s", written, today)
    return summary


async def purge_captures() -> Dict[str, int]:
    """Delete ALL weekly_capture rows (for --full destructive reseeds only)."""
    from src.repositories.kpi_history import get_kpi_history_repository

    repo = await get_kpi_history_repository()
    deleted: Dict[str, int] = {}
    for kpi_id in CAPTURE_KPI_IDS:
        deleted[kpi_id] = await repo.delete_source(kpi_id, CAPTURE_SOURCE)
    return deleted


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO)
    if "--purge" in sys.argv[1:]:
        print("KPI capture purge:", asyncio.run(purge_captures()))
        return
    kpi_ids = [a for a in sys.argv[1:] if not a.startswith("--")] or None
    print("KPI weekly capture summary:", asyncio.run(run_capture(kpi_ids)))


if __name__ == "__main__":
    main()
