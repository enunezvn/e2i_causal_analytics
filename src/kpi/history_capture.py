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

Brand axis: KPIs in :data:`BRAND_CAPTURE_KPI_IDS` are ALSO captured once per
portfolio brand (:data:`CAPTURE_BRANDS`) through the same calculator path with
``context={"brand": <brand>}`` — the honest forward-accruing substrate for the
Time-Series brand selector / "Compare Brands" overlay on present-state KPIs.
Only KPIs whose live calculator returns a DISTINCT brand-scoped reading are
listed (measured against the live API before wiring); a KPI whose calculator
ignores or lacks a brand parameter is captured globally only — three identical
lines would be a fabricated brand axis.

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
    # #1360: Trigger Funnel Conversion — frontier-anchored 30d funnel reading
    # (migration 118); no honest monthly backfill, so history accrues forward
    # via weekly capture like WS2-TR-002/003.
    "WS2-TR-009",
)

#: Portfolio brands captured per-brand for :data:`BRAND_CAPTURE_KPI_IDS`.
#: Mirrors ``src.ml.synthetic.config.Brand`` (the DGP's brand domain — the only
#: values ``triggers.brand_id`` / ``patient_journeys.brand`` carry); canonical
#: case matches the backfill's brand rows and the calculators' case-sensitive
#: ``brand::text = $1`` predicates. Lockstep-tested against the enum.
CAPTURE_BRANDS: tuple = ("Fabhalta", "Kisqali", "Remibrutinib")

#: Capture KPIs whose LIVE calculator honors ``context["brand"]`` with a
#: distinct per-brand reading (verified 2026-09-05 against
#: ``GET /api/kpis/{id}?brand=...`` on prod: every brand differed from the
#: global and from each other). Deliberately NOT here:
#: - WS1-DQ-002 — hcp_profiles has no brand column (calculator docstring);
#: - WS1-DQ-003/004/005/007/009, WS3-BI-004 — calculators take no brand
#:   parameter (a brand-scoped ask returns the global figure);
#: - BR-005 — single-brand by definition.
BRAND_CAPTURE_KPI_IDS: frozenset = frozenset(
    {
        # data_quality_source_coverage_patients: $1 brand on numerator + universe
        "WS1-DQ-001",
        # data_quality_geographic_consistency: $1 brand on source + universe shares
        "WS1-DQ-006",
        # business_impact_patient_touch_rate: $1 brand over v_patient_eligibility
        "WS3-BI-003",
        # trigger_performance_recall_brand (113)
        "WS2-TR-002",
        # trigger_performance_action_rate_uplift_brand (113)
        "WS2-TR-003",
        # trigger_effectiveness_funnel_conversion: $1 brand, nullable (118)
        "WS2-TR-009",
    }
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
    summary: Dict[str, Any] = {
        "written": {},
        "written_by_brand": {},
        "errors": {},
        "date": today,
    }
    points: List[Dict[str, Any]] = []
    for kpi_id in targets:
        # '' = the global reading (every capture KPI); brand-capable KPIs are
        # additionally read once per portfolio brand. Each scope is independent:
        # one brand failing never blocks the global point or the other brands.
        scopes = [""] + (list(CAPTURE_BRANDS) if kpi_id in BRAND_CAPTURE_KPI_IDS else [])
        for brand in scopes:
            key = f"{kpi_id}[{brand}]" if brand else kpi_id
            # The global call keeps its historical shape (no context kwarg);
            # brand scopes route through the calculators' context["brand"].
            kwargs: Dict[str, Any] = {"context": {"brand": brand}} if brand else {}
            try:
                result = await asyncio.to_thread(
                    calculator.calculate, kpi_id, use_cache=False, force_refresh=True, **kwargs
                )
                error = getattr(result, "error", None)
                value = getattr(result, "value", None)
                if error or value is None:
                    summary["errors"][key] = str(error or "no value")
                    continue
                points.append(
                    {
                        "kpi_id": kpi_id,
                        "brand": brand,
                        "region": "",
                        "metric_date": today,
                        "value": float(value),
                        "status": _status_str(result),
                        "source": CAPTURE_SOURCE,
                        "is_synthetic": True,
                    }
                )
            except Exception as e:  # noqa: BLE001
                summary["errors"][key] = str(e)
                logger.error("KPI capture failed for %s: %s", key, e, exc_info=True)

    if points:
        written = await repo.upsert_points(points)
        for p in points:
            if p["brand"]:
                summary["written_by_brand"].setdefault(p["kpi_id"], {})[p["brand"]] = p["value"]
            else:
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
