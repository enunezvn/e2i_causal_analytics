"""Canonical answer for ``e2i_data_query_tool(query_type='kpi')`` on the Rx-volume family.

Canonical TRx lane, Task 15A (codex r1 HIGH, revised codex r2 HIGH, r3 HIGH). The
data-query tool returned raw business_metrics ROWS while ``kpi_calculate_tool``
returned the canonical aggregate: one KPI at two shapes. Rebuilding the aggregate
from the monthly series API was not equivalent either -- its coverage filter, a
rounded lookback start and an out-of-window fallback could pick another month or
value than the headline.

For TRx / NRx / NBRx / TRx Share this module therefore makes the SAME call
``kpi_calculate_tool`` makes -- ``calculator.calculate(kpi_id, context=...)`` with the
same brand/region context -- so statement selection (``canonical_query_call``,
migration 143), the synthetic twin, the cache and the errors are one code path.

``time_range`` is a LOOKBACK, not a reporting window: it never chooses the month, but
the headline is served only when its month COMPLETED inside the lookback (see
``ELIGIBILITY_RULE``); otherwise the answer is an explicit ``out_of_range`` refusal
naming the frontier month and pointing at ``kpi_calculate_tool(window=...)``. Windowed
asks belong to that tool. Every other ``kpi_name`` keeps the stored-row path.

WHY THIS IS LOAD-BEARING AND NOT MERELY TIDY: pre-lane the canonical KPIs rested on
treatment_events, so a stored-row answer tripped ``cross_substrate_conflict`` and the
reader was warned. Once the lane repoints them at business_metrics the two agree on
SUBSTRATE and that fence goes silent, while the SHAPES still differ. Measured at both
shas 2026-09-17: origin/main ``4bcd37e76`` FIRES, lane ``0fe8e3f95`` SILENT. Routing is
what keeps the answer honest -- there is no warning behind it any more.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from src.kpi.measure_basis import BUSINESS_METRICS_BASIS
from src.kpi.volume_family import CANONICAL_VOLUME_KPI_IDS

NOTE = (
    "canonical aggregate from kpi_calculate_tool's own calculation: the latest complete "
    "calendar month at the global data frontier, regions summed unless a region is named, "
    "served only when that month completed inside time_range; for a reporting window call "
    "kpi_calculate_tool with window=..."
)
#: codex r3 HIGH: time_range is honoured, never silently discarded. A canonical monthly
#: figure is eligible iff it became COMPLETE inside the lookback: the first day after its
#: month is on or after ``since``. Strict month overlap (month end >= since) would refuse
#: the tool's DEFAULT last_30_days on every 31st (2026-10-31: since 2026-10-01, September
#: ends 2026-09-30); this rule serves the default every day (the cron appends month N on
#: the first Monday of N, so N-1 is the frontier all month) and still refuses last_7_days
#: on 2026-09-15 (August completed 2026-09-01, before since 2026-09-08). The upper edge
#: needs no test: the frontier is always before the in-progress month.
ELIGIBILITY_RULE = "served only when the frontier month completed on or after the lookback start"
_ROW_NAME: Dict[str, str] = {
    "WS3-BI-005": "trx",
    "WS3-BI-006": "nrx",
    "WS3-BI-007": "nbrx",
    "WS3-BI-008": "trx_share",
}
#: Stored metric keys the vocabulary maps onto the family but which are a DIFFERENT
#: stored quantity: business_metrics 'market_share' is the modeled brand market share,
#: not TRx Share, so it keeps the stored-row path. This guard is load-bearing, not
#: decorative -- measured 2026-09-17, ``recognize_kpi("market share") -> WS3-BI-008``.
_STORED_ONLY_KEYS = frozenset({"market_share"})


def canonical_volume_kpi_id(kpi_name: Optional[str]) -> Optional[str]:
    """The canonical volume KPI a data-query ``kpi_name`` names, else None."""
    if not kpi_name:
        return None
    # Ask the shared #2130 vocabulary which STORED key the name means rather than
    # re-deriving it: a narrower local normalization sent "market shares" and
    # "market/share" to canonical TRx Share (codex iter9 HIGH).
    from src.kpi.business_metric_vocabulary import canonical_business_metric_name

    stored_key = canonical_business_metric_name(kpi_name)
    if stored_key in _STORED_ONLY_KEYS:
        return None
    if stored_key is None and "(" in kpi_name:
        # The vocabulary accepts a parenthetical only when it restates the SAME
        # quantity ("Total Prescriptions (TRx)"); it refuses one naming another
        # axis ("Total Prescriptions (patients)"). Broad recognition below would
        # still find the base name and route it canonically (codex iter10 HIGH).
        return None
    from src.services.kpi_resolution import recognize_kpi

    kpi = recognize_kpi(kpi_name)
    return kpi.id if kpi is not None and kpi.id in CANONICAL_VOLUME_KPI_IDS else None


def _iso(value: Any) -> Optional[str]:
    return str(value)[:10] if value else None


def _utc_today() -> date:
    """The clock ``_get_time_filter`` uses (UTC)."""
    return datetime.now(timezone.utc).date()


def _completed_on(data_month: str) -> date:
    """The first day after the month: the day its figure became complete."""
    year, month = int(data_month[:4]), int(data_month[5:7])
    return date(year + month // 12, month % 12 + 1, 1)


async def canonical_volume_stored_rows(
    kpi_name: Optional[str],
    filters: Dict[str, Any],
    window_start: str,
    limit: int,
    *,
    calculator: Any = None,
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """The tool response for a canonical volume KPI, or None for any other ``kpi_name``.

    ``window_start`` (the tool's lookback start, ``since.date()``) gates eligibility
    under ``ELIGIBILITY_RULE``; it never selects the month. ``limit`` is accepted for
    the hook's signature -- the answer is one aggregate row or an explicit refusal, so
    there is nothing to cap.

    Returning ``None`` means "not mine": the caller must fall through to the stored-row
    path, and the calculator is not called at all.
    """
    kpi_id = canonical_volume_kpi_id(kpi_name)
    if kpi_id is None:
        return None
    brand = filters.get("brand")
    region = filters.get("region")
    # The context kpi_calculate_tool builds for the same scope: brand / region only when named.
    context: Dict[str, Any] = {}
    if brand:
        context["brand"] = brand
    if region:
        context["region"] = region
    if calculator is None:
        from src.api.routes.kpi import get_kpi_calculator

        calculator = get_kpi_calculator()
    # calculate() is synchronous and hits Redis/Postgres; keep the event loop free.
    result = await asyncio.to_thread(calculator.calculate, kpi_id, context=context)
    metadata = result.metadata or {}
    envelope: Dict[str, Any] = {
        "success": result.error is None,
        "query_type": "kpi",
        "kpi_id": kpi_id,
        "canonical_aggregate": True,
        "filters_applied": dict(filters),
        "lookback_start": window_start,
        "data_source": "synthetic" if metadata.get("include_synthetic") else "database",
        "measure_basis": BUSINESS_METRICS_BASIS,
        "cross_substrate_conflict": None,
        "note": NOTE,
    }
    if result.error is not None:
        return {
            **envelope,
            "count": 0,
            "data": [],
            "data_month": None,
            "data_through": None,
            "error": str(result.error),
        }
    computed = metadata.get("context") or {}
    data_month = _iso(computed.get("data_month"))
    data_through = _iso(computed.get("data_through"))
    completed = _completed_on(data_month) if data_month else None
    if completed is None or completed < date.fromisoformat(window_start[:10]):
        # Never an out-of-range headline: name the frontier, give no value.
        return {
            **envelope,
            "success": False,
            "count": 0,
            "data": [],
            "data_month": None,
            "data_through": None,
            "error": (
                f"out of range: the latest complete month is {data_month} (complete on "
                f"{completed}), before the requested lookback starting {window_start}; "
                "call kpi_calculate_tool with window=... for that month"
            ),
            "out_of_range": {
                "requested_since": window_start,
                "as_of": (today or _utc_today()).isoformat(),
                "frontier_month": data_month,
                "frontier_complete_on": completed.isoformat() if completed else None,
                "rule": ELIGIBILITY_RULE,
                "use_instead": "kpi_calculate_tool(window=...)",
            },
        }
    row = {
        "metric_date": data_month,
        "metric_name": _ROW_NAME[kpi_id],
        "kpi_id": kpi_id,
        "brand": brand or "",
        "region": region or "",
        "value": result.value,
        "data_through": data_through,
    }
    return {
        **envelope,
        "count": 1,
        "data": [row],
        "data_month": data_month,
        "data_through": data_through,
    }
