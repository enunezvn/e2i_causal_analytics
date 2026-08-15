"""On-the-fly axis-segmented monthly KPI history (severity tier / line of therapy).

The materialized ``kpi_history`` table (migration 079) has no patient-segment
dimension, so ``/api/kpis/{id}/history`` cannot honestly serve per-tier
trends (threading an axis param into that read would silently return the
unsegmented series). This module computes them live from the migration-110
registry queries (``business_impact_*_monthly_by_{segment,line}``), which
return (month_start, bucket, value) rows for ALL buckets of the axis in one
``kpi_query`` RPC call, plus the global prescription date range
(data_min/data_max) used for edge trimming.

Month bucketing (calendar month) and partial-edge-month trimming mirror
:mod:`src.kpi.history_backfill` exactly, so the per-tier lines PARTITION the
headline materialized series month by month (validated read-only against the
live DB 2026-07-18: Remibrutinib 2026-06 TRx low+medium+high = 272+715+335 =
1322 == the kpi_history headline point; LOT 0-3 = 331+316+313+362 likewise).

Only the Rx-volume family supports axes — the same set migration 105 scoped:
WS3-BI-005 TRx, WS3-BI-006 NRx, WS3-BI-007 NBRx. TRx Share has no axis
monthly variant (it has never had a windowed sibling either).
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from src.kpi.history_backfill import _complete_months, _to_date
from src.kpi.synthetic_mode import monthly_axis_query_id

logger = logging.getLogger(__name__)

# KPI registry code -> base query family (mirrors BusinessImpactCalculator).
SEGMENTED_KPI_QUERY_FAMILIES: Dict[str, str] = {
    "WS3-BI-005": "business_impact_trx",
    "WS3-BI-006": "business_impact_nrx",
    "WS3-BI-007": "business_impact_nbrx",
}

# Canonical axis names as the API speaks them -> migration-110 suffix part.
AXIS_SUFFIXES: Dict[str, str] = {
    "segment": "segment",
    "therapy_line": "line",
}

# patient_journeys.segment_assignment values, in display order.
SEGMENT_BUCKETS: List[str] = ["low_severity", "medium_severity", "high_severity"]
# patient_journeys.prior_therapy_lines domain (0-3), as the text buckets the
# migration-110 statements emit.
THERAPY_LINE_BUCKETS: List[str] = ["0", "1", "2", "3"]

_SEGMENT_LABELS = {
    "low_severity": "Low severity",
    "medium_severity": "Medium severity",
    "high_severity": "High severity",
}


def canonical_buckets(axis: str) -> List[str]:
    """Display-ordered bucket keys for an axis."""
    return SEGMENT_BUCKETS if axis == "segment" else THERAPY_LINE_BUCKETS


def bucket_label(axis: str, key: str) -> str:
    """Human label for a bucket key.

    ``prior_therapy_lines`` counts PRIOR lines, so '0' is treatment-naive for
    the current therapy — label it literally rather than inventing a 1L/2L
    remapping the substrate does not define.
    """
    if axis == "segment":
        return _SEGMENT_LABELS.get(key, key)
    return f"{key} prior line{'' if key == '1' else 's'}"


def segmented_query_id(kpi_id: str, axis: str) -> str:
    """The registry query this KPI/axis pair actually runs (#1640).

    Shared with the route so the substrate it declares comes from the query
    that RAN -- including the ``_include_synthetic`` variant, which reads a
    different set of tables. Two copies of this derivation could drift into a
    label that names the wrong query.
    """
    base = SEGMENTED_KPI_QUERY_FAMILIES[kpi_id]
    return monthly_axis_query_id(base, axis=AXIS_SUFFIXES[axis])


async def fetch_segmented_rows(
    kpi_id: str, *, axis: str, brand: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Run the migration-110 monthly-by-axis query for one KPI via kpi_query.

    Returns the raw (month_start, bucket, value, data_min, data_max) rows;
    empty list on error (logged), matching the sibling repository behavior.
    """
    import inspect

    query_id = segmented_query_id(kpi_id, axis)
    try:
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        result_or_coro = client.rpc(
            "kpi_query", {"query_id": query_id, "params": [brand]}
        ).execute()
        result = await result_or_coro if inspect.isawaitable(result_or_coro) else result_or_coro
        return result.data if getattr(result, "data", None) else []
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to fetch segmented history via {query_id}: {e}", exc_info=True)
        return []


def shape_segmented_series(
    rows: List[Dict[str, Any]],
    *,
    axis: str,
    value: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Shape raw monthly-by-axis rows into per-bucket series + data_through.

    Mirrors the backfill's frontier honesty: only calendar months FULLY
    covered by [data_min, data_max] are emitted (partial edge months would
    render a fake cliff), and months inside the span with no events are
    genuine zeros, so every bucket is zero-filled across the trimmed span.
    """
    parsed: Dict[Tuple[date, str], float] = {}
    data_min: Optional[date] = None
    data_max: Optional[date] = None
    observed: List[str] = []
    for r in rows:
        m = _to_date(r.get("month_start"))
        bucket = r.get("bucket")
        if m is None or bucket is None or r.get("value") is None:
            continue
        bucket = str(bucket)
        parsed[(m, bucket)] = float(r["value"])
        if bucket not in observed:
            observed.append(bucket)
        if data_min is None:
            data_min = _to_date(r.get("data_min"))
            data_max = _to_date(r.get("data_max"))

    if not parsed or data_min is None or data_max is None:
        return [], data_max.isoformat() if data_max else None

    months = _complete_months([data_min, data_max])
    lo = _to_date(start_date) if start_date else None
    hi = _to_date(end_date) if end_date else None
    months = [m for m in months if (lo is None or m >= lo) and (hi is None or m <= hi)]

    # Canonical buckets first (zero-filled even when absent — a tier with no
    # events is a genuine zero series), then any unexpected extras observed.
    buckets = canonical_buckets(axis) + [b for b in observed if b not in canonical_buckets(axis)]
    if value is not None:
        buckets = [b for b in buckets if b == value]

    series: List[Dict[str, Any]] = []
    for b in buckets:
        points = [{"metric_date": m.isoformat(), "value": parsed.get((m, b), 0.0)} for m in months]
        series.append(
            {"key": b, "label": bucket_label(axis, b), "count": len(points), "points": points}
        )
    return series, data_max.isoformat()
