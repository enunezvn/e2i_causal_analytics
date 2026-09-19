"""Canonical monthly Rx-volume series — the read API a forecaster calls (canonical TRx lane).

Returns a brand (optionally region) monthly TRx / NRx / NBRx series from the
canonical ``business_metrics`` rows through the vetted
``canonical_volume_monthly_series`` statement (migration 143), with:

* COMPLETE calendar months only: the in-progress month is excluded twice —
  by the statement (``metric_date < date_trunc('month', CURRENT_DATE)``) and
  here against ``as_of`` — and reported in ``dropped_in_progress``;
* months with fewer aggregated rows than the series' fullest month (a region
  or brand missing that month) excluded and reported in ``dropped_incomplete``
  instead of being passed off as a level;
* ``data_through`` = the last day of the last complete month served.

Forecasting itself (forecast_kpi_tool, TimesFM worker, 6.5 routing) is Lane B;
this module is its data seam and has no forecasting logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.kpi.synthetic_mode import resolve_kpi_query_id

SERIES_QUERY_ID = "canonical_volume_monthly_series"
#: Mirrors the statement's OWN guard (``WHERE $1::text IN ('trx','nrx','nbrx')``),
#: which is the capability; tests/unit/test_kpi/test_canonical_volume_series.py
#: checks this tuple against that SQL rather than trusting two declarations.
SUPPORTED_METRICS: Tuple[str, ...] = ("trx", "nrx", "nbrx")


class CanonicalSeriesError(ValueError):
    """The series cannot be served honestly (unsupported metric, no client)."""


@dataclass(frozen=True)
class MonthlyVolumePoint:
    month: date
    value: float
    n_rows: int


@dataclass(frozen=True)
class CanonicalVolumeSeries:
    metric: str
    brand: Optional[str]
    region: Optional[str]
    points: Tuple[MonthlyVolumePoint, ...]
    data_through: Optional[date]
    as_of: date
    dropped_in_progress: Tuple[date, ...]
    dropped_incomplete: Tuple[date, ...]
    query_id: str


def month_end(month: date) -> date:
    first_of_next = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return first_of_next - timedelta(days=1)


def _check_metric(metric: str) -> None:
    if metric not in SUPPORTED_METRICS:
        raise CanonicalSeriesError(
            f"unsupported metric {metric!r}; the canonical volume series serves {SUPPORTED_METRICS}"
        )


def shape_monthly_series(
    rows: Iterable[Dict[str, Any]],
    *,
    metric: str,
    brand: Optional[str],
    region: Optional[str],
    as_of: date,
    query_id: str,
) -> CanonicalVolumeSeries:
    """Pure: statement rows -> a complete-month series."""
    _check_metric(metric)
    current = as_of.replace(day=1)
    parsed: List[Tuple[date, float, int]] = []
    for row in rows:
        raw, value = row.get("metric_date"), row.get("value")
        if raw is None or value is None:
            continue
        month = date.fromisoformat(str(raw)[:10]).replace(day=1)
        parsed.append((month, float(value), int(row.get("n_rows") or 0)))
    parsed.sort()
    in_progress = tuple(m for m, _, _ in parsed if m >= current)
    complete = [(m, v, n) for m, v, n in parsed if m < current]
    fullest = max((n for _, _, n in complete), default=0)
    incomplete = tuple(m for m, _, n in complete if n < fullest)
    points = tuple(MonthlyVolumePoint(m, v, n) for m, v, n in complete if n == fullest)
    return CanonicalVolumeSeries(
        metric=metric,
        brand=brand,
        region=region,
        points=points,
        data_through=month_end(points[-1].month) if points else None,
        as_of=as_of,
        dropped_in_progress=in_progress,
        dropped_incomplete=incomplete,
        query_id=query_id,
    )


def fetch_canonical_volume_series(
    metric: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    *,
    client: Any = None,
    as_of: Optional[date] = None,
) -> CanonicalVolumeSeries:
    """Read one canonical monthly series (sync supabase client, like measure_basis)."""
    _check_metric(metric)
    if client is None:
        from src.api.dependencies.supabase_client import get_supabase

        client = get_supabase()
    if client is None:
        raise CanonicalSeriesError("no Supabase client: the canonical series cannot be read")
    query_id = resolve_kpi_query_id(SERIES_QUERY_ID)
    rows = (
        client.rpc("kpi_query", {"query_id": query_id, "params": [metric, brand, region]})
        .execute()
        .data
        or []
    )
    return shape_monthly_series(
        rows,
        metric=metric,
        brand=brand,
        region=region,
        as_of=as_of or date.today(),
        query_id=query_id,
    )


async def afetch_canonical_volume_series(
    metric: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    *,
    client: Any = None,
    as_of: Optional[date] = None,
) -> CanonicalVolumeSeries:
    """Async wrapper (the reader is the sync client) for async callers."""
    return await asyncio.to_thread(
        fetch_canonical_volume_series, metric, brand, region, client=client, as_of=as_of
    )
