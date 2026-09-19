"""kpi_history backfill for the canonical Rx-volume KPIs WS3-BI-005..008 (canonical TRx lane).

A DIRECT monthly source, like ROI: ``business_metrics`` already is a monthly
brand x region series, so every point is a SUM of real rows — no recount, no
synthesis. National (``brand=''``) = sum over brands x regions; per brand = sum
over regions; per region and brand x region straight from
``business_metrics.region`` (#1536, the migration-125 ROI precedent). WS3-BI-008
TRx Share = brand / portfolio within the same month (and region), per brand and
brand x region only — a portfolio share of the portfolio is undefined.

Synthetic policy (codex r1 HIGH): the history reads exactly the rows the headline
statement reads — ``is_synthetic = false`` unless ``kpi_include_synthetic()`` (the
base-vs-``_include_synthetic`` twin choice) — and a point is ``is_synthetic`` iff
at least one of its input rows was (#895 taint rule). Real-derived history is
never marked synthetic, and real and synthetic rows are never mixed silently.

The in-progress calendar month is never emitted — the same rule migration 143's
statements enforce.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple, TypeVar

from src.kpi.synthetic_mode import kpi_include_synthetic
from src.kpi.volume_family import has_dimensions

CANONICAL_SOURCE = "business_metrics.value"
#: KPI id -> the ``business_metrics.metric_type`` whose rows the point sums. The
#: share reads trx rows (its numerator AND denominator are trx), which is why this
#: is not the statements' result-key map. Guarded against the deployed SQL by
#: tests/unit/test_kpi/test_history_canonical_volume.py rather than against a
#: second hand-maintained map.
CANONICAL_METRIC_FOR_KPI: Dict[str, str] = {
    "WS3-BI-005": "trx",
    "WS3-BI-006": "nrx",
    "WS3-BI-007": "nbrx",
    "WS3-BI-008": "trx",
}
SHARE_KPI_ID = "WS3-BI-008"
_PAGE_SIZE = 5000
_CACHE_KEY = "canonical_volume_rows"

Cell = Tuple[str, str, str]  # (brand, region, month ISO)
Pair = Tuple[str, str]  # (brand | region, month ISO)
Acc = Tuple[float, bool]  # (sum, any input synthetic)
#: ``_add`` is generic over its key so each accumulator keeps its OWN key type.
#: Annotating the accumulators ``Dict[Hashable, Acc]`` instead would type every
#: key as ``Hashable``, and mypy then refuses each ``for (b, m), ... in`` unpack
#: ("Hashable object is not iterable") — one loose annotation, 12 errors.
_Key = TypeVar("_Key", bound=Hashable)


def _flag(value: Any) -> bool:
    """PostgREST returns booleans; tolerate string forms (mirrors
    ``src.repositories.provenance.coerce_provenance_flag``)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "1", "yes"}
    return bool(value)


def _add(acc: Dict[_Key, Acc], key: _Key, value: float, synthetic: bool) -> None:
    total, tainted = acc.get(key, (0.0, False))
    acc[key] = (total + value, tainted or synthetic)


def _point(
    kpi_meta: Any, brand: str, region: str, month: str, value: float, synthetic: bool
) -> Dict[str, Any]:
    # Function-local: history_backfill imports this module for HANDLERS.
    from src.kpi.history_backfill import _status_for

    return {
        "kpi_id": str(kpi_meta.id),
        "brand": brand,
        "region": region,
        "metric_date": month,
        "value": value,
        "status": _status_for(kpi_meta, value),
        "source": CANONICAL_SOURCE,
        "is_synthetic": synthetic,
    }


def _cells(
    rows: Iterable[Dict[str, Any]], metric: str, as_of: date, include_synthetic: bool
) -> Dict[Cell, Acc]:
    current = as_of.replace(day=1).isoformat()
    cells: Dict[Cell, Acc] = {}
    for row in rows:
        if row.get("metric_type") != metric:
            continue
        synthetic = _flag(row.get("is_synthetic"))
        if synthetic and not include_synthetic:
            continue
        if not has_dimensions(row):
            # The ONE NULL-dimension rule the migration-143 statements also apply
            # (volume_family.DIMENSIONED_ROW_SQL, codex r2): a row without a brand or
            # region is excluded from EVERY point, national included.
            continue
        month = str(row.get("metric_date") or "")[:10]
        value = row.get("value")
        brand = str(row["brand"])
        region = str(row["region"]).lower()
        if not month or value is None or month >= current:
            continue
        _add(cells, (brand, region, month), float(value), synthetic)
    return cells


def aggregate_canonical_points(
    rows: Iterable[Dict[str, Any]], kpi_meta: Any, as_of: date, include_synthetic: bool
) -> List[Dict[str, Any]]:
    """Pure: business_metrics rows -> kpi_history points for one canonical KPI."""
    kpi_id = str(kpi_meta.id)
    cells = _cells(rows, CANONICAL_METRIC_FOR_KPI[kpi_id], as_of, include_synthetic)
    if kpi_id == SHARE_KPI_ID:
        return _share_points(cells, kpi_meta)
    national: Dict[str, Acc] = {}
    per_brand: Dict[Pair, Acc] = {}
    per_region: Dict[Pair, Acc] = {}
    for (brand, region, month), (value, synthetic) in cells.items():
        _add(national, month, value, synthetic)
        _add(per_brand, (brand, month), value, synthetic)
        _add(per_region, (region, month), value, synthetic)
    points = [_point(kpi_meta, "", "", m, v, s) for m, (v, s) in sorted(national.items())]
    points += [_point(kpi_meta, b, "", m, v, s) for (b, m), (v, s) in sorted(per_brand.items())]
    points += [_point(kpi_meta, "", r, m, v, s) for (r, m), (v, s) in sorted(per_region.items())]
    points += [_point(kpi_meta, b, r, m, v, s) for (b, r, m), (v, s) in sorted(cells.items())]
    return points


def _share_points(cells: Dict[Cell, Acc], kpi_meta: Any) -> List[Dict[str, Any]]:
    portfolio: Dict[str, Acc] = {}
    portfolio_region: Dict[Pair, Acc] = {}
    per_brand: Dict[Pair, Acc] = {}
    for (brand, region, month), (value, synthetic) in cells.items():
        _add(portfolio, month, value, synthetic)
        _add(portfolio_region, (region, month), value, synthetic)
        _add(per_brand, (brand, month), value, synthetic)
    points: List[Dict[str, Any]] = []
    for (b, m), (v, s) in sorted(per_brand.items()):
        denom, denom_synthetic = portfolio[m]
        if denom > 0:
            points.append(_point(kpi_meta, b, "", m, v / denom, s or denom_synthetic))
    for (b, r, m), (v, s) in sorted(cells.items()):
        denom, denom_synthetic = portfolio_region[(r, m)]
        if denom > 0:
            points.append(_point(kpi_meta, b, r, m, v / denom, s or denom_synthetic))
    return points


async def fetch_canonical_rows(
    client: Any, as_of: date, include_synthetic: bool, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """trx/nrx/nbrx rows before the in-progress month under the synthetic policy."""
    key = f"{_CACHE_KEY}:{'synthetic' if include_synthetic else 'real'}"
    if cache is not None and key in cache:
        cached: List[Dict[str, Any]] = cache[key]
        return cached
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        query = (
            client.table("business_metrics")
            .select("metric_id,metric_date,metric_type,brand,region,value,is_synthetic")
            .in_("metric_type", ["trx", "nrx", "nbrx"])
            .lt("metric_date", as_of.replace(day=1).isoformat())
        )
        if not include_synthetic:
            query = query.eq("is_synthetic", False)
        result = await query.order("metric_id").range(offset, offset + _PAGE_SIZE - 1).execute()
        page = result.data or []
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    if cache is not None:
        cache[key] = rows
    return rows


async def backfill_canonical_volume(
    client: Any,
    kpi_meta: Any,
    cache: Optional[Dict[str, Any]] = None,
    as_of: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """HANDLERS entry for WS3-BI-005..008 (the synthetic mode is read at call time)."""
    as_of = as_of or date.today()
    include_synthetic = kpi_include_synthetic()
    rows = await fetch_canonical_rows(client, as_of, include_synthetic, cache)
    return aggregate_canonical_points(rows, kpi_meta, as_of, include_synthetic)
