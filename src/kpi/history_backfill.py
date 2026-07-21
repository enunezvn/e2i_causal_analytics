"""
KPI history walk-forward backfill.
=================================

Materializes REAL monthly KPI points into ``kpi_history`` (migration 079) for the
Time-Series "KPI history" view. Two honest mechanisms, dispatched per KPI:

1. **Direct monthly source** — the KPI's value already lives as a monthly series
   (e.g. ``WS3-BI-010`` ROI <- ``business_metrics.roi``). Read it, group by month.
2. **As-of recompute** (Batch 2) — recompute the KPI "as of" each month from a
   dated source table, mirroring the CURRENT ``kpi_query_registry`` semantics of
   each KPI (numerators/denominators/filters byte-for-byte where possible; the
   only change is the window: calendar month instead of trailing-30-days-from-
   frontier).

Anti-fabrication: a KPI gets a handler ONLY when its history can be produced from
real, time-dimensioned data. No synthesis, no smoothing, no invented points. KPIs
with no honest temporal source are intentionally NOT registered (the UI shows an
empty-state, never a synthesized flat line).

Batch-2 coverage (each handler documents the registry query it mirrors):

- ``WS3-BI-005/006`` TRx/NRx  <- treatment_events.event_date (global + per-brand)
- ``WS3-BI-007/008`` NBRx/TRx Share <- treatment_events (per-brand ONLY: the live
  calculator fails loud without a brand — a "global NBRx/share" is undefined, so
  no ``brand=''`` rows are fabricated)
- ``WS3-BI-009`` Conversion Rate <- triggers x treatment_events (30-day
  trigger->prescription window, right-censored at the frontier exactly like the
  live query)
- ``WS3-BI-001/002`` MAU/WAU <- user_sessions.session_start (the substrate keeps
  only ~90 days of sessions, so this is a SHORT honest series — 2-3 points)
- ``WS2-TR-001/004/005/006/007/008`` <- triggers.trigger_timestamp
  (TR-004/006 denominator = delivered/viewed ONLY, per migrations 092/090)
- ``BR-001`` <- treatment_events UAS7 baseline events (monthly patient cohorts)
- ``BR-003`` <- patient_journeys x treatment_events PNH tests (cumulative AS-OF
  recompute at each month-end — both numerator and denominator are real dated
  events, so the series is an evolving rate, not a coverage ramp)
- ``BR-004`` <- first Kisqali Rx per patient x journey_start_date (monthly
  first-Rx cohorts, median days)

Batch-3 coverage:

- ``BR-002`` <- hcp_intent_surveys.survey_date (monthly mean
  intent_to_prescribe_change over quality Remibrutinib surveys — the
  ``survey_month`` series behind ``v_kpi_intent_to_prescribe``)

Intentionally SKIPPED (no honest monthly recast of the CURRENT semantics):

- ``CM-001..005`` — per-analysis causal estimates carrying CIs/p-values, not a
  time series.
- ``WS1-MP-*`` — model-metric trends are already served from the walk-forward
  ``ml_performance_metrics`` table (Model Performance page); duplicating them
  into kpi_history would blur provenance.
- ``WS1-DQ-*``, ``WS3-BI-003/004``, ``BR-005`` — coverage/eligibility against a
  present-state universe (undated ``coverage_status``, as-of-now eligibility
  views): a backdated recompute would be a ramp artifact, not KPI history.
- ``WS2-TR-002`` (recall) — the live query windows BOTH the outcome set and the
  trigger-precede join; the two windows straddle month boundaries, so a
  calendar-month recast is not a clean mirror of the current reading. Deferred.
- ``WS2-TR-003`` (uplift) — the live query is an ALL-TIME two-arm contrast with
  no time dimension; slicing it by month would be a different experiment
  reading, not this KPI's history.

Reseed churn: the weekly synthetic reseed (``scripts/reseed_synthetic.sh``,
``--anchor-to-now``) SHIFTS every substrate timestamp, so months from a previous
seed would otherwise linger in kpi_history forever. ``run_backfill`` therefore
DELETES the existing (kpi_id, source) rows before upserting the fresh set
(replace semantics), and the reseed script invokes this module as its final step.

Frontier honesty: interval handlers emit points ONLY for calendar months fully
covered by the handler's source rows (first/last partial months are dropped);
the cumulative as-of handler (BR-003) emits a point for every month-end at or
before its sources' frontier, because an as-of reading is complete by
construction. ``is_synthetic=True`` on every point.

Run:  python -m src.kpi.history_backfill           # all registered KPIs
      python -m src.kpi.history_backfill WS3-BI-010 # one KPI
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from src.kpi.models import KPIStatus

logger = logging.getLogger(__name__)

# A handler returns a list of point dicts ready for kpi_history upsert.
# Signature: (client, kpi_meta, cache=None) — ``cache`` is a per-run dict that
# lets handlers share paginated source fetches (six trigger KPIs read the same
# 37k rows; four Rx KPIs the same 70k).
Handler = Callable[..., Awaitable[List[Dict[str, Any]]]]

# ---------------------------------------------------------------------------
# Per-KPI provenance tags. run_backfill deletes (kpi_id, source) rows before
# upserting so a reseed's shifted months replace — never accumulate on — the
# previous seed's.
# ---------------------------------------------------------------------------
HANDLER_SOURCES: Dict[str, str] = {
    "WS3-BI-010": "business_metrics.roi",
    "WS3-BI-005": "treatment_events.event_date",
    "WS3-BI-006": "treatment_events.event_date",
    "WS3-BI-007": "treatment_events.event_date",
    "WS3-BI-008": "treatment_events.event_date",
    "WS3-BI-009": "triggers+treatment_events",
    "WS3-BI-001": "user_sessions.session_start",
    "WS3-BI-002": "user_sessions.session_start",
    "WS2-TR-001": "triggers.trigger_timestamp",
    "WS2-TR-004": "triggers.trigger_timestamp",
    "WS2-TR-005": "triggers.trigger_timestamp",
    "WS2-TR-006": "triggers.trigger_timestamp",
    "WS2-TR-007": "triggers.trigger_timestamp",
    "WS2-TR-008": "triggers.trigger_timestamp",
    "BR-001": "treatment_events.uas7_baseline",
    "BR-002": "hcp_intent_surveys.survey_date",
    "BR-003": "patient_journeys+treatment_events.pnh",
    "BR-004": "treatment_events+patient_journeys",
}

# Direction mirror of the live calculators (NOT re-derived from the YAML):
# trigger_performance.py -> {TR-005, TR-006, TR-007, TR-008};
# brand_specific.py      -> {BR-001, BR-004}. Everything else higher-is-better.
LOWER_IS_BETTER: frozenset = frozenset(
    {"WS2-TR-005", "WS2-TR-006", "WS2-TR-007", "WS2-TR-008", "BR-001", "BR-004"}
)

# Page size for paginated PostgREST reads. The local PostgREST has no
# db-max-rows cap, but paging keeps memory bounded and survives one being set.
_PAGE_SIZE = 20000

# BR-001: UAS7 cutoff — mirrors brand_specific._calc_remi_ah_uncontrolled's
# context default (EAACI/GA2LEN guideline cutoff, PMID 34536239).
_UAS7_UNCONTROLLED_THRESHOLD = 7.0

# BR-003: real PNH LOINCs — mirrors the migration-091 registry SQL verbatim.
_PNH_LOINCS = frozenset({"55164-8", "35468-8", "90735-2", "44007-3"})


def _status_for(kpi_meta: Any, value: float, lower_is_better: bool = False) -> Optional[str]:
    """Evaluate a value against the KPI threshold; None when unavailable."""
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


# ---------------------------------------------------------------------------
# Month/date helpers
# ---------------------------------------------------------------------------


def _to_date(value: Any) -> Optional[date]:
    """Parse a PostgREST date/timestamptz string to a date (None on garbage)."""
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _next_month(d: date) -> date:
    return date(d.year + (d.month == 12), (d.month % 12) + 1, 1)


def _month_end(month: date) -> date:
    return _next_month(month) - timedelta(days=1)


def _complete_months(dates: Iterable[date]) -> List[date]:
    """Calendar months FULLY covered by [min(dates), max(dates)].

    A leading month is complete only when the data starts on its 1st; a
    trailing month only when the data reaches its last day. Partial edge
    months are dropped — a mid-month frontier would otherwise render an
    artificially truncated point that reads as a real collapse/spike.
    """
    dates = list(dates)
    if not dates:
        return []
    lo, hi = min(dates), max(dates)
    first = _month_start(lo) if lo.day == 1 else _next_month(lo)
    if hi == _month_end(_month_start(hi)):
        last = _month_start(hi)
    else:
        last = _month_start(_month_start(hi) - timedelta(days=1))
    months: List[date] = []
    m = first
    while m <= last:
        months.append(m)
        m = _next_month(m)
    return months


# ---------------------------------------------------------------------------
# Shared fetch + point helpers
# ---------------------------------------------------------------------------


async def _fetch_all(
    client: Any,
    table: str,
    columns: str,
    order_col: str,
    eq_filters: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch ALL matching rows via deterministic keyset-free pagination.

    Ordered by a UNIQUE column so offset pages never skip/duplicate rows.
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        rows: List[Dict[str, Any]] = cache[cache_key]
        return rows
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        query = client.table(table).select(columns)
        for col, val in (eq_filters or {}).items():
            query = query.eq(col, val)
        result = await query.order(order_col).range(offset, offset + _PAGE_SIZE - 1).execute()
        page = result.data or []
        out.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    if cache is not None and cache_key is not None:
        cache[cache_key] = out
    return out


def _point(kpi_meta: Any, brand: str, month: date, value: float) -> Dict[str, Any]:
    """Build one kpi_history row dict (metric_date = month start)."""
    kpi_id = str(kpi_meta.id)
    return {
        "kpi_id": kpi_id,
        "brand": brand,
        "region": "",
        "metric_date": month.isoformat(),
        "value": value,
        "status": _status_for(kpi_meta, value, kpi_id in LOWER_IS_BETTER),
        "source": HANDLER_SOURCES[kpi_id],
        "is_synthetic": True,
    }


async def _fetch_prescriptions(
    client: Any, cache: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """All prescription events (shared by the four Rx KPIs + BI-009 + BR-004)."""
    return await _fetch_all(
        client,
        "treatment_events",
        "patient_id,brand,event_date,sequence_number",
        "treatment_event_id",
        eq_filters={"event_type": "prescription"},
        cache=cache,
        cache_key="prescriptions",
    )


async def _fetch_triggers(client: Any, cache: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All triggers (shared by the six WS2-TR KPIs + BI-009)."""
    return await _fetch_all(
        client,
        "triggers",
        "trigger_id,patient_id,trigger_timestamp,delivery_status,acceptance_status,"
        "false_positive_flag,lead_time_days,outcome_tracked,outcome_value,"
        "previous_trigger_id,change_failed",
        "trigger_id",
        cache=cache,
        cache_key="triggers",
    )


async def _fetch_journeys(client: Any, cache: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All patient journeys (shared by BR-003 denominator + BR-004 join)."""
    return await _fetch_all(
        client,
        "patient_journeys",
        "patient_id,journey_start_date,brand,primary_diagnosis_code",
        "patient_journey_id",
        cache=cache,
        cache_key="patient_journeys",
    )


def _rx_dated(rows: List[Dict[str, Any]]) -> List[tuple]:
    """(date, row) pairs for rows with a parseable event_date."""
    out = []
    for r in rows:
        d = _to_date(r.get("event_date"))
        if d is not None:
            out.append((d, r))
    return out


# ---------------------------------------------------------------------------
# WS3-BI-010 — Batch 1 (direct monthly source)
# ---------------------------------------------------------------------------


async def _backfill_roi(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
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

    def _roi_point(brand: str, date_str: str, vals: List[float]) -> Dict[str, Any]:
        value = sum(vals) / len(vals)
        return {
            "kpi_id": kpi_meta.id,
            "brand": brand,
            "region": "",
            "metric_date": date_str,
            "value": value,
            "status": _status_for(kpi_meta, value),
            "source": "business_metrics.roi",
            "is_synthetic": True,
        }

    for date_str, vals in global_acc.items():
        points.append(_roi_point("", date_str, vals))
    for (brand, date_str), vals in brand_acc.items():
        points.append(_roi_point(brand, date_str, vals))
    return points


# ---------------------------------------------------------------------------
# WS3-BI-005/006/007/008 — prescription volume family (as-of monthly recount)
# ---------------------------------------------------------------------------


async def _backfill_trx(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS3-BI-005 TRx: COUNT(prescriptions) per month, global + per-brand.

    Mirrors ``business_impact_trx`` (COUNT over event_type='prescription' with
    an optional brand filter); the trailing-30-day window becomes the calendar
    month. Zero-count months inside the covered span are genuine zeros.
    """
    dated = _rx_dated(await _fetch_prescriptions(client, cache))
    months = _complete_months([d for d, _ in dated])
    per_month: Dict[date, int] = defaultdict(int)
    per_brand_month: Dict[tuple, int] = defaultdict(int)
    brands = set()
    for d, r in dated:
        m = _month_start(d)
        per_month[m] += 1
        b = r.get("brand")
        if b:
            brands.add(b)
            per_brand_month[(b, m)] += 1
    points = [_point(kpi_meta, "", m, float(per_month.get(m, 0))) for m in months]
    for b in sorted(brands):
        points.extend(_point(kpi_meta, b, m, float(per_brand_month.get((b, m), 0))) for m in months)
    return points


async def _backfill_nrx(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS3-BI-006 NRx: COUNT(prescriptions with sequence_number=1) per month.

    Mirrors ``business_impact_nrx`` (sequence_number = 1 + optional brand
    filter). Global + per-brand, calendar-month window.
    """
    dated = _rx_dated(await _fetch_prescriptions(client, cache))
    months = _complete_months([d for d, _ in dated])
    per_month: Dict[date, int] = defaultdict(int)
    per_brand_month: Dict[tuple, int] = defaultdict(int)
    brands = set()
    for d, r in dated:
        b = r.get("brand")
        if b:
            brands.add(b)
        if r.get("sequence_number") != 1:
            continue
        m = _month_start(d)
        per_month[m] += 1
        if b:
            per_brand_month[(b, m)] += 1
    points = [_point(kpi_meta, "", m, float(per_month.get(m, 0))) for m in months]
    for b in sorted(brands):
        points.extend(_point(kpi_meta, b, m, float(per_brand_month.get((b, m), 0))) for m in months)
    return points


async def _backfill_nbrx(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS3-BI-007 NBRx: patients whose FIRST prescription of brand X lands in
    the month.

    Mirrors ``business_impact_nbrx``: first_brand = MIN(event_date) per patient
    over the brand-filtered prescriptions, counted by the month the first date
    falls in. Per-brand ONLY — the live calculator fails loud without a brand
    ("new-to-brand" is undefined globally), so no brand='' rows are written.
    """
    dated = _rx_dated(await _fetch_prescriptions(client, cache))
    months = _complete_months([d for d, _ in dated])
    first_by_brand_patient: Dict[tuple, date] = {}
    for d, r in dated:
        b = r.get("brand")
        if not b:
            continue
        key = (b, r.get("patient_id"))
        prev = first_by_brand_patient.get(key)
        if prev is None or d < prev:
            first_by_brand_patient[key] = d
    per_brand_month: Dict[tuple, int] = defaultdict(int)
    brands = set()
    for (b, _pid), first in first_by_brand_patient.items():
        brands.add(b)
        per_brand_month[(b, _month_start(first))] += 1
    points: List[Dict[str, Any]] = []
    for b in sorted(brands):
        points.extend(_point(kpi_meta, b, m, float(per_brand_month.get((b, m), 0))) for m in months)
    return points


async def _backfill_trx_share(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS3-BI-008 TRx Share: brand prescriptions / category prescriptions per
    month.

    Mirrors ``business_impact_trx_share`` (brand COUNT / windowed category
    COUNT). Per-brand ONLY (the live calculator fails loud without a brand).
    Months with an empty category are skipped, mirroring NULLIF -> NULL.
    """
    dated = _rx_dated(await _fetch_prescriptions(client, cache))
    months = _complete_months([d for d, _ in dated])
    per_month: Dict[date, int] = defaultdict(int)
    per_brand_month: Dict[tuple, int] = defaultdict(int)
    brands = set()
    for d, r in dated:
        m = _month_start(d)
        per_month[m] += 1
        b = r.get("brand")
        if b:
            brands.add(b)
            per_brand_month[(b, m)] += 1
    points: List[Dict[str, Any]] = []
    for b in sorted(brands):
        for m in months:
            total = per_month.get(m, 0)
            if total == 0:
                continue
            points.append(_point(kpi_meta, b, m, per_brand_month.get((b, m), 0) / total))
    return points


async def _backfill_conversion_rate(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS3-BI-009 Conversion Rate: triggers converting to a prescription within
    30 days, grouped by trigger month.

    Mirrors ``business_impact_conversion_rate``: a trigger converts when the
    SAME patient has a prescription with trigger_date <= event_date <=
    trigger_date + 30 days. Like the live query, the follow-up window is
    right-censored at the prescription frontier (a trigger near the frontier
    that has had no time to convert counts as unconverted). Global only —
    the live query is brand-agnostic.
    """
    triggers = await _fetch_triggers(client, cache)
    rx_by_patient: Dict[Any, List[date]] = defaultdict(list)
    for d, r in _rx_dated(await _fetch_prescriptions(client, cache)):
        rx_by_patient[r.get("patient_id")].append(d)
    for dates in rx_by_patient.values():
        dates.sort()

    trig_dates: List[date] = []
    triggered: Dict[date, int] = defaultdict(int)
    converted: Dict[date, int] = defaultdict(int)
    for t in triggers:
        d = _to_date(t.get("trigger_timestamp"))
        if d is None:
            continue
        trig_dates.append(d)
        m = _month_start(d)
        triggered[m] += 1
        rx_dates = rx_by_patient.get(t.get("patient_id"))
        if not rx_dates:
            continue
        i = bisect_left(rx_dates, d)
        if i < len(rx_dates) and rx_dates[i] <= d + timedelta(days=30):
            converted[m] += 1

    points: List[Dict[str, Any]] = []
    for m in _complete_months(trig_dates):
        den = triggered.get(m, 0)
        if den == 0:
            continue
        points.append(_point(kpi_meta, "", m, converted.get(m, 0) / den))
    return points


# ---------------------------------------------------------------------------
# WS3-BI-001/002 — active users (short honest series: ~90 days of sessions)
# ---------------------------------------------------------------------------


async def _backfill_mau(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS3-BI-001 MAU: distinct user_id per calendar month.

    Mirrors the PRIMARY live path (v_kpi_active_users.monthly_active_users,
    which is already COUNT(DISTINCT user_id) grouped by month). The substrate
    keeps ~90 days of sessions, so only 2-3 full months exist — a short honest
    series beats a long fabricated one.
    """
    rows = await _fetch_all(
        client,
        "user_sessions",
        "user_id,session_start",
        "session_id",
        cache=cache,
        cache_key="user_sessions",
    )
    users_by_month: Dict[date, set] = defaultdict(set)
    dates: List[date] = []
    for r in rows:
        d = _to_date(r.get("session_start"))
        if d is None:
            continue
        dates.append(d)
        users_by_month[_month_start(d)].add(r.get("user_id"))
    return [
        _point(kpi_meta, "", m, float(len(users_by_month.get(m, set()))))
        for m in _complete_months(dates)
    ]


async def _backfill_wau(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS3-BI-002 WAU: mean of the weekly distinct-user counts for the
    Monday-anchored weeks that fall FULLY inside the month.

    The live WAU is a point-in-time trailing-7-day distinct count; its honest
    monthly summary is the average of the complete weekly readings inside the
    month (weeks straddling a month boundary are attributed to neither month).
    Documented choice: mean (not max) — a typical week, not the best week.
    """
    rows = await _fetch_all(
        client,
        "user_sessions",
        "user_id,session_start",
        "session_id",
        cache=cache,
        cache_key="user_sessions",
    )
    users_by_week: Dict[date, set] = defaultdict(set)
    dates: List[date] = []
    for r in rows:
        d = _to_date(r.get("session_start"))
        if d is None:
            continue
        dates.append(d)
        users_by_week[d - timedelta(days=d.weekday())].add(r.get("user_id"))
    points: List[Dict[str, Any]] = []
    for m in _complete_months(dates):
        end = _month_end(m)
        weekly = [
            len(users)
            for week_start, users in users_by_week.items()
            if week_start >= m and week_start + timedelta(days=6) <= end
        ]
        if not weekly:
            continue
        points.append(_point(kpi_meta, "", m, sum(weekly) / len(weekly)))
    return points


# ---------------------------------------------------------------------------
# WS2-TR-* — trigger performance (monthly recast of the registry ratios)
# ---------------------------------------------------------------------------


def _is_delivered(row: Dict[str, Any]) -> bool:
    return row.get("delivery_status") in ("delivered", "viewed")


async def _trigger_monthly_ratio(
    client: Any,
    kpi_meta: Any,
    cache: Optional[Dict[str, Any]],
    numerator: Callable[[Dict[str, Any]], bool],
    denominator: Callable[[Dict[str, Any]], bool],
    mature_days: int = 0,
) -> List[Dict[str, Any]]:
    """Shared monthly num/den recast over triggers.trigger_timestamp.

    Months with an empty denominator are skipped (mirrors NULLIF -> NULL: no
    fabricated 0.0 on an empty cohort).

    ``mature_days`` > 0 excludes triggers whose forward outcome window has not
    fully elapsed at the data frontier (max trigger date): a trigger fired 5
    days before the frontier has only 5 days of its 30d conversion window in
    the data, so counting it would read as a false decline in the final month.
    Calendar-completeness (``_complete_months``) still uses ALL dates — a month
    is complete or not regardless of which of its triggers have matured.

    Rows with an unparseable ``trigger_timestamp`` are dropped at bucketing
    (they have no month), so every bucketed row carries a valid parsed date.
    """
    by_month: Dict[date, List[Tuple[date, Dict[str, Any]]]] = defaultdict(list)
    dates: List[date] = []
    for r in await _fetch_triggers(client, cache):
        d = _to_date(r.get("trigger_timestamp"))
        if d is None:
            continue
        dates.append(d)
        by_month[_month_start(d)].append((d, r))
    cutoff: Optional[date] = None
    if mature_days > 0 and dates:
        cutoff = max(dates) - timedelta(days=mature_days)
    points: List[Dict[str, Any]] = []
    for m in _complete_months(dates):
        pairs = by_month.get(m, [])
        if cutoff is not None:
            pairs = [(d, r) for d, r in pairs if d <= cutoff]
        den = sum(1 for _, r in pairs if denominator(r))
        if den == 0:
            continue
        num = sum(1 for _, r in pairs if numerator(r))
        points.append(_point(kpi_meta, "", m, num / den))
    return points


async def _backfill_tr001_precision(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS2-TR-001 definition v2 (migration 113): accepted-and-converted /
    accepted-and-tracked — the declared "trigger accepted AND downstream
    outcome achieved" truth shape.

    Mirrors ``trigger_performance_precision`` v2 EXACTLY (lockstep: a backfill
    computing the old tracked-only ratio would write historical points on a
    different definition than the live reading), including the 30d maturation
    guard — the live SQL scores only triggers whose conversion window has fully
    elapsed (window (frontier-60d, frontier-30d]); without the same guard here
    the final backfilled month would read a false decline from late-month
    triggers whose windows extend past the data frontier.
    """
    return await _trigger_monthly_ratio(
        client,
        kpi_meta,
        cache,
        numerator=lambda r: r.get("acceptance_status") == "accepted"
        and bool(r.get("outcome_tracked"))
        and float(r.get("outcome_value") or 0) > 0,
        denominator=lambda r: r.get("acceptance_status") == "accepted"
        and bool(r.get("outcome_tracked")),
        mature_days=30,
    )


async def _backfill_tr004_acceptance(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS2-TR-004: accepted / delivered-or-viewed.

    Mirrors ``trigger_performance_acceptance_rate`` (migration 092, #1124): the
    denominator counts ONLY delivery_status IN ('delivered','viewed') — never
    all-non-null — and the numerator counts acceptance_status='accepted'
    unrestricted, exactly as the registry SQL does.
    """
    return await _trigger_monthly_ratio(
        client,
        kpi_meta,
        cache,
        numerator=lambda r: r.get("acceptance_status") == "accepted",
        denominator=_is_delivered,
    )


async def _backfill_tr005_false_alert(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS2-TR-005: false_positive_flag / all triggers (lower-is-better).

    Mirrors ``trigger_performance_false_alert_rate`` (#1118 tracked flags).
    """
    return await _trigger_monthly_ratio(
        client,
        kpi_meta,
        cache,
        numerator=lambda r: bool(r.get("false_positive_flag")),
        denominator=lambda r: True,
    )


async def _backfill_tr006_override(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS2-TR-006: overridden / delivered-or-viewed (lower-is-better).

    Mirrors ``trigger_performance_override_rate`` (migration 090, #1119
    delivered denominator).
    """
    return await _trigger_monthly_ratio(
        client,
        kpi_meta,
        cache,
        numerator=lambda r: r.get("acceptance_status") == "overridden",
        denominator=_is_delivered,
    )


async def _backfill_tr007_lead_time(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS2-TR-007: median lead_time_days per month (lower-is-better).

    Mirrors ``trigger_performance_lead_time`` — PERCENTILE_CONT(0.5) over
    non-null lead_time_days; statistics.median matches (mean of the middle
    two on even n). Months with no non-null values are skipped.
    """
    by_month: Dict[date, List[float]] = defaultdict(list)
    dates: List[date] = []
    for r in await _fetch_triggers(client, cache):
        d = _to_date(r.get("trigger_timestamp"))
        if d is None:
            continue
        dates.append(d)
        lead = r.get("lead_time_days")
        if lead is not None:
            by_month[_month_start(d)].append(float(lead))
    points: List[Dict[str, Any]] = []
    for m in _complete_months(dates):
        vals = by_month.get(m)
        if not vals:
            continue
        points.append(_point(kpi_meta, "", m, float(statistics.median(vals))))
    return points


async def _backfill_tr008_cfr(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """WS2-TR-008: change_failed / (previous_trigger_id IS NOT NULL)
    (lower-is-better). Mirrors ``trigger_performance_cfr``.
    """
    return await _trigger_monthly_ratio(
        client,
        kpi_meta,
        cache,
        numerator=lambda r: bool(r.get("change_failed")),
        denominator=lambda r: r.get("previous_trigger_id") is not None,
    )


# ---------------------------------------------------------------------------
# BR-* — brand-specific (written under brand='' — each KPI is single-brand by
# definition, and both history consumers query the global scope)
# ---------------------------------------------------------------------------


async def _backfill_br001_ah_uncontrolled(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """BR-001: share of the month's UAS7-assessed patients that are
    uncontrolled (any UAS7 >= 7 that month). Lower-is-better.

    Mirrors ``brand_specific_remi_ah_uncontrolled`` (per-patient
    bool_or(UAS7 >= threshold) over baseline_antihistamine R06A events) with
    the all-time cohort recast to the month each baseline assessment lands in.
    The UAS7 cutoff mirrors the calculator's context default (7).
    """
    rows = await _fetch_all(
        client,
        "treatment_events",
        "patient_id,event_date,lab_values",
        "treatment_event_id",
        eq_filters={
            "brand": "Remibrutinib",
            "event_subtype": "baseline_antihistamine",
            "drug_class": "R06A",
        },
        cache=cache,
        cache_key="br001_events",
    )
    # (month) -> patient -> uncontrolled?
    by_month: Dict[date, Dict[Any, bool]] = defaultdict(dict)
    dates: List[date] = []
    for r in rows:
        lab = r.get("lab_values") or {}
        if not isinstance(lab, dict) or lab.get("assay") != "UAS7":
            continue
        raw_value = lab.get("value")
        if raw_value is None:
            continue
        try:
            uas7 = float(raw_value)
        except (TypeError, ValueError):
            continue
        d = _to_date(r.get("event_date"))
        if d is None:
            continue
        dates.append(d)
        m = _month_start(d)
        pid = r.get("patient_id")
        uncontrolled = uas7 >= _UAS7_UNCONTROLLED_THRESHOLD
        by_month[m][pid] = by_month[m].get(pid, False) or uncontrolled
    points: List[Dict[str, Any]] = []
    for m in _complete_months(dates):
        patients = by_month.get(m)
        if not patients:
            continue
        points.append(_point(kpi_meta, "", m, sum(patients.values()) / len(patients)))
    return points


async def _backfill_br002_intent_delta(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """BR-002: mean intent-to-prescribe change across the month's quality
    Remibrutinib surveys.

    Mirrors the canonical semantics behind
    ``brand_specific_remi_intent_delta_primary`` — ``v_kpi_intent_to_prescribe``'s
    AVG(intent_to_prescribe_change) per survey month with
    ``response_quality_flag = true`` — of which the live registry read is the
    latest-month head, so the all-time recast IS the view's ``survey_month``
    axis. Rows with NULL ``intent_to_prescribe_change`` are excluded (baseline
    surveys carry no delta), matching both registry variants' ``IS NOT NULL``
    guard.
    """
    rows = await _fetch_all(
        client,
        "hcp_intent_surveys",
        "survey_id,survey_date,intent_to_prescribe_change",
        "survey_id",
        eq_filters={"brand": "Remibrutinib", "response_quality_flag": True},
        cache=cache,
        cache_key="br002_surveys",
    )
    by_month: Dict[date, List[float]] = defaultdict(list)
    dates: List[date] = []
    for r in rows:
        change = r.get("intent_to_prescribe_change")
        if change is None:
            continue
        d = _to_date(r.get("survey_date"))
        if d is None:
            continue
        dates.append(d)
        by_month[_month_start(d)].append(float(change))
    points: List[Dict[str, Any]] = []
    for m in _complete_months(dates):
        values = by_month.get(m)
        if not values:
            continue
        points.append(_point(kpi_meta, "", m, sum(values) / len(values)))
    return points


async def _backfill_br003_pnh_tested(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """BR-003: cumulative AS-OF recompute — at each month-end, share of the
    D59.5-eligible Fabhalta patients (journey started by then) with a PNH
    flow-cytometry test (real PNH LOINC) by then.

    Mirrors ``brand_specific_fabhalta_pnh_tested`` (migration 091 numerator)
    evaluated as it WOULD have read at each historical month-end. Both sides
    are real dated events, so the series is an evolving rate — not a coverage
    ramp. As-of readings are complete by construction, so every month-end at
    or before BOTH sources' frontiers is emitted (no partial-month drop).
    """
    journeys = await _fetch_journeys(client, cache)
    eligible_start: Dict[Any, date] = {}
    for r in journeys:
        if r.get("brand") != "Fabhalta" or r.get("primary_diagnosis_code") != "D59.5":
            continue
        d = _to_date(r.get("journey_start_date"))
        if d is None:
            continue
        pid = r.get("patient_id")
        if pid not in eligible_start or d < eligible_start[pid]:
            eligible_start[pid] = d

    pnh_rows = await _fetch_all(
        client,
        "treatment_events",
        "patient_id,event_date,loinc_codes",
        "treatment_event_id",
        eq_filters={"event_subtype": "pnh_flow_cytometry"},
        cache=cache,
        cache_key="pnh_events",
    )
    first_test: Dict[Any, date] = {}
    pnh_dates: List[date] = []
    for r in pnh_rows:
        codes = r.get("loinc_codes") or []
        if not _PNH_LOINCS.intersection(codes):
            continue
        d = _to_date(r.get("event_date"))
        if d is None:
            continue
        pnh_dates.append(d)
        pid = r.get("patient_id")
        if pid not in first_test or d < first_test[pid]:
            first_test[pid] = d

    if not eligible_start or not pnh_dates:
        # Structurally-empty side -> no honest series (mirror of the #1116
        # fail-loud: never render a substrate gap as a plausible 0%).
        return []

    frontier = min(max(eligible_start.values()), max(pnh_dates))
    points: List[Dict[str, Any]] = []
    m = _month_start(min(eligible_start.values()))
    while _month_end(m) <= frontier:
        asof = _month_end(m)
        eligible = [pid for pid, start in eligible_start.items() if start <= asof]
        if eligible:
            tested = sum(1 for pid in eligible if first_test.get(pid, date.max) <= asof)
            points.append(_point(kpi_meta, "", m, tested / len(eligible)))
        m = _next_month(m)
    return points


async def _backfill_br004_dx_adoption(
    client: Any, kpi_meta: Any, cache: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """BR-004: median days from journey start to FIRST Kisqali prescription,
    grouped by the month the first prescription lands in. Lower-is-better.

    Mirrors ``brand_specific_kisqali_dx_adoption``: MIN(event_date) per patient
    over Kisqali prescriptions, joined to ALL of the patient's journeys with
    journey_start_date <= first Rx (the SQL join multiplies per journey row —
    preserved verbatim), PERCENTILE_CONT(0.5) over the day deltas.
    """
    dated = _rx_dated(await _fetch_prescriptions(client, cache))
    first_rx: Dict[Any, date] = {}
    for d, r in dated:
        if r.get("brand") != "Kisqali":
            continue
        pid = r.get("patient_id")
        if pid not in first_rx or d < first_rx[pid]:
            first_rx[pid] = d

    starts_by_patient: Dict[Any, List[date]] = defaultdict(list)
    for r in await _fetch_journeys(client, cache):
        d = _to_date(r.get("journey_start_date"))
        if d is not None:
            starts_by_patient[r.get("patient_id")].append(d)

    deltas_by_month: Dict[date, List[float]] = defaultdict(list)
    for pid, rx_date in first_rx.items():
        for start in starts_by_patient.get(pid, []):
            if rx_date >= start:
                deltas_by_month[_month_start(rx_date)].append(float((rx_date - start).days))

    # Cohort completeness follows the prescription feed (the month's first-Rx
    # cohort is closed once the month is fully covered by prescription data).
    points: List[Dict[str, Any]] = []
    for m in _complete_months([d for d, _ in dated]):
        deltas = deltas_by_month.get(m)
        if not deltas:
            continue
        points.append(_point(kpi_meta, "", m, float(statistics.median(deltas))))
    return points


# KPI_ID -> handler. Only honestly-backfillable KPIs are registered (Batch 1:
# ROI. Batch 2: the as-of recompute family documented in the module docstring.
# Batch 3: BR-002 monthly intent-delta).
HANDLERS: Dict[str, Handler] = {
    "WS3-BI-010": _backfill_roi,
    "WS3-BI-005": _backfill_trx,
    "WS3-BI-006": _backfill_nrx,
    "WS3-BI-007": _backfill_nbrx,
    "WS3-BI-008": _backfill_trx_share,
    "WS3-BI-009": _backfill_conversion_rate,
    "WS3-BI-001": _backfill_mau,
    "WS3-BI-002": _backfill_wau,
    "WS2-TR-001": _backfill_tr001_precision,
    "WS2-TR-004": _backfill_tr004_acceptance,
    "WS2-TR-005": _backfill_tr005_false_alert,
    "WS2-TR-006": _backfill_tr006_override,
    "WS2-TR-007": _backfill_tr007_lead_time,
    "WS2-TR-008": _backfill_tr008_cfr,
    "BR-001": _backfill_br001_ah_uncontrolled,
    "BR-002": _backfill_br002_intent_delta,
    "BR-003": _backfill_br003_pnh_tested,
    "BR-004": _backfill_br004_dx_adoption,
}


async def run_backfill(kpi_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute, replace and upsert history points for the registered KPIs.

    Replace semantics: existing (kpi_id, source) rows are deleted AFTER the
    fresh points computed successfully and BEFORE the upsert, so a reseed's
    shifted timeline never leaves stale months behind — and a failing handler
    never wipes the previous good series.
    """
    from src.kpi.registry import KPIRegistry
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.kpi_history import get_kpi_history_repository

    client = await get_async_supabase_client()
    if client is None:
        raise RuntimeError("No Supabase client for KPI history backfill")
    repo = await get_kpi_history_repository()
    registry = KPIRegistry()

    targets = kpi_ids or list(HANDLERS.keys())
    cache: Dict[str, Any] = {}
    summary: Dict[str, Any] = {"written": {}, "deleted": {}, "skipped": [], "errors": {}}
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
            points = await handler(client, kpi_meta, cache)
            deleted = await repo.delete_source(kpi_id, HANDLER_SOURCES[kpi_id])
            written = await repo.upsert_points(points)
            summary["deleted"][kpi_id] = deleted
            summary["written"][kpi_id] = written
            logger.info(
                "KPI history backfill %s: %d points (replaced %d)", kpi_id, written, deleted
            )
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
