"""KPI-aware data resolution for the Tool Composer (issue #810).

Background
----------
The Tool Composer answers multi-faceted analytical queries. Its cohort loader
(:func:`src.services.cohort_resolution.resolve_cohort_frame`) always resolves the
patient-clinical ``patient_journeys`` grain. But many flagship queries are about a
**defined KPI** — e.g. *"what drove <brand> conversion in <region>?"* — whose
outcome does NOT live in ``patient_journeys``.

"Conversion" is the defined ``Conversion Rate`` KPI (``WS3-BI-009``):
*percentage of triggers resulting in a prescription within 30 days*
(``triggers ⋈ treatment_events``; see ``config/kpi_definitions.yaml`` and the
allowlist SQL ``business_impact_conversion_rate`` in migration 044). Resolving the
patient grain can never bind that outcome, so the causal core fails.

This service makes the pipeline KPI-aware:

1. :func:`recognize_kpi` — map a query to a defined KPI via the KPI registry
   (``src/kpi/registry.py``, 46 KPIs) + a small KPI-vocabulary alias map.
2. :func:`resolve_kpi_frame` — materialize the **analyzable** frame for that KPI
   from its REAL substrate, returning the outcome column + candidate driver
   columns so the planner can bind the causal outcome to the KPI.

Dynamic, not hardcoded
----------------------
Brand and region are **parameters**, matched case-insensitively against the
ACTUAL distinct values present in the data (``treatment_events.brand`` /
``hcp_profiles.geographic_region``) — there is no hardcoded brand or region list,
and nothing is special-cased to a particular brand/region. An input value not
present in the data fails closed (``None``), never a wrong-population or
fabricated frame.

Anti-mocking discipline
-----------------------
Never fabricates a frame. Returns ``None`` (fail closed) on unrecognized
brand/region, missing substrate, or empty results — callers then proceed without
``estimation_data`` and the composable tools fail closed in turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.kpi.models import KPIMetadata
from src.kpi.registry import get_registry

logger = logging.getLogger(__name__)

# Conversion Rate KPI id (config/kpi_definitions.yaml -> WS3-BI-009).
CONVERSION_KPI_ID = "WS3-BI-009"

# Conversion window: a delivered trigger is "converted" if a prescription occurs
# within this many days after it (authoritative SQL business_impact_conversion_rate).
_CONVERSION_WINDOW_DAYS = 30

# Candidate causal driver columns on the trigger grain (those that exist are
# exposed; the planner picks treatment/segments from them). These are trigger
# FEATURES, not the outcome.
_DRIVER_COLUMNS = [
    "trigger_type",
    "delivery_channel",
    "priority",
    "confidence_score",
    "lead_time_days",
    "acceptance_status",
]

# Columns fetched from the triggers table for the conversion substrate.
_TRIGGER_SELECT = [
    "trigger_id",
    "patient_id",
    "hcp_id",
    "trigger_timestamp",
    *_DRIVER_COLUMNS,
]

# Generous per-request row cap; the substrate tables are small (~4k triggers,
# ~10k events). A WARNING fires if a fetch hits this (possible truncation).
_MAX_ROWS = 100_000

# KPI vocabulary aliases: common user terms -> KPI id. This maps the FIXED,
# defined KPI vocabulary (46 KPIs) to ids; it is NOT brand/region hardcoding.
_ALIASES: Dict[str, str] = {
    "conversion": CONVERSION_KPI_ID,
    "conversion rate": CONVERSION_KPI_ID,
    "nbrx": "WS3-BI-007",
    "new-to-brand": "WS3-BI-007",
    "new to brand": "WS3-BI-007",
    "nrx": "WS3-BI-006",
    "new prescription": "WS3-BI-006",
    "trx share": "WS3-BI-008",
    "market share": "WS3-BI-008",
    "trx": "WS3-BI-005",
    "total prescription": "WS3-BI-005",
    "return on investment": "WS3-BI-010",
    "roi": "WS3-BI-010",
    "hcp coverage": "WS3-BI-004",
    "patient touch": "WS3-BI-003",
}


@dataclass
class KpiFrame:
    """A materialized, analyzable frame for a KPI causal question.

    Attributes:
        frame: the real per-unit DataFrame (e.g. per-trigger) carrying the
            outcome column + driver columns.
        outcome_column: the causal outcome column name (e.g. ``"converted"``).
        driver_columns: candidate causal driver/segment columns present in the
            frame.
        kpi_id: the resolved KPI id (e.g. ``"WS3-BI-009"``).
        kpi_name: the human-readable KPI name (e.g. ``"Conversion Rate"``).
        is_truncated: ``True`` when a source fetch hit the ``_MAX_ROWS`` cap, so the
            substrate may be a truncated sample (no silent caps — surfaced to the
            caller for logging / response provenance).
    """

    frame: pd.DataFrame
    outcome_column: str
    driver_columns: List[str]
    kpi_id: str
    kpi_name: str
    is_truncated: bool = False


# ---------------------------------------------------------------------------
# KPI recognition (registry-driven, dynamic across all 46 KPIs)
# ---------------------------------------------------------------------------
def recognize_kpi(query: Optional[str]) -> Optional[KPIMetadata]:
    """Recognize a defined KPI referenced by ``query``, else ``None``.

    Matches the query against KPI-vocabulary aliases first (longest alias wins
    for specificity), then falls back to a conservative match on the registry's
    KPI names. Brand/region in the query are ignored here (they are resolved
    separately and dynamically).
    """
    if not query:
        return None
    q = " ".join(str(query).lower().split())
    registry = get_registry()

    # 1) alias match — longest alias first so "conversion rate" beats "rate".
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if alias in q:
            kpi = registry.get(_ALIASES[alias])
            if kpi is not None:
                return kpi

    # 2) dynamic fallback: a distinctive KPI-name token appears in the query.
    stop = {"rate", "score", "total", "new", "of", "the", "and", "to", "per", "median"}
    for kpi in registry.get_all():
        for tok in (
            str(kpi.name).lower().replace("-", " ").replace("(", " ").replace(")", " ").split()
        ):
            tok = tok.strip()
            if len(tok) >= 4 and tok not in stop and tok in q:
                return kpi
    return None


# ---------------------------------------------------------------------------
# Pure outcome construction (real logic; unit-tested without a DB)
# ---------------------------------------------------------------------------
def _compute_conversion_outcome(
    triggers: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int = _CONVERSION_WINDOW_DAYS,
) -> pd.Series:
    """Per-trigger binary ``converted``: a prescription for the trigger's patient
    within ``[trigger_date, trigger_date + window_days]`` (inclusive, date-level).

    Mirrors the authoritative ``business_impact_conversion_rate`` SQL. Pure: takes
    real-shaped frames, computes the real outcome — no DB, no fabrication.
    """
    if triggers is None or len(triggers) == 0:
        return pd.Series([], dtype=int)

    trig_ts = pd.to_datetime(triggers["trigger_timestamp"], errors="coerce", utc=True)

    by_patient: Dict[Any, List[Any]] = {}
    if events is not None and len(events) and "patient_id" in events.columns:
        ev_dt = pd.to_datetime(events["event_date"], errors="coerce", utc=True)
        for pid, d in zip(events["patient_id"], ev_dt, strict=False):
            if pd.notna(d):
                by_patient.setdefault(pid, []).append(d.date())

    out: List[int] = []
    for pid, ts in zip(triggers["patient_id"], trig_ts, strict=False):
        if pd.isna(ts):
            out.append(0)
            continue
        lo = ts.date()
        hi = (ts + pd.Timedelta(days=window_days)).date()
        dates = by_patient.get(pid, [])
        out.append(int(any(lo <= d <= hi for d in dates)))
    return pd.Series(out, index=triggers.index, dtype=int)


def _assemble_conversion_frame(
    triggers: pd.DataFrame,
    hcp_regions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    region_canonical: Optional[str],
    window_days: int = _CONVERSION_WINDOW_DAYS,
) -> Optional[KpiFrame]:
    """Build the conversion ``KpiFrame`` from already-fetched frames (pure).

    ``events`` must already be brand-filtered prescriptions (the brand filter is
    applied on the prescription side, since ``triggers`` carries no usable brand).
    Region is applied here via the ``triggers.hcp_id ⋈ hcp_regions`` join. Fails
    closed (``None``) on empty triggers or an unrecognized/empty region.
    """
    if triggers is None or len(triggers) == 0:
        return None

    df = triggers.copy()

    if region_canonical:
        if hcp_regions is None or len(hcp_regions) == 0:
            return None
        reg = hcp_regions.copy()
        reg_norm = reg["geographic_region"].astype(str).str.strip().str.lower()
        in_region = set(reg.loc[reg_norm == region_canonical.strip().lower(), "hcp_id"])
        if not in_region:
            return None
        df = df[df["hcp_id"].isin(in_region)].copy()
        if len(df) == 0:
            return None

    df["converted"] = _compute_conversion_outcome(df, events, window_days).to_numpy()

    drivers = [c for c in _DRIVER_COLUMNS if c in df.columns]
    # Derived clean binary treatment from acceptance_status (real, not fabricated).
    if "acceptance_status" in df.columns:
        df["accepted"] = (
            df["acceptance_status"].astype(str).str.strip().str.lower() == "accepted"
        ).astype(int)
        drivers.append("accepted")

    return KpiFrame(
        frame=df.reset_index(drop=True),
        outcome_column="converted",
        driver_columns=drivers,
        kpi_id=CONVERSION_KPI_ID,
        kpi_name="Conversion Rate",
    )


# ---------------------------------------------------------------------------
# Dynamic brand/region resolution against the REAL data values
# ---------------------------------------------------------------------------
def _match_against_distinct(value: Optional[str], distinct: set[str]) -> Optional[str]:
    """Case-insensitively match ``value`` to a member of ``distinct`` (the actual
    data values). Returns the canonical (data) spelling, or ``None`` if absent."""
    if not value or not str(value).strip():
        return None
    norm = str(value).strip().lower()
    for d in distinct:
        if str(d).strip().lower() == norm:
            return str(d)
    return None


def _default_client() -> Any:
    from src.repositories import get_supabase_client

    return get_supabase_client()


def _resolve_brand_canonical(
    client: Any, brand: str, *, include_synthetic: bool = False
) -> tuple[Optional[str], bool]:
    """Resolve ``brand`` to its canonical data spelling, case-insensitively, against
    the real ``treatment_events.brand`` values.

    ``treatment_events.brand`` is a PostgreSQL ENUM (``brand_type``) so ``ILIKE`` is
    unavailable (``operator does not exist: brand_type ~~* unknown``); we scan the
    distinct values and match in Python. To avoid a SILENT truncation, the scan's
    cap is detected and returned: a no-match while the scan was capped is reported
    as ``(None, True)`` so the caller can distinguish "brand truly absent" from
    "brand may exist beyond the row cap" — it is never a silent fail-closed.

    Returns ``(canonical_or_None, scan_truncated)``.
    """
    value = str(brand).strip()
    if not value:
        return None, False
    _bq = (
        client.table("treatment_events")
        .select("brand")
        .eq("event_type", "prescription")
        .not_.is_("brand", "null")
    )
    # Shard 07 R10: the brand distinct-scan default-excludes synthetic so a real-mode
    # resolution never canonicalizes against a synthetic-only brand value.
    from src.repositories.provenance import apply_provenance_filter

    _bq = apply_provenance_filter(_bq, include_synthetic)
    rows = getattr(_bq.limit(_MAX_ROWS).execute(), "data", None) or []
    scan_truncated = len(rows) >= _MAX_ROWS
    if scan_truncated:
        logger.warning(
            "kpi_resolution: brand distinct-scan hit the %d-row cap; a brand beyond "
            "the cap could be missed.",
            _MAX_ROWS,
        )
    distinct_brands = {str(r["brand"]) for r in rows if r.get("brand")}
    return _match_against_distinct(value, distinct_brands), scan_truncated


# Tables this module reads that carry the is_synthetic provenance column (Shard 01).
# A read on one of these default-excludes synthetic rows unless the caller opts in.
_PROVENANCE_TAGGABLE = frozenset(
    {"triggers", "treatment_events", "hcp_profiles", "patient_journeys",
     "business_metrics", "ml_predictions", "episodic_memories"}
)


def _fetch_df(
    client: Any,
    table: str,
    columns: str,
    *,
    brand: Optional[str] = None,
    include_synthetic: bool = False,
) -> pd.DataFrame:
    q = client.table(table).select(columns)
    if table == "treatment_events":
        q = q.eq("event_type", "prescription")
        if brand:
            q = q.eq("brand", brand)
    # Shard 07 R10: default-exclude is_synthetic on taggable tables (gated so a table
    # without the column never 42703s).
    if table in _PROVENANCE_TAGGABLE:
        from src.repositories.provenance import apply_provenance_filter

        q = apply_provenance_filter(q, include_synthetic)
    rows = getattr(q.limit(_MAX_ROWS).execute(), "data", None) or []
    if len(rows) >= _MAX_ROWS:
        logger.warning(
            "kpi_resolution: %s fetch hit the %d-row cap; results may be truncated.",
            table,
            _MAX_ROWS,
        )
    return pd.DataFrame(rows)


def _build_conversion_frame(
    brand: Optional[str],
    region: Optional[str],
    *,
    supabase_client: Optional[Any] = None,
    window_days: int = _CONVERSION_WINDOW_DAYS,
    include_synthetic: bool = False,
) -> Optional[KpiFrame]:
    """Materialize the conversion substrate (triggers ⋈ treatment_events) for a
    dynamic ``(brand, region)`` from the REAL tables. Fails closed on
    unrecognized brand/region or empty data; never fabricates.

    Shard 07 R10: every source read default-excludes is_synthetic; a validation run
    opts in with ``include_synthetic=True`` so it can measure the synthetic substrate.
    """
    client = supabase_client if supabase_client is not None else _default_client()

    triggers = _fetch_df(
        client, "triggers", ",".join(_TRIGGER_SELECT), include_synthetic=include_synthetic
    )
    if triggers is None or len(triggers) == 0:
        logger.info("kpi_resolution: no triggers available -> fail closed")
        return None

    hcp_regions = _fetch_df(
        client, "hcp_profiles", "hcp_id,geographic_region",
        include_synthetic=include_synthetic,
    )

    # Region resolved DYNAMICALLY against the real geographic_region values.
    region_canonical: Optional[str] = None
    if region and str(region).strip():
        distinct_regions = {
            str(r) for r in (hcp_regions.get("geographic_region", pd.Series(dtype=str)).dropna())
        }
        region_canonical = _match_against_distinct(region, distinct_regions)
        if region_canonical is None:
            logger.info("kpi_resolution: unrecognized region %r -> fail closed", region)
            return None

    # Brand resolved DYNAMICALLY against the real treatment_events.brand values.
    # brand is a PG enum -> distinct scan; the scan's cap is tracked so a no-match
    # under a truncated scan is never a silent fail-closed.
    brand_canonical: Optional[str] = None
    brand_scan_truncated = False
    if brand and str(brand).strip():
        brand_canonical, brand_scan_truncated = _resolve_brand_canonical(
            client, brand, include_synthetic=include_synthetic
        )
        if brand_canonical is None:
            logger.info(
                "kpi_resolution: unrecognized brand %r (brand_scan_truncated=%s) -> fail closed",
                brand,
                brand_scan_truncated,
            )
            return None

    events = _fetch_df(
        client, "treatment_events", "patient_id,event_date,event_type,brand",
        brand=brand_canonical, include_synthetic=include_synthetic,
    )

    # No silent caps: if any source fetch (incl. the brand distinct-scan) hit the
    # row cap, the substrate may be a truncated sample — flag it so the caller /
    # response can surface it.
    truncated = (
        len(triggers) >= _MAX_ROWS
        or len(events) >= _MAX_ROWS
        or len(hcp_regions) >= _MAX_ROWS
        or brand_scan_truncated
    )

    kf = _assemble_conversion_frame(
        triggers, hcp_regions, events, region_canonical=region_canonical, window_days=window_days
    )
    if kf is None:
        logger.info(
            "kpi_resolution: conversion frame empty for brand=%r region=%r -> fail closed",
            brand,
            region,
        )
        return None
    kf.is_truncated = truncated
    return kf


# ---------------------------------------------------------------------------
# Dispatch — per-KPI substrate builders (extension point)
# ---------------------------------------------------------------------------
# Map KPI id -> substrate builder. Conversion (WS3-BI-009) is implemented; other
# KPIs return None (honest "no builder yet") until their substrate is added.
_BUILDERS: Dict[str, Callable[..., Optional[KpiFrame]]] = {
    CONVERSION_KPI_ID: _build_conversion_frame,
}


def resolve_kpi_frame(
    kpi: Optional[KPIMetadata],
    brand: Optional[str],
    region: Optional[str],
    *,
    supabase_client: Optional[Any] = None,
    window_days: int = _CONVERSION_WINDOW_DAYS,
    include_synthetic: bool = False,
) -> Optional[KpiFrame]:
    """Resolve the analyzable :class:`KpiFrame` for ``kpi`` at ``(brand, region)``.

    Returns ``None`` (fail closed, never fabricated) when ``kpi`` is ``None``, has
    no substrate builder yet, or no real data resolves.

    Raises:
        Genuine infrastructure errors (e.g. a Supabase connection / auth failure
        from the client) propagate — the caller logs and proceeds WITHOUT data
        (both wired callers, ``chatbot_tools`` and the orchestrator dispatcher,
        wrap this in a best-effort guard). This mirrors the
        :func:`src.services.cohort_resolution.resolve_cohort_frame` contract: the
        service never silently swallows infra failures into a fabricated/empty
        frame; the composable tools then fail closed honestly.
    """
    if kpi is None:
        return None
    builder = _BUILDERS.get(kpi.id)
    if builder is None:
        logger.info(
            "kpi_resolution: no substrate builder for KPI %s (%s) yet -> None",
            kpi.id,
            kpi.name,
        )
        return None
    return builder(
        brand, region, supabase_client=supabase_client,
        window_days=window_days, include_synthetic=include_synthetic,
    )
