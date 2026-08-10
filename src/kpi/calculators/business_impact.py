"""
WS3 Business Impact KPI Calculators

Implements calculators for business impact metrics:
- Monthly Active Users (MAU)
- Weekly Active Users (WAU)
- Patient Touch Rate
- HCP Coverage
- TRx, NRx, NBRx
- TRx Share
- Conversion Rate
- ROI
"""

import logging
from typing import Any

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)
from src.kpi.synthetic_mode import (
    biologic_query_id,
    brand_scoped_query_id,
    ige_tier_query_id,
    line_query_id,
    region_query_id,
    resolve_kpi_query_id,
    segment_query_id,
    windowed_axis_query_id,
    windowed_query_id,
)

logger = logging.getLogger(__name__)

# Brands for which the biologic-status / IgE-tertile axes are REAL. The
# ``biologic_experienced`` and ``ige_level`` columns are populated ONLY for these
# brands in the DGP (``clinical_codes.BRAND_ELIGIBILITY_FIELDS`` -- Remibrutinib /
# CSU); every other brand is 100% NULL by design. A breakdown on those axes for a
# non-eligible brand would be fabricated, so the calculator fails closed rather
# than return a silent 0 (which is indistinguishable from a genuine zero). Kept
# as a local constant (not imported from the synthetic DGP module) to keep the
# serving path decoupled -- a consistency test locks it to the SSOT, mirroring
# ``causal._BRAND_CLINICAL_COVARIATES``.
_BIOLOGIC_AXIS_BRANDS: frozenset[str] = frozenset({"Remibrutinib"})


class BusinessImpactCalculator(KPICalculatorBase):
    """Calculator for WS3 Business Impact KPIs."""

    def __init__(self, db_client: Any = None):
        """Initialize with database client.

        Args:
            db_client: Database client for executing queries.
        """
        self._db_client = db_client

    @property
    def db_client(self) -> Any:
        """Get database client, lazily initializing if needed."""
        if self._db_client is None:
            from src.repositories import get_supabase_client

            self._db_client = get_supabase_client()
        return self._db_client

    def supports(self, kpi: KPIMetadata) -> bool:
        """Check if this calculator supports the given KPI."""
        return kpi.workstream == Workstream.WS3_BUSINESS

    def calculate(self, kpi: KPIMetadata, context: dict[str, Any] | None = None) -> KPIResult:
        """Calculate a business impact KPI.

        Args:
            kpi: The KPI metadata defining what to calculate.
            context: Optional context with brand, date_range, etc.

        Returns:
            KPIResult with calculated value and status.
        """
        context = context or {}

        calculator_map = {
            "WS3-BI-001": self._calc_mau,
            "WS3-BI-002": self._calc_wau,
            "WS3-BI-003": self._calc_patient_touch_rate,
            "WS3-BI-004": self._calc_hcp_coverage,
            "WS3-BI-005": self._calc_trx,
            "WS3-BI-006": self._calc_nrx,
            "WS3-BI-007": self._calc_nbrx,
            "WS3-BI-008": self._calc_trx_share,
            "WS3-BI-009": self._calc_conversion_rate,
            "WS3-BI-010": self._calc_roi,
        }

        calc_func = calculator_map.get(kpi.id)
        if calc_func is None:
            return KPIResult(
                kpi_id=kpi.id,
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            value = calc_func(context)
            # Volume metrics (TRx, NRx, NBRx) carry threshold: null in the YAML,
            # so _evaluate_status routes them to INFORMATIONAL (no special case).
            status = self._evaluate_status(kpi, value)
            return KPIResult(
                kpi_id=kpi.id,
                value=value,
                status=status,
                cached=False,
                error=None,
                metadata={"context": context},
            )
        except Exception as e:
            return KPIResult(
                kpi_id=kpi.id,
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
                error=str(e),
            )

    def _evaluate_status(
        self, kpi: KPIMetadata, value: float | None, lower_is_better: bool = False
    ) -> KPIStatus:
        """Evaluate KPI value against thresholds."""
        if value is None:
            return KPIStatus.UNKNOWN
        if kpi.threshold is None:
            # No threshold by design -> tracked for trend/context only.
            return KPIStatus.INFORMATIONAL
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

    # Region-scoped query id helper shared across calculators (migrations
    # 077/078). Kept as a thin alias so the call sites below stay readable.
    _region_variant = staticmethod(region_query_id)

    @staticmethod
    def _stash_data_through(context: dict[str, Any], result: list[dict[str, Any]] | None) -> None:
        """Surface the row's ``data_through`` provenance into the per-call context.

        The frontier-anchored registry rows (migration 089) report the as-of
        date their window ends at (``MAX(<domain ts>)`` -- the substrate is
        calendar-fixed, so windows anchor to the data frontier, not NOW()).
        ``calculate()`` embeds the context in ``KPIResult.metadata``, which lets
        the chatbot cite the real period instead of implying wall-clock
        recency. Rows without the column (explicit ``*_windowed*`` variants,
        pre-089 deployments) leave the key absent -- honest absence, never a
        fabricated date.
        """
        if result and isinstance(result[0], dict) and result[0].get("data_through") is not None:
            context["data_through"] = result[0]["data_through"]

    @staticmethod
    def _guard_brand_scoped_axis(axis_label: str, brand: str | None) -> None:
        """Fail closed when a biologic-status / IgE breakdown is requested for a
        brand whose DGP does not populate those columns.

        ``biologic_experienced`` and ``ige_level`` are REAL only for
        :data:`_BIOLOGIC_AXIS_BRANDS` (Remibrutinib / CSU); every other brand is
        100% NULL by design. Rather than return a silent 0 (indistinguishable
        from a genuine zero), raise so BOTH ``/api/kpis`` and the chatbot surface
        an explicit "not available for <brand>" -- parity with the #1216 refusal
        to fabricate a biologic/IgE sub-population. The membership check is
        case-insensitive; the underlying SQL ``brand::text = $1`` is
        case-sensitive, so a canonical-cased brand ("Remibrutinib") is expected.
        """
        eligible = ", ".join(sorted(_BIOLOGIC_AXIS_BRANDS))
        norm = (brand or "").strip()
        if not norm:
            raise RuntimeError(
                f"{axis_label} breakdown requires a brand and is available only for "
                f"{eligible}: the biologic-status / IgE columns are unpopulated for "
                f"every other brand by design."
            )
        if norm.title() not in _BIOLOGIC_AXIS_BRANDS:
            raise RuntimeError(
                f"{axis_label} breakdown is not available for {norm}: biologic-status / "
                f"IgE data exists only for {eligible} (other brands are 100% NULL by "
                f"design -- reporting a split would fabricate it)."
            )

    def _resolve_windowed_call(
        self,
        base_query_id: str,
        *,
        brand: str | None,
        region: str | None,
        window: dict[str, Any] | None,
        segment: str | None = None,
        therapy_line: int | str | None = None,
        biologic: str | None = None,
        ige_tier: str | None = None,
    ) -> tuple[str, list[Any]]:
        """Compose (query_id, positional params) for a windowable KPI.

        Param order respects the kpi_query 4-param cap:
          no axis, no region:  [brand, start, end]
          region:               [brand, region, start, end]
          segment/therapy_line/biologic/ige_tier: [brand, axis_value(, start, end)]
        With no window, falls back to the existing base / _region behavior.

        PRECEDENCE (migrations 105/108): a patient axis -- severity tier
        (``segment``), line-of-therapy (``therapy_line``), biologic status
        (``biologic``), or IgE tertile (``ige_tier``) -- takes precedence over
        ``region`` and NONE are ever combined: the kpi_query RPC caps positional
        params at 4, so brand+region+axis+window can't all fit in one statement.
        Within the axis family the order is segment > therapy_line > biologic >
        ige_tier if a caller somehow supplies more than one. Region-scoped-AND-
        axis-scoped reads would need a follow-up migration (a 5th param slot).
        ``therapy_line`` is checked via ``is not None`` (not truthiness) because
        line 0 is a real, commonly-populated bucket -- ``if therapy_line:`` would
        silently drop it.

        The biologic/ige_tier axes are brand-gated (columns exist only for
        :data:`_BIOLOGIC_AXIS_BRANDS`); :meth:`_guard_brand_scoped_axis` fails
        closed for any other brand BEFORE a query is built.
        """
        if segment is not None:
            if window is None:
                return segment_query_id(base_query_id), [brand, segment]
            return (
                windowed_axis_query_id(base_query_id, axis="segment"),
                [brand, segment, window["start"], window["end"]],
            )
        if therapy_line is not None:
            if window is None:
                return line_query_id(base_query_id), [brand, therapy_line]
            return (
                windowed_axis_query_id(base_query_id, axis="line"),
                [brand, therapy_line, window["start"], window["end"]],
            )
        if biologic is not None:
            self._guard_brand_scoped_axis("biologic-status", brand)
            if window is None:
                return biologic_query_id(base_query_id), [brand, biologic]
            return (
                windowed_axis_query_id(base_query_id, axis="biologic"),
                [brand, biologic, window["start"], window["end"]],
            )
        if ige_tier is not None:
            self._guard_brand_scoped_axis("IgE-tier", brand)
            if window is None:
                return ige_tier_query_id(base_query_id), [brand, ige_tier]
            return (
                windowed_axis_query_id(base_query_id, axis="ige_tier"),
                [brand, ige_tier, window["start"], window["end"]],
            )
        if window is None:
            if region:
                return region_query_id(base_query_id), [brand, region]
            return base_query_id, [brand]
        qid = windowed_query_id(base_query_id, region=bool(region))
        if region:
            return qid, [brand, region, window["start"], window["end"]]
        return qid, [brand, window["start"], window["end"]]

    def _calc_mau(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-001: Monthly Active Users.

        Unique users with at least one session in past 30 days.
        Uses v_kpi_active_users view if available.
        """
        # Try view first
        result = self._execute_query("business_impact_mau_view", [])
        if result and result[0].get("mau") is not None:
            return float(result[0]["mau"])

        # Fall back to direct calculation
        result = self._execute_query("business_impact_mau_fallback", [])
        if result and result[0].get("mau") is not None:
            return float(result[0]["mau"])
        raise RuntimeError("KPI WS3-BI-001 unavailable: no data for monthly active users")

    def _calc_wau(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-002: Weekly Active Users.

        Unique users with at least one session in past 7 days.
        """
        # Try view first
        result = self._execute_query("business_impact_wau_view", [])
        if result and result[0].get("wau") is not None:
            return float(result[0]["wau"])

        # Fall back to direct calculation
        result = self._execute_query("business_impact_wau_fallback", [])
        if result and result[0].get("wau") is not None:
            return float(result[0]["wau"])
        raise RuntimeError("KPI WS3-BI-002 unavailable: no data for weekly active users")

    def _calc_patient_touch_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-003: Patient Touch Rate.

        Fraction of code-anchored ELIGIBLE patients with a trigger-driven
        touchpoint. Eligibility is DERIVED from the real primary_diagnosis_code
        (membership in the brand's qualifying ICD-10 set, via the
        v_patient_eligibility view), NOT a blanket is_eligible flag — which does
        not exist; that absence was the #574 fail-loud reason — and NOT the
        ~93%-NULL journey_status. A "touch" is a trigger that was actually
        DELIVERED (delivery_status IN ('delivered','viewed')); pending/failed/
        expired triggers never reached anyone, so counting any trigger would be
        the degenerate ~99.5% relabel #574 forbids.

        #577: returns the FRACTION touched/eligible in [0,1] (the division is
        done in SQL — sibling parity with conversion_rate / hcp_coverage). $1 is
        an optional brand filter ('' => all brands). Fails loud when there is no
        eligible cohort (NULLIF -> NULL touch_rate); a genuine 0.0 (cohort exists
        but none delivered-touched) is a legitimate value and is returned, not
        raised.
        """
        brand = context.get("brand") or ""
        result = self._execute_query("business_impact_patient_touch_rate", [brand])
        if not result or result[0].get("touch_rate") is None:
            raise RuntimeError(
                "KPI WS3-BI-003 unavailable: no code-anchored eligible patient cohort "
                "(apply the #577 patient-touch view + registry, migration 050)"
            )
        return float(result[0]["touch_rate"])

    def _calc_hcp_coverage(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-004: HCP Coverage.

        Percentage of priority HCPs with active engagement.
        """
        result = self._execute_query("business_impact_hcp_coverage", [])
        if result and result[0].get("coverage") is not None:
            return float(result[0]["coverage"])
        raise RuntimeError("KPI WS3-BI-004 unavailable: no data for HCP coverage")

    def _calc_trx(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-005: Total Prescriptions (TRx).

        Total prescription volume. No threshold (volume metric). When a region
        is supplied, routes to the region-scoped variant (migration 077); brand
        stays an optional filter. When a window is supplied, routes to the
        `_windowed[_region]` variant with [brand(, region), start, end]. When a
        severity tier or line-of-therapy is supplied, routes to the
        `_segment[_windowed]` / `_line[_windowed]` variant instead (migration
        105; takes precedence over region -- see `_resolve_windowed_call`).
        """
        query_id, params = self._resolve_windowed_call(
            "business_impact_trx",
            brand=context.get("brand"),
            region=context.get("region"),
            window=context.get("window"),
            segment=context.get("segment"),
            therapy_line=context.get("therapy_line"),
            biologic=context.get("biologic"),
            ige_tier=context.get("ige_tier"),
        )
        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("trx") is not None:
            return float(result[0]["trx"])
        raise RuntimeError("KPI WS3-BI-005 unavailable: no data for total prescriptions (TRx)")

    def _calc_nrx(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-006: New Prescriptions (NRx).

        First-time prescriptions for a patient. No threshold (volume metric).
        When a region is supplied, routes to the region-scoped variant
        (migration 077); brand stays an optional filter. When a window is
        supplied, routes to the `_windowed[_region]` variant with
        [brand(, region), start, end]. When a severity tier or
        line-of-therapy is supplied, routes to the `_segment[_windowed]` /
        `_line[_windowed]` variant instead (migration 105; takes precedence
        over region -- see `_resolve_windowed_call`).
        """
        query_id, params = self._resolve_windowed_call(
            "business_impact_nrx",
            brand=context.get("brand"),
            region=context.get("region"),
            window=context.get("window"),
            segment=context.get("segment"),
            therapy_line=context.get("therapy_line"),
            biologic=context.get("biologic"),
            ige_tier=context.get("ige_tier"),
        )
        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("nrx") is not None:
            return float(result[0]["nrx"])
        raise RuntimeError("KPI WS3-BI-006 unavailable: no data for new prescriptions (NRx)")

    def _calc_nbrx(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-007: New-to-Brand Prescriptions (NBRx).

        First prescription of specific brand for a patient.
        No threshold (volume metric). Region/window/segment/therapy_line
        routing mirrors `_calc_trx` / `_calc_nrx` (migrations 077/084/105).
        """
        brand = context.get("brand")
        if not brand:
            # NBRx is new-to-brand by definition: with no brand the metric is undefined,
            # not zero. Fail loud rather than fabricate a plausible 0 prescriptions.
            raise RuntimeError(
                "KPI WS3-BI-007 unavailable: no brand specified for new-to-brand prescriptions (NBRx)"
            )

        query_id, params = self._resolve_windowed_call(
            "business_impact_nbrx",
            brand=brand,
            region=context.get("region"),
            window=context.get("window"),
            segment=context.get("segment"),
            therapy_line=context.get("therapy_line"),
            biologic=context.get("biologic"),
            ige_tier=context.get("ige_tier"),
        )
        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("nbrx") is not None:
            return float(result[0]["nbrx"])
        raise RuntimeError(
            "KPI WS3-BI-007 unavailable: no data for new-to-brand prescriptions (NBRx)"
        )

    def _calc_trx_share(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-008: TRx Share.

        A brand's share of the TRACKED PORTFOLIO's prescriptions — the
        denominator is every prescription in ``treatment_events``, and only the
        portfolio brands (Fabhalta / Kisqali / Remibrutinib) exist there. This
        is NOT market share against external competitors; competitor brands
        (e.g. Xolair, Dupixent) are not in the data model at all.

        Region/segment/therapy_line routing goes through
        `_resolve_windowed_call`. Windowed variants exist for the plain and
        segment/line-scoped reads (migration 111); region/biologic/ige_tier
        have no windowed sibling, so a window combined with those axes fails
        loud rather than silently dropping either filter.
        """
        brand = context.get("brand")
        if not brand:
            # TRx Share is a brand's share of category: with no brand the metric is
            # undefined, not zero. Fail loud rather than fabricate a plausible 0% share.
            raise RuntimeError("KPI WS3-BI-008 unavailable: no brand specified for TRx share")

        window = context.get("window")
        if window is not None and (
            context.get("region")
            or context.get("biologic") is not None
            or context.get("ige_tier") is not None
        ):
            raise RuntimeError(
                "KPI WS3-BI-008: a time window on TRx share can be combined only "
                "with the severity-tier (segment) or line-of-therapy axis; "
                "windowed region/biologic/IgE-tier share variants are not "
                "registered (migration 111 covers plain/segment/line only)."
            )

        query_id, params = self._resolve_windowed_call(
            "business_impact_trx_share",
            brand=brand,
            region=context.get("region"),
            window=window,
            segment=context.get("segment"),
            therapy_line=context.get("therapy_line"),
            biologic=context.get("biologic"),
            ige_tier=context.get("ige_tier"),
        )
        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("share") is not None:
            return float(result[0]["share"])
        raise RuntimeError("KPI WS3-BI-008 unavailable: no data for TRx share")

    def _calc_conversion_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-009: Conversion Rate.

        Percentage of triggers resulting in a prescription within 30 days.
        Brand/segment/therapy_line/window routing (migration 111) — before it,
        this method honored ONLY ``region`` and silently dropped every other
        filter, so a "Remibrutinib high-severity" ask got the overall portfolio
        figure echoed back under the brand's name (session_1784387374342).

        BRAND SEMANTICS: a brand-scoped read counts triggers with that
        ``triggers.brand_id`` converting to a SAME-brand prescription; with no
        brand it reduces to the certified base semantics (all triggers, any
        prescription — verified equal to the base statement in the migration
        111 dry-run). The 30-day trigger→Rx conversion horizon is the KPI's
        definition and never changes; a window bounds WHICH triggers count.

        Axis precedence mirrors `_resolve_windowed_call` (segment >
        therapy_line > region). Unsupported combinations fail loud instead of
        silently dropping a filter: biologic/IgE-tier (triggers carry no such
        dimension) and region+window / region+brand (no such registry
        variants; the legacy `_region` read is region-only).
        """
        brand = context.get("brand")
        region = context.get("region")
        segment = context.get("segment")
        therapy_line = context.get("therapy_line")
        window = context.get("window")

        if context.get("biologic") is not None or context.get("ige_tier") is not None:
            raise RuntimeError(
                "KPI WS3-BI-009 does not support the biologic-status / IgE-tier "
                "axes: conversion is computed over triggers, which carry no "
                "biologic/IgE dimension. Supported: brand, severity tier "
                "(segment), line of therapy, time window, or region alone."
            )

        base = "business_impact_conversion_rate"
        if segment is not None:
            if window is None:
                query_id, params = segment_query_id(base), [brand, segment]
            else:
                query_id, params = (
                    windowed_axis_query_id(base, axis="segment"),
                    [brand, segment, window["start"], window["end"]],
                )
        elif therapy_line is not None:
            if window is None:
                query_id, params = line_query_id(base), [brand, therapy_line]
            else:
                query_id, params = (
                    windowed_axis_query_id(base, axis="line"),
                    [brand, therapy_line, window["start"], window["end"]],
                )
        elif window is not None:
            if region:
                raise RuntimeError(
                    "KPI WS3-BI-009: a time window on conversion rate cannot be "
                    "combined with a region filter (no windowed-region variant "
                    "is registered; migration 111 covers brand/segment/line)."
                )
            query_id, params = (
                windowed_query_id(base, region=False),
                [brand, window["start"], window["end"]],
            )
        elif region:
            if brand:
                raise RuntimeError(
                    "KPI WS3-BI-009: brand and region cannot be combined for "
                    "conversion rate (the region variant predates the brand-"
                    "scoped reads and takes region only)."
                )
            query_id, params = self._region_variant(base), [region]
        elif brand:
            query_id, params = brand_scoped_query_id(base), [brand]
        else:
            query_id, params = base, []

        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("conversion_rate") is not None:
            return float(result[0]["conversion_rate"])
        raise RuntimeError("KPI WS3-BI-009 unavailable: no data for conversion rate")

    # #1532 temporal-variability band: minimum monthly observations a slice
    # needs before its band is shown (below this, n is still reported but the
    # band is suppressed — a 3-month range dressed as a 12-month band would be
    # the same plausible-but-wrong shape #1527 rejected for pooled STDDEV).
    _ROI_BAND_MIN_N = 6

    _ROI_BAND_SEMANTICS = (
        "Range of each (metric_name, brand, region) slice's monthly ROI values "
        "over the trailing 12 months of data — recent temporal variability of "
        "the slice's ROI, NOT uncertainty about the current value. Monthly data "
        "gives n=1 per slice in the 30-day headline window, so no interval on "
        "the headline is possible (#1527); the headline stays a point estimate."
    )

    def _calc_roi(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-010: Return on Investment.

        Value generated per dollar invested. When ``business_metrics`` answers,
        the per-slice trailing-12-month temporal-variability band (#1532) rides
        the context into ``KPIResult.metadata`` (the ``funnel_stages`` seam).
        """
        # Try business_metrics table first
        result = self._execute_query("business_impact_roi_business_metrics", [])
        if result and result[0].get("avg_roi") is not None:
            # Provenance reflects whichever source actually answered (the two
            # probes' frontiers diverge; that is why ROI has no static
            # reporting_window note in the chatbot map).
            self._stash_data_through(context, result)
            self._stash_roi_temporal_band(context)
            return float(result[0]["avg_roi"])

        # Try agent_activities table. No band here: the band describes
        # business_metrics slices; attaching it to an agent_activities headline
        # would pair a figure with dispersion from a different substrate.
        result = self._execute_query("business_impact_roi_agent_activities", [])
        if result and result[0].get("avg_roi") is not None:
            self._stash_data_through(context, result)
            return float(result[0]["avg_roi"])

        raise RuntimeError("KPI WS3-BI-010 unavailable: no data for return on investment (ROI)")

    def _stash_roi_temporal_band(self, context: dict[str, Any]) -> None:
        """Attach the #1532 per-slice band to the context (best-effort).

        The band is supplementary metadata: a failure here must never take the
        headline down with it (the headline itself keeps the #574 fail-loud
        contract via ``_execute_query``), and omitting the band on error is
        honest absence — nothing downstream renders a fabricated range.
        """
        try:
            rows = self._execute_query(
                "business_impact_roi_temporal_band",
                [context.get("brand"), context.get("region")],
            )
        except Exception as exc:  # noqa: BLE001 - supplementary metadata only
            logger.warning("WS3-BI-010 temporal band query failed (band omitted): %s", exc)
            return
        band = self._assemble_roi_temporal_band(rows)
        if band is not None:
            context["temporal_variability_band"] = band

    @classmethod
    def _assemble_roi_temporal_band(
        cls, rows: list[dict[str, Any]] | None, min_n: int | None = None
    ) -> dict[str, Any] | None:
        """Pure: band-query rows -> ``temporal_variability_band`` payload.

        #1532 contract: every slice reports its actual ``n``; the band itself
        appears only at ``n >= min_n`` (default 6) AND with real aggregates —
        otherwise ``band`` is None with ``band_suppressed`` True, never a
        fabricated range. Empty/absent rows -> None (honest absence: real-mode
        on an all-synthetic substrate has zero slices). The payload never uses
        confidence-interval naming (the #1526 sensitivity_band discipline).
        """
        if not rows:
            return None
        floor = cls._ROI_BAND_MIN_N if min_n is None else min_n
        slices: list[dict[str, Any]] = []
        for row in rows:
            n = int(row.get("n") or 0)
            entry: dict[str, Any] = {
                "metric_name": row.get("metric_name"),
                "brand": row.get("brand"),
                "region": row.get("region"),
                "n": n,
            }
            has_stats = row.get("roi_min") is not None and row.get("roi_max") is not None
            if n >= floor and has_stats:
                entry["band"] = {
                    "roi_min": float(row["roi_min"]),
                    "roi_max": float(row["roi_max"]),
                    "roi_mean": (
                        float(row["roi_mean"]) if row.get("roi_mean") is not None else None
                    ),
                    "roi_stddev": (
                        float(row["roi_stddev"]) if row.get("roi_stddev") is not None else None
                    ),
                }
                entry["band_suppressed"] = False
            else:
                entry["band"] = None
                entry["band_suppressed"] = True
            slices.append(entry)
        return {
            "semantics": cls._ROI_BAND_SEMANTICS,
            "window": "trailing 12 months ending at the business_metrics data frontier",
            "min_n": floor,
            "slices": slices,
        }

    def _execute_query(self, query_id: str, params: list[Any]) -> list[dict[str, Any]] | None:
        """Execute a vetted KPI query by id and return results.

        Runs a pre-vetted statement from the kpi_query_registry via the
        kpi_query allowlist RPC, identified by query_id. The params list
        binds positionally to $1..$N in the registered statement.
        """
        # #574: do NOT swallow RPC failures into None — callers convert None -> 0.0,
        # fabricating a zero KPI on a dead/misconfigured backend. Let exceptions propagate
        # to calculate(), which surfaces them as KPIResult(error=...). A successful query
        # with no rows still returns [] (a genuine empty, not an error).
        # Demo/review: swap to the _include_synthetic twin under the
        # E2I_KPI_INCLUDE_SYNTHETIC flag (no-op otherwise). See synthetic_mode.py.
        query_id = resolve_kpi_query_id(query_id)
        response = self.db_client.rpc(
            "kpi_query", {"query_id": query_id, "params": params}
        ).execute()
        return response.data  # type: ignore[no-any-return]
