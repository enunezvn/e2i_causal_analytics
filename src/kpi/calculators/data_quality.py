"""
WS1 Data Quality KPI Calculators

Implements calculators for data quality metrics:
- Source coverage (patients, HCPs)
- Cross-source match rate
- Stacking lift
- Completeness pass rate
- Geographic consistency
- Data lag
- Time-to-release

(WS1-DQ-008 "Label Quality (IAA)" was removed in T8 — a working metric, corpus
Fleiss κ ≈ 0.76, deprioritized by product decision. The DB objects
v_kpi_label_quality + ml_annotations are retained.)
"""

from typing import Any

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)
from src.kpi.synthetic_mode import region_query_id, resolve_kpi_query_id


class DataQualityCalculator(KPICalculatorBase):
    """Calculator for WS1 Data Quality KPIs."""

    def __init__(self, db_client: Any = None):
        """Initialize with database client.

        Args:
            db_client: Database client for executing queries.
                      If None, uses default Supabase client.
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
        return kpi.workstream == Workstream.WS1_DATA_QUALITY

    def calculate(self, kpi: KPIMetadata, context: dict[str, Any] | None = None) -> KPIResult:
        """Calculate a data quality KPI.

        Args:
            kpi: The KPI metadata defining what to calculate.
            context: Optional context with brand, date_range, etc.

        Returns:
            KPIResult with calculated value and status.
        """
        context = context or {}

        # Route to specific calculator based on KPI ID
        calculator_map = {
            "WS1-DQ-001": self._calc_source_coverage_patients,
            "WS1-DQ-002": self._calc_source_coverage_hcps,
            "WS1-DQ-003": self._calc_cross_source_match,
            "WS1-DQ-004": self._calc_stacking_lift,
            "WS1-DQ-005": self._calc_completeness_pass_rate,
            "WS1-DQ-006": self._calc_geographic_consistency,
            "WS1-DQ-007": self._calc_data_lag,
            "WS1-DQ-009": self._calc_time_to_release,
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

    # WS1 data-quality metrics where a LOWER value is better, so the threshold
    # direction must invert (target < warning < critical are "bad" bounds going up):
    #   DQ-006 geographic gap (dimensionless), DQ-007 data lag (query days / thresholds
    #   days — unit-consistent).
    # Declared explicitly (mirrors ModelPerformance/BrandSpecific) because the base
    # KPICalculatorBase._is_lower_better name-heuristic misses "Geographic
    # Consistency" (#577). Without this, a gap value was scored higher-is-better —
    # e.g. the real 0.1049 geographic gap reported GOOD when it is CRITICAL (> 0.10).
    #
    # DQ-009 (time-to-release) is ALSO lower-is-better and is now INCLUDED (#580). It was
    # previously excluded only because its registry query returned DAYS (avg_ttr_hours/24.0
    # AS median_ttr_days) while kpi_definitions.yaml WS1-DQ-009 declares unit HOURS with
    # thresholds 24/48/72 (hours) — a unit mismatch that scored the value against the wrong
    # units. Migration 054 re-registered the row to return avg_ttr_hours (HOURS, the honest
    # name for the view's AVG(time_to_release_hours)); _calc_time_to_release now reads that
    # hours key, so DQ-009 is unit-consistent and evaluated lower-is-better here. Both legs
    # (drop /24.0 + add to this set) must stay together — either alone re-breaks the unit.
    _LOWER_IS_BETTER_IDS = {"WS1-DQ-006", "WS1-DQ-007", "WS1-DQ-009"}

    def _evaluate_status(self, kpi: KPIMetadata, value: float | None) -> KPIStatus:
        """Evaluate KPI value against thresholds (direction-aware)."""
        if value is None:
            return KPIStatus.UNKNOWN
        if kpi.threshold is None:
            # No threshold by design -> tracked for trend/context only.
            return KPIStatus.INFORMATIONAL
        lower_is_better = kpi.id in self._LOWER_IS_BETTER_IDS
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

    @staticmethod
    def _region_scoped(
        base_query_id: str, context: dict[str, Any], base_params: list[Any]
    ) -> tuple[str, list[Any]]:
        """Route to the region-scoped query variant (migration 078) when a region
        is selected, else the base query with its own params.

        The region variants take region as ``$1`` (max_params 1). For coverage,
        the region cut is region-only (region takes precedence over brand);
        ``base_params`` is used verbatim only in the non-region case, so
        region=None stays byte-identical to today (certified gates unaffected).
        Only the three region-decomposable data-quality KPIs have a variant;
        geographic_consistency (cross-region by nature) and the view-backed KPIs
        do not, and keep their portfolio value when a region is selected.
        """
        region = context.get("region")
        if region:
            return region_query_id(base_query_id), [region]
        return base_query_id, base_params

    def _calc_source_coverage_patients(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-001: Source Coverage - Patients.

        Formula: covered_patients / reference_patients
        """
        brand = context.get("brand")
        query_id, params = self._region_scoped(
            "data_quality_source_coverage_patients", context, [brand]
        )
        result = self._execute_query(query_id, params)
        if not result or result[0].get("total") is None or result[0]["total"] <= 0:
            raise RuntimeError(
                "KPI WS1-DQ-001 unavailable: no reference patients to compute "
                "source coverage over (empty result or zero reference universe)"
            )
        # A genuine 0 covered over a real reference universe is a legitimate 0.0 coverage.
        return float(result[0]["covered"] / result[0]["total"])

    def _calc_source_coverage_hcps(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-002: Source Coverage - HCPs.

        Formula: covered_hcps / reference_universe(universe_type='hcp').target_count

        #577: wired to real data as a GLOBAL coverage ratio. Numerator = distinct
        HCPs with ``hcp_profiles.coverage_status = true``; denominator =
        ``SUM(reference_universe.target_count)`` for ``universe_type='hcp'``.
        No brand param: hcp_profiles has no brand column, so the numerator is not
        brand-attributable — banding only the denominator by brand would yield an
        incoherent ratio (global covered HCPs over one brand's universe). Per-brand
        HCP coverage needs a brand-attributable coverage source (future).
        """
        query_id, params = self._region_scoped("data_quality_source_coverage_hcps", context, [])
        result = self._execute_query(query_id, params)
        if not result or result[0].get("total") is None or result[0]["total"] <= 0:
            raise RuntimeError(
                "KPI WS1-DQ-002 unavailable: no reference HCP universe to compute "
                "source coverage over (empty result or zero reference universe)"
            )
        # A genuine 0 covered over a real reference universe is a legitimate 0.0 coverage.
        return float(result[0]["covered"] / result[0]["total"])

    def _calc_cross_source_match(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-003: Cross-source Match Rate.

        Uses v_kpi_cross_source_match view.
        """
        result = self._execute_query("data_quality_cross_source_match", [])
        if not result or result[0].get("match_rate") is None:
            raise RuntimeError("KPI WS1-DQ-003 unavailable: no data for cross-source match rate")
        # A genuine 0.0 match_rate (sources exist but none matched) is legitimate.
        return float(result[0]["match_rate"])

    def _calc_stacking_lift(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-004: Stacking Lift.

        Uses v_kpi_stacking_lift view.
        """
        result = self._execute_query("data_quality_stacking_lift", [])
        if not result or result[0].get("lift_score") is None:
            raise RuntimeError("KPI WS1-DQ-004 unavailable: no data for stacking lift")
        # A genuine realized lift_score (including a real 1.0 neutral or < 1.0) is returned.
        return float(result[0]["lift_score"])

    def _calc_completeness_pass_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-005: Completeness Pass Rate.

        Formula: records_passing_completeness / total_records
        """
        query_id, params = self._region_scoped("data_quality_completeness_pass_rate", context, [])
        result = self._execute_query(query_id, params)
        if not result or result[0].get("pass_rate") is None:
            raise RuntimeError("KPI WS1-DQ-005 unavailable: no data for completeness pass rate")
        # A genuine 0.0 pass_rate (records exist but none passed) is a legitimate value.
        return float(result[0]["pass_rate"])

    def _calc_geographic_consistency(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-006: Geographic Consistency.

        Formula: max_region(|share_source - share_universe|) — the maximum
        absolute gap between the source's regional distribution and the
        reference universe's regional distribution (lower is better).

        #577: wired to the authoritative formula (config/kpi_definitions.yaml +
        docs/data/06-KPI-REFERENCE.md). Source share = patient_journeys by
        geographic_region; universe share = reference_universe(universe_type=
        'patient') by region. The pre-#574 stub joined a non-existent
        agent_activities.hcp_id AND measured region self-consistency (the wrong
        metric); this implements the documented share-gap instead.
        """
        brand = context.get("brand")
        result = self._execute_query("data_quality_geographic_consistency", [brand])
        if not result or result[0].get("max_gap") is None:
            raise RuntimeError("KPI WS1-DQ-006 unavailable: no data for geographic consistency gap")
        # A genuine 0.0 max_gap (source distribution perfectly matches the universe) is
        # a legitimate best-case value and is returned, not raised.
        return float(result[0]["max_gap"])

    def _calc_data_lag(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-007: Data Lag (Median).

        Uses v_kpi_data_lag view.
        Returns median lag in days (lower is better).
        """
        result = self._execute_query("data_quality_data_lag", [])
        if not result or result[0].get("median_lag_days") is None:
            raise RuntimeError("KPI WS1-DQ-007 unavailable: no data for median data lag")
        # A genuine 0.0 median lag (data lands same-day) is a legitimate best-case value.
        return float(result[0]["median_lag_days"])

    def _calc_time_to_release(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-009: Time-to-Release (TTR).

        Uses v_kpi_time_to_release view. Returns the average time-to-release in
        HOURS (lower is better), matching the kpi_definitions.yaml WS1-DQ-009 unit
        and thresholds (target 24 / warning 48 / critical 72, hours). #580: the
        registry row returns ``avg_ttr_hours`` (the view's AVG(time_to_release_hours))
        — previously ``(avg_ttr_hours / 24.0) AS median_ttr_days``, which both
        converted to days (mismatching the hour thresholds) and mislabeled an AVG
        as a 'median'.
        """
        result = self._execute_query("data_quality_time_to_release", [])
        if not result or result[0].get("avg_ttr_hours") is None:
            raise RuntimeError("KPI WS1-DQ-009 unavailable: no data for time-to-release")
        # A genuine 0.0 hours (instantaneous release) is a legitimate best-case value.
        return float(result[0]["avg_ttr_hours"])

    def _execute_query(self, query_id: str, params: list[Any]) -> list[dict[str, Any]] | None:
        """Run a vetted KPI statement via the `kpi_query` allowlist RPC.

        #574: the calculators no longer build raw SQL. `query_id` selects a
        pre-vetted statement from `kpi_query_registry`; `params` bind to its
        `$1..$N` placeholders. This replaces the now-dead `execute_sql` RPC.

        F-007 iter-2 (#421): the prior implementation caught all exceptions
        and returned `None`, which callers translated into `0.0` (e.g.,
        `_calc_source_coverage_patients` returns `0.0` when result is None).
        That cascaded user-visible "0%" KPI values from silent Supabase
        failures — RPC unreachable, missing function, auth error all looked
        identical to "no rows", which looked identical to "perfectly zero".

        Now: exceptions propagate. The outer `DataQualityCalculator.calculate`
        (line 86-104) catches them and emits `KPIResult(value=None,
        error=str(e))` — the user sees a real error message instead of a
        fabricated zero.

        An empty result set (`response.data == []`) is still returned as
        `[]` — that's the legitimate "no rows" case the caller can handle
        with its own logic (e.g., "0 covered patients / 100 reference =
        0.0%" is correct; "RPC failed" should NOT silently become "0.0%").

        Args:
            query_id: Registry id of a vetted statement in
                `kpi_query_registry` (its `$1..$N` placeholders bind `params`).
            params: Query parameters bound to `$1..$N`.

        Returns:
            List of result rows as dictionaries (possibly empty).

        Raises:
            RuntimeError: if no Supabase client is configured.
            Exception: any exception raised by `self.db_client.rpc(...)`
                propagates up to the calling KPI helper, which propagates to
                `calculate()`'s outer try/except — surfacing as
                `KPIResult.error`.
        """
        if self.db_client is None:
            raise RuntimeError(
                "DataQualityCalculator has no Supabase client; cannot execute KPI query"
            )
        # Demo/review: swap to the _include_synthetic twin under the
        # E2I_KPI_INCLUDE_SYNTHETIC flag (no-op otherwise). See synthetic_mode.py.
        query_id = resolve_kpi_query_id(query_id)
        response = self.db_client.rpc(
            "kpi_query", {"query_id": query_id, "params": params}
        ).execute()
        data = getattr(response, "data", None)
        if data is None:
            return []
        return list(data)
