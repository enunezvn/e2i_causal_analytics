"""
WS1 Data Quality KPI Calculators

Implements calculators for data quality metrics:
- Source coverage (patients, HCPs)
- Cross-source match rate
- Stacking lift
- Completeness pass rate
- Geographic consistency
- Data lag
- Label quality
- Time-to-release
"""

from typing import Any

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


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
            "WS1-DQ-008": self._calc_label_quality,
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

    def _evaluate_status(self, kpi: KPIMetadata, value: float | None) -> KPIStatus:
        """Evaluate KPI value against thresholds."""
        if value is None or kpi.threshold is None:
            return KPIStatus.UNKNOWN
        return kpi.threshold.evaluate(value)

    def _calc_source_coverage_patients(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-001: Source Coverage - Patients.

        Formula: covered_patients / reference_patients
        """
        brand = context.get("brand")
        result = self._execute_query("data_quality_source_coverage_patients", [brand])
        if result and result[0]["total"] > 0:
            return float(result[0]["covered"] / result[0]["total"])
        return 0.0

    def _calc_source_coverage_hcps(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-002: Source Coverage - HCPs.

        Formula: covered_hcps / reference_hcps
        """
        context.get("brand")
        raise RuntimeError("KPI WS1-DQ-002 unavailable: reference_hcps table does not exist (#574)")

    def _calc_cross_source_match(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-003: Cross-source Match Rate.

        Uses v_kpi_cross_source_match view.
        """
        result = self._execute_query("data_quality_cross_source_match", [])
        if result:
            return float(result[0]["match_rate"])
        return 0.0

    def _calc_stacking_lift(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-004: Stacking Lift.

        Uses v_kpi_stacking_lift view.
        """
        result = self._execute_query("data_quality_stacking_lift", [])
        if result:
            return float(result[0]["lift_score"])
        return 1.0  # Neutral lift

    def _calc_completeness_pass_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-005: Completeness Pass Rate.

        Formula: records_passing_completeness / total_records
        """
        result = self._execute_query("data_quality_completeness_pass_rate", [])
        if result:
            return result[0]["pass_rate"] or 0.0
        return 0.0

    def _calc_geographic_consistency(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-006: Geographic Consistency.

        Formula: consistent_geo_records / total_geo_records
        """
        raise RuntimeError(
            "KPI WS1-DQ-006 unavailable: agent_activities has no hcp_id column for the join (#574)"
        )

    def _calc_data_lag(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-007: Data Lag (Median).

        Uses v_kpi_data_lag view.
        Returns median lag in days (lower is better).
        """
        result = self._execute_query("data_quality_data_lag", [])
        if result:
            return float(result[0]["median_lag_days"])
        return 0.0

    def _calc_label_quality(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-008: Label Quality (IAA).

        Uses v_kpi_label_quality view.
        Returns inter-annotator agreement score.
        """
        raise RuntimeError(
            "KPI WS1-DQ-008 unavailable: no iaa_score source column available (#574)"
        )

    def _calc_time_to_release(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-009: Time-to-Release (TTR).

        Uses v_kpi_time_to_release view.
        Returns median time in days (lower is better).
        """
        result = self._execute_query("data_quality_time_to_release", [])
        if result:
            return float(result[0]["median_ttr_days"])
        return 0.0

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
        response = self.db_client.rpc(
            "kpi_query", {"query_id": query_id, "params": params}
        ).execute()
        data = getattr(response, "data", None)
        if data is None:
            return []
        return list(data)
