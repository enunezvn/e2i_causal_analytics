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

    def calculate(
        self, kpi: KPIMetadata, context: dict[str, Any] | None = None
    ) -> KPIResult:
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
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            value = calc_func(context)
            status = self._evaluate_status(kpi, value)
            return KPIResult(
                kpi_id=kpi.id,
                value=value,
                status=status,
                metadata={"context": context},
            )
        except Exception as e:
            return KPIResult(
                kpi_id=kpi.id,
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
        query = """
            SELECT
                COUNT(DISTINCT pj.patient_id) as covered,
                COUNT(DISTINCT ru.patient_id) as total
            FROM patient_journeys pj
            FULL OUTER JOIN reference_universe ru ON pj.patient_id = ru.patient_id
            WHERE ($1::text IS NULL OR pj.brand = $1 OR ru.brand = $1)
        """
        result = self._execute_query(query, [brand])
        if result and result[0]["total"] > 0:
            return result[0]["covered"] / result[0]["total"]
        return 0.0

    def _calc_source_coverage_hcps(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-002: Source Coverage - HCPs.

        Formula: covered_hcps / reference_hcps
        """
        brand = context.get("brand")
        query = """
            SELECT
                COUNT(DISTINCT aa.hcp_id) as covered,
                COUNT(DISTINCT ru.hcp_id) as total
            FROM agent_activities aa
            FULL OUTER JOIN reference_hcps ru ON aa.hcp_id = ru.hcp_id
            WHERE ($1::text IS NULL OR aa.brand = $1 OR ru.brand = $1)
        """
        result = self._execute_query(query, [brand])
        if result and result[0]["total"] > 0:
            return result[0]["covered"] / result[0]["total"]
        return 0.0

    def _calc_cross_source_match(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-003: Cross-source Match Rate.

        Uses v_kpi_cross_source_match view.
        """
        query = "SELECT match_rate FROM v_kpi_cross_source_match LIMIT 1"
        result = self._execute_query(query, [])
        if result:
            return result[0]["match_rate"]
        return 0.0

    def _calc_stacking_lift(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-004: Stacking Lift.

        Uses v_kpi_stacking_lift view.
        """
        query = "SELECT lift_score FROM v_kpi_stacking_lift LIMIT 1"
        result = self._execute_query(query, [])
        if result:
            return result[0]["lift_score"]
        return 1.0  # Neutral lift

    def _calc_completeness_pass_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-005: Completeness Pass Rate.

        Formula: records_passing_completeness / total_records
        """
        query = """
            SELECT
                AVG(CASE
                    WHEN patient_id IS NOT NULL
                    AND brand IS NOT NULL
                    AND event_date IS NOT NULL
                    THEN 1.0 ELSE 0.0
                END) as pass_rate
            FROM patient_journeys
            WHERE created_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result:
            return result[0]["pass_rate"] or 0.0
        return 0.0

    def _calc_geographic_consistency(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-006: Geographic Consistency.

        Formula: consistent_geo_records / total_geo_records
        """
        query = """
            SELECT
                AVG(CASE
                    WHEN hcp_region IS NOT NULL
                    AND hcp_region = patient_region
                    THEN 1.0 ELSE 0.0
                END) as consistency_rate
            FROM patient_journeys pj
            JOIN agent_activities aa ON pj.hcp_id = aa.hcp_id
            WHERE pj.created_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result:
            return result[0]["consistency_rate"] or 0.0
        return 0.0

    def _calc_data_lag(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-007: Data Lag (Median).

        Uses v_kpi_data_lag view.
        Returns median lag in days (lower is better).
        """
        query = "SELECT median_lag_days FROM v_kpi_data_lag LIMIT 1"
        result = self._execute_query(query, [])
        if result:
            return result[0]["median_lag_days"]
        return 0.0

    def _calc_label_quality(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-008: Label Quality (IAA).

        Uses v_kpi_label_quality view.
        Returns inter-annotator agreement score.
        """
        query = "SELECT iaa_score FROM v_kpi_label_quality LIMIT 1"
        result = self._execute_query(query, [])
        if result:
            return result[0]["iaa_score"]
        return 0.0

    def _calc_time_to_release(self, context: dict[str, Any]) -> float:
        """Calculate WS1-DQ-009: Time-to-Release (TTR).

        Uses v_kpi_time_to_release view.
        Returns median time in days (lower is better).
        """
        query = "SELECT median_ttr_days FROM v_kpi_time_to_release LIMIT 1"
        result = self._execute_query(query, [])
        if result:
            return result[0]["median_ttr_days"]
        return 0.0

    def _execute_query(
        self, query: str, params: list[Any]
    ) -> list[dict[str, Any]] | None:
        """Execute a SQL query and return results.

        Args:
            query: SQL query string with $1, $2, etc. placeholders.
            params: Query parameters.

        Returns:
            List of result rows as dictionaries, or None on error.
        """
        try:
            # Use raw SQL execution via Supabase RPC or direct connection
            # This is a simplified implementation - actual implementation
            # would depend on the specific database client
            response = self.db_client.rpc("execute_sql", {"query": query}).execute()
            return response.data
        except Exception:
            # Fall back to mock data for testing
            return None
