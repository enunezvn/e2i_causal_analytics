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

from typing import Any

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


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
            from src.repositories.base import get_supabase_client

            self._db_client = get_supabase_client()
        return self._db_client

    def supports(self, kpi: KPIMetadata) -> bool:
        """Check if this calculator supports the given KPI."""
        return kpi.workstream == Workstream.WS3_BUSINESS

    def calculate(
        self, kpi: KPIMetadata, context: dict[str, Any] | None = None
    ) -> KPIResult:
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
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            value = calc_func(context)
            # Volume metrics (TRx, NRx, NBRx) don't have thresholds
            if kpi.id in {"WS3-BI-005", "WS3-BI-006", "WS3-BI-007"}:
                status = KPIStatus.UNKNOWN
            else:
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

    def _evaluate_status(
        self, kpi: KPIMetadata, value: float | None, lower_is_better: bool = False
    ) -> KPIStatus:
        """Evaluate KPI value against thresholds."""
        if value is None or kpi.threshold is None:
            return KPIStatus.UNKNOWN
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

    def _calc_mau(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-001: Monthly Active Users.

        Unique users with at least one session in past 30 days.
        Uses v_kpi_active_users view if available.
        """
        # Try view first
        query = """
            SELECT mau
            FROM v_kpi_active_users
            ORDER BY calculated_at DESC
            LIMIT 1
        """
        result = self._execute_query(query, [])
        if result and result[0].get("mau") is not None:
            return float(result[0]["mau"])

        # Fall back to direct calculation
        query = """
            SELECT COUNT(DISTINCT user_id) as mau
            FROM user_sessions
            WHERE session_start >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result and result[0].get("mau") is not None:
            return float(result[0]["mau"])
        return 0.0

    def _calc_wau(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-002: Weekly Active Users.

        Unique users with at least one session in past 7 days.
        """
        # Try view first
        query = """
            SELECT wau
            FROM v_kpi_active_users
            ORDER BY calculated_at DESC
            LIMIT 1
        """
        result = self._execute_query(query, [])
        if result and result[0].get("wau") is not None:
            return float(result[0]["wau"])

        # Fall back to direct calculation
        query = """
            SELECT COUNT(DISTINCT user_id) as wau
            FROM user_sessions
            WHERE session_start >= NOW() - INTERVAL '7 days'
        """
        result = self._execute_query(query, [])
        if result and result[0].get("wau") is not None:
            return float(result[0]["wau"])
        return 0.0

    def _calc_patient_touch_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-003: Patient Touch Rate.

        Percentage of eligible patients with trigger-driven touchpoint.
        """
        query = """
            WITH eligible AS (
                SELECT COUNT(DISTINCT patient_id) as total
                FROM patient_journeys
                WHERE is_eligible = true
            ),
            touched AS (
                SELECT COUNT(DISTINCT t.patient_id) as total
                FROM triggers t
                INNER JOIN patient_journeys pj ON pj.patient_id = t.patient_id
                WHERE pj.is_eligible = true
                AND t.fired_at >= NOW() - INTERVAL '30 days'
            )
            SELECT touched.total::float / NULLIF(eligible.total, 0) as touch_rate
            FROM eligible, touched
        """
        result = self._execute_query(query, [])
        if result and result[0].get("touch_rate") is not None:
            return result[0]["touch_rate"]
        return 0.0

    def _calc_hcp_coverage(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-004: HCP Coverage.

        Percentage of priority HCPs with active engagement.
        """
        query = """
            SELECT
                COUNT(CASE WHEN coverage_status = 'covered' THEN 1 END)::float /
                NULLIF(COUNT(CASE WHEN priority_tier <= 2 THEN 1 END), 0)
                as coverage
            FROM hcp_profiles
        """
        result = self._execute_query(query, [])
        if result and result[0].get("coverage") is not None:
            return result[0]["coverage"]
        return 0.0

    def _calc_trx(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-005: Total Prescriptions (TRx).

        Total prescription volume. No threshold (volume metric).
        """
        brand = context.get("brand")
        brand_filter = "AND brand = $1" if brand else ""

        query = f"""
            SELECT COUNT(*) as trx
            FROM treatment_events
            WHERE event_type = 'prescription'
            AND event_date >= NOW() - INTERVAL '30 days'
            {brand_filter}
        """
        params = [brand] if brand else []
        result = self._execute_query(query, params)
        if result and result[0].get("trx") is not None:
            return float(result[0]["trx"])
        return 0.0

    def _calc_nrx(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-006: New Prescriptions (NRx).

        First-time prescriptions for a patient. No threshold (volume metric).
        """
        brand = context.get("brand")
        brand_filter = "AND brand = $1" if brand else ""

        query = f"""
            SELECT COUNT(*) as nrx
            FROM treatment_events
            WHERE event_type = 'prescription'
            AND sequence_number = 1
            AND event_date >= NOW() - INTERVAL '30 days'
            {brand_filter}
        """
        params = [brand] if brand else []
        result = self._execute_query(query, params)
        if result and result[0].get("nrx") is not None:
            return float(result[0]["nrx"])
        return 0.0

    def _calc_nbrx(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-007: New-to-Brand Prescriptions (NBRx).

        First prescription of specific brand for a patient.
        No threshold (volume metric).
        """
        brand = context.get("brand")
        if not brand:
            return 0.0

        query = """
            WITH first_brand AS (
                SELECT patient_id, MIN(event_date) as first_date
                FROM treatment_events
                WHERE event_type = 'prescription'
                AND brand = $1
                GROUP BY patient_id
            )
            SELECT COUNT(*) as nbrx
            FROM first_brand
            WHERE first_date >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [brand])
        if result and result[0].get("nbrx") is not None:
            return float(result[0]["nbrx"])
        return 0.0

    def _calc_trx_share(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-008: TRx Share.

        Brand prescription share of total category.
        """
        brand = context.get("brand")
        if not brand:
            return 0.0

        query = """
            WITH category AS (
                SELECT COUNT(*) as total
                FROM treatment_events
                WHERE event_type = 'prescription'
                AND event_date >= NOW() - INTERVAL '30 days'
            ),
            brand_rx AS (
                SELECT COUNT(*) as total
                FROM treatment_events
                WHERE event_type = 'prescription'
                AND brand = $1
                AND event_date >= NOW() - INTERVAL '30 days'
            )
            SELECT brand_rx.total::float / NULLIF(category.total, 0) as share
            FROM category, brand_rx
        """
        result = self._execute_query(query, [brand])
        if result and result[0].get("share") is not None:
            return result[0]["share"]
        return 0.0

    def _calc_conversion_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-009: Conversion Rate.

        Percentage of triggers resulting in prescription.
        """
        query = """
            WITH triggered AS (
                SELECT COUNT(DISTINCT trigger_id) as total
                FROM triggers
                WHERE fired_at >= NOW() - INTERVAL '30 days'
            ),
            converted AS (
                SELECT COUNT(DISTINCT t.trigger_id) as total
                FROM triggers t
                INNER JOIN treatment_events te ON te.patient_id = t.patient_id
                WHERE t.fired_at >= NOW() - INTERVAL '30 days'
                AND te.event_type = 'prescription'
                AND te.event_date >= t.fired_at
                AND te.event_date <= t.fired_at + INTERVAL '30 days'
            )
            SELECT converted.total::float / NULLIF(triggered.total, 0) as conversion_rate
            FROM triggered, converted
        """
        result = self._execute_query(query, [])
        if result and result[0].get("conversion_rate") is not None:
            return result[0]["conversion_rate"]
        return 0.0

    def _calc_roi(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-010: Return on Investment.

        Value generated per dollar invested.
        """
        # Try business_metrics table first
        query = """
            SELECT AVG(roi) as avg_roi
            FROM business_metrics
            WHERE metric_date >= NOW() - INTERVAL '30 days'
            AND roi IS NOT NULL
        """
        result = self._execute_query(query, [])
        if result and result[0].get("avg_roi") is not None:
            return result[0]["avg_roi"]

        # Try agent_activities table
        query = """
            SELECT AVG(roi_estimate) as avg_roi
            FROM agent_activities
            WHERE activity_timestamp >= NOW() - INTERVAL '30 days'
            AND roi_estimate IS NOT NULL
        """
        result = self._execute_query(query, [])
        if result and result[0].get("avg_roi") is not None:
            return result[0]["avg_roi"]

        return 0.0

    def _execute_query(
        self, query: str, params: list[Any]
    ) -> list[dict[str, Any]] | None:
        """Execute a SQL query and return results."""
        try:
            response = self.db_client.rpc("execute_sql", {"query": query}).execute()
            return response.data
        except Exception:
            return None
