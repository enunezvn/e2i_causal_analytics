"""
Brand-Specific KPI Calculators

Implements calculators for brand-specific metrics:
- Remibrutinib: AH Uncontrolled %, Intent-to-Prescribe Δ
- Fabhalta: % PNH Tested
- Kisqali: Dx Adoption, Oncologist Reach
"""

from typing import Any

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


class BrandSpecificCalculator(KPICalculatorBase):
    """Calculator for Brand-Specific KPIs."""

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
        return kpi.workstream == Workstream.BRAND_SPECIFIC

    def calculate(self, kpi: KPIMetadata, context: dict[str, Any] | None = None) -> KPIResult:
        """Calculate a brand-specific KPI.

        Args:
            kpi: The KPI metadata defining what to calculate.
            context: Optional context with brand, date_range, etc.

        Returns:
            KPIResult with calculated value and status.
        """
        context = context or {}

        calculator_map = {
            "BR-001": self._calc_remi_ah_uncontrolled,
            "BR-002": self._calc_remi_intent_delta,
            "BR-003": self._calc_fabhalta_pnh_tested,
            "BR-004": self._calc_kisqali_dx_adoption,
            "BR-005": self._calc_kisqali_oncologist_reach,
        }

        calc_func = calculator_map.get(kpi.id)
        if calc_func is None:
            return KPIResult(
                kpi_id=kpi.id,
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            value = calc_func(context)
            # BR-001 (uncontrolled %) and BR-004 (days) are lower-is-better
            lower_is_better = kpi.id in {"BR-001", "BR-004"}
            status = self._evaluate_status(kpi, value, lower_is_better)
            return KPIResult(
                kpi_id=kpi.id,
                value=value,
                status=status,
                metadata={"context": context, "lower_is_better": lower_is_better},
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

    def _calc_remi_ah_uncontrolled(self, context: dict[str, Any]) -> float:
        """Calculate BR-001: Remi - AH Uncontrolled %.

        Percentage of antihistamine patients with uncontrolled symptoms.
        Lower is better (more controlled = good).
        """
        query = """
            WITH ah_patients AS (
                SELECT DISTINCT pj.patient_id
                FROM patient_journeys pj
                INNER JOIN treatment_events te ON te.patient_id = pj.patient_id
                WHERE te.brand IN ('antihistamine', 'H1-blocker')
                AND pj.diagnosis LIKE '%CSU%'
            ),
            uncontrolled AS (
                SELECT COUNT(DISTINCT ap.patient_id) as total
                FROM ah_patients ap
                INNER JOIN treatment_events te ON te.patient_id = ap.patient_id
                WHERE te.treatment_response IN ('inadequate', 'uncontrolled', 'refractory')
            ),
            total_ah AS (
                SELECT COUNT(*) as total FROM ah_patients
            )
            SELECT uncontrolled.total::float / NULLIF(total_ah.total, 0) as uncontrolled_pct
            FROM uncontrolled, total_ah
        """
        result = self._execute_query(query, [])
        if result and result[0].get("uncontrolled_pct") is not None:
            return result[0]["uncontrolled_pct"]
        return 0.0

    def _calc_remi_intent_delta(self, context: dict[str, Any]) -> float:
        """Calculate BR-002: Remi - Intent-to-Prescribe Δ.

        Change in HCP intent-to-prescribe score after intervention.
        Uses v_kpi_intent_to_prescribe view if available.
        """
        # Try view first
        query = """
            SELECT avg_intent_change as intent_delta
            FROM v_kpi_intent_to_prescribe
            WHERE brand = 'Remibrutinib'
            ORDER BY survey_month DESC
            LIMIT 1
        """
        result = self._execute_query(query, [])
        if result and result[0].get("intent_delta") is not None:
            return result[0]["intent_delta"]

        # Fall back to direct calculation
        query = """
            SELECT AVG(intent_to_prescribe_change) as intent_delta
            FROM hcp_intent_surveys
            WHERE brand = 'Remibrutinib'
            AND survey_date >= NOW() - INTERVAL '90 days'
            AND intent_to_prescribe_change IS NOT NULL
        """
        result = self._execute_query(query, [])
        if result and result[0].get("intent_delta") is not None:
            return result[0]["intent_delta"]
        return 0.0

    def _calc_fabhalta_pnh_tested(self, context: dict[str, Any]) -> float:
        """Calculate BR-003: Fabhalta - % PNH Tested.

        Percentage of eligible patients tested for PNH.
        """
        query = """
            WITH eligible AS (
                SELECT COUNT(DISTINCT patient_id) as total
                FROM patient_journeys
                WHERE diagnosis LIKE '%PNH%'
                OR (diagnosis LIKE '%anemia%' AND is_eligible = true)
            ),
            tested AS (
                SELECT COUNT(DISTINCT patient_id) as total
                FROM treatment_events
                WHERE test_type IN ('flow_cytometry', 'pnh_panel', 'gpi_anchor')
            )
            SELECT tested.total::float / NULLIF(eligible.total, 0) as tested_pct
            FROM eligible, tested
        """
        result = self._execute_query(query, [])
        if result and result[0].get("tested_pct") is not None:
            return result[0]["tested_pct"]
        return 0.0

    def _calc_kisqali_dx_adoption(self, context: dict[str, Any]) -> float:
        """Calculate BR-004: Kisqali - Dx Adoption.

        Median time from diagnosis to first Kisqali prescription (days).
        Lower is better.
        """
        query = """
            WITH first_kisqali AS (
                SELECT
                    te.patient_id,
                    MIN(te.event_date) as first_rx_date
                FROM treatment_events te
                WHERE te.brand = 'Kisqali'
                AND te.event_type = 'prescription'
                GROUP BY te.patient_id
            )
            SELECT
                PERCENTILE_CONT(0.5) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (fk.first_rx_date - pj.diagnosis_date)) / 86400
                ) as median_days
            FROM first_kisqali fk
            INNER JOIN patient_journeys pj ON pj.patient_id = fk.patient_id
            WHERE pj.diagnosis_date IS NOT NULL
            AND fk.first_rx_date >= pj.diagnosis_date
        """
        result = self._execute_query(query, [])
        if result and result[0].get("median_days") is not None:
            return float(result[0]["median_days"])
        return 0.0

    def _calc_kisqali_oncologist_reach(self, context: dict[str, Any]) -> float:
        """Calculate BR-005: Kisqali - Oncologist Reach.

        Percentage of oncologists with Kisqali engagement.
        """
        query = """
            WITH oncologists AS (
                SELECT COUNT(DISTINCT hcp_id) as total
                FROM hcp_profiles
                WHERE specialty LIKE '%oncolog%'
            ),
            engaged AS (
                SELECT COUNT(DISTINCT t.hcp_id) as total
                FROM triggers t
                INNER JOIN hcp_profiles hp ON hp.hcp_id = t.hcp_id
                WHERE hp.specialty LIKE '%oncolog%'
                AND t.brand = 'Kisqali'
                AND t.fired_at >= NOW() - INTERVAL '90 days'
            )
            SELECT engaged.total::float / NULLIF(oncologists.total, 0) as reach
            FROM oncologists, engaged
        """
        result = self._execute_query(query, [])
        if result and result[0].get("reach") is not None:
            return result[0]["reach"]
        return 0.0

    def _execute_query(self, query: str, params: list[Any]) -> list[dict[str, Any]] | None:
        """Execute a SQL query and return results."""
        try:
            response = self.db_client.rpc("execute_sql", {"query": query}).execute()
            return response.data
        except Exception:
            return None
