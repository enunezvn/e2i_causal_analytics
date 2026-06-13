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
from src.kpi.synthetic_mode import resolve_kpi_query_id


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
            # Volume metrics (TRx, NRx, NBRx) don't have thresholds
            if kpi.id in {"WS3-BI-005", "WS3-BI-006", "WS3-BI-007"}:
                status = KPIStatus.UNKNOWN
            else:
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
        if value is None or kpi.threshold is None:
            return KPIStatus.UNKNOWN
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

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
        return 0.0

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
        return 0.0

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
        return 0.0

    def _calc_trx(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-005: Total Prescriptions (TRx).

        Total prescription volume. No threshold (volume metric).
        """
        brand = context.get("brand")
        result = self._execute_query("business_impact_trx", [brand])
        if result and result[0].get("trx") is not None:
            return float(result[0]["trx"])
        return 0.0

    def _calc_nrx(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-006: New Prescriptions (NRx).

        First-time prescriptions for a patient. No threshold (volume metric).
        """
        brand = context.get("brand")
        result = self._execute_query("business_impact_nrx", [brand])
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

        result = self._execute_query("business_impact_nbrx", [brand])
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

        result = self._execute_query("business_impact_trx_share", [brand])
        if result and result[0].get("share") is not None:
            return float(result[0]["share"])
        return 0.0

    def _calc_conversion_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-009: Conversion Rate.

        Percentage of triggers resulting in prescription.
        """
        result = self._execute_query("business_impact_conversion_rate", [])
        if result and result[0].get("conversion_rate") is not None:
            return float(result[0]["conversion_rate"])
        return 0.0

    def _calc_roi(self, context: dict[str, Any]) -> float:
        """Calculate WS3-BI-010: Return on Investment.

        Value generated per dollar invested.
        """
        # Try business_metrics table first
        result = self._execute_query("business_impact_roi_business_metrics", [])
        if result and result[0].get("avg_roi") is not None:
            return float(result[0]["avg_roi"])

        # Try agent_activities table
        result = self._execute_query("business_impact_roi_agent_activities", [])
        if result and result[0].get("avg_roi") is not None:
            return float(result[0]["avg_roi"])

        return 0.0

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
