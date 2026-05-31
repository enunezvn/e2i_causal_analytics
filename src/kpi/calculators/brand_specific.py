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
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
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
                cached=False,
                error=None,
                metadata={"context": context, "lower_is_better": lower_is_better},
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

    def _calc_remi_ah_uncontrolled(self, context: dict[str, Any]) -> float:
        """Calculate BR-001: Remi - AH Uncontrolled %.

        Percentage of antihistamine patients with uncontrolled symptoms.
        Lower is better (more controlled = good).
        """
        raise RuntimeError(
            "KPI BR-001 unavailable: 'antihistamine' is not a valid "
            "event_type/brand category; metric cannot be computed without "
            "misrepresenting it (#574)"
        )

    def _calc_remi_intent_delta(self, context: dict[str, Any]) -> float:
        """Calculate BR-002: Remi - Intent-to-Prescribe Δ.

        Change in HCP intent-to-prescribe score after intervention.
        Uses v_kpi_intent_to_prescribe view if available.
        """
        # Try view first
        result = self._execute_query("brand_specific_remi_intent_delta_primary", [])
        if result and result[0].get("intent_delta") is not None:
            return float(result[0]["intent_delta"])

        # Fall back to direct calculation
        result = self._execute_query("brand_specific_remi_intent_delta_fallback", [])
        if result and result[0].get("intent_delta") is not None:
            return float(result[0]["intent_delta"])
        return 0.0

    def _calc_fabhalta_pnh_tested(self, context: dict[str, Any]) -> float:
        """Calculate BR-003: Fabhalta - % PNH Tested.

        Percentage of eligible patients tested for PNH.
        """
        raise RuntimeError(
            "KPI BR-003 unavailable: patient_journeys has no is_eligible column (#574)"
        )

    def _calc_kisqali_dx_adoption(self, context: dict[str, Any]) -> float:
        """Calculate BR-004: Kisqali - Dx Adoption.

        Median time from diagnosis to first Kisqali prescription (days).
        Lower is better.
        """
        result = self._execute_query("brand_specific_kisqali_dx_adoption", [])
        if result and result[0].get("median_days") is not None:
            return float(result[0]["median_days"])
        return 0.0

    def _calc_kisqali_oncologist_reach(self, context: dict[str, Any]) -> float:
        """Calculate BR-005: Kisqali - Oncologist Reach.

        Percentage of oncologists with Kisqali engagement.
        """
        result = self._execute_query("brand_specific_kisqali_oncologist_reach", [])
        if result and result[0].get("reach") is not None:
            return float(result[0]["reach"])
        return 0.0

    def _execute_query(self, query_id: str, params: list[Any]) -> list[dict[str, Any]] | None:
        """Run a vetted statement from kpi_query_registry by id.

        The statement identified by ``query_id`` is executed via the
        ``kpi_query`` allowlist RPC; ``params`` bind positionally to its
        ``$1..$N`` placeholders.
        """
        # #574: do NOT swallow RPC failures into None — callers convert None -> 0.0,
        # fabricating a zero KPI on a dead/misconfigured backend. Let exceptions propagate
        # to calculate(), which surfaces them as KPIResult(error=...). A successful query
        # with no rows still returns [] (a genuine empty, not an error).
        response = self.db_client.rpc(
            "kpi_query", {"query_id": query_id, "params": params}
        ).execute()
        return response.data  # type: ignore[no-any-return]
