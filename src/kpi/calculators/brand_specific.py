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
from src.kpi.synthetic_mode import resolve_kpi_query_id


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

        Percentage of antihistamine(R06A)-treated CSU patients whose disease remains
        uncontrolled (UAS7 >= 7, the EAACI/GA2LEN guideline cutoff, PMID 34536239).
        Lower is better (more controlled = good).

        #577: computed over the generated CSU cohort (baseline_antihistamine events
        carrying a UAS7 reading). The UAS7 cutoff is passed as the bound param so it
        is explicit/auditable. Fails loud if there is no antihistamine-treated cohort
        (an empty denominator must NOT become a fabricated 0% "fully controlled").
        """
        threshold = context.get("uas7_uncontrolled_threshold", 7)
        result = self._execute_query("brand_specific_remi_ah_uncontrolled", [threshold])
        if not result or result[0].get("uncontrolled_rate") is None:
            raise RuntimeError(
                "KPI BR-001 unavailable: no antihistamine-treated CSU cohort "
                "(apply the #577 brand-specific seed, migration 046)"
            )
        return float(result[0]["uncontrolled_rate"])

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
        # Both the primary view and the fallback yielded no intent-delta row -> fail loud.
        # A genuine 0.0 delta (HCPs scored, no net change) is returned by the branches
        # above; this path means there is no scored intent cohort at all, NOT a real zero.
        raise RuntimeError(
            "KPI BR-002 unavailable: no intent-to-prescribe data "
            "(primary view and fallback both returned nothing)"
        )

    def _calc_fabhalta_pnh_tested(self, context: dict[str, Any]) -> float:
        """Calculate BR-003: Fabhalta - % PNH Tested.

        Percentage of PNH-eligible (ICD-10 D59.5) patients who received a PNH
        flow-cytometry diagnostic test (lab_test carrying a real PNH LOINC).

        #577: eligibility is DERIVED from the real D59.5 diagnosis (not a blanket
        is_eligible flag); the numerator counts genuine PNH-flow lab events. Fails
        loud if there is no D59.5-eligible cohort; a genuine 0.0 (cohort exists but
        none tested) is a legitimate value.
        """
        result = self._execute_query("brand_specific_fabhalta_pnh_tested", [])
        if not result or result[0].get("tested_rate") is None:
            raise RuntimeError(
                "KPI BR-003 unavailable: no PNH-eligible (D59.5) cohort "
                "(apply the #577 brand-specific seed, migration 046)"
            )
        return float(result[0]["tested_rate"])

    def _calc_kisqali_dx_adoption(self, context: dict[str, Any]) -> float:
        """Calculate BR-004: Kisqali - Dx Adoption.

        Median time from diagnosis to first Kisqali prescription (days).
        Lower is better.
        """
        result = self._execute_query("brand_specific_kisqali_dx_adoption", [])
        if result and result[0].get("median_days") is not None:
            return float(result[0]["median_days"])
        # No dx->first-Rx pairs -> fail loud (a fabricated 0 days would read as an instant,
        # perfect adoption under the lower-is-better band). A genuine median of 0.0 from the
        # query is still returned by the branch above.
        raise RuntimeError(
            "KPI BR-004 unavailable: no data for Kisqali dx-to-prescription median days"
        )

    def _calc_kisqali_oncologist_reach(self, context: dict[str, Any]) -> float:
        """Calculate BR-005: Kisqali - Oncologist Reach.

        Percentage of oncologists with Kisqali engagement.
        """
        result = self._execute_query("brand_specific_kisqali_oncologist_reach", [])
        if result and result[0].get("reach") is not None:
            return float(result[0]["reach"])
        # No oncologist universe -> fail loud (a fabricated 0% reach would be mistaken for
        # a real "no oncologist engaged"). A genuine 0.0 reach (universe exists, none
        # engaged) is returned by the branch above.
        raise RuntimeError("KPI BR-005 unavailable: no data for Kisqali oncologist reach")

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
        # Demo/review: swap to the _include_synthetic twin under the
        # E2I_KPI_INCLUDE_SYNTHETIC flag (no-op otherwise). See synthetic_mode.py.
        query_id = resolve_kpi_query_id(query_id)
        response = self.db_client.rpc(
            "kpi_query", {"query_id": query_id, "params": params}
        ).execute()
        return response.data  # type: ignore[no-any-return]
