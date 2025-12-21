"""ROI Calculator Node for Gap Analyzer Agent.

This node calculates ROI estimates for closing performance gaps using
pharmaceutical-specific economics from ROICalculationService.

Implements the full ROI methodology from docs/roi_methodology.md:
- 6 Value Drivers (TRx Lift, Patient ID, Action Rate, ITP, Data Quality, Drift)
- Bootstrap Monte Carlo simulations (1,000) for confidence intervals
- Attribution Framework (Full/Partial/Shared/Minimal)
- Risk Adjustment (Technical/Organizational/Data/Timeline factors)

Reference: docs/roi_methodology.md, src/services/roi_calculation.py
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.services.roi_calculation import (
    AttributionLevel,
    CostInput,
    RiskAssessment,
    RiskLevel,
    ROICalculationService,
    ROIResult,
    ValueDriverInput,
    ValueDriverType,
)

from ..state import (
    ConfidenceIntervalDict,
    GapAnalyzerState,
    PerformanceGap,
    ROIEstimate,
)

logger = logging.getLogger(__name__)


class ROICalculatorNode:
    """Calculate ROI estimates for performance gaps.

    Uses pharmaceutical-specific economics from ROICalculationService with:
    - 6 value drivers at pharma-specific unit rates
    - Bootstrap confidence intervals (1,000 simulations)
    - Attribution framework (full/partial/shared/minimal)
    - Risk adjustment (4 factors)
    """

    # Mapping from KPI metric to primary value driver
    METRIC_TO_DRIVER: Dict[str, ValueDriverType] = {
        "trx": ValueDriverType.TRX_LIFT,
        "nrx": ValueDriverType.TRX_LIFT,
        "patient_count": ValueDriverType.PATIENT_IDENTIFICATION,
        "patient_identification": ValueDriverType.PATIENT_IDENTIFICATION,
        "trigger_acceptance": ValueDriverType.ACTION_RATE,
        "trigger_count": ValueDriverType.ACTION_RATE,
        "hcp_engagement_score": ValueDriverType.INTENT_TO_PRESCRIBE,
        "itp": ValueDriverType.INTENT_TO_PRESCRIBE,
        "conversion_rate": ValueDriverType.INTENT_TO_PRESCRIBE,
        "data_quality": ValueDriverType.DATA_QUALITY,
        "model_accuracy": ValueDriverType.DRIFT_PREVENTION,
        "market_share": ValueDriverType.TRX_LIFT,  # Translates to TRx impact
    }

    # Default cost category for gap initiatives
    DEFAULT_COST_CATEGORY = "algorithm_optimization"

    # Engineering cost per day
    ENGINEERING_RATE = 2500.0  # USD per day

    def __init__(
        self,
        roi_service: Optional[ROICalculationService] = None,
        use_bootstrap: bool = True,
        n_simulations: int = 1000,
    ):
        """Initialize ROI calculator with service.

        Args:
            roi_service: Injected ROICalculationService (or created if None)
            use_bootstrap: Whether to compute bootstrap confidence intervals
            n_simulations: Number of Monte Carlo simulations for bootstrap
        """
        self.roi_service = roi_service or ROICalculationService(n_simulations=n_simulations)
        self.use_bootstrap = use_bootstrap

    async def execute(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Execute ROI calculation workflow.

        Args:
            state: Current gap analyzer state with gaps_detected

        Returns:
            Updated state with roi_estimates, total_addressable_value, roi_latency_ms
        """
        start_time = time.time()

        try:
            gaps_detected = state.get("gaps_detected", [])

            if not gaps_detected:
                return {
                    "roi_estimates": [],
                    "total_addressable_value": 0.0,
                    "roi_latency_ms": 0,
                    "warnings": ["No gaps detected for ROI calculation"],
                    "status": "prioritizing",
                }

            # Calculate ROI for each gap using ROICalculationService
            roi_estimates: List[ROIEstimate] = []

            for gap in gaps_detected:
                roi_estimate = self._calculate_roi(gap)
                roi_estimates.append(roi_estimate)

            # Calculate total addressable value (attributed value)
            total_addressable_value = sum(est["estimated_revenue_impact"] for est in roi_estimates)

            roi_latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"ROI calculated for {len(roi_estimates)} gaps, "
                f"total addressable value: ${total_addressable_value:,.0f}"
            )

            return {
                "roi_estimates": roi_estimates,
                "total_addressable_value": total_addressable_value,
                "roi_latency_ms": roi_latency_ms,
                "status": "prioritizing",
            }

        except Exception as e:
            logger.error(f"ROI calculation failed: {e}")
            roi_latency_ms = int((time.time() - start_time) * 1000)
            return {
                "errors": [
                    {
                        "node": "roi_calculator",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                ],
                "roi_latency_ms": roi_latency_ms,
                "status": "failed",
            }

    def _calculate_roi(self, gap: PerformanceGap) -> ROIEstimate:
        """Calculate ROI estimate for a single gap using ROICalculationService.

        Implements full ROI methodology:
        1. Map metric to value driver
        2. Estimate intervention costs
        3. Determine attribution level from gap type
        4. Apply risk adjustment
        5. Run bootstrap simulations for confidence interval

        Args:
            gap: Performance gap to analyze

        Returns:
            ROI estimate with confidence interval, attribution, risk adjustment
        """
        metric = gap["metric"]
        gap_size = abs(gap["gap_size"])
        gap_type = gap["gap_type"]

        # Map metric to value driver
        driver_type = self._get_value_driver(metric)

        # Create value driver input
        value_driver = self._create_value_driver_input(driver_type, gap_size, gap)

        # Estimate costs for closing the gap
        cost_input = self._estimate_intervention_costs(metric, gap_size)

        # Determine attribution level from gap type
        attribution = self._determine_attribution(gap_type)

        # Assess risks based on gap characteristics
        risk_assessment = self._assess_risks(gap)

        # Calculate ROI using the full service
        roi_result: ROIResult = self.roi_service.calculate_roi(
            value_drivers=[value_driver],
            cost_input=cost_input,
            attribution_level=attribution,
            risk_assessment=risk_assessment,
        )

        # Build confidence interval dict
        confidence_interval: Optional[ConfidenceIntervalDict] = None
        if roi_result.confidence_interval:
            ci = roi_result.confidence_interval
            confidence_interval = {
                "lower_bound": ci.lower_bound,
                "median": ci.median,
                "upper_bound": ci.upper_bound,
                "probability_positive": ci.probability_positive,
                "probability_target": ci.probability_target,
            }

        # Build assumptions list
        assumptions = self._build_assumptions(metric, driver_type, attribution, risk_assessment)

        # Legacy confidence (use probability_positive if available)
        legacy_confidence = (
            confidence_interval["probability_positive"]
            if confidence_interval
            else self._calculate_legacy_confidence(gap)
        )

        # Convert ROIResult to ROIEstimate TypedDict
        roi_estimate: ROIEstimate = {
            "gap_id": gap["gap_id"],
            "estimated_revenue_impact": roi_result.attributed_value,
            "estimated_cost_to_close": roi_result.implementation_cost,
            "expected_roi": roi_result.base_roi,
            "risk_adjusted_roi": roi_result.risk_adjusted_roi,
            "payback_period_months": self._calculate_payback_months(
                roi_result.attributed_value, roi_result.implementation_cost
            ),
            "confidence_interval": confidence_interval,
            "attribution_level": attribution.value,
            "attribution_rate": roi_result.attribution_rate,
            "total_risk_adjustment": roi_result.total_risk_adjustment,
            "value_by_driver": roi_result.value_by_driver,
            "confidence": legacy_confidence,
            "assumptions": assumptions,
        }

        return roi_estimate

    def _get_value_driver(self, metric: str) -> ValueDriverType:
        """Map KPI metric to primary value driver type.

        Args:
            metric: KPI metric name

        Returns:
            Corresponding value driver type
        """
        return self.METRIC_TO_DRIVER.get(metric, ValueDriverType.TRX_LIFT)

    def _create_value_driver_input(
        self,
        driver_type: ValueDriverType,
        gap_size: float,
        gap: PerformanceGap,
    ) -> ValueDriverInput:
        """Create value driver input for ROI calculation.

        Maps gap size to appropriate driver quantity based on type.

        Args:
            driver_type: Type of value driver
            gap_size: Absolute gap size
            gap: Full gap details for context

        Returns:
            ValueDriverInput for ROI service
        """
        # Convert gap size to driver-appropriate quantity
        # For TRx: gap_size is TRx count
        # For Patient ID: gap_size might need conversion
        # For Action Rate: gap_size is trigger count
        # etc.

        return ValueDriverInput(
            driver_type=driver_type,
            quantity=gap_size,
            # Optional fields based on driver type
            hcp_count=(
                int(gap_size / 10) if driver_type == ValueDriverType.INTENT_TO_PRESCRIBE else None
            ),
            trigger_count=int(gap_size) if driver_type == ValueDriverType.ACTION_RATE else None,
            fp_reduction=(
                int(gap_size * 0.3) if driver_type == ValueDriverType.DATA_QUALITY else None
            ),
            fn_reduction=(
                int(gap_size * 0.7) if driver_type == ValueDriverType.DATA_QUALITY else None
            ),
            auc_drop_prevented=0.02 if driver_type == ValueDriverType.DRIFT_PREVENTION else None,
            baseline_model_value=(
                gap_size * 850 if driver_type == ValueDriverType.DRIFT_PREVENTION else None
            ),
        )

    def _estimate_intervention_costs(
        self,
        metric: str,
        gap_size: float,
    ) -> CostInput:
        """Estimate intervention costs for closing a gap.

        Cost components:
        - Engineering effort (based on gap size)
        - Data acquisition (if data quality or patient ID)
        - Change management (if organizational change needed)

        Args:
            metric: KPI metric
            gap_size: Size of gap to close

        Returns:
            CostInput for ROI calculation
        """
        # Engineering cost (scaled by gap complexity)
        engineering_days = self._estimate_engineering_days(metric, gap_size)

        # Data acquisition (for patient ID and data quality metrics)
        data_source_costs: Dict[str, float] = {}
        incremental_data_cost = 0.0
        if metric in ["patient_identification", "patient_count", "data_quality"]:
            incremental_data_cost = gap_size * 100  # ~$100 per patient identified
            data_source_costs["IQVIA APLD"] = incremental_data_cost

        # Change management (for org-level changes)
        change_management_cost = 0.0
        if gap_size > 100 or metric in ["conversion_rate", "market_share"]:
            change_management_cost = min(50000, gap_size * 200)  # Cap at $50k

        return CostInput(
            engineering_days=engineering_days,
            engineering_day_rate=self.ENGINEERING_RATE,
            data_source_costs=data_source_costs,
            incremental_data_cost=incremental_data_cost,
            change_management_cost=change_management_cost,
        )

    def _estimate_engineering_days(self, metric: str, gap_size: float) -> float:
        """Estimate engineering days required to close a gap.

        Args:
            metric: KPI metric
            gap_size: Size of gap

        Returns:
            Estimated engineering days
        """
        # Base days by metric type
        base_days = {
            "trx": 5,
            "nrx": 5,
            "patient_count": 10,
            "patient_identification": 10,
            "trigger_acceptance": 8,
            "conversion_rate": 15,
            "hcp_engagement_score": 8,
            "data_quality": 12,
            "model_accuracy": 20,
            "market_share": 10,
        }

        base = base_days.get(metric, 10)

        # Scale by gap size (larger gaps = more effort)
        if gap_size > 1000:
            scale = 2.0
        elif gap_size > 100:
            scale = 1.5
        elif gap_size > 10:
            scale = 1.0
        else:
            scale = 0.5

        return base * scale

    def _determine_attribution(self, gap_type: str) -> AttributionLevel:
        """Determine attribution level from gap type.

        Attribution reflects how much of the improvement can be attributed
        to the initiative:
        - vs_target: Full (100%) - direct target setting
        - vs_benchmark: Partial (65%) - peer comparison has some noise
        - vs_potential: Shared (35%) - multiple factors for top decile
        - temporal: Minimal (10%) - many confounders over time

        Args:
            gap_type: Type of gap comparison

        Returns:
            Attribution level
        """
        attribution_mapping = {
            "vs_target": AttributionLevel.FULL,
            "vs_benchmark": AttributionLevel.PARTIAL,
            "vs_potential": AttributionLevel.SHARED,
            "temporal": AttributionLevel.MINIMAL,
        }
        return attribution_mapping.get(gap_type, AttributionLevel.PARTIAL)

    def _assess_risks(self, gap: PerformanceGap) -> RiskAssessment:
        """Assess risk factors for closing a gap.

        Risk factors:
        - Technical complexity: Based on metric type
        - Organizational change: Based on gap size
        - Data dependencies: Based on metric data requirements
        - Timeline uncertainty: Based on gap percentage

        Args:
            gap: Performance gap

        Returns:
            Risk assessment for ROI adjustment
        """
        metric = gap["metric"]
        gap_size = abs(gap["gap_size"])
        gap_pct = abs(gap["gap_percentage"])

        # Technical complexity
        complex_metrics = ["model_accuracy", "data_quality", "conversion_rate"]
        if metric in complex_metrics:
            technical = RiskLevel.HIGH
        elif metric in ["patient_identification", "hcp_engagement_score"]:
            technical = RiskLevel.MEDIUM
        else:
            technical = RiskLevel.LOW

        # Organizational change
        if gap_size > 500:
            organizational = RiskLevel.HIGH
        elif gap_size > 100:
            organizational = RiskLevel.MEDIUM
        else:
            organizational = RiskLevel.LOW

        # Data dependencies
        data_heavy = ["patient_identification", "patient_count", "data_quality"]
        if metric in data_heavy:
            data_deps = RiskLevel.HIGH
        elif metric in ["trigger_acceptance", "model_accuracy"]:
            data_deps = RiskLevel.MEDIUM
        else:
            data_deps = RiskLevel.LOW

        # Timeline uncertainty
        if gap_pct > 50:
            timeline = RiskLevel.HIGH
        elif gap_pct > 20:
            timeline = RiskLevel.MEDIUM
        else:
            timeline = RiskLevel.LOW

        return RiskAssessment(
            technical_complexity=technical,
            organizational_change=organizational,
            data_dependencies=data_deps,
            timeline_uncertainty=timeline,
        )

    def _calculate_payback_months(
        self,
        revenue_impact: float,
        cost: float,
    ) -> int:
        """Calculate payback period in months.

        Args:
            revenue_impact: Annual revenue impact
            cost: One-time implementation cost

        Returns:
            Payback period in months (1-24)
        """
        if revenue_impact <= 0:
            return 24

        monthly_revenue = revenue_impact / 12
        if monthly_revenue <= 0:
            return 24

        months = int(cost / monthly_revenue)
        return max(1, min(months, 24))  # Clamp to 1-24

    def _build_assumptions(
        self,
        metric: str,
        driver_type: ValueDriverType,
        attribution: AttributionLevel,
        risk: RiskAssessment,
    ) -> List[str]:
        """Build list of assumptions for transparency.

        Args:
            metric: KPI metric
            driver_type: Value driver used
            attribution: Attribution level
            risk: Risk assessment

        Returns:
            List of assumption statements
        """
        # Get unit value from service
        unit_values = {
            ValueDriverType.TRX_LIFT: "$850/TRx",
            ValueDriverType.PATIENT_IDENTIFICATION: "$1,200/patient",
            ValueDriverType.ACTION_RATE: "$45/pp/1000 triggers",
            ValueDriverType.INTENT_TO_PRESCRIBE: "$320/HCP/pp",
            ValueDriverType.DATA_QUALITY: "$200/FP, $650/FN",
            ValueDriverType.DRIFT_PREVENTION: "2x value multiplier",
        }

        assumptions = [
            f"Value driver: {driver_type.value}",
            f"Unit value: {unit_values.get(driver_type, 'N/A')}",
            f"Attribution level: {attribution.value} ({self._get_attribution_pct(attribution)})",
            f"Risk adjustment applied based on {self._summarize_risks(risk)}",
            "Bootstrap CI from 1,000 Monte Carlo simulations",
        ]

        # Add metric-specific assumptions
        if metric in ["trx", "nrx"]:
            assumptions.append("Market conditions assumed stable")
        elif metric == "market_share":
            assumptions.append("Market size assumed constant")
        elif metric == "conversion_rate":
            assumptions.append("Patient journey optimization feasible")

        return assumptions

    def _get_attribution_pct(self, level: AttributionLevel) -> str:
        """Get attribution percentage string."""
        pcts = {
            AttributionLevel.FULL: "100%",
            AttributionLevel.PARTIAL: "65%",
            AttributionLevel.SHARED: "35%",
            AttributionLevel.MINIMAL: "10%",
        }
        return pcts.get(level, "N/A")

    def _summarize_risks(self, risk: RiskAssessment) -> str:
        """Summarize risk factors."""
        high_count = sum(
            1
            for r in [
                risk.technical_complexity,
                risk.organizational_change,
                risk.data_dependencies,
                risk.timeline_uncertainty,
            ]
            if r == RiskLevel.HIGH
        )
        if high_count >= 2:
            return "multiple high-risk factors"
        elif high_count == 1:
            return "one high-risk factor"
        else:
            return "low-to-medium risk factors"

    def _calculate_legacy_confidence(self, gap: PerformanceGap) -> float:
        """Calculate legacy confidence score for backwards compatibility.

        Args:
            gap: Performance gap

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.7  # Base confidence

        # Gap size factor
        gap_size = abs(gap["gap_size"])
        if gap_size > 100:
            confidence += 0.1
        elif gap_size < 10:
            confidence -= 0.1

        # Gap percentage factor
        gap_pct = abs(gap["gap_percentage"])
        if 10 <= gap_pct <= 50:
            confidence += 0.1
        elif gap_pct > 100:
            confidence -= 0.2

        # Gap type factor
        gap_type_confidence = {
            "vs_target": 0.1,
            "vs_benchmark": 0.05,
            "vs_potential": 0.0,
            "temporal": -0.05,
        }
        confidence += gap_type_confidence.get(gap["gap_type"], 0.0)

        return max(0.0, min(1.0, confidence))
