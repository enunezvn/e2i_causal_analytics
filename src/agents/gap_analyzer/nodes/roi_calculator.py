"""ROI Calculator Node for Gap Analyzer Agent.

This node calculates ROI estimates for closing performance gaps using
pharmaceutical-specific economics and assumptions.

ROI Formula:
- Revenue Impact = gap_size × metric_multiplier
- Cost to Close = gap_size × intervention_cost
- ROI = (revenue_impact - cost_to_close) / cost_to_close
- Payback Period = cost_to_close / (revenue_impact / 12)

Economics:
- revenue_per_trx: $500
- cost_per_hcp_visit: $150
- cost_per_sample: $25
- conversion_rate_improvement: 5%
"""

import time
from typing import Dict, List, Any
import numpy as np

from ..state import GapAnalyzerState, PerformanceGap, ROIEstimate


class ROICalculatorNode:
    """Calculate ROI estimates for performance gaps.

    Uses pharmaceutical-specific economic assumptions.
    """

    # Default economic assumptions
    DEFAULT_ASSUMPTIONS = {
        "revenue_per_trx": 500.0,  # Revenue per prescription (USD)
        "cost_per_hcp_visit": 150.0,  # Cost per HCP visit (USD)
        "cost_per_sample": 25.0,  # Cost per sample (USD)
        "conversion_rate_improvement": 0.05,  # 5% improvement
        "time_to_impact_months": 3,  # Months to see results
        "annual_multiplier": 1.0,  # Assume annual impact
    }

    # Metric-specific multipliers for revenue calculation
    METRIC_MULTIPLIERS = {
        "trx": 500.0,  # $500 per TRx
        "nrx": 600.0,  # $600 per NRx (higher margin)
        "market_share": 10000.0,  # $10k per 1% market share point
        "conversion_rate": 50000.0,  # $50k per 1% conversion improvement
        "hcp_engagement_score": 2000.0,  # $2k per point improvement
    }

    # Intervention costs by metric type
    INTERVENTION_COSTS = {
        "trx": 100.0,  # $100 per TRx gap (sampling, visits)
        "nrx": 150.0,  # $150 per NRx gap (targeting, messaging)
        "market_share": 5000.0,  # $5k per 1% market share gap (campaigns)
        "conversion_rate": 20000.0,  # $20k per 1% conversion gap (programs)
        "hcp_engagement_score": 1000.0,  # $1k per point gap (engagement)
    }

    def __init__(self):
        """Initialize ROI calculator."""
        self.assumptions = self.DEFAULT_ASSUMPTIONS.copy()

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

            # Calculate ROI for each gap
            roi_estimates: List[ROIEstimate] = []

            for gap in gaps_detected:
                roi_estimate = self._calculate_roi(gap)
                roi_estimates.append(roi_estimate)

            # Calculate total addressable value
            total_addressable_value = sum(
                est["estimated_revenue_impact"] for est in roi_estimates
            )

            roi_latency_ms = int((time.time() - start_time) * 1000)

            return {
                "roi_estimates": roi_estimates,
                "total_addressable_value": total_addressable_value,
                "roi_latency_ms": roi_latency_ms,
                "status": "prioritizing",
            }

        except Exception as e:
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
        """Calculate ROI estimate for a single gap.

        Args:
            gap: Performance gap to analyze

        Returns:
            ROI estimate with revenue impact, cost, ROI ratio, payback period
        """
        metric = gap["metric"]
        gap_size = abs(gap["gap_size"])

        # Get metric-specific multipliers
        metric_multiplier = self._get_metric_multiplier(metric)
        intervention_cost_per_unit = self._get_intervention_cost(metric)

        # Calculate revenue impact (annual)
        estimated_revenue_impact = gap_size * metric_multiplier

        # Calculate cost to close gap
        estimated_cost_to_close = gap_size * intervention_cost_per_unit

        # Calculate ROI ratio
        expected_roi = (
            (estimated_revenue_impact - estimated_cost_to_close)
            / estimated_cost_to_close
            if estimated_cost_to_close > 0
            else float("inf")
        )

        # Calculate payback period (months)
        monthly_revenue = estimated_revenue_impact / 12
        payback_period_months = (
            int(estimated_cost_to_close / monthly_revenue)
            if monthly_revenue > 0
            else 24
        )
        payback_period_months = min(payback_period_months, 24)  # Cap at 24 months

        # Calculate confidence based on gap characteristics
        confidence = self._calculate_confidence(gap)

        # Document assumptions
        assumptions = self._get_assumptions(metric)

        roi_estimate: ROIEstimate = {
            "gap_id": gap["gap_id"],
            "estimated_revenue_impact": estimated_revenue_impact,
            "estimated_cost_to_close": estimated_cost_to_close,
            "expected_roi": expected_roi,
            "payback_period_months": payback_period_months,
            "confidence": confidence,
            "assumptions": assumptions,
        }

        return roi_estimate

    def _get_metric_multiplier(self, metric: str) -> float:
        """Get revenue multiplier for a metric.

        Args:
            metric: KPI name

        Returns:
            Revenue multiplier (USD per unit)
        """
        return self.METRIC_MULTIPLIERS.get(metric, 1000.0)  # Default $1k

    def _get_intervention_cost(self, metric: str) -> float:
        """Get intervention cost per unit for a metric.

        Args:
            metric: KPI name

        Returns:
            Cost per unit (USD)
        """
        return self.INTERVENTION_COSTS.get(metric, 500.0)  # Default $500

    def _calculate_confidence(self, gap: PerformanceGap) -> float:
        """Calculate confidence in ROI estimate.

        Confidence factors:
        - Gap size (larger = more reliable)
        - Gap percentage (moderate gaps more reliable than extreme)
        - Gap type (vs_target > vs_benchmark > vs_potential > temporal)

        Args:
            gap: Performance gap

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.7  # Base confidence

        # Gap size factor (larger gaps = higher confidence)
        gap_size = abs(gap["gap_size"])
        if gap_size > 100:
            confidence += 0.1
        elif gap_size < 10:
            confidence -= 0.1

        # Gap percentage factor (moderate gaps more reliable)
        gap_pct = abs(gap["gap_percentage"])
        if 10 <= gap_pct <= 50:
            confidence += 0.1
        elif gap_pct > 100:
            confidence -= 0.2  # Very large gaps may be unreliable

        # Gap type factor
        gap_type_confidence = {
            "vs_target": 0.1,  # Targets are well-defined
            "vs_benchmark": 0.05,  # Peer data may vary
            "vs_potential": 0.0,  # Top decile may not be achievable
            "temporal": -0.05,  # Prior period may not repeat
        }
        confidence += gap_type_confidence.get(gap["gap_type"], 0.0)

        # Clip to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _get_assumptions(self, metric: str) -> List[str]:
        """Get documented assumptions for ROI calculation.

        Args:
            metric: KPI name

        Returns:
            List of assumption statements
        """
        base_assumptions = [
            f"Revenue per TRx: ${self.assumptions['revenue_per_trx']:.0f}",
            f"Cost per HCP visit: ${self.assumptions['cost_per_hcp_visit']:.0f}",
            f"Time to impact: {self.assumptions['time_to_impact_months']} months",
        ]

        metric_specific = {
            "trx": [
                "Assumes direct correlation between gap closure and TRx volume",
                "Market conditions remain stable",
            ],
            "nrx": [
                "Assumes new prescriber acquisition is sustainable",
                "No significant competitive actions",
            ],
            "market_share": [
                "Assumes market size remains constant",
                "Competitor actions accounted for in benchmark",
            ],
            "conversion_rate": [
                "Assumes conversion improvements are sustainable",
                "Patient journey optimization is feasible",
            ],
            "hcp_engagement_score": [
                "Assumes engagement correlates with prescribing behavior",
                "Multichannel touchpoints are effective",
            ],
        }

        return base_assumptions + metric_specific.get(metric, [])
