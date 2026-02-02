"""Policy Learner Node for Heterogeneous Optimizer Agent.

This node learns optimal treatment allocation policy based on CATE estimates.
Uses CATE to recommend allocation changes.
"""

import logging
import time
from typing import Any, Dict, List

from ..state import HeterogeneousOptimizerState, PolicyRecommendation

logger = logging.getLogger(__name__)


class PolicyLearnerNode:
    """Learn optimal treatment allocation policy.

    Recommends treatment rate adjustments based on CATE estimates:
    - High responders (CATE >= 1.5x ATE): Increase treatment
    - Low responders (CATE <= 0.5x ATE): Decrease treatment
    - Average responders: Maintain current rate
    """

    def __init__(self):
        self.min_cate_for_treatment = 0.01  # Minimum CATE to recommend treatment

    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        """Execute policy learning."""
        start_time = time.time()
        logger.info(
            "Starting policy learning",
            extra={
                "node": "policy_learner",
                "segment_count": len(state.get("cate_by_segment", {})),
                "high_responders": len(state.get("high_responders", [])),
                "low_responders": len(state.get("low_responders", [])),
            },
        )

        if state.get("status") == "failed":
            logger.warning("Skipping policy learning - previous node failed")
            return state

        try:
            cate_by_segment = state["cate_by_segment"]
            high_responders = state["high_responders"]
            low_responders = state["low_responders"]
            ate = state["overall_ate"]

            # Generate policy recommendations
            recommendations = []

            for _segment_var, results in cate_by_segment.items():
                for result in results:
                    rec = self._generate_recommendation(result, ate)
                    if rec:
                        recommendations.append(rec)

            # Sort by expected incremental outcome
            recommendations.sort(key=lambda x: x["expected_incremental_outcome"], reverse=True)

            # Calculate total expected lift if policy is implemented
            total_lift = sum(r["expected_incremental_outcome"] for r in recommendations)

            # Generate summary
            summary = self._generate_allocation_summary(
                recommendations, high_responders, low_responders, ate
            )

            total_time = (
                state.get("estimation_latency_ms", 0)
                + state.get("analysis_latency_ms", 0)
                + int((time.time() - start_time) * 1000)
            )

            # Count increase/decrease recommendations
            increase_count = sum(
                1
                for r in recommendations
                if r["recommended_treatment_rate"] > r["current_treatment_rate"]
            )
            decrease_count = sum(
                1
                for r in recommendations
                if r["recommended_treatment_rate"] < r["current_treatment_rate"]
            )

            logger.info(
                "Policy learning complete",
                extra={
                    "node": "policy_learner",
                    "recommendation_count": len(recommendations),
                    "increase_recommendations": increase_count,
                    "decrease_recommendations": decrease_count,
                    "expected_total_lift": total_lift,
                    "total_latency_ms": total_time,
                },
            )

            return {
                **state,
                "policy_recommendations": recommendations[:20],  # Top 20
                "expected_total_lift": total_lift,
                "optimal_allocation_summary": summary,
                "total_latency_ms": total_time,
                "status": "completed",
            }

        except Exception as e:
            logger.error(
                "Policy learning failed",
                extra={"node": "policy_learner", "error": str(e)},
                exc_info=True,
            )
            return {
                **state,
                "errors": [{"node": "policy_learner", "error": str(e)}],
                "status": "failed",
            }

    def _generate_recommendation(self, result: Dict[str, Any], ate: float) -> PolicyRecommendation:
        """Generate policy recommendation for a segment.

        Args:
            result: CATE result dictionary
            ate: Overall average treatment effect

        Returns:
            Policy recommendation
        """

        cate = result["cate_estimate"]
        segment_name = result["segment_name"]
        segment_value = result["segment_value"]
        sample_size = result["sample_size"]

        # Determine recommended treatment rate change
        current_rate = 0.5  # Assume current 50% coverage

        if cate <= 0 or cate < self.min_cate_for_treatment:
            # Negative or very low responder - minimize treatment
            recommended_rate = 0.1
        elif cate >= ate * 1.5 and ate > 0:
            # High responder - increase treatment (only if ate > 0)
            recommended_rate = min(0.9, current_rate + 0.2)
        elif cate <= ate * 0.5 and ate > 0:
            # Low responder - decrease treatment (only if ate > 0)
            recommended_rate = max(0.1, current_rate - 0.2)
        else:
            # Average responder - maintain
            recommended_rate = 0.5

        # Calculate expected incremental outcome
        rate_change = recommended_rate - current_rate
        expected_lift = rate_change * cate * sample_size

        # Confidence based on sample size and significance
        confidence = min(0.9, 0.5 + (sample_size / 1000) * 0.3)
        if result.get("statistical_significance"):
            confidence = min(confidence + 0.1, 0.95)

        return PolicyRecommendation(
            segment=f"{segment_name}={segment_value}",
            current_treatment_rate=current_rate,
            recommended_treatment_rate=recommended_rate,
            expected_incremental_outcome=expected_lift,
            confidence=confidence,
        )

    def _generate_allocation_summary(
        self,
        recommendations: List[PolicyRecommendation],
        high_responders: List,
        low_responders: List,
        ate: float,
    ) -> str:
        """Generate natural language summary of optimal allocation.

        Args:
            recommendations: Policy recommendations
            high_responders: High responder segments
            low_responders: Low responder segments
            ate: Overall ATE

        Returns:
            Summary string
        """

        increase_recs = [
            r
            for r in recommendations
            if r["recommended_treatment_rate"] > r["current_treatment_rate"]
        ]
        decrease_recs = [
            r
            for r in recommendations
            if r["recommended_treatment_rate"] < r["current_treatment_rate"]
        ]

        summary_parts = [
            f"Treatment effect heterogeneity detected (ATE: {ate:.3f}).",
            f"Identified {len(high_responders)} high-responder segments and {len(low_responders)} low-responder segments.",
        ]

        if increase_recs:
            top_increase = increase_recs[0]
            summary_parts.append(
                f"Recommend increasing treatment in {len(increase_recs)} segments, "
                f"starting with {top_increase['segment']}."
            )

        if decrease_recs:
            summary_parts.append(
                f"Recommend decreasing treatment in {len(decrease_recs)} segments to optimize resource allocation."
            )

        total_lift = sum(r["expected_incremental_outcome"] for r in recommendations)
        summary_parts.append(
            f"Expected total outcome lift from reallocation: {total_lift:.1f} units."
        )

        return " ".join(summary_parts)
