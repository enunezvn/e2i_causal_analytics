"""Segment Analyzer Node for Heterogeneous Optimizer Agent.

This node analyzes segments to identify high/low responders based on CATE estimates.
Pure computation - no LLM needed.
"""

import time
from typing import Any, Dict, List

from ..state import CATEResult, HeterogeneousOptimizerState, SegmentProfile


class SegmentAnalyzerNode:
    """Analyze segments to identify high/low responders.

    High responders: CATE >= 1.5x ATE
    Low responders: CATE <= 0.5x ATE
    """

    def __init__(self):
        self.high_responder_threshold = 1.5  # 1.5x ATE
        self.low_responder_threshold = 0.5  # 0.5x ATE

    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        """Execute segment analysis."""
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        try:
            ate = state["overall_ate"]
            cate_by_segment = state["cate_by_segment"]
            top_count = state.get("top_segments_count", 10)

            # Flatten all segment results
            all_segments = []
            total_size = 0

            for segment_var, results in cate_by_segment.items():
                for result in results:
                    total_size += result["sample_size"]
                    all_segments.append({"segment_var": segment_var, "result": result})

            # Identify high responders
            high_responders = self._identify_responders(
                all_segments, ate, total_size, "high", self.high_responder_threshold
            )[:top_count]

            # Identify low responders
            low_responders = self._identify_responders(
                all_segments, ate, total_size, "low", self.low_responder_threshold
            )[:top_count]

            # Create segment comparison
            comparison = self._create_comparison(high_responders, low_responders, ate)

            analysis_time = int((time.time() - start_time) * 1000)

            return {
                **state,
                "high_responders": high_responders,
                "low_responders": low_responders,
                "segment_comparison": comparison,
                "analysis_latency_ms": analysis_time,
                "status": "optimizing",
            }

        except Exception as e:
            return {
                **state,
                "errors": [{"node": "segment_analyzer", "error": str(e)}],
                "status": "failed",
            }

    def _identify_responders(
        self,
        all_segments: List[Dict],
        ate: float,
        total_size: int,
        responder_type: str,
        threshold: float,
    ) -> List[SegmentProfile]:
        """Identify high or low responder segments.

        Args:
            all_segments: All segment results
            ate: Overall average treatment effect
            total_size: Total sample size
            responder_type: 'high' or 'low'
            threshold: Multiplier for ATE threshold

        Returns:
            List of segment profiles matching criteria
        """

        profiles = []

        for seg in all_segments:
            result = seg["result"]
            cate = result["cate_estimate"]

            # Determine if segment qualifies
            if responder_type == "high":
                qualifies = ate > 0 and cate >= ate * threshold
            else:
                qualifies = ate > 0 and cate <= ate * threshold

            if not qualifies:
                continue

            profile = SegmentProfile(
                segment_id=f"{seg['segment_var']}_{result['segment_value']}",
                responder_type=responder_type,
                cate_estimate=cate,
                defining_features=[
                    {
                        "variable": seg["segment_var"],
                        "value": result["segment_value"],
                        "effect_size": cate / ate if ate != 0 else 0,
                    }
                ],
                size=result["sample_size"],
                size_percentage=result["sample_size"] / total_size * 100 if total_size > 0 else 0,
                recommendation=self._generate_recommendation(
                    seg["segment_var"], result, responder_type
                ),
            )
            profiles.append(profile)

        # Sort by CATE (descending for high, ascending for low)
        reverse = responder_type == "high"
        profiles.sort(key=lambda x: x["cate_estimate"], reverse=reverse)

        return profiles

    def _generate_recommendation(
        self, segment_var: str, result: CATEResult, responder_type: str
    ) -> str:
        """Generate action recommendation for segment.

        Args:
            segment_var: Segment variable name
            result: CATE result for this segment
            responder_type: 'high' or 'low'

        Returns:
            Action recommendation string
        """

        segment_value = result["segment_value"]
        cate = result["cate_estimate"]

        if responder_type == "high":
            return f"Prioritize treatment for {segment_var}={segment_value} (CATE: {cate:.3f}). High response expected."
        else:
            return f"De-prioritize treatment for {segment_var}={segment_value} (CATE: {cate:.3f}). Consider alternative interventions."

    def _create_comparison(
        self,
        high_responders: List[SegmentProfile],
        low_responders: List[SegmentProfile],
        ate: float,
    ) -> Dict[str, Any]:
        """Create comparison summary between high and low responders.

        Args:
            high_responders: High responder segments
            low_responders: Low responder segments
            ate: Overall ATE

        Returns:
            Comparison dictionary
        """

        high_avg_cate = (
            sum(h["cate_estimate"] for h in high_responders) / len(high_responders)
            if high_responders
            else 0
        )
        low_avg_cate = (
            sum(l["cate_estimate"] for l in low_responders) / len(low_responders)
            if low_responders
            else 0
        )

        return {
            "overall_ate": ate,
            "high_responder_avg_cate": high_avg_cate,
            "low_responder_avg_cate": low_avg_cate,
            "effect_ratio": high_avg_cate / low_avg_cate if low_avg_cate != 0 else float("inf"),
            "high_responder_count": len(high_responders),
            "low_responder_count": len(low_responders),
        }
