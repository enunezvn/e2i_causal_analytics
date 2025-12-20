"""Formatter Node for Gap Analyzer Agent.

This node generates the final output:
- Executive summary
- Key insights
- Formatted opportunities
- Performance metadata
"""

import time
from typing import Dict, List, Any, Optional

from ..state import GapAnalyzerState, PrioritizedOpportunity


class FormatterNode:
    """Format gap analyzer output for consumption."""

    def __init__(self):
        """Initialize formatter."""
        pass

    async def execute(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Execute formatting workflow.

        Args:
            state: Complete gap analyzer state

        Returns:
            Updated state with executive_summary and key_insights
        """
        start_time = time.time()

        try:
            prioritized_opportunities = state.get("prioritized_opportunities", [])
            quick_wins = state.get("quick_wins", [])
            strategic_bets = state.get("strategic_bets", [])
            total_addressable_value = state.get("total_addressable_value", 0.0)
            total_gap_value = state.get("total_gap_value", 0.0)
            segments_analyzed = state.get("segments_analyzed", 0)

            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                prioritized_opportunities=prioritized_opportunities,
                quick_wins=quick_wins,
                strategic_bets=strategic_bets,
                total_addressable_value=total_addressable_value,
                total_gap_value=total_gap_value,
                segments_analyzed=segments_analyzed,
            )

            # Extract key insights
            key_insights = self._extract_key_insights(
                prioritized_opportunities=prioritized_opportunities,
                quick_wins=quick_wins,
                strategic_bets=strategic_bets,
            )

            # Calculate total latency
            detection_latency = state.get("detection_latency_ms", 0)
            roi_latency = state.get("roi_latency_ms", 0)
            prioritization_latency = state.get("prioritization_latency_ms", 0)
            formatting_latency_ms = int((time.time() - start_time) * 1000)

            total_latency_ms = (
                detection_latency
                + roi_latency
                + prioritization_latency
                + formatting_latency_ms
            )

            return {
                "executive_summary": executive_summary,
                "key_insights": key_insights,
                "total_latency_ms": total_latency_ms,
                "status": "completed",
            }

        except Exception as e:
            formatting_latency_ms = int((time.time() - start_time) * 1000)
            return {
                "errors": [
                    {
                        "node": "formatter",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                ],
                "total_latency_ms": formatting_latency_ms,
                "status": "failed",
            }

    def _generate_executive_summary(
        self,
        prioritized_opportunities: List[PrioritizedOpportunity],
        quick_wins: List[PrioritizedOpportunity],
        strategic_bets: List[PrioritizedOpportunity],
        total_addressable_value: float,
        total_gap_value: float,
        segments_analyzed: int,
    ) -> str:
        """Generate executive summary.

        Args:
            prioritized_opportunities: All prioritized opportunities
            quick_wins: Quick win opportunities
            strategic_bets: Strategic bet opportunities
            total_addressable_value: Total potential revenue
            total_gap_value: Total gap size
            segments_analyzed: Number of segments analyzed

        Returns:
            Executive summary text
        """
        if not prioritized_opportunities:
            return (
                f"Analysis complete. No significant performance gaps identified "
                f"across {segments_analyzed} segments that meet the minimum threshold."
            )

        num_opportunities = len(prioritized_opportunities)
        num_quick_wins = len(quick_wins)
        num_strategic_bets = len(strategic_bets)

        # Top opportunity details
        top_opp = prioritized_opportunities[0]
        top_metric = top_opp["gap"]["metric"]
        top_segment = top_opp["gap"]["segment_value"]
        top_roi = top_opp["roi_estimate"]["expected_roi"]
        top_revenue = top_opp["roi_estimate"]["estimated_revenue_impact"]

        summary = (
            f"Identified {num_opportunities} ROI opportunities across {segments_analyzed} segments "
            f"with total addressable value of ${total_addressable_value:,.0f}. "
        )

        if num_quick_wins > 0:
            summary += (
                f"Found {num_quick_wins} quick wins (low difficulty, high ROI) "
                f"for immediate action. "
            )

        if num_strategic_bets > 0:
            summary += (
                f"Identified {num_strategic_bets} strategic bets (high impact, high investment) "
                f"for long-term growth. "
            )

        summary += (
            f"Top opportunity: Close {top_metric} gap in {top_segment} "
            f"for ${top_revenue:,.0f} annual impact at {top_roi:.1f}x ROI."
        )

        return summary

    def _extract_key_insights(
        self,
        prioritized_opportunities: List[PrioritizedOpportunity],
        quick_wins: List[PrioritizedOpportunity],
        strategic_bets: List[PrioritizedOpportunity],
    ) -> List[str]:
        """Extract key insights from opportunities.

        Args:
            prioritized_opportunities: All prioritized opportunities
            quick_wins: Quick win opportunities
            strategic_bets: Strategic bet opportunities

        Returns:
            List of 3-5 key insights
        """
        insights = []

        if not prioritized_opportunities:
            insights.append("No significant performance gaps detected above threshold")
            return insights

        # Insight 1: Top opportunity
        top_opp = prioritized_opportunities[0]
        insights.append(
            f"Highest ROI opportunity: {top_opp['gap']['metric']} in "
            f"{top_opp['gap']['segment_value']} ({top_opp['gap']['segment']}) "
            f"at {top_opp['roi_estimate']['expected_roi']:.1f}x ROI"
        )

        # Insight 2: Segment concentration
        segment_distribution = self._analyze_segment_distribution(
            prioritized_opportunities
        )
        if segment_distribution:
            top_segment, count = segment_distribution[0]
            insights.append(
                f"Performance gaps concentrated in {top_segment} "
                f"({count}/{len(prioritized_opportunities)} opportunities)"
            )

        # Insight 3: Metric patterns
        metric_distribution = self._analyze_metric_distribution(
            prioritized_opportunities
        )
        if metric_distribution:
            top_metric, count = metric_distribution[0]
            insights.append(
                f"Primary gap type: {top_metric} "
                f"({count}/{len(prioritized_opportunities)} opportunities)"
            )

        # Insight 4: Quick wins availability
        if quick_wins:
            total_qw_revenue = sum(
                qw["roi_estimate"]["estimated_revenue_impact"] for qw in quick_wins
            )
            insights.append(
                f"{len(quick_wins)} quick wins available "
                f"with ${total_qw_revenue:,.0f} total potential impact"
            )
        else:
            insights.append("No quick wins identified; focus on medium/high difficulty opportunities")

        # Insight 5: Strategic focus
        if strategic_bets:
            total_sb_revenue = sum(
                sb["roi_estimate"]["estimated_revenue_impact"] for sb in strategic_bets
            )
            insights.append(
                f"{len(strategic_bets)} strategic bets offer "
                f"${total_sb_revenue:,.0f} long-term value with significant investment"
            )

        return insights[:5]  # Limit to 5 insights

    def _analyze_segment_distribution(
        self, opportunities: List[PrioritizedOpportunity]
    ) -> List[tuple[str, int]]:
        """Analyze distribution of opportunities across segments.

        Args:
            opportunities: Prioritized opportunities

        Returns:
            List of (segment_value, count) sorted by count descending
        """
        segment_counts: Dict[str, int] = {}

        for opp in opportunities:
            segment_value = opp["gap"]["segment_value"]
            segment_counts[segment_value] = segment_counts.get(segment_value, 0) + 1

        # Sort by count descending
        sorted_segments = sorted(
            segment_counts.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_segments

    def _analyze_metric_distribution(
        self, opportunities: List[PrioritizedOpportunity]
    ) -> List[tuple[str, int]]:
        """Analyze distribution of opportunities across metrics.

        Args:
            opportunities: Prioritized opportunities

        Returns:
            List of (metric, count) sorted by count descending
        """
        metric_counts: Dict[str, int] = {}

        for opp in opportunities:
            metric = opp["gap"]["metric"]
            metric_counts[metric] = metric_counts.get(metric, 0) + 1

        # Sort by count descending
        sorted_metrics = sorted(
            metric_counts.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_metrics
