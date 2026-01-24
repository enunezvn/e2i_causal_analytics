"""Formatter Node for Gap Analyzer Agent.

This node generates the final output:
- Executive summary
- Key insights
- Formatted opportunities
- Performance metadata

Additionally handles:
- Memory contribution (episodic + working memory)
- DSPy training signal collection
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

from ..state import GapAnalyzerState, PrioritizedOpportunity

logger = logging.getLogger(__name__)


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
                detection_latency + roi_latency + prioritization_latency + formatting_latency_ms
            )

            # Build result for memory contribution
            result = {
                "executive_summary": executive_summary,
                "key_insights": key_insights,
                "prioritized_opportunities": prioritized_opportunities,
                "quick_wins": quick_wins,
                "strategic_bets": strategic_bets,
                "total_addressable_value": total_addressable_value,
                "confidence": state.get("prioritization_confidence", 0.8),
            }

            # Contribute to memory (async, non-blocking on failure)
            memory_contribution = await self._contribute_to_memory(result, state)

            # Collect DSPy training signal (async, non-blocking on failure)
            await self._collect_dspy_signal(state, result, total_latency_ms)

            return {
                "executive_summary": executive_summary,
                "key_insights": key_insights,
                "total_latency_ms": total_latency_ms,
                "memory_contribution": memory_contribution,
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

    async def _contribute_to_memory(
        self,
        result: Dict[str, Any],
        state: GapAnalyzerState,
    ) -> Dict[str, int]:
        """Contribute gap analysis to memory systems.

        Non-blocking: failures are logged but don't affect workflow.

        Args:
            result: Formatted gap analysis result
            state: Current workflow state

        Returns:
            Memory contribution counts
        """
        try:
            from ..memory_hooks import contribute_to_memory

            memory_counts = await contribute_to_memory(
                result=result,
                state=dict(state),
                session_id=state.get("session_id"),
                region=state.get("region"),
            )
            logger.info(
                f"Memory contribution: episodic={memory_counts.get('episodic_stored', 0)}, "
                f"working={memory_counts.get('working_cached', 0)}"
            )
            return memory_counts
        except Exception as e:
            logger.warning(f"Memory contribution failed (non-fatal): {e}")
            return {"episodic_stored": 0, "working_cached": 0}

    async def _collect_dspy_signal(
        self,
        state: GapAnalyzerState,
        result: Dict[str, Any],
        total_latency_ms: float,
    ) -> None:
        """Collect DSPy training signal for feedback_learner.

        Non-blocking: failures are logged but don't affect workflow.

        Args:
            state: Current workflow state
            result: Formatted gap analysis result
            total_latency_ms: Total workflow latency
        """
        try:
            from ..dspy_integration import get_gap_analyzer_signal_collector

            collector = get_gap_analyzer_signal_collector()

            # Initialize signal
            signal = collector.collect_analysis_signal(
                session_id=state.get("session_id", ""),
                query=state.get("query", ""),
                brand=state.get("brand", ""),
                metrics_analyzed=state.get("metrics", []),
                segments_analyzed=state.get("segments_analyzed", 0),
            )

            # Update with detection phase data
            gaps_detected = state.get("gaps_detected", [])
            gap_types = list({g.get("gap_type", "unknown") for g in gaps_detected})
            collector.update_detection(
                signal=signal,
                gaps_detected_count=len(gaps_detected),
                total_gap_value=state.get("total_gap_value", 0.0),
                gap_types=gap_types,
                detection_latency_ms=state.get("detection_latency_ms", 0.0),
            )

            # Update with ROI phase data
            prioritized_opps = state.get("prioritized_opportunities", [])
            roi_estimates = [o.get("roi_estimate", {}) for o in prioritized_opps]
            avg_roi = 0.0
            high_roi_count = 0
            if roi_estimates:
                rois = [r.get("expected_roi", 0) for r in roi_estimates if r.get("expected_roi")]
                if rois:
                    avg_roi = sum(rois) / len(rois)
                    high_roi_count = sum(1 for r in rois if r > 2.0)

            collector.update_roi(
                signal=signal,
                roi_estimates_count=len(roi_estimates),
                total_addressable_value=state.get("total_addressable_value", 0.0),
                avg_expected_roi=avg_roi,
                high_roi_count=high_roi_count,
                roi_latency_ms=state.get("roi_latency_ms", 0.0),
            )

            # Update with prioritization phase data
            quick_wins = state.get("quick_wins", [])
            strategic_bets = state.get("strategic_bets", [])
            key_insights = result.get("key_insights", [])
            executive_summary = result.get("executive_summary", "")

            collector.update_prioritization(
                signal=signal,
                quick_wins_count=len(quick_wins),
                strategic_bets_count=len(strategic_bets),
                prioritization_confidence=state.get("prioritization_confidence", 0.8),
                executive_summary_length=len(executive_summary),
                key_insights_count=len(key_insights),
                actionable_recommendations=len(quick_wins) + len(strategic_bets),
                total_latency_ms=total_latency_ms,
            )

            logger.debug(
                f"DSPy signal collected: reward={signal.compute_reward():.3f}, "
                f"gaps={len(gaps_detected)}, quick_wins={len(quick_wins)}"
            )
        except Exception as e:
            logger.warning(f"DSPy signal collection failed (non-fatal): {e}")

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
        segment_distribution = self._analyze_segment_distribution(prioritized_opportunities)
        if segment_distribution:
            top_segment, count = segment_distribution[0]
            insights.append(
                f"Performance gaps concentrated in {top_segment} "
                f"({count}/{len(prioritized_opportunities)} opportunities)"
            )

        # Insight 3: Metric patterns
        metric_distribution = self._analyze_metric_distribution(prioritized_opportunities)
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
            insights.append(
                "No quick wins identified; focus on medium/high difficulty opportunities"
            )

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
        sorted_segments = sorted(segment_counts.items(), key=lambda x: x[1], reverse=True)

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
        sorted_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)

        return sorted_metrics
