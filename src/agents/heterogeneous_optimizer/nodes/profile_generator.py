"""Profile Generator Node for Heterogeneous Optimizer Agent.

This node generates visualization data and executive summaries.
Pure computation - no LLM needed.

Additionally handles:
- DSPy training signal collection and routing
"""

import logging
import time
from typing import Any, Dict, List

from ..state import HeterogeneousOptimizerState

logger = logging.getLogger(__name__)


class ProfileGeneratorNode:
    """Generate visualization data and executive summaries."""

    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        """Execute profile generation."""
        start_time = time.time()
        logger.info(
            "Starting profile generation",
            extra={
                "node": "profile_generator",
                "high_responders": len(state.get("high_responders", [])),
                "low_responders": len(state.get("low_responders", [])),
                "policy_recommendations": len(state.get("policy_recommendations", [])),
            },
        )

        if state.get("status") == "failed":
            logger.warning("Skipping profile generation - previous node failed")
            return state

        try:
            # Generate CATE plot data
            cate_plot_data = self._generate_cate_plot_data(state)

            # Generate segment grid data
            segment_grid_data = self._generate_segment_grid_data(state)

            # Generate executive summary
            executive_summary = self._generate_executive_summary(state)

            # Generate key insights
            key_insights = self._generate_key_insights(state)

            generation_time = int((time.time() - start_time) * 1000)

            logger.info(
                "Profile generation complete",
                extra={
                    "node": "profile_generator",
                    "segment_plot_count": len(cate_plot_data.get("segments", [])),
                    "insight_count": len(key_insights),
                    "latency_ms": generation_time,
                },
            )

            # Collect and route DSPy training signal (non-blocking)
            await self._collect_dspy_signal(
                state=state,
                executive_summary=executive_summary,
                key_insights=key_insights,
                generation_time=generation_time,
            )

            return {
                **state,
                "cate_plot_data": cate_plot_data,
                "segment_grid_data": segment_grid_data,
                "executive_summary": executive_summary,
                "key_insights": key_insights,
                "status": "completed",
            }

        except Exception as e:
            logger.error(
                "Profile generation failed",
                extra={"node": "profile_generator", "error": str(e)},
                exc_info=True,
            )
            return {
                **state,
                "errors": [{"node": "profile_generator", "error": str(e)}],
                "status": "failed",
            }

    def _generate_cate_plot_data(self, state: HeterogeneousOptimizerState) -> Dict[str, Any]:
        """Generate data for CATE visualization plots.

        Returns data structure suitable for plotting CATE estimates by segment.
        """

        cate_by_segment = state.get("cate_by_segment", {})
        overall_ate = state.get("overall_ate", 0)

        plot_data = {
            "overall_ate": overall_ate,
            "segments": [],
        }

        for segment_var, results in cate_by_segment.items():
            for result in results:
                plot_data["segments"].append(
                    {
                        "segment_var": segment_var,
                        "segment_value": result["segment_value"],
                        "cate": result["cate_estimate"],
                        "ci_lower": result["cate_ci_lower"],
                        "ci_upper": result["cate_ci_upper"],
                        "sample_size": result["sample_size"],
                        "significant": result["statistical_significance"],
                    }
                )

        # Sort by CATE for visualization
        plot_data["segments"].sort(key=lambda x: x["cate"], reverse=True)

        return plot_data

    def _generate_segment_grid_data(self, state: HeterogeneousOptimizerState) -> Dict[str, Any]:
        """Generate segment grid data for heatmap visualization.

        Returns data structure suitable for segment comparison heatmap.
        """

        high_responders = state.get("high_responders", [])
        low_responders = state.get("low_responders", [])
        segment_comparison = state.get("segment_comparison", {})

        grid_data = {
            "comparison_metrics": segment_comparison,
            "high_responder_segments": [
                {
                    "segment_id": h["segment_id"],
                    "cate": h["cate_estimate"],
                    "size_pct": h["size_percentage"],
                }
                for h in high_responders
            ],
            "low_responder_segments": [
                {
                    "segment_id": l["segment_id"],
                    "cate": l["cate_estimate"],
                    "size_pct": l["size_percentage"],
                }
                for l in low_responders
            ],
        }

        return grid_data

    def _generate_executive_summary(self, state: HeterogeneousOptimizerState) -> str:
        """Generate executive summary of heterogeneous optimization analysis."""

        overall_ate = state.get("overall_ate", 0)
        heterogeneity_score = state.get("heterogeneity_score", 0)
        high_responders = state.get("high_responders", [])
        low_responders = state.get("low_responders", [])
        expected_total_lift = state.get("expected_total_lift", 0)
        optimal_allocation_summary = state.get("optimal_allocation_summary", "")

        if not high_responders and not low_responders:
            return (
                f"Heterogeneous effect analysis complete. "
                f"Overall treatment effect: {overall_ate:.3f}. "
                f"Limited heterogeneity detected (score: {heterogeneity_score:.2f}). "
                f"Treatment effects are relatively uniform across segments."
            )

        summary_parts = [
            "Heterogeneous treatment effect analysis complete.",
            f"Overall treatment effect: {overall_ate:.3f}.",
            f"Heterogeneity score: {heterogeneity_score:.2f} (0=uniform, 1=highly heterogeneous).",
        ]

        if high_responders:
            top_high = high_responders[0]
            summary_parts.append(
                f"Top high-responder: {top_high['segment_id']} "
                f"(CATE: {top_high['cate_estimate']:.3f}, "
                f"{top_high['size_percentage']:.1f}% of population)."
            )

        if low_responders:
            top_low = low_responders[0]
            summary_parts.append(
                f"Top low-responder: {top_low['segment_id']} "
                f"(CATE: {top_low['cate_estimate']:.3f}, "
                f"{top_low['size_percentage']:.1f}% of population)."
            )

        if optimal_allocation_summary:
            summary_parts.append(optimal_allocation_summary)

        if expected_total_lift != 0:
            summary_parts.append(
                f"Implementing optimal allocation policy could yield "
                f"{abs(expected_total_lift):.1f} units of {'incremental' if expected_total_lift > 0 else 'avoided'} outcome."
            )

        return " ".join(summary_parts)

    def _generate_key_insights(self, state: HeterogeneousOptimizerState) -> List[str]:
        """Generate key insights from heterogeneous optimization analysis."""

        insights = []

        overall_ate = state.get("overall_ate", 0)
        heterogeneity_score = state.get("heterogeneity_score", 0)
        high_responders = state.get("high_responders", [])
        low_responders = state.get("low_responders", [])
        feature_importance = state.get("feature_importance", {})
        segment_comparison = state.get("segment_comparison", {})

        # Insight 1: Overall treatment effect
        if overall_ate > 0:
            insights.append(
                f"Treatment has positive overall effect (ATE: {overall_ate:.3f}), "
                f"but effect varies significantly across segments."
            )
        elif overall_ate < 0:
            insights.append(
                f"Treatment has negative overall effect (ATE: {overall_ate:.3f}). "
                f"Consider segment-specific interventions."
            )
        else:
            insights.append(
                f"Treatment has minimal overall effect (ATE: {overall_ate:.3f}). "
                f"Heterogeneity analysis reveals segment-specific opportunities."
            )

        # Insight 2: Heterogeneity level
        if heterogeneity_score > 0.7:
            insights.append(
                f"High treatment effect heterogeneity detected (score: {heterogeneity_score:.2f}). "
                f"Segment-specific strategies strongly recommended."
            )
        elif heterogeneity_score > 0.4:
            insights.append(
                f"Moderate treatment effect heterogeneity (score: {heterogeneity_score:.2f}). "
                f"Targeting high-responder segments can improve outcomes."
            )
        else:
            insights.append(
                f"Low treatment effect heterogeneity (score: {heterogeneity_score:.2f}). "
                f"Effects are relatively uniform across segments."
            )

        # Insight 3: High vs low responders
        if high_responders and low_responders:
            effect_ratio = segment_comparison.get("effect_ratio", 1)
            if effect_ratio > 3:
                insights.append(
                    f"High-responder segments show {effect_ratio:.1f}x stronger effects than low-responders. "
                    f"Resource reallocation could significantly improve efficiency."
                )

        # Insight 4: Feature importance
        if feature_importance:
            top_feature = max(feature_importance.items(), key=lambda x: x[1])
            insights.append(
                f"Treatment effect most strongly moderated by '{top_feature[0]}' "
                f"(importance: {top_feature[1]:.3f}). Use this for targeting."
            )

        # Insight 5: Segment-specific recommendations
        if high_responders:
            high_count = len(high_responders)
            insights.append(
                f"Prioritize treatment for {high_count} high-responder segments "
                f"to maximize outcome gains."
            )

        return insights[:5]  # Limit to top 5 insights

    async def _collect_dspy_signal(
        self,
        state: HeterogeneousOptimizerState,
        executive_summary: str,
        key_insights: List[str],
        generation_time: int,
    ) -> None:
        """Collect and route DSPy training signal to feedback_learner.

        Non-blocking: failures are logged but don't affect workflow.

        Args:
            state: Complete workflow state
            executive_summary: Generated executive summary
            key_insights: Generated key insights
            generation_time: Profile generation latency in ms
        """
        try:
            from src.agents.heterogeneous_optimizer.dspy_integration import (
                get_heterogeneous_optimizer_signal_collector,
            )
            from src.agents.tier2_signal_router import route_heterogeneous_optimizer_signal

            collector = get_heterogeneous_optimizer_signal_collector()

            # Initialize signal with input context
            signal = collector.collect_optimization_signal(
                session_id=state.get("session_id", ""),
                query=state.get("query", ""),
                treatment_var=state.get("treatment_var", ""),
                outcome_var=state.get("outcome_var", ""),
                segment_vars_count=len(state.get("segment_vars", [])),
                effect_modifiers_count=len(state.get("effect_modifiers", [])),
            )

            # Update with CATE estimation phase
            cate_by_segment = state.get("cate_by_segment", {})
            total_segments = sum(len(results) for results in cate_by_segment.values())
            significant_count = sum(
                1
                for results in cate_by_segment.values()
                for r in results
                if r.get("statistical_significance", False)
            )

            collector.update_cate_estimation(
                signal=signal,
                overall_ate=state.get("overall_ate", 0.0),
                heterogeneity_score=state.get("heterogeneity_score", 0.0),
                cate_segments_count=total_segments,
                significant_cate_count=significant_count,
                estimation_latency_ms=state.get("estimation_latency_ms", 0.0),
            )

            # Update with segment discovery phase
            high_responders = state.get("high_responders", [])
            low_responders = state.get("low_responders", [])

            # Calculate responder spread (difference between avg high and avg low CATE)
            responder_spread = 0.0
            if high_responders and low_responders:
                avg_high = sum(h.get("cate_estimate", 0) for h in high_responders) / len(
                    high_responders
                )
                avg_low = sum(l.get("cate_estimate", 0) for l in low_responders) / len(
                    low_responders
                )
                responder_spread = abs(avg_high - avg_low)

            collector.update_segment_discovery(
                signal=signal,
                high_responders_count=len(high_responders),
                low_responders_count=len(low_responders),
                responder_spread=responder_spread,
                analysis_latency_ms=state.get("segment_latency_ms", 0.0),
            )

            # Calculate total latency
            total_latency_ms = (
                state.get("estimation_latency_ms", 0)
                + state.get("segment_latency_ms", 0)
                + state.get("hierarchical_latency_ms", 0)
                + state.get("policy_latency_ms", 0)
                + generation_time
            )

            # Update with policy learning phase (final)
            policy_recommendations = state.get("policy_recommendations", [])
            actionable_count = sum(
                1 for p in policy_recommendations if p.get("actionable", True)
            )

            collector.update_policy_learning(
                signal=signal,
                policy_recommendations_count=len(policy_recommendations),
                expected_total_lift=state.get("expected_total_lift", 0.0),
                actionable_policies=actionable_count,
                executive_summary_length=len(executive_summary),
                key_insights_count=len(key_insights),
                visualization_data_complete=True,
                total_latency_ms=total_latency_ms,
                confidence_score=state.get("confidence_score", 0.8),
            )

            # Route to feedback_learner
            await route_heterogeneous_optimizer_signal(signal.to_dict())

            logger.debug(
                f"DSPy signal collected: reward={signal.compute_reward():.3f}, "
                f"heterogeneity={state.get('heterogeneity_score', 0):.2f}"
            )

        except Exception as e:
            logger.warning(f"DSPy signal collection failed (non-fatal): {e}")
