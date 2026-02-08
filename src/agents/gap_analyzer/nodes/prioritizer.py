"""Prioritizer Node for Gap Analyzer Agent.

This node prioritizes gaps by ROI and categorizes them into:
- Quick Wins: Low difficulty, high ROI (top 5)
- Strategic Bets: High impact, high difficulty (top 5)

Categorization Criteria:
- Quick Wins: cost < $10k AND gap < 10% AND ROI > 1
- Strategic Bets: cost > $50k AND ROI > 2
- Implementation Difficulty: Based on cost, gap size, complexity

V4.4: Added causal evidence filtering and confidence adjustments.
"""

import time
from typing import Any, Dict, List, Literal, Tuple, cast

from ..state import (
    GapAnalyzerState,
    PerformanceGap,
    PrioritizedOpportunity,
    ROIEstimate,
)

# V4.4: Causal evidence adjustment constants
DIRECT_CAUSE_BOOST = 1.2  # Boost for direct causes
NO_CAUSAL_EVIDENCE_PENALTY = 0.7  # Penalty for predictive-only features
HIGH_CAUSAL_SCORE_THRESHOLD = 0.6  # Threshold for "high" causal importance


class PrioritizerNode:
    """Prioritize gaps by ROI and categorize into quick wins and strategic bets."""

    def __init__(self):
        """Initialize prioritizer."""
        pass

    async def execute(self, state: GapAnalyzerState) -> Dict[str, Any]:
        """Execute prioritization workflow.

        Args:
            state: Current gap analyzer state with gaps_detected and roi_estimates

        Returns:
            Updated state with prioritized_opportunities, quick_wins, strategic_bets
        """
        start_time = time.time()

        try:
            gaps_detected = state.get("gaps_detected", [])
            roi_estimates = state.get("roi_estimates", [])
            max_opportunities = state.get("max_opportunities", 10)

            if not gaps_detected or not roi_estimates:
                return {
                    "prioritized_opportunities": [],
                    "quick_wins": [],
                    "strategic_bets": [],
                    "warnings": ["No gaps or ROI estimates available for prioritization"],
                    "status": "completed",
                }

            # Create gap_id -> gap mapping
            gap_map = {gap["gap_id"]: gap for gap in gaps_detected}

            # Create gap_id -> roi mapping
            roi_map = {roi["gap_id"]: roi for roi in roi_estimates}

            # Combine gaps with ROI estimates
            opportunities = []
            for gap_id in gap_map:
                if gap_id not in roi_map:
                    continue

                gap = gap_map[gap_id]
                roi_estimate = roi_map[gap_id]

                # Assess implementation difficulty
                difficulty = self._assess_difficulty(gap, roi_estimate)

                # Generate recommended action
                action = self._generate_action(gap, roi_estimate, difficulty)

                # Estimate time to impact
                time_to_impact = self._estimate_time_to_impact(difficulty)

                opportunity: PrioritizedOpportunity = {
                    "rank": 0,  # Will be set after sorting
                    "gap": gap,
                    "roi_estimate": roi_estimate,
                    "recommended_action": action,
                    "implementation_difficulty": difficulty,
                    "time_to_impact": time_to_impact,
                }

                opportunities.append(opportunity)

            # V4.4: Apply causal evidence adjustments if available
            causal_evidence_warnings: List[str] = []
            if self._has_causal_evidence(state):
                causal_rankings = state.get("causal_rankings", [])
                direct_cause_features = state.get("direct_cause_features", [])
                predictive_only_features = state.get("predictive_only_features", [])

                # Build causal lookup
                causal_lookup = self._build_causal_feature_lookup(causal_rankings or [])

                # Apply causal adjustments
                opportunities, causal_evidence_warnings = self._apply_causal_evidence_adjustments(
                    opportunities,
                    causal_lookup,
                    direct_cause_features or [],
                    predictive_only_features or [],
                )

            # Sort by expected ROI (descending) - may have been adjusted by causal evidence
            opportunities.sort(key=lambda o: o["roi_estimate"]["expected_roi"], reverse=True)

            # Assign ranks
            for rank, opp in enumerate(opportunities, start=1):
                opp["rank"] = rank

            # Limit to max_opportunities
            prioritized_opportunities = opportunities[:max_opportunities]

            # Categorize into quick wins and strategic bets
            quick_wins = self._identify_quick_wins(opportunities)[:5]
            strategic_bets = self._identify_strategic_bets(opportunities)[:5]

            prioritization_latency_ms = int((time.time() - start_time) * 1000)

            result: Dict[str, Any] = {
                "prioritized_opportunities": prioritized_opportunities,
                "quick_wins": quick_wins,
                "strategic_bets": strategic_bets,
                "prioritization_latency_ms": prioritization_latency_ms,
                "status": "completed",
            }

            # V4.4: Add causal evidence warnings if any
            if causal_evidence_warnings:
                result["causal_evidence_warnings"] = causal_evidence_warnings

            return result

        except Exception as e:
            prioritization_latency_ms = int((time.time() - start_time) * 1000)
            return {
                "errors": [
                    {
                        "node": "prioritizer",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                ],
                "prioritization_latency_ms": prioritization_latency_ms,
                "status": "failed",
            }

    def _assess_difficulty(
        self, gap: PerformanceGap, roi_estimate: ROIEstimate
    ) -> Literal["low", "medium", "high"]:
        """Assess implementation difficulty.

        Factors:
        - Cost to close (higher = harder)
        - Gap size (larger = harder)
        - Gap percentage (extreme = harder)

        Args:
            gap: Performance gap
            roi_estimate: ROI estimate

        Returns:
            Difficulty level: "low", "medium", "high"
        """
        cost = roi_estimate["estimated_cost_to_close"]
        gap_size = abs(gap["gap_size"])
        gap_pct = abs(gap["gap_percentage"])

        # Difficulty score (0-3)
        score = 0

        # Cost factor
        if cost > 50000:
            score += 1
        elif cost < 10000:
            score -= 1

        # Gap size factor (metric-specific)
        metric = gap["metric"]
        if metric in ["trx", "nrx"]:
            if gap_size > 100:
                score += 1
        elif metric in ["market_share", "conversion_rate"]:
            if gap_pct > 20:
                score += 1

        # Gap percentage factor
        if gap_pct > 50:
            score += 1
        elif gap_pct < 10:
            score -= 1

        # Map score to difficulty
        if score <= 0:
            return "low"
        elif score == 1:
            return "medium"
        else:
            return "high"

    def _generate_action(
        self,
        gap: PerformanceGap,
        roi_estimate: ROIEstimate,
        difficulty: Literal["low", "medium", "high"],
    ) -> str:
        """Generate recommended action for closing the gap.

        Args:
            gap: Performance gap
            roi_estimate: ROI estimate
            difficulty: Implementation difficulty

        Returns:
            Specific action recommendation
        """
        metric = gap["metric"]
        segment = gap["segment"]
        segment_value = gap["segment_value"]
        gap_type = gap["gap_type"]

        # Metric-specific action templates
        action_templates = {
            "trx": {
                "low": f"Launch targeted sampling campaign in {segment_value} ({segment}) to drive TRx growth",
                "medium": f"Implement multichannel engagement strategy for HCPs in {segment_value} to increase TRx",
                "high": f"Execute comprehensive market access and HCP engagement program in {segment_value} to close TRx gap",
            },
            "nrx": {
                "low": f"Deploy HCP educational webinars in {segment_value} to boost new prescriptions",
                "medium": f"Launch new prescriber acquisition campaign targeting {segment_value} specialists",
                "high": f"Develop strategic partnership program with KOLs in {segment_value} for NRx growth",
            },
            "market_share": {
                "low": f"Increase rep frequency in {segment_value} to capture share",
                "medium": f"Launch competitive positioning campaign in {segment_value}",
                "high": f"Execute full-scale market penetration strategy in {segment_value} with expanded resources",
            },
            "conversion_rate": {
                "low": f"Optimize patient starter program messaging for {segment_value}",
                "medium": f"Redesign patient journey touchpoints for {segment_value} segment",
                "high": f"Implement comprehensive patient support and HCP enablement program in {segment_value}",
            },
            "hcp_engagement_score": {
                "low": f"Increase digital touchpoints with HCPs in {segment_value}",
                "medium": f"Launch omnichannel engagement initiative for {segment_value} providers",
                "high": f"Build strategic HCP partnership program with personalized engagement for {segment_value}",
            },
        }

        # Get action template
        templates = action_templates.get(
            metric,
            {
                "low": f"Address performance gap in {segment_value}",
                "medium": f"Implement targeted intervention in {segment_value}",
                "high": f"Execute strategic initiative in {segment_value}",
            },
        )

        action = templates.get(difficulty, templates["medium"])

        # Add gap type context
        if gap_type == "vs_benchmark":
            action += " (benchmark-driven)"
        elif gap_type == "vs_potential":
            action += " (top-decile target)"
        elif gap_type == "temporal":
            action += " (restore prior performance)"

        return action

    def _estimate_time_to_impact(self, difficulty: Literal["low", "medium", "high"]) -> str:
        """Estimate time to see results.

        Args:
            difficulty: Implementation difficulty

        Returns:
            Time range estimate (e.g., "1-3 months")
        """
        time_estimates = {
            "low": "1-3 months",
            "medium": "3-6 months",
            "high": "6-12 months",
        }

        return time_estimates[difficulty]

    def _identify_quick_wins(
        self, opportunities: List[PrioritizedOpportunity]
    ) -> List[PrioritizedOpportunity]:
        """Identify quick win opportunities.

        Criteria:
        - Low implementation difficulty
        - ROI > 1.0
        - Cost < $10k (optional, for clarity)

        Args:
            opportunities: All prioritized opportunities

        Returns:
            List of quick wins (sorted by ROI)
        """
        quick_wins = [
            opp
            for opp in opportunities
            if opp["implementation_difficulty"] == "low"
            and opp["roi_estimate"]["expected_roi"] > 1.0
        ]

        # Sort by ROI
        quick_wins.sort(key=lambda o: o["roi_estimate"]["expected_roi"], reverse=True)

        return quick_wins

    def _identify_strategic_bets(
        self, opportunities: List[PrioritizedOpportunity]
    ) -> List[PrioritizedOpportunity]:
        """Identify strategic bet opportunities.

        Criteria:
        - High implementation difficulty
        - ROI > 2.0 (high impact)
        - Cost > $50k (significant investment)

        Args:
            opportunities: All prioritized opportunities

        Returns:
            List of strategic bets (sorted by ROI)
        """
        strategic_bets = [
            opp
            for opp in opportunities
            if opp["implementation_difficulty"] == "high"
            and opp["roi_estimate"]["expected_roi"] > 2.0
            and opp["roi_estimate"]["estimated_cost_to_close"] > 50000
        ]

        # Sort by ROI
        strategic_bets.sort(key=lambda o: o["roi_estimate"]["expected_roi"], reverse=True)

        return strategic_bets

    # =========================================================================
    # V4.4: Causal Evidence Filtering Methods
    # =========================================================================

    def _build_causal_feature_lookup(
        self, causal_rankings: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Build lookup from feature name to causal ranking info.

        Args:
            causal_rankings: List of FeatureRanking dicts from DriverRanker

        Returns:
            Dict mapping feature_name to ranking info
        """
        lookup: Dict[str, Dict[str, Any]] = {}
        for ranking in causal_rankings:
            feature_name = ranking.get("feature_name", "")
            if feature_name:
                lookup[feature_name] = {
                    "causal_rank": ranking.get("causal_rank"),
                    "predictive_rank": ranking.get("predictive_rank"),
                    "causal_score": ranking.get("causal_score", 0.0),
                    "predictive_score": ranking.get("predictive_score", 0.0),
                    "rank_difference": ranking.get("rank_difference", 0),
                    "is_direct_cause": ranking.get("is_direct_cause", False),
                    "path_length": ranking.get("path_length"),
                }
        return lookup

    def _get_gap_feature_name(self, gap: PerformanceGap) -> str:
        """Extract feature name from gap for causal lookup.

        The gap's metric and segment form the feature identifier.

        Args:
            gap: Performance gap

        Returns:
            Feature name for causal lookup
        """
        # Primary feature is the metric being measured
        return gap["metric"]

    def _apply_causal_evidence_adjustments(
        self,
        opportunities: List[PrioritizedOpportunity],
        causal_lookup: Dict[str, Dict[str, Any]],
        direct_cause_features: List[str],
        predictive_only_features: List[str],
    ) -> Tuple[List[PrioritizedOpportunity], List[str]]:
        """Adjust opportunity ROI based on causal evidence.

        V4.4: Boost opportunities targeting direct causes,
        penalize those based only on predictive importance.

        Args:
            opportunities: List of opportunities to adjust
            causal_lookup: Feature name to causal ranking lookup
            direct_cause_features: Features with direct causal edge to target
            predictive_only_features: Features with predictive but no causal signal

        Returns:
            Tuple of (adjusted opportunities, causal evidence warnings)
        """
        adjusted_opportunities = []
        warnings: List[str] = []

        for opp in opportunities:
            gap = opp["gap"]
            feature_name = self._get_gap_feature_name(gap)
            roi_estimate = opp["roi_estimate"]
            original_roi = roi_estimate["expected_roi"]

            # Get causal info for this feature
            causal_info = causal_lookup.get(feature_name)

            adjustment_factor = 1.0
            adjustment_reason = None

            if causal_info:
                causal_score = causal_info.get("causal_score", 0.0)
                is_direct_cause = causal_info.get("is_direct_cause", False)

                # Boost for direct causes
                if is_direct_cause or feature_name in direct_cause_features:
                    adjustment_factor = DIRECT_CAUSE_BOOST
                    adjustment_reason = "direct_cause_boost"
                # Boost for high causal score
                elif causal_score >= HIGH_CAUSAL_SCORE_THRESHOLD:
                    adjustment_factor = 1.0 + (causal_score - HIGH_CAUSAL_SCORE_THRESHOLD) * 0.5
                    adjustment_reason = "high_causal_score"
                # Penalize predictive-only features
                elif feature_name in predictive_only_features:
                    adjustment_factor = NO_CAUSAL_EVIDENCE_PENALTY
                    adjustment_reason = "predictive_only_penalty"
                    warnings.append(
                        f"Gap '{gap['gap_id']}' targets '{feature_name}' which lacks causal evidence. "
                        f"ROI adjusted by {NO_CAUSAL_EVIDENCE_PENALTY:.0%}."
                    )
            else:
                # No causal info available - add warning but don't adjust
                warnings.append(
                    f"Gap '{gap['gap_id']}' targets '{feature_name}' with no causal analysis available."
                )

            # Apply adjustment to ROI
            if adjustment_factor != 1.0:
                adjusted_roi = original_roi * adjustment_factor
                # Create updated ROI estimate with causal adjustment
                adjusted_roi_estimate = dict(roi_estimate)
                adjusted_roi_estimate["expected_roi"] = adjusted_roi
                adjusted_roi_estimate["causal_adjustment_factor"] = adjustment_factor
                adjusted_roi_estimate["causal_adjustment_reason"] = adjustment_reason

                # Create adjusted opportunity
                adjusted_opp: PrioritizedOpportunity = {
                    "rank": opp["rank"],
                    "gap": opp["gap"],
                    "roi_estimate": cast(ROIEstimate, adjusted_roi_estimate),
                    "recommended_action": opp["recommended_action"],
                    "implementation_difficulty": opp["implementation_difficulty"],
                    "time_to_impact": opp["time_to_impact"],
                }
                adjusted_opportunities.append(adjusted_opp)
            else:
                adjusted_opportunities.append(opp)

        return adjusted_opportunities, warnings

    def _has_causal_evidence(self, state: GapAnalyzerState) -> bool:
        """Check if causal discovery results are available and valid.

        Args:
            state: Current gap analyzer state

        Returns:
            True if causal evidence is available for filtering
        """
        causal_rankings = state.get("causal_rankings", [])
        discovery_gate_decision = state.get("discovery_gate_decision")

        # Causal evidence is available if:
        # 1. We have causal rankings
        # 2. Discovery gate decision is accept or review (not reject)
        return bool(causal_rankings) and discovery_gate_decision in ("accept", "review")
