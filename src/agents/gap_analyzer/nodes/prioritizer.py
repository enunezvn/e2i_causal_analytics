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

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from ..action_templates import render_action
from ..opportunity_classification import classify_bucket, is_low_value
from ..state import (
    GapAnalyzerState,
    PerformanceGap,
    PrioritizedOpportunity,
    ROIEstimate,
)

logger = logging.getLogger(__name__)

# V4.4: Causal evidence adjustment constants
DIRECT_CAUSE_BOOST = 1.2  # Boost for direct causes
NO_CAUSAL_EVIDENCE_PENALTY = 0.7  # Penalty for predictive-only features
HIGH_CAUSAL_SCORE_THRESHOLD = 0.6  # Threshold for "high" causal importance

# #357: Instrument-availability bonus — credibility boost when a STRONG instrument
# (first-stage F >= 10, Staiger-Stock) is available for the opportunity's feature.
STRONG_INSTRUMENT_BONUS = 1.15  # < DIRECT_CAUSE_BOOST (1.2): availability of a strong
#                                 identification strategy is supporting evidence, not as
#                                 strong as a confirmed direct-cause edge.
STRONG_INSTRUMENT_F_FLOOR = 10.0  # Staiger-Stock strong threshold; mirrors
#                                   IVDiagnostics.is_weak_instrument (F < 10) and
#                                   _classify_instrument_strength (f_stat >= 10 -> STRONG).
#                                   Inclusive ">= 10" boundary (belt-and-suspenders gate).


class PrioritizerNode:
    """Prioritize gaps by ROI and categorize into quick wins and strategic bets."""

    def __init__(self, label_criteria_provider: Optional[object] = None):
        """Initialize prioritizer.

        Args:
            label_criteria_provider: Injectable indicated-population provider for the
                label-gater (tests pass a fixed one). Lazily defaulted to
                ``LabelCriteriaProvider`` only when the gate is actually enabled — avoids
                import cost / network at construction time when label_segmentation is off.
        """
        self._label_provider = label_criteria_provider

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
                    "steady_plays": [],
                    "strategic_bets": [],
                    "suppressed_count": 0,
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

                # Assess implementation difficulty + capture the human-readable
                # rationale (the cost/gap-size factors that drove the rating —
                # previously computed-then-discarded; T6 surfaces it for drill-down).
                difficulty = self._assess_difficulty(gap, roi_estimate)
                difficulty_rationale = self._explain_difficulty(gap, roi_estimate, difficulty)

                # Generate recommended action (#1835: brand-aware — the same gap
                # for Kisqali and Remibrutinib must not read identically).
                action = self._generate_action(gap, roi_estimate, difficulty, state.get("brand"))

                # Estimate time to impact
                time_to_impact = self._estimate_time_to_impact(difficulty)

                opportunity: PrioritizedOpportunity = {
                    "rank": 0,  # Will be set after sorting
                    "gap": gap,
                    "roi_estimate": roi_estimate,
                    "recommended_action": action,
                    "implementation_difficulty": difficulty,
                    "time_to_impact": time_to_impact,
                    "difficulty_rationale": difficulty_rationale,
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

            # #357: Apply instrument-availability bonus if available. Independent of and
            # ADDITIVE to the V4.4 causal adjustment above — runs after it, so when a
            # feature is both a direct cause AND strong-instrumented the two compound.
            if self._has_instrument_evidence(state):
                instrument_lookup = state.get("instrument_strength_by_feature") or {}
                opportunities, instrument_warnings = self._apply_instrument_availability_bonus(
                    opportunities,
                    instrument_lookup,
                )
                causal_evidence_warnings.extend(instrument_warnings)

            # Label-gater (opt-in): resolve the indicated population once (fail-open) and
            # annotate each opportunity's gap segment with its on/off-label verdict. None
            # => gating is a no-op (existing behaviour). Annotation precedes the sort so
            # off-label opportunities can be demoted below.
            population = self._resolve_population(state)
            if population is not None:
                for opp in opportunities:
                    self._apply_label_gate(opp["roi_estimate"], opp["gap"], population)

            # T6: suppress low-value noise and stamp the 3-bucket category. An
            # opportunity whose FINAL (causal/instrument-adjusted) expected_roi is
            # at or below break-even returns no more than its cost — it is hidden
            # rather than dumped into a junk "other" bucket. Survivors are tagged
            # with exactly one of quick_win / steady_play / strategic_bet via the
            # shared classification SSOT (so the agent, the headline counts, and
            # the per-card badge can never disagree).
            surviving: List[PrioritizedOpportunity] = []
            suppressed_count = 0
            for opp in opportunities:
                if is_low_value(opp["roi_estimate"]["expected_roi"]):
                    suppressed_count += 1
                    continue
                opp["category"] = classify_bucket(
                    opp["implementation_difficulty"],
                    opp["roi_estimate"]["expected_roi"],
                    opp["roi_estimate"]["estimated_cost_to_close"],
                )
                surviving.append(opp)
            opportunities = surviving

            # Rank: on-label/indeterminate first, off-label opportunities demoted to the
            # bottom (codex#7 — explicit partition-by-verdict; expected_roi order is
            # PRESERVED within each partition, the ROI value itself is never tampered
            # with). Ascending key (no reverse): off_label False sorts before True, and
            # -expected_roi sorts higher ROI first. When the gate is off every off_label
            # defaults False, so this reduces to the prior expected_roi-descending sort.
            opportunities.sort(
                key=lambda o: (
                    o["roi_estimate"].get("off_label", False),
                    -o["roi_estimate"]["expected_roi"],
                )
            )

            # Assign ranks from the partitioned order.
            for rank, opp in enumerate(opportunities, start=1):
                opp["rank"] = rank

            # Limit to max_opportunities
            prioritized_opportunities = opportunities[:max_opportunities]

            # Categorize by the stamped 3-bucket category AFTER the partitioned sort
            # so the highlight lists respect the off-label demotion (each list is a
            # filter on the already-sorted, already-classified survivors). Capped at
            # the top 5 per bucket for the highlight lists; the API endpoint derives
            # truthful brand-level counts from the same classifier.
            quick_wins = self._identify_quick_wins(opportunities)[:5]
            steady_plays = self._identify_steady_plays(opportunities)[:5]
            strategic_bets = self._identify_strategic_bets(opportunities)[:5]

            prioritization_latency_ms = int((time.time() - start_time) * 1000)

            result: Dict[str, Any] = {
                "prioritized_opportunities": prioritized_opportunities,
                "quick_wins": quick_wins,
                "steady_plays": steady_plays,
                "strategic_bets": strategic_bets,
                "suppressed_count": suppressed_count,
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

    def _resolve_population(self, state: GapAnalyzerState):
        """Resolve the indicated population for the run, or None when the label-gater
        is off / brand absent / lookup fails (fail-open — never breaks prioritization).

        Indication: ``state["indication"]`` if present, else None (the provider falls
        back to the brand's primary indication — gap_analyzer loads business_metrics, not
        patient_journeys, so diagnosis-based indication resolution is not available here).
        """
        if not state.get("label_segmentation") or not state.get("brand"):
            return None
        provider = self._label_provider
        if provider is None:
            from src.services.clinical_context.label_criteria_provider import (
                LabelCriteriaProvider,
            )

            provider = LabelCriteriaProvider()
            self._label_provider = provider
        try:
            return provider.derive(state["brand"], state.get("indication"))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — fail-open
            logger.warning(
                "label-gater: indicated-population lookup failed; skipping gate",
                extra={"node": "prioritizer", "brand": state.get("brand"), "error": str(exc)},
            )
            return None

    def _apply_label_gate(self, roi_estimate: ROIEstimate, gap: PerformanceGap, population) -> None:
        """Annotate an opportunity's ROI estimate with its label verdict (fail-open, in
        place). ``off_label`` is set ONLY for a label-evidenced violation; the
        opportunity is surfaced either way (rank demotion, not deletion; ROI untouched).
        """
        try:
            from src.services.clinical_context.label_gate import (
                descriptor_from_segment,
                evaluate_segment,
            )

            descriptor = descriptor_from_segment(gap["segment"], gap["segment_value"])
            verdict = evaluate_segment([descriptor], population)
        except Exception as exc:  # noqa: BLE001 — fail-open
            logger.warning(
                "label-gater: segment gate failed; leaving opportunity unflagged",
                extra={"node": "prioritizer", "error": str(exc)},
            )
            return
        roi_estimate["label_verdict"] = verdict.verdict
        roi_estimate["off_label"] = verdict.verdict == "off_label"
        roi_estimate["label_evidence_confirmed"] = verdict.confirmed_by_label
        if verdict.reason:
            roi_estimate["off_label_reason"] = verdict.reason

    def _difficulty_factors(
        self, gap: PerformanceGap, roi_estimate: ROIEstimate
    ) -> Tuple[int, List[str]]:
        """Compute the difficulty score AND the human-readable factors behind it.

        Single source of truth for both ``_assess_difficulty`` (which maps the
        score to a level) and ``_explain_difficulty`` (which renders the factors).

        Factors:
        - Cost to close (higher = harder)
        - Gap size (larger = harder)
        - Gap percentage (extreme = harder)

        Returns:
            (score, factor_phrases) where each phrase explains one +/- driver.
        """
        cost = roi_estimate["estimated_cost_to_close"]
        gap_size = abs(gap["gap_size"])
        gap_pct = abs(gap["gap_percentage"])
        metric = gap["metric"]

        score = 0
        factors: List[str] = []

        # Cost factor
        if cost > 50000:
            score += 1
            factors.append(f"high cost to close (${cost:,.0f} > $50k)")
        elif cost < 10000:
            score -= 1
            factors.append(f"low cost to close (${cost:,.0f} < $10k)")

        # Gap size factor (metric-specific)
        if metric in ["trx", "nrx"]:
            if gap_size > 100:
                score += 1
                factors.append(f"large {metric.upper()} gap ({gap_size:,.0f} units > 100)")
        elif metric in ["market_share", "conversion_rate"]:
            if gap_pct > 20:
                score += 1
                factors.append(f"wide {metric.replace('_', ' ')} gap ({gap_pct:.0f}% > 20%)")

        # Gap percentage factor
        if gap_pct > 50:
            score += 1
            factors.append(f"extreme gap size ({gap_pct:.0f}% > 50%)")
        elif gap_pct < 10:
            score -= 1
            factors.append(f"small gap size ({gap_pct:.0f}% < 10%)")

        return score, factors

    def _assess_difficulty(
        self, gap: PerformanceGap, roi_estimate: ROIEstimate
    ) -> Literal["low", "medium", "high"]:
        """Assess implementation difficulty (low / medium / high)."""
        score, _ = self._difficulty_factors(gap, roi_estimate)
        if score <= 0:
            return "low"
        elif score == 1:
            return "medium"
        else:
            return "high"

    def _explain_difficulty(
        self,
        gap: PerformanceGap,
        roi_estimate: ROIEstimate,
        difficulty: Literal["low", "medium", "high"],
    ) -> str:
        """Render a one-line rationale for the difficulty rating (T6 drill-down).

        Cites the cost/gap-size factors that pushed the score up or down, so the
        UI can answer "why is this rated medium effort?". Falls back to a baseline
        statement when no factor moved the needle.
        """
        _, factors = self._difficulty_factors(gap, roi_estimate)
        if factors:
            return f"Rated {difficulty} implementation effort: {'; '.join(factors)}."
        return (
            f"Rated {difficulty} implementation effort: no cost or gap-size factor "
            "pushed it above the baseline."
        )

    def _generate_action(
        self,
        gap: PerformanceGap,
        roi_estimate: ROIEstimate,
        difficulty: Literal["low", "medium", "high"],
        brand: Optional[str] = None,
    ) -> str:
        """Generate the recommended action for closing the gap (#1835: brand-aware).

        The wording names the brand and its SSOT-derived HCP audience (see
        ``action_templates``) so two brands never share a sentence for the same
        gap; an unknown / missing brand falls open to the neutral templates.

        Args:
            gap: Performance gap
            roi_estimate: ROI estimate (unused by the templates; kept for the
                call-site contract)
            difficulty: Implementation difficulty
            brand: ``state["brand"]`` as the API received it (any casing)

        Returns:
            Specific action recommendation, <= 160 chars (executive-brief budget)
        """
        return render_action(
            metric=gap["metric"],
            difficulty=difficulty,
            segment=gap["segment"],
            segment_value=gap["segment_value"],
            gap_type=gap["gap_type"],
            brand=brand,
        )

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
        """Highlight list of quick-win opportunities (category == "quick_win").

        A filter on the already-classified, already-partition-sorted survivors —
        so order (on-label first, ROI desc) is preserved without re-sorting. The
        ``quick_win`` category itself is defined by the shared classification SSOT
        (low difficulty AND ROI > 1).
        """
        return [opp for opp in opportunities if opp.get("category") == "quick_win"]

    def _identify_steady_plays(
        self, opportunities: List[PrioritizedOpportunity]
    ) -> List[PrioritizedOpportunity]:
        """Highlight list of steady-play opportunities (category == "steady_play")
        — the meaningful middle ground (neither quick win nor strategic bet)."""
        return [opp for opp in opportunities if opp.get("category") == "steady_play"]

    def _identify_strategic_bets(
        self, opportunities: List[PrioritizedOpportunity]
    ) -> List[PrioritizedOpportunity]:
        """Highlight list of strategic-bet opportunities (category == "strategic_bet").

        ``strategic_bet`` is defined by the shared classification SSOT (high
        difficulty AND ROI > 2 AND cost > $50k) — so a merely high-difficulty
        opportunity is NEVER a phantom strategic bet here.
        """
        return [opp for opp in opportunities if opp.get("category") == "strategic_bet"]

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

                # Create adjusted opportunity. Spread the original so non-ROI keys
                # (e.g. difficulty_rationale) are carried through, not dropped.
                adjusted_opp = cast(
                    PrioritizedOpportunity,
                    {**opp, "roi_estimate": cast(ROIEstimate, adjusted_roi_estimate)},
                )
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

    # =========================================================================
    # #357: Instrument-Availability Bonus Methods
    # =========================================================================

    def _has_instrument_evidence(self, state: GapAnalyzerState) -> bool:
        """Check if per-feature instrument strength is available.

        Independent of the V4.4 causal gate: the instrument bonus can apply even when
        causal_rankings is absent, and vice-versa (the two mechanisms are additive and
        decoupled).

        Args:
            state: Current gap analyzer state

        Returns:
            True if instrument strength data is present (non-empty).
        """
        return bool(state.get("instrument_strength_by_feature"))

    def _apply_instrument_availability_bonus(
        self,
        opportunities: List[PrioritizedOpportunity],
        instrument_lookup: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[PrioritizedOpportunity], List[str]]:
        """Boost opportunity ROI when a STRONG instrument is available (#357, Option-3).

        Asymmetric by design (bonus-only, D-4): only ``InstrumentStrength.STRONG`` earns a
        bonus; MODERATE/WEAK/VERY_WEAK/absent -> factor 1.0 (no penalty — absence of an
        instrument is not evidence the opportunity is bad).

        Belt-and-suspenders gate: requires BOTH ``instrument_strength == "strong"`` AND a
        real ``first_stage_f_stat >= STRONG_INSTRUMENT_F_FLOOR`` — guards against any
        upstream that sets the enum without a real F-stat.

        Compounding with V4.4 (D-3 = multiply): this runs AFTER the V4.4 causal adjustment
        and multiplies the already-causal-adjusted ``expected_roi``. The instrument factor
        is recorded in NEW ROI keys (``instrument_adjustment_factor`` /
        ``instrument_adjustment_reason``) so the V4.4 record is never overwritten.

        Args:
            opportunities: Opportunities to adjust (possibly already V4.4-adjusted)
            instrument_lookup: feature_name -> IVDiagnostics.to_dict() output

        Returns:
            Tuple of (adjusted opportunities, instrument warnings)
        """
        adjusted_opportunities: List[PrioritizedOpportunity] = []
        warnings: List[str] = []

        for opp in opportunities:
            gap = opp["gap"]
            feature_name = self._get_gap_feature_name(gap)
            roi_estimate = opp["roi_estimate"]
            original_roi = roi_estimate["expected_roi"]

            diag = instrument_lookup.get(feature_name)

            adjustment_factor = 1.0
            adjustment_reason: Optional[str] = None

            if diag is not None:
                strength = diag.get("instrument_strength")
                try:
                    f_stat = float(diag.get("first_stage_f_stat", 0.0))
                except (TypeError, ValueError):
                    f_stat = 0.0

                if strength == "strong" and f_stat >= STRONG_INSTRUMENT_F_FLOOR:
                    adjustment_factor = STRONG_INSTRUMENT_BONUS
                    adjustment_reason = "strong_instrument_bonus"
                    warnings.append(
                        f"Gap '{gap['gap_id']}' targets '{feature_name}' which has a strong "
                        f"instrument (first-stage F={f_stat:.1f}). "
                        f"ROI boosted by {STRONG_INSTRUMENT_BONUS:.0%}."
                    )

            if adjustment_factor != 1.0:
                adjusted_roi_estimate = dict(roi_estimate)
                adjusted_roi_estimate["expected_roi"] = original_roi * adjustment_factor
                adjusted_roi_estimate["instrument_adjustment_factor"] = adjustment_factor
                adjusted_roi_estimate["instrument_adjustment_reason"] = adjustment_reason

                # Spread the original so non-ROI keys (e.g. difficulty_rationale)
                # are carried through, not dropped.
                adjusted_opp = cast(
                    PrioritizedOpportunity,
                    {**opp, "roi_estimate": cast(ROIEstimate, adjusted_roi_estimate)},
                )
                adjusted_opportunities.append(adjusted_opp)
            else:
                adjusted_opportunities.append(opp)

        return adjusted_opportunities, warnings
