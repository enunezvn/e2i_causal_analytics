"""Policy Learner Node for Heterogeneous Optimizer Agent.

This node learns optimal treatment allocation policy based on CATE estimates.
Uses CATE to recommend allocation changes.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..state import HeterogeneousOptimizerState, PolicyRecommendation

logger = logging.getLogger(__name__)


class PolicyLearnerNode:
    """Learn optimal treatment allocation policy.

    Recommends treatment rate adjustments, significance-gated:
    - High-responder TIER (classified upstream by segment_analyzer — strict
      1.5x|ATE| with a relative top-half fallback) with a positive CATE: increase.
    - Genuinely HARMFUL subgroup (CATE < 0): minimise, regardless of tier.
    - Everything else (average / below-average-but-beneficial / not significant):
      maintain.

    The decision boundary for the tiers lives in segment_analyzer; this node acts
    on the tier it assigned (so the policy can never contradict the responder tiers
    the page displays) plus the per-segment CATE sign. See _generate_recommendation.
    """

    def __init__(self, label_criteria_provider: Optional[object] = None):
        # Injectable for tests; lazily defaulted (avoids import cost when the
        # label-gater is off, and network at construction time).
        self._label_provider = label_criteria_provider

    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        """Execute policy learning."""
        start_time = time.time()
        logger.info(
            "Starting policy learning",
            extra={
                "node": "policy_learner",
                "segment_count": len(state.get("cate_by_segment") or {}),
                "high_responders": len(state.get("high_responders") or []),
                "low_responders": len(state.get("low_responders") or []),
            },
        )

        if state.get("status") == "failed":
            logger.warning("Skipping policy learning - previous node failed")
            return state

        # #437 fail-close: upstream cate_estimator did NOT set status=failed
        # but produced no overall_ate. Policy recommendations are scored against
        # the baseline ATE; without it the silent-default ``or 0.0`` previously
        # produced zero-lift "neutral" recommendations indistinguishable from a
        # legitimate honest zero. Honest zero (``overall_ate == 0.0``) is
        # distinct from absence (``None``) and remains supported.
        if state.get("overall_ate") is None:
            logger.error(
                "Fail-closed: upstream cate_estimator produced no overall_ate",
                extra={"node": "policy_learner"},
            )
            return {
                **state,
                "errors": [
                    {
                        "node": "policy_learner",
                        "error": (
                            "upstream cate_estimator produced overall_ate=None "
                            "without setting status=failed; refusing to "
                            "synthesize policy recommendations without baseline ATE"
                        ),
                    }
                ],
                "status": "failed",
            }

        try:
            cate_by_segment_raw = state.get("cate_by_segment")
            cate_by_segment = cate_by_segment_raw if cate_by_segment_raw is not None else {}
            high_responders = state.get("high_responders") or []
            low_responders = state.get("low_responders") or []
            raw_ate = state.get("overall_ate")
            ate: float = float(raw_ate) if raw_ate is not None else 0.0

            # #437 row 4: explicit warning on partial data (real ATE but empty
            # cate_by_segment) so consumers see SKIPPED-not-substituted.
            partial_data_warning: str | None = None
            if not cate_by_segment:
                partial_data_warning = (
                    "cate_by_segment is empty; policy recommendations cannot "
                    "be derived per-segment and total expected lift will be 0"
                )
                logger.warning(partial_data_warning, extra={"node": "policy_learner"})

            # Resolve the indicated population once (fail-open) if the label-gater
            # is enabled for this run. None => gating is a no-op (existing behaviour).
            population = self._resolve_population(state)

            # Generate policy recommendations. Direction is decided per-segment by
            # the ABOVE-ATE significance gate in _generate_recommendation (a segment
            # is reallocated only if its CATE CI is significantly above the ATE, or
            # significantly harmful) — so a homogeneous effect honestly yields ~0
            # lift even when segment_analyzer's relative fallback still lists nominal
            # high/low responders for display.
            recommendations = []

            for segment_var, results in cate_by_segment.items():
                for result in results:
                    rec = self._generate_recommendation(
                        dict(result),  # type: ignore[arg-type]
                        ate,
                    )
                    if rec:
                        if population is not None:
                            self._apply_label_gate(rec, dict(result), population)
                        recommendations.append(rec)

            # Rank: on-label/indeterminate first (then by expected incremental
            # outcome), off-label segments demoted to the bottom (codex#7 — explicit
            # partition-by-verdict, metric preserved WITHIN each partition; the
            # outcome value itself is never tampered with).
            recommendations.sort(
                key=lambda r: (r.get("off_label", False), -r["expected_incremental_outcome"])
            )

            # Expected lift WITHOUT cross-dimension double-counting (see
            # _aggregate_total_lift). Reported as the best single-segmentation-axis
            # targeting opportunity, not the sum across overlapping dimensions.
            total_lift, best_dim = self._aggregate_total_lift(recommendations)

            # Percentage-point lift = best-axis incremental outcomes / cohort size.
            # cohort_size = total patients in any ONE dimension (each dimension
            # partitions the same cohort). This is the HEADLINE metric: an absolute
            # change in the outcome RATE (e.g. +4.9pp adherence), which is honest and
            # interpretable, unlike a raw count that scales with cohort size. ~0 for a
            # homogeneous effect under the above-ATE gate.
            cohort_size = (
                sum(r["sample_size"] for r in next(iter(cate_by_segment.values())))
                if cate_by_segment
                else 0
            )
            expected_lift_pp = (total_lift / cohort_size) if cohort_size > 0 else 0.0

            # Generate summary
            summary = self._generate_allocation_summary(
                recommendations,
                high_responders,
                low_responders,
                ate,
                total_lift,
                best_dim,
                expected_lift_pp,
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

            output: Dict[str, Any] = {
                **state,
                "policy_recommendations": recommendations[:20],  # Top 20
                "expected_total_lift": total_lift,
                "expected_lift_pp": expected_lift_pp,
                "optimal_allocation_summary": summary,
                "total_latency_ms": total_time,
                "status": "completed",
            }
            if partial_data_warning is not None:
                existing_warnings = state.get("warnings") or []
                output["warnings"] = [*existing_warnings, partial_data_warning]
            return output  # type: ignore[return-value]

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

    def _generate_recommendation(
        self,
        result: Dict[str, Any],
        ate: float,
    ) -> PolicyRecommendation:
        """Generate policy recommendation for a segment.

        ABOVE-ATE significance gate. A segment is only reallocated when its CATE is
        statistically distinguishable from the OVERALL ATE — i.e. it is a *genuine*
        differential responder, not merely the top half of a near-uniform effect:

        - INCREASE iff the segment's CATE confidence interval lies ENTIRELY ABOVE
          the ATE (``cate_ci_lower > ate``) AND the effect is positive — the segment
          responds significantly MORE than average. (A segment whose CI overlaps the
          ATE is statistically average; concentrating treatment there is noise-
          driven, so it is MAINTAINED.)
        - DECREASE iff the CATE confidence interval lies ENTIRELY BELOW ZERO
          (``cate_ci_upper < 0``) — a genuinely HARMFUL subgroup, minimised wherever
          it occurs (sign-based, tier-independent, correct for a net-harmful ATE<0).
        - Otherwise MAINTAIN.

        WHY this is stricter than per-segment significance (CATE != 0): for a
        beneficial-but-UNIFORM effect every band is significantly positive yet none
        is meaningfully ABOVE average, so the honest targeting lift is ~0. Gating on
        "significantly above the ATE" (using the CIs the estimator already produced)
        makes the lift collapse to ~0 for homogeneous effects and only surface a real
        number when genuine heterogeneity exists.

        Args:
            result: CATE result dictionary (incl. cate_ci_lower / cate_ci_upper)
            ate: Overall average treatment effect — the bar a segment must clear

        Returns:
            Policy recommendation
        """

        cate = result["cate_estimate"]
        segment_name = result["segment_name"]
        segment_value = result["segment_value"]
        sample_size = result["sample_size"]
        ci_lower = result.get("cate_ci_lower")
        ci_upper = result.get("cate_ci_upper")

        # Determine recommended treatment rate change
        current_rate = 0.5  # Assume current 50% coverage

        if cate > 0 and ci_lower is not None and ci_lower > ate:
            # CI lies entirely ABOVE the ATE -> significantly above-average responder.
            recommended_rate = min(0.9, current_rate + 0.2)
        elif ci_upper is not None and ci_upper < 0:
            # CI lies entirely BELOW zero -> significantly harmful subgroup; minimise.
            recommended_rate = 0.1
        else:
            # Statistically average (CI overlaps the ATE) / not harmful — maintain.
            recommended_rate = current_rate

        # Calculate expected incremental outcome. Sign is correct in both action
        # branches: increasing a positive-CATE segment (+rate x +cate) and reducing
        # a negative-CATE segment (-rate x -cate) both yield a positive lift.
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

    def _resolve_population(self, state: HeterogeneousOptimizerState):
        """Resolve the indicated population for the run, or None when the label-gater
        is off / brand absent / lookup fails (fail-open — never breaks policy learning)."""
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
                extra={"node": "policy_learner", "brand": state.get("brand"), "error": str(exc)},
            )
            return None

    def _apply_label_gate(
        self, rec: PolicyRecommendation, result: Dict[str, Any], population
    ) -> None:
        """Annotate a recommendation with its label verdict (fail-open, in place).
        off_label is set ONLY for a label-evidenced violation; the recommendation is
        surfaced either way (rank demotion, not deletion; outcome value untouched)."""
        try:
            from src.services.clinical_context.label_gate import (
                descriptor_from_segment,
                evaluate_segment,
            )

            descriptor = descriptor_from_segment(result["segment_name"], result["segment_value"])
            verdict = evaluate_segment([descriptor], population)
        except Exception as exc:  # noqa: BLE001 — fail-open
            logger.warning(
                "label-gater: segment gate failed; leaving recommendation unflagged",
                extra={"node": "policy_learner", "error": str(exc)},
            )
            return
        rec["label_verdict"] = verdict.verdict
        rec["off_label"] = verdict.verdict == "off_label"
        rec["label_evidence_confirmed"] = verdict.confirmed_by_label
        if verdict.reason:
            rec["off_label_reason"] = verdict.reason

    @staticmethod
    def _aggregate_total_lift(
        recommendations: List[PolicyRecommendation],
    ) -> Tuple[float, Optional[str]]:
        """Expected lift WITHOUT double-counting across overlapping segmentation
        dimensions.

        ``cate_by_segment`` carries SEVERAL segmentation dimensions
        (disease_severity_band, age_band, geographic_region, …). Each dimension
        is a DIFFERENT partition of the SAME cohort, so a patient appears once per
        dimension. Summing ``expected_incremental_outcome`` across ALL
        recommendations therefore counts each patient once per dimension (~Nx
        inflation — the reported +1289 was ~5x too high). WITHIN a single dimension
        the bands ARE disjoint, so that per-dimension sum is valid.

        Report the best single-dimension policy: the maximum per-dimension total
        ("optimal targeting on one segmentation axis"). Returns (lift, dimension).
        Combining multiple axes for more lift would require patient-level joint
        membership, which the aggregated segment state does not carry.
        """
        by_dim: Dict[str, float] = defaultdict(float)
        for r in recommendations:
            dim = str(r.get("segment", "")).split("=", 1)[0]
            by_dim[dim] += r.get("expected_incremental_outcome", 0.0) or 0.0
        if not by_dim:
            return 0.0, None
        best_dim = max(by_dim, key=lambda d: by_dim[d])
        return by_dim[best_dim], best_dim

    def _generate_allocation_summary(
        self,
        recommendations: List[PolicyRecommendation],
        high_responders: List,
        low_responders: List,
        ate: float,
        total_lift: float,
        best_dim: Optional[str],
        expected_lift_pp: float,
    ) -> str:
        """Generate natural language summary of optimal allocation.

        Args:
            recommendations: Policy recommendations
            high_responders: High responder segments
            low_responders: Low responder segments
            ate: Overall ATE
            total_lift: De-double-counted expected lift (best single-axis count)
            best_dim: The segmentation dimension that total_lift comes from
            expected_lift_pp: The same lift as a percentage-point change in the
                outcome rate (count / cohort) — the headline metric

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

        if not increase_recs and not decrease_recs:
            summary_parts.append(
                "No segment responds significantly above the average effect, so there "
                "is no reliable differential-targeting opportunity — expected lift "
                "from targeting is ~0. A uniform rollout is appropriate."
            )
            return " ".join(summary_parts)

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

        axis = f" by targeting the {best_dim} axis" if best_dim else ""
        summary_parts.append(
            f"Expected outcome lift{axis}: +{expected_lift_pp * 100:.1f} percentage "
            f"points (~{total_lift:.0f} incremental patients on the best single "
            "segmentation axis — NOT summed across overlapping dimensions, which "
            "would double-count patients)."
        )

        return " ".join(summary_parts)
