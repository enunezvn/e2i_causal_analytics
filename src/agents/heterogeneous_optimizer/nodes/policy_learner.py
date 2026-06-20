"""Policy Learner Node for Heterogeneous Optimizer Agent.

This node learns optimal treatment allocation policy based on CATE estimates.
Uses CATE to recommend allocation changes.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..state import HeterogeneousOptimizerState, PolicyRecommendation

logger = logging.getLogger(__name__)


class PolicyLearnerNode:
    """Learn optimal treatment allocation policy.

    Recommends treatment rate adjustments based on CATE estimates:
    - High responders (CATE >= 1.5x ATE): Increase treatment
    - Low responders (CATE <= 0.5x ATE): Decrease treatment
    - Average responders: Maintain current rate
    """

    def __init__(self, label_criteria_provider: Optional[object] = None):
        self.min_cate_for_treatment = 0.01  # Minimum CATE to recommend treatment
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

            # Generate policy recommendations
            recommendations = []

            for _segment_var, results in cate_by_segment.items():
                for result in results:
                    rec = self._generate_recommendation(dict(result), ate)  # type: ignore[arg-type]
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

            output: Dict[str, Any] = {
                **state,
                "policy_recommendations": recommendations[:20],  # Top 20
                "expected_total_lift": total_lift,
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
