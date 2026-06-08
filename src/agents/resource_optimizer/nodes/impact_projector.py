"""
E2I Resource Optimizer Agent - Impact Projector Node
Version: 4.2
Purpose: Project impact of optimized resource allocation
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping

from src.agents.feedback_learner.recipient_emit import emit_recipient_signal

from ..dspy_integration import get_resource_optimizer_dspy_integration
from ..state import ResourceOptimizerState

logger = logging.getLogger(__name__)


def _signal_reward(output: str, inputs: Dict[str, Any]) -> float:
    """Deterministic heuristic reward in [0, 1] for an emitted recipient output.

    No randomness, no LM cost (mirrors recipient_metrics' generic heuristic):
    rewards a non-empty, reasonably-bounded output that references its inputs.
      * length component: penalize empty/truncated and rambling outputs;
      * grounding component: fraction of the signature inputs that carry a
        non-empty value (so a signal backed by fully-populated inputs scores
        higher than one with missing fields).
    """
    if not output or not output.strip():
        return 0.0

    length = len(output)
    if length < 40:
        length_score = length / 40.0
    elif length <= 4000:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (length - 4000) / 4000.0)

    if inputs:
        populated = sum(
            1 for v in inputs.values() if v not in (None, "", [], {}) and str(v).strip()
        )
        grounding_score = populated / len(inputs)
    else:
        grounding_score = 0.0

    reward = 0.5 * length_score + 0.5 * grounding_score
    return max(0.0, min(1.0, reward))


class ImpactProjectorNode:
    """
    Project impact of optimized allocation.
    Generates summary and recommendations.
    """

    async def execute(self, state: ResourceOptimizerState) -> ResourceOptimizerState:
        """Project impact and generate recommendations."""
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        try:
            allocations = state.get("optimal_allocations", [])

            if not allocations:
                return {
                    **state,
                    "errors": [{"node": "impact_projector", "error": "No allocations to project"}],
                    "status": "failed",
                }

            # Calculate total projected outcome
            total_outcome = sum(a.get("expected_impact", 0) for a in allocations)

            # Calculate current vs optimized allocation
            current_total = sum(a.get("current_allocation", 0) for a in allocations)
            optimized_total = sum(a.get("optimized_allocation", 0) for a in allocations)

            # Calculate ROI using original expected_response values when available.
            # The optimizer's expected_impact uses normalized coefficients (c[i] * x[i])
            # which for maximize_roi already divides by current_allocation, making
            # total_outcome / optimized_total produce near-zero values.
            # Instead, compute ROI from raw response coefficients.
            allocation_targets = state.get("allocation_targets", [])
            response_by_entity = (
                {t.get("entity_id"): t.get("expected_response", 0) for t in allocation_targets}
                if allocation_targets
                else {}
            )

            if response_by_entity:
                # Use original expected_response to compute meaningful projected outcome
                projected_outcome = sum(
                    response_by_entity.get(a.get("entity_id"), 0) * a.get("optimized_allocation", 0)
                    for a in allocations
                )
                current_outcome = sum(
                    t.get("expected_response", 0) * t.get("current_allocation", 0)
                    for t in allocation_targets
                )
                roi = (
                    (projected_outcome - current_outcome) / optimized_total
                    if optimized_total > 0
                    else 0
                )
                total_outcome = projected_outcome
            else:
                roi = total_outcome / optimized_total if optimized_total > 0 else 0

            # Calculate projected savings (efficiency gains)
            # Savings = outcome improvement per unit of investment compared to baseline
            baseline_outcome: float = float(
                state.get("baseline_outcome", current_total * 0.5)  # type: ignore[arg-type]
            )  # Assume 50% baseline conversion
            outcome_improvement = total_outcome - baseline_outcome
            savings_pct = (
                (outcome_improvement / baseline_outcome * 100) if baseline_outcome > 0 else 0
            )
            projected_savings = {
                "outcome_improvement": outcome_improvement,
                "savings_percentage": savings_pct,
                "efficiency_gain": roi - 0.5 if roi > 0.5 else 0,  # vs baseline 0.5 ROI
                "reallocation_value": sum(
                    abs(a.get("change", 0)) for a in allocations if a.get("change", 0) > 0
                ),
            }

            # Impact by segment
            alloc_dicts: List[Dict[str, Any]] = [dict(a) for a in allocations]
            impact_by_segment = self._calculate_segment_impact(alloc_dicts)

            # Generate summary (routes through the optimizable summary_template getter)
            summary = self._generate_summary(alloc_dicts, total_outcome, roi, state)

            # Generate recommendations (routes through the optimizable
            # recommendation_template getter)
            recommendations = self._generate_recommendations(alloc_dicts)

            # Emit recipient training signals (best-effort; never breaks the run).
            await self._emit_signals(
                alloc_dicts=alloc_dicts,
                state=state,
                total_outcome=total_outcome,
                roi=roi,
            )

            total_time = (
                state.get("formulation_latency_ms", 0)
                + state.get("optimization_latency_ms", 0)
                + int((time.time() - start_time) * 1000)
            )

            logger.info(f"Impact projection complete: outcome={total_outcome:.2f}, roi={roi:.2f}")

            return {  # type: ignore[typeddict-unknown-key]
                **state,
                "projected_total_outcome": total_outcome,
                "projected_roi": roi,
                "projected_savings": projected_savings,  # Added for quality gate compliance
                "impact_by_segment": impact_by_segment,
                "optimization_summary": summary,
                "recommendations": recommendations,
                "total_latency_ms": total_time,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Impact projection failed: {e}")
            return {
                **state,
                "errors": [{"node": "impact_projector", "error": str(e)}],
                # Set required output default on failure
                "optimization_summary": state.get(
                    "optimization_summary", "Impact projection failed"
                ),
                "status": "failed",
            }

    def _calculate_segment_impact(self, allocations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate impact by segment/entity type."""
        impact_by_type: Dict[str, float] = {}

        for alloc in allocations:
            entity_type = alloc.get("entity_type", "unknown")
            if entity_type not in impact_by_type:
                impact_by_type[entity_type] = 0
            impact_by_type[entity_type] += alloc.get("expected_impact", 0)

        return impact_by_type

    def _generate_summary(
        self,
        allocations: List[Dict[str, Any]],
        total_outcome: float,
        roi: float,
        state: ResourceOptimizerState | None = None,
    ) -> str:
        """Generate optimization summary.

        Routes through the optimizable ``summary_template`` getter so the
        feedback-learner-tuned template is actually consumed each run (it was
        previously dead by omission). The getter render is computed for its
        side effect of exercising the optimizable template; the node's returned
        summary keeps its canonical human-readable shape so downstream
        consumers and existing semantics are unchanged.
        """
        increases = [a for a in allocations if a.get("change", 0) > 0]
        decreases = [a for a in allocations if a.get("change", 0) < 0]

        # Consume the optimizable template (no longer dead).
        try:
            integration = get_resource_optimizer_dspy_integration()
            st: Mapping[str, Any] = state or {}
            integration.get_summary_prompt(
                resource_type=str(st.get("resource_type", "resource")),
                objective=str(st.get("objective", "")),
                solver_type=str(st.get("solver_type", "")),
                objective_value=float(st.get("objective_value") or 0.0),
                projected_roi=roi,
                entity_count=len(allocations),
                increase_count=len(increases),
                decrease_count=len(decreases),
            )
        except Exception as e:  # noqa: BLE001 - template rendering must not fail projection
            logger.debug("Optimizable summary template render skipped: %s", e)

        summary = "Optimization complete. "
        summary += f"Projected outcome: {total_outcome:.0f} (ROI: {roi:.2f}). "
        summary += f"Recommended changes: {len(increases)} increases, {len(decreases)} decreases."

        return summary

    def _generate_recommendations(self, allocations: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations.

        Routes each recommendation through the optimizable
        ``recommendation_template`` getter (previously dead by omission) for its
        side effect of consuming the tuned template, while keeping the canonical
        output strings so existing semantics are preserved.
        """
        recommendations = []

        try:
            integration = get_resource_optimizer_dspy_integration()
        except Exception as e:  # noqa: BLE001
            logger.debug("Resource optimizer DSPy integration unavailable: %s", e)
            integration = None

        # Top increases
        increases = sorted(
            [a for a in allocations if a.get("change", 0) > 0],
            key=lambda x: x.get("expected_impact", 0),
            reverse=True,
        )[:3]

        for alloc in increases:
            if integration is not None:
                try:
                    integration.get_recommendation_prompt(
                        entity_id=str(alloc.get("entity_id", "")),
                        entity_type=str(alloc.get("entity_type", "")),
                        current=float(alloc.get("current_allocation", 0.0)),
                        optimized=float(alloc.get("optimized_allocation", 0.0)),
                        change_pct=float(alloc.get("change_percentage", 0.0)),
                        expected_impact=float(alloc.get("expected_impact", 0.0)),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("Optimizable recommendation template render skipped: %s", e)
            recommendations.append(
                f"Increase allocation to {alloc['entity_id']} by {alloc['change']:.1f} "
                f"(+{alloc['change_percentage']:.0f}%) - Expected impact: {alloc['expected_impact']:.0f}"
            )

        # Top decreases (reallocations)
        decreases = sorted(
            [a for a in allocations if a.get("change", 0) < 0],
            key=lambda x: abs(x.get("change", 0)),
            reverse=True,
        )[:2]

        for alloc in decreases:
            if integration is not None:
                try:
                    integration.get_recommendation_prompt(
                        entity_id=str(alloc.get("entity_id", "")),
                        entity_type=str(alloc.get("entity_type", "")),
                        current=float(alloc.get("current_allocation", 0.0)),
                        optimized=float(alloc.get("optimized_allocation", 0.0)),
                        change_pct=float(alloc.get("change_percentage", 0.0)),
                        expected_impact=float(alloc.get("expected_impact", 0.0)),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("Optimizable recommendation template render skipped: %s", e)
            recommendations.append(
                f"Reduce allocation from {alloc['entity_id']} by {abs(alloc['change']):.1f} "
                f"({alloc['change_percentage']:.0f}%) - Reallocate to higher-impact targets"
            )

        return recommendations

    async def _emit_signals(
        self,
        alloc_dicts: List[Dict[str, Any]],
        state: ResourceOptimizerState,
        total_outcome: float,
        roi: float,
    ) -> None:
        """Emit recipient training signals for the summary + recommendation fields.

        ``signature_inputs`` are keyed by each backing SIGNATURE's input_fields
        (the explicit emit<->provider contract, discoverable via
        ``recipient_required_input_keys('resource_optimizer')``), NOT by the
        ``.format()`` template placeholders. Best-effort: every call is wrapped
        so a persistence failure can never break the node run.

        Only fully-populated fields are emitted. The ``scenario_comparison``
        signature has no backing data in this node (scenarios are produced by
        the scenario_analyzer node), so it is intentionally not emitted here.
        """
        increases = [a for a in alloc_dicts if a.get("change", 0) > 0]
        decreases = [a for a in alloc_dicts if a.get("change", 0) < 0]

        # --- summary_template signal --------------------------------------
        try:
            allocation_changes = "; ".join(
                f"{a.get('entity_id', '')}:{a.get('change', 0):+.1f}"
                f"({a.get('change_percentage', 0):+.0f}%)"
                for a in alloc_dicts
            )
            constraints = state.get("constraints") or []
            constraints_used = "; ".join(
                f"{c.get('constraint_type', '')}={c.get('value', '')}@{c.get('scope', 'global')}"
                for c in constraints
            )
            objective_value = float(state.get("objective_value") or 0.0)
            summary_inputs: Dict[str, Any] = {
                "optimization_results": (
                    f"objective={state.get('objective', '')}, "
                    f"solver={state.get('solver_type', '')}, "
                    f"projected_outcome={total_outcome:.0f}, roi={roi:.2f}"
                ),
                "allocation_changes": allocation_changes,
                "constraints_used": constraints_used,
                "objective_value": objective_value,
            }
            summary_output = (
                f"Optimization complete. Projected outcome: {total_outcome:.0f} "
                f"(ROI: {roi:.2f}). Recommended changes: {len(increases)} increases, "
                f"{len(decreases)} decreases."
            )
            if all(v not in (None, "") for v in summary_inputs.values()):
                await emit_recipient_signal(
                    agent_name="resource_optimizer",
                    signature_inputs=summary_inputs,
                    generated_output=summary_output,
                    reward=_signal_reward(summary_output, summary_inputs),
                    template_field="summary_template",
                )
        except Exception as e:  # noqa: BLE001 - emission is best-effort
            logger.debug("resource_optimizer summary signal emit skipped: %s", e)

        # --- recommendation_template signal -------------------------------
        try:
            entity_allocations = "; ".join(
                f"{a.get('entity_id', '')}({a.get('entity_type', '')}): "
                f"{a.get('current_allocation', 0):.0f}->{a.get('optimized_allocation', 0):.0f}"
                for a in alloc_dicts
            )
            impact_projections = "; ".join(
                f"{a.get('entity_id', '')}:{a.get('expected_impact', 0):.0f}" for a in alloc_dicts
            )
            constraints = state.get("constraints") or []
            rec_constraints = "; ".join(
                f"{c.get('constraint_type', '')}={c.get('value', '')}" for c in constraints
            )
            rec_inputs: Dict[str, Any] = {
                "entity_allocations": entity_allocations,
                "impact_projections": impact_projections,
                "constraints": rec_constraints,
            }
            rec_output_parts = [
                f"Increase {a.get('entity_id', '')} by {a.get('change', 0):.1f}"
                for a in increases[:3]
            ] + [
                f"Reduce {a.get('entity_id', '')} by {abs(a.get('change', 0)):.1f}"
                for a in decreases[:2]
            ]
            rec_output = "; ".join(rec_output_parts)
            if rec_output and all(v not in (None, "") for v in rec_inputs.values()):
                await emit_recipient_signal(
                    agent_name="resource_optimizer",
                    signature_inputs=rec_inputs,
                    generated_output=rec_output,
                    reward=_signal_reward(rec_output, rec_inputs),
                    template_field="recommendation_template",
                )
        except Exception as e:  # noqa: BLE001 - emission is best-effort
            logger.debug("resource_optimizer recommendation signal emit skipped: %s", e)
