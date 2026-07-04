"""
E2I Resource Optimizer Agent - Impact Projector Node
Version: 4.2
Purpose: Project impact of optimized resource allocation
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Mapping

from src.agents.feedback_learner.recipient_emit import emit_recipient_signal

from ..dspy_integration import get_resource_optimizer_dspy_integration
from ..response_model import problem_gamma, target_response_value
from ..state import ResourceOptimizerState

logger = logging.getLogger(__name__)

# Territory ids in this platform follow "<region>-T<nn>" (e.g. "south-T02");
# when they do, impact shares are grouped by region — a segmentation with more
# than one slice. Anything else falls back to grouping by entity_type.
_REGION_ID_RE = re.compile(r"^([A-Za-z]+)-T\d+$")


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

            # Projected outcome at the optimized allocation. expected_impact is
            # already the shared-response-model outcome per entity (same curve
            # the solver optimized), so the total is a straight sum.
            total_outcome = sum(a.get("expected_impact", 0) for a in allocations)

            # Baseline: outcome of the CURRENT allocation under the same curve
            # (the curve is anchored at current, so this equals
            # expected_response * current_allocation for every gamma).
            gamma = problem_gamma(state.get("_problem"))
            allocation_targets = state.get("allocation_targets", [])
            current_outcome = sum(
                target_response_value(t, t.get("current_allocation", 0) or 0.0, gamma)
                for t in allocation_targets
            )

            # "projected_roi" is the RELATIVE OUTCOME LIFT of the optimized
            # allocation vs the current one (0.083 = +8.3%). The previous
            # definition — outcome delta divided by total SPEND — mixed outcome
            # units with dollars and produced numbers with no readable meaning.
            roi = (
                (total_outcome - current_outcome) / current_outcome if current_outcome > 0 else 0.0
            )

            # Efficiency block: improvements vs the honest baseline (current
            # allocation's outcome), not vs an invented 50%-conversion floor.
            outcome_improvement = total_outcome - current_outcome
            savings_pct = (
                (outcome_improvement / current_outcome * 100) if current_outcome > 0 else 0
            )
            projected_savings = {
                "outcome_improvement": outcome_improvement,
                "savings_percentage": savings_pct,
                "efficiency_gain": roi,
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
        """Percentage share of the projected outcome by segment (sums to ~100).

        Segments are regions when entity ids follow "<region>-T<nn>", falling
        back to entity_type otherwise. The previous version summed RAW
        expected_impact keyed by entity_type — with territory-only inputs that
        was a single "territory" slice whose value was a dimensionless solver
        number, which the UI then rendered with a % suffix.
        """
        totals: Dict[str, float] = {}

        for alloc in allocations:
            m = _REGION_ID_RE.match(str(alloc.get("entity_id", "")))
            segment = m.group(1) if m else alloc.get("entity_type", "unknown")
            totals[segment] = totals.get(segment, 0.0) + alloc.get("expected_impact", 0)

        grand_total = sum(totals.values())
        if grand_total <= 0:
            return {}
        return {k: round(v / grand_total * 100, 1) for k, v in totals.items()}

    def _generate_summary(
        self,
        allocations: List[Dict[str, Any]],
        total_outcome: float,
        roi: float,
        state: ResourceOptimizerState | None = None,
    ) -> str:
        """Generate optimization summary via the optimizable ``summary_template``.

        The getter output is the RETURNED summary (so the feedback-learner-tuned
        template actually reaches the user — not just rendered and discarded).
        The default template renders the canonical "Optimization complete ..."
        sentence plus a metadata clause, so existing summary substrings hold. If
        the getter / ``.format()`` fails for any reason, fall back to the inline
        canonical string so the node never breaks.
        """
        increases = [a for a in allocations if a.get("change", 0) > 0]
        decreases = [a for a in allocations if a.get("change", 0) < 0]

        inline_summary = (
            "Optimization complete. "
            f"Projected outcome: {total_outcome:.0f} (outcome lift vs current: {roi:+.1%}). "
            f"Recommended changes: {len(increases)} increases, {len(decreases)} decreases."
        )

        # Consume the optimizable template (no longer dead) and USE its output.
        try:
            integration = get_resource_optimizer_dspy_integration()
            st: Mapping[str, Any] = state or {}
            return integration.get_summary_prompt(
                resource_type=str(st.get("resource_type", "resource")),
                objective=str(st.get("objective", "")),
                solver_type=str(st.get("solver_type", "")),
                # objective_value carries the projected outcome value the node
                # historically reported as "Projected outcome".
                objective_value=float(total_outcome),
                projected_roi=roi,
                entity_count=len(allocations),
                increase_count=len(increases),
                decrease_count=len(decreases),
            )
        except Exception as e:  # noqa: BLE001 - fall back to inline so projection never breaks
            logger.debug("Optimizable summary template render failed; using inline: %s", e)
            return inline_summary

    def _recommendation_line(
        self,
        integration: Any,
        alloc: Dict[str, Any],
        inline: str,
    ) -> str:
        """Render one recommendation line via the optimizable template.

        Returns the getter output (so the tuned ``recommendation_template`` flows
        to the user); falls back to ``inline`` if the getter/format fails.
        """
        if integration is None:
            return inline
        # A None change_percentage means "new allocation from zero current" —
        # the template requires a float and would render it as +0%, hiding the
        # move, so the inline (which spells out "new allocation") wins.
        if alloc.get("change_percentage") is None:
            return inline
        try:
            return str(
                integration.get_recommendation_prompt(
                    entity_id=str(alloc.get("entity_id", "")),
                    entity_type=str(alloc.get("entity_type", "")),
                    current=float(alloc.get("current_allocation", 0.0)),
                    optimized=float(alloc.get("optimized_allocation", 0.0)),
                    change_pct=float(alloc.get("change_percentage", 0.0)),
                    expected_impact=float(alloc.get("expected_impact", 0.0)),
                )
            )
        except Exception as e:  # noqa: BLE001 - fall back to inline so projection never breaks
            logger.debug("Optimizable recommendation template render failed; using inline: %s", e)
            return inline

    def _generate_recommendations(self, allocations: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations via the optimizable template.

        Each line is rendered by ``get_recommendation_prompt`` (the
        previously-dead getter) and RETURNED, so the feedback-learner-tuned
        ``recommendation_template`` actually reaches the user. Each line falls
        back to its inline canonical string if the getter/format fails.
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
            pct = alloc.get("change_percentage")
            pct_str = f"+{pct:.0f}%" if pct is not None else "new allocation"
            inline = (
                f"Increase allocation to {alloc['entity_id']} by {alloc['change']:,.0f} "
                f"({pct_str}) - Projected outcome at new "
                f"allocation: {alloc['expected_impact']:,.0f}"
            )
            recommendations.append(self._recommendation_line(integration, alloc, inline))

        # Top decreases (reallocations)
        decreases = sorted(
            [a for a in allocations if a.get("change", 0) < 0],
            key=lambda x: abs(x.get("change", 0)),
            reverse=True,
        )[:2]

        for alloc in decreases:
            inline = (
                f"Reduce allocation from {alloc['entity_id']} by {abs(alloc['change']):,.0f} "
                f"({(alloc.get('change_percentage') or 0.0):.0f}%) - Reallocate to "
                f"higher-impact targets"
            )
            recommendations.append(self._recommendation_line(integration, alloc, inline))

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

        Emission is guarded ONLY on truly-mandatory fields (the optimization
        results and the allocation/impact descriptions). An empty constraints
        string is a VALID signature value for an unconstrained run (a normal
        case), NOT a missing-data sentinel — guarding on it would silently drop
        every unconstrained run's training data, so it is excluded from the
        guard. The ``scenario_comparison`` signature has no backing data in this
        node (scenarios are produced by the scenario_analyzer node), so it is
        intentionally not emitted here.
        """
        increases = [a for a in alloc_dicts if a.get("change", 0) > 0]
        decreases = [a for a in alloc_dicts if a.get("change", 0) < 0]

        # --- summary_template signal --------------------------------------
        try:
            allocation_changes = "; ".join(
                f"{a.get('entity_id', '')}:{a.get('change', 0):+.1f}"
                f"({(a.get('change_percentage') or 0):+.0f}%)"
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
                f"(outcome lift vs current: {roi:+.1%}). "
                f"Recommended changes: {len(increases)} increases, "
                f"{len(decreases)} decreases."
            )
            # Guard only mandatory fields. ``constraints_used`` (empty on an
            # unconstrained run) and ``objective_value`` (may legitimately be 0)
            # are NOT guarded — an empty/zero value there is valid, not missing.
            summary_required = ("optimization_results", "allocation_changes")
            if all(summary_inputs[k] not in (None, "") for k in summary_required):
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
            # Guard only mandatory fields. ``constraints`` (empty on an
            # unconstrained run) is a valid value, not missing — excluded.
            rec_required = ("entity_allocations", "impact_projections")
            if rec_output and all(rec_inputs[k] not in (None, "") for k in rec_required):
                await emit_recipient_signal(
                    agent_name="resource_optimizer",
                    signature_inputs=rec_inputs,
                    generated_output=rec_output,
                    reward=_signal_reward(rec_output, rec_inputs),
                    template_field="recommendation_template",
                )
        except Exception as e:  # noqa: BLE001 - emission is best-effort
            logger.debug("resource_optimizer recommendation signal emit skipped: %s", e)
