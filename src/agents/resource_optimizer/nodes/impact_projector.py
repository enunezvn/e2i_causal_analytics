"""
E2I Resource Optimizer Agent - Impact Projector Node
Version: 4.2
Purpose: Project impact of optimized resource allocation
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ..state import ResourceOptimizerState

logger = logging.getLogger(__name__)


class ImpactProjectorNode:
    """
    Project impact of optimized allocation.
    Generates summary and recommendations.
    """

    async def execute(
        self, state: ResourceOptimizerState
    ) -> ResourceOptimizerState:
        """Project impact and generate recommendations."""
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        try:
            allocations = state.get("optimal_allocations", [])

            if not allocations:
                return {
                    **state,
                    "errors": [
                        {"node": "impact_projector", "error": "No allocations to project"}
                    ],
                    "status": "failed",
                }

            # Calculate total projected outcome
            total_outcome = sum(a.get("expected_impact", 0) for a in allocations)

            # Calculate total investment
            total_allocation = sum(
                a.get("optimized_allocation", 0) for a in allocations
            )

            # Calculate ROI
            roi = total_outcome / total_allocation if total_allocation > 0 else 0

            # Impact by segment
            impact_by_segment = self._calculate_segment_impact(allocations)

            # Generate summary
            summary = self._generate_summary(allocations, total_outcome, roi)

            # Generate recommendations
            recommendations = self._generate_recommendations(allocations)

            total_time = (
                state.get("formulation_latency_ms", 0)
                + state.get("optimization_latency_ms", 0)
                + int((time.time() - start_time) * 1000)
            )

            logger.info(
                f"Impact projection complete: outcome={total_outcome:.2f}, roi={roi:.2f}"
            )

            return {
                **state,
                "projected_total_outcome": total_outcome,
                "projected_roi": roi,
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
                "status": "failed",
            }

    def _calculate_segment_impact(
        self, allocations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
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
    ) -> str:
        """Generate optimization summary."""
        increases = [a for a in allocations if a.get("change", 0) > 0]
        decreases = [a for a in allocations if a.get("change", 0) < 0]

        summary = f"Optimization complete. "
        summary += f"Projected outcome: {total_outcome:.0f} (ROI: {roi:.2f}). "
        summary += (
            f"Recommended changes: {len(increases)} increases, {len(decreases)} decreases."
        )

        return summary

    def _generate_recommendations(
        self, allocations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Top increases
        increases = sorted(
            [a for a in allocations if a.get("change", 0) > 0],
            key=lambda x: x.get("expected_impact", 0),
            reverse=True,
        )[:3]

        for alloc in increases:
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
            recommendations.append(
                f"Reduce allocation from {alloc['entity_id']} by {abs(alloc['change']):.1f} "
                f"({alloc['change_percentage']:.0f}%) - Reallocate to higher-impact targets"
            )

        return recommendations
