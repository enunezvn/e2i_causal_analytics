"""
E2I Resource Optimizer Agent - Scenario Analyzer Node
Version: 4.2
Purpose: What-if scenario analysis for resource allocation
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ..state import ResourceOptimizerState, ScenarioResult

logger = logging.getLogger(__name__)


class ScenarioAnalyzerNode:
    """
    Analyze what-if scenarios for resource allocation.
    Compares different allocation strategies.
    """

    async def execute(
        self, state: ResourceOptimizerState
    ) -> ResourceOptimizerState:
        """Execute scenario analysis."""
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        # Skip if scenarios not requested
        if not state.get("run_scenarios"):
            return {**state, "status": "projecting"}

        try:
            allocations = state.get("optimal_allocations", [])
            targets = state.get("allocation_targets", [])
            constraints = state.get("constraints", [])
            scenario_count = state.get("scenario_count", 3)

            # Generate scenarios
            scenarios = self._generate_scenarios(
                allocations, targets, constraints, scenario_count
            )

            # Perform sensitivity analysis
            sensitivity = self._analyze_sensitivity(allocations, targets)

            logger.info(
                f"Scenario analysis complete: {len(scenarios)} scenarios analyzed"
            )

            return {
                **state,
                "scenarios": scenarios,
                "sensitivity_analysis": sensitivity,
                "status": "projecting",
            }

        except Exception as e:
            logger.error(f"Scenario analysis failed: {e}")
            return {
                **state,
                "errors": [{"node": "scenario_analyzer", "error": str(e)}],
                "status": "failed",
            }

    def _generate_scenarios(
        self,
        allocations: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        count: int,
    ) -> List[ScenarioResult]:
        """Generate what-if scenarios."""
        scenarios = []

        # Get budget from constraints
        budget = None
        for c in constraints:
            if c.get("constraint_type") == "budget":
                budget = c.get("value", 0)
                break

        if budget is None:
            budget = sum(a.get("optimized_allocation", 0) for a in allocations)

        # Scenario 1: Current allocation (baseline)
        current_total = sum(
            t.get("current_allocation", 0) for t in targets
        )
        current_outcome = sum(
            t.get("current_allocation", 0) * t.get("expected_response", 1.0)
            for t in targets
        )
        scenarios.append(
            ScenarioResult(
                scenario_name="Current Allocation (Baseline)",
                total_allocation=current_total,
                projected_outcome=current_outcome,
                roi=current_outcome / current_total if current_total > 0 else 0,
                constraint_violations=[],
            )
        )

        # Scenario 2: Optimized allocation
        opt_total = sum(a.get("optimized_allocation", 0) for a in allocations)
        opt_outcome = sum(a.get("expected_impact", 0) for a in allocations)
        scenarios.append(
            ScenarioResult(
                scenario_name="Optimized Allocation",
                total_allocation=opt_total,
                projected_outcome=opt_outcome,
                roi=opt_outcome / opt_total if opt_total > 0 else 0,
                constraint_violations=[],
            )
        )

        # Scenario 3: Equal distribution
        if targets:
            equal_alloc = budget / len(targets)
            equal_outcome = sum(
                equal_alloc * t.get("expected_response", 1.0) for t in targets
            )
            scenarios.append(
                ScenarioResult(
                    scenario_name="Equal Distribution",
                    total_allocation=budget,
                    projected_outcome=equal_outcome,
                    roi=equal_outcome / budget if budget > 0 else 0,
                    constraint_violations=[],
                )
            )

        # Scenario 4: Focus on top performers
        if len(targets) > 2 and count >= 4:
            sorted_targets = sorted(
                targets, key=lambda t: t.get("expected_response", 0), reverse=True
            )
            top_half = sorted_targets[: len(sorted_targets) // 2]
            focus_alloc = budget / len(top_half) if top_half else 0
            focus_outcome = sum(
                focus_alloc * t.get("expected_response", 1.0) for t in top_half
            )
            scenarios.append(
                ScenarioResult(
                    scenario_name="Focus Top Performers",
                    total_allocation=budget,
                    projected_outcome=focus_outcome,
                    roi=focus_outcome / budget if budget > 0 else 0,
                    constraint_violations=[],
                )
            )

        return scenarios[:count]

    def _analyze_sensitivity(
        self,
        allocations: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Analyze sensitivity of allocation to changes."""
        sensitivity = {}

        for i, target in enumerate(targets):
            entity_id = target.get("entity_id", f"entity_{i}")
            response = target.get("expected_response", 1.0)

            # Sensitivity = marginal impact of 1 unit change
            sensitivity[entity_id] = response

        return sensitivity
