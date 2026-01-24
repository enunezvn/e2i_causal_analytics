"""
E2I Resource Optimizer Agent - Problem Formulator Node
Version: 4.2
Purpose: Formulate optimization problem from inputs
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..state import AllocationTarget, Constraint, ResourceOptimizerState

logger = logging.getLogger(__name__)


class ProblemFormulatorNode:
    """
    Formulate optimization problem from inputs.
    Converts targets and constraints into optimization matrices.
    """

    async def execute(self, state: ResourceOptimizerState) -> ResourceOptimizerState:
        """Formulate the optimization problem."""
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        try:
            targets = state.get("allocation_targets", [])
            constraints = state.get("constraints", [])
            objective = state.get("objective", "maximize_outcome")

            # Validate inputs
            validation_errors = self._validate_inputs(targets, constraints)
            if validation_errors:
                return {
                    **state,
                    "errors": [{"node": "formulator", "error": e} for e in validation_errors],
                    "status": "failed",
                }

            # Build optimization problem representation
            problem = self._build_problem(targets, constraints, objective)

            # Determine appropriate solver
            solver_type = self._select_solver(problem, state.get("solver_type"))

            formulation_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Problem formulated: {len(targets)} targets, "
                f"{len(constraints)} constraints, solver={solver_type}"
            )

            return {
                **state,
                "_problem": problem,  # Internal use for optimizer
                "solver_type": solver_type,
                "formulation_latency_ms": formulation_time,
                "status": "optimizing",
            }

        except Exception as e:
            logger.error(f"Problem formulation failed: {e}")
            return {
                **state,
                "errors": [{"node": "formulator", "error": str(e)}],
                "status": "failed",
            }

    def _validate_inputs(
        self,
        targets: List[AllocationTarget],
        constraints: List[Constraint],
    ) -> List[str]:
        """Validate optimization inputs."""
        errors = []

        if not targets:
            errors.append("No allocation targets provided")

        # Check for budget constraint
        budget_constraints = [c for c in constraints if c.get("constraint_type") == "budget"]
        if not budget_constraints:
            errors.append("No budget constraint specified")

        # Check for negative values
        for target in targets:
            if target.get("expected_response", 0) < 0:
                errors.append(f"Negative response coefficient for {target.get('entity_id')}")

        # Check for valid entity IDs
        for target in targets:
            if not target.get("entity_id"):
                errors.append("Target missing entity_id")

        return errors

    def _build_problem(
        self,
        targets: List[AllocationTarget],
        constraints: List[Constraint],
        objective: str,
    ) -> Dict[str, Any]:
        """Build optimization problem representation."""
        n = len(targets)

        # Objective coefficients (response per unit allocation)
        if objective in ["maximize_outcome", "balance"]:
            c = [t.get("expected_response", 1.0) for t in targets]
        elif objective == "maximize_roi":
            c = [
                t.get("expected_response", 1.0) / max(t.get("current_allocation", 1), 1)
                for t in targets
            ]
        else:  # minimize_cost
            c = [-1.0] * n

        # Variable bounds
        lb = [t.get("min_allocation", 0) or 0 for t in targets]
        ub = [t.get("max_allocation") or float("inf") for t in targets]

        # Variable types for MILP
        var_types = []
        for t in targets:
            if t.get("is_binary", False):
                var_types.append("binary")
            elif t.get("is_integer", False):
                var_types.append("integer")
            else:
                var_types.append("continuous")

        # Fixed costs for binary selection (adjust objective if present)
        fixed_costs = [t.get("fixed_cost", 0.0) for t in targets]

        # Allocation units (for discrete step sizes)
        allocation_units = [t.get("allocation_unit") for t in targets]

        # Constraint matrices
        a_ub = []
        b_ub = []
        a_eq = []
        b_eq = []

        # Cardinality constraints (min/max entities to select)
        min_entities: Optional[int] = None
        max_entities: Optional[int] = None

        for constraint in constraints:
            constraint_type = constraint.get("constraint_type", "")
            value = constraint.get("value", 0)

            if constraint_type == "budget":
                # Sum of allocations <= budget
                a_ub.append([1.0] * n)
                b_ub.append(value)
            elif constraint_type == "min_total":
                # Sum of allocations >= min (negate for <=)
                a_ub.append([-1.0] * n)
                b_ub.append(-value)
            elif constraint_type == "exact_total":
                # Sum of allocations == value
                a_eq.append([1.0] * n)
                b_eq.append(value)
            elif constraint_type == "cardinality":
                # Cardinality constraint (min/max entities to select)
                min_entities = constraint.get("min_entities")
                max_entities = constraint.get("max_entities")
                if max_entities is None and value > 0:
                    max_entities = int(value)

        return {
            "c": c,
            "lb": lb,
            "ub": ub,
            "a_eq": a_eq if a_eq else None,
            "b_eq": b_eq if b_eq else None,
            "a_ub": a_ub if a_ub else None,
            "b_ub": b_ub if b_ub else None,
            "n": n,
            "targets": targets,
            "objective": objective,
            # MILP extensions
            "var_types": var_types,
            "fixed_costs": fixed_costs,
            "allocation_units": allocation_units,
            "min_entities": min_entities,
            "max_entities": max_entities,
            "has_integer_vars": any(t != "continuous" for t in var_types),
        }

    def _select_solver(self, problem: Dict[str, Any], requested: Optional[str]) -> str:
        """Select appropriate solver for the problem."""
        # Check for integer/binary variables or cardinality constraints
        has_integer = problem.get("has_integer_vars", False)
        has_cardinality = (
            problem.get("min_entities") is not None
            or problem.get("max_entities") is not None
        )

        if has_integer or has_cardinality:
            return "milp"
        elif requested:
            return requested
        else:
            return "linear"
