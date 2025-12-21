"""
E2I Resource Optimizer Agent - Optimizer Node
Version: 4.2
Purpose: Core optimization engine for resource allocation
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ..state import AllocationResult, ResourceOptimizerState

logger = logging.getLogger(__name__)


class OptimizerNode:
    """
    Core optimization engine.
    Solves the formulated problem using appropriate solver.
    """

    async def execute(self, state: ResourceOptimizerState) -> ResourceOptimizerState:
        """Execute optimization."""
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        try:
            problem = state.get("_problem")
            if not problem:
                return {
                    **state,
                    "errors": [{"node": "optimizer", "error": "No problem formulated"}],
                    "status": "failed",
                }

            solver_type = state.get("solver_type", "linear")

            if solver_type == "linear":
                result = self._solve_linear(problem)
            elif solver_type == "milp":
                result = self._solve_milp(problem)
            else:
                result = self._solve_nonlinear(problem)

            if result["status"] != "optimal":
                warnings = list(state.get("warnings") or [])
                warnings.append(f"Solver returned: {result['status']}")
                return {
                    **state,
                    "solver_status": result["status"],
                    "warnings": warnings,
                    "status": "failed" if result["status"] == "infeasible" else "analyzing",
                }

            # Build allocation results
            allocations = self._build_allocations(
                result["x"],
                problem["targets"],
                problem["c"],
            )

            optimization_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Optimization complete: status={result['status']}, "
                f"objective={result['objective']:.2f}, time={optimization_time}ms"
            )

            return {
                **state,
                "optimal_allocations": allocations,
                "objective_value": float(result["objective"]),
                "solver_status": "optimal",
                "solve_time_ms": result.get("solve_time_ms", 0),
                "optimization_latency_ms": optimization_time,
                "status": "analyzing" if state.get("run_scenarios") else "projecting",
            }

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                **state,
                "errors": [{"node": "optimizer", "error": str(e)}],
                "status": "failed",
            }

    def _solve_linear(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve linear programming problem."""
        start = time.time()

        try:
            import numpy as np
            from scipy.optimize import linprog

            # Negate c for maximization (linprog minimizes)
            c = np.array([-x for x in problem["c"]])

            bounds = list(zip(problem["lb"], problem["ub"], strict=False))

            # Convert constraints
            a_ub = np.array(problem["a_ub"]) if problem["a_ub"] else None
            b_ub = np.array(problem["b_ub"]) if problem["b_ub"] else None
            a_eq = np.array(problem["a_eq"]) if problem["a_eq"] else None
            b_eq = np.array(problem["b_eq"]) if problem["b_eq"] else None

            result = linprog(
                c,
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            solve_time = int((time.time() - start) * 1000)

            if result.success:
                return {
                    "status": "optimal",
                    "x": result.x.tolist(),
                    "objective": -result.fun,  # Negate back
                    "solve_time_ms": solve_time,
                }
            else:
                status = "infeasible" if "infeasible" in str(result.message).lower() else "failed"
                return {
                    "status": status,
                    "x": None,
                    "objective": None,
                }

        except ImportError:
            # Fallback to simple proportional allocation
            return self._solve_proportional(problem)

    def _solve_milp(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve mixed-integer linear programming."""
        # Fall back to linear for now
        return self._solve_linear(problem)

    def _solve_nonlinear(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve nonlinear optimization."""
        start = time.time()

        try:
            import numpy as np
            from scipy.optimize import minimize

            c = np.array(problem["c"])

            def objective(x):
                return -np.dot(c, x)  # Negate for maximization

            bounds = list(zip(problem["lb"], problem["ub"], strict=False))
            x0 = [t.get("current_allocation", 1.0) for t in problem["targets"]]

            # Build constraints
            constraints = []
            if problem["a_ub"]:
                a_ub = np.array(problem["a_ub"])
                b_ub = np.array(problem["b_ub"])
                for i in range(len(b_ub)):
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, i=i, a=a_ub, b=b_ub: b[i] - np.dot(a[i], x),
                        }
                    )

            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            solve_time = int((time.time() - start) * 1000)

            return {
                "status": "optimal" if result.success else "failed",
                "x": result.x.tolist() if result.success else None,
                "objective": -result.fun if result.success else None,
                "solve_time_ms": solve_time,
            }

        except ImportError:
            return self._solve_proportional(problem)

    def _solve_proportional(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback solver using proportional allocation."""
        start = time.time()

        problem["targets"]
        n = problem["n"]
        c = problem["c"]

        # Find budget from constraints
        budget = None
        if problem["b_ub"]:
            budget = problem["b_ub"][0]  # Assume first is budget

        if budget is None:
            return {"status": "failed", "x": None, "objective": None}

        # Proportional allocation based on response coefficients
        total_response = sum(c)
        if total_response <= 0:
            # Equal allocation
            x = [budget / n] * n
        else:
            # Weighted by response
            x = [(r / total_response) * budget for r in c]

        # Apply bounds
        for i in range(n):
            x[i] = max(problem["lb"][i], min(x[i], problem["ub"][i]))

        objective = sum(c[i] * x[i] for i in range(n))

        solve_time = int((time.time() - start) * 1000)

        return {
            "status": "optimal",
            "x": x,
            "objective": objective,
            "solve_time_ms": solve_time,
        }

    def _build_allocations(
        self,
        x: List[float],
        targets: List[Dict[str, Any]],
        c: List[float],
    ) -> List[AllocationResult]:
        """Build allocation results from solution."""
        allocations = []

        for i, target in enumerate(targets):
            current = target.get("current_allocation", 0)
            optimized = float(x[i])
            change = optimized - current

            allocations.append(
                AllocationResult(
                    entity_id=target["entity_id"],
                    entity_type=target.get("entity_type", "hcp"),
                    current_allocation=current,
                    optimized_allocation=optimized,
                    change=change,
                    change_percentage=(change / current * 100) if current > 0 else 0,
                    expected_impact=float(c[i] * optimized),
                )
            )

        # Sort by change magnitude
        allocations.sort(key=lambda a: abs(a["change"]), reverse=True)

        return allocations
