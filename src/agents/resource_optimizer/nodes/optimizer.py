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
        """
        Solve mixed-integer linear programming using PuLP.

        Supports:
        - Continuous, integer, and binary decision variables
        - Budget and capacity constraints
        - Cardinality constraints (min/max entities to select)
        - Fixed costs for binary selection
        - Discrete allocation units
        """
        start = time.time()

        try:
            from pulp import (
                LpMaximize,
                LpProblem,
                LpStatus,
                LpVariable,
                lpSum,
                value,
                PULP_CBC_CMD,
            )
        except ImportError:
            logger.warning("PuLP not available, falling back to linear solver")
            return self._solve_linear(problem)

        n = problem["n"]
        c = problem["c"]
        lb = problem["lb"]
        ub = problem["ub"]
        var_types = problem.get("var_types", ["continuous"] * n)
        fixed_costs = problem.get("fixed_costs", [0.0] * n)
        min_entities = problem.get("min_entities")
        max_entities = problem.get("max_entities")

        # Create the problem
        prob = LpProblem("ResourceOptimization", LpMaximize)

        # Create decision variables
        x = []  # Allocation variables
        y = []  # Binary selection variables (for cardinality constraints)

        for i in range(n):
            var_type = var_types[i]

            if var_type == "binary":
                # Binary variable: 0 or 1
                var = LpVariable(f"x_{i}", cat="Binary")
            elif var_type == "integer":
                # Integer variable with bounds
                var = LpVariable(
                    f"x_{i}",
                    lowBound=lb[i],
                    upBound=ub[i] if ub[i] != float("inf") else None,
                    cat="Integer",
                )
            else:
                # Continuous variable with bounds
                var = LpVariable(
                    f"x_{i}",
                    lowBound=lb[i],
                    upBound=ub[i] if ub[i] != float("inf") else None,
                    cat="Continuous",
                )
            x.append(var)

            # Add binary selection indicator if cardinality constraints exist
            if min_entities is not None or max_entities is not None:
                y_var = LpVariable(f"y_{i}", cat="Binary")
                y.append(y_var)

        # Objective function: maximize total response minus fixed costs
        if any(fc > 0 for fc in fixed_costs) and y:
            # Include fixed costs in objective
            prob += lpSum(c[i] * x[i] - fixed_costs[i] * y[i] for i in range(n))
        else:
            prob += lpSum(c[i] * x[i] for i in range(n))

        # Add inequality constraints (A_ub @ x <= b_ub)
        if problem.get("a_ub") and problem.get("b_ub"):
            for j, (row, b) in enumerate(zip(problem["a_ub"], problem["b_ub"])):
                prob += lpSum(row[i] * x[i] for i in range(n)) <= b, f"ineq_{j}"

        # Add equality constraints (A_eq @ x == b_eq)
        if problem.get("a_eq") and problem.get("b_eq"):
            for j, (row, b) in enumerate(zip(problem["a_eq"], problem["b_eq"])):
                prob += lpSum(row[i] * x[i] for i in range(n)) == b, f"eq_{j}"

        # Add cardinality constraints (link allocation to selection)
        if y:
            for i in range(n):
                # If entity is selected (y[i]=1), allocation can be positive
                # If not selected (y[i]=0), allocation must be 0
                big_m = ub[i] if ub[i] != float("inf") else 1e6
                prob += x[i] <= big_m * y[i], f"link_upper_{i}"
                # Ensure minimum allocation if selected
                if lb[i] > 0:
                    prob += x[i] >= lb[i] * y[i], f"link_lower_{i}"

            # Min/max entities constraints
            if min_entities is not None:
                prob += lpSum(y[i] for i in range(n)) >= min_entities, "min_entities"
            if max_entities is not None:
                prob += lpSum(y[i] for i in range(n)) <= max_entities, "max_entities"

        # Solve with CBC solver (no output)
        try:
            solver = PULP_CBC_CMD(msg=0)
            prob.solve(solver)
        except Exception:
            # Fallback to default solver
            prob.solve()

        solve_time = int((time.time() - start) * 1000)

        # Check solution status
        status = LpStatus[prob.status]

        if status == "Optimal":
            # Extract solution values
            solution = [value(x[i]) or 0.0 for i in range(n)]

            # For binary variables, ensure they're exactly 0 or 1
            for i in range(n):
                if var_types[i] == "binary":
                    solution[i] = round(solution[i])

            # For integer variables, round to nearest integer
            for i in range(n):
                if var_types[i] == "integer":
                    solution[i] = round(solution[i])

            objective_value = value(prob.objective)

            return {
                "status": "optimal",
                "x": solution,
                "objective": objective_value,
                "solve_time_ms": solve_time,
                "solver": "pulp_cbc",
            }
        elif status == "Infeasible":
            return {
                "status": "infeasible",
                "x": None,
                "objective": None,
                "solve_time_ms": solve_time,
            }
        else:
            # Suboptimal, unbounded, or other status
            return {
                "status": status.lower(),
                "x": None,
                "objective": None,
                "solve_time_ms": solve_time,
            }

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
