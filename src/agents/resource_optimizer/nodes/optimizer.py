"""
E2I Resource Optimizer Agent - Optimizer Node
Version: 4.2
Purpose: Core optimization engine for resource allocation
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..response_model import problem_gamma, target_response_value
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
                problem,
            )

            optimization_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Optimization complete: status={result['status']}, "
                f"objective={result['objective']:.2f}, time={optimization_time}ms"
            )

            warnings = list(state.get("warnings") or [])
            underspend_note = self._underspend_note(problem, result["x"])
            if underspend_note:
                warnings.append(underspend_note)

            return {
                **state,
                "optimal_allocations": allocations,
                "objective_value": float(result["objective"]),
                "solver_status": "optimal",
                "solve_time_ms": result.get("solve_time_ms", 0),
                "optimization_latency_ms": optimization_time,
                "warnings": warnings,
                "status": "analyzing" if state.get("run_scenarios") else "projecting",
            }

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                **state,
                "errors": [{"node": "optimizer", "error": str(e)}],
                # Set required output defaults on failure
                "optimal_allocations": state.get("optimal_allocations", []),
                "objective_value": state.get("objective_value", 0.0),
                "solver_status": "failed",
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
                PULP_CBC_CMD,
                LpMaximize,
                LpProblem,
                LpStatus,
                LpVariable,
                lpSum,
                value,
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
            for j, (row, b) in enumerate(zip(problem["a_ub"], problem["b_ub"], strict=False)):
                prob += lpSum(row[i] * x[i] for i in range(n)) <= b, f"ineq_{j}"

        # Add equality constraints (A_eq @ x == b_eq)
        if problem.get("a_eq") and problem.get("b_eq"):
            for j, (row, b) in enumerate(zip(problem["a_eq"], problem["b_eq"], strict=False)):
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
        """Solve the (possibly concave) allocation problem with SLSQP.

        The objective routes through the shared response model
        (``response_model.py``): with a concave power curve the optimum is
        interior — marginal returns get equalized across entities — instead
        of the bang-bang all-at-bounds solution a linear objective produces.

        Per-objective shapes (all in outcome units from the same curve):
        - maximize_outcome: max sum_i f_i(x_i)
        - maximize_roi:     max sum_i f_i(x_i) − hurdle * sum_i x_i, with the
                            hurdle priced at the portfolio's CURRENT average
                            marginal return. This is classic ROI equalization:
                            each territory is funded until its marginal return
                            meets the hurdle, so above-average-productivity
                            territories grow and below-average ones shrink.
                            (A literal outcome/spend ratio is NOT used: any
                            concave response has monotonically decreasing
                            average product, so ratio-max always collapses to
                            the minimum allowed spend — a wall of at-the-bound
                            cuts, not an allocation strategy.)
        - minimize_cost:    min sum_i x_i  s.t. outcome(x) >= outcome(current)
                            (cheapest allocation that keeps today's outcome)
        - balance:          max outcome ratio − penalty on large relative
                            reallocations (prefers smaller moves unless the
                            outcome gain justifies them)
        """
        start = time.time()

        try:
            import numpy as np
            from scipy.optimize import minimize

            targets = problem["targets"]
            gamma = problem_gamma(problem)
            objective_name = problem.get("objective", "maximize_outcome")
            n = problem["n"]

            # --- Scaling ----------------------------------------------------
            # Allocations are O(1e5-1e6) while outcome coefficients can be
            # O(1e-3): unscaled, the objective's gradient entries fall below
            # SLSQP's termination tolerance and the solver "converges" at the
            # starting point without moving. Solve in z = x / scale (z ~ 1 at
            # the current allocation) with an O(1)-normalized objective.
            cur = np.array([t.get("current_allocation", 0.0) or 0.0 for t in targets])
            scale = np.where(cur > 0, cur, 1.0)

            def to_x(z: Any) -> Any:
                return np.asarray(z) * scale

            def outcome(x: Any) -> float:
                return float(
                    sum(
                        target_response_value(t, xi, gamma)
                        for t, xi in zip(targets, x, strict=False)
                    )
                )

            current_outcome = outcome(cur)
            outcome_norm = current_outcome if current_outcome > 0 else 1.0
            total_scale = float(np.sum(scale))

            if objective_name == "maximize_roi":
                # ROI equalization: net value with capital priced at the
                # portfolio's current average MARGINAL return. For the power
                # curve the current marginal in territory i is gamma * r_i, so
                # the allocation-weighted average marginal is
                # gamma * outcome(cur) / spend(cur).
                marginal_gamma = gamma if gamma is not None else 1.0
                hurdle = marginal_gamma * outcome_norm / total_scale

                def objective(z: Any) -> float:
                    x = to_x(z)
                    return -(outcome(x) - hurdle * float(np.sum(x))) / outcome_norm

            elif objective_name == "minimize_cost":

                def objective(z: Any) -> float:
                    return float(np.sum(to_x(z))) / total_scale

            elif objective_name == "balance":
                # Outcome ratio minus a mean-squared relative-change penalty:
                # dimensionless on both sides so the trade-off is scale-free.
                safe_cur = np.where(cur > 0, cur, 1.0)
                balance_lambda = 0.5

                def objective(z: Any) -> float:
                    x = to_x(z)
                    rel_change = (x - cur) / safe_cur
                    penalty = balance_lambda * float(np.mean(rel_change**2))
                    return -(outcome(x) / outcome_norm) + penalty

            else:  # maximize_outcome

                def objective(z: Any) -> float:
                    return -outcome(to_x(z)) / outcome_norm

            lb = np.asarray(problem["lb"], dtype=float) / scale
            ub_raw = np.asarray(
                [u if u != float("inf") else np.inf for u in problem["ub"]], dtype=float
            )
            ub = ub_raw / scale
            bounds = list(zip(lb.tolist(), ub.tolist(), strict=False))

            # Build constraints (rows stay linear in z: row . (scale * z))
            constraints = []
            if problem["a_ub"]:
                a_ub = np.array(problem["a_ub"]) * scale
                b_ub = np.array(problem["b_ub"])
                for i in range(len(b_ub)):
                    norm = max(abs(b_ub[i]), 1.0)
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda z, i=i, a=a_ub, b=b_ub, nrm=norm: (b[i] - np.dot(a[i], z))
                            / nrm,
                        }
                    )
            if problem.get("a_eq"):
                a_eq = np.array(problem["a_eq"]) * scale
                b_eq = np.array(problem["b_eq"])
                for i in range(len(b_eq)):
                    norm = max(abs(b_eq[i]), 1.0)
                    constraints.append(
                        {
                            "type": "eq",
                            "fun": lambda z, i=i, a=a_eq, b=b_eq, nrm=norm: (b[i] - np.dot(a[i], z))
                            / nrm,
                        }
                    )
            if objective_name == "minimize_cost":
                # Keep (at least) the current outcome while cutting spend.
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda z: (outcome(to_x(z)) - current_outcome) / outcome_norm,
                    }
                )

            # Start from the current allocation clipped into bounds; scale down
            # first if the budget row is violated so SLSQP starts near-feasible.
            z0 = np.clip(np.ones(n), lb, ub)
            if problem["a_ub"]:
                for row, b in zip(problem["a_ub"], problem["b_ub"], strict=False):
                    used = float(np.dot(np.asarray(row) * scale, z0))
                    if used > b > 0:
                        z0 = np.clip(z0 * (b / used), lb, ub)

            result = minimize(
                objective,
                z0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )

            solve_time = int((time.time() - start) * 1000)

            if not result.success:
                return {
                    "status": "failed",
                    "x": None,
                    "objective": None,
                    "solve_time_ms": solve_time,
                }

            x_opt = to_x(result.x).tolist()
            # Report the projected outcome as the objective value regardless of
            # the internal objective shape (ratio/penalized forms are solver
            # internals; the outcome is the number every downstream consumer
            # reads).
            return {
                "status": "optimal",
                "x": x_opt,
                "objective": outcome(x_opt),
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

    @staticmethod
    def _underspend_note(problem: Dict[str, Any], x: List[float]) -> Optional[str]:
        """Honest note when maximize_roi intentionally leaves budget unallocated.

        The hurdle objective declines to spend dollars whose marginal return is
        below the current-average marginal — economically correct, but a silent
        underspend lets downstream narratives claim the full budget is "under
        optimization" when it isn't. minimize_cost underspends BY DESIGN (that
        is its objective), so no note is emitted for it.
        """
        if problem.get("objective") != "maximize_roi":
            return None
        if not problem.get("b_ub"):
            return None
        budget = float(problem["b_ub"][0])
        spend = float(sum(x))
        if budget <= 0 or spend >= budget * 0.995:
            return None
        return (
            f"maximize_roi deployed ${spend:,.0f} of the ${budget:,.0f} budget; "
            f"${budget - spend:,.0f} was intentionally left unallocated because its "
            f"marginal return falls below the hurdle rate (the portfolio's current "
            f"average marginal return) — spending it would reduce ROI."
        )

    def _build_allocations(
        self,
        x: List[float],
        targets: List[Dict[str, Any]],
        problem: Dict[str, Any],
    ) -> List[AllocationResult]:
        """Build allocation results from solution.

        ``expected_impact`` is the projected OUTCOME at the optimized
        allocation, computed from the shared response model — the same units
        for every objective. (It was previously ``c[i] * x[i]``, which for
        maximize_roi used current-allocation-normalized coefficients and
        produced dimensionless solver internals that the UI then displayed
        as if they meant something.)
        """
        allocations = []
        gamma = problem_gamma(problem)

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
                    change_percentage=(
                        (change / current * 100)
                        if current > 0
                        else (None if optimized > 0 else 0.0)
                    ),
                    expected_impact=float(target_response_value(target, optimized, gamma)),
                )
            )

        # Sort by change magnitude
        allocations.sort(key=lambda a: abs(a["change"]), reverse=True)

        return allocations
