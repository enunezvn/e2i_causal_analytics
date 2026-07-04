"""Response-model unit tests + the interior-optimum guard for the SLSQP solver.

The guard encodes WHY the concave model exists: with a linear response the
optimum is always bang-bang (every entity at a bound), which rendered as a
wall of +/-50% changes on the dashboard. If a future change reverts the
default to linear (or breaks the solver scaling so SLSQP stalls at x0), these
tests fail.
"""

import pytest

from src.agents.resource_optimizer.nodes.optimizer import OptimizerNode
from src.agents.resource_optimizer.nodes.problem_formulator import ProblemFormulatorNode
from src.agents.resource_optimizer.response_model import (
    DEFAULT_GAMMA,
    problem_gamma,
    response_marginal,
    response_value,
)

# Real-shaped synthetic seeding: TRx-per-notional-dollar responses across
# territories with different productivity (as seeded from territory_metrics).
_TERRITORIES = [
    ("south-T02", 200, 535),
    ("south-T05", 195, 508),
    ("northeast-T07", 115, 356),
    ("northeast-T08", 113, 278),
    ("midwest-T06", 111, 297),
    ("west-T03", 105, 280),
    ("west-T07", 92, 235),
]


def _targets():
    targets = []
    for tid, hcp, trx in _TERRITORIES:
        cur = hcp * 1500.0
        targets.append(
            {
                "entity_id": tid,
                "entity_type": "territory",
                "current_allocation": cur,
                "min_allocation": cur * 0.5,
                "max_allocation": cur * 1.5,
                "expected_response": trx / cur,
            }
        )
    return targets


class TestResponseModel:
    def test_linear_when_gamma_none(self):
        assert response_value(2.0, 100.0, 50.0, None) == 100.0
        assert response_marginal(2.0, 100.0, 50.0, None) == 2.0

    def test_power_curve_anchored_at_current(self):
        # f(cur) == r * cur for every gamma — the curve pivots on the current point.
        for gamma in (0.3, 0.6, 0.9):
            assert response_value(2.0, 100.0, 100.0, gamma) == pytest.approx(200.0)

    def test_power_curve_diminishing_returns(self):
        # Above current: concave curve is BELOW the linear extrapolation.
        assert response_value(2.0, 100.0, 150.0, 0.6) < 2.0 * 150.0
        # Below current: concave curve is ABOVE the linear one (first dollars count more).
        assert response_value(2.0, 100.0, 50.0, 0.6) > 2.0 * 50.0
        # Marginal decreases with allocation.
        m_low = response_marginal(2.0, 100.0, 50.0, 0.6)
        m_cur = response_marginal(2.0, 100.0, 100.0, 0.6)
        m_high = response_marginal(2.0, 100.0, 150.0, 0.6)
        assert m_low > m_cur > m_high > 0

    def test_fallback_to_linear_on_bad_inputs(self):
        # Non-positive current allocation cannot anchor the curve.
        assert response_value(2.0, 0.0, 50.0, 0.6) == 100.0
        # Out-of-range gamma is ignored.
        assert response_value(2.0, 100.0, 50.0, 1.5) == 100.0

    def test_problem_gamma_extraction(self):
        assert problem_gamma({"response_model": {"type": "power", "gamma": 0.6}}) == 0.6
        assert problem_gamma({"response_model": {"type": "linear"}}) is None
        assert problem_gamma({}) is None
        assert problem_gamma(None) is None


class TestFormulatorResponseModelSelection:
    def test_default_is_concave_nonlinear(self):
        node = ProblemFormulatorNode()
        problem = node._build_problem(
            _targets(),
            [{"constraint_type": "budget", "value": 1_000_000.0}],
            "maximize_roi",
            requested_solver=None,
        )
        assert problem["response_model"] == {"type": "power", "gamma": DEFAULT_GAMMA}
        assert node._select_solver(problem, None) == "nonlinear"

    def test_explicit_linear_keeps_lp_semantics(self):
        node = ProblemFormulatorNode()
        problem = node._build_problem(
            _targets(),
            [{"constraint_type": "budget", "value": 1_000_000.0}],
            "maximize_outcome",
            requested_solver="linear",
        )
        assert problem["response_model"] == {"type": "linear"}
        assert node._select_solver(problem, "linear") == "linear"

    def test_discrete_vars_force_linear_response(self):
        node = ProblemFormulatorNode()
        targets = _targets()
        targets[0]["is_integer"] = True
        problem = node._build_problem(
            targets,
            [{"constraint_type": "budget", "value": 1_000_000.0}],
            "maximize_outcome",
            requested_solver=None,
        )
        assert problem["response_model"] == {"type": "linear"}
        assert node._select_solver(problem, None) == "milp"


class TestInteriorOptimum:
    """The whole point of the concave model: no more ±50% walls."""

    def _solve(self, objective: str):
        formulator = ProblemFormulatorNode()
        targets = _targets()
        budget = sum(t["current_allocation"] for t in targets)
        problem = formulator._build_problem(
            targets,
            [{"constraint_type": "budget", "value": budget}],
            objective,
            requested_solver=None,
        )
        node = OptimizerNode()
        result = node._solve_nonlinear(problem)
        assert result["status"] == "optimal"
        allocations = node._build_allocations(result["x"], targets, problem)
        return targets, budget, allocations

    def test_roi_solution_is_differentiated_not_bang_bang(self):
        targets, budget, allocations = self._solve("maximize_roi")

        at_bounds = [a for a in allocations if abs(abs(a["change_percentage"]) - 50.0) < 0.01]
        assert len(at_bounds) < len(allocations), (
            "every territory landed exactly at its ±50% bound — the bang-bang "
            "LP artifact the concave response model exists to prevent"
        )
        # The solver actually moved money (it must not stall at the start point)...
        assert any(abs(a["change_percentage"]) > 1.0 for a in allocations)
        # ...and produced DIFFERENT relative changes for different territories.
        changes = sorted(round(a["change_percentage"], 1) for a in allocations)
        assert len(set(changes)) >= 3

    def test_roi_solution_respects_budget(self):
        targets, budget, allocations = self._solve("maximize_roi")
        spent = sum(a["optimized_allocation"] for a in allocations)
        assert spent <= budget * 1.001

    def test_roi_moves_money_toward_higher_productivity(self):
        targets, budget, allocations = self._solve("maximize_roi")
        by_id = {a["entity_id"]: a for a in allocations}
        r = {t["entity_id"]: t["expected_response"] for t in targets}
        best = max(r, key=lambda k: r[k])
        worst = min(r, key=lambda k: r[k])
        assert by_id[best]["change_percentage"] > by_id[worst]["change_percentage"]

    def test_minimize_cost_keeps_current_outcome(self):
        targets, budget, allocations = self._solve("minimize_cost")
        current_outcome = sum(
            t["expected_response"]
            * t["current_allocation"]
            * (1.0)  # curve anchored at current: f(cur) = r*cur
            for t in targets
        )
        projected = sum(a["expected_impact"] for a in allocations)
        spent = sum(a["optimized_allocation"] for a in allocations)
        assert projected >= current_outcome * 0.999
        assert spent < sum(t["current_allocation"] for t in targets)

    def test_expected_impact_is_outcome_units(self):
        """expected_impact must be the response-curve outcome (TRx-equivalents),
        comparable to the territory's real activity — not c[i]*x[i] solver
        internals."""
        targets, budget, allocations = self._solve("maximize_outcome")
        trx_by_id = {tid: trx for tid, _, trx in _TERRITORIES}
        for a in allocations:
            real_trx = trx_by_id[a["entity_id"]]
            # Within ±50% allocation moves, the concave outcome stays within
            # ~±30% of the anchored (current) outcome.
            assert 0.6 * real_trx < a["expected_impact"] < 1.4 * real_trx


def _skewed_targets():
    """One highly productive territory + two duds (codex review repro):
    the star hits its 1.5x cap, and the hurdle objective declines to redeploy
    the freed-up money into below-hurdle territories."""
    targets = []
    for tid, r in [("south-T01", 10.0), ("west-T01", 1.0), ("west-T02", 1.0)]:
        targets.append(
            {
                "entity_id": tid,
                "entity_type": "territory",
                "current_allocation": 100.0,
                "min_allocation": 50.0,
                "max_allocation": 150.0,
                "expected_response": r,
            }
        )
    return targets


class TestUnderspendHonesty:
    """maximize_roi may intentionally leave budget unallocated (marginal return
    below the hurdle). That is economically correct — but the run must SAY so,
    or downstream narratives claim the full budget was deployed."""

    def _formulate(self, targets, budget, objective="maximize_roi"):
        return ProblemFormulatorNode()._build_problem(
            targets,
            [{"constraint_type": "budget", "value": budget}],
            objective,
            requested_solver=None,
        )

    def test_skewed_inputs_underspend_and_emit_note(self):
        problem = self._formulate(_skewed_targets(), 300.0)
        node = OptimizerNode()
        result = node._solve_nonlinear(problem)
        assert result["status"] == "optimal"
        spend = sum(result["x"])
        assert spend < 300.0 * 0.995  # the skew genuinely produces underspend
        note = node._underspend_note(problem, result["x"])
        assert note is not None
        assert "unallocated" in note
        assert f"${300.0 - spend:,.0f}" in note

    def test_realistic_inputs_deploy_full_budget_no_note(self):
        targets = _targets()
        budget = sum(t["current_allocation"] for t in targets)
        problem = self._formulate(targets, budget)
        node = OptimizerNode()
        result = node._solve_nonlinear(problem)
        assert result["status"] == "optimal"
        assert sum(result["x"]) == pytest.approx(budget, rel=5e-3)
        assert node._underspend_note(problem, result["x"]) is None

    def test_note_is_roi_only(self):
        # minimize_cost underspends BY DESIGN — no note for it.
        assert (
            OptimizerNode._underspend_note({"objective": "minimize_cost", "b_ub": [300.0]}, [100.0])
            is None
        )
        assert (
            OptimizerNode._underspend_note(
                {"objective": "maximize_roi", "b_ub": [300.0]}, [100.0, 100.0]
            )
            is not None
        )
        # Within 0.5% of budget counts as fully deployed (solver tolerance).
        assert (
            OptimizerNode._underspend_note(
                {"objective": "maximize_roi", "b_ub": [300.0]}, [150.0, 149.5]
            )
            is None
        )


class TestZeroCurrentAllocation:
    """A new allocation funded from zero current must not render as a 0%
    change — a percentage of zero is undefined, and 0% reads as 'no move'."""

    def test_new_allocation_reports_null_percentage(self):
        targets = [
            {
                "entity_id": "new-T01",
                "entity_type": "territory",
                "current_allocation": 0.0,
                "min_allocation": 0.0,
                "max_allocation": 100.0,
                "expected_response": 1.0,
            },
            {
                "entity_id": "old-T01",
                "entity_type": "territory",
                "current_allocation": 100.0,
                "min_allocation": 50.0,
                "max_allocation": 150.0,
                "expected_response": 1.0,
            },
        ]
        allocations = OptimizerNode()._build_allocations(
            [50.0, 50.0], targets, {"objective": "maximize_roi"}
        )
        by_id = {a["entity_id"]: a for a in allocations}
        assert by_id["new-T01"]["change_percentage"] is None
        assert by_id["new-T01"]["change"] == pytest.approx(50.0)
        assert by_id["old-T01"]["change_percentage"] == pytest.approx(-50.0)

    def test_zero_current_zero_optimized_is_honest_zero(self):
        targets = [
            {
                "entity_id": "empty-T01",
                "entity_type": "territory",
                "current_allocation": 0.0,
                "min_allocation": 0.0,
                "max_allocation": 10.0,
                "expected_response": 1.0,
            }
        ]
        allocations = OptimizerNode()._build_allocations([0.0], targets, {})
        assert allocations[0]["change_percentage"] == 0.0
