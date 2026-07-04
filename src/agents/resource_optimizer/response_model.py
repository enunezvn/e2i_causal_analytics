"""
E2I Resource Optimizer Agent - Response Model
Version: 4.3
Purpose: Shared response-curve model (outcome as a function of allocation)

Every node that converts an allocation into a projected outcome MUST route
through these helpers so the solver objective, the per-entity expected_impact,
the projected totals, the scenario outcomes, and the sensitivity figures are
all computed from the SAME curve. (Before 4.3 each node had its own inline
``response * allocation`` arithmetic, which silently disagreed across
objectives — e.g. ``maximize_roi`` normalized its coefficients so the reported
"impact" values were dimensionless solver internals.)

Curve shapes
------------
linear (``gamma is None``)
    outcome(x) = r * x — constant marginal return. This is the historical
    behavior and remains what LP/MILP solvers optimize (their objectives must
    stay linear in x).

concave power (``0 < gamma < 1``)
    outcome(x) = r * cur * (x / cur) ** gamma — anchored so it passes through
    the linear value at the current allocation (outcome(cur) = r * cur), with
    diminishing returns beyond it. This models promotional-response saturation:
    the first dollars in a territory buy more outcome than the last dollars.

Why concave matters: with a LINEAR objective, box bounds (e.g. 0.5x-1.5x of
current) and one budget row, the LP optimum is always "bang-bang" — every
entity slammed to a bound except a single budget-balancing fractional one. On
the dashboard that rendered as a wall of +/-50% changes, which reads as
broken. A concave objective has an interior optimum that equalizes marginal
returns, producing differentiated, defensible reallocations.

``DEFAULT_GAMMA = 0.6`` keeps the marginal-return exponent 1/(1-gamma) = 2.5:
a 14% productivity edge between two territories translates to roughly a
1.14**2.5 ~ 1.4x allocation tilt — visible but not extreme.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

DEFAULT_GAMMA = 0.6

# Floor (as a fraction of current allocation) used when evaluating the power
# curve's marginal at x -> 0, where the true derivative diverges.
_MARGINAL_X_FLOOR = 1e-9


def response_value(
    expected_response: float,
    current_allocation: float,
    x: float,
    gamma: Optional[float] = None,
) -> float:
    """Projected outcome contribution for allocation ``x``.

    Falls back to the linear form when ``gamma`` is None, out of range, or the
    current allocation is non-positive (the power curve is anchored at the
    current allocation, so it needs cur > 0 to be well-defined).
    """
    r = float(expected_response)
    cur = float(current_allocation)
    x = max(0.0, float(x))
    if gamma is None or not (0.0 < gamma < 1.0) or cur <= 0.0:
        return r * x
    return float(r * cur * (x / cur) ** gamma)


def response_marginal(
    expected_response: float,
    current_allocation: float,
    x: float,
    gamma: Optional[float] = None,
) -> float:
    """Marginal outcome per additional allocation unit at ``x`` (d outcome / d x)."""
    r = float(expected_response)
    cur = float(current_allocation)
    if gamma is None or not (0.0 < gamma < 1.0) or cur <= 0.0:
        return r
    x = max(_MARGINAL_X_FLOOR * cur, float(x))
    return float(r * gamma * (x / cur) ** (gamma - 1.0))


def problem_gamma(problem: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract the concavity exponent from a formulated problem dict.

    Returns None (linear response) unless the problem carries a
    ``response_model`` of type "power" with a valid gamma.
    """
    if not problem:
        return None
    rm = problem.get("response_model") or {}
    if rm.get("type") != "power":
        return None
    gamma = rm.get("gamma")
    if isinstance(gamma, (int, float)) and 0.0 < float(gamma) < 1.0:
        return float(gamma)
    return None


def target_response_value(
    target: Mapping[str, Any], x: float, gamma: Optional[float] = None
) -> float:
    """``response_value`` for an AllocationTarget-shaped mapping."""
    return response_value(
        target.get("expected_response", 1.0) or 0.0,
        target.get("current_allocation", 0.0) or 0.0,
        x,
        gamma,
    )


def target_response_marginal(
    target: Mapping[str, Any], x: float, gamma: Optional[float] = None
) -> float:
    """``response_marginal`` for an AllocationTarget-shaped mapping."""
    return response_marginal(
        target.get("expected_response", 1.0) or 0.0,
        target.get("current_allocation", 0.0) or 0.0,
        x,
        gamma,
    )
