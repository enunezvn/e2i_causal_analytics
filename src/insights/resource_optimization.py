"""Resource-optimization strategic insight: DSPy interpretation of a solver run.

Turns the optimizer's raw output (allocation moves, projected lift, solver
status, provenance) into a business read — where the budget moves, WHY (marginal
productivity differences), what the projected gain is worth, and what the
synthetic-data caveat means for acting on it. Falls back to a deterministic
factual summary when the LM is unavailable (never fabricates)."""

from __future__ import annotations

import logging
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class ResourceOptimizationInsightSignature(dspy.Signature):
        """Interpret a resource-allocation optimization for a commercial pharma
        strategist, STRICTLY grounded in the provided numbers. Use ONLY the
        moves, lift, budget, solver facts, and causal context given; NEVER
        invent dollar amounts, territory names, or outcomes. Explain the
        reallocation STRATEGY (where resources move and the productivity logic
        driving it — money flows toward territories with higher marginal
        returns until diminishing returns equalize them), quantify what the
        projected outcome lift means relative to the current allocation, and
        judge actionability. Where `causal_context` lists modeled drivers,
        connect the moves to those levers QUALITATIVELY with their stated
        provenance — do not treat them as estimated effects of this run.
        ALWAYS close with the caveat given in `caveats` (the run is on
        clearly-labelled synthetic data: directionally meaningful, but the
        dollar values are illustrative, so validate against real budget data
        before acting).

        Write every output as PLAIN PROSE — no markdown syntax: no asterisks,
        no underscore emphasis, no backticks, no # heading markers, no
        bullet-list markers; plain numbered enumeration like "1." is fine."""

        scope: str = dspy.InputField(
            desc="Brand scope, resource type, objective, solver status, entity count"
        )
        moves: str = dspy.InputField(
            desc="Top allocation increases and decreases with % changes and amounts"
        )
        outcome: str = dspy.InputField(
            desc="Projected outcome lift vs current allocation, and total budget"
        )
        causal_context: str = dspy.InputField(
            desc=(
                "Registry-modeled causal drivers of the commercial outcome KPIs "
                "(may say none are available). Cite only drivers listed here, "
                "qualitatively, with their stated provenance; NEVER invent "
                "effects or figures beyond those given."
            )
        )
        caveats: str = dspy.InputField(desc="Data-provenance caveats that MUST be stated")

        interpretation: str = dspy.OutputField(
            desc="Business read: strategy behind the moves, expected gain, actionability"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, actionable takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    ResourceOptimizationInsightSignature = None  # type: ignore[assignment,misc]


def _fmt_move(m: dict[str, Any]) -> str:
    entity = str(m.get("entity_id", "?"))
    pct = m.get("change_percentage")
    change = m.get("change")
    parts = [entity]
    if pct is not None:
        try:
            parts.append(f"{float(pct):+.1f}%")
        except (TypeError, ValueError):
            pass
    if change is not None:
        try:
            parts.append(f"({float(change):+,.0f})")
        except (TypeError, ValueError):
            pass
    return " ".join(parts)


def build_grounding(
    objective: str,
    brand: str | None,
    resource_type: str | None,
    solver_status: str | None,
    entity_count: int | None,
    total_budget: float | None,
    projected_lift_pct: float | None,
    top_increases: list[dict[str, Any]],
    top_decreases: list[dict[str, Any]],
    synthetic: bool,
    optimization_summary: str = "",
    recommendations: list[str] | None = None,
    total_spend: float | None = None,
    causal_drivers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    brand_label = brand or "All brands"
    scope = (
        f"{brand_label} / {resource_type or 'budget'} / objective={objective or 'unknown'} / "
        f"solver={solver_status or 'unknown'} / {entity_count or 0} territories"
    )
    inc = "; ".join(_fmt_move(m) for m in top_increases[:5]) or "none"
    dec = "; ".join(_fmt_move(m) for m in top_decreases[:5]) or "none"
    moves = f"Increases: {inc}. Decreases: {dec}."
    lift_str = f"{projected_lift_pct:+.1f}%" if projected_lift_pct is not None else "—"
    budget_str = f"${total_budget:,.0f}" if total_budget else "—"
    # Underspend narration must match the objective's economics: maximize_roi
    # leaves money unallocated when its marginal return falls below the hurdle,
    # minimize_cost underspends BY DESIGN (savings while preserving outcome),
    # and any other objective gets the honest deployed-vs-budget fact with no
    # invented rationale. Saying "total budget under optimization" when the
    # optimizer declined to spend part of it would narrate money as deployed.
    spend_f = float(total_spend) if total_spend is not None else 0.0
    budget_f = float(total_budget) if total_budget is not None else 0.0
    underspend = total_spend is not None and budget_f > 0 and spend_f < budget_f * 0.995
    if underspend:
        deploy_str = f"recommends deploying ${spend_f:,.0f} of the ${budget_f:,.0f} budget"
        unspent_f = budget_f - spend_f
        if objective == "maximize_roi":
            rationale = (
                f" (${unspent_f:,.0f} intentionally unallocated — "
                f"its marginal return falls below the hurdle rate)"
            )
        elif objective == "minimize_cost":
            rationale = (
                f" (${unspent_f:,.0f} in savings while preserving the current outcome level)"
            )
        else:
            rationale = f" (${unspent_f:,.0f} left unallocated)"
        outcome = (
            f"Projected outcome lift vs current allocation: {lift_str}; {deploy_str}{rationale}."
        )
    else:
        outcome = (
            f"Projected outcome lift vs current allocation: {lift_str}; "
            f"total budget under optimization: {budget_str}."
        )
    if synthetic:
        caveats = (
            "This run used a clearly-labelled SYNTHETIC allocation problem (notional "
            "budgets seeded from territory activity; no real per-entity budget source "
            "is wired). The reallocation directions are meaningful; the dollar values "
            "are illustrative and must be validated against real budget data before "
            "acting."
        )
    else:
        caveats = (
            "Allocation targets were supplied by the caller; validate the response "
            "coefficients before acting on the projected lift."
        )
    grounding = [
        {"label": "Brand", "value": brand_label},
        {"label": "Objective", "value": str(objective or "—")},
        {"label": "Projected lift", "value": lift_str},
        {"label": "Budget", "value": budget_str},
        {"label": "Solver", "value": str(solver_status or "unknown")},
    ]
    if underspend:
        grounding.insert(4, {"label": "Deployed", "value": f"${spend_f:,.0f}"})
    # Causal-registry context (commercial grain, 2026-07-07): modeled drivers
    # of the outcome KPIs give the WHY behind the moves. Server-derived by the
    # route (same read path + provenance gate as the chatbot); absent/empty
    # degrades to an honest "none available" line, never silence-as-coverage.
    from src.insights.causal_context import format_causal_context

    drivers = causal_drivers or []
    causal_context = format_causal_context(drivers)
    if drivers:
        grounding.append({"label": "Modeled drivers", "value": f"{len(drivers)} registry chain(s)"})
    return {
        "scope": scope,
        "moves": moves,
        "outcome": outcome,
        "caveats": caveats,
        "causal_context": causal_context,
        "has_causal_context": bool(drivers),
        "grounding": grounding,
        "optimization_summary": (optimization_summary or "").strip(),
        "recommendations": normalize_list(recommendations or []),
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    """Deterministic factual summary — surfaces the agent's own output verbatim."""
    summary = g.get("optimization_summary") or ""
    if not summary:
        return {
            "insight": "No optimization narrative is available yet — run an optimization.",
            "key_takeaways": g.get("recommendations", []),
            "grounding": g["grounding"],
            "is_fallback": True,
        }
    causal_line = f" {g['causal_context']}" if g.get("has_causal_context") else ""
    insight = (
        f"{summary} Scope: {g['scope']}. {g['outcome']}{causal_line} {g['caveats']} "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": g.get("recommendations", []),
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        ResourceOptimizationInsightSignature,
        scope=g["scope"],
        moves=g["moves"],
        outcome=g["outcome"],
        causal_context=g.get(
            "causal_context", "No modeled causal drivers are available for this scope."
        ),
        caveats=g["caveats"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": normalize_list(getattr(pred, "key_takeaways", [])),
        "grounding": g["grounding"],
        "is_fallback": False,
    }
