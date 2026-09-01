"""Digital-twin strategic insight: interpret twin readiness + simulation evidence.

Grounded in the brand's REAL twin-model inventory, simulation history and the
per-intervention effect coverage (the identification gate). The substrate is
synthetic-gold — the narrative must keep saying so (honesty-critical: these are
showcase capabilities, not real-world effects).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class DigitalTwinInsightSignature(dspy.Signature):
        """Interpret a pharmaceutical brand's digital-twin simulation program for a
        commercial analyst, STRICTLY grounded in the provided counts and results.
        Use ONLY the numbers and names given; NEVER invent effects, models, or
        confidence values. The effects are estimated from a SYNTHETIC gold-standard
        cohort (a capability showcase, not real-world evidence) — say so plainly.
        Explain what the simulation evidence implies about which intervention levers
        look strongest and what to pre-screen next; if history is thin, say so
        rather than over-reading it.

        Write every output as PLAIN PROSE — no markdown syntax: no asterisks,
        no underscore emphasis, no backticks, no # heading markers, no
        bullet-list markers, no numbered-list markers — write flowing prose."""

        scope: str = dspy.InputField(desc="Brand scope of this twin program view")
        model_summary: str = dspy.InputField(desc="Trained twin models available")
        simulation_summary: str = dspy.InputField(
            desc="Simulation history: totals, recommendation mix, deploy rate"
        )
        latest_result: str = dspy.InputField(
            desc="Most recent completed simulation: intervention, ATE, CI, recommendation"
        )
        intervention_coverage: str = dspy.InputField(
            desc="Which catalog interventions carry an identified causal effect"
        )

        interpretation: str = dspy.OutputField(
            desc="What the twin evidence says about intervention strategy for this brand"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 specific, grounded takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    DigitalTwinInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(
    brand: str,
    models: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
    effect_available: dict[str, bool],
    catalog: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Derive the grounded summary strings from REAL rows (no fabrication).

    ``models``/``simulations`` are raw repository rows; ``effect_available`` is
    the per-intervention identification map served by /intervention-types.
    """
    model_names = [str(m.get("model_name") or m.get("model_id") or "unnamed") for m in models]
    model_summary = (
        f"{len(models)} active twin model(s) for {brand}: " + ", ".join(model_names)
        if models
        else f"No active twin model for {brand} — simulations unavailable."
    )

    completed = [s for s in simulations if str(s.get("simulation_status") or "") == "completed"]
    rec_mix: Counter = Counter(str(s.get("recommendation") or "unknown") for s in completed)
    deploys = rec_mix.get("deploy", 0)
    deploy_rate = (deploys / len(completed)) if completed else 0.0
    simulation_summary = (
        f"{len(simulations)} simulation(s) recorded for {brand}, {len(completed)} completed; "
        f"recommendation mix: "
        + (", ".join(f"{k}={v}" for k, v in rec_mix.most_common()) or "none")
        + f"; deploy rate {deploy_rate:.0%}."
    )

    latest = completed[0] if completed else None
    if latest is not None:
        ate = latest.get("simulated_ate")
        lo = latest.get("simulated_ci_lower")
        hi = latest.get("simulated_ci_upper")
        ci = (
            f" (95% CI {float(lo):+.3f}..{float(hi):+.3f})"
            if lo is not None and hi is not None
            else ""
        )
        latest_result = (
            f"Latest completed simulation: {latest.get('intervention_type', 'unknown')} -> "
            f"ATE {float(ate):+.3f}{ci}, recommendation '{latest.get('recommendation', '?')}', "
            f"provenance {latest.get('data_provenance') or 'unknown'}."
            if ate is not None
            else "Latest completed simulation has no recorded effect estimate."
        )
    else:
        latest_result = "No completed simulation yet for this brand."

    labels = dict(catalog)
    identified = [labels.get(v, v) for v, ok in effect_available.items() if ok]
    missing = [labels.get(v, v) for v, _label in catalog if not effect_available.get(v)]
    intervention_coverage = (
        f"{len(identified)} of {len(catalog)} catalog interventions carry an identified "
        f"causal effect (cohort-estimated from the synthetic-gold cohort): "
        + (", ".join(sorted(identified)) or "none")
        + (f". Not yet identified: {', '.join(sorted(missing))}." if missing else ".")
    )

    return {
        "scope": brand,
        "model_summary": model_summary,
        "simulation_summary": simulation_summary,
        "latest_result": latest_result,
        "intervention_coverage": intervention_coverage,
        "grounding": [
            {"label": "Twin models", "value": str(len(models))},
            {"label": "Simulations", "value": str(len(simulations))},
            {"label": "Deploy rate", "value": f"{deploy_rate:.0%}"},
            {"label": "Identified interventions", "value": f"{len(identified)}/{len(catalog)}"},
        ],
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}: {g['model_summary']} {g['simulation_summary']} "
        f"{g['latest_result']} {g['intervention_coverage']} "
        "Effects are estimated from the synthetic-gold cohort (capability showcase, "
        "not real-world evidence). (Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["simulation_summary"], g["intervention_coverage"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    """LLM interpretation grounded in ``g``, or a deterministic factual fallback."""
    pred = run_signature(
        DigitalTwinInsightSignature,
        scope=g["scope"],
        model_summary=g["model_summary"],
        simulation_summary=g["simulation_summary"],
        latest_result=g["latest_result"],
        intervention_coverage=g["intervention_coverage"],
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
