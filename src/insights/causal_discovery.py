"""Causal-discovery strategic insight: portfolio-level read of discovered effects."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.insights.column_labels import column_label
from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class CausalDiscoveryInsightSignature(dspy.Signature):
        """Interpret a leaderboard of agent-validated causal effects for a brand
        analyst, STRICTLY grounded in the provided effects. Use ONLY the treatments,
        outcomes, ATEs, CIs, gate statuses, and estimators given; NEVER invent effects
        or numbers. Emphasise which effects are robust and ACTIONABLE (gate=proceed,
        CI excludes 0) vs which need review; if none are robust, say so plainly.

        For the robust effects, translate each into a CONCRETE, EXECUTABLE commercial
        action grounded in that specific treatment -> outcome — say HOW to act, not just
        THAT it is actionable. Name the lever to pull (the treatment), the segment or
        audience to prioritize, and the metric to monitor and how to validate it. Vague
        filler such as "make data-driven decisions", "leverage the effect", or "monitor
        outcomes" — without saying WHICH action, for WHOM, measured HOW — is NOT
        acceptable.

        The registry context lists commercial chains modeled OUTSIDE this
        leaderboard's estimation scope (curated, directional): you may mention them
        as additional modeled coverage, but NEVER present them as discovered
        effects and NEVER attribute numbers to them.

        You are also given the brand's labeled CLINICAL POSITIONING (target
        population + line of therapy). GATE every recommendation by it: if a
        discovered effect, segment, or audience implies targeting a population
        OUTSIDE the brand's labeled target population or line of therapy, say so
        explicitly and do NOT recommend acting on it — even when the modeled
        effect is favourable. A statistically strong effect in a clinically
        off-target population (e.g. treatment-naive patients for an
        antihistamine-refractory indication) is NOT an actionable commercial
        recommendation. When the clinical positioning is empty, proceed without
        this gate rather than inventing one.

        Write every output as PLAIN PROSE — no markdown syntax: no asterisks,
        no underscore emphasis, no backticks, no # heading markers, no
        bullet-list markers, no numbered-list markers — write flowing prose."""

        scope: str = dspy.InputField(desc="Brand + analysis grain")
        effects_table: str = dspy.InputField(
            desc="Ranked effects: treatment->outcome, ATE [CI], gate, estimator"
        )
        gate_summary: str = dspy.InputField(desc="Counts by gate status")
        registry_context: str = dspy.InputField(
            desc="Commercial chains modeled in the registry, outside estimation scope (no figures)"
        )
        clinical_positioning: str = dspy.InputField(
            desc=(
                "The brand's labeled target population + line of therapy; gate every "
                "recommendation by it (empty when unavailable — then no clinical gate)"
            )
        )

        interpretation: str = dspy.OutputField(
            desc=(
                "Which effects to act on and HOW — grounded in ATE/CI/gate, with the "
                "concrete action each robust effect implies (lever, target segment, metric)"
            )
        )
        key_takeaways: list = dspy.OutputField(
            desc=(
                "3-5 grounded takeaways, each a SPECIFIC action a brand team could execute — "
                "name the lever, the target segment, and the metric; NOT vague guidance like "
                "'make data-driven decisions' or 'leverage the effect'"
            )
        )

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    CausalDiscoveryInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(
    brand: str,
    grain: str,
    effects: list[dict[str, Any]],
    causal_drivers: list[dict[str, Any]] | None = None,
    clinical_positioning: str = "",
) -> dict[str, Any]:
    def _rank(e: dict[str, Any]) -> float:
        return abs(float(e.get("ate") or 0))

    ranked = sorted(effects, key=_rank, reverse=True)
    rows = []
    for e in ranked[:8]:
        # #1895: effects are named by their display label — the leaderboard
        # rows above this insight read the same label, never the raw column.
        rows.append(
            f"{column_label(str(e.get('treatment') or ''))}->"
            f"{column_label(str(e.get('outcome') or ''))}: "
            f"ATE {float(e.get('ate') or 0):+.3f} "
            f"[{float(e.get('ate_ci_lower') or 0):+.3f}, {float(e.get('ate_ci_upper') or 0):+.3f}], "
            f"gate={e.get('status')}, est={e.get('selected_estimator')}"
        )
    effects_table = "\n".join(rows) or "no effects discovered"
    gates: Counter = Counter(e.get("status", "unknown") for e in effects)
    gate_summary = ", ".join(f"{g}={c}" for g, c in gates.most_common()) or "none"
    # Commercial chains live OUTSIDE the leaderboard's estimation scope (the
    # dataset-spec grain guard excludes commercial nodes from runs) — digit-free
    # so curated values can never pose as discovered ATEs.
    from src.insights.causal_context import format_driver_names, format_qualitative_context

    drivers = causal_drivers or []
    named = format_driver_names(drivers)
    chips = [
        {"label": "Effects", "value": str(len(effects))},
        {"label": "Proceed", "value": str(gates.get("proceed", 0))},
        {"label": "Review", "value": str(gates.get("review", 0))},
    ]
    if named:
        chips.append({"label": "Registry chains", "value": str(len(named))})
    positioning = (clinical_positioning or "").strip()
    if positioning:
        chips.append({"label": "Clinical positioning", "value": "applied"})
    return {
        "scope": f"{brand} / {grain}",
        "effects_table": effects_table,
        "gate_summary": gate_summary,
        "registry_context": format_qualitative_context(drivers),
        "has_registry_context": bool(named),
        "clinical_positioning": positioning,
        "grounding": chips,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}, discovered effects (by |ATE|):\n{g['effects_table']}\n"
        f"Gate distribution: {g['gate_summary']}. "
        + (f"{g['registry_context']} " if g.get("has_registry_context") else "")
        + (
            f"Clinical positioning: {g['clinical_positioning']} "
            if g.get("clinical_positioning")
            else ""
        )
        + "(Factual summary — LLM interpretation unavailable.)"
    )
    first_line = g["effects_table"].splitlines()[0] if g["effects_table"] else g["gate_summary"]
    return {
        "insight": insight,
        "key_takeaways": [f"Gates: {g['gate_summary']}", first_line],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        CausalDiscoveryInsightSignature,
        scope=g["scope"],
        effects_table=g["effects_table"],
        gate_summary=g["gate_summary"],
        registry_context=g["registry_context"],
        clinical_positioning=g.get("clinical_positioning", ""),
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
