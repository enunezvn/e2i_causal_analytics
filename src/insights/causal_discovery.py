"""Causal-discovery strategic insight: portfolio-level read of discovered effects."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class CausalDiscoveryInsightSignature(dspy.Signature):
        """Interpret a leaderboard of agent-validated causal effects for a brand
        analyst, STRICTLY grounded in the provided effects. Use ONLY the treatments,
        outcomes, ATEs, CIs, gate statuses, and estimators given; NEVER invent effects
        or numbers. Emphasise which effects are robust and ACTIONABLE (gate=proceed,
        CI excludes 0) vs which need review; if none are robust, say so plainly."""

        scope: str = dspy.InputField(desc="Brand + analysis grain")
        effects_table: str = dspy.InputField(
            desc="Ranked effects: treatment->outcome, ATE [CI], gate, estimator"
        )
        gate_summary: str = dspy.InputField(desc="Counts by gate status")

        interpretation: str = dspy.OutputField(
            desc="Which effects to act on and why, grounded in ATE/CI/gate"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, actionable takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    CausalDiscoveryInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(brand: str, grain: str, effects: list[dict[str, Any]]) -> dict[str, Any]:
    def _rank(e: dict[str, Any]) -> float:
        return abs(float(e.get("ate") or 0))

    ranked = sorted(effects, key=_rank, reverse=True)
    rows = []
    for e in ranked[:8]:
        rows.append(
            f"{e.get('treatment')}->{e.get('outcome')}: "
            f"ATE {float(e.get('ate') or 0):+.3f} "
            f"[{float(e.get('ate_ci_lower') or 0):+.3f}, {float(e.get('ate_ci_upper') or 0):+.3f}], "
            f"gate={e.get('status')}, est={e.get('selected_estimator')}"
        )
    effects_table = "\n".join(rows) or "no effects discovered"
    gates: Counter = Counter(e.get("status", "unknown") for e in effects)
    gate_summary = ", ".join(f"{g}={c}" for g, c in gates.most_common()) or "none"
    return {
        "scope": f"{brand} / {grain}",
        "effects_table": effects_table,
        "gate_summary": gate_summary,
        "grounding": [
            {"label": "Effects", "value": str(len(effects))},
            {"label": "Proceed", "value": str(gates.get("proceed", 0))},
            {"label": "Review", "value": str(gates.get("review", 0))},
        ],
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}, discovered effects (by |ATE|):\n{g['effects_table']}\n"
        f"Gate distribution: {g['gate_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
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
