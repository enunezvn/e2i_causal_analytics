"""Predictive-cohort strategic insight: who to target and why (cohort + SHAP)."""
from __future__ import annotations

import logging
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class PredictiveCohortInsightSignature(dspy.Signature):
        """Turn a scored out-of-sample cohort into a targeting read for a commercial
        analyst, STRICTLY grounded in the provided numbers. Use ONLY the score
        distribution, named top targets, and SHAP driver importances given; NEVER
        invent entities, probabilities, or features. Say who to prioritise, what
        drives their scores, and how confident the ranking is."""

        model_version: str = dspy.InputField(desc="Scoring model version")
        distribution_summary: str = dspy.InputField(desc="n scored, mean probability")
        top_targets_summary: str = dspy.InputField(desc="Top-ranked entities with probabilities")
        drivers_summary: str = dspy.InputField(desc="Top SHAP feature drivers + importances")

        interpretation: str = dspy.OutputField(desc="Targeting read grounded in the numbers")
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded targeting takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    PredictiveCohortInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(
    model_version: str,
    n_scored: int,
    mean_prob: float,
    top_targets: list[dict[str, Any]],
    top_drivers: list[dict[str, Any]],
) -> dict[str, Any]:
    distribution_summary = f"{n_scored} entities scored, mean probability {mean_prob:.3f}"
    top_targets_summary = "; ".join(
        f"{t.get('entity_id')} ({float(t.get('probability') or 0):.2f})" for t in top_targets[:5]
    ) or "none"
    drivers_summary = "; ".join(
        f"{d.get('feature')} ({float(d.get('importance') or 0):.2f})" for d in top_drivers[:5]
    ) or "none"
    return {
        "model_version": model_version,
        "distribution_summary": distribution_summary,
        "top_targets_summary": top_targets_summary,
        "drivers_summary": drivers_summary,
        "grounding": [
            {"label": "Scored", "value": str(n_scored)},
            {"label": "Mean p", "value": f"{mean_prob:.3f}"},
            {"label": "Top targets", "value": str(min(len(top_targets), 5))},
        ],
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"Model {g['model_version']}: {g['distribution_summary']}. "
        f"Highest-probability targets: {g['top_targets_summary']}. "
        f"Main drivers: {g['drivers_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["distribution_summary"], f"Drivers: {g['drivers_summary']}"],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        PredictiveCohortInsightSignature,
        model_version=g["model_version"],
        distribution_summary=g["distribution_summary"],
        top_targets_summary=g["top_targets_summary"],
        drivers_summary=g["drivers_summary"],
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
