"""Adapt the resource-optimizer's existing summary/recommendations to the uniform
strategic-insight payload (no new LLM call — surfaces what the agent already made)."""
from __future__ import annotations

from typing import Any

from src.insights.common import normalize_list


def to_insight(
    optimization_summary: str,
    recommendations: list[str] | None,
    projected_lift_pct: float | None,
    solver_status: str | None,
) -> dict[str, Any]:
    # NOTE on is_fallback semantics (intentional, differs from the LLM endpoints):
    # this endpoint never calls an LLM — it surfaces the resource-optimizer agent's
    # EXISTING summary/recommendations. So is_fallback=False when a real summary is
    # present (it is genuine agent output, NOT a degraded LLM fallback — the card must
    # not show a "LLM unavailable" badge over real agent text), and is_fallback=True
    # only when there is no summary yet (no optimization run). The provenance label
    # ("Resource optimizer (existing agent output)") disambiguates on the client.
    summary = (optimization_summary or "").strip()
    recs = normalize_list(recommendations or [])
    grounding: list[dict[str, str]] = [{"label": "Solver", "value": str(solver_status or "unknown")}]
    if projected_lift_pct is not None:
        grounding.insert(0, {"label": "Projected lift", "value": f"{projected_lift_pct:+.1f}%"})
    if not summary:
        return {
            "insight": "No optimization narrative is available yet — run an optimization.",
            "key_takeaways": recs,
            "grounding": grounding,
            "is_fallback": True,
        }
    return {
        "insight": summary,
        "key_takeaways": recs,
        "grounding": grounding,
        "is_fallback": False,
    }
