"""Feedback-learning strategic insight: what the Tier-5 loop is learning and
what deserves attention (pattern hot-spots, unapplied updates, weak signals)."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class FeedbackLearningInsightSignature(dspy.Signature):
        """Interpret the state of a self-improvement feedback loop for an
        analytics-platform operator, STRICTLY grounded in the provided figures.
        Use ONLY the numbers and pattern/update descriptions given; never invent
        counts, severities, or agent names. State whether the loop is actively
        learning or starved, what the detected patterns imply about agent
        quality, and the single most useful next action (e.g. review pending
        updates / investigate a low-reward agent / keep collecting)."""

        activity_summary: str = dspy.InputField(desc="Cycles, last run, feedback inflow volumes")
        patterns_summary: str = dspy.InputField(desc="Detected patterns by severity + examples")
        updates_summary: str = dspy.InputField(desc="Knowledge updates proposed vs applied")
        signal_quality_summary: str = dspy.InputField(desc="Reward stats per agent/component")

        interpretation: str = dspy.OutputField(desc="Loop health diagnosis grounded in the figures")
        key_takeaways: list = dspy.OutputField(
            desc="3-5 grounded takeaways incl. recommended action"
        )

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    FeedbackLearningInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(
    cycles_24h: int,
    last_cycle_at: str | None,
    thumbs_7d: int,
    signals_7d: int,
    avg_reward_7d: float | None,
    patterns: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    low_reward_agents: list[tuple[str, float]],
) -> dict[str, Any]:
    """Build the server-derived grounding for the insight (figures + chips).

    ``patterns``/``updates`` are the persisted page artifacts (dicts with
    ``severity``/``pattern_type``/``description`` and ``status``/``update_type``
    respectively); ``low_reward_agents`` is (agent, avg reward) pairs below the
    attention threshold, already computed by the caller.
    """
    inflow = thumbs_7d + signals_7d
    activity_summary = (
        f"{cycles_24h} learning cycle(s) in the last 24h; "
        f"last cycle {last_cycle_at or 'never'}; "
        f"7-day feedback inflow {inflow} items "
        f"({thumbs_7d} explicit thumbs, {signals_7d} cognitive reward signals"
        + (f", avg reward {avg_reward_7d:.2f}" if avg_reward_7d is not None else "")
        + ")"
    )

    if patterns:
        sev_counts = Counter(str(p.get("severity", "unknown")) for p in patterns)
        sev_str = ", ".join(f"{n} {sev}" for sev, n in sev_counts.most_common())
        examples = "; ".join(
            f"[{p.get('severity')}] {str(p.get('description', ''))[:120]}" for p in patterns[:3]
        )
        patterns_summary = f"{len(patterns)} pattern(s) detected ({sev_str}). Examples: {examples}"
    else:
        patterns_summary = "no patterns detected yet"

    if updates:
        applied = sum(1 for u in updates if str(u.get("status", "")) == "applied")
        pending = sum(1 for u in updates if str(u.get("status", "")) in ("proposed", "approved"))
        updates_summary = (
            f"{len(updates)} knowledge update(s): {applied} applied, {pending} pending review"
        )
    else:
        updates_summary = "no knowledge updates proposed yet"

    if low_reward_agents:
        weak = ", ".join(f"{a} ({r:.2f})" for a, r in low_reward_agents)
        signal_quality_summary = f"agents/components with 7-day avg reward below 0.5: {weak}"
    elif signals_7d:
        signal_quality_summary = "no agent/component below the 0.5 reward attention threshold"
    else:
        signal_quality_summary = "no reward signals in the window"

    chips: list[dict[str, str]] = [
        {"label": "Cycles 24h", "value": str(cycles_24h)},
        {"label": "Feedback 7d", "value": str(inflow)},
        {"label": "Patterns", "value": str(len(patterns))},
        {"label": "Updates", "value": str(len(updates))},
    ]
    if avg_reward_7d is not None:
        chips.append({"label": "Avg reward", "value": f"{avg_reward_7d:.2f}"})

    return {
        "activity_summary": activity_summary,
        "patterns_summary": patterns_summary,
        "updates_summary": updates_summary,
        "signal_quality_summary": signal_quality_summary,
        "grounding": chips,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"Feedback loop: {g['activity_summary']}. {g['patterns_summary']}. "
        f"{g['updates_summary']}. Signal quality: {g['signal_quality_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["activity_summary"], g["patterns_summary"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        FeedbackLearningInsightSignature,
        activity_summary=g["activity_summary"],
        patterns_summary=g["patterns_summary"],
        updates_summary=g["updates_summary"],
        signal_quality_summary=g["signal_quality_summary"],
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
