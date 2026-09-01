"""Experiments strategic insight: DSPy interpretation of the A/B portfolio.

Turns the in-silico A/B portfolio (per-intervention-channel effect estimates
from ab_experiment_results, grouped by the migration-100 intervention_channel
taxonomy) into a business read — WHAT the portfolio tests, WHICH interventions
show statistically supported lift, what adopting the winners would be worth
qualitatively, and which channels honestly show nothing. Falls back to a
deterministic factual summary when the LM is unavailable (never fabricates).

All arithmetic (per-channel means, significance shares, ranking) happens HERE,
server-side; the LM only narrates the numbers it is given.
"""

from __future__ import annotations

import logging
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class ExperimentsInsightSignature(dspy.Signature):
        """Interpret an in-silico A/B experimentation portfolio for a
        commercial pharma strategist, STRICTLY grounded in the provided
        numbers. Use ONLY the channel effects, significance counts, and scope
        given; NEVER invent effect sizes, dollar values, or channels. Explain
        what the portfolio is testing (each experiment randomizes one
        engagement intervention against standard practice on a synthetic HCP
        panel), rank the interventions by the evidence given, state what
        adopting the strongest channels would mean for the brand's outcome
        QUALITATIVELY (scale the given percentage-point effects to "per 100
        targeted HCPs" framing only — no invented dollars), and be explicit
        about channels whose tests show NO significant effect: a null result
        is decision-relevant (stop investing) and must not be spun as a win.
        ALWAYS close with the caveat given in `caveats`.

        Write every output as PLAIN PROSE — no markdown syntax: no asterisks,
        no underscore emphasis, no backticks, no # heading markers, no
        bullet-list markers, no numbered-list markers — write flowing prose."""

        scope: str = dspy.InputField(
            desc="Brand scope, number of running experiments with results, channels tested"
        )
        channel_effects: str = dspy.InputField(
            desc="Per-channel test counts, mean effects (percentage points), significance shares"
        )
        highlights: str = dspy.InputField(
            desc="Strongest evidence-backed channel(s) and null channel(s), precomputed"
        )
        caveats: str = dspy.InputField(desc="Data-provenance caveats that MUST be stated")

        interpretation: str = dspy.OutputField(
            desc="Business read: what is tested, which interventions win, value if adopted"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, actionable takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    ExperimentsInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(brand: str | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate experiment×result rows into the insight's grounding.

    Args:
        brand: Brand scope label (None/"All" = whole portfolio).
        rows: One dict per experiment: ``intervention_channel`` plus its
            (possibly empty) ``ab_experiment_results`` list from the PostgREST
            embed. Only rows with a channel and at least one result count.

    Returns:
        Grounding dict for generate_insight()/_fallback().
    """
    # Human labels come from the same catalog the generator plants (mirrored
    # from the digital-twin taxonomy); unknown values degrade to the raw value.
    from src.ml.synthetic.generators.experiment_generator import CHANNEL_LABELS

    per_channel: dict[str, dict[str, Any]] = {}
    n_experiments = 0
    for row in rows:
        channel = row.get("intervention_channel")
        results = row.get("ab_experiment_results") or []
        if not channel or not results:
            continue
        # One 'final' analysis per experiment in the substrate; take the first.
        res = results[0]
        effect = res.get("effect_estimate")
        if effect is None:
            continue
        n_experiments += 1
        agg = per_channel.setdefault(channel, {"n": 0, "effect_sum": 0.0, "significant": 0})
        agg["n"] += 1
        agg["effect_sum"] += float(effect)
        agg["significant"] += 1 if res.get("is_significant") else 0

    channels = []
    for channel, agg in per_channel.items():
        mean_effect = agg["effect_sum"] / agg["n"]
        channels.append(
            {
                "channel": channel,
                "label": CHANNEL_LABELS.get(channel, channel),
                "n": agg["n"],
                "mean_effect_pp": round(mean_effect * 100, 1),
                "significant": agg["significant"],
            }
        )
    channels.sort(key=lambda c: c["mean_effect_pp"], reverse=True)

    brand_label = brand if brand and brand != "All" else "All brands"
    scope = (
        f"{brand_label} / {n_experiments} running in-silico A/B experiments with "
        f"final results / {len(channels)} intervention channels tested"
    )
    channel_effects = (
        "; ".join(
            f"{c['label']}: {c['n']} tests, mean effect {c['mean_effect_pp']:+.1f}pp, "
            f"{c['significant']}/{c['n']} significant"
            for c in channels
        )
        or "none"
    )

    # Evidence-backed = majority of that channel's tests significant.
    winners = [c for c in channels if c["n"] > 0 and c["significant"] * 2 > c["n"]]
    nulls = [c for c in channels if c["significant"] == 0]
    if winners:
        top = winners[0]
        highlight_win = (
            f"Strongest evidence: {top['label']} "
            f"({top['mean_effect_pp']:+.1f}pp mean effect, "
            f"{top['significant']}/{top['n']} tests significant) — roughly "
            f"{abs(top['mean_effect_pp']):.0f} additional conversions per 100 "
            f"targeted HCPs if the effect holds at rollout."
        )
    else:
        highlight_win = "No channel has majority-significant evidence yet."
    if nulls:
        highlight_null = (
            "No significant effect in any test for: " + ", ".join(c["label"] for c in nulls) + "."
        )
    else:
        highlight_null = "Every tested channel shows at least one significant result."
    highlights = f"{highlight_win} {highlight_null}"

    caveats = (
        "These are in-silico A/B tests on the clearly-labelled SYNTHETIC HCP "
        "panel (known ground-truth effects planted for estimator validation). "
        "Effect directions and channel rankings are meaningful as a "
        "methodology showcase; validate against real-world experiments before "
        "shifting budget."
    )

    grounding = [
        {"label": "Brand", "value": brand_label},
        {"label": "Experiments", "value": str(n_experiments)},
        {"label": "Channels tested", "value": str(len(channels))},
    ]
    if winners:
        grounding.append(
            {
                "label": "Top channel",
                "value": f"{winners[0]['label']} ({winners[0]['mean_effect_pp']:+.1f}pp)",
            }
        )
    if nulls:
        grounding.append({"label": "Null channels", "value": ", ".join(c["label"] for c in nulls)})

    return {
        "scope": scope,
        "channel_effects": channel_effects,
        "highlights": highlights,
        "caveats": caveats,
        "grounding": grounding,
        "channels": channels,
        "n_experiments": n_experiments,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    """Deterministic factual summary — narrates the computed aggregates verbatim."""
    if not g["n_experiments"]:
        return {
            "insight": (
                "No running A/B experiments with final results are available for "
                "this scope, so no portfolio interpretation can be produced — run "
                "monitoring or widen the brand scope."
            ),
            "key_takeaways": [],
            "grounding": g["grounding"],
            "is_fallback": True,
        }
    insight = (
        f"A/B portfolio read. Scope: {g['scope']}. Channel results: "
        f"{g['channel_effects']}. {g['highlights']} {g['caveats']} "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    takeaways = [
        f"{c['label']}: {c['mean_effect_pp']:+.1f}pp mean effect "
        f"({c['significant']}/{c['n']} significant)"
        for c in g["channels"][:5]
    ]
    return {
        "insight": insight,
        "key_takeaways": takeaways,
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    if not g["n_experiments"]:
        return _fallback(g)
    pred = run_signature(
        ExperimentsInsightSignature,
        scope=g["scope"],
        channel_effects=g["channel_effects"],
        highlights=g["highlights"],
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
