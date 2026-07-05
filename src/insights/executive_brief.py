"""Executive-brief strategic insight: DSPy distillation of gap/ROI figures.

The /ai-insights Executive AI Brief previously fell back to the cognitive-RAG
endpoint, whose answer read as a *description* rather than a strategic
distillation (review finding 1: "it seems more like a description ... the
system is meant to make intelligent strategic distillation that would help
decision making"). This module turns the brand's REAL gap-analysis figures
(the same /gaps/opportunities feed the sibling Priority-Actions card renders)
into a decision aid: the single highest-impact decision, its quantified
stakes, a ranked action sequence, an honest actionability judgment, and the
suppression caveat. Falls back to a deterministic factual summary when the LM
is unavailable (never fabricates)."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class ExecutiveBriefInsightSignature(dspy.Signature):
        """Write an EXECUTIVE BRIEF for a pharma commercial leader, STRICTLY
        grounded in the provided figures. Use ONLY the opportunity values, ROI
        multiples, gap percentages, and mix counts given; NEVER invent dollar
        amounts, segments, metrics, or trends. Structure the brief as a
        decision aid, not a description: (1) LEAD with the single
        HIGHEST-IMPACT DECISION the numbers support and its quantified stakes
        (value at stake, expected ROI); (2) lay out the recommended ACTION
        SEQUENCE across the ranked opportunities — what to do first and why,
        trading ROI against implementation effort; (3) judge ACTIONABILITY
        honestly — flag when the portfolio is thin, concentrated in a single
        metric or segment, or when everything sits below break-even. ALWAYS
        close by stating the caveat given in `caveats`."""

        scope: str = dspy.InputField(
            desc="Brand, total addressable opportunity value, opportunity mix counts"
        )
        opportunities: str = dspy.InputField(
            desc="Ranked opportunities with ROI multiple, revenue impact, gap %, segment, effort"
        )
        caveats: str = dspy.InputField(desc="Data caveats that MUST be stated")

        interpretation: str = dspy.OutputField(
            desc=(
                "Executive brief: highest-impact decision with quantified stakes, "
                "ranked action sequence, honest actionability judgment"
            )
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, decision-ready takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    ExecutiveBriefInsightSignature = None  # type: ignore[assignment,misc]


def _money(value: float | None) -> str:
    """Compact USD, mirroring the frontend gap drill-down ($5.0M / $300K / $42)."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${round(v)}"


def _truncate(s: str, max_len: int) -> str:
    """Bound one free-text field so a verbose opportunity can't crowd out the rest."""
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _opportunity_line(o: dict[str, Any]) -> str:
    rank = o.get("rank", "?")
    action = _truncate(str(o.get("recommended_action", "")).strip() or "(no action text)", 160)
    roi = o.get("expected_roi")
    roi_str = f"{float(roi):.1f}x ROI" if roi is not None else "ROI —"
    rev = _money(o.get("revenue_impact"))
    gap_pct = o.get("gap_percentage")
    gap_str = f"{float(gap_pct):.0f}%" if gap_pct is not None else "—"
    metric = str(o.get("gap_metric", "")).upper() or "—"
    seg = _truncate(str(o.get("segment_value", "")).strip() or "—", 60)
    effort = str(o.get("implementation_difficulty") or "unknown")
    return (
        f"{rank}. {action} — {roi_str}, {rev} revenue impact, "
        f"closing a {gap_str} {metric} gap in {seg} ({effort} effort)."
    )


def build_grounding(
    brand: str,
    total_addressable_value: float | None,
    quick_wins_count: int,
    steady_plays_count: int,
    strategic_bets_count: int,
    suppressed_count: int,
    opportunities: list[dict[str, Any]],
) -> dict[str, Any]:
    mix = (
        f"{quick_wins_count} quick win(s), {steady_plays_count} steady play(s), "
        f"{strategic_bets_count} strategic bet(s)"
    )
    scope = (
        f"{brand} / total addressable opportunity value "
        f"{_money(total_addressable_value)} / mix: {mix}"
    )
    ranked = sorted(opportunities, key=lambda o: o.get("rank", 0))[:5]
    if ranked:
        opp_text = " ".join(_opportunity_line(o) for o in ranked)
    elif suppressed_count > 0:
        # All-suppressed is REAL signal (mirrors the T6 gap-analyzer honest
        # narrative): the right brief is "don't invest now", not silence.
        opp_text = (
            "None surfaced: every identified opportunity fell below the "
            "break-even threshold and was suppressed."
        )
    else:
        opp_text = "None: no gap-analysis signal is available for this brand."
    caveat_parts: list[str] = []
    if suppressed_count > 0:
        noun = "opportunity was" if suppressed_count == 1 else "opportunities were"
        caveat_parts.append(
            f"{suppressed_count} low-value {noun} suppressed (below break-even) "
            "and excluded from these figures."
        )
    caveat_parts.append(
        "Figures come from the gap analyzer's ROI model on current data; "
        "validate them before committing budget."
    )
    caveats = " ".join(caveat_parts)
    grounding = [
        {"label": "Brand", "value": brand},
        {"label": "Addressable value", "value": _money(total_addressable_value)},
        {"label": "Mix", "value": mix},
    ]
    if ranked:
        top = ranked[0]
        roi = top.get("expected_roi")
        grounding.append(
            {
                "label": "Top ROI",
                "value": f"{float(roi):.1f}x" if roi is not None else "—",
            }
        )
    if suppressed_count > 0:
        grounding.append({"label": "Suppressed", "value": str(suppressed_count)})
    return {
        "brand": brand,
        "scope": scope,
        "opportunities": opp_text,
        "caveats": caveats,
        "grounding": grounding,
        "has_signal": bool(ranked) or suppressed_count > 0,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    """Deterministic factual summary built verbatim from the grounded figures."""
    if not g["has_signal"]:
        return {
            "insight": (
                f"No gap-analysis signal is available for {g['brand']} yet — run a "
                "gap analysis to generate an executive brief."
            ),
            "key_takeaways": [],
            "grounding": g["grounding"],
            "is_fallback": True,
        }
    insight = (
        f"Scope: {g['scope']}. Ranked opportunities: {g['opportunities']} "
        f"{g['caveats']} (Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


# ---- Numeric grounding guard ---------------------------------------------------
# The signature TELLS the LM not to invent figures, but an executive brief is the
# highest-stakes surface for plausible-wrong values, so the prompt is not the only
# defense: every high-risk numeric claim in the LM output (currency, percentage,
# ROI multiple) must literally appear in the grounding strings it was given, or
# the whole response is discarded for the deterministic fallback. This is a
# DETERMINISTIC check on unit-classed numbers only — semantic claims (segment
# names, actions) cannot be validated mechanically without a false sense of
# safety, so those remain covered by the prompt contract.

_MONEY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\b", re.I)
_PCT_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*%")
_MULT_RE = re.compile(r"\b([\d,]+(?:\.\d+)?)\s*[x×]\b", re.I)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


def _numeric_claims(text: str) -> set[tuple[str, float]]:
    """Extract (kind, canonical value) for every $-amount, percentage and multiple."""
    claims: set[tuple[str, float]] = set()
    for m in _MONEY_RE.finditer(text):
        value = float(m.group(1).replace(",", ""))
        value *= _SCALE.get((m.group(2) or "").lower(), 1.0)
        claims.add(("money", round(value, 2)))
    for m in _PCT_RE.finditer(text):
        claims.add(("pct", round(float(m.group(1).replace(",", "")), 2)))
    for m in _MULT_RE.finditer(text):
        claims.add(("mult", round(float(m.group(1).replace(",", "")), 2)))
    return claims


def _is_grounded(candidate: str, sources: list[str]) -> bool:
    """True iff every numeric claim in ``candidate`` appears in the sources."""
    grounded: set[tuple[str, float]] = set()
    for s in sources:
        grounded |= _numeric_claims(s)
    return _numeric_claims(candidate) <= grounded


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    # No real signal -> the honest factual answer, never an LLM riff on nothing.
    if not g["has_signal"]:
        return _fallback(g)
    pred = run_signature(
        ExecutiveBriefInsightSignature,
        scope=g["scope"],
        opportunities=g["opportunities"],
        caveats=g["caveats"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    takeaways = normalize_list(getattr(pred, "key_takeaways", []))
    sources = [g["scope"], g["opportunities"], g["caveats"]]
    ungrounded = [t for t in [interpretation, *takeaways] if not _is_grounded(t, sources)]
    if ungrounded:
        # Fail closed: a single invented figure poisons trust in the whole
        # brief, so the labelled deterministic fallback replaces it entirely.
        logger.warning(
            "executive-brief LM output carried %d ungrounded numeric claim(s); "
            "using factual fallback",
            len(ungrounded),
        )
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": takeaways,
        "grounding": g["grounding"],
        "is_fallback": False,
    }
