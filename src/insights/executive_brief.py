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
        metric or segment, or when everything sits below break-even. When
        citing figures, cite ONLY figures given above and keep each sentence's
        figures to a SINGLE opportunity — never pair one opportunity's dollar
        value with another's ROI, gap, or segment — and name a segment or gap
        metric only in the sentence that cites that same opportunity's own
        figures. ALWAYS close by stating the caveat given in `caveats`."""

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
    opp_lines = [_opportunity_line(o) for o in ranked]
    if ranked:
        opp_text = " ".join(opp_lines)
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
        # Per-UNIT source strings for the grounding guard: each opportunity is
        # its own unit so a sentence pairing one opportunity's dollar value
        # with another's ROI/gap can be detected (codex PR-5 round 2) — a flat
        # value set would accept any swapped combination.
        "sources": [scope, caveats, *opp_lines],
        # Structured attribute tokens per unit (segment, gap metric): a numeric
        # sentence must not name ANOTHER unit's segment/metric even when its
        # figures all trace to one unit — "field triggers in South for $1.2M at
        # 3.2x" is a false attribution with fully-grounded numbers (codex PR-5
        # round 3). Actions are free prose and stay prompt-governed: matching
        # them mechanically would be a false sense of safety.
        "source_tokens": [set(), set()]
        + [
            {
                t.lower()
                for t in (
                    str(o.get("segment_value", "")).strip(),
                    str(o.get("gap_metric", "")).strip(),
                )
                if len(t) >= 3
            }
            for o in ranked
        ],
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
# defense. Deterministic layers (each fails closed to the labelled fallback):
#   1. money / percentage / ROI-multiple claims must appear in the grounding and
#      bind per-SENTENCE to a single source unit (rounds 1-2);
#   2. segment / gap-metric tokens named in a numeric sentence must belong to
#      that same unit (round 3);
#   3. labelled portfolio counts (quick wins / steady plays / strategic bets /
#      suppressed) must match the grounded counts globally (round 4).
# What stays prompt-governed — documented, not accidental: free-prose action
# wording, spelled-out numbers ("two quick wins"), and count phrasings outside
# the server-controlled vocabulary; matching those mechanically would be a
# false sense of safety.

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


# Server-controlled portfolio-count vocabulary (codex PR-5 round 4): the
# signature hands the LM the mix/suppression counts, so an invented "99 quick
# wins" is a fabricated portfolio-breadth claim the money/pct/multiple guard
# never saw. Counts are PORTFOLIO-level facts — there is exactly one set, so a
# count claim has no cross-opportunity pairing risk and validates against the
# whole grounding (unlike money/pct/mult, which stay unit-bound). Count
# phrasings outside this vocabulary ("the top 3 plays") and spelled-out
# numbers ("two quick wins") remain prompt-governed — the same documented
# boundary as free-prose actions.
_COUNT_KEYWORDS = {
    "quick_win": r"quick[- ]wins?",
    "steady_play": r"steady[- ]plays?",
    "strategic_bet": r"strategic[- ]bets?",
    "suppressed": r"suppress\w*",
}
# Number-first: up to three letter-words may sit between the number and its
# label ("3 low-value opportunities were suppressed"); digits act as barriers
# so one count can never borrow another count's label across a list.
_COUNT_GAP = r"(?:\W+[A-Za-z()-]+){0,3}?\W+"


def _count_claims(text: str) -> set[tuple[str, float]]:
    """Extract (count:<label>, value) claims for the server-controlled labels."""
    claims: set[tuple[str, float]] = set()
    for kind, kw in _COUNT_KEYWORDS.items():
        for m in re.finditer(rf"\b(\d+)\b{_COUNT_GAP}{kw}", text, re.I):
            claims.add((f"count:{kind}", float(m.group(1))))
        # Label-first allows only space/colon/equals/dash between label and
        # number ("suppressed 42", "quick wins: 2") — list punctuation would
        # mint spurious claims from prose like "2 quick wins, 1 steady play"
        # (the comma would hand quick_win the NEXT item's count).
        for m in re.finditer(rf"{kw}[ \t:=-]+(\d+)\b", text, re.I):
            claims.add((f"count:{kind}", float(m.group(1))))
    return claims


# Sentences (and semicolon clauses) are the pairing unit: a figure cited next
# to another figure inside one sentence claims a RELATIONSHIP between them.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def _is_grounded(
    candidate: str,
    sources: list[str],
    source_tokens: list[set[str]] | None = None,
) -> bool:
    """True iff every numeric sentence binds to a SINGLE source unit.

    Global value-membership is not enough: an LM can pair opportunity A's
    dollar value with opportunity B's ROI/gap and every number still "appears
    somewhere" (codex PR-5 round 2). Each sentence's numeric claims must
    therefore be a subset of ONE source unit's claims (scope, caveats, or one
    opportunity line) — AND, when ``source_tokens`` is given, that same unit
    must own every segment/metric token the sentence names, so grounded
    figures cannot be re-attributed to another opportunity's segment or
    metric (codex PR-5 round 3). Legitimate prose that mixes units inside a
    single numeric sentence falls back — fail-closed by design; the signature
    instructs the LM to keep each sentence's figures and segment/metric
    mentions to a single opportunity.

    Labelled portfolio COUNTS (quick wins / steady plays / strategic bets /
    suppressed) are validated too, but globally: they are portfolio-level
    facts with exactly one grounded set, so an invented "99 quick wins" is
    rejected while a correct restatement passes regardless of which sentence
    it shares with an opportunity's figures (codex PR-5 round 4).
    """
    unit_claims = [_numeric_claims(s) for s in sources]
    grounded_counts: set[tuple[str, float]] = set()
    for s in sources:
        grounded_counts |= _count_claims(s)
    tokens = source_tokens if source_tokens is not None else [set() for _ in sources]
    all_tokens: set[str] = set().union(*tokens) if tokens else set()
    for sentence in _SENTENCE_SPLIT_RE.split(candidate):
        if not _count_claims(sentence) <= grounded_counts:
            return False
        claims = _numeric_claims(sentence)
        if not claims:
            continue
        mentioned = {
            t for t in all_tokens if re.search(rf"\b{re.escape(t)}\b", sentence, re.IGNORECASE)
        }
        if not any(
            claims <= unit_claims[i] and mentioned <= tokens[i] for i in range(len(sources))
        ):
            return False
    return True


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
    sources = g["sources"]
    source_tokens = g.get("source_tokens")
    ungrounded = [
        t for t in [interpretation, *takeaways] if not _is_grounded(t, sources, source_tokens)
    ]
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
