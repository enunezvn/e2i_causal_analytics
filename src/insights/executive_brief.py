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
is unavailable (never fabricates).

Figure integrity is enforced by SERVER-SIDE INJECTION, not by parsing the
LM's English: the LM only ever sees placeholder tokens ({TOTAL}, {ROI_1},
{SEG_1}, ...) — never a digit — and the server substitutes the real values
after validating the tokens. Fabricating or swapping a figure is structurally
impossible because the model has no numbers to garble; the only checks needed
are exact set arithmetic over the server-defined token vocabulary. This
replaced a per-sentence numeric-grounding guard that rejected 9/10 faithful
samples (the LM habitually writes multi-opportunity comparison sentences the
single-source-per-sentence rule forbade), pinning the page to the factual
fallback; the placeholder contract measured 10/10 clean on the same live
grounding.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class ExecutiveBriefInsightSignature(dspy.Signature):
        """Write an EXECUTIVE BRIEF for a pharma commercial leader. Every
        figure in the inputs is a placeholder token like {ROI_1} or {TOTAL};
        the server will substitute the real values after you write. Rules:
        (1) express EVERY figure using ONLY the placeholder tokens exactly as
        given, curly braces included; (2) NEVER write digits or spelled-out
        numbers — no counts, no percentages, no rankings written as numerals;
        (3) opportunities are ranked best-first by ROI (rank 1 is highest).
        Structure: lead with the single highest-impact decision and its
        stakes; then the action sequence across the ranked opportunities;
        then an honest actionability judgment; ALWAYS close with the caveat
        given in `caveats`. Where `causal_context` names modeled causal
        levers, you may connect recommended actions to those levers BY NAME
        with their stated provenance — no figures are provided for them and
        you must not invent any."""

        scope: str = dspy.InputField(
            desc="Brand, total addressable opportunity value, opportunity mix counts"
        )
        opportunities: str = dspy.InputField(
            desc="Ranked opportunities with ROI multiple, revenue impact, gap %, segment, effort"
        )
        causal_context: str = dspy.InputField(
            desc=(
                "Registry-modeled causal levers by NAME only (may say none are "
                "available); reference qualitatively, never with numbers"
            )
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


def _strip_segment_suffix(action: str, segment: str) -> str:
    """Drop a trailing "in [the] <segment>" from the action prose.

    The LM-facing opportunity line appends "in {SEG_n}" itself; leaving the
    real segment name in the action would both read twice and hand the LM a
    prose alias that bypasses the token-index attribution check.
    """
    if len(segment) < 3:
        return action
    return re.sub(rf"\s+in\s+(?:the\s+)?{re.escape(segment)}\s*$", "", action, flags=re.IGNORECASE)


def _lm_opportunity_line(pos: int, o: dict[str, Any]) -> str:
    """Placeholder-token variant of ``_opportunity_line`` (token index = pos)."""
    seg = _truncate(str(o.get("segment_value", "")).strip() or "—", 60)
    action = _truncate(str(o.get("recommended_action", "")).strip() or "(no action text)", 160)
    action = _strip_segment_suffix(action, seg)
    metric = str(o.get("gap_metric", "")).upper() or "—"
    effort = str(o.get("implementation_difficulty") or "unknown")
    return (
        f"Rank {pos}: {action} in {{SEG_{pos}}} — {{ROI_{pos}}} ROI, "
        f"{{IMPACT_{pos}}} revenue impact, closing a {{GAP_{pos}}} {metric} gap "
        f"({effort} effort)."
    )


def build_grounding(
    brand: str,
    total_addressable_value: float | None,
    quick_wins_count: int,
    steady_plays_count: int,
    strategic_bets_count: int,
    suppressed_count: int,
    opportunities: list[dict[str, Any]],
    causal_drivers: list[str] | None = None,
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

    # LM-facing variants: identical structure, every figure a placeholder
    # token. The injection map is the server-owned token vocabulary — the LM
    # cannot introduce a figure that isn't in it.
    injection: dict[str, str] = {
        "{TOTAL}": _money(total_addressable_value),
        "{QUICK}": str(quick_wins_count),
        "{STEADY}": str(steady_plays_count),
        "{BETS}": str(strategic_bets_count),
    }
    lm_scope = (
        f"{brand} / total addressable opportunity value {{TOTAL}} / mix: "
        "{QUICK} quick win(s), {STEADY} steady play(s), {BETS} strategic bet(s)"
    )
    lm_opp_lines: list[str] = []
    # Token index is the LIST position (1-based), not the feed's rank field:
    # positions are unique by construction, so duplicate/missing ranks can
    # never collide two opportunities onto one token.
    for pos, o in enumerate(ranked, start=1):
        roi = o.get("expected_roi")
        gap_pct = o.get("gap_percentage")
        injection[f"{{ROI_{pos}}}"] = f"{float(roi):.1f}x" if roi is not None else "—"
        injection[f"{{IMPACT_{pos}}}"] = _money(o.get("revenue_impact"))
        injection[f"{{GAP_{pos}}}"] = f"{float(gap_pct):.0f}%" if gap_pct is not None else "—"
        injection[f"{{SEG_{pos}}}"] = _truncate(str(o.get("segment_value", "")).strip() or "—", 60)
        lm_opp_lines.append(_lm_opportunity_line(pos, o))
    lm_opportunities = " ".join(lm_opp_lines) if lm_opp_lines else opp_text
    lm_caveat_parts: list[str] = []
    if suppressed_count > 0:
        injection["{SUPPRESSED}"] = str(suppressed_count)
        noun = "opportunity was" if suppressed_count == 1 else "opportunities were"
        lm_caveat_parts.append(
            f"{{SUPPRESSED}} low-value {noun} suppressed (below break-even) "
            "and excluded from these figures."
        )
    lm_caveat_parts.append(
        "Figures come from the gap analyzer's ROI model on current data; "
        "validate them before committing budget."
    )
    # Causal levers (commercial grain, 2026-07-07) — NAMES ONLY, digit-free by
    # construction: the placeholder guard fails closed on ANY numeric char in
    # LM output, so a lever like "persistent_180d" fed raw would poison every
    # sample into fallback. Defensive filter here; the route humanizes via
    # causal_context.format_driver_names, which drops digit-bearing names too.
    levers = [n for n in (causal_drivers or []) if not any(ch.isnumeric() for ch in n)]
    if levers:
        causal_context = (
            "Modeled causal levers from the causal-path registry "
            "(curated synthetic knowledge, provenance-labeled): " + "; ".join(levers) + "."
        )
        grounding.append({"label": "Causal levers", "value": str(len(levers))})
    else:
        causal_context = "No modeled causal levers are available for this brand."
    return {
        "brand": brand,
        "scope": scope,
        "opportunities": opp_text,
        "caveats": caveats,
        "grounding": grounding,
        "has_signal": bool(ranked) or suppressed_count > 0,
        "lm_scope": lm_scope,
        "lm_opportunities": lm_opportunities,
        "lm_caveats": " ".join(lm_caveat_parts),
        # Same digit-free string on both channels — nothing to inject, and no
        # separate figure-bearing display variant to drift from the LM's view.
        "causal_context": causal_context,
        "lm_causal_context": causal_context,
        "has_causal_context": bool(levers),
        "injection": injection,
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
    causal_line = f" {g['causal_context']}" if g.get("has_causal_context") else ""
    insight = (
        f"Scope: {g['scope']}. Ranked opportunities: {g['opportunities']}{causal_line} "
        f"{g['caveats']} (Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


# ---- Placeholder contract validation --------------------------------------------
# The LM's inputs contain no figures, so its output must not either: every
# number the user sees is injected server-side from the grounded feed. The
# checks below are exact operations on the server-defined token vocabulary —
# no parsing of the LM's English — and each fails closed to the labelled
# factual fallback. What stays prompt-governed (documented, not accidental):
# spelled-out numbers ("three opportunities") and segment names the LM echoes
# from the free-prose action text; neither can mint a numeric figure.

_PLACEHOLDER_RE = re.compile(r"\{[A-Z]+(?:_\d+)?\}")
_SEG_TOKEN_RE = re.compile(r"\{SEG_(\d+)\}")
_METRIC_TOKEN_RE = re.compile(r"\{(?:ROI|IMPACT|GAP)_(\d+)\}")

# Sentences (and semicolon clauses) are the pairing unit: a metric token cited
# next to a segment token inside one sentence claims a relationship between them.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def _placeholder_violation(text: str, vocab: set[str]) -> str | None:
    """First violation of the placeholder contract in ``text``, or None.

    Three checks: (1) every token used must exist in the server vocabulary;
    (2) no numeric character may survive outside a token — str.isnumeric, not
    just ``\\d``, so circled/superscript/Roman glyphs ("②x", "²", "Ⅲ") cannot
    render as figures (codex PR-1153 round 1); this also traps malformed or
    lowercased tokens, whose embedded index digits are left behind by the
    strict token regex; (3) within a sentence that names segment tokens,
    every metric token's index must be among that sentence's segment indices —
    "{SEG_1} yields {ROI_2}" re-attributes rank 2's figure to rank 1's segment
    even though both values are real.
    """
    used = set(_PLACEHOLDER_RE.findall(text))
    unknown = used - vocab
    if unknown:
        return f"unknown placeholder(s): {sorted(unknown)}"
    if any(ch.isnumeric() for ch in _PLACEHOLDER_RE.sub("", text)):
        return "numeric characters outside placeholder tokens"
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        segs = {m.group(1) for m in _SEG_TOKEN_RE.finditer(sentence)}
        metrics = {m.group(1) for m in _METRIC_TOKEN_RE.finditer(sentence)}
        if segs and metrics and not metrics <= segs:
            return "metric token paired with another opportunity's segment token"
    return None


def _inject(text: str, injection: dict[str, str]) -> str:
    """Substitute real values for tokens in one pass (an injected value can
    never itself be re-substituted)."""
    return _PLACEHOLDER_RE.sub(lambda m: injection.get(m.group(0), m.group(0)), text)


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    # No real signal -> the honest factual answer, never an LLM riff on nothing.
    if not g["has_signal"]:
        return _fallback(g)
    vocab = set(g["injection"])
    # Two independent draws: lm_cache=False forces a fresh sample per attempt —
    # the long-lived API process's in-memory DSPy cache would otherwise replay
    # the identical rejected completion on every retry.
    for attempt in (1, 2):
        pred = run_signature(
            ExecutiveBriefInsightSignature,
            lm_cache=False,
            scope=g["lm_scope"],
            opportunities=g["lm_opportunities"],
            causal_context=g.get(
                "lm_causal_context", "No modeled causal levers are available for this brand."
            ),
            caveats=g["lm_caveats"],
        )
        if pred is None:
            # LM unavailable/errored (not a contract violation): retrying via
            # run_signature is its caller's concern; fall back honestly.
            return _fallback(g)
        interpretation = str(getattr(pred, "interpretation", "")).strip()
        takeaways = normalize_list(getattr(pred, "key_takeaways", []))
        violations = [
            v for v in (_placeholder_violation(u, vocab) for u in [interpretation, *takeaways]) if v
        ]
        if interpretation and not violations:
            return {
                "insight": _inject(interpretation, g["injection"]),
                "key_takeaways": [_inject(t, g["injection"]) for t in takeaways],
                "grounding": g["grounding"],
                "is_fallback": False,
            }
        logger.warning(
            "executive-brief sample %d violated the placeholder contract (%s); %s",
            attempt,
            "; ".join(violations) or "empty interpretation",
            "retrying with a fresh sample" if attempt == 1 else "using factual fallback",
        )
    return _fallback(g)
