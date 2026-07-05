"""HTE strategic insight: interpret ONE completed segment-level CATE analysis.

Grounding contract (mirrors ``executive_brief``): the route derives EVERY figure
SERVER-SIDE from the persisted segment-analysis record (``analysis_id`` is the
only caller input), and the LM output passes a fail-closed numeric guard before
it is served — any numeric claim the grounding cannot vouch for (including a
flipped sign, a pp/% unit swap, or a name digit re-used out of context)
downgrades the response to the deterministic factual fallback. Effects on the
binary clinical outcomes are probability deltas presented in PERCENTAGE POINTS
(pp), matching the /ai-insights HTE card's display unit.

The insight must keep two questions separate (the card's core honesty issue):

* "Is the treatment working in a segment?"  -> per-segment CI vs zero.
* "Should we target segments differentially?" -> segments must beat the
  OVERALL ATE (the above-ATE lift gate); per-segment significance vs zero is
  NOT a targeting license.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class HTEInsightSignature(dspy.Signature):
        """Interpret ONE completed segment-level CATE (heterogeneous treatment
        effect) analysis for a pharma brand analyst, STRICTLY grounded in the
        figures provided. Use ONLY the numbers given — never invent, re-derive,
        round differently, or extrapolate. Effects are probability deltas in
        percentage points (pp); never call them "percent growth".

        Answer BOTH questions, keeping them clearly separate:
        (1) Is the treatment effect real? Read the overall ATE and how many
        segments' CIs exclude zero.
        (2) Is differential targeting warranted? Use ONLY the provided
        targeting verdict (expected lift / allocation summary) — a segment
        being significant vs zero does NOT by itself justify targeting it;
        that requires the segment to beat the overall ATE.

        Name the strongest and weakest segments with their pp effects, state
        the single most decision-relevant implication, and ALWAYS close with
        the caveat that these are model-based estimates from one causal-forest
        analysis on an observational cohort."""

        scope: str = dspy.InputField(desc="Treatment -> outcome, brand filter, cohort size")
        effect_summary: str = dspy.InputField(
            desc="Overall ATE (pp), significant-segment count, heterogeneity score"
        )
        segments: str = dspy.InputField(
            desc="Per-segment CATE lines: value, pp effect, CI, n, significance"
        )
        targeting: str = dspy.InputField(
            desc="Differential-targeting verdict (expected lift, allocation summary)"
        )

        interpretation: str = dspy.OutputField(
            desc="Grounded strategic read: effect reality, heterogeneity, targeting verdict, caveat"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, decision-oriented takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    HTEInsightSignature = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pp(value: Any) -> str | None:
    """Format a probability delta as signed percentage points at 1 decimal."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    v = f * 100.0
    return f"{v:+.1f}pp"


def _fmt_int(value: Any) -> str | None:
    try:
        i = int(value)
    except (TypeError, ValueError):
        return None
    return f"{i:,}"


# ---------------------------------------------------------------------------
# Vouched phrases — name digits are grounded only IN CONTEXT
# ---------------------------------------------------------------------------

# Natural-word expansions for unit letters embedded in variable names, so
# "persistent_180d" keeps "180-day persistence" grounded.
_UNIT_WORDS = {"d": "day", "w": "week", "m": "month", "y": "year"}


def _phrase_variants(raw: Any) -> set[str]:
    """A digit-bearing name plus its natural paraphrases.

    These phrases are stripped from text BEFORE numeric-claim extraction, so
    their internal digits pass the guard only in context: "persistent_180d"
    and "180-day persistence" are fine, a bare re-use like "Treat 180
    patients" still trips it.
    """
    s = str(raw or "").strip()
    if not s or not re.search(r"\d", s):
        return set()
    out = {s}
    for sep in (" ", "-", ""):
        out.add(s.replace("_", sep))
    band = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)  # age bands like "50-65"
    if band:
        a, b = band.group(1), band.group(2)
        out.update({f"{a}-{b}", f"{a}–{b}", f"{a}—{b}", f"{a} - {b}", f"{a} to {b}"})
    for num, unit in re.findall(r"(\d+)([A-Za-z]+)", s):
        out.update({f"{num}{unit}", f"{num} {unit}"})
        word = _UNIT_WORDS.get(unit.lower())
        if word:
            out.update({f"{num}-{word}", f"{num} {word}", f"{num}-{word}s", f"{num} {word}s"})
    return out


# A vouched phrase immediately followed by a unit word is a numeric claim in
# disguise ("50 to 65 percent" from the 50-65 age band), not a name mention —
# leave it in place so its digits face the guard.
_PHRASE_UNIT_LOOKAHEAD = r"(?!\s*(?:%|pp\b|percent\b|percentage\b|points?\b))"


def _strip_phrases(text: str, phrases: list[str]) -> str:
    """Remove vouched phrases (longest first) so only bare numbers remain."""
    for p in phrases:
        if p:
            text = re.sub(re.escape(p) + _PHRASE_UNIT_LOOKAHEAD, " ", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Fail-closed output guard
# ---------------------------------------------------------------------------

# A numeric claim is (sign, number, unit): "+11.1pp" / "-2.8" / "95%". The
# sign counts only when directly attached to the number, so a markdown bullet
# ("- 11.1pp") reads unsigned and "top-2" keeps its hyphen inside the word.
_CLAIM_RE = re.compile(
    r"(?:(?<![\w.\-])([+\-−]))?"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(pp\b|%|percentage[\s-]points?\b|percent\b))?",
    re.IGNORECASE,
)
# Count-fraction claims: "13/14", "13 of 14", "13 out of 14", "13-of-14".
_FRACTION_RE = re.compile(r"\b(\d+)\s*(?:/|-?\s*(?:out\s+of|of)\s*-?)\s*(\d+)\b", re.IGNORECASE)


def _extract_claims(text: str) -> list[tuple[str, str, str]]:
    """(sign, comma-stripped number, normalized unit) for every number."""
    claims: list[tuple[str, str, str]] = []
    for m in _CLAIM_RE.finditer(text):
        sign = "-" if m.group(1) in ("-", "−") else (m.group(1) or "")
        unit_raw = (m.group(3) or "").lower()
        if unit_raw == "pp" or "point" in unit_raw:
            unit = "pp"
        elif unit_raw:
            unit = "%"
        else:
            unit = ""
        claims.append((sign, m.group(2).replace(",", ""), unit))
    return claims


def _claim_vouched(
    sign: str, num: str, unit: str, vouched: dict[str, set[tuple[str, str]]]
) -> bool:
    """True iff a grounded rendering of ``num`` exists that the claim does not
    contradict: omitting the sign or unit is fine, flipping the sign or
    swapping pp for % is not."""
    for v_sign, v_unit in vouched.get(num, ()):
        if (not sign or sign == v_sign) and (not unit or unit == v_unit):
            return True
    return False


def _is_grounded(candidate: str, g: dict[str, Any]) -> bool:
    """True iff every numeric claim in ``candidate`` is vouched by the grounding.

    Fail-closed: any digit sequence the grounding did not render (different
    rounding, re-derived deltas, invented figures, flipped signs, swapped
    units, name digits re-used out of context) rejects the whole output.
    Count fractions whose denominator is the segment total must state the
    true significant count.
    """
    text = _strip_phrases(candidate, g["phrases"])
    vouched: dict[str, set[tuple[str, str]]] = g["vouched"]
    for frac in _FRACTION_RE.finditer(text):
        m_str, k_str = frac.group(1), frac.group(2)
        if k_str == str(g["total_count"]):
            # Any "m of <segment total>" claim is a significance-count claim:
            # the numerator must be the actual significant count. Both digits
            # being individually vouched is NOT enough ("3 of 3" from a true
            # 2-of-3 would otherwise pass).
            if m_str != str(g["sig_count"]):
                return False
            continue
        if not (_claim_vouched("", m_str, "", vouched) and _claim_vouched("", k_str, "", vouched)):
            return False
    return all(_claim_vouched(s, n, u, vouched) for s, n, u in _extract_claims(text))


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def build_grounding(record: dict[str, Any]) -> dict[str, Any]:
    """Build the grounded prompt inputs + the guard's vouched vocabulary.

    ``record`` is a plain-dict projection of the persisted
    ``SegmentAnalysisResponse`` (see the /insights/hte route). The guard
    vocabulary is EXTRACTED from the exact strings rendered into the prompt —
    sign and unit included — so the LM can only echo figures as given.
    """
    treatment = record.get("treatment_var") or "the treatment"
    outcome = record.get("outcome_var") or "the outcome"
    brand = record.get("brand")
    ci_level = record.get("confidence_level")
    ci_pct = f"{round(float(ci_level) * 100):d}" if ci_level else "95"

    rows: list[dict[str, Any]] = []
    for dimension, results in (record.get("cate_by_segment") or {}).items():
        for r in results or []:
            rows.append({**r, "dimension": dimension})

    overall_ate = record.get("overall_ate")
    has_signal = bool(rows) and overall_ate is not None

    n_total = sum(int(r.get("sample_size") or 0) for r in rows)
    # Segment dimensions partition the same cohort, so the cohort size is the
    # per-dimension sum, not the all-rows sum.
    dims = {r["dimension"] for r in rows}
    if dims:
        n_total = max(
            sum(int(r.get("sample_size") or 0) for r in rows if r["dimension"] == d) for d in dims
        )

    sig_count = sum(1 for r in rows if r.get("statistical_significance"))
    total_count = len(rows)

    ate_pp = _pp(overall_ate)
    het = record.get("heterogeneity_score")
    het_str = f"{float(het):.2f}" if het is not None and math.isfinite(float(het)) else None

    scope = (
        f"{treatment} -> {outcome}"
        + (f", brand filter {brand}" if brand else ", all brands")
        + (f", cohort n={_fmt_int(n_total)}" if n_total else "")
        + f", {ci_pct}% CIs"
    )

    effect_summary = (
        f"Overall ATE {ate_pp or '—'}; "
        f"{sig_count} of {total_count} segments have {ci_pct}% CIs excluding zero; "
        + (
            f"heterogeneity score {het_str} (0-1 scale)"
            if het_str
            else "heterogeneity score unavailable"
        )
    )

    seg_lines: list[str] = []
    ordered = sorted(rows, key=lambda r: float(r.get("cate_estimate") or 0.0), reverse=True)
    for r in ordered:
        name = f"{r['dimension']}={r.get('segment_value')}"
        cate_s = _pp(r.get("cate_estimate")) or "—"
        lo_s = _pp(r.get("cate_ci_lower")) or "—"
        hi_s = _pp(r.get("cate_ci_upper")) or "—"
        n_s = _fmt_int(r.get("sample_size")) or "—"
        sig = "significant" if r.get("statistical_significance") else "not significant"
        seg_lines.append(f"{name}: {cate_s} [CI {lo_s} to {hi_s}], n={n_s}, {sig}")

    # expected_lift_pp is stored as a probability FRACTION despite its name:
    # policy_learner validates it in [0, 1] and multiplies by 100 only at
    # display. Render through _pp like every other effect — a 0.10 lift is
    # +10.0pp, not +0.1pp.
    lift_s = _pp(record.get("expected_lift_pp"))
    allocation = str(record.get("optimal_allocation_summary") or "").strip()
    if len(allocation) > 300:
        allocation = allocation[:297] + "..."
    targeting = (f"Expected lift from differential targeting: {lift_s}. " if lift_s else "") + (
        allocation if allocation else "No allocation summary available."
    )

    grounding_chips = [
        {"label": "Overall ATE", "value": ate_pp or "—"},
        {"label": "Significant segments", "value": f"{sig_count}/{total_count}"},
        {"label": "Heterogeneity", "value": het_str or "—"},
        {"label": "n", "value": _fmt_int(n_total) or "—"},
    ]

    # Digit-bearing NAMES (variables, brand, dimensions, segment values) are
    # grounded as PHRASES, not free-floating numbers: their digits pass the
    # guard only in context.
    phrases: set[str] = set()
    for name_part in (treatment, outcome, brand, *dims):
        phrases |= _phrase_variants(name_part)
    for r in rows:
        phrases |= _phrase_variants(r.get("segment_value"))
    phrase_list = sorted(phrases, key=len, reverse=True)

    vouched: dict[str, set[tuple[str, str]]] = {}
    grounded_text = _strip_phrases(
        "\n".join([scope, effect_summary, *seg_lines, targeting]), phrase_list
    )
    for sign, num, unit in _extract_claims(grounded_text):
        vouched.setdefault(num, set()).add((sign, unit))

    return {
        "scope": scope,
        "effect_summary": effect_summary,
        "segments": "\n".join(seg_lines),
        "targeting": targeting,
        "grounding": grounding_chips,
        "phrases": phrase_list,
        "vouched": vouched,
        "has_signal": has_signal,
        "sig_count": sig_count,
        "total_count": total_count,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    if not g["has_signal"]:
        return {
            "insight": (
                "The persisted analysis contains no per-segment CATE results, so no "
                "grounded heterogeneity interpretation can be produced. Re-run the "
                "segment-level CATE analysis to generate one."
            ),
            "key_takeaways": [],
            "grounding": [],
            "is_fallback": True,
        }
    insight = (
        f"For {g['scope']}: {g['effect_summary']}. {g['targeting']} "
        "These are model-based estimates from one causal-forest analysis on an "
        "observational cohort. (Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["effect_summary"], g["targeting"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    if not g["has_signal"]:
        return _fallback(g)
    pred = run_signature(
        HTEInsightSignature,
        scope=g["scope"],
        effect_summary=g["effect_summary"],
        segments=g["segments"],
        targeting=g["targeting"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    takeaways = normalize_list(getattr(pred, "key_takeaways", []))
    rejected = [t for t in [interpretation, *takeaways] if not _is_grounded(t, g)]
    if rejected:
        logger.warning(
            "HTE insight LM output carried %d ungrounded numeric claim(s); using factual fallback",
            len(rejected),
        )
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": takeaways,
        "grounding": g["grounding"],
        "is_fallback": False,
    }
