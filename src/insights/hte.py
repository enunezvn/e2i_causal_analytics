"""HTE strategic insight: interpret ONE completed segment-level CATE analysis.

Grounding contract (mirrors ``executive_brief``): the route derives EVERY figure
SERVER-SIDE from the persisted segment-analysis record (``analysis_id`` is the
only caller input), and the LM output passes a fail-closed numeric guard before
it is served — any numeric claim the grounding cannot vouch for downgrades the
response to the deterministic factual fallback. Effects on the binary clinical
outcomes are probability deltas presented in PERCENTAGE POINTS (pp), matching
the /ai-insights HTE card's display unit.

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
# Grounding
# ---------------------------------------------------------------------------


def build_grounding(record: dict[str, Any]) -> dict[str, Any]:
    """Build the grounded prompt inputs + vouched numeric sets from a record.

    ``record`` is a plain-dict projection of the persisted
    ``SegmentAnalysisResponse`` (see the /insights/hte route). Only figures
    rendered into the strings below are vouched for the output guard.
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

    # Vouched numeric vocabulary (the guard's whitelist). Keyed by canonical
    # unsigned string; every figure rendered into the prompt must be added.
    vouched: set[str] = set()
    segment_tokens: dict[str, set[str]] = {}

    def _vouch(s: str | None) -> str:
        if not s:
            return "—"
        for m in re.findall(r"\d[\d,]*(?:\.\d+)?", s):
            vouched.add(m.replace(",", ""))
            vouched.add(m)
        return s

    ate_pp = _pp(overall_ate)
    het = record.get("heterogeneity_score")
    het_str = f"{float(het):.2f}" if het is not None and math.isfinite(float(het)) else None

    scope = (
        f"{treatment} -> {outcome}"
        + (f", brand filter {brand}" if brand else ", all brands")
        + (f", cohort n={_fmt_int(n_total)}" if n_total else "")
        + f", {ci_pct}% CIs"
    )
    _vouch(scope)

    effect_summary = (
        f"Overall ATE {_vouch(ate_pp)}; "
        f"{sig_count} of {total_count} segments have {ci_pct}% CIs excluding zero; "
        + (
            f"heterogeneity score {_vouch(het_str)} (0-1 scale)"
            if het_str
            else "heterogeneity score unavailable"
        )
    )
    vouched.update({str(sig_count), str(total_count), ci_pct})

    seg_lines: list[str] = []
    ordered = sorted(rows, key=lambda r: float(r.get("cate_estimate") or 0.0), reverse=True)
    for r in ordered:
        name = f"{r['dimension']}={r.get('segment_value')}"
        cate_s = _vouch(_pp(r.get("cate_estimate")))
        lo_s = _vouch(_pp(r.get("cate_ci_lower")))
        hi_s = _vouch(_pp(r.get("cate_ci_upper")))
        n_s = _vouch(_fmt_int(r.get("sample_size")))
        sig = "significant" if r.get("statistical_significance") else "not significant"
        seg_lines.append(f"{name}: {cate_s} [CI {lo_s} to {hi_s}], n={n_s}, {sig}")
        # Segment-name numerals (age bands like "50-65") must not trip the guard.
        toks = set(re.findall(r"\d[\d,]*(?:\.\d+)?", str(r.get("segment_value") or "")))
        vouched.update(toks)
        segment_tokens[str(r.get("segment_value") or "")] = {
            t.replace(",", "")
            for t in re.findall(r"\d[\d,]*(?:\.\d+)?", f"{cate_s} {lo_s} {hi_s} {n_s}")
        }

    # expected_lift_pp is ALREADY in percentage points — format directly.
    lift_pp = record.get("expected_lift_pp")
    lift_s = (
        f"{float(lift_pp):+.1f}pp"
        if lift_pp is not None and math.isfinite(float(lift_pp))
        else None
    )
    allocation = str(record.get("optimal_allocation_summary") or "").strip()
    if len(allocation) > 300:
        allocation = allocation[:297] + "..."
    targeting = (
        f"Expected lift from differential targeting: {_vouch(lift_s)}. " if lift_s else ""
    ) + (_vouch(allocation) if allocation else "No allocation summary available.")

    grounding_chips = [
        {"label": "Overall ATE", "value": ate_pp or "—"},
        {"label": "Significant segments", "value": f"{sig_count}/{total_count}"},
        {"label": "Heterogeneity", "value": het_str or "—"},
        {"label": "n", "value": _fmt_int(n_total) or "—"},
    ]

    # Variable names can carry digits (persistent_180d) — vouch them so the
    # guard does not flag the design description itself.
    for name in (treatment, outcome):
        vouched.update(re.findall(r"\d+", str(name)))

    return {
        "scope": scope,
        "effect_summary": effect_summary,
        "segments": "\n".join(seg_lines),
        "targeting": targeting,
        "grounding": grounding_chips,
        "vouched": vouched,
        "segment_tokens": segment_tokens,
        "has_signal": has_signal,
        "sig_count": sig_count,
        "total_count": total_count,
    }


# ---------------------------------------------------------------------------
# Fail-closed output guard
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# "13/14" or "13 of 14" count-fraction claims.
_FRACTION_RE = re.compile(r"\b(\d+)\s*(?:/|of)\s*(\d+)\b")


def _numeric_claims(text: str) -> list[str]:
    """Every numeric token in ``text``, comma-stripped."""
    return [m.replace(",", "") for m in _NUM_RE.findall(text)]


def _is_grounded(candidate: str, g: dict[str, Any]) -> bool:
    """True iff every numeric claim in ``candidate`` is vouched by the grounding.

    Fail-closed: any digit sequence the grounding did not render (different
    rounding, re-derived deltas, invented figures) rejects the whole output.
    Count fractions ("m of k" / "m/k") must additionally match the actual
    significant/total pair or be composed of vouched integers.
    """
    vouched: set[str] = g["vouched"]
    for frac in _FRACTION_RE.finditer(candidate):
        m_str, k_str = frac.group(1), frac.group(2)
        if k_str == str(g["total_count"]):
            # Any "m of <segment total>" claim is a significance-count claim:
            # the numerator must be the actual significant count. Both digits
            # being individually vouched is NOT enough ("3 of 3" from a true
            # 2-of-3 would otherwise pass).
            if m_str != str(g["sig_count"]):
                return False
            continue
        if m_str not in vouched or k_str not in vouched:
            return False
    for claim in _numeric_claims(candidate):
        if claim not in vouched:
            return False
    return True


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
