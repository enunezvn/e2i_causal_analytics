"""Home-dashboard strategic insight: read of the computed KPI grid scoped to the
selected brand + territory. Grounding is SERVER-derived (registry KPIs recomputed
under the same brand/region context the dashboard uses) — caller-posted figures
are never accepted.

Constraint-aware (2026-07-20): the signature performs two-channel triage over a
data_constraint_context block (src.insights.data_constraint_context), and every
generation passes a fail-closed DIGIT-SUBSET guard — each digit sequence in the
output must literally appear in the grounding or the constraint context (one
fresh-sample retry, then the factual fallback). This surface previously served
LM output with zero validation while exec-brief/HTE both grew guards after
instruction-only proved insufficient."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class HomeKpiInsightSignature(dspy.Signature):
        """Interpret an executive KPI dashboard for a pharma commercial analyst,
        STRICTLY grounded in the provided KPI values. Use ONLY the KPI names,
        values, units, and statuses given; NEVER invent metrics or numbers.
        Lead with what most needs attention (critical, then warning), then what
        is working (good); treat informational KPIs as context, not alarms.
        Read the picture for the STATED brand/territory scope — do not
        generalise beyond it. Rows tagged [sibling brand: X] are computed
        portfolio-wide and belong to ANOTHER brand; NEVER attribute them to the
        selected brand — at most mention them as portfolio context. Rows tagged
        [lower is better] are gap/rate metrics where a SMALLER value is good —
        never read them as scores. If few or no KPIs computed for this scope,
        say so plainly instead of speculating.

        Perform TWO-CHANNEL triage using data_constraint_context:

        Channel 1 — interpretation + key_takeaways: full-strength,
        lever-specific recommendations for every KPI the reader can act on —
        expressly INCLUDING commercial levers (HCP targeting and call-plan
        coverage, field-force allocation, access/copay and patient-services
        pull-through, conversion-funnel actions, budget reallocation,
        diagnostic-activation programs) and any lever the KPI's classification
        names; the lever list is open, never a closed whitelist. KPIs without a
        flagged constraint deserve full-strength recommendations; a constraint
        may only EXPLAIN a KPI whose classification carries a measurement
        caveat — never blanket-attribute a commercial gap to prevalence or
        data lag.

        Channel 2 — structural_considerations: constraints (claims lag, vendor
        coverage, disease prevalence, metric artifacts) are context that
        explains readings, and fixing them belongs to a DIFFERENT actor (data
        strategy / platform). Name each binding constraint, the KPIs it gates,
        and what would improve if it were lifted (e.g. faster claims feeds,
        tokenized data linking). NEVER recommend that this reader fix a
        constraint they cannot change, and NEVER treat a caveat-flagged value
        as a real performance level."""

        scope: str = dspy.InputField(desc="Brand + territory the KPI values are scoped to")
        kpi_table: str = dspy.InputField(desc="Computed KPIs: name [workstream]: value (status)")
        status_summary: str = dspy.InputField(desc="Counts of KPI statuses")
        coverage: str = dspy.InputField(desc="How many of the defined KPIs computed for this scope")
        data_constraint_context: str = dspy.InputField(
            desc="Measurement constraints + per-KPI actionability classification; "
            "cite it — attribute constraints, do not recommend fixing them"
        )

        interpretation: str = dspy.OutputField(
            desc="What the KPI picture says for this scope and where to act, grounded in the values"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, actionable takeaways")
        structural_considerations: str = dspy.OutputField(
            desc="Channel 2: structural constraints — escalation and investment "
            "considerations for data-strategy/platform owners (empty string if none)"
        )

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    HomeKpiInsightSignature = None  # type: ignore[assignment,misc]


# Statuses that most need attention sort first in the grounding table so a
# truncated table never drops the alarming rows.
_STATUS_RANK = {"critical": 0, "warning": 1, "good": 2, "informational": 3, "unknown": 4}
_MAX_TABLE_ROWS = 40


def _fmt_value(value: float, unit: str | None, value_format: str | None) -> str:
    """Render a KPI value the way the dashboard does: 'percent' KPIs are 0-1
    ratios shown as NN.N%; everything else as-is plus the unit."""
    if value_format == "percent":
        return f"{value * 100:.1f}%"
    rendered = f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}".strip() if unit else rendered


def build_grounding(
    brand: str,
    region: str | None,
    metas: list[Any],
    results: list[Any],
) -> dict[str, Any]:
    """Join registry metadata with batch results; keep only KPIs that actually
    computed (real value, no error) — the same visibility rule the home grid uses."""
    meta_by_id = {m.id: m for m in metas}
    computed = [r for r in results if r.value is not None and not r.error]

    def _status(r: Any) -> str:
        s = r.status
        return s.value if hasattr(s, "value") else str(s)

    rows = sorted(computed, key=lambda r: (_STATUS_RANK.get(_status(r), 5), r.kpi_id))
    lines = []
    for r in rows[:_MAX_TABLE_ROWS]:
        m = meta_by_id.get(r.kpi_id)
        name = m.name if m else r.kpi_id
        ws = m.workstream.value if m and hasattr(m.workstream, "value") else ""
        value = _fmt_value(
            float(r.value),
            m.unit if m else None,
            m.value_format if m else None,
        )
        # Brand-specific KPIs compute portfolio-wide, so another brand's rows
        # appear even under a brand scope — tag them so the LM can't misread
        # them as the selected brand's performance.
        tag = ""
        if brand != "All" and m and m.brand and m.brand != brand:
            tag = f" [sibling brand: {m.brand}]"
        # Gap/rate metrics where smaller is good (registry direction field,
        # e.g. Geographic Consistency Gap) — hint inline so neither the LM nor
        # a human reader mistakes the value for a score.
        if m and getattr(m, "direction", None) == "lower_is_better":
            tag = f" [lower is better]{tag}"
        lines.append(f"{name} [{ws}]: {value} ({_status(r)}){tag}")
    kpi_table = "\n".join(lines) or "no KPIs computed for this scope"

    statuses: Counter = Counter(_status(r) for r in computed)
    status_summary = ", ".join(f"{s}={c}" for s, c in statuses.most_common()) or "none"
    territory = region.title() if region else "All US"
    scope = f"{'All brands (portfolio)' if brand == 'All' else brand} / {territory}"
    coverage = f"{len(computed)} of {len(metas)} defined KPIs computed for this scope"

    chips = [
        {"label": "Brand", "value": "All" if brand == "All" else brand},
        {"label": "Territory", "value": territory},
        {"label": "Computed", "value": f"{len(computed)}/{len(metas)}"},
    ]
    for s in ("critical", "warning", "good"):
        if statuses.get(s):
            chips.append({"label": s.title(), "value": str(statuses[s])})
    return {
        "scope": scope,
        "kpi_table": kpi_table,
        "status_summary": status_summary,
        "coverage": coverage,
        "grounding": chips,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}: {g['coverage']}.\n{g['kpi_table']}\n"
        f"Status distribution: {g['status_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    first_line = g["kpi_table"].splitlines()[0] if g["kpi_table"] else g["coverage"]
    return {
        "insight": insight,
        "key_takeaways": [g["coverage"], f"Statuses: {g['status_summary']}", first_line],
        "grounding": g["grounding"],
        "structural_considerations": "",
        "is_fallback": True,
    }


# Digit sequences (integers or decimals) — the unit of the fail-closed subset
# guard. "67.5" is ONE token: an LM re-rounding it to "67" or "68" fails the
# guard rather than serving a figure the grounding cannot vouch for.
# Thousands separators are normalized first ("12,345" ≡ "12345"): _fmt_value
# renders large volumes with commas and the LM may cite either form — both
# must tokenize identically or a verbatim quote gets falsely rejected.
#
# SIGN-AWARE (codex review): a negative grounded value ("-8.0%") tokenizes
# WITH its sign, so an LM stripping the minus ("an 8.0% uplift") is rejected —
# reporting a decline as a gain is precisely the wrong-direction narrative
# this surface exists to prevent. A minus counts as part of the number only
# when it is not preceded by a digit/letter/dot, keeping ranges ("1-3
# months" -> {1, 3}) and hyphenated identifiers ("adherent_180d" -> {180})
# tokenizing exactly as before; unicode minus/en-dash are normalized first.
#
# Known limitations (deliberate, weighed against false-rejection cost):
# * Small shared integers ("2 of 44" licenses "2") cannot catch small-number
#   invention — the guard is a SUBSET check, not a semantics checker; range
#   fidelity ("1-3 months" vs "3-month lag") is the prompt's job, and making
#   ranges atomic would reject honest phrasings like "up to 3 months".
# * Re-rounded forms ("~10%" for 10.2%) are rejected BY DESIGN — the retry +
#   factual fallback is the recovery path (serving degraded-but-true beats
#   serving a figure the grounding cannot vouch for).
# * Spelled-out numbers ("ten percent") carry no digits and bypass the guard;
#   the signature instructs citing the given values verbatim.
_DIGIT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d)")
_MINUS_VARIANTS_RE = re.compile(r"[−–]")  # − (minus sign), – (en dash)

_NO_CONTEXT = "No data-constraint context is available for this scope."


def _digit_sequences(text: str) -> set[str]:
    text = _THOUSANDS_RE.sub("", _MINUS_VARIANTS_RE.sub("-", text))
    out: set[str] = set()
    for m in _DIGIT_RE.finditer(text):
        tok = m.group()
        if tok.startswith("-"):
            prev = text[m.start() - 1] if m.start() > 0 else " "
            if prev.isalnum() or prev == ".":
                tok = tok[1:]  # "1-3" / "x-3": hyphen, not a sign
        out.add(tok)
    return out


def _digit_violation(outputs: list[str], corpus: str) -> str | None:
    """First digit sequence in ``outputs`` that the grounding corpus cannot
    vouch for, or None. Subset check, fail-closed — mirrors the exec-brief
    lesson (instruction-only grounding proved insufficient) without its full
    placeholder machinery."""
    allowed = _digit_sequences(corpus)
    for text in outputs:
        for seq in _digit_sequences(text):
            if seq not in allowed:
                return seq
    return None


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    context = g.get("data_constraint_context") or _NO_CONTEXT
    corpus = "\n".join([g["scope"], g["kpi_table"], g["status_summary"], g["coverage"], context])
    # Attempt 2 forces a fresh sample (lm_cache=False) — the long-lived API
    # process's in-memory DSPy cache would otherwise replay the identical
    # rejected completion on the retry (exec-brief precedent).
    for attempt in (1, 2):
        pred = run_signature(
            HomeKpiInsightSignature,
            lm_cache=attempt == 1,
            scope=g["scope"],
            kpi_table=g["kpi_table"],
            status_summary=g["status_summary"],
            coverage=g["coverage"],
            data_constraint_context=context,
        )
        if pred is None:
            return _fallback(g)
        interpretation = str(getattr(pred, "interpretation", "")).strip()
        takeaways = normalize_list(getattr(pred, "key_takeaways", []))
        structural = str(getattr(pred, "structural_considerations", "") or "").strip()
        violation = _digit_violation([interpretation, *takeaways, structural], corpus)
        if interpretation and violation is None:
            return {
                "insight": interpretation,
                "key_takeaways": takeaways,
                "grounding": g["grounding"],
                "structural_considerations": structural,
                "is_fallback": False,
            }
        logger.warning(
            "home-KPI insight attempt %d rejected (digit %r absent from grounding corpus%s); %s",
            attempt,
            violation,
            "" if interpretation else "; empty interpretation",
            "retrying with a fresh sample" if attempt == 1 else "serving factual fallback",
        )
    return _fallback(g)
