"""Home-dashboard strategic insight: read of the computed KPI grid scoped to the
selected brand + territory. Grounding is SERVER-derived (registry KPIs recomputed
under the same brand/region context the dashboard uses) — caller-posted figures
are never accepted."""

from __future__ import annotations

import logging
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
        selected brand — at most mention them as portfolio context. If few or
        no KPIs computed for this scope, say so plainly instead of
        speculating."""

        scope: str = dspy.InputField(desc="Brand + territory the KPI values are scoped to")
        kpi_table: str = dspy.InputField(desc="Computed KPIs: name [workstream]: value (status)")
        status_summary: str = dspy.InputField(desc="Counts of KPI statuses")
        coverage: str = dspy.InputField(desc="How many of the defined KPIs computed for this scope")

        interpretation: str = dspy.OutputField(
            desc="What the KPI picture says for this scope and where to act, grounded in the values"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, actionable takeaways")

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
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        HomeKpiInsightSignature,
        scope=g["scope"],
        kpi_table=g["kpi_table"],
        status_summary=g["status_summary"],
        coverage=g["coverage"],
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
