"""CohortProfiler Agent — Tier 0: population profiling for chat.

Companion to :mod:`src.agents.cohort_constructor`. Where cohort_constructor
*materializes* an eligible patient list (a real ``DataFrame`` + audit trail) for
the ML pipeline (``scope_definer → cohort_constructor → data_preparer``) and
therefore cannot run from a free-text chat query, this agent answers the *chat*
form of a cohort question — "size / define a cohort of ... patients" — with REAL
numbers: the prescribing population broken down by the clinical segment axes that
already exist (disease-severity tier and line-of-therapy), for a brand.

It reuses the exact KPI calculation path the CopilotKit chat tool uses
(``get_kpi_calculator().calculate`` with a ``segment`` / ``therapy_line`` context
— the mig-105 registry variants shipped in PR #1208), so the numbers it reports
are the same real, DB-backed counts the live chat UI already returns. It NEVER
fabricates: if a brand has no prescribing rows it says so, and a total data
failure fails closed (``status="failed"``) rather than emitting an empty table.

Design rationale (REASON-BEFORE-RULES): the orchestrator classifier routes
COHORT_DEFINITION queries here (was cohort_constructor, which dead-ended: it
failed closed and its explainer fallback then also failed closed with nothing to
explain — verified by container replay). Keeping a dedicated agent — rather than
overloading cohort_constructor or the explainer's input resolver — keeps three
distinct concepts separate: population *profiling* (this), cohort *materialization*
(cohort_constructor, ML pipeline), and *explanation* of upstream analysis (explainer).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The three commercial brands on the synthetic substrate. Brand predicate in the
# KPI registry is a case-SENSITIVE exact match (``brand::text = $1``), so callers
# must be normalised to this canonical casing before hitting the calculator.
SUPPORTED_BRANDS: Tuple[str, ...] = ("Remibrutinib", "Fabhalta", "Kisqali")

# NRx (new prescriptions) — the "new prescribing patients" proxy used to SIZE the
# population. Reusing the mig-105 ``_segment`` / ``_line`` variants keeps this in
# lock-step with the chat tool and the /api/kpis breakdown.
_NRX_KPI_ID = "WS3-BI-006"

# (context value, human label) for each severity tier and line-of-therapy bucket.
_SEVERITY_TIERS: Tuple[Tuple[str, str], ...] = (
    ("low_severity", "Low severity"),
    ("medium_severity", "Medium severity"),
    ("high_severity", "High severity"),
)
_THERAPY_LINES: Tuple[Tuple[str, str], ...] = (
    ("0", "0 prior lines (1st line)"),
    ("1", "1 prior line (2nd line)"),
    ("2", "2 prior lines (3rd line)"),
    ("3", "3+ prior lines (4th line+)"),
)


class CohortProfilerAgent:
    """Profiles the eligible prescribing population by clinical segment axes."""

    def __init__(self) -> None:
        # No graph / LLM / mlflow ceremony — this agent is pure computation over
        # the shared KPI calculator, mirroring the health_score fast-path style.
        self._log = logger

    # ------------------------------------------------------------------ public
    async def analyze(self, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the segment breakdown and return a synthesizer-ready result.

        Named ``analyze`` (the dispatcher's default method spec) rather than
        ``run`` because the Tier 1-5 ``AGENT_METHOD_MAP`` contract is pinned to 13
        agents by the registry-consistency guard; Tier-0 agents like this one and
        cohort_constructor dispatch via the fall-through ``analyze`` contract and
        deliberately carry no method-map entry.

        ``agent_input`` is the merged orchestrator payload; the input resolver
        grounds ``brand`` (canonical casing) from the parsed query / user_context.
        When no brand is grounded we profile every supported brand rather than
        fabricate a default.
        """
        brand = self._canonical_brand(agent_input.get("brand"))
        brands: List[str] = [brand] if brand else list(SUPPORTED_BRANDS)

        try:
            calculator = self._get_calculator()
        except Exception as e:  # pragma: no cover - defensive (import/init)
            return self._failed(f"KPI calculator unavailable: {e}")

        profiles: List[Dict[str, Any]] = []
        for b in brands:
            profile = await self._profile_brand(calculator, b)
            if profile is not None:
                profiles.append(profile)

        if not profiles:
            # Every requested brand returned no prescribing population — this is a
            # genuine empty/failure state, not a zero to narrate. Fail closed.
            return self._failed(
                "no prescribing population found for "
                + (brand or "any supported brand")
                + " — nothing to profile (no values were fabricated)"
            )

        narrative = self._render(profiles, brand_requested=brand)
        return {
            "status": "completed",
            "narrative": narrative,
            "cohort_profile": {"segment_axis": "severity+line_of_therapy", "brands": profiles},
            "confidence": 0.9,
            "recommendations": [
                "Materialize the eligible patient list for ML via the cohort "
                "pipeline (scope_definer → cohort_constructor) when you need the "
                "actual patient rows, not just the population size.",
            ],
        }

    # --------------------------------------------------------------- internals
    def _get_calculator(self) -> Any:
        # Local import avoids an agent <-> api.routes import cycle at module load
        # (same pattern as chatbot_tools.kpi_calculate_tool).
        from src.api.routes.kpi import get_kpi_calculator

        return get_kpi_calculator()

    def _canonical_brand(self, raw: Any) -> Optional[str]:
        """Map a possibly mis-cased brand string to canonical casing, else None."""
        if not isinstance(raw, str) or not raw.strip():
            return None
        low = raw.strip().lower()
        for b in SUPPORTED_BRANDS:
            if b.lower() == low:
                return b
        return None

    async def _value(self, calculator: Any, context: Dict[str, Any]) -> Optional[float]:
        """One calculator call → its scalar value (None on missing/error)."""
        try:
            result = await asyncio.to_thread(calculator.calculate, _NRX_KPI_ID, context=context)
        except Exception as e:  # pragma: no cover - defensive (DB/calc error)
            self._log.warning("cohort_profiler: calculate failed for %s: %s", context, e)
            return None
        value = result.get("value") if isinstance(result, dict) else getattr(result, "value", None)
        return value if isinstance(value, (int, float)) else None

    async def _profile_brand(self, calculator: Any, brand: str) -> Optional[Dict[str, Any]]:
        """Real NRx headline + severity + line breakdown for one brand."""
        headline = await self._value(calculator, {"brand": brand})
        if not headline:
            return None  # no prescribing population for this brand — skip honestly

        severity = {}
        for value, _label in _SEVERITY_TIERS:
            severity[value] = await self._value(calculator, {"brand": brand, "segment": value})
        line = {}
        for value, _label in _THERAPY_LINES:
            line[value] = await self._value(calculator, {"brand": brand, "therapy_line": value})

        return {"brand": brand, "headline_nrx": headline, "severity": severity, "line": line}

    def _render(self, profiles: List[Dict[str, Any]], brand_requested: Optional[str]) -> str:
        """Markdown narrative with the real per-segment counts."""
        parts: List[str] = []
        scope = brand_requested or "all brands"
        parts.append(f"**Patient cohort profile — {scope}**")
        parts.append(
            "Eligible prescribing population sized by the clinical segment axes "
            "that exist in the data today (new prescriptions, most recent 30 days):"
        )
        for p in profiles:
            parts.append(f"\n### {p['brand']} — {self._fmt(p['headline_nrx'])} new-Rx patients")
            parts.append("\n_By disease-severity tier:_\n")
            parts.append("| Severity tier | New-Rx patients |\n|---|---|")
            for value, label in _SEVERITY_TIERS:
                parts.append(f"| {label} | {self._fmt(p['severity'].get(value))} |")
            parts.append("\n_By line of therapy:_\n")
            parts.append("| Line of therapy | New-Rx patients |\n|---|---|")
            for value, label in _THERAPY_LINES:
                parts.append(f"| {label} | {self._fmt(p['line'].get(value))} |")
        parts.append(
            "\n_These are population sizes, not a patient list. To materialize the "
            "actual eligible patients for an ML pipeline — with full inclusion/"
            "exclusion criteria and an audit trail — use the cohort pipeline "
            "(scope_definer → cohort_constructor)._"
        )
        return "\n".join(parts)

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "n/a"
        return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"

    def _failed(self, message: str) -> Dict[str, Any]:
        """Honest fail-closed result (dispatcher fails the dispatch on this)."""
        return {"status": "failed", "errors": [{"error": message}], "narrative": ""}
