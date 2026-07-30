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

#1356 extension (ratified ``extend:cohort_profiler`` ruling, 2026-07-29; parts
1 + 2 — part 3, propensity ranking, is blocked on #1354):

* **Parameter binding** — the ask is parsed (:mod:`.ask`) and its parameters
  BIND into the profile query: brand (named directly or via indication, fixing
  benchmark q11's canned all-brands answer), age criteria (servable —
  ``patient_journeys.age_at_diagnosis`` is fully populated), and, for criteria
  the data model cannot serve (e.g. "diagnosed in 2024": zero diagnosis events,
  no diagnosis-date column), an HONEST accounting naming exactly which criteria
  were applied and which could not be — never answering a different question
  than was asked. If nothing the ask pinned down can be served, it fails closed
  with guidance instead of returning a canned profile.
* **HCP-entity cohorts** — "HCPs who prescribed >50 TRx last quarter" runs a
  per-HCP aggregation over an explicit half-open window with a threshold filter
  (allowlisted ``kpi_query`` statements registered in migration 117, over the
  same ``treatment_events`` prescription substrate as the platform TRx KPI),
  returning cohort size + specialty / priority-tier breakdowns that mirror the
  patient-profile shape.
* **Cache identity** — the 26.4ms byte-identical q11/q15 repeat was the
  (context-keyed) Redis KPI cache serving two asks that had collapsed to the
  SAME parameterless call set. Binding the parameters restores the keying:
  patient KPI calls now carry the bound brand in their cache context, and the
  criteria/HCP paths go through the allowlist RPC with every ask parameter
  (brand, age bounds, window, threshold) bound positionally — two different
  asks can no longer share a payload.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .ask import CohortAsk, Criterion, Window, parse_cohort_ask

logger = logging.getLogger(__name__)

# The three commercial brands on the synthetic substrate. Brand predicate in the
# KPI registry is a case-SENSITIVE exact match (``brand::text = $1``), so callers
# must be normalised to this canonical casing before hitting the calculator.
SUPPORTED_BRANDS: Tuple[str, ...] = ("Remibrutinib", "Fabhalta", "Kisqali")

# NRx (new prescriptions) — the "new prescribing patients" proxy used to SIZE the
# population. Reusing the mig-105 ``_segment`` / ``_line`` variants keeps this in
# lock-step with the chat tool and the /api/kpis breakdown.
_NRX_KPI_ID = "WS3-BI-006"

# Allowlisted kpi_query statement ids (migration 117). Each has an
# ``_include_synthetic`` twin following the ADDITIVE-variant idiom of
# ``synthetic_mode.region_query_id`` (deliberately absent from
# SYNTHETIC_TWINNED_QUERY_IDS, which is locked to migrations 066/085/095).
_HCP_COHORT_QUERY_ID = "cohort_profiler_hcp_trx_cohort"
_PATIENT_CRITERIA_QUERY_ID = "cohort_profiler_patient_criteria_profile"

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

_MATERIALIZE_FOOTER = (
    "\n_These are population sizes, not a patient list. To materialize the "
    "actual eligible patients for an ML pipeline — with full inclusion/"
    "exclusion criteria and an audit trail — use the cohort pipeline "
    "(scope_definer → cohort_constructor)._"
)


def _profiler_query_id(base: str) -> str:
    """Synthetic-visibility variant id for the #1356 allowlist statements.

    Follows the additive-variant idiom (``synthetic_mode.region_query_id``):
    these ids are ADDITIVE and deliberately absent from
    ``SYNTHETIC_TWINNED_QUERY_IDS`` (locked to migrations 066/085/095 by CI), so
    the ``_include_synthetic`` suffix is appended HERE under the showcase flag.
    """
    from src.kpi.synthetic_mode import kpi_include_synthetic

    return f"{base}_include_synthetic" if kpi_include_synthetic() else base


class CohortProfilerAgent:
    """Profiles the eligible prescribing population by clinical segment axes."""

    def __init__(self) -> None:
        # No graph / LLM / mlflow ceremony — this agent is pure computation over
        # the shared KPI calculator, mirroring the health_score fast-path style.
        self._log = logger

    # ------------------------------------------------------------------ public
    async def analyze(self, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the ask-bound cohort profile and return a synthesizer-ready result.

        Named ``analyze`` (the dispatcher's default method spec) rather than
        ``run`` because the Tier 1-5 ``AGENT_METHOD_MAP`` contract is pinned to 13
        agents by the registry-consistency guard; Tier-0 agents like this one and
        cohort_constructor dispatch via the fall-through ``analyze`` contract and
        deliberately carry no method-map entry.

        ``agent_input`` is the merged orchestrator payload; the input resolver
        grounds ``brand`` (canonical casing) from the parsed query / user_context
        when the NLP layer supplied one — and since #1356 the raw query text is
        parsed as the fallback, so a brand/criteria/threshold named in the ask
        itself binds even when no structured entities were grounded (the exact
        q11/q15 failure mode). When no brand is grounded anywhere we profile
        every supported brand rather than fabricate a default.
        """
        query = str(agent_input.get("query") or "")
        ask = parse_cohort_ask(query, brand_hint=agent_input.get("brand"), today=self._today())

        if ask.entity_type == "hcp":
            return await self._analyze_hcp(ask)
        return await self._analyze_patients(ask)

    # ----------------------------------------------------------- patient path
    async def _analyze_patients(self, ask: CohortAsk) -> Dict[str, Any]:
        servable = [c for c in ask.criteria if c.servable]
        unserved = [c for c in ask.criteria if not c.servable]

        # The ask pinned down ONLY things the data model cannot serve (and no
        # brand): a canned profile would answer a different question than was
        # asked. Fail closed with guidance instead (#1356 part 1).
        if unserved and not servable and not ask.brand:
            details = "; ".join(f"'{c.label}' — {c.guidance}" for c in unserved)
            return self._failed(
                "no requested criterion can be served by the data model: " + details
            )

        if servable:
            return await self._profile_patients_with_criteria(ask, servable, unserved)

        # No extra servable criteria → the original mig-105 KPI-calculator path
        # (numbers stay in lock-step with the live chat UI), brand-bound.
        return await self._profile_patients_legacy(ask, unserved)

    async def _profile_patients_legacy(
        self, ask: CohortAsk, unserved: List[Criterion]
    ) -> Dict[str, Any]:
        brand = ask.brand
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

        applied = [f"brand = {brand}"] if brand else []
        narrative = self._render(profiles, brand_requested=brand)
        if applied or unserved:
            narrative += self._render_criteria_accounting(applied, unserved)
        return {
            "status": "completed",
            "narrative": narrative,
            "cohort_profile": {
                "segment_axis": "severity+line_of_therapy",
                "brands": profiles,
                "criteria_applied": applied,
                "criteria_not_applied": [
                    {"label": c.label, "guidance": c.guidance} for c in unserved
                ],
            },
            "confidence": 0.9,
            "recommendations": [
                "Materialize the eligible patient list for ML via the cohort "
                "pipeline (scope_definer → cohort_constructor) when you need the "
                "actual patient rows, not just the population size.",
            ],
        }

    async def _profile_patients_with_criteria(
        self, ask: CohortAsk, servable: List[Criterion], unserved: List[Criterion]
    ) -> Dict[str, Any]:
        """Criteria-bound profile via the allowlisted mig-117 statement.

        Binds brand + age bounds into ONE grouped statement over the same
        NRx substrate as the mig-105 path (prescription events, sequence 1,
        most recent 30 days of data) joined to ``patient_journeys`` for
        ``age_at_diagnosis`` / ``segment_assignment`` / ``prior_therapy_lines``
        (all fully populated — verified READ-ONLY 2026-07-30).
        """
        age_min = next((c.value for c in servable if c.kind == "age_min"), None)
        age_max = next((c.value for c in servable if c.kind == "age_max"), None)

        try:
            rows = await self._rpc_rows(
                _profiler_query_id(_PATIENT_CRITERIA_QUERY_ID),
                [ask.brand, age_min, age_max],
            )
        except Exception as e:
            return self._failed(f"criteria-bound cohort query unavailable: {e}")

        severity: Dict[str, float] = {}
        line: Dict[str, float] = {}
        headline = 0.0
        for row in rows:
            n = float(row.get("nrx") or 0)
            headline += n
            sev = str(row.get("severity") or "")
            tl = str(row.get("therapy_line") if row.get("therapy_line") is not None else "")
            if sev:
                severity[sev] = severity.get(sev, 0.0) + n
            if tl:
                line[tl] = line.get(tl, 0.0) + n

        if not rows or not headline:
            return self._failed(
                "no prescribing population matches the servable criteria ("
                + self._applied_criteria_text(ask, servable)
                + ") — nothing to profile (no values were fabricated)"
            )

        scope = ask.brand or "all brands"
        profile = {
            "brand": scope,
            "headline_nrx": headline,
            "severity": severity,
            "line": line,
        }
        applied = self._applied_criteria_list(ask, servable)

        parts: List[str] = [f"**Patient cohort profile — {scope} (criteria-bound)**"]
        parts.append(
            "Eligible prescribing population matching the servable criteria "
            "(new prescriptions, most recent 30 days of data):"
        )
        parts.append(f"\n### {scope} — {self._fmt(headline)} new-Rx patients matching criteria")
        parts.append("\n_By disease-severity tier:_\n")
        parts.append("| Severity tier | New-Rx patients |\n|---|---|")
        for value, label in _SEVERITY_TIERS:
            parts.append(f"| {label} | {self._fmt(severity.get(value))} |")
        parts.append("\n_By line of therapy:_\n")
        parts.append("| Line of therapy | New-Rx patients |\n|---|---|")
        for value, label in _THERAPY_LINES:
            parts.append(f"| {label} | {self._fmt(line.get(value))} |")
        narrative = "\n".join(parts)
        narrative += self._render_criteria_accounting(applied, unserved)
        narrative += _MATERIALIZE_FOOTER

        return {
            "status": "completed",
            "narrative": narrative,
            "cohort_profile": {
                "segment_axis": "severity+line_of_therapy",
                "entity": "patient",
                "brands": [profile],
                "criteria_applied": applied,
                "criteria_not_applied": [
                    {"label": c.label, "guidance": c.guidance} for c in unserved
                ],
            },
            "confidence": 0.9,
            "recommendations": [
                "Materialize the eligible patient list for ML via the cohort "
                "pipeline (scope_definer → cohort_constructor) when you need the "
                "actual patient rows, not just the population size.",
            ],
        }

    @staticmethod
    def _applied_criteria_list(ask: CohortAsk, servable: List[Criterion]) -> List[str]:
        applied: List[str] = []
        if ask.brand:
            applied.append(f"brand = {ask.brand}")
        for c in servable:
            if c.kind == "age_min":
                applied.append(f"age at diagnosis > {c.value} ('{c.label}')")
            elif c.kind == "age_max":
                applied.append(f"age at diagnosis < {c.value} ('{c.label}')")
        return applied

    def _applied_criteria_text(self, ask: CohortAsk, servable: List[Criterion]) -> str:
        return "; ".join(self._applied_criteria_list(ask, servable)) or "none"

    @staticmethod
    def _render_criteria_accounting(applied: List[str], unserved: List[Criterion]) -> str:
        """Honest per-criterion accounting: EXACTLY which bound, which could not."""
        parts: List[str] = ["\n\n**Criteria accounting**"]
        if applied:
            parts.append("- Applied: " + "; ".join(applied) + ".")
        for c in unserved:
            parts.append(f"- NOT applied — '{c.label}' could not be served: {c.guidance}")
        parts.append("- No other criteria were applied.")
        return "\n".join(parts)

    # --------------------------------------------------------------- HCP path
    async def _analyze_hcp(self, ask: CohortAsk) -> Dict[str, Any]:
        """HCP-entity cohort with a quantitative KPI threshold (#1356 part 2).

        Per-HCP aggregation over an explicit half-open window with a threshold
        filter, over the SAME ``treatment_events`` prescription substrate as the
        platform TRx KPI (``business_impact_trx``), joined to ``hcp_profiles``
        for the specialty / priority-tier segment axes. A zero-match cohort over
        a NONZERO prescribing base is an honest answer (the threshold filtered
        everyone out), distinguished from a genuine empty by a threshold-free
        probe — only the latter fails closed.
        """
        if ask.threshold is not None and not ask.threshold.servable:
            return self._failed(
                f"cannot serve the '{ask.threshold.label}' threshold: {ask.threshold.guidance}"
            )

        window = ask.window or self._default_hcp_window()
        thr = ask.threshold.min_exclusive if ask.threshold else 0
        params: List[Any] = [ask.brand, window.start.isoformat(), window.end.isoformat(), thr]
        qid = _profiler_query_id(_HCP_COHORT_QUERY_ID)

        try:
            rows = await self._rpc_rows(qid, params)
            base_rows: Optional[List[Dict[str, Any]]] = None
            if not rows and thr > 0:
                # Distinguish "threshold filtered everyone out" (honest zero)
                # from "no prescribing data at all" (genuine empty).
                base_rows = await self._rpc_rows(qid, [ask.brand, params[1], params[2], 0])
        except Exception as e:
            return self._failed(f"HCP cohort query unavailable: {e}")

        if not rows and not base_rows:
            return self._failed(
                "no prescribing HCPs found for "
                + (ask.brand or "any supported brand")
                + f" in {window.label} ({window.start.isoformat()} → "
                + f"{(window.end - timedelta(days=1)).isoformat()}) — nothing to "
                "profile (no values were fabricated)"
            )

        cohort_size = 0
        total_trx = 0.0
        max_trx = 0.0
        specialty: Dict[str, int] = {}
        tiers: Dict[str, int] = {}
        for row in rows:
            n = int(row.get("n_hcps") or 0)
            cohort_size += n
            total_trx += float(row.get("total_trx") or 0)
            max_trx = max(max_trx, float(row.get("max_trx") or 0))
            spec = str(row.get("specialty") or "unknown")
            specialty[spec] = specialty.get(spec, 0) + n
            tier = row.get("priority_tier")
            tier_key = str(tier) if tier is not None else "unknown"
            tiers[tier_key] = tiers.get(tier_key, 0) + n

        base_size = sum(int(r.get("n_hcps") or 0) for r in base_rows) if base_rows else None
        narrative = self._render_hcp(
            ask, window, thr, cohort_size, total_trx, max_trx, specialty, tiers, base_size
        )

        return {
            "status": "completed",
            "narrative": narrative,
            "cohort_profile": {
                "entity": "hcp",
                "segment_axis": "specialty+priority_tier",
                "brand": ask.brand,
                "window": {
                    "label": window.label,
                    "start": window.start.isoformat(),
                    "end_exclusive": window.end.isoformat(),
                    "explicit": window.explicit,
                },
                "threshold": {
                    "metric": "trx",
                    "min_exclusive": thr,
                    "stated": ask.threshold.label if ask.threshold else None,
                },
                "cohort_size": cohort_size,
                "specialty": specialty,
                "priority_tier": tiers,
                "trx_total": total_trx,
                "trx_max": max_trx,
            },
            "confidence": 0.9,
            "recommendations": [
                "'High-value' here is threshold-filtered TRx volume only; "
                "model-scored adoption-propensity ranking is planned once the "
                "hcp_adoption models are promoted (#1354).",
            ],
        }

    def _default_hcp_window(self) -> Window:
        today = self._today()
        return Window(
            label="most recent 90 days (no time window was named in the ask)",
            start=today - timedelta(days=90),
            end=today + timedelta(days=1),
            explicit=False,
        )

    def _render_hcp(
        self,
        ask: CohortAsk,
        window: Window,
        thr: int,
        cohort_size: int,
        total_trx: float,
        max_trx: float,
        specialty: Dict[str, int],
        tiers: Dict[str, int],
        base_size: Optional[int],
    ) -> str:
        scope = ask.brand or "all brands"
        window_disp = f"{window.start.isoformat()} → {(window.end - timedelta(days=1)).isoformat()}"
        thr_disp = ask.threshold.label if ask.threshold else f"more than {thr} TRx"

        parts: List[str] = [f"**HCP cohort profile — {scope}**"]
        parts.append(f"HCPs with {thr_disp} (prescription events, {window.label}: {window_disp}):")
        if cohort_size == 0:
            parts.append(
                f"\n**0 HCPs** met the threshold (> {thr} TRx) in this window"
                + (
                    f" — of {base_size:,} HCPs with any prescriptions in the same window."
                    if base_size is not None
                    else "."
                )
            )
            parts.append(
                "_This is a real zero over a real prescribing base, not missing "
                "data: lower the threshold or widen the window to get a non-empty "
                "cohort._"
            )
        else:
            parts.append(
                f"\n### {cohort_size:,} HCPs — {self._fmt(total_trx)} TRx combined "
                f"(top prescriber: {self._fmt(max_trx)} TRx)"
            )
            parts.append("\n_By specialty:_\n")
            parts.append("| Specialty | HCPs |\n|---|---|")
            for spec, n in sorted(specialty.items(), key=lambda kv: -kv[1]):
                parts.append(f"| {spec} | {n:,} |")
            parts.append("\n_By priority tier:_\n")
            parts.append("| Priority tier | HCPs |\n|---|---|")
            for tier, n in sorted(tiers.items()):
                parts.append(f"| Tier {tier} | {n:,} |")
        if not window.explicit:
            parts.append(
                f"\n_No time window was named — defaulted to the {window.label.split(' (')[0]} "
                f"({window_disp}). Name a window (e.g. 'last quarter') to change it._"
            )
        parts.append(
            "\n_These are cohort sizes from per-HCP TRx aggregation (same "
            "prescription substrate as the platform TRx KPI), not an outreach "
            "list with contact routing._"
        )
        return "\n".join(parts)

    # --------------------------------------------------------------- internals
    def _get_calculator(self) -> Any:
        # Local import avoids an agent <-> api.routes import cycle at module load
        # (same pattern as chatbot_tools.kpi_calculate_tool).
        from src.api.routes.kpi import get_kpi_calculator

        return get_kpi_calculator()

    def _get_db_client(self) -> Any:
        # Same client + allowlist-RPC path as the KPI calculators
        # (BusinessImpactCalculator._execute_query): vetted read-only statements
        # from kpi_query_registry only — never raw SQL from the agent.
        from src.repositories import get_supabase_client

        return get_supabase_client()

    def _today(self) -> date:
        """Injectable clock for deterministic window math in tests."""
        return date.today()

    async def _rpc_rows(self, query_id: str, params: List[Any]) -> List[Dict[str, Any]]:
        """One allowlisted kpi_query RPC call → its rows (sync client offloaded)."""
        client = self._get_db_client()

        def _call() -> Any:
            return client.rpc("kpi_query", {"query_id": query_id, "params": params}).execute()

        response = await asyncio.to_thread(_call)
        data = getattr(response, "data", None)
        return list(data or [])

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
        parts.append(_MATERIALIZE_FOOTER)
        return "\n".join(parts)

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "n/a"
        return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"

    def _failed(self, message: str) -> Dict[str, Any]:
        """Honest fail-closed result (dispatcher fails the dispatch on this)."""
        return {"status": "failed", "errors": [{"error": message}], "narrative": ""}
