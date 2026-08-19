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
* **Volume tiers (#1736)** — "Segment HCPs by prescription volume into
  high, medium, and low tiers" (eval 4.3, undeliverable-promise shape across
  two runs) buckets the same per-HCP TRx cohort into value-based terciles
  computed WITHIN the queried scope (mig-130 statements), returning real
  counts per high/medium/low tier with the measured cut points disclosed —
  never the DISTINCT ``hcp_profiles.priority_tier`` targeting attribute.
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

from .ask import CohortAsk, Criterion, Window, merge_cohort_asks, parse_cohort_ask

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
# Region-bound sibling (mig-129, #1693): same statement + $5 region matched
# case-insensitively against hcp_profiles.geographic_region; selected only
# when the ask names a region so the 4-param base id keeps serving unscoped
# asks (additive-variant idiom, and pre-migration code paths stay valid).
_HCP_COHORT_REGION_QUERY_ID = "cohort_profiler_hcp_trx_cohort_region"
# Volume-tier siblings (mig-130, #1736): the SAME per-HCP TRx cohort CTE,
# bucketed into high/medium/low by value-based terciles computed WITHIN the
# queried scope (brand/window/threshold/region — measured 2026-08-19: the
# northeast cohort's cuts are 1/5 while the global cohort's are 2/5, so the
# cuts must follow the scope). Cut points ride along in every row and are
# disclosed as measured, scope-relative values. Additive-variant idiom: these
# ids are selected only when the ask names volume tiers, so the
# single-threshold ids above keep serving their existing consumers.
_HCP_VOLUME_TIER_QUERY_ID = "cohort_profiler_hcp_volume_tiers"
_HCP_VOLUME_TIER_REGION_QUERY_ID = "cohort_profiler_hcp_volume_tiers_region"
_PATIENT_CRITERIA_QUERY_ID = "cohort_profiler_patient_criteria_profile"
# Windowed sibling: [brand, start, end, min_age_exclusive, max_age_exclusive].
# The mig-044 RPC once capped at 4 positional params (so max-age was dropped);
# mig-120 (#1388) raised the cap to 6 and mig-122 (#1402) restored the max-age
# bound as $5 — both age bounds now co-bind with the window.
_PATIENT_CRITERIA_WINDOWED_QUERY_ID = "cohort_profiler_patient_criteria_profile_windowed"

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

# Rendering order + labels for the volume tiers (#1736). Vocabulary alignment:
# domain_vocabulary.yaml hcp_segments names these segments high_volume /
# medium_volume / low_volume; the keys here are the mig-130 statements'
# volume_tier values.
_VOLUME_TIER_ORDER: Tuple[Tuple[str, str], ...] = (
    ("high", "High volume"),
    ("medium", "Medium volume"),
    ("low", "Low volume"),
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

        # #1698: ``query`` is the chat model's rewrite, and the measured 2.1
        # defect is that rewrite silently dropping servable criteria. When the
        # live chat path threads the user's original ask alongside it, parse
        # that too and merge — a dropped criterion must still reach the
        # servable-binding / criteria_not_applied accounting below.
        raw = str(agent_input.get("raw_user_query") or "")
        if raw and raw != query:
            ask = merge_cohort_asks(
                ask,
                parse_cohort_ask(raw, brand_hint=agent_input.get("brand"), today=self._today()),
            )

        if ask.entity_type == "hcp":
            return await self._analyze_hcp(ask)
        return await self._analyze_patients(ask)

    # ----------------------------------------------------------- patient path
    async def _analyze_patients(self, ask: CohortAsk) -> Dict[str, Any]:
        servable = [c for c in ask.criteria if c.servable]
        unserved = [c for c in ask.criteria if not c.servable]

        # Region (#1693) binds on HCP cohorts only today (mig-129,
        # hcp_profiles.geographic_region). On a patient ask it flows through
        # the honest per-criterion accounting instead of silently profiling
        # an unscoped population as region-filtered.
        region_criteria = [c for c in servable if c.kind == "region"]
        if region_criteria:
            servable = [c for c in servable if c.kind != "region"]
            for c in region_criteria:
                unserved.append(
                    Criterion(
                        kind=c.kind,
                        label=c.label,
                        servable=False,
                        text_value=c.text_value,
                        guidance=(
                            "geographic filters are served on HCP-entity cohorts "
                            "(hcp_profiles.geographic_region) — re-ask as an HCP "
                            "cohort (e.g. 'HCPs in the northeast'), or materialize "
                            "a region-scoped patient cohort via the ML pipeline "
                            "(scope_definer → cohort_constructor)"
                        ),
                    )
                )

        # A KPI threshold on a PATIENT-entity ask is recognized but NOT
        # servable on this path today (no allowlisted per-patient KPI
        # aggregation exists) — it flows through the same honest per-criterion
        # accounting as any other unservable criterion, never silently dropped
        # (#1356 codex iter-1 finding 1: 'size the <brand> cohort' and the same
        # ask + '>50 TRx' are materially different questions).
        if ask.threshold is not None:
            unserved.append(
                Criterion(
                    kind="kpi_threshold",
                    label=ask.threshold.label,
                    servable=False,
                    value=ask.threshold.min_exclusive,
                    guidance=(
                        "quantitative prescribing thresholds are served on "
                        "HCP-entity cohorts only today (TRx metric) — re-ask as "
                        "e.g. 'HCPs who prescribed more than 50 TRx last "
                        "quarter', or materialize per-patient criteria via the "
                        "ML cohort pipeline (scope_definer → cohort_constructor)"
                    ),
                )
            )

        # Volume tiers on a PATIENT ask (#1736): per-HCP TRx is a PRESCRIBER
        # axis — recognized, honestly not servable here, never silently
        # dropped (mirrors the threshold accounting above).
        if ask.volume_tiers:
            unserved.append(
                Criterion(
                    kind="volume_tiers",
                    label="high/medium/low prescription-volume tiers",
                    servable=False,
                    guidance=(
                        "prescription-volume tiers bucket PRESCRIBERS by "
                        "per-HCP TRx — re-ask as an HCP segmentation (e.g. "
                        "'Segment HCPs by prescription volume into high, "
                        "medium and low tiers'), or use the severity / "
                        "line-of-therapy axes for patient populations"
                    ),
                )
            )

        # A recognized WINDOW on a patient ask BINDS (#1356 codex iter-2): it
        # bounds the NRx counting window — 'patients with new prescriptions in
        # [start, end)' — the platform's established windowed-KPI semantic
        # (mig-084/105 `_windowed` variants, calculator context['window'],
        # already part of the KPI cache key). Both age bounds now co-bind with
        # the window: mig-120 (#1388) raised the kpi_query cap to 6 params and
        # mig-122 (#1402) restored the windowed statement's max-age bound as $5,
        # so max-age + window is served rather than disclosed as not-applied.

        # The ask pinned down ONLY things the data model cannot serve (and no
        # brand, no window): a canned profile would answer a different question
        # than was asked. Fail closed with guidance instead (#1356 part 1).
        if unserved and not servable and not ask.brand and ask.window is None:
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

        window = ask.window
        profiles: List[Dict[str, Any]] = []
        for b in brands:
            profile = await self._profile_brand(calculator, b, window=window)
            if profile is not None:
                profiles.append(profile)

        if not profiles:
            # Every requested brand returned no prescribing population — this is a
            # genuine empty/failure state, not a zero to narrate. Fail closed.
            return self._failed(
                "no prescribing population found for "
                + (brand or "any supported brand")
                + (f" in {window.label}" if window else "")
                + " — nothing to profile (no values were fabricated)"
            )

        applied = [f"brand = {brand}"] if brand else []
        if window:
            applied.append(self._window_applied_text(window))
        narrative = self._render(profiles, brand_requested=brand, window=window)
        if applied or unserved:
            narrative += self._render_criteria_accounting(applied, unserved)
        return {
            "status": "completed",
            "narrative": narrative,
            "cohort_profile": {
                "segment_axis": "severity+line_of_therapy",
                "brands": profiles,
                "window": self._window_profile(window),
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
        most recent 30 days of data — or the ask's explicit window via the
        `_windowed` sibling, params [brand, start, end, min_age, max_age])
        joined to ``patient_journeys`` for ``age_at_diagnosis`` /
        ``segment_assignment`` / ``prior_therapy_lines`` (all fully populated —
        verified READ-ONLY 2026-07-30).
        """
        age_min = next((c.value for c in servable if c.kind == "age_min"), None)
        age_max = next((c.value for c in servable if c.kind == "age_max"), None)
        window = ask.window

        try:
            if window is not None:
                # Both age bounds bind alongside the window now the kpi_query
                # RPC allows 6 positional params (#1388 / mig-120) and mig-122
                # (#1402) restored the windowed statement's $5 max-age bound.
                rows = await self._rpc_rows(
                    _profiler_query_id(_PATIENT_CRITERIA_WINDOWED_QUERY_ID),
                    [
                        ask.brand,
                        window.start.isoformat(),
                        window.end.isoformat(),
                        age_min,
                        age_max,
                    ],
                )
            else:
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
        if window:
            applied.append(self._window_applied_text(window))

        parts: List[str] = [f"**Patient cohort profile — {scope} (criteria-bound)**"]
        parts.append(
            "Eligible prescribing population matching the servable criteria "
            f"(new prescriptions, {self._window_phrase(window)}):"
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
                "window": self._window_profile(window),
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

    # ------------------------------------------------- window helpers (iter-2)
    @staticmethod
    def _window_phrase(window: Optional[Window]) -> str:
        """Narrative phrase for the NRx counting window — explicit dates when a
        window is bound, the mig-105 default wording otherwise."""
        if window is None:
            return "most recent 30 days of data"
        end_inclusive = (window.end - timedelta(days=1)).isoformat()
        return f"{window.label}: {window.start.isoformat()} → {end_inclusive}"

    @staticmethod
    def _window_applied_text(window: Window) -> str:
        end_inclusive = (window.end - timedelta(days=1)).isoformat()
        return f"window = {window.label} ({window.start.isoformat()} → {end_inclusive})"

    @staticmethod
    def _window_profile(window: Optional[Window]) -> Optional[Dict[str, Any]]:
        if window is None:
            return None
        return {
            "label": window.label,
            "start": window.start.isoformat(),
            "end_exclusive": window.end.isoformat(),
            "explicit": window.explicit,
        }

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

        # Region (#1693) BINDS on this path via the mig-129 `_region` statement
        # (hcp_profiles.geographic_region). Every other recognized criterion
        # (age / diagnosis-year) is a patient-journey attribute — none bind to
        # an HCP cohort today. They must surface in the per-criterion
        # accounting, never vanish (#1356 codex iter-1 finding 2); and if they
        # are the ONLY specifics in the ask (no brand, no threshold, no region,
        # no explicit window), profiling all prescribing HCPs would answer a
        # different question — fail closed with guidance.
        region = next((c for c in ask.criteria if c.kind == "region"), None)
        unserved = [self._hcp_unservable(c) for c in ask.criteria if c.kind != "region"]
        if unserved and not (
            ask.brand
            or ask.threshold
            or region
            or ask.volume_tiers
            or (ask.window and ask.window.explicit)
        ):
            details = "; ".join(f"'{c.label}' — {c.guidance}" for c in unserved)
            return self._failed("no requested criterion can be served on an HCP cohort: " + details)

        window = ask.window or self._default_hcp_window()
        thr = ask.threshold.min_exclusive if ask.threshold else 0
        if ask.volume_tiers:
            # #1736: the tier ask is served by the mig-130 tercile statements
            # (counts per high/medium/low tier); threshold/region/window all
            # compose exactly as on the single-threshold path below.
            return await self._analyze_hcp_volume_tiers(ask, window, thr, region, unserved)
        params: List[Any] = [ask.brand, window.start.isoformat(), window.end.isoformat(), thr]
        if region is not None:
            params.append(region.text_value)
            qid = _profiler_query_id(_HCP_COHORT_REGION_QUERY_ID)
        else:
            qid = _profiler_query_id(_HCP_COHORT_QUERY_ID)

        try:
            rows = await self._rpc_rows(qid, params)
            base_rows: Optional[List[Dict[str, Any]]] = None
            if not rows and thr > 0:
                # Distinguish "threshold filtered everyone out" (honest zero)
                # from "no prescribing data at all" (genuine empty). The probe
                # keeps the region bind so the contrast stays within scope.
                base_params = [ask.brand, params[1], params[2], 0]
                if region is not None:
                    base_params.append(region.text_value)
                base_rows = await self._rpc_rows(qid, base_params)
        except Exception as e:
            return self._failed(f"HCP cohort query unavailable: {e}")

        if not rows and not base_rows:
            return self._failed(
                "no prescribing HCPs found for "
                + (ask.brand or "any supported brand")
                + (f" in the {region.text_value} region" if region is not None else "")
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
            ask, window, thr, cohort_size, total_trx, max_trx, specialty, tiers, base_size, region
        )
        if unserved or region is not None:
            applied = []
            if ask.brand:
                applied.append(f"brand = {ask.brand}")
            if region is not None:
                applied.append(
                    f"region = {region.text_value} "
                    f"(hcp_profiles.geographic_region, '{region.label}')"
                )
            if ask.threshold:
                applied.append(f"TRx threshold ({ask.threshold.label})")
            applied.append(
                f"window = {window.label} ({window.start.isoformat()} → "
                f"{(window.end - timedelta(days=1)).isoformat()})"
            )
            narrative += self._render_criteria_accounting(applied, unserved)

        return {
            "status": "completed",
            "narrative": narrative,
            "cohort_profile": {
                "entity": "hcp",
                "segment_axis": "specialty+priority_tier",
                "brand": ask.brand,
                # #1693/#1694 scope honesty: region is None unless it was
                # ACTUALLY bound as a filter — synthesis must never assert a
                # regional scope this field does not carry.
                "region": region.text_value if region is not None else None,
                "region_applied": region is not None,
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
                "criteria_not_applied": [
                    {"label": c.label, "guidance": c.guidance} for c in unserved
                ],
            },
            "confidence": 0.9,
            "recommendations": [
                "'High-value' here is threshold-filtered TRx volume only; "
                "model-scored adoption-propensity ranking is planned once the "
                "hcp_adoption models are promoted (#1354).",
            ],
        }

    @staticmethod
    def _hcp_unservable(c: Criterion) -> Criterion:
        """Re-tag a recognized criterion with HCP-path guidance.

        All recognized criteria (age bounds, diagnosis-year) are
        patient-journey attributes; none bind to an HCP cohort today, so on
        this path each is unservable — with guidance saying why.
        """
        if c.kind in ("age_min", "age_max"):
            guidance = (
                "age criteria bind to patient_journeys.age_at_diagnosis (a "
                "patient attribute); HCP cohorts have no age axis today — "
                "re-ask as a patient cohort, or drop the age bound"
            )
        else:
            guidance = c.guidance or "not servable on an HCP cohort"
        return Criterion(
            kind=c.kind, label=c.label, servable=False, value=c.value, guidance=guidance
        )

    def _default_hcp_window(self) -> Window:
        # Inclusive-today semantics: exactly 90 dates in [today-89, today+1).
        today = self._today()
        return Window(
            label="most recent 90 days (no time window was named in the ask)",
            start=today - timedelta(days=89),
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
        region: Optional[Criterion] = None,
    ) -> str:
        scope = ask.brand or "all brands"
        if region is not None:
            scope += f", {region.text_value} region"
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

    # ------------------------------------------------- volume tiers (#1736)
    async def _analyze_hcp_volume_tiers(
        self,
        ask: CohortAsk,
        window: Window,
        thr: int,
        region: Optional[Criterion],
        unserved: List[Criterion],
    ) -> Dict[str, Any]:
        """HCP volume-tier segmentation: real counts per high/medium/low tier.

        Serves the eval-4.3 promise ("counts per tier, plus specialty where
        available") in ONE allowlisted call: the mig-130 statements bucket the
        per-HCP TRx cohort into value-based terciles computed WITHIN the
        queried scope (brand / window / threshold / region) and return the
        measured cut points in every row, so the tiers are disclosed as
        scope-relative measurements — never fixed global constants, and never
        the DISTINCT ``hcp_profiles.priority_tier`` targeting attribute.
        """
        params: List[Any] = [ask.brand, window.start.isoformat(), window.end.isoformat(), thr]
        if region is not None:
            params.append(region.text_value)
            qid = _profiler_query_id(_HCP_VOLUME_TIER_REGION_QUERY_ID)
        else:
            qid = _profiler_query_id(_HCP_VOLUME_TIER_QUERY_ID)

        try:
            rows = await self._rpc_rows(qid, params)
            base_rows: Optional[List[Dict[str, Any]]] = None
            if not rows and thr > 0:
                # Distinguish "threshold filtered everyone out" (honest zero)
                # from "no prescribing data at all" (genuine empty) — same
                # probe idiom as the single-threshold path.
                base_params: List[Any] = [ask.brand, params[1], params[2], 0]
                if region is not None:
                    base_params.append(region.text_value)
                base_rows = await self._rpc_rows(qid, base_params)
        except Exception as e:
            return self._failed(f"HCP volume-tier query unavailable: {e}")

        if not rows and not base_rows:
            return self._failed(
                "no prescribing HCPs found for "
                + (ask.brand or "any supported brand")
                + (f" in the {region.text_value} region" if region is not None else "")
                + f" in {window.label} ({window.start.isoformat()} → "
                + f"{(window.end - timedelta(days=1)).isoformat()}) — nothing to "
                "segment into volume tiers (no values were fabricated)"
            )

        tiers: Dict[str, Dict[str, Any]] = {
            key: {"n_hcps": 0, "trx_total": 0.0, "trx_min": None, "trx_max": None}
            for key, _label in _VOLUME_TIER_ORDER
        }
        specialty: Dict[str, int] = {}
        cut_low: Optional[int] = None
        cut_medium: Optional[int] = None
        cohort_size = 0
        total_trx = 0.0
        for row in rows:
            key = str(row.get("volume_tier") or "")
            if key not in tiers:  # pragma: no cover - defensive (unknown bucket)
                continue
            n = int(row.get("n_hcps") or 0)
            trx = float(row.get("total_trx") or 0)
            bucket = tiers[key]
            bucket["n_hcps"] += n
            bucket["trx_total"] += trx
            row_min = row.get("min_trx")
            row_max = row.get("max_trx")
            if row_min is not None:
                bucket["trx_min"] = (
                    row_min if bucket["trx_min"] is None else min(bucket["trx_min"], row_min)
                )
            if row_max is not None:
                bucket["trx_max"] = (
                    row_max if bucket["trx_max"] is None else max(bucket["trx_max"], row_max)
                )
            cohort_size += n
            total_trx += trx
            spec = str(row.get("specialty") or "unknown")
            specialty[spec] = specialty.get(spec, 0) + n
            if cut_low is None:
                cut_low = row.get("cut_low_max")
                cut_medium = row.get("cut_medium_max")

        base_size = sum(int(r.get("n_hcps") or 0) for r in base_rows) if base_rows else None
        narrative = self._render_hcp_volume_tiers(
            ask,
            window,
            thr,
            cohort_size,
            tiers,
            specialty,
            cut_low,
            cut_medium,
            base_size,
            region,
        )
        applied: List[str] = []
        if ask.brand:
            applied.append(f"brand = {ask.brand}")
        if region is not None:
            applied.append(
                f"region = {region.text_value} (hcp_profiles.geographic_region, '{region.label}')"
            )
        if ask.threshold:
            applied.append(f"TRx threshold ({ask.threshold.label})")
        applied.append(self._window_applied_text(window))
        if unserved or region is not None or ask.threshold:
            narrative += self._render_criteria_accounting(applied, unserved)

        return {
            "status": "completed",
            "narrative": narrative,
            "cohort_profile": {
                "entity": "hcp",
                "segment_axis": "volume_tier+specialty",
                "brand": ask.brand,
                "region": region.text_value if region is not None else None,
                "region_applied": region is not None,
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
                "volume_tiers": tiers,
                "tier_boundaries": {
                    "method": (
                        "value-based terciles (percentile_disc 1/3 and 2/3) of "
                        "the per-HCP TRx distribution within this scope; ties "
                        "share a tier"
                    ),
                    "low_max_trx": cut_low,
                    "medium_max_trx": cut_medium,
                },
                "specialty": specialty,
                "trx_total": total_trx,
                "criteria_not_applied": [
                    {"label": c.label, "guidance": c.guidance} for c in unserved
                ],
            },
            "confidence": 0.9,
            "recommendations": [
                "Tier boundaries are scope-relative terciles measured from this "
                "cohort's TRx distribution — for a FIXED cutoff cohort instead, "
                "state an explicit threshold (e.g. 'HCPs with more than 10 TRx').",
            ],
        }

    def _render_hcp_volume_tiers(
        self,
        ask: CohortAsk,
        window: Window,
        thr: int,
        cohort_size: int,
        tiers: Dict[str, Dict[str, Any]],
        specialty: Dict[str, int],
        cut_low: Optional[int],
        cut_medium: Optional[int],
        base_size: Optional[int],
        region: Optional[Criterion] = None,
    ) -> str:
        scope = ask.brand or "all brands"
        if region is not None:
            scope += f", {region.text_value} region"
        window_disp = f"{window.start.isoformat()} → {(window.end - timedelta(days=1)).isoformat()}"

        parts: List[str] = [f"**HCP volume-tier segmentation — {scope}**"]
        thr_note = f", TRx > {thr}" if thr else ""
        parts.append(
            "Prescribing HCPs bucketed into high/medium/low prescription-volume "
            f"tiers by per-HCP TRx (prescription events, {window.label}: "
            f"{window_disp}{thr_note}):"
        )
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
                "data: lower the threshold or widen the window, then re-run the "
                "volume-tier segmentation._"
            )
        else:
            parts.append(f"\n### {cohort_size:,} HCPs — counts per tier")
            parts.append("\n| Volume tier | Per-HCP TRx | HCPs | TRx combined |\n|---|---|---|---|")
            for key, label in _VOLUME_TIER_ORDER:
                bucket = tiers[key]
                rng = (
                    f"{bucket['trx_min']}–{bucket['trx_max']}"
                    if bucket["trx_min"] is not None
                    else "—"
                )
                parts.append(
                    f"| {label} | {rng} | {bucket['n_hcps']:,} | {self._fmt(bucket['trx_total'])} |"
                )
            parts.append(
                "\n_Tier boundaries are value-based terciles of THIS cohort's "
                f"per-HCP TRx distribution — measured cut points: low ≤ {cut_low} "
                f"< medium ≤ {cut_medium} < high. They are scope-relative "
                "measurements, not fixed global constants; HCPs with equal TRx "
                "always share a tier. This axis is computed from prescribing "
                "volume and is distinct from the static priority-tier targeting "
                "attribute._"
            )
            parts.append("\n_By specialty (all tiers combined):_\n")
            parts.append("| Specialty | HCPs |\n|---|---|")
            for spec, n in sorted(specialty.items(), key=lambda kv: -kv[1]):
                parts.append(f"| {spec} | {n:,} |")
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

    async def _profile_brand(
        self, calculator: Any, brand: str, window: Optional[Window] = None
    ) -> Optional[Dict[str, Any]]:
        """Real NRx headline + severity + line breakdown for one brand.

        A bound ``window`` rides in each calculator context: the business_impact
        calculator routes it to the mig-084/105 ``_windowed`` /
        ``_segment_windowed`` / ``_line_windowed`` registry variants, and the
        KPI cache keys on it — so a windowed and a windowless ask can never
        share cached values (#1356 codex iter-2).
        """

        def _ctx(base: Dict[str, Any]) -> Dict[str, Any]:
            if window is not None:
                base["window"] = {
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                }
            return base

        headline = await self._value(calculator, _ctx({"brand": brand}))
        if not headline:
            return None  # no prescribing population for this brand — skip honestly

        severity = {}
        for value, _label in _SEVERITY_TIERS:
            severity[value] = await self._value(
                calculator, _ctx({"brand": brand, "segment": value})
            )
        line = {}
        for value, _label in _THERAPY_LINES:
            line[value] = await self._value(
                calculator, _ctx({"brand": brand, "therapy_line": value})
            )

        return {"brand": brand, "headline_nrx": headline, "severity": severity, "line": line}

    def _render(
        self,
        profiles: List[Dict[str, Any]],
        brand_requested: Optional[str],
        window: Optional[Window] = None,
    ) -> str:
        """Markdown narrative with the real per-segment counts."""
        parts: List[str] = []
        scope = brand_requested or "all brands"
        parts.append(f"**Patient cohort profile — {scope}**")
        parts.append(
            "Eligible prescribing population sized by the clinical segment axes "
            f"that exist in the data today (new prescriptions, {self._window_phrase(window)}):"
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
