"""Capability catalog for the chat suggestion pills (``POST /api/chat/suggestions``).

Why this exists
---------------
Measured 2026-09-05 (docs/demos/results/2026-09-05_pill_suggestions_review/):
42% of the live suggestion pills asked for analyses the E2I assistant cannot
deliver -- SHAP recomputation, territory detail, trends of causal-registry
outcomes -- because the pill prompt described the assistant's abilities in one
prose sentence. A prompt carrying a catalog derived from code and data moved
the unanswerable share to 9% in a faithful prototype.

Everything list-shaped here is DERIVED, never transcribed (#1638 roster
pattern): KPI names from the registry, trend coverage from
``v_kpi_history_coverage``, axis-capable KPIs from the segmented-history
families, causal outcomes from the causal-path registry, agents from the
factory. The only hand-written text is the axis/composition RULES (guarded by
a test against ``kpi_calculate_tool``'s signature) and the per-route hints.

Design: docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from src.agents.factory import build_agent_roster_block
from src.kpi.models import Workstream
from src.kpi.registry import get_registry
from src.kpi.segmented_history import SEGMENTED_KPI_QUERY_FAMILIES

logger = logging.getLogger(__name__)

CoverageLoader = Callable[[], Awaitable[List[Dict[str, Any]]]]
OutcomesLoader = Callable[[], Awaitable[List[str]]]


@dataclass(frozen=True)
class KpiEntry:
    """One registry KPI as the prompt needs it."""

    id: str
    name: str
    workstream: str  # Workstream.value, e.g. "ws3_business"
    brand: Optional[str]  # brand-specific KPIs name their brand


@dataclass(frozen=True)
class CapabilityCatalog:
    """What the assistant can answer, as data. Rendered by :func:`render_catalog_block`."""

    kpis: Tuple[KpiEntry, ...]
    trend_kpi_ids: FrozenSet[str]  # have a materialized monthly series
    per_brand_only_trend_ids: FrozenSet[str]  # trend exists only in per-brand scopes
    axis_kpi_ids: FrozenSet[str]  # accept severity / therapy-line splits
    causal_outcomes: Tuple[str, ...]  # distinct end_node names in the causal registry
    agent_roster: str  # prompt-ready roster block from the factory
    degraded: Tuple[str, ...] = ()  # DB-backed fields that failed to load
    loaded_at: float = 0.0  # time.monotonic() at build

    def kpi_name(self, kpi_id: str) -> str:
        for entry in self.kpis:
            if entry.id == kpi_id:
                return entry.name
        return kpi_id


async def _default_coverage_loader() -> List[Dict[str, Any]]:
    # Build the repository directly (not via get_kpi_history_repository(), which
    # caches a client-less instance forever after one failed init) so a client
    # failure raises here, lands in ``degraded`` and is retried on the next refresh.
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.kpi_history import KPIHistoryRepository

    repo = KPIHistoryRepository(supabase_client=await get_async_supabase_client())
    return await repo.get_coverage()


async def _default_outcomes_loader() -> List[str]:
    from src.kpi.synthetic_mode import kpi_include_synthetic
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.causal_path import CausalPathRepository

    client = await get_async_supabase_client()
    repo = CausalPathRepository(client)
    return await repo.get_distinct_outcomes(include_synthetic=kpi_include_synthetic())


def _kpi_entries() -> Tuple[KpiEntry, ...]:
    entries = [
        KpiEntry(id=k.id, name=k.name, workstream=k.workstream.value, brand=k.brand)
        for k in get_registry().get_all()
    ]
    return tuple(sorted(entries, key=lambda e: (e.workstream, e.id)))


def _trend_sets(rows: Sequence[Dict[str, Any]]) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    scopes: Dict[str, set[str]] = {}
    for row in rows:
        kpi_id = str(row.get("kpi_id") or "")
        try:
            points = int(row.get("points") or 0)
        except (TypeError, ValueError):
            points = 0
        # Brand axis only: region-scoped rows never make a KPI trendable (kpi.py coverage endpoint semantics).
        region = str(row.get("region") or "")
        if not kpi_id or points <= 0 or region:
            continue
        brand = row.get("brand")
        scopes.setdefault(kpi_id, set()).add("" if brand is None else str(brand))
    trend = frozenset(scopes)
    per_brand_only = frozenset(k for k, brands in scopes.items() if "" not in brands)
    return trend, per_brand_only


async def build_capability_catalog(
    *,
    coverage_loader: Optional[CoverageLoader] = None,
    outcomes_loader: Optional[OutcomesLoader] = None,
) -> CapabilityCatalog:
    """Build the catalog. Never raises for a DB-backed field: it records it in ``degraded``."""
    degraded: List[str] = []

    rows: List[Dict[str, Any]] = []
    try:
        rows = list(await (coverage_loader or _default_coverage_loader)())
    except Exception as exc:  # noqa: BLE001 - degrade, never 502 the pills
        logger.warning("capability catalog: trend coverage unavailable: %s", exc)
    if not rows:
        logger.warning("capability catalog: trend coverage empty; marking degraded")
        degraded.append("trend_coverage")

    outcomes: List[str] = []
    try:
        outcomes = [str(o) for o in await (outcomes_loader or _default_outcomes_loader)() if o]
    except Exception as exc:  # noqa: BLE001
        logger.warning("capability catalog: causal outcomes unavailable: %s", exc)
    if not outcomes:
        logger.warning("capability catalog: causal outcomes empty; marking degraded")
        degraded.append("causal_outcomes")

    kpis = _kpi_entries()
    known = {e.id for e in kpis}
    trend, per_brand_only = _trend_sets(rows)
    # Offer only registry KPIs: an id that lingers in the coverage view after a
    # registry change must not reach the prompt as a bare id.
    trend &= known
    per_brand_only &= known
    return CapabilityCatalog(
        kpis=kpis,
        trend_kpi_ids=trend,
        per_brand_only_trend_ids=per_brand_only,
        axis_kpi_ids=frozenset(SEGMENTED_KPI_QUERY_FAMILIES),
        causal_outcomes=tuple(sorted(set(outcomes))),
        agent_roster=build_agent_roster_block(),
        degraded=tuple(degraded),
        loaded_at=time.monotonic(),
    )


# =============================================================================
# RENDERER
# =============================================================================

_WORKSTREAM_ORDER: Tuple[Tuple[Workstream, str], ...] = (
    (Workstream.WS3_BUSINESS, "Business impact"),
    (Workstream.WS2_TRIGGERS, "Trigger performance"),
    (Workstream.WS1_MODEL_PERFORMANCE, "Model performance"),
    (Workstream.WS1_DATA_QUALITY, "Data quality"),
    (Workstream.BRAND_SPECIFIC, "Brand-specific"),
    (Workstream.CAUSAL_METRICS, "Causal-effect metrics"),
)

# Hand-written RULES (not a list). The axis words are pinned to
# kpi_calculate_tool's parameter names by test_axis_vocabulary_matches_kpi_calculate_tool.
AXIS_PARAMETER_NAMES: Tuple[str, ...] = (
    "segment",
    "therapy_line",
    "region",
    "biologic",
    "ige_tier",
)
AXIS_RULES = (
    "Breakdown axes, AT MOST ONE per ask: segment = patient severity tier (low/medium/high); "
    "therapy_line = line of therapy (0-3); region = US census region (northeast/south/midwest/west); "
    "and - Remibrutinib ONLY - biologic status (naive/experienced) or ige_tier (low/medium/high). "
    'An optional time window ("last 3 months", "Q1 2025", "2025-01-01 to 2025-03-31") composes with '
    "segment/therapy_line but NOT with region/biologic/ige_tier for share, conversion or trigger KPIs. "
    "TRx share is share of the tracked 3-brand portfolio, NOT share versus competitors."
)

NEVER_BLOCK = (
    "NEVER PROPOSE (no tool serves these): named HCP or patient lists / rosters / exports; "
    'territory-level detail; competitor brands\' share or volume; TRx/NRx/NBRx "by HCP segment" '
    "(patient axes only); trends over time of SHAP values, CATE / treatment effects, predicted "
    "probabilities, gap sizes or optimizer allocations; recomputing, validating, re-deriving or "
    "EXTENDING an on-screen SHAP, optimizer, prediction, gap or CATE result (another segment, more "
    "features, per-territory detail, robustness, thresholds); two breakdown axes at once; causal "
    "drivers scoped to a region, month or segment; drivers OF a driver (unless it is itself a "
    "section-C outcome); thresholds, dose-response or nonlinearity questions; on-demand sensitivity / "
    'subgroup / "controlling for" analyses; live experiment status, lift or results; agent accuracy / '
    "error rates; audit-cycle metrics; data refresh schedules or pipeline latency; campaign-level ROI; "
    'toggling page UI (e.g. nowcast overlay); undefined ratios such as "conversion from NRx to NBRx"; '
    "emails, external data, CRM or any write action; treating a section-C outcome as a KPI (its rate, "
    "value, trend, chart or breakdown)."
)


def _names(catalog: CapabilityCatalog, ids: FrozenSet[str], *, mark_per_brand: bool = False) -> str:
    parts: List[str] = []
    for kpi_id in sorted(ids):
        name = catalog.kpi_name(kpi_id)
        if mark_per_brand and kpi_id in catalog.per_brand_only_trend_ids:
            name += " (per brand only)"
        parts.append(name)
    return ", ".join(parts)


def render_catalog_block(catalog: CapabilityCatalog) -> str:
    """Render the catalog as the prompt's A-H capability sections plus the NEVER list."""
    lines: List[str] = ["WHAT THE ASSISTANT CAN DO (every pill must map to exactly one of A-H):"]

    lines.append(
        "A. KPI values - the current value of any registry KPI, per brand, optionally over a time "
        "window. Registry KPIs by area:"
    )
    for workstream, label in _WORKSTREAM_ORDER:
        names = [
            e.name + (f" ({e.brand} only)" if e.brand else "")
            for e in catalog.kpis
            if e.workstream == workstream.value
        ]
        if names:
            lines.append(f"   - {label}: {'; '.join(names)}")
    lines.append("   " + AXIS_RULES)

    axis_names = _names(catalog, catalog.axis_kpi_ids)
    if "trend_coverage" in catalog.degraded or not catalog.trend_kpi_ids:
        trend_clause = (
            "a monthly trend line for the KPIs with a materialized history (the coverage list is "
            f"unavailable right now - the Rx-volume KPIs {axis_names} always have one; propose trends "
            "only for those)"
        )
    else:
        trend_clause = f"a monthly trend line for {_names(catalog, catalog.trend_kpi_ids, mark_per_brand=True)}"
    lines.append(
        f"B. Charts: {trend_clause}; ONE chart comparing severity tiers or lines of therapy for "
        f"{axis_names}; any other registry KPI as a current-value chart; several KPIs side by side."
    )

    if "causal_outcomes" in catalog.degraded:
        lines.append(
            "C. Causal drivers, causal paths and treatment effects from the causal-path registry, per "
            "brand, with confidence and refutation evidence, for the registry's patient-journey and "
            "commercial outcomes. The outcome list is unavailable right now: propose at most ONE "
            'causal-driver pill, phrased "what drives <the outcome or KPI named on screen> for '
            '<brand>?", and invent no outcome names.'
        )
    else:
        lines.append(
            "C. Causal drivers, causal paths and treatment effects from the causal-path registry, per "
            "brand, with confidence and refutation evidence, for these OUTCOMES only: "
            f"{', '.join(catalog.causal_outcomes)}. These outcomes are registry NODES, not KPIs: they "
            "cannot be computed, trended, charted or broken down by region, segment or month - a "
            'driver question is "what drives <outcome> for <brand>?", nothing finer. The registry has '
            "NO time, region or segment dimension."
        )

    lines.append(
        "D. Segments: KPI breakdowns by ONE of the axes in A; a ranking of HCP segments by predicted "
        "likelihood to prescribe a brand, by specialty OR by geographic region; aggregate HCP / "
        "patient cohort profiles (counts by specialty, tier, severity - never named individuals)."
    )
    lines.append(
        "E. Clinical and regulatory context per brand: FDA-label indications, mechanism of action, "
        "pivotal trial endpoints, real-world evidence, competitor landscape (as context, not as data)."
    )
    lines.append(
        "F. Platform: the agents below and what each does; an agent's recent activity; the system "
        "health score; experiment design, drift checks and gap/ROI opportunity analysis run through "
        "the orchestrator."
    )
    lines.extend("   " + line for line in catalog.agent_roster.splitlines())
    lines.append("G. Internal document / knowledge-base search.")
    lines.append(
        "H. Dashboard actions: navigate to a page, set the brand or region filter, set the date range."
    )
    lines.append("")
    lines.append(NEVER_BLOCK)
    return "\n".join(lines)


# =============================================================================
# ROUTE HINTS - used only when page_content is empty
# =============================================================================
# One sentence per app route (frontend/src/router/routes.tsx, auth routes
# excluded): what the page shows and which catalog letters fit it. A renamed
# route simply falls back to today's behaviour (path + brand only).

ROUTE_HINTS: Dict[str, str] = {
    "/": (
        "Home dashboard: KPI tiles (TRx, market share, HCP reach), active campaigns, model accuracy, "
        "system health and the top gap opportunity; pills should ask for KPI values or trends (A/B), "
        "drivers of the gap's KPI (C) or platform health (F)."
    ),
    "/documentation": (
        "How E2I Works: explains the platform, its agents and analyses; pills should ask what the "
        "assistant can analyse, which agents exist (F) or where a KPI is defined (A)."
    ),
    "/ai-insights": (
        "Executive Insights: brand-level narrative of KPI movements and causal drivers; pills should "
        "ask for KPI values and trends (A/B) and causal drivers (C)."
    ),
    "/knowledge-graph": (
        "Knowledge Graph: causal paths between drivers and outcomes; pills should ask for the drivers "
        "of a registry outcome (C) or the KPIs those outcomes relate to (A)."
    ),
    "/causal-analysis": (
        "Causal Analysis: treatment-effect estimates driver -> outcome with confidence and refutation; "
        "pills should ask for drivers or paths of an outcome (C), never a trend of an effect."
    ),
    "/causal-discovery": (
        "Causal Discovery: discovered causal graphs over the patient-journey data; pills should ask for "
        "drivers of a registry outcome (C) or the related KPIs (A)."
    ),
    "/segment-analysis": (
        "Segment Analysis: KPI and effect differences across patient axes (severity tier, line of "
        "therapy, biologic/IgE for Remibrutinib); pills should ask for KPI breakdowns by ONE axis (A/D)."
    ),
    "/expert-reviews": (
        "Expert Reviews: human review queue for agent outputs; pills should ask about agents and "
        "platform status (F) or the KPIs under review (A)."
    ),
    "/predictive-analytics": (
        "Predictive Analytics: scored cohorts and predicted probabilities from the ML models; pills "
        "should ask for the HCP segment likelihood ranking (D), model-performance KPIs (A) or clinical "
        "context (E), never per-patient predictions."
    ),
    "/model-performance": (
        "Model Performance: ROC-AUC, PR-AUC, F1, calibration and PSI drift KPIs; pills should ask for "
        "those KPI values or charts (A/B)."
    ),
    "/feature-importance": (
        "Feature Importance: SHAP feature rankings for the brand models; pills should turn a feature "
        "into a catalog ask - HCP segment likelihood (D), a KPI breakdown (A) or clinical context (E) - "
        "never SHAP recomputation."
    ),
    "/time-series": (
        "Time Series: monthly KPI history with nowcast; pills should ask for trend charts and period "
        "comparisons of ONE KPI (B) or KPI values over a window (A)."
    ),
    "/intervention-impact": (
        "Intervention Impact: measured effects of interventions on outcomes; pills should ask for "
        "drivers or treatment effects of a registry outcome (C) and the affected KPIs (A/B)."
    ),
    "/digital-twin": (
        "Digital Twin: simulated intervention scenarios; pills should ask for the causal drivers behind "
        "a scenario's outcome (C) or the baseline KPI values (A)."
    ),
    "/gap-analysis": (
        "Gap Analysis: KPI gaps versus target by segment with expected ROI; pills should ask for the "
        "underlying KPI value or breakdown (A), its trend (B) or its drivers (C), never a trend of the "
        "gap itself."
    ),
    "/resource-optimization": (
        "Resource Optimization: recommended field-force allocation by territory; pills should ask for "
        "regional KPI breakdowns (A), ROI (A/B) or causal drivers (C), never territory-level detail."
    ),
    "/experiments": (
        "Experiments: health, enrollment and interim analyses of running A/B tests; pills should ask "
        "for experiment design through the orchestrator (F) or the KPIs an experiment targets (A), "
        "never live lift, enrollment or results."
    ),
    "/kpi-dictionary": (
        "KPI Dictionary: registry definitions of every KPI; pills should ask for the value, definition, "
        "chart or drivers of specific KPIs (A/B/C)."
    ),
    "/data-quality": (
        "Data Quality: source coverage, match rate and freshness KPIs; pills should ask for those KPI "
        "values or charts (A/B)."
    ),
    "/system-health": (
        "System Health: platform health score and component status; pills should ask about the health "
        "score and the agents (F)."
    ),
    "/monitoring": (
        "Monitoring: drift and model-monitoring KPIs; pills should ask for PSI/drift and "
        "model-performance KPI values or charts (A/B) or a drift check via the orchestrator (F)."
    ),
    "/analytics": (
        "Agent Analytics: query volume, latency and success counts per agent; pills should ask which "
        "agents exist, what they do and what they did recently (F), never agent accuracy, latency or "
        "error rates."
    ),
    "/agent-orchestration": (
        "Agent Orchestration: the agent tiers and their recent activity; pills should ask which agents "
        "exist, what they do and what they did recently (F)."
    ),
    "/memory-architecture": (
        "Memory Architecture: how the assistant's memory tiers work; pills should ask about the "
        "platform and agents (F) or search internal documents (G)."
    ),
    "/audit-chain": (
        "Audit Chain: provenance of agent decisions; pills should ask about agents and their activity "
        "(F), never audit-cycle metrics."
    ),
    "/feedback-learning": (
        "Feedback Learning: how user feedback improves the agents; pills should ask about agents (F) "
        "or internal documents (G)."
    ),
    "/admin": (
        "Administration: users and settings; pills should stay on platform status and agents (F)."
    ),
}


def route_hint(page: Optional[str]) -> str:
    """Hint for ``page`` ('' when unknown). Tolerates a query string and a trailing slash."""
    if not page:
        return ""
    path = page.split("?", 1)[0].split("#", 1)[0].rstrip("/").lower() or "/"
    return ROUTE_HINTS.get(path, "")


# =============================================================================
# VALIDATOR - narrow, deterministic, tuned to the pill families graded NO
# =============================================================================


class SuggestionLike(Protocol):
    @property
    def title(self) -> str: ...

    @property
    def message(self) -> str: ...


P = TypeVar("P", bound=SuggestionLike)

# A time-boxed journey flag (persistent_180d) is never a KPI.
_DURATION_OUTCOME_RE = re.compile(r"_\d+d$", re.I)

# "the <outcome> rate / trend / by region ..." - the outcome used as a metric.
_VALUE_ASK_RE = re.compile(
    r"\b(?:rates?|values?|levels?|trends?|chart|plot|graph|over time|monthly|month-over-month|"
    r"quarterly|weekly|by (?:census )?region|by (?:severity )?tier|by segment|by line|breakdown|"
    r"distribution|percentage|how many|count|volume)\b",
    re.I,
)
# ... unless the pill is a causal question, which section C serves.
_CAUSAL_ASK_RE = re.compile(
    r"\b(?:driv\w*|caus\w*|paths?|effects?|why|influenc\w*|factors?|impacts?|refut\w*|confiden\w*)\b",
    re.I,
)

# Registry KPI NAMES the catalog prompt offers as answerable must never be
# dropped by an off-platform rule, even though their names share vocabulary
# with the rule's trigger words. Exemptions carved out below:
#   - Geographic Consistency Gap (WS1-DQ-006) vs. gap_recompute
#   - SHAP Coverage (WS1-MP-007) vs. shap_or_feature_importance
#   - Conditional ATE (CATE) (CM-002) vs. uplift_by_segment
_OFF_PLATFORM_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "shap_or_feature_importance",
        re.compile(
            r"\bSHAP\b(?! coverage)|\bfeature[- ]importances?\b|\bfeature rankings?\b|"
            r"\btop(?:-| )?\d* ?features\b",
            re.I,
        ),
    ),
    ("territory_detail", re.compile(r"\bterritor(?:y|ies)\b|\bT-\d{3}\b", re.I)),
    (
        "individual_prediction",
        re.compile(
            r"\bpredicted (?:\d+-day )?(?:[a-z_]+ )?probabilit(?:y|ies)\b|\bmean predicted probability\b|"
            r"\bpropensity scores?\b|"
            r"\b(?:each|individual|specific) (?:HCP|patient|prescriber)s?\b"
            r"(?!\s+(?:segment|specialt|tier|group|cohort))|"
            r"\b(?:HCP|patient) (?:list|roster)s?\b",
            re.I,
        ),
    ),
    (
        "gap_recompute",
        re.compile(
            r"(?<!consistency )\bgap\b[^.?]{0,160}\b(?:trend|evolv\w*|evolution|over the (?:past|last)|"
            r"chart|plot|month)\b|"
            r"\b(?:chart|plot|trend of)\b[^.?]{0,160}(?<!consistency )\bgap\b",
            re.I,
        ),
    ),
    (
        "uplift_by_segment",
        re.compile(
            r"\bCATE\b(?<!\(CATE)|\bheterogen\w*\b|\btreatment effects? (?:by|across|for) (?:patient )?"
            r"(?:segment|tier|subgroup|cohort)s?\b|\bsubgroup analys\w*|\bsensitivity analys\w*|"
            r"\bcontrolling for\b",
            re.I,
        ),
    ),
    (
        "off_platform_action",
        re.compile(
            r"\be-?mails?\b|\bexport(?:s|ed|ing)?\b|\bCRM\b|\bVeeva\b|"
            r"\bsend (?:a |an )?(?:report|message|alert|email)\b",
            re.I,
        ),
    ),
    (
        "competitor_data",
        re.compile(
            r"\bcompetitors?'?s? (?:market )?(?:share|volume|TRx|NRx|sales)\b|"
            r"\b(?:vs\.?|versus|against) (?:the )?competitors?\b",
            re.I,
        ),
    ),
)


def journey_outcomes(catalog: CapabilityCatalog) -> Tuple[str, ...]:
    """Outcomes with no KPI counterpart - the ones a pill can mistake for a metric.

    Time-boxed journey flags (``persistent_180d``) are never KPIs; anything else
    the KPI recognizer cannot resolve (``adopted``) is treated the same. Outcomes
    the recognizer reads as a KPI mention (``roi``, ``trx_volume``, and also
    ``treatment_initiated`` -> the causal-metric KPI) are left to the prompt.
    """
    from src.services.kpi_resolution import recognize_kpi

    out: List[str] = []
    for outcome in catalog.causal_outcomes:
        if _DURATION_OUTCOME_RE.search(outcome) or recognize_kpi(outcome.replace("_", " ")) is None:
            out.append(outcome)
    return tuple(out)


def match_unsupported_rule(text: str, journey: Sequence[str]) -> Optional[str]:
    """Name of the rule ``text`` violates, or None when the pill is supported."""
    for name, pattern in _OFF_PLATFORM_RULES:
        if pattern.search(text):
            return name
    lowered = text.lower()
    for outcome in journey:
        needle = outcome.lower()
        spaced = needle.replace("_", " ")
        if (
            re.search(rf"\b{re.escape(needle)}\b", lowered) is None
            and re.search(rf"\b{re.escape(spaced)}\b", lowered) is None
        ):
            continue
        if _VALUE_ASK_RE.search(lowered) and not _CAUSAL_ASK_RE.search(lowered):
            return f"outcome_as_kpi:{outcome}"
    return None


def filter_unsupported_pills(
    pills: Sequence[P], catalog: CapabilityCatalog
) -> Tuple[List[P], List[Tuple[P, str]]]:
    """Split ``pills`` into (kept, [(dropped, rule), ...]) preserving order."""
    journey = journey_outcomes(catalog)
    kept: List[P] = []
    dropped: List[Tuple[P, str]] = []
    for pill in pills:
        rule = match_unsupported_rule(f"{pill.title} {pill.message}", journey)
        if rule is None:
            kept.append(pill)
        else:
            dropped.append((pill, rule))
    return kept, dropped
