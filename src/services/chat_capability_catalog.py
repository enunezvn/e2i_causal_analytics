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

import asyncio
import dataclasses
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

CATALOG_TTL_SECONDS = 600.0
# A degraded catalog (a DB-backed field failed) retries sooner, but not on
# every pill request - a down database must not be hammered.
DEGRADED_TTL_SECONDS = 60.0
# Each DB loader gets its own budget so a stalled connection cannot hold the
# refreshing request for the client's full connect+read timeouts (10 s + 30 s
# per loader). Measured 2026-09-06 on the droplet: 30-70 ms per query plus
# 130 ms client creation; one unreproduced 18 s cold read. A timeout is an
# ordinary exception below: the field is marked degraded, the last-good lists
# carry forward and the refresh is retried after DEGRADED_TTL_SECONDS. The two
# loaders stay sequential on purpose: one in-flight query at a time against a
# possibly stalled database, so a refresh costs at most twice this budget
# (#1901 item 1).
CATALOG_LOADER_TIMEOUT_SECONDS = 5.0


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
        rows = list(
            await asyncio.wait_for(
                (coverage_loader or _default_coverage_loader)(), CATALOG_LOADER_TIMEOUT_SECONDS
            )
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never 502 the pills
        logger.warning(
            "capability catalog: trend coverage unavailable: %s: %s", type(exc).__name__, exc
        )
    if not rows:
        logger.warning("capability catalog: trend coverage empty; marking degraded")
        degraded.append("trend_coverage")

    outcomes: List[str] = []
    try:
        outcomes = [
            str(o)
            for o in await asyncio.wait_for(
                (outcomes_loader or _default_outcomes_loader)(), CATALOG_LOADER_TIMEOUT_SECONDS
            )
            if o
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "capability catalog: causal outcomes unavailable: %s: %s", type(exc).__name__, exc
        )
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
    "territory-level detail; per-HCP or per-patient predictions; model retraining; competitor brands' "
    'share or volume; TRx/NRx/NBRx "by HCP segment" '
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
    # fallbacks key on EMPTY data: a degraded refresh that carried last-good lists
    # forward still renders them
    if not catalog.trend_kpi_ids:
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

    if not catalog.causal_outcomes:
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

# Competitor SHARE / VOLUME / performance DATA is NEVER (the catalog's TRx share is
# portfolio share); the competitor landscape as clinical context (section E) is
# served, so "versus competitors" needs one of these words in the same clause.
_COMPETITOR_DATA_WORDS = (
    r"(?:share|volume|TRx|NRx|NBRx|sales|revenue|uptake|growth|prescriptions?|scripts?|"
    r"adoption|persistence|adherence|rates?|perform\w*|outperform\w*|benchmark\w*|"
    r"(?:patient |treatment |new |brand )?starts?|initiations?|switch(?:es|ing)?|units|demand)"
)
_COMPETITOR_NOUN = r"(?:the )?competit(?:ors?|ion)"

# Registry KPI NAMES the catalog prompt offers as answerable must never be
# dropped by an off-platform rule, even though their names share vocabulary
# with the rule's trigger words. Exemptions carved out below:
#   - Geographic Consistency Gap (WS1-DQ-006) vs. gap_recompute
#   - SHAP Coverage (WS1-MP-007) vs. shap_or_feature_importance
#   - Conditional ATE (CATE) (CM-002) vs. uplift_by_segment - the exemption
#     covers only the BARE value ask; "(CATE)" followed by a segment axis or
#     a time form is re-caught by the trailing alternatives below
_OFF_PLATFORM_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "shap_or_feature_importance",
        re.compile(
            r"\bSHAP\b(?! coverage)|\bfeature[- ]importances?\b|\bfeature rankings?\b|"
            r"\btop(?:-| )?\d* ?features\b(?!\s+(?:of|in|on) (?:the |this |our )?"
            r"(?:(?:E2I|e2i)\b(?!'s|\s+model)|(?:E2I |e2i )?(?:platform|dashboard|app|application|tool|product|site)\b))",
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
            r"\bcontrolling for\b"
            r"|\(CATE\)[^.?]{0,120}\b(?:by|across|for|per|between) (?:patient |each )?"
            r"(?:segment|tier|subgroup|cohort|severity|biologic|IgE)\w*"
            r"|\(CATE\)[^.?]{0,120}\b(?:trends?|over time|over the (?:past|last)|monthly|month-over-month)\b"
            # Conditional ATE (CATE) is a registry KPI and section B advertises a
            # current-value chart for any registry KPI, so a bare "Chart the
            # Conditional ATE (CATE)" is answerable; TREND forms are NEVER -
            # "(CATE) ... <time word>" (the alternative above), "trend of ...
            # (CATE)", and a month/week/quarter word before the KPI.
            r"|\btrend of\b[^.?]{0,120}\(CATE\)"
            r"|\b(?:monthly|weekly|quarterly)\b[^.?]{0,120}\(CATE\)",
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
            rf"\bcompetit(?:ors?|ion)'?s? (?:market )?(?:share|volume|TRx|NRx|NBRx|sales)\b"
            rf"|\b{_COMPETITOR_DATA_WORDS}\b[^.?]{{0,80}}\b(?:vs\.?|versus|against|with) {_COMPETITOR_NOUN}\b"
            rf"|\b(?:vs\.?|versus|against|with) {_COMPETITOR_NOUN}\b[^.?]{{0,80}}\b{_COMPETITOR_DATA_WORDS}\b",
            re.I,
        ),
    ),
)


# Part C publishes the page summary to the assistant as a readable, so a pill
# MAY read, rank or compare SHAP, CATE, gap, prediction or optimizer territory
# values that are literally on screen (the pill prompt says so; the
# /resource-optimization summary publishes the allocation count, projected
# ROI and outcome, and the largest increase and decrease, not the territory
# table itself). The artefact rules listed below therefore yield when the
# question names the on-screen artefact AND asks for nothing that would
# extend it (another axis, a trend, a recomputation). The extends-list
# mirrors the pill prompt's own forbidden verbs (recompute, validate, extend,
# explain WHY) and the artefact rules' own trend/axis vocabulary, so the
# exemption cannot keep an ask that extends the artefact; it does not prove
# the summary carries the row the pill names.
_ON_SCREEN_ARTEFACT_RULES = frozenset(
    {
        "shap_or_feature_importance",
        "gap_recompute",
        "uplift_by_segment",
        "individual_prediction",
        "territory_detail",
    }
)
_ON_SCREEN_RE = re.compile(
    r"\bon[- ]screen\b|\bon the (?:page|screen)\b|\b(?:shown|displayed|visible)\b", re.I
)
_EXTENDS_ON_SCREEN_RE = re.compile(
    r"\bre-?comput\w*|\bre-?calculat\w*|\bre-?run\b|\bvalidat\w*|\bextend\w*|\banother\b|"
    r"\bmore features\b|\bwhy\b|\breasons?\b|\bbecause\b|\bdrivers? behind\b|\bwhat drives\b|"
    r"\bby (?:census |HCP )?(?:region|territory|segment|tier|specialty|severity|biologic|IgE|cohort|subgroup)\w*|"
    r"\bper[- ]territory\b|\btrends?\b|\bover time\b|\bover the (?:past|last)\b|\bmonth\w*|"
    r"\bsince\b|\bchang\w*|\bthreshold\w*|\brobust\w*|\bsensitivit\w*",
    re.I,
)

# Section D advertises segment-level likelihood (predict_hcp_segment_likelihood_tool:
# mean propensity by specialty or geographic region), so an AGGREGATE likelihood ask
# is answerable; individual_prediction yields to it unless an individual marker is
# present.
_AGGREGATE_LIKELIHOOD_RE = re.compile(
    r"\b(?:by|per|across|for each|which|top) "
    r"(?:(?:HCP )?specialt(?:y|ies)|(?:geographic |census )?regions?|(?:HCP )?segments?)\b",
    re.I,
)
_INDIVIDUAL_ASK_RE = re.compile(
    r"\b(?:each|individual|specific|this|that|single) (?:HCP|patient|prescriber|doctor)s?\b|"
    r"\b(?:HCP|patient)[- ]?\d+\b|\bpropensity scores?\b|\b(?:HCP|patient) (?:list|roster)s?\b|"
    # ranked or listed HCPs with an axis word are still an HCP list, not a segment aggregate
    r"\btop \d* ?(?:HCP|prescriber|physician|doctor|patient)s\b|"
    r"\b(?:which|list|name|rank) (?:the )?(?:HCP|prescriber|physician|doctor|patient)s\b|"
    r"\b(?:HCP|prescriber)s (?:shown|with)\b",
    re.I,
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
    """Name of the rule ``text`` violates, or None when the pill is supported.

    On-screen READ questions (Part C) bypass the artefact rules named in
    ``_ON_SCREEN_ARTEFACT_RULES`` unless they also ask to extend the
    artefact. Aggregate HCP-segment likelihood asks (by specialty or
    region, section D) bypass individual_prediction unless an individual
    HCP or patient is named.
    """
    on_screen_read = bool(_ON_SCREEN_RE.search(text)) and not _EXTENDS_ON_SCREEN_RE.search(text)
    for name, pattern in _OFF_PLATFORM_RULES:
        if pattern.search(text):
            if on_screen_read and name in _ON_SCREEN_ARTEFACT_RULES:
                continue
            if (
                name == "individual_prediction"
                and _AGGREGATE_LIKELIHOOD_RE.search(text)
                and not _INDIVIDUAL_ASK_RE.search(text)
            ):
                continue
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


# =============================================================================
# CACHE - lazy, in-process, TTL; no startup hook (CI runs TestClient lifespans
# on a 30 s thread timeout, and a lazy cache adds no work there)
# =============================================================================


def _keep_last_good_fields(
    fresh: CapabilityCatalog, previous: Optional[CapabilityCatalog]
) -> CapabilityCatalog:
    """On a degraded refresh, carry the previous catalog's good DB-backed fields forward.

    ``degraded`` still names the fields whose refresh failed, so the cache
    retries in ``DEGRADED_TTL_SECONDS`` and the outage stays visible; the
    renderer shows the carried-forward lists because it keys on empty data,
    not on the marker.
    """
    if previous is None or not fresh.degraded:
        return fresh
    updates: Dict[str, Any] = {}
    if "trend_coverage" in fresh.degraded and "trend_coverage" not in previous.degraded:
        updates["trend_kpi_ids"] = previous.trend_kpi_ids
        updates["per_brand_only_trend_ids"] = previous.per_brand_only_trend_ids
    if "causal_outcomes" in fresh.degraded and "causal_outcomes" not in previous.degraded:
        updates["causal_outcomes"] = previous.causal_outcomes
    if not updates:
        return fresh
    return dataclasses.replace(fresh, **updates)


class _CatalogCache:
    """Process-wide catalog with a TTL and a single-flight refresh.

    One build runs at a time: concurrent cold (or expired) callers await the
    build already in flight instead of each hitting the DB, and a slow degraded
    build can no longer overwrite a faster good one. The future is created on
    the running loop inside ``get()`` - never at import time, where an asyncio
    primitive would bind to whichever loop exists first. A build that
    ``reset()`` orphaned mid-flight still serves its own waiters but neither
    writes the cache nor clears a newer build's future; the guard is a
    per-build token so an eager task factory (build finished inside
    ``ensure_future``) publishes too.
    """

    def __init__(self) -> None:
        self._catalog: Optional[CapabilityCatalog] = None
        self._inflight: Optional["asyncio.Future[CapabilityCatalog]"] = None
        # The build allowed to publish, as a token rather than the task object:
        # under an eager task factory the build can finish inside
        # ensure_future(), before ``_inflight`` is even assigned, and a
        # task-identity check would then never publish nor clear (#1901 item 5).
        self._generation: Optional[object] = None

    async def get(
        self,
        *,
        now: Optional[float] = None,
        coverage_loader: Optional[CoverageLoader] = None,
        outcomes_loader: Optional[OutcomesLoader] = None,
    ) -> CapabilityCatalog:
        current = time.monotonic() if now is None else now
        cached = self._catalog
        if cached is not None:
            ttl = DEGRADED_TTL_SECONDS if cached.degraded else CATALOG_TTL_SECONDS
            if current - cached.loaded_at < ttl:
                return cached
        if self._inflight is None:
            token = object()
            self._generation = token
            future = asyncio.ensure_future(
                self._refresh(token, cached, now, coverage_loader, outcomes_loader)
            )
            # An eager build that never suspended has already finished, clearing
            # itself and publishing if it succeeded; storing its finished future
            # would pin it forever.
            if not future.done():
                self._inflight = future
            return await asyncio.shield(future)
        return await asyncio.shield(self._inflight)

    async def _refresh(
        self,
        token: object,
        previous: Optional[CapabilityCatalog],
        now: Optional[float],
        coverage_loader: Optional[CoverageLoader],
        outcomes_loader: Optional[OutcomesLoader],
    ) -> CapabilityCatalog:
        try:
            fresh = await build_capability_catalog(
                coverage_loader=coverage_loader, outcomes_loader=outcomes_loader
            )
            fresh = _keep_last_good_fields(fresh, previous)
            if now is not None:
                fresh = dataclasses.replace(fresh, loaded_at=now)
            if self._generation is token:
                self._catalog = fresh
            return fresh
        finally:
            if self._generation is token:
                self._inflight = None
                self._generation = None

    def reset(self) -> None:
        # _inflight and _generation are written and cleared together; clearing one
        # without the other strands the cache (no build may ever clear the slot).
        self._catalog = None
        self._inflight = None
        self._generation = None


_cache = _CatalogCache()


async def get_capability_catalog() -> CapabilityCatalog:
    """The process-wide cached catalog (built lazily on first use)."""
    return await _cache.get()


def reset_capability_catalog_cache() -> None:
    """Test hook: forget the cached catalog."""
    _cache.reset()
