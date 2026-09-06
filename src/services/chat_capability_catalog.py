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
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

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
