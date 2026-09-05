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

    trend, per_brand_only = _trend_sets(rows)
    return CapabilityCatalog(
        kpis=_kpi_entries(),
        trend_kpi_ids=trend,
        per_brand_only_trend_ids=per_brand_only,
        axis_kpi_ids=frozenset(SEGMENTED_KPI_QUERY_FAMILIES),
        causal_outcomes=tuple(sorted(set(outcomes))),
        agent_roster=build_agent_roster_block(),
        degraded=tuple(degraded),
        loaded_at=time.monotonic(),
    )
