"""Unit tests for src.services.chat_capability_catalog.

Everything DB-backed is injected through the two loader callables, so these
tests run without Supabase. The KPI registry (YAML) and the agent roster
(factory config) are real: they are code, and the point of the catalog is
that its lists come from code.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest  # noqa: F401

from src.kpi.segmented_history import SEGMENTED_KPI_QUERY_FAMILIES
from src.services import chat_capability_catalog as cat

# =============================================================================
# FIXTURE DATA
# =============================================================================

COVERAGE_ROWS: List[Dict[str, Any]] = [
    {"kpi_id": "WS3-BI-005", "brand": "", "region": "", "points": 24},
    {"kpi_id": "WS3-BI-005", "brand": "Kisqali", "region": "", "points": 24},
    # NBRx: per-brand scopes only, no '' row -> per_brand_only
    {"kpi_id": "WS3-BI-007", "brand": "Kisqali", "region": "", "points": 24},
    {"kpi_id": "WS3-BI-007", "brand": "Fabhalta", "region": "", "points": 24},
    # zero points is not a trend
    {"kpi_id": "WS3-BI-010", "brand": "", "region": "", "points": 0},
    # region-scoped row does not make WS3-BI-010 trendable
    {"kpi_id": "WS3-BI-010", "brand": "", "region": "west", "points": 24},
    # junk row is skipped
    {"kpi_id": "", "brand": None, "points": "x"},
]

OUTCOMES: List[str] = [
    "treatment_initiated",
    "persistent_180d",
    "trx_volume",
    "nrx_volume",
    "discontinued_180d",
    "roi",
    "adopted",
    "trx_market_share",
    "nbrx_volume",
    "intent_to_prescribe",
    "adherent_180d",
    "action_taken",
    "low_gap_180d",
    "conversion_flag",
]


async def _coverage() -> List[Dict[str, Any]]:
    return list(COVERAGE_ROWS)


async def _outcomes() -> List[str]:
    return list(OUTCOMES)


async def _boom() -> Any:
    raise RuntimeError("db down")


async def _empty() -> list:
    return []


async def make_catalog(coverage=_coverage, outcomes=_outcomes) -> cat.CapabilityCatalog:
    return await cat.build_capability_catalog(coverage_loader=coverage, outcomes_loader=outcomes)


# =============================================================================
# BUILDER
# =============================================================================


async def test_kpis_come_from_the_registry():
    c = await make_catalog()
    ids = {k.id for k in c.kpis}
    assert "WS3-BI-005" in ids
    assert len(ids) >= 40
    assert c.kpi_name("WS3-BI-005") == "Total Prescriptions (TRx)"
    # unknown ids fall back to the id itself (never KeyError in a prompt)
    assert c.kpi_name("NOPE-1") == "NOPE-1"


async def test_trend_sets_from_coverage_rows():
    c = await make_catalog()
    assert c.trend_kpi_ids == frozenset({"WS3-BI-005", "WS3-BI-007"})
    assert c.per_brand_only_trend_ids == frozenset({"WS3-BI-007"})
    assert "WS3-BI-010" not in c.trend_kpi_ids


async def test_axis_kpis_from_segmented_history_families():
    c = await make_catalog()
    assert c.axis_kpi_ids == frozenset(SEGMENTED_KPI_QUERY_FAMILIES)


async def test_outcomes_sorted_deduped_and_roster_present():
    async def dup() -> List[str]:
        return ["roi", "roi", "adopted", ""]

    c = await make_catalog(outcomes=dup)
    assert c.causal_outcomes == ("adopted", "roi")
    assert "The E2I system has" in c.agent_roster
    assert c.degraded == ()


async def test_loader_failure_marks_degraded_and_does_not_raise():
    c = await make_catalog(coverage=_boom, outcomes=_boom)
    assert set(c.degraded) == {"trend_coverage", "causal_outcomes"}
    assert c.trend_kpi_ids == frozenset()
    assert c.causal_outcomes == ()
    # code-derived fields survive a DB outage
    assert len(c.kpis) >= 40
    assert c.axis_kpi_ids == frozenset(SEGMENTED_KPI_QUERY_FAMILIES)


async def test_empty_results_are_degraded_too(caplog):
    # KPIHistoryRepository.get_coverage returns [] on error AND when it has no
    # client; an empty coverage view is not a realistic prod state.
    with caplog.at_level("WARNING", logger="src.services.chat_capability_catalog"):
        c = await make_catalog(coverage=_empty, outcomes=_empty)
    assert set(c.degraded) == {"trend_coverage", "causal_outcomes"}
    messages = [r.getMessage() for r in caplog.records]
    assert any("trend coverage empty" in m for m in messages)
    assert any("causal outcomes empty" in m for m in messages)
