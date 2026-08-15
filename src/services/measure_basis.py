"""Measure-basis SSOT: what a KPI figure MEASURES, and what it may be compared with (#1640).

Deliberately cheap to import. Every surface that emits a KPI figure needs this
-- the chat tools, the orchestrator's ``kpi_lookup`` payload, the Home KPI
summary tiles -- and none of them can afford to import
``src.api.routes.chatbot_tools`` to get it (~30s: orchestrator/tool_composer/RAG
stacks). Same precedent as #1475, which moved ``KPI_SEMANTIC_NOTES`` here for
exactly this reason. This module imports nothing heavier than the KPI registry,
and resolves KPI names function-locally.

The rule: two figures are comparable only if their substrate declarations are
EQUAL, and an undeclared substrate is never comparable with anything.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: What ``business_metrics.value`` actually is (#1640).
#:
#: Measured against the live DB 2026-08-15: the national ``business_metrics``
#: TRx total for 2026-08 is 825,242, against 11,298 trailing-30-day
#: ``treatment_events`` prescription events for the same brand — **73.0x**, and
#: stable month over month (2026-07: 830,103; 2026-06: 812,266). Not a window,
#: grain or bucket-width artifact: ``BusinessMetricsGenerator`` draws Kisqali
#: TRx from a base of 50,000 per region per month with ``REGION_FACTORS``
#: northeast 1.15 and 2% monthly trend, so national 50,000 x 4.00 x 4.26 =
#: 852,000 against the measured 825,242. An event count cannot be fractional;
#: these values are (min 5,923.95, max 307,229.18).
#:
#: So the two numbers are different QUANTITIES sharing the name "TRx", and no
#: window or grain conversion maps one onto the other.
BUSINESS_METRICS_BASIS: Dict[str, Any] = {
    "substrate": ["business_metrics"],
    "computed": False,
    "grain": "brand x region x calendar month",
    "measure": "modeled market-scale monthly level",
    "note": (
        "business_metrics.value is a MODELED market-scale level, not a count of "
        "observed events. Do NOT compare it with, sum it against, or divide it by a "
        "figure computed from treatment_events (which is what kpi_calculate_tool "
        "returns for volume KPIs): measured 2026-08-15, the national business_metrics "
        "TRx total is ~73x the trailing-30-day treatment_events prescription count for "
        "the same brand. If both appear in one answer, say plainly that they measure "
        "different things and never present one as a check on the other."
    ),
}


def measure_basis_for_kpi(kpi: Any) -> Dict[str, Any]:
    """Declare what a computed KPI figure measures, DERIVED from the registry.

    ``KPIMetadata.tables`` is the existing SSOT for a KPI's substrate (measured:
    all 45 registry KPIs populate it), so this needs no hand-maintained map —
    and deriving gets the one genuine exception right for free. WS3-BI-010
    (ROI) really does read ``business_metrics``, so it IS comparable with
    ``e2i_data_query_tool``'s stored ROI, where a blanket "the KPI tool means
    treatment_events" rule would wrongly fence it off.
    """
    tables = sorted(getattr(kpi, "tables", None) or [])
    return {
        "substrate": tables,
        "computed": True,
        "measure": (
            f"computed on demand from {', '.join(tables)}"
            if tables
            else "computed on demand; substrate not declared"
        ),
        "note": (
            "Computed from the operational substrate at query time — NOT read from the "
            "business_metrics snapshot table. Only compare with another figure whose "
            "substrate matches; e2i_data_query_tool(query_type='kpi') returns "
            "business_metrics rows, which for volume KPIs measure something different "
            "(see its measure_basis)."
        ),
    }


def bases_are_comparable(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> bool:
    """Two figures are comparable only when they rest on the same substrate.

    Deliberately fails CLOSED on a missing declaration: an undeclared basis is
    not evidence of agreement.
    """
    if not left or not right:
        return False
    left_tables = set(left.get("substrate") or [])
    right_tables = set(right.get("substrate") or [])
    if not left_tables or not right_tables:
        # An UNDECLARED substrate is not evidence of agreement, and two
        # undeclared ones are not evidence of agreement with each other --
        # `sorted([]) == sorted([])` would have said they matched.
        return False
    # EQUALITY, not intersection. Intersection certified `conversion_rate`
    # (['triggers', 'treatment_events']) comparable with TRx
    # (['treatment_events']) on the shared leg, but a ratio and a count are not
    # comparable just because one SQL leg overlaps. It also certified ROI
    # against stored business_metrics rows, when ROI's declaration is a UNION of
    # possible sources -- its calculator can fall back to agent_activities -- so
    # the declaration cannot say which leg actually ran. Equality fails closed
    # in both cases.
    return left_tables == right_tables


def cross_substrate_conflict(kpi_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """State, in code, that a stored figure is not the computed one (#1640).

    Called on every ``e2i_data_query_tool(query_type='kpi')`` return. If the
    asked-for metric is one the registry can COMPUTE, and that computation rests
    on a different substrate from the stored rows being returned, the payload
    says so here rather than leaving it to the answer layer to notice.

    This is the fence's only load-bearing caller of :func:`bases_are_comparable`
    -- the first version of this change shipped that helper with no caller at
    all, which labelled the problem without fencing anything.
    """
    if not kpi_name:
        return None
    # Function-local: this module defers its kpi_resolution import to the bottom
    # (noqa: E402) and _query_kpis is defined above that point.
    from src.services.kpi_resolution import recognize_kpi

    kpi = recognize_kpi(kpi_name)
    if kpi is None:
        return None
    computed = measure_basis_for_kpi(kpi)
    if bases_are_comparable(computed, BUSINESS_METRICS_BASIS):
        return None
    return {
        "this_tool": "e2i_data_query_tool",
        "this_substrate": list(BUSINESS_METRICS_BASIS["substrate"]),
        "other_tool": "kpi_calculate_tool",
        "other_substrate": list(computed["substrate"]),
        "kpi_id": kpi.id,
        "note": (
            f"The rows above are stored business_metrics values. {kpi.name} can also be "
            f"COMPUTED by kpi_calculate_tool from {', '.join(computed['substrate']) or 'a different substrate'}, "
            "and the two are NOT comparable: they measure different things and differ by a "
            "large, roughly constant factor (measured ~73x for TRx). If both appear in one "
            "answer, label each with its source and never present one as a check, total, "
            "correction or share-of for the other."
        ),
    }
