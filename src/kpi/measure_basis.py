"""Measure-basis SSOT: what a KPI figure MEASURES, and what it may be compared with (#1640).

Deliberately cheap to import. Every surface that emits a KPI figure needs this
-- the chat tools, the orchestrator's ``kpi_lookup`` payload, the Home KPI
summary tiles -- and none of them can afford to import
``src.api.routes.chatbot_tools`` to get it (~30s: orchestrator/tool_composer/RAG
stacks). Same spirit as #1475, which moved ``KPI_SEMANTIC_NOTES`` out of that
module for exactly this reason.

It lives under ``src.kpi`` rather than ``src.services`` because
``src/services/__init__.py`` eagerly imports ``alert_routing`` and friends, so
anything under that package drags in ``aiohttp`` (measured: 0.54s / 394 modules
vs 0.43s / 344 without). An orchestrator or chat runtime that has the KPI deps
but not the alert-routing ones could not import the dispatcher. ``src.kpi`` is
also where ``KPIMetadata`` -- the declaration this rule is derived FROM --
already lives. KPI-name resolution stays function-local, so the ``src.services``
cost is paid only on the conflict path that needs it.

The rule: two figures are comparable only if their substrate declarations are
EQUAL, and an undeclared substrate is never comparable with anything.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

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


def measure_basis_for_kpi(
    kpi: Any, result_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Declare what a computed KPI figure measures, DERIVED from the registry.

    ``KPIMetadata.tables`` is the existing SSOT for a KPI's substrate (measured:
    all 45 registry KPIs populate it), so this needs no hand-maintained map —
    and deriving gets the one genuine exception right for free. WS3-BI-010
    (ROI) really does read ``business_metrics``, so it IS comparable with
    ``e2i_data_query_tool``'s stored ROI, where a blanket "the KPI tool means
    treatment_events" rule would wrongly fence it off.
    """
    declared = sorted(getattr(kpi, "tables", None) or [])
    # A calculator that recorded which branch answered wins over the registry's
    # union of POSSIBLE sources (#1640). ROI is the case that matters: it tries
    # business_metrics first and only falls back to agent_activities when
    # unscoped and empty, so the union both over-claims and over-fences.
    actual = ((result_metadata or {}).get("context") or {}).get("measure_basis_substrate")
    tables = sorted(actual) if actual else declared
    return {
        "substrate": tables,
        # Whether a CALCULATOR recorded the branch that answered. False does not
        # mean the substrate is wrong: for a single-source KPI, and for the 11
        # registry KPIs whose two tables are JOINED in one query (measured), the
        # declared set is exact. It is a superset only for a KPI whose
        # calculator falls back between sources -- ROI is the one such case in
        # the registry today, and it now records which branch answered. Nothing
        # here can derive fallback-ness statically, so the payload states what
        # it knows instead of guessing.
        "runtime_confirmed": bool(actual),
        "declared_sources": declared,
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


# --------------------------------------------------------------------------
# Deriving a substrate from the registry SQL a KPI actually runs (#1640).
#
# These live here rather than in the CopilotKit route because three surfaces
# need them -- the Home summary tiles, the KPI point-value endpoint, and the
# history/segmented/nowcast series that back the chat trend chart. The chart
# is the surface the issue is about: `renderKpiTrend` fetches real history
# through `getKPIHistory` (E2ICopilotProvider.tsx:1008), so a TRx series can
# land beside a business_metrics TRx figure in one answer.
#
# The DB import is function-local, so this module stays cheap for callers that
# only need the constants above.
# --------------------------------------------------------------------------


#: Words that can follow FROM/JOIN or an alias but are never a table name.
#: `LATERAL` and `SELECT` head a subquery; the rest end the table list, and
#: treating one as a table (or as an alias to skip past) yields a phantom.
_CLAUSE_KEYWORDS = frozenset(
    {
        "where",
        "group",
        "order",
        "having",
        "limit",
        "offset",
        "union",
        "intersect",
        "except",
        "on",
        "using",
        "window",
        "left",
        "right",
        "full",
        "inner",
        "outer",
        "cross",
        "join",
        "lateral",
        "natural",
        "fetch",
        "for",
        "returning",
        "with",
    }
)
_NOT_A_TABLE = _CLAUSE_KEYWORDS | {"select", ""}


def tables_in_sql(sql: str) -> list[str]:
    """Real tables a registry query reads, derived from the SQL itself (#1640).

    A PHANTOM table is a wrong basis, which is worse than a narrow one, so the
    forms that produce one are handled explicitly:

    * ``FROM public.treatment_events`` yields ``treatment_events``, not
      ``public`` -- measured, 12 of the 306 registry queries are
      schema-qualified today;
    * ``JOIN LATERAL (...)`` and ``FROM (SELECT ...)`` name no table here, and
      the inner query's own FROM clauses are picked up on their own;
    * quoted identifiers are read.

    CTE names are excluded: ``WITH first_brand AS (...) SELECT ... FROM
    first_brand`` reads ``treatment_events``, not a table called
    ``first_brand``.
    """
    ident = r'(?:"([a-z_][a-z0-9_]*)"|([a-z_][a-z0-9_]*))'
    ctes = {
        (m.group(1) or m.group(2)).lower()
        for m in re.finditer(
            # RECURSIVE follows WITH; MATERIALIZED / NOT MATERIALIZED follow AS.
            # Missing either does not merely narrow the basis -- the CTE name
            # then survives as a PHANTOM table. Measured against the live
            # registry 2026-08-15: 0 of 306 rows use either form today, but the
            # migration-044 CHECK admits them (it only requires the text to
            # start with `with` or `select`).
            rf"(?:with\s+(?:recursive\s+)?|,\s*){ident}\s+as\s*"
            r"(?:not\s+)?(?:materialized\s+)?\(",
            sql,
            re.IGNORECASE,
        )
    }
    # A FROM clause can list several tables: `FROM a, b` and `FROM a x, b y`
    # are old-style cross joins, and 40 of the 306 registry queries use the
    # form (measured 2026-08-15) -- `FROM oncologists, engaged`,
    # `FROM triggered, converted`. Matching only the FIRST reference after the
    # keyword drops the rest.
    #
    # Today that costs nothing: in all 40, every reference past the first is a
    # CTE, so first-only and full-list agree on all 306 queries (measured: 0
    # differences, 0 phantoms either way). The scan is fixed anyway because
    # the agreement is a property of the current SQL, not of the rule.
    table_ref = rf"(?:{ident}\s*\.\s*)?{ident}"
    found: set[str] = set()
    for m in re.finditer(r"\b(?:from|join)\b", sql, re.IGNORECASE):
        rest = sql[m.end() :]
        while True:
            ref = re.match(rf"\s*(?:lateral\s+)?{table_ref}", rest, re.IGNORECASE)
            if not ref:
                # A subquery or `(` -- it names no table here, and its own FROM
                # clauses are picked up by the outer scan on their own.
                break
            # groups: 1/2 = optional schema (quoted/bare), 3/4 = table
            name = (ref.group(3) or ref.group(4) or "").lower()
            if name in _NOT_A_TABLE:
                break
            tail = rest[ref.end() :]
            if re.match(r"\s*\(", tail):
                # `FROM generate_series(...)` -- a set-returning function, not a
                # table. Adding it is a phantom (the pre-existing scan did), so
                # stop this clause rather than guess past the argument list.
                # Measured 2026-08-15: no registry query uses the form, so this
                # closes a latent case rather than a live one.
                break
            found.add(name)
            # Continue ONLY across a comma, and only after skipping an optional
            # alias -- anything else (WHERE, ON, GROUP BY, a closing paren)
            # ends the table list. Scanning past those would pull column names
            # in as phantom tables.
            alias = re.match(r"\s+(?:as\s+)?[a-z_][a-z0-9_]*", tail, re.IGNORECASE)
            if alias and alias.group(0).strip().lower().split()[-1] not in _CLAUSE_KEYWORDS:
                tail = tail[alias.end() :]
            comma = re.match(r"\s*,", tail)
            if not comma:
                break
            rest = tail[comma.end() :]
    return sorted(found - ctes)


#: How long a registry read stays good. Short enough that a migration is picked
#: up without a restart, long enough that Home does not re-query per render.
SUBSTRATE_CACHE_TTL_SECONDS = 300

_CACHE: dict[tuple[str, ...], tuple[float, dict[str, list[str]]]] = {}


def reset_substrate_cache() -> None:
    """Drop the memoized registry reads (tests, and after a migration)."""
    _CACHE.clear()


def read_query_substrates(query_ids: tuple[str, ...]) -> dict[str, list[str]]:
    """``{query_id: [table, ...]}`` read from ``kpi_query_registry``."""
    from src.api.dependencies.supabase_client import get_supabase

    client = get_supabase()
    if client is None:
        return {}
    try:
        rows = (
            client.table("kpi_query_registry")
            .select("query_id,sql")
            .in_("query_id", list(query_ids))
            .execute()
        ).data or []
    except Exception as e:  # fail closed: no basis beats a wrong basis
        logger.warning(
            f"[measure_basis] kpi_query_registry unreadable, omitting measure_basis: {e}"
        )
        return {}
    return {r["query_id"]: tables_in_sql(r.get("sql") or "") for r in rows}


def query_substrates_cached(query_ids: tuple[str, ...]) -> dict[str, list[str]]:
    """Memoized registry read, with a TTL and NO caching of failures.

    An ``lru_cache`` here meant a single transient read failure cached ``{}``
    for the life of the process, so Home would emit numeric tiles with no basis
    until restart — and a registry migration could never be picked up either.
    A failure is not a result; only a non-empty read is remembered.
    """
    entry = _CACHE.get(query_ids)
    if entry is not None and (time.monotonic() - entry[0]) < SUBSTRATE_CACHE_TTL_SECONDS:
        return entry[1]
    result = read_query_substrates(query_ids)
    if result:
        _CACHE[query_ids] = (time.monotonic(), result)
    return result


def materialized_history_basis(kpi: Any, rows: Optional[list] = None) -> Dict[str, Any]:
    """Declare what a ``kpi_history`` SERIES measures (#1640).

    ``/api/kpis/{id}/history`` reads the materialized ``kpi_history`` table,
    not the calculator -- so the substrate is ``kpi_history``, and saying
    ``treatment_events`` here would claim the series was computed live, which
    is what the SEGMENTED endpoint does and this one explicitly does not.
    ``materialized_from`` keeps the provenance without overstating it.

    This is the surface #1640 is about: `renderKpiTrend` charts this series,
    and the same answer can carry a business_metrics TRx figure from
    ``e2i_data_query_tool`` -- measured ~73x apart.
    """
    # Provenance comes from the ROWS, not from the registry declaration. Every
    # kpi_history row carries the backfill's `source` tag, and for ROI that tag
    # is `business_metrics.roi` -- while `KPIMetadata.tables` is the UNION of
    # its calculator's POSSIBLE sources, which includes agent_activities.
    # Declaring the union would both over-claim provenance and FENCE a
    # history-vs-current ROI comparison that is business-metrics-backed on both
    # sides. Measured live 2026-08-15: all 3,280 WS3-BI-010 rows are
    # `business_metrics.roi`; all 720 WS3-BI-005 rows are
    # `treatment_events.event_date`.
    sources = sorted({str(row.get("source")) for row in (rows or []) if row.get("source")})
    runtime_confirmed = bool(sources)
    if not sources:
        # An empty series is not "provenance unknown": the backfill registers a
        # tag per KPI, and that is what these rows WOULD carry.
        from src.kpi.history_backfill import HANDLER_SOURCES

        registered = HANDLER_SOURCES.get(str(getattr(kpi, "id", "")), "")
        sources = [registered] if registered else []
    return {
        "substrate": ["kpi_history"],
        "computed": False,
        "materialized_from": sources,
        "runtime_confirmed": runtime_confirmed,
        "grain": "kpi x brand x region x calendar month",
        "measure": (
            "monthly materialized history of the computed KPI"
            + (f" (materialized from {', '.join(sources)})" if sources else "")
        ),
        "note": (
            "Read from the materialized kpi_history table -- the stored form of the "
            "COMPUTED KPI. Compare only with a figure resting on the same source: "
            "`materialized_from` names it, and it is NOT always the same one "
            "(ROI history is backfilled from business_metrics.roi, so it IS "
            "comparable with stored ROI; the Rx-volume family is backfilled from "
            "treatment_events, so it is NOT). For a treatment_events-backed series, "
            "do NOT plot or compare it against e2i_data_query_tool(query_type='kpi') "
            "business_metrics values -- measured 2026-08-15, those are ~73x larger "
            "because they are a modeled market-scale level rather than a count of "
            "observed events."
        ),
    }


async def registry_query_basis(query_id: str) -> Optional[Dict[str, Any]]:
    """The basis of a series computed live by one registry query (#1640).

    Returns ``None`` when the registry cannot be read: no basis beats a wrong
    basis, and a caller that omits the field is honest about not knowing.

    Off the event loop -- the reader is the sync supabase client, and the
    routes that need this are async. The 300s cache means the hop is paid at
    most once per query per worker per window.
    """
    tables = (await asyncio.to_thread(query_substrates_cached, (query_id,))).get(query_id)
    if not tables:
        return None
    return {
        "substrate": list(tables),
        "computed": True,
        # Not a declaration read off the registry metadata: these are the
        # tables of the SQL this request ran.
        "runtime_confirmed": True,
        "query_id": query_id,
        "measure": f"computed live from {', '.join(tables)}",
        "note": (
            "Computed from the operational substrate at query time. Do NOT plot or "
            "compare it against e2i_data_query_tool(query_type='kpi') values for a "
            "volume KPI: measured 2026-08-15, those are ~73x larger because they are a "
            "modeled market-scale level rather than a count of observed events."
        ),
    }


def substrates_agree(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> bool:
    """Do these two figures rest on the same substrate?

    **This answers "same source", NOT "same measure", and is only meaningful for
    ONE metric surfaced two ways** -- which is the only way it is called
    (:func:`cross_substrate_conflict`, comparing a metric's computed form
    against the stored rows for that same metric).

    The limit is real and measured: Trigger Recall (WS2-TR-002) and Conversion
    Rate (WS3-BI-009) both declare ``['treatment_events', 'triggers']``, so this
    returns True for them -- correctly, they DO share a substrate -- while a
    recall and a conversion rate are obviously not interchangeable figures. Do
    not reach for this as a general "may I compare these two numbers" oracle;
    it cannot answer that and does not claim to.

    Fails CLOSED on a missing declaration: an undeclared substrate is not
    evidence of agreement, and two undeclared ones are not evidence of agreement
    with each other.
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
    if substrates_agree(computed, BUSINESS_METRICS_BASIS):
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


#: Superseded name. `bases_are_comparable` read as a general comparability
#: oracle, which this is not -- it compares SUBSTRATES, for one metric surfaced
#: two ways. Kept so existing imports keep working.
bases_are_comparable = substrates_agree
