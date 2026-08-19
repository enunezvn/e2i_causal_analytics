"""
E2I Chatbot Tools for LangGraph Integration.

Provides LangGraph-compatible tools for the E2I chatbot agent:
- e2i_data_query_tool: Unified access to ALL E2I analytics data
- causal_analysis_tool: Run causal analysis via hybrid RAG search
- agent_routing_tool: Route to specific tier agents (keyword-based)
- conversation_memory_tool: Retrieve chat history
- document_retrieval_tool: Hybrid RAG search
- orchestrator_tool: Execute queries through the full 22-agent orchestrator system
- tool_composer_tool: Process multi-faceted queries via Tool Composer pipeline

Adapted from Pydantic AI patterns to LangGraph @tool decorators.
"""

import asyncio
import contextvars
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from src.agents.tool_composer import compose_query
from src.api.routes.chatbot_dspy import (
    CHATBOT_DSPY_ROUTING_ENABLED,
    VALID_AGENTS,
    route_agent_dspy,
    route_agent_hardcoded,
)
from src.api.routes.cognitive import get_orchestrator
from src.kpi.synthetic_mode import kpi_include_synthetic
from src.memory.services.factories import get_async_supabase_client
from src.rag.retriever import hybrid_search
from src.repositories import (
    AgentActivityRepository,
    BusinessMetricRepository,
    CausalPathRepository,
    CausalValidationRepository,
    TriggerRepository,
)
from src.repositories.chatbot_conversation import (
    get_chatbot_conversation_repository,
)
from src.repositories.chatbot_message import (
    get_chatbot_message_repository,
)
from src.services import cohort_resolution, kpi_resolution
from src.services.enum_labels import (
    BRAND_ENUM_LABELS,
    REGION_ENUM_LABELS,
    resolve_brand_label,
    resolve_region_label,
)
from src.utils.redaction import redact_query

logger = logging.getLogger(__name__)

# #1698: the verbatim latest user message, stashed by the copilotkit handler
# before the graph runs. The model's orchestrator_tool ``query`` arg is a lossy
# rewrite — the measured 2.1 defect stripped servable cohort criteria from it —
# and ToolNode invokes tools with the model's args only, so a side channel is
# the only way the original ask can reach tool-side honesty accounting. A
# contextvar (same rationale as copilotkit's _session_id_context) leaves every
# non-chat caller (benchmarks, direct invocations) unaffected: the var is unset
# there and behavior is unchanged.
_raw_user_query_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "e2i_raw_user_query", default=None
)


def set_raw_user_query(text: Optional[str]) -> "contextvars.Token[Optional[str]]":
    """Stash the verbatim latest user message for tool-side honesty accounting."""
    return _raw_user_query_context.set(text or None)


def reset_raw_user_query(token: "contextvars.Token[Optional[str]]") -> None:
    _raw_user_query_context.reset(token)


# Try to import Opik for tracing
try:
    from src.mlops.opik_connector import OpikConnector

    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    logger.debug("Opik not available for agent routing tracing")


# =============================================================================
# ENUMS AND MODELS
# =============================================================================


class E2IQueryType(str, Enum):
    """Supported query types for E2I data queries."""

    KPI = "kpi"
    CAUSAL_CHAIN = "causal_chain"
    AGENT_ANALYSIS = "agent_analysis"
    TRIGGERS = "triggers"
    EXPERIMENTS = "experiments"
    PREDICTIONS = "predictions"
    RECOMMENDATIONS = "recommendations"
    DRIFT_REPORTS = "drift_reports"


class TimeRange(str, Enum):
    """Time range options for queries."""

    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_YEAR = "last_year"
    ALL_TIME = "all_time"


class E2IDataQueryInput(BaseModel):
    """Input schema for e2i_data_query_tool."""

    query_type: E2IQueryType = Field(
        description="Type of E2I data to query: kpi, causal_chain, agent_analysis, triggers, experiments, predictions, recommendations, drift_reports"
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter, resolved case-insensitively against the actual data values (e.g., Kisqali)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region filter, resolved case-insensitively against the actual data values (e.g., Northeast)",
    )
    kpi_name: Optional[str] = Field(
        default=None,
        description="Specific KPI name for KPI queries (e.g., TRx, NRx, conversion_rate)",
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Agent name filter for agent_analysis queries",
    )
    time_range: TimeRange = Field(
        default=TimeRange.LAST_30_DAYS,
        description="Time range for the query",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional filters as key-value pairs",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_type": "kpi",
                "brand": "Kisqali",
                "region": "Northeast",
                "kpi_name": "TRx",
                "time_range": "last_30_days",
                "limit": 10,
            }
        }
    )


class CausalAnalysisInput(BaseModel):
    """Input schema for causal_analysis_tool."""

    kpi_name: str = Field(
        description="KPI to analyze (e.g., TRx, NRx, conversion_rate, market_share)"
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter, resolved case-insensitively against the actual data values (e.g., Kisqali)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region filter, resolved case-insensitively against the actual data values (e.g., Northeast)",
    )
    time_period: Optional[str] = Field(
        default="last_30_days",
        description="Time period for analysis",
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for causal relationships",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_name": "TRx",
                "brand": "Kisqali",
                "region": "Northeast",
                "time_period": "last_30_days",
                "min_confidence": 0.7,
            }
        }
    )


class AgentRoutingInput(BaseModel):
    """Input schema for agent_routing_tool."""

    query: str = Field(description="The user's query to route")
    target_agent: Optional[str] = Field(
        default=None,
        description="Specific agent to route to (if known)",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for routing decision",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Why did TRx drop for Kisqali in Q3?",
                "target_agent": "causal_impact",
                "context": {"intent": "causal_analysis", "brand": "Kisqali"},
            }
        }
    )


class ConversationMemoryInput(BaseModel):
    """Input schema for conversation_memory_tool."""

    session_id: str = Field(description="Session ID to retrieve history for")
    message_count: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recent messages to retrieve",
    )
    include_tool_calls: bool = Field(
        default=True,
        description="Whether to include tool call details",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "sess_abc123def456",
                "message_count": 10,
                "include_tool_calls": True,
            }
        }
    )


class DocumentRetrievalInput(BaseModel):
    """Input schema for document_retrieval_tool."""

    query: str = Field(description="Search query for document retrieval")
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter for documents",
    )
    kpi_name: Optional[str] = Field(
        default=None,
        description="KPI name for targeted retrieval",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Kisqali conversion rate trends",
                "k": 5,
                "brand": "Kisqali",
                "kpi_name": "conversion_rate",
            }
        }
    )


class OrchestratorToolInput(BaseModel):
    """Input schema for orchestrator_tool."""

    query: str = Field(
        description="The query to process through the E2I orchestrator and 22-agent system"
    )
    target_agent: Optional[str] = Field(
        default=None,
        description="Specific agent to route to (e.g., causal_impact, experiment_designer, drift_monitor)",
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand context for the query, resolved case-insensitively against the actual data values (e.g., Kisqali)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region context for the query, resolved case-insensitively against the actual data values (e.g., Northeast)",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context continuity",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Analyze the impact of rep visits on Kisqali TRx in the Northeast",
                "target_agent": "causal_impact",
                "brand": "Kisqali",
                "region": "Northeast",
                "session_id": "sess_abc123",
            }
        }
    )


class ToolComposerToolInput(BaseModel):
    """Input schema for tool_composer_tool."""

    query: str = Field(
        description="Multi-faceted query requiring decomposition and multi-agent processing"
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand context for the query, resolved case-insensitively against the actual data values (e.g., Kisqali)",
    )
    region: Optional[str] = Field(
        default=None,
        description=(
            "Region context for the query, resolved case-insensitively against the "
            "actual geographic_region values in the data (US census regions)."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context continuity",
    )
    data_source: Optional[str] = Field(
        default=None,
        description=(
            "Optional cohort data source (table name or parquet/s3 path) used "
            "to resolve a real DataFrame for (brand, region) and supply it to "
            "the tools as estimation_data. If omitted or unresolvable, the "
            "composable tools fail-closed honestly."
        ),
    )
    max_parallel: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of parallel tool executions",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Compare TRx trends across Kisqali, Fabhalta, and Remibrutinib, then explain causal factors",
                "brand": None,
                "region": "Northeast",
                "session_id": "sess_abc123",
                "max_parallel": 3,
            }
        }
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_time_filter(time_range: TimeRange) -> datetime:
    """Convert time range enum to datetime filter."""
    now = datetime.now(timezone.utc)
    if time_range == TimeRange.LAST_7_DAYS:
        return now - timedelta(days=7)
    elif time_range == TimeRange.LAST_30_DAYS:
        return now - timedelta(days=30)
    elif time_range == TimeRange.LAST_90_DAYS:
        return now - timedelta(days=90)
    elif time_range == TimeRange.LAST_YEAR:
        return now - timedelta(days=365)
    else:  # ALL_TIME
        return datetime(2020, 1, 1)


def _normalize_metric_name(kpi_name: str) -> str:
    """Map a display-form KPI name to the stored metric_name key.

    business_metrics.metric_name values are lowercase snake_case (trx, nrx,
    market_share, conversion_rate, hcp_engagement_score) while LLM tool calls
    pass display forms ("TRx", "Market Share") — an exact-match filter on
    those returns 0 rows.
    """
    return kpi_name.strip().lower().replace("-", "_").replace(" ", "_")


# business_metrics.region / business_metrics.brand are Postgres ENUM columns
# (region_type, brand_type). The labels, the platform's region synonym table
# and the resolvers all live in src.services.enum_labels (#1505) — cohort
# resolution shares them, so an enum change lands in one place.
#
# This surface asks for the SYNONYM-TOLERANT contract: an LLM tool call
# naturally produces display and colloquial forms ("Northeast", "North East",
# "NE", "the Pacific"), and unlike metric_name (plain text, mismatch = 0 rows)
# a non-label string in an enum cast raises 22P02 and fails the ENTIRE KPI
# query (#1501). An input that maps to no label means "no row can ever hold
# it"; the caller returns the 0-row result the filter implies instead of
# erroring.


def _normalize_region(region: str) -> Optional[str]:
    """Map a display-form region or synonym to its region_type enum label.

    Since #1565 the shared resolver also strips natural-phrasing noise tokens
    ("the Northeast region" -> northeast) and accepts "west coast"; anything
    still unresolved is genuinely ambiguous ("East", "east coast") or unknown,
    so the caller pairs its honest failure with ``_REGION_CLARIFY_HINT``.
    """
    return resolve_region_label(region, allow_synonyms=True)


#: #1565: an unresolvable region should end in a QUESTION, not a dead end.
#: The error keeps naming the known labels (failure-closed, #1501/#1538
#: precedent); this hint tells the LLM what to ask the user. Phrasings like
#: "East Coast" are unresolvable BY DESIGN — the Atlantic seaboard spans the
#: northeast AND south census regions — so only the user can disambiguate.
_REGION_CLARIFY_HINT = (
    "Ask the user which US census region they mean: northeast, south, "
    "midwest, or west. Phrasings like 'East' or 'East Coast' span more than "
    "one census region, so the data cannot be scoped to them without the "
    "user's choice."
)


_normalize_brand = resolve_brand_label


# #1640: the measure-basis rule is an SSOT in src/kpi/measure_basis.py so
# the orchestrator and the Home KPI summary can import it without paying this
# module's ~30s import cost (#1475 precedent). Private aliases kept so existing
# consumers and tests read unchanged.
from src.kpi.measure_basis import (  # noqa: E402,F401
    BUSINESS_METRICS_BASIS as _BUSINESS_METRICS_BASIS,
)
from src.kpi.measure_basis import (  # noqa: E402,F401
    bases_are_comparable,  # re-exported under its public name for consumers
)
from src.kpi.measure_basis import (  # noqa: E402,F401
    cross_substrate_conflict as _cross_substrate_conflict,
)
from src.kpi.measure_basis import (  # noqa: E402,F401
    measure_basis_for_kpi as _measure_basis_for_kpi,
)


async def _query_kpis(
    brand: Optional[str],
    region: Optional[str],
    kpi_name: Optional[str],
    since: datetime,
    limit: int,
) -> Dict[str, Any]:
    """Query KPI metrics from business_metrics table (newest first, windowed).

    Synthetic provenance rides the SSOT deployment gate inside
    ``apply_provenance_filter`` (showcase instances include synthetic rows,
    real-mode deployments exclude them); ``data_source`` labels the answer
    honestly either way (kpi_calculate_tool precedent).
    """
    try:
        filters: Dict[str, Any] = {}
        unmatched: List[str] = []
        if brand:
            normalized_brand = _normalize_brand(brand)
            if normalized_brand is None:
                unmatched.append(
                    f"brand {brand!r} does not match any known brand "
                    f"({', '.join(sorted(BRAND_ENUM_LABELS))})"
                )
            else:
                filters["brand"] = normalized_brand
        if region:
            normalized_region = _normalize_region(region)
            if normalized_region is None:
                unmatched.append(
                    f"region {region!r} does not match any known region "
                    f"({', '.join(REGION_ENUM_LABELS)})"
                )
            else:
                filters["region"] = normalized_region
        if kpi_name:
            # business_metrics uses 'metric_name' column, not 'kpi_name'
            filters["metric_name"] = _normalize_metric_name(kpi_name)

        window_start = since.date().isoformat()

        if unmatched:
            # brand/region are enum columns: an unmappable value can never
            # match a row, and passing it through would 22P02 the entire
            # query (#1501). Return the 0-row result the filter implies —
            # the same outcome _normalize_metric_name's passthrough has on
            # the plain-text metric_name column — with an honest note.
            requested = dict(filters)
            if brand and "brand" not in filters:
                requested["brand"] = brand
            if region and "region" not in filters:
                requested["region"] = region
            response: Dict[str, Any] = {
                "success": True,
                "query_type": "kpi",
                "count": 0,
                "data": [],
                "filters_applied": requested,
                "window_start": window_start,
                "data_source": "synthetic" if kpi_include_synthetic() else "database",
                "measure_basis": _BUSINESS_METRICS_BASIS,
                # No rows, so no stored figure to be confused with anything.
                # The notice is a caveat ON the rows above it (#1640 codex
                # iter-3): _query_kpis filters metric_name with
                # _normalize_metric_name while the notice resolves through
                # recognize_kpi, and those diverge -- "total prescriptions"
                # filters 'total_prescriptions', which is never a stored key,
                # yet resolved to TRx. Gating on rows closes that: a key the
                # table does not use cannot return rows, so the mismatch can
                # never reach a reader.
                "cross_substrate_conflict": None,
                "note": "; ".join(unmatched) + "; returned 0 rows",
            }
            if region and "region" not in filters:
                # #1565: unresolvable region -> clarify, not a dead end.
                response["hint"] = _REGION_CLARIFY_HINT
            return response

        client = await get_async_supabase_client()
        repo = BusinessMetricRepository(client)

        metrics = await repo.query_metrics(
            filters=filters,
            since=window_start,
            limit=limit,
        )

        return {
            "success": True,
            "query_type": "kpi",
            "count": len(metrics),
            "data": metrics,
            "filters_applied": filters,
            "window_start": window_start,
            "data_source": "synthetic" if kpi_include_synthetic() else "database",
            "measure_basis": _BUSINESS_METRICS_BASIS,
            "cross_substrate_conflict": (_cross_substrate_conflict(kpi_name) if metrics else None),
        }
    except Exception as e:
        logger.error(f"KPI query failed: {e}")
        return {"success": False, "error": str(e), "query_type": "kpi"}


def _format_causal_path(
    row: Dict[str, Any], refutation_evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Map a ``causal_paths`` registry row onto the chat-facing chain shape.

    ``confidence`` here is the registry's method-attributed ``confidence_level``
    (a real 0-1 causal confidence) — never a retrieval similarity score.

    #1352: ``validation_status`` carries the migration-119 pinned semantics
    ('validated' == "RefutationSuite evidence exists and passed"), and
    ``refutation_evidence`` is the per-path summary from
    :func:`_summarize_refutation_rows` — ``None`` means the evidence lookup
    succeeded and found nothing on record (see
    :func:`_refutation_evidence_entry` for the lookup-failed state).
    """
    return {
        "path_id": row.get("path_id"),
        "cause": row.get("start_node"),
        "effect": row.get("end_node"),
        "via": list(row.get("intermediate_nodes") or []),
        "effect_size": row.get("causal_effect_size"),
        "confidence": row.get("confidence_level"),
        "method": row.get("method_used"),
        "time_lag_days": row.get("time_lag_days"),
        "business_impact_estimate": row.get("business_impact_estimate"),
        "brand": row.get("brand"),
        "validation_status": row.get("validation_status"),
        "refutation_evidence": refutation_evidence,
    }


def _summarize_refutation_rows(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Aggregate one path's ``causal_validations`` rows into a chat summary.

    Gate priority mirrors ``CausalValidationRepository.get_gate_decision``
    (block > review > proceed), extended over the full ``gate_decision`` enum
    (reject counts as blocking, augment as review-band, accept as proceed).
    ``evidence_is_synthetic`` reads the migration-119 provenance label
    (``details_json.is_synthetic``) so seeded synthetic evidence can never
    masquerade as real RefutationSuite output in an answer.
    """
    if not rows:
        return None

    def _status_count(status: str) -> int:
        return sum(1 for r in rows if r.get("status") == status)

    gates = {r.get("gate_decision") for r in rows}
    if gates & {"block", "reject"}:
        gate = "block"
    elif gates & {"review", "augment"}:
        gate = "review"
    else:
        gate = "proceed"

    confidences = [
        float(r["confidence_score"]) for r in rows if r.get("confidence_score") is not None
    ]

    def _details(r: Dict[str, Any]) -> Dict[str, Any]:
        raw = r.get("details_json")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                return {}
        return {}

    timestamps = [str(r["created_at"]) for r in rows if r.get("created_at")]
    return {
        "tests_total": len(rows),
        "tests_passed": _status_count("passed"),
        "tests_failed": _status_count("failed"),
        "tests_warning": _status_count("warning"),
        "gate_decision": gate,
        "confidence_score": (sum(confidences) / len(confidences)) if confidences else None,
        "evidence_is_synthetic": any(bool(_details(r).get("is_synthetic")) for r in rows),
        "latest_test_at": max(timestamps) if timestamps else None,
    }


def _refutation_evidence_entry(
    path_id: Optional[str], summaries: Optional[Dict[str, Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Resolve one path's refutation-evidence entry, keeping three states
    honestly distinct:

    * summary dict — evidence rows exist for this path;
    * ``None`` — the lookup succeeded and there is genuinely no refutation
      evidence on record;
    * lookup-failed marker — the evidence query errored (``summaries is
      None``); this must never be presented as absence of evidence.
    """
    if summaries is None:
        return {
            "lookup_failed": True,
            "note": (
                "refutation-evidence lookup unavailable for this answer — "
                "do not read this as 'no evidence exists'"
            ),
        }
    if not path_id:
        return None
    return summaries.get(path_id)


async def _fetch_refutation_summaries(
    client: Any, paths: List[Dict[str, Any]]
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Batch-fetch and summarize refutation evidence for a list of path rows.

    Returns ``None`` on lookup failure (the caller degrades via
    :func:`_refutation_evidence_entry`'s lookup-failed marker — evidence is
    enrichment, never a gate on answering).
    """
    path_ids = [str(p.get("path_id")) for p in paths if p.get("path_id")]
    if not path_ids:
        return {}
    try:
        repo = CausalValidationRepository(client)
        rows_by_path = await repo.get_rows_for_paths(path_ids)
    except Exception as e:
        logger.warning(f"Refutation-evidence lookup failed (degrading honestly): {e}")
        return None
    summaries: Dict[str, Dict[str, Any]] = {}
    for pid, rows in rows_by_path.items():
        summary = _summarize_refutation_rows(rows)
        if summary is not None:
            summaries[pid] = summary
    return summaries


async def _query_causal_chains(
    brand: Optional[str],
    kpi_name: Optional[str],
    since: datetime,
    limit: int,
    min_confidence: float = 0.5,
    include_synthetic: Optional[bool] = None,
) -> Dict[str, Any]:
    """Query causal relationships from the ``causal_paths`` registry.

    Provenance (#893): synthetic causal paths must never surface AS REAL
    insight. ``include_synthetic=None`` (the chat default) resolves to the
    platform gate ``kpi_include_synthetic()`` — the same convention as the KPI
    tools: in synthetic-showcase mode paths surface labeled
    ``data_source: "synthetic"``; in real mode the filter fails closed (same
    semantics as #872). The explicit True/False override exists for
    agent-context/validation callers only and is deliberately NOT exposed in
    the LLM tool schema.

    2026-07-07 rewire: the ``kpi_name`` branch previously detoured into
    ``hybrid_search`` and returned RRF-scored RAG documents as "causal chains".
    It now queries the registry (real ``confidence_level``) like everything
    else. ``since`` is not applied: registry paths are current modeled
    knowledge, not dated events.
    """
    try:
        client = await get_async_supabase_client()
        repo = CausalPathRepository(client)
        if include_synthetic is None:
            include_synthetic = kpi_include_synthetic()
        data_source = "synthetic" if include_synthetic else "database"

        if kpi_name:
            paths = await repo.search_paths_for_outcome(
                kpi_name,
                brand=brand,
                min_confidence=min_confidence,
                limit=limit,
                include_synthetic=include_synthetic,
            )
            # #1352: surface validation provenance (status + refutation
            # evidence summary) so answers can cite it (the q07 gap).
            summaries = await _fetch_refutation_summaries(client, paths)
            return {
                "success": True,
                "query_type": "causal_chain",
                "count": len(paths),
                "data": [
                    _format_causal_path(
                        p,
                        refutation_evidence=_refutation_evidence_entry(p.get("path_id"), summaries),
                    )
                    for p in paths
                ],
                "kpi_analyzed": kpi_name,
                "data_source": data_source,
            }

        filters: dict[str, str] = {"brand": brand} if brand else {}
        paths = await repo.get_many(
            filters=filters, limit=limit, include_synthetic=include_synthetic
        )
        # Raw-row branch: rows already carry validation_status; attach the
        # same per-path refutation-evidence entry for parity (#1352).
        summaries = await _fetch_refutation_summaries(client, paths)
        for p in paths:
            p["refutation_evidence"] = _refutation_evidence_entry(p.get("path_id"), summaries)
        return {
            "success": True,
            "query_type": "causal_chain",
            "count": len(paths),
            "data": paths,
            "data_source": data_source,
        }
    except Exception as e:
        logger.error(f"Causal chain query failed: {e}")
        return {"success": False, "error": str(e), "query_type": "causal_chain"}


async def _query_agent_analysis(
    agent_name: Optional[str],
    brand: Optional[str],
    since: datetime,
    limit: int,
) -> Dict[str, Any]:
    """Query agent analysis outputs from the agent_activities table.

    #1355 rewire: ``since`` was accepted but never applied and ``brand`` was
    silently dropped ("brand isn't a column"), so an empty-window answer was
    indistinguishable from a brand gap (the q16 misreading). Both now apply:

    * ``since`` filters ``activity_timestamp`` (the time_range window every
      sibling query already honors);
    * ``brand`` resolves through the ``analysis_results->>'brand'`` JSONB
      field — the documented writer contract (the DGP seed generator and the
      runtime activity writer ``src/agents/activity_writer.py`` both stamp
      ``analysis_results.brand`` on brand-scoped analyses).

    Provenance rides the repository's default-exclude predicate; on a
    synthetic-gold showcase instance the ``E2I_INCLUDE_SYNTHETIC`` deployment
    gate surfaces seeded rows, and ``data_source`` labels the answer honestly
    (the KPI/causal-chain convention).
    """
    try:
        from src.repositories.provenance import deployment_includes_synthetic

        client = await get_async_supabase_client()
        repo = AgentActivityRepository(client)

        activities = await repo.query_activities(
            agent_name=agent_name,
            brand=brand,
            since=since,
            limit=limit,
        )

        return {
            "success": True,
            "query_type": "agent_analysis",
            "count": len(activities),
            "data": activities,
            "agent_filter": agent_name,
            "brand_filter": brand,
            "window_start": since.isoformat() if since else None,
            "data_source": "synthetic" if deployment_includes_synthetic() else "database",
        }
    except Exception as e:
        logger.error(f"Agent analysis query failed: {e}")
        return {"success": False, "error": str(e), "query_type": "agent_analysis"}


async def _query_triggers(
    brand: Optional[str],
    region: Optional[str],
    since: datetime,
    limit: int,
) -> Dict[str, Any]:
    """Query triggers/alerts from triggers table.

    #1727: the requested time window IS applied — ``since`` filters
    ``trigger_timestamp`` (the #1355 ``_query_agent_analysis`` pattern), and
    the payload states the window (``time_period_applied: True`` +
    ``window_start``). ``brand``/``region`` remain NOT applied (#1700,
    wording tightened by #1718): triggers rows DO carry a ``brand_id``
    column — region is the one genuinely absent from the row schema — but
    neither is applied as a filter here, and the payload says so explicitly
    (``brand_applied``/``region_applied``: False, plus a ``scope_note`` when
    a brand or region was requested) — in the 2026-08-18 A.9-seed run the
    synthesis layer attributed the conversation's region to region-less
    trigger rows because the payload gave it nothing honest to attribute
    scope to. Mirrors #1695's fields on ``causal_analysis_tool``.
    """
    try:
        client = await get_async_supabase_client()
        repo = TriggerRepository(client)

        # #1727: the time window is real (trigger_timestamp >= since). Brand/
        # region are NOT applied — rows carry brand_id (region does not exist
        # in this table), but this query deliberately returns the cross-brand
        # set for the window. Parameters kept for API compatibility (#1718).
        triggers = await repo.get_triggers_since(since, limit=limit)

        response: Dict[str, Any] = {
            "success": True,
            "query_type": "triggers",
            "count": len(triggers),
            "data": triggers,
            # #1700: brand/region filters are not applied above — without
            # these fields the synthesis layer reads the request's brand/
            # region context as applied filters and fabricates scope in prose.
            "brand_applied": False,
            "region_applied": False,
            "time_period_applied": True,
            "window_start": since.isoformat(),
        }
        ignored = [
            f"{name} {value!r}" for name, value in (("brand", brand), ("region", region)) if value
        ]
        if ignored:
            # #1718: distinguish column EXISTENCE from filter APPLICATION. The
            # earlier "the triggers table has no brand or region columns"
            # claim was false for brand — rows DO carry a brand_id column
            # (mixed values across brands); only region is absent from the row
            # schema. An answer quoting the note verbatim would inherit the
            # false schema claim, and it suppressed legitimate per-brand
            # tallies the rows support. What is true (and stated below): the
            # scopes were NOT applied as filters (while the #1727 time window
            # WAS).
            verb = "was" if len(ignored) == 1 else "were"
            noun = "a filter" if len(ignored) == 1 else "filters"
            response["scope_note"] = (
                f"{' and '.join(ignored)} {verb} NOT applied as {noun} — these "
                "results are cross-brand and cross-region (rows carry a "
                "brand_id tag, but region does not exist in this table). The "
                f"requested time window WAS applied (triggers since "
                f"{since.date().isoformat()}). Do not present them as "
                "specific to any brand or region."
            )
        return response
    except Exception as e:
        logger.error(f"Triggers query failed: {e}")
        return {"success": False, "error": str(e), "query_type": "triggers"}


async def _query_via_rag(
    query_type: str,
    query: str,
    filters: Optional[Dict[str, Any]],
    limit: int,
) -> Dict[str, Any]:
    """Fallback RAG query for experiments, predictions, recommendations, drift_reports."""
    try:
        results = await hybrid_search(
            query=f"{query_type}: {query}",
            k=limit,
            filters=filters,
        )

        return {
            "success": True,
            "query_type": query_type,
            "count": len(results),
            "data": [
                {
                    "source_id": r.source_id,
                    "content": r.content,
                    "score": r.score,
                    "source": r.source,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "retrieval_method": "hybrid_rag",
        }
    except Exception as e:
        logger.error(f"RAG query for {query_type} failed: {e}")
        return {"success": False, "error": str(e), "query_type": query_type}


# =============================================================================
# LANGGRAPH TOOLS
# =============================================================================


@tool(args_schema=E2IDataQueryInput)
async def e2i_data_query_tool(
    query_type: E2IQueryType,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    kpi_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    time_range: TimeRange = TimeRange.LAST_30_DAYS,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query E2I analytics data across multiple data types.

    This tool provides unified access to ALL E2I analytics data including:
    - KPIs: STORED snapshot rows from business_metrics. For a COMPUTED KPI VALUE
      for a brand (e.g. "what is the NBRx/NRx/TRx/market share for Kisqali?"),
      prefer ``kpi_calculate_tool`` — it resolves the KPI definition and calculates
      from the real substrate, whereas this returns the raw stored rows (and 0 for
      a derived KPI like NBRx that is not materialized here).
    - Causal chains: Discovered cause-effect relationships
    - Agent analyses: Outputs from the 22-agent system
    - Triggers: Alerts and explanations for metric changes
    - Experiments: A/B test designs and results
    - Predictions: ML model predictions
    - Recommendations: Generated recommendations
    - Drift reports: Data/model drift detection results

    Use this tool when users ask about E2I business metrics, causal relationships,
    agent outputs, or any analytical data from the platform.

    Args:
        query_type: Type of data to query (kpi, causal_chain, agent_analysis, etc.)
        brand: Optional brand filter, resolved case-insensitively against the actual data values
        region: Optional region filter, resolved case-insensitively against the actual data values
        kpi_name: Specific KPI name for KPI/causal queries
        agent_name: Agent name filter for agent_analysis queries (brand and
            time_range also apply there — brand resolves through the
            ``analysis_results->>'brand'`` JSONB field, #1355)
        time_range: Time range for the query (last_7_days, last_30_days, etc.)
        limit: Maximum results (1-100)
        filters: Additional key-value filters

    Returns:
        Dict with success status, query results, and metadata
    """
    logger.info(f"E2I data query: type={query_type}, brand={brand}, kpi={kpi_name}")

    since = _get_time_filter(time_range)

    if query_type == E2IQueryType.KPI:
        return await _query_kpis(brand, region, kpi_name, since, limit)

    elif query_type == E2IQueryType.CAUSAL_CHAIN:
        return await _query_causal_chains(brand, kpi_name, since, limit)

    elif query_type == E2IQueryType.AGENT_ANALYSIS:
        return await _query_agent_analysis(agent_name, brand, since, limit)

    elif query_type == E2IQueryType.TRIGGERS:
        return await _query_triggers(brand, region, since, limit)

    else:
        # Use RAG for experiments, predictions, recommendations, drift_reports
        query_str = f"{brand or ''} {region or ''} {kpi_name or ''}".strip() or "recent"
        combined_filters = filters or {}
        if brand:
            combined_filters["brand"] = brand
        return await _query_via_rag(query_type.value, query_str, combined_filters, limit)


@tool(args_schema=CausalAnalysisInput)
async def causal_analysis_tool(
    kpi_name: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    time_period: Optional[str] = "last_30_days",
    min_confidence: float = 0.7,
) -> Dict[str, Any]:
    """
    Find modeled causal drivers for an outcome in the causal-path registry.

    Queries the ``causal_paths`` registry — method-attributed causal
    relationships (DoWhy/EconML style) with REAL 0-1 ``confidence_level``
    values — for chains whose cause/effect nodes match the requested outcome.
    (2026-07-07 rewire: the previous implementation filtered RAG rank-fusion
    scores, ceiling ~0.03, against this 0-1 threshold — it could never return
    a chain for any query.)

    SUBSTRATE COVERAGE — read this before answering: the registry models
    patient-journey outcomes (treatment_initiated, persistent_180d,
    conversion_flag …) AND, since the 2026-07-07 commercial grain, the core
    commercial KPIs — TRx / NRx / NBRx / TRx market share / ROI /
    intent-to-prescribe (curated synthetic chains, surfaced provenance-labeled).
    When the response carries ``causal_chains_found: 0`` with
    ``substrate_coverage``, tell the user plainly that the causal registry
    does not cover that KPI, and offer the ``outcomes_covered`` it does model
    — do NOT imply an analysis ran and found nothing above the confidence
    threshold, and do NOT dress other tools' correlational data up as causal
    drivers.

    Args:
        kpi_name: Outcome to analyze; matched (tokenized, case-insensitive)
            against the registry's cause/effect node names.
        brand: Brand filter, matched case-insensitively.
        region: Echoed back for context; the registry has no region dimension,
            so it is NOT a filter. #1694: the response says so explicitly
            (``region_applied: False`` plus a ``scope_note`` when a region was
            requested) — measured live, region='Northeast' and region=null
            return byte-identical path sets, and without the disclaimer the
            synthesis layer presented the echo as an applied filter.
        time_period: Echoed back for context; registry paths are current
            modeled knowledge, not dated events, so it is NOT a filter
            (``time_period_applied: False`` in the response).
        min_confidence: Minimum ``confidence_level`` (0-1).

    Returns:
        Dict with success, causal_chains_found, results (cause/effect/via/
        effect_size/confidence/method), data_source provenance label, and —
        when the registry doesn't cover the outcome — substrate_coverage.
    """
    logger.info(f"Causal analysis: kpi={kpi_name}, brand={brand}, confidence>={min_confidence}")

    try:
        client = await get_async_supabase_client()
        repo = CausalPathRepository(client)
        # Provenance (#893): same platform gate as the KPI tools — synthetic
        # paths surface only in showcase mode, labeled; real mode fails closed.
        include_synthetic = kpi_include_synthetic()
        data_source = "synthetic" if include_synthetic else "database"

        paths = await repo.search_paths_for_outcome(
            kpi_name,
            brand=brand,
            min_confidence=min_confidence,
            # #1716: limit counts DISTINCT (cause, outcome, brand) paths —
            # the repository dedups duplicate registry rows BEFORE the cap.
            # 25 (was 15): measured 2026-08-19, 'treatment_initiated' matches
            # 24 distinct driver questions and the directly relevant
            # trigger_accepted -> treatment_initiated path (0.892) ranked
            # 17th by confidence, so a 15-distinct cap still crowded out the
            # answer to eval turn 4.7.
            limit=25,
            include_synthetic=include_synthetic,
        )

        # #1352: attach validation provenance (pinned validation_status +
        # refutation-evidence summary) so the answer can cite whether each
        # chain actually passed refutation testing (the q07 gap).
        summaries = await _fetch_refutation_summaries(client, paths)

        response: Dict[str, Any] = {
            "success": True,
            "kpi_analyzed": kpi_name,
            "brand": brand,
            "region": region,
            "causal_chains_found": len(paths),
            "min_confidence_applied": min_confidence,
            # #1694: the registry has no region/time dimension, so the echoed
            # region/time_period above are NOT filters — measured live, the
            # same path_ids come back for region='Northeast' and region=null.
            # Without these fields the synthesis layer reads the echo as an
            # applied filter and fabricates regional scoping in prose.
            "region_applied": False,
            "time_period_applied": False,
            "results": [
                _format_causal_path(
                    p, refutation_evidence=_refutation_evidence_entry(p.get("path_id"), summaries)
                )
                for p in paths
            ],
            "analysis_type": "causal_paths_registry",
            "data_source": data_source,
        }
        if region:
            response["scope_note"] = (
                f"The causal-path registry has no region dimension — region {region!r} "
                "was NOT applied as a filter; these results are brand-level. Do not "
                "present them as specific to that region."
            )
        if not paths:
            outcomes = await repo.get_distinct_outcomes(include_synthetic=include_synthetic)
            response["substrate_coverage"] = {
                "outcomes_covered": outcomes,
                "note": (
                    f"The causal-path registry does not cover '{kpi_name}' — see "
                    "outcomes_covered for what it does model (patient-journey and "
                    "commercial-KPI outcomes). This is a substrate coverage gap — "
                    "it is NOT evidence that no causal drivers exist for this KPI."
                ),
            }
        return response

    except Exception as e:
        logger.error(f"Causal analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "kpi_analyzed": kpi_name,
        }


@tool(args_schema=AgentRoutingInput)
async def agent_routing_tool(
    query: str,
    target_agent: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Route a query to the appropriate E2I agent tier using DSPy.

    The E2I system has 22 agents organized in 6 tiers:
    - Tier 0: ML Foundation (scope_definer, data_preparer, feature_analyzer, etc.)
    - Tier 1: Orchestration (orchestrator, tool_composer)
    - Tier 2: Causal Analytics (causal_impact, gap_analyzer, heterogeneous_optimizer)
    - Tier 3: Monitoring (drift_monitor, experiment_designer, experiment_monitor, health_score)
    - Tier 4: Predictions (prediction_synthesizer, resource_optimizer)
    - Tier 5: Learning (explainer, feedback_learner)

    Uses DSPy-based routing with intelligent agent selection and fallback
    to keyword matching when DSPy is unavailable.

    Args:
        query: The user's query to route
        target_agent: Specific agent to route to (if known)
        context: Additional context for routing decision (intent, brand_context)

    Returns:
        Dict with routing decision, confidence, rationale, and agent recommendation
    """
    logger.info(f"Agent routing: query={redact_query(query)}, target={target_agent}")

    # Initialize Opik tracing if available
    opik_span = None
    if OPIK_AVAILABLE:
        try:
            opik = OpikConnector()
            opik_span = opik.start_span(  # type: ignore[attr-defined]
                name="agent_routing",
                metadata={
                    "query_preview": redact_query(query, max_len=100),
                    "target_agent": target_agent,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to start Opik span: {e}")

    try:
        # If target agent specified, validate and return
        if target_agent:
            if target_agent in VALID_AGENTS:
                result = {
                    "success": True,
                    "routed_to": target_agent,
                    "secondary_agents": [],
                    "routing_confidence": 1.0,
                    "rationale": "Explicit agent selection",
                    "routing_method": "explicit",
                    "query_analyzed": redact_query(query, max_len=100),
                }
                # Log to Opik
                if opik_span:
                    opik_span.set_metadata(
                        {
                            "routed_to": target_agent,
                            "routing_confidence": 1.0,
                            "routing_method": "explicit",
                        }
                    )
                return result
            else:
                return {
                    "success": False,
                    "error": f"Unknown agent: {target_agent}",
                    "available_agents": list(VALID_AGENTS),
                }

        # Extract context values
        intent = ""
        brand_context = ""
        if context:
            intent = context.get("intent", "")
            brand_context = context.get("brand_context", "") or context.get("brand", "")

        # Use DSPy routing with fallback to hardcoded
        (
            primary_agent,
            secondary_agents,
            confidence,
            rationale,
            routing_method,
        ) = await route_agent_dspy(
            query=query,
            intent=intent,
            brand_context=brand_context,
            collect_signal=True,
        )

        result = {
            "success": True,
            "routed_to": primary_agent,
            "secondary_agents": secondary_agents,
            "routing_confidence": confidence,
            "rationale": rationale,
            "routing_method": routing_method,
            "dspy_enabled": CHATBOT_DSPY_ROUTING_ENABLED,
            "query_analyzed": redact_query(query, max_len=100),
        }

        # Log routing decision to Opik
        if opik_span:
            opik_span.set_metadata(
                {
                    "routed_to": primary_agent,
                    "secondary_agents": secondary_agents,
                    "routing_confidence": confidence,
                    "routing_method": routing_method,
                    "intent": intent,
                    "brand_context": brand_context,
                }
            )

        return result

    except Exception as e:
        logger.error(f"Agent routing failed: {e}")
        # Fallback to hardcoded routing on error
        try:
            primary_agent, secondary_agents, confidence, rationale = route_agent_hardcoded(
                query, intent="" if not context else context.get("intent", "")
            )
            return {
                "success": True,
                "routed_to": primary_agent,
                "secondary_agents": secondary_agents,
                "routing_confidence": confidence,
                "rationale": rationale,
                "routing_method": "hardcoded_fallback",
                "fallback_reason": str(e),
                "query_analyzed": redact_query(query, max_len=100),
            }
        except Exception as fallback_error:
            return {
                "success": False,
                "error": str(e),
                "fallback_error": str(fallback_error),
                "query_analyzed": redact_query(query, max_len=100),
            }
    finally:
        # End Opik span
        if opik_span:
            try:
                opik_span.end()
            except Exception:
                pass


@tool(args_schema=ConversationMemoryInput)
async def conversation_memory_tool(
    session_id: str,
    message_count: int = 10,
    include_tool_calls: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve conversation history from a chat session.

    This tool provides access to:
    - Recent messages in a conversation
    - Tool calls and results
    - Agent attributions
    - RAG context used

    Use this tool to provide context-aware responses based on
    previous conversation turns.

    Args:
        session_id: Session ID to retrieve history for
        message_count: Number of recent messages (1-50)
        include_tool_calls: Whether to include tool call details

    Returns:
        Dict with conversation history and metadata
    """
    logger.info(f"Conversation memory: session={session_id}, count={message_count}")

    try:
        client = await get_async_supabase_client()
        msg_repo = get_chatbot_message_repository(client)
        conv_repo = get_chatbot_conversation_repository(client)

        # Get conversation metadata
        conversation = await conv_repo.get_by_session_id(session_id)
        if not conversation:
            return {
                "success": False,
                "error": "Conversation not found",
                "session_id": session_id,
            }

        # Get recent messages
        messages = await msg_repo.get_recent_messages(session_id, count=message_count)

        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted = {
                "role": msg.get("role"),
                "content": msg.get("content"),
                "created_at": msg.get("created_at"),
                "agent_name": msg.get("agent_name"),
            }
            if include_tool_calls:
                formatted["tool_calls"] = msg.get("tool_calls", [])
                formatted["tool_results"] = msg.get("tool_results", [])
            formatted_messages.append(formatted)

        return {
            "success": True,
            "session_id": session_id,
            "conversation_title": conversation.get("title"),
            "brand_context": conversation.get("brand_context"),
            "region_context": conversation.get("region_context"),
            "message_count": len(formatted_messages),
            "messages": formatted_messages,
        }

    except Exception as e:
        logger.error(f"Conversation memory retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id,
        }


@tool(args_schema=DocumentRetrievalInput)
async def document_retrieval_tool(
    query: str,
    k: int = 5,
    brand: Optional[str] = None,
    kpi_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve relevant documents using hybrid RAG search.

    This tool performs hybrid retrieval combining:
    - Dense vector search (semantic similarity)
    - Sparse BM25 search (keyword matching)
    - Graph traversal (causal relationships)

    Use this tool when users need information from the knowledge base
    about E2I analytics, procedures, or historical data.

    Args:
        query: Search query for document retrieval
        k: Number of documents to retrieve (1-20)
        brand: Optional brand filter
        kpi_name: Optional KPI name for targeted retrieval

    Returns:
        Dict with retrieved documents and relevance scores
    """
    logger.info(f"Document retrieval: query={redact_query(query)}, k={k}, brand={brand}")

    try:
        filters = {}
        if brand:
            filters["brand"] = brand

        results = await hybrid_search(
            query=query,
            k=k,
            kpi_name=kpi_name,
            filters=filters if filters else None,
        )

        documents = [
            {
                "source_id": r.source_id,
                "content": r.content,
                "relevance_score": r.score,
                "source": r.source,
                "retrieval_method": r.retrieval_method,
                "metadata": r.metadata,
            }
            for r in results
        ]

        return {
            "success": True,
            "query": query,
            "document_count": len(documents),
            "documents": documents,
            "filters_applied": {"brand": brand, "kpi_name": kpi_name},
        }

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }


def _project_failure_details(failure_details: Any) -> List[Dict[str, Any]]:
    """Trim per-agent failure metadata to what chat surfaces may see (#1549).

    ``_build_output``'s failure_details entries carry the dispatcher's raw
    internal ``error`` audit strings (plus ``latency_ms``). The AG-UI
    synthesis prompt serializes tool payloads without projection or
    redaction, and on the partial_success path ``response_text`` does NOT
    already narrate the errors — so a passthrough would put raw internals in
    front of the synthesizer that this surface never exposed before. Mirror
    of /chat's surfaced-``user_action`` pattern (``chat_bridge`` #1451): keep
    the agent name, the dispatcher-authored user-facing next step, and a
    coarse category DERIVED deterministically from the real error — never
    the raw string. Unexpected shapes degrade to omission, not a raise
    (orchestrator-supplied data, same tolerance as chat_bridge).
    """
    projected: List[Dict[str, Any]] = []
    for detail in failure_details or []:
        if not isinstance(detail, dict):
            continue
        error_text = str(detail.get("error") or "")
        low = error_text.lower()
        if "needs structured inputs" in low or "missing:" in low:
            reason = "missing_required_inputs"
        elif "timeout" in low or "timed out" in low:
            reason = "timeout"
        elif error_text:
            reason = "agent_error"
        else:
            reason = "unknown"
        entry: Dict[str, Any] = {
            "agent_name": detail.get("agent_name"),
            "reason": reason,
        }
        user_action = detail.get("user_action")
        if isinstance(user_action, str) and user_action.strip():
            entry["user_action"] = user_action.strip()
        projected.append(entry)
    return projected


@tool(args_schema=OrchestratorToolInput)
async def orchestrator_tool(
    query: str,
    target_agent: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a query through the E2I orchestrator and 22-agent system.

    This tool provides access to the full E2I multi-agent architecture:
    - Tier 0: ML Foundation (data prep, feature analysis, model training)
    - Tier 1: Orchestration (orchestrator, tool_composer)
    - Tier 2: Causal Analytics (causal_impact, gap_analyzer, heterogeneous_optimizer)
    - Tier 3: Monitoring (drift_monitor, experiment_designer, experiment_monitor, health_score)
    - Tier 4: Predictions (prediction_synthesizer, resource_optimizer)
    - Tier 5: Learning (explainer, feedback_learner)

    Use this tool for:
    - Complex causal analysis requiring the causal_impact agent
    - Experiment design through experiment_designer agent
    - Drift detection and model health checks
    - Multi-agent orchestrated queries
    - Any query that benefits from the full agent pipeline

    This tool routes through the real orchestrator, NOT just keyword matching.

    Args:
        query: The query to process through the orchestrator
        target_agent: Optional specific agent to route to
        brand: Brand context, resolved case-insensitively against the actual data values
        region: Region context, resolved case-insensitively against the actual data values
        session_id: Session ID for context continuity

    Returns:
        Dict with orchestrator response, agents dispatched, and confidence
    """
    logger.info(f"Orchestrator tool: query={redact_query(query)}, target_agent={target_agent}")

    try:
        orchestrator = get_orchestrator()

        if orchestrator is None:
            logger.warning("Orchestrator unavailable, using fallback")
            # Fallback to hybrid search when orchestrator unavailable
            fallback_results = await hybrid_search(
                query=query,
                k=10,
                filters={"brand": brand} if brand else None,
            )
            return {
                "success": True,
                "fallback": True,
                "reason": "Orchestrator unavailable - using RAG fallback",
                "query": query,
                "result_count": len(fallback_results),
                "results": [
                    {
                        "content": r.content,
                        "score": r.score,
                        "source": r.source,
                    }
                    for r in fallback_results[:5]
                ],
            }

        # Build user context for orchestrator
        user_context = {}
        if brand:
            user_context["brand"] = brand
        if region:
            user_context["region"] = region
        if target_agent:
            user_context["target_agent"] = target_agent

        # #1698: thread the user's original ask alongside the model's rewrite.
        # The cohort accounting parses BOTH texts, so a criterion the rewrite
        # dropped is still bound as a filter or honestly reported as
        # criteria_not_applied — never silently lost.
        raw_user_query = _raw_user_query_context.get()
        if raw_user_query and raw_user_query != query:
            user_context["raw_user_query"] = raw_user_query

        # Generate session_id if not provided
        effective_session_id = session_id or f"chatbot-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Call the orchestrator
        orchestrator_result = await orchestrator.run(
            {
                "query": query,
                "session_id": effective_session_id,
                "user_context": user_context,
            }
        )

        # Extract key fields from orchestrator response
        response_text = orchestrator_result.get("response_text", "")
        response_confidence = orchestrator_result.get("response_confidence", 0.85)
        agents_dispatched = orchestrator_result.get("agents_dispatched", [])

        # #1549 truthful envelope: success must reflect the run's REAL status.
        # (failure_details are projected via _project_failure_details — see
        # its docstring for the leak-surface rationale.)
        # _build_output always emits "status" ("completed" / "partial_success"
        # / "failed"); a fail-closed run (zero successful agents — the
        # synthesizer's "Please try again or rephrase your question."
        # abstention) was previously re-promoted to a hardcoded success=True,
        # so the AG-UI synthesis prompt presented the ask-back as grounded
        # evidence and tool_evidence/_grade_copilot_turn rewarded it (#1257
        # rule keys on ``success``). A missing "status" (non-orchestrator-
        # shaped dicts) keeps the old lenient contract. partial_success stays
        # success=True — it carries real evidence from the successful agents —
        # with the failure metadata propagated so the synthesizer can caveat.
        #
        # NOTE: no "analysis_results" key. _build_output never emits one, so
        # the old `orchestrator_result.get("analysis_results", {})` was ALWAYS
        # {} — silent evidence-loss noise with zero consumers repo-wide
        # (verified 2026-08-12: src/ hits are the agent_activities JSONB
        # column, frontend/ has none, tests only fabricated it in mocks).
        status = orchestrator_result.get("status")
        run_failed = status == "failed"
        failed_agents = orchestrator_result.get("failed_agents") or []
        failure_details = _project_failure_details(orchestrator_result.get("failure_details"))

        payload: Dict[str, Any] = {
            "success": not run_failed,
            "fallback": False,
            "query": query,
            # The honest response text is ALWAYS preserved — on a failed run
            # it carries the synthesizer's fail-closed abstention, which must
            # still reach the user via the synthesis prompt.
            "response": response_text,
            "confidence": response_confidence,
            "agents_dispatched": agents_dispatched,
            "target_agent_requested": target_agent,
            "context": {
                "brand": brand,
                "region": region,
                "session_id": effective_session_id,
            },
        }
        if status is not None:
            payload["status"] = status
        if orchestrator_result.get("has_partial_failure"):
            payload["has_partial_failure"] = True
        if failed_agents:
            payload["failed_agents"] = failed_agents
        if failure_details:
            payload["failure_details"] = failure_details
        return payload

    except Exception as e:
        logger.error(f"Orchestrator tool failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "fallback": True,
        }


def _resolve_cohort_frame(
    brand: Optional[str],
    region: Optional[str],
    data_source: Optional[str],
) -> Optional["pd.DataFrame"]:
    """Resolve a real cohort DataFrame for (brand, region).

    Delegates to the shared :func:`cohort_resolution.resolve_cohort_frame`
    service (issue #779), which:

    * with an explicit ``data_source`` -> uses the tier0
      ``CohortConstructorAgent`` loader (preserves the original R4 behavior);
    * WITHOUT a ``data_source`` -> resolves the canonical ``patient_journeys``
      table filtered by brand + ``geographic_region``.

    Returns the cohort frame on success, or ``None`` when nothing resolves.
    RAISES on genuine loader/infra failure so the caller can log-and-proceed
    (the composable tools then fail closed honestly -- never substituting
    fabricated data).
    """
    return cohort_resolution.resolve_cohort_frame(brand, region, data_source=data_source)


@tool(args_schema=ToolComposerToolInput)
async def tool_composer_tool(
    query: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    session_id: Optional[str] = None,
    max_parallel: int = 3,
    data_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process multi-faceted queries through the E2I Tool Composer.

    The Tool Composer handles complex queries that require:
    - Multiple sub-questions answered by different agents
    - Cross-agent analysis (e.g., causal + drift + predictions)
    - Comparison across brands, KPIs, or time periods
    - Analysis combined with recommendations

    Pipeline:
    1. DECOMPOSE: Break query into atomic sub-questions
    2. PLAN: Map sub-questions to tools and create execution DAG
    3. EXECUTE: Run tools in dependency order with parallel execution
    4. SYNTHESIZE: Aggregate results into coherent response

    Example queries:
    - "Compare TRx trends across all brands and explain the causal factors"
    - "Show me the health score and recommendations for Kisqali"
    - "What caused the NRx drop and should we run an experiment?"

    Args:
        query: Multi-faceted query requiring decomposition
        brand: Brand context, resolved case-insensitively against the actual data values
        region: Region context, resolved case-insensitively against the actual data values
        session_id: Session ID for context continuity
        max_parallel: Maximum parallel tool executions (1-5)

    Returns:
        Dict with synthesized response from multiple agent outputs
    """
    logger.info(f"Tool composer: query={redact_query(query)}, brand={brand}")

    try:
        # Build context for Tool Composer
        context: Dict[str, Any] = {
            "brand": brand,
            "region": region,
            "session_id": session_id or f"composer-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "max_parallel": max_parallel,
        }

        # Issue #810: KPI-aware data resolution. When the query targets a defined
        # KPI (e.g. "what drove <brand> conversion ..."), resolve the KPI's REAL
        # substrate (e.g. triggers ⋈ treatment_events for Conversion Rate) with
        # the KPI outcome materialized, rather than the patient-clinical cohort.
        # This takes precedence over the cohort frame for KPI-outcome questions.
        kpi_resolved = False
        try:
            kpi = kpi_resolution.recognize_kpi(query)
            if kpi is not None:
                kpi_frame = kpi_resolution.resolve_kpi_frame(kpi, brand, region)
                if kpi_frame is not None:
                    context["estimation_data"] = kpi_frame.frame
                    context["kpi_outcome"] = kpi_frame.outcome_column
                    context["kpi_name"] = kpi_frame.kpi_name
                    context["kpi_driver_columns"] = kpi_frame.driver_columns
                    kpi_resolved = True
                    logger.info(
                        "Tool composer: resolved KPI '%s' substrate (%d rows, "
                        "outcome=%s, truncated=%s) for brand=%s region=%s",
                        kpi_frame.kpi_name,
                        len(kpi_frame.frame),
                        kpi_frame.outcome_column,
                        kpi_frame.is_truncated,
                        brand,
                        region,
                    )
        except Exception as kpi_err:
            logger.warning(
                f"Tool composer: KPI resolution failed, falling back to cohort: {kpi_err}"
            )

        # Rec#1a: resolve a REAL cohort DataFrame for (brand, region) and thread
        # it into the composer context under the canonical "estimation_data" key.
        # The executor auto-injects it into composable tool kwargs. Best-effort:
        # if the loader is unavailable or raises, log and proceed -- the tools
        # then fail-closed honestly rather than fabricating values. Skipped when a
        # KPI substrate already resolved (the KPI frame is the right outcome data).
        if not kpi_resolved:
            try:
                estimation_frame = _resolve_cohort_frame(brand, region, data_source)
                if estimation_frame is not None:
                    context["estimation_data"] = estimation_frame
                    logger.info(
                        "Tool composer: resolved estimation_data frame "
                        f"({len(estimation_frame)} rows) for brand={brand} region={region}"
                    )
            except Exception as resolve_err:
                logger.warning(
                    f"Tool composer: cohort resolution failed, proceeding without "
                    f"estimation_data: {resolve_err}"
                )

        # #1557: do NOT pre-build a shared LLM client here. Injecting one puts
        # ToolComposer in dependency-injection mode ("every phase SHARES it
        # unchanged"), which bypassed the #1365 per-phase sizing: the plan
        # phase ran with adaptive thinking ON against the shared 4096 budget,
        # thinking tokens ate the budget, and the planner JSON truncated at the
        # ceiling (PlanningError, eval turn 2.6). With no client injected the
        # composer builds a correctly-SIZED client per phase from the factory
        # (plan -> thinking disabled, truncation impossible) — the same
        # deliberate choice ToolComposerAgent._get_composer already makes. The
        # tiers are equivalent anyway: "reasoning" and "standard" map to the
        # same model for both providers, so nothing here depended on the tier.
        result = await compose_query(query=query, context=context)

        # Extract composition results from Pydantic CompositionResult model
        # result.decomposition contains sub_questions, result.plan has execution info,
        # result.execution has tool outputs, result.response has synthesized answer
        return {
            # F6 fail-closed: surface the REAL composition outcome. A total tool
            # failure (0/N succeeded) yields result.success=False / status=FAILED;
            # do NOT re-promote it to a hardcoded success envelope. The honest
            # synthesized_response + confidence (0.0) already flow through below.
            "success": result.success,
            "status": result.status.value,
            "query": query,
            "sub_questions": [
                {"id": sq.id, "question": sq.question, "intent": sq.intent}
                for sq in result.decomposition.sub_questions
            ],
            "tools_executed": result.execution.tools_executed,
            "execution_order": result.plan.get_execution_order(),
            "parallel_groups": result.plan.parallel_groups,
            "synthesized_response": result.response.answer,
            "confidence": result.response.confidence,
            "agent_outputs": result.execution.get_all_outputs(),
            "context": {
                "brand": brand,
                "region": region,
                "session_id": context["session_id"],
            },
        }

    except Exception as e:
        logger.error(f"Tool composer failed: {e}")
        # Fallback to orchestrator for simpler processing
        try:
            logger.info("Tool composer fallback: attempting orchestrator")
            orchestrator = get_orchestrator()
            if orchestrator:
                fallback_result = await orchestrator.run(
                    {
                        "query": query,
                        "session_id": session_id
                        or f"fallback-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "user_context": {"brand": brand, "region": region},
                    }
                )
                # #1549: mirror orchestrator_tool's truthful envelope — the
                # fallback of the fallback can itself fail closed, and a
                # hardcoded success=True here counted that abstention as
                # grounded evidence downstream. The honest response text is
                # preserved either way.
                fb_status = fallback_result.get("status")
                fb_payload: Dict[str, Any] = {
                    "success": fb_status != "failed",
                    "fallback": True,
                    "fallback_reason": f"Tool composer error: {str(e)}",
                    "query": query,
                    "response": fallback_result.get("response_text", ""),
                    "confidence": fallback_result.get("response_confidence", 0.7),
                }
                if fb_status is not None:
                    fb_payload["status"] = fb_status
                if fallback_result.get("has_partial_failure"):
                    fb_payload["has_partial_failure"] = True
                fb_failed_agents = fallback_result.get("failed_agents") or []
                fb_failure_details = _project_failure_details(
                    fallback_result.get("failure_details")
                )
                if fb_failed_agents:
                    fb_payload["failed_agents"] = fb_failed_agents
                if fb_failure_details:
                    fb_payload["failure_details"] = fb_failure_details
                return fb_payload
        except Exception as fallback_error:
            logger.error(f"Orchestrator fallback also failed: {fallback_error}")

        return {
            "success": False,
            "error": str(e),
            "query": query,
        }


# =============================================================================
# TOOL EXPORTS
# =============================================================================


# =============================================================================
# KPI ENGINE TOOL — compute a DEFINED KPI on demand (the registry's calculable KPIs)
# =============================================================================


class KpiCalculateInput(BaseModel):
    """Input schema for kpi_calculate_tool."""

    kpi_name: str = Field(
        description=(
            "The KPI to compute, e.g. 'NBRx' (new-to-brand Rx), 'TRx', 'NRx', "
            "'market share', 'conversion rate', 'ROI', 'HCP coverage'. Resolved "
            "against the defined KPI registry."
        )
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand filter (e.g. Remibrutinib, Fabhalta, Kisqali), case-insensitive.",
    )
    region: Optional[str] = Field(
        default=None,
        description=(
            "Optional geographic region filter. US census regions: northeast, "
            "south, midwest, west (case-insensitive; synonyms like 'north "
            "east', 'NE', 'new england', 'pacific', 'west coast' and natural "
            "phrasings like 'the Northeast region' resolve to a label; any "
            "other value errors honestly with a clarify hint). Ambiguous "
            "phrasings ('East', 'East Coast') span more than one census "
            "region and will NOT resolve — ask the user which census region "
            "they mean instead of guessing. The response's region_status tells "
            "you whether the figure is region-scoped: only 'applied' means it "
            "is; 'not_applicable' means this KPI has no region variant and "
            "the value is global — NEVER present it as region-specific."
        ),
    )
    segment: Optional[str] = Field(
        default=None,
        description=(
            "Optional severity tier filter: one of low_severity, "
            "medium_severity, high_severity. Mutually exclusive with "
            "region/therapy_line."
        ),
    )
    therapy_line: Optional[str] = Field(
        default=None,
        description=(
            "Optional line-of-therapy filter: one of '0', '1', '2', '3'. "
            "Mutually exclusive with region/segment."
        ),
    )
    biologic: Optional[str] = Field(
        default=None,
        description=(
            "Optional biologic-status filter: 'naive' or 'experienced'. "
            "AVAILABLE FOR REMIBRUTINIB ONLY -- for any other brand the tool "
            "returns an error (the data is 100% NULL by design); do NOT retry "
            "or fabricate a split. Mutually exclusive with "
            "region/segment/therapy_line/ige_tier."
        ),
    )
    ige_tier: Optional[str] = Field(
        default=None,
        description=(
            "Optional IgE-tertile filter: 'low', 'medium', or 'high' "
            "(data-driven tertiles, not a clinical threshold). AVAILABLE FOR "
            "REMIBRUTINIB ONLY -- other brands return an error; do NOT fabricate. "
            "Mutually exclusive with region/segment/therapy_line/biologic."
        ),
    )
    trigger_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional trigger-type filter for the TRIGGER-EFFECTIVENESS KPIs "
            "ONLY (trigger precision, acceptance rate, override rate, trigger "
            "funnel conversion -- #1360). Live values include "
            "prescription_opportunity, engagement_gap, adherence_risk, "
            "cross_sell, competitive_threat, churn_prevention, reactivation, "
            "treatment_switch. Any other KPI returns an error -- the filter "
            "is never silently dropped."
        ),
    )
    window: Optional[str] = Field(
        default=None,
        description=(
            "Time window, e.g. 'last 3 months', 'last year', 'Q1 2025', or "
            "'2025-01-01 to 2025-03-31'. Supported for TRx/NRx/NBRx, TRx "
            "share, conversion rate (alone or combined with segment/"
            "therapy_line), and the trigger-effectiveness KPIs (alone or "
            "combined with brand/trigger_type; NOT combinable with region -- "
            "the tool errors honestly). ALWAYS pass this when the user names "
            "a period. Omit for the engine's default window (the most recent "
            "30 days of available data)."
        ),
    )


# Reporting window per KPI id, MIRRORED from the vetted SQL in the kpi_query
# allowlist registry (database/migrations/044/066, re-anchored by 089).
# The WS3 KPIs count over a 30-day window that ends at the DATA FRONTIER
# (`MAX(<domain ts>)` -- migration 089), NOT wall-clock NOW(): the synthetic
# gold-standard substrate is calendar-fixed by design, so a NOW()-anchored
# window silently decays to an empty set (the 2026-07-03 "NBRx = 0.0"
# incident). The engine surfaces the concrete as-of date per answer as
# `data_through`; this static note names the window SEMANTICS and the domain
# the frontier belongs to. Surfacing the real window lets the chatbot CITE it
# instead of presenting the figure as "the last 30 calendar days". Only KPIs
# whose window we have verified against the registry are listed; for any other
# KPI the field is omitted (honest absence over a guessed period -- ROI stays
# out: its two source probes' frontiers diverge). KEEP IN SYNC with those
# migrations.
KPI_REPORTING_WINDOWS = {
    "WS3-BI-005": "most recent 30 days of prescription data",  # TRx
    "WS3-BI-006": "most recent 30 days of prescription data",  # NRx
    "WS3-BI-007": "most recent 30 days of prescription data",  # NBRx
    "WS3-BI-008": "most recent 30 days of prescription data",  # TRx share
    "WS3-BI-009": "most recent 30 days of trigger data",  # Conversion rate
    # #1360 trigger-effectiveness family (migrations 089/113/118). Precision's
    # default cohort is LAGGED so the 30-day conversion window has matured --
    # describing it as a plain trailing window would misstate the figure.
    "WS2-TR-001": (
        "30-day trigger cohort ending 30 days before the trigger-data "
        "frontier (the conversion window must mature)"
    ),
    "WS2-TR-004": "most recent 30 days of trigger data",  # Acceptance rate
    # #1713: verified against migration 089 (same frontier-anchored 30-day
    # trigger window as WS2-TR-006). Omitting it while its sibling disclosed a
    # window was the substrate of the eval-certified borrowing defect
    # (Override Rate's window label asserted in prose for False Alert Rate).
    "WS2-TR-005": "most recent 30 days of trigger data",  # False alert rate
    "WS2-TR-006": "most recent 30 days of trigger data",  # Override rate
    "WS2-TR-009": "most recent 30 days of trigger data",  # Funnel conversion
}

# Definition clarifications the synthesizer MUST carry into the answer.
# SSOT moved to src/services/kpi_resolution.py (#1475): the orchestrator's
# explainer resolver binds the same notes, and importing THIS module costs ~30s
# (orchestrator/tool_composer/RAG stacks) — unaffordable in a sync resolver.
# Re-exported here so every existing consumer keeps working unchanged.
from src.services.kpi_resolution import KPI_SEMANTIC_NOTES  # noqa: E402


def _kpi_result_to_response(
    kpi: Any, result: Any, *, brand: Optional[str] = None, region: Optional[str] = None
) -> Dict[str, Any]:
    """Map a ``KPIResult`` onto the chatbot tool response (pure; unit-tested, no DB).

    Surfaces ``data_source='synthetic'`` when the engine answered from the
    synthetic-gold substrate so the chatbot/FE badges the figure honestly rather
    than passing it off as real-world data.

    Echoes the ``brand`` the figure was computed for so the synthesizer can
    name it instead of re-asking. ``region`` is echoed from the engine's REGION
    PROVENANCE (#1538): it carries the region ONLY when ``region_status ==
    "applied"`` (a region-scoped query variant computed the value); when the
    engine reports ``not_applicable`` the value is global — ``region`` is None
    and ``region_note`` says so explicitly. Copies the engine's window provenance
    (``window_requested``/``window_applied``/``window_status``) so the chatbot can
    state exactly which period the figure covers. The static ``reporting_window``
    note is included ONLY when ``window_status == 'default'`` (no custom window was
    applied); when a custom window was honored or was not applicable, the stale
    default-window note would contradict the real answer.

    ``data_through`` (migration 089): the frontier-anchored default windows end
    at the domain's latest data date, not wall-clock now -- the substrate is
    calendar-fixed by design. When the engine surfaced that as-of date (stashed
    into metadata context by the calculator), copy it up so the chatbot cites
    e.g. "30 days ending 2025-04-23" instead of implying recency. Honest
    absence when the engine did not report one.
    """
    if getattr(result, "error", None):
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "kpi_id": kpi.id,
            "kpi_name": kpi.name,
            "error": result.error,
        }
    metadata = getattr(result, "metadata", None) or {}
    include_synthetic = bool(metadata.get("include_synthetic"))
    window_status = getattr(result, "window_status", "default")
    # Region echo is PROVENANCE-BASED (#1538), not the caller's argument: only
    # a fixed set of calculators route to region-scoped variants; echoing the
    # requested region beside a value the engine computed globally is the
    # #1534 defect class (a scope the response names but the SQL never
    # applied), and the synthesizer would caption the figure with it.
    region_status = getattr(result, "region_status", "default")
    region_applied = getattr(result, "region_applied", None)
    response: Dict[str, Any] = {
        "success": True,
        "query_type": "kpi_calculate",
        "kpi_id": kpi.id,
        "kpi_name": kpi.name,
        "value": result.value,
        "status": result.status,
        "data_source": "synthetic" if include_synthetic else "database",
        # #1640: what this number MEASURES, derived from the registry's own
        # `tables` declaration. Without it, an event-ledger count and a
        # business_metrics level both arrive labelled "TRx" and read as a
        # contradiction (or worse, as a reconciliation).
        "measure_basis": _measure_basis_for_kpi(kpi, metadata),
        "brand": brand,
        "region": region_applied if region_status == "applied" else None,
        "region_status": region_status,
        "window_requested": getattr(result, "window_requested", None),
        "window_applied": getattr(result, "window_applied", None),
        "window_status": window_status,
    }
    if region_status == "not_applicable":
        requested_region = getattr(result, "region_requested", None) or region
        response["region_note"] = (
            f"the region filter ({requested_region}) was NOT applied — "
            f"{kpi.name} has no region-scoped variant, so this value is "
            "global/portfolio-level. Do not present it as region-specific."
        )
    data_through = (metadata.get("context") or {}).get("data_through")
    if data_through is not None:
        response["data_through"] = data_through
    # #1360: WS2-TR-009 surfaces its stage counts (delivered -> viewed ->
    # accepted -> actioned -> outcome) so the synthesizer can narrate the whole
    # funnel, not just the headline rate. Absent for every other KPI.
    funnel_stages = (metadata.get("context") or {}).get("funnel_stages")
    if funnel_stages is not None:
        response["funnel_stages"] = funnel_stages
    # #1532: WS3-BI-010 surfaces the per-slice trailing-12-month
    # temporal-variability band (range of monthly ROI values — NOT a
    # confidence interval; #1527 established none is possible on the 30-day
    # headline). Honestly absent when the agent_activities fallback answered
    # or real-mode has zero slices. Gated on the KPI id: the band is an
    # ROI-only estimand, and a cached result whose metadata was polluted by
    # the pre-fix shared-context leak must never show an ROI band beside a
    # different KPI's figure (codex iter-1 finding 2).
    if kpi.id == "WS3-BI-010":
        temporal_band = (metadata.get("context") or {}).get("temporal_variability_band")
        if temporal_band is not None:
            response["temporal_variability_band"] = temporal_band
    if window_status == "default":
        window = KPI_REPORTING_WINDOWS.get(kpi.id)
        if window:
            response["reporting_window"] = window
    # #1713: direction glosses on `status` ("above/below threshold") are only
    # checkable when the payload names the metric's polarity — the 2026-08-19
    # eval wrote "flagged warning (below healthy threshold)" for WS2-TR-005,
    # which is lower-is-better (warning means ABOVE its threshold). Surface
    # the exact flag the calculator evaluated the status WITH (stashed in
    # KPIResult.metadata by the trigger/brand/model-performance calculators);
    # honest absence when the calculator did not report one — never a polarity
    # the evaluation didn't use.
    lower_is_better = metadata.get("lower_is_better")
    if lower_is_better is not None:
        response["direction"] = "lower_is_better" if lower_is_better else "higher_is_better"
    semantic_note = KPI_SEMANTIC_NOTES.get(kpi.id)
    if semantic_note:
        response["semantic_note"] = semantic_note
    return response


# Cumulative prescription-volume KPIs (event counts over a window). ONLY these
# get the trailing-30d coverage probe below: for a ratio/share KPI (e.g.
# WS3-BI-008 TRx Share) the trailing value is not additive, so the share math
# would fire false warnings.
_VOLUME_KPI_IDS = frozenset({"WS3-BI-005", "WS3-BI-006", "WS3-BI-007"})

_COVERAGE_MIN_WINDOW_DAYS = 45  # a window this short IS its own trailing period
_COVERAGE_WARN_FACTOR = 2.0  # warn when trailing share > 2x the uniform share


async def _window_coverage_probe(
    kpi: Any,
    result: Any,
    window: Any,
    calculator: Any,
    context: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Disclose intra-window data-density asymmetry for volume KPIs.

    2026-07-07 session review: a "90-day baseline" (15,767) was 96% composed of
    the same most-recent-30-days it was compared against (15,239), because the
    substrate is dense only in the recent window — the requested window was
    honestly "applied", but nothing disclosed the asymmetry, and the chatbot
    concluded a fabricated "softening". For cumulative volume KPIs with an
    applied window longer than 45 days, compute the same KPI over the window's
    trailing 30 days and report its share of the total; attach
    ``coverage_warning`` when that share exceeds 2x the uniform expectation.
    Probe failures degrade silently — the main figure must never be blocked or
    altered by the probe.

    An in-progress calendar window ("this quarter"/"this year" from #1546, but
    equally an explicit "Q3 2026", bare "2026", or a dict spec) carries a
    future ``window.end``; unclamped, the trailing sub-window lay entirely in
    the future, so share was 0 and the warning silently suppressed (PR #1554
    codex iter-1). Both the trailing window and the 45-day gate therefore use
    the ELAPSED span, clamped to ``now``.
    """
    try:
        if window is None or kpi.id not in _VOLUME_KPI_IDS:
            return None
        if getattr(result, "window_status", None) != "applied":
            return None
        window_value = getattr(result, "value", None)
        if not isinstance(window_value, (int, float)) or window_value <= 0:
            return None
        if now is None:
            now = datetime.now(timezone.utc)
        effective_end = min(window.end, now)
        window_days = (effective_end - window.start).days
        if window_days <= _COVERAGE_MIN_WINDOW_DAYS:
            return None

        trailing_window = {
            "start": (effective_end - timedelta(days=30)).isoformat(),
            "end": effective_end.isoformat(),
        }
        trailing_result = await asyncio.to_thread(
            calculator.calculate, kpi.id, context={**context, "window": trailing_window}
        )
        trailing_value = getattr(trailing_result, "value", None)
        if not isinstance(trailing_value, (int, float)):
            return None

        share = trailing_value / window_value
        expected = 30.0 / window_days
        coverage: Dict[str, Any] = {
            "window_days": window_days,
            "trailing_30d_value": trailing_value,
            "trailing_30d_share": round(share, 4),
            "uniform_expected_share": round(expected, 4),
        }
        if share > _COVERAGE_WARN_FACTOR * expected:
            coverage["coverage_warning"] = (
                f"{share:.0%} of this {window_days}-day total falls in its most recent "
                "30 days — the data is not evenly distributed across the window. Do NOT "
                "treat the full-window figure as a baseline for the recent period; "
                "compare against a prior non-overlapping window instead."
            )
        return coverage
    except Exception as exc:  # noqa: BLE001 - the probe must never break the main figure
        logger.warning("kpi_calculate_tool: window coverage probe failed: %s", exc)
        return None


# The four trigger-effectiveness KPIs the #1360 ruling assigned to the chat KPI
# path -- the ONLY KPIs whose calculator reads context['trigger_type'] (the
# migration-118 statement families). The guard below keeps the filter from
# silently dropping on any other KPI.
_TRIGGER_EFFECTIVENESS_KPI_IDS = frozenset({"WS2-TR-001", "WS2-TR-004", "WS2-TR-006", "WS2-TR-009"})

#: Coordinators that turn two KPI mentions into two SEPARATE asks (#1637). Two
#: mentions alone do not: "TRx market share" is a modifier chain naming exactly
#: one KPI (WS3-BI-008). Only an explicit coordinator between the mentions means
#: the caller asked for both.
#:
#: Includes COMPARISON forms (codex iter-2): "TRx vs NRx" is two metrics just as
#: much as "TRx and NRx" is, and comparison is the more natural phrasing for the
#: ask — omitting it left the original single-call failure intact for exactly the
#: shape most likely to produce it.
#: Asymmetric by design: a MISSED coordinator degrades to the behaviour that
#: shipped before this guard existed (one metric answered), while a FALSE
#: coordinator refuses a question the tool can actually answer. So the list is
#: grown deliberately and each addition is measured against the observed corpus.
#: Bare "with" is excluded for exactly that reason -- it is a preposition that
#: forms MODIFIER relationships, and "market share with respect to TRx" names one
#: KPI, not two. The explicitly additive compounds are safe.
#: Punctuation joiners are here too (codex iter-11). kpi_resolution normalizes
#: "-", ".", "–", "—" as SEPARATORS so a single label like "TRx-share" resolves;
#: left out of this list, the same characters BETWEEN two distinct metrics let
#: "TRx-NRx" and "TRx.NRx" be answered as one. Anything that separates words for
#: matching must also be readable as coordination between two metrics.
#:
#: Punctuation inside ONE recognized phrase is unaffected: the gap is measured
#: between distinct metric spans, and "TRx-share" matches a single alias covering
#: both words, so there is no second span and no gap to read.
_KPI_COORDINATOR_RE = re.compile(
    r"(?:\band\b|\bor\b|\bplus\b|\bas well as\b|\balongside\b"
    r"|\b(?:along|together) with\b|\bvs\.?\b|\bversus\b"
    r"|\bcompared (?:to|with)\b|\bagainst\b|&|,|/|\+|-|–|—|\.|_)"
)

#: Backstop on the multi-mention scan. The scan ends naturally once every mention
#: is masked; this only bounds the case-sensitive abbreviation branch, which reads
#: the UNMASKED original. Set well above any realistic ask, and hitting it is
#: logged rather than silently truncating the scan.
_MAX_KPI_MENTION_SCANS = 8


@tool(args_schema=KpiCalculateInput)
async def kpi_calculate_tool(
    kpi_name: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    segment: Optional[str] = None,
    therapy_line: Optional[str] = None,
    biologic: Optional[str] = None,
    ige_tier: Optional[str] = None,
    trigger_type: Optional[str] = None,
    window: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a DEFINED KPI on demand via the KPI engine (the registry's calculable KPIs).

    Use this for "what is the <KPI> for <brand>?" questions — NBRx (new-to-brand
    Rx), TRx, NRx, market share, conversion rate, ROI, HCP coverage, etc. Unlike
    ``e2i_data_query_tool`` (which reads the materialized ``business_metrics``
    fixture and returns 0 for a KPI that isn't stored there), this RESOLVES the KPI
    name to its definition and CALCULATES it from the real substrate (e.g. NBRx =
    count of each patient's first-brand prescription over ``treatment_events``).

    TRIGGER-EFFECTIVENESS KPIs (#1360): trigger precision, acceptance rate,
    override rate, and trigger funnel conversion are chat-KPI-path KPIs served
    here. They accept ``brand``, ``region``, ``trigger_type`` and ``window``
    filters — with ONE constraint: ``region`` does NOT compose with an explicit
    ``window`` (the tool errors honestly rather than silently dropping the
    region). WS2-TR-009 responses carry ``funnel_stages`` (delivered ->
    viewed -> accepted -> actioned -> outcome counts) alongside the headline.

    TIME WINDOW: pass ``window`` (e.g. "last 3 months", "last year", "Q1 2025",
    "2025-01-01 to 2025-03-31") to compute the volume KPIs (TRx/NRx/NBRx), TRx
    share, conversion rate, or the trigger-effectiveness KPIs over that period
    — ALWAYS pass it when the user names one. A window composes with the
    ``segment`` / ``therapy_line`` axes (e.g. per-tier conversion rate over the
    last year); it does NOT compose with region/biologic/ige_tier for share or
    conversion, nor with region for the trigger-effectiveness KPIs (the tool
    errors honestly). The engine reports back ``window_status`` ("applied" when the
    requested window was honored, "not_applicable" when the KPI has no time
    dimension, "default" when no window was requested), plus ``window_requested``
    and ``window_applied``. BASELINE COMPARISONS: to compare a recent period
    against a baseline, request a PRIOR NON-OVERLAPPING window of the same
    length (e.g. "2026-05-08 to 2026-06-07" as the baseline for "2026-06-07 to
    2026-07-07") — never a longer window that CONTAINS the recent period. For
    volume KPIs over windows longer than 45 days the response may carry
    ``window_coverage`` (the trailing-30d share of the window total); when it
    includes a ``coverage_warning``, the data is unevenly distributed across
    the window and the figure must NOT be used as a baseline. When no custom
    window applies, the engine's default
    window is disclosed via ``reporting_window`` — it is FRONTIER-ANCHORED
    (migration 089): "the most recent 30 days of data", ending at the domain's
    latest data date (``data_through``), NOT the last 30 calendar days. State
    the brand and the period your answer actually covers — cite ``data_through``
    when present, and do NOT imply a figure covers a period it does not.

    Args:
        kpi_name: the KPI to compute (resolved against the defined KPI registry).
        brand: optional brand filter (case-insensitive).
        region: optional geographic region filter (US census regions:
            northeast/south/midwest/west; synonyms and natural phrasings like
            'the Northeast region' or 'west coast' resolve; ambiguous or
            unknown values error with the known-label list plus a hint to ask
            the user which census region they mean). The response's
            ``region_status`` says whether the figure is actually
            region-scoped ("applied") or global ("not_applicable" — never
            present those as region-specific).
        segment: optional severity tier filter (low_severity, medium_severity,
            high_severity); mutually exclusive with region/therapy_line.
        therapy_line: optional line-of-therapy filter ('0'-'3'); mutually
            exclusive with region/segment.
        biologic: optional biologic-status filter ('naive'/'experienced'),
            REMIBRUTINIB ONLY -- returns an error for other brands (data is
            NULL by design); mutually exclusive with the other axes.
        ige_tier: optional IgE-tertile filter ('low'/'medium'/'high',
            data-driven), REMIBRUTINIB ONLY -- returns an error for other
            brands; mutually exclusive with the other axes.
        trigger_type: optional trigger-type filter, TRIGGER-EFFECTIVENESS KPIs
            ONLY (#1360) -- returns an error for any other KPI (never a
            silent drop).
        window: optional time window (rolling or absolute); omit for the
            engine's default window (most recent 30 days of data).

    Returns:
        Dict with success, kpi_id, kpi_name, value, status, data_source, brand,
        region + region_status (+ region_note when a requested region was not
        applied), the window provenance fields, data_through (the as-of date
        the default window ends at, when the engine reports one), and (when no
        custom window applies) reporting_window.

    STATUS SEMANTICS: "good"/"warning"/"critical" = value vs the KPI's defined
    thresholds; "informational" = the KPI has NO target BY DESIGN (volume
    metrics like TRx/NRx/NBRx and causal effect sizes are tracked for
    trend/context, not scored) — the value is real, do not call it a problem;
    "unknown" = the status could not be evaluated (missing data or calculation
    error) — do not speculate about causes beyond any error field present.
    """
    _span = kpi_resolution.recognize_kpi_span(kpi_name)
    if _span is None:
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "error": f"'{kpi_name}' did not resolve to a defined KPI.",
            "hint": "Try a defined KPI like NBRx, TRx, NRx, market share, conversion rate, or ROI.",
        }
    kpi, _normalized, _kpi_start, _kpi_end = _span

    # #1637: this tool computes ONE KPI, but ``kpi_name`` is free text. A
    # COORDINATED ask ("false alert rate and override rate") silently resolved to
    # whichever alias matched first and was answered as if complete -- the eval's
    # turn 4.6 shape, where the answer then blamed the tool for the metric it
    # never asked for. Refuse instead, naming both, so the caller issues one call
    # per metric.
    #
    # Gated on an explicit coordinator between the mentions, because bare
    # multi-mention detection is NOT sufficient: measured against the 20 distinct
    # kpi_name values the model actually passed across the 51-turn 2026-08-15 run,
    # an ungated guard refuses "TRx market share" (32 calls) -- a MODIFIER chain
    # naming the single KPI WS3-BI-008, not two metrics. Adjacency means one
    # metric; "and"/"vs"/"&"/"," between the spans means two.
    #
    # EVERY further mention is examined, not just the first (codex iter-4).
    # recognize_distinct_metric returns one match in vocabulary order, so
    # "TRx market share and ROI" handed back the ADJACENT "TRx" -- gap " ", no
    # coordinator -- and the tool computed TRx share while silently dropping ROI.
    # Stopping at the first mention re-created the exact fail-silent this guard
    # exists to close, just one mention further along.
    # The gap is read from the PUNCTUATION-PRESERVING normalization, not from
    # _normalized (#1637 codex iter-10). kpi_resolution normalizes "/" to a space
    # so "TRx/share" resolves as the single KPI WS3-BI-008 rather than falling
    # through to a token match on TRx alone -- but "/" is also a coordinator, and
    # reading the gap from the normalized string would erase it and let "TRx/NRx"
    # be answered as one metric. Both normalizations are length-preserving, so the
    # spans index into either string identically.
    _punctuated = " ".join(kpi_name.lower().split())
    _masked = _normalized[:_kpi_start] + " " * (_kpi_end - _kpi_start) + _normalized[_kpi_end:]
    _coordinated: List[str] = []
    _seen_ids = {kpi.id}
    # The loop terminates on its own: each pass masks the span it found, so the
    # probe runs out of mentions. _MAX_KPI_MENTION_SCANS is a backstop for the
    # case-sensitive abbreviation branch, which reads the UNMASKED original and
    # would otherwise keep returning the same mention. Exhausting it is logged
    # rather than silently truncating the scan.
    _scan_complete = True
    for _ in range(_MAX_KPI_MENTION_SCANS):
        _other = kpi_resolution.recognize_distinct_metric(
            _masked, exclude_id=kpi.id, original_query=kpi_name
        )
        if _other is None:
            break
        _other_kpi, _other_start, _other_end = _other
        # Only lack of PROGRESS may end the scan (codex iter-5). A repeated id
        # must not: "TRx market share for TRx and ROI" mentions TRx twice, and
        # breaking on the repeat abandoned the scan before reaching the
        # coordinated ROI -- answering one KPI as complete, the same fail-silent
        # one mention further along again. A zero-width span cannot be masked, so
        # that is the genuine no-progress case (the case-sensitive abbreviation
        # branch reports (0, 0) once its lowercase occurrence is masked away).
        if _other_end <= _other_start:
            break
        _masked = _masked[:_other_start] + " " * (_other_end - _other_start) + _masked[_other_end:]
        if _other_kpi.id in _seen_ids:
            continue
        _seen_ids.add(_other_kpi.id)
        _gap = _punctuated[min(_kpi_end, _other_end) : max(_kpi_start, _other_start)]
        if _KPI_COORDINATOR_RE.search(_gap):
            _coordinated.append(str(_other_kpi.name))
    else:
        # for/else: the loop ran the full range without breaking, i.e. it never
        # ran out of mentions -- the cap stopped it, not the text.
        _scan_complete = False
    if not _scan_complete:
        # Fail CLOSED, not just loudly (codex iter-6). Having established that a
        # further coordinated metric may be unexamined, computing one KPI and
        # returning success is exactly the false-complete this guard exists to
        # prevent -- the warning would document the wrong answer, not avoid it.
        logger.warning(
            "kpi_calculate: metric-mention scan hit its %d-scan cap for %r "
            "(found %s); refusing rather than answering a possibly-partial ask",
            _MAX_KPI_MENTION_SCANS,
            kpi_name,
            sorted(_seen_ids),
        )
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "error": (
                f"{kpi_name!r} contains too many metric mentions to determine "
                f"reliably whether more than one KPI was asked for; this tool "
                f"computes one KPI per call."
            ),
            "hint": "Call kpi_calculate_tool once per metric, naming each metric on its own.",
        }

    if _coordinated:
        _all_named = sorted({str(kpi.name), *_coordinated})
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "error": (
                f"{kpi_name!r} names more than one KPI ({' and '.join(_all_named)}); "
                f"this tool computes one KPI per call."
            ),
            # Enumerate EVERY named metric, not the first two: with three or more
            # the caller would otherwise be steered into dropping the rest, which
            # is the failure this guard exists to prevent.
            "hint": (
                "Call kpi_calculate_tool once per metric — "
                + ", then ".join(repr(n) for n in _all_named)
                + " — and report each."
            ),
        }

    # #1360: only the trigger-effectiveness calculators read
    # context['trigger_type']; on any other KPI the key would be silently
    # ignored while the response implied the filter applied (the
    # dead-'territory'-key incident). Fail fast, before touching the DB.
    if trigger_type and kpi.id not in _TRIGGER_EFFECTIVENESS_KPI_IDS:
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "kpi_id": kpi.id,
            "kpi_name": kpi.name,
            "error": (
                f"trigger_type only applies to the trigger-effectiveness KPIs "
                f"(trigger precision, acceptance rate, override rate, trigger "
                f"funnel conversion), not {kpi.name}."
            ),
        }

    # Parse the requested window BEFORE touching the calculator: an unparseable
    # window is a user-input error, not a calculation error, so fail fast with a
    # helpful hint and never hit the DB.
    from src.services.time_window import WindowParseError, parse_window

    try:
        parsed = parse_window(window)
    except WindowParseError as e:
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "error": str(e),
            "hint": "Try 'last 3 months', 'Q1 2025', or '2025-01-01 to 2025-03-31'.",
        }

    # Brand is an enum column and the input schema promises case-insensitivity:
    # resolve 'kisqali' to its real label before it reaches any calculator
    # (the scoped ROI query matches brand::text = $1 exactly — migration 125),
    # and fail fast with the known-brand list on an unmappable value (the
    # _query_kpis #1501 precedent) instead of a misleading no-rows error.
    if brand:
        normalized_brand = _normalize_brand(brand)
        if normalized_brand is None:
            return {
                "success": False,
                "query_type": "kpi_calculate",
                "kpi_id": kpi.id,
                "kpi_name": kpi.name,
                "error": (
                    f"brand {brand!r} does not match any known brand "
                    f"({', '.join(sorted(BRAND_ENUM_LABELS))})"
                ),
            }
        brand = normalized_brand

    # Region gets the same treatment (#1538, the brand precedent above): the
    # input schema promises synonym tolerance ('North East'/'NE'/'new
    # england'), and an unmappable value can never match a row — fail fast
    # with the known-label list before touching any calculator/DB instead of
    # returning a misleading 0/None under a junk region.
    if region:
        normalized_region = _normalize_region(region)
        if normalized_region is None:
            return {
                "success": False,
                "query_type": "kpi_calculate",
                "kpi_id": kpi.id,
                "kpi_name": kpi.name,
                "error": (
                    f"region {region!r} does not match any known region "
                    f"({', '.join(REGION_ENUM_LABELS)})"
                ),
                # #1565: ambiguity produces a question, not a dead end.
                "hint": _REGION_CLARIFY_HINT,
            }
        region = normalized_region

    context: Dict[str, Any] = {}
    if brand:
        context["brand"] = brand
    if region:
        # KPI calculators read ``context.get("region")`` (business_impact /
        # trigger_performance / data_quality); "territory" was a dead key, so a
        # region filter silently dropped -> region-agnostic windowed query while
        # the response still echoed the region. Pass the key the engine reads.
        context["region"] = region
    if segment:
        context["segment"] = segment
    if therapy_line:
        context["therapy_line"] = therapy_line
    if biologic:
        context["biologic"] = biologic
    if ige_tier:
        context["ige_tier"] = ige_tier
    if trigger_type:
        # TriggerPerformanceCalculator routes the migration-118 effectiveness
        # family on context['trigger_type'] (#1360).
        context["trigger_type"] = trigger_type
    if parsed is not None:
        context["window"] = parsed.as_dict()

    try:
        # Local import avoids a chatbot_tools <-> kpi route import cycle at load.
        from src.api.routes.kpi import get_kpi_calculator

        calculator = get_kpi_calculator()
        # calculate() is synchronous (a DB RPC) -> off-load to a worker thread so
        # the chatbot event loop is never blocked (mirrors the cognitive-RAG fix).
        result = await asyncio.to_thread(calculator.calculate, kpi.id, context=context)
    except Exception as exc:  # noqa: BLE001 - surface as a tool error, never fabricate
        logger.error("kpi_calculate_tool: calculation failed for %s: %s", kpi.id, exc)
        return {
            "success": False,
            "query_type": "kpi_calculate",
            "kpi_id": kpi.id,
            "kpi_name": kpi.name,
            "error": str(exc),
        }

    response = _kpi_result_to_response(kpi, result, brand=brand, region=region)
    coverage = await _window_coverage_probe(kpi, result, parsed, calculator, context)
    if coverage is not None:
        response["window_coverage"] = coverage
    return response


# =============================================================================
# CLINICAL CONTEXT TOOL (FDA-label / mechanism / competitor landscape)
# =============================================================================

# Lazily-built shared ClinicalContextService (real ChEMBL / ClinicalTrials.gov /
# PubMed / OpenFDA clients). Its fragment cache is a module-level global, so a
# single instance reuses live-API results across chatbot calls. Built on first
# use to keep the chatbot_tools import graph cheap (the ChEMBL kg client is a
# lazy import inside the service).
_clinical_context_service: Optional[Any] = None


def _get_clinical_context_service() -> Any:
    """Return the shared ClinicalContextService, building it on first use."""
    global _clinical_context_service
    if _clinical_context_service is None:
        from src.services.clinical_context import ClinicalContextService

        _clinical_context_service = ClinicalContextService()
    return _clinical_context_service


class ClinicalContextInput(BaseModel):
    """Input schema for clinical_context_tool."""

    brand: str = Field(
        description="Brand to fetch label/clinical context for (Kisqali, Fabhalta, or Remibrutinib)"
    )
    outcome: Optional[str] = Field(
        default="treatment_initiated",
        description=(
            "Optional synthetic outcome to frame against the brand's real pivotal "
            "endpoint (e.g. treatment_initiated, persistent_180d). Only affects the "
            "mapped-endpoint framing; the FDA-label indications, mechanism, and "
            "competitor landscape do NOT depend on it."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"brand": "Fabhalta", "outcome": "treatment_initiated"}}
    )


@tool(args_schema=ClinicalContextInput)
async def clinical_context_tool(
    brand: str,
    outcome: Optional[str] = "treatment_initiated",
) -> Dict[str, Any]:
    """
    Fetch REAL, source-attributed clinical/regulatory context for a Novartis brand
    to GROUND and TAILOR commercial and causal/strategic insight.

    Returns the FDA-label approved indications, limitations of use, and boxed
    warning (OpenFDA); the drug's mechanism of action (ChEMBL); the disease's
    pivotal trial endpoints (ClinicalTrials.gov); a real-world-evidence citation
    (PubMed); and the competitor landscape within the indication (curated
    reference). Every item carries its own source label plus an honesty label
    stating the synthetic-estimate / real-context boundary.

    Use this whenever a user asks about a brand's label indications, approved use,
    mechanism of action, on-/off-label boundaries, or the competitive/therapeutic
    landscape — and to anchor commercial/causal recommendations in the regulatory
    reality (e.g. on-label HCP targeting, competitive density within an
    indication, how the label boundary shapes causal drivers). This surfaces
    FACTUAL regulatory/biomedical context; it is NOT individualized prescribing or
    medical advice.

    Args:
        brand: Kisqali, Fabhalta, or Remibrutinib
        outcome: Optional synthetic outcome to frame against the real pivotal
            endpoint; does not affect the label indications themselves.

    Returns:
        Dict with success status, the clinical-context payload (approved
        indications, mechanism, pivotal endpoints, real-world evidence, competitor
        landscape), and the honesty label.
    """
    logger.info(f"Clinical context: brand={brand}, outcome={outcome}")

    try:
        service = _get_clinical_context_service()
        # get_context fans out synchronous httpx calls (ChEMBL + CT.gov + PubMed +
        # OpenFDA); off-load to a worker thread so a slow / rate-limited upstream
        # cannot block the chatbot event loop (mirrors the kpi_calculate_tool fix).
        payload = await asyncio.to_thread(
            service.get_context, brand, outcome or "treatment_initiated"
        )
    except KeyError:
        # brand_map has no profile for this brand (no enrichment facts).
        return {
            "success": False,
            "query_type": "clinical_context",
            "brand": brand,
            "error": (
                f"No clinical-context profile for brand '{brand}'. "
                "Known brands: Kisqali, Fabhalta, Remibrutinib."
            ),
        }
    except Exception as exc:  # noqa: BLE001 - surface as a tool error, never fabricate
        logger.error("clinical_context_tool: fetch failed for %s: %s", brand, exc)
        return {
            "success": False,
            "query_type": "clinical_context",
            "brand": brand,
            "error": str(exc),
        }

    return {
        "success": True,
        "query_type": "clinical_context",
        "brand": payload.get("brand", brand),
        "clinical_context": payload,
    }


# =============================================================================
# HCP SEGMENT LIKELIHOOD TOOL (#1354, demo Q3.3 / benchmark q14)
# =============================================================================


class HcpSegmentLikelihoodInput(BaseModel):
    """Input schema for predict_hcp_segment_likelihood_tool."""

    brand: str = Field(
        description=(
            "Brand to rank HCP segments for (Remibrutinib, Fabhalta, or Kisqali), "
            "case-insensitive. REQUIRED — there is no default; a missing brand "
            "fails closed (no silent brand)."
        )
    )
    segment_by: Optional[str] = Field(
        default="specialty",
        description=(
            "HCP segment axis to rank by: 'specialty' (default, the primary "
            "clinical archetype) or 'geographic_region'. These are the covariates "
            "the champion model is scored on."
        ),
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description=(
            "Optional horizon phrase from the ask (e.g. 'next quarter'). Echoed as "
            "context only — the model scores CURRENT adoption propensity, not a "
            "horizon-specific increase."
        ),
    )


@tool(args_schema=HcpSegmentLikelihoodInput)
async def predict_hcp_segment_likelihood_tool(
    brand: str,
    segment_by: Optional[str] = "specialty",
    time_horizon: Optional[str] = None,
) -> Dict[str, Any]:
    """Rank HCP segments by predicted likelihood-to-prescribe for a brand (#1354).

    Use this for "which HCP segments / specialties / regions are most likely to
    increase <brand> prescriptions?" (demo question 3.3). It scores the brand's
    PROMOTED HCP-adoption champion (a calibrated model, out-of-sample AUC ~0.77)
    over the REAL addressable HCP cohort and rolls the per-HCP adoption
    propensities up to a per-segment ranking with real n and standard errors —
    NOT a regional TRx proxy.

    Honesty: the served quantity is adoption propensity (the platform's
    "likelihood to prescribe"), NOT a horizon-conditioned increase — a stated
    horizon is context only. Thin segments are flagged ``low_confidence``. Fails
    closed (``success: False``) with an honest error when no champion is promoted
    for the brand or the scoring substrate is unavailable — never fabricates.

    Args:
        brand: Remibrutinib | Fabhalta | Kisqali (case-insensitive, required).
        segment_by: 'specialty' (default) or 'geographic_region'.
        time_horizon: optional horizon phrase, echoed as context only.
    """
    if not brand or not str(brand).strip():
        return {
            "success": False,
            "query_type": "hcp_segment_likelihood",
            "error": "A brand is required to rank HCP segments (no default brand).",
        }
    axis = segment_by or "specialty"
    try:
        # Lazy import keeps the chatbot_tools import graph cheap and avoids any
        # route<->service import cycle at load.
        from src.services.hcp_segment_likelihood import (
            ChampionNotPromotedError,
            SegmentScoringError,
            build_segment_ranking_narrative,
            score_hcp_segments,
        )

        result = await score_hcp_segments(brand, segment_by=axis)
    except ValueError as exc:
        return {
            "success": False,
            "query_type": "hcp_segment_likelihood",
            "error": str(exc),
            "segments": [],
        }
    except (ChampionNotPromotedError, SegmentScoringError) as exc:
        # Honest fail-closed: no promoted champion / empty substrate.
        return {
            "success": False,
            "query_type": "hcp_segment_likelihood",
            "brand": brand,
            "error": str(exc),
            "segments": [],
        }
    except Exception as exc:  # noqa: BLE001 - surface as a tool error, never fabricate
        logger.error("predict_hcp_segment_likelihood_tool failed for %s: %s", brand, exc)
        return {
            "success": False,
            "query_type": "hcp_segment_likelihood",
            "brand": brand,
            "error": str(exc),
            "segments": [],
        }

    narrative = build_segment_ranking_narrative(result, top_n=5, horizon=time_horizon)
    return {
        "success": True,
        "query_type": "hcp_segment_likelihood",
        "brand": result.brand,
        "model_name": result.model_name,
        "prediction_target": result.prediction_target,
        "segment_by": result.segment_by,
        "n_scored": result.n_scored,
        "overall_mean_propensity": result.overall_mean_propensity,
        "holdout_auc": result.holdout_auc,
        "feature_source": result.feature_source,
        "segments": [
            {
                "segment": s.segment,
                "n": s.n,
                "mean_propensity": s.mean_propensity,
                "se_propensity": s.se_propensity,
                "min_propensity": s.min_propensity,
                "max_propensity": s.max_propensity,
                "low_confidence": s.low_confidence,
            }
            for s in result.segments
        ],
        "narrative": narrative,
    }


# List of all E2I chatbot tools for use in LangGraph ToolNode
E2I_CHATBOT_TOOLS = [
    e2i_data_query_tool,
    kpi_calculate_tool,
    causal_analysis_tool,
    clinical_context_tool,
    agent_routing_tool,
    conversation_memory_tool,
    document_retrieval_tool,
    orchestrator_tool,
    tool_composer_tool,
    predict_hcp_segment_likelihood_tool,
]

# Tool name to function mapping
E2I_TOOL_MAP = {
    "e2i_data_query_tool": e2i_data_query_tool,
    "kpi_calculate_tool": kpi_calculate_tool,
    "causal_analysis_tool": causal_analysis_tool,
    "clinical_context_tool": clinical_context_tool,
    "agent_routing_tool": agent_routing_tool,
    "conversation_memory_tool": conversation_memory_tool,
    "document_retrieval_tool": document_retrieval_tool,
    "orchestrator_tool": orchestrator_tool,
    "tool_composer_tool": tool_composer_tool,
    "predict_hcp_segment_likelihood_tool": predict_hcp_segment_likelihood_tool,
}


def get_e2i_chatbot_tools() -> List:
    """Get list of all E2I chatbot tools for LangGraph integration."""
    return E2I_CHATBOT_TOOLS


def get_tool_by_name(name: str):
    """Get a specific tool by name."""
    return E2I_TOOL_MAP.get(name)
