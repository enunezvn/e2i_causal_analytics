"""Shared agent_activities runtime writer (#1355).

Until this module, only experiment_designer's memory hooks persisted
agent_activities rows (#883 §5) — every other Tier-2/3 agent completed analyses
that ``_query_agent_analysis`` (the chat agent-analysis tool), the
business_impact_roi_agent_activities KPI and the RAG index could never see.

``persist_agent_activity`` lifts the experiment_designer idiom into one SSOT so
heterogeneous_optimizer / causal_impact / gap_analyzer (and future agents) all
write the SAME real-column payload:

* REAL columns only — schema SSOT database/core/e2i_ml_complete_v3_schema.sql
  :610 (+ migration 063 ``is_synthetic``); ``search_vector`` is GENERATED and
  must never be written.
* ``is_synthetic=False`` — runtime rows are REAL provenance, distinct from the
  DGP-seeded substrate (AgentActivitiesGenerator stamps True).
* Brand contract: callers put ``brand`` INSIDE ``analysis_results`` — that is
  the field the chat read path filters on (``analysis_results->>'brand'``,
  see AgentActivityRepository.query_activities).
* NEVER raises: a failed activity write must never fail the analysis itself
  (log-and-continue, mirroring memory_hooks' swallow-and-warn semantics).

Uses the sync service-role client from ``src.repositories.get_supabase_client``
— the same client the experiment_designer hooks insert with.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# varchar(30) PK on the live table.
_ACTIVITY_ID_MAX = 30

#: Kill switch for the IMPLICIT (factory-created) real client. Armed by
#: tests/conftest.py for every pytest session: on 2026-07-30 the pre-existing
#: agent unit suites invoked ``contribute_to_memory``, the factory picked up
#: real service-role creds (tests/conftest ``load_dotenv(override=True)``) and
#: 16 real rows landed in the LIVE agent_activities table (same bug family as
#: the #1371 CHATBOT_MLFLOW_METRICS hang: a default-on external write reachable
#: from unit tests). An EXPLICITLY passed client is unaffected — tests inject
#: fakes, and a deliberate caller can still opt in. Read fresh per call.
_DISABLE_ENV = "E2I_DISABLE_AGENT_ACTIVITY_WRITER"
_TRUTHY = ("1", "true", "yes")


def _writer_disabled() -> bool:
    return os.getenv(_DISABLE_ENV, "0").strip().lower() in _TRUTHY


def _default_client_factory():
    from src.repositories import get_supabase_client

    return get_supabase_client()


def _clamp(value: Optional[float], lo: float, hi: float) -> Optional[float]:
    if value is None:
        return None
    try:
        return min(max(float(value), lo), hi)
    except (TypeError, ValueError):
        return None


def build_activity_id(agent_name: str, salt: str = "") -> str:
    """Content-hashed id ``act_<16hex>`` (20 chars <= varchar(30)).

    Timestamped so repeated analyses by the same agent never collide.
    sha256 is content addressing, not security.
    """
    content = f"{agent_name}:{salt}:{datetime.now(timezone.utc).isoformat()}"
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"act_{digest}"[:_ACTIVITY_ID_MAX]


def persist_agent_activity(
    *,
    agent_name: str,
    agent_tier: str,
    activity_type: str,
    analysis_results: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None,
    processing_duration_ms: Optional[int] = None,
    records_processed: Optional[int] = None,
    causal_paths_analyzed: Optional[int] = None,
    confidence_level: Optional[float] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    impact_estimate: Optional[float] = None,
    roi_estimate: Optional[float] = None,
    workstream: Optional[str] = None,
    status: str = "completed",
    supabase_client: Any = None,
    _client_factory: Callable[[], Any] = _default_client_factory,
) -> Optional[str]:
    """Insert one real (``is_synthetic=False``) agent_activities row.

    Returns the ``activity_id`` when the insert landed, ``None`` otherwise —
    never raises and never fabricates success (the #883 §5 lesson: report a
    stored row only when ``response.data`` confirms it).

    ``analysis_results`` should carry ``brand`` when the analysis is
    brand-scoped — the chat agent-analysis query resolves brand through
    ``analysis_results->>'brand'``.

    ``E2I_DISABLE_AGENT_ACTIVITY_WRITER`` (armed session-wide by
    tests/conftest.py) blocks the IMPLICIT factory-created client so no unit
    test can ever write the live table; an explicitly passed
    ``supabase_client`` (test fakes, deliberate callers) is unaffected.
    """
    try:
        if supabase_client is None and _writer_disabled():
            logger.debug("agent_activities write disabled (%s) for %s", _DISABLE_ENV, agent_name)
            return None
        client = supabase_client if supabase_client is not None else _client_factory()
        if client is None:
            logger.warning("agent_activities write skipped for %s: no Supabase client", agent_name)
            return None

        activity_id = build_activity_id(agent_name, salt=activity_type)
        payload: Dict[str, Any] = {
            "activity_id": activity_id,
            "agent_name": agent_name,
            "agent_tier": agent_tier,
            "activity_timestamp": datetime.now(timezone.utc).isoformat(),
            "activity_type": activity_type,
            "analysis_results": analysis_results,
            "status": status,
            # Runtime rows are REAL provenance (the DGP seed generator is the
            # only writer that stamps True).
            "is_synthetic": False,
        }
        if input_data is not None:
            payload["input_data"] = input_data
        if processing_duration_ms is not None:
            payload["processing_duration_ms"] = int(processing_duration_ms)
        if records_processed is not None:
            payload["records_processed"] = int(records_processed)
        if causal_paths_analyzed is not None:
            payload["causal_paths_analyzed"] = int(causal_paths_analyzed)
        if confidence_level is not None:
            # numeric(4,3); contract is a 0-1 confidence — clamp defensively.
            payload["confidence_level"] = round(_clamp(confidence_level, 0.0, 1.0) or 0.0, 3)
        if recommendations is not None:
            payload["recommendations"] = recommendations
        if impact_estimate is not None:
            # numeric(15,2)
            clamped_impact = _clamp(impact_estimate, -1e13, 1e13)
            if clamped_impact is not None:
                payload["impact_estimate"] = round(clamped_impact, 2)
        if roi_estimate is not None:
            # numeric(5,2) caps at +/-999.99
            clamped_roi = _clamp(roi_estimate, -999.99, 999.99)
            if clamped_roi is not None:
                payload["roi_estimate"] = round(clamped_roi, 2)
        if workstream is not None:
            payload["workstream"] = workstream

        response = client.table("agent_activities").insert(payload).execute()
        if getattr(response, "data", None):
            logger.info("Stored agent activity %s for %s", activity_id, agent_name)
            return activity_id
        logger.warning("agent_activities insert returned no data for %s", agent_name)
    except Exception as e:  # log-and-continue: never fail the analysis
        logger.warning("Failed to store agent activity for %s: %s", agent_name, e)
    return None
