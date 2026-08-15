"""
E2I Agent Orchestration API
============================

FastAPI endpoints for agent status monitoring and orchestration.

Endpoints:
- GET /agents/status: Get status of all 22 agents in the tier hierarchy

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, model_validator

from src.api.dependencies.auth import require_auth
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["Agent Orchestration"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS
# =============================================================================


class AgentStatusEnum(str, Enum):
    """Agent status values."""

    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"


class AgentTierEnum(int, Enum):
    """Agent tier levels (0-5)."""

    ML_FOUNDATION = 0
    ORCHESTRATION = 1
    CAUSAL_ANALYTICS = 2
    MONITORING = 3
    ML_PREDICTIONS = 4
    SELF_IMPROVEMENT = 5


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class AgentInfo(BaseModel):
    """Information about a single agent."""

    id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    tier: int = Field(..., ge=0, le=5, description="Agent tier (0-5)")
    status: AgentStatusEnum = Field(..., description="Current agent status")
    last_activity: Optional[str] = Field(None, description="ISO timestamp of last activity")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")


class AgentStatusResponse(BaseModel):
    """Response containing status of all agents."""

    agents: List[AgentInfo] = Field(..., description="List of all agents with their status")
    total_agents: int = Field(..., description="Total number of agents")
    # ``total`` is an alias for ``total_agents`` so the frontend contract
    # (api-schemas.ts AgentStatusResponseSchema declares ``total``) resolves to a
    # real value instead of falling through to undefined. Kept alongside
    # ``total_agents`` (existing consumers) rather than renamed, since both name
    # the same count. Optional on input + auto-derived from ``total_agents`` so
    # existing construction sites need not pass it; the wire always carries it.
    total: Optional[int] = Field(
        default=None, description="Total number of agents (alias of total_agents)"
    )
    active_count: int = Field(..., description="Number of active agents")
    processing_count: int = Field(..., description="Number of processing agents")
    error_count: int = Field(..., description="Number of agents in error state")
    timestamp: datetime = Field(..., description="Response timestamp")

    @model_validator(mode="after")
    def _mirror_total(self) -> "AgentStatusResponse":
        """``total`` is a pure alias of ``total_agents`` — mirror it
        unconditionally so an inconsistent explicit ``total`` can never be
        serialized (the two must always agree)."""
        self.total = self.total_agents
        return self


class AgentActivityItem(BaseModel):
    """A single agent action, sourced from ``audit_chain_entries``."""

    entry_id: str = Field(..., description="audit_chain_entries.entry_id")
    agent_id: str = Field(..., description="Registry agent id (kebab-case)")
    agent_name: str = Field(..., description="Human-readable agent name")
    tier: int = Field(..., ge=0, le=5, description="Agent tier (0-5)")
    action: str = Field(..., description="Human-readable action label")
    action_type: str = Field(..., description="Raw action_type slug")
    timestamp: str = Field(..., description="ISO timestamp of the action")
    duration_ms: Optional[int] = Field(None, description="Action duration in ms")
    status: str = Field(..., description="completed | in_progress | failed")
    details: Optional[str] = Field(None, description="Query text / context, if any")


class AgentActivityResponse(BaseModel):
    """Response containing recent agent activity (newest first)."""

    activities: List[AgentActivityItem] = Field(
        default_factory=list, description="Recent agent actions, newest first"
    )
    total: int = Field(..., description="Number of activities returned")
    window_hours: int = Field(..., description="Look-back window in hours")
    timestamp: datetime = Field(..., description="Response timestamp")


# =============================================================================
# SAMPLE DATA
# =============================================================================

# Default agent configuration matching the 22-agent tier hierarchy
AGENT_REGISTRY = [
    # Tier 0 - ML Foundation (9 agents)
    AgentInfo(
        id="scope-definer",
        name="Scope Definer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        last_activity=None,
        capabilities=["problem_scoping", "requirement_analysis"],
    ),
    AgentInfo(
        id="data-preparer",
        name="Data Preparer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        last_activity=None,
        capabilities=["data_validation", "preprocessing"],
    ),
    AgentInfo(
        id="feature-analyzer",
        name="Feature Analyzer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        last_activity=None,
        capabilities=["feature_engineering", "selection"],
    ),
    AgentInfo(
        id="model-selector",
        name="Model Selector",
        tier=0,
        status=AgentStatusEnum.IDLE,
        last_activity=None,
        capabilities=["model_comparison", "benchmarking"],
    ),
    AgentInfo(
        id="model-trainer",
        name="Model Trainer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        last_activity=None,
        capabilities=["training", "hyperparameter_tuning"],
    ),
    AgentInfo(
        id="model-deployer",
        name="Model Deployer",
        tier=0,
        status=AgentStatusEnum.IDLE,
        last_activity=None,
        capabilities=["deployment", "versioning"],
    ),
    AgentInfo(
        id="observability-connector",
        name="Observability Connector",
        tier=0,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["mlflow", "opik", "monitoring"],
    ),
    AgentInfo(
        id="cohort-constructor",
        name="Cohort Constructor",
        tier=0,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["cohort_definition", "patient_eligibility", "inclusion_exclusion"],
    ),
    AgentInfo(
        id="cohort-profiler",
        name="Cohort Profiler",
        tier=0,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["cohort_sizing", "segment_breakdown", "population_profiling"],
    ),
    # Tier 1 - Orchestration (2 agents)
    AgentInfo(
        id="orchestrator",
        name="Orchestrator",
        tier=1,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["routing", "coordination", "agent_dispatch"],
    ),
    AgentInfo(
        id="tool-composer",
        name="Tool Composer",
        tier=1,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["tool_orchestration", "query_decomposition"],
    ),
    # Tier 2 - Causal Analytics (3 agents)
    AgentInfo(
        id="causal-impact",
        name="Causal Impact",
        tier=2,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["causal_tracing", "effect_estimation", "dowhy"],
    ),
    AgentInfo(
        id="gap-analyzer",
        name="Gap Analyzer",
        tier=2,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["opportunity_detection", "roi_analysis"],
    ),
    AgentInfo(
        id="heterogeneous-optimizer",
        name="Heterogeneous Optimizer",
        tier=2,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["cate_analysis", "segment_optimization", "econml"],
    ),
    # Tier 3 - Monitoring (4 agents)
    AgentInfo(
        id="drift-monitor",
        name="Drift Monitor",
        tier=3,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["data_drift", "model_drift", "alerting"],
    ),
    AgentInfo(
        id="experiment-designer",
        name="Experiment Designer",
        tier=3,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["ab_testing", "sample_size", "power_analysis"],
    ),
    AgentInfo(
        id="experiment-monitor",
        name="Experiment Monitor",
        tier=3,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["experiment_monitoring", "srm_detection", "interim_analysis"],
    ),
    AgentInfo(
        id="health-score",
        name="Health Score",
        tier=3,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["system_health", "performance_metrics"],
    ),
    # Tier 4 - ML Predictions (2 agents)
    AgentInfo(
        id="prediction-synthesizer",
        name="Prediction Synthesizer",
        tier=4,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["prediction_aggregation", "ensemble"],
    ),
    AgentInfo(
        id="resource-optimizer",
        name="Resource Optimizer",
        tier=4,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["resource_allocation", "optimization"],
    ),
    # Tier 5 - Self-Improvement (2 agents)
    AgentInfo(
        id="explainer",
        name="Explainer",
        tier=5,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["nl_generation", "insight_explanation"],
    ),
    AgentInfo(
        id="feedback-learner",
        name="Feedback Learner",
        tier=5,
        status=AgentStatusEnum.ACTIVE,
        last_activity=None,
        capabilities=["feedback_integration", "self_improvement"],
    ),
]


# Fast lookup of registry agents by id (kebab-case) for status/activity joins.
_REGISTRY_BY_ID: Dict[str, AgentInfo] = {a.id: a for a in AGENT_REGISTRY}


# =============================================================================
# LIVE ACTIVITY DERIVATION (audit_chain_entries)
# =============================================================================
#
# Statuses and the Activity Feed are derived from ``audit_chain_entries`` — the
# per-agent-node execution log already consumed by /analytics/summary. This
# replaces the previously-hardcoded statuses (ACTIVE x15) and the empty
# Activity Feed with live, honest signal. When telemetry is unavailable every
# agent reports IDLE and the feed is empty — never a fabricated value.

# An agent is reported ACTIVE only if it recorded an action within this window.
_ACTIVE_WINDOW = timedelta(minutes=15)

# Default look-back for the Activity Feed.
_ACTIVITY_WINDOW_HOURS_DEFAULT = 24

# ``health_score_quick`` is the automated background health poller: it emits the
# overwhelming majority of audit rows (~99% of the last 24h — workflow_start /
# component / compose scaffolding). It is REAL activity, but surfacing it in the
# feed would bury genuine agent work under a wall of identical polling entries,
# so it is excluded from the FEED only. It is still counted for the health-score
# agent's live STATUS (the poller running means health-score is active). Add
# further automated pollers here if they appear.
_AUTOMATED_POLLER_AGENTS = {"health_score_quick"}

# Intra-workflow scaffolding actions: the orchestration plumbing emitted around
# every workflow (genesis + compose/component steps). Real, but low-signal for
# an Activity Feed — excluded from the FEED only so it surfaces meaningful agent
# work (estimate_cate, model_training, srm_detector, ...). Kept for STATUS.
_SCAFFOLDING_ACTIONS = {"workflow_start", "component", "compose"}

# audit_chain_entries.agent_name is snake_case; AGENT_REGISTRY ids are
# kebab-case. Most map by a simple replace; poller/variant names alias to their
# canonical agent.
_AGENT_NAME_ALIASES = {
    "health_score_quick": "health-score",
}

# Acronyms to preserve when humanizing an action_type slug for display.
_ACTION_ACRONYMS = {"cate": "CATE", "srm": "SRM", "roi": "ROI", "ml": "ML"}


def _normalize_agent_name(raw: str) -> str:
    """Map an ``audit_chain_entries.agent_name`` to its AGENT_REGISTRY id."""
    if raw in _AGENT_NAME_ALIASES:
        return _AGENT_NAME_ALIASES[raw]
    return raw.replace("_", "-")


def _humanize_slug(slug: str) -> str:
    """Turn a snake_case slug ('estimate_cate') into a label ('Estimate CATE')."""
    if not slug:
        return "Activity"
    return " ".join(_ACTION_ACRONYMS.get(w.lower(), w.capitalize()) for w in slug.split("_"))


def _row_status(validation_passed: Optional[bool], action_type: str) -> str:
    """Per-row activity status. Mirrors the analytics convention: a NULL
    validation counts as completed — only an explicit ``False`` or an
    ``*_error`` action is a failure (never a fabricated success/failure)."""
    if validation_passed is False or (action_type or "").endswith("_error"):
        return "failed"
    return "completed"


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp from the audit chain; None if unparseable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _get_supabase_client():
    """Get the Supabase client for audit-log reads (None if unavailable)."""
    try:
        from src.api.dependencies.supabase_client import get_supabase

        return get_supabase()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not get Supabase client for agent activity: {e}")
        return None


def _fetch_audit_rows(
    since: datetime,
    *,
    feed_view: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fetch recent ``audit_chain_entries`` rows (newest first).

    ``feed_view=True`` (Activity Feed): excludes the automated health poller AND
    intra-workflow scaffolding actions so the feed surfaces meaningful agent
    work. ``feed_view=False`` (status derivation): keeps every row, since the
    poller running is itself evidence the health-score agent is active.

    Returns ``[]`` on any error (including no configured DB) so callers degrade
    gracefully rather than fabricating data.
    """
    db = _get_supabase_client()
    if db is None:
        return []
    try:
        query = (
            db.table("audit_chain_entries")
            .select(
                "entry_id, agent_name, agent_tier, action_type, created_at, "
                "duration_ms, validation_passed, query_text, brand"
            )
            .gte("created_at", since.isoformat())
        )
        if feed_view:
            for poller in _AUTOMATED_POLLER_AGENTS:
                query = query.neq("agent_name", poller)
            for action in _SCAFFOLDING_ACTIONS:
                query = query.neq("action_type", action)
        result = query.order("created_at", desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        logger.warning(f"Could not fetch agent audit rows: {e}")
        return []


def _derive_live_statuses(rows: List[Dict[str, Any]], now: datetime) -> Dict[str, Dict[str, Any]]:
    """Compute ``{registry_id: {status, last_activity}}`` from recent rows.

    ``rows`` must be newest-first, so the first row seen for an agent is its
    latest action. An agent active within :data:`_ACTIVE_WINDOW` is ACTIVE;
    one whose latest action failed is ERROR; otherwise IDLE.
    """
    live: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        agent_id = _normalize_agent_name(row.get("agent_name") or "")
        if agent_id in live:
            continue  # newest-first => already captured this agent's latest
        ts = _parse_ts(row.get("created_at"))
        if ts is None:
            continue
        if _row_status(row.get("validation_passed"), row.get("action_type") or "") == "failed":
            status = AgentStatusEnum.ERROR
        elif (now - ts) <= _ACTIVE_WINDOW:
            status = AgentStatusEnum.ACTIVE
        else:
            status = AgentStatusEnum.IDLE
        live[agent_id] = {"status": status, "last_activity": row.get("created_at")}
    return live


def _apply_live_statuses(live: Dict[str, Dict[str, Any]]) -> List[AgentInfo]:
    """Overlay live status + last_activity onto the registry WITHOUT mutating
    it (uses ``model_copy``). Agents with no recent telemetry report IDLE /
    ``last_activity=None`` — honest "no observed activity", never fabricated."""
    agents: List[AgentInfo] = []
    for agent in AGENT_REGISTRY:
        info = live.get(agent.id)
        if info is not None:
            agents.append(
                agent.model_copy(
                    update={
                        "status": info["status"],
                        "last_activity": info["last_activity"],
                    }
                )
            )
        else:
            agents.append(
                agent.model_copy(update={"status": AgentStatusEnum.IDLE, "last_activity": None})
            )
    return agents


def _row_to_activity(row: Dict[str, Any]) -> Optional[AgentActivityItem]:
    """Map an audit row to an Activity Feed item (None if it has no timestamp)."""
    created = row.get("created_at")
    if not created:
        return None
    raw = row.get("agent_name") or ""
    agent_id = _normalize_agent_name(raw)
    registry = _REGISTRY_BY_ID.get(agent_id)
    if registry is not None:
        tier = registry.tier
        name = registry.name
    else:
        # Unmapped source (e.g. a pipeline-level marker): fall back to the row's
        # own tier, clamped to the valid range, and a humanized name.
        tier = max(0, min(5, int(row.get("agent_tier") or 0)))
        name = _humanize_slug(raw)
    action_type = row.get("action_type") or ""
    details = (row.get("query_text") or "").strip() or None
    return AgentActivityItem(
        entry_id=str(row.get("entry_id") or created),
        agent_id=agent_id,
        agent_name=name,
        tier=tier,
        action=_humanize_slug(action_type),
        action_type=action_type,
        timestamp=created,
        duration_ms=row.get("duration_ms"),
        status=_row_status(row.get("validation_passed"), action_type),
        details=details,
    )


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "/status",
    response_model=AgentStatusResponse,
    summary="Get all agents status",
    operation_id="get_agent_status",
)
async def get_agent_status() -> AgentStatusResponse:
    """
    Get status of all agents in the orchestration system.

    Returns all 22 agents across 6 tiers with their capabilities. Status and
    ``last_activity`` are derived live from ``audit_chain_entries`` — an agent
    that recorded an action within the last 15 minutes is ACTIVE; otherwise
    IDLE. When telemetry is unavailable every agent reports IDLE (the registry's
    default statuses are never surfaced as a fabricated "active").
    """
    now = datetime.now(timezone.utc)
    rows = _fetch_audit_rows(now - _ACTIVE_WINDOW, feed_view=False, limit=2000)
    agents = _apply_live_statuses(_derive_live_statuses(rows, now))

    active_count = sum(1 for a in agents if a.status == AgentStatusEnum.ACTIVE)
    processing_count = sum(1 for a in agents if a.status == AgentStatusEnum.PROCESSING)
    error_count = sum(1 for a in agents if a.status == AgentStatusEnum.ERROR)

    return AgentStatusResponse(
        agents=agents,
        total_agents=len(agents),
        total=len(agents),
        active_count=active_count,
        processing_count=processing_count,
        error_count=error_count,
        timestamp=now,
    )


@router.get(
    "/activity",
    response_model=AgentActivityResponse,
    summary="Get recent agent activity",
    operation_id="get_agent_activity",
)
async def get_agent_activity(
    hours: int = Query(
        _ACTIVITY_WINDOW_HOURS_DEFAULT, ge=1, le=168, description="Look-back window (hours)"
    ),
    limit: int = Query(50, ge=1, le=200, description="Max activities to return"),
    user: Dict[str, Any] = Depends(require_auth),
) -> AgentActivityResponse:
    """
    Recent agent actions from the audit chain (newest first).

    Sourced from ``audit_chain_entries``. The automated health poller
    (``health_score_quick``) is excluded so the feed surfaces meaningful agent
    work; an empty list is an honest "no recent activity", never a fabricated
    entry. Auth-gated (reads may include query text), matching the other
    audit-reading endpoints.
    """
    now = datetime.now(timezone.utc)
    rows = _fetch_audit_rows(now - timedelta(hours=hours), feed_view=True, limit=limit)
    activities = [item for item in (_row_to_activity(r) for r in rows) if item is not None]
    return AgentActivityResponse(
        activities=activities,
        total=len(activities),
        window_hours=hours,
        timestamp=now,
    )
