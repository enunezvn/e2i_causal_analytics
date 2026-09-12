"""Response schema for GET /admin/observability/tool-composer (spec §8).

The verdict word comes first on every tool row, because it is the reading: the counts explain it.
Measured latency is null until there are 20 successful runs to measure, and the DECLARED latency
is a separate, labelled field — the two are never merged, so a declared number can never be read
as a measurement.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CompositionCounts(BaseModel):
    """How the window's compositions ended."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(description="Episodes started in the window")
    success: int = 0
    partial: int = 0
    failed: int = 0
    cancelled: int = 0
    unfinished: int = Field(default=0, description="No terminal outcome recorded yet")
    abandoned: int = Field(
        default=0,
        description="Unfinished and silent past the heartbeat window: the worker went away",
    )
    by_plan_source: Dict[str, int] = Field(
        default_factory=dict, description="llm / plan_cache / kpi_deterministic"
    )
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None


class ToolReliabilityRow(BaseModel):
    """One tool's evidence in the window, verdict first."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    verdict: str = Field(
        description="no_runs | too_few_runs | caveat | reliable | inconclusive (spec §7.1)"
    )
    category: Optional[str] = None
    source_agent: Optional[str] = None
    n_invoked: int = 0
    n_succeeded: int = 0
    n_refused: int = Field(default=0, description="Declining to answer; never a health failure")
    n_health_failures: int = 0
    n_health: int = Field(default=0, description="succeeded + health failures: the denominator")
    n_retried: int = 0
    n_synthetic: int = 0
    p50_latency_ms: Optional[float] = Field(
        default=None, description="Measured; null below 20 successful runs"
    )
    p95_latency_ms: Optional[float] = None
    declared_latency_ms: Optional[float] = Field(
        default=None, description="The registry's declared number, never a measurement"
    )
    most_common_health_error: Optional[str] = None
    last_executed_at: Optional[str] = None


class RecentFailure(BaseModel):
    """A composition that failed or only partly succeeded."""

    model_config = ConfigDict(extra="forbid")

    composition_id: str
    outcome: Optional[str] = None
    status: Optional[str] = None
    failed_phase: Optional[str] = None
    error_type: Optional[str] = Field(
        default=None, description="The exception class; never its message (spec §5.5)"
    )
    entry_point: Optional[str] = None
    plan_source: Optional[str] = None
    query_preview: str = Field(default="", description="Redacted, at most 100 characters")
    step_classes: List[Dict[str, Any]] = Field(
        default_factory=list, description="The steps that did not succeed, with their classes"
    )
    last_activity_at: Optional[str] = None
    total_latency_ms: Optional[float] = None


class ToolComposerObservability(BaseModel):
    """The Observability tab's tool-composer section."""

    model_config = ConfigDict(extra="forbid")

    window_days: int
    include_synthetic: bool = Field(
        description="Whether synthetic-substrate runs are counted in this deployment"
    )
    compositions: CompositionCounts
    tools: List[ToolReliabilityRow] = Field(default_factory=list)
    recent_failures: List[RecentFailure] = Field(default_factory=list)
