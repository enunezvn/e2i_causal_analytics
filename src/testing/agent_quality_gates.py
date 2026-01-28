"""Agent Quality Gates Configuration.

Defines per-agent quality thresholds for Tier 1-5 agent testing.
Each agent has specific success criteria beyond just contract validation.

Quality gates enforce:
1. Required output fields (must be present)
2. Data quality checks (type, range, content validation)
3. Status-based failure detection (agents that return error states)
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class DataQualityCheck(TypedDict, total=False):
    """Data quality check configuration."""

    type: Literal["str", "int", "float", "bool", "list", "dict", "tuple"]
    not_null: bool
    min_value: float | int
    max_value: float | int
    min_length: int
    max_length: int
    must_be: Any  # Value must equal this
    must_not_be: Any  # Value must not equal this
    in_set: list[Any]  # Value must be in this set
    not_contains: list[str]  # String must not contain these substrings (for error detection)


class DataSourceRequirement(TypedDict, total=False):
    """Data source validation configuration for an agent.

    Used to enforce that agents use real data sources instead of mocks.
    """

    reject_mock: bool  # If True, reject mock data sources
    acceptable_sources: list[str]  # List of acceptable source types (e.g., ["supabase", "tier0"])


class AgentQualityGate(TypedDict, total=False):
    """Quality gate configuration for a single agent."""

    # Fields that MUST be present in output
    required_output_fields: list[str]

    # Minimum percentage of contract required fields that must be present
    min_required_fields_pct: float

    # Data quality checks for specific fields
    data_quality_checks: dict[str, DataQualityCheck]

    # Status values that indicate failure
    fail_on_status: list[str]

    # Custom description for documentation
    description: str

    # Data source validation requirements
    data_source_requirement: DataSourceRequirement


# Per-agent quality gate definitions
AGENT_QUALITY_GATES: dict[str, AgentQualityGate] = {
    "orchestrator": {
        "description": "Coordinates all agents, routes queries to appropriate handlers",
        "required_output_fields": ["status", "response_text"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "status": {"type": "str", "not_null": True},
            "response_text": {"type": "str", "min_length": 1},
        },
        "fail_on_status": ["error", "failed"],
    },
    "tool_composer": {
        "description": "Multi-faceted query decomposition and tool orchestration",
        "required_output_fields": ["success"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "success": {"type": "bool", "must_be": True},
        },
        "fail_on_status": ["failed"],
    },
    "causal_impact": {
        "description": "Traces causal chains, estimates treatment effects",
        "required_output_fields": ["status"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "status": {"type": "str", "not_null": True, "must_not_be": "error"},
        },
        "fail_on_status": ["error", "failed"],
    },
    "gap_analyzer": {
        "description": "Identifies ROI opportunities and performance gaps",
        "required_output_fields": ["executive_summary"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "executive_summary": {
                "type": "str",
                "not_null": True,
                "not_contains": ["Error:", "error:", "Failed:", "failed:"],
            },
        },
        "fail_on_status": ["error", "failed"],
    },
    "heterogeneous_optimizer": {
        "description": "Segment-level CATE analysis for heterogeneous treatment effects",
        "required_output_fields": ["status"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "status": {"type": "str", "not_null": True},
        },
        "fail_on_status": ["error", "failed"],
    },
    "drift_monitor": {
        "description": "Monitors data and model drift",
        "required_output_fields": ["overall_drift_score"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "overall_drift_score": {"type": "float", "not_null": True},
        },
        "fail_on_status": ["error", "failed"],
    },
    "experiment_designer": {
        "description": "Designs A/B tests with Digital Twin pre-screening",
        "required_output_fields": ["design_type"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "design_type": {"type": "str", "not_null": True},
        },
        "fail_on_status": ["error", "failed"],
    },
    "health_score": {
        "description": "Monitors system health metrics",
        "required_output_fields": ["overall_health_score"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "overall_health_score": {
                "type": "float",
                "not_null": True,
                "min_value": 0.0,
                "max_value": 100.0,
                # Reject perfect 100.0 scores - indicates mock data usage
                # Real systems always have some variance in health checks
                "must_not_be": 100.0,
            },
            "component_health_score": {
                "type": "float",
                # Reject perfect 1.0 component scores - indicates mock data
                # Real component health checks have latency variance
                "must_not_be": 1.0,
            },
        },
        "fail_on_status": ["error", "failed"],
    },
    "prediction_synthesizer": {
        "description": "Aggregates ML predictions from multiple models",
        "required_output_fields": ["status"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "status": {"type": "str", "not_null": True, "must_not_be": "failed"},
        },
        "fail_on_status": ["failed", "error"],
    },
    "resource_optimizer": {
        "description": "Optimizes resource allocation",
        "required_output_fields": ["status"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "status": {"type": "str", "not_null": True},
        },
        "fail_on_status": ["error", "failed", "infeasible"],
    },
    "explainer": {
        "description": "Generates natural language explanations",
        "required_output_fields": ["executive_summary"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "executive_summary": {"type": "str", "min_length": 10},
        },
        "fail_on_status": ["error", "failed"],
    },
    "feedback_learner": {
        "description": "Self-improvement from feedback",
        "required_output_fields": ["status"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "status": {"type": "str", "in_set": ["completed", "partial", "success"]},
        },
        "fail_on_status": ["error", "failed"],
    },
}


def get_quality_gate(agent_name: str) -> AgentQualityGate | None:
    """Get quality gate configuration for an agent.

    Args:
        agent_name: Name of the agent

    Returns:
        AgentQualityGate or None if not configured
    """
    return AGENT_QUALITY_GATES.get(agent_name)


def list_configured_agents() -> list[str]:
    """List all agents with quality gates configured."""
    return list(AGENT_QUALITY_GATES.keys())
