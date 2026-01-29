"""Agent Quality Gates Configuration.

Defines per-agent quality thresholds for Tier 1-5 agent testing.
Each agent has specific success criteria beyond just contract validation.

Quality gates enforce:
1. Required output fields (must be present)
2. Data quality checks (type, range, content validation)
3. Status-based failure detection (agents that return error states)
4. **Semantic validation** - checks meaning, not just structure (v4.3)
"""

from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict


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

    # Semantic validator function: (output: dict) -> tuple[bool, str]
    # Returns (is_valid, reason) where reason explains why validation failed
    # This validates MEANING, not just structure
    semantic_validator: Callable[[dict[str, Any]], tuple[bool, str]]


# =============================================================================
# SEMANTIC VALIDATORS
# These functions validate MEANING, not just structure.
# Each returns (is_valid: bool, reason: str)
# =============================================================================


def _validate_tool_composer(output: dict[str, Any]) -> tuple[bool, str]:
    """Tool composer must have real tool execution, not 'unable to assess'."""
    # Check all possible response fields
    response_text = ""
    for field in ["response", "answer", "synthesis_response"]:
        if field in output:
            response_text += " " + str(output.get(field, ""))

    response_lower = response_text.lower()

    # Check for explicit failure indicators
    failure_phrases = ["unable to", "cannot", "failed to", "could not"]
    for phrase in failure_phrases:
        if phrase in response_lower:
            confidence = output.get("confidence", 0)
            if confidence < 0.5:
                return (
                    False,
                    f"Tool composer failed: contains '{phrase}' with confidence {confidence}",
                )

    # Check confidence threshold
    confidence = output.get("confidence", 0)
    if isinstance(confidence, (int, float)) and confidence < 0.3:
        return (False, f"Confidence {confidence} below minimum threshold 0.3")

    # Check for successful tool execution
    tools_executed = output.get("tools_executed", 0)
    tools_succeeded = output.get("tools_succeeded", 0)
    if tools_executed > 0 and tools_succeeded == 0:
        return (False, f"All {tools_executed} tools failed - no successful execution")

    return (True, "Passed semantic validation")


def _validate_prediction_synthesizer(output: dict[str, Any]) -> tuple[bool, str]:
    """Predictions require model diversity or explicit insufficient data warning."""
    models_succeeded = output.get("models_succeeded", 0)
    status = output.get("status", "")

    # If only 1 model, must have warnings about insufficient diversity
    if models_succeeded < 2:
        interpretation = output.get("prediction_interpretation", {})
        recommendations = interpretation.get("recommendations", [])

        # Check if there's an appropriate warning
        has_warning = any(
            "insufficient" in str(r).lower() or "single" in str(r).lower() or "cannot validate" in str(r).lower()
            for r in recommendations
        )
        if not has_warning and status != "failed":
            return (
                False,
                f"Single model ({models_succeeded}) without insufficient data warning is dangerous",
            )

    # Check for dangerous recommendations on zero predictions
    ensemble = output.get("ensemble_prediction", {})
    point_estimate = ensemble.get("point_estimate", -1) if ensemble else -1

    if point_estimate == 0.0 and models_succeeded == 1:
        interpretation = output.get("prediction_interpretation", {})
        reliability = interpretation.get("reliability_assessment", "")
        if reliability not in ("UNVALIDATED", "UNRELIABLE"):
            return (
                False,
                "Zero prediction from single model must be marked UNVALIDATED/UNRELIABLE",
            )

    return (True, "Passed semantic validation")


def _validate_drift_monitor(output: dict[str, Any]) -> tuple[bool, str]:
    """Drift monitor must provide recommended actions for high drift."""
    drift_score = output.get("overall_drift_score", 0)

    # High drift requires recommended actions
    if drift_score > 0.7:
        recommended_actions = output.get("recommended_actions", [])
        if not recommended_actions:
            return (
                False,
                f"High drift ({drift_score:.2f}) detected but no recommended_actions provided",
            )

    return (True, "Passed semantic validation")


def _validate_experiment_designer(output: dict[str, Any]) -> tuple[bool, str]:
    """Experiment designer must calculate real sample sizes, not N/A."""
    # Check top-level fields first, then fall back to nested power_analysis
    required_sample = output.get("required_sample_size")
    power = output.get("statistical_power")

    # Check nested power_analysis if top-level not found
    power_analysis = output.get("power_analysis", {})
    if required_sample is None and power_analysis:
        required_sample = power_analysis.get("required_sample_size")
    if power is None and power_analysis:
        power = power_analysis.get("achieved_power")

    # Check for N/A or None values in critical fields
    if required_sample in (None, "N/A", "n/a", 0):
        return (
            False,
            f"required_sample_size is {required_sample} - must be calculated",
        )

    if power in (None, "N/A", "n/a"):
        return (
            False,
            f"statistical_power is {power} - must be calculated",
        )

    return (True, "Passed semantic validation")


def _validate_health_score(output: dict[str, Any]) -> tuple[bool, str]:
    """Health score must provide diagnostic details for low component scores."""
    component_score = output.get("component_health_score", 1.0)
    overall_score = output.get("overall_health_score", 100)

    # If component score is low, diagnostics should be present
    if component_score and component_score < 0.8:
        # Check all possible diagnostic field names
        diagnostics = (
            output.get("health_diagnosis")
            or output.get("health_diagnostics")
            or output.get("component_details")
        )
        if not diagnostics:
            return (
                False,
                f"Component score {component_score} is degraded but no diagnostics provided",
            )

        # Check that diagnosis has root causes
        if isinstance(diagnostics, dict) and not diagnostics.get("root_causes"):
            return (
                False,
                f"Component score {component_score} is degraded but diagnosis has no root causes",
            )

    return (True, "Passed semantic validation")


def _validate_resource_optimizer(output: dict[str, Any]) -> tuple[bool, str]:
    """Resource optimizer must calculate actual savings, not N/A."""
    projected_savings = output.get("projected_savings")
    projected_roi = output.get("projected_roi")
    status = output.get("status", "")

    # For completed optimization, must have either savings or ROI
    if status in ("completed", "optimal"):
        if projected_savings in (None, "N/A", "n/a") and projected_roi in (None, "N/A", "n/a"):
            return (
                False,
                "Neither projected_savings nor projected_roi calculated for completed optimization",
            )

        # If savings is a dict, check it has meaningful values
        if isinstance(projected_savings, dict) and not any(projected_savings.values()):
            return (
                False,
                "projected_savings dict is empty - must contain actual calculations",
            )

    return (True, "Passed semantic validation")


def _validate_explainer(output: dict[str, Any]) -> tuple[bool, str]:
    """Explainer must surface recommendations in output, not just count them."""
    key_findings_count = output.get("key_findings_count", 0)
    recommendations_count = output.get("recommendations_count", 0)

    if recommendations_count > 0:
        # Check if recommendations are actually surfaced
        executive_summary = output.get("executive_summary", "")
        recommendations = output.get("recommendations") or output.get("recommendations_text")

        if not recommendations and "recommendation" not in executive_summary.lower():
            return (
                False,
                f"Has {recommendations_count} recommendations but none surfaced in output",
            )

    return (True, "Passed semantic validation")


def _validate_heterogeneous_optimizer(output: dict[str, Any]) -> tuple[bool, str]:
    """Heterogeneous optimizer must provide strategic interpretation."""
    overall_ate = output.get("overall_ate")
    heterogeneity_score = output.get("heterogeneity_score")

    # If we have data, we need interpretation
    if overall_ate is not None and heterogeneity_score is not None:
        strategic_interpretation = output.get("strategic_interpretation") or output.get("interpretation")
        if not strategic_interpretation:
            return (
                False,
                f"Has ATE={overall_ate} and heterogeneity={heterogeneity_score} but no strategic interpretation",
            )

    return (True, "Passed semantic validation")


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
        "semantic_validator": _validate_tool_composer,
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
        "semantic_validator": _validate_heterogeneous_optimizer,
    },
    "drift_monitor": {
        "description": "Monitors data and model drift",
        "required_output_fields": ["overall_drift_score"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "overall_drift_score": {"type": "float", "not_null": True},
        },
        "fail_on_status": ["error", "failed"],
        "semantic_validator": _validate_drift_monitor,
    },
    "experiment_designer": {
        "description": "Designs A/B tests with Digital Twin pre-screening",
        "required_output_fields": ["design_type"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "design_type": {"type": "str", "not_null": True},
        },
        "fail_on_status": ["error", "failed"],
        "semantic_validator": _validate_experiment_designer,
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
        "semantic_validator": _validate_health_score,
    },
    "prediction_synthesizer": {
        "description": "Aggregates ML predictions from multiple models",
        "required_output_fields": ["status"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "status": {"type": "str", "not_null": True, "must_not_be": "failed"},
        },
        "fail_on_status": ["failed", "error"],
        "semantic_validator": _validate_prediction_synthesizer,
    },
    "resource_optimizer": {
        "description": "Optimizes resource allocation",
        "required_output_fields": ["status"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "status": {"type": "str", "not_null": True},
        },
        "fail_on_status": ["error", "failed", "infeasible"],
        "semantic_validator": _validate_resource_optimizer,
    },
    "explainer": {
        "description": "Generates natural language explanations",
        "required_output_fields": ["executive_summary"],
        "min_required_fields_pct": 0.4,
        "data_quality_checks": {
            "executive_summary": {"type": "str", "min_length": 10},
        },
        "fail_on_status": ["error", "failed"],
        "semantic_validator": _validate_explainer,
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


def run_semantic_validation(agent_name: str, output: dict[str, Any]) -> tuple[bool, str]:
    """Run semantic validation for an agent's output.

    This validates MEANING, not just structure. It catches issues like:
    - Recommendations on invalid data
    - N/A values for calculated fields
    - Success status with empty outputs

    Args:
        agent_name: Name of the agent
        output: Agent's output dictionary

    Returns:
        (is_valid, reason) tuple where reason explains validation result
    """
    gate = get_quality_gate(agent_name)
    if not gate:
        return (True, f"No quality gate configured for {agent_name}")

    validator = gate.get("semantic_validator")
    if not validator:
        return (True, f"No semantic validator configured for {agent_name}")

    try:
        return validator(output)
    except Exception as e:
        return (False, f"Semantic validation error: {e}")
