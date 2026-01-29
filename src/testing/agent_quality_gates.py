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

import re
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


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict or object attribute.

    This handles both dict-like objects (using .get()) and
    dataclass/object outputs (using getattr()).
    """
    if obj is None:
        return default

    # Try dict-like access first
    if hasattr(obj, "get") and callable(getattr(obj, "get")):
        return obj.get(key, default)

    # Fall back to attribute access
    return getattr(obj, key, default)


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

    # Fail if reporting success but no tools were actually executed
    success = output.get("success", False)
    status = output.get("status", "")
    if success or str(status).lower() in ("success", "completed"):
        if not tools_executed:
            return (
                False,
                "Reports success but tools_executed is 0 - no tools were actually run",
            )

    # Detect fabricated sample sizes (tier0 data has ~600 rows)
    sample_match = re.search(r"sample.size.of.(\d+)", response_text, re.IGNORECASE)
    if sample_match and int(sample_match.group(1)) > 1000:
        return (
            False,
            f"Fabricated sample size of {sample_match.group(1)} (tier0 data has ~600 rows)",
        )

    # Detect fabricated entity IDs (E001, E002, etc.)
    fabricated_ids = re.findall(r"\bE\d{3}\b", response_text)
    if len(fabricated_ids) >= 2:
        return (
            False,
            f"Fabricated entity IDs detected: {fabricated_ids[:5]} - not real data",
        )

    return (True, "Passed semantic validation")


def _validate_prediction_synthesizer(output: dict[str, Any]) -> tuple[bool, str]:
    """Predictions require model diversity or explicit insufficient data warning."""
    models_succeeded = _safe_get(output, "models_succeeded", 0)
    status = _safe_get(output, "status", "")

    # If only 1 model, must have warnings about insufficient diversity
    if models_succeeded < 2:
        interpretation = _safe_get(output, "prediction_interpretation", {}) or {}
        recommendations = _safe_get(interpretation, "recommendations", []) or []
        reliability = _safe_get(interpretation, "reliability_assessment", "")
        anomaly_flags = _safe_get(interpretation, "anomaly_flags", []) or []

        # Check if there's an appropriate warning anywhere
        warning_keywords = ["insufficient", "single", "cannot validate", "unvalidated", "caution"]

        # Check recommendations
        has_warning = any(
            any(kw in str(r).lower() for kw in warning_keywords)
            for r in recommendations
        )

        # Check anomaly flags (which also contain warnings)
        if not has_warning:
            has_warning = any(
                any(kw in str(_safe_get(a, "message", "")).lower() for kw in warning_keywords)
                for a in anomaly_flags
            )

        # Check reliability assessment directly
        if not has_warning and reliability in ("UNVALIDATED", "UNRELIABLE"):
            has_warning = True

        # Check state-level warnings
        if not has_warning:
            state_warnings = _safe_get(output, "warnings", []) or []
            has_warning = any(
                any(kw in str(w).lower() for kw in warning_keywords)
                for w in state_warnings
            )

        if not has_warning and status != "failed":
            return (
                False,
                f"Single model ({models_succeeded}) without insufficient data warning is dangerous",
            )

    # Check for dangerous recommendations on zero predictions
    ensemble = _safe_get(output, "ensemble_prediction", {})
    point_estimate = _safe_get(ensemble, "point_estimate", -1) if ensemble else -1

    if point_estimate == 0.0 and models_succeeded == 1:
        interpretation = _safe_get(output, "prediction_interpretation", {}) or {}
        reliability = _safe_get(interpretation, "reliability_assessment", "")
        if reliability not in ("UNVALIDATED", "UNRELIABLE"):
            return (
                False,
                "Zero prediction from single model must be marked UNVALIDATED/UNRELIABLE",
            )

    # Fail on CANNOT_ASSESS in critical prediction fields
    prediction_summary = str(_safe_get(output, "prediction_summary", ""))
    risk_assessment = str(_safe_get(output, "risk_assessment", ""))
    if "CANNOT_ASSESS" in prediction_summary or "CANNOT_ASSESS" in risk_assessment:
        return (
            False,
            "CANNOT_ASSESS in prediction output - agent unable to produce useful predictions",
        )

    # Fail on zero prediction with zero context (no real analysis occurred)
    similar_cases = _safe_get(output, "similar_cases", None)
    historical_accuracy = _safe_get(output, "historical_accuracy", None)
    if (
        point_estimate == 0.0
        and (not similar_cases or similar_cases == [])
        and (historical_accuracy == 0.0 or historical_accuracy is None)
    ):
        return (
            False,
            "Zero prediction with no similar cases and no historical accuracy - no real analysis",
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

    # Completed drift monitoring must cover at least 2 of 3 drift types
    status = output.get("status", "")
    if status in ("completed", "success"):
        drift_types_with_results = 0
        for drift_key in ("data_drift_results", "model_drift_results", "concept_drift_results"):
            results = output.get(drift_key)
            if results and results != {} and results != []:
                drift_types_with_results += 1
        if drift_types_with_results < 2:
            return (
                False,
                f"Only {drift_types_with_results}/3 drift types have results - "
                "must analyze at least 2 types (data, model, concept)",
            )

    return (True, "Passed semantic validation")


def _validate_experiment_designer(output: dict[str, Any]) -> tuple[bool, str]:
    """Experiment designer must calculate real sample sizes, not N/A."""
    # Check top-level fields first, then fall back to nested power_analysis
    # Use _safe_get to handle both dict and dataclass outputs
    required_sample = _safe_get(output, "required_sample_size")
    power = _safe_get(output, "statistical_power")

    # Check nested power_analysis if top-level not found
    power_analysis = _safe_get(output, "power_analysis", {})
    if required_sample is None and power_analysis:
        required_sample = _safe_get(power_analysis, "required_sample_size")
        if required_sample is None:
            required_sample = _safe_get(power_analysis, "sample_size")
    if power is None and power_analysis:
        power = _safe_get(power_analysis, "achieved_power")
        if power is None:
            power = _safe_get(power_analysis, "power")

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
    component_score = _safe_get(output, "component_health_score", 1.0)
    overall_score = _safe_get(output, "overall_health_score", 100)

    # If component score is low, some form of diagnostic info should be present
    if component_score and component_score < 0.8:
        # Check all possible diagnostic field names - use _safe_get for object/dict compatibility
        diagnostics = (
            _safe_get(output, "health_diagnosis")
            or _safe_get(output, "health_diagnostics")
            or _safe_get(output, "component_details")
            or _safe_get(output, "component_health_details")
            or _safe_get(output, "diagnostics")
        )

        # Also check if there's a detailed breakdown in the health checks
        if not diagnostics:
            health_checks = _safe_get(output, "health_checks", {})
            if health_checks:
                diagnostics = health_checks  # Health checks can serve as diagnostics

        # Critical issues and warnings also count as diagnostics
        critical_issues = _safe_get(output, "critical_issues", [])
        warnings = _safe_get(output, "warnings", [])
        health_summary = _safe_get(output, "health_summary", "")

        # Accept any of the following as valid diagnostics:
        # 1. Explicit health_diagnosis or similar fields
        # 2. Non-empty critical_issues list
        # 3. Non-empty warnings list
        # 4. Health summary containing diagnostic info (> 50 chars with "issue" or "degraded")
        has_diagnostics = bool(diagnostics)
        has_issues = bool(critical_issues)
        has_warnings = bool(warnings)
        has_summary_diagnostics = (
            len(health_summary) > 50
            and ("issue" in health_summary.lower() or "degraded" in health_summary.lower())
        )

        if not (has_diagnostics or has_issues or has_warnings or has_summary_diagnostics):
            return (
                False,
                f"Component score {component_score} is degraded but no diagnostics provided",
            )

        # If we have explicit diagnosis with expected structure, validate root causes
        if diagnostics and isinstance(diagnostics, dict) and "root_causes" in diagnostics:
            if not diagnostics.get("root_causes"):
                # Only fail if we don't have other diagnostic sources
                if not (has_issues or has_warnings):
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

    # Completed optimization with negligible ROI indicates no real optimization occurred
    if status in ("completed", "optimal") and isinstance(projected_roi, (int, float)):
        if projected_roi < 0.05:
            return (
                False,
                f"Projected ROI of {projected_roi:.2%} is below 5% threshold - "
                "optimization produced negligible value",
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

    # Reject meta-descriptions instead of actual explanations
    exec_summary = output.get("executive_summary", "")
    if (
        re.match(r"^Analysis complete with \d+ findings", exec_summary)
        and len(exec_summary) < 100
    ):
        return (
            False,
            "Executive summary is a meta-description, not an actual explanation: "
            f"'{exec_summary[:80]}...'",
        )

    # Reject output with excessive raw unformatted floats (indicates no real formatting)
    combined_text = exec_summary + " " + str(output.get("detailed_explanation", ""))
    raw_float_matches = re.findall(r"\d+\.\d{6,}", combined_text)
    if len(raw_float_matches) >= 3:
        return (
            False,
            f"Found {len(raw_float_matches)} unformatted raw floats (e.g., {raw_float_matches[0]}) - "
            "explainer must format numbers for human readability",
        )

    return (True, "Passed semantic validation")


def _validate_heterogeneous_optimizer(output: dict[str, Any]) -> tuple[bool, str]:
    """Heterogeneous optimizer must provide strategic interpretation."""
    overall_ate = _safe_get(output, "overall_ate")
    heterogeneity_score = _safe_get(output, "heterogeneity_score")

    # If we have data, we need interpretation
    if overall_ate is not None and heterogeneity_score is not None:
        # Check multiple possible interpretation fields
        strategic_interpretation = (
            _safe_get(output, "strategic_interpretation")
            or _safe_get(output, "interpretation")
            or _safe_get(output, "key_insights")  # key_insights can serve as interpretation
        )

        # Also check executive_summary which contains interpretation
        if not strategic_interpretation:
            exec_summary = _safe_get(output, "executive_summary", "")
            if exec_summary and len(exec_summary) > 50:
                # Executive summary with substantial content can serve as interpretation
                strategic_interpretation = exec_summary

        if not strategic_interpretation:
            return (
                False,
                f"Has ATE={overall_ate} and heterogeneity={heterogeneity_score} but no strategic interpretation",
            )

    # Completed optimization must identify at least some responder segments
    het_status = _safe_get(output, "status", "")
    if het_status in ("completed", "success"):
        high_responders = _safe_get(output, "high_responders", None)
        low_responders = _safe_get(output, "low_responders", None)
        if (
            (high_responders is not None and not high_responders)
            and (low_responders is not None and not low_responders)
        ):
            return (
                False,
                "Both high_responders and low_responders are empty - "
                "no segment differentiation found",
            )

    return (True, "Passed semantic validation")


def _validate_feedback_learner(output: dict[str, Any]) -> tuple[bool, str]:
    """Feedback learner must actually learn something when status is completed."""
    status = output.get("status", "")

    # "partial" status is honest - insufficient data acknowledged
    if status == "partial":
        return (True, "Partial status accepted - insufficient data acknowledged")

    # For completed status, at least ONE learning activity must have occurred
    if status in ("completed", "success"):
        # Check list-based outputs
        detected_patterns = output.get("detected_patterns", [])
        learning_recommendations = output.get("learning_recommendations", [])
        applied_updates = output.get("applied_updates", [])
        feedback_items = output.get("feedback_items", [])

        has_list_content = any(
            bool(lst)
            for lst in [
                detected_patterns,
                learning_recommendations,
                applied_updates,
                feedback_items,
            ]
        )

        # Check counter-based outputs
        feedback_count = output.get("feedback_count", 0)
        pattern_count = output.get("pattern_count", 0)
        has_counter_content = (
            isinstance(feedback_count, (int, float)) and feedback_count > 0
        ) or (isinstance(pattern_count, (int, float)) and pattern_count > 0)

        if not has_list_content and not has_counter_content:
            return (
                False,
                "Completed with 0 items processed, 0 patterns, 0 recommendations - "
                "no learning occurred",
            )

    return (True, "Passed semantic validation")


def _validate_orchestrator(output: dict[str, Any]) -> tuple[bool, str]:
    """Orchestrator must dispatch unique agents and produce substantive responses."""
    status = output.get("status", "")

    if status in ("completed", "success"):
        # agents_dispatched must be non-empty
        agents_dispatched = output.get("agents_dispatched", [])
        if isinstance(agents_dispatched, list) and not agents_dispatched:
            return (
                False,
                "Completed with no agents dispatched - orchestrator did not route to any agent",
            )

        # No duplicate agent dispatches
        if isinstance(agents_dispatched, list) and len(agents_dispatched) > 0:
            if len(set(agents_dispatched)) < len(agents_dispatched):
                duplicates = [
                    a for a in set(agents_dispatched) if agents_dispatched.count(a) > 1
                ]
                return (
                    False,
                    f"Duplicate agents dispatched: {duplicates} - "
                    "orchestrator should not dispatch the same agent twice",
                )

    # Response must be substantive (>=50 chars)
    response_text = output.get("response_text", "")
    if isinstance(response_text, str) and len(response_text) < 50:
        return (
            False,
            f"Response text is only {len(response_text)} chars - "
            "must be >=50 chars for substantive content",
        )

    return (True, "Passed semantic validation")


def _validate_causal_impact(output: dict[str, Any]) -> tuple[bool, str]:
    """Causal impact must produce valid ATE within its own confidence interval."""
    status = output.get("status", "")

    if status in ("completed", "success"):
        ate_estimate = _safe_get(output, "ate_estimate")

        # ATE must be numeric
        if ate_estimate is None or not isinstance(ate_estimate, (int, float)):
            return (
                False,
                f"ate_estimate is {ate_estimate} - must be a numeric value",
            )

        # Confidence interval must be a 2-element sequence
        ci = _safe_get(output, "confidence_interval")
        if ci is None or not isinstance(ci, (list, tuple)) or len(ci) != 2:
            return (
                False,
                f"confidence_interval is {ci} - must be a 2-element list/tuple [lower, upper]",
            )

        # ATE must fall within its own CI (basic sanity check)
        ci_lower, ci_upper = ci[0], ci[1]
        if isinstance(ci_lower, (int, float)) and isinstance(ci_upper, (int, float)):
            if not (ci_lower <= ate_estimate <= ci_upper):
                return (
                    False,
                    f"ATE {ate_estimate} falls outside its own CI [{ci_lower}, {ci_upper}]",
                )

    return (True, "Passed semantic validation")


def _validate_gap_analyzer(output: dict[str, Any]) -> tuple[bool, str]:
    """Gap analyzer must find actionable opportunities with real value."""
    status = output.get("status", "")

    if status in ("completed", "success"):
        # Must identify at least one opportunity
        opportunities = output.get("prioritized_opportunities", [])
        if not opportunities:
            return (
                False,
                "Completed with no prioritized_opportunities - no gaps identified",
            )

        # Total addressable value must be positive
        total_value = output.get("total_addressable_value", 0)
        if isinstance(total_value, (int, float)) and total_value <= 0:
            return (
                False,
                f"total_addressable_value is {total_value} - must be > 0 for completed analysis",
            )

        # Executive summary must be substantive
        exec_summary = output.get("executive_summary", "")
        if isinstance(exec_summary, str) and len(exec_summary) < 50:
            return (
                False,
                f"executive_summary is only {len(exec_summary)} chars - "
                "must be >=50 chars for actionable insights",
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
            "response_text": {"type": "str", "min_length": 50},
        },
        "fail_on_status": ["error", "failed"],
        "semantic_validator": _validate_orchestrator,
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
        "semantic_validator": _validate_causal_impact,
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
        "semantic_validator": _validate_gap_analyzer,
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
        "semantic_validator": _validate_feedback_learner,
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
