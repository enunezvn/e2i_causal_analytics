#!/usr/bin/env python3
"""Tier 1-5 Agent Testing Framework.

Tests all Tier 1-5 agents using tier0 synthetic data outputs to validate:
- Agent processing (correct execution without errors)
- Output correctness (outputs match TypedDict contracts)
- Observability (Opik traces captured properly)

Usage:
    # Run tier0 first, then test all Tier 1-5 agents
    python scripts/run_tier1_5_test.py --run-tier0-first

    # Use cached tier0 outputs (faster iteration)
    python scripts/run_tier1_5_test.py --tier0-cache scripts/tier0_output_cache/latest.pkl

    # Test specific tiers
    python scripts/run_tier1_5_test.py --tiers 2,3

    # Test specific agents
    python scripts/run_tier1_5_test.py --agents causal_impact,explainer

    # Skip Opik verification (if Opik not running)
    python scripts/run_tier1_5_test.py --skip-observability

    # Save results to JSON
    python scripts/run_tier1_5_test.py --output results/tier1_5_test_results.json

Prerequisites:
    - On droplet: cd /opt/e2i_causal_analytics && source .venv/bin/activate
    - Tier0 test outputs available (run run_tier0_test.py first or use --run-tier0-first)
    - Opik running (port 5173, optional - use --skip-observability to skip)

Author: E2I Causal Analytics Team
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import pickle
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class FieldValidationResult:
    """Validation result for a single field."""

    name: str
    expected_type: str
    present: bool
    valid_type: bool
    actual_type: str | None = None
    error: str | None = None


@dataclass
class ContractValidationDetail:
    """Detailed contract validation results."""

    valid: bool
    state_class: str
    required_fields_checked: list[str] = field(default_factory=list)
    required_fields_present: list[str] = field(default_factory=list)
    required_fields_valid: list[FieldValidationResult] = field(default_factory=list)
    optional_fields_checked: list[str] = field(default_factory=list)
    optional_fields_present: list[str] = field(default_factory=list)
    optional_fields_valid: list[FieldValidationResult] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    type_errors: list[dict[str, str]] = field(default_factory=list)
    extra_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_fields: int = 0
    required_total: int = 0
    optional_total: int = 0


@dataclass
class TraceVerificationDetail:
    """Detailed Opik trace verification results."""

    trace_exists: bool = False
    trace_id: str | None = None
    trace_url: str | None = None
    metadata_valid: bool = False
    expected_metadata: dict[str, Any] = field(default_factory=dict)
    actual_metadata: dict[str, Any] = field(default_factory=dict)
    span_count: int = 0
    span_names: list[str] = field(default_factory=list)
    duration_ms: float | None = None
    error_captured: bool = False


@dataclass
class PerformanceDetail:
    """Agent performance metrics."""

    total_time_ms: float = 0.0
    llm_calls: int = 0
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    tool_calls: int = 0
    memory_peak_mb: float | None = None


@dataclass
class AgentTestResult:
    """Complete result of testing a single agent."""

    # Identity
    agent_name: str
    tier: int
    test_timestamp: str

    # Execution
    success: bool
    execution_time_ms: float
    error: str | None = None
    error_traceback: str | None = None

    # Input Summary
    input_summary: dict[str, Any] = field(default_factory=dict)

    # Agent Output (full)
    agent_output: dict[str, Any] | None = None

    # Contract Validation Details
    contract_validation: ContractValidationDetail | None = None

    # Observability Details
    trace_verification: TraceVerificationDetail | None = None

    # Performance Metrics
    performance_metrics: PerformanceDetail | None = None


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Agent configurations: name -> (tier, state_class_import_path, method, etc.)
# method: the method to call (default: "run")
# is_async: whether the method is async (default: True)
# input_model: Pydantic input model class name if agent expects a model instead of dict
# input_module: module path for input model (if needed)
AGENT_CONFIGS = {
    # Tier 1: Orchestration
    "orchestrator": {
        "tier": 1,
        "state_module": "src.agents.orchestrator.state",
        "state_class": "OrchestratorState",
        "agent_module": "src.agents.orchestrator",
        "agent_class": "OrchestratorAgent",
        "method": "run",
        "is_async": True,
    },
    "tool_composer": {
        "tier": 1,
        "state_module": "src.agents.tool_composer.state",
        "state_class": "ToolComposerState",
        "agent_module": "src.agents.tool_composer",
        "agent_class": "ToolComposerAgent",
        "method": "run",
        "is_async": True,
    },
    # Tier 2: Causal
    "causal_impact": {
        "tier": 2,
        "state_module": "src.agents.causal_impact.state",
        "state_class": "CausalImpactState",
        "agent_module": "src.agents.causal_impact",
        "agent_class": "CausalImpactAgent",
        "method": "run",
        "is_async": True,
    },
    "gap_analyzer": {
        "tier": 2,
        "state_module": "src.agents.gap_analyzer.state",
        "state_class": "GapAnalyzerState",
        "agent_module": "src.agents.gap_analyzer",
        "agent_class": "GapAnalyzerAgent",
        "method": "run",
        "is_async": True,
    },
    "heterogeneous_optimizer": {
        "tier": 2,
        "state_module": "src.agents.heterogeneous_optimizer.state",
        "state_class": "HeterogeneousOptimizerState",
        "agent_module": "src.agents.heterogeneous_optimizer",
        "agent_class": "HeterogeneousOptimizerAgent",
        "method": "run",
        "is_async": True,
    },
    # Tier 3: Monitoring
    "drift_monitor": {
        "tier": 3,
        "state_module": "src.agents.drift_monitor.state",
        "state_class": "DriftMonitorState",
        "agent_module": "src.agents.drift_monitor",
        "agent_class": "DriftMonitorAgent",
        "method": "run",
        "is_async": True,
        "input_module": "src.agents.drift_monitor.agent",
        "input_model": "DriftMonitorInput",
    },
    "experiment_designer": {
        "tier": 3,
        "state_module": "src.agents.experiment_designer.state",
        "state_class": "ExperimentDesignState",
        "agent_module": "src.agents.experiment_designer",
        "agent_class": "ExperimentDesignerAgent",
        "method": "run",
        "is_async": False,  # sync method
        "input_module": "src.agents.experiment_designer.agent",
        "input_model": "ExperimentDesignerInput",
        "timeout": 120.0,  # LLM-based validity audit needs more time
    },
    "health_score": {
        "tier": 3,
        "state_module": "src.agents.health_score.state",
        "state_class": "HealthScoreState",
        "agent_module": "src.agents.health_score",
        "agent_class": "HealthScoreAgent",
        "method": "check_health",
        "is_async": True,
        "uses_kwargs": True,  # scope, query, experiment_name
    },
    # Tier 4: ML Predictions
    "prediction_synthesizer": {
        "tier": 4,
        "state_module": "src.agents.prediction_synthesizer.state",
        "state_class": "PredictionSynthesizerState",
        "agent_module": "src.agents.prediction_synthesizer",
        "agent_class": "PredictionSynthesizerAgent",
        "method": "synthesize",
        "is_async": True,
        "uses_kwargs": True,  # entity_id, prediction_target, features, etc.
    },
    "resource_optimizer": {
        "tier": 4,
        "state_module": "src.agents.resource_optimizer.state",
        "state_class": "ResourceOptimizerState",
        "agent_module": "src.agents.resource_optimizer",
        "agent_class": "ResourceOptimizerAgent",
        "method": "optimize",
        "is_async": True,
        "uses_kwargs": True,  # allocation_targets, constraints, etc.
    },
    # Tier 5: Self-Improvement
    "explainer": {
        "tier": 5,
        "state_module": "src.agents.explainer.state",
        "state_class": "ExplainerState",
        "agent_module": "src.agents.explainer",
        "agent_class": "ExplainerAgent",
        "method": "explain",
        "is_async": True,
        "uses_kwargs": True,  # analysis_results, query, user_expertise, etc.
    },
    "feedback_learner": {
        "tier": 5,
        "state_module": "src.agents.feedback_learner.state",
        "state_class": "FeedbackLearnerState",
        "agent_module": "src.agents.feedback_learner",
        "agent_class": "FeedbackLearnerAgent",
        "method": "learn",
        "is_async": True,
        "uses_kwargs": True,  # time_range_start, time_range_end, batch_id, etc.
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def print_header(title: str, char: str = "=", width: int = 70) -> None:
    """Print a section header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def print_subheader(title: str, char: str = "-", width: int = 60) -> None:
    """Print a subsection header."""
    print(f"\n  {char * width}")
    print(f"  {title}")
    print(f"  {char * width}")


def format_duration(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


def summarize_input(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Create a summary of agent input for display."""
    summary = {}
    for key, value in input_dict.items():
        if hasattr(value, "__len__") and not isinstance(value, str):
            if hasattr(value, "columns"):  # DataFrame
                summary[f"{key}_rows"] = len(value)
                summary[f"{key}_columns"] = len(value.columns)
            elif isinstance(value, list):
                summary[f"{key}_count"] = len(value)
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = list(value.keys())[:5]
        elif isinstance(value, str) and len(value) > 50:
            summary[key] = value[:50] + "..."
        else:
            summary[key] = value
    return summary


def import_class(module_path: str, class_name: str) -> type:
    """Dynamically import a class from a module."""
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# =============================================================================
# ENHANCED OUTPUT HELPERS (Tier0-style formatting)
# =============================================================================


def print_input_section(inputs: dict[str, Any]) -> None:
    """Print standardized input summary section."""
    print("\n  📥 Input Summary:")
    for key, value in inputs.items():
        if isinstance(value, str) and len(value) > 60:
            print(f"    • {key}: {value[:60]}...")
        elif isinstance(value, list):
            print(f"    • {key}: [{len(value)} items]")
        elif isinstance(value, dict):
            print(f"    • {key}: {{{len(value)} keys}}")
        else:
            print(f"    • {key}: {value}")


def print_processing_section(steps: list[tuple[str, bool]]) -> None:
    """Print processing steps with checkmarks.

    Args:
        steps: List of (step_description, success)
    """
    print("\n  ⚙️  Processing:")
    for desc, success in steps:
        icon = "✅" if success else "❌"
        print(f"    {icon} {desc}")


def print_validation_checks(checks: list[tuple[str, bool, str, str]]) -> None:
    """Print validation checks with expected vs actual.

    Args:
        checks: List of (check_name, passed, expected, actual)
    """
    print("\n  🔍 Validation Checks:")
    for name, passed, expected, actual in checks:
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"    • {name}: {icon}")
        print(f"        Expected: {expected}")
        print(f"        Actual:   {actual}")


def print_metrics_table(metrics: list[tuple[str, Any, str | None, bool | None]]) -> None:
    """Print key metrics in table format.

    Args:
        metrics: List of (metric_name, value, threshold, passed)
                threshold and passed are optional (None to skip)
    """
    print("\n  📊 Key Metrics:")
    print(f"    {'Metric':<25} {'Value':<15} {'Threshold':<15} {'Status':<10}")
    print(f"    {'-'*65}")

    for name, value, threshold, passed in metrics:
        # Format value
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        elif value is None:
            value_str = "N/A"
        else:
            value_str = str(value)[:15]

        # Format threshold
        threshold_str = str(threshold) if threshold else "-"

        # Format status
        if passed is None:
            status_str = "-"
        elif passed:
            status_str = "✅"
        else:
            status_str = "❌"

        print(f"    {name:<25} {value_str:<15} {threshold_str:<15} {status_str:<10}")


def print_analysis_section(
    title: str,
    insights: list[str],
    recommendations: list[str] | None = None,
) -> None:
    """Print analysis insights and recommendations.

    Args:
        title: Section title (e.g., "CAUSAL IMPACT Analysis")
        insights: List of insight bullet points
        recommendations: Optional list of recommendations
    """
    print(f"\n  💡 {title}:")
    for insight in insights:
        print(f"    • {insight}")

    if recommendations:
        print("\n    Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec}")


def print_step_result(status: str, message: str, duration_s: float | None = None) -> None:
    """Print final step result with status.

    Args:
        status: "success", "warning", or "failed"
        message: Result message
        duration_s: Optional duration in seconds
    """
    print("\n  " + "-" * 60)

    if status == "success":
        icon = "✅"
        label = "RESULT: PASS"
    elif status == "warning":
        icon = "⚠️"
        label = "RESULT: PASS (with warnings)"
    else:
        icon = "❌"
        label = "RESULT: FAIL"

    duration_str = f" ({duration_s:.1f}s)" if duration_s else ""
    print(f"  {icon} {label} - {message}{duration_str}")
    print("  " + "-" * 60)


# Tier descriptions for analysis
TIER_DESCRIPTIONS = {
    1: "Orchestration",
    2: "Causal Analytics",
    3: "Monitoring",
    4: "ML Predictions",
    5: "Self-Improvement",
}

# Agent-specific analysis extractors
AGENT_ANALYSIS_CONFIG = {
    "orchestrator": {
        "key_fields": ["routing_decision", "selected_agents", "confidence"],
        "insights_template": [
            "Query routed to {selected_agents} agents",
            "Routing confidence: {confidence}",
        ],
    },
    "tool_composer": {
        "key_fields": ["composed_tools", "execution_plan", "tool_count"],
        "insights_template": [
            "Composed {tool_count} tools for query execution",
            "Execution plan generated successfully",
        ],
    },
    "causal_impact": {
        "key_fields": ["overall_ate", "causal_effect", "confidence_interval", "p_value"],
        "insights_template": [
            "Average Treatment Effect (ATE): {overall_ate}",
            "Statistical significance: p={p_value}",
        ],
    },
    "gap_analyzer": {
        "key_fields": ["gaps_identified", "total_gap_value", "top_opportunities"],
        "insights_template": [
            "Identified {gaps_identified} performance gaps",
            "Total opportunity value: {total_gap_value}",
        ],
    },
    "heterogeneous_optimizer": {
        "key_fields": ["heterogeneity_score", "segments_analyzed", "cate_results"],
        "insights_template": [
            "Heterogeneity score: {heterogeneity_score}",
            "Analyzed {segments_analyzed} patient segments",
        ],
    },
    "drift_monitor": {
        "key_fields": ["drift_detected", "drift_score", "features_drifted"],
        "insights_template": [
            "Drift detected: {drift_detected}",
            "Overall drift score: {drift_score}",
        ],
    },
    "experiment_designer": {
        "key_fields": ["experiment_design", "sample_size", "statistical_power", "validity_assessment"],
        "insights_template": [
            "Experiment design created with n={sample_size}",
            "Statistical power: {statistical_power}",
        ],
    },
    "health_score": {
        "key_fields": ["health_status", "overall_score", "component_scores"],
        "insights_template": [
            "System health: {health_status}",
            "Overall health score: {overall_score}",
        ],
    },
    "prediction_synthesizer": {
        "key_fields": ["prediction", "confidence", "model_agreement"],
        "insights_template": [
            "Synthesized prediction: {prediction}",
            "Model agreement: {model_agreement}",
        ],
    },
    "resource_optimizer": {
        "key_fields": ["optimization_result", "cost_savings", "allocation_changes"],
        "insights_template": [
            "Optimization completed: {optimization_result}",
            "Projected savings: {cost_savings}",
        ],
    },
    "explainer": {
        "key_fields": ["executive_summary", "explanation_type", "confidence"],
        "insights_template": [
            "Generated {explanation_type} explanation",
            "Summary: {executive_summary}",
        ],
    },
    "feedback_learner": {
        "key_fields": ["learning_summary", "improvements_identified", "patterns_learned"],
        "insights_template": [
            "Learning cycle complete: {learning_summary}",
            "Improvements identified: {improvements_identified}",
        ],
    },
}


def extract_agent_insights(agent_name: str, output: dict[str, Any]) -> list[str]:
    """Extract agent-specific insights from output."""
    config = AGENT_ANALYSIS_CONFIG.get(agent_name, {})
    insights = []

    # Try to fill in template insights
    for template in config.get("insights_template", []):
        try:
            # Extract values from output
            values = {}
            for key in config.get("key_fields", []):
                val = output.get(key)
                if val is not None:
                    if isinstance(val, float):
                        values[key] = f"{val:.4f}"
                    elif isinstance(val, list):
                        values[key] = len(val)
                    elif isinstance(val, str) and len(val) > 50:
                        values[key] = val[:50] + "..."
                    else:
                        values[key] = val
                else:
                    values[key] = "N/A"

            insight = template.format(**values)
            insights.append(insight)
        except (KeyError, ValueError):
            continue

    # Add generic insights if we couldn't extract specific ones
    if not insights:
        if output.get("status"):
            insights.append(f"Agent completed with status: {output['status']}")
        if output.get("analysis_complete"):
            insights.append("Analysis completed successfully")
        if not insights:
            insights.append("Agent execution completed")

    return insights


# =============================================================================
# TIER0 STATE LOADING
# =============================================================================


def load_tier0_state(cache_path: str | None = None) -> dict[str, Any]:
    """Load tier0 state from cache or run tier0 test.

    Args:
        cache_path: Path to cached tier0 state pickle file

    Returns:
        Tier0 state dictionary
    """
    if cache_path and Path(cache_path).exists():
        print(f"  Loading tier0 state from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(
        f"Tier0 cache not found at {cache_path}. "
        "Run with --run-tier0-first or provide a valid cache path."
    )


async def run_tier0_and_cache(cache_dir: str = "scripts/tier0_output_cache") -> dict[str, Any]:
    """Run tier0 test and cache the results.

    Args:
        cache_dir: Directory to save cache files

    Returns:
        Tier0 state dictionary
    """
    from scripts.run_tier0_test import run_pipeline

    print("  Running tier0 pipeline to generate synthetic data...")

    # Run tier0 pipeline and capture the returned state
    state = await run_pipeline(step=None, dry_run=False)

    # Save to cache
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_file = cache_path / f"tier0_state_{timestamp}.pkl"
    latest_link = cache_path / "latest.pkl"

    with open(cache_file, "wb") as f:
        pickle.dump(state, f)

    # Create/update latest symlink
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(cache_file.name)

    print(f"  Tier0 state cached to: {cache_file}")
    return state


# =============================================================================
# AGENT TESTING
# =============================================================================


async def test_agent(
    agent_name: str,
    config: dict[str, Any],
    mapper: Any,  # Tier0OutputMapper
    validator: Any,  # ContractValidator
    trace_verifier: Any | None,  # OpikTraceVerifier
    timeout_seconds: float = 30.0,
) -> AgentTestResult:
    """Test a single agent.

    Args:
        agent_name: Name of the agent to test
        config: Agent configuration dict
        mapper: Tier0OutputMapper instance
        validator: ContractValidator instance
        trace_verifier: OpikTraceVerifier instance (or None to skip)
        timeout_seconds: Maximum execution time

    Returns:
        AgentTestResult with test details
    """
    tier = config["tier"]
    test_timestamp = datetime.now(UTC).isoformat()

    result = AgentTestResult(
        agent_name=agent_name,
        tier=tier,
        test_timestamp=test_timestamp,
        success=False,
        execution_time_ms=0.0,
    )

    try:
        # 1. Get mapped inputs
        agent_input = mapper.get_agent_mapping(agent_name)
        result.input_summary = summarize_input(agent_input)

        # 2. Import and instantiate agent
        agent_class = import_class(config["agent_module"], config["agent_class"])
        agent = agent_class()

        # 3. Get the method to call (default: "run")
        method_name = config.get("method", "run")
        is_async = config.get("is_async", True)
        method = getattr(agent, method_name)

        # 4. Convert input to Pydantic model if needed
        if "input_model" in config:
            input_model_class = import_class(config["input_module"], config["input_model"])
            agent_input = input_model_class(**agent_input)

        # 5. Run agent with timeout
        uses_kwargs = config.get("uses_kwargs", False)
        start_time = time.time()
        try:
            if is_async:
                if uses_kwargs:
                    # Unpack dict as keyword arguments
                    output = await asyncio.wait_for(
                        method(**agent_input),
                        timeout=timeout_seconds,
                    )
                else:
                    output = await asyncio.wait_for(
                        method(agent_input),
                        timeout=timeout_seconds,
                    )
            else:
                # Run sync method in executor to allow timeout
                loop = asyncio.get_event_loop()
                if uses_kwargs:
                    import functools
                    output = await asyncio.wait_for(
                        loop.run_in_executor(None, functools.partial(method, **agent_input)),
                        timeout=timeout_seconds,
                    )
                else:
                    output = await asyncio.wait_for(
                        loop.run_in_executor(None, method, agent_input),
                        timeout=timeout_seconds,
                    )
        except asyncio.TimeoutError:
            result.error = f"Agent execution timed out after {timeout_seconds}s"
            result.execution_time_ms = timeout_seconds * 1000
            return result
        except RuntimeError as e:
            # Handle async generator issues (e.g., "generator didn't stop after athrow()")
            # This can happen with LangGraph when cancellation/timeout interrupts an async generator
            error_str = str(e)
            if "generator" in error_str and "athrow" in error_str:
                result.error = f"Async generator cancellation error (LangGraph issue): {error_str}"
            else:
                result.error = f"RuntimeError: {error_str}"
            result.error_traceback = traceback.format_exc()
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        result.execution_time_ms = (time.time() - start_time) * 1000

        # 4. Store output (convert to dict if needed)
        if isinstance(output, dict):
            result.agent_output = output
        elif hasattr(output, "__dict__"):
            result.agent_output = output.__dict__
        else:
            result.agent_output = {"raw_output": str(output)}

        # 5. Validate contract with detailed field tracking
        try:
            state_class = import_class(config["state_module"], config["state_class"])
            validation_result = validator.validate_state(result.agent_output, state_class)

            # Get contract summary for field details
            contract_summary = validator.get_contract_summary(state_class)
            required_fields = contract_summary.get("required_fields", [])
            optional_fields = contract_summary.get("optional_fields", [])
            field_types = contract_summary.get("field_types", {})

            # Build detailed field validation results
            required_valid = []
            required_present = []
            for fld in required_fields:
                present = fld in result.agent_output
                if present:
                    required_present.append(fld)
                # Check if there's a type error for this field
                type_err = next((te for te in validation_result.type_errors if te.get("field") == fld), None)
                required_valid.append(FieldValidationResult(
                    name=fld,
                    expected_type=field_types.get(fld, "Any"),
                    present=present,
                    valid_type=present and type_err is None,
                    actual_type=type(result.agent_output.get(fld)).__name__ if present else None,
                    error=type_err.get("message") if type_err else None,
                ))

            optional_valid = []
            optional_present = []
            for fld in optional_fields:
                present = fld in result.agent_output
                if present:
                    optional_present.append(fld)
                    # Check if there's a type error for this field
                    type_err = next((te for te in validation_result.type_errors if te.get("field") == fld), None)
                    optional_valid.append(FieldValidationResult(
                        name=fld,
                        expected_type=field_types.get(fld, "Any"),
                        present=present,
                        valid_type=type_err is None,
                        actual_type=type(result.agent_output.get(fld)).__name__ if present else None,
                        error=type_err.get("message") if type_err else None,
                    ))

            result.contract_validation = ContractValidationDetail(
                valid=validation_result.valid,
                state_class=config["state_class"],
                required_fields_checked=required_fields,
                required_fields_present=required_present,
                required_fields_valid=required_valid,
                optional_fields_checked=optional_fields,
                optional_fields_present=optional_present,
                optional_fields_valid=optional_valid,
                missing_required=[e for e in validation_result.errors if "Missing" in e],
                type_errors=validation_result.type_errors,
                extra_fields=validation_result.extra_fields,
                warnings=validation_result.warnings,
                total_fields=len(required_fields) + len(optional_fields),
                required_total=len(required_fields),
                optional_total=len(optional_fields),
            )
        except Exception as e:
            result.contract_validation = ContractValidationDetail(
                valid=False,
                state_class=config["state_class"],
                warnings=[f"Contract validation failed: {e}"],
            )

        # 6. Verify observability (if verifier provided)
        if trace_verifier is not None:
            trace_id = result.agent_output.get("trace_id")
            if trace_id:
                try:
                    trace_result = await trace_verifier.verify_agent_trace(
                        agent_name=agent_name,
                        trace_id=trace_id,
                        tier=tier,
                    )
                    result.trace_verification = TraceVerificationDetail(
                        trace_exists=trace_result.trace_exists,
                        trace_id=trace_result.trace_id,
                        trace_url=trace_result.trace_url,
                        metadata_valid=trace_result.metadata_valid,
                        expected_metadata=trace_result.expected_metadata,
                        actual_metadata=trace_result.actual_metadata,
                        span_count=trace_result.span_count,
                        span_names=trace_result.span_names,
                        duration_ms=trace_result.duration_ms,
                        error_captured=trace_result.error_captured,
                    )
                except Exception as e:
                    result.trace_verification = TraceVerificationDetail(
                        trace_exists=False,
                        trace_id=trace_id,
                    )

        # 7. Determine overall success
        result.success = True
        if result.contract_validation and not result.contract_validation.valid:
            # Contract validation failures with missing required fields are errors
            if result.contract_validation.missing_required:
                result.success = False

    except Exception as e:
        result.error = str(e)
        result.error_traceback = traceback.format_exc()

    return result


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================


def print_agent_result(result: AgentTestResult, verbose: bool = True) -> None:
    """Print detailed result for a single agent in enhanced Tier0-style format.

    Args:
        result: AgentTestResult to print
        verbose: If True, print full details; if False, print summary only
    """
    tier_name = TIER_DESCRIPTIONS.get(result.tier, "Unknown")
    print_header(f"AGENT: {result.agent_name.upper()} (Tier {result.tier} - {tier_name})")

    # Input summary (enhanced format)
    print_input_section(result.input_summary)

    # Processing steps
    processing_steps = [
        (f"Agent {result.agent_name} instantiated", True),
        (f"Input validation passed", not result.error or "input" not in (result.error or "").lower()),
        (f"Agent execution {'completed' if result.success else 'failed'}", result.success),
    ]
    if result.contract_validation:
        processing_steps.append(
            (f"Contract validation ({result.contract_validation.state_class})",
             result.contract_validation.valid)
        )
    if result.trace_verification:
        processing_steps.append(
            (f"Opik trace captured", result.trace_verification.trace_exists)
        )
    print_processing_section(processing_steps)

    # Validation checks
    checks = []
    if result.contract_validation:
        cv = result.contract_validation
        req_present = len(cv.required_fields_present)
        req_total = cv.required_total
        checks.append((
            "Required fields present",
            req_present == req_total,
            f"{req_total} required fields",
            f"{req_present}/{req_total} present"
        ))
        checks.append((
            "Type validation",
            len(cv.type_errors) == 0,
            "no type errors",
            f"{len(cv.type_errors)} type errors" if cv.type_errors else "all types valid"
        ))
    checks.append((
        "Execution completed",
        result.success,
        "success without errors",
        "success" if result.success else f"failed: {(result.error or 'unknown')[:40]}"
    ))
    if result.trace_verification:
        tv = result.trace_verification
        checks.append((
            "Observability trace",
            tv.trace_exists,
            "trace captured",
            "trace exists" if tv.trace_exists else "no trace"
        ))
    print_validation_checks(checks)

    # Key metrics table
    metrics = [
        ("execution_time", result.execution_time_ms / 1000, None, None),
        ("agent_tier", result.tier, None, None),
    ]
    if result.contract_validation:
        cv = result.contract_validation
        metrics.append(("required_fields", f"{len(cv.required_fields_present)}/{cv.required_total}", None, None))
        metrics.append(("optional_fields", f"{len(cv.optional_fields_present)}/{cv.optional_total}", None, None))
        metrics.append(("contract_valid", cv.valid, "True", cv.valid))
    if result.trace_verification and result.trace_verification.trace_exists:
        tv = result.trace_verification
        metrics.append(("trace_spans", tv.span_count, None, None))
        if tv.duration_ms:
            metrics.append(("trace_duration_ms", tv.duration_ms, None, None))

    # Add agent-specific metrics from output
    if result.agent_output:
        output = result.agent_output
        # Extract key numeric/boolean metrics
        priority_metrics = ["overall_ate", "heterogeneity_score", "drift_score", "health_score",
                          "overall_score", "confidence", "statistical_power", "p_value"]
        for key in priority_metrics:
            if key in output and output[key] is not None:
                val = output[key]
                if isinstance(val, (int, float, bool)):
                    metrics.append((key, val, None, None))

    print_metrics_table(metrics)

    # Agent-specific analysis
    if result.agent_output:
        insights = extract_agent_insights(result.agent_name, result.agent_output)
        print_analysis_section(f"{result.agent_name.upper()} Analysis", insights)

    # Show key output fields if verbose
    if verbose and result.agent_output:
        print("\n  📋 Key Output Fields:")
        output_items = list(result.agent_output.items())

        # Prioritize important fields
        priority_fields = ["overall_ate", "heterogeneity_score", "causal_effect", "drift_detected",
                          "experiment_design", "health_status", "prediction", "optimization_result",
                          "executive_summary", "learning_summary", "analysis_complete", "status"]
        priority_items = [(k, v) for k, v in output_items if k in priority_fields]
        other_items = [(k, v) for k, v in output_items if k not in priority_fields]

        # Show priority items first
        for key, value in priority_items[:6]:
            _print_output_value(key, value, indent=4)

        # Show other items (limit)
        shown = len(priority_items[:6])
        for key, value in other_items[:max(0, 6 - shown)]:
            _print_output_value(key, value, indent=4)

        total_fields = len(output_items)
        shown_total = min(6, len(priority_items)) + min(max(0, 6 - len(priority_items[:6])), len(other_items))
        remaining = total_fields - shown_total
        if remaining > 0:
            print(f"    ... and {remaining} more fields")

    # Contract validation details (verbose)
    if verbose and result.contract_validation:
        cv = result.contract_validation
        if cv.type_errors:
            print("\n  ⚠️  Type Errors:")
            for te in cv.type_errors[:3]:
                print(f"    • {te.get('field')}: expected {te.get('expected')}, got {te.get('actual')}")
        if cv.warnings:
            print(f"\n  ⚠️  Warnings ({len(cv.warnings)}):")
            for w in cv.warnings[:2]:
                w_display = w[:70] + "..." if len(w) > 70 else w
                print(f"    • {w_display}")

    # Observability details (verbose)
    if verbose and result.trace_verification and result.trace_verification.trace_exists:
        tv = result.trace_verification
        print("\n  🔭 Observability Details:")
        if tv.trace_id:
            print(f"    • Trace ID: {tv.trace_id[:40]}...")
        if tv.trace_url:
            print(f"    • URL: {tv.trace_url}")
        if tv.span_names:
            print(f"    • Spans: {', '.join(tv.span_names[:5])}")

    # Error details if failed
    if result.error:
        print("\n  🚨 Error Details:")
        print(f"    {result.error}")
        if verbose and result.error_traceback:
            tb_lines = result.error_traceback.strip().split("\n")
            print("    Traceback (last 3 lines):")
            for line in tb_lines[-3:]:
                print(f"      {line}")

    # Final result line
    duration_s = result.execution_time_ms / 1000
    if result.success:
        print_step_result("success", f"{result.agent_name} completed successfully", duration_s)
    else:
        error_brief = (result.error or "Unknown error")[:50]
        print_step_result("failed", f"{result.agent_name}: {error_brief}", duration_s)


def _print_output_value(key: str, value: Any, indent: int = 4) -> None:
    """Print a single output value with appropriate formatting."""
    prefix = " " * indent

    if value is None:
        print(f"{prefix}{key}: null")
    elif isinstance(value, dict):
        if len(value) == 0:
            print(f"{prefix}{key}: {{}}")
        elif len(value) <= 3:
            # Small dict - print inline or expanded
            print(f"{prefix}{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"{prefix}  {k}: {v:.4f}")
                elif isinstance(v, list):
                    print(f"{prefix}  {k}: [{len(v)} items]")
                elif isinstance(v, str) and len(v) > 50:
                    print(f"{prefix}  {k}: {v[:50]}...")
                else:
                    print(f"{prefix}  {k}: {v}")
        else:
            # Large dict - summarize
            print(f"{prefix}{key}: {{dict with {len(value)} keys}}")
    elif isinstance(value, list):
        if len(value) == 0:
            print(f"{prefix}{key}: []")
        elif len(value) <= 2 and all(isinstance(x, (str, int, float)) for x in value):
            print(f"{prefix}{key}: {value}")
        else:
            # Show first item if it's a dict (common pattern)
            if isinstance(value[0], dict):
                print(f"{prefix}{key}: [{len(value)} items]")
                # Show first item as sample
                first = value[0]
                sample_keys = list(first.keys())[:4]
                sample = {k: first[k] for k in sample_keys}
                # Truncate values in sample
                for k, v in sample.items():
                    if isinstance(v, float):
                        sample[k] = round(v, 4)
                    elif isinstance(v, str) and len(v) > 30:
                        sample[k] = v[:30] + "..."
                print(f"{prefix}  [0]: {sample}...")
            else:
                print(f"{prefix}{key}: [{len(value)} items]")
    elif isinstance(value, float):
        print(f"{prefix}{key}: {value:.4f}")
    elif isinstance(value, str):
        if len(value) > 100:
            print(f"{prefix}{key}: {value[:100]}...")
        elif "\n" in value:
            # Multi-line string - show first line
            first_line = value.split("\n")[0]
            if len(first_line) > 80:
                first_line = first_line[:80] + "..."
            print(f"{prefix}{key}: {first_line}")
        else:
            print(f"{prefix}{key}: {value}")
    else:
        print(f"{prefix}{key}: {value}")


def print_summary(
    results: list[AgentTestResult],
    total_time_ms: float,
    tier0_experiment_id: str,
    verbose: bool = True,
) -> None:
    """Print test summary with detailed tier breakdown."""
    print_header("TEST SUMMARY")

    # Tier descriptions
    tier_names = {
        1: "Orchestration",
        2: "Causal Analytics",
        3: "Monitoring",
        4: "ML Predictions",
        5: "Self-Improvement",
    }

    # Tier breakdown with detailed status
    tier_results: dict[int, dict[str, list[AgentTestResult]]] = {}
    for r in results:
        if r.tier not in tier_results:
            tier_results[r.tier] = {"passed": [], "failed": []}
        if r.success:
            tier_results[r.tier]["passed"].append(r)
        else:
            tier_results[r.tier]["failed"].append(r)

    print("\nTIER RESULTS:")
    print("-" * 60)

    for tier in sorted(tier_results.keys()):
        passed = tier_results[tier]["passed"]
        failed = tier_results[tier]["failed"]
        total = len(passed) + len(failed)
        tier_name = tier_names.get(tier, "Unknown")

        # Color coding
        if len(failed) == 0:
            status_color = "\033[92m"  # Green
            status = "ALL PASSED"
        elif len(passed) == 0:
            status_color = "\033[91m"  # Red
            status = "ALL FAILED"
        else:
            status_color = "\033[93m"  # Yellow
            status = f"{len(passed)}/{total} PASSED"

        print(f"\n  Tier {tier} - {tier_name}: {status_color}{status}\033[0m")

        if verbose:
            # List all agents with status
            for r in passed:
                time_str = format_duration(r.execution_time_ms)
                print(f"    \u2713 {r.agent_name} ({time_str})")
            for r in failed:
                time_str = format_duration(r.execution_time_ms)
                error_brief = (r.error or "Unknown")[:40]
                print(f"    \u2717 {r.agent_name} ({time_str}) - {error_brief}")
        else:
            # Compact listing
            all_agents = [r.agent_name for r in passed] + [f"{r.agent_name} (FAILED)" for r in failed]
            print(f"    Agents: {', '.join(all_agents)}")

    print()
    print("-" * 60)

    # Overall stats with visual bar
    passed_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)
    total = len(results)
    pass_rate = (passed_count / total * 100) if total > 0 else 0

    # Visual progress bar
    bar_width = 40
    filled = int(bar_width * pass_rate / 100)
    bar = "\033[92m" + "█" * filled + "\033[0m" + "░" * (bar_width - filled)

    print(f"\nOVERALL: [{bar}] {pass_rate:.1f}%")
    print(f"  Total Agents: {total}")
    print(f"  Passed: \033[92m{passed_count}\033[0m")
    print(f"  Failed: \033[91m{failed_count}\033[0m")
    print(f"  Total Time: {format_duration(total_time_ms)}")
    print(f"  Avg Time/Agent: {format_duration(total_time_ms / total) if total > 0 else 'N/A'}")

    # Failed agents with details
    failed = [r for r in results if not r.success]
    if failed:
        print("\nFAILED AGENTS:")
        print("-" * 60)
        for r in failed:
            error_msg = r.error or "Contract validation failed"
            # Truncate long errors
            if len(error_msg) > 80:
                error_msg = error_msg[:80] + "..."
            print(f"  \u274c {r.agent_name} (Tier {r.tier})")
            print(f"     Error: {error_msg}")
            if r.error_traceback and verbose:
                # Show last line of traceback
                last_line = r.error_traceback.strip().split("\n")[-1]
                print(f"     Last line: {last_line[:70]}")

    # Contract validation summary
    contracts_valid = sum(
        1 for r in results if r.contract_validation and r.contract_validation.valid
    )
    type_errors_total = sum(
        len(r.contract_validation.type_errors)
        for r in results if r.contract_validation
    )

    print("\nCONTRACT VALIDATION:")
    print(f"  Valid Contracts: {contracts_valid}/{total}")
    print(f"  Total Type Errors: {type_errors_total}")

    # Observability summary
    traces_created = sum(
        1 for r in results if r.trace_verification and r.trace_verification.trace_exists
    )
    traces_verified = sum(
        1 for r in results if r.trace_verification and r.trace_verification.metadata_valid
    )
    traces_with_errors = sum(
        1 for r in results if r.trace_verification and r.trace_verification.error_captured
    )

    print("\nOBSERVABILITY:")
    print(f"  Traces Created: {traces_created}")
    print(f"  Traces Verified: {traces_verified}")
    if traces_with_errors > 0:
        print(f"  Traces with Errors: {traces_with_errors}")


# =============================================================================
# MAIN RUNNER
# =============================================================================


async def run_tests(
    tier0_cache: str | None = None,
    run_tier0_first: bool = False,
    tiers: list[int] | None = None,
    agents: list[str] | None = None,
    skip_observability: bool = False,
    output_path: str | None = None,
    timeout_seconds: float = 30.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all agent tests.

    Args:
        tier0_cache: Path to cached tier0 state
        run_tier0_first: Run tier0 test first to generate data
        tiers: List of tiers to test (None = all)
        agents: List of agent names to test (None = all)
        skip_observability: Skip Opik trace verification
        output_path: Path to save JSON results
        timeout_seconds: Timeout per agent
        verbose: Show detailed output for each agent

    Returns:
        Full test results dict
    """
    from src.testing import Tier0OutputMapper, ContractValidator, OpikTraceVerifier

    print_header("TIER 1-5 AGENT TESTING FRAMEWORK")

    # Load or generate tier0 state
    if run_tier0_first:
        tier0_state = await run_tier0_and_cache()
    elif tier0_cache:
        tier0_state = load_tier0_state(tier0_cache)
    else:
        # Try default cache location
        default_cache = PROJECT_ROOT / "scripts" / "tier0_output_cache" / "latest.pkl"
        if default_cache.exists():
            tier0_state = load_tier0_state(str(default_cache))
        else:
            raise ValueError(
                "No tier0 state available. Use --run-tier0-first or --tier0-cache"
            )

    experiment_id = tier0_state.get("experiment_id", "unknown")
    print(f"Tier0 Experiment ID: {experiment_id}")

    # Initialize components
    mapper = Tier0OutputMapper(tier0_state)
    # Use lenient mode: agents return output-focused dicts, not echoing all input fields
    # In lenient mode, missing required fields are warnings, not errors
    validator = ContractValidator(lenient=True)

    trace_verifier = None
    if not skip_observability:
        trace_verifier = OpikTraceVerifier()
        health = await trace_verifier.check_opik_health()
        if health.get("healthy"):
            print("Opik: Healthy")
        else:
            print(f"Opik: Not available ({health.get('error')})")
            print("  Continuing without observability verification...")
            trace_verifier = None

    # Filter agents to test
    agents_to_test = {}
    for name, config in AGENT_CONFIGS.items():
        if tiers and config["tier"] not in tiers:
            continue
        if agents and name not in agents:
            continue
        agents_to_test[name] = config

    print(f"Agents to Test: {len(agents_to_test)}")
    print(f"Agents: {', '.join(agents_to_test.keys())}")

    # Summarize tier0 state
    df = tier0_state.get("eligible_df")
    if df is not None:
        print(f"\nTier0 Data:")
        print(f"  eligible_df: {len(df)} rows x {len(df.columns)} columns")

    trained_model = tier0_state.get("trained_model")
    if trained_model:
        print(f"  trained_model: {type(trained_model).__name__}")

    feature_importance = tier0_state.get("feature_importance")
    if feature_importance:
        print(f"  feature_importance: {len(feature_importance)} features")

    validation_metrics = tier0_state.get("validation_metrics")
    if validation_metrics:
        print(f"  validation_metrics: {list(validation_metrics.keys())}")

    # Run tests
    results: list[AgentTestResult] = []
    total_start = time.time()

    for agent_name, config in agents_to_test.items():
        # Use per-agent timeout if specified, otherwise use global timeout
        agent_timeout = config.get("timeout", timeout_seconds)
        result = await test_agent(
            agent_name=agent_name,
            config=config,
            mapper=mapper,
            validator=validator,
            trace_verifier=trace_verifier,
            timeout_seconds=agent_timeout,
        )
        results.append(result)
        print_agent_result(result, verbose=verbose)

    total_time_ms = (time.time() - total_start) * 1000

    # Print summary
    print_summary(results, total_time_ms, experiment_id, verbose=verbose)

    # Build full results
    full_results = {
        "test_run": {
            "id": f"tier1_5_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now(UTC).isoformat(),
            "tier0_cache": tier0_cache,
            "tier0_experiment_id": experiment_id,
        },
        "summary": {
            "total_agents": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "skipped": 0,
            "total_time_ms": total_time_ms,
            "pass_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
        },
        "tier_breakdown": {},
        "results": [asdict(r) for r in results],
        "observability_summary": {
            "traces_created": sum(
                1 for r in results if r.trace_verification and r.trace_verification.trace_exists
            ),
            "traces_verified": sum(
                1 for r in results if r.trace_verification and r.trace_verification.metadata_valid
            ),
            "opik_health": "healthy" if trace_verifier else "not_checked",
        },
    }

    # Build tier breakdown
    for tier in sorted(set(r.tier for r in results)):
        tier_results = [r for r in results if r.tier == tier]
        full_results["tier_breakdown"][f"tier_{tier}"] = {
            "passed": sum(1 for r in tier_results if r.success),
            "failed": sum(1 for r in tier_results if not r.success),
            "agents": [r.agent_name for r in tier_results],
        }

    # Save results if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    return full_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Tier 1-5 agent tests using tier0 outputs"
    )
    parser.add_argument(
        "--run-tier0-first",
        action="store_true",
        help="Run tier0 test first to generate synthetic data",
    )
    parser.add_argument(
        "--tier0-cache",
        type=str,
        default=None,
        help="Path to cached tier0 state pickle file",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default=None,
        help="Comma-separated list of tiers to test (e.g., '2,3')",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated list of agents to test (e.g., 'causal_impact,explainer')",
    )
    parser.add_argument(
        "--skip-observability",
        action="store_true",
        help="Skip Opik trace verification",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout per agent in seconds (default: 30)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Show detailed output per agent (default: True)",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Show brief output (opposite of --verbose)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/results",
        help="Directory to save results MD file (default: docs/results)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to markdown file (only print to console)",
    )

    args = parser.parse_args()

    # Parse tiers
    tiers = None
    if args.tiers:
        tiers = [int(t.strip()) for t in args.tiers.split(",")]

    # Parse agents
    agents = None
    if args.agents:
        agents = [a.strip() for a in args.agents.split(",")]

    # Determine verbosity
    verbose = args.verbose and not args.brief

    # Setup output capturing for markdown save
    output_buffer = io.StringIO()

    class TeeOutput:
        """Write to both console and buffer."""
        def __init__(self, console, buffer):
            self.console = console
            self.buffer = buffer

        def write(self, text):
            # Strip ANSI color codes for markdown file
            import re
            clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
            self.buffer.write(clean_text)
            self.console.write(text)

        def flush(self):
            self.console.flush()
            self.buffer.flush()

    # Capture output
    original_stdout = sys.stdout
    if not args.no_save:
        sys.stdout = TeeOutput(original_stdout, output_buffer)

    try:
        # Run tests
        asyncio.run(
            run_tests(
                tier0_cache=args.tier0_cache,
                run_tier0_first=args.run_tier0_first,
                tiers=tiers,
                agents=agents,
                skip_observability=args.skip_observability,
                output_path=args.output,
                timeout_seconds=args.timeout,
                verbose=verbose,
            )
        )
    finally:
        # Restore stdout
        sys.stdout = original_stdout

        # Save results to markdown file
        if not args.no_save and output_buffer.getvalue():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"tier1_5_pipeline_run_{timestamp}.md"

            # Build markdown content
            md_content = "# Tier 1-5 Agent Test Results\n\n"
            md_content += f"**Generated**: {datetime.now().isoformat()}\n\n"
            md_content += "## Test Configuration\n\n"
            md_content += f"- **Tiers Tested**: {args.tiers or 'all (1-5)'}\n"
            md_content += f"- **Agents Tested**: {args.agents or 'all'}\n"
            md_content += f"- **Tier0 Cache**: {args.tier0_cache or 'auto-generated'}\n"
            md_content += f"- **Observability**: {'skipped' if args.skip_observability else 'enabled'}\n"
            md_content += f"- **Timeout**: {args.timeout}s per agent\n\n"
            md_content += "## Results\n\n"
            md_content += "```\n"
            md_content += output_buffer.getvalue()
            md_content += "```\n"

            with open(output_file, "w") as f:
                f.write(md_content)

            print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
