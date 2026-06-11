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

# Suppress warnings and configure services for host-based test execution
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
os.environ.setdefault("E2I_TESTING_MODE", "true")
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:8084")

from dotenv import load_dotenv

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")


def _install_networkx_dowhy_compat() -> None:
    """Alias the renamed networkx d-separation API for dowhy 0.12 (harness-only).

    The pinned dowhy==0.12 calls ``nx.algorithms.d_separated`` inside
    ``CausalModel.identify_effect`` (dowhy/causal_graph.py), but networkx
    RENAMED that function to ``is_d_separator`` (3.3) and REMOVED the old name
    (3.5; the pinned networkx is 3.6.1). Without the alias the causal_impact
    refutation node fail-closes before any refuter runs ("DoWhy CausalModel
    reconstruction failed ... no attribute 'd_separated'") — a dependency
    incompatibility, not an analysis failure (already noted in
    tests/integration/test_synthetic_causal_gates.py gate_3). The rename is
    semantics-preserving (identical (G, x, y, z) -> bool contract), so
    aliasing lets the REAL DoWhy refutation suite execute; nothing is mocked.
    Harness-scoped on purpose: prod owns its own dependency-resolution fix.
    """
    try:
        import networkx as nx
    except ImportError:  # harness deps guarantee networkx; stay quiet if absent
        return
    if not hasattr(nx.algorithms, "d_separated") and hasattr(nx.algorithms, "is_d_separator"):
        nx.algorithms.d_separated = nx.algorithms.is_d_separator
        nx.d_separated = nx.is_d_separator


_install_networkx_dowhy_compat()


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
class QualityGateDetail:
    """Detailed quality gate validation results."""

    passed: bool
    total_checks: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    failed_check_names: list[str] = field(default_factory=list)
    failed_check_messages: list[str] = field(default_factory=list)
    required_output_fields_present: list[str] = field(default_factory=list)
    required_output_fields_missing: list[str] = field(default_factory=list)
    status_failure: bool = False
    status_value: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DataSourceDetail:
    """Detailed data source validation results."""

    passed: bool
    detected_source: str = "unknown"
    acceptable_sources: list[str] = field(default_factory=list)
    reject_mock: bool = False
    message: str = ""
    evidence: list[str] = field(default_factory=list)


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

    # Quality Gate Details
    quality_gate: QualityGateDetail | None = None

    # Data Source Validation Details
    data_source: DataSourceDetail | None = None

    # Observability Details
    trace_verification: TraceVerificationDetail | None = None

    # Performance Metrics
    performance_metrics: PerformanceDetail | None = None


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Per-agent dispatch + harness metadata is owned by
# src.agents.orchestrator._agent_method_map::AGENT_METHOD_MAP. The harness was
# previously duplicating these values in a literal AGENT_CONFIGS dict, which
# allowed silent drift between the production dispatcher and the harness.
# Issue #252 unified them: AGENT_CONFIGS is now computed once at import time
# from AGENT_METHOD_MAP via get_harness_configs(). Per-harness-only fields
# (tier, agent module/class, state module/class, timeout) layer on top of the
# shared dispatch spec.
from src.agents.orchestrator._agent_method_map import get_harness_configs

AGENT_CONFIGS = get_harness_configs()

# Base per-agent timeout when neither the agent config nor the CLI provides one.
DEFAULT_AGENT_TIMEOUT_SECONDS = 30.0


def resolve_agent_timeout(config: dict[str, Any], cli_timeout: float | None) -> float:
    """Resolve the effective per-agent timeout.

    An EXPLICIT CLI ``--timeout`` acts as a FLOOR, never a silent cap: a
    per-agent configured timeout LONGER than the CLI value is preserved
    (heavy agents keep their budget), but a SHORTER configured value (e.g.
    experiment_monitor's 20s) must not cap the run below what the caller
    explicitly asked for — on a LOKY-serialized box that capped
    ``--timeout 180`` down to 20s and timed the agent out. Without a CLI
    value, the per-agent config (or the 30s base default) applies unchanged.
    """
    configured = config.get("timeout")
    if cli_timeout is None:
        return float(configured if configured is not None else DEFAULT_AGENT_TIMEOUT_SECONDS)
    if configured is None:
        return float(cli_timeout)
    return float(max(configured, cli_timeout))


# =============================================================================
# AGENT-SPECIFIC CONFIGURATION
# =============================================================================


class Tier0ModelClient:
    """Wraps a tier0 trained sklearn model as a prediction client."""

    def __init__(
        self, model, model_id: str = "tier0_model", feature_names: list[str] | None = None
    ):
        self.model = model
        self.model_id = model_id
        # Get feature names from model if available, otherwise use provided names
        if hasattr(model, "feature_names_in_"):
            self.feature_names = list(model.feature_names_in_)
        else:
            self.feature_names = feature_names or []

    async def predict(
        self,
        entity_id: str,
        features: dict[str, Any],
        time_horizon: str,
    ) -> dict[str, Any]:
        """Make prediction using the wrapped model."""
        import time

        import numpy as np

        start = time.time()

        try:
            # Reorder features to match model's expected order
            if self.feature_names:
                feature_values = [features.get(name, 0.0) for name in self.feature_names]
            else:
                feature_values = list(features.values())

            X = np.array([feature_values], dtype=float)

            # Handle missing/NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Get prediction
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                # proba is array of class probabilities, e.g., [0.3, 0.7] for binary
                proba_list = [float(p) for p in proba]  # Convert to list of floats
                prediction = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                prediction = float(self.model.predict(X)[0])
                proba_list = None  # No probability estimates available

            # Confidence based on distance from 0.5
            confidence = abs(prediction - 0.5) * 2

            return {
                "model_id": self.model_id,
                "model_type": "logistic_regression",
                "prediction": prediction,
                "proba": proba_list,  # Return full class probabilities as list
                "confidence": max(0.5, min(1.0, confidence + 0.5)),
                "latency_ms": int((time.time() - start) * 1000),
                "features_used": self.feature_names or list(features.keys()),
            }
        except Exception as e:
            # Return a default prediction on error
            return {
                "model_id": self.model_id,
                "model_type": "logistic_regression",
                "prediction": 0.5,
                "proba": None,  # Explicitly set to None
                "confidence": 0.3,
                "latency_ms": int((time.time() - start) * 1000),
                "features_used": self.feature_names or [],
                "error": str(e),
            }


class PopulationBaselineClient:
    """Returns population mean discontinuation rate as a baseline prediction model.

    This provides a legitimate statistical model (population prior) that serves
    as a second model for ensemble validation. The prediction_synthesizer requires
    models_count >= 2 for ensemble reliability assessment.
    """

    def __init__(self, population_rate: float, model_id: str = "population_baseline"):
        self.population_rate = population_rate
        self.model_id = model_id

    async def predict(
        self,
        entity_id: str,
        features: dict[str, Any],
        time_horizon: str,
    ) -> dict[str, Any]:
        """Return population baseline prediction."""
        import time as _time

        start = _time.time()
        return {
            "model_id": self.model_id,
            "model_type": "population_baseline",
            "prediction": self.population_rate,
            "proba": [1.0 - self.population_rate, self.population_rate],
            "confidence": 0.6,  # Moderate confidence — prior is less precise
            "latency_ms": int((_time.time() - start) * 1000),
            "features_used": [],
        }


def _get_agent_kwargs(
    agent_name: str,
    enforce_real_data: bool = True,
    tier0_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get agent-specific constructor kwargs for testing.

    This function returns kwargs that should be passed to agent constructors
    to ensure they use real data sources instead of mock fallbacks.

    Args:
        agent_name: Name of the agent
        enforce_real_data: If True, configure agents to require real data sources
        tier0_state: Optional tier0 state dict with trained_model, etc.

    Returns:
        Dict of kwargs to pass to agent constructor
    """
    if not enforce_real_data:
        return {}

    if agent_name == "health_score":
        # Inject real health client for health_score agent
        try:
            from src.agents.health_score.health_client import get_health_client_for_testing

            return {"health_client": get_health_client_for_testing()}
        except ImportError:
            return {}

    elif agent_name == "gap_analyzer":
        # gap_analyzer: tier0_output_mapper passes tier0_data which contains
        # the patient-level data. The GapDetectorNode will aggregate it
        # into performance metrics per segment.
        return {}

    elif agent_name == "heterogeneous_optimizer":
        # heterogeneous_optimizer: require real data
        # Note: This is set at the node level, not agent level
        # The agent will use real data by default, but we can't easily
        # inject require_real_data=True without modifying agent constructor
        return {}

    elif agent_name == "prediction_synthesizer":
        # prediction_synthesizer: inject tier0 trained model + population baseline
        # as two model clients. Ensemble validation requires models_count >= 2
        # for reliability_assessment != "UNVALIDATED".
        # Disable Opik tracing to avoid async generator cancellation issues with timeouts
        if tier0_state and tier0_state.get("trained_model"):
            model = tier0_state["trained_model"]
            model_id = tier0_state.get("experiment_id", "tier0_model")
            feature_names = tier0_state.get("feature_names")  # Get from tier0 state

            # Compute the population base rate of the ACTUAL modeling target
            # for the baseline model. The synthetic-CSU frames model
            # scope_spec.prediction_target (e.g. treatment_initiated) and
            # carry a CONSTANT-0 legacy discontinuation_flag — using that
            # would feed a fabricated 0.0 prior. Legacy caches (no
            # prediction_target) keep the discontinuation rate.
            eligible_df = tier0_state.get("eligible_df")
            target_col = (tier0_state.get("scope_spec") or {}).get("prediction_target")
            if eligible_df is not None and target_col and target_col in eligible_df.columns:
                pop_rate = float(eligible_df[target_col].mean())
            elif eligible_df is not None and "discontinuation_flag" in eligible_df.columns:
                pop_rate = float(eligible_df["discontinuation_flag"].mean())
            else:
                pop_rate = 0.3  # Reasonable default

            return {
                "model_clients": {
                    model_id: Tier0ModelClient(model, model_id, feature_names),
                    "population_baseline": PopulationBaselineClient(pop_rate),
                },
                "enable_opik": False,  # Prevent async generator issues with timeouts
                "enable_memory": False,  # Simplify test execution
            }
        return {"enable_opik": False, "enable_memory": False}

    elif agent_name == "causal_impact":
        # causal_impact.run() now contributes to tri-memory on success (#788). In the
        # keyless harness SUPABASE_URL/OPENAI are unset, so store_causal_analysis would
        # fail functionally — but FIRST it triggers a cold-start load of the
        # all-MiniLM-L6-v2 fallback embedder (~144s) via insert_episodic_memory_with_text,
        # blowing the 90s per-agent timeout (the same flake the tool_composer branch
        # below avoids). Disable memory for the harness only; the memory write path is
        # covered by its own unit + faithful integration tests and the #785 populate run.
        return {"enable_memory": False}

    elif agent_name == "tool_composer":
        # tool_composer: disable the planner's episodic-memory lookup in the
        # keyless harness. SUPABASE_URL is unset, so the lookup always fails
        # functionally — but it FIRST triggers a cold-start load of the
        # all-MiniLM-L6-v2 fallback embedder (~144s cold vs ~2s warm). As the
        # first agent to hit this path, tool_composer absorbs the cold start
        # and blows the 90s per-agent timeout (the flake that forced the
        # admin-merges). Setting use_episodic_memory=False makes
        # ToolPlanner._check_episodic_memory short-circuit to [] BEFORE
        # find_similar_compositions loads the embedder.
        #
        # The config flows: ToolComposerAgent(config=...) -> ToolComposer(config=...)
        # -> _init_phase_handlers reads config["phases"]["plan"]["use_episodic_memory"]
        # -> ToolPlanner(use_episodic_memory=False). This disables a
        # NON-FUNCTIONAL lookup only; it does not mask any real failure (the
        # agent is NOT in TIER1_5_EXPECTED_FAIL_AGENTS).
        return {"config": {"phases": {"plan": {"use_episodic_memory": False}}}}

    elif agent_name == "orchestrator":
        # The harness exercises the orchestrator's OWN contract (intent routing +
        # synthesis); it instantiates it WITHOUT a sub-agent registry (the real
        # Tier 2-5 agents are validated as their own harness entries). Since #814
        # the dispatcher fails CLOSED for an unregistered agent instead of
        # fabricating a result, so a registry-less orchestrator would dispatch
        # every routed agent to a structured failure and the orchestrator's
        # contract would regress. allow_mock=True opts this test-only path into
        # the canned dispatcher scaffold (production never sets it -> fail-closed).
        return {"allow_mock": True}

    return {}


def _get_graph_kwargs(agent_name: str, enforce_real_data: bool = True) -> dict[str, Any]:
    """Get kwargs for graph creation functions.

    Some agents use graph factories that accept configuration options.

    Args:
        agent_name: Name of the agent
        enforce_real_data: If True, configure graphs to require real data

    Returns:
        Dict of kwargs for graph creation
    """
    if not enforce_real_data:
        return {}

    if agent_name == "gap_analyzer":
        return {"use_mock": False}

    return {}


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
    """Create a summary of agent input for display.

    No truncation - full values shown for debugging and clarity.
    """
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
        else:
            # No truncation - show full value
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
    """Print standardized input summary section.

    No truncation - full values shown for clarity.
    """
    print("\n  📥 Input Summary:")
    for key, value in inputs.items():
        if isinstance(value, str):
            # No truncation - show full string
            print(f"    • {key}: {value}")
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
    print(f"    {'-' * 65}")

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
# NOTE: Field names must match actual agent state output fields
AGENT_ANALYSIS_CONFIG = {
    "orchestrator": {
        "key_fields": ["agents_dispatched", "response_confidence", "final_response"],
        "insights_template": [
            "Query routed to {agents_dispatched} agents",
            "Routing confidence: {response_confidence}",
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
        "key_fields": ["ate_estimate", "confidence_interval", "p_value", "confidence"],
        "insights_template": [
            "Average Treatment Effect (ATE): {ate_estimate}",
            "Statistical significance: p={p_value}",
        ],
    },
    "gap_analyzer": {
        "key_fields": ["prioritized_opportunities", "total_gap_value", "total_addressable_value"],
        "insights_template": [
            "Identified {prioritized_opportunities} performance gaps",
            "Total opportunity value: {total_gap_value}",
        ],
    },
    "heterogeneous_optimizer": {
        "key_fields": ["heterogeneity_score", "overall_ate", "cate_by_segment"],
        "insights_template": [
            "Overall ATE: {overall_ate}",
            "Heterogeneity score: {heterogeneity_score}",
        ],
    },
    "drift_monitor": {
        "key_fields": [
            "overall_drift_score",
            "features_with_drift",
            "recommended_actions",
            "drift_interpretation",
        ],
        "insights_template": [
            "Overall drift score: {overall_drift_score}",
            "Features with drift: {features_with_drift}",
            "Recommended actions: {recommended_actions}",
        ],
    },
    "experiment_designer": {
        "key_fields": [
            "experiment_design",
            "required_sample_size",
            "statistical_power",
            "validity_assessment",
        ],
        "insights_template": [
            "Experiment design created with n={required_sample_size}",
            "Statistical power: {statistical_power}",
        ],
    },
    "health_score": {
        "key_fields": [
            "overall_health_score",
            "health_grade",
            "critical_issues",
            "recommendations",
        ],
        "insights_template": [
            "Health grade: {health_grade}",
            "Overall health score: {overall_health_score}",
            "Recommendations: {recommendations}",
        ],
    },
    "experiment_monitor": {
        "key_fields": [
            "experiments_checked",
            "healthy_count",
            "warning_count",
            "critical_count",
            "monitor_summary",
        ],
        "insights_template": [
            "Experiments checked: {experiments_checked}",
            "Health breakdown: {healthy_count} healthy, {warning_count} warning, {critical_count} critical",
            "Summary: {monitor_summary}",
        ],
    },
    "prediction_synthesizer": {
        "key_fields": ["ensemble_prediction", "prediction_summary", "models_succeeded"],
        "insights_template": [
            "Prediction summary: {prediction_summary}",
            "Models succeeded: {models_succeeded}",
        ],
    },
    "resource_optimizer": {
        "key_fields": [
            "optimization_summary",
            "projected_roi",
            "recommendations",
            "projected_savings",
        ],
        "insights_template": [
            "Optimization completed: {optimization_summary}",
            "Projected ROI: {projected_roi}",
        ],
    },
    "explainer": {
        "key_fields": ["executive_summary", "extracted_insights", "key_themes"],
        "insights_template": [
            "Executive summary: {executive_summary}",
            "Key themes: {key_themes}",
        ],
    },
    "feedback_learner": {
        "key_fields": [
            "feedback_summary",
            "learning_summary",
            "learning_recommendations",
            "detected_patterns",
        ],
        "insights_template": [
            "Learning summary: {learning_summary}",
            "Patterns detected: {detected_patterns}",
        ],
    },
}


def extract_agent_insights(agent_name: str, output: dict[str, Any]) -> list[str]:
    """Extract agent-specific insights from output.

    Supports nested field access via dot notation (e.g., 'ensemble_prediction.point_estimate').
    """
    config = AGENT_ANALYSIS_CONFIG.get(agent_name, {})
    insights = []

    # Track list fields that need expanded display
    list_fields_to_expand: dict[str, list[str]] = {}

    # Try to fill in template insights
    for template in config.get("insights_template", []):
        try:
            # Extract values from output
            values = {}
            for key in config.get("key_fields", []):
                # Handle nested keys with dot notation
                if "." in key:
                    parts = key.split(".")
                    val = output
                    for part in parts:
                        if val is not None and isinstance(val, dict):
                            val = val.get(part)
                        else:
                            val = None
                            break
                else:
                    val = output.get(key)

                if val is not None:
                    if isinstance(val, float):
                        values[key] = f"{val:.4f}"
                    elif isinstance(val, list):
                        values[key] = len(val)
                        # Track string lists for expanded display below the template line
                        if all(isinstance(x, str) for x in val) and len(val) > 0:
                            if f"{{{key}}}" in template:
                                list_fields_to_expand[key] = val
                    elif isinstance(val, dict):
                        # For dicts, show key count or first few keys
                        keys_preview = list(val.keys())[:3]
                        values[key] = (
                            f"{{{', '.join(keys_preview)}...}}" if len(val) > 3 else str(val)
                        )
                    elif isinstance(val, str):
                        # Show full string value (no truncation)
                        values[key] = val
                    else:
                        values[key] = val
                else:
                    values[key] = "N/A"

            insight = template.format(**values)
            insights.append(insight)

            # Expand string list items as sub-insights after the template line
            for key, items in list_fields_to_expand.items():
                if f"{{{key}}}" in template:
                    for i, item in enumerate(items, 1):
                        insights.append(f"  {i}. {item}")
            list_fields_to_expand.clear()
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

    Notes:
        Block 4 / Finding #12: when the cached state contains a
        ``split_assignments`` mapping, downstream tier1-5 consumers MUST
        reuse those assignments instead of re-deriving splits — otherwise
        cache reloads would invalidate the train/val/test isolation that
        was established at tier0 time. We surface this as a non-fatal
        notice here so callers can pass ``pre_assigned_splits`` back into
        ``run_pipeline`` if they re-run tier0 against the same data.
    """
    if cache_path and Path(cache_path).exists():
        print(f"  Loading tier0 state from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            state = pickle.load(f)
        if isinstance(state, dict):
            assignments = state.get("split_assignments")
            if assignments:
                print(
                    f"  Tier0 cache contains {len(assignments)} pre-assigned "
                    f"split labels (strategy={state.get('split_strategy', 'unknown')}). "
                    "Downstream consumers will REUSE these assignments — "
                    "re-derivation is forbidden by Block 4 contract."
                )
            else:
                print(
                    "  ⚠️  Tier0 cache has no split_assignments; running on "
                    "an older cache. Re-run tier0 to populate them."
                )
        return state

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
    from scripts.run_tier0_test import CONFIG, run_pipeline

    print("  Running tier0 pipeline to generate synthetic data...")

    # Disable MLflow for local testing (not required for tier1-5 validation)
    CONFIG.enable_mlflow = False

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
    quality_validator: Any,  # QualityGateValidator
    data_source_validator: Any | None,  # DataSourceValidator
    trace_verifier: Any | None,  # OpikTraceVerifier
    timeout_seconds: float = 30.0,
    enforce_real_data: bool = True,
) -> AgentTestResult:
    """Test a single agent.

    Args:
        agent_name: Name of the agent to test
        config: Agent configuration dict
        mapper: Tier0OutputMapper instance
        validator: ContractValidator instance
        quality_validator: QualityGateValidator instance
        data_source_validator: DataSourceValidator instance (or None to skip)
        trace_verifier: OpikTraceVerifier instance (or None to skip)
        timeout_seconds: Maximum execution time
        enforce_real_data: If True, configure agents to use real data sources

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

        # 2. Import and instantiate agent with real-data configuration
        agent_class = import_class(config["agent_module"], config["agent_class"])
        # Pass tier0_state so agents can access trained_model, deployment_manifest, etc.
        agent_kwargs = _get_agent_kwargs(
            agent_name,
            enforce_real_data=enforce_real_data,
            tier0_state=mapper.state,
        )
        agent = agent_class(**agent_kwargs)

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
                type_err = next(
                    (te for te in validation_result.type_errors if te.get("field") == fld), None
                )
                required_valid.append(
                    FieldValidationResult(
                        name=fld,
                        expected_type=field_types.get(fld, "Any"),
                        present=present,
                        valid_type=present and type_err is None,
                        actual_type=type(result.agent_output.get(fld)).__name__
                        if present
                        else None,
                        error=type_err.get("message") if type_err else None,
                    )
                )

            optional_valid = []
            optional_present = []
            for fld in optional_fields:
                present = fld in result.agent_output
                if present:
                    optional_present.append(fld)
                    # Check if there's a type error for this field
                    type_err = next(
                        (te for te in validation_result.type_errors if te.get("field") == fld), None
                    )
                    optional_valid.append(
                        FieldValidationResult(
                            name=fld,
                            expected_type=field_types.get(fld, "Any"),
                            present=present,
                            valid_type=type_err is None,
                            actual_type=type(result.agent_output.get(fld)).__name__
                            if present
                            else None,
                            error=type_err.get("message") if type_err else None,
                        )
                    )

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

        # 6. Validate quality gate
        try:
            # Calculate contract required fields percentage
            contract_req_pct = 0.0
            contract_req_total = -1
            if result.contract_validation:
                cv = result.contract_validation
                contract_req_total = cv.required_total
                if cv.required_total > 0:
                    contract_req_pct = len(cv.required_fields_present) / cv.required_total

            quality_result = quality_validator.validate(
                agent_name=agent_name,
                output=result.agent_output,
                contract_required_fields_pct=contract_req_pct,
                contract_required_total=contract_req_total,
            )

            result.quality_gate = QualityGateDetail(
                passed=quality_result.passed,
                total_checks=quality_result.total_checks,
                checks_passed=quality_result.checks_passed,
                checks_failed=quality_result.checks_failed,
                failed_check_names=[c.check_name for c in quality_result.failed_checks],
                failed_check_messages=[c.message for c in quality_result.failed_checks],
                required_output_fields_present=quality_result.required_output_fields_present,
                required_output_fields_missing=quality_result.required_output_fields_missing,
                status_failure=quality_result.status_failure,
                status_value=quality_result.status_value,
                warnings=quality_result.warnings,
            )
        except Exception as e:
            result.quality_gate = QualityGateDetail(
                passed=False,
                warnings=[f"Quality gate validation failed: {e}"],
            )

        # 7. Validate data source (if validator provided)
        if data_source_validator is not None:
            try:
                ds_result = data_source_validator.validate(
                    agent_name=agent_name,
                    agent_output=result.agent_output,
                    execution_logs=[],  # Could capture logs if needed
                    agent_instance=agent,
                )
                result.data_source = DataSourceDetail(
                    passed=ds_result.passed,
                    detected_source=ds_result.detected_source.value,
                    acceptable_sources=[s.value for s in ds_result.acceptable_sources],
                    reject_mock=ds_result.reject_mock,
                    message=ds_result.message,
                    evidence=ds_result.evidence,
                )
            except Exception as e:
                result.data_source = DataSourceDetail(
                    passed=False,
                    message=f"Data source validation failed: {e}",
                )

        # 8. Verify observability (if verifier provided)
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
                except Exception:
                    result.trace_verification = TraceVerificationDetail(
                        trace_exists=False,
                        trace_id=trace_id,
                    )

        # 9. Determine overall success
        # Success requires ALL of:
        # 1. No execution errors
        # 2. Quality gate passes (primary check - agent-specific)
        # 3. Data source validation passes (if validator provided)
        data_source_ok = (
            result.data_source is None  # No validator = OK
            or result.data_source.passed
        )
        result.success = (
            result.error is None
            and result.quality_gate is not None
            and result.quality_gate.passed
            and data_source_ok
        )

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
        (
            "Input validation passed",
            not result.error or "input" not in (result.error or "").lower(),
        ),
        ("Agent execution completed", result.error is None),
    ]
    if result.contract_validation:
        processing_steps.append(
            (
                f"Contract validation ({result.contract_validation.state_class})",
                result.contract_validation.valid,
            )
        )
    if result.quality_gate:
        processing_steps.append(("Quality gate validation", result.quality_gate.passed))
    if result.trace_verification:
        processing_steps.append(("Opik trace captured", result.trace_verification.trace_exists))
    print_processing_section(processing_steps)

    # Validation checks
    checks = []
    if result.contract_validation:
        cv = result.contract_validation
        req_present = len(cv.required_fields_present)
        req_total = cv.required_total
        checks.append(
            (
                "Required fields present",
                req_present == req_total,
                f"{req_total} required fields",
                f"{req_present}/{req_total} present",
            )
        )
        checks.append(
            (
                "Type validation",
                len(cv.type_errors) == 0,
                "no type errors",
                f"{len(cv.type_errors)} type errors" if cv.type_errors else "all types valid",
            )
        )
    if result.quality_gate:
        qg = result.quality_gate
        checks.append(
            (
                "Quality gate",
                qg.passed,
                f"{qg.total_checks} checks pass",
                f"{qg.checks_passed}/{qg.total_checks} passed"
                if qg.total_checks > 0
                else "no checks",
            )
        )
        if qg.status_failure:
            checks.append(("Status check", False, "no failure status", f"status={qg.status_value}"))
    checks.append(
        (
            "Overall result",
            result.success,
            "success",
            "PASS" if result.success else f"FAIL: {(result.error or 'quality gate failed')[:40]}",
        )
    )
    if result.trace_verification:
        tv = result.trace_verification
        checks.append(
            (
                "Observability trace",
                tv.trace_exists,
                "trace captured",
                "trace exists" if tv.trace_exists else "no trace",
            )
        )
    print_validation_checks(checks)

    # Key metrics table
    metrics = [
        ("execution_time", result.execution_time_ms / 1000, None, None),
        ("agent_tier", result.tier, None, None),
    ]
    if result.contract_validation:
        cv = result.contract_validation
        metrics.append(
            (
                "required_fields",
                f"{len(cv.required_fields_present)}/{cv.required_total}",
                None,
                None,
            )
        )
        metrics.append(
            (
                "optional_fields",
                f"{len(cv.optional_fields_present)}/{cv.optional_total}",
                None,
                None,
            )
        )
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
        priority_metrics = [
            "overall_ate",
            "heterogeneity_score",
            "drift_score",
            "health_score",
            "overall_score",
            "confidence",
            "statistical_power",
            "p_value",
        ]
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

    # Show key output fields (full output - no truncation)
    if result.agent_output:
        print("\n  📋 Key Output Fields:")
        output_items = list(result.agent_output.items())

        # Prioritize important fields (shown first for readability)
        priority_fields = [
            "status",
            "executive_summary",
            "learning_summary",
            "overall_ate",
            "heterogeneity_score",
            "causal_effect",
            "overall_drift_score",
            "recommended_actions",
            "drift_interpretation",
            "overall_health_score",
            "health_grade",
            "health_summary",
            "recommendations",
            "prediction",
            "prediction_summary",
            "optimization_result",
            "optimization_summary",
            "drift_detected",
            "experiment_design",
            "analysis_complete",
        ]
        priority_items = [(k, v) for k, v in output_items if k in priority_fields]
        other_items = [(k, v) for k, v in output_items if k not in priority_fields]

        # Show all priority items first, then all remaining items
        for key, value in priority_items:
            _print_output_value(key, value, indent=4)
        for key, value in other_items:
            _print_output_value(key, value, indent=4)

    # Quality gate details (verbose)
    if verbose and result.quality_gate:
        qg = result.quality_gate
        print("\n  📊 Quality Gate:")
        print(f"    • Checks: {qg.checks_passed}/{qg.total_checks} passed")
        if qg.required_output_fields_present:
            print(f"    • Required output fields: {', '.join(qg.required_output_fields_present)}")
        if qg.required_output_fields_missing:
            print(f"    • Missing output fields: {', '.join(qg.required_output_fields_missing)}")
        if qg.status_failure:
            print(f"    • ❌ Status failure: {qg.status_value}")
        if qg.failed_check_messages:
            print("    • Failed checks:")
            for msg in qg.failed_check_messages:
                # No truncation - show full message
                print(f"      - {msg}")
        if qg.passed:
            print("    • QUALITY GATE: ✅ PASS")
        else:
            print("    • QUALITY GATE: ❌ FAIL")

    # Contract validation details (verbose)
    if verbose and result.contract_validation:
        cv = result.contract_validation
        if cv.type_errors:
            print("\n  ⚠️  Type Errors:")
            for te in cv.type_errors[:3]:
                print(
                    f"    • {te.get('field')}: expected {te.get('expected')}, got {te.get('actual')}"
                )
        if cv.warnings:
            print(f"\n  ⚠️  Warnings ({len(cv.warnings)}):")
            for w in cv.warnings:
                # No truncation - show full warning
                print(f"    • {w}")

    # Observability details (verbose)
    if verbose and result.trace_verification and result.trace_verification.trace_exists:
        tv = result.trace_verification
        print("\n  🔭 Observability Details:")
        if tv.trace_id:
            print(f"    • Trace ID: {tv.trace_id}")
        if tv.trace_url:
            print(f"    • URL: {tv.trace_url}")
        if tv.span_names:
            print(f"    • Spans: {', '.join(tv.span_names)}")

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
        # No truncation - show full error
        error_brief = result.error or "Unknown error"
        print_step_result("failed", f"{result.agent_name}: {error_brief}", duration_s)


def _print_output_value(key: str, value: Any, indent: int = 4) -> None:
    """Print a single output value with appropriate formatting.

    No truncation is applied - full values are shown for debugging.
    """
    prefix = " " * indent

    if value is None:
        print(f"{prefix}{key}: null")
    elif isinstance(value, dict):
        if len(value) == 0:
            print(f"{prefix}{key}: {{}}")
        elif len(value) <= 5:
            # Small/medium dict - print expanded (no truncation)
            print(f"{prefix}{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"{prefix}  {k}: {v:.4f}")
                elif isinstance(v, list):
                    print(f"{prefix}  {k}: [{len(v)} items]")
                elif isinstance(v, dict):
                    print(f"{prefix}  {k}: {{dict with {len(v)} keys}}")
                elif isinstance(v, str):
                    # Show full string value (no truncation)
                    print(f"{prefix}  {k}: {v}")
                else:
                    print(f"{prefix}  {k}: {v}")
        else:
            # Large dict - show keys and first few values
            print(f"{prefix}{key}: {{dict with {len(value)} keys}}")
            for i, (k, v) in enumerate(value.items()):
                if i >= 3:
                    print(f"{prefix}  ... and {len(value) - 3} more keys")
                    break
                if isinstance(v, float):
                    print(f"{prefix}  {k}: {v:.4f}")
                elif isinstance(v, (list, dict)):
                    print(f"{prefix}  {k}: [{type(v).__name__} with {len(v)} items]")
                else:
                    print(f"{prefix}  {k}: {v}")
    elif isinstance(value, list):
        if len(value) == 0:
            print(f"{prefix}{key}: []")
        elif len(value) <= 2 and all(isinstance(x, (str, int, float)) for x in value):
            print(f"{prefix}{key}: {value}")
        elif all(isinstance(x, str) for x in value):
            # List of strings - show all items for full visibility
            print(f"{prefix}{key}: [{len(value)} items]")
            for i, item in enumerate(value):
                print(f"{prefix}  [{i}]: {item}")
        else:
            # Show first item if it's a dict (common pattern)
            if isinstance(value[0], dict):
                print(f"{prefix}{key}: [{len(value)} items]")
                # Show first item as sample (no truncation)
                first = value[0]
                sample_keys = list(first.keys())[:4]
                sample = {k: first[k] for k in sample_keys}
                # Format values in sample (no truncation)
                for k, v in sample.items():
                    if isinstance(v, float):
                        sample[k] = round(v, 4)
                    # No string truncation - show full value
                if len(first) > 4:
                    print(f"{prefix}  [0]: {sample}... (+{len(first) - 4} more keys)")
                else:
                    print(f"{prefix}  [0]: {sample}")
            else:
                print(f"{prefix}{key}: [{len(value)} items]")
    elif isinstance(value, float):
        print(f"{prefix}{key}: {value:.4f}")
    elif isinstance(value, str):
        if "\n" in value:
            # Multi-line string - show all lines with proper formatting
            lines = value.split("\n")
            print(f"{prefix}{key}:")
            for line in lines[:10]:  # Limit to first 10 lines for readability
                print(f"{prefix}  {line}")
            if len(lines) > 10:
                print(f"{prefix}  ... ({len(lines) - 10} more lines)")
        else:
            # Single-line string - show full value (no truncation)
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
            all_agents = [r.agent_name for r in passed] + [
                f"{r.agent_name} (FAILED)" for r in failed
            ]
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
            # No truncation - show full error
            error_msg = r.error or "Contract validation failed"
            print(f"  \u274c {r.agent_name} (Tier {r.tier})")
            print(f"     Error: {error_msg}")
            if r.error_traceback and verbose:
                # Show last line of traceback (full)
                last_line = r.error_traceback.strip().split("\n")[-1]
                print(f"     Last line: {last_line}")

    # Quality gate summary
    quality_gates_passed = sum(1 for r in results if r.quality_gate and r.quality_gate.passed)
    quality_gates_failed = sum(1 for r in results if r.quality_gate and not r.quality_gate.passed)
    status_failures = sum(1 for r in results if r.quality_gate and r.quality_gate.status_failure)

    print("\nQUALITY GATES:")
    print(f"  Passed: \033[92m{quality_gates_passed}\033[0m/{total}")
    print(f"  Failed: \033[91m{quality_gates_failed}\033[0m/{total}")
    if status_failures > 0:
        print(f"  Status Failures: {status_failures} (agents returned error/failed status)")

    # Quality gate failure details
    qg_failed = [r for r in results if r.quality_gate and not r.quality_gate.passed]
    if qg_failed:
        print("\n  Quality Gate Failures:")
        for r in qg_failed:
            qg = r.quality_gate
            if qg.status_failure:
                print(f"    • {r.agent_name}: status={qg.status_value}")
            elif qg.failed_check_messages:
                # No truncation - show full message
                msg = qg.failed_check_messages[0]
                print(f"    • {r.agent_name}: {msg}")
            else:
                print(f"    • {r.agent_name}: {qg.checks_failed} checks failed")

    # Contract validation summary
    contracts_valid = sum(
        1 for r in results if r.contract_validation and r.contract_validation.valid
    )
    type_errors_total = sum(
        len(r.contract_validation.type_errors) for r in results if r.contract_validation
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
    timeout_seconds: float | None = None,
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
        timeout_seconds: Explicit per-agent timeout FLOOR (None = use each
            agent's configured timeout, defaulting to 30s)
        verbose: Show detailed output for each agent

    Returns:
        Full test results dict
    """
    from src.testing import (
        ContractValidator,
        DataSourceValidator,
        OpikTraceVerifier,
        QualityGateValidator,
        Tier0OutputMapper,
    )

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
            raise ValueError("No tier0 state available. Use --run-tier0-first or --tier0-cache")

    experiment_id = tier0_state.get("experiment_id", "unknown")
    print(f"Tier0 Experiment ID: {experiment_id}")

    # Initialize components
    mapper = Tier0OutputMapper(tier0_state)
    # ContractValidator validates TypedDict structure
    # QualityGateValidator enforces per-agent quality thresholds
    # DataSourceValidator ensures agents use real data sources (not mocks)
    validator = ContractValidator()
    quality_validator = QualityGateValidator()
    data_source_validator = DataSourceValidator()

    print("Data Source Validation: Enabled")

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
        print("\nTier0 Data:")
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
        # Per-agent timeout; an explicit CLI --timeout is a FLOOR (see
        # resolve_agent_timeout) so a short per-agent config value cannot
        # silently cap the run below what the caller asked for.
        agent_timeout = resolve_agent_timeout(config, timeout_seconds)
        result = await test_agent(
            agent_name=agent_name,
            config=config,
            mapper=mapper,
            validator=validator,
            quality_validator=quality_validator,
            data_source_validator=data_source_validator,
            trace_verifier=trace_verifier,
            timeout_seconds=agent_timeout,
            enforce_real_data=True,
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
        "quality_gate_summary": {
            "passed": sum(1 for r in results if r.quality_gate and r.quality_gate.passed),
            "failed": sum(1 for r in results if r.quality_gate and not r.quality_gate.passed),
            "status_failures": sum(
                1 for r in results if r.quality_gate and r.quality_gate.status_failure
            ),
            "failed_agents": [
                {
                    "agent": r.agent_name,
                    "status_failure": r.quality_gate.status_failure if r.quality_gate else False,
                    "status_value": r.quality_gate.status_value if r.quality_gate else None,
                    "failed_checks": r.quality_gate.failed_check_names if r.quality_gate else [],
                }
                for r in results
                if r.quality_gate and not r.quality_gate.passed
            ],
        },
        "observability_summary": {
            "traces_created": sum(
                1 for r in results if r.trace_verification and r.trace_verification.trace_exists
            ),
            "traces_verified": sum(
                1 for r in results if r.trace_verification and r.trace_verification.metadata_valid
            ),
            "opik_health": "healthy" if trace_verifier else "not_checked",
        },
        "data_source_summary": {
            "validated": sum(1 for r in results if r.data_source is not None),
            "passed": sum(1 for r in results if r.data_source and r.data_source.passed),
            "failed": sum(1 for r in results if r.data_source and not r.data_source.passed),
            "mock_detected": sum(
                1 for r in results if r.data_source and r.data_source.detected_source == "mock"
            ),
            "failed_agents": [
                {
                    "agent": r.agent_name,
                    "detected_source": r.data_source.detected_source if r.data_source else None,
                    "message": r.data_source.message if r.data_source else None,
                }
                for r in results
                if r.data_source and not r.data_source.passed
            ],
        },
    }

    # Build tier breakdown
    for tier in sorted({r.tier for r in results}):
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


def _parse_expected_fail(raw: str | None) -> set[str]:
    """Parse the comma-separated expected-fail allow-list into a set of names.

    Names are case-folded so the allow-list matches the (snake_case lowercase)
    agent names case-insensitively — a maintainer typo like ``Orchestrator`` must
    still match ``orchestrator`` rather than mis-routing it to a hard fail
    (#616 hardening, codex M-1).
    """
    if not raw:
        return set()
    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def summarize_results(
    results_path: str,
    expected_fail: str | None,
    step_summary_path: str | None,
) -> int:
    """Render a per-agent table + enforce the expected-fail allow-list (#616).

    Reads the harness results JSON (the same ``full_results`` the run writes to
    ``docs/results/tier1_5_pipeline_latest.json``), writes a human-readable
    pass/fail table to ``step_summary_path`` (``$GITHUB_STEP_SUMMARY`` in CI), and
    returns a process exit code:

    - ``0`` when every failing agent (if any) is on ``expected_fail`` — the #600
      monitored-alarm contract is preserved for the known set.
    - ``1`` when ANY agent NOT on ``expected_fail`` failed — a NEW regression
      beyond the known set, which must HARD FAIL so a green check can no longer
      silently mask it.
    - ``1`` when the results file is missing/unreadable (the harness was expected
      to have produced it; absence is itself a failure signal, not a pass).

    This is intentionally separate from the alarm-only ``run-harness`` step: that
    step keeps its ``::warning`` + ``exit 0`` (issue #600); this step is the
    honest gate (issue #616).
    """
    allow = _parse_expected_fail(expected_fail)

    path = Path(results_path)
    if not path.exists():
        msg = f"Tier 1-5 results JSON not found at {results_path}"
        print(f"::error title=Tier 1-5 results missing::{msg}")
        _write_step_summary(step_summary_path, f"## Tier 1-5 Agent Harness\n\n❌ {msg}\n")
        return 1

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        msg = f"Could not parse Tier 1-5 results JSON: {exc}"
        print(f"::error title=Tier 1-5 results unreadable::{msg}")
        _write_step_summary(step_summary_path, f"## Tier 1-5 Agent Harness\n\n❌ {msg}\n")
        return 1

    rows = data.get("results") or []
    if not rows:
        # Fail CLOSED: a present-but-empty results set means the harness ran no
        # agent (it always runs the full AGENT_METHOD_MAP). A vacuous 0/0 must
        # not slip through as a pass (#616 hardening, codex L-2).
        msg = (
            "Tier 1-5 results JSON contains no agent rows — the harness ran no "
            "agent; treating as a hard failure rather than a vacuous 0/0 pass"
        )
        print(f"::error title=Tier 1-5 empty results::{msg}")
        _write_step_summary(step_summary_path, f"## Tier 1-5 Agent Harness\n\n❌ {msg}\n")
        return 1

    summary = data.get("summary") or {}
    total = summary.get("total_agents", len(rows))
    passed = summary.get("passed", sum(1 for r in rows if r.get("success")))

    # Categorise failures relative to the allow-list (case-insensitive — see
    # _parse_expected_fail, codex M-1).
    failed_rows = [r for r in rows if not r.get("success")]
    new_failures = [r for r in failed_rows if (r.get("agent_name") or "").lower() not in allow]
    known_failures = [r for r in failed_rows if (r.get("agent_name") or "").lower() in allow]

    lines: list[str] = []
    lines.append("## Tier 1-5 Agent Harness")
    lines.append("")
    lines.append(f"**{passed}/{total} agents passed** (keyless smoke harness).")
    lines.append("")
    lines.append("| Agent | Tier | Result | Data source | Quality gate | Detail |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=lambda x: (x.get("tier") or 0, x.get("agent_name") or "")):
        name = r.get("agent_name", "?")
        tier = r.get("tier", "?")
        success = bool(r.get("success"))
        ds = r.get("data_source") or {}
        detected = ds.get("detected_source", "—")
        # Honest label: marked-mock agents are a "plumbing-only PASS".
        if detected == "mock":
            ds_label = "mock (plumbing-only)"
        else:
            ds_label = str(detected)
        qg = r.get("quality_gate") or {}
        qg_label = "pass" if qg.get("passed") else ("fail" if qg else "—")
        if success:
            result_label = "✅ PASS"
            detail = ""
        elif name.lower() in allow:
            result_label = "⚠️ KNOWN-FAIL"
            detail = "expected-fail allow-list (#600 alarm)"
        else:
            result_label = "❌ NEW-FAIL"
            detail = str(r.get("error") or "see results artifact")[:80]
        lines.append(f"| {name} | {tier} | {result_label} | {ds_label} | {qg_label} | {detail} |")
    lines.append("")

    if known_failures:
        names = ", ".join(sorted(r.get("agent_name", "?") for r in known_failures))
        lines.append(f"⚠️ **Known (allow-listed) failures** — non-blocking (#600): {names}")
        lines.append("")
    if new_failures:
        names = ", ".join(sorted(r.get("agent_name", "?") for r in new_failures))
        lines.append(
            f"❌ **NEW failures (not on allow-list)** — hard-failing this check (#616): {names}"
        )
        lines.append("")
        print(
            "::error title=Tier 1-5 NEW agent regression::"
            f"{names} failed and are NOT on TIER1_5_EXPECTED_FAIL_AGENTS. "
            "This is a new regression and hard-fails the check (issue #616). "
            "See the tier1-5-results artifact for per-agent detail."
        )
    else:
        lines.append("✅ No new (non-allow-listed) agent failures.")
        lines.append("")

    _write_step_summary(step_summary_path, "\n".join(lines) + "\n")

    return 1 if new_failures else 0


def _write_step_summary(step_summary_path: str | None, content: str) -> None:
    """Append ``content`` to the GitHub step summary file (or print if absent)."""
    if not step_summary_path:
        print(content)
        return
    try:
        with open(step_summary_path, "a", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as exc:
        # Never let summary-writing failure mask the gate result; just log.
        print(f"::warning::Could not write step summary to {step_summary_path}: {exc}")
        print(content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Tier 1-5 agent tests using tier0 outputs")
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
        default=None,
        help=(
            "Per-agent timeout FLOOR in seconds: per-agent configured timeouts "
            "LONGER than this are preserved, shorter ones are raised to it. "
            "Default: each agent's configured timeout (30s base)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
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
    # Summarize-only mode (issue #616): do NOT run the harness; read an existing
    # results JSON, emit a per-agent table to the step summary, and enforce the
    # expected-fail allow-list. Used by the tier1-5-test.yml honest-gate step.
    parser.add_argument(
        "--summarize",
        type=str,
        default=None,
        metavar="RESULTS_JSON",
        help="Summarize an existing results JSON + enforce the expected-fail "
        "allow-list, then exit (does not run the harness).",
    )
    parser.add_argument(
        "--expected-fail",
        type=str,
        default="",
        help="Comma-separated agents whose failure is allow-listed "
        "(non-blocking, #600). Used only with --summarize.",
    )
    parser.add_argument(
        "--step-summary",
        type=str,
        default=None,
        help="Path to append the markdown summary to (e.g. $GITHUB_STEP_SUMMARY). "
        "Used only with --summarize.",
    )

    args = parser.parse_args()

    # Summarize-only mode short-circuits the full harness run (issue #616).
    if args.summarize is not None:
        sys.exit(
            summarize_results(
                results_path=args.summarize,
                expected_fail=args.expected_fail,
                step_summary_path=args.step_summary,
            )
        )

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

            clean_text = re.sub(r"\x1b\[[0-9;]*m", "", text)
            self.buffer.write(clean_text)
            self.console.write(text)

        def flush(self):
            self.console.flush()
            self.buffer.flush()

    # Capture output
    original_stdout = sys.stdout
    if not args.no_save:
        sys.stdout = TeeOutput(original_stdout, output_buffer)

    # Track whether any agent failed so we can propagate a non-zero
    # exit. Issue #263 acceptance: the harness must hard-fail the
    # GitHub Actions check when any tier1-5 agent regresses; the
    # workflow's ``continue-on-error`` flip is necessary but not
    # sufficient without a real exit code here. ``run_tests`` records
    # failures in ``summary["failed"]`` but does not raise.
    full_results: dict[str, Any] | None = None
    try:
        # Run tests
        full_results = asyncio.run(
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
            md_content += (
                f"- **Observability**: {'skipped' if args.skip_observability else 'enabled'}\n"
            )
            md_content += (
                f"- **Timeout**: {args.timeout}s floor per agent\n\n"
                if args.timeout is not None
                else "- **Timeout**: per-agent config (30s base)\n\n"
            )
            md_content += "## Results\n\n"
            md_content += "```\n"
            md_content += output_buffer.getvalue()
            md_content += "```\n"

            with open(output_file, "w") as f:
                f.write(md_content)

            print(f"\n📄 Results saved to: {output_file}")

    # Propagate non-zero exit when any agent failed. Defensive on the
    # results-dict shape: ``run_tests`` always returns ``summary.failed``
    # as an int, but if any future refactor swaps the shape we err on
    # the side of failing loudly rather than silently exiting 0.
    if full_results is None:
        sys.exit(1)
    summary = full_results.get("summary") or {}
    failed_count = summary.get("failed")
    if not isinstance(failed_count, int) or failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
