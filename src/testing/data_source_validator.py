"""Data Source Validator for Agent Outputs.

Validates that agents use appropriate data sources (real Supabase vs mock data).
Detects mock data usage patterns and enforces data source requirements per agent.

The problem this solves:
- Some agents silently fall back to mock/hardcoded data when Supabase is unavailable
- This leads to agents passing tests with fake data that doesn't reflect reality
- For example, health_score returns 100% when using mock data

Detection strategies:
1. health_score: Perfect 100% scores indicate mock data (real systems have variance)
2. gap_analyzer/heterogeneous_optimizer: Check for MockDataConnector in logs/metadata
3. resource_optimizer: Computational only (no external data needed)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources agents can use."""

    SUPABASE = "supabase"  # Real Supabase synthetic/production data
    MOCK = "mock"  # Mock/hardcoded fallback data
    TIER0_PASSTHROUGH = "tier0"  # Data passed through from tier0 pipeline
    COMPUTATIONAL = "computational"  # Agent is purely computational (no external data)
    UNKNOWN = "unknown"  # Could not determine data source


@dataclass
class DataSourceValidationResult:
    """Result of data source validation for an agent."""

    agent_name: str
    passed: bool
    detected_source: DataSourceType
    acceptable_sources: list[DataSourceType] = field(default_factory=list)
    reject_mock: bool = False
    message: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Get summary string."""
        status = "PASS" if self.passed else "FAIL"
        return f"{status}: {self.agent_name} uses {self.detected_source.value} ({self.message})"


class DataSourceValidator:
    """Validates that agents use appropriate data sources.

    Each agent has specific data source requirements:
    - Some agents MUST use Supabase data (reject mock)
    - Some agents are computational only (no external data needed)
    - Some agents can use tier0 passthrough data

    Usage:
        validator = DataSourceValidator()
        result = validator.validate(
            agent_name="health_score",
            agent_output={"overall_health_score": 100.0, ...},
        )
        if not result.passed:
            print(f"Data source validation failed: {result.message}")
    """

    # Agent data source requirements
    # acceptable: list of allowed data source types
    # reject_mock: if True, explicitly reject mock data even if otherwise acceptable
    AGENT_DATA_SOURCE_REQUIREMENTS: dict[str, dict[str, Any]] = {
        "health_score": {
            "acceptable": [DataSourceType.SUPABASE],
            "reject_mock": True,
            "description": "Must check real system components",
        },
        "gap_analyzer": {
            "acceptable": [DataSourceType.SUPABASE, DataSourceType.TIER0_PASSTHROUGH],
            "reject_mock": True,
            "description": "Must analyze real or tier0 data for meaningful gaps",
        },
        "heterogeneous_optimizer": {
            "acceptable": [DataSourceType.SUPABASE, DataSourceType.TIER0_PASSTHROUGH],
            "reject_mock": True,
            "description": "Must have real data for CATE estimation",
        },
        "resource_optimizer": {
            "acceptable": [DataSourceType.COMPUTATIONAL, DataSourceType.TIER0_PASSTHROUGH],
            "reject_mock": False,
            "description": "Purely computational, uses input constraints only",
        },
        "causal_impact": {
            "acceptable": [DataSourceType.SUPABASE, DataSourceType.TIER0_PASSTHROUGH],
            "reject_mock": False,
            "description": "Can use tier0 data for causal analysis",
        },
        "drift_monitor": {
            "acceptable": [DataSourceType.TIER0_PASSTHROUGH, DataSourceType.SUPABASE],
            "reject_mock": False,
            "description": "Compares reference and current data from tier0",
        },
        "experiment_designer": {
            "acceptable": [DataSourceType.TIER0_PASSTHROUGH, DataSourceType.COMPUTATIONAL],
            "reject_mock": False,
            "description": "Designs experiments based on tier0 population data",
        },
        "prediction_synthesizer": {
            "acceptable": [DataSourceType.TIER0_PASSTHROUGH],
            "reject_mock": False,
            "description": "Synthesizes predictions from tier0 models",
        },
        "explainer": {
            "acceptable": [DataSourceType.TIER0_PASSTHROUGH],
            "reject_mock": False,
            "description": "Explains tier0 analysis results",
        },
        "feedback_learner": {
            "acceptable": [DataSourceType.SUPABASE, DataSourceType.TIER0_PASSTHROUGH],
            "reject_mock": False,
            "description": "Learns from feedback stored in Supabase or tier0",
        },
        "orchestrator": {
            "acceptable": [DataSourceType.COMPUTATIONAL],
            "reject_mock": False,
            "description": "Routes queries, no external data needed",
        },
        "tool_composer": {
            "acceptable": [DataSourceType.COMPUTATIONAL],
            "reject_mock": False,
            "description": "Composes tools, no external data needed",
        },
    }

    def __init__(
        self,
        custom_requirements: dict[str, dict[str, Any]] | None = None,
    ):
        """Initialize validator.

        Args:
            custom_requirements: Override default requirements for specific agents
        """
        self.requirements = {**self.AGENT_DATA_SOURCE_REQUIREMENTS}
        if custom_requirements:
            self.requirements.update(custom_requirements)

    def validate(
        self,
        agent_name: str,
        agent_output: dict[str, Any],
        execution_logs: list[str] | None = None,
        agent_instance: Any | None = None,
    ) -> DataSourceValidationResult:
        """Validate that agent used appropriate data source.

        Args:
            agent_name: Name of the agent
            agent_output: Agent's output dictionary
            execution_logs: Optional captured logs from execution
            agent_instance: Optional agent instance for inspection

        Returns:
            DataSourceValidationResult with validation details
        """
        # Get requirements for this agent
        reqs = self.requirements.get(agent_name)
        if reqs is None:
            return DataSourceValidationResult(
                agent_name=agent_name,
                passed=True,
                detected_source=DataSourceType.UNKNOWN,
                message="No data source requirements configured for agent",
            )

        acceptable = reqs.get("acceptable", [])
        reject_mock = reqs.get("reject_mock", False)

        # Detect data source
        detected, evidence = self._detect_data_source(
            agent_name=agent_name,
            agent_output=agent_output,
            execution_logs=execution_logs or [],
            agent_instance=agent_instance,
        )

        # Check if detected source is acceptable
        is_acceptable = detected in acceptable
        is_mock_rejected = reject_mock and detected == DataSourceType.MOCK

        passed = is_acceptable and not is_mock_rejected

        # Build message
        if passed:
            message = f"Data source '{detected.value}' is acceptable"
        elif is_mock_rejected:
            message = f"Mock data detected but reject_mock=True for {agent_name}"
        else:
            acceptable_names = [s.value for s in acceptable]
            message = f"Data source '{detected.value}' not in acceptable sources: {acceptable_names}"

        return DataSourceValidationResult(
            agent_name=agent_name,
            passed=passed,
            detected_source=detected,
            acceptable_sources=acceptable,
            reject_mock=reject_mock,
            message=message,
            evidence=evidence,
        )

    def _detect_data_source(
        self,
        agent_name: str,
        agent_output: dict[str, Any],
        execution_logs: list[str],
        agent_instance: Any | None,
    ) -> tuple[DataSourceType, list[str]]:
        """Detect which data source the agent used.

        Returns:
            Tuple of (detected source type, evidence list)
        """
        evidence: list[str] = []

        # Agent-specific detection logic
        if agent_name == "health_score":
            return self._detect_health_score_source(agent_output, evidence)
        elif agent_name == "gap_analyzer":
            return self._detect_gap_analyzer_source(agent_output, execution_logs, evidence)
        elif agent_name == "heterogeneous_optimizer":
            return self._detect_heterogeneous_optimizer_source(
                agent_output, execution_logs, agent_instance, evidence
            )
        elif agent_name == "resource_optimizer":
            # Resource optimizer is purely computational
            evidence.append("resource_optimizer is computational-only agent")
            return DataSourceType.COMPUTATIONAL, evidence
        elif agent_name in ("orchestrator", "tool_composer"):
            evidence.append(f"{agent_name} is computational-only (routing/composition)")
            return DataSourceType.COMPUTATIONAL, evidence
        elif agent_name == "drift_monitor":
            return self._detect_drift_monitor_source(agent_output, evidence)
        elif agent_name == "experiment_designer":
            return self._detect_experiment_designer_source(agent_output, evidence)
        else:
            # Default: check for tier0 passthrough indicators
            return self._detect_tier0_passthrough(agent_output, evidence)

    def _detect_health_score_source(
        self,
        agent_output: dict[str, Any],
        evidence: list[str],
    ) -> tuple[DataSourceType, list[str]]:
        """Detect data source for health_score agent.

        Mock indicator: Perfect 100% scores (real systems have variance)
        """
        overall_score = agent_output.get("overall_health_score")
        component_score = agent_output.get("component_health_score")

        # Check for perfect scores (mock indicator)
        if overall_score == 100.0:
            evidence.append(f"overall_health_score is exactly 100.0 (mock indicator)")
            return DataSourceType.MOCK, evidence

        if component_score == 1.0:
            # Check if all components are healthy (another mock indicator)
            component_statuses = agent_output.get("component_statuses", [])
            if component_statuses:
                all_healthy = all(
                    s.get("status") == "healthy" for s in component_statuses
                )
                if all_healthy and len(component_statuses) >= 3:
                    evidence.append(
                        f"All {len(component_statuses)} components report 'healthy' "
                        f"with component_health_score=1.0 (mock indicator)"
                    )
                    return DataSourceType.MOCK, evidence

        # Real data would have some variance
        evidence.append(f"Health scores show variance (overall={overall_score})")
        return DataSourceType.SUPABASE, evidence

    def _detect_gap_analyzer_source(
        self,
        agent_output: dict[str, Any],
        execution_logs: list[str],
        evidence: list[str],
    ) -> tuple[DataSourceType, list[str]]:
        """Detect data source for gap_analyzer agent.

        Mock indicators:
        - "MockDataConnector" in logs
        - use_mock=True in agent configuration
        """
        # Check execution logs for mock connector
        log_text = "\n".join(execution_logs)
        if "MockDataConnector" in log_text or "mock" in log_text.lower():
            evidence.append("'MockDataConnector' found in execution logs")
            return DataSourceType.MOCK, evidence

        # Check if output has data_source metadata
        data_source = agent_output.get("data_source")
        if data_source:
            if "mock" in str(data_source).lower():
                evidence.append(f"data_source field indicates mock: {data_source}")
                return DataSourceType.MOCK, evidence
            evidence.append(f"data_source field indicates: {data_source}")
            return DataSourceType.SUPABASE, evidence

        # Check for tier0 passthrough indicators
        if agent_output.get("tier0_experiment_id"):
            evidence.append("tier0_experiment_id present (tier0 passthrough)")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        # Default to SUPABASE if no mock indicators
        evidence.append("No mock indicators found, assuming Supabase")
        return DataSourceType.SUPABASE, evidence

    def _detect_heterogeneous_optimizer_source(
        self,
        agent_output: dict[str, Any],
        execution_logs: list[str],
        agent_instance: Any | None,
        evidence: list[str],
    ) -> tuple[DataSourceType, list[str]]:
        """Detect data source for heterogeneous_optimizer agent.

        Mock indicators:
        - MockDataConnector in data_connector
        - "Mock" in logs
        - Fallback to mock in _fetch_data
        """
        # Check execution logs
        log_text = "\n".join(execution_logs)
        if "MockDataConnector" in log_text:
            evidence.append("'MockDataConnector' found in execution logs")
            return DataSourceType.MOCK, evidence

        if "Falling back to MockDataConnector" in log_text:
            evidence.append("Fallback to MockDataConnector detected in logs")
            return DataSourceType.MOCK, evidence

        # Check agent instance if available
        if agent_instance is not None:
            # Try to find data_connector in the agent or its nodes
            connector_type = self._get_data_connector_type(agent_instance)
            if connector_type:
                if "Mock" in connector_type:
                    evidence.append(f"Agent data_connector is {connector_type}")
                    return DataSourceType.MOCK, evidence
                evidence.append(f"Agent data_connector is {connector_type}")
                return DataSourceType.SUPABASE, evidence

        # Check for tier0 passthrough
        if agent_output.get("tier0_experiment_id"):
            evidence.append("tier0_experiment_id present")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        # Default: no mock indicators found
        evidence.append("No mock indicators found, assuming Supabase")
        return DataSourceType.SUPABASE, evidence

    def _detect_tier0_passthrough(
        self,
        agent_output: dict[str, Any],
        evidence: list[str],
    ) -> tuple[DataSourceType, list[str]]:
        """Default detection for tier0 passthrough.

        Tier0 passthrough indicators:
        - tier0_experiment_id field
        - analysis based on tier0 data
        """
        if agent_output.get("tier0_experiment_id"):
            evidence.append("tier0_experiment_id present in output")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        # If status indicates success and no error, assume tier0 passthrough
        status = agent_output.get("status")
        if status in ("completed", "success", "analyzing"):
            evidence.append(f"Agent completed with status={status}, assuming tier0 passthrough")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        evidence.append("Could not determine data source")
        return DataSourceType.UNKNOWN, evidence

    def _detect_drift_monitor_source(
        self,
        agent_output: dict[str, Any],
        evidence: list[str],
    ) -> tuple[DataSourceType, list[str]]:
        """Detect data source for drift_monitor agent.

        Drift monitor receives tier0 data through the mapper and processes it.
        Key indicators: data_drift_results, overall_drift_score exist.
        """
        # Check for tier0 passthrough via standard fields
        if agent_output.get("tier0_experiment_id"):
            evidence.append("tier0_experiment_id present in output")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        # Check for drift results (indicates actual processing)
        has_drift_results = (
            "data_drift_results" in agent_output
            or "overall_drift_score" in agent_output
            or "features_with_drift" in agent_output
        )
        if has_drift_results:
            evidence.append("Drift results present, using tier0 passthrough data")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        # Fall back to status check
        status = agent_output.get("status")
        if status in ("completed", "success"):
            evidence.append(f"Agent completed with status={status}")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        evidence.append("Could not determine drift_monitor data source")
        return DataSourceType.UNKNOWN, evidence

    def _detect_experiment_designer_source(
        self,
        agent_output: dict[str, Any],
        evidence: list[str],
    ) -> tuple[DataSourceType, list[str]]:
        """Detect data source for experiment_designer agent.

        Experiment designer is primarily computational (designs experiments based on
        population parameters from tier0 data). It doesn't query external databases.
        """
        # Check for tier0 passthrough via standard fields
        if agent_output.get("tier0_experiment_id"):
            evidence.append("tier0_experiment_id present in output")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        # Check for design output (indicates actual processing)
        has_design = (
            "design_type" in agent_output
            or "experiment_template" in agent_output
            or "design_rationale" in agent_output
        )
        if has_design:
            evidence.append("Experiment design present, computational agent")
            return DataSourceType.COMPUTATIONAL, evidence

        # Fall back to status check
        status = agent_output.get("status")
        if status in ("completed", "success"):
            evidence.append(f"Agent completed with status={status}")
            return DataSourceType.COMPUTATIONAL, evidence

        evidence.append("Could not determine experiment_designer data source")
        return DataSourceType.UNKNOWN, evidence

    def _get_data_connector_type(self, agent_instance: Any) -> str | None:
        """Extract data connector type name from agent instance.

        Searches for data_connector attribute in agent or its internal nodes.
        """
        # Direct attribute check
        if hasattr(agent_instance, "data_connector"):
            return type(agent_instance.data_connector).__name__

        # Check graph nodes (for LangGraph agents)
        if hasattr(agent_instance, "_graph"):
            graph = agent_instance._graph
            if hasattr(graph, "nodes"):
                for node in graph.nodes.values():
                    if hasattr(node, "data_connector"):
                        return type(node.data_connector).__name__

        # Check internal nodes dict
        if hasattr(agent_instance, "_nodes"):
            for node in agent_instance._nodes.values():
                if hasattr(node, "data_connector"):
                    return type(node.data_connector).__name__

        return None

    def get_requirements(self, agent_name: str) -> dict[str, Any] | None:
        """Get data source requirements for an agent."""
        return self.requirements.get(agent_name)

    def list_agents_with_requirements(self) -> list[str]:
        """List all agents with configured data source requirements."""
        return list(self.requirements.keys())
