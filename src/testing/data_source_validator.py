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
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# In-band marker stamped on every keyless-harness mock LLM response / client
# (see ``src/utils/mock_llm.py``: ``MOCK_MARKER`` + ``MarkedMockChatLLM``). We
# duplicate the literal here rather than import it so the validator stays free of
# an agent-runtime dependency and is importable in lightweight contexts.
MOCK_LLM_MARKER = "mock_response_for_dev_only"

# Opt-in flag that authorises agents to fall back to a MARKED mock LLM when a
# provider key is absent (#606 item C). Mirrors ``src/utils/mock_llm.MOCK_LLM_FLAG``.
_MOCK_LLM_FLAG = "E2I_ALLOW_MOCK_LLM"
_MOCK_LLM_TRUTHY = {"1", "true", "yes", "on"}

# Provider -> the API-key env var that ``src/utils/llm_factory`` requires. When
# the configured provider's key is absent AND the opt-in flag is set, an
# LLM-dependent agent constructs a MARKED mock (the exact gate this mirrors).
_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Agents whose primary "computation" is an LLM call. In the keyless harness these
# transparently fall back to a MARKED mock LLM, so a COMPUTATIONAL "PASS" would
# silently equate canned reasoning with real reasoning (#616). We detect the
# mock and downgrade them to an explicit MOCK / "plumbing-only PASS".
_LLM_DEPENDENT_AGENTS = frozenset({"orchestrator", "tool_composer", "experiment_designer"})


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
            # LLM-dependent (#616): in the keyless harness it falls back to a
            # MARKED mock LLM. MOCK is therefore an ACCEPTABLE source — it still
            # exercises the routing/synthesis plumbing — but is recorded as MOCK
            # (a "plumbing-only PASS") so the signal stays honest. reject_mock is
            # False because a marked dev-only mock here is expected, not a leak.
            "acceptable": [DataSourceType.COMPUTATIONAL, DataSourceType.MOCK],
            "reject_mock": False,
            "description": "Routes queries; LLM-driven (marked-mock in keyless harness)",
        },
        "tool_composer": {
            "acceptable": [DataSourceType.COMPUTATIONAL, DataSourceType.MOCK],
            "reject_mock": False,
            "description": "Composes tools; LLM-driven (marked-mock in keyless harness)",
        },
        "experiment_designer": {
            # validity_audit node falls back to a MARKED MockValidityLLM in the
            # keyless harness (#471/#606), surfacing the marker in agent_output.
            "acceptable": [
                DataSourceType.TIER0_PASSTHROUGH,
                DataSourceType.COMPUTATIONAL,
                DataSourceType.MOCK,
            ],
            "reject_mock": False,
            "description": "Designs experiments; LLM validity-audit (marked-mock in keyless harness)",
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
            message = (
                f"Data source '{detected.value}' not in acceptable sources: {acceptable_names}"
            )

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
            # LLM-driven agents (#616). In the keyless harness they fall back to a
            # MARKED mock LLM; detect that REAL marker (not "computational"
            # silently masking canned reasoning) and record MOCK. Falls through to
            # COMPUTATIONAL only when no mock marker is reachable (e.g. a real key
            # is present, or the agent ran with an injected real LLM).
            mock_evidence = self._detect_marked_mock_llm(
                agent_name=agent_name,
                agent_output=agent_output,
                execution_logs=execution_logs,
                agent_instance=agent_instance,
            )
            if mock_evidence:
                evidence.extend(mock_evidence)
                return DataSourceType.MOCK, evidence
            evidence.append(f"{agent_name} is computational-only (routing/composition)")
            return DataSourceType.COMPUTATIONAL, evidence
        elif agent_name == "drift_monitor":
            return self._detect_drift_monitor_source(agent_output, evidence)
        elif agent_name == "experiment_designer":
            return self._detect_experiment_designer_source(
                agent_output, execution_logs, agent_instance, evidence
            )
        elif agent_name == "causal_impact":
            return self._detect_causal_impact_source(agent_output, evidence)
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
            evidence.append("overall_health_score is exactly 100.0 (mock indicator)")
            return DataSourceType.MOCK, evidence

        if component_score == 1.0:
            # Check if all components are healthy (another mock indicator)
            component_statuses = agent_output.get("component_statuses", [])
            if component_statuses:
                all_healthy = all(s.get("status") == "healthy" for s in component_statuses)
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

    def _detect_causal_impact_source(
        self,
        agent_output: dict[str, Any],
        evidence: list[str],
    ) -> tuple[DataSourceType, list[str]]:
        """Detect data source for causal_impact agent.

        H1/H2: a causal analysis that ran end-to-end used tier0 data REGARDLESS
        of its robustness verdict. After the fail-open remediation a refutation
        that BLOCKS a weak claim yields ``status="failed"`` — so the default
        ``_detect_tier0_passthrough`` (which infers tier0 only from
        ``status in {completed, success}``) would mis-label an honest fail-closed
        block as ``UNKNOWN``. Detect the real analysis directly: a numeric ATE
        plus an executed refutation suite proves the agent processed the tier0
        frame, whether the gate PROCEEDed, REVIEWed, or BLOCKed.
        """
        if agent_output.get("tier0_experiment_id"):
            evidence.append("tier0_experiment_id present in output")
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        ate = agent_output.get("ate_estimate")
        refutation_total = agent_output.get("refutation_tests_total", 0) or 0
        if isinstance(ate, (int, float)) and not isinstance(ate, bool) and refutation_total > 0:
            evidence.append(
                "Numeric ATE + executed refutation suite "
                f"({refutation_total} tests) → tier0 passthrough (verdict-independent)"
            )
            return DataSourceType.TIER0_PASSTHROUGH, evidence

        # Fall back to the default status-based heuristic (covers completed/
        # success outputs that lack an explicit refutation count).
        return self._detect_tier0_passthrough(agent_output, evidence)

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
        execution_logs: list[str],
        agent_instance: Any | None,
        evidence: list[str],
    ) -> tuple[DataSourceType, list[str]]:
        """Detect data source for experiment_designer agent.

        Experiment designer is primarily computational (designs experiments based on
        population parameters from tier0 data). It doesn't query external databases.
        In the keyless harness its validity_audit node uses a MARKED MockValidityLLM
        (#471/#606) whose marker surfaces in ``agent_output`` — detect that first so
        a canned validity audit is recorded as MOCK, not COMPUTATIONAL (#616).
        """
        # Marked-mock LLM takes precedence: a canned validity audit must not be
        # reported as a real computational design.
        mock_evidence = self._detect_marked_mock_llm(
            agent_name="experiment_designer",
            agent_output=agent_output,
            execution_logs=execution_logs,
            agent_instance=agent_instance,
        )
        if mock_evidence:
            evidence.extend(mock_evidence)
            return DataSourceType.MOCK, evidence

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

    def _detect_marked_mock_llm(
        self,
        agent_name: str,
        agent_output: dict[str, Any],
        execution_logs: list[str],
        agent_instance: Any | None,
    ) -> list[str]:
        """Detect a MARKED keyless-harness mock LLM via REAL signals (#616).

        Returns a non-empty evidence list IFF a genuine marked-mock indicator is
        found; an empty list means "no mock marker reachable" (caller keeps the
        agent's normal source). Three independent, faithful signals are checked —
        none fabricates a marker that isn't there:

        1. ``agent_output`` carries the in-band ``mock_response_for_dev_only``
           marker (e.g. experiment_designer's validity_audit writes it into the
           structured output dict).
        2. ``agent_instance`` retains a live ``MarkedMockChatLLM`` (or any object
           exposing the marker attribute) — e.g. tool_composer's ``llm_client``
           after ``run()``.
        3. The keyless-mock GATE the agents themselves use is active: the opt-in
           flag ``E2I_ALLOW_MOCK_LLM`` is set AND the configured provider's API
           key is absent — the exact condition under which ``llm_or_marked_mock``
           / agent constructors fall back to the marked mock. This is the only
           signal that reaches the orchestrator, whose graph nodes (and their mock
           LLMs) are constructed transiently per-call and never retained on the
           instance.
        """
        # Signal 1: marker in the output payload.
        if self._marker_in_output(agent_output):
            return [f"{agent_name} output carries the {MOCK_LLM_MARKER} marker (marked-mock LLM)"]

        # Signal 2: live marked-mock LLM reachable on the instance.
        if agent_instance is not None and self._instance_has_marked_mock(agent_instance):
            return [
                f"{agent_name} instance retains a MarkedMockChatLLM "
                f"({MOCK_LLM_MARKER}=True) — keyless-harness mock"
            ]

        # Signal 3: marker text in captured execution logs.
        log_text = "\n".join(execution_logs or [])
        if MOCK_LLM_MARKER in log_text or "MarkedMockChatLLM" in log_text:
            return [f"{agent_name} execution logs reference the marked-mock LLM"]

        # Signal 4: the keyless-mock gate is active (mechanism-faithful — same
        # gate the agent uses). Scoped to LLM-dependent agents so a missing key
        # in an unrelated context can't mislabel a non-LLM agent.
        if agent_name in _LLM_DEPENDENT_AGENTS and self._keyless_mock_gate_active():
            return [
                f"{agent_name} ran under the keyless-mock gate "
                f"({_MOCK_LLM_FLAG} set + provider key absent) — falls back to a marked-mock LLM"
            ]

        return []

    @staticmethod
    def _marker_in_output(agent_output: dict[str, Any]) -> bool:
        """True if the dev-only mock marker appears anywhere in the output dict.

        Bounded recursion over nested dicts/lists; the marker is a simple
        truthy ``mock_response_for_dev_only`` key wherever the agent stamped it.
        """

        def _scan(value: Any, depth: int) -> bool:
            if depth > 4:
                return False
            if isinstance(value, dict):
                if value.get(MOCK_LLM_MARKER):
                    return True
                return any(_scan(v, depth + 1) for v in value.values())
            if isinstance(value, (list, tuple)):
                return any(_scan(v, depth + 1) for v in value)
            return False

        return _scan(agent_output or {}, 0)

    @staticmethod
    def _instance_has_marked_mock(agent_instance: Any, max_depth: int = 5) -> bool:
        """True if a live MarkedMockChatLLM (marker attribute) is reachable.

        Traverses the instance's ``__dict__`` graph (bounded depth, cycle-safe).
        Matches by the class-level ``mock_response_for_dev_only`` attribute rather
        than importing the class, so the validator stays dependency-light. Catches
        agents that retain the mock LLM on the instance (e.g. tool_composer's
        ``llm_client``); agents that build/discard mock LLMs transiently per call
        (e.g. orchestrator) are covered by the gate signal instead.
        """
        seen: set[int] = set()

        def _scan(obj: Any, depth: int) -> bool:
            if depth > max_depth or obj is None:
                return False
            oid = id(obj)
            if oid in seen:
                return False
            seen.add(oid)
            if isinstance(obj, (str, bytes, int, float, bool)):
                return False
            # The marker is a class/instance attribute set True on the mock LLM.
            if getattr(obj, MOCK_LLM_MARKER, False) is True:
                return True
            attrs = getattr(obj, "__dict__", None)
            if isinstance(attrs, dict):
                for v in attrs.values():
                    if _scan(v, depth + 1):
                        return True
            if isinstance(obj, dict):
                for v in obj.values():
                    if _scan(v, depth + 1):
                        return True
            if isinstance(obj, (list, tuple, set)):
                for v in obj:
                    if _scan(v, depth + 1):
                        return True
            return False

        return _scan(agent_instance, 0)

    @staticmethod
    def _keyless_mock_gate_active() -> bool:
        """True if the keyless-mock fallback gate is active in this environment.

        Mirrors ``src/utils/mock_llm.mock_llm_allowed`` + the llm_factory key
        check: the opt-in flag is truthy AND the configured provider's API key is
        absent. Under exactly this gate an LLM-dependent agent constructs a MARKED
        mock instead of raising — so it is a faithful proxy for the mock, not a
        fabricated marker.
        """
        flag = os.environ.get(_MOCK_LLM_FLAG, "").strip().lower()
        if flag not in _MOCK_LLM_TRUTHY:
            return False
        provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
        key_env = _PROVIDER_KEY_ENV.get(provider, "OPENAI_API_KEY")
        return not os.environ.get(key_env)

    def get_requirements(self, agent_name: str) -> dict[str, Any] | None:
        """Get data source requirements for an agent."""
        return self.requirements.get(agent_name)

    def list_agents_with_requirements(self) -> list[str]:
        """List all agents with configured data source requirements."""
        return list(self.requirements.keys())
