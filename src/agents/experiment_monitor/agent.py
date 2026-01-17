"""Experiment Monitor Agent.

This module provides the main agent class for experiment monitoring.

Tier: 3 (Monitoring)
Agent Type: Standard (Fast Path)
Performance Target: <5s per experiment check
"""

import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.experiment_monitor.graph import experiment_monitor_graph
from src.agents.experiment_monitor.memory_hooks import (
    contribute_to_memory,
    get_experiment_monitor_memory_hooks,
)
from src.agents.experiment_monitor.state import (
    ExperimentMonitorState,
    ExperimentSummary,
    MonitorAlert,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMonitorInput:
    """Input for experiment monitor agent.

    Attributes:
        query: Natural language query describing monitoring request
        experiment_ids: Optional list of specific experiment IDs to check
        check_all_active: If True, check all active experiments
        srm_threshold: P-value threshold for SRM detection (default: 0.001)
        enrollment_threshold: Minimum daily enrollment rate (default: 5.0)
        fidelity_threshold: Maximum acceptable prediction error (default: 0.2)
        stale_data_threshold_hours: Hours after which data is stale (default: 24.0)
        check_interim: Whether to check for interim analysis triggers
    """

    query: str = ""
    experiment_ids: Optional[List[str]] = None
    check_all_active: bool = True
    srm_threshold: float = 0.001
    enrollment_threshold: float = 5.0
    fidelity_threshold: float = 0.2
    stale_data_threshold_hours: float = 24.0
    check_interim: bool = True


@dataclass
class ExperimentMonitorOutput:
    """Output from experiment monitor agent.

    Attributes:
        experiments: List of experiment summaries with health status
        alerts: List of generated alerts
        experiments_checked: Number of experiments checked
        healthy_count: Number of healthy experiments
        warning_count: Number of experiments with warnings
        critical_count: Number of experiments with critical issues
        monitor_summary: Human-readable summary
        recommended_actions: List of recommended actions
        check_latency_ms: Total execution time in milliseconds
        errors: List of any errors encountered
    """

    experiments: List[ExperimentSummary] = field(default_factory=list)
    alerts: List[MonitorAlert] = field(default_factory=list)
    experiments_checked: int = 0
    healthy_count: int = 0
    warning_count: int = 0
    critical_count: int = 0
    monitor_summary: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    check_latency_ms: int = 0
    errors: List[str] = field(default_factory=list)


class ExperimentMonitorAgent:
    """Experiment Monitor Agent.

    This agent monitors active A/B experiments for:
    1. Health issues (enrollment rates, data quality)
    2. Sample Ratio Mismatch (SRM)
    3. Interim analysis triggers
    4. Digital Twin fidelity (optional)

    Usage:
        from src.agents.experiment_monitor import ExperimentMonitorAgent, ExperimentMonitorInput

        agent = ExperimentMonitorAgent()
        result = await agent.run_async(ExperimentMonitorInput(
            query="Check all active experiments for issues",
            check_all_active=True
        ))

        print(f"Experiments checked: {result.experiments_checked}")
        print(f"Alerts: {len(result.alerts)}")
    """

    def __init__(self, enable_memory: bool = True):
        """Initialize experiment monitor agent.

        Args:
            enable_memory: Whether to enable memory integration (default: True)
        """
        self.graph = experiment_monitor_graph
        self.enable_memory = enable_memory
        self._memory_hooks = None

    @property
    def memory_hooks(self):
        """Lazy-load memory hooks."""
        if self._memory_hooks is None and self.enable_memory:
            try:
                self._memory_hooks = get_experiment_monitor_memory_hooks()
            except Exception as e:
                logger.warning(f"Failed to initialize memory hooks: {e}")
        return self._memory_hooks

    async def run_async(
        self,
        input_data: ExperimentMonitorInput,
        session_id: Optional[str] = None,
    ) -> ExperimentMonitorOutput:
        """Run the experiment monitor agent asynchronously.

        Args:
            input_data: ExperimentMonitorInput with monitoring parameters
            session_id: Optional session ID for memory tracking (generates UUID if not provided)

        Returns:
            ExperimentMonitorOutput with monitoring results
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Build initial state
        initial_state: ExperimentMonitorState = {
            "query": input_data.query,
            "experiment_ids": input_data.experiment_ids or [],
            "check_all_active": input_data.check_all_active,
            "srm_threshold": input_data.srm_threshold,
            "enrollment_threshold": input_data.enrollment_threshold,
            "fidelity_threshold": input_data.fidelity_threshold,
            "stale_data_threshold_hours": input_data.stale_data_threshold_hours,
            "check_interim": input_data.check_interim,
            "experiments": [],
            "srm_issues": [],
            "enrollment_issues": [],
            "stale_data_issues": [],
            "fidelity_issues": [],
            "interim_triggers": [],
            "alerts": [],
            "monitor_summary": "",
            "recommended_actions": [],
            "check_latency_ms": 0,
            "experiments_checked": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        # Execute graph
        final_state = await self.graph.ainvoke(initial_state)

        # Build output
        experiments = final_state.get("experiments", [])
        output = ExperimentMonitorOutput(
            experiments=experiments,
            alerts=final_state.get("alerts", []),
            experiments_checked=final_state.get("experiments_checked", 0),
            healthy_count=sum(1 for e in experiments if e.get("health_status") == "healthy"),
            warning_count=sum(1 for e in experiments if e.get("health_status") == "warning"),
            critical_count=sum(1 for e in experiments if e.get("health_status") == "critical"),
            monitor_summary=final_state.get("monitor_summary", ""),
            recommended_actions=final_state.get("recommended_actions", []),
            check_latency_ms=final_state.get("check_latency_ms", 0),
            errors=[e["error"] for e in final_state.get("errors", [])],
        )

        # Contribute to memory if enabled
        if self.enable_memory:
            try:
                result_dict = self._output_to_dict(output)
                memory_stats = await contribute_to_memory(
                    result=result_dict,
                    state=final_state,
                    memory_hooks=self.memory_hooks,
                    session_id=session_id,
                )
                logger.debug(
                    f"Memory contribution complete: {memory_stats.get('alerts_stored', 0)} alerts, "
                    f"{memory_stats.get('check_stored', 0)} checks stored"
                )
            except Exception as e:
                logger.warning(f"Memory contribution failed (non-blocking): {e}")

        return output

    def _output_to_dict(self, output: ExperimentMonitorOutput) -> Dict[str, Any]:
        """Convert output dataclass to dictionary for memory storage.

        Args:
            output: ExperimentMonitorOutput instance

        Returns:
            Dictionary representation
        """
        return {
            "experiments": output.experiments,
            "alerts": [
                a if isinstance(a, dict) else asdict(a) if hasattr(a, "__dataclass_fields__") else a
                for a in output.alerts
            ],
            "experiments_checked": output.experiments_checked,
            "healthy_count": output.healthy_count,
            "warning_count": output.warning_count,
            "critical_count": output.critical_count,
            "monitor_summary": output.monitor_summary,
            "recommended_actions": output.recommended_actions,
            "check_latency_ms": output.check_latency_ms,
            "errors": output.errors,
        }

    def run(
        self,
        input_data: ExperimentMonitorInput,
        session_id: Optional[str] = None,
    ) -> ExperimentMonitorOutput:
        """Run the experiment monitor agent synchronously.

        Args:
            input_data: ExperimentMonitorInput with monitoring parameters
            session_id: Optional session ID for memory tracking

        Returns:
            ExperimentMonitorOutput with monitoring results
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(self.run_async(input_data, session_id))
            return loop.run_until_complete(self.run_async(input_data, session_id))
        except RuntimeError:
            return asyncio.run(self.run_async(input_data, session_id))
