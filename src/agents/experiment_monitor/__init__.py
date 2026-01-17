"""Experiment Monitor Agent.

This module provides the ExperimentMonitorAgent for monitoring active A/B experiments.

Tier: 3 (Monitoring)
Agent Type: Standard (Fast Path)
Performance Target: <5s per experiment check

Features:
    - Health monitoring (enrollment rates, data quality)
    - Sample Ratio Mismatch (SRM) detection
    - Interim analysis triggers
    - Digital Twin fidelity tracking
    - Alert generation with severity levels

Usage:
    from src.agents.experiment_monitor import (
        ExperimentMonitorAgent,
        ExperimentMonitorInput,
        ExperimentMonitorOutput,
    )

    agent = ExperimentMonitorAgent()
    result = await agent.run_async(ExperimentMonitorInput(
        query="Check all active experiments for issues",
        check_all_active=True
    ))

    print(f"Experiments checked: {result.experiments_checked}")
    print(f"Alerts: {len(result.alerts)}")
"""

from src.agents.experiment_monitor.agent import (
    ExperimentMonitorAgent,
    ExperimentMonitorInput,
    ExperimentMonitorOutput,
)
from src.agents.experiment_monitor.dspy_integration import (
    DSPY_AVAILABLE,
    ExperimentMonitorDSPyIntegration,
    ExperimentMonitorPrompts,
    get_experiment_monitor_dspy_integration,
    reset_dspy_integration,
)
from src.agents.experiment_monitor.graph import experiment_monitor_graph
from src.agents.experiment_monitor.memory_hooks import (
    ExperimentMonitorMemoryHooks,
    MonitoringContext,
    MonitoringRecord,
    contribute_to_memory,
    get_experiment_monitor_memory_hooks,
    reset_memory_hooks,
)
from src.agents.experiment_monitor.state import (
    EnrollmentIssue,
    ErrorDetails,
    ExperimentMonitorState,
    ExperimentSummary,
    FidelityIssue,
    InterimTrigger,
    MonitorAlert,
    SRMIssue,
)

__all__ = [
    # Main agent
    "ExperimentMonitorAgent",
    "ExperimentMonitorInput",
    "ExperimentMonitorOutput",
    # Graph
    "experiment_monitor_graph",
    # State types
    "ExperimentMonitorState",
    "ExperimentSummary",
    "SRMIssue",
    "EnrollmentIssue",
    "FidelityIssue",
    "InterimTrigger",
    "MonitorAlert",
    "ErrorDetails",
    # DSPy integration
    "ExperimentMonitorDSPyIntegration",
    "ExperimentMonitorPrompts",
    "get_experiment_monitor_dspy_integration",
    "reset_dspy_integration",
    "DSPY_AVAILABLE",
    # Memory hooks
    "ExperimentMonitorMemoryHooks",
    "MonitoringContext",
    "MonitoringRecord",
    "contribute_to_memory",
    "get_experiment_monitor_memory_hooks",
    "reset_memory_hooks",
]
