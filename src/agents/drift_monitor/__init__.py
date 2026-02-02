"""Drift Monitor Agent.

Tier: 3 (Monitoring)
Agent Type: Standard (Fast Path)
Performance Target: <10s for 50 features

This agent detects drift in:
1. Data distributions (PSI + KS test)
2. Model predictions
3. Concept (feature-target relationships)

Key Features:
- Statistical drift detection (no LLM usage)
- Multi-level severity (none, low, medium, high, critical)
- Alert generation with recommended actions
- Composite drift score calculation

Usage:
    from src.agents.drift_monitor import DriftMonitorAgent, DriftMonitorInput

    agent = DriftMonitorAgent()
    result = agent.run(DriftMonitorInput(
        query="Check for drift in engagement features",
        features_to_monitor=["hcp_engagement_frequency", "trx_total"],
        model_id="model_v1",
        time_window="7d"
    ))

    print(f"Drift Score: {result.overall_drift_score:.3f}")
    print(f"Features with drift: {result.features_with_drift}")
    print(f"Alerts: {len(result.alerts)}")

Documentation:
    - Specialist: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md
    - Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

from src.agents.drift_monitor.agent import (
    DriftMonitorAgent,
    DriftMonitorInput,
    DriftMonitorOutput,
)
from src.agents.drift_monitor.mlflow_tracker import (
    DriftMonitorContext,
    DriftMonitorMetrics,
    DriftMonitorMLflowTracker,
)
from src.agents.drift_monitor.mlflow_tracker import (
    create_tracker as create_mlflow_tracker,
)
from src.agents.drift_monitor.state import (
    DriftAlert,
    DriftMonitorState,
    DriftResult,
)

__all__ = [
    "DriftMonitorAgent",
    "DriftMonitorInput",
    "DriftMonitorOutput",
    "DriftMonitorState",
    "DriftResult",
    "DriftAlert",
    # MLflow tracking
    "DriftMonitorMLflowTracker",
    "DriftMonitorMetrics",
    "DriftMonitorContext",
    "create_mlflow_tracker",
]
