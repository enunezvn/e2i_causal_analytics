"""GEPA Feedback Metrics for E2I Agents.

This module provides specialized GEPA metrics for different agent types:
- CausalImpactGEPAMetric: For Tier 2 Hybrid causal analysis agents
- ExperimentDesignerGEPAMetric: For Tier 3 Hybrid experiment design agents
- FeedbackLearnerGEPAMetric: For Tier 5 Deep self-improvement agents
- StandardAgentGEPAMetric: For all Standard agents (SLA + accuracy)

All metrics return ScoreWithFeedback objects for GEPA's reflective evolution.
"""

from src.optimization.gepa.metrics.base import (
    DSPyTrace,
    E2IGEPAMetric,
    ScoreWithFeedback,
)
from src.optimization.gepa.metrics.causal_impact_metric import CausalImpactGEPAMetric
from src.optimization.gepa.metrics.experiment_designer_metric import (
    ExperimentDesignerGEPAMetric,
)
from src.optimization.gepa.metrics.feedback_learner_metric import (
    FeedbackLearnerGEPAMetric,
)
from src.optimization.gepa.metrics.standard_agent_metric import StandardAgentGEPAMetric

# Agent type to metric class mapping
AGENT_METRICS = {
    # Tier 2: Causal Analytics
    "causal_impact": CausalImpactGEPAMetric,
    # Tier 3: Monitoring
    "experiment_designer": ExperimentDesignerGEPAMetric,
    # Tier 5: Self-Improvement
    "feedback_learner": FeedbackLearnerGEPAMetric,
    "explainer": FeedbackLearnerGEPAMetric,  # Uses same deep metric
    # All other agents use StandardAgentGEPAMetric
}


def get_metric_for_agent(agent_name: str) -> E2IGEPAMetric:
    """Get the appropriate GEPA metric instance for an agent.

    Args:
        agent_name: Name of the agent (e.g., 'causal_impact', 'orchestrator')

    Returns:
        An instantiated metric appropriate for this agent type
    """
    metric_class = AGENT_METRICS.get(agent_name, StandardAgentGEPAMetric)
    return metric_class()


__all__ = [
    # Base
    "E2IGEPAMetric",
    "ScoreWithFeedback",
    "DSPyTrace",
    # Specialized Metrics
    "CausalImpactGEPAMetric",
    "ExperimentDesignerGEPAMetric",
    "FeedbackLearnerGEPAMetric",
    "StandardAgentGEPAMetric",
    # Factory
    "get_metric_for_agent",
    "AGENT_METRICS",
]
