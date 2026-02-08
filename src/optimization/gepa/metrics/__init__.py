"""GEPA Feedback Metrics for E2I Agents.

This module provides specialized GEPA metrics for different agent types:
- EvidenceSynthesisGEPAMetric: For DSPy module optimization (interpretation quality)
- CausalImpactGEPAMetric: For Tier 2 full pipeline evaluation (with DoWhy tools)
- ExperimentDesignerGEPAMetric: For Tier 3 Hybrid experiment design agents
- FeedbackLearnerGEPAMetric: For Tier 5 Deep self-improvement agents
- ToolComposerGEPAMetric: For Tier 1 Tool Composer 4-phase pipeline
- StandardAgentGEPAMetric: For all Standard agents (SLA + accuracy)

Metrics return float scores for GEPA's aggregation and reflective evolution.
"""

from typing import cast

from src.optimization.gepa.metrics.base import (
    DSPyTrace,
    E2IGEPAMetric,
    ScoreWithFeedback,
)
from src.optimization.gepa.metrics.causal_impact_metric import CausalImpactGEPAMetric
from src.optimization.gepa.metrics.evidence_synthesis_metric import (
    EvidenceSynthesisGEPAMetric,
)
from src.optimization.gepa.metrics.experiment_designer_metric import (
    ExperimentDesignerGEPAMetric,
)
from src.optimization.gepa.metrics.feedback_learner_metric import (
    FeedbackLearnerGEPAMetric,
)
from src.optimization.gepa.metrics.standard_agent_metric import StandardAgentGEPAMetric
from src.optimization.gepa.metrics.tool_composer_metric import ToolComposerGEPAMetric

# Agent type to metric class mapping
# Note: For DSPy module optimization, use EvidenceSynthesisGEPAMetric
# CausalImpactGEPAMetric is for full pipeline evaluation with DoWhy tools
AGENT_METRICS = {
    # Tier 1: Orchestration
    "tool_composer": ToolComposerGEPAMetric,
    # Tier 2: Causal Analytics - DSPy module optimization
    "causal_impact": EvidenceSynthesisGEPAMetric,
    "causal_impact_pipeline": CausalImpactGEPAMetric,  # Full pipeline eval
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
    return cast(E2IGEPAMetric, metric_class())


__all__ = [
    # Base
    "E2IGEPAMetric",
    "ScoreWithFeedback",
    "DSPyTrace",
    # Specialized Metrics
    "EvidenceSynthesisGEPAMetric",
    "CausalImpactGEPAMetric",
    "ExperimentDesignerGEPAMetric",
    "FeedbackLearnerGEPAMetric",
    "ToolComposerGEPAMetric",
    "StandardAgentGEPAMetric",
    # Factory
    "get_metric_for_agent",
    "AGENT_METRICS",
]
