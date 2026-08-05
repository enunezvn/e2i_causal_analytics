"""GEPA Feedback Metrics for E2I Agents.

This module provides specialized GEPA metrics for different agent types:
- EvidenceSynthesisGEPAMetric: For DSPy module optimization (interpretation quality)
- CausalImpactGEPAMetric: For Tier 2 full pipeline evaluation (with DoWhy tools)
- ExperimentDesignerGEPAMetric: For Tier 3 Hybrid experiment design agents
- FeedbackLearnerGEPAMetric: For Tier 5 Deep self-improvement agents
- ToolComposerGEPAMetric: For Tier 1 Tool Composer 4-phase pipeline
- StandardAgentGEPAMetric: For all Standard agents (SLA + accuracy)
- RagasGEPAMetric: For RAG-shaped agents (RAGAS retrieval + grounding quality)

Metrics return float scores for GEPA's aggregation and reflective evolution.

Return-shape warning (dspy 3.1.0): GEPA coerces a metric return with
``s["score"] if hasattr(s, "score") else s``, so a plain ``{"score", "feedback"}``
dict reaches ``dspy.Evaluate``'s summation intact and raises ``TypeError:
unsupported operand type(s) for +: 'int' and 'dict'``. The older metrics in this
package still return dicts and are relied on being wrapped at the call site
(see ``recipient_optimizer._wrap_metric``); RagasGEPAMetric returns a
``dspy.Prediction`` directly.
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
from src.optimization.gepa.metrics.ragas_metric import (
    RagasGEPAMetric,
    RagasMetricUnavailableError,
    RagasUnjudgeableExampleError,
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
    # RAG: RAGAS retrieval + grounding quality (#1486).
    #
    # This entry optimizes PROMPTS. The nightly GEPA cycle evolves DSPy
    # signature instructions; `retrieval_configurations` in
    # database/ml/022_self_improvement_tables.sql is a separate search space
    # with a separate optimizer and is deliberately NOT fed from here.
    #
    # `explainer` is deliberately NOT remapped to this metric even though
    # #1486 proposed it: ExplanationSynthesisSignature takes analysis_results /
    # user_expertise / focus_areas / output_format and carries no retrieved
    # contexts, so RAGAS would be grading retrieval that never happened.
    #
    # Constructing this metric verifies the RAGAS judge can run and RAISES if it
    # cannot — deliberately, and only for callers that ask for it by name. A
    # per-example refusal is swallowed by dspy into failure_score 0.0 (measured
    # on dspy 3.1.0), so a keyless environment would otherwise score every
    # candidate 0.0 and GEPA would optimize against fabricated signal.
    "cognitive_rag": RagasGEPAMetric,
    # All other agents use StandardAgentGEPAMetric
}


def get_metric_for_agent(agent_name: str) -> E2IGEPAMetric:
    """Get the appropriate GEPA metric instance for an agent.

    Args:
        agent_name: Name of the agent (e.g., 'causal_impact', 'orchestrator')

    Returns:
        An instantiated metric appropriate for this agent type

    Raises:
        RagasMetricUnavailableError: only for agents mapped to RagasGEPAMetric,
            when the RAGAS judge cannot run. Failing here is the point — see the
            note on the "cognitive_rag" entry in AGENT_METRICS.
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
    "RagasGEPAMetric",
    "RagasMetricUnavailableError",
    "RagasUnjudgeableExampleError",
    # Factory
    "get_metric_for_agent",
    "AGENT_METRICS",
]
