"""
E2I Feedback Learner Agent - Tier 5 Self-Improvement
Version: 4.2
Purpose: Learn from user feedback to improve system performance

DSPy Integration Support:
- CognitiveContext from CognitiveRAG 4-phase cycle
- Training signals for MIPROv2 optimization
- Memory contribution helpers

Self-Improvement Support:
- RubricEvaluator for AI-as-judge response evaluation
- Configuration loading from self_improvement.yaml
- Pattern detection and improvement suggestions
"""

from .agent import (
    FeedbackLearnerAgent,
    FeedbackLearnerInput,
    FeedbackLearnerOutput,
    process_feedback_batch,
)

# Config exports
from .config import (
    SelfImprovementConfig,
    load_self_improvement_config,
)
from .dspy_integration import (
    DSPY_AVAILABLE,
    AgentTrainingSignal,
    FeedbackLearnerCognitiveContext,
    FeedbackLearnerOptimizer,
    FeedbackLearnerTrainingSignal,
    create_memory_contribution,
)

# Evaluation exports
from .evaluation import (
    CriterionScore,
    EvaluationContext,
    ImprovementDecision,
    PatternFlag,
    RubricEvaluation,
    RubricEvaluator,
)
from .graph import build_feedback_learner_graph, build_simple_feedback_learner_graph

# Node exports
from .nodes import RubricNode
from .state import (
    DetectedPattern,
    FeedbackItem,
    FeedbackLearnerState,
    FeedbackSummary,
    KnowledgeUpdate,
    LearningRecommendation,
)

__all__ = [
    # Agent
    "FeedbackLearnerAgent",
    "FeedbackLearnerInput",
    "FeedbackLearnerOutput",
    # Graph builders
    "build_feedback_learner_graph",
    "build_simple_feedback_learner_graph",
    # State types
    "FeedbackLearnerState",
    "FeedbackItem",
    "DetectedPattern",
    "LearningRecommendation",
    "KnowledgeUpdate",
    "FeedbackSummary",
    # DSPy Integration
    "FeedbackLearnerCognitiveContext",
    "FeedbackLearnerTrainingSignal",
    "AgentTrainingSignal",
    "FeedbackLearnerOptimizer",
    "create_memory_contribution",
    "DSPY_AVAILABLE",
    # Rubric Evaluation
    "RubricEvaluator",
    "RubricEvaluation",
    "EvaluationContext",
    "CriterionScore",
    "ImprovementDecision",
    "PatternFlag",
    "RubricNode",
    # Configuration
    "SelfImprovementConfig",
    "load_self_improvement_config",
    # Convenience functions
    "process_feedback_batch",
]
