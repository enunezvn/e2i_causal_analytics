"""
E2I Feedback Learner Agent - Tier 5 Self-Improvement
Version: 4.2
Purpose: Learn from user feedback to improve system performance

DSPy Integration Support:
- CognitiveContext from CognitiveRAG 4-phase cycle
- Training signals for MIPROv2 optimization
- Memory contribution helpers
"""

from .agent import (
    FeedbackLearnerAgent,
    FeedbackLearnerInput,
    FeedbackLearnerOutput,
    process_feedback_batch,
)
from .dspy_integration import (
    DSPY_AVAILABLE,
    AgentTrainingSignal,
    FeedbackLearnerCognitiveContext,
    FeedbackLearnerOptimizer,
    FeedbackLearnerTrainingSignal,
    create_memory_contribution,
)
from .graph import build_feedback_learner_graph, build_simple_feedback_learner_graph
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
    # Convenience functions
    "process_feedback_batch",
]
