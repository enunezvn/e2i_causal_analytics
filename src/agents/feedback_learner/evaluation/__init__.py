"""
Rubric evaluation module for the Feedback Learner agent.

This module provides AI-as-judge evaluation of chatbot responses against
the E2I Causal Analytics rubric criteria.

Exports:
    - RubricEvaluator: Main evaluator class
    - Models: EvaluationContext, RubricEvaluation, CriterionScore, etc.
    - Criteria: DEFAULT_CRITERIA, get_default_criteria, etc.
"""

from src.agents.feedback_learner.evaluation.criteria import (
    DEFAULT_CRITERIA,
    DEFAULT_OVERRIDE_CONDITIONS,
    DEFAULT_THRESHOLDS,
    RubricCriterion,
    get_criterion_by_name,
    get_default_criteria,
    get_total_weight,
    validate_weights,
)
from src.agents.feedback_learner.evaluation.models import (
    CriterionScore,
    DecisionThresholds,
    EvaluationContext,
    ImprovementDecision,
    ImprovementSource,
    OverrideCondition,
    PatternFlag,
    RubricConfig,
    RubricCriterionConfig,
    RubricEvaluation,
)
from src.agents.feedback_learner.evaluation.rubric_evaluator import (
    RubricEvaluator,
)

__all__ = [
    # Main evaluator
    "RubricEvaluator",
    # Models
    "ImprovementDecision",
    "ImprovementSource",
    "CriterionScore",
    "EvaluationContext",
    "PatternFlag",
    "RubricEvaluation",
    "RubricConfig",
    "RubricCriterionConfig",
    "DecisionThresholds",
    "OverrideCondition",
    # Criteria
    "RubricCriterion",
    "DEFAULT_CRITERIA",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_OVERRIDE_CONDITIONS",
    "get_default_criteria",
    "get_criterion_by_name",
    "get_total_weight",
    "validate_weights",
]
