"""
Pydantic models for rubric evaluation.

Models for evaluation context, criterion scores, and evaluation results.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ImprovementDecision(str, Enum):
    """Decision types based on rubric scores."""

    ACCEPTABLE = "acceptable"  # Score >= 4.0
    SUGGESTION = "suggestion"  # Score 3.0-3.9
    AUTO_UPDATE = "auto_update"  # Score 2.0-2.9
    ESCALATE = "escalate"  # Score < 2.0


class ImprovementSource(str, Enum):
    """Source of improvement for audit trail."""

    AUTO = "auto"
    SUGGESTION_APPROVED = "suggestion_approved"
    MANUAL = "manual"
    ROLLBACK = "rollback"


class CriterionScore(BaseModel):
    """Score for a single criterion."""

    criterion: str
    score: float = Field(ge=1.0, le=5.0)
    reasoning: str
    evidence: Optional[str] = None


class EvaluationContext(BaseModel):
    """Context for evaluation."""

    user_query: str
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    final_response: str
    session_id: Optional[str] = None
    agent_names: List[str] = Field(default_factory=list)
    messages_evaluated: int = 1
    retrieved_contexts: List[str] = Field(default_factory=list)


class PatternFlag(BaseModel):
    """Pattern flag for recurring weakness detection."""

    pattern_type: str
    score: float
    reasoning: str
    criterion: str


class RubricEvaluation(BaseModel):
    """Complete evaluation result."""

    weighted_score: float = Field(ge=1.0, le=5.0)
    criterion_scores: List[CriterionScore]
    decision: ImprovementDecision
    overall_analysis: str
    pattern_flags: List[PatternFlag] = Field(default_factory=list)
    improvement_suggestion: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_acceptable(self) -> bool:
        """Check if evaluation passed."""
        return self.decision == ImprovementDecision.ACCEPTABLE

    @property
    def needs_action(self) -> bool:
        """Check if evaluation requires improvement action."""
        return self.decision in [
            ImprovementDecision.SUGGESTION,
            ImprovementDecision.AUTO_UPDATE,
            ImprovementDecision.ESCALATE,
        ]

    def to_learning_signal_format(self) -> Dict[str, Any]:
        """Convert to format for learning_signals table."""
        return {
            "rubric_scores": {
                s.criterion: {"score": s.score, "reasoning": s.reasoning}
                for s in self.criterion_scores
            },
            "rubric_total": self.weighted_score,
            "improvement_details": {
                "decision": self.decision.value,
                "pattern_flags": [p.model_dump() for p in self.pattern_flags],
                "suggestion": self.improvement_suggestion,
            },
        }


class RubricConfig(BaseModel):
    """Configuration for rubric evaluation."""

    criteria: Dict[str, "RubricCriterionConfig"]
    decision_thresholds: "DecisionThresholds"
    override_conditions: List["OverrideCondition"] = Field(default_factory=list)


class RubricCriterionConfig(BaseModel):
    """Configuration for a single rubric criterion."""

    name: str
    weight: float = Field(ge=0.0, le=1.0)
    description: str
    scoring_guide: Dict[int, str]


class DecisionThresholds(BaseModel):
    """Thresholds for improvement decisions."""

    acceptable: float = 4.0
    suggestion: float = 3.0
    auto_update: float = 2.0


class OverrideCondition(BaseModel):
    """Override condition for decision logic."""

    condition: str
    threshold: float
    action: str


# Forward reference resolution
RubricConfig.model_rebuild()
