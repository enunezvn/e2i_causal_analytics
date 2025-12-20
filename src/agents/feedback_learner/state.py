"""
E2I Feedback Learner Agent - State Definitions
Version: 4.2
Purpose: TypedDict state for feedback learning workflow

Includes DSPy integration support:
- CognitiveContext from CognitiveRAG 4-phase cycle
- Training signal fields for MIPROv2 optimization
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

# Import cognitive context from DSPy integration module
from .dspy_integration import (
    FeedbackLearnerCognitiveContext,
    AgentTrainingSignal,
    FeedbackLearnerTrainingSignal,
)


class FeedbackItem(TypedDict):
    """Individual feedback item."""

    feedback_id: str
    timestamp: str
    feedback_type: Literal["rating", "correction", "outcome", "explicit"]
    source_agent: str
    query: str
    agent_response: str
    user_feedback: Any  # Rating, correction text, or outcome data
    metadata: Dict[str, Any]


class DetectedPattern(TypedDict):
    """Detected pattern in feedback."""

    pattern_id: str
    pattern_type: Literal[
        "accuracy_issue",
        "latency_issue",
        "relevance_issue",
        "format_issue",
        "coverage_gap",
    ]
    description: str
    frequency: int
    severity: Literal["low", "medium", "high", "critical"]
    affected_agents: List[str]
    example_feedback_ids: List[str]
    root_cause_hypothesis: str


class LearningRecommendation(TypedDict):
    """Recommendation for improvement."""

    recommendation_id: str
    category: Literal[
        "prompt_update",
        "model_retrain",
        "data_update",
        "config_change",
        "new_capability",
    ]
    description: str
    affected_agents: List[str]
    expected_impact: str
    implementation_effort: Literal["low", "medium", "high"]
    priority: int
    proposed_change: Optional[str]


class KnowledgeUpdate(TypedDict):
    """Update to knowledge base."""

    update_id: str
    knowledge_type: Literal[
        "experiment", "baseline", "agent_config", "prompt", "threshold"
    ]
    key: str
    old_value: Any
    new_value: Any
    justification: str
    effective_date: str


class FeedbackSummary(TypedDict):
    """Summary statistics for collected feedback."""

    total_count: int
    by_type: Dict[str, int]
    by_agent: Dict[str, int]
    average_rating: Optional[float]
    rating_count: int


class FeedbackLearnerState(TypedDict):
    """Complete state for Feedback Learner agent."""

    # === INPUT ===
    batch_id: str
    time_range_start: str
    time_range_end: str
    focus_agents: Optional[List[str]]

    # === COGNITIVE CONTEXT (From CognitiveRAG) ===
    cognitive_context: Optional[FeedbackLearnerCognitiveContext]

    # === DSPY TRAINING SIGNALS (For MIPROv2 Optimization) ===
    training_signal: Optional[FeedbackLearnerTrainingSignal]

    # === FEEDBACK DATA ===
    feedback_items: Optional[List[FeedbackItem]]
    feedback_summary: Optional[FeedbackSummary]

    # === PATTERN ANALYSIS ===
    detected_patterns: Optional[List[DetectedPattern]]
    pattern_clusters: Optional[Dict[str, List[str]]]

    # === LEARNING OUTPUTS ===
    learning_recommendations: Optional[List[LearningRecommendation]]
    priority_improvements: Optional[List[str]]

    # === KNOWLEDGE UPDATES ===
    proposed_updates: Optional[List[KnowledgeUpdate]]
    applied_updates: Optional[List[str]]

    # === SUMMARY ===
    learning_summary: Optional[str]
    metrics_before: Optional[Dict[str, float]]
    metrics_after: Optional[Dict[str, float]]

    # === EXECUTION METADATA ===
    collection_latency_ms: int
    analysis_latency_ms: int
    extraction_latency_ms: int
    update_latency_ms: int
    total_latency_ms: int
    model_used: Optional[str]

    # === ERROR HANDLING ===
    errors: List[Dict[str, Any]]
    warnings: List[str]
    status: Literal[
        "pending",
        "collecting",
        "analyzing",
        "extracting",
        "updating",
        "completed",
        "failed",
    ]
