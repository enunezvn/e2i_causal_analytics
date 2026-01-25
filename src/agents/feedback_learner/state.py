"""
E2I Feedback Learner Agent - State Definitions
Version: 4.3
Purpose: TypedDict state for feedback learning workflow

Includes DSPy integration support:
- CognitiveContext from CognitiveRAG 4-phase cycle
- Training signal fields for MIPROv2 optimization
- ValidationOutcome from Causal Validation Protocol
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict
from uuid import UUID

# Import cognitive context from DSPy integration module
from .dspy_integration import (
    FeedbackLearnerCognitiveContext,
    FeedbackLearnerTrainingSignal,
)

# Import validation outcome for causal validation learning
# NOTE: Must be unconditional (not TYPE_CHECKING) because TypedDict evaluates
# forward references at runtime when used with get_type_hints() or similar
from src.causal_engine.validation_outcome import ValidationOutcome


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
    knowledge_type: Literal["experiment", "baseline", "agent_config", "prompt", "threshold"]
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


# =============================================================================
# DISCOVERY FEEDBACK TYPES (V4.4)
# =============================================================================


class DiscoveryFeedbackItem(TypedDict):
    """Feedback specific to causal discovery results."""

    feedback_id: str
    timestamp: str
    feedback_type: Literal[
        "user_correction",  # User corrects discovered edges
        "expert_review",  # Domain expert validates/rejects
        "outcome_validation",  # Observed outcomes vs predicted
        "gate_override",  # User overrides gate decision
    ]
    discovery_run_id: str
    algorithm_used: str
    # The DAG that was evaluated
    dag_adjacency: Optional[Dict[str, List[str]]]
    dag_nodes: Optional[List[str]]
    # Gate decision that was made
    original_gate_decision: Literal["accept", "review", "reject", "augment"]
    # Feedback content
    user_decision: Optional[Literal["accept", "reject", "modify"]]
    edge_corrections: Optional[List[Dict[str, Any]]]  # Added/removed edges
    comments: Optional[str]
    # Validation metrics
    accuracy_score: Optional[float]  # 0-1 if measured
    metadata: Dict[str, Any]


class DiscoveryAccuracyTracking(TypedDict):
    """Tracking accuracy by discovery algorithm."""

    algorithm: str
    total_runs: int
    accepted_runs: int
    rejected_runs: int
    modified_runs: int
    average_accuracy: float
    edge_precision: float  # Correct edges / predicted edges
    edge_recall: float  # Correct edges / true edges


class DiscoveryParameterRecommendation(TypedDict):
    """Recommended parameter changes for discovery algorithms."""

    recommendation_id: str
    algorithm: str
    parameter_name: str
    current_value: Any
    recommended_value: Any
    justification: str
    expected_accuracy_improvement: float
    confidence: float


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

    # === DISCOVERY FEEDBACK (V4.4) ===
    discovery_feedback_items: Optional[List[DiscoveryFeedbackItem]]
    discovery_accuracy_tracking: Optional[Dict[str, DiscoveryAccuracyTracking]]
    discovery_parameter_recommendations: Optional[List[DiscoveryParameterRecommendation]]
    has_discovery_feedback: bool

    # === CAUSAL VALIDATION OUTCOMES (From Causal Impact Agent) ===
    validation_outcomes: Optional[List[ValidationOutcome]]

    # === PATTERN ANALYSIS ===
    detected_patterns: Optional[List[DetectedPattern]]
    pattern_clusters: Optional[Dict[str, List[str]]]

    # === LEARNING OUTPUTS ===
    learning_recommendations: Optional[List[LearningRecommendation]]
    priority_improvements: Optional[List[str]]

    # === KNOWLEDGE UPDATES ===
    proposed_updates: Optional[List[KnowledgeUpdate]]
    applied_updates: Optional[List[str]]

    # === RUBRIC EVALUATION ===
    rubric_evaluation_context: Optional[Dict[str, Any]]
    rubric_evaluation: Optional[Dict[str, Any]]
    rubric_weighted_score: Optional[float]
    rubric_decision: Optional[str]
    rubric_pattern_flags: Optional[List[Dict[str, Any]]]
    rubric_improvement_suggestion: Optional[str]
    rubric_latency_ms: Optional[int]
    rubric_error: Optional[str]

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

    # === AUDIT CHAIN ===
    audit_workflow_id: Optional[UUID]
