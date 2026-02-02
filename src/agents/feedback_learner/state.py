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

from typing import Any, Dict, List, Literal, NotRequired, Optional, TypedDict
from uuid import UUID

# Import validation outcome for causal validation learning
# NOTE: Must be unconditional (not TYPE_CHECKING) because TypedDict evaluates
# forward references at runtime when used with get_type_hints() or similar
from src.causal_engine.validation_outcome import ValidationOutcome

# Import cognitive context from DSPy integration module
from .dspy_integration import (
    FeedbackLearnerCognitiveContext,
    FeedbackLearnerTrainingSignal,
)


class FeedbackItem(TypedDict):
    """Individual feedback item."""

    feedback_id: str
    timestamp: str
    feedback_type: Literal["rating", "correction", "outcome", "explicit", "implicit"]
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

    # === INPUT (NotRequired - provided by caller) ===
    batch_id: NotRequired[str]
    time_range_start: NotRequired[str]
    time_range_end: NotRequired[str]
    focus_agents: NotRequired[List[str]]

    # === COGNITIVE CONTEXT (From CognitiveRAG) ===
    cognitive_context: NotRequired[FeedbackLearnerCognitiveContext]

    # === DSPY TRAINING SIGNALS (For MIPROv2 Optimization) ===
    training_signal: NotRequired[FeedbackLearnerTrainingSignal]

    # === FEEDBACK DATA ===
    feedback_items: NotRequired[List[FeedbackItem]]
    feedback_summary: NotRequired[FeedbackSummary]

    # === DISCOVERY FEEDBACK (V4.4) ===
    discovery_feedback_items: NotRequired[List[DiscoveryFeedbackItem]]
    discovery_accuracy_tracking: NotRequired[Dict[str, DiscoveryAccuracyTracking]]
    discovery_parameter_recommendations: NotRequired[List[DiscoveryParameterRecommendation]]
    has_discovery_feedback: NotRequired[bool]

    # === CAUSAL VALIDATION OUTCOMES (From Causal Impact Agent) ===
    validation_outcomes: NotRequired[List[ValidationOutcome]]

    # === PATTERN ANALYSIS ===
    detected_patterns: NotRequired[List[DetectedPattern]]
    pattern_clusters: NotRequired[Dict[str, List[str]]]

    # === LEARNING OUTPUTS ===
    learning_recommendations: NotRequired[List[LearningRecommendation]]
    priority_improvements: NotRequired[List[str]]

    # === KNOWLEDGE UPDATES ===
    proposed_updates: NotRequired[List[KnowledgeUpdate]]
    applied_updates: NotRequired[List[str]]

    # === RUBRIC EVALUATION ===
    rubric_evaluation_context: NotRequired[Dict[str, Any]]
    rubric_evaluation: NotRequired[Dict[str, Any]]
    rubric_weighted_score: NotRequired[float]
    rubric_decision: NotRequired[str]
    rubric_pattern_flags: NotRequired[List[Dict[str, Any]]]
    rubric_improvement_suggestion: NotRequired[str]
    rubric_latency_ms: NotRequired[int]
    rubric_error: NotRequired[str]

    # === SUMMARY (Required output) ===
    learning_summary: str

    # === METRICS (NotRequired - may not always be computed) ===
    metrics_before: NotRequired[Dict[str, float]]
    metrics_after: NotRequired[Dict[str, float]]

    # === EXECUTION METADATA (NotRequired - populated during execution) ===
    collection_latency_ms: NotRequired[int]
    analysis_latency_ms: NotRequired[int]
    extraction_latency_ms: NotRequired[int]
    update_latency_ms: NotRequired[int]
    total_latency_ms: NotRequired[int]
    model_used: NotRequired[str]

    # === ERROR HANDLING (Required outputs) ===
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
    audit_workflow_id: NotRequired[UUID]
