"""
E2I Feedback Learner Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures, training signals, and MIPROv2 optimization for feedback learning

This module implements the DSPy integration patterns for the Feedback Learner agent,
enabling continuous self-improvement through:
1. Training signal collection from all agents
2. MIPROv2 prompt optimization
3. Cognitive context enrichment from CognitiveRAG
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

logger = logging.getLogger(__name__)


# =============================================================================
# 1. COGNITIVE CONTEXT (From CognitiveRAG)
# =============================================================================


class FeedbackLearnerCognitiveContext(TypedDict):
    """
    Cognitive context enriched by CognitiveRAG for feedback analysis.

    This context is injected at the start of each learning cycle from the
    4-phase cognitive workflow's Summarizer and Investigator phases.
    """

    # Evidence synthesis from Summarizer phase
    synthesized_summary: str

    # Historical patterns from episodic memory
    historical_patterns: List[Dict[str, Any]]

    # Successful optimization examples from semantic memory
    optimization_examples: List[Dict[str, Any]]

    # Agent-specific performance baselines
    agent_baselines: Dict[str, Dict[str, float]]

    # Prior learning outcomes from procedural memory
    prior_learnings: List[Dict[str, Any]]

    # Cross-agent correlation insights
    correlation_insights: List[Dict[str, Any]]

    # Confidence in retrieved evidence
    evidence_confidence: float


# =============================================================================
# 2. TRAINING SIGNAL STRUCTURE
# =============================================================================


class AgentTrainingSignal(TypedDict):
    """Training signal emitted by any E2I agent for DSPy optimization."""

    # Signal metadata
    signal_id: str
    source_agent: str
    timestamp: str

    # Input context
    input_context: Dict[str, Any]

    # Agent output
    output: Dict[str, Any]

    # Ground truth / feedback
    user_feedback: Optional[Dict[str, Any]]
    outcome_observed: Optional[Dict[str, Any]]

    # Pre-computed metrics
    latency_ms: float
    token_count: Optional[int]

    # Phase information (for CognitiveRAG integration)
    cognitive_phase: Optional[str]


# =============================================================================
# 3. FEEDBACK LEARNER TRAINING SIGNAL
# =============================================================================


@dataclass
class FeedbackLearnerTrainingSignal:
    """
    Training signal for MIPROv2 optimization of feedback learning prompts.

    This signal captures the full context of a learning cycle including:
    - Input: Collected feedback batch and cognitive context
    - Output: Detected patterns, recommendations, applied updates
    - Outcomes: Improvement in downstream agent metrics

    The compute_reward() method produces a scalar that MIPROv2 uses to
    optimize the DSPy signatures used in pattern detection and recommendation.
    """

    # === Input Context ===
    batch_id: str
    feedback_count: int
    time_range_start: str
    time_range_end: str
    focus_agents: List[str] = field(default_factory=list)
    cognitive_context: Optional[Dict[str, Any]] = None

    # === Processing Outputs ===
    patterns_detected: int = 0
    recommendations_generated: int = 0
    updates_applied: int = 0

    # === Quality Metrics ===
    pattern_accuracy: float = 0.0  # Validated by human review
    recommendation_actionability: float = 0.0  # Percentage implemented
    update_effectiveness: float = 0.0  # Downstream metric improvement

    # === Rubric Evaluation Metrics ===
    rubric_weighted_score: Optional[float] = None  # AI-as-judge rubric score (1-5)
    rubric_decision: Optional[str] = None  # ImprovementDecision value
    rubric_pattern_flags: int = 0  # Number of quality issues flagged

    # === Latency ===
    collection_latency_ms: float = 0.0
    analysis_latency_ms: float = 0.0
    extraction_latency_ms: float = 0.0
    update_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # === LLM Usage ===
    model_used: str = "deterministic"
    llm_calls: int = 0
    total_tokens: int = 0

    # === Outcome (Delayed) ===
    metric_improvement_7d: Optional[float] = None
    metric_improvement_30d: Optional[float] = None
    agent_satisfaction_delta: Optional[float] = None

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward score for MIPROv2 optimization.

        Weighting (with rubric):
        - pattern_accuracy: 0.20 (finding real patterns)
        - recommendation_actionability: 0.20 (practical recommendations)
        - update_effectiveness: 0.20 (updates that work)
        - rubric_quality: 0.20 (AI-as-judge rubric score)
        - efficiency: 0.10 (latency vs. feedback processed)
        - coverage: 0.10 (patterns per feedback item)

        Returns:
            Float reward in range [0.0, 1.0]
        """
        # Adjust weights based on whether rubric evaluation is available
        if self.rubric_weighted_score is not None:
            weights = {
                "pattern_accuracy": 0.20,
                "recommendation_actionability": 0.20,
                "update_effectiveness": 0.20,
                "rubric_quality": 0.20,
                "efficiency": 0.10,
                "coverage": 0.10,
            }
        else:
            weights = {
                "pattern_accuracy": 0.25,
                "recommendation_actionability": 0.25,
                "update_effectiveness": 0.25,
                "efficiency": 0.15,
                "coverage": 0.10,
            }

        # Pattern accuracy (0-1)
        accuracy_score = min(1.0, self.pattern_accuracy)

        # Recommendation actionability (0-1)
        actionability_score = min(1.0, self.recommendation_actionability)

        # Update effectiveness (0-1, allow negative for harmful updates)
        effectiveness_score = max(0.0, min(1.0, self.update_effectiveness))

        # Efficiency: feedback processed per second
        # Target: 100 feedback items in <30s = 3.33 items/s
        target_throughput = 3.33
        if self.total_latency_ms > 0:
            actual_throughput = (self.feedback_count * 1000) / self.total_latency_ms
            efficiency_score = min(1.0, actual_throughput / target_throughput)
        else:
            efficiency_score = 1.0 if self.feedback_count == 0 else 0.5

        # Coverage: patterns detected per feedback item
        # Target: 1 pattern per 10 feedback items
        target_ratio = 0.1
        if self.feedback_count > 0:
            actual_ratio = self.patterns_detected / self.feedback_count
            coverage_score = min(1.0, actual_ratio / target_ratio)
        else:
            coverage_score = 0.0

        # Rubric quality: normalize 1-5 scale to 0-1
        # Score >= 4.0 is "acceptable", so 4.0 maps to 0.75, 5.0 maps to 1.0
        if self.rubric_weighted_score is not None:
            rubric_score = max(0.0, min(1.0, (self.rubric_weighted_score - 1.0) / 4.0))
        else:
            rubric_score = 0.0

        # Weighted sum
        reward = (
            weights["pattern_accuracy"] * accuracy_score
            + weights["recommendation_actionability"] * actionability_score
            + weights["update_effectiveness"] * effectiveness_score
            + weights["efficiency"] * efficiency_score
            + weights["coverage"] * coverage_score
        )

        # Add rubric score if available
        if self.rubric_weighted_score is not None and "rubric_quality" in weights:
            reward += weights["rubric_quality"] * rubric_score

        return round(reward, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": f"fbl_{self.batch_id}",
            "source_agent": "feedback_learner",
            "timestamp": self.created_at,
            "input_context": {
                "batch_id": self.batch_id,
                "feedback_count": self.feedback_count,
                "time_range_start": self.time_range_start,
                "time_range_end": self.time_range_end,
                "focus_agents": self.focus_agents,
                "has_cognitive_context": self.cognitive_context is not None,
            },
            "output": {
                "patterns_detected": self.patterns_detected,
                "recommendations_generated": self.recommendations_generated,
                "updates_applied": self.updates_applied,
            },
            "quality_metrics": {
                "pattern_accuracy": self.pattern_accuracy,
                "recommendation_actionability": self.recommendation_actionability,
                "update_effectiveness": self.update_effectiveness,
            },
            "rubric_evaluation": {
                "weighted_score": self.rubric_weighted_score,
                "decision": self.rubric_decision,
                "pattern_flags": self.rubric_pattern_flags,
            },
            "latency": {
                "collection_ms": self.collection_latency_ms,
                "analysis_ms": self.analysis_latency_ms,
                "extraction_ms": self.extraction_latency_ms,
                "update_ms": self.update_latency_ms,
                "total_ms": self.total_latency_ms,
            },
            "llm_usage": {
                "model": self.model_used,
                "calls": self.llm_calls,
                "tokens": self.total_tokens,
            },
            "outcomes": {
                "improvement_7d": self.metric_improvement_7d,
                "improvement_30d": self.metric_improvement_30d,
                "satisfaction_delta": self.agent_satisfaction_delta,
            },
            "reward": self.compute_reward(),
        }


# =============================================================================
# 4. DSPy SIGNATURES FOR FEEDBACK LEARNING
# =============================================================================

# Note: These signatures require dspy to be installed.
# Import is conditional to allow module to work without dspy.

try:
    import dspy

    class PatternDetectionSignature(dspy.Signature):
        """
        Detect systematic patterns in user feedback.

        Analyzes feedback items to identify recurring issues,
        common complaints, and improvement opportunities.
        """

        feedback_batch: str = dspy.InputField(
            desc="JSON array of feedback items with ratings, corrections, and comments"
        )
        agent_baselines: str = dspy.InputField(desc="Current performance baselines for each agent")
        historical_patterns: str = dspy.InputField(desc="Previously detected patterns for context")

        patterns: list = dspy.OutputField(
            desc="List of detected patterns with type, severity, affected_agents, frequency"
        )
        confidence: float = dspy.OutputField(desc="Confidence in pattern detection (0.0-1.0)")
        root_causes: list = dspy.OutputField(desc="Hypothesized root causes for each pattern")

    class RecommendationGenerationSignature(dspy.Signature):
        """
        Generate improvement recommendations from detected patterns.

        Produces actionable recommendations categorized by:
        - prompt_update: Changes to agent prompts
        - model_retrain: Need for model retraining
        - data_update: Data quality or coverage issues
        - config_change: Configuration adjustments
        - new_capability: New feature requirements
        """

        detected_patterns: str = dspy.InputField(desc="Patterns detected in feedback analysis")
        prior_learnings: str = dspy.InputField(desc="What worked in past learning cycles")
        optimization_examples: str = dspy.InputField(desc="Successful optimization examples")

        recommendations: list = dspy.OutputField(
            desc="Prioritized list of recommendations with category, description, expected_impact"
        )
        implementation_order: list = dspy.OutputField(desc="Recommended order of implementation")
        risk_assessment: str = dspy.OutputField(
            desc="Potential risks of implementing recommendations"
        )

    class KnowledgeUpdateSignature(dspy.Signature):
        """
        Determine knowledge base updates from recommendations.

        Translates recommendations into concrete updates to:
        - Agent prompts
        - Configuration parameters
        - Knowledge graph relationships
        - Procedural memory
        """

        recommendation: str = dspy.InputField(
            desc="Single recommendation to translate to knowledge update"
        )
        current_knowledge: str = dspy.InputField(
            desc="Current state of the knowledge to be updated"
        )

        update_type: str = dspy.OutputField(
            desc="Type: experiment | baseline | agent_config | prompt | threshold"
        )
        key: str = dspy.OutputField(desc="Key path to update in knowledge store")
        old_value: str = dspy.OutputField(desc="Current value (for rollback)")
        new_value: str = dspy.OutputField(desc="Proposed new value")
        justification: str = dspy.OutputField(desc="Reason for this update")

    class LearningSummarySignature(dspy.Signature):
        """
        Generate executive summary of learning cycle outcomes.

        Produces a concise summary suitable for:
        - Stakeholder reporting
        - Agent handoff context
        - Knowledge graph storage
        """

        patterns: str = dspy.InputField(desc="Detected patterns")
        recommendations: str = dspy.InputField(desc="Generated recommendations")
        applied_updates: str = dspy.InputField(desc="Updates that were applied")
        feedback_stats: str = dspy.InputField(desc="Feedback statistics summary")

        summary: str = dspy.OutputField(desc="Executive summary in 2-3 paragraphs")
        key_insights: list = dspy.OutputField(desc="Top 3-5 key insights from this learning cycle")
        next_steps: list = dspy.OutputField(desc="Recommended follow-up actions")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Feedback Learner agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic pattern detection")

    # Placeholder classes when dspy is not available
    PatternDetectionSignature = None
    RecommendationGenerationSignature = None
    KnowledgeUpdateSignature = None
    LearningSummarySignature = None


# =============================================================================
# 5. MIPROv2 OPTIMIZATION HELPERS
# =============================================================================


class FeedbackLearnerOptimizer:
    """
    MIPROv2 optimizer for Feedback Learner agent prompts.

    Uses collected training signals to optimize the DSPy signatures
    used in pattern detection, recommendation generation, and
    knowledge updates.
    """

    def __init__(self, signal_store: Optional[Any] = None):
        """
        Initialize optimizer with signal store.

        Args:
            signal_store: Store for retrieving historical training signals
        """
        self.signal_store = signal_store
        self._cached_examples = {}

    def pattern_metric(self, example, prediction, trace=None) -> float:
        """
        Metric for pattern detection optimization.

        Good pattern detection should:
        1. Find patterns that are actionable
        2. Correctly identify affected agents
        3. Produce accurate root cause hypotheses
        """
        score = 0.0

        # Patterns should be specific
        if hasattr(prediction, "patterns") and prediction.patterns:
            for pattern in prediction.patterns:
                if isinstance(pattern, dict):
                    # Has required fields
                    if all(k in pattern for k in ["type", "severity", "affected_agents"]):
                        score += 0.1
                    # Has root cause
                    if pattern.get("root_cause_hypothesis"):
                        score += 0.05

        # Confidence should be calibrated (penalize over/under confidence)
        if hasattr(prediction, "confidence"):
            conf = prediction.confidence
            if 0.3 <= conf <= 0.9:
                score += 0.2

        # Root causes should be specific
        if hasattr(prediction, "root_causes") and prediction.root_causes:
            score += min(0.3, len(prediction.root_causes) * 0.1)

        return min(1.0, score)

    def recommendation_metric(self, example, prediction, trace=None) -> float:
        """
        Metric for recommendation generation optimization.

        Good recommendations should be:
        1. Actionable (specific, implementable)
        2. Prioritized correctly
        3. Have realistic expected impacts
        """
        score = 0.0

        # Recommendations should be actionable
        if hasattr(prediction, "recommendations") and prediction.recommendations:
            for rec in prediction.recommendations:
                if isinstance(rec, dict):
                    # Has category
                    if rec.get("category") in [
                        "prompt_update",
                        "model_retrain",
                        "data_update",
                        "config_change",
                        "new_capability",
                    ]:
                        score += 0.1
                    # Has expected impact
                    if rec.get("expected_impact"):
                        score += 0.05

        # Implementation order should be provided
        if hasattr(prediction, "implementation_order") and prediction.implementation_order:
            score += 0.2

        # Risk assessment should be thoughtful
        if hasattr(prediction, "risk_assessment") and len(str(prediction.risk_assessment)) > 50:
            score += 0.2

        return min(1.0, score)

    async def optimize(
        self,
        phase: Literal["pattern", "recommendation", "update", "summary"],
        training_signals: List[Dict[str, Any]],
        budget: int = 50,
    ) -> Optional[Any]:
        """
        Run MIPROv2 optimization for a specific phase.

        Args:
            phase: Which signature to optimize
            training_signals: Historical training signals
            budget: Number of optimization trials

        Returns:
            Optimized DSPy module or None if optimization fails
        """
        if not DSPY_AVAILABLE:
            logger.warning("DSPy not available for optimization")
            return None

        import dspy
        from dspy.teleprompt import MIPROv2

        # Convert signals to examples
        examples = self._signals_to_examples(training_signals, phase)

        if len(examples) < 5:
            logger.warning(f"Insufficient examples ({len(examples)}) for optimization")
            return None

        metrics = {
            "pattern": self.pattern_metric,
            "recommendation": self.recommendation_metric,
            # Add more as needed
        }

        signatures = {
            "pattern": PatternDetectionSignature,
            "recommendation": RecommendationGenerationSignature,
        }

        if phase not in metrics or phase not in signatures:
            logger.warning(f"Unknown optimization phase: {phase}")
            return None

        optimizer = MIPROv2(
            metric=metrics[phase], num_candidates=10, max_bootstrapped_demos=4, num_threads=4
        )

        module = dspy.ChainOfThought(signatures[phase])

        optimized = optimizer.compile(module, trainset=examples, num_trials=budget)

        return optimized

    def _signals_to_examples(self, signals: List[Dict[str, Any]], phase: str) -> List:
        """Convert training signals to DSPy Examples."""
        if not DSPY_AVAILABLE:
            return []

        import dspy

        examples = []
        for signal in signals:
            # Filter by phase and success
            if signal.get("source_agent") != "feedback_learner":
                continue

            reward = signal.get("reward", 0)
            if reward < 0.5:  # Only use successful examples
                continue

            try:
                if phase == "pattern":
                    example = dspy.Example(
                        feedback_batch=str(
                            signal.get("input_context", {}).get("feedback_batch", [])
                        ),
                        agent_baselines="{}",
                        historical_patterns="[]",
                        patterns=signal.get("output", {}).get("patterns", []),
                        confidence=0.8,
                        root_causes=[],
                    ).with_inputs("feedback_batch", "agent_baselines", "historical_patterns")
                    examples.append(example)

            except Exception as e:
                logger.debug(f"Failed to convert signal to example: {e}")

        return examples


# =============================================================================
# 6. MEMORY CONTRIBUTION HELPERS
# =============================================================================


def create_memory_contribution(
    signal: FeedbackLearnerTrainingSignal,
    memory_type: Literal["episodic", "semantic", "procedural"] = "semantic",
) -> Dict[str, Any]:
    """
    Create memory contribution from training signal.

    Feedback Learner primarily contributes to SEMANTIC memory (knowledge graph)
    with new patterns, successful optimizations, and causal relationships.

    Args:
        signal: Training signal from learning cycle
        memory_type: Target memory type

    Returns:
        Memory contribution ready for storage
    """
    contribution = {
        "source_agent": "feedback_learner",
        "memory_type": memory_type,
        "created_at": signal.created_at,
        "batch_id": signal.batch_id,
    }

    if memory_type == "semantic":
        # Semantic memory: knowledge graph entities and relationships
        contribution.update(
            {
                "index": "learning_outcomes",
                "ttl_days": 365,
                "entities": [
                    {
                        "type": "LearningCycle",
                        "id": signal.batch_id,
                        "properties": {
                            "feedback_count": signal.feedback_count,
                            "patterns_detected": signal.patterns_detected,
                            "recommendations_generated": signal.recommendations_generated,
                            "updates_applied": signal.updates_applied,
                            "reward": signal.compute_reward(),
                        },
                    }
                ],
                "relationships": (
                    [
                        {
                            "type": "IMPROVED",
                            "from": {"type": "LearningCycle", "id": signal.batch_id},
                            "to": {"type": "Agent", "id": agent},
                            "properties": {"cycle_date": signal.created_at},
                        }
                        for agent in signal.focus_agents
                    ]
                    if signal.focus_agents
                    else []
                ),
            }
        )

    elif memory_type == "episodic":
        # Episodic memory: specific learning experiences
        contribution.update(
            {
                "index": "learning_experiences",
                "ttl_days": 180,
                "content": {
                    "batch_id": signal.batch_id,
                    "summary": f"Processed {signal.feedback_count} feedback items, "
                    f"detected {signal.patterns_detected} patterns, "
                    f"generated {signal.recommendations_generated} recommendations",
                    "reward": signal.compute_reward(),
                },
            }
        )

    elif memory_type == "procedural":
        # Procedural memory: successful learning procedures
        if signal.compute_reward() >= 0.7:  # Only store high-quality procedures
            contribution.update(
                {
                    "index": "learning_procedures",
                    "ttl_days": 365,
                    "procedure": {
                        "trigger": "feedback_batch",
                        "conditions": {
                            "min_feedback_count": 10,
                            "focus_agents": signal.focus_agents,
                        },
                        "steps": [
                            "collect_feedback",
                            "analyze_patterns",
                            "extract_learnings",
                            "update_knowledge",
                        ],
                        "expected_outcome": f"reward >= {signal.compute_reward():.2f}",
                    },
                }
            )

    return contribution


# =============================================================================
# 7. EXPORTS
# =============================================================================

__all__ = [
    # Cognitive Context
    "FeedbackLearnerCognitiveContext",
    # Training Signals
    "AgentTrainingSignal",
    "FeedbackLearnerTrainingSignal",
    # DSPy Signatures (may be None if dspy not installed)
    "PatternDetectionSignature",
    "RecommendationGenerationSignature",
    "KnowledgeUpdateSignature",
    "LearningSummarySignature",
    # Optimization
    "FeedbackLearnerOptimizer",
    "DSPY_AVAILABLE",
    # Memory
    "create_memory_contribution",
]
