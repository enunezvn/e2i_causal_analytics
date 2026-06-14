"""
E2I Feedback Learner Agent - DSPy Integration Module
Version: 4.3
Purpose: DSPy signatures, training signals, and GEPA/MIPROv2 optimization for feedback learning

This module implements the DSPy integration patterns for the Feedback Learner agent,
enabling continuous self-improvement through:
1. Training signal collection from all agents
2. GEPA prompt optimization (primary, 10%+ improvement over MIPROv2)
3. MIPROv2 prompt optimization (fallback)
4. Cognitive context enrichment from CognitiveRAG

GEPA Migration (v4.3):
- Added GEPA as the default optimizer for Feedback Learner
- FeedbackLearnerOptimizer now supports optimizer_type="gepa" or "miprov2"
- Integrated with FeedbackLearnerGEPAMetric for reflective evaluation
- Module versioning support via save_optimized_module
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict, Union

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # These names are resolved lazily at runtime via PEP 562 ``__getattr__`` (so
    # importing dspy stays off the fast path). Declaring them here lets static
    # tools (ruff F822 on ``__all__``, mypy) see them without an eager import.
    from src.optimization.gepa import (  # noqa: F401
        create_gepa_optimizer,
        get_metric_for_agent,
        load_optimized_module,
        save_optimized_module,
    )
    from src.optimization.gepa.metrics import FeedbackLearnerGEPAMetric  # noqa: F401

    PatternDetectionSignature: Any
    RecommendationGenerationSignature: Any
    KnowledgeUpdateSignature: Any
    LearningSummarySignature: Any

# Type alias for optimizer selection
OptimizerType = Literal["miprov2", "gepa"]


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
    # F-015 (issue #424): pattern_accuracy is Optional[float] = None when no
    # ground-truth pattern-validation labels are available. Computing this
    # honestly requires labeled validation data (see #426 / F-015-PhaseB).
    # `compute_reward()` skips the term and redistributes its weight when None,
    # rather than anchoring on a fabricated default. Treat `0.0` as "validated
    # poorly" (real signal), and `None` as "no measurement" (skip).
    pattern_accuracy: Optional[float] = None
    recommendation_actionability: float = 0.0  # Percentage implemented
    # F15 (audit): None when the knowledge_store apply-backend is unwired (the
    # ratio is structurally unmeasurable). compute_reward skips it like
    # pattern_accuracy rather than anchoring on a misleading 0.0.
    # #837: when a real knowledge_stores backend IS wired, this is the fraction
    # of proposed updates DURABLY PERSISTED to the store (read-back confirmed) —
    # a real measure of update application, NOT of downstream behavioural impact
    # (agent-side consumption of the recorded learning is a separate loop).
    update_effectiveness: Optional[float] = None  # applied(persisted)/proposed; None when unwired

    # === Rubric Evaluation Metrics ===
    rubric_weighted_score: Optional[float] = None  # AI-as-judge rubric score (1-5)
    rubric_decision: Optional[str] = None  # ImprovementDecision value
    rubric_pattern_flags: int = 0  # Number of quality issues flagged

    # === Carried content (for faithful DSPy example conversion — audit F6) ===
    # Bounded samples of the cycle's actual content so the optimizer trains on
    # real inputs/outputs instead of empty placeholders.
    feedback_batch: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    applied_updates: List[Dict[str, Any]] = field(default_factory=list)
    learning_summary: str = ""

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
        - pattern_accuracy: 0.20 (finding real patterns; skipped if None)
        - recommendation_actionability: 0.20 (practical recommendations)
        - update_effectiveness: 0.20 (updates that work)
        - rubric_quality: 0.20 (AI-as-judge rubric score)
        - efficiency: 0.10 (latency vs. feedback processed)
        - coverage: 0.10 (patterns per feedback item)

        F-015 (issue #424): if `pattern_accuracy` is None (no ground-truth
        labels available), the term is skipped and its weight is redistributed
        proportionally across the remaining present terms — rather than
        substituted with 0.0 (which would penalize an unmeasurable property).

        Returns:
            Float reward in range [0.0, 1.0]
        """
        # Base weights — adjusted below based on which signals are present.
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

        # Recommendation actionability (0-1)
        actionability_score = min(1.0, self.recommendation_actionability)

        # Update effectiveness (0-1). F15: None means structurally unmeasurable
        # (no knowledge_store apply-backend) -> omitted from the reward below.

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

        # Build the (weight, score) list, omitting unmeasured terms.
        # Pattern accuracy is omitted entirely when None (F-015).
        weight_score_pairs: list[tuple[float, float]] = [
            (weights["recommendation_actionability"], actionability_score),
            (weights["efficiency"], efficiency_score),
            (weights["coverage"], coverage_score),
        ]
        if self.pattern_accuracy is not None:
            accuracy_score = min(1.0, max(0.0, self.pattern_accuracy))
            weight_score_pairs.append((weights["pattern_accuracy"], accuracy_score))
        # F15: omit update_effectiveness entirely when None (unmeasurable),
        # mirroring pattern_accuracy, so its weight is redistributed.
        if self.update_effectiveness is not None:
            effectiveness_score = max(0.0, min(1.0, self.update_effectiveness))
            weight_score_pairs.append((weights["update_effectiveness"], effectiveness_score))
        if self.rubric_weighted_score is not None and "rubric_quality" in weights:
            weight_score_pairs.append((weights["rubric_quality"], rubric_score))

        # Redistribute by normalizing present weights to sum to 1.0.
        total_weight = sum(w for w, _ in weight_score_pairs)
        if total_weight <= 0:
            return 0.0
        reward = sum(w * s for w, s in weight_score_pairs) / total_weight

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
                # F6: carry a bounded sample of the real feedback batch.
                "feedback_batch": self.feedback_batch[:20],
            },
            "output": {
                "patterns_detected": self.patterns_detected,
                "recommendations_generated": self.recommendations_generated,
                "updates_applied": self.updates_applied,
                # F6: carry the actual detected content for conversion.
                "patterns": self.patterns[:20],
                "recommendations": self.recommendations[:20],
                "applied_updates": self.applied_updates[:20],
                "learning_summary": self.learning_summary,
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
#
# DSPy availability is probed WITHOUT importing dspy. ``import dspy`` loads
# ~714 MB; this module sits on the import chain of the Health Score Fast Path
# agent ("Zero LLM usage in critical path") via ``recipient_emit`` ->
# ``feedback_learner.__init__`` -> ``feedback_learner.agent`` -> here, so it must
# not pull dspy in at import time. The Signature subclasses below are needed ONLY
# by the GEPA/MIPROv2 optimizer paths, so they are built lazily on first
# attribute access (PEP 562 module ``__getattr__``) and ``dspy`` is imported only
# then.
DSPY_AVAILABLE: bool = importlib.util.find_spec("dspy") is not None
if DSPY_AVAILABLE:
    logger.info("DSPy detected for Feedback Learner agent; import deferred until needed")
else:
    logger.warning("DSPy not available - using deterministic pattern detection")


@functools.lru_cache(maxsize=1)
def _get_feedback_signatures() -> Dict[str, Any]:
    """Build (once) and return the feedback_learner DSPy Signature classes.

    Importing dspy here keeps the ~714 MB import off the fast-path module-import
    chain; it happens only when a GEPA/MIPROv2 optimizer path actually requests
    these signatures. Returns an all-``None`` mapping when dspy is not installed
    (mirrors the historical placeholder behavior).
    """
    if not DSPY_AVAILABLE:
        return {
            "PatternDetectionSignature": None,
            "RecommendationGenerationSignature": None,
            "KnowledgeUpdateSignature": None,
            "LearningSummarySignature": None,
        }

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

    logger.info("DSPy signatures built for Feedback Learner agent")
    return {
        "PatternDetectionSignature": PatternDetectionSignature,
        "RecommendationGenerationSignature": RecommendationGenerationSignature,
        "KnowledgeUpdateSignature": KnowledgeUpdateSignature,
        "LearningSummarySignature": LearningSummarySignature,
    }


# Names resolved lazily via PEP 562: ``from ...dspy_integration import
# PatternDetectionSignature`` (and attribute access) builds the class on first
# use without importing dspy at module import time.
_LAZY_SIGNATURE_NAMES = frozenset(
    {
        "PatternDetectionSignature",
        "RecommendationGenerationSignature",
        "KnowledgeUpdateSignature",
        "LearningSummarySignature",
    }
)


# =============================================================================
# 4.1 GEPA OPTIMIZER SUPPORT
# =============================================================================

# Conditional GEPA support.
#
# ``src.optimization.gepa`` eagerly imports dspy (~714 MB) when its package
# ``__init__`` executes — and even ``importlib.util.find_spec("src.optimization.
# gepa")`` would trigger that, because resolving a DOTTED name imports the parent
# packages (``src.optimization/__init__.py`` eagerly imports gepa -> dspy). So we
# avoid importlib entirely here and probe GEPA availability the only fully
# import-free way: GEPA requires dspy, and its (first-party) package source must
# exist on disk. The real symbols are loaded lazily on first optimizer use via
# ``_get_gepa_symbols()`` / PEP 562 ``__getattr__``.
_GEPA_PKG_INIT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # src/
    "optimization",
    "gepa",
    "__init__.py",
)
GEPA_AVAILABLE: bool = DSPY_AVAILABLE and os.path.isfile(_GEPA_PKG_INIT)
if GEPA_AVAILABLE:
    logger.info("GEPA optimizer detected for Feedback Learner agent; import deferred until needed")
else:
    logger.info("GEPA not available - using MIPROv2 optimizer")

_GEPA_SYMBOL_NAMES = frozenset(
    {
        "create_gepa_optimizer",
        "get_metric_for_agent",
        "load_optimized_module",
        "save_optimized_module",
        "FeedbackLearnerGEPAMetric",
    }
)


@functools.lru_cache(maxsize=1)
def _get_gepa_symbols() -> Dict[str, Any]:
    """Import (once) and return the GEPA optimizer symbols.

    Deferred so the eager dspy import inside ``src.optimization.gepa`` stays off
    the fast-path module-import chain. Returns an all-``None`` mapping when GEPA
    is not importable (mirrors the historical placeholder behavior).
    """
    if not GEPA_AVAILABLE:
        return dict.fromkeys(_GEPA_SYMBOL_NAMES)

    try:
        from src.optimization.gepa import (
            create_gepa_optimizer,
            get_metric_for_agent,
            load_optimized_module,
            save_optimized_module,
        )
        from src.optimization.gepa.metrics import FeedbackLearnerGEPAMetric

        logger.info("GEPA optimizer symbols loaded for Feedback Learner agent")
        return {
            "create_gepa_optimizer": create_gepa_optimizer,
            "get_metric_for_agent": get_metric_for_agent,
            "load_optimized_module": load_optimized_module,
            "save_optimized_module": save_optimized_module,
            "FeedbackLearnerGEPAMetric": FeedbackLearnerGEPAMetric,
        }
    except ImportError:
        logger.info("GEPA import failed at use time - using MIPROv2 optimizer")
        return dict.fromkeys(_GEPA_SYMBOL_NAMES)


def __getattr__(name: str) -> Any:
    """Lazily resolve DSPy Signature classes and GEPA symbols (PEP 562).

    Keeps the public import surface unchanged
    (``from ...dspy_integration import PatternDetectionSignature`` /
    ``save_optimized_module`` still work) while deferring the ~714 MB dspy import
    until a name is actually accessed.
    """
    if name in _LAZY_SIGNATURE_NAMES:
        return _get_feedback_signatures()[name]
    if name in _GEPA_SYMBOL_NAMES:
        return _get_gepa_symbols()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# 5. GEPA OPTIMIZATION TRIGGER
# =============================================================================


@dataclass
class GEPAOptimizationTrigger:
    """
    Determines when to trigger GEPA optimization based on accumulated signals.

    The trigger evaluates multiple conditions:
    1. Minimum signal count: Enough training data for optimization
    2. Reward delta: Significant change in performance
    3. Cooldown period: Prevent excessive optimization runs
    4. Pattern severity: Critical patterns may force optimization

    Usage:
        trigger = GEPAOptimizationTrigger()
        should_trigger, reason = trigger.should_trigger(
            signal_count=150,
            current_reward=0.72,
            baseline_reward=0.65,
            last_optimization=datetime(2025, 1, 1),
            has_critical_patterns=False,
        )
    """

    # Minimum signals required for optimization
    # ~1 signal/cycle; 20 ≈ reachable in normal operation
    min_signals: int = 20

    # Minimum reward improvement delta to trigger
    min_reward_delta: float = 0.05

    # Minimum hours between optimization runs
    cooldown_hours: int = 24

    # Maximum hours before forcing optimization (regardless of delta)
    max_hours_without_optimization: int = 168  # 7 days

    # Critical pattern severity forces immediate optimization
    critical_pattern_triggers: bool = True

    def should_trigger(
        self,
        signal_count: int,
        current_reward: float,
        baseline_reward: float = 0.0,
        last_optimization: Optional[datetime] = None,
        has_critical_patterns: bool = False,
    ) -> tuple[bool, str]:
        """
        Determine if GEPA optimization should be triggered.

        Args:
            signal_count: Number of accumulated training signals
            current_reward: Average reward from recent learning cycles
            baseline_reward: Reward from last optimization baseline
            last_optimization: Timestamp of last optimization run
            has_critical_patterns: Whether critical patterns were detected

        Returns:
            Tuple of (should_trigger, reason)
        """
        now = datetime.now(timezone.utc)

        # Check cooldown period
        if last_optimization is not None:
            hours_since = (now - last_optimization).total_seconds() / 3600
            if hours_since < self.cooldown_hours:
                return (
                    False,
                    f"Cooldown active: {hours_since:.1f}h < {self.cooldown_hours}h",
                )
        else:
            # No previous optimization - skip forced check, rely on reward delta
            hours_since = 0.0

        # Critical patterns override other checks
        if has_critical_patterns and self.critical_pattern_triggers:
            if signal_count >= self.min_signals // 2:  # Require half the signals
                return (
                    True,
                    f"Critical patterns detected with {signal_count} signals",
                )

        # Check minimum signal count
        if signal_count < self.min_signals:
            return (
                False,
                f"Insufficient signals: {signal_count} < {self.min_signals}",
            )

        # Force optimization if too long since last run
        if hours_since > self.max_hours_without_optimization:
            return (
                True,
                f"Forced: {hours_since:.1f}h since last optimization",
            )

        # Check reward delta
        reward_delta = current_reward - baseline_reward
        if reward_delta >= self.min_reward_delta:
            return (
                True,
                f"Reward improved: {reward_delta:.3f} >= {self.min_reward_delta}",
            )
        elif reward_delta <= -self.min_reward_delta:
            return (
                True,
                f"Reward degraded: {reward_delta:.3f} (needs recovery)",
            )

        return (
            False,
            f"No trigger: delta={reward_delta:.3f}, signals={signal_count}",
        )

    def get_recommended_budget(
        self,
        signal_count: int,
        hours_since_last: float,
        has_critical_patterns: bool = False,
    ) -> str:
        """
        Get recommended GEPA budget based on context.

        Args:
            signal_count: Number of accumulated signals
            hours_since_last: Hours since last optimization
            has_critical_patterns: Whether critical patterns exist

        Returns:
            Budget preset: "light", "medium", or "heavy"
        """
        # Critical patterns need thorough optimization
        if has_critical_patterns:
            return "heavy"

        # Long time since optimization - be thorough
        if hours_since_last > self.max_hours_without_optimization:
            return "heavy"

        # Many signals available - use them
        if signal_count > self.min_signals * 3:
            return "heavy"
        elif signal_count > self.min_signals * 2:
            return "medium"
        else:
            return "light"


# =============================================================================
# 6. DSPy OPTIMIZATION HELPERS (MIPROv2 + GEPA)
# =============================================================================


class FeedbackLearnerOptimizer:
    """
    DSPy optimizer for Feedback Learner agent prompts.

    Supports both MIPROv2 and GEPA optimizers. Uses collected training
    signals to optimize the DSPy signatures used in pattern detection,
    recommendation generation, and knowledge updates.

    GEPA provides 10%+ improvement over MIPROv2 via:
    - Reflective evolution with rich textual feedback
    - Pareto frontier for multi-objective optimization
    - Better exploration of prompt space
    """

    def __init__(
        self,
        signal_store: Optional[Any] = None,
        optimizer_type: OptimizerType = "gepa",
    ):
        """
        Initialize optimizer with signal store.

        Args:
            signal_store: Store for retrieving historical training signals
            optimizer_type: Optimizer to use - "gepa" (recommended) or "miprov2"
        """
        self.signal_store = signal_store
        self._cached_examples: Dict[str, Any] = {}

        # Select optimizer based on availability and preference
        if optimizer_type == "gepa" and GEPA_AVAILABLE:
            self.optimizer_type = "gepa"
            logger.info("Using GEPA optimizer for Feedback Learner")
        elif DSPY_AVAILABLE:
            self.optimizer_type = "miprov2"
            if optimizer_type == "gepa":
                logger.warning("GEPA not available, falling back to MIPROv2")
            else:
                logger.info("Using MIPROv2 optimizer for Feedback Learner")
        else:
            self.optimizer_type = None  # type: ignore[assignment]
            logger.warning("No optimizer available - optimization disabled")

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

    def summary_metric(self, example, prediction, trace=None) -> float:
        """Metric for the learning-summary phase (MIPROv2).

        The LearningSummarySignature outputs `summary`/`key_insights`/`next_steps`
        — the recommendation_metric scores none of these (it would be signal-deaf
        for this phase). Score the actual summary outputs instead.
        """
        score = 0.0
        summary = str(getattr(prediction, "summary", "") or "").strip()
        if len(summary) >= 40:
            score += 0.4
        elif summary:
            score += 0.2
        if getattr(prediction, "key_insights", None):
            score += 0.3
        if getattr(prediction, "next_steps", None):
            score += 0.3
        return min(1.0, score)

    async def optimize(
        self,
        phase: Literal["pattern", "recommendation", "update", "summary"],
        training_signals: List[Dict[str, Any]],
        budget: Union[int, str] = "medium",
    ) -> Optional[Any]:
        """
        Run optimization for a specific phase using GEPA or MIPROv2.

        Args:
            phase: Which signature to optimize
            training_signals: Historical training signals
            budget: For GEPA: "light", "medium", "heavy". For MIPROv2: int (trials)

        Returns:
            Optimized DSPy module or None if optimization fails
        """
        if self.optimizer_type == "gepa":
            gepa_budget = budget if isinstance(budget, str) else "medium"
            return await self._optimize_with_gepa(phase, training_signals, gepa_budget)
        elif self.optimizer_type == "miprov2":
            # Convert string budget to int for MIPROv2
            mipro_budget = (
                {"light": 20, "medium": 50, "heavy": 100}.get(budget, 50)
                if isinstance(budget, str)
                else budget
            )
            return await self._optimize_with_miprov2(phase, training_signals, mipro_budget)
        else:
            logger.warning("No optimizer available")
            return None

    async def _optimize_with_gepa(
        self,
        phase: Literal["pattern", "recommendation", "update", "summary"],
        training_signals: List[Dict[str, Any]],
        budget: str = "medium",
    ) -> Optional[Any]:
        """
        Run GEPA optimization for a specific phase.

        GEPA uses reflective evolution with rich textual feedback,
        providing 10%+ improvement over MIPROv2.

        Args:
            phase: Which signature to optimize
            training_signals: Historical training signals
            budget: GEPA budget preset - "light", "medium", "heavy"

        Returns:
            Optimized DSPy module or None if optimization fails
        """
        if not GEPA_AVAILABLE or not DSPY_AVAILABLE:
            logger.warning("GEPA or DSPy not available for optimization")
            return None

        import dspy

        # Resolve dspy-backed Signature classes and GEPA symbols lazily (the
        # import that loads dspy happens here, not at module import time).
        _sigs = _get_feedback_signatures()
        _gepa = _get_gepa_symbols()
        get_metric_for_agent = _gepa["get_metric_for_agent"]
        create_gepa_optimizer = _gepa["create_gepa_optimizer"]

        # Convert signals to examples
        examples = self._signals_to_examples(training_signals, phase)
        trainset = examples[: int(len(examples) * 0.8)]
        valset = examples[int(len(examples) * 0.8) :]

        if len(trainset) < 5:
            logger.warning(f"Insufficient examples ({len(trainset)}) for GEPA optimization")
            return None

        signatures = {
            "pattern": _sigs["PatternDetectionSignature"],
            "recommendation": _sigs["RecommendationGenerationSignature"],
            "update": _sigs["KnowledgeUpdateSignature"],
            "summary": _sigs["LearningSummarySignature"],
        }

        if phase not in signatures or signatures[phase] is None:
            logger.warning(f"Unknown or unavailable optimization phase: {phase}")
            return None

        # Get GEPA metric for feedback learner
        metric = get_metric_for_agent("feedback_learner")

        # Create GEPA optimizer. create_gepa_optimizer maps the budget preset to
        # GEPA's `auto` parameter; passing budget= would land in **kwargs and
        # raise TypeError in GEPA() (no `budget` param). Use auto=budget.
        optimizer = create_gepa_optimizer(
            metric=metric,
            trainset=trainset,
            valset=valset,
            auto=budget,
            enable_tool_optimization=False,  # Feedback Learner is Deep agent, not Hybrid
            seed=42,
        )

        # Create module
        module = dspy.ChainOfThought(signatures[phase])

        # Bind the configured LM directly to the module so GEPA's parallel
        # evaluation workers use it. dspy's thread-local settings.lm does NOT
        # propagate to the optimizer's worker threads, so relying on the global
        # config alone yields "No LM is loaded" in every rollout (scores 0.0).
        lm = getattr(dspy.settings, "lm", None)
        if lm is not None and hasattr(module, "set_lm"):
            module.set_lm(lm)

        # Run optimization
        logger.info(f"Starting GEPA optimization for {phase} phase with budget={budget}")
        # Pass the held-out valset so GEPA validates on unseen examples (F6).
        optimized = optimizer.compile(module, trainset=trainset, valset=valset)

        # Saving is handled by the orchestrator (optimization_runner) using the
        # sync save_optimized_module(module, agent_name, metadata=...) signature.
        return optimized

    async def _optimize_with_miprov2(
        self,
        phase: Literal["pattern", "recommendation", "update", "summary"],
        training_signals: List[Dict[str, Any]],
        budget: int = 50,
    ) -> Optional[Any]:
        """
        Run MIPROv2 optimization for a specific phase (legacy).

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

        # Resolve dspy-backed Signature classes lazily (loads dspy here, not at
        # module import time).
        _sigs = _get_feedback_signatures()

        # Convert signals to examples
        examples = self._signals_to_examples(training_signals, phase)

        if len(examples) < 5:
            logger.warning(f"Insufficient examples ({len(examples)}) for optimization")
            return None

        signatures = {
            "pattern": _sigs["PatternDetectionSignature"],
            "recommendation": _sigs["RecommendationGenerationSignature"],
            "summary": _sigs["LearningSummarySignature"],
        }
        metrics = {
            "pattern": self.pattern_metric,
            "recommendation": self.recommendation_metric,
            "summary": self.summary_metric,
        }
        if phase == "update" or phase not in signatures or phase not in metrics:
            logger.warning(f"Phase '{phase}' not optimizable via MIPROv2; skipping")
            return None

        optimizer = MIPROv2(
            metric=metrics[phase], num_candidates=10, max_bootstrapped_demos=4, num_threads=4
        )

        module = dspy.ChainOfThought(signatures[phase])

        optimized = optimizer.compile(module, trainset=examples, num_trials=budget)

        return optimized

    def _signals_to_examples(self, signals: List[Dict[str, Any]], phase: str) -> List:
        """Convert persisted training signals to DSPy Examples for a phase.

        Builds real, content-bearing examples for 'pattern', 'recommendation',
        and 'summary'. The 'update' phase is intentionally skipped: its
        KnowledgeUpdateSignature needs a (recommendation, current_knowledge)
        pair we do not persist, so producing examples would be fabrication.
        """
        if not DSPY_AVAILABLE:
            return []

        import json as _json

        import dspy

        if phase == "update":
            logger.info(
                "Skipping 'update' phase optimization: no paired current_knowledge "
                "examples are persisted (audit F6 — honest skip, not a silent stub)."
            )
            return []

        examples: list = []
        for signal in signals:
            if signal.get("source_agent") != "feedback_learner":
                continue
            if signal.get("reward", 0) < 0.5:  # only successful cycles
                continue

            ic = signal.get("input_context", {}) or {}
            out = signal.get("output", {}) or {}
            feedback_batch = ic.get("feedback_batch", []) or []
            patterns = out.get("patterns", []) or []
            recommendations = out.get("recommendations", []) or []
            applied_updates = out.get("applied_updates", []) or []
            learning_summary = out.get("learning_summary", "") or ""

            # Skip degenerate signals that carry no usable content for any phase
            # (recommendations included so a recs-only cycle still trains the
            # recommendation phase).
            if not feedback_batch and not patterns and not recommendations and not learning_summary:
                continue

            try:
                if phase == "pattern":
                    ex = dspy.Example(
                        feedback_batch=_json.dumps(feedback_batch),
                        agent_baselines=_json.dumps(ic.get("agent_baselines", {})),
                        historical_patterns=_json.dumps(ic.get("historical_patterns", [])),
                        patterns=patterns,
                        confidence=0.8,
                        root_causes=[
                            p.get("root_cause_hypothesis")
                            for p in patterns
                            if isinstance(p, dict) and p.get("root_cause_hypothesis")
                        ],
                    ).with_inputs("feedback_batch", "agent_baselines", "historical_patterns")
                    examples.append(ex)

                elif phase == "recommendation":
                    if not patterns and not recommendations:
                        continue
                    ex = dspy.Example(
                        detected_patterns=_json.dumps(patterns),
                        prior_learnings=_json.dumps(ic.get("prior_learnings", [])),
                        optimization_examples=_json.dumps(ic.get("optimization_examples", [])),
                        recommendations=recommendations,
                        implementation_order=[
                            r.get("category") for r in recommendations if isinstance(r, dict)
                        ],
                        risk_assessment=str(out.get("risk_assessment", "")),
                    ).with_inputs("detected_patterns", "prior_learnings", "optimization_examples")
                    examples.append(ex)

                elif phase == "summary":
                    if not learning_summary and not patterns:
                        continue
                    ex = dspy.Example(
                        patterns=_json.dumps(patterns),
                        recommendations=_json.dumps(recommendations),
                        applied_updates=_json.dumps(applied_updates),
                        feedback_stats=_json.dumps(ic),
                        summary=learning_summary,
                        key_insights=[],
                        next_steps=[],
                    ).with_inputs(
                        "patterns", "recommendations", "applied_updates", "feedback_stats"
                    )
                    examples.append(ex)

            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to convert signal to example ({phase}): {e}")

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
    contribution: Dict[str, Any] = {
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
    "GEPAOptimizationTrigger",
    "DSPY_AVAILABLE",
    "GEPA_AVAILABLE",
    "OptimizerType",
    # Memory
    "create_memory_contribution",
]
