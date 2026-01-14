"""
DSPy Integration for E2I Chatbot Intent Classification and Agent Routing.

This module provides DSPy-based classification with:
- Intent Classification: Structured signatures for reliable classification
- Agent Routing: Intelligent routing to the appropriate E2I tier agents
- Confidence scores for each prediction
- Training signal collection for continuous improvement
- Graceful fallback to hardcoded patterns if DSPy unavailable

Phases 3-4 of the CopilotKit-DSPy observability integration plan.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .chatbot_state import IntentType

logger = logging.getLogger(__name__)

# Check DSPy availability
try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using hardcoded intent classification")

# Feature flag for DSPy intent classification
CHATBOT_DSPY_INTENT_ENABLED = os.getenv("CHATBOT_DSPY_INTENT", "true").lower() == "true"

# DSPy LLM configuration flag (to ensure we only configure once)
_dspy_lm_configured = False


def _ensure_dspy_configured():
    """Ensure DSPy LLM is configured before use."""
    global _dspy_lm_configured
    if not DSPY_AVAILABLE or _dspy_lm_configured:
        return

    try:
        # Check if already configured
        if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            _dspy_lm_configured = True
            return

        # Configure DSPy with Anthropic Claude
        lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
        dspy.configure(lm=lm)
        _dspy_lm_configured = True
        logger.info("Configured DSPy LLM for chatbot intent classification")
    except Exception as e:
        logger.warning(f"Failed to configure DSPy LLM: {e}")


# =============================================================================
# DSPY SIGNATURES
# =============================================================================

if DSPY_AVAILABLE:

    class ChatbotIntentClassificationSignature(dspy.Signature):
        """
        Classify user query intent for E2I pharmaceutical analytics chatbot.

        The chatbot handles queries about pharmaceutical commercial operations
        including KPIs (TRx, NRx, market share), causal analysis, recommendations,
        and general assistance.

        Intent Categories:
        - kpi_query: Questions about specific metrics (TRx, NRx, market share, volume)
        - causal_analysis: Questions about why something happened, drivers, impacts
        - agent_status: Questions about system agents or their status
        - recommendation: Requests for suggestions, improvements, strategies
        - search: Requests to find or look up specific information
        - multi_faceted: Complex queries requiring multiple agents/analyses
        - greeting: Hello, hi, hey type messages
        - help: Questions about what the chatbot can do
        - general: Anything else that doesn't fit above categories
        """

        query: str = dspy.InputField(desc="User's query text")
        conversation_context: str = dspy.InputField(
            desc="Recent conversation history (last 2-3 messages) for context",
            default="",
        )
        brand_context: str = dspy.InputField(
            desc="Current brand filter (e.g., Kisqali, Remibrutinib, Fabhalta)",
            default="",
        )

        intent: str = dspy.OutputField(
            desc="Primary intent classification. Must be one of: kpi_query, causal_analysis, agent_status, recommendation, search, multi_faceted, greeting, help, general"
        )
        confidence: float = dspy.OutputField(
            desc="Confidence score between 0.0 and 1.0 for the classification"
        )
        reasoning: str = dspy.OutputField(
            desc="Brief explanation of why this intent was chosen"
        )

    class AgentRoutingSignature(dspy.Signature):
        """
        Route user queries to the appropriate E2I tier agent.

        The E2I system has 18 agents organized in 6 tiers:
        - Tier 0: ML Foundation (scope_definer, data_preparer, feature_analyzer, model_selector, model_trainer, model_deployer, observability_connector)
        - Tier 1: Orchestration (orchestrator, tool_composer)
        - Tier 2: Causal Analytics (causal_impact, gap_analyzer, heterogeneous_optimizer)
        - Tier 3: Monitoring (drift_monitor, experiment_designer, health_score)
        - Tier 4: Predictions (prediction_synthesizer, resource_optimizer)
        - Tier 5: Learning (explainer, feedback_learner)

        Agent Capabilities:
        - causal_impact: Analyzes WHY metrics changed, identifies causal drivers and effects
        - gap_analyzer: Finds ROI opportunities, underperforming segments, growth potential
        - heterogeneous_optimizer: Segment-level analysis, CATE estimation, heterogeneous effects
        - drift_monitor: Detects data drift, model drift, anomalies, distribution shifts
        - experiment_designer: Designs A/B tests, experiments, trials, hypothesis testing
        - health_score: System health metrics, performance scores, status checks
        - prediction_synthesizer: Future predictions, forecasts, trend projections
        - resource_optimizer: Resource allocation, optimization recommendations
        - explainer: Natural language explanations, summaries, interpretations
        - feedback_learner: Learns from feedback, improves over time
        """

        query: str = dspy.InputField(desc="User's query to route to an agent")
        intent: str = dspy.InputField(
            desc="Classified intent of the query (kpi_query, causal_analysis, etc.)",
            default="",
        )
        brand_context: str = dspy.InputField(
            desc="Current brand filter (Kisqali, Remibrutinib, Fabhalta)",
            default="",
        )

        primary_agent: str = dspy.OutputField(
            desc="Primary agent to route to. Must be one of: causal_impact, gap_analyzer, heterogeneous_optimizer, drift_monitor, experiment_designer, health_score, prediction_synthesizer, resource_optimizer, explainer, feedback_learner"
        )
        secondary_agents: str = dspy.OutputField(
            desc="Comma-separated list of secondary agents that may be helpful (can be empty)"
        )
        routing_confidence: float = dspy.OutputField(
            desc="Confidence score between 0.0 and 1.0 for the routing decision"
        )
        rationale: str = dspy.OutputField(
            desc="Brief explanation of why this agent was chosen"
        )


# =============================================================================
# TRAINING SIGNAL COLLECTION
# =============================================================================


@dataclass
class IntentTrainingSignal:
    """
    Training signal for intent classification optimization.

    Captures classification attempts with their outcomes for
    feedback_learner to optimize DSPy prompts.
    """

    query: str
    conversation_context: str
    brand_context: str
    predicted_intent: str
    confidence: float
    reasoning: str
    classification_method: str  # "dspy" or "hardcoded"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Reward signals (populated after interaction)
    user_followed_up: Optional[bool] = None  # Did user ask follow-up?
    response_helpful: Optional[bool] = None  # User feedback if available
    correct_routing: Optional[bool] = None  # Did routing work correctly?

    def compute_reward(self) -> float:
        """
        Compute reward score for this classification signal.

        Returns:
            Reward between 0.0 and 1.0
        """
        # Base reward from confidence (if classification succeeded)
        reward = self.confidence * 0.3

        # Reward for correct routing (if known)
        if self.correct_routing is True:
            reward += 0.4
        elif self.correct_routing is False:
            reward -= 0.2

        # Reward for helpful response (if user feedback available)
        if self.response_helpful is True:
            reward += 0.3
        elif self.response_helpful is False:
            reward -= 0.1

        # Penalty for low confidence (indicates uncertain classification)
        if self.confidence < 0.5:
            reward -= 0.1

        return max(0.0, min(1.0, reward))


class IntentTrainingSignalCollector:
    """
    Collects training signals for DSPy intent classification optimization.

    Maintains a buffer of recent signals and provides methods to
    submit them for training.
    """

    def __init__(self, buffer_size: int = 1000):
        self._buffer: List[IntentTrainingSignal] = []
        self._buffer_size = buffer_size

    def add_signal(self, signal: IntentTrainingSignal) -> None:
        """Add a training signal to the buffer."""
        self._buffer.append(signal)
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size :]

    def get_signals(self, limit: int = 100) -> List[IntentTrainingSignal]:
        """Get recent signals from buffer."""
        return self._buffer[-limit:]

    def get_high_quality_signals(
        self, min_confidence: float = 0.7, limit: int = 50
    ) -> List[IntentTrainingSignal]:
        """Get high-confidence signals for training."""
        return [s for s in self._buffer if s.confidence >= min_confidence][-limit:]

    def clear(self) -> None:
        """Clear the signal buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# Global signal collector instance
_signal_collector: Optional[IntentTrainingSignalCollector] = None


def get_signal_collector() -> IntentTrainingSignalCollector:
    """Get the global signal collector instance."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = IntentTrainingSignalCollector()
    return _signal_collector


# =============================================================================
# DSPY MODULE
# =============================================================================

if DSPY_AVAILABLE:

    class ChatbotIntentClassifier(dspy.Module):
        """
        DSPy module for classifying chatbot query intents.

        Uses Chain of Thought reasoning to improve classification accuracy.
        """

        def __init__(self):
            super().__init__()
            self.classifier = dspy.ChainOfThought(ChatbotIntentClassificationSignature)

        def forward(
            self,
            query: str,
            conversation_context: str = "",
            brand_context: str = "",
        ) -> dspy.Prediction:
            """
            Classify the intent of a user query.

            Args:
                query: User's query text
                conversation_context: Recent conversation history
                brand_context: Current brand filter

            Returns:
                DSPy Prediction with intent, confidence, reasoning
            """
            result = self.classifier(
                query=query,
                conversation_context=conversation_context,
                brand_context=brand_context,
            )
            return result

    class ChatbotAgentRouter(dspy.Module):
        """
        DSPy module for routing queries to appropriate E2I tier agents.

        Uses Chain of Thought reasoning to select the best agent based on
        query content, intent, and context.
        """

        def __init__(self):
            super().__init__()
            self.router = dspy.ChainOfThought(AgentRoutingSignature)

        def forward(
            self,
            query: str,
            intent: str = "",
            brand_context: str = "",
        ) -> dspy.Prediction:
            """
            Route a query to the appropriate agent.

            Args:
                query: User's query text
                intent: Classified intent of the query
                brand_context: Current brand filter

            Returns:
                DSPy Prediction with primary_agent, secondary_agents,
                routing_confidence, rationale
            """
            result = self.router(
                query=query,
                intent=intent,
                brand_context=brand_context,
            )
            return result


# =============================================================================
# HARDCODED FALLBACK
# =============================================================================


def _matches_pattern(query_lower: str, patterns: list[str]) -> bool:
    """
    Check if query matches any pattern using word boundaries.

    For multi-word patterns, checks substring match.
    For single-word patterns, checks word boundary match.
    """
    for pattern in patterns:
        if " " in pattern:
            if pattern in query_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(pattern)}\b", query_lower):
                return True
    return False


def _is_multi_faceted_query(query: str) -> bool:
    """
    Detect if query needs Tool Composer for multi-faceted processing.

    Multi-faceted queries require aggregating results from multiple agents
    or analyzing multiple aspects of the same question.
    """
    query_lower = query.lower()

    facets = {
        # Query contains comparative/conjunction keywords suggesting multiple questions
        "conjunction_keywords": any(
            w in query_lower
            for w in ["compare", "trends", "explain", "also", "and then", "both"]
        ),
        # Query mentions multiple KPIs
        "multiple_kpis": len(
            re.findall(
                r"(trx|nrx|market share|conversion|volume|patient starts)", query_lower
            )
        )
        > 1,
        # Query spans cross-agent capabilities
        "cross_agent": any(
            w in query_lower
            for w in ["drift", "health", "causal", "experiment", "prediction"]
        ),
        # Query mentions multiple brands
        "multiple_brands": len(
            re.findall(r"(kisqali|fabhalta|remibrutinib|all brands)", query_lower)
        )
        > 1,
        # Query asks for both analysis AND recommendations
        "analysis_and_recommendation": (
            "why" in query_lower or "what caused" in query_lower
        )
        and any(w in query_lower for w in ["recommend", "suggest", "should"]),
    }

    return sum(facets.values()) >= 2


def classify_intent_hardcoded(query: str) -> Tuple[str, float, str]:
    """
    Classify intent using hardcoded patterns (fallback).

    Returns:
        Tuple of (intent, confidence, reasoning)
    """
    query_lower = query.lower()

    # Greeting patterns
    if _matches_pattern(
        query_lower, ["hello", "hi", "hey", "good morning", "good afternoon"]
    ):
        return (IntentType.GREETING, 0.95, "Matched greeting pattern")

    # Help patterns
    if _matches_pattern(query_lower, ["help", "what can you", "how do i", "guide me"]):
        return (IntentType.HELP, 0.95, "Matched help request pattern")

    # Multi-faceted check (before individual intents)
    if _is_multi_faceted_query(query):
        return (IntentType.MULTI_FACETED, 0.85, "Complex query requiring multiple analyses")

    # KPI patterns
    if _matches_pattern(
        query_lower,
        ["kpi", "trx", "nrx", "market share", "conversion", "metric", "volume"],
    ):
        return (IntentType.KPI_QUERY, 0.90, "Matched KPI keyword pattern")

    # Causal patterns
    if _matches_pattern(
        query_lower,
        ["why", "cause", "caused", "impact", "effect", "driver", "causal", "because"],
    ):
        return (IntentType.CAUSAL_ANALYSIS, 0.90, "Matched causal analysis pattern")

    # Agent patterns
    if _matches_pattern(
        query_lower, ["agent", "tier", "orchestrator", "status", "system"]
    ):
        return (IntentType.AGENT_STATUS, 0.90, "Matched agent status pattern")

    # Recommendation patterns
    if _matches_pattern(
        query_lower, ["recommend", "suggest", "improve", "optimize", "strategy"]
    ):
        return (IntentType.RECOMMENDATION, 0.90, "Matched recommendation pattern")

    # Search patterns
    if _matches_pattern(
        query_lower, ["search", "find", "look for", "show me", "trend"]
    ):
        return (IntentType.SEARCH, 0.85, "Matched search pattern")

    return (IntentType.GENERAL, 0.70, "No specific pattern matched - defaulting to general")


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

# Cached DSPy classifier instance
_dspy_classifier: Optional["ChatbotIntentClassifier"] = None


def _get_dspy_classifier() -> Optional["ChatbotIntentClassifier"]:
    """Get or create the DSPy classifier instance."""
    global _dspy_classifier
    if not DSPY_AVAILABLE or not CHATBOT_DSPY_INTENT_ENABLED:
        return None

    # Ensure DSPy LLM is configured before creating classifier
    _ensure_dspy_configured()

    if _dspy_classifier is None:
        try:
            _dspy_classifier = ChatbotIntentClassifier()
            logger.info("Initialized DSPy ChatbotIntentClassifier")
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy classifier: {e}")
            return None
    return _dspy_classifier


# Valid intent values for validation
VALID_INTENTS = {
    IntentType.KPI_QUERY,
    IntentType.CAUSAL_ANALYSIS,
    IntentType.AGENT_STATUS,
    IntentType.RECOMMENDATION,
    IntentType.SEARCH,
    IntentType.MULTI_FACETED,
    IntentType.GREETING,
    IntentType.HELP,
    IntentType.GENERAL,
}


def _normalize_intent(intent: str) -> str:
    """Normalize intent string to valid IntentType value."""
    intent_lower = intent.lower().strip()

    # Direct mapping
    if intent_lower in VALID_INTENTS:
        return intent_lower

    # Handle variations
    mappings = {
        "kpi": IntentType.KPI_QUERY,
        "kpi query": IntentType.KPI_QUERY,
        "causal": IntentType.CAUSAL_ANALYSIS,
        "causal analysis": IntentType.CAUSAL_ANALYSIS,
        "agent": IntentType.AGENT_STATUS,
        "agent status": IntentType.AGENT_STATUS,
        "recommendations": IntentType.RECOMMENDATION,
        "multi-faceted": IntentType.MULTI_FACETED,
        "multifaceted": IntentType.MULTI_FACETED,
        "complex": IntentType.MULTI_FACETED,
        "greetings": IntentType.GREETING,
        "hi": IntentType.GREETING,
        "hello": IntentType.GREETING,
    }

    return mappings.get(intent_lower, IntentType.GENERAL)


def _validate_confidence(confidence: Any) -> float:
    """Validate and normalize confidence score."""
    try:
        conf = float(confidence)
        return max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        return 0.5  # Default confidence if parsing fails


async def classify_intent_dspy(
    query: str,
    conversation_context: str = "",
    brand_context: str = "",
    collect_signal: bool = True,
) -> Tuple[str, float, str, str]:
    """
    Classify intent using DSPy with fallback to hardcoded patterns.

    Args:
        query: User's query text
        conversation_context: Recent conversation history
        brand_context: Current brand filter
        collect_signal: Whether to collect training signal

    Returns:
        Tuple of (intent, confidence, reasoning, classification_method)
    """
    classifier = _get_dspy_classifier()
    classification_method = "dspy"
    intent = None
    confidence = 0.0
    reasoning = ""

    if classifier is not None:
        try:
            # Run DSPy classification
            result = classifier(
                query=query,
                conversation_context=conversation_context,
                brand_context=brand_context,
            )

            # Extract and validate results
            intent = _normalize_intent(str(result.intent))
            confidence = _validate_confidence(result.confidence)
            reasoning = str(getattr(result, "reasoning", "DSPy classification"))

            logger.debug(
                f"DSPy classified '{query[:50]}...' as {intent} "
                f"(confidence={confidence:.2f})"
            )

        except Exception as e:
            logger.warning(f"DSPy classification failed, using fallback: {e}")
            classifier = None  # Trigger fallback

    # Fallback to hardcoded if DSPy unavailable or failed
    if classifier is None:
        intent, confidence, reasoning = classify_intent_hardcoded(query)
        classification_method = "hardcoded"
        logger.debug(f"Hardcoded classified '{query[:50]}...' as {intent}")

    # Collect training signal
    if collect_signal:
        signal = IntentTrainingSignal(
            query=query,
            conversation_context=conversation_context,
            brand_context=brand_context,
            predicted_intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            classification_method=classification_method,
        )
        get_signal_collector().add_signal(signal)

    return (intent, confidence, reasoning, classification_method)


# =============================================================================
# SYNCHRONOUS WRAPPER (for compatibility)
# =============================================================================


def classify_intent_sync(
    query: str,
    conversation_context: str = "",
    brand_context: str = "",
) -> Tuple[str, float, str, str]:
    """
    Synchronous wrapper for intent classification.

    Uses hardcoded patterns only (DSPy requires async for proper LLM calls).
    Use classify_intent_dspy for full DSPy support.

    Returns:
        Tuple of (intent, confidence, reasoning, classification_method)
    """
    intent, confidence, reasoning = classify_intent_hardcoded(query)
    return (intent, confidence, reasoning, "hardcoded")


# =============================================================================
# AGENT ROUTING (PHASE 4)
# =============================================================================

# Feature flag for DSPy agent routing
CHATBOT_DSPY_ROUTING_ENABLED = os.getenv("CHATBOT_DSPY_ROUTING", "true").lower() == "true"

# Valid agents for routing
VALID_AGENTS = {
    "causal_impact",
    "gap_analyzer",
    "heterogeneous_optimizer",
    "drift_monitor",
    "experiment_designer",
    "health_score",
    "prediction_synthesizer",
    "resource_optimizer",
    "explainer",
    "feedback_learner",
}

# Agent capability keywords for fallback routing
AGENT_CAPABILITIES = {
    "causal_impact": ["why", "cause", "caused", "effect", "impact", "driver", "factor", "causal"],
    "gap_analyzer": ["gap", "opportunity", "roi", "underperforming", "potential", "growth"],
    "heterogeneous_optimizer": ["segment", "cate", "heterogeneous", "subgroup", "cohort"],
    "drift_monitor": ["drift", "change", "shift", "anomaly", "deviation", "distribution"],
    "experiment_designer": ["test", "experiment", "a/b", "trial", "hypothesis", "design"],
    "health_score": ["health", "status", "score", "performance", "metric", "check"],
    "prediction_synthesizer": ["predict", "forecast", "future", "trend", "projection", "model"],
    "resource_optimizer": ["allocate", "resource", "optimize", "budget", "efficiency"],
    "explainer": ["explain", "summarize", "interpret", "describe", "understand"],
    "feedback_learner": ["feedback", "improve", "learn", "optimize", "tune"],
}


def _normalize_agent(agent: str) -> str:
    """Normalize agent string to valid agent name."""
    agent_lower = agent.lower().strip().replace(" ", "_").replace("-", "_")

    # Direct match
    if agent_lower in VALID_AGENTS:
        return agent_lower

    # Handle common variations
    mappings = {
        "causal": "causal_impact",
        "causal_analysis": "causal_impact",
        "gap": "gap_analyzer",
        "drift": "drift_monitor",
        "experiment": "experiment_designer",
        "health": "health_score",
        "prediction": "prediction_synthesizer",
        "predict": "prediction_synthesizer",
        "resource": "resource_optimizer",
        "explain": "explainer",
        "feedback": "feedback_learner",
    }

    return mappings.get(agent_lower, "explainer")


def route_agent_hardcoded(
    query: str,
    intent: str = "",
) -> Tuple[str, List[str], float, str]:
    """
    Route query to agent using keyword matching (fallback).

    Returns:
        Tuple of (primary_agent, secondary_agents, confidence, rationale)
    """
    query_lower = query.lower()
    scores: Dict[str, int] = {}

    # Score each agent based on keyword matches
    for agent, keywords in AGENT_CAPABILITIES.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[agent] = score

    # Intent-based boosting
    if intent == IntentType.CAUSAL_ANALYSIS:
        scores["causal_impact"] = scores.get("causal_impact", 0) + 2
    elif intent == IntentType.KPI_QUERY:
        scores["health_score"] = scores.get("health_score", 0) + 1
    elif intent == IntentType.RECOMMENDATION:
        scores["gap_analyzer"] = scores.get("gap_analyzer", 0) + 1

    if scores:
        # Sort by score descending
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_agent = sorted_agents[0][0]
        primary_score = sorted_agents[0][1]

        # Get secondary agents with scores > 0
        secondary_agents = [a for a, s in sorted_agents[1:3] if s > 0]

        # Calculate confidence based on score distribution
        total_score = sum(scores.values())
        confidence = min(0.95, 0.5 + (primary_score / max(total_score, 1)) * 0.4)

        return (
            primary_agent,
            secondary_agents,
            confidence,
            f"Keyword match (score: {primary_score}, total: {total_score})",
        )

    # Default to explainer for general queries
    return (
        "explainer",
        [],
        0.6,
        "Default routing (no specific keywords matched)",
    )


@dataclass
class RoutingTrainingSignal:
    """
    Training signal for agent routing optimization.

    Captures routing attempts with their outcomes for
    feedback_learner to optimize DSPy prompts.
    """

    query: str
    intent: str
    brand_context: str
    predicted_agent: str
    secondary_agents: List[str]
    confidence: float
    rationale: str
    routing_method: str  # "dspy" or "hardcoded"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Reward signals (populated after interaction)
    correct_agent: Optional[bool] = None  # Was this the right agent?
    response_quality: Optional[float] = None  # Quality of agent response (0-1)

    def compute_reward(self) -> float:
        """Compute reward score for this routing signal."""
        reward = self.confidence * 0.3

        if self.correct_agent is True:
            reward += 0.5
        elif self.correct_agent is False:
            reward -= 0.2

        if self.response_quality is not None:
            reward += self.response_quality * 0.2

        return max(0.0, min(1.0, reward))


# Routing signal collector
class RoutingTrainingSignalCollector:
    """Collects training signals for agent routing optimization."""

    def __init__(self, buffer_size: int = 1000):
        self._buffer: List[RoutingTrainingSignal] = []
        self._buffer_size = buffer_size

    def add_signal(self, signal: RoutingTrainingSignal) -> None:
        """Add a routing signal to the buffer."""
        self._buffer.append(signal)
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size:]

    def get_signals(self, limit: int = 100) -> List[RoutingTrainingSignal]:
        """Get recent signals from buffer."""
        return self._buffer[-limit:]

    def clear(self) -> None:
        """Clear the signal buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# Global routing signal collector
_routing_signal_collector: Optional[RoutingTrainingSignalCollector] = None


def get_routing_signal_collector() -> RoutingTrainingSignalCollector:
    """Get the global routing signal collector instance."""
    global _routing_signal_collector
    if _routing_signal_collector is None:
        _routing_signal_collector = RoutingTrainingSignalCollector()
    return _routing_signal_collector


# Cached DSPy router instance
_dspy_router: Optional["ChatbotAgentRouter"] = None


def _get_dspy_router() -> Optional["ChatbotAgentRouter"]:
    """Get or create the DSPy router instance."""
    global _dspy_router
    if not DSPY_AVAILABLE or not CHATBOT_DSPY_ROUTING_ENABLED:
        return None

    # Ensure DSPy LLM is configured before creating router
    _ensure_dspy_configured()

    if _dspy_router is None:
        try:
            _dspy_router = ChatbotAgentRouter()
            logger.info("Initialized DSPy ChatbotAgentRouter")
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy router: {e}")
            return None
    return _dspy_router


async def route_agent_dspy(
    query: str,
    intent: str = "",
    brand_context: str = "",
    collect_signal: bool = True,
) -> Tuple[str, List[str], float, str, str]:
    """
    Route query to agent using DSPy with fallback to keyword matching.

    Args:
        query: User's query text
        intent: Classified intent of the query
        brand_context: Current brand filter
        collect_signal: Whether to collect training signal

    Returns:
        Tuple of (primary_agent, secondary_agents, confidence, rationale, routing_method)
    """
    router = _get_dspy_router()
    routing_method = "dspy"
    primary_agent = None
    secondary_agents: List[str] = []
    confidence = 0.0
    rationale = ""

    if router is not None:
        try:
            # Run DSPy routing
            result = router(
                query=query,
                intent=intent,
                brand_context=brand_context,
            )

            # Extract and validate results
            primary_agent = _normalize_agent(str(result.primary_agent))

            # Parse secondary agents (comma-separated)
            secondary_str = str(getattr(result, "secondary_agents", ""))
            if secondary_str and secondary_str.lower() not in ("", "none", "empty"):
                secondary_agents = [
                    _normalize_agent(a.strip())
                    for a in secondary_str.split(",")
                    if a.strip() and a.strip().lower() not in ("none", "empty")
                ]
                # Remove primary from secondary if present
                secondary_agents = [a for a in secondary_agents if a != primary_agent]

            confidence = _validate_confidence(result.routing_confidence)
            rationale = str(getattr(result, "rationale", "DSPy routing"))

            logger.debug(
                f"DSPy routed '{query[:50]}...' to {primary_agent} "
                f"(confidence={confidence:.2f})"
            )

        except Exception as e:
            logger.warning(f"DSPy routing failed, using fallback: {e}")
            router = None  # Trigger fallback

    # Fallback to hardcoded if DSPy unavailable or failed
    if router is None:
        primary_agent, secondary_agents, confidence, rationale = route_agent_hardcoded(
            query, intent
        )
        routing_method = "hardcoded"
        logger.debug(f"Hardcoded routed '{query[:50]}...' to {primary_agent}")

    # Collect training signal
    if collect_signal:
        signal = RoutingTrainingSignal(
            query=query,
            intent=intent,
            brand_context=brand_context,
            predicted_agent=primary_agent,
            secondary_agents=secondary_agents,
            confidence=confidence,
            rationale=rationale,
            routing_method=routing_method,
        )
        get_routing_signal_collector().add_signal(signal)

    return (primary_agent, secondary_agents, confidence, rationale, routing_method)


def route_agent_sync(
    query: str,
    intent: str = "",
) -> Tuple[str, List[str], float, str, str]:
    """
    Synchronous wrapper for agent routing.

    Uses hardcoded patterns only (DSPy requires async for proper LLM calls).
    Use route_agent_dspy for full DSPy support.

    Returns:
        Tuple of (primary_agent, secondary_agents, confidence, rationale, routing_method)
    """
    primary_agent, secondary_agents, confidence, rationale = route_agent_hardcoded(
        query, intent
    )
    return (primary_agent, secondary_agents, confidence, rationale, "hardcoded")


# =============================================================================
# COGNITIVE RAG (PHASE 5)
# =============================================================================

# Feature flag for cognitive RAG
CHATBOT_COGNITIVE_RAG_ENABLED = os.getenv("CHATBOT_COGNITIVE_RAG", "true").lower() == "true"

# E2I domain vocabulary for query rewriting
E2I_DOMAIN_VOCABULARY = """
Brands: Kisqali, Remibrutinib, Fabhalta
Regions: Northeast, Southeast, Midwest, West, Southwest
KPIs: TRx, NRx, Market Share, Conversion Rate, Volume, Growth Rate
HCP Types: Oncologist, Rheumatologist, Dermatologist, Hematologist
Patient Stages: Diagnosis, Treatment, Maintenance, Follow-up
Time References: MTD, QTD, YTD, Last Quarter, Last Month, Last Week
"""

# Memory types for multi-hop retrieval
MEMORY_TYPES = ["episodic", "semantic", "procedural"]


if DSPY_AVAILABLE:

    class QueryRewriteSignature(dspy.Signature):
        """
        Rewrite user query for optimal retrieval across memory stores.
        Pharmaceutical domain-aware query expansion for E2I analytics.
        """

        original_query: str = dspy.InputField(desc="The user's original natural language question")
        conversation_context: str = dspy.InputField(
            desc="Recent conversation history for context", default=""
        )
        domain_vocabulary: str = dspy.InputField(
            desc="Available domain terms: brands, regions, KPIs, HCP types"
        )

        rewritten_query: str = dspy.OutputField(
            desc="Optimized query for hybrid retrieval (semantic + sparse + graph)"
        )
        search_keywords: str = dspy.OutputField(
            desc="Comma-separated key terms for full-text search"
        )
        graph_entities: str = dspy.OutputField(
            desc="Comma-separated entities to anchor graph traversal"
        )

    class HopDecisionSignature(dspy.Signature):
        """
        Decide the next retrieval hop based on accumulated evidence.
        Multi-hop investigation for comprehensive context gathering.
        """

        investigation_goal: str = dspy.InputField(desc="What we're trying to discover")
        current_evidence: str = dspy.InputField(desc="Evidence collected so far (JSON)")
        hop_number: int = dspy.InputField(desc="Current hop number (1-4)")
        available_memories: str = dspy.InputField(
            desc="Comma-separated memory types not yet queried"
        )

        next_memory: str = dspy.OutputField(
            desc="Next memory type to query: episodic | semantic | procedural | STOP"
        )
        retrieval_query: str = dspy.OutputField(desc="Specific query for the next memory store")
        reasoning: str = dspy.OutputField(desc="Why this hop is needed or why to stop")
        confidence: float = dspy.OutputField(
            desc="Confidence that more evidence is needed (0.0-1.0, low=sufficient evidence)"
        )

    class EvidenceRelevanceSignature(dspy.Signature):
        """
        Score retrieved evidence for relevance to investigation goal.
        Filters noise and ranks evidence quality for pharmaceutical analytics.
        """

        investigation_goal: str = dspy.InputField(desc="What we're trying to discover")
        evidence_item: str = dspy.InputField(desc="A single piece of retrieved evidence")
        source_memory: str = dspy.InputField(desc="Which memory store this came from")

        relevance_score: float = dspy.OutputField(desc="Relevance score 0.0-1.0")
        key_insight: str = dspy.OutputField(desc="The key insight this evidence provides")
        follow_up_needed: bool = dspy.OutputField(
            desc="Whether this evidence suggests follow-up queries"
        )

    class ChatbotQueryRewriter(dspy.Module):
        """DSPy module for query rewriting in cognitive RAG."""

        def __init__(self):
            super().__init__()
            self.rewrite = dspy.ChainOfThought(QueryRewriteSignature)

        def forward(
            self,
            original_query: str,
            conversation_context: str = "",
            domain_vocabulary: str = E2I_DOMAIN_VOCABULARY,
        ):
            return self.rewrite(
                original_query=original_query,
                conversation_context=conversation_context,
                domain_vocabulary=domain_vocabulary,
            )

    class ChatbotHopDecider(dspy.Module):
        """DSPy module for multi-hop retrieval decisions."""

        def __init__(self):
            super().__init__()
            self.decide = dspy.ChainOfThought(HopDecisionSignature)

        def forward(
            self,
            investigation_goal: str,
            current_evidence: str,
            hop_number: int,
            available_memories: str,
        ):
            return self.decide(
                investigation_goal=investigation_goal,
                current_evidence=current_evidence,
                hop_number=hop_number,
                available_memories=available_memories,
            )

    class ChatbotEvidenceScorer(dspy.Module):
        """DSPy module for evidence relevance scoring."""

        def __init__(self):
            super().__init__()
            self.score = dspy.Predict(EvidenceRelevanceSignature)

        def forward(
            self,
            investigation_goal: str,
            evidence_item: str,
            source_memory: str,
        ):
            return self.score(
                investigation_goal=investigation_goal,
                evidence_item=evidence_item,
                source_memory=source_memory,
            )


@dataclass
class RAGTrainingSignal:
    """
    Training signal for cognitive RAG optimization.

    Captures RAG retrieval outcomes for feedback_learner optimization.
    """

    query: str
    rewritten_query: str
    search_keywords: List[str]
    graph_entities: List[str]
    evidence_count: int
    hop_count: int
    avg_relevance_score: float
    retrieval_method: str  # "cognitive" or "basic"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Outcome signals
    response_quality: Optional[float] = None
    user_feedback: Optional[str] = None

    def compute_reward(self) -> float:
        """Compute reward score for this RAG signal."""
        reward = 0.0

        # Evidence quality rewards
        if self.evidence_count > 0:
            reward += min(0.3, self.evidence_count * 0.1)

        # Relevance rewards
        reward += self.avg_relevance_score * 0.3

        # Hop efficiency (fewer hops for good results = better)
        if self.evidence_count >= 3 and self.hop_count <= 2:
            reward += 0.2

        # User feedback
        if self.response_quality is not None:
            reward += self.response_quality * 0.2

        return max(0.0, min(1.0, reward))


class RAGTrainingSignalCollector:
    """Collects training signals for cognitive RAG optimization."""

    def __init__(self, buffer_size: int = 1000):
        self._buffer: List[RAGTrainingSignal] = []
        self._buffer_size = buffer_size

    def add_signal(self, signal: RAGTrainingSignal) -> None:
        """Add a RAG signal to the buffer."""
        self._buffer.append(signal)
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size:]

    def get_signals(self, limit: int = 100) -> List[RAGTrainingSignal]:
        """Get recent signals from buffer."""
        return self._buffer[-limit:]

    def clear(self) -> None:
        """Clear the signal buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# Global RAG signal collector
_rag_signal_collector: Optional[RAGTrainingSignalCollector] = None


def get_rag_signal_collector() -> RAGTrainingSignalCollector:
    """Get the global RAG signal collector instance."""
    global _rag_signal_collector
    if _rag_signal_collector is None:
        _rag_signal_collector = RAGTrainingSignalCollector()
    return _rag_signal_collector


# Cached DSPy query rewriter instance
_dspy_query_rewriter: Optional["ChatbotQueryRewriter"] = None


def _get_dspy_query_rewriter() -> Optional["ChatbotQueryRewriter"]:
    """Get or create the DSPy query rewriter instance."""
    global _dspy_query_rewriter
    if not DSPY_AVAILABLE or not CHATBOT_COGNITIVE_RAG_ENABLED:
        return None

    # Ensure DSPy LLM is configured before creating query rewriter
    _ensure_dspy_configured()

    if _dspy_query_rewriter is None:
        try:
            _dspy_query_rewriter = ChatbotQueryRewriter()
            logger.info("Initialized DSPy ChatbotQueryRewriter")
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy query rewriter: {e}")
            return None
    return _dspy_query_rewriter


def rewrite_query_hardcoded(
    query: str,
    brand_context: str = "",
) -> Tuple[str, List[str], List[str]]:
    """
    Rewrite query using rule-based extraction (fallback).

    Returns:
        Tuple of (rewritten_query, search_keywords, graph_entities)
    """
    query_lower = query.lower()

    # Extract keywords based on patterns
    keywords = []
    entities = []

    # Brand extraction
    for brand in ["kisqali", "remibrutinib", "fabhalta"]:
        if brand in query_lower:
            keywords.append(brand)
            entities.append(brand.title())

    # Add brand context if not already in query
    if brand_context and brand_context.lower() not in query_lower:
        keywords.append(brand_context.lower())
        entities.append(brand_context)

    # KPI extraction
    kpis = ["trx", "nrx", "market share", "conversion", "volume", "growth"]
    for kpi in kpis:
        if kpi in query_lower:
            keywords.append(kpi)

    # Region extraction
    regions = ["northeast", "southeast", "midwest", "west", "southwest"]
    for region in regions:
        if region in query_lower:
            keywords.append(region)
            entities.append(region.title())

    # Time extraction
    times = ["mtd", "qtd", "ytd", "last quarter", "last month", "q1", "q2", "q3", "q4"]
    for time_ref in times:
        if time_ref in query_lower:
            keywords.append(time_ref)

    # If no keywords extracted, use important words from query
    if not keywords:
        # Remove common words and extract nouns/adjectives
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "why", "how",
            "when", "where", "who", "which", "in", "on", "at", "to", "for", "of",
            "and", "or", "but", "with", "from", "by", "about", "can", "could",
            "would", "should", "do", "does", "did", "have", "has", "had", "be",
            "been", "being", "this", "that", "these", "those", "my", "your", "our",
        }
        words = re.findall(r'\b[a-z]+\b', query_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2][:5]

    # Build rewritten query
    rewritten_parts = [query]
    if entities and entities[0] not in query:
        rewritten_parts.append(f"Context: {', '.join(entities)}")

    return (
        " ".join(rewritten_parts),
        keywords,
        entities,
    )


async def rewrite_query_dspy(
    query: str,
    conversation_context: str = "",
    brand_context: str = "",
    collect_signal: bool = False,
) -> Tuple[str, List[str], List[str], str]:
    """
    Rewrite query using DSPy with fallback to hardcoded extraction.

    Args:
        query: User's original query
        conversation_context: Recent conversation history
        brand_context: Current brand filter
        collect_signal: Whether to collect training signal

    Returns:
        Tuple of (rewritten_query, search_keywords, graph_entities, rewrite_method)
    """
    rewriter = _get_dspy_query_rewriter()
    rewrite_method = "dspy"
    rewritten_query = query
    search_keywords: List[str] = []
    graph_entities: List[str] = []

    if rewriter is not None:
        try:
            # Build context with brand
            full_context = conversation_context
            if brand_context:
                full_context = f"Brand: {brand_context}\n{conversation_context}"

            result = rewriter(
                original_query=query,
                conversation_context=full_context,
                domain_vocabulary=E2I_DOMAIN_VOCABULARY,
            )

            rewritten_query = str(result.rewritten_query)

            # Parse keywords (comma-separated)
            keywords_str = str(getattr(result, "search_keywords", ""))
            if keywords_str:
                search_keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

            # Parse entities (comma-separated)
            entities_str = str(getattr(result, "graph_entities", ""))
            if entities_str:
                graph_entities = [e.strip() for e in entities_str.split(",") if e.strip()]

            logger.debug(
                f"DSPy rewrote '{query[:30]}...' to '{rewritten_query[:50]}...' "
                f"(keywords: {len(search_keywords)}, entities: {len(graph_entities)})"
            )

        except Exception as e:
            logger.warning(f"DSPy query rewrite failed, using fallback: {e}")
            rewriter = None

    # Fallback to hardcoded
    if rewriter is None:
        rewritten_query, search_keywords, graph_entities = rewrite_query_hardcoded(
            query, brand_context
        )
        rewrite_method = "hardcoded"
        logger.debug(f"Hardcoded rewrote query (keywords: {search_keywords})")

    return (rewritten_query, search_keywords, graph_entities, rewrite_method)


async def score_evidence_dspy(
    investigation_goal: str,
    evidence_item: str,
    source_memory: str = "episodic",
) -> Tuple[float, str, bool]:
    """
    Score evidence relevance using DSPy (or fallback to heuristics).

    Returns:
        Tuple of (relevance_score, key_insight, follow_up_needed)
    """
    if not DSPY_AVAILABLE or not CHATBOT_COGNITIVE_RAG_ENABLED:
        # Heuristic scoring
        score = 0.5
        goal_words = set(investigation_goal.lower().split())
        evidence_words = set(evidence_item.lower().split())
        overlap = len(goal_words & evidence_words)
        score = min(1.0, 0.3 + overlap * 0.1)
        return (score, evidence_item[:100], overlap > 2)

    try:
        scorer = ChatbotEvidenceScorer()
        result = scorer(
            investigation_goal=investigation_goal,
            evidence_item=evidence_item,
            source_memory=source_memory,
        )
        return (
            _validate_confidence(result.relevance_score),
            str(result.key_insight),
            bool(result.follow_up_needed),
        )
    except Exception as e:
        logger.warning(f"DSPy evidence scoring failed: {e}")
        return (0.5, evidence_item[:100], False)


@dataclass
class CognitiveRAGResult:
    """Result of cognitive RAG retrieval."""

    rewritten_query: str
    search_keywords: List[str]
    graph_entities: List[str]
    evidence: List[Dict[str, Any]]
    hop_count: int
    avg_relevance_score: float
    retrieval_method: str  # "cognitive" or "basic"


async def cognitive_rag_retrieve(
    query: str,
    conversation_context: str = "",
    brand_context: str = "",
    intent: str = "",
    k: int = 5,
    enable_multi_hop: bool = False,
    collect_signal: bool = True,
) -> CognitiveRAGResult:
    """
    Perform cognitive RAG retrieval with query rewriting and optional multi-hop.

    This is the main entry point for Phase 5 cognitive RAG integration.

    Args:
        query: User's query
        conversation_context: Recent conversation history
        brand_context: Current brand filter
        intent: Classified intent (for retrieval strategy)
        k: Number of results to return
        enable_multi_hop: Whether to use multi-hop retrieval (slower but more comprehensive)
        collect_signal: Whether to collect training signal

    Returns:
        CognitiveRAGResult with rewritten query, evidence, and metadata
    """
    # Step 1: Rewrite query for better retrieval
    rewritten_query, search_keywords, graph_entities, rewrite_method = await rewrite_query_dspy(
        query=query,
        conversation_context=conversation_context,
        brand_context=brand_context,
    )

    evidence: List[Dict[str, Any]] = []
    hop_count = 1
    relevance_scores: List[float] = []

    # Step 2: Execute retrieval (import here to avoid circular imports)
    try:
        from src.rag.retriever import hybrid_search

        # Use rewritten query for hybrid search
        results = await hybrid_search(
            query=rewritten_query,
            k=k,
            kpi_name=None,  # Will be extracted from keywords if needed
            filters={"brand": brand_context} if brand_context else None,
        )

        # Step 3: Score and filter evidence
        for r in results:
            score, insight, _ = await score_evidence_dspy(
                investigation_goal=f"Answer: {query}",
                evidence_item=r.content[:500],
                source_memory="episodic",
            )

            relevance_scores.append(score)

            if score >= 0.3:  # Minimum relevance threshold
                evidence.append({
                    "source_id": r.source_id,
                    "content": r.content[:500],
                    "score": r.score,
                    "relevance_score": score,
                    "key_insight": insight,
                    "source": r.source,
                })

    except Exception as e:
        logger.error(f"Cognitive RAG retrieval failed: {e}")
        # Return empty result on failure
        return CognitiveRAGResult(
            rewritten_query=query,
            search_keywords=[],
            graph_entities=[],
            evidence=[],
            hop_count=0,
            avg_relevance_score=0.0,
            retrieval_method="failed",
        )

    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    retrieval_method = "cognitive" if rewrite_method == "dspy" else "basic"

    # Collect training signal
    if collect_signal:
        signal = RAGTrainingSignal(
            query=query,
            rewritten_query=rewritten_query,
            search_keywords=search_keywords,
            graph_entities=graph_entities,
            evidence_count=len(evidence),
            hop_count=hop_count,
            avg_relevance_score=avg_relevance,
            retrieval_method=retrieval_method,
        )
        get_rag_signal_collector().add_signal(signal)

    return CognitiveRAGResult(
        rewritten_query=rewritten_query,
        search_keywords=search_keywords,
        graph_entities=graph_entities,
        evidence=evidence,
        hop_count=hop_count,
        avg_relevance_score=avg_relevance,
        retrieval_method=retrieval_method,
    )


# =============================================================================
# PHASE 6: EVIDENCE SYNTHESIS DSPY
# =============================================================================

# Feature flag for DSPy evidence synthesis
CHATBOT_DSPY_SYNTHESIS_ENABLED = os.getenv("CHATBOT_DSPY_SYNTHESIS", "true").lower() == "true"


# DSPy Signatures for Evidence Synthesis
if DSPY_AVAILABLE:

    class EvidenceSynthesisSignature(dspy.Signature):
        """
        Synthesize a response from retrieved evidence for pharmaceutical analytics queries.

        Given a user query and retrieved evidence from the RAG system, produce a
        well-structured, accurate response that:
        - Directly addresses the user's question
        - Cites evidence sources appropriately
        - Provides confidence levels for claims
        - Is grounded in the retrieved evidence
        - Maintains focus on commercial pharma analytics (not clinical)

        The response should be informative, actionable, and suitable for business decision-making.
        """

        query: str = dspy.InputField(desc="User's original question")
        intent: str = dspy.InputField(
            desc="Classified intent (kpi_query, causal_analysis, recommendation, etc.)",
            default="general",
        )
        evidence: str = dspy.InputField(
            desc="Retrieved evidence items formatted as source citations with content"
        )
        brand_context: str = dspy.InputField(
            desc="Brand filter context (Kisqali, Remibrutinib, Fabhalta)",
            default="",
        )
        conversation_context: str = dspy.InputField(
            desc="Recent conversation history for continuity",
            default="",
        )

        response: str = dspy.OutputField(
            desc="Well-structured response that addresses the query using the evidence"
        )
        confidence_statement: str = dspy.OutputField(
            desc="Brief statement about confidence level (e.g., 'High confidence based on 3 corroborating sources' or 'Moderate confidence - limited evidence available')"
        )
        evidence_citations: str = dspy.OutputField(
            desc="Comma-separated list of source_ids that were used to formulate the response"
        )
        follow_up_suggestions: str = dspy.OutputField(
            desc="1-2 suggested follow-up questions the user might want to ask",
            default="",
        )


# =============================================================================
# EVIDENCE SYNTHESIS DSPY MODULE
# =============================================================================

_chatbot_synthesizer = None


def _get_dspy_synthesizer():
    """Get or create the DSPy ChatbotSynthesizer singleton."""
    global _chatbot_synthesizer
    if _chatbot_synthesizer is None and DSPY_AVAILABLE:
        # Ensure DSPy LLM is configured before creating synthesizer
        _ensure_dspy_configured()
        try:
            _chatbot_synthesizer = dspy.ChainOfThought(EvidenceSynthesisSignature)
            logger.info("Initialized DSPy ChatbotSynthesizer module")
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy ChatbotSynthesizer: {e}")
            return None
    return _chatbot_synthesizer


# =============================================================================
# SYNTHESIS TRAINING SIGNAL COLLECTION
# =============================================================================

@dataclass
class SynthesisTrainingSignal:
    """Training signal for evidence synthesis optimization."""

    query: str
    intent: str
    evidence_count: int
    response_length: int
    confidence_level: str  # "high", "moderate", "low"
    citations_count: int
    synthesis_method: str  # "dspy" or "hardcoded"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Feedback for optimization (filled later)
    user_rating: Optional[float] = None  # 1-5 scale
    was_helpful: Optional[bool] = None
    had_hallucination: Optional[bool] = None

    def compute_reward(self) -> float:
        """Compute reward score for training optimization."""
        reward = 0.0

        # Base reward for completing synthesis
        reward += 0.2

        # Reward for using evidence (citations)
        if self.citations_count > 0:
            reward += min(0.3, self.citations_count * 0.1)  # Up to 0.3 for citations

        # Reward for appropriate response length (not too short, not too long)
        if 100 <= self.response_length <= 500:
            reward += 0.2
        elif 50 <= self.response_length <= 800:
            reward += 0.1

        # Reward for using DSPy (structured synthesis)
        if self.synthesis_method == "dspy":
            reward += 0.1

        # Penalty for hallucination
        if self.had_hallucination:
            reward -= 0.5

        # Reward for user feedback
        if self.user_rating is not None:
            reward += (self.user_rating - 3) * 0.1  # -0.2 to +0.2

        if self.was_helpful:
            reward += 0.2

        return max(0.0, min(1.0, reward))


class SynthesisTrainingSignalCollector:
    """Collects synthesis training signals for feedback_learner optimization."""

    def __init__(self, max_buffer_size: int = 1000):
        self._signals: List[SynthesisTrainingSignal] = []
        self._max_buffer_size = max_buffer_size

    def add_signal(self, signal: SynthesisTrainingSignal) -> None:
        """Add a training signal to the buffer."""
        self._signals.append(signal)
        if len(self._signals) > self._max_buffer_size:
            # Remove oldest signals
            self._signals = self._signals[-self._max_buffer_size:]
        logger.debug(f"Added synthesis training signal, buffer size: {len(self._signals)}")

    def get_signals(self, limit: int = 100) -> List[SynthesisTrainingSignal]:
        """Get recent training signals."""
        return self._signals[-limit:]

    def clear(self) -> None:
        """Clear the signal buffer."""
        self._signals.clear()


# Global synthesis signal collector singleton
_synthesis_signal_collector: Optional[SynthesisTrainingSignalCollector] = None


def get_synthesis_signal_collector() -> SynthesisTrainingSignalCollector:
    """Get the global synthesis signal collector singleton."""
    global _synthesis_signal_collector
    if _synthesis_signal_collector is None:
        _synthesis_signal_collector = SynthesisTrainingSignalCollector()
    return _synthesis_signal_collector


# =============================================================================
# SYNTHESIS RESULT DATACLASS
# =============================================================================

@dataclass
class SynthesisResult:
    """Result from evidence synthesis."""

    response: str
    confidence_statement: str
    evidence_citations: List[str]  # List of source_ids
    follow_up_suggestions: List[str]
    synthesis_method: str  # "dspy" or "hardcoded"
    confidence_level: str  # "high", "moderate", "low"


# =============================================================================
# HARDCODED SYNTHESIS FALLBACK
# =============================================================================

def synthesize_response_hardcoded(
    query: str,
    intent: str,
    evidence: List[Dict[str, Any]],
    brand_context: str = "",
) -> Tuple[str, str, List[str], List[str], str]:
    """
    Hardcoded synthesis fallback when DSPy is unavailable.

    Returns:
        Tuple of (response, confidence_statement, citations, follow_ups, confidence_level)
    """
    # Extract citations from evidence
    citations = [e.get("source_id", "unknown") for e in evidence if e.get("source_id")]

    # Build response based on evidence
    if not evidence:
        response = (
            f"I don't have specific data to answer your question about "
            f"{brand_context + ' ' if brand_context else ''}"
            f"at this time. Could you please rephrase your question or try a different query?"
        )
        confidence_statement = "Low confidence - no relevant evidence found"
        confidence_level = "low"
        follow_ups = [
            "What specific KPI would you like to explore?",
            "Would you like to see available data for a different time period?",
        ]
    else:
        # Build response from evidence
        evidence_summary = []
        for i, e in enumerate(evidence[:3], 1):
            content = e.get("content", "")[:150]
            source = e.get("source", "data")
            evidence_summary.append(f"- **Source {i}** ({source}): {content}...")

        evidence_text = "\n".join(evidence_summary)

        # Determine confidence based on evidence quality
        avg_score = sum(e.get("relevance_score", e.get("score", 0.5)) for e in evidence) / len(evidence)
        if avg_score >= 0.7 and len(evidence) >= 2:
            confidence_level = "high"
            confidence_statement = f"High confidence based on {len(evidence)} corroborating sources (avg relevance: {avg_score:.0%})"
        elif avg_score >= 0.5 or len(evidence) >= 2:
            confidence_level = "moderate"
            confidence_statement = f"Moderate confidence based on {len(evidence)} source(s) (avg relevance: {avg_score:.0%})"
        else:
            confidence_level = "low"
            confidence_statement = f"Low confidence - limited evidence (avg relevance: {avg_score:.0%})"

        # Generate response based on intent
        if intent == "kpi_query":
            response = (
                f"Based on the retrieved data{' for ' + brand_context if brand_context else ''}:\n\n"
                f"{evidence_text}\n\n"
                f"These metrics provide insights into current performance trends."
            )
            follow_ups = [
                "Would you like to see the trend over a different time period?",
                "Should I analyze the drivers behind these numbers?",
            ]
        elif intent == "causal_analysis":
            response = (
                f"Analyzing the causal factors{' for ' + brand_context if brand_context else ''}:\n\n"
                f"{evidence_text}\n\n"
                f"These findings suggest potential cause-effect relationships in the data."
            )
            follow_ups = [
                "Would you like to explore any specific driver in more detail?",
                "Should I compare this with a different segment or region?",
            ]
        elif intent == "recommendation":
            response = (
                f"Based on the analysis{' for ' + brand_context if brand_context else ''}:\n\n"
                f"{evidence_text}\n\n"
                f"Consider these insights when making strategic decisions."
            )
            follow_ups = [
                "Would you like specific action items based on these recommendations?",
                "Should I prioritize these by potential impact?",
            ]
        else:
            response = (
                f"Here's what I found{' for ' + brand_context if brand_context else ''}:\n\n"
                f"{evidence_text}\n\n"
                f"Let me know if you'd like more details on any specific aspect."
            )
            follow_ups = [
                "Would you like me to dig deeper into any of these findings?",
                "Is there a specific aspect you'd like me to focus on?",
            ]

    return response, confidence_statement, citations, follow_ups, confidence_level


# =============================================================================
# MAIN SYNTHESIS FUNCTION
# =============================================================================

async def synthesize_response_dspy(
    query: str,
    intent: str,
    evidence: List[Dict[str, Any]],
    brand_context: str = "",
    conversation_context: str = "",
    collect_signal: bool = True,
) -> SynthesisResult:
    """
    Synthesize a response from evidence using DSPy (with hardcoded fallback).

    This function uses DSPy's ChainOfThought to generate structured responses
    with confidence statements and proper evidence citations.

    Args:
        query: User's original question
        intent: Classified intent type
        evidence: Retrieved evidence items from RAG
        brand_context: Optional brand filter
        conversation_context: Recent conversation for continuity
        collect_signal: Whether to collect training signals

    Returns:
        SynthesisResult with response, confidence, and citations
    """
    synthesis_method = "hardcoded"
    response = ""
    confidence_statement = ""
    evidence_citations: List[str] = []
    follow_up_suggestions: List[str] = []
    confidence_level = "low"

    # Format evidence for DSPy
    evidence_text = ""
    if evidence:
        evidence_parts = []
        for e in evidence[:5]:  # Top 5 for context window
            source_id = e.get("source_id", "unknown")
            content = e.get("content", "")[:300]
            score = e.get("relevance_score", e.get("score", 0.0))
            source = e.get("source", "data")
            evidence_parts.append(f"[{source_id}] ({source}, relevance: {score:.0%}): {content}")
        evidence_text = "\n\n".join(evidence_parts)

    # Try DSPy synthesis if enabled
    if CHATBOT_DSPY_SYNTHESIS_ENABLED and DSPY_AVAILABLE:
        try:
            synthesizer = _get_dspy_synthesizer()
            if synthesizer:
                result = synthesizer(
                    query=query,
                    intent=intent,
                    evidence=evidence_text if evidence_text else "No evidence retrieved.",
                    brand_context=brand_context,
                    conversation_context=conversation_context,
                )

                response = result.response
                confidence_statement = result.confidence_statement
                follow_up_suggestions_str = getattr(result, "follow_up_suggestions", "")

                # Parse evidence citations (comma-separated source_ids)
                citations_str = result.evidence_citations
                if citations_str:
                    evidence_citations = [
                        c.strip() for c in citations_str.split(",") if c.strip()
                    ]

                # Parse follow-up suggestions
                if follow_up_suggestions_str:
                    follow_up_suggestions = [
                        s.strip() for s in follow_up_suggestions_str.split("?")
                        if s.strip() and len(s.strip()) > 10
                    ]
                    # Re-add question marks
                    follow_up_suggestions = [s + "?" if not s.endswith("?") else s for s in follow_up_suggestions[:2]]

                # Determine confidence level from statement
                confidence_lower = confidence_statement.lower()
                if "high" in confidence_lower:
                    confidence_level = "high"
                elif "moderate" in confidence_lower or "medium" in confidence_lower:
                    confidence_level = "moderate"
                else:
                    confidence_level = "low"

                synthesis_method = "dspy"
                logger.debug(f"DSPy synthesis complete: {len(response)} chars, {len(evidence_citations)} citations")

        except Exception as e:
            logger.warning(f"DSPy synthesis failed, using fallback: {e}")

    # Fallback to hardcoded synthesis
    if synthesis_method == "hardcoded":
        response, confidence_statement, evidence_citations, follow_up_suggestions, confidence_level = (
            synthesize_response_hardcoded(query, intent, evidence, brand_context)
        )

    # Collect training signal
    if collect_signal:
        signal = SynthesisTrainingSignal(
            query=query,
            intent=intent,
            evidence_count=len(evidence),
            response_length=len(response),
            confidence_level=confidence_level,
            citations_count=len(evidence_citations),
            synthesis_method=synthesis_method,
        )
        get_synthesis_signal_collector().add_signal(signal)

    return SynthesisResult(
        response=response,
        confidence_statement=confidence_statement,
        evidence_citations=evidence_citations,
        follow_up_suggestions=follow_up_suggestions,
        synthesis_method=synthesis_method,
        confidence_level=confidence_level,
    )


# =============================================================================
# PHASE 7: UNIFIED TRAINING SIGNAL COLLECTION
# =============================================================================

@dataclass
class ChatbotSessionSignal:
    """
    Unified training signal for a complete chatbot session.

    Aggregates signals from all phases (intent, routing, RAG, synthesis)
    to provide comprehensive training data for feedback_learner.
    """

    # Session identification
    session_id: str
    thread_id: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Query context
    query: str = ""
    brand_context: str = ""
    region_context: str = ""

    # Phase 3: Intent Classification
    predicted_intent: str = ""
    intent_confidence: float = 0.0
    intent_method: str = ""  # "dspy" or "hardcoded"
    intent_reasoning: str = ""

    # Phase 4: Agent Routing
    predicted_agent: str = ""
    secondary_agents: List[str] = field(default_factory=list)
    routing_confidence: float = 0.0
    routing_method: str = ""  # "dspy" or "hardcoded"
    routing_rationale: str = ""

    # Phase 5: Cognitive RAG
    rewritten_query: str = ""
    search_keywords: List[str] = field(default_factory=list)
    graph_entities: List[str] = field(default_factory=list)
    evidence_count: int = 0
    hop_count: int = 0
    avg_relevance_score: float = 0.0
    rag_method: str = ""  # "cognitive" or "basic"

    # Phase 6: Evidence Synthesis
    response_length: int = 0
    synthesis_confidence: str = ""  # "high", "moderate", "low"
    citations_count: int = 0
    synthesis_method: str = ""  # "dspy" or "hardcoded"
    follow_up_count: int = 0

    # User feedback (populated after interaction)
    user_rating: Optional[float] = None  # 1-5 scale
    was_helpful: Optional[bool] = None
    user_followed_up: Optional[bool] = None
    had_hallucination: Optional[bool] = None

    # Timing metrics
    total_duration_ms: Optional[float] = None
    intent_duration_ms: Optional[float] = None
    routing_duration_ms: Optional[float] = None
    rag_duration_ms: Optional[float] = None
    synthesis_duration_ms: Optional[float] = None

    def compute_accuracy_reward(self) -> float:
        """
        Compute accuracy reward based on classification and routing correctness.

        Returns:
            Reward between 0.0 and 1.0
        """
        reward = 0.0

        # Intent classification confidence
        if self.intent_method == "dspy":
            reward += self.intent_confidence * 0.2
        else:
            reward += self.intent_confidence * 0.15

        # Routing confidence
        if self.routing_method == "dspy":
            reward += self.routing_confidence * 0.2
        else:
            reward += self.routing_confidence * 0.15

        # Evidence retrieval quality
        if self.evidence_count >= 3:
            reward += 0.15
        elif self.evidence_count >= 1:
            reward += 0.1

        # Relevance score
        reward += self.avg_relevance_score * 0.15

        # User feedback if available
        if self.was_helpful is True:
            reward += 0.3
        elif self.was_helpful is False:
            reward -= 0.2

        return max(0.0, min(1.0, reward))

    def compute_efficiency_reward(self) -> float:
        """
        Compute efficiency reward based on response time and resource usage.

        Returns:
            Reward between 0.0 and 1.0
        """
        reward = 0.5  # Base reward

        # DSPy usage rewards (structured processing)
        dspy_usage = sum([
            1 if self.intent_method == "dspy" else 0,
            1 if self.routing_method == "dspy" else 0,
            1 if self.rag_method == "cognitive" else 0,
            1 if self.synthesis_method == "dspy" else 0,
        ])
        reward += dspy_usage * 0.05  # Up to 0.2 bonus for full DSPy

        # Hop efficiency (fewer hops for good results = better)
        if self.evidence_count >= 3 and self.hop_count <= 2:
            reward += 0.15
        elif self.evidence_count >= 1 and self.hop_count <= 3:
            reward += 0.1

        # Response length efficiency
        if 100 <= self.response_length <= 500:
            reward += 0.1
        elif 50 <= self.response_length <= 800:
            reward += 0.05

        # Timing rewards (if available)
        if self.total_duration_ms is not None:
            if self.total_duration_ms < 2000:  # Under 2 seconds
                reward += 0.1
            elif self.total_duration_ms < 5000:  # Under 5 seconds
                reward += 0.05
            elif self.total_duration_ms > 10000:  # Over 10 seconds
                reward -= 0.1

        return max(0.0, min(1.0, reward))

    def compute_satisfaction_reward(self) -> float:
        """
        Compute satisfaction reward based on user feedback and response quality.

        Returns:
            Reward between 0.0 and 1.0
        """
        reward = 0.3  # Base reward for completing the interaction

        # User rating (1-5 scale)
        if self.user_rating is not None:
            reward += (self.user_rating - 3) * 0.15  # -0.3 to +0.3

        # Helpful response
        if self.was_helpful is True:
            reward += 0.25
        elif self.was_helpful is False:
            reward -= 0.2

        # User engagement (followed up with another question)
        if self.user_followed_up is True:
            reward += 0.1

        # Hallucination penalty
        if self.had_hallucination is True:
            reward -= 0.4

        # Synthesis confidence
        if self.synthesis_confidence == "high":
            reward += 0.15
        elif self.synthesis_confidence == "moderate":
            reward += 0.1
        elif self.synthesis_confidence == "low":
            reward += 0.0

        # Citations (grounded responses)
        if self.citations_count >= 2:
            reward += 0.1
        elif self.citations_count >= 1:
            reward += 0.05

        return max(0.0, min(1.0, reward))

    def compute_unified_reward(self) -> Dict[str, float]:
        """
        Compute unified reward scores across all dimensions.

        Returns:
            Dict with accuracy, efficiency, satisfaction, and overall scores
        """
        accuracy = self.compute_accuracy_reward()
        efficiency = self.compute_efficiency_reward()
        satisfaction = self.compute_satisfaction_reward()

        # Weighted overall score
        overall = (
            accuracy * 0.35 +  # Classification/routing correctness
            efficiency * 0.25 +  # Resource efficiency
            satisfaction * 0.40  # User satisfaction (most important)
        )

        return {
            "accuracy": accuracy,
            "efficiency": efficiency,
            "satisfaction": satisfaction,
            "overall": overall,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for database storage."""
        return {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "brand_context": self.brand_context,
            "region_context": self.region_context,
            "predicted_intent": self.predicted_intent,
            "intent_confidence": self.intent_confidence,
            "intent_method": self.intent_method,
            "intent_reasoning": self.intent_reasoning,
            "predicted_agent": self.predicted_agent,
            "secondary_agents": self.secondary_agents,
            "routing_confidence": self.routing_confidence,
            "routing_method": self.routing_method,
            "routing_rationale": self.routing_rationale,
            "rewritten_query": self.rewritten_query,
            "search_keywords": self.search_keywords,
            "graph_entities": self.graph_entities,
            "evidence_count": self.evidence_count,
            "hop_count": self.hop_count,
            "avg_relevance_score": self.avg_relevance_score,
            "rag_method": self.rag_method,
            "response_length": self.response_length,
            "synthesis_confidence": self.synthesis_confidence,
            "citations_count": self.citations_count,
            "synthesis_method": self.synthesis_method,
            "follow_up_count": self.follow_up_count,
            "user_rating": self.user_rating,
            "was_helpful": self.was_helpful,
            "user_followed_up": self.user_followed_up,
            "had_hallucination": self.had_hallucination,
            "total_duration_ms": self.total_duration_ms,
            "intent_duration_ms": self.intent_duration_ms,
            "routing_duration_ms": self.routing_duration_ms,
            "rag_duration_ms": self.rag_duration_ms,
            "synthesis_duration_ms": self.synthesis_duration_ms,
            **self.compute_unified_reward(),
        }


class ChatbotSignalCollector:
    """
    Unified training signal collector for the chatbot workflow.

    Aggregates signals from all phases (intent, routing, RAG, synthesis)
    and provides methods for computing rewards and persisting to database.
    """

    def __init__(self, buffer_size: int = 500):
        self._signals: List[ChatbotSessionSignal] = []
        self._buffer_size = buffer_size
        self._pending_sessions: Dict[str, ChatbotSessionSignal] = {}

    def start_session(
        self,
        session_id: str,
        thread_id: str,
        query: str,
        user_id: Optional[str] = None,
        brand_context: str = "",
        region_context: str = "",
    ) -> ChatbotSessionSignal:
        """
        Start collecting signals for a new session.

        Args:
            session_id: Unique session identifier
            thread_id: Conversation thread ID
            query: User's query
            user_id: Optional user identifier
            brand_context: Brand filter context
            region_context: Region filter context

        Returns:
            New ChatbotSessionSignal instance
        """
        signal = ChatbotSessionSignal(
            session_id=session_id,
            thread_id=thread_id,
            query=query,
            user_id=user_id,
            brand_context=brand_context,
            region_context=region_context,
        )
        self._pending_sessions[session_id] = signal
        logger.debug(f"Started signal collection for session: {session_id}")
        return signal

    def get_session(self, session_id: str) -> Optional[ChatbotSessionSignal]:
        """Get a pending session by ID."""
        return self._pending_sessions.get(session_id)

    def update_intent(
        self,
        session_id: str,
        intent: str,
        confidence: float,
        method: str,
        reasoning: str = "",
        duration_ms: Optional[float] = None,
    ) -> None:
        """Update intent classification results for a session."""
        signal = self._pending_sessions.get(session_id)
        if signal:
            signal.predicted_intent = intent
            signal.intent_confidence = confidence
            signal.intent_method = method
            signal.intent_reasoning = reasoning
            signal.intent_duration_ms = duration_ms
            logger.debug(f"Updated intent signal for session {session_id}: {intent}")

    def update_routing(
        self,
        session_id: str,
        agent: str,
        secondary_agents: List[str],
        confidence: float,
        method: str,
        rationale: str = "",
        duration_ms: Optional[float] = None,
    ) -> None:
        """Update agent routing results for a session."""
        signal = self._pending_sessions.get(session_id)
        if signal:
            signal.predicted_agent = agent
            signal.secondary_agents = secondary_agents
            signal.routing_confidence = confidence
            signal.routing_method = method
            signal.routing_rationale = rationale
            signal.routing_duration_ms = duration_ms
            logger.debug(f"Updated routing signal for session {session_id}: {agent}")

    def update_rag(
        self,
        session_id: str,
        rewritten_query: str,
        keywords: List[str],
        entities: List[str],
        evidence_count: int,
        hop_count: int,
        avg_relevance: float,
        method: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Update RAG retrieval results for a session."""
        signal = self._pending_sessions.get(session_id)
        if signal:
            signal.rewritten_query = rewritten_query
            signal.search_keywords = keywords
            signal.graph_entities = entities
            signal.evidence_count = evidence_count
            signal.hop_count = hop_count
            signal.avg_relevance_score = avg_relevance
            signal.rag_method = method
            signal.rag_duration_ms = duration_ms
            logger.debug(
                f"Updated RAG signal for session {session_id}: "
                f"{evidence_count} evidence, {hop_count} hops"
            )

    def update_synthesis(
        self,
        session_id: str,
        response_length: int,
        confidence: str,
        citations_count: int,
        method: str,
        follow_up_count: int = 0,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Update synthesis results for a session."""
        signal = self._pending_sessions.get(session_id)
        if signal:
            signal.response_length = response_length
            signal.synthesis_confidence = confidence
            signal.citations_count = citations_count
            signal.synthesis_method = method
            signal.follow_up_count = follow_up_count
            signal.synthesis_duration_ms = duration_ms
            logger.debug(
                f"Updated synthesis signal for session {session_id}: "
                f"{response_length} chars, {confidence} confidence"
            )

    def update_feedback(
        self,
        session_id: str,
        user_rating: Optional[float] = None,
        was_helpful: Optional[bool] = None,
        user_followed_up: Optional[bool] = None,
        had_hallucination: Optional[bool] = None,
    ) -> None:
        """Update user feedback for a session."""
        signal = self._pending_sessions.get(session_id)
        if signal:
            if user_rating is not None:
                signal.user_rating = user_rating
            if was_helpful is not None:
                signal.was_helpful = was_helpful
            if user_followed_up is not None:
                signal.user_followed_up = user_followed_up
            if had_hallucination is not None:
                signal.had_hallucination = had_hallucination
            logger.debug(f"Updated feedback for session {session_id}")

    def finalize_session(
        self,
        session_id: str,
        total_duration_ms: Optional[float] = None,
    ) -> Optional[ChatbotSessionSignal]:
        """
        Finalize a session and move signal to completed buffer.

        Args:
            session_id: Session ID to finalize
            total_duration_ms: Total interaction duration

        Returns:
            Finalized signal or None if session not found
        """
        signal = self._pending_sessions.pop(session_id, None)
        if signal:
            signal.total_duration_ms = total_duration_ms

            # Add to completed signals buffer
            self._signals.append(signal)
            if len(self._signals) > self._buffer_size:
                self._signals = self._signals[-self._buffer_size:]

            rewards = signal.compute_unified_reward()
            logger.info(
                f"Finalized session {session_id} - "
                f"rewards: accuracy={rewards['accuracy']:.2f}, "
                f"efficiency={rewards['efficiency']:.2f}, "
                f"satisfaction={rewards['satisfaction']:.2f}, "
                f"overall={rewards['overall']:.2f}"
            )
            return signal
        return None

    def get_signals(self, limit: int = 100) -> List[ChatbotSessionSignal]:
        """Get recent completed signals."""
        return self._signals[-limit:]

    def get_high_quality_signals(
        self,
        min_overall_reward: float = 0.6,
        limit: int = 50,
    ) -> List[ChatbotSessionSignal]:
        """Get high-quality signals for training."""
        high_quality = [
            s for s in self._signals
            if s.compute_unified_reward()["overall"] >= min_overall_reward
        ]
        return high_quality[-limit:]

    def get_signals_for_training(
        self,
        phase: str = "all",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get signals formatted for training the specified phase.

        Args:
            phase: "intent", "routing", "rag", "synthesis", or "all"
            limit: Maximum number of signals to return

        Returns:
            List of dictionaries with relevant training data
        """
        signals = self._signals[-limit:]

        if phase == "intent":
            return [
                {
                    "query": s.query,
                    "context": s.brand_context,
                    "target_intent": s.predicted_intent,
                    "confidence": s.intent_confidence,
                    "reward": s.compute_accuracy_reward(),
                }
                for s in signals if s.intent_method
            ]
        elif phase == "routing":
            return [
                {
                    "query": s.query,
                    "intent": s.predicted_intent,
                    "context": s.brand_context,
                    "target_agent": s.predicted_agent,
                    "confidence": s.routing_confidence,
                    "reward": s.compute_accuracy_reward(),
                }
                for s in signals if s.routing_method
            ]
        elif phase == "rag":
            return [
                {
                    "query": s.query,
                    "rewritten": s.rewritten_query,
                    "keywords": s.search_keywords,
                    "entities": s.graph_entities,
                    "evidence_count": s.evidence_count,
                    "relevance": s.avg_relevance_score,
                    "reward": s.compute_efficiency_reward(),
                }
                for s in signals if s.rag_method
            ]
        elif phase == "synthesis":
            return [
                {
                    "query": s.query,
                    "intent": s.predicted_intent,
                    "evidence_count": s.evidence_count,
                    "response_length": s.response_length,
                    "confidence": s.synthesis_confidence,
                    "citations": s.citations_count,
                    "reward": s.compute_satisfaction_reward(),
                }
                for s in signals if s.synthesis_method
            ]
        else:
            return [s.to_dict() for s in signals]

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics for collected signals."""
        if not self._signals:
            return {
                "total_signals": 0,
                "pending_sessions": len(self._pending_sessions),
            }

        rewards = [s.compute_unified_reward() for s in self._signals]
        dspy_usage = {
            "intent": sum(1 for s in self._signals if s.intent_method == "dspy"),
            "routing": sum(1 for s in self._signals if s.routing_method == "dspy"),
            "rag": sum(1 for s in self._signals if s.rag_method == "cognitive"),
            "synthesis": sum(1 for s in self._signals if s.synthesis_method == "dspy"),
        }

        return {
            "total_signals": len(self._signals),
            "pending_sessions": len(self._pending_sessions),
            "avg_accuracy": sum(r["accuracy"] for r in rewards) / len(rewards),
            "avg_efficiency": sum(r["efficiency"] for r in rewards) / len(rewards),
            "avg_satisfaction": sum(r["satisfaction"] for r in rewards) / len(rewards),
            "avg_overall": sum(r["overall"] for r in rewards) / len(rewards),
            "dspy_usage": dspy_usage,
            "high_quality_count": sum(1 for r in rewards if r["overall"] >= 0.6),
        }

    def clear(self) -> None:
        """Clear all signals."""
        self._signals.clear()
        self._pending_sessions.clear()

    def __len__(self) -> int:
        return len(self._signals)

    async def persist_signal_to_database(
        self,
        signal: ChatbotSessionSignal,
    ) -> Optional[int]:
        """
        Persist a training signal to the database.

        Args:
            signal: The ChatbotSessionSignal to persist

        Returns:
            The database ID of the inserted signal, or None if failed
        """
        try:
            from src.memory.services.factories import get_async_supabase_service_client

            # Use service role client to bypass RLS for internal signal collection
            client = await get_async_supabase_service_client()
            if not client:
                logger.warning("No Supabase service client available for signal persistence")
                return None

            # Compute rewards
            rewards = signal.compute_unified_reward()

            # Convert user_id to UUID if present
            user_id = None
            if signal.user_id:
                try:
                    import uuid
                    user_id = str(uuid.UUID(signal.user_id))
                except (ValueError, TypeError):
                    # user_id is not a valid UUID, skip it
                    pass

            # Prepare signal data
            signal_data = {
                "session_id": signal.session_id,
                "thread_id": signal.thread_id,
                "user_id": user_id,
                "query": signal.query,
                "brand_context": signal.brand_context or "",
                "region_context": signal.region_context or "",
                "predicted_intent": signal.predicted_intent or "",
                "intent_confidence": signal.intent_confidence,
                "intent_method": signal.intent_method or "",
                "intent_reasoning": signal.intent_reasoning or "",
                "predicted_agent": signal.predicted_agent or "",
                "secondary_agents": signal.secondary_agents or [],
                "routing_confidence": signal.routing_confidence,
                "routing_method": signal.routing_method or "",
                "routing_rationale": signal.routing_rationale or "",
                "rewritten_query": signal.rewritten_query or "",
                "search_keywords": signal.search_keywords or [],
                "graph_entities": signal.graph_entities or [],
                "evidence_count": signal.evidence_count,
                "hop_count": signal.hop_count,
                "avg_relevance_score": signal.avg_relevance_score,
                "rag_method": signal.rag_method or "",
                "response_length": signal.response_length,
                "synthesis_confidence": signal.synthesis_confidence or "",
                "citations_count": signal.citations_count,
                "synthesis_method": signal.synthesis_method or "",
                "follow_up_count": signal.follow_up_count,
                "user_rating": signal.user_rating,
                "was_helpful": signal.was_helpful,
                "user_followed_up": signal.user_followed_up,
                "had_hallucination": signal.had_hallucination,
                "total_duration_ms": signal.total_duration_ms,
                "intent_duration_ms": signal.intent_duration_ms,
                "routing_duration_ms": signal.routing_duration_ms,
                "rag_duration_ms": signal.rag_duration_ms,
                "synthesis_duration_ms": signal.synthesis_duration_ms,
                "reward_accuracy": rewards["accuracy"],
                "reward_efficiency": rewards["efficiency"],
                "reward_satisfaction": rewards["satisfaction"],
                "reward_overall": rewards["overall"],
                "session_timestamp": signal.timestamp.isoformat(),
            }

            result = await client.table("chatbot_training_signals").insert(
                signal_data
            ).execute()

            if result.data and len(result.data) > 0:
                db_id = result.data[0].get("id")
                logger.debug(f"Persisted training signal to database: id={db_id}")
                return db_id
            return None

        except Exception as e:
            logger.warning(f"Failed to persist training signal: {e}")
            return None


# Global unified signal collector singleton
_chatbot_signal_collector: Optional[ChatbotSignalCollector] = None


def get_chatbot_signal_collector() -> ChatbotSignalCollector:
    """Get the global chatbot signal collector singleton."""
    global _chatbot_signal_collector
    if _chatbot_signal_collector is None:
        _chatbot_signal_collector = ChatbotSignalCollector()
    return _chatbot_signal_collector
