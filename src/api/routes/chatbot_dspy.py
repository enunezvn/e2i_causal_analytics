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
