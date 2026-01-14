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
