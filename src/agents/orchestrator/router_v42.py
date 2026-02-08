"""
E2I Orchestrator Router - Updated for Tool Composer Integration
Version: 4.2
Purpose: Route queries to appropriate agents, including Tool Composer for multi-faceted queries

This file shows how the Orchestrator router is updated to integrate Tool Composer.
The router now includes MULTI_FACETED as a query classification that triggers
the Tool Composer instead of routing to a single agent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, cast

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY INTENT CLASSIFICATION
# ============================================================================


class QueryIntent(str, Enum):
    """
    Classification of query intents for routing.

    V4.2 adds MULTI_FACETED for Tool Composer routing.
    """

    # Single-agent intents
    CAUSAL = "causal"  # → Causal Impact Agent
    COMPARATIVE = "comparative"  # → Gap Analyzer
    HETEROGENEOUS = "heterogeneous"  # → Heterogeneous Optimizer
    EXPERIMENTAL = "experimental"  # → Experiment Designer
    MONITORING = "monitoring"  # → Drift Monitor / Health Score
    PREDICTIVE = "predictive"  # → Prediction Synthesizer
    EXPLORATORY = "exploratory"  # → Explainer
    VALIDATION = "validation"  # → Causal Impact (validation workflow)

    # V4.2: Multi-agent intent
    MULTI_FACETED = "multi_faceted"  # → Tool Composer

    # Fallback
    UNKNOWN = "unknown"  # → Explainer (best effort)


@dataclass
class RoutingDecision:
    """Result of query routing"""

    intent: QueryIntent
    target: str  # Agent name or "tool_composer"
    confidence: float
    reasoning: str
    extracted_entities: Dict[str, Any]


# ============================================================================
# MULTI-FACETED DETECTION
# ============================================================================


class MultiFacetedDetector:
    """
    Detects whether a query should be routed to Tool Composer.

    Heuristics:
    1. Multiple question marks
    2. Chained reasoning phrases ("and then", "what if", "after that")
    3. Multiple entity types referenced (3+)
    4. Cross-domain analysis (causal + predictive + comparison)
    """

    # Phrases that suggest chained reasoning
    CHAINING_PHRASES = [
        r"\band\s+then\b",
        r"\bwhat\s+if\b",
        r"\bafter\s+that\b",
        r"\bfollowed\s+by\b",
        r"\band\s+predict\b",
        r"\band\s+estimate\b",
        r"\band\s+simulate\b",
        r"\band\s+what\s+would\b",
        r"\bthen\s+what\b",
    ]

    # Phrases suggesting multi-domain analysis
    MULTI_DOMAIN_PHRASES = [
        r"compare.*and.*predict",
        r"impact.*and.*gap",
        r"effect.*and.*forecast",
        r"causal.*and.*roi",
        r"analyze.*then.*recommend",
        r"segment.*and.*simulate",
    ]

    # Entity type patterns
    ENTITY_PATTERNS = {
        "region": r"\b(northeast|midwest|south|west|region|territory)\b",
        "brand": r"\b(remibrutinib|fabhalta|kisqali|brand|drug)\b",
        "hcp": r"\b(hcp|physician|doctor|oncologist|rheumatologist|provider)\b",
        "time": r"\b(q[1-4]|quarter|month|year|week|2024|2023|ytd)\b",
        "metric": r"\b(rx|prescription|volume|share|revenue|nps|access)\b",
        "intervention": r"\b(campaign|visit|program|message|detailing|speaker)\b",
    }

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def detect(self, query: str) -> Tuple[bool, float, str]:
        """
        Detect if query is multi-faceted.

        Returns:
            (is_multi_faceted, confidence, reasoning)
        """
        query_lower = query.lower()
        signals = []

        # Check for multiple questions
        question_count = query.count("?")
        if question_count > 1:
            signals.append(f"Multiple questions ({question_count})")

        # Check for chaining phrases
        for pattern in self.CHAINING_PHRASES:
            if re.search(pattern, query_lower):
                signals.append(f"Chaining phrase: {pattern}")
                break

        # Check for multi-domain phrases
        for pattern in self.MULTI_DOMAIN_PHRASES:
            if re.search(pattern, query_lower):
                signals.append(f"Multi-domain pattern: {pattern}")
                break

        # Count entity types
        entity_types_found = []
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            if re.search(pattern, query_lower):
                entity_types_found.append(entity_type)

        if len(entity_types_found) >= 3:
            signals.append(
                f"Multiple entity types ({len(entity_types_found)}): {entity_types_found}"
            )

        # Calculate confidence
        confidence = len(signals) / 4.0  # Max 4 signal types
        confidence = min(confidence, 1.0)

        is_multi_faceted = confidence >= self.confidence_threshold
        reasoning = "; ".join(signals) if signals else "No multi-faceted signals detected"

        return is_multi_faceted, confidence, reasoning


# ============================================================================
# QUERY ROUTER
# ============================================================================


class QueryRouter:
    """
    Routes queries to appropriate agents or Tool Composer.

    V4.2 Update: Now checks for MULTI_FACETED queries first and routes
    to Tool Composer when detected.
    """

    def __init__(
        self,
        multi_faceted_detector: Optional[MultiFacetedDetector] = None,
        enable_tool_composer: bool = True,
    ):
        self.multi_faceted_detector = multi_faceted_detector or MultiFacetedDetector()
        self.enable_tool_composer = enable_tool_composer

        # Intent patterns for single-agent routing
        self.intent_patterns = {
            QueryIntent.CAUSAL: [
                r"\bimpact\b.*\bof\b",
                r"\bcausal\b",
                r"\beffect\b.*\bon\b",
                r"\bcause\b",
                r"\bdriv(e|ing|er)\b",
                r"\battribut",
            ],
            QueryIntent.COMPARATIVE: [
                r"\bgap\b",
                r"\bcompare\b",
                r"\bdifference\b",
                r"\bvs\.?\b",
                r"\bversus\b",
                r"\bunderperform",
                r"\boutperform",
            ],
            QueryIntent.HETEROGENEOUS: [
                r"\bsegment\b",
                r"\bvary\b.*\bby\b",
                r"\bheterogeneous\b",
                r"\bdiffer.*\bacross\b",
                r"\bwhich\b.*\brespond\b",
                r"\btarget\b.*\bgroup\b",
            ],
            QueryIntent.EXPERIMENTAL: [
                r"\ba/b\s*test\b",
                r"\bexperiment\b",
                r"\btest\s+design\b",
                r"\bsample\s+size\b",
                r"\bpower\b.*\banalysis\b",
                r"\brandomiz",
            ],
            QueryIntent.MONITORING: [
                r"\bdrift\b",
                r"\bhealth\b.*\bscore\b",
                r"\bdegradation\b",
                r"\bperformance\b.*\bover\s+time\b",
                r"\bpsi\b",
                r"\bstability\b",
            ],
            QueryIntent.PREDICTIVE: [
                r"\bpredict\b",
                r"\bforecast\b",
                r"\bproject\b",
                r"\brisk\b.*\bscore\b",
                r"\bpropensity\b",
                r"\blikelihood\b",
            ],
            QueryIntent.VALIDATION: [
                r"\brefut\b",
                r"\bvalidat\b",
                r"\brobust\b",
                r"\bsensitivity\b",
                r"\be-?value\b",
            ],
        }

        # Agent routing map
        self.agent_map = {
            QueryIntent.CAUSAL: "causal_impact",
            QueryIntent.COMPARATIVE: "gap_analyzer",
            QueryIntent.HETEROGENEOUS: "heterogeneous_optimizer",
            QueryIntent.EXPERIMENTAL: "experiment_designer",
            QueryIntent.MONITORING: "drift_monitor",
            QueryIntent.PREDICTIVE: "prediction_synthesizer",
            QueryIntent.VALIDATION: "causal_impact",
            QueryIntent.EXPLORATORY: "explainer",
            QueryIntent.MULTI_FACETED: "tool_composer",  # V4.2
            QueryIntent.UNKNOWN: "explainer",
        }

    def route(self, query: str, entities: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route a query to the appropriate agent.

        Args:
            query: The user's query
            entities: Pre-extracted entities (optional)

        Returns:
            RoutingDecision with target agent and reasoning
        """
        entities = entities or {}

        # V4.2: Check for multi-faceted query FIRST
        if self.enable_tool_composer:
            is_multi, mf_confidence, mf_reasoning = self.multi_faceted_detector.detect(query)

            if is_multi:
                logger.info(f"Query classified as MULTI_FACETED: {mf_reasoning}")
                return RoutingDecision(
                    intent=QueryIntent.MULTI_FACETED,
                    target="tool_composer",
                    confidence=mf_confidence,
                    reasoning=mf_reasoning,
                    extracted_entities=entities,
                )

        # Single-agent classification
        intent, confidence, reasoning = self._classify_intent(query)
        target = self.agent_map.get(intent, "explainer")

        logger.info(f"Query classified as {intent.value}: {reasoning}")

        return RoutingDecision(
            intent=intent,
            target=target,
            confidence=confidence,
            reasoning=reasoning,
            extracted_entities=entities,
        )

    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float, str]:
        """Classify query intent using pattern matching"""
        query_lower = query.lower()
        scores: Dict[QueryIntent, float] = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matched_patterns = []

            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
                    matched_patterns.append(pattern)

            if score > 0:
                scores[intent] = score / len(patterns)

        if not scores:
            return QueryIntent.UNKNOWN, 0.5, "No intent patterns matched"

        # Get highest scoring intent
        best_intent = max(scores, key=lambda k: scores.get(k, 0.0))
        confidence = scores[best_intent]

        reasoning = f"Matched {best_intent.value} patterns with confidence {confidence:.2f}"

        return best_intent, confidence, reasoning


# ============================================================================
# ORCHESTRATOR INTEGRATION
# ============================================================================


class OrchestratorRouter:
    """
    Main orchestrator routing logic with Tool Composer integration.

    This class shows how the Orchestrator uses the router to dispatch
    queries to either single agents or the Tool Composer.
    """

    def __init__(
        self,
        router: Optional[QueryRouter] = None,
        tool_composer: Optional[Any] = None,  # ToolComposer instance
        agents: Optional[Dict[str, Any]] = None,  # Agent instances
    ):
        self.router = router or QueryRouter()
        self.tool_composer = tool_composer
        self.agents = agents or {}

    async def handle_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an incoming query by routing to appropriate handler.

        Args:
            query: User's query
            context: Optional context (filters, user info, etc.)

        Returns:
            Response dictionary with answer and metadata
        """
        context = context or {}

        # Route the query
        decision = self.router.route(query)

        logger.info(
            f"Routing decision: {decision.target} "
            f"(intent={decision.intent.value}, confidence={decision.confidence:.2f})"
        )

        # Dispatch based on routing decision
        if decision.intent == QueryIntent.MULTI_FACETED:
            # Route to Tool Composer
            return await self._handle_multi_faceted(query, context, decision)
        else:
            # Route to single agent
            return await self._handle_single_agent(query, context, decision)

    async def _handle_multi_faceted(
        self, query: str, context: Dict[str, Any], decision: RoutingDecision
    ) -> Dict[str, Any]:
        """Handle multi-faceted query via Tool Composer"""
        if self.tool_composer is None:
            logger.warning("Tool Composer not configured, falling back to Explainer")
            return await self._handle_single_agent(
                query,
                context,
                RoutingDecision(
                    intent=QueryIntent.EXPLORATORY,
                    target="explainer",
                    confidence=0.5,
                    reasoning="Tool Composer fallback",
                    extracted_entities={},
                ),
            )

        # Use Tool Composer integration
        from ..tool_composer import ToolComposerIntegration

        integration = ToolComposerIntegration(self.tool_composer)
        result = await integration.handle_multi_faceted_query(
            query=query, extracted_entities=decision.extracted_entities, user_context=context
        )

        # Add routing metadata
        result["routing"] = {
            "intent": decision.intent.value,
            "target": decision.target,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }

        return result

    async def _handle_single_agent(
        self, query: str, context: Dict[str, Any], decision: RoutingDecision
    ) -> Dict[str, Any]:
        """Handle query via single agent"""
        agent = self.agents.get(decision.target)

        if agent is None:
            logger.warning(f"Agent '{decision.target}' not found, using fallback")
            # Return a basic response
            return {
                "success": False,
                "response": f"Agent '{decision.target}' is not available",
                "routing": {
                    "intent": decision.intent.value,
                    "target": decision.target,
                    "confidence": decision.confidence,
                },
            }

        # Call the agent's handle method
        result: Dict[str, Any] = await agent.handle(query, context)

        # Add routing metadata
        result["routing"] = {
            "intent": decision.intent.value,
            "target": decision.target,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }

        return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test the router
    router = QueryRouter(enable_tool_composer=True)

    test_queries = [
        # Should route to Tool Composer
        "Compare the causal impact of rep visits vs speaker programs for oncologists, and predict which would work better in the Midwest",
        "What's driving the gap between our top and bottom territories, and what experiments should we run to close it?",
        "Which HCPs showed the biggest response to Q3 changes, and what if we extended that to the South?",
        # Should route to single agents
        "What was the impact of the Q3 campaign?",  # Causal Impact
        "Which territories are underperforming?",  # Gap Analyzer
        "Design an A/B test for the new messaging",  # Experiment Designer
        "What is the churn risk for patient segment A?",  # Prediction Synthesizer
    ]

    for query in test_queries:
        decision = router.route(query)
        print(f"\nQuery: {query[:60]}...")
        print(f"  → {decision.target} ({decision.intent.value}, {decision.confidence:.2f})")
        print(f"  Reasoning: {decision.reasoning}")
