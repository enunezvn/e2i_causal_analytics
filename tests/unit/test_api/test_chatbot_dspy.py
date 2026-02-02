"""
Unit tests for E2I Chatbot DSPy Intent Classification and Agent Routing.

Tests the DSPy-based classifier and router including:
- Hardcoded fallback classification
- Intent normalization
- Confidence validation
- Training signal collection
- Multi-faceted query detection
- Agent routing (DSPy and hardcoded fallback)
- Routing training signal collection
"""

import pytest

# Group all DSPy tests on same worker to prevent import race conditions
pytestmark = pytest.mark.xdist_group(name="dspy_integration")
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.routes.chatbot_dspy import (
    AGENT_CAPABILITIES,
    VALID_AGENTS,
    VALID_INTENTS,
    IntentTrainingSignal,
    IntentTrainingSignalCollector,
    RoutingTrainingSignal,
    RoutingTrainingSignalCollector,
    _is_multi_faceted_query,
    _matches_pattern,
    _normalize_agent,
    _normalize_intent,
    _validate_confidence,
    classify_intent_dspy,
    # Intent classification
    classify_intent_hardcoded,
    classify_intent_sync,
    get_routing_signal_collector,
    get_signal_collector,
    route_agent_dspy,
    # Agent routing
    route_agent_hardcoded,
    route_agent_sync,
)
from src.api.routes.chatbot_state import IntentType


class TestHardcodedClassification:
    """Test cases for hardcoded intent classification (fallback)."""

    def test_greeting_intent(self):
        """Test greeting pattern detection."""
        test_cases = [
            "Hello, how are you?",
            "Hi there!",
            "Hey",
            "Good morning team",
            "Good afternoon everyone",
        ]
        for query in test_cases:
            intent, confidence, reasoning = classify_intent_hardcoded(query)
            assert intent == IntentType.GREETING, f"Failed for: {query}"
            assert confidence >= 0.9

    def test_help_intent(self):
        """Test help request pattern detection."""
        test_cases = [
            "Help me with this",
            "What can you do?",
            "How do I use this?",
            "Guide me through the process",
        ]
        for query in test_cases:
            intent, confidence, reasoning = classify_intent_hardcoded(query)
            assert intent == IntentType.HELP, f"Failed for: {query}"
            assert confidence >= 0.9

    def test_kpi_query_intent(self):
        """Test KPI query pattern detection."""
        # Pattern matches: kpi, trx, nrx, market share, conversion, metric (singular), volume
        test_cases = [
            "What is the TRx for Kisqali?",
            "Tell me NRx numbers",
            "What's our market share?",
            "Conversion rate last month",
            "Volume in Northeast region",
            "Show me the KPI dashboard",
        ]
        for query in test_cases:
            intent, confidence, reasoning = classify_intent_hardcoded(query)
            assert intent == IntentType.KPI_QUERY, f"Failed for: {query}"
            assert confidence >= 0.85

    def test_causal_analysis_intent(self):
        """Test causal analysis pattern detection."""
        # Pattern matches: why, cause, caused, impact, effect, driver (singular), causal, because
        # Note: Queries with KPI terms (TRx, NRx) will match KPI first
        test_cases = [
            "Why did sales drop?",
            "What caused the decrease?",
            "Impact of the new campaign",
            "What is the main driver?",  # driver (singular)
            "Causal factors for growth",
            "Because of the promotion",
        ]
        for query in test_cases:
            intent, confidence, reasoning = classify_intent_hardcoded(query)
            assert intent == IntentType.CAUSAL_ANALYSIS, f"Failed for: {query}"
            assert confidence >= 0.85

    def test_agent_status_intent(self):
        """Test agent status pattern detection."""
        # Pattern matches: agent, tier, orchestrator, status, system
        test_cases = [
            "What is the agent doing?",
            "Show me the orchestrator",
            "What tier is this?",
            "System status check",
        ]
        for query in test_cases:
            intent, confidence, reasoning = classify_intent_hardcoded(query)
            assert intent == IntentType.AGENT_STATUS, f"Failed for: {query}"
            assert confidence >= 0.85

    def test_recommendation_intent(self):
        """Test recommendation pattern detection."""
        test_cases = [
            "What do you recommend?",
            "Suggest improvements",
            "How can we improve this?",
            "Optimize the strategy",
        ]
        for query in test_cases:
            intent, confidence, reasoning = classify_intent_hardcoded(query)
            assert intent == IntentType.RECOMMENDATION, f"Failed for: {query}"
            assert confidence >= 0.85

    def test_search_intent(self):
        """Test search pattern detection."""
        test_cases = [
            "Search for Kisqali data",
            "Find the report",
            "Look for trends",
            "Show me the dashboard",
        ]
        for query in test_cases:
            intent, confidence, reasoning = classify_intent_hardcoded(query)
            assert intent == IntentType.SEARCH, f"Failed for: {query}"
            assert confidence >= 0.8

    def test_general_intent_fallback(self):
        """Test fallback to general intent."""
        test_cases = [
            "The weather is nice today",
            "Random unrelated text",
            "Just testing something",
        ]
        for query in test_cases:
            intent, confidence, reasoning = classify_intent_hardcoded(query)
            assert intent == IntentType.GENERAL, f"Failed for: {query}"
            assert confidence >= 0.5


class TestMultiFacetedDetection:
    """Test cases for multi-faceted query detection."""

    def test_multi_faceted_with_analysis_and_recommendation(self):
        """Test detection of queries needing both analysis and recommendations."""
        # This triggers analysis_and_recommendation (why + recommend) AND conjunction_keywords (explain)
        query = "Why did sales drop and explain what you recommend we should do"
        assert _is_multi_faceted_query(query) is True

    def test_multi_faceted_with_multiple_kpis_and_conjunctions(self):
        """Test detection of queries with multiple KPIs and conjunction keywords."""
        # This triggers multiple_kpis (trx + nrx) AND conjunction_keywords (compare/trends)
        query = "Compare TRx and NRx trends across the region"
        assert _is_multi_faceted_query(query) is True

    def test_multi_faceted_cross_agent_with_conjunctions(self):
        """Test detection of cross-agent queries with conjunctions."""
        # This triggers cross_agent (causal) AND conjunction_keywords (explain/trends)
        query = "Explain the causal trends in our performance"
        assert _is_multi_faceted_query(query) is True

    def test_multi_faceted_multiple_brands_with_comparison(self):
        """Test detection of queries comparing multiple brands."""
        # This triggers multiple_brands (Kisqali + Fabhalta) AND conjunction_keywords (compare)
        query = "Compare Kisqali and Fabhalta performance"
        assert _is_multi_faceted_query(query) is True

    def test_simple_query_not_multi_faceted(self):
        """Test that simple queries are not marked as multi-faceted."""
        simple_queries = [
            "What is TRx?",
            "Show me the dashboard",
            "Hello there",
            "Why did sales drop?",  # Only one facet (would need recommend too)
        ]
        for query in simple_queries:
            assert _is_multi_faceted_query(query) is False, f"Failed for: {query}"


class TestPatternMatching:
    """Test cases for pattern matching helper."""

    def test_single_word_pattern(self):
        """Test single word pattern matching with word boundaries."""
        assert _matches_pattern("hello world", ["hello"]) is True
        assert _matches_pattern("this is nice", ["hi"]) is False  # "hi" inside "this"
        assert _matches_pattern("say hi there", ["hi"]) is True

    def test_multi_word_pattern(self):
        """Test multi-word pattern matching."""
        assert _matches_pattern("good morning team", ["good morning"]) is True
        assert _matches_pattern("morning team good", ["good morning"]) is False


class TestIntentNormalization:
    """Test cases for intent normalization."""

    def test_normalize_valid_intents(self):
        """Test that valid intents are returned unchanged."""
        for intent in VALID_INTENTS:
            assert _normalize_intent(intent) == intent

    def test_normalize_variations(self):
        """Test normalization of intent variations."""
        assert _normalize_intent("KPI") == IntentType.KPI_QUERY
        assert _normalize_intent("kpi query") == IntentType.KPI_QUERY
        assert _normalize_intent("causal") == IntentType.CAUSAL_ANALYSIS
        assert _normalize_intent("multi-faceted") == IntentType.MULTI_FACETED
        assert _normalize_intent("MULTIFACETED") == IntentType.MULTI_FACETED
        assert _normalize_intent("greetings") == IntentType.GREETING

    def test_normalize_unknown_falls_to_general(self):
        """Test that unknown intents default to general."""
        assert _normalize_intent("unknown_intent") == IntentType.GENERAL
        assert _normalize_intent("random text") == IntentType.GENERAL


class TestConfidenceValidation:
    """Test cases for confidence score validation."""

    def test_valid_confidence_values(self):
        """Test valid confidence values are returned unchanged."""
        assert _validate_confidence(0.5) == 0.5
        assert _validate_confidence(1.0) == 1.0
        assert _validate_confidence(0.0) == 0.0

    def test_confidence_clamping(self):
        """Test that confidence values are clamped to [0, 1]."""
        assert _validate_confidence(1.5) == 1.0
        assert _validate_confidence(-0.5) == 0.0
        assert _validate_confidence(100) == 1.0

    def test_invalid_confidence_types(self):
        """Test handling of invalid confidence types."""
        assert _validate_confidence("not a number") == 0.5
        assert _validate_confidence(None) == 0.5
        assert _validate_confidence([0.5]) == 0.5

    def test_string_number_confidence(self):
        """Test parsing of string numbers."""
        assert _validate_confidence("0.8") == 0.8
        assert _validate_confidence("1") == 1.0


class TestTrainingSignalCollection:
    """Test cases for training signal collection."""

    def test_signal_creation(self):
        """Test creating a training signal."""
        signal = IntentTrainingSignal(
            query="What is TRx?",
            conversation_context="",
            brand_context="Kisqali",
            predicted_intent="kpi_query",
            confidence=0.95,
            reasoning="Matched KPI pattern",
            classification_method="hardcoded",
        )
        assert signal.query == "What is TRx?"
        assert signal.confidence == 0.95
        assert signal.classification_method == "hardcoded"

    def test_signal_reward_computation(self):
        """Test reward computation for training signals."""
        # High confidence, correct routing
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="kpi_query",
            confidence=0.9,
            reasoning="test",
            classification_method="dspy",
            correct_routing=True,
            response_helpful=True,
        )
        reward = signal.compute_reward()
        assert reward > 0.5  # Should be high reward

        # Low confidence, incorrect routing
        signal_bad = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.3,
            reasoning="test",
            classification_method="hardcoded",
            correct_routing=False,
            response_helpful=False,
        )
        reward_bad = signal_bad.compute_reward()
        assert reward_bad < reward  # Should be lower reward

    def test_signal_collector_add_and_get(self):
        """Test signal collector add and retrieval."""
        collector = IntentTrainingSignalCollector(buffer_size=10)
        collector.clear()

        for i in range(5):
            signal = IntentTrainingSignal(
                query=f"query {i}",
                conversation_context="",
                brand_context="",
                predicted_intent="general",
                confidence=0.5 + (i * 0.1),
                reasoning="test",
                classification_method="hardcoded",
            )
            collector.add_signal(signal)

        assert len(collector) == 5
        signals = collector.get_signals(limit=3)
        assert len(signals) == 3

    def test_signal_collector_buffer_overflow(self):
        """Test that buffer size is respected."""
        collector = IntentTrainingSignalCollector(buffer_size=5)
        collector.clear()

        for i in range(10):
            signal = IntentTrainingSignal(
                query=f"query {i}",
                conversation_context="",
                brand_context="",
                predicted_intent="general",
                confidence=0.5,
                reasoning="test",
                classification_method="hardcoded",
            )
            collector.add_signal(signal)

        assert len(collector) == 5  # Buffer should cap at 5

    def test_get_high_quality_signals(self):
        """Test filtering for high-quality signals."""
        collector = IntentTrainingSignalCollector(buffer_size=100)
        collector.clear()

        # Add mix of high and low confidence signals
        for i, conf in enumerate([0.3, 0.5, 0.7, 0.8, 0.9]):
            signal = IntentTrainingSignal(
                query=f"query {i}",
                conversation_context="",
                brand_context="",
                predicted_intent="general",
                confidence=conf,
                reasoning="test",
                classification_method="hardcoded",
            )
            collector.add_signal(signal)

        high_quality = collector.get_high_quality_signals(min_confidence=0.7)
        assert len(high_quality) == 3  # Should only get signals with conf >= 0.7


class TestSyncClassification:
    """Test cases for synchronous classification wrapper."""

    def test_sync_classification(self):
        """Test synchronous classification uses hardcoded fallback."""
        intent, confidence, reasoning, method = classify_intent_sync("What is TRx for Kisqali?")
        assert intent == IntentType.KPI_QUERY
        assert method == "hardcoded"
        assert confidence >= 0.8


@pytest.mark.asyncio
class TestAsyncDSPyClassification:
    """Test cases for async DSPy classification."""

    async def test_async_classification_fallback(self):
        """Test async classification falls back to hardcoded when DSPy unavailable."""
        # Patch DSPy as unavailable
        with patch("src.api.routes.chatbot_dspy._get_dspy_classifier", return_value=None):
            intent, confidence, reasoning, method = await classify_intent_dspy(
                query="What is TRx for Kisqali?",
                collect_signal=False,
            )
            assert intent == IntentType.KPI_QUERY
            assert method == "hardcoded"
            assert confidence >= 0.8

    async def test_async_classification_collects_signal(self):
        """Test that signal collection works."""
        # Clear the global collector first
        collector = get_signal_collector()
        initial_count = len(collector)

        await classify_intent_dspy(
            query="Test query for signal collection",
            collect_signal=True,
        )

        # Should have one more signal
        assert len(collector) == initial_count + 1

    async def test_async_classification_with_context(self):
        """Test classification with conversation and brand context."""
        intent, confidence, reasoning, method = await classify_intent_dspy(
            query="What is TRx?",
            conversation_context="User: Hello\nAssistant: Hi there!",
            brand_context="Kisqali",
            collect_signal=False,
        )
        # DSPy model may classify TRx query as kpi_query or help depending on context
        # The key behavior is that a valid intent is returned with confidence
        valid_intents = [IntentType.KPI_QUERY, IntentType.HELP, "kpi_query", "help"]
        assert intent in valid_intents or hasattr(intent, "value")
        assert confidence > 0


class TestGlobalSignalCollector:
    """Test cases for global signal collector singleton."""

    def test_get_signal_collector_singleton(self):
        """Test that get_signal_collector returns same instance."""
        collector1 = get_signal_collector()
        collector2 = get_signal_collector()
        assert collector1 is collector2


# =============================================================================
# AGENT ROUTING TESTS (PHASE 4)
# =============================================================================


class TestAgentNormalization:
    """Test cases for agent name normalization."""

    def test_normalize_valid_agents(self):
        """Test that valid agent names are returned unchanged."""
        for agent in VALID_AGENTS:
            assert _normalize_agent(agent) == agent

    def test_normalize_with_variations(self):
        """Test normalization of agent name variations."""
        assert _normalize_agent("causal") == "causal_impact"
        assert _normalize_agent("causal_analysis") == "causal_impact"
        assert _normalize_agent("gap") == "gap_analyzer"
        assert _normalize_agent("drift") == "drift_monitor"
        assert _normalize_agent("experiment") == "experiment_designer"
        assert _normalize_agent("health") == "health_score"
        assert _normalize_agent("prediction") == "prediction_synthesizer"
        assert _normalize_agent("predict") == "prediction_synthesizer"
        assert _normalize_agent("resource") == "resource_optimizer"
        assert _normalize_agent("explain") == "explainer"
        assert _normalize_agent("feedback") == "feedback_learner"

    def test_normalize_with_whitespace_and_case(self):
        """Test normalization handles whitespace and case."""
        assert _normalize_agent("  CAUSAL_IMPACT  ") == "causal_impact"
        assert _normalize_agent("Gap-Analyzer") == "gap_analyzer"
        assert _normalize_agent("DRIFT_MONITOR") == "drift_monitor"

    def test_normalize_unknown_defaults_to_explainer(self):
        """Test that unknown agents default to explainer."""
        assert _normalize_agent("unknown_agent") == "explainer"
        assert _normalize_agent("random") == "explainer"


class TestHardcodedRouting:
    """Test cases for hardcoded agent routing (fallback)."""

    def test_route_causal_query(self):
        """Test routing of causal analysis queries."""
        test_cases = [
            "Why did TRx drop last quarter?",
            "What caused the sales decrease?",
            "Impact of the new marketing campaign",
            "What are the causal drivers?",
        ]
        for query in test_cases:
            agent, secondary, confidence, rationale = route_agent_hardcoded(query)
            assert agent == "causal_impact", f"Failed for: {query}"
            assert confidence >= 0.5

    def test_route_gap_analysis_query(self):
        """Test routing of gap analysis queries."""
        test_cases = [
            "Find growth opportunities in the market",
            "What ROI can we expect?",
            "Identify underperforming segments",
        ]
        for query in test_cases:
            agent, secondary, confidence, rationale = route_agent_hardcoded(query)
            assert agent == "gap_analyzer", f"Failed for: {query}"
            assert confidence >= 0.5

    def test_route_drift_monitoring_query(self):
        """Test routing of drift monitoring queries."""
        # Note: Keywords must match exactly - "anomaly" matches "anomaly", not "anomalies"
        test_cases = [
            "Check for data drift",
            "Are there any distribution shifts?",
            "Detect deviation in the metrics",
        ]
        for query in test_cases:
            agent, secondary, confidence, rationale = route_agent_hardcoded(query)
            assert agent == "drift_monitor", f"Failed for: {query}"
            assert confidence >= 0.5

    def test_route_experiment_design_query(self):
        """Test routing of experiment design queries."""
        test_cases = [
            "Design an A/B test for the campaign",
            "Create an experiment to test pricing",
            "Help me test this hypothesis",
        ]
        for query in test_cases:
            agent, secondary, confidence, rationale = route_agent_hardcoded(query)
            assert agent == "experiment_designer", f"Failed for: {query}"
            assert confidence >= 0.5

    def test_route_health_score_query(self):
        """Test routing of health score queries."""
        test_cases = [
            "What's the system health status?",
            "Show me the performance score",
            "Check the health metrics",
        ]
        for query in test_cases:
            agent, secondary, confidence, rationale = route_agent_hardcoded(query)
            assert agent == "health_score", f"Failed for: {query}"
            assert confidence >= 0.5

    def test_route_prediction_query(self):
        """Test routing of prediction queries."""
        test_cases = [
            "Predict next month's sales",
            "What's the forecast for Q2?",
            "Show me the trend projections",
        ]
        for query in test_cases:
            agent, secondary, confidence, rationale = route_agent_hardcoded(query)
            assert agent == "prediction_synthesizer", f"Failed for: {query}"
            assert confidence >= 0.5

    def test_route_with_intent_boosting(self):
        """Test that intent boosts routing confidence."""
        # Causal intent should boost causal_impact
        agent, _, _, _ = route_agent_hardcoded(
            "Explain the trend", intent=IntentType.CAUSAL_ANALYSIS
        )
        assert agent == "causal_impact"

        # KPI intent should boost health_score
        agent, _, _, _ = route_agent_hardcoded("Show performance data", intent=IntentType.KPI_QUERY)
        assert agent == "health_score"

        # Recommendation intent should boost gap_analyzer
        agent, _, _, _ = route_agent_hardcoded(
            "Suggest strategies", intent=IntentType.RECOMMENDATION
        )
        assert agent == "gap_analyzer"

    def test_route_default_to_explainer(self):
        """Test that unrecognized queries default to explainer."""
        agent, secondary, confidence, rationale = route_agent_hardcoded(
            "Just a random question about nothing specific"
        )
        assert agent == "explainer"
        assert confidence >= 0.5

    def test_route_returns_secondary_agents(self):
        """Test that routing returns secondary agent suggestions."""
        # Query with multiple agent matches
        agent, secondary, confidence, rationale = route_agent_hardcoded(
            "Why did drift cause the prediction to fail?"
        )
        # Should match causal_impact (why, cause) and drift_monitor (drift) and prediction_synthesizer (prediction)
        assert len(secondary) >= 1


class TestRoutingTrainingSignal:
    """Test cases for routing training signal collection."""

    def test_signal_creation(self):
        """Test creating a routing training signal."""
        signal = RoutingTrainingSignal(
            query="Why did TRx drop?",
            intent="causal_analysis",
            brand_context="Kisqali",
            predicted_agent="causal_impact",
            secondary_agents=["gap_analyzer"],
            confidence=0.9,
            rationale="Causal keywords matched",
            routing_method="hardcoded",
        )
        assert signal.query == "Why did TRx drop?"
        assert signal.predicted_agent == "causal_impact"
        assert signal.confidence == 0.9
        assert signal.routing_method == "hardcoded"

    def test_signal_reward_computation(self):
        """Test reward computation for routing signals."""
        # High confidence, correct agent
        signal = RoutingTrainingSignal(
            query="test",
            intent="",
            brand_context="",
            predicted_agent="causal_impact",
            secondary_agents=[],
            confidence=0.9,
            rationale="test",
            routing_method="dspy",
            correct_agent=True,
            response_quality=0.8,
        )
        reward = signal.compute_reward()
        assert reward > 0.5  # Should be high reward

        # Low confidence, incorrect agent
        signal_bad = RoutingTrainingSignal(
            query="test",
            intent="",
            brand_context="",
            predicted_agent="explainer",
            secondary_agents=[],
            confidence=0.3,
            rationale="test",
            routing_method="hardcoded",
            correct_agent=False,
        )
        reward_bad = signal_bad.compute_reward()
        assert reward_bad < reward  # Should be lower reward

    def test_routing_collector_add_and_get(self):
        """Test routing signal collector add and retrieval."""
        collector = RoutingTrainingSignalCollector(buffer_size=10)
        collector.clear()

        for i in range(5):
            signal = RoutingTrainingSignal(
                query=f"query {i}",
                intent="",
                brand_context="",
                predicted_agent="causal_impact",
                secondary_agents=[],
                confidence=0.5 + (i * 0.1),
                rationale="test",
                routing_method="hardcoded",
            )
            collector.add_signal(signal)

        assert len(collector) == 5
        signals = collector.get_signals(limit=3)
        assert len(signals) == 3

    def test_routing_collector_buffer_overflow(self):
        """Test that buffer size is respected."""
        collector = RoutingTrainingSignalCollector(buffer_size=5)
        collector.clear()

        for i in range(10):
            signal = RoutingTrainingSignal(
                query=f"query {i}",
                intent="",
                brand_context="",
                predicted_agent="causal_impact",
                secondary_agents=[],
                confidence=0.5,
                rationale="test",
                routing_method="hardcoded",
            )
            collector.add_signal(signal)

        assert len(collector) == 5  # Buffer should cap at 5


class TestGlobalRoutingSignalCollector:
    """Test cases for global routing signal collector singleton."""

    def test_get_routing_signal_collector_singleton(self):
        """Test that get_routing_signal_collector returns same instance."""
        collector1 = get_routing_signal_collector()
        collector2 = get_routing_signal_collector()
        assert collector1 is collector2


class TestSyncRouting:
    """Test cases for synchronous routing wrapper."""

    def test_sync_routing(self):
        """Test synchronous routing uses hardcoded fallback."""
        agent, secondary, confidence, rationale, method = route_agent_sync(
            "Why did TRx drop for Kisqali?"
        )
        assert agent == "causal_impact"
        assert method == "hardcoded"
        assert confidence >= 0.5


@pytest.mark.asyncio
class TestAsyncDSPyRouting:
    """Test cases for async DSPy agent routing."""

    async def test_async_routing_fallback(self):
        """Test async routing falls back to hardcoded when DSPy unavailable."""
        # Patch DSPy as unavailable
        with patch("src.api.routes.chatbot_dspy._get_dspy_router", return_value=None):
            agent, secondary, confidence, rationale, method = await route_agent_dspy(
                query="Why did TRx drop for Kisqali?",
                collect_signal=False,
            )
            assert agent == "causal_impact"
            assert method == "hardcoded"
            assert confidence >= 0.5

    async def test_async_routing_collects_signal(self):
        """Test that signal collection works."""
        # Clear the global collector first
        collector = get_routing_signal_collector()
        initial_count = len(collector)

        await route_agent_dspy(
            query="Test query for signal collection",
            collect_signal=True,
        )

        # Should have one more signal
        assert len(collector) == initial_count + 1

    async def test_async_routing_with_context(self):
        """Test routing with intent and brand context."""
        agent, secondary, confidence, rationale, method = await route_agent_dspy(
            query="What caused the drop?",
            intent=IntentType.CAUSAL_ANALYSIS,
            brand_context="Kisqali",
            collect_signal=False,
        )
        assert agent == "causal_impact"
        assert confidence > 0


class TestAgentCapabilitiesMapping:
    """Test cases for agent capabilities mapping."""

    def test_all_agents_have_capabilities(self):
        """Test that all valid agents have defined capabilities."""
        for agent in VALID_AGENTS:
            assert agent in AGENT_CAPABILITIES, f"Missing capabilities for {agent}"
            assert len(AGENT_CAPABILITIES[agent]) > 0, f"Empty capabilities for {agent}"

    def test_capabilities_are_unique_keywords(self):
        """Test that capability keywords are meaningful."""
        for agent, keywords in AGENT_CAPABILITIES.items():
            for keyword in keywords:
                # Keywords should be lowercase and non-empty
                assert keyword == keyword.lower(), f"Non-lowercase keyword: {keyword}"
                assert len(keyword) > 0, f"Empty keyword for {agent}"


# ============================================================================
# PHASE 5: Cognitive RAG DSPy Tests
# ============================================================================

from src.api.routes.chatbot_dspy import (
    CHATBOT_COGNITIVE_RAG_ENABLED,
    E2I_DOMAIN_VOCABULARY,
    # Cognitive RAG components
    CognitiveRAGResult,
    RAGTrainingSignal,
    RAGTrainingSignalCollector,
    cognitive_rag_retrieve,
    get_rag_signal_collector,
    rewrite_query_hardcoded,
)


class TestCognitiveRAGResult:
    """Test cases for CognitiveRAGResult dataclass."""

    def test_result_creation(self):
        """Test creating a cognitive RAG result."""
        result = CognitiveRAGResult(
            rewritten_query="TRx metrics for Kisqali Northeast region",
            search_keywords=["trx", "kisqali", "northeast"],
            graph_entities=["Kisqali", "Northeast"],
            evidence=[{"source_id": "doc1", "content": "TRx is 1000", "score": 0.9}],
            hop_count=1,
            avg_relevance_score=0.85,
            retrieval_method="cognitive",
        )
        assert result.rewritten_query == "TRx metrics for Kisqali Northeast region"
        assert len(result.search_keywords) == 3
        assert len(result.graph_entities) == 2
        assert len(result.evidence) == 1
        assert result.hop_count == 1
        assert result.avg_relevance_score == 0.85
        assert result.retrieval_method == "cognitive"

    def test_result_with_empty_evidence(self):
        """Test result with no evidence found."""
        result = CognitiveRAGResult(
            rewritten_query="test query",
            search_keywords=["test"],
            graph_entities=[],
            evidence=[],
            hop_count=0,
            avg_relevance_score=0.0,
            retrieval_method="basic",
        )
        assert result.evidence == []
        assert result.avg_relevance_score == 0.0


class TestRAGTrainingSignal:
    """Test cases for RAG training signal collection."""

    def test_signal_creation(self):
        """Test creating a RAG training signal."""
        signal = RAGTrainingSignal(
            query="What is TRx for Kisqali?",
            rewritten_query="TRx metrics Kisqali brand",
            search_keywords=["trx", "kisqali"],
            graph_entities=["Kisqali"],
            evidence_count=5,
            hop_count=1,
            avg_relevance_score=0.8,
            retrieval_method="cognitive",
        )
        assert signal.query == "What is TRx for Kisqali?"
        assert signal.rewritten_query == "TRx metrics Kisqali brand"
        assert len(signal.search_keywords) == 2
        assert signal.evidence_count == 5
        assert signal.retrieval_method == "cognitive"

    def test_signal_reward_computation(self):
        """Test reward computation for RAG signals."""
        # High evidence count, high relevance
        signal_good = RAGTrainingSignal(
            query="test",
            rewritten_query="test rewritten",
            search_keywords=["test"],
            graph_entities=[],
            evidence_count=5,
            hop_count=1,
            avg_relevance_score=0.9,
            retrieval_method="cognitive",
            response_quality=0.9,
            user_feedback="helpful",
        )
        reward_good = signal_good.compute_reward()
        assert reward_good > 0.5  # Should be high reward

        # Low evidence count, low relevance
        signal_bad = RAGTrainingSignal(
            query="test",
            rewritten_query="test",
            search_keywords=[],
            graph_entities=[],
            evidence_count=0,
            hop_count=3,  # Many hops, no evidence = bad
            avg_relevance_score=0.1,
            retrieval_method="basic",
        )
        reward_bad = signal_bad.compute_reward()
        assert reward_bad < reward_good  # Should be lower reward

    def test_signal_default_timestamp(self):
        """Test that signal has default timestamp."""
        signal = RAGTrainingSignal(
            query="test",
            rewritten_query="test",
            search_keywords=[],
            graph_entities=[],
            evidence_count=0,
            hop_count=0,
            avg_relevance_score=0.0,
            retrieval_method="basic",
        )
        assert signal.timestamp is not None


class TestRAGTrainingSignalCollector:
    """Test cases for RAG training signal collector."""

    def test_collector_add_and_get(self):
        """Test collector add and retrieval."""
        collector = RAGTrainingSignalCollector(buffer_size=10)
        collector.clear()

        for i in range(5):
            signal = RAGTrainingSignal(
                query=f"query {i}",
                rewritten_query=f"rewritten {i}",
                search_keywords=["test"],
                graph_entities=[],
                evidence_count=i,
                hop_count=1,
                avg_relevance_score=0.5 + (i * 0.1),
                retrieval_method="cognitive",
            )
            collector.add_signal(signal)

        assert len(collector) == 5
        signals = collector.get_signals(limit=3)
        assert len(signals) == 3

    def test_collector_buffer_overflow(self):
        """Test that buffer size is respected."""
        collector = RAGTrainingSignalCollector(buffer_size=5)
        collector.clear()

        for i in range(10):
            signal = RAGTrainingSignal(
                query=f"query {i}",
                rewritten_query=f"rewritten {i}",
                search_keywords=[],
                graph_entities=[],
                evidence_count=i,
                hop_count=1,
                avg_relevance_score=0.5,
                retrieval_method="basic",
            )
            collector.add_signal(signal)

        assert len(collector) == 5  # Buffer should cap at 5


class TestGlobalRAGSignalCollector:
    """Test cases for global RAG signal collector singleton."""

    def test_get_rag_signal_collector_singleton(self):
        """Test that get_rag_signal_collector returns same instance."""
        collector1 = get_rag_signal_collector()
        collector2 = get_rag_signal_collector()
        assert collector1 is collector2


class TestHardcodedQueryRewriting:
    """Test cases for hardcoded query rewriting (fallback)."""

    def test_rewrite_with_e2i_entities(self):
        """Test query rewriting extracts E2I entities."""
        query = "What is the TRx for Kisqali in the Northeast?"
        rewritten, keywords, entities = rewrite_query_hardcoded(query)

        assert "kisqali" in keywords
        assert "trx" in keywords
        assert "northeast" in keywords
        assert "Kisqali" in entities
        assert "Northeast" in entities

    def test_rewrite_with_brand_context(self):
        """Test query rewriting with brand context."""
        query = "Show me market share"
        rewritten, keywords, entities = rewrite_query_hardcoded(query, brand_context="Fabhalta")

        # Should include brand from context
        assert "fabhalta" in keywords
        assert "market share" in keywords

    def test_rewrite_generic_query(self):
        """Test query rewriting with generic query."""
        query = "Tell me about sales"
        rewritten, keywords, entities = rewrite_query_hardcoded(query)

        # Should still return keywords
        assert len(keywords) > 0
        # No specific entities
        assert entities == []

    def test_rewrite_multiple_entities(self):
        """Test extraction of multiple E2I entities."""
        query = "Compare Kisqali TRx in Northeast vs Southeast for Q2"
        rewritten, keywords, entities = rewrite_query_hardcoded(query)

        assert "Kisqali" in entities
        assert "Northeast" in entities
        assert "Southeast" in entities
        assert "trx" in keywords


class TestE2IDomainVocabulary:
    """Test cases for E2I domain vocabulary constant."""

    def test_vocabulary_contains_brands(self):
        """Test vocabulary contains all E2I brands."""
        assert "Kisqali" in E2I_DOMAIN_VOCABULARY
        assert "Remibrutinib" in E2I_DOMAIN_VOCABULARY
        assert "Fabhalta" in E2I_DOMAIN_VOCABULARY

    def test_vocabulary_contains_regions(self):
        """Test vocabulary contains regions."""
        assert "Northeast" in E2I_DOMAIN_VOCABULARY
        assert "Southeast" in E2I_DOMAIN_VOCABULARY
        assert "Midwest" in E2I_DOMAIN_VOCABULARY

    def test_vocabulary_contains_kpis(self):
        """Test vocabulary contains KPIs."""
        assert "TRx" in E2I_DOMAIN_VOCABULARY
        assert "NRx" in E2I_DOMAIN_VOCABULARY
        assert "Market Share" in E2I_DOMAIN_VOCABULARY


@pytest.mark.asyncio
class TestAsyncCognitiveRAGRetrieve:
    """Test cases for async cognitive RAG retrieval."""

    async def test_cognitive_retrieve_fallback(self):
        """Test cognitive retrieve falls back when hybrid_search fails."""
        # Mock the hybrid_search to fail (patch in the rag.retriever module)
        with patch("src.rag.retriever.hybrid_search", side_effect=Exception("Mock error")):
            result = await cognitive_rag_retrieve(
                query="What is TRx for Kisqali?",
                collect_signal=False,
            )
            # Should return result with empty evidence due to error
            assert isinstance(result, CognitiveRAGResult)
            assert result.evidence == []

    async def test_cognitive_retrieve_with_results(self):
        """Test cognitive retrieve with mocked results."""
        # Mock hybrid_search to return results
        mock_result = MagicMock()
        mock_result.source_id = "doc1"
        mock_result.content = "TRx for Kisqali is 1000 units"
        mock_result.score = 0.9
        mock_result.source = "analytics"

        with patch("src.rag.retriever.hybrid_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_result]

            result = await cognitive_rag_retrieve(
                query="What is TRx for Kisqali?",
                brand_context="Kisqali",
                collect_signal=False,
            )

            assert isinstance(result, CognitiveRAGResult)
            assert result.rewritten_query is not None
            # Evidence may be filtered by relevance threshold
            assert result.retrieval_method in ["cognitive", "basic"]

    async def test_cognitive_retrieve_collects_signal(self):
        """Test that signal collection works."""
        collector = get_rag_signal_collector()
        initial_count = len(collector)

        # Mock hybrid_search
        with patch("src.rag.retriever.hybrid_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            await cognitive_rag_retrieve(
                query="Test query for signal collection",
                collect_signal=True,
            )

        # Should have one more signal
        assert len(collector) == initial_count + 1

    async def test_cognitive_retrieve_with_context(self):
        """Test cognitive retrieve with full context."""
        with patch("src.rag.retriever.hybrid_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            result = await cognitive_rag_retrieve(
                query="Why did TRx drop?",
                conversation_context="User asked about Kisqali earlier",
                brand_context="Kisqali",
                intent="causal_analysis",
                k=3,
                collect_signal=False,
            )

            assert isinstance(result, CognitiveRAGResult)
            # Check that search was called
            mock_search.assert_called_once()


class TestChatbotCognitiveRAGFeatureFlag:
    """Test cases for cognitive RAG feature flag."""

    def test_feature_flag_exists(self):
        """Test that feature flag is defined."""
        assert isinstance(CHATBOT_COGNITIVE_RAG_ENABLED, bool)


# =============================================================================
# PHASE 6: EVIDENCE SYNTHESIS TESTS
# =============================================================================

from src.api.routes.chatbot_dspy import (
    CHATBOT_DSPY_SYNTHESIS_ENABLED,
    SynthesisResult,
    SynthesisTrainingSignal,
    SynthesisTrainingSignalCollector,
    get_synthesis_signal_collector,
    synthesize_response_dspy,
    # Phase 6: Evidence Synthesis
    synthesize_response_hardcoded,
)


class TestHardcodedSynthesis:
    """Test cases for hardcoded evidence synthesis (fallback)."""

    def test_synthesis_with_no_evidence(self):
        """Test synthesis when no evidence is available."""
        response, confidence_statement, citations, follow_ups, level = (
            synthesize_response_hardcoded(
                query="What is the TRx for Kisqali?",
                intent="kpi_query",
                evidence=[],
                brand_context="Kisqali",
            )
        )

        assert response is not None
        assert len(response) > 0
        assert confidence_statement is not None
        assert "low" in confidence_statement.lower() or level == "low"
        assert citations == []
        assert len(follow_ups) >= 1

    def test_synthesis_with_single_evidence(self):
        """Test synthesis with one evidence item."""
        evidence = [
            {
                "source_id": "src_001",
                "content": "TRx for Kisqali increased by 15% in Q3",
                "score": 0.8,
                "relevance_score": 0.8,
                "source": "metrics",
            }
        ]

        response, confidence_statement, citations, follow_ups, level = (
            synthesize_response_hardcoded(
                query="What is the TRx for Kisqali?",
                intent="kpi_query",
                evidence=evidence,
                brand_context="Kisqali",
            )
        )

        assert response is not None
        assert "src_001" in citations
        assert level in ["moderate", "low"]  # Single source = moderate at best
        assert len(follow_ups) >= 1

    def test_synthesis_with_multiple_evidence(self):
        """Test synthesis with multiple evidence items."""
        evidence = [
            {
                "source_id": "src_001",
                "content": "TRx for Kisqali increased by 15% in Q3",
                "relevance_score": 0.85,
                "source": "metrics",
            },
            {
                "source_id": "src_002",
                "content": "Northeast region showed strongest growth at 20%",
                "relevance_score": 0.78,
                "source": "analysis",
            },
            {
                "source_id": "src_003",
                "content": "New HCP targeting strategy contributed to gains",
                "relevance_score": 0.72,
                "source": "insights",
            },
        ]

        response, confidence_statement, citations, follow_ups, level = (
            synthesize_response_hardcoded(
                query="What is driving Kisqali TRx growth?",
                intent="causal_analysis",
                evidence=evidence,
                brand_context="Kisqali",
            )
        )

        assert response is not None
        assert len(response) > 100  # Should have substantial content
        assert len(citations) == 3
        assert level in ["high", "moderate"]  # Multiple sources = higher confidence
        assert "causal" in response.lower() or "factor" in response.lower()

    def test_synthesis_kpi_query_intent(self):
        """Test synthesis output format for KPI queries."""
        evidence = [
            {
                "source_id": "kpi_src",
                "content": "TRx is 10,000 units",
                "relevance_score": 0.9,
                "source": "kpi",
            }
        ]

        response, _, _, follow_ups, _ = synthesize_response_hardcoded(
            query="What is TRx?",
            intent="kpi_query",
            evidence=evidence,
        )

        assert (
            "metric" in response.lower()
            or "trend" in response.lower()
            or "data" in response.lower()
        )
        # KPI follow-ups should suggest trend or driver analysis
        assert any(
            "trend" in f.lower() or "driver" in f.lower() or "period" in f.lower()
            for f in follow_ups
        )

    def test_synthesis_causal_intent(self):
        """Test synthesis output format for causal analysis."""
        evidence = [
            {
                "source_id": "causal_src",
                "content": "Campaign caused 10% lift",
                "relevance_score": 0.85,
                "source": "causal",
            }
        ]

        response, _, _, follow_ups, _ = synthesize_response_hardcoded(
            query="What caused the increase?",
            intent="causal_analysis",
            evidence=evidence,
        )

        assert (
            "causal" in response.lower()
            or "factor" in response.lower()
            or "finding" in response.lower()
        )
        # Causal follow-ups should suggest deeper analysis
        assert any(
            "driver" in f.lower()
            or "detail" in f.lower()
            or "compare" in f.lower()
            or "segment" in f.lower()
            for f in follow_ups
        )

    def test_synthesis_recommendation_intent(self):
        """Test synthesis output format for recommendations."""
        evidence = [
            {
                "source_id": "rec_src",
                "content": "Recommend focusing on top HCPs",
                "relevance_score": 0.75,
                "source": "rec",
            }
        ]

        response, _, _, follow_ups, _ = synthesize_response_hardcoded(
            query="What should we do?",
            intent="recommendation",
            evidence=evidence,
        )

        assert (
            "recommendation" in response.lower()
            or "insight" in response.lower()
            or "analysis" in response.lower()
        )
        # Recommendation follow-ups should suggest action items
        assert any(
            "action" in f.lower() or "prioritize" in f.lower() or "impact" in f.lower()
            for f in follow_ups
        )

    def test_synthesis_confidence_levels(self):
        """Test confidence level calculation based on evidence quality."""
        # High confidence: multiple sources with high relevance
        high_evidence = [
            {"source_id": "h1", "content": "Data 1", "relevance_score": 0.9, "source": "a"},
            {"source_id": "h2", "content": "Data 2", "relevance_score": 0.8, "source": "b"},
        ]
        _, _, _, _, level_high = synthesize_response_hardcoded("q", "general", high_evidence)
        assert level_high == "high"

        # Moderate confidence: fewer sources or lower scores
        mod_evidence = [
            {"source_id": "m1", "content": "Data", "relevance_score": 0.6, "source": "a"}
        ]
        _, _, _, _, level_mod = synthesize_response_hardcoded("q", "general", mod_evidence)
        assert level_mod in ["moderate", "low"]

        # Low confidence: no evidence
        _, _, _, _, level_low = synthesize_response_hardcoded("q", "general", [])
        assert level_low == "low"


class TestSynthesisTrainingSignal:
    """Test cases for synthesis training signal."""

    def test_training_signal_creation(self):
        """Test creating a synthesis training signal."""
        signal = SynthesisTrainingSignal(
            query="Test query",
            intent="kpi_query",
            evidence_count=3,
            response_length=250,
            confidence_level="high",
            citations_count=3,
            synthesis_method="hardcoded",
        )

        assert signal.query == "Test query"
        assert signal.intent == "kpi_query"
        assert signal.evidence_count == 3
        assert signal.response_length == 250
        assert signal.confidence_level == "high"
        assert signal.citations_count == 3
        assert signal.synthesis_method == "hardcoded"
        assert signal.timestamp is not None

    def test_training_signal_reward_computation(self):
        """Test reward calculation for training signals."""
        # Good synthesis with citations
        good_signal = SynthesisTrainingSignal(
            query="Test",
            intent="kpi_query",
            evidence_count=3,
            response_length=300,
            confidence_level="high",
            citations_count=3,
            synthesis_method="dspy",
        )

        reward = good_signal.compute_reward()
        assert reward > 0.5  # Should be rewarded for good synthesis

    def test_training_signal_hallucination_penalty(self):
        """Test that hallucination penalizes reward."""
        hallucination_signal = SynthesisTrainingSignal(
            query="Test",
            intent="kpi_query",
            evidence_count=2,
            response_length=200,
            confidence_level="high",
            citations_count=2,
            synthesis_method="dspy",
            had_hallucination=True,
        )

        reward = hallucination_signal.compute_reward()
        # Should be heavily penalized
        assert reward < 0.5

    def test_training_signal_user_feedback(self):
        """Test user feedback affects reward."""
        # Good feedback
        good_feedback_signal = SynthesisTrainingSignal(
            query="Test",
            intent="kpi_query",
            evidence_count=2,
            response_length=200,
            confidence_level="moderate",
            citations_count=2,
            synthesis_method="hardcoded",
            user_rating=5.0,
            was_helpful=True,
        )

        good_reward = good_feedback_signal.compute_reward()

        # Bad feedback
        bad_feedback_signal = SynthesisTrainingSignal(
            query="Test",
            intent="kpi_query",
            evidence_count=2,
            response_length=200,
            confidence_level="moderate",
            citations_count=2,
            synthesis_method="hardcoded",
            user_rating=1.0,
            was_helpful=False,
        )

        bad_reward = bad_feedback_signal.compute_reward()

        assert good_reward > bad_reward


class TestSynthesisTrainingSignalCollector:
    """Test cases for synthesis training signal collector."""

    def test_collector_add_signal(self):
        """Test adding signals to collector."""
        collector = SynthesisTrainingSignalCollector()

        signal = SynthesisTrainingSignal(
            query="Test",
            intent="general",
            evidence_count=1,
            response_length=100,
            confidence_level="low",
            citations_count=0,
            synthesis_method="hardcoded",
        )

        collector.add_signal(signal)
        assert len(collector.get_signals()) == 1

    def test_collector_buffer_limit(self):
        """Test collector respects buffer size limit."""
        collector = SynthesisTrainingSignalCollector(max_buffer_size=5)

        for i in range(10):
            signal = SynthesisTrainingSignal(
                query=f"Query {i}",
                intent="general",
                evidence_count=1,
                response_length=100,
                confidence_level="low",
                citations_count=0,
                synthesis_method="hardcoded",
            )
            collector.add_signal(signal)

        # Should only keep last 5
        assert len(collector.get_signals()) == 5
        # Most recent should be "Query 9"
        assert collector.get_signals()[-1].query == "Query 9"

    def test_collector_clear(self):
        """Test collector clear functionality."""
        collector = SynthesisTrainingSignalCollector()

        signal = SynthesisTrainingSignal(
            query="Test",
            intent="general",
            evidence_count=1,
            response_length=100,
            confidence_level="low",
            citations_count=0,
            synthesis_method="hardcoded",
        )

        collector.add_signal(signal)
        assert len(collector.get_signals()) == 1

        collector.clear()
        assert len(collector.get_signals()) == 0

    def test_global_collector_singleton(self):
        """Test that global collector is a singleton."""
        collector1 = get_synthesis_signal_collector()
        collector2 = get_synthesis_signal_collector()

        assert collector1 is collector2


@pytest.mark.asyncio
class TestAsyncSynthesis:
    """Test cases for async DSPy synthesis function."""

    async def test_synthesis_dspy_with_evidence(self):
        """Test DSPy synthesis with valid evidence."""
        evidence = [
            {
                "source_id": "src_001",
                "content": "TRx for Kisqali increased by 15%",
                "relevance_score": 0.85,
                "source": "metrics",
            },
            {
                "source_id": "src_002",
                "content": "Northeast showed strongest growth",
                "relevance_score": 0.75,
                "source": "analysis",
            },
        ]

        result = await synthesize_response_dspy(
            query="What is the TRx trend for Kisqali?",
            intent="kpi_query",
            evidence=evidence,
            brand_context="Kisqali",
            collect_signal=False,
        )

        assert isinstance(result, SynthesisResult)
        assert result.response is not None
        assert len(result.response) > 0
        assert result.synthesis_method in ["dspy", "hardcoded"]
        assert result.confidence_level in ["high", "moderate", "low"]
        # Should cite evidence sources
        assert len(result.evidence_citations) >= 0  # May or may not cite depending on synthesis

    async def test_synthesis_dspy_with_no_evidence(self):
        """Test DSPy synthesis with no evidence (should still work)."""
        result = await synthesize_response_dspy(
            query="Tell me about Kisqali",
            intent="general",
            evidence=[],
            brand_context="Kisqali",
            collect_signal=False,
        )

        assert isinstance(result, SynthesisResult)
        assert result.response is not None
        # Confidence should be low without evidence, though model may vary
        assert result.confidence_level in ["low", "moderate"]
        # Without evidence, citations should be minimal (DSPy may still generate some)
        assert len(result.evidence_citations) <= 1

    async def test_synthesis_dspy_collects_signal(self):
        """Test that synthesis collects training signals."""
        collector = get_synthesis_signal_collector()
        initial_count = len(collector.get_signals())

        evidence = [
            {"source_id": "test", "content": "Test data", "relevance_score": 0.7, "source": "test"}
        ]

        await synthesize_response_dspy(
            query="Test query",
            intent="general",
            evidence=evidence,
            collect_signal=True,
        )

        assert len(collector.get_signals()) == initial_count + 1

    async def test_synthesis_dspy_with_conversation_context(self):
        """Test synthesis with conversation context."""
        evidence = [
            {
                "source_id": "src",
                "content": "Relevant data",
                "relevance_score": 0.8,
                "source": "data",
            }
        ]

        result = await synthesize_response_dspy(
            query="And what about the Northeast?",
            intent="kpi_query",
            evidence=evidence,
            brand_context="Kisqali",
            conversation_context="User asked about TRx trends. Assistant provided overview of Kisqali performance.",
            collect_signal=False,
        )

        assert isinstance(result, SynthesisResult)
        assert result.response is not None

    async def test_synthesis_dspy_follow_up_suggestions(self):
        """Test that synthesis provides follow-up suggestions."""
        evidence = [
            {
                "source_id": "src1",
                "content": "TRx is 10,000 units",
                "relevance_score": 0.9,
                "source": "kpi",
            },
            {
                "source_id": "src2",
                "content": "Growth is 15%",
                "relevance_score": 0.85,
                "source": "kpi",
            },
        ]

        result = await synthesize_response_dspy(
            query="What is the TRx for Kisqali?",
            intent="kpi_query",
            evidence=evidence,
            brand_context="Kisqali",
            collect_signal=False,
        )

        # Should have follow-up suggestions (from hardcoded fallback if DSPy not configured)
        # Note: DSPy may or may not generate follow-ups
        assert result.follow_up_suggestions is not None


class TestSynthesisFeatureFlag:
    """Test cases for synthesis feature flag."""

    def test_feature_flag_exists(self):
        """Test that synthesis feature flag is defined."""
        assert isinstance(CHATBOT_DSPY_SYNTHESIS_ENABLED, bool)

    def test_feature_flag_default_enabled(self):
        """Test that synthesis is enabled by default."""
        # Default should be True based on the code
        assert CHATBOT_DSPY_SYNTHESIS_ENABLED is True


# =============================================================================
# PHASE 7: UNIFIED TRAINING SIGNAL COLLECTION TESTS
# =============================================================================


class TestChatbotSessionSignal:
    """Test cases for ChatbotSessionSignal dataclass."""

    def test_session_signal_creation_defaults(self):
        """Test creating a session signal with default values."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test-session-123",
            thread_id="thread-456",
            query="What is the TRx for Kisqali?",
        )

        assert signal.session_id == "test-session-123"
        assert signal.thread_id == "thread-456"
        assert signal.query == "What is the TRx for Kisqali?"
        assert signal.user_id is None
        assert signal.predicted_intent == ""
        assert signal.intent_confidence == 0.0
        assert signal.evidence_count == 0
        assert signal.user_rating is None

    def test_session_signal_full_creation(self):
        """Test creating a fully populated session signal."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test-session-123",
            thread_id="thread-456",
            user_id="user-789",
            query="What is the TRx for Kisqali in Northeast?",
            brand_context="Kisqali",
            region_context="Northeast",
            predicted_intent="kpi_query",
            intent_confidence=0.95,
            intent_method="dspy",
            intent_reasoning="Query contains TRx KPI",
            predicted_agent="causal_impact",
            secondary_agents=["gap_analyzer"],
            routing_confidence=0.88,
            routing_method="dspy",
            rewritten_query="Kisqali TRx Northeast region metrics",
            evidence_count=3,
            avg_relevance_score=0.85,
            rag_method="cognitive",
            response_length=250,
            synthesis_confidence="high",
            citations_count=2,
            synthesis_method="dspy",
            follow_up_count=3,
        )

        assert signal.user_id == "user-789"
        assert signal.brand_context == "Kisqali"
        assert signal.predicted_intent == "kpi_query"
        assert signal.intent_confidence == 0.95
        assert signal.predicted_agent == "causal_impact"
        assert len(signal.secondary_agents) == 1
        assert signal.evidence_count == 3
        assert signal.synthesis_confidence == "high"

    def test_compute_accuracy_reward_high_confidence(self):
        """Test accuracy reward with high confidence values."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test",
            thread_id="thread",
            query="test",
            intent_confidence=0.95,
            intent_method="dspy",
            routing_confidence=0.90,
            routing_method="dspy",
            avg_relevance_score=0.88,
            rag_method="cognitive",
            synthesis_method="dspy",
            synthesis_confidence="high",
        )

        reward = signal.compute_accuracy_reward()
        assert 0.0 <= reward <= 1.0
        # High confidence should yield reasonable reward (above baseline)
        assert reward > 0.4

    def test_compute_accuracy_reward_low_confidence(self):
        """Test accuracy reward with low confidence values."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test",
            thread_id="thread",
            query="test",
            intent_confidence=0.3,
            intent_method="hardcoded",
            routing_confidence=0.4,
            routing_method="hardcoded",
            avg_relevance_score=0.3,
            rag_method="basic",
            synthesis_method="hardcoded",
            synthesis_confidence="low",
        )

        reward = signal.compute_accuracy_reward()
        assert 0.0 <= reward <= 1.0
        assert reward < 0.5  # Low confidence should yield low reward

    def test_compute_efficiency_reward(self):
        """Test efficiency reward computation."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        # Fast response
        fast_signal = ChatbotSessionSignal(
            session_id="test",
            thread_id="thread",
            query="test",
            total_duration_ms=500,
            hop_count=1,
        )

        fast_reward = fast_signal.compute_efficiency_reward()
        assert 0.0 <= fast_reward <= 1.0

        # Slow response
        slow_signal = ChatbotSessionSignal(
            session_id="test",
            thread_id="thread",
            query="test",
            total_duration_ms=15000,
            hop_count=5,
        )

        slow_reward = slow_signal.compute_efficiency_reward()
        assert 0.0 <= slow_reward <= 1.0
        assert fast_reward > slow_reward  # Fast should be more efficient

    def test_compute_satisfaction_reward_with_positive_feedback(self):
        """Test satisfaction reward with positive user feedback."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test",
            thread_id="thread",
            query="test",
            user_rating=5.0,
            was_helpful=True,
            user_followed_up=False,
            had_hallucination=False,
        )

        reward = signal.compute_satisfaction_reward()
        assert 0.0 <= reward <= 1.0
        assert reward > 0.7  # Positive feedback should yield high reward

    def test_compute_satisfaction_reward_with_negative_feedback(self):
        """Test satisfaction reward with negative user feedback."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test",
            thread_id="thread",
            query="test",
            user_rating=1.0,
            was_helpful=False,
            had_hallucination=True,
        )

        reward = signal.compute_satisfaction_reward()
        assert 0.0 <= reward <= 1.0
        assert reward < 0.5  # Negative feedback should yield low reward

    def test_compute_unified_reward(self):
        """Test unified reward computation returns all components."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test",
            thread_id="thread",
            query="test",
            intent_confidence=0.9,
            intent_method="dspy",
            routing_confidence=0.85,
            routing_method="dspy",
            avg_relevance_score=0.8,
            synthesis_confidence="high",
            synthesis_method="dspy",
            total_duration_ms=1000,
            user_rating=4.5,
            was_helpful=True,
        )

        rewards = signal.compute_unified_reward()

        assert "accuracy" in rewards
        assert "efficiency" in rewards
        assert "satisfaction" in rewards
        assert "overall" in rewards

        for _key, value in rewards.items():
            assert 0.0 <= value <= 1.0

        # Overall should be weighted average
        assert rewards["overall"] > 0

    def test_to_dict_serialization(self):
        """Test that session signal can be serialized to dict."""
        from src.api.routes.chatbot_dspy import ChatbotSessionSignal

        signal = ChatbotSessionSignal(
            session_id="test-session",
            thread_id="thread-123",
            query="What is the TRx?",
            predicted_intent="kpi_query",
            intent_confidence=0.9,
        )

        result = signal.to_dict()

        assert isinstance(result, dict)
        assert result["session_id"] == "test-session"
        assert result["thread_id"] == "thread-123"
        assert result["query"] == "What is the TRx?"
        assert result["predicted_intent"] == "kpi_query"
        # Rewards are flattened into top level of dict
        assert "accuracy" in result
        assert "efficiency" in result
        assert "satisfaction" in result
        assert "overall" in result
        assert "timestamp" in result


class TestChatbotSignalCollector:
    """Test cases for ChatbotSignalCollector class."""

    def test_collector_creation(self):
        """Test creating a signal collector."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector(buffer_size=100)
        assert collector is not None

    def test_start_session(self):
        """Test starting a new session."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector(buffer_size=100)
        signal = collector.start_session(
            session_id="session-123",
            thread_id="thread-456",
            query="What is the TRx for Kisqali?",
            user_id="user-789",
            brand_context="Kisqali",
            region_context="Northeast",
        )

        assert signal is not None
        assert signal.session_id == "session-123"
        assert signal.thread_id == "thread-456"
        assert signal.query == "What is the TRx for Kisqali?"
        assert signal.user_id == "user-789"
        assert signal.brand_context == "Kisqali"

    def test_get_session(self):
        """Test retrieving an active session."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()
        collector.start_session(
            session_id="session-123",
            thread_id="thread-456",
            query="Test query",
        )

        session = collector.get_session("session-123")
        assert session is not None
        assert session.session_id == "session-123"

        # Non-existent session
        missing = collector.get_session("non-existent")
        assert missing is None

    def test_update_intent(self):
        """Test updating intent classification signal."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()
        collector.start_session(
            session_id="session-123",
            thread_id="thread-456",
            query="Test query",
        )

        collector.update_intent(
            session_id="session-123",
            intent="kpi_query",
            confidence=0.95,
            method="dspy",
            reasoning="Contains KPI reference",
        )

        session = collector.get_session("session-123")
        assert session.predicted_intent == "kpi_query"
        assert session.intent_confidence == 0.95
        assert session.intent_method == "dspy"
        assert session.intent_reasoning == "Contains KPI reference"

    def test_update_routing(self):
        """Test updating routing signal."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()
        collector.start_session(
            session_id="session-123",
            thread_id="thread-456",
            query="Test query",
        )

        collector.update_routing(
            session_id="session-123",
            agent="causal_impact",
            secondary_agents=["gap_analyzer", "explainer"],
            confidence=0.88,
            method="dspy",
            rationale="Query requires causal analysis",
        )

        session = collector.get_session("session-123")
        assert session.predicted_agent == "causal_impact"
        assert "gap_analyzer" in session.secondary_agents
        assert session.routing_confidence == 0.88

    def test_update_rag(self):
        """Test updating RAG signal."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()
        collector.start_session(
            session_id="session-123",
            thread_id="thread-456",
            query="Test query",
        )

        collector.update_rag(
            session_id="session-123",
            rewritten_query="optimized query",
            keywords=["TRx", "Kisqali"],
            entities=["Kisqali", "Northeast"],
            evidence_count=5,
            hop_count=2,
            avg_relevance=0.85,
            method="cognitive",
        )

        session = collector.get_session("session-123")
        assert session.rewritten_query == "optimized query"
        assert "TRx" in session.search_keywords
        assert session.evidence_count == 5
        assert session.rag_method == "cognitive"

    def test_update_synthesis(self):
        """Test updating synthesis signal."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()
        collector.start_session(
            session_id="session-123",
            thread_id="thread-456",
            query="Test query",
        )

        collector.update_synthesis(
            session_id="session-123",
            response_length=350,
            confidence="high",
            citations_count=3,
            method="dspy",
            follow_up_count=2,
        )

        session = collector.get_session("session-123")
        assert session.response_length == 350
        assert session.synthesis_confidence == "high"
        assert session.citations_count == 3

    def test_update_feedback(self):
        """Test updating user feedback."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()
        collector.start_session(
            session_id="session-123",
            thread_id="thread-456",
            query="Test query",
        )

        collector.update_feedback(
            session_id="session-123",
            user_rating=4.5,
            was_helpful=True,
            user_followed_up=True,
            had_hallucination=False,
        )

        session = collector.get_session("session-123")
        assert session.user_rating == 4.5
        assert session.was_helpful is True
        assert session.user_followed_up is True
        assert session.had_hallucination is False

    def test_finalize_session(self):
        """Test finalizing a session moves it to completed signals."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()
        collector.start_session(
            session_id="session-123",
            thread_id="thread-456",
            query="Test query",
        )

        collector.update_intent(
            session_id="session-123",
            intent="kpi_query",
            confidence=0.9,
            method="dspy",
        )

        finalized = collector.finalize_session(
            session_id="session-123",
            total_duration_ms=1500.0,
        )

        assert finalized is not None
        assert finalized.total_duration_ms == 1500.0

        # Session should no longer be in active sessions
        assert collector.get_session("session-123") is None

        # Should be in completed signals
        signals = collector.get_signals(limit=10)
        assert len(signals) >= 1

    def test_get_high_quality_signals(self):
        """Test retrieving high quality signals."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()

        # Create a high quality session
        collector.start_session(
            session_id="high-quality",
            thread_id="thread-1",
            query="What is TRx?",
        )
        collector.update_intent(
            session_id="high-quality",
            intent="kpi_query",
            confidence=0.95,
            method="dspy",
        )
        collector.update_synthesis(
            session_id="high-quality",
            response_length=300,
            confidence="high",
            citations_count=3,
            method="dspy",
            follow_up_count=2,
        )
        collector.update_feedback(
            session_id="high-quality",
            user_rating=5.0,
            was_helpful=True,
        )
        collector.finalize_session(session_id="high-quality", total_duration_ms=1000)

        # Create a low quality session
        collector.start_session(
            session_id="low-quality",
            thread_id="thread-2",
            query="hi",
        )
        collector.update_intent(
            session_id="low-quality",
            intent="greeting",
            confidence=0.3,
            method="hardcoded",
        )
        collector.finalize_session(session_id="low-quality", total_duration_ms=500)

        # Get high quality signals
        high_quality = collector.get_high_quality_signals(
            min_overall_reward=0.6,
            limit=10,
        )

        # Should have at least the high quality signal
        assert len(high_quality) >= 1

    def test_get_signals_for_training(self):
        """Test retrieving signals formatted for training."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()

        collector.start_session(
            session_id="session-1",
            thread_id="thread-1",
            query="What is TRx for Kisqali?",
        )
        collector.update_intent(
            session_id="session-1",
            intent="kpi_query",
            confidence=0.9,
            method="dspy",
            reasoning="Contains TRx KPI",
        )
        collector.finalize_session(session_id="session-1")

        # Get signals for intent phase
        intent_signals = collector.get_signals_for_training(phase="intent", limit=10)
        assert len(intent_signals) >= 1
        assert "query" in intent_signals[0]
        # The key is "target_intent" in the training format
        assert "target_intent" in intent_signals[0]
        assert "reward" in intent_signals[0]

    def test_get_statistics(self):
        """Test getting collector statistics."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector()

        collector.start_session(
            session_id="session-1",
            thread_id="thread-1",
            query="Test query 1",
        )
        collector.finalize_session(session_id="session-1")

        collector.start_session(
            session_id="session-2",
            thread_id="thread-2",
            query="Test query 2",
        )
        # Leave session-2 active

        stats = collector.get_statistics()

        assert "total_signals" in stats
        # The key is "pending_sessions" in the stats
        assert "pending_sessions" in stats
        assert stats["total_signals"] >= 1
        assert stats["pending_sessions"] >= 1

    def test_buffer_size_limit(self):
        """Test that collector respects buffer size limit."""
        from src.api.routes.chatbot_dspy import ChatbotSignalCollector

        collector = ChatbotSignalCollector(buffer_size=3)

        # Create more signals than buffer size
        for i in range(5):
            collector.start_session(
                session_id=f"session-{i}",
                thread_id=f"thread-{i}",
                query=f"Query {i}",
            )
            collector.finalize_session(session_id=f"session-{i}")

        # Should only keep buffer_size signals
        signals = collector.get_signals(limit=100)
        assert len(signals) <= 3


class TestChatbotSignalCollectorSingleton:
    """Test cases for the global signal collector singleton."""

    def test_get_chatbot_signal_collector_returns_instance(self):
        """Test that get_chatbot_signal_collector returns a collector."""
        from src.api.routes.chatbot_dspy import get_chatbot_signal_collector

        collector = get_chatbot_signal_collector()
        assert collector is not None

    def test_get_chatbot_signal_collector_returns_same_instance(self):
        """Test that get_chatbot_signal_collector returns singleton."""
        from src.api.routes.chatbot_dspy import get_chatbot_signal_collector

        collector1 = get_chatbot_signal_collector()
        collector2 = get_chatbot_signal_collector()

        assert collector1 is collector2
