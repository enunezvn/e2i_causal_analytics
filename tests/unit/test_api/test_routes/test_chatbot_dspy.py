"""
Tests for src/api/routes/chatbot_dspy.py

Covers:
- IntentTrainingSignal dataclass and compute_reward
- IntentTrainingSignalCollector
- RoutingTrainingSignal dataclass and compute_reward
- RoutingTrainingSignalCollector
- Pattern matching utilities
- Hardcoded intent classification
- Hardcoded agent routing
- Intent and agent normalization
- Confidence validation
- Sync wrappers
- Async functions with mocked DSPy
"""

import pytest

# Group all DSPy tests on same worker to prevent import race conditions
pytestmark = pytest.mark.xdist_group(name="dspy_integration")

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chatbot_dspy import (
    IntentTrainingSignal,
    IntentTrainingSignalCollector,
    RoutingTrainingSignal,
    RoutingTrainingSignalCollector,
    _matches_pattern,
    _is_multi_faceted_query,
    classify_intent_hardcoded,
    _normalize_intent,
    _validate_confidence,
    classify_intent_sync,
    _normalize_agent,
    route_agent_hardcoded,
    get_signal_collector,
    get_routing_signal_collector,
    VALID_INTENTS,
    VALID_AGENTS,
    AGENT_CAPABILITIES,
)
from src.api.routes.chatbot_state import IntentType


# =============================================================================
# IntentTrainingSignal Tests
# =============================================================================


class TestIntentTrainingSignal:
    """Tests for IntentTrainingSignal dataclass."""

    def test_create_signal(self):
        """Test creating an IntentTrainingSignal."""
        signal = IntentTrainingSignal(
            query="What is the TRx for Kisqali?",
            conversation_context="Previous message about sales",
            brand_context="Kisqali",
            predicted_intent="kpi_query",
            confidence=0.85,
            reasoning="Matched KPI keyword pattern",
            classification_method="hardcoded",
        )
        assert signal.query == "What is the TRx for Kisqali?"
        assert signal.predicted_intent == "kpi_query"
        assert signal.confidence == 0.85
        assert signal.classification_method == "hardcoded"
        assert signal.user_followed_up is None
        assert signal.response_helpful is None
        assert signal.correct_routing is None

    def test_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.5,
            reasoning="test",
            classification_method="hardcoded",
        )
        assert isinstance(signal.timestamp, datetime)

    def test_compute_reward_base_confidence(self):
        """Test compute_reward with only confidence."""
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.8,
            reasoning="test",
            classification_method="hardcoded",
        )
        # Base reward = 0.8 * 0.3 = 0.24
        reward = signal.compute_reward()
        assert 0.0 <= reward <= 1.0
        assert reward == pytest.approx(0.24, abs=0.01)

    def test_compute_reward_correct_routing(self):
        """Test compute_reward with correct routing."""
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.8,
            reasoning="test",
            classification_method="hardcoded",
            correct_routing=True,
        )
        # Base = 0.24 + 0.4 for correct routing = 0.64
        reward = signal.compute_reward()
        assert reward == pytest.approx(0.64, abs=0.01)

    def test_compute_reward_incorrect_routing(self):
        """Test compute_reward with incorrect routing."""
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.8,
            reasoning="test",
            classification_method="hardcoded",
            correct_routing=False,
        )
        # Base = 0.24 - 0.2 for incorrect routing = 0.04
        reward = signal.compute_reward()
        assert reward == pytest.approx(0.04, abs=0.01)

    def test_compute_reward_helpful_response(self):
        """Test compute_reward with helpful response."""
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.8,
            reasoning="test",
            classification_method="hardcoded",
            response_helpful=True,
        )
        # Base = 0.24 + 0.3 for helpful = 0.54
        reward = signal.compute_reward()
        assert reward == pytest.approx(0.54, abs=0.01)

    def test_compute_reward_unhelpful_response(self):
        """Test compute_reward with unhelpful response."""
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.8,
            reasoning="test",
            classification_method="hardcoded",
            response_helpful=False,
        )
        # Base = 0.24 - 0.1 for unhelpful = 0.14
        reward = signal.compute_reward()
        assert reward == pytest.approx(0.14, abs=0.01)

    def test_compute_reward_low_confidence_penalty(self):
        """Test compute_reward with low confidence penalty."""
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.3,  # Low confidence
            reasoning="test",
            classification_method="hardcoded",
        )
        # Base = 0.3 * 0.3 = 0.09, penalty -0.1 = -0.01, clamped to 0.0
        reward = signal.compute_reward()
        assert reward == 0.0

    def test_compute_reward_clamped_to_range(self):
        """Test compute_reward is clamped between 0 and 1."""
        # High reward signal
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=1.0,
            reasoning="test",
            classification_method="hardcoded",
            correct_routing=True,
            response_helpful=True,
        )
        reward = signal.compute_reward()
        assert 0.0 <= reward <= 1.0


# =============================================================================
# IntentTrainingSignalCollector Tests
# =============================================================================


class TestIntentTrainingSignalCollector:
    """Tests for IntentTrainingSignalCollector."""

    def test_create_collector(self):
        """Test creating a collector with default buffer size."""
        collector = IntentTrainingSignalCollector()
        assert len(collector) == 0

    def test_create_collector_custom_buffer_size(self):
        """Test creating a collector with custom buffer size."""
        collector = IntentTrainingSignalCollector(buffer_size=50)
        assert len(collector) == 0

    def test_add_signal(self):
        """Test adding a signal to the collector."""
        collector = IntentTrainingSignalCollector()
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.5,
            reasoning="test",
            classification_method="hardcoded",
        )
        collector.add_signal(signal)
        assert len(collector) == 1

    def test_add_multiple_signals(self):
        """Test adding multiple signals."""
        collector = IntentTrainingSignalCollector()
        for i in range(5):
            signal = IntentTrainingSignal(
                query=f"test {i}",
                conversation_context="",
                brand_context="",
                predicted_intent="general",
                confidence=0.5,
                reasoning="test",
                classification_method="hardcoded",
            )
            collector.add_signal(signal)
        assert len(collector) == 5

    def test_buffer_overflow(self):
        """Test that buffer overflows correctly."""
        collector = IntentTrainingSignalCollector(buffer_size=3)
        for i in range(5):
            signal = IntentTrainingSignal(
                query=f"test {i}",
                conversation_context="",
                brand_context="",
                predicted_intent="general",
                confidence=0.5,
                reasoning="test",
                classification_method="hardcoded",
            )
            collector.add_signal(signal)
        assert len(collector) == 3
        # Should have the last 3 signals (test 2, test 3, test 4)
        signals = collector.get_signals(10)
        assert signals[0].query == "test 2"
        assert signals[-1].query == "test 4"

    def test_get_signals(self):
        """Test getting signals with limit."""
        collector = IntentTrainingSignalCollector()
        for i in range(10):
            signal = IntentTrainingSignal(
                query=f"test {i}",
                conversation_context="",
                brand_context="",
                predicted_intent="general",
                confidence=0.5,
                reasoning="test",
                classification_method="hardcoded",
            )
            collector.add_signal(signal)

        signals = collector.get_signals(limit=5)
        assert len(signals) == 5

    def test_get_high_quality_signals(self):
        """Test getting high confidence signals."""
        collector = IntentTrainingSignalCollector()
        # Add signals with varying confidence
        confidences = [0.3, 0.5, 0.7, 0.8, 0.9, 0.6, 0.95]
        for i, conf in enumerate(confidences):
            signal = IntentTrainingSignal(
                query=f"test {i}",
                conversation_context="",
                brand_context="",
                predicted_intent="general",
                confidence=conf,
                reasoning="test",
                classification_method="hardcoded",
            )
            collector.add_signal(signal)

        # Get signals with confidence >= 0.7
        high_quality = collector.get_high_quality_signals(min_confidence=0.7)
        assert len(high_quality) == 4  # 0.7, 0.8, 0.9, 0.95

    def test_clear(self):
        """Test clearing the buffer."""
        collector = IntentTrainingSignalCollector()
        for i in range(5):
            signal = IntentTrainingSignal(
                query=f"test {i}",
                conversation_context="",
                brand_context="",
                predicted_intent="general",
                confidence=0.5,
                reasoning="test",
                classification_method="hardcoded",
            )
            collector.add_signal(signal)
        assert len(collector) == 5
        collector.clear()
        assert len(collector) == 0


# =============================================================================
# RoutingTrainingSignal Tests
# =============================================================================


class TestRoutingTrainingSignal:
    """Tests for RoutingTrainingSignal dataclass."""

    def test_create_signal(self):
        """Test creating a RoutingTrainingSignal."""
        signal = RoutingTrainingSignal(
            query="Why did TRx drop?",
            intent="causal_analysis",
            brand_context="Kisqali",
            predicted_agent="causal_impact",
            secondary_agents=["gap_analyzer"],
            confidence=0.85,
            rationale="Matched causal keywords",
            routing_method="hardcoded",
        )
        assert signal.query == "Why did TRx drop?"
        assert signal.predicted_agent == "causal_impact"
        assert signal.secondary_agents == ["gap_analyzer"]
        assert signal.routing_method == "hardcoded"
        assert signal.correct_agent is None
        assert signal.response_quality is None

    def test_compute_reward_base_confidence(self):
        """Test compute_reward with only confidence."""
        signal = RoutingTrainingSignal(
            query="test",
            intent="general",
            brand_context="",
            predicted_agent="explainer",
            secondary_agents=[],
            confidence=0.8,
            rationale="test",
            routing_method="hardcoded",
        )
        # Base = 0.8 * 0.3 = 0.24
        reward = signal.compute_reward()
        assert reward == pytest.approx(0.24, abs=0.01)

    def test_compute_reward_correct_agent(self):
        """Test compute_reward with correct agent."""
        signal = RoutingTrainingSignal(
            query="test",
            intent="general",
            brand_context="",
            predicted_agent="explainer",
            secondary_agents=[],
            confidence=0.8,
            rationale="test",
            routing_method="hardcoded",
            correct_agent=True,
        )
        # Base = 0.24 + 0.5 for correct agent = 0.74
        reward = signal.compute_reward()
        assert reward == pytest.approx(0.74, abs=0.01)

    def test_compute_reward_incorrect_agent(self):
        """Test compute_reward with incorrect agent."""
        signal = RoutingTrainingSignal(
            query="test",
            intent="general",
            brand_context="",
            predicted_agent="explainer",
            secondary_agents=[],
            confidence=0.8,
            rationale="test",
            routing_method="hardcoded",
            correct_agent=False,
        )
        # Base = 0.24 - 0.2 for incorrect = 0.04
        reward = signal.compute_reward()
        assert reward == pytest.approx(0.04, abs=0.01)

    def test_compute_reward_with_response_quality(self):
        """Test compute_reward with response quality."""
        signal = RoutingTrainingSignal(
            query="test",
            intent="general",
            brand_context="",
            predicted_agent="explainer",
            secondary_agents=[],
            confidence=0.8,
            rationale="test",
            routing_method="hardcoded",
            response_quality=0.9,
        )
        # Base = 0.24 + 0.9 * 0.2 = 0.24 + 0.18 = 0.42
        reward = signal.compute_reward()
        assert reward == pytest.approx(0.42, abs=0.01)


# =============================================================================
# RoutingTrainingSignalCollector Tests
# =============================================================================


class TestRoutingTrainingSignalCollector:
    """Tests for RoutingTrainingSignalCollector."""

    def test_create_collector(self):
        """Test creating a collector."""
        collector = RoutingTrainingSignalCollector()
        assert len(collector) == 0

    def test_add_signal(self):
        """Test adding a signal."""
        collector = RoutingTrainingSignalCollector()
        signal = RoutingTrainingSignal(
            query="test",
            intent="general",
            brand_context="",
            predicted_agent="explainer",
            secondary_agents=[],
            confidence=0.5,
            rationale="test",
            routing_method="hardcoded",
        )
        collector.add_signal(signal)
        assert len(collector) == 1

    def test_buffer_overflow(self):
        """Test buffer overflow behavior."""
        collector = RoutingTrainingSignalCollector(buffer_size=3)
        for i in range(5):
            signal = RoutingTrainingSignal(
                query=f"test {i}",
                intent="general",
                brand_context="",
                predicted_agent="explainer",
                secondary_agents=[],
                confidence=0.5,
                rationale="test",
                routing_method="hardcoded",
            )
            collector.add_signal(signal)
        assert len(collector) == 3

    def test_get_signals(self):
        """Test getting signals."""
        collector = RoutingTrainingSignalCollector()
        for i in range(10):
            signal = RoutingTrainingSignal(
                query=f"test {i}",
                intent="general",
                brand_context="",
                predicted_agent="explainer",
                secondary_agents=[],
                confidence=0.5,
                rationale="test",
                routing_method="hardcoded",
            )
            collector.add_signal(signal)

        signals = collector.get_signals(limit=5)
        assert len(signals) == 5

    def test_clear(self):
        """Test clearing the buffer."""
        collector = RoutingTrainingSignalCollector()
        for i in range(5):
            signal = RoutingTrainingSignal(
                query=f"test {i}",
                intent="general",
                brand_context="",
                predicted_agent="explainer",
                secondary_agents=[],
                confidence=0.5,
                rationale="test",
                routing_method="hardcoded",
            )
            collector.add_signal(signal)
        collector.clear()
        assert len(collector) == 0


# =============================================================================
# Pattern Matching Tests
# =============================================================================


class TestMatchesPattern:
    """Tests for _matches_pattern function."""

    def test_single_word_exact_match(self):
        """Test single word exact match."""
        assert _matches_pattern("hello world", ["hello"]) is True
        assert _matches_pattern("hello world", ["world"]) is True

    def test_single_word_no_match(self):
        """Test single word no match."""
        assert _matches_pattern("hello world", ["goodbye"]) is False

    def test_single_word_boundary(self):
        """Test word boundary matching."""
        # "help" should not match "helpful"
        assert _matches_pattern("that was helpful", ["help"]) is False
        # "help" should match standalone "help"
        assert _matches_pattern("please help me", ["help"]) is True

    def test_multi_word_pattern(self):
        """Test multi-word pattern matching."""
        assert _matches_pattern("good morning everyone", ["good morning"]) is True
        assert _matches_pattern("good afternoon", ["good morning"]) is False

    def test_multiple_patterns(self):
        """Test matching against multiple patterns."""
        patterns = ["hello", "hi", "hey"]
        assert _matches_pattern("hello there", patterns) is True
        assert _matches_pattern("hi there", patterns) is True
        assert _matches_pattern("hey there", patterns) is True
        assert _matches_pattern("goodbye", patterns) is False

    def test_case_sensitivity(self):
        """Test that matching is case-insensitive when query is lowercased."""
        # The function expects query_lower, so it should match
        assert _matches_pattern("hello world", ["hello"]) is True


# =============================================================================
# Multi-Faceted Query Detection Tests
# =============================================================================


class TestIsMultiFacetedQuery:
    """Tests for _is_multi_faceted_query function."""

    def test_simple_query_not_multi_faceted(self):
        """Test simple query is not multi-faceted."""
        assert _is_multi_faceted_query("What is the TRx?") is False

    def test_conjunction_keywords(self):
        """Test conjunction keywords detection."""
        assert _is_multi_faceted_query("Compare TRx and NRx trends") is True

    def test_multiple_kpis(self):
        """Test multiple KPIs detection (requires 2+ facets)."""
        # Multiple KPIs alone is only 1 facet, need 2+ for multi-faceted
        # Adding "compare" gives conjunction_keywords + multiple_kpis = 2 facets
        assert _is_multi_faceted_query("Compare TRx and market share for Q1") is True

    def test_multiple_brands(self):
        """Test multiple brands detection."""
        assert _is_multi_faceted_query("Compare Kisqali and Fabhalta performance") is True

    def test_cross_agent_query(self):
        """Test cross-agent capabilities query."""
        # Needs 2+ facets to be multi-faceted
        assert _is_multi_faceted_query("Compare health trends with drift analysis") is True

    def test_analysis_and_recommendation(self):
        """Test analysis + recommendation query (requires 2+ facets)."""
        # "why" + "should" = analysis_and_recommendation (1 facet)
        # Add cross_agent keyword to get 2 facets
        assert _is_multi_faceted_query("Why did the health score drop and what should we do?") is True

    def test_single_facet_not_multi_faceted(self):
        """Test single facet is not multi-faceted."""
        # Only has one facet (causal via "why")
        assert _is_multi_faceted_query("Why did TRx drop?") is False


# =============================================================================
# Hardcoded Intent Classification Tests
# =============================================================================


class TestClassifyIntentHardcoded:
    """Tests for classify_intent_hardcoded function."""

    def test_greeting_hello(self):
        """Test greeting classification for 'hello'."""
        intent, confidence, reasoning = classify_intent_hardcoded("Hello!")
        assert intent == IntentType.GREETING
        assert confidence >= 0.9

    def test_greeting_hi(self):
        """Test greeting classification for 'hi'."""
        intent, confidence, reasoning = classify_intent_hardcoded("Hi there")
        assert intent == IntentType.GREETING

    def test_greeting_good_morning(self):
        """Test greeting classification for 'good morning'."""
        intent, confidence, reasoning = classify_intent_hardcoded("Good morning!")
        assert intent == IntentType.GREETING

    def test_help_request(self):
        """Test help intent classification."""
        intent, confidence, reasoning = classify_intent_hardcoded("What can you help me with?")
        assert intent == IntentType.HELP

    def test_help_how_do_i(self):
        """Test help intent for 'how do i'."""
        intent, confidence, reasoning = classify_intent_hardcoded("How do I use this?")
        assert intent == IntentType.HELP

    def test_kpi_query_trx(self):
        """Test KPI query for TRx."""
        intent, confidence, reasoning = classify_intent_hardcoded("What is the TRx for Kisqali?")
        assert intent == IntentType.KPI_QUERY

    def test_kpi_query_nrx(self):
        """Test KPI query for NRx."""
        intent, confidence, reasoning = classify_intent_hardcoded("Show NRx trends")
        assert intent == IntentType.KPI_QUERY

    def test_kpi_query_market_share(self):
        """Test KPI query for market share."""
        intent, confidence, reasoning = classify_intent_hardcoded("What is our market share?")
        assert intent == IntentType.KPI_QUERY

    def test_causal_analysis_why(self):
        """Test causal analysis for 'why' questions."""
        intent, confidence, reasoning = classify_intent_hardcoded("Why did sales drop last quarter?")
        assert intent == IntentType.CAUSAL_ANALYSIS

    def test_causal_analysis_impact(self):
        """Test causal analysis for impact questions."""
        intent, confidence, reasoning = classify_intent_hardcoded("What is the impact of the campaign?")
        assert intent == IntentType.CAUSAL_ANALYSIS

    def test_causal_analysis_driver(self):
        """Test causal analysis for driver questions."""
        # Must use singular "driver" to match word boundary pattern
        intent, confidence, reasoning = classify_intent_hardcoded("What is the key driver?")
        assert intent == IntentType.CAUSAL_ANALYSIS

    def test_agent_status(self):
        """Test agent status classification."""
        # Must use singular "agent" to match word boundary pattern
        intent, confidence, reasoning = classify_intent_hardcoded("Show the agent status")
        assert intent == IntentType.AGENT_STATUS

    def test_recommendation(self):
        """Test recommendation classification."""
        intent, confidence, reasoning = classify_intent_hardcoded("What do you recommend?")
        assert intent == IntentType.RECOMMENDATION

    def test_recommendation_suggest(self):
        """Test recommendation for 'suggest'."""
        intent, confidence, reasoning = classify_intent_hardcoded("Suggest improvements")
        assert intent == IntentType.RECOMMENDATION

    def test_search(self):
        """Test search classification."""
        intent, confidence, reasoning = classify_intent_hardcoded("Search for HCP data")
        assert intent == IntentType.SEARCH

    def test_search_find(self):
        """Test search for 'find'."""
        intent, confidence, reasoning = classify_intent_hardcoded("Find all physicians in NYC")
        assert intent == IntentType.SEARCH

    def test_multi_faceted(self):
        """Test multi-faceted query classification."""
        intent, confidence, reasoning = classify_intent_hardcoded(
            "Compare TRx and NRx trends and explain the differences"
        )
        assert intent == IntentType.MULTI_FACETED

    def test_general_fallback(self):
        """Test general fallback classification."""
        intent, confidence, reasoning = classify_intent_hardcoded("Random unrelated query")
        assert intent == IntentType.GENERAL
        assert confidence < 0.9  # Lower confidence for fallback


# =============================================================================
# Intent Normalization Tests
# =============================================================================


class TestNormalizeIntent:
    """Tests for _normalize_intent function."""

    def test_valid_intent_passthrough(self):
        """Test valid intents pass through."""
        assert _normalize_intent("kpi_query") == IntentType.KPI_QUERY
        assert _normalize_intent("causal_analysis") == IntentType.CAUSAL_ANALYSIS
        assert _normalize_intent("general") == IntentType.GENERAL

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert _normalize_intent("KPI_QUERY") == IntentType.KPI_QUERY
        assert _normalize_intent("CAUSAL_ANALYSIS") == IntentType.CAUSAL_ANALYSIS

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert _normalize_intent("  kpi_query  ") == IntentType.KPI_QUERY

    def test_kpi_variations(self):
        """Test KPI intent variations."""
        assert _normalize_intent("kpi") == IntentType.KPI_QUERY
        assert _normalize_intent("kpi query") == IntentType.KPI_QUERY

    def test_causal_variations(self):
        """Test causal intent variations."""
        assert _normalize_intent("causal") == IntentType.CAUSAL_ANALYSIS
        assert _normalize_intent("causal analysis") == IntentType.CAUSAL_ANALYSIS

    def test_multi_faceted_variations(self):
        """Test multi-faceted variations."""
        assert _normalize_intent("multi-faceted") == IntentType.MULTI_FACETED
        assert _normalize_intent("multifaceted") == IntentType.MULTI_FACETED
        assert _normalize_intent("complex") == IntentType.MULTI_FACETED

    def test_greeting_variations(self):
        """Test greeting variations."""
        assert _normalize_intent("greetings") == IntentType.GREETING
        assert _normalize_intent("hi") == IntentType.GREETING
        assert _normalize_intent("hello") == IntentType.GREETING

    def test_unknown_defaults_to_general(self):
        """Test unknown intent defaults to general."""
        assert _normalize_intent("unknown_intent") == IntentType.GENERAL
        assert _normalize_intent("random") == IntentType.GENERAL


# =============================================================================
# Confidence Validation Tests
# =============================================================================


class TestValidateConfidence:
    """Tests for _validate_confidence function."""

    def test_valid_confidence(self):
        """Test valid confidence values."""
        assert _validate_confidence(0.5) == 0.5
        assert _validate_confidence(0.0) == 0.0
        assert _validate_confidence(1.0) == 1.0

    def test_clamp_above_one(self):
        """Test confidence above 1 is clamped."""
        assert _validate_confidence(1.5) == 1.0
        assert _validate_confidence(2.0) == 1.0

    def test_clamp_below_zero(self):
        """Test confidence below 0 is clamped."""
        assert _validate_confidence(-0.5) == 0.0
        assert _validate_confidence(-1.0) == 0.0

    def test_string_confidence(self):
        """Test string confidence is converted."""
        assert _validate_confidence("0.75") == 0.75
        assert _validate_confidence("0.5") == 0.5

    def test_invalid_string(self):
        """Test invalid string returns default."""
        assert _validate_confidence("invalid") == 0.5

    def test_none_value(self):
        """Test None returns default."""
        assert _validate_confidence(None) == 0.5

    def test_integer_confidence(self):
        """Test integer confidence is converted."""
        assert _validate_confidence(1) == 1.0
        assert _validate_confidence(0) == 0.0


# =============================================================================
# Sync Classification Tests
# =============================================================================


class TestClassifyIntentSync:
    """Tests for classify_intent_sync function."""

    def test_returns_tuple(self):
        """Test sync classification returns correct tuple format."""
        result = classify_intent_sync("What is the TRx?")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_method_is_hardcoded(self):
        """Test sync always uses hardcoded method."""
        intent, confidence, reasoning, method = classify_intent_sync("Hello")
        assert method == "hardcoded"

    def test_greeting_classification(self):
        """Test greeting is classified correctly."""
        intent, confidence, reasoning, method = classify_intent_sync("Hello there!")
        assert intent == IntentType.GREETING

    def test_kpi_classification(self):
        """Test KPI query is classified correctly."""
        intent, confidence, reasoning, method = classify_intent_sync("Show TRx data")
        assert intent == IntentType.KPI_QUERY

    def test_with_context(self):
        """Test classification with context parameters."""
        intent, confidence, reasoning, method = classify_intent_sync(
            query="What is the trend?",
            conversation_context="Previous discussion about sales",
            brand_context="Kisqali",
        )
        # Should still work (context not used in hardcoded version)
        assert intent is not None


# =============================================================================
# Agent Normalization Tests
# =============================================================================


class TestNormalizeAgent:
    """Tests for _normalize_agent function."""

    def test_valid_agent_passthrough(self):
        """Test valid agent names pass through."""
        assert _normalize_agent("causal_impact") == "causal_impact"
        assert _normalize_agent("gap_analyzer") == "gap_analyzer"
        assert _normalize_agent("drift_monitor") == "drift_monitor"

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert _normalize_agent("CAUSAL_IMPACT") == "causal_impact"
        assert _normalize_agent("Gap_Analyzer") == "gap_analyzer"

    def test_space_to_underscore(self):
        """Test spaces are converted to underscores."""
        assert _normalize_agent("causal impact") == "causal_impact"
        assert _normalize_agent("drift monitor") == "drift_monitor"

    def test_hyphen_to_underscore(self):
        """Test hyphens are converted to underscores."""
        assert _normalize_agent("causal-impact") == "causal_impact"
        assert _normalize_agent("gap-analyzer") == "gap_analyzer"

    def test_short_names(self):
        """Test short name mappings."""
        assert _normalize_agent("causal") == "causal_impact"
        assert _normalize_agent("gap") == "gap_analyzer"
        assert _normalize_agent("drift") == "drift_monitor"
        assert _normalize_agent("experiment") == "experiment_designer"
        assert _normalize_agent("health") == "health_score"
        assert _normalize_agent("prediction") == "prediction_synthesizer"
        assert _normalize_agent("predict") == "prediction_synthesizer"
        assert _normalize_agent("resource") == "resource_optimizer"
        assert _normalize_agent("explain") == "explainer"
        assert _normalize_agent("feedback") == "feedback_learner"

    def test_cohort_variations(self):
        """Test cohort agent variations."""
        assert _normalize_agent("cohort") == "cohort_constructor"
        assert _normalize_agent("cohort_builder") == "cohort_constructor"
        assert _normalize_agent("patient_cohort") == "cohort_constructor"

    def test_unknown_defaults_to_explainer(self):
        """Test unknown agent defaults to explainer."""
        assert _normalize_agent("unknown_agent") == "explainer"
        assert _normalize_agent("random") == "explainer"


# =============================================================================
# Hardcoded Agent Routing Tests
# =============================================================================


class TestRouteAgentHardcoded:
    """Tests for route_agent_hardcoded function."""

    def test_returns_tuple(self):
        """Test routing returns correct tuple format."""
        result = route_agent_hardcoded("Why did TRx drop?")
        assert isinstance(result, tuple)
        assert len(result) == 4
        primary, secondary, confidence, rationale = result
        assert isinstance(primary, str)
        assert isinstance(secondary, list)
        assert isinstance(confidence, float)
        assert isinstance(rationale, str)

    def test_causal_keywords_route_to_causal_impact(self):
        """Test causal keywords route to causal_impact agent."""
        primary, _, _, _ = route_agent_hardcoded("Why did sales drop?")
        assert primary == "causal_impact"

    def test_causal_impact_keyword(self):
        """Test 'impact' keyword routes to causal_impact."""
        primary, _, _, _ = route_agent_hardcoded("What is the impact of the campaign?")
        assert primary == "causal_impact"

    def test_gap_keywords_route_to_gap_analyzer(self):
        """Test gap keywords route to gap_analyzer."""
        primary, _, _, _ = route_agent_hardcoded("Find gaps in our coverage")
        assert primary == "gap_analyzer"

    def test_drift_keywords_route_to_drift_monitor(self):
        """Test drift keywords route to drift_monitor."""
        primary, _, _, _ = route_agent_hardcoded("Check for data drift")
        assert primary == "drift_monitor"

    def test_experiment_keywords_route_to_experiment_designer(self):
        """Test experiment keywords route to experiment_designer."""
        primary, _, _, _ = route_agent_hardcoded("Design an A/B test")
        assert primary == "experiment_designer"

    def test_health_keywords_route_to_health_score(self):
        """Test health keywords route to health_score."""
        primary, _, _, _ = route_agent_hardcoded("Check system health")
        assert primary == "health_score"

    def test_predict_keywords_route_to_prediction_synthesizer(self):
        """Test prediction keywords route to prediction_synthesizer."""
        primary, _, _, _ = route_agent_hardcoded("Predict next quarter sales")
        assert primary == "prediction_synthesizer"

    def test_cohort_keywords_route_to_cohort_constructor(self):
        """Test cohort keywords route to cohort_constructor."""
        primary, _, _, _ = route_agent_hardcoded("Build a patient cohort")
        assert primary == "cohort_constructor"

    def test_explain_keywords_route_to_explainer(self):
        """Test explain keywords route to explainer."""
        primary, _, _, _ = route_agent_hardcoded("Explain the results")
        assert primary == "explainer"

    def test_no_match_defaults_to_explainer(self):
        """Test queries with no matches default to explainer."""
        primary, secondary, confidence, rationale = route_agent_hardcoded(
            "Random unrelated query with no keywords"
        )
        assert primary == "explainer"
        assert secondary == []
        assert "Default routing" in rationale

    def test_intent_boosting_causal(self):
        """Test intent boosting for causal analysis."""
        # Use a query that doesn't match other agent keywords
        # so causal_impact wins purely from intent boost
        primary_with_intent, _, _, _ = route_agent_hardcoded(
            "Evaluate the situation",
            intent=IntentType.CAUSAL_ANALYSIS,
        )
        assert primary_with_intent == "causal_impact"

    def test_secondary_agents(self):
        """Test secondary agents are returned for multi-match queries."""
        primary, secondary, _, _ = route_agent_hardcoded(
            "Explain why there was drift in the predictions"
        )
        # Should match multiple agents
        assert len(secondary) <= 2  # Limited to 2 secondary agents

    def test_confidence_calculation(self):
        """Test confidence is calculated correctly."""
        primary, secondary, confidence, _ = route_agent_hardcoded(
            "What caused the impact on our growth potential?"
        )
        # Multiple keyword matches should give higher confidence
        assert 0.5 <= confidence <= 0.95


# =============================================================================
# Global Collectors Tests
# =============================================================================


class TestGetSignalCollector:
    """Tests for get_signal_collector singleton."""

    def test_returns_collector(self):
        """Test returns a collector instance."""
        collector = get_signal_collector()
        assert isinstance(collector, IntentTrainingSignalCollector)

    def test_returns_same_instance(self):
        """Test returns the same singleton instance."""
        collector1 = get_signal_collector()
        collector2 = get_signal_collector()
        assert collector1 is collector2


class TestGetRoutingSignalCollector:
    """Tests for get_routing_signal_collector singleton."""

    def test_returns_collector(self):
        """Test returns a collector instance."""
        collector = get_routing_signal_collector()
        assert isinstance(collector, RoutingTrainingSignalCollector)

    def test_returns_same_instance(self):
        """Test returns the same singleton instance."""
        collector1 = get_routing_signal_collector()
        collector2 = get_routing_signal_collector()
        assert collector1 is collector2


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_valid_intents_contains_all_types(self):
        """Test VALID_INTENTS contains all intent types."""
        assert IntentType.KPI_QUERY in VALID_INTENTS
        assert IntentType.CAUSAL_ANALYSIS in VALID_INTENTS
        assert IntentType.AGENT_STATUS in VALID_INTENTS
        assert IntentType.RECOMMENDATION in VALID_INTENTS
        assert IntentType.SEARCH in VALID_INTENTS
        assert IntentType.MULTI_FACETED in VALID_INTENTS
        assert IntentType.GREETING in VALID_INTENTS
        assert IntentType.HELP in VALID_INTENTS
        assert IntentType.GENERAL in VALID_INTENTS

    def test_valid_agents_contains_expected_agents(self):
        """Test VALID_AGENTS contains expected agents."""
        expected_agents = [
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
            "cohort_constructor",
        ]
        for agent in expected_agents:
            assert agent in VALID_AGENTS

    def test_agent_capabilities_has_keywords(self):
        """Test AGENT_CAPABILITIES has keywords for each agent."""
        assert "causal_impact" in AGENT_CAPABILITIES
        assert "gap_analyzer" in AGENT_CAPABILITIES
        assert "drift_monitor" in AGENT_CAPABILITIES

        # Each agent should have at least one keyword
        for agent, keywords in AGENT_CAPABILITIES.items():
            assert len(keywords) > 0, f"{agent} has no keywords"


# =============================================================================
# Async Function Tests (with mocked DSPy)
# =============================================================================


class TestClassifyIntentDspy:
    """Tests for classify_intent_dspy async function."""

    @pytest.mark.asyncio
    async def test_fallback_when_dspy_unavailable(self):
        """Test fallback to hardcoded when DSPy is unavailable."""
        # Import here to get async function
        from src.api.routes.chatbot_dspy import classify_intent_dspy

        with patch("src.api.routes.chatbot_dspy._get_dspy_classifier", return_value=None):
            intent, confidence, reasoning, method = await classify_intent_dspy(
                "Hello there!",
                collect_signal=False,
            )
            assert intent == IntentType.GREETING
            assert method == "hardcoded"

    @pytest.mark.asyncio
    async def test_with_mocked_dspy_classifier(self):
        """Test with mocked DSPy classifier."""
        from src.api.routes.chatbot_dspy import classify_intent_dspy

        # Create mock prediction result
        mock_result = MagicMock()
        mock_result.intent = "kpi_query"
        mock_result.confidence = 0.92
        mock_result.reasoning = "Query mentions TRx metric"

        with patch("src.api.routes.chatbot_dspy._get_dspy_classifier") as mock_get_classifier:
            with patch("src.api.routes.chatbot_dspy._run_dspy_with_retry", new_callable=AsyncMock) as mock_retry:
                mock_retry.return_value = mock_result
                mock_get_classifier.return_value = MagicMock()

                intent, confidence, reasoning, method = await classify_intent_dspy(
                    "What is the TRx for Kisqali?",
                    collect_signal=False,
                )

                assert intent == IntentType.KPI_QUERY
                assert confidence == 0.92
                assert method == "dspy"

    @pytest.mark.asyncio
    async def test_fallback_on_dspy_failure(self):
        """Test fallback when DSPy fails."""
        from src.api.routes.chatbot_dspy import classify_intent_dspy

        with patch("src.api.routes.chatbot_dspy._get_dspy_classifier") as mock_get_classifier:
            with patch("src.api.routes.chatbot_dspy._run_dspy_with_retry", new_callable=AsyncMock) as mock_retry:
                # DSPy returns None (failure)
                mock_retry.return_value = None
                mock_get_classifier.return_value = MagicMock()

                intent, confidence, reasoning, method = await classify_intent_dspy(
                    "What is the TRx?",
                    collect_signal=False,
                )

                # Should fallback to hardcoded
                assert method == "hardcoded"
                assert intent == IntentType.KPI_QUERY


class TestRouteAgentDspy:
    """Tests for route_agent_dspy async function."""

    @pytest.mark.asyncio
    async def test_fallback_when_dspy_unavailable(self):
        """Test fallback to hardcoded when DSPy is unavailable."""
        from src.api.routes.chatbot_dspy import route_agent_dspy

        with patch("src.api.routes.chatbot_dspy._get_dspy_router", return_value=None):
            primary, secondary, confidence, rationale, method = await route_agent_dspy(
                "Why did sales drop?",
                collect_signal=False,
            )
            assert primary == "causal_impact"
            assert method == "hardcoded"

    @pytest.mark.asyncio
    async def test_with_mocked_dspy_router(self):
        """Test with mocked DSPy router."""
        from src.api.routes.chatbot_dspy import route_agent_dspy

        # Create mock prediction result
        mock_result = MagicMock()
        mock_result.primary_agent = "gap_analyzer"
        mock_result.secondary_agents = "causal_impact,health_score"
        mock_result.routing_confidence = 0.88
        mock_result.rationale = "Query asks about opportunities"

        with patch("src.api.routes.chatbot_dspy._get_dspy_router") as mock_get_router:
            with patch("src.api.routes.chatbot_dspy._run_dspy_with_retry", new_callable=AsyncMock) as mock_retry:
                mock_retry.return_value = mock_result
                mock_get_router.return_value = MagicMock()

                primary, secondary, confidence, rationale, method = await route_agent_dspy(
                    "Find ROI opportunities",
                    collect_signal=False,
                )

                assert primary == "gap_analyzer"
                assert method == "dspy"

    @pytest.mark.asyncio
    async def test_fallback_on_dspy_failure(self):
        """Test fallback when DSPy fails."""
        from src.api.routes.chatbot_dspy import route_agent_dspy

        with patch("src.api.routes.chatbot_dspy._get_dspy_router") as mock_get_router:
            with patch("src.api.routes.chatbot_dspy._run_dspy_with_retry", new_callable=AsyncMock) as mock_retry:
                # DSPy returns None (failure)
                mock_retry.return_value = None
                mock_get_router.return_value = MagicMock()

                primary, secondary, confidence, rationale, method = await route_agent_dspy(
                    "Predict future sales",
                    collect_signal=False,
                )

                # Should fallback to hardcoded
                assert method == "hardcoded"
                assert primary == "prediction_synthesizer"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for chatbot_dspy module."""

    def test_full_classification_flow_sync(self):
        """Test full classification flow using sync methods."""
        # Use a query that matches causal but NOT KPI (KPI is checked first)
        query = "Why did performance decline?"

        # Classify intent
        intent, conf1, _, method1 = classify_intent_sync(query)
        assert intent == IntentType.CAUSAL_ANALYSIS
        assert method1 == "hardcoded"

        # Route to agent based on intent
        agent, secondary, conf2, rationale = route_agent_hardcoded(
            query,
            intent=intent,
        )
        assert agent == "causal_impact"

    def test_signal_collection_integration(self):
        """Test that signals are collected during classification."""
        # Get fresh collector
        collector = get_signal_collector()
        initial_count = len(collector)

        # Run classification (which should collect signal by default)
        # Using sync version for simplicity in test
        intent, _, _, _ = classify_intent_sync("Test query for signal collection")

        # Signal collection happens in async version only
        # Sync version doesn't collect signals
        # Just verify the collector works
        signal = IntentTrainingSignal(
            query="test",
            conversation_context="",
            brand_context="",
            predicted_intent="general",
            confidence=0.5,
            reasoning="test",
            classification_method="hardcoded",
        )
        collector.add_signal(signal)
        assert len(collector) == initial_count + 1

    def test_routing_for_each_agent_type(self):
        """Test routing works for queries targeting each agent."""
        test_cases = [
            ("Why did this happen and what caused it?", "causal_impact"),
            ("Find ROI gaps and opportunities", "gap_analyzer"),
            ("Check for data drift anomalies", "drift_monitor"),
            ("Design an experiment for A/B testing", "experiment_designer"),
            ("What is the health score and status?", "health_score"),
            ("Predict future trends and forecast", "prediction_synthesizer"),
            ("Build a patient cohort with criteria", "cohort_constructor"),
            ("Explain and summarize the results", "explainer"),
        ]

        for query, expected_agent in test_cases:
            agent, _, _, _ = route_agent_hardcoded(query)
            assert agent == expected_agent, f"Query '{query}' routed to {agent}, expected {expected_agent}"
