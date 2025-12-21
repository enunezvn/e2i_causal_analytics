"""Tests for intent_classifier node."""

import pytest

from src.agents.orchestrator.nodes.intent_classifier import (
    IntentClassifierNode,
    classify_intent,
)


class TestIntentClassifierNode:
    """Test IntentClassifierNode."""

    @pytest.mark.asyncio
    async def test_pattern_classify_causal_effect(self):
        """Test pattern matching for causal effect queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify(
            "what is the impact of hcp engagement on conversions?"
        )

        assert result["primary_intent"] == "causal_effect"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_performance_gap(self):
        """Test pattern matching for performance gap queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify(
            "where are the roi opportunities for improving performance?"
        )

        assert result["primary_intent"] == "performance_gap"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_segment_analysis(self):
        """Test pattern matching for segment analysis queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("which segments respond best to our messaging?")

        assert result["primary_intent"] == "segment_analysis"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_experiment_design(self):
        """Test pattern matching for experiment design queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("help me design an a/b test for our campaign")

        assert result["primary_intent"] == "experiment_design"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_prediction(self):
        """Test pattern matching for prediction queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("what will be the forecast for next quarter?")

        assert result["primary_intent"] == "prediction"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_resource_allocation(self):
        """Test pattern matching for resource allocation queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("how should we allocate our budget across regions?")

        assert result["primary_intent"] == "resource_allocation"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_explanation(self):
        """Test pattern matching for explanation queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("explain how this model works")

        assert result["primary_intent"] == "explanation"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_system_health(self):
        """Test pattern matching for system health queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("what is the system health status?")

        assert result["primary_intent"] == "system_health"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_drift_check(self):
        """Test pattern matching for drift check queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("has there been any data drift recently?")

        assert result["primary_intent"] == "drift_check"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_feedback(self):
        """Test pattern matching for feedback queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("learn from previous campaign results")

        assert result["primary_intent"] == "feedback"
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_general(self):
        """Test pattern matching for general/unclear queries."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("hello there")

        assert result["primary_intent"] == "general"
        assert result["confidence"] < 0.8

    @pytest.mark.asyncio
    async def test_pattern_classify_multi_intent(self):
        """Test pattern matching with multiple intents."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify(
            "what causes conversion drops and how do we improve performance?"
        )

        # Should detect primary intent
        assert result["primary_intent"] in ["causal_effect", "performance_gap"]
        # Should detect secondary intents
        assert len(result["secondary_intents"]) > 0
        assert result["requires_multi_agent"] is True

    @pytest.mark.asyncio
    async def test_execute_with_high_confidence(self):
        """Test execute with high confidence pattern match."""
        classifier = IntentClassifierNode()

        state = {"query": "what drives patient conversion rates?"}

        result = await classifier.execute(state)

        assert "intent" in result
        assert result["intent"]["primary_intent"] == "causal_effect"
        assert result["intent"]["confidence"] >= 0.8
        assert result["current_phase"] == "routing"
        assert result["classification_latency_ms"] >= 0  # Can be 0 for fast pattern matching

    @pytest.mark.asyncio
    async def test_execute_with_low_confidence_fallback(self):
        """Test execute with low confidence triggers LLM fallback."""
        classifier = IntentClassifierNode()

        # Ambiguous query
        state = {"query": "tell me about remibrutinib"}

        result = await classifier.execute(state)

        assert "intent" in result
        assert "primary_intent" in result["intent"]
        assert result["current_phase"] == "routing"

    @pytest.mark.asyncio
    async def test_classify_intent_function(self):
        """Test standalone classify_intent function."""
        state = {"query": "predict next quarter conversions"}

        result = await classify_intent(state)

        assert "intent" in result
        assert result["intent"]["primary_intent"] == "prediction"


class TestPatternMatching:
    """Test pattern matching edge cases."""

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        """Test case-insensitive pattern matching."""
        classifier = IntentClassifierNode()

        result1 = classifier._pattern_classify("WHAT DRIVES CONVERSIONS?")
        result2 = classifier._pattern_classify("what drives conversions?")

        assert result1["primary_intent"] == result2["primary_intent"]

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test partial word matching."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("i want to understand what is impacting our results")

        assert result["primary_intent"] == "causal_effect"

    @pytest.mark.asyncio
    async def test_multiple_patterns_same_intent(self):
        """Test multiple patterns matching same intent."""
        classifier = IntentClassifierNode()

        result1 = classifier._pattern_classify("what causes conversion drops?")
        result2 = classifier._pattern_classify("why did conversions decrease?")
        result3 = classifier._pattern_classify("what drives the change in conversion?")

        assert (
            result1["primary_intent"]
            == result2["primary_intent"]
            == result3["primary_intent"]
            == "causal_effect"
        )

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test empty query handling."""
        classifier = IntentClassifierNode()

        result = classifier._pattern_classify("")

        assert result["primary_intent"] == "general"
        assert result["confidence"] < 0.8
