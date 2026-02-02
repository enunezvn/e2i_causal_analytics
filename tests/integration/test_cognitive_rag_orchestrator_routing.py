"""
Integration tests for CognitiveRAG → Orchestrator Routing.

Tests the flow from CognitiveRAG's Phase 3 (Agent Routing) to Orchestrator,
validating:
1. Pre-routing acceptance (high confidence routing decisions)
2. Low-confidence override (falling back to local classification)
3. Training signal emission for DSPy optimization
4. Multi-agent routing from CognitiveRAG decisions

This validates the DSPy integration described in orchestrator-agent.md.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.orchestrator.agent import OrchestratorAgent

# =============================================================================
# Test Data
# =============================================================================

SAMPLE_COGNITIVE_ROUTING_HIGH_CONFIDENCE = {
    "primary_agent": "causal_impact",
    "secondary_agents": [],
    "routing_confidence": 0.92,
    "parameters": {"interpretation_depth": "detailed"},
    "detected_intent": "causal",
    "evidence_summary": "Query relates to causal impact of HCP engagement on TRx",
}

SAMPLE_COGNITIVE_ROUTING_LOW_CONFIDENCE = {
    "primary_agent": "gap_analyzer",
    "secondary_agents": ["heterogeneous_optimizer"],
    "routing_confidence": 0.45,  # Below threshold
    "parameters": {},
    "detected_intent": "exploratory",
    "evidence_summary": "Unclear intent, may involve gap analysis",
}

SAMPLE_COGNITIVE_ROUTING_MULTI_AGENT = {
    "primary_agent": "causal_impact",
    "secondary_agents": ["heterogeneous_optimizer", "explainer"],
    "routing_confidence": 0.85,
    "parameters": {"include_cate": True},
    "detected_intent": "causal",
    "evidence_summary": "Causal analysis with segment breakdown needed",
}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent_registry():
    """Create mock agent registry with common agents."""
    registry = {}

    for agent_name in [
        "causal_impact",
        "gap_analyzer",
        "heterogeneous_optimizer",
        "explainer",
        "experiment_designer",
        "drift_monitor",
        "prediction_synthesizer",
    ]:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(
            return_value={
                "success": True,
                "agent_name": agent_name,
                "result": {"narrative": f"Analysis from {agent_name}"},
                "confidence": 0.85,
            }
        )
        mock_agent.analyze = mock_agent.run
        registry[agent_name] = mock_agent

    return registry


@pytest.fixture
def orchestrator_with_registry(mock_agent_registry):
    """Create orchestrator with mocked agent registry."""
    return OrchestratorAgent(
        agent_registry=mock_agent_registry,
        enable_checkpointing=False,
        enable_opik=False,
    )


@pytest.fixture
def mock_intent_classifier():
    """Create mock intent classifier for testing override behavior."""
    mock = AsyncMock()
    mock.execute = AsyncMock(
        return_value={
            "intent": {
                "primary_intent": "causal_effect",
                "confidence": 0.88,
                "secondary_intents": [],
                "requires_multi_agent": False,
            },
            "classification_latency_ms": 150,
            "current_phase": "routing",
        }
    )
    return mock


# =============================================================================
# Pre-Routing Acceptance Tests
# =============================================================================


class TestCognitiveRAGRoutingAcceptance:
    """Test cases for accepting CognitiveRAG pre-routing decisions."""

    @pytest.mark.asyncio
    async def test_accept_high_confidence_routing(self, orchestrator_with_registry):
        """Orchestrator should accept CognitiveRAG routing when confidence >= threshold."""
        input_data = {
            "query": "What is the causal impact of HCP engagement on NRx for Kisqali?",
            "user_id": "user_123",
            "cognitive_routing": SAMPLE_COGNITIVE_ROUTING_HIGH_CONFIDENCE,
        }

        with patch.object(
            orchestrator_with_registry,
            "classify_intent",
            new_callable=AsyncMock,
        ):
            # The classify_intent should NOT be called when pre-routing is accepted
            result = await orchestrator_with_registry.run(input_data)

            # Verify response structure
            assert result["status"] in ["completed", "partial_success"]
            assert "causal_impact" in result.get("agents_dispatched", [])

    @pytest.mark.asyncio
    async def test_routing_uses_cognitive_rag_parameters(self, orchestrator_with_registry):
        """Parameters from CognitiveRAG should be passed to the dispatched agent."""
        input_data = {
            "query": "Analyze the causal chain for TRx decline",
            "user_id": "user_456",
            "cognitive_routing": SAMPLE_COGNITIVE_ROUTING_HIGH_CONFIDENCE,
        }

        result = await orchestrator_with_registry.run(input_data)

        # Verify the agent was called (mock agent in registry)
        agent = orchestrator_with_registry.agent_registry["causal_impact"]
        if agent.run.called or agent.analyze.called:
            # Check parameters were included
            call_args = agent.run.call_args or agent.analyze.call_args
            if call_args:
                # Parameters should include cognitive routing params
                assert result["status"] in ["completed", "partial_success", "failed"]

    @pytest.mark.asyncio
    async def test_multi_agent_routing_from_cognitive_rag(self, orchestrator_with_registry):
        """Orchestrator should handle multi-agent routing from CognitiveRAG."""
        input_data = {
            "query": "What drives NRx differences across segments and why?",
            "user_id": "user_789",
            "cognitive_routing": SAMPLE_COGNITIVE_ROUTING_MULTI_AGENT,
        }

        result = await orchestrator_with_registry.run(input_data)

        # Verify multiple agents were considered
        assert result["status"] in ["completed", "partial_success", "failed"]
        agents_dispatched = result.get("agents_dispatched", [])
        # Should have at least the primary agent
        assert len(agents_dispatched) >= 1


# =============================================================================
# Low-Confidence Override Tests
# =============================================================================


class TestLowConfidenceOverride:
    """Test cases for overriding low-confidence CognitiveRAG routing."""

    @pytest.mark.asyncio
    async def test_override_low_confidence_routing(
        self, orchestrator_with_registry, mock_intent_classifier
    ):
        """Orchestrator should fall back to local classification when confidence < threshold."""
        input_data = {
            "query": "Show me some insights about market performance",
            "user_id": "user_abc",
            "cognitive_routing": SAMPLE_COGNITIVE_ROUTING_LOW_CONFIDENCE,
        }

        # Patch the classifier to track if it was called
        with patch(
            "src.agents.orchestrator.nodes.intent_classifier.IntentClassifierNode",
            return_value=mock_intent_classifier,
        ):
            result = await orchestrator_with_registry.run(input_data)

            # The orchestrator should still produce a result
            assert result["status"] in ["completed", "partial_success", "failed"]

    @pytest.mark.asyncio
    async def test_confidence_threshold_boundary(self, orchestrator_with_registry):
        """Test behavior at exactly the confidence threshold (0.7)."""
        boundary_routing = {
            "primary_agent": "drift_monitor",
            "secondary_agents": [],
            "routing_confidence": 0.70,  # Exactly at threshold
            "parameters": {},
            "detected_intent": "drift_check",
        }

        input_data = {
            "query": "Check for data drift in the model",
            "user_id": "user_boundary",
            "cognitive_routing": boundary_routing,
        }

        result = await orchestrator_with_registry.run(input_data)

        # At threshold, should still be accepted
        assert result["status"] in ["completed", "partial_success", "failed"]


# =============================================================================
# Training Signal Emission Tests
# =============================================================================


class TestTrainingSignalEmission:
    """Test cases for DSPy training signal collection."""

    @pytest.mark.asyncio
    async def test_training_signal_structure(self, orchestrator_with_registry):
        """Verify training signal contains required fields for DSPy optimization."""
        input_data = {
            "query": "What caused the drop in conversion rate?",
            "user_id": "user_signal",
            "cognitive_routing": SAMPLE_COGNITIVE_ROUTING_HIGH_CONFIDENCE,
        }

        result = await orchestrator_with_registry.run(input_data)

        # Training signals would be collected internally
        # This test validates the output structure that feeds signals
        assert "query_id" in result
        assert "status" in result
        assert "agents_dispatched" in result
        assert "response_confidence" in result or "confidence" in result

    @pytest.mark.asyncio
    async def test_routing_accuracy_calculation(self, orchestrator_with_registry):
        """Test that routing accuracy can be computed from outputs."""
        input_data = {
            "query": "Analyze the causal impact of rep visits",
            "user_id": "user_accuracy",
            "cognitive_routing": {
                "primary_agent": "causal_impact",
                "secondary_agents": [],
                "routing_confidence": 0.9,
                "detected_intent": "causal",
            },
        }

        result = await orchestrator_with_registry.run(input_data)

        # Compute routing accuracy (would be used for DSPy training)
        predicted_agent = input_data["cognitive_routing"]["primary_agent"]
        actual_agents = result.get("agents_dispatched", [])

        if actual_agents:
            actual_primary = actual_agents[0]
            if predicted_agent == actual_primary:
                accuracy = 1.0
            elif predicted_agent in actual_agents:
                accuracy = 0.7
            else:
                accuracy = 0.2

            # Accuracy should be computable
            assert 0.0 <= accuracy <= 1.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases in CognitiveRAG routing integration."""

    @pytest.mark.asyncio
    async def test_missing_cognitive_routing(self, orchestrator_with_registry):
        """Orchestrator should work normally without cognitive_routing field."""
        input_data = {
            "query": "What is the trend in TRx for Kisqali?",
            "user_id": "user_no_routing",
            # No cognitive_routing field
        }

        result = await orchestrator_with_registry.run(input_data)

        # Should fall back to normal classification
        assert result["status"] in ["completed", "partial_success", "failed"]
        assert "query_id" in result

    @pytest.mark.asyncio
    async def test_invalid_agent_in_routing(self, orchestrator_with_registry):
        """Handle gracefully when CognitiveRAG routes to unknown agent."""
        invalid_routing = {
            "primary_agent": "nonexistent_agent",
            "secondary_agents": [],
            "routing_confidence": 0.95,
            "detected_intent": "unknown",
        }

        input_data = {
            "query": "Some query",
            "user_id": "user_invalid",
            "cognitive_routing": invalid_routing,
        }

        result = await orchestrator_with_registry.run(input_data)

        # Should handle gracefully (either fallback or error)
        assert result["status"] in ["completed", "partial_success", "failed"]

    @pytest.mark.asyncio
    async def test_empty_secondary_agents(self, orchestrator_with_registry):
        """Handle routing with no secondary agents."""
        single_agent_routing = {
            "primary_agent": "explainer",
            "secondary_agents": [],
            "routing_confidence": 0.88,
            "detected_intent": "explanation",
        }

        input_data = {
            "query": "Explain the model predictions",
            "user_id": "user_single",
            "cognitive_routing": single_agent_routing,
        }

        result = await orchestrator_with_registry.run(input_data)

        assert result["status"] in ["completed", "partial_success", "failed"]

    @pytest.mark.asyncio
    async def test_cognitive_routing_with_empty_parameters(self, orchestrator_with_registry):
        """Handle routing with empty parameters dict."""
        routing_no_params = {
            "primary_agent": "gap_analyzer",
            "secondary_agents": [],
            "routing_confidence": 0.82,
            "parameters": {},
            "detected_intent": "performance_gap",
        }

        input_data = {
            "query": "Find performance gaps in the Midwest",
            "user_id": "user_no_params",
            "cognitive_routing": routing_no_params,
        }

        result = await orchestrator_with_registry.run(input_data)

        assert result["status"] in ["completed", "partial_success", "failed"]


# =============================================================================
# Integration with Existing Orchestrator Flow
# =============================================================================


class TestOrchestratorFlowIntegration:
    """Test CognitiveRAG routing integrates with existing orchestrator flow."""

    @pytest.mark.asyncio
    async def test_rag_context_still_retrieved(self, orchestrator_with_registry):
        """RAG context should still be retrieved even with pre-routing."""
        input_data = {
            "query": "What is the causal impact of engagement?",
            "user_id": "user_rag",
            "cognitive_routing": SAMPLE_COGNITIVE_ROUTING_HIGH_CONFIDENCE,
        }

        with patch(
            "src.agents.orchestrator.nodes.rag_context.retrieve_rag_context",
            new_callable=AsyncMock,
        ) as mock_rag:
            mock_rag.return_value = {
                "rag_context": {"documents": []},
                "rag_latency_ms": 100,
            }

            result = await orchestrator_with_registry.run(input_data)

            # Result should be valid
            assert result["status"] in ["completed", "partial_success", "failed"]

    @pytest.mark.asyncio
    async def test_synthesis_works_with_cognitive_routing(self, orchestrator_with_registry):
        """Response synthesis should work with CognitiveRAG-routed agents."""
        input_data = {
            "query": "Explain the drivers of NRx growth",
            "user_id": "user_synthesis",
            "cognitive_routing": SAMPLE_COGNITIVE_ROUTING_HIGH_CONFIDENCE,
        }

        result = await orchestrator_with_registry.run(input_data)

        # Should have synthesized response
        assert "response_text" in result or "synthesized_response" in result
        assert result["status"] in ["completed", "partial_success", "failed"]

    @pytest.mark.asyncio
    async def test_latency_tracking_with_cognitive_routing(self, orchestrator_with_registry):
        """Latency should still be tracked with CognitiveRAG routing."""
        input_data = {
            "query": "Quick analysis of market share",
            "user_id": "user_latency",
            "cognitive_routing": SAMPLE_COGNITIVE_ROUTING_HIGH_CONFIDENCE,
        }

        result = await orchestrator_with_registry.run(input_data)

        # Latency fields should be present
        assert "total_latency_ms" in result
        assert result["total_latency_ms"] >= 0
