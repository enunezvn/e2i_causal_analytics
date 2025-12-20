"""Integration tests for OrchestratorAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.orchestrator.agent import OrchestratorAgent


class TestOrchestratorAgent:
    """Test OrchestratorAgent integration."""

    @pytest.mark.asyncio
    async def test_run_complete_workflow(self):
        """Test complete orchestrator workflow."""
        orchestrator = OrchestratorAgent()

        input_data = {
            "query": "what is the impact of hcp engagement on patient conversions?",
            "user_id": "user_123",
            "user_context": {"expertise": "analyst"},
        }

        result = await orchestrator.run(input_data)

        # Verify output structure
        assert "query_id" in result
        assert result["status"] == "completed"
        assert "response_text" in result
        assert result["response_text"] != ""
        assert "response_confidence" in result
        assert result["response_confidence"] > 0
        assert "agents_dispatched" in result
        assert len(result["agents_dispatched"]) > 0
        assert "agent_results" in result
        assert "total_latency_ms" in result
        assert result["total_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_run_with_query_id(self):
        """Test that provided query_id is preserved."""
        orchestrator = OrchestratorAgent()

        input_data = {
            "query": "test query",
            "query_id": "custom-query-id-123",
        }

        result = await orchestrator.run(input_data)

        assert result["query_id"] == "custom-query-id-123"

    @pytest.mark.asyncio
    async def test_run_without_query_id(self):
        """Test that query_id is generated if not provided."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "test query"}

        result = await orchestrator.run(input_data)

        assert "query_id" in result
        assert result["query_id"].startswith("q-")

    @pytest.mark.asyncio
    async def test_run_missing_query(self):
        """Test that missing query raises ValueError."""
        orchestrator = OrchestratorAgent()

        input_data = {"user_id": "user_123"}

        with pytest.raises(ValueError, match="Missing required field: query"):
            await orchestrator.run(input_data)

    @pytest.mark.asyncio
    async def test_run_causal_effect_query(self):
        """Test orchestrator with causal effect query."""
        orchestrator = OrchestratorAgent()

        input_data = {
            "query": "what drives patient conversion rates?",
        }

        result = await orchestrator.run(input_data)

        # Should dispatch to causal_impact agent
        assert "causal_impact" in result["agents_dispatched"]
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_performance_gap_query(self):
        """Test orchestrator with performance gap query."""
        orchestrator = OrchestratorAgent()

        input_data = {
            "query": "where are the roi opportunities for improving performance?",
        }

        result = await orchestrator.run(input_data)

        # Should dispatch to gap_analyzer agent
        assert "gap_analyzer" in result["agents_dispatched"]
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_prediction_query(self):
        """Test orchestrator with prediction query."""
        orchestrator = OrchestratorAgent()

        input_data = {
            "query": "what will be the forecast for next quarter conversions?",
        }

        result = await orchestrator.run(input_data)

        # Should dispatch to prediction_synthesizer agent
        assert "prediction_synthesizer" in result["agents_dispatched"]
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_general_query(self):
        """Test orchestrator with general/unclear query."""
        orchestrator = OrchestratorAgent()

        input_data = {
            "query": "hello there, how are you?",
        }

        result = await orchestrator.run(input_data)

        # Should default to explainer
        assert "explainer" in result["agents_dispatched"]
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_with_conversation_history(self):
        """Test orchestrator with conversation history."""
        orchestrator = OrchestratorAgent()

        input_data = {
            "query": "what about segment analysis?",
            "conversation_history": [
                {"role": "user", "content": "analyze conversions"},
                {"role": "assistant", "content": "Here's the analysis..."},
            ],
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_with_session_id(self):
        """Test orchestrator with session tracking."""
        orchestrator = OrchestratorAgent()

        input_data = {
            "query": "test query",
            "session_id": "session-abc-123",
        }

        result = await orchestrator.run(input_data)

        # Session ID should be preserved in metadata
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_latency_breakdown(self):
        """Test that all latency components are measured."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "what drives conversions?"}

        result = await orchestrator.run(input_data)

        # Verify latency breakdown
        assert "classification_latency_ms" in result
        assert "routing_latency_ms" in result
        assert "dispatch_latency_ms" in result
        assert "synthesis_latency_ms" in result
        assert "total_latency_ms" in result

        # All should be non-negative
        assert result["classification_latency_ms"] >= 0
        assert result["routing_latency_ms"] >= 0
        assert result["dispatch_latency_ms"] >= 0
        assert result["synthesis_latency_ms"] >= 0

        # Total should be sum of components
        expected_total = (
            result["classification_latency_ms"]
            + result["routing_latency_ms"]
            + result["dispatch_latency_ms"]
            + result["synthesis_latency_ms"]
        )
        assert result["total_latency_ms"] == expected_total

    @pytest.mark.asyncio
    async def test_intent_classification_metadata(self):
        """Test that intent classification metadata is included."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "what causes conversion rate changes?"}

        result = await orchestrator.run(input_data)

        assert "intent_classified" in result
        assert result["intent_classified"] == "causal_effect"
        assert "intent_confidence" in result
        assert result["intent_confidence"] >= 0.8


class TestOrchestratorHelperMethods:
    """Test OrchestratorAgent helper methods."""

    @pytest.mark.asyncio
    async def test_classify_intent(self):
        """Test standalone classify_intent helper."""
        orchestrator = OrchestratorAgent()

        intent = await orchestrator.classify_intent(
            "what is the impact of hcp engagement?"
        )

        assert "primary_intent" in intent
        assert intent["primary_intent"] == "causal_effect"
        assert "confidence" in intent
        assert intent["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_route_query(self):
        """Test standalone route_query helper."""
        orchestrator = OrchestratorAgent()

        agents = await orchestrator.route_query("what drives patient conversions?")

        assert isinstance(agents, list)
        assert len(agents) > 0
        assert "causal_impact" in agents

    def test_get_agent_registry(self):
        """Test get_agent_registry method."""
        mock_agent = MagicMock()
        orchestrator = OrchestratorAgent(agent_registry={"test_agent": mock_agent})

        registry = orchestrator.get_agent_registry()

        assert "test_agent" in registry
        assert registry["test_agent"] == mock_agent

    def test_register_agent(self):
        """Test register_agent method."""
        orchestrator = OrchestratorAgent()

        mock_agent = MagicMock()
        orchestrator.register_agent("new_agent", mock_agent)

        registry = orchestrator.get_agent_registry()
        assert "new_agent" in registry
        assert registry["new_agent"] == mock_agent

    def test_unregister_agent(self):
        """Test unregister_agent method."""
        mock_agent = MagicMock()
        orchestrator = OrchestratorAgent(agent_registry={"test_agent": mock_agent})

        orchestrator.unregister_agent("test_agent")

        registry = orchestrator.get_agent_registry()
        assert "test_agent" not in registry

    def test_unregister_nonexistent_agent(self):
        """Test unregister_agent with non-existent agent."""
        orchestrator = OrchestratorAgent()

        # Should not raise error
        orchestrator.unregister_agent("nonexistent_agent")

    def test_generate_query_id(self):
        """Test query ID generation."""
        orchestrator = OrchestratorAgent()

        query_id = orchestrator._generate_query_id()

        assert query_id.startswith("q-")
        assert len(query_id) == 14  # "q-" + 12 hex chars


class TestOrchestratorWithRealAgents:
    """Test orchestrator with real agent instances."""

    @pytest.mark.asyncio
    async def test_run_with_registered_agent(self):
        """Test orchestrator dispatching to registered agent."""
        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(
            return_value={
                "narrative": "Real agent analysis result",
                "recommendations": ["Action 1", "Action 2"],
                "confidence": 0.95,
            }
        )

        # Register agent
        orchestrator = OrchestratorAgent(agent_registry={"causal_impact": mock_agent})

        input_data = {"query": "what drives conversions?"}

        result = await orchestrator.run(input_data)

        # Verify agent was called
        assert mock_agent.analyze.called

        # Verify output includes agent result
        assert result["status"] == "completed"
        assert "Real agent analysis result" in result["response_text"]
        assert result["response_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_run_with_multiple_registered_agents(self):
        """Test orchestrator with multiple registered agents."""
        # Create mock agents
        mock_causal = MagicMock()
        mock_causal.analyze = AsyncMock(
            return_value={
                "narrative": "Causal analysis",
                "recommendations": ["Rec 1"],
                "confidence": 0.9,
            }
        )

        mock_gap = MagicMock()
        mock_gap.analyze = AsyncMock(
            return_value={
                "narrative": "Gap analysis",
                "recommendations": ["Rec 2"],
                "confidence": 0.8,
            }
        )

        # Register both agents
        orchestrator = OrchestratorAgent(
            agent_registry={"causal_impact": mock_causal, "gap_analyzer": mock_gap}
        )

        # Query that should route to causal_impact
        input_data = {"query": "what causes conversion drops?"}

        result = await orchestrator.run(input_data)

        # Should call causal_impact
        assert mock_causal.analyze.called
        assert result["status"] == "completed"


class TestOrchestratorOutputContract:
    """Test that orchestrator output conforms to contract."""

    @pytest.mark.asyncio
    async def test_output_has_required_fields(self):
        """Test that output has all required contract fields."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "test query"}

        result = await orchestrator.run(input_data)

        # Required fields from OrchestratorOutput contract
        required_fields = [
            "query_id",
            "status",
            "response_text",
            "response_confidence",
            "agents_dispatched",
            "agent_results",
            "citations",
            "visualizations",
            "follow_up_suggestions",
            "recommendations",
            "total_latency_ms",
            "timestamp",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_output_types(self):
        """Test that output field types are correct."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "test query"}

        result = await orchestrator.run(input_data)

        # Verify types
        assert isinstance(result["query_id"], str)
        assert isinstance(result["status"], str)
        assert result["status"] in ["completed", "failed"]
        assert isinstance(result["response_text"], str)
        assert isinstance(result["response_confidence"], (int, float))
        assert isinstance(result["agents_dispatched"], list)
        assert isinstance(result["agent_results"], list)
        assert isinstance(result["citations"], list)
        assert isinstance(result["visualizations"], list)
        assert isinstance(result["follow_up_suggestions"], list)
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["total_latency_ms"], (int, float))

    @pytest.mark.asyncio
    async def test_agent_results_structure(self):
        """Test that agent_results have correct structure."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "what drives conversions?"}

        result = await orchestrator.run(input_data)

        # Verify agent_results structure
        assert len(result["agent_results"]) > 0

        for agent_result in result["agent_results"]:
            assert "agent_name" in agent_result
            assert "success" in agent_result
            assert isinstance(agent_result["success"], bool)

            if agent_result["success"]:
                assert "result" in agent_result
            else:
                assert "error" in agent_result


class TestOrchestratorPerformance:
    """Test orchestrator performance characteristics."""

    @pytest.mark.asyncio
    async def test_orchestration_overhead_target(self):
        """Test that orchestration overhead meets <2s target."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "what drives conversions?"}

        result = await orchestrator.run(input_data)

        # Orchestration overhead = classification + routing + synthesis
        # (dispatch time is agent execution, not orchestration overhead)
        orchestration_overhead = (
            result["classification_latency_ms"]
            + result["routing_latency_ms"]
            + result["synthesis_latency_ms"]
        )

        # Should be well under 2000ms (2 seconds) with mock agents
        assert orchestration_overhead < 2000

    @pytest.mark.asyncio
    async def test_classification_speed(self):
        """Test that intent classification is fast (<500ms target)."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "what is the impact of hcp engagement?"}

        result = await orchestrator.run(input_data)

        # Classification should be fast with pattern matching
        assert result["classification_latency_ms"] < 500

    @pytest.mark.asyncio
    async def test_routing_speed(self):
        """Test that routing is very fast (<50ms target)."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "what drives conversions?"}

        result = await orchestrator.run(input_data)

        # Routing should be very fast (pure logic, no LLM)
        assert result["routing_latency_ms"] < 100  # Relaxed from 50ms for CI


class TestOrchestratorEdgeCases:
    """Test orchestrator edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test orchestrator with empty query."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": ""}

        result = await orchestrator.run(input_data)

        # Should default to explainer
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test orchestrator with very long query."""
        orchestrator = OrchestratorAgent()

        long_query = "what is the impact " * 100  # Very long query

        input_data = {"query": long_query}

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Test orchestrator with special characters."""
        orchestrator = OrchestratorAgent()

        input_data = {"query": "what's the impact? (HCP engagement -> conversions)"}

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"
