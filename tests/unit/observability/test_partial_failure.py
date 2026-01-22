"""Tests for partial failure handling in the orchestrator.

This module tests graceful degradation when some agents fail while others succeed.
Key scenarios:
- Partial success: some agents succeed, some fail
- Complete failure: all agents fail
- Complete success: all agents succeed
- User-friendly warning message generation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from src.agents.orchestrator.agent import OrchestratorAgent


class TestOrchestratorPartialFailure:
    """Test OrchestratorAgent's partial failure handling."""

    @pytest.fixture
    def orchestrator(self) -> OrchestratorAgent:
        """Create orchestrator instance for testing."""
        return OrchestratorAgent(agent_registry={}, enable_checkpointing=False)

    def test_build_output_complete_success(self, orchestrator: OrchestratorAgent):
        """Test _build_output when all agents succeed."""
        state = {
            "query_id": "q-test123",
            "status": "completed",
            "synthesized_response": "Analysis complete.",
            "response_confidence": 0.95,
            "agent_results": [
                {"agent_name": "gap_analyzer", "success": True, "latency_ms": 100},
                {"agent_name": "causal_impact", "success": True, "latency_ms": 150},
            ],
            "total_latency_ms": 500,
            "intent": {"primary_intent": "causal_effect", "confidence": 0.9},
        }

        output = orchestrator._build_output(state)

        assert output["status"] == "completed"
        assert output["has_partial_failure"] is False
        assert output["successful_agents"] == ["gap_analyzer", "causal_impact"]
        assert output["failed_agents"] == []
        assert output["failure_details"] is None
        assert output["response_text"] == "Analysis complete."

    def test_build_output_partial_failure(self, orchestrator: OrchestratorAgent):
        """Test _build_output when some agents fail but others succeed."""
        state = {
            "query_id": "q-test456",
            "status": "completed",
            "synthesized_response": "Partial analysis available.",
            "response_confidence": 0.7,
            "agent_results": [
                {"agent_name": "gap_analyzer", "success": True, "latency_ms": 100},
                {
                    "agent_name": "causal_impact",
                    "success": False,
                    "error": "Connection timed out",
                    "latency_ms": 5000,
                },
                {"agent_name": "explainer", "success": True, "latency_ms": 200},
            ],
            "total_latency_ms": 5500,
            "intent": {"primary_intent": "causal_effect", "confidence": 0.85},
        }

        output = orchestrator._build_output(state)

        assert output["status"] == "partial_success"
        assert output["has_partial_failure"] is True
        assert output["successful_agents"] == ["gap_analyzer", "explainer"]
        assert output["failed_agents"] == ["causal_impact"]
        assert output["failure_details"] is not None
        assert len(output["failure_details"]) == 1
        assert output["failure_details"][0]["agent_name"] == "causal_impact"
        assert output["failure_details"][0]["error"] == "Connection timed out"
        assert output["failure_details"][0]["latency_ms"] == 5000

    def test_build_output_complete_failure(self, orchestrator: OrchestratorAgent):
        """Test _build_output when all agents fail."""
        state = {
            "query_id": "q-test789",
            "status": "failed",
            "synthesized_response": "",
            "response_confidence": 0.0,
            "agent_results": [
                {
                    "agent_name": "gap_analyzer",
                    "success": False,
                    "error": "Service unavailable",
                    "latency_ms": 1000,
                },
                {
                    "agent_name": "causal_impact",
                    "success": False,
                    "error": "Timeout",
                    "latency_ms": 5000,
                },
            ],
            "error": "All agents failed",
            "error_type": "agent_failure",
            "total_latency_ms": 6000,
        }

        output = orchestrator._build_output(state)

        # When all agents fail, it's not "partial_success" - it stays "failed"
        assert output["status"] == "failed"
        assert output["has_partial_failure"] is False  # No successful agents
        assert output["successful_agents"] == []
        assert output["failed_agents"] == ["gap_analyzer", "causal_impact"]
        assert output["failure_details"] is not None
        assert len(output["failure_details"]) == 2
        assert output["orchestrator_error"] == "All agents failed"

    def test_build_output_no_agents_dispatched(self, orchestrator: OrchestratorAgent):
        """Test _build_output when no agents were dispatched."""
        state = {
            "query_id": "q-empty",
            "status": "completed",
            "synthesized_response": "No analysis needed.",
            "response_confidence": 1.0,
            "agent_results": [],
            "total_latency_ms": 50,
        }

        output = orchestrator._build_output(state)

        assert output["status"] == "completed"
        assert output["has_partial_failure"] is False
        assert output["successful_agents"] == []
        assert output["failed_agents"] == []
        assert output["failure_details"] is None

    def test_build_output_includes_all_latency_fields(
        self, orchestrator: OrchestratorAgent
    ):
        """Test that _build_output includes all latency breakdown fields."""
        state = {
            "query_id": "q-latency",
            "status": "completed",
            "synthesized_response": "Done.",
            "response_confidence": 0.9,
            "agent_results": [
                {"agent_name": "gap_analyzer", "success": True, "latency_ms": 100},
            ],
            "total_latency_ms": 500,
            "classification_latency_ms": 50,
            "rag_latency_ms": 75,
            "routing_latency_ms": 25,
            "dispatch_latency_ms": 300,
            "synthesis_latency_ms": 50,
            "intent": {"primary_intent": "gap_analysis", "confidence": 0.92},
        }

        output = orchestrator._build_output(state)

        assert output["total_latency_ms"] == 500
        assert output["classification_latency_ms"] == 50
        assert output["rag_latency_ms"] == 75
        assert output["routing_latency_ms"] == 25
        assert output["dispatch_latency_ms"] == 300
        assert output["synthesis_latency_ms"] == 50
        assert output["intent_classified"] == "gap_analysis"
        assert output["intent_confidence"] == 0.92

    def test_build_output_handles_missing_error_field(
        self, orchestrator: OrchestratorAgent
    ):
        """Test graceful handling when error field is missing from failed result."""
        state = {
            "query_id": "q-missing-error",
            "status": "completed",
            "synthesized_response": "Partial result.",
            "response_confidence": 0.6,
            "agent_results": [
                {"agent_name": "explainer", "success": True, "latency_ms": 100},
                {
                    "agent_name": "causal_impact",
                    "success": False,
                    # No "error" field
                    "latency_ms": 3000,
                },
            ],
            "total_latency_ms": 3200,
        }

        output = orchestrator._build_output(state)

        assert output["status"] == "partial_success"
        assert output["failure_details"][0]["error"] == "Unknown error"


class TestChatbotGraphPartialFailureWarning:
    """Test the partial failure warning message builder."""

    def test_build_partial_failure_warning_single_agent(self):
        """Test warning message for single agent failure."""
        # Import the helper function
        from src.api.routes.chatbot_graph import _build_partial_failure_warning

        failed_agents = ["causal_impact"]
        failure_details = [
            {
                "agent_name": "causal_impact",
                "error": "Connection timed out",
                "latency_ms": 5000,
            }
        ]

        warning = _build_partial_failure_warning(failed_agents, failure_details)

        assert "One analysis component" in warning
        assert "causal_impact" in warning
        assert "took too long" in warning  # Timeout message
        assert "results above are based on" in warning

    def test_build_partial_failure_warning_multiple_agents(self):
        """Test warning message for multiple agent failures."""
        from src.api.routes.chatbot_graph import _build_partial_failure_warning

        failed_agents = ["causal_impact", "drift_monitor"]
        failure_details = [
            {
                "agent_name": "causal_impact",
                "error": "Service unavailable",
                "latency_ms": 2000,
            },
            {
                "agent_name": "drift_monitor",
                "error": "Connection refused",
                "latency_ms": 1000,
            },
        ]

        warning = _build_partial_failure_warning(failed_agents, failure_details)

        assert "2 analysis components" in warning
        assert "causal_impact" in warning
        assert "drift_monitor" in warning
        # Both errors contain "connection" or "unavailable" keywords,
        # so both get converted to user-friendly "temporarily unavailable" message
        assert "unavailable" in warning

    def test_build_partial_failure_warning_empty_list(self):
        """Test that empty failure list returns empty string."""
        from src.api.routes.chatbot_graph import _build_partial_failure_warning

        warning = _build_partial_failure_warning([], [])

        assert warning == ""

    def test_build_partial_failure_warning_timeout_message(self):
        """Test timeout errors get user-friendly message."""
        from src.api.routes.chatbot_graph import _build_partial_failure_warning

        failed_agents = ["prediction_synthesizer"]
        failure_details = [
            {
                "agent_name": "prediction_synthesizer",
                "error": "Operation timed out after 30 seconds",
                "latency_ms": 30000,
            }
        ]

        warning = _build_partial_failure_warning(failed_agents, failure_details)

        assert "took too long" in warning
        assert "Operation timed out" not in warning  # Raw error not shown

    def test_build_partial_failure_warning_connection_message(self):
        """Test connection errors get user-friendly message."""
        from src.api.routes.chatbot_graph import _build_partial_failure_warning

        failed_agents = ["gap_analyzer"]
        failure_details = [
            {
                "agent_name": "gap_analyzer",
                "error": "Connection refused: upstream service unavailable",
                "latency_ms": 500,
            }
        ]

        warning = _build_partial_failure_warning(failed_agents, failure_details)

        assert "temporarily unavailable" in warning

    def test_build_partial_failure_warning_long_error_truncated(self):
        """Test that very long error messages are truncated."""
        from src.api.routes.chatbot_graph import _build_partial_failure_warning

        long_error = "x" * 200  # 200 character error message
        failed_agents = ["test_agent"]
        failure_details = [
            {
                "agent_name": "test_agent",
                "error": long_error,
                "latency_ms": 1000,
            }
        ]

        warning = _build_partial_failure_warning(failed_agents, failure_details)

        # Error should be truncated to 100 chars + "..."
        assert "..." in warning
        assert len(warning) < len(long_error) + 100  # Warning is shorter than raw error


class TestOrchestratorRunNoRuntimeError:
    """Test that OrchestratorAgent.run() doesn't raise RuntimeError for partial failures."""

    @pytest.fixture
    def orchestrator(self) -> OrchestratorAgent:
        """Create orchestrator with mocked graph."""
        orch = OrchestratorAgent(agent_registry={}, enable_checkpointing=False)
        return orch

    @pytest.mark.asyncio
    async def test_run_returns_result_on_partial_failure(
        self, orchestrator: OrchestratorAgent
    ):
        """Test that run() returns results even when some agents fail."""
        # Mock the graph to return a partial failure state
        mock_final_state = {
            "query_id": "q-partial",
            "status": "completed",
            "synthesized_response": "Here's what we found from the successful agents.",
            "response_confidence": 0.75,
            "agent_results": [
                {"agent_name": "explainer", "success": True, "latency_ms": 100},
                {
                    "agent_name": "causal_impact",
                    "success": False,
                    "error": "Timeout",
                    "latency_ms": 5000,
                },
            ],
            "total_latency_ms": 5200,
            "classification_latency_ms": 50,
            "rag_latency_ms": 0,
            "routing_latency_ms": 25,
            "dispatch_latency_ms": 5100,
            "synthesis_latency_ms": 25,
            "intent": {"primary_intent": "explanation", "confidence": 0.88},
        }

        orchestrator.graph = AsyncMock()
        orchestrator.graph.ainvoke = AsyncMock(return_value=mock_final_state)

        # Should NOT raise RuntimeError
        result = await orchestrator.run({"query": "What happened with sales?"})

        assert result["status"] == "partial_success"
        assert result["has_partial_failure"] is True
        assert "Here's what we found" in result["response_text"]
        assert result["successful_agents"] == ["explainer"]
        assert result["failed_agents"] == ["causal_impact"]

    @pytest.mark.asyncio
    async def test_run_returns_result_on_complete_failure(
        self, orchestrator: OrchestratorAgent
    ):
        """Test that run() returns structured result even when all agents fail."""
        mock_final_state = {
            "query_id": "q-failed",
            "status": "failed",
            "synthesized_response": "",
            "response_confidence": 0.0,
            "agent_results": [
                {
                    "agent_name": "gap_analyzer",
                    "success": False,
                    "error": "Service unavailable",
                    "latency_ms": 1000,
                },
            ],
            "error": "All agents failed",
            "error_type": "agent_failure",
            "total_latency_ms": 1100,
            "classification_latency_ms": 50,
            "rag_latency_ms": 0,
            "routing_latency_ms": 25,
            "dispatch_latency_ms": 1000,
            "synthesis_latency_ms": 25,
        }

        orchestrator.graph = AsyncMock()
        orchestrator.graph.ainvoke = AsyncMock(return_value=mock_final_state)

        # Should NOT raise RuntimeError even on complete failure
        result = await orchestrator.run({"query": "Analyze the gap."})

        assert result["status"] == "failed"
        assert result["has_partial_failure"] is False
        assert result["failed_agents"] == ["gap_analyzer"]
        assert result["orchestrator_error"] == "All agents failed"


class TestPartialFailureIntegration:
    """Integration tests for partial failure handling end-to-end."""

    @pytest.mark.asyncio
    async def test_partial_failure_preserves_successful_results(self):
        """Test that successful agent results are preserved despite failures."""
        orchestrator = OrchestratorAgent(agent_registry={}, enable_checkpointing=False)

        # Simulate a state where one agent succeeded with good results
        mock_final_state = {
            "query_id": "q-integration",
            "status": "completed",
            "synthesized_response": "Gap analysis found 3 opportunities worth $2M.",
            "response_confidence": 0.82,
            "agent_results": [
                {
                    "agent_name": "gap_analyzer",
                    "success": True,
                    "latency_ms": 150,
                    "output": {
                        "gaps": [
                            {"territory": "Northeast", "potential": "$800K"},
                            {"territory": "West", "potential": "$700K"},
                            {"territory": "South", "potential": "$500K"},
                        ]
                    },
                },
                {
                    "agent_name": "causal_impact",
                    "success": False,
                    "error": "Model inference failed: GPU memory exhausted",
                    "latency_ms": 8000,
                },
            ],
            "total_latency_ms": 8500,
            "classification_latency_ms": 50,
            "rag_latency_ms": 0,
            "routing_latency_ms": 25,
            "dispatch_latency_ms": 8400,
            "synthesis_latency_ms": 25,
            "intent": {"primary_intent": "gap_analysis", "confidence": 0.91},
        }

        orchestrator.graph = AsyncMock()
        orchestrator.graph.ainvoke = AsyncMock(return_value=mock_final_state)

        result = await orchestrator.run({"query": "Find opportunities in all regions."})

        # Successful results should be preserved
        assert result["status"] == "partial_success"
        assert "Gap analysis found 3 opportunities" in result["response_text"]
        assert result["response_confidence"] == 0.82

        # Failed agent info should be available for display
        assert result["has_partial_failure"] is True
        assert len(result["failure_details"]) == 1
        assert "causal_impact" in result["failed_agents"]
        assert "GPU memory" in result["failure_details"][0]["error"]

        # Successful agent should be tracked
        assert "gap_analyzer" in result["successful_agents"]
