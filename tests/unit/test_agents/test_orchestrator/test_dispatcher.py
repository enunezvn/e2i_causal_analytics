"""Tests for dispatcher node."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator.nodes.dispatcher import (
    DispatcherNode,
    dispatch_to_agents,
)


class TestDispatcherNode:
    """Test DispatcherNode."""

    @pytest.mark.asyncio
    async def test_dispatch_single_agent_mock(self):
        """Test dispatching to a single agent with mock execution."""
        dispatcher = DispatcherNode()

        state = {
            "query": "what drives conversions?",
            "dispatch_plan": [
                {
                    "agent_name": "causal_impact",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                }
            ],
            "parallel_groups": [["causal_impact"]],
        }

        result = await dispatcher.execute(state)

        assert "agent_results" in result
        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["agent_name"] == "causal_impact"
        assert result["agent_results"][0]["success"] is True
        assert result["agent_results"][0]["result"] is not None
        assert "narrative" in result["agent_results"][0]["result"]
        assert result["current_phase"] == "synthesizing"
        assert result["dispatch_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_dispatch_multiple_agents_parallel(self):
        """Test dispatching to multiple agents in parallel."""
        dispatcher = DispatcherNode()

        state = {
            "query": "what drives conversions and how do we improve?",
            "dispatch_plan": [
                {
                    "agent_name": "causal_impact",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                },
                {
                    "agent_name": "gap_analyzer",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 20000,
                    "fallback_agent": None,
                },
            ],
            "parallel_groups": [["causal_impact", "gap_analyzer"]],
        }

        result = await dispatcher.execute(state)

        assert len(result["agent_results"]) == 2
        agent_names = [r["agent_name"] for r in result["agent_results"]]
        assert "causal_impact" in agent_names
        assert "gap_analyzer" in agent_names
        assert all(r["success"] for r in result["agent_results"])

    @pytest.mark.asyncio
    async def test_dispatch_sequential_groups(self):
        """Test sequential execution of parallel groups."""
        dispatcher = DispatcherNode()

        state = {
            "query": "analyze conversions by segment",
            "dispatch_plan": [
                {
                    "agent_name": "causal_impact",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                },
                {
                    "agent_name": "heterogeneous_optimizer",
                    "priority": 2,
                    "parameters": {},
                    "timeout_ms": 25000,
                    "fallback_agent": None,
                },
            ],
            "parallel_groups": [["causal_impact"], ["heterogeneous_optimizer"]],
        }

        result = await dispatcher.execute(state)

        # Both agents should execute (sequentially by group)
        assert len(result["agent_results"]) == 2
        assert result["agent_results"][0]["agent_name"] == "causal_impact"
        assert result["agent_results"][1]["agent_name"] == "heterogeneous_optimizer"

    @pytest.mark.asyncio
    async def test_dispatch_with_real_agent(self):
        """Test dispatching with a real agent in registry."""
        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(
            return_value={
                "narrative": "Real agent response",
                "recommendations": ["Test recommendation"],
                "confidence": 0.95,
            }
        )

        # Create dispatcher with agent registry
        dispatcher = DispatcherNode(agent_registry={"test_agent": mock_agent})

        state = {
            "query": "test query",
            "user_context": {"expertise": "analyst"},
            "dispatch_plan": [
                {
                    "agent_name": "test_agent",
                    "priority": 1,
                    "parameters": {"param1": "value1"},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                }
            ],
            "parallel_groups": [["test_agent"]],
        }

        result = await dispatcher.execute(state)

        # Verify agent was called
        assert mock_agent.analyze.called
        call_args = mock_agent.analyze.call_args[0][0]
        assert call_args["query"] == "test query"
        assert call_args["user_context"] == {"expertise": "analyst"}
        assert call_args["parameters"] == {"param1": "value1"}

        # Verify result
        assert result["agent_results"][0]["success"] is True
        assert result["agent_results"][0]["result"]["narrative"] == "Real agent response"

    @pytest.mark.asyncio
    async def test_dispatch_timeout(self):
        """Test agent timeout handling."""
        # Create mock agent that takes too long
        mock_agent = MagicMock()

        async def slow_analyze(_):
            await asyncio.sleep(2)  # 2 seconds
            return {"narrative": "Should not see this"}

        mock_agent.analyze = AsyncMock(side_effect=slow_analyze)

        dispatcher = DispatcherNode(agent_registry={"slow_agent": mock_agent})

        state = {
            "query": "test query",
            "dispatch_plan": [
                {
                    "agent_name": "slow_agent",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 100,  # 100ms timeout
                    "fallback_agent": None,
                }
            ],
            "parallel_groups": [["slow_agent"]],
        }

        result = await dispatcher.execute(state)

        # Should timeout
        assert result["agent_results"][0]["success"] is False
        assert "timed out" in result["agent_results"][0]["error"].lower()
        assert result["agent_results"][0]["latency_ms"] == 100

    @pytest.mark.asyncio
    async def test_dispatch_exception_handling(self):
        """Test exception handling during dispatch."""
        # Create mock agent that raises exception
        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(side_effect=ValueError("Test error"))

        dispatcher = DispatcherNode(agent_registry={"error_agent": mock_agent})

        state = {
            "query": "test query",
            "dispatch_plan": [
                {
                    "agent_name": "error_agent",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                }
            ],
            "parallel_groups": [["error_agent"]],
        }

        result = await dispatcher.execute(state)

        # Should capture exception
        assert result["agent_results"][0]["success"] is False
        assert "Test error" in result["agent_results"][0]["error"]

    @pytest.mark.asyncio
    async def test_dispatch_fallback_on_error(self):
        """Test fallback agent invocation on error."""
        # Create mock agent that fails
        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(side_effect=RuntimeError("Primary failed"))

        dispatcher = DispatcherNode(agent_registry={"failing_agent": mock_agent})

        state = {
            "query": "test query",
            "dispatch_plan": [
                {
                    "agent_name": "failing_agent",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": "explainer",  # Fallback to explainer (mock)
                }
            ],
            "parallel_groups": [["failing_agent"]],
        }

        result = await dispatcher.execute(state)

        # Should have 2 results: failed primary and successful fallback
        assert len(result["agent_results"]) == 2
        assert result["agent_results"][0]["success"] is False  # Primary failed
        assert result["agent_results"][0]["agent_name"] == "failing_agent"
        assert result["agent_results"][1]["success"] is True  # Fallback succeeded
        assert result["agent_results"][1]["agent_name"] == "explainer"

    @pytest.mark.asyncio
    async def test_dispatch_fallback_on_timeout(self):
        """Test fallback agent invocation on timeout."""
        # Create mock agent that times out
        mock_agent = MagicMock()

        async def slow_analyze(_):
            await asyncio.sleep(1)
            return {"narrative": "Too slow"}

        mock_agent.analyze = AsyncMock(side_effect=slow_analyze)

        dispatcher = DispatcherNode(agent_registry={"slow_agent": mock_agent})

        state = {
            "query": "test query",
            "dispatch_plan": [
                {
                    "agent_name": "slow_agent",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 50,  # 50ms timeout
                    "fallback_agent": "causal_impact",  # Mock fallback
                }
            ],
            "parallel_groups": [["slow_agent"]],
        }

        result = await dispatcher.execute(state)

        # Should have 2 results: timed out primary and successful fallback
        assert len(result["agent_results"]) == 2
        assert result["agent_results"][0]["success"] is False
        assert "timed out" in result["agent_results"][0]["error"].lower()
        assert result["agent_results"][1]["success"] is True
        assert result["agent_results"][1]["agent_name"] == "causal_impact"

    @pytest.mark.asyncio
    async def test_dispatch_no_fallback_on_error(self):
        """Test behavior when error occurs and no fallback is specified."""
        # Create mock agent that fails
        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(side_effect=ValueError("Agent error"))

        dispatcher = DispatcherNode(agent_registry={"failing_agent": mock_agent})

        state = {
            "query": "test query",
            "dispatch_plan": [
                {
                    "agent_name": "failing_agent",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,  # No fallback
                }
            ],
            "parallel_groups": [["failing_agent"]],
        }

        result = await dispatcher.execute(state)

        # Should have 1 failed result, no fallback
        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["success"] is False

    @pytest.mark.asyncio
    async def test_dispatch_latency_measurement(self):
        """Test dispatch latency measurement."""
        dispatcher = DispatcherNode()

        state = {
            "query": "test query",
            "dispatch_plan": [
                {
                    "agent_name": "causal_impact",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                }
            ],
            "parallel_groups": [["causal_impact"]],
        }

        result = await dispatcher.execute(state)

        assert "dispatch_latency_ms" in result
        assert result["dispatch_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_dispatch_empty_plan(self):
        """Test dispatch with empty dispatch plan."""
        dispatcher = DispatcherNode()

        state = {"query": "test query", "dispatch_plan": [], "parallel_groups": []}

        result = await dispatcher.execute(state)

        assert result["agent_results"] == []
        assert result["current_phase"] == "synthesizing"

    @pytest.mark.asyncio
    async def test_dispatch_to_agents_function(self):
        """Test standalone dispatch_to_agents function."""
        state = {
            "query": "test query",
            "dispatch_plan": [
                {
                    "agent_name": "causal_impact",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                }
            ],
            "parallel_groups": [["causal_impact"]],
        }

        result = await dispatch_to_agents(state)

        assert "agent_results" in result
        assert len(result["agent_results"]) == 1


class TestMockAgentExecution:
    """Test mock agent execution."""

    @pytest.mark.asyncio
    async def test_mock_causal_impact(self):
        """Test mock execution for causal_impact agent."""
        dispatcher = DispatcherNode()

        dispatch = {
            "agent_name": "causal_impact",
            "priority": 1,
            "parameters": {},
            "timeout_ms": 30000,
            "fallback_agent": None,
        }

        result = await dispatcher._mock_agent_execution(dispatch, {})

        assert result["agent_name"] == "causal_impact"
        assert result["success"] is True
        assert "narrative" in result["result"]
        assert "recommendations" in result["result"]
        assert "confidence" in result["result"]
        assert result["latency_ms"] == 50

    @pytest.mark.asyncio
    async def test_mock_all_agent_types(self):
        """Test mock execution for all agent types."""
        dispatcher = DispatcherNode()

        agent_names = [
            "causal_impact",
            "gap_analyzer",
            "heterogeneous_optimizer",
            "prediction_synthesizer",
            "explainer",
            "resource_optimizer",
            "health_score",
            "drift_monitor",
            "experiment_designer",
            "feedback_learner",
        ]

        for agent_name in agent_names:
            dispatch = {
                "agent_name": agent_name,
                "priority": 1,
                "parameters": {},
                "timeout_ms": 30000,
                "fallback_agent": None,
            }

            result = await dispatcher._mock_agent_execution(dispatch, {})

            assert result["agent_name"] == agent_name
            assert result["success"] is True
            assert result["result"] is not None
            assert "narrative" in result["result"]

    @pytest.mark.asyncio
    async def test_mock_unknown_agent(self):
        """Test mock execution for unknown agent type."""
        dispatcher = DispatcherNode()

        dispatch = {
            "agent_name": "unknown_agent",
            "priority": 1,
            "parameters": {},
            "timeout_ms": 30000,
            "fallback_agent": None,
        }

        result = await dispatcher._mock_agent_execution(dispatch, {})

        # Should return default mock response
        assert result["success"] is True
        assert "Mock response from unknown_agent" in result["result"]["narrative"]


class TestAgentInputPreparation:
    """Test agent input preparation."""

    def test_prepare_agent_input_basic(self):
        """Test basic agent input preparation."""
        dispatcher = DispatcherNode()

        state = {
            "query": "what drives conversions?",
            "user_context": {"expertise": "analyst"},
        }

        dispatch = {"agent_name": "causal_impact", "parameters": {"depth": "deep"}}

        agent_input = dispatcher._prepare_agent_input(state, dispatch)

        assert agent_input["query"] == "what drives conversions?"
        assert agent_input["user_context"] == {"expertise": "analyst"}
        assert agent_input["parameters"] == {"depth": "deep"}

    def test_prepare_agent_input_no_context(self):
        """Test agent input preparation with no user context."""
        dispatcher = DispatcherNode()

        state = {"query": "test query"}
        dispatch = {"agent_name": "causal_impact", "parameters": {}}

        agent_input = dispatcher._prepare_agent_input(state, dispatch)

        assert agent_input["query"] == "test query"
        assert agent_input["user_context"] == {}
        assert agent_input["parameters"] == {}

    def test_prepare_agent_input_no_parameters(self):
        """Test agent input preparation with no parameters."""
        dispatcher = DispatcherNode()

        state = {"query": "test query", "user_context": {"expertise": "executive"}}
        dispatch = {"agent_name": "causal_impact"}

        agent_input = dispatcher._prepare_agent_input(state, dispatch)

        assert agent_input["parameters"] == {}


class TestParallelExecution:
    """Test parallel execution behavior."""

    @pytest.mark.asyncio
    async def test_parallel_execution_within_group(self):
        """Test that agents in same group execute in parallel."""
        import time as time_module
        from functools import partial

        # Create mock agents
        execution_times = []

        async def track_execution(agent_name, _input):
            start = time_module.time()
            await asyncio.sleep(0.1)  # 100ms
            end = time_module.time()
            execution_times.append((agent_name, start, end))
            return {
                "narrative": f"Response from {agent_name}",
                "recommendations": [],
                "confidence": 0.8,
            }

        mock_agent1 = MagicMock()
        mock_agent1.analyze = partial(track_execution, "agent1")

        mock_agent2 = MagicMock()
        mock_agent2.analyze = partial(track_execution, "agent2")

        dispatcher = DispatcherNode(agent_registry={"agent1": mock_agent1, "agent2": mock_agent2})

        state = {
            "query": "test",
            "dispatch_plan": [
                {
                    "agent_name": "agent1",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                },
                {
                    "agent_name": "agent2",
                    "priority": 1,
                    "parameters": {},
                    "timeout_ms": 30000,
                    "fallback_agent": None,
                },
            ],
            "parallel_groups": [["agent1", "agent2"]],
        }

        await dispatcher.execute(state)

        # Verify both executed
        assert len(execution_times) == 2

        # Verify they overlapped (parallel execution)
        # If sequential, agent2 would start after agent1 ends
        # If parallel, agent2 starts before agent1 ends
        _agent1_start, agent1_end = execution_times[0][1], execution_times[0][2]
        agent2_start, _agent2_end = execution_times[1][1], execution_times[1][2]

        # Agent2 should start before Agent1 finishes (parallel)
        assert agent2_start < agent1_end
