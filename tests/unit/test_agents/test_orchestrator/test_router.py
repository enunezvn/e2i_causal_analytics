"""Tests for router node."""

import pytest
from src.agents.orchestrator.nodes.router import RouterNode, route_to_agents


class TestRouterNode:
    """Test RouterNode."""

    @pytest.mark.asyncio
    async def test_route_causal_effect(self):
        """Test routing for causal_effect intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "causal_effect",
                "confidence": 0.95,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert "dispatch_plan" in result
        assert len(result["dispatch_plan"]) == 1
        assert result["dispatch_plan"][0]["agent_name"] == "causal_impact"
        assert result["dispatch_plan"][0]["priority"] == 1
        assert result["dispatch_plan"][0]["timeout_ms"] == 30000
        assert result["dispatch_plan"][0]["fallback_agent"] == "explainer"
        assert result["current_phase"] == "dispatching"
        assert result["routing_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_route_performance_gap(self):
        """Test routing for performance_gap intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "performance_gap",
                "confidence": 0.92,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "gap_analyzer"
        assert result["dispatch_plan"][0]["priority"] == 1
        assert result["dispatch_plan"][0]["timeout_ms"] == 20000
        assert result["dispatch_plan"][0]["fallback_agent"] is None

    @pytest.mark.asyncio
    async def test_route_segment_analysis(self):
        """Test routing for segment_analysis intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "segment_analysis",
                "confidence": 0.88,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "heterogeneous_optimizer"
        assert result["dispatch_plan"][0]["fallback_agent"] == "gap_analyzer"

    @pytest.mark.asyncio
    async def test_route_experiment_design(self):
        """Test routing for experiment_design intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "experiment_design",
                "confidence": 0.90,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "experiment_designer"
        assert result["dispatch_plan"][0]["timeout_ms"] == 60000
        assert result["dispatch_plan"][0]["parameters"] == {
            "preregistration_formality": "medium"
        }

    @pytest.mark.asyncio
    async def test_route_prediction(self):
        """Test routing for prediction intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "prediction",
                "confidence": 0.93,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "prediction_synthesizer"
        assert result["dispatch_plan"][0]["timeout_ms"] == 15000

    @pytest.mark.asyncio
    async def test_route_resource_allocation(self):
        """Test routing for resource_allocation intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "resource_allocation",
                "confidence": 0.87,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "resource_optimizer"
        assert result["dispatch_plan"][0]["timeout_ms"] == 20000

    @pytest.mark.asyncio
    async def test_route_explanation(self):
        """Test routing for explanation intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "explanation",
                "confidence": 0.91,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "explainer"
        assert result["dispatch_plan"][0]["timeout_ms"] == 45000
        assert result["dispatch_plan"][0]["parameters"] == {"depth": "standard"}

    @pytest.mark.asyncio
    async def test_route_system_health(self):
        """Test routing for system_health intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "system_health",
                "confidence": 0.96,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "health_score"
        assert result["dispatch_plan"][0]["timeout_ms"] == 5000

    @pytest.mark.asyncio
    async def test_route_drift_check(self):
        """Test routing for drift_check intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "drift_check",
                "confidence": 0.89,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "drift_monitor"
        assert result["dispatch_plan"][0]["timeout_ms"] == 10000

    @pytest.mark.asyncio
    async def test_route_feedback(self):
        """Test routing for feedback intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "feedback",
                "confidence": 0.85,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert result["dispatch_plan"][0]["agent_name"] == "feedback_learner"
        assert result["dispatch_plan"][0]["timeout_ms"] == 30000

    @pytest.mark.asyncio
    async def test_route_multi_agent_causal_segment(self):
        """Test multi-agent routing for causal_effect + segment_analysis."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "causal_effect",
                "confidence": 0.90,
                "secondary_intents": ["segment_analysis"],
                "requires_multi_agent": True,
            }
        }

        result = await router.execute(state)

        assert len(result["dispatch_plan"]) == 2
        assert result["dispatch_plan"][0]["agent_name"] == "causal_impact"
        assert result["dispatch_plan"][0]["priority"] == 1
        assert result["dispatch_plan"][1]["agent_name"] == "heterogeneous_optimizer"
        assert result["dispatch_plan"][1]["priority"] == 2

        # Check parallel groups
        assert len(result["parallel_groups"]) == 2
        assert result["parallel_groups"][0] == ["causal_impact"]
        assert result["parallel_groups"][1] == ["heterogeneous_optimizer"]

    @pytest.mark.asyncio
    async def test_route_multi_agent_performance_resource(self):
        """Test multi-agent routing for performance_gap + resource_allocation."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "performance_gap",
                "confidence": 0.88,
                "secondary_intents": ["resource_allocation"],
                "requires_multi_agent": True,
            }
        }

        result = await router.execute(state)

        assert len(result["dispatch_plan"]) == 2
        assert result["dispatch_plan"][0]["agent_name"] == "gap_analyzer"
        assert result["dispatch_plan"][0]["priority"] == 1
        assert result["dispatch_plan"][1]["agent_name"] == "resource_optimizer"
        assert result["dispatch_plan"][1]["priority"] == 2

    @pytest.mark.asyncio
    async def test_route_multi_agent_prediction_explanation(self):
        """Test multi-agent routing for prediction + explanation."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "prediction",
                "confidence": 0.91,
                "secondary_intents": ["explanation"],
                "requires_multi_agent": True,
            }
        }

        result = await router.execute(state)

        assert len(result["dispatch_plan"]) == 2
        assert result["dispatch_plan"][0]["agent_name"] == "prediction_synthesizer"
        assert result["dispatch_plan"][0]["priority"] == 1
        assert result["dispatch_plan"][1]["agent_name"] == "explainer"
        assert result["dispatch_plan"][1]["priority"] == 2

    @pytest.mark.asyncio
    async def test_route_no_intent(self):
        """Test default routing when no intent is provided."""
        router = RouterNode()

        state = {}

        result = await router.execute(state)

        assert len(result["dispatch_plan"]) == 1
        assert result["dispatch_plan"][0]["agent_name"] == "explainer"
        assert result["dispatch_plan"][0]["parameters"] == {"depth": "minimal"}
        assert result["dispatch_plan"][0]["timeout_ms"] == 30000
        assert result["parallel_groups"] == [["explainer"]]

    @pytest.mark.asyncio
    async def test_route_unknown_intent(self):
        """Test default routing for unknown intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "unknown_intent_type",
                "confidence": 0.60,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        # Should default to explainer
        assert len(result["dispatch_plan"]) == 1
        assert result["dispatch_plan"][0]["agent_name"] == "explainer"
        assert result["dispatch_plan"][0]["parameters"] == {"depth": "minimal"}

    @pytest.mark.asyncio
    async def test_route_general_intent(self):
        """Test routing for general intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "general",
                "confidence": 0.50,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        # General intent should default to explainer
        assert result["dispatch_plan"][0]["agent_name"] == "explainer"

    @pytest.mark.asyncio
    async def test_parallel_groups_single_agent(self):
        """Test parallel groups creation for single agent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "causal_effect",
                "confidence": 0.95,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert len(result["parallel_groups"]) == 1
        assert result["parallel_groups"][0] == ["causal_impact"]

    @pytest.mark.asyncio
    async def test_routing_latency_measurement(self):
        """Test that routing latency is measured."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "prediction",
                "confidence": 0.93,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        assert "routing_latency_ms" in result
        assert result["routing_latency_ms"] >= 0
        assert result["routing_latency_ms"] < 100  # Should be very fast (<100ms)

    @pytest.mark.asyncio
    async def test_route_to_agents_function(self):
        """Test standalone route_to_agents function."""
        state = {
            "intent": {
                "primary_intent": "causal_effect",
                "confidence": 0.95,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await route_to_agents(state)

        assert "dispatch_plan" in result
        assert result["dispatch_plan"][0]["agent_name"] == "causal_impact"


class TestRouterHelperMethods:
    """Test router helper methods."""

    def test_group_by_priority_single_priority(self):
        """Test grouping agents with same priority."""
        router = RouterNode()

        dispatches = [
            {"agent_name": "agent1", "priority": 1},
            {"agent_name": "agent2", "priority": 1},
            {"agent_name": "agent3", "priority": 1},
        ]

        groups = router._group_by_priority(dispatches)

        assert len(groups) == 1
        assert set(groups[0]) == {"agent1", "agent2", "agent3"}

    def test_group_by_priority_multiple_priorities(self):
        """Test grouping agents with different priorities."""
        router = RouterNode()

        dispatches = [
            {"agent_name": "agent1", "priority": 1},
            {"agent_name": "agent2", "priority": 2},
            {"agent_name": "agent3", "priority": 1},
            {"agent_name": "agent4", "priority": 3},
        ]

        groups = router._group_by_priority(dispatches)

        assert len(groups) == 3
        assert set(groups[0]) == {"agent1", "agent3"}  # Priority 1
        assert set(groups[1]) == {"agent2"}  # Priority 2
        assert set(groups[2]) == {"agent4"}  # Priority 3

    def test_group_by_priority_empty(self):
        """Test grouping with empty dispatch list."""
        router = RouterNode()

        groups = router._group_by_priority([])

        assert groups == []

    def test_get_dispatch_for_agent_existing(self):
        """Test getting dispatch config for existing agent."""
        router = RouterNode()

        dispatch = router._get_dispatch_for_agent("causal_impact", priority=2)

        assert dispatch["agent_name"] == "causal_impact"
        assert dispatch["priority"] == 2  # Override priority
        assert dispatch["timeout_ms"] == 30000
        assert dispatch["fallback_agent"] == "explainer"
        assert dispatch["parameters"] == {"interpretation_depth": "standard"}

    def test_get_dispatch_for_agent_default(self):
        """Test getting dispatch config for unknown agent."""
        router = RouterNode()

        dispatch = router._get_dispatch_for_agent("unknown_agent", priority=3)

        assert dispatch["agent_name"] == "unknown_agent"
        assert dispatch["priority"] == 3
        assert dispatch["timeout_ms"] == 30000
        assert dispatch["fallback_agent"] is None
        assert dispatch["parameters"] == {}


class TestIntentToAgentMapping:
    """Test INTENT_TO_AGENTS mapping coverage."""

    def test_all_intents_have_mappings(self):
        """Test that all expected intents have agent mappings."""
        router = RouterNode()

        expected_intents = [
            "causal_effect",
            "performance_gap",
            "segment_analysis",
            "experiment_design",
            "prediction",
            "resource_allocation",
            "explanation",
            "system_health",
            "drift_check",
            "feedback",
        ]

        for intent in expected_intents:
            assert (
                intent in router.INTENT_TO_AGENTS
            ), f"Missing mapping for intent: {intent}"

    def test_all_dispatches_have_required_fields(self):
        """Test that all agent dispatches have required fields."""
        router = RouterNode()

        required_fields = [
            "agent_name",
            "priority",
            "parameters",
            "timeout_ms",
            "fallback_agent",
        ]

        for intent, dispatches in router.INTENT_TO_AGENTS.items():
            for dispatch in dispatches:
                for field in required_fields:
                    assert (
                        field in dispatch
                    ), f"Missing field '{field}' in {intent} dispatch"

    def test_timeout_configurations(self):
        """Test timeout configurations are reasonable."""
        router = RouterNode()

        for intent, dispatches in router.INTENT_TO_AGENTS.items():
            for dispatch in dispatches:
                timeout = dispatch["timeout_ms"]
                assert timeout > 0, f"Invalid timeout for {intent}"
                assert timeout <= 60000, f"Timeout too high for {intent}"

    def test_multi_agent_patterns_exist(self):
        """Test that multi-agent patterns are defined."""
        router = RouterNode()

        expected_patterns = [
            ("causal_effect", "segment_analysis"),
            ("performance_gap", "resource_allocation"),
            ("prediction", "explanation"),
        ]

        for pattern in expected_patterns:
            assert (
                pattern in router.MULTI_AGENT_PATTERNS
            ), f"Missing multi-agent pattern: {pattern}"

    def test_multi_agent_patterns_have_priorities(self):
        """Test that multi-agent patterns define priorities."""
        router = RouterNode()

        for pattern, agents in router.MULTI_AGENT_PATTERNS.items():
            priorities = [priority for _, priority in agents]
            assert (
                len(priorities) == len(set(priorities)) or len(priorities) <= 1
            ), f"Duplicate priorities in pattern {pattern}"
            assert all(p > 0 for p in priorities), f"Invalid priority in pattern {pattern}"
