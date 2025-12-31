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
        assert result["dispatch_plan"][0]["priority"] == "critical"
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
        assert result["dispatch_plan"][0]["priority"] == "critical"
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
        assert result["dispatch_plan"][0]["parameters"] == {"preregistration_formality": "medium"}

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
        assert result["dispatch_plan"][0]["priority"] == "critical"
        assert result["dispatch_plan"][1]["agent_name"] == "heterogeneous_optimizer"
        assert result["dispatch_plan"][1]["priority"] == "high"

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
        assert result["dispatch_plan"][0]["priority"] == "critical"
        assert result["dispatch_plan"][1]["agent_name"] == "resource_optimizer"
        assert result["dispatch_plan"][1]["priority"] == "high"

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
        assert result["dispatch_plan"][0]["priority"] == "critical"
        assert result["dispatch_plan"][1]["agent_name"] == "explainer"
        assert result["dispatch_plan"][1]["priority"] == "high"

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
            {"agent_name": "agent1", "priority": "critical"},
            {"agent_name": "agent2", "priority": "critical"},
            {"agent_name": "agent3", "priority": "critical"},
        ]

        groups = router._group_by_priority(dispatches)

        assert len(groups) == 1
        assert set(groups[0]) == {"agent1", "agent2", "agent3"}

    def test_group_by_priority_multiple_priorities(self):
        """Test grouping agents with different priorities."""
        router = RouterNode()

        dispatches = [
            {"agent_name": "agent1", "priority": "critical"},
            {"agent_name": "agent2", "priority": "high"},
            {"agent_name": "agent3", "priority": "critical"},
            {"agent_name": "agent4", "priority": "medium"},
        ]

        groups = router._group_by_priority(dispatches)

        assert len(groups) == 3
        assert set(groups[0]) == {"agent1", "agent3"}  # Priority critical
        assert set(groups[1]) == {"agent2"}  # Priority high
        assert set(groups[2]) == {"agent4"}  # Priority medium

    def test_group_by_priority_empty(self):
        """Test grouping with empty dispatch list."""
        router = RouterNode()

        groups = router._group_by_priority([])

        assert groups == []

    def test_get_dispatch_for_agent_existing(self):
        """Test getting dispatch config for existing agent."""
        router = RouterNode()

        dispatch = router._get_dispatch_for_agent("causal_impact", priority="high")

        assert dispatch["agent_name"] == "causal_impact"
        assert dispatch["priority"] == "high"  # Override priority
        assert dispatch["timeout_ms"] == 30000
        assert dispatch["fallback_agent"] == "explainer"
        assert dispatch["parameters"] == {"interpretation_depth": "standard"}

    def test_get_dispatch_for_agent_default(self):
        """Test getting dispatch config for unknown agent."""
        router = RouterNode()

        dispatch = router._get_dispatch_for_agent("unknown_agent", priority="medium")

        assert dispatch["agent_name"] == "unknown_agent"
        assert dispatch["priority"] == "medium"
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
            assert intent in router.INTENT_TO_AGENTS, f"Missing mapping for intent: {intent}"

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
                    assert field in dispatch, f"Missing field '{field}' in {intent} dispatch"

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
            assert pattern in router.MULTI_AGENT_PATTERNS, f"Missing multi-agent pattern: {pattern}"

    def test_multi_agent_patterns_have_priorities(self):
        """Test that multi-agent patterns define priorities."""
        router = RouterNode()
        valid_priorities = {"critical", "high", "medium", "low"}

        for pattern, agents in router.MULTI_AGENT_PATTERNS.items():
            priorities = [priority for _, priority in agents]
            assert (
                len(priorities) == len(set(priorities)) or len(priorities) <= 1
            ), f"Duplicate priorities in pattern {pattern}"
            assert all(
                p in valid_priorities for p in priorities
            ), f"Invalid priority in pattern {pattern}"


# ============================================================================
# V4.4: Discovery Routing Tests
# ============================================================================


class TestShouldApplyDiscoveryRouting:
    """Test _should_apply_discovery_routing method."""

    def test_apply_when_enable_discovery_true(self):
        """Should apply discovery routing when enable_discovery is True."""
        router = RouterNode()

        state = {"enable_discovery": True}
        assert router._should_apply_discovery_routing(state) is True

    def test_apply_when_propagate_dag_true(self):
        """Should apply discovery routing when propagate_discovered_dag is True."""
        router = RouterNode()

        state = {"propagate_discovered_dag": True}
        assert router._should_apply_discovery_routing(state) is True

    def test_not_apply_when_both_false(self):
        """Should not apply discovery routing when both flags are False."""
        router = RouterNode()

        state = {"enable_discovery": False, "propagate_discovered_dag": False}
        assert router._should_apply_discovery_routing(state) is False

    def test_not_apply_when_missing(self):
        """Should not apply discovery routing when flags are missing."""
        router = RouterNode()

        state = {}
        assert router._should_apply_discovery_routing(state) is False

    def test_not_apply_when_gate_rejected(self):
        """Should not apply discovery routing when gate decision is reject."""
        router = RouterNode()

        state = {
            "enable_discovery": True,
            "discovery_gate_decision": "reject",
        }
        assert router._should_apply_discovery_routing(state) is False

    def test_apply_when_gate_accept(self):
        """Should apply discovery routing when gate decision is accept."""
        router = RouterNode()

        state = {
            "enable_discovery": True,
            "discovery_gate_decision": "accept",
        }
        assert router._should_apply_discovery_routing(state) is True

    def test_apply_when_gate_review(self):
        """Should apply discovery routing when gate decision is review."""
        router = RouterNode()

        state = {
            "enable_discovery": True,
            "discovery_gate_decision": "review",
        }
        assert router._should_apply_discovery_routing(state) is True

    def test_apply_when_gate_augment(self):
        """Should apply discovery routing when gate decision is augment."""
        router = RouterNode()

        state = {
            "enable_discovery": True,
            "discovery_gate_decision": "augment",
        }
        assert router._should_apply_discovery_routing(state) is True


class TestEnhanceWithDiscoveryData:
    """Test _enhance_with_discovery_data method."""

    @pytest.fixture
    def sample_dag_adjacency(self):
        """Sample DAG adjacency matrix."""
        return [[0, 1, 0], [0, 0, 1], [0, 0, 0]]

    @pytest.fixture
    def sample_dag_nodes(self):
        """Sample DAG nodes."""
        return ["treatment", "segment", "outcome"]

    @pytest.fixture
    def sample_state(self, sample_dag_adjacency, sample_dag_nodes):
        """Sample state with discovery data."""
        return {
            "enable_discovery": True,
            "propagate_discovered_dag": True,
            "discovery_config": {"algorithms": ["ges", "pc"], "threshold": 0.5},
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {"treatment->segment": "DIRECTED"},
            "discovery_gate_decision": "accept",
            "discovery_gate_confidence": 0.85,
        }

    def test_enhance_discovery_aware_agent(self, sample_state):
        """Should enhance discovery-aware agent with DAG data."""
        router = RouterNode()

        dispatch_plan = [
            {
                "agent_name": "causal_impact",
                "priority": "critical",
                "parameters": {"interpretation_depth": "standard"},
                "timeout_ms": 30000,
                "fallback_agent": "explainer",
            }
        ]

        enhanced, aware_agents = router._enhance_with_discovery_data(
            dispatch_plan, sample_state
        )

        assert len(enhanced) == 1
        assert enhanced[0]["agent_name"] == "causal_impact"
        assert "discovered_dag_adjacency" in enhanced[0]["parameters"]
        assert "discovered_dag_nodes" in enhanced[0]["parameters"]
        assert "discovery_gate_decision" in enhanced[0]["parameters"]
        assert enhanced[0]["parameters"]["discovery_gate_confidence"] == 0.85
        assert "causal_impact" in aware_agents

    def test_preserve_existing_parameters(self, sample_state):
        """Should preserve existing agent parameters."""
        router = RouterNode()

        dispatch_plan = [
            {
                "agent_name": "causal_impact",
                "priority": "critical",
                "parameters": {"interpretation_depth": "deep", "custom": "value"},
                "timeout_ms": 30000,
                "fallback_agent": "explainer",
            }
        ]

        enhanced, _ = router._enhance_with_discovery_data(dispatch_plan, sample_state)

        assert enhanced[0]["parameters"]["interpretation_depth"] == "deep"
        assert enhanced[0]["parameters"]["custom"] == "value"

    def test_non_discovery_aware_agent_unchanged(self, sample_state):
        """Should not modify non-discovery-aware agents."""
        router = RouterNode()

        dispatch_plan = [
            {
                "agent_name": "health_score",
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 5000,
                "fallback_agent": None,
            }
        ]

        enhanced, aware_agents = router._enhance_with_discovery_data(
            dispatch_plan, sample_state
        )

        assert len(enhanced) == 1
        assert "discovered_dag_adjacency" not in enhanced[0]["parameters"]
        assert aware_agents == []

    def test_multiple_discovery_aware_agents(self, sample_state):
        """Should enhance multiple discovery-aware agents."""
        router = RouterNode()

        dispatch_plan = [
            {
                "agent_name": "causal_impact",
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 30000,
                "fallback_agent": None,
            },
            {
                "agent_name": "heterogeneous_optimizer",
                "priority": "high",
                "parameters": {},
                "timeout_ms": 25000,
                "fallback_agent": None,
            },
        ]

        enhanced, aware_agents = router._enhance_with_discovery_data(
            dispatch_plan, sample_state
        )

        assert len(enhanced) == 2
        assert set(aware_agents) == {"causal_impact", "heterogeneous_optimizer"}
        assert "discovered_dag_adjacency" in enhanced[0]["parameters"]
        assert "discovered_dag_adjacency" in enhanced[1]["parameters"]

    def test_no_dag_data_only_config(self):
        """Should add discovery_config even without DAG data."""
        router = RouterNode()

        state = {
            "enable_discovery": True,
            "propagate_discovered_dag": True,
            "discovery_config": {"algorithms": ["ges"]},
        }

        dispatch_plan = [
            {
                "agent_name": "gap_analyzer",
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 20000,
                "fallback_agent": None,
            }
        ]

        enhanced, aware_agents = router._enhance_with_discovery_data(
            dispatch_plan, state
        )

        assert "discovery_config" in enhanced[0]["parameters"]
        # No DAG data, so not in aware_agents
        assert aware_agents == []

    def test_discovery_aware_agents_list(self):
        """Should correctly identify all discovery-aware agents."""
        router = RouterNode()

        expected_agents = [
            "causal_impact",
            "gap_analyzer",
            "heterogeneous_optimizer",
            "experiment_designer",
        ]

        assert router.DISCOVERY_AWARE_AGENTS == expected_agents


class TestDiscoveryRoutingIntegration:
    """Integration tests for discovery routing in execute method."""

    @pytest.fixture
    def sample_dag_adjacency(self):
        """Sample DAG adjacency matrix."""
        return [[0, 1, 0], [0, 0, 1], [0, 0, 0]]

    @pytest.fixture
    def sample_dag_nodes(self):
        """Sample DAG nodes."""
        return ["treatment", "segment", "outcome"]

    @pytest.mark.asyncio
    async def test_discovery_routing_applied_for_causal_effect(
        self, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should apply discovery routing for causal_effect intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "causal_effect",
                "confidence": 0.95,
                "secondary_intents": [],
                "requires_multi_agent": False,
            },
            "enable_discovery": True,
            "propagate_discovered_dag": True,
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
            "discovery_gate_confidence": 0.85,
        }

        result = await router.execute(state)

        assert result["discovery_routing_applied"] is True
        assert result["discovery_aware_agents"] == ["causal_impact"]
        assert "discovered_dag_adjacency" in result["dispatch_plan"][0]["parameters"]

    @pytest.mark.asyncio
    async def test_discovery_routing_not_applied_for_non_aware_agent(
        self, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should not apply discovery routing for non-discovery-aware agents."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "system_health",
                "confidence": 0.95,
                "secondary_intents": [],
                "requires_multi_agent": False,
            },
            "enable_discovery": True,
            "propagate_discovered_dag": True,
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
        }

        result = await router.execute(state)

        assert result["discovery_routing_applied"] is False
        assert result["discovery_aware_agents"] is None

    @pytest.mark.asyncio
    async def test_discovery_routing_skipped_on_reject(
        self, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should skip discovery routing when gate is rejected."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "causal_effect",
                "confidence": 0.95,
                "secondary_intents": [],
                "requires_multi_agent": False,
            },
            "enable_discovery": True,
            "propagate_discovered_dag": True,
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "reject",
        }

        result = await router.execute(state)

        assert result["discovery_routing_applied"] is False
        assert "discovered_dag_adjacency" not in result["dispatch_plan"][0]["parameters"]

    @pytest.mark.asyncio
    async def test_multi_agent_discovery_routing(
        self, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should apply discovery routing to multiple discovery-aware agents."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "causal_effect",
                "confidence": 0.90,
                "secondary_intents": ["segment_analysis"],
                "requires_multi_agent": True,
            },
            "enable_discovery": True,
            "propagate_discovered_dag": True,
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
        }

        result = await router.execute(state)

        assert result["discovery_routing_applied"] is True
        # Both causal_impact and heterogeneous_optimizer are discovery-aware
        assert set(result["discovery_aware_agents"]) == {
            "causal_impact",
            "heterogeneous_optimizer",
        }

    @pytest.mark.asyncio
    async def test_discovery_routing_with_no_dag_data(self):
        """Should handle discovery routing without DAG data."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "segment_analysis",
                "confidence": 0.88,
                "secondary_intents": [],
                "requires_multi_agent": False,
            },
            "enable_discovery": True,
            "discovery_config": {"algorithms": ["ges"]},
        }

        result = await router.execute(state)

        # Discovery routing applied but no agents received DAG
        assert result["discovery_routing_applied"] is False
        assert "discovery_config" in result["dispatch_plan"][0]["parameters"]

    @pytest.mark.asyncio
    async def test_experiment_designer_receives_dag(
        self, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should pass DAG data to experiment_designer agent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "experiment_design",
                "confidence": 0.90,
                "secondary_intents": [],
                "requires_multi_agent": False,
            },
            "enable_discovery": True,
            "propagate_discovered_dag": True,
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {"treatment<->segment": "BIDIRECTED"},
            "discovery_gate_decision": "accept",
            "discovery_gate_confidence": 0.80,
        }

        result = await router.execute(state)

        assert result["discovery_routing_applied"] is True
        assert "experiment_designer" in result["discovery_aware_agents"]
        params = result["dispatch_plan"][0]["parameters"]
        assert params["discovered_dag_adjacency"] == sample_dag_adjacency
        assert params["discovered_dag_nodes"] == sample_dag_nodes
        assert params["discovered_dag_edge_types"] == {"treatment<->segment": "BIDIRECTED"}
        assert params["discovery_gate_decision"] == "accept"
        assert params["discovery_gate_confidence"] == 0.80
