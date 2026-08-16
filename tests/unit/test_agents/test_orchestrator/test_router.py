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
        # 300s as of #1419: critical-gates-first refutation on the 5k stratified
        # subsample measures ~223s on the live 37k conversion frame (#1351's
        # 120s predates the refutation suite becoming enforceable).
        assert result["dispatch_plan"][0]["timeout_ms"] == 300000
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
        # #1635: raised 150s -> 240s. The old 150s sat BELOW the agent's own
        # serial internal step ceilings (design_reasoning 120s + validity_audit
        # 90s = 210s declared in code), so the dispatch timeout could fire
        # mid-graph and discard a design the agent was about to return. See
        # TestDispatchBudgetsAreJustified for the composition and the ceiling.
        assert result["dispatch_plan"][0]["timeout_ms"] == 240000
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
        # #1634: raised 5000ms -> 20000ms. The old value was the only budget in
        # INTENT_TO_AGENTS with no justifying comment and was never measured;
        # the agent's measured cold worst case is 3673ms (n=5, faithful
        # chat-path wiring), which 5000ms cleared by only 1.36x on an idle box.
        assert result["dispatch_plan"][0]["timeout_ms"] == 20000

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
    async def test_route_cohort_definition(self):
        """Test routing for cohort_definition intent."""
        router = RouterNode()

        state = {
            "intent": {
                "primary_intent": "cohort_definition",
                "confidence": 0.92,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }

        result = await router.execute(state)

        # cohort_definition chat queries route to cohort_profiler (real per-segment
        # counts), NOT cohort_constructor (ML-pipeline agent that can't run from a
        # chat payload — it dead-ended). See router.py cohort_definition dispatch.
        assert result["dispatch_plan"][0]["agent_name"] == "cohort_profiler"
        assert result["dispatch_plan"][0]["priority"] == "critical"
        assert result["dispatch_plan"][0]["timeout_ms"] == 30000  # ≤8 DB KPI calls/brand
        assert result["dispatch_plan"][0]["parameters"] == {}
        # No fallback: profiling either has real data or fails closed honestly; an
        # explainer fallback would only re-fail with nothing to explain.
        assert result["dispatch_plan"][0]["fallback_agent"] is None
        assert result["current_phase"] == "dispatching"

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
        assert dispatch["timeout_ms"] == 300000  # #1419 measured refutation budget
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
            "cohort_definition",  # Tier 0: Patient cohort construction
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

        # Per-intent ceilings. Most agents <= 120s; tool_composer's
        # documented SLA is 180s (4-phase decompose/plan/execute/synthesize);
        # heterogeneous_optimizer (segment_analysis) is 420s as of 2026-06-11 —
        # the full 37,378-row gold conversion frame's CausalForestDML +
        # per-segment effect_interval + hierarchical uplift MEASURED 269.7s
        # serialized (gate 11, clean substrate); 420s = measured + ~55% headroom.
        # experiment_design is 240s as of 2026-08-16 (#1635) — the previous 150s
        # (#1351) sat BELOW the agent's own serial internal step ceilings
        # (design_reasoning wait_for 120s + a 60s fallback LLM; validity_audit
        # wait_for 90s, which degrades rather than fails), so the dispatch
        # timeout could fire mid-graph and discard a design the agent was about
        # to return. 240s covers the realistic degraded path (~235s) and stays
        # 60s under the host-nginx proxy_read_timeout 300s ceiling.
        # causal_effect is 300s as of 2026-07-31 (#1419) — critical-gates-first
        # refutation on the 5k stratified subsample measures ~223s on the live
        # 37,371-row conversion frame (recon ~12s + e-value ~2s + 30 placebo
        # sims x ~2.13s + 20 rcc sims x ~2.59s); 300s = measured + ~35% headroom
        # and equals the host-nginx proxy_read_timeout ceiling.
        max_timeout_ms = {
            "multi_faceted": 180_000,
            "segment_analysis": 420_000,
            "experiment_design": 240_000,
            "causal_effect": 300_000,
        }
        for intent, dispatches in router.INTENT_TO_AGENTS.items():
            for dispatch in dispatches:
                timeout = dispatch["timeout_ms"]
                cap = max_timeout_ms.get(intent, 120_000)
                assert timeout > 0, f"Invalid timeout for {intent}"
                assert timeout <= cap, f"Timeout {timeout}ms for {intent} exceeds cap {cap}ms"

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
            assert len(priorities) == len(set(priorities)) or len(priorities) <= 1, (
                f"Duplicate priorities in pattern {pattern}"
            )
            assert all(p in valid_priorities for p in priorities), (
                f"Invalid priority in pattern {pattern}"
            )


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

        enhanced, aware_agents = router._enhance_with_discovery_data(dispatch_plan, sample_state)

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

        enhanced, aware_agents = router._enhance_with_discovery_data(dispatch_plan, sample_state)

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

        enhanced, aware_agents = router._enhance_with_discovery_data(dispatch_plan, sample_state)

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

        enhanced, aware_agents = router._enhance_with_discovery_data(dispatch_plan, state)

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
    async def test_multi_agent_discovery_routing(self, sample_dag_adjacency, sample_dag_nodes):
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
    async def test_experiment_designer_receives_dag(self, sample_dag_adjacency, sample_dag_nodes):
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


class TestDispatchBudgetsAreJustified:
    """#1634 / #1635: every per-agent dispatch budget must be defensible against
    the agent's MEASURED runtime, and a budget the agent cannot meet must not
    turn a served capability into a user-facing dead end.

    These tests deliberately pin the *reasoning*, not just the constant: each
    asserts the budget against the measurement (or the code-derived internal
    ceiling) that justifies it, so a future edit that reverts the number to an
    unmeasured guess fails here rather than silently in a chat turn.
    """

    # docker/nginx/host-nginx.conf:119 -- ``proxy_read_timeout 300s``. A dispatch
    # emits NO bytes while it runs (measured: turn 3.4's first_progress_ms equals
    # its total_ms), so this bounds the SILENT window end-to-end and is a hard
    # ceiling on any single-agent budget that wants to reach the user.
    HOST_NGINX_SILENT_WINDOW_CEILING_MS = 300_000

    # src/agents/experiment_designer/nodes/design_reasoning.py -- the primary
    # reasoning LLM is wrapped in ``asyncio.wait_for(..., timeout=120)``.
    DESIGN_REASONING_PRIMARY_CEILING_MS = 120_000
    # src/agents/experiment_designer/nodes/validity_audit.py -- the audit LLM is
    # wrapped in ``asyncio.wait_for(..., timeout=90)`` and DEGRADES on expiry
    # (validity_audit_status="timed_out") rather than failing the run.
    VALIDITY_AUDIT_CEILING_MS = 90_000

    @staticmethod
    def _dispatch_for(plan, agent_name):
        for dispatch in plan:
            if dispatch["agent_name"] == agent_name:
                return dispatch
        raise AssertionError(f"{agent_name} not in dispatch plan")

    @staticmethod
    async def _plan(primary_intent, confidence=0.95):
        router = RouterNode()
        state = {
            "intent": {
                "primary_intent": primary_intent,
                "confidence": confidence,
                "secondary_intents": [],
                "requires_multi_agent": False,
            }
        }
        result = await router.execute(state)
        return result["dispatch_plan"]

    @pytest.mark.asyncio
    async def test_system_health_budget_covers_measured_cold_runtime(self):
        """#1634: the 5000ms budget was never measured -- it shipped with the
        initial platform commit and is the only entry in INTENT_TO_AGENTS with
        no justifying comment.

        MEASURED 2026-08-16 on the faithful chat-path wiring
        (``factory._health_score_kwargs()`` -- the same four real backends
        ``create_agent_registry`` injects), full graph, all four dimensions
        reported measured=True, grade A, zero errors:
          cold (fresh process, n=5): 2311 / 2342 / 2922 / 3000 / 3673 ms
          warm (same process):        107 - 594 ms
        A chat dispatch hits the COLD path after a worker respawn, so the worst
        cold run (3673 ms) is the number the budget must clear. 5000 ms leaves
        only 1.36x over it on an idle box with no gunicorn contention -- which is
        why the live turn timed out at 5000 ms while the agent served the same
        ask in 14.7 s wall under the previous route.
        """
        plan = await self._plan("system_health", confidence=0.96)
        dispatch = self._dispatch_for(plan, "health_score")

        measured_cold_worst_ms = 3673
        assert dispatch["timeout_ms"] >= 4 * measured_cold_worst_ms, (
            "system_health budget must clear the measured cold worst case with "
            "real headroom for container contention"
        )
        assert dispatch["timeout_ms"] == 20_000

    @pytest.mark.asyncio
    async def test_experiment_design_budget_covers_agent_internal_ceilings(self):
        """#1635: 150000 ms sat BELOW the agent's own internal step ceilings, so
        the dispatch timeout could fire mid-graph and discard a design the agent
        was about to return.

        The experiment_designer graph is sequential (context_loader ->
        design_reasoning -> power_analysis -> validity_audit -> template_generator),
        and its two LLM steps declare 120 s + 90 s of internal budget on that
        serial path -- 210 s, i.e. 1.4x the old 150 s dispatch budget. The budget
        must cover the composition it wraps.
        """
        plan = await self._plan("experiment_design", confidence=0.90)
        dispatch = self._dispatch_for(plan, "experiment_designer")

        serial_internal_ceiling_ms = (
            self.DESIGN_REASONING_PRIMARY_CEILING_MS + self.VALIDITY_AUDIT_CEILING_MS
        )
        assert dispatch["timeout_ms"] >= serial_internal_ceiling_ms, (
            "dispatch budget must cover the agent's own serial internal step "
            "ceilings, or it cuts off runs the agent would have completed"
        )
        assert dispatch["timeout_ms"] == 240_000

    @pytest.mark.asyncio
    async def test_experiment_design_budget_stays_under_host_nginx_ceiling(self):
        """#1635: the budget may not exceed the host-nginx silent-window ceiling.

        A budget above 300 s cannot reach the user at all -- nginx closes the
        connection first, so the agent would be doing work no one receives.
        """
        plan = await self._plan("experiment_design", confidence=0.90)
        dispatch = self._dispatch_for(plan, "experiment_designer")

        assert dispatch["timeout_ms"] < self.HOST_NGINX_SILENT_WINDOW_CEILING_MS

    @pytest.mark.asyncio
    async def test_experiment_design_keeps_no_failing_closed_fallback(self):
        """#1635: ``fallback_agent`` stays None -- deliberately, with a disproof.

        ``explainer`` is the platform's universal fallback, but
        ``_resolve_explainer_input`` (dispatcher.py) fails CLOSED at step (4)
        when there are no upstream results to explain: on an experiment_designer
        timeout it is the ONLY dispatched agent, so an explainer fallback would
        re-fail with nothing to explain and reproduce the same dead end -- the
        precedent already documented on the ``cohort_definition`` entry.

        The methodological fallback is restored by the BUDGET instead: the
        agent's own graceful-degradation path (validity_audit self-caps at 90 s
        and proceeds to template_generator) yields a real design, which the old
        150 s dispatch budget was cutting off.
        """
        plan = await self._plan("experiment_design", confidence=0.90)
        dispatch = self._dispatch_for(plan, "experiment_designer")

        assert dispatch["fallback_agent"] is None

    @pytest.mark.asyncio
    async def test_no_dispatch_budget_is_an_unjustified_round_guess(self):
        """#1634 guard: the two budgets this pair repaired must not silently
        revert to their previous unmeasured values."""
        health = self._dispatch_for(await self._plan("system_health", 0.96), "health_score")
        design = self._dispatch_for(
            await self._plan("experiment_design", 0.90), "experiment_designer"
        )

        assert health["timeout_ms"] != 5_000, "reverted to the unmeasured 5000ms guess"
        assert design["timeout_ms"] != 150_000, "reverted to the sub-composition 150s budget"
