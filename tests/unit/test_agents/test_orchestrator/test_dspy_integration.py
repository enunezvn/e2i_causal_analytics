"""
Tests for Orchestrator Agent DSPy Integration.

Tests the DSPy Hub role implementation including:
- Training signal dataclass
- Reward computation
- Signal collector
- Hub coordination
- Singleton access patterns
"""

import pytest
from datetime import datetime, timezone

from src.agents.orchestrator.dspy_integration import (
    RoutingTrainingSignal,
    OrchestratorSignalCollector,
    OrchestratorDSPyHub,
    get_orchestrator_signal_collector,
    get_orchestrator_dspy_hub,
    reset_dspy_integration,
    DSPY_AVAILABLE,
)


class TestRoutingTrainingSignal:
    """Tests for RoutingTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with defaults."""
        signal = RoutingTrainingSignal()

        assert signal.signal_id == ""
        assert signal.session_id == ""
        assert signal.query == ""
        assert signal.query_pattern == ""
        assert signal.intent == ""
        assert signal.entities_extracted == []
        assert signal.agents_selected == []
        assert signal.routing_confidence == 0.0
        assert signal.routing_rationale == ""
        assert signal.agents_succeeded == 0
        assert signal.agents_failed == 0
        assert signal.was_rerouted is False
        assert signal.user_satisfaction is None
        assert signal.created_at is not None

    def test_custom_initialization(self):
        """Test signal with custom values."""
        signal = RoutingTrainingSignal(
            session_id="orch-session-123",
            query="What is the causal impact of rep visits on TRx?",
            query_pattern="CAUSAL",
            intent="CAUSAL_ANALYSIS",
            entities_extracted=["rep_visits", "TRx", "Remibrutinib"],
            agents_selected=["causal_impact", "gap_analyzer"],
            routing_confidence=0.92,
        )

        assert signal.session_id == "orch-session-123"
        assert signal.query_pattern == "CAUSAL"
        assert len(signal.agents_selected) == 2
        assert signal.routing_confidence == 0.92

    def test_compute_reward_minimal(self):
        """Test reward with minimal data."""
        signal = RoutingTrainingSignal()
        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0
        # Minimal gets: 0 routing accuracy (no agents), 0.25 efficiency (no latency),
        # 0.20 no-rerouting bonus, 0.10 partial satisfaction = 0.55
        assert reward < 0.6  # Minimal data should yield moderate-low reward

    def test_compute_reward_high_quality(self):
        """Test reward with high-quality routing."""
        signal = RoutingTrainingSignal(
            agents_selected=["causal_impact", "gap_analyzer"],
            agents_succeeded=2,
            agents_failed=0,
            total_latency_ms=3000,  # Under 5s target
            was_rerouted=False,
            user_satisfaction=5.0,
        )
        reward = signal.compute_reward()

        assert reward > 0.8  # High-quality should score well
        assert reward <= 1.0

    def test_compute_reward_routing_accuracy(self):
        """Test that routing accuracy affects reward."""
        # All agents succeeded
        signal_accurate = RoutingTrainingSignal(
            agents_selected=["causal_impact", "gap_analyzer"],
            agents_succeeded=2,
            agents_failed=0,
        )

        # Some agents failed
        signal_inaccurate = RoutingTrainingSignal(
            agents_selected=["causal_impact", "gap_analyzer"],
            agents_succeeded=1,
            agents_failed=1,
        )

        assert signal_accurate.compute_reward() > signal_inaccurate.compute_reward()

    def test_compute_reward_efficiency_impact(self):
        """Test that latency affects reward."""
        # Fast routing
        signal_fast = RoutingTrainingSignal(
            agents_selected=["causal_impact"],
            agents_succeeded=1,
            total_latency_ms=2000,
        )

        # Slow routing
        signal_slow = RoutingTrainingSignal(
            agents_selected=["causal_impact"],
            agents_succeeded=1,
            total_latency_ms=20000,
        )

        assert signal_fast.compute_reward() > signal_slow.compute_reward()

    def test_compute_reward_no_rerouting_bonus(self):
        """Test that no rerouting gets bonus."""
        # No rerouting needed
        signal_direct = RoutingTrainingSignal(
            agents_selected=["causal_impact"],
            agents_succeeded=1,
            was_rerouted=False,
        )

        # Had to reroute
        signal_rerouted = RoutingTrainingSignal(
            agents_selected=["causal_impact"],
            agents_succeeded=1,
            was_rerouted=True,
        )

        assert signal_direct.compute_reward() > signal_rerouted.compute_reward()

    def test_compute_reward_satisfaction_impact(self):
        """Test that user satisfaction affects reward."""
        signal_satisfied = RoutingTrainingSignal(
            agents_selected=["causal_impact"],
            agents_succeeded=1,
            user_satisfaction=5.0,
        )

        signal_unsatisfied = RoutingTrainingSignal(
            agents_selected=["causal_impact"],
            agents_succeeded=1,
            user_satisfaction=1.0,
        )

        assert signal_satisfied.compute_reward() > signal_unsatisfied.compute_reward()

    def test_to_dict_structure(self):
        """Test dictionary serialization structure."""
        signal = RoutingTrainingSignal(
            session_id="sess-123",
            intent="CAUSAL_ANALYSIS",
            agents_selected=["causal_impact"],
            routing_confidence=0.85,
        )

        result = signal.to_dict()

        assert result["source_agent"] == "orchestrator"
        assert result["dspy_type"] == "hub"
        assert "input_context" in result
        assert "routing_decision" in result
        assert "outcome" in result
        assert "quality_metrics" in result
        assert "reward" in result

    def test_to_dict_query_truncation(self):
        """Test long query truncation."""
        signal = RoutingTrainingSignal(query="A" * 600)
        result = signal.to_dict()

        assert len(result["input_context"]["query"]) <= 500

    def test_to_dict_rationale_truncation(self):
        """Test long rationale truncation."""
        signal = RoutingTrainingSignal(routing_rationale="B" * 300)
        result = signal.to_dict()

        assert len(result["routing_decision"]["routing_rationale"]) <= 200


class TestOrchestratorSignalCollector:
    """Tests for OrchestratorSignalCollector."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = OrchestratorSignalCollector()

        assert collector._signals_buffer == []
        assert collector._buffer_limit == 100

    def test_collect_routing_signal(self):
        """Test signal collection initiation."""
        collector = OrchestratorSignalCollector()

        signal = collector.collect_routing_signal(
            session_id="orch-sess-123",
            query="Analyze TRx trends",
            query_pattern="DESCRIPTIVE",
            intent="TREND_ANALYSIS",
            entities=["TRx", "Remibrutinib"],
            agents_selected=["gap_analyzer"],
            routing_confidence=0.88,
            routing_rationale="Query matches gap analysis pattern",
        )

        assert isinstance(signal, RoutingTrainingSignal)
        assert signal.session_id == "orch-sess-123"
        assert signal.query_pattern == "DESCRIPTIVE"
        assert signal.agents_selected == ["gap_analyzer"]

    def test_update_with_outcome_adds_to_buffer(self):
        """Test outcome update adds signal to buffer."""
        collector = OrchestratorSignalCollector()
        signal = RoutingTrainingSignal(
            session_id="test-session",
            agents_selected=["causal_impact"],
        )

        assert len(collector._signals_buffer) == 0

        collector.update_with_outcome(
            signal,
            agents_succeeded=1,
            agents_failed=0,
            total_latency_ms=2500,
            was_rerouted=False,
        )

        assert len(collector._signals_buffer) == 1
        assert signal.agents_succeeded == 1
        assert signal.total_latency_ms == 2500

    def test_buffer_limit_enforcement(self):
        """Test buffer respects limit."""
        collector = OrchestratorSignalCollector()
        collector._buffer_limit = 5

        for i in range(7):
            signal = RoutingTrainingSignal(
                session_id=f"orch-{i}",
                agents_selected=["causal_impact"],
            )
            collector.update_with_outcome(
                signal,
                agents_succeeded=1,
                agents_failed=0,
                total_latency_ms=2000,
            )

        assert len(collector._signals_buffer) == 5
        assert collector._signals_buffer[0].session_id == "orch-2"

    def test_update_with_feedback(self):
        """Test delayed feedback update."""
        collector = OrchestratorSignalCollector()
        signal = RoutingTrainingSignal()

        updated = collector.update_with_feedback(
            signal,
            user_satisfaction=4.5,
            answer_quality=0.9,
        )

        assert updated.user_satisfaction == 4.5
        assert updated.answer_quality == 0.9

    def test_get_signals_for_training(self):
        """Test signal retrieval for training."""
        collector = OrchestratorSignalCollector()

        for i in range(3):
            signal = RoutingTrainingSignal(
                session_id=f"orch-{i}",
                agents_selected=["causal_impact"],
            )
            collector.update_with_outcome(
                signal,
                agents_succeeded=1,
                agents_failed=0,
                total_latency_ms=2000,
            )

        all_signals = collector.get_signals_for_training(min_reward=0.0)
        assert len(all_signals) == 3

    def test_clear_buffer(self):
        """Test buffer clearing."""
        collector = OrchestratorSignalCollector()

        signal = RoutingTrainingSignal(agents_selected=["causal_impact"])
        collector.update_with_outcome(
            signal,
            agents_succeeded=1,
            agents_failed=0,
            total_latency_ms=2000,
        )

        assert len(collector._signals_buffer) == 1

        collector.clear_buffer()

        assert len(collector._signals_buffer) == 0


class TestOrchestratorDSPyHub:
    """Tests for OrchestratorDSPyHub."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test hub initializes correctly."""
        hub = OrchestratorDSPyHub()

        assert hub.dspy_type == "hub"
        assert hub._pending_optimization_requests == []

    @pytest.mark.asyncio
    async def test_request_optimization(self):
        """Test optimization request."""
        hub = OrchestratorDSPyHub()

        request_id = await hub.request_optimization(
            agent_name="causal_impact",
            signature_name="CausalGraphSignature",
            training_signals=[{"signal": "test"}],
            priority="high",
        )

        assert request_id.startswith("opt_causal_impact_")
        assert len(hub._pending_optimization_requests) == 1
        assert hub._pending_optimization_requests[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_pending_requests(self):
        """Test pending request retrieval."""
        hub = OrchestratorDSPyHub()

        await hub.request_optimization(
            agent_name="gap_analyzer",
            signature_name="GapDetectionSignature",
            training_signals=[],
        )

        pending = hub.get_pending_requests()
        assert len(pending) == 1
        assert pending[0]["agent_name"] == "gap_analyzer"


class TestSingletonAccess:
    """Tests for singleton access patterns."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_get_signal_collector_creates_singleton(self):
        """Test signal collector singleton creation."""
        collector1 = get_orchestrator_signal_collector()
        collector2 = get_orchestrator_signal_collector()

        assert collector1 is collector2

    def test_get_dspy_hub_creates_singleton(self):
        """Test DSPy hub singleton creation."""
        hub1 = get_orchestrator_dspy_hub()
        hub2 = get_orchestrator_dspy_hub()

        assert hub1 is hub2

    def test_reset_dspy_integration(self):
        """Test singleton reset."""
        collector1 = get_orchestrator_signal_collector()
        hub1 = get_orchestrator_dspy_hub()

        reset_dspy_integration()

        collector2 = get_orchestrator_signal_collector()
        hub2 = get_orchestrator_dspy_hub()

        assert collector1 is not collector2
        assert hub1 is not hub2


class TestDSPySignatures:
    """Tests for DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE reflects actual availability."""
        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_agent_routing_signature(self):
        """Test AgentRoutingSignature exists."""
        from src.agents.orchestrator.dspy_integration import AgentRoutingSignature

        import dspy
        assert issubclass(AgentRoutingSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_intent_classification_signature(self):
        """Test IntentClassificationSignature exists."""
        from src.agents.orchestrator.dspy_integration import IntentClassificationSignature

        import dspy
        assert issubclass(IntentClassificationSignature, dspy.Signature)


class TestFullSignalLifecycle:
    """Integration tests for complete signal lifecycle."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_complete_signal_lifecycle(self):
        """Test full signal collection through all phases."""
        collector = get_orchestrator_signal_collector()

        # Phase 1: Route query
        signal = collector.collect_routing_signal(
            session_id="lifecycle-orch-test",
            query="What is the causal impact of rep visits on TRx?",
            query_pattern="CAUSAL",
            intent="CAUSAL_ANALYSIS",
            entities=["rep_visits", "TRx"],
            agents_selected=["causal_impact", "gap_analyzer"],
            routing_confidence=0.92,
            routing_rationale="Causal question maps to causal_impact agent",
        )

        # Phase 2: Execution outcome (adds to buffer)
        signal = collector.update_with_outcome(
            signal,
            agents_succeeded=2,
            agents_failed=0,
            total_latency_ms=3500,
            was_rerouted=False,
        )

        # Phase 3: Delayed feedback
        signal = collector.update_with_feedback(
            signal,
            user_satisfaction=4.5,
            answer_quality=0.88,
        )

        # Verify final state
        assert signal.session_id == "lifecycle-orch-test"
        assert signal.agents_succeeded == 2
        assert signal.was_rerouted is False
        assert signal.user_satisfaction == 4.5

        # Verify in buffer
        signals = collector.get_signals_for_training()
        assert len(signals) == 1

        # Verify reward is good
        reward = signal.compute_reward()
        assert reward > 0.7
