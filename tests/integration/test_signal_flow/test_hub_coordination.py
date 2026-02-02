"""
E2I Signal Flow Integration Tests - Batch 3: Hub Coordination

Tests orchestrator and feedback_learner hub coordination for DSPy optimization.

Hub agents:
- orchestrator (Hub role)
- feedback_learner (Hybrid role - central optimization hub)

Run: pytest tests/integration/test_signal_flow/test_hub_coordination.py -v
"""

from datetime import datetime, timezone

# =============================================================================
# FEEDBACK LEARNER HUB TESTS
# =============================================================================


class TestFeedbackLearnerTrainingSignal:
    """Test FeedbackLearnerTrainingSignal for hub operations."""

    def test_import_training_signal(self):
        """Verify FeedbackLearnerTrainingSignal can be imported."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        assert FeedbackLearnerTrainingSignal is not None

    def test_create_training_signal(self):
        """Create a valid training signal for learning cycle."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="batch_001",
            feedback_count=50,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T23:59:59Z",
            focus_agents=["causal_impact", "gap_analyzer"],
            patterns_detected=5,
            recommendations_generated=3,
            updates_applied=2,
        )

        assert signal.batch_id == "batch_001"
        assert signal.feedback_count == 50
        assert len(signal.focus_agents) == 2

    def test_compute_reward(self):
        """Test reward computation for learning cycle."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="batch_002",
            feedback_count=100,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T23:59:59Z",
            patterns_detected=10,
            recommendations_generated=5,
            updates_applied=3,
            pattern_accuracy=0.8,
            recommendation_actionability=0.7,
            update_effectiveness=0.85,
            total_latency_ms=25000,
        )

        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0
        assert reward >= 0.5, "High quality learning cycle should have good reward"

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="batch_003",
            feedback_count=75,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T12:00:00Z",
        )

        signal_dict = signal.to_dict()

        assert signal_dict["source_agent"] == "feedback_learner"
        assert "input_context" in signal_dict
        assert "output" in signal_dict
        assert "quality_metrics" in signal_dict
        assert "latency" in signal_dict
        assert "reward" in signal_dict


class TestFeedbackLearnerOptimizer:
    """Test FeedbackLearnerOptimizer for MIPROv2 optimization."""

    def test_import_optimizer(self):
        """Verify FeedbackLearnerOptimizer can be imported."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerOptimizer,
        )

        assert FeedbackLearnerOptimizer is not None

    def test_create_optimizer(self):
        """Create optimizer instance."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerOptimizer,
        )

        optimizer = FeedbackLearnerOptimizer()

        assert optimizer is not None
        assert hasattr(optimizer, "pattern_metric")
        assert hasattr(optimizer, "recommendation_metric")

    def test_pattern_metric(self):
        """Test pattern detection metric."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerOptimizer,
        )

        optimizer = FeedbackLearnerOptimizer()

        # Mock prediction with patterns
        class MockPrediction:
            patterns = [
                {
                    "type": "recurring_issue",
                    "severity": "high",
                    "affected_agents": ["causal_impact"],
                },
                {
                    "type": "performance_drop",
                    "severity": "medium",
                    "affected_agents": ["gap_analyzer"],
                },
            ]
            confidence = 0.75
            root_causes = ["data_quality", "prompt_drift"]

        score = optimizer.pattern_metric(None, MockPrediction())

        assert 0.0 <= score <= 1.0

    def test_recommendation_metric(self):
        """Test recommendation generation metric."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerOptimizer,
        )

        optimizer = FeedbackLearnerOptimizer()

        # Mock prediction with recommendations
        class MockPrediction:
            recommendations = [
                {"category": "prompt_update", "expected_impact": "high"},
                {"category": "config_change", "expected_impact": "medium"},
            ]
            implementation_order = ["prompt_update", "config_change"]
            risk_assessment = "Low risk - changes can be easily rolled back if issues arise."

        score = optimizer.recommendation_metric(None, MockPrediction())

        assert 0.0 <= score <= 1.0


class TestAgentTrainingSignal:
    """Test generic AgentTrainingSignal TypedDict."""

    def test_import_agent_training_signal(self):
        """Verify AgentTrainingSignal can be imported."""
        from src.agents.feedback_learner.dspy_integration import (
            AgentTrainingSignal,
        )

        assert AgentTrainingSignal is not None

    def test_create_agent_training_signal(self):
        """Create a valid AgentTrainingSignal dict."""
        from src.agents.feedback_learner.dspy_integration import (
            AgentTrainingSignal,
        )

        signal: AgentTrainingSignal = {
            "signal_id": "test_001",
            "source_agent": "causal_impact",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_context": {"query": "test query"},
            "output": {"result": "test result"},
            "user_feedback": None,
            "outcome_observed": None,
            "latency_ms": 1500.0,
            "token_count": 500,
            "cognitive_phase": "agent",
        }

        assert signal["signal_id"] == "test_001"
        assert signal["source_agent"] == "causal_impact"


class TestMemoryContribution:
    """Test memory contribution from training signals."""

    def test_import_memory_contribution(self):
        """Verify create_memory_contribution can be imported."""
        from src.agents.feedback_learner.dspy_integration import (
            create_memory_contribution,
        )

        assert create_memory_contribution is not None

    def test_create_semantic_memory_contribution(self):
        """Create semantic memory contribution."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="batch_test",
            feedback_count=50,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T12:00:00Z",
            patterns_detected=5,
            recommendations_generated=3,
            focus_agents=["causal_impact"],
        )

        contribution = create_memory_contribution(signal, "semantic")

        assert contribution["source_agent"] == "feedback_learner"
        assert contribution["memory_type"] == "semantic"
        assert "entities" in contribution
        assert "relationships" in contribution

    def test_create_episodic_memory_contribution(self):
        """Create episodic memory contribution."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="batch_test",
            feedback_count=50,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T12:00:00Z",
        )

        contribution = create_memory_contribution(signal, "episodic")

        assert contribution["memory_type"] == "episodic"
        assert "content" in contribution

    def test_create_procedural_memory_contribution(self):
        """Create procedural memory contribution for high-quality signals."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        # High quality signal
        signal = FeedbackLearnerTrainingSignal(
            batch_id="batch_test",
            feedback_count=100,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T23:59:59Z",
            patterns_detected=10,
            pattern_accuracy=0.9,
            recommendation_actionability=0.8,
            update_effectiveness=0.85,
        )

        contribution = create_memory_contribution(signal, "procedural")

        assert contribution["memory_type"] == "procedural"
        # Should have procedure if reward >= 0.7
        if signal.compute_reward() >= 0.7:
            assert "procedure" in contribution


# =============================================================================
# ORCHESTRATOR HUB TESTS
# =============================================================================


class TestOrchestratorDSPyIntegration:
    """Test orchestrator DSPy hub integration."""

    def test_import_orchestrator_dspy(self):
        """Verify orchestrator DSPy integration can be imported."""
        from src.agents.orchestrator.dspy_integration import (
            OrchestratorDSPyHub,
        )

        assert OrchestratorDSPyHub is not None

    def test_create_orchestrator_hub(self):
        """Create orchestrator DSPy hub."""
        from src.agents.orchestrator.dspy_integration import (
            OrchestratorDSPyHub,
        )

        hub = OrchestratorDSPyHub()

        assert hub.dspy_type == "hub"

    def test_hub_has_request_optimization(self):
        """Hub should have request_optimization method."""
        from src.agents.orchestrator.dspy_integration import (
            OrchestratorDSPyHub,
        )

        hub = OrchestratorDSPyHub()

        assert hasattr(hub, "request_optimization")

    def test_hub_has_pending_requests(self):
        """Hub should have get_pending_requests method."""
        from src.agents.orchestrator.dspy_integration import (
            OrchestratorDSPyHub,
        )

        hub = OrchestratorDSPyHub()

        assert hasattr(hub, "get_pending_requests")


class TestOrchestratorTrainingSignal:
    """Test orchestrator routing training signal."""

    def test_import_routing_signal(self):
        """Verify RoutingTrainingSignal can be imported."""
        from src.agents.orchestrator.dspy_integration import (
            RoutingTrainingSignal,
        )

        assert RoutingTrainingSignal is not None

    def test_create_routing_signal(self):
        """Create a routing training signal."""
        from src.agents.orchestrator.dspy_integration import (
            RoutingTrainingSignal,
        )

        signal = RoutingTrainingSignal(
            signal_id="route_001",
            session_id="session_test",
            query="What is the causal impact of marketing?",
            intent="causal_analysis",
            agents_selected=["causal_impact"],
        )

        assert signal.signal_id == "route_001"
        assert "causal_impact" in signal.agents_selected

    def test_routing_signal_compute_reward(self):
        """Test reward computation for routing."""
        from src.agents.orchestrator.dspy_integration import (
            RoutingTrainingSignal,
        )

        signal = RoutingTrainingSignal(
            signal_id="route_002",
            session_id="session_test",
            query="Test query",
            query_pattern="gap_analysis",
            intent="find_opportunities",
            agents_selected=["gap_analyzer"],
            routing_confidence=0.9,
            agents_succeeded=1,
            agents_failed=0,
            total_latency_ms=2000,
            was_rerouted=False,
            user_satisfaction=4.5,
        )

        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0


# =============================================================================
# SIGNAL FLOW COORDINATION TESTS
# =============================================================================


class TestSignalFlowCoordination:
    """Test coordination between hub and sender agents."""

    def test_signals_can_flow_to_feedback_learner(self):
        """Signals from senders can be processed by feedback_learner."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        # Simulate sender signals
        sender_signals = [
            CausalAnalysisTrainingSignal(
                signal_id=f"ci_{i}",
                session_id=f"session_{i}",
            ).to_dict()
            for i in range(10)
        ]

        # Simulate feedback_learner processing
        learner_signal = FeedbackLearnerTrainingSignal(
            batch_id="batch_001",
            feedback_count=len(sender_signals),
            time_range_start=sender_signals[0]["timestamp"],
            time_range_end=sender_signals[-1]["timestamp"],
            focus_agents=list({s["source_agent"] for s in sender_signals}),
            patterns_detected=2,
        )

        assert learner_signal.feedback_count == 10
        assert "causal_impact" in learner_signal.focus_agents

    def test_hybrid_agents_both_send_and_receive(self):
        """Hybrid agents can both generate signals and receive prompts."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        # feedback_learner generates its own training signal
        signal = FeedbackLearnerTrainingSignal(
            batch_id="hybrid_test",
            feedback_count=50,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T12:00:00Z",
        )

        signal_dict = signal.to_dict()

        # Signal has sender characteristics
        assert signal_dict["source_agent"] == "feedback_learner"
        assert "reward" in signal_dict

        # Also has optimizer for receiving optimized prompts
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerOptimizer,
        )

        optimizer = FeedbackLearnerOptimizer()
        assert hasattr(optimizer, "optimize")


class TestOptimizationThresholds:
    """Test optimization trigger thresholds per SignalFlowContract."""

    def test_min_signals_threshold(self):
        """Test minimum signals for optimization."""
        # Per SignalFlowContract: min_signals_for_optimization = 100
        MIN_SIGNALS = 100

        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        # Below threshold
        small_batch = [
            CausalAnalysisTrainingSignal(signal_id=f"ci_{i}").to_dict() for i in range(50)
        ]

        # At threshold
        full_batch = [
            CausalAnalysisTrainingSignal(signal_id=f"ci_{i}").to_dict() for i in range(100)
        ]

        should_optimize_small = len(small_batch) >= MIN_SIGNALS
        should_optimize_full = len(full_batch) >= MIN_SIGNALS

        assert not should_optimize_small
        assert should_optimize_full

    def test_quality_threshold(self):
        """Test signal quality threshold for optimization."""
        # Per SignalFlowContract: min_signal_quality = 0.6
        MIN_QUALITY = 0.6

        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        # Create signals with varying quality
        signals = [
            CausalAnalysisTrainingSignal(
                signal_id=f"ci_{i}",
                refutation_tests_passed=i,
                refutation_tests_failed=1,
            ).to_dict()
            for i in range(5)
        ]

        # Filter by quality
        quality_signals = [s for s in signals if s["reward"] >= MIN_QUALITY]

        # Not all signals should pass quality threshold
        assert len(quality_signals) < len(signals)
