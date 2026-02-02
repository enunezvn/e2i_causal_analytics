"""
Tests for Heterogeneous Optimizer Agent DSPy Integration.

Tests the DSPy Sender role implementation including:
- Training signal dataclass
- Reward computation
- Signal collector
- Singleton access patterns

Note: This module is marked to run sequentially (not in parallel) because
the dspy import has race conditions during parallel pytest-xdist execution.
"""

import pytest

# Mark entire module to run on same worker - prevents import race conditions
pytestmark = pytest.mark.xdist_group(name="dspy_integration")

from src.agents.heterogeneous_optimizer.dspy_integration import (
    DSPY_AVAILABLE,
    HeterogeneousOptimizationTrainingSignal,
    HeterogeneousOptimizerSignalCollector,
    get_heterogeneous_optimizer_signal_collector,
    reset_dspy_integration,
)


class TestHeterogeneousOptimizationTrainingSignal:
    """Tests for HeterogeneousOptimizationTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with defaults."""
        signal = HeterogeneousOptimizationTrainingSignal()

        assert signal.signal_id == ""
        assert signal.session_id == ""
        assert signal.query == ""
        assert signal.treatment_var == ""
        assert signal.outcome_var == ""
        assert signal.segment_vars_count == 0
        assert signal.effect_modifiers_count == 0
        assert signal.overall_ate == 0.0
        assert signal.heterogeneity_score == 0.0
        assert signal.user_satisfaction is None
        assert signal.created_at is not None

    def test_custom_initialization(self):
        """Test signal with custom values."""
        signal = HeterogeneousOptimizationTrainingSignal(
            session_id="het-session-123",
            query="Find treatment effect heterogeneity",
            treatment_var="hcp_visits",
            outcome_var="prescriptions",
            segment_vars_count=5,
            effect_modifiers_count=3,
        )

        assert signal.session_id == "het-session-123"
        assert signal.treatment_var == "hcp_visits"
        assert signal.outcome_var == "prescriptions"
        assert signal.segment_vars_count == 5
        assert signal.effect_modifiers_count == 3

    def test_compute_reward_minimal(self):
        """Test reward with minimal data."""
        signal = HeterogeneousOptimizationTrainingSignal()
        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0
        assert reward < 0.5  # Minimal data should yield low reward

    def test_compute_reward_high_quality(self):
        """Test reward with high-quality CATE analysis."""
        signal = HeterogeneousOptimizationTrainingSignal(
            heterogeneity_score=0.5,  # Moderate heterogeneity (ideal)
            cate_segments_count=10,
            significant_cate_count=8,  # 80% significant
            high_responders_count=4,
            low_responders_count=3,
            responder_spread=0.35,  # 35% effect difference
            policy_recommendations_count=5,
            actionable_policies=4,
            expected_total_lift=0.15,
            total_latency_ms=8000,  # Under 12s target
            user_satisfaction=5.0,
        )
        reward = signal.compute_reward()

        assert reward > 0.7
        assert reward <= 1.0

    def test_compute_reward_heterogeneity_score_impact(self):
        """Test that moderate heterogeneity scores better."""
        # Moderate heterogeneity (ideal: 0.3-0.7)
        signal_moderate = HeterogeneousOptimizationTrainingSignal(
            heterogeneity_score=0.5,
            cate_segments_count=10,
            significant_cate_count=8,
        )

        # Very high heterogeneity (less actionable)
        signal_extreme = HeterogeneousOptimizationTrainingSignal(
            heterogeneity_score=0.95,
            cate_segments_count=10,
            significant_cate_count=8,
        )

        assert signal_moderate.compute_reward() >= signal_extreme.compute_reward()

    def test_compute_reward_segment_separation(self):
        """Test that responder separation impacts reward."""
        # Good segment separation
        signal_good_sep = HeterogeneousOptimizationTrainingSignal(
            high_responders_count=5,
            low_responders_count=4,
            responder_spread=0.4,  # 40% difference
        )

        # Poor segment separation
        signal_poor_sep = HeterogeneousOptimizationTrainingSignal(
            high_responders_count=5,
            low_responders_count=4,
            responder_spread=0.05,  # Only 5% difference
        )

        assert signal_good_sep.compute_reward() > signal_poor_sep.compute_reward()

    def test_compute_reward_missing_responders(self):
        """Test reward when missing high or low responders."""
        # Has both
        signal_both = HeterogeneousOptimizationTrainingSignal(
            high_responders_count=3,
            low_responders_count=3,
        )

        # Missing low responders
        signal_no_low = HeterogeneousOptimizationTrainingSignal(
            high_responders_count=3,
            low_responders_count=0,
        )

        assert signal_both.compute_reward() > signal_no_low.compute_reward()

    def test_compute_reward_policy_quality(self):
        """Test that policy quality impacts reward."""
        # Good policies
        signal_good = HeterogeneousOptimizationTrainingSignal(
            policy_recommendations_count=5,
            actionable_policies=4,
            expected_total_lift=0.20,
        )

        # No actionable policies
        signal_poor = HeterogeneousOptimizationTrainingSignal(
            policy_recommendations_count=5,
            actionable_policies=0,
            expected_total_lift=0.0,
        )

        assert signal_good.compute_reward() > signal_poor.compute_reward()

    def test_compute_reward_latency_impact(self):
        """Test that latency affects reward."""
        signal_fast = HeterogeneousOptimizationTrainingSignal(total_latency_ms=6000)
        signal_slow = HeterogeneousOptimizationTrainingSignal(total_latency_ms=40000)

        assert signal_fast.compute_reward() > signal_slow.compute_reward()

    def test_to_dict_structure(self):
        """Test dictionary serialization structure."""
        signal = HeterogeneousOptimizationTrainingSignal(
            session_id="sess-123",
            treatment_var="treatment",
            overall_ate=0.12,
            heterogeneity_score=0.45,
        )

        result = signal.to_dict()

        assert result["source_agent"] == "heterogeneous_optimizer"
        assert result["dspy_type"] == "sender"
        assert "input_context" in result
        assert "cate_estimation" in result
        assert "segment_discovery" in result
        assert "policy_learning" in result
        assert "output" in result
        assert "outcome" in result
        assert "reward" in result

    def test_to_dict_query_truncation(self):
        """Test long query truncation."""
        signal = HeterogeneousOptimizationTrainingSignal(query="A" * 600)
        result = signal.to_dict()

        assert len(result["input_context"]["query"]) <= 500


class TestHeterogeneousOptimizerSignalCollector:
    """Tests for HeterogeneousOptimizerSignalCollector."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = HeterogeneousOptimizerSignalCollector()

        assert collector.dspy_type == "sender"
        assert collector._signals_buffer == []
        assert collector._buffer_limit == 100

    def test_collect_optimization_signal(self):
        """Test signal collection initiation."""
        collector = HeterogeneousOptimizerSignalCollector()

        signal = collector.collect_optimization_signal(
            session_id="het-sess-123",
            query="Analyze CATE for marketing effect",
            treatment_var="marketing_intensity",
            outcome_var="conversion_rate",
            segment_vars_count=6,
            effect_modifiers_count=4,
        )

        assert isinstance(signal, HeterogeneousOptimizationTrainingSignal)
        assert signal.session_id == "het-sess-123"
        assert signal.treatment_var == "marketing_intensity"
        assert signal.segment_vars_count == 6

    def test_update_cate_estimation(self):
        """Test CATE estimation phase update."""
        collector = HeterogeneousOptimizerSignalCollector()
        signal = HeterogeneousOptimizationTrainingSignal()

        updated = collector.update_cate_estimation(
            signal,
            overall_ate=0.15,
            heterogeneity_score=0.45,
            cate_segments_count=12,
            significant_cate_count=9,
            estimation_latency_ms=4000,
        )

        assert updated.overall_ate == 0.15
        assert updated.heterogeneity_score == 0.45
        assert updated.cate_segments_count == 12
        assert updated.significant_cate_count == 9
        assert updated.estimation_latency_ms == 4000

    def test_update_segment_discovery(self):
        """Test segment discovery phase update."""
        collector = HeterogeneousOptimizerSignalCollector()
        signal = HeterogeneousOptimizationTrainingSignal()

        updated = collector.update_segment_discovery(
            signal,
            high_responders_count=5,
            low_responders_count=4,
            responder_spread=0.32,
            analysis_latency_ms=2500,
        )

        assert updated.high_responders_count == 5
        assert updated.low_responders_count == 4
        assert updated.responder_spread == 0.32
        assert updated.analysis_latency_ms == 2500

    def test_update_policy_learning_adds_to_buffer(self):
        """Test policy learning update adds signal to buffer."""
        collector = HeterogeneousOptimizerSignalCollector()
        signal = HeterogeneousOptimizationTrainingSignal()

        assert len(collector._signals_buffer) == 0

        collector.update_policy_learning(
            signal,
            policy_recommendations_count=6,
            expected_total_lift=0.18,
            actionable_policies=5,
            executive_summary_length=280,
            key_insights_count=4,
            visualization_data_complete=True,
            total_latency_ms=9000,
            confidence_score=0.82,
        )

        assert len(collector._signals_buffer) == 1
        assert signal.policy_recommendations_count == 6
        assert signal.expected_total_lift == 0.18
        assert signal.visualization_data_complete is True

    def test_buffer_limit_enforcement(self):
        """Test buffer respects limit."""
        collector = HeterogeneousOptimizerSignalCollector()
        collector._buffer_limit = 5

        for i in range(7):
            signal = HeterogeneousOptimizationTrainingSignal(session_id=f"het-{i}")
            collector.update_policy_learning(
                signal,
                policy_recommendations_count=3,
                expected_total_lift=0.10,
                actionable_policies=2,
                executive_summary_length=150,
                key_insights_count=3,
                visualization_data_complete=True,
                total_latency_ms=8000,
                confidence_score=0.7,
            )

        assert len(collector._signals_buffer) == 5
        assert collector._signals_buffer[0].session_id == "het-2"

    def test_update_with_feedback(self):
        """Test delayed feedback update."""
        collector = HeterogeneousOptimizerSignalCollector()
        signal = HeterogeneousOptimizationTrainingSignal()

        updated = collector.update_with_feedback(signal, user_satisfaction=4.5)

        assert updated.user_satisfaction == 4.5

    def test_get_signals_for_training(self):
        """Test signal retrieval for training."""
        collector = HeterogeneousOptimizerSignalCollector()

        for i, het_score in enumerate([0.3, 0.5, 0.7]):
            signal = HeterogeneousOptimizationTrainingSignal(
                session_id=f"het-{i}",
                heterogeneity_score=het_score,
            )
            collector.update_policy_learning(
                signal,
                policy_recommendations_count=4,
                expected_total_lift=0.12,
                actionable_policies=3,
                executive_summary_length=200,
                key_insights_count=3,
                visualization_data_complete=True,
                total_latency_ms=9000,
                confidence_score=0.75,
            )

        all_signals = collector.get_signals_for_training(min_reward=0.0)
        assert len(all_signals) == 3

    def test_get_high_heterogeneity_examples(self):
        """Test retrieval of high heterogeneity examples."""
        collector = HeterogeneousOptimizerSignalCollector()

        for i, het_score in enumerate([0.1, 0.35, 0.55, 0.2]):
            signal = HeterogeneousOptimizationTrainingSignal(
                session_id=f"het-{i}",
                heterogeneity_score=het_score,
            )
            collector.update_policy_learning(
                signal,
                policy_recommendations_count=3,
                expected_total_lift=0.10,
                actionable_policies=2,
                executive_summary_length=150,
                key_insights_count=2,
                visualization_data_complete=True,
                total_latency_ms=8000,
                confidence_score=0.7,
            )

        high_het = collector.get_high_heterogeneity_examples(min_heterogeneity=0.3, limit=10)

        assert len(high_het) == 2
        for ex in high_het:
            assert ex["cate_estimation"]["heterogeneity_score"] >= 0.3

    def test_clear_buffer(self):
        """Test buffer clearing."""
        collector = HeterogeneousOptimizerSignalCollector()

        signal = HeterogeneousOptimizationTrainingSignal()
        collector.update_policy_learning(
            signal,
            policy_recommendations_count=2,
            expected_total_lift=0.08,
            actionable_policies=1,
            executive_summary_length=100,
            key_insights_count=2,
            visualization_data_complete=False,
            total_latency_ms=7000,
            confidence_score=0.6,
        )

        assert len(collector._signals_buffer) == 1

        collector.clear_buffer()

        assert len(collector._signals_buffer) == 0


class TestSingletonAccess:
    """Tests for singleton access patterns."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_get_signal_collector_creates_singleton(self):
        """Test singleton creation."""
        collector1 = get_heterogeneous_optimizer_signal_collector()
        collector2 = get_heterogeneous_optimizer_signal_collector()

        assert collector1 is collector2

    def test_reset_dspy_integration(self):
        """Test singleton reset."""
        collector1 = get_heterogeneous_optimizer_signal_collector()
        reset_dspy_integration()
        collector2 = get_heterogeneous_optimizer_signal_collector()

        assert collector1 is not collector2


class TestDSPySignatures:
    """Tests for DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE reflects actual availability."""
        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_cate_interpretation_signature(self):
        """Test CATEInterpretationSignature exists."""
        import dspy

        from src.agents.heterogeneous_optimizer.dspy_integration import CATEInterpretationSignature

        assert issubclass(CATEInterpretationSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_policy_recommendation_signature(self):
        """Test PolicyRecommendationSignature exists."""
        import dspy

        from src.agents.heterogeneous_optimizer.dspy_integration import (
            PolicyRecommendationSignature,
        )

        assert issubclass(PolicyRecommendationSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_segment_profile_signature(self):
        """Test SegmentProfileSignature exists."""
        import dspy

        from src.agents.heterogeneous_optimizer.dspy_integration import SegmentProfileSignature

        assert issubclass(SegmentProfileSignature, dspy.Signature)


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
        collector = get_heterogeneous_optimizer_signal_collector()

        # Phase 1: Initialize
        signal = collector.collect_optimization_signal(
            session_id="lifecycle-het-test",
            query="Who responds best to HCP engagement?",
            treatment_var="hcp_engagement_score",
            outcome_var="prescription_uplift",
            segment_vars_count=8,
            effect_modifiers_count=5,
        )

        # Phase 2: CATE Estimation
        signal = collector.update_cate_estimation(
            signal,
            overall_ate=0.18,
            heterogeneity_score=0.52,
            cate_segments_count=15,
            significant_cate_count=12,
            estimation_latency_ms=4500,
        )

        # Phase 3: Segment Discovery
        signal = collector.update_segment_discovery(
            signal,
            high_responders_count=5,
            low_responders_count=4,
            responder_spread=0.38,
            analysis_latency_ms=3000,
        )

        # Phase 4: Policy Learning (adds to buffer)
        signal = collector.update_policy_learning(
            signal,
            policy_recommendations_count=6,
            expected_total_lift=0.22,
            actionable_policies=5,
            executive_summary_length=350,
            key_insights_count=5,
            visualization_data_complete=True,
            total_latency_ms=10000,
            confidence_score=0.85,
        )

        # Phase 5: Delayed feedback
        signal = collector.update_with_feedback(signal, user_satisfaction=4.8)

        # Verify final state
        assert signal.session_id == "lifecycle-het-test"
        assert signal.overall_ate == 0.18
        assert signal.heterogeneity_score == 0.52
        assert signal.high_responders_count == 5
        assert signal.policy_recommendations_count == 6
        assert signal.user_satisfaction == 4.8

        # Verify in buffer
        signals = collector.get_signals_for_training()
        assert len(signals) == 1

        # Verify reward is good
        reward = signal.compute_reward()
        assert reward > 0.5
