"""
Tests for Causal Impact Agent DSPy Integration.

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

from src.agents.causal_impact.dspy_integration import (
    DSPY_AVAILABLE,
    CausalAnalysisTrainingSignal,
    CausalImpactSignalCollector,
    get_causal_impact_signal_collector,
    reset_dspy_integration,
)


class TestCausalAnalysisTrainingSignal:
    """Tests for CausalAnalysisTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with defaults."""
        signal = CausalAnalysisTrainingSignal()

        assert signal.signal_id == ""
        assert signal.session_id == ""
        assert signal.query == ""
        assert signal.treatment_var == ""
        assert signal.outcome_var == ""
        assert signal.confounders_count == 0
        assert signal.dag_nodes_count == 0
        assert signal.ate_estimate == 0.0
        assert signal.refutation_tests_passed == 0
        assert signal.user_satisfaction is None
        assert signal.created_at is not None

    def test_custom_initialization(self):
        """Test signal with custom values."""
        signal = CausalAnalysisTrainingSignal(
            session_id="test-session-123",
            query="What is the causal effect of marketing on sales?",
            treatment_var="marketing_spend",
            outcome_var="sales_revenue",
            confounders_count=5,
        )

        assert signal.session_id == "test-session-123"
        assert signal.treatment_var == "marketing_spend"
        assert signal.outcome_var == "sales_revenue"
        assert signal.confounders_count == 5

    def test_compute_reward_minimal(self):
        """Test reward with minimal data."""
        signal = CausalAnalysisTrainingSignal()
        reward = signal.compute_reward()

        # Should have some base reward from efficiency + partial satisfaction
        assert 0.0 <= reward <= 1.0
        # With no tests, no significance, minimal depth
        assert reward < 0.5

    def test_compute_reward_high_quality(self):
        """Test reward with high-quality analysis."""
        signal = CausalAnalysisTrainingSignal(
            refutation_tests_passed=4,
            refutation_tests_failed=0,
            overall_robust=True,
            statistical_significance=True,
            ate_estimate=0.15,
            ate_ci_width=0.02,  # Narrow CI
            interpretation_depth="deep",
            key_findings_count=4,
            recommendations_count=3,
            total_latency_ms=10000,  # Under 15s target
            user_satisfaction=5.0,
        )
        reward = signal.compute_reward()

        assert reward > 0.7  # High-quality should score well
        assert reward <= 1.0

    def test_compute_reward_refutation_weighting(self):
        """Test that refutation robustness impacts reward."""
        # All tests pass
        signal_robust = CausalAnalysisTrainingSignal(
            refutation_tests_passed=4,
            refutation_tests_failed=0,
        )

        # All tests fail
        signal_not_robust = CausalAnalysisTrainingSignal(
            refutation_tests_passed=0,
            refutation_tests_failed=4,
        )

        assert signal_robust.compute_reward() > signal_not_robust.compute_reward()

    def test_compute_reward_latency_impact(self):
        """Test that latency affects reward."""
        # Fast analysis
        signal_fast = CausalAnalysisTrainingSignal(
            total_latency_ms=5000,  # 5s
        )

        # Slow analysis
        signal_slow = CausalAnalysisTrainingSignal(
            total_latency_ms=50000,  # 50s
        )

        assert signal_fast.compute_reward() > signal_slow.compute_reward()

    def test_to_dict_structure(self):
        """Test dictionary serialization structure."""
        signal = CausalAnalysisTrainingSignal(
            session_id="sess-123",
            query="Test query",
            treatment_var="treatment",
            outcome_var="outcome",
            ate_estimate=0.25,
            refutation_tests_passed=3,
        )

        result = signal.to_dict()

        # Check required keys
        assert "signal_id" in result
        assert "source_agent" in result
        assert result["source_agent"] == "causal_impact"
        assert result["dspy_type"] == "sender"
        assert "timestamp" in result
        assert "input_context" in result
        assert "graph_building" in result
        assert "estimation" in result
        assert "refutation" in result
        assert "sensitivity" in result
        assert "interpretation" in result
        assert "outcome" in result
        assert "reward" in result

    def test_to_dict_input_context(self):
        """Test input_context in serialized dict."""
        signal = CausalAnalysisTrainingSignal(
            query="A" * 600,  # Long query
            treatment_var="treatment",
            outcome_var="outcome",
            confounders_count=3,
        )

        result = signal.to_dict()

        # Query should be truncated to 500 chars
        assert len(result["input_context"]["query"]) <= 500
        assert result["input_context"]["treatment_var"] == "treatment"
        assert result["input_context"]["outcome_var"] == "outcome"
        assert result["input_context"]["confounders_count"] == 3


class TestCausalImpactSignalCollector:
    """Tests for CausalImpactSignalCollector."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = CausalImpactSignalCollector()

        assert collector.dspy_type == "sender"
        assert collector._signals_buffer == []
        assert collector._buffer_limit == 100

    def test_collect_analysis_signal(self):
        """Test signal collection initiation."""
        collector = CausalImpactSignalCollector()

        signal = collector.collect_analysis_signal(
            session_id="sess-123",
            query="Test causal query",
            treatment_var="treatment",
            outcome_var="outcome",
            confounders_count=4,
        )

        assert isinstance(signal, CausalAnalysisTrainingSignal)
        assert signal.session_id == "sess-123"
        assert signal.treatment_var == "treatment"
        assert signal.confounders_count == 4

    def test_update_graph_building(self):
        """Test graph building phase update."""
        collector = CausalImpactSignalCollector()
        signal = collector.collect_analysis_signal(
            session_id="sess-123",
            query="Test",
            treatment_var="t",
            outcome_var="o",
            confounders_count=2,
        )

        updated = collector.update_graph_building(
            signal,
            dag_nodes_count=10,
            dag_edges_count=15,
            adjustment_sets_found=3,
            graph_confidence=0.85,
        )

        assert updated.dag_nodes_count == 10
        assert updated.dag_edges_count == 15
        assert updated.adjustment_sets_found == 3
        assert updated.graph_confidence == 0.85

    def test_update_estimation(self):
        """Test estimation phase update."""
        collector = CausalImpactSignalCollector()
        signal = CausalAnalysisTrainingSignal()

        updated = collector.update_estimation(
            signal,
            method="propensity_score",
            ate_estimate=0.15,
            ate_ci_lower=0.10,
            ate_ci_upper=0.20,
            statistical_significance=True,
            effect_size="medium",
            sample_size=1000,
        )

        assert updated.estimation_method == "propensity_score"
        assert updated.ate_estimate == 0.15
        assert updated.ate_ci_width == 0.10  # 0.20 - 0.10
        assert updated.statistical_significance is True
        assert updated.effect_size == "medium"
        assert updated.sample_size == 1000

    def test_update_refutation(self):
        """Test refutation phase update."""
        collector = CausalImpactSignalCollector()
        signal = CausalAnalysisTrainingSignal()

        updated = collector.update_refutation(
            signal,
            tests_passed=3,
            tests_failed=1,
            overall_robust=True,
        )

        assert updated.refutation_tests_passed == 3
        assert updated.refutation_tests_failed == 1
        assert updated.overall_robust is True

    def test_update_sensitivity(self):
        """Test sensitivity phase update."""
        collector = CausalImpactSignalCollector()
        signal = CausalAnalysisTrainingSignal()

        updated = collector.update_sensitivity(
            signal,
            e_value=2.5,
            robust_to_confounding=True,
        )

        assert updated.e_value == 2.5
        assert updated.robust_to_confounding is True

    def test_update_interpretation_adds_to_buffer(self):
        """Test interpretation update adds signal to buffer."""
        collector = CausalImpactSignalCollector()
        signal = CausalAnalysisTrainingSignal()

        assert len(collector._signals_buffer) == 0

        collector.update_interpretation(
            signal,
            interpretation_depth="standard",
            narrative_length=500,
            key_findings_count=3,
            recommendations_count=2,
            total_latency_ms=8000,
            confidence_score=0.75,
        )

        assert len(collector._signals_buffer) == 1
        assert signal.interpretation_depth == "standard"
        assert signal.key_findings_count == 3

    def test_buffer_limit_enforcement(self):
        """Test buffer respects limit."""
        collector = CausalImpactSignalCollector()
        collector._buffer_limit = 5

        # Add 7 signals
        for i in range(7):
            signal = CausalAnalysisTrainingSignal(session_id=f"sess-{i}")
            collector.update_interpretation(
                signal,
                interpretation_depth="minimal",
                narrative_length=100,
                key_findings_count=1,
                recommendations_count=1,
                total_latency_ms=5000,
                confidence_score=0.5,
            )

        # Should only have 5 (oldest removed)
        assert len(collector._signals_buffer) == 5
        # First signal should be sess-2 (sess-0 and sess-1 removed)
        assert collector._signals_buffer[0].session_id == "sess-2"

    def test_update_with_feedback(self):
        """Test delayed feedback update."""
        collector = CausalImpactSignalCollector()
        signal = CausalAnalysisTrainingSignal()

        updated = collector.update_with_feedback(signal, user_satisfaction=4.5)

        assert updated.user_satisfaction == 4.5

    def test_get_signals_for_training(self):
        """Test signal retrieval for training."""
        collector = CausalImpactSignalCollector()

        # Add signals with varying rewards
        for i, robust in enumerate([True, False, True]):
            signal = CausalAnalysisTrainingSignal(
                session_id=f"sess-{i}",
                refutation_tests_passed=4 if robust else 0,
                refutation_tests_failed=0 if robust else 4,
            )
            collector.update_interpretation(
                signal,
                interpretation_depth="standard",
                narrative_length=200,
                key_findings_count=2,
                recommendations_count=2,
                total_latency_ms=10000,
                confidence_score=0.7 if robust else 0.3,
            )

        # Get all signals
        all_signals = collector.get_signals_for_training(min_reward=0.0)
        assert len(all_signals) == 3

        # Get filtered signals (higher reward threshold)
        filtered = collector.get_signals_for_training(min_reward=0.3)
        assert len(filtered) >= 1  # At least the robust ones

    def test_get_signals_with_limit(self):
        """Test signal retrieval with limit."""
        collector = CausalImpactSignalCollector()

        # Add 5 signals
        for i in range(5):
            signal = CausalAnalysisTrainingSignal(session_id=f"sess-{i}")
            collector.update_interpretation(
                signal,
                interpretation_depth="minimal",
                narrative_length=100,
                key_findings_count=1,
                recommendations_count=1,
                total_latency_ms=5000,
                confidence_score=0.5,
            )

        limited = collector.get_signals_for_training(min_reward=0.0, limit=2)
        assert len(limited) == 2

    def test_get_robust_examples(self):
        """Test retrieval of robust examples only."""
        collector = CausalImpactSignalCollector()

        # Add mix of robust and non-robust
        for i, robust in enumerate([True, False, True, False]):
            signal = CausalAnalysisTrainingSignal(
                session_id=f"sess-{i}",
                overall_robust=robust,
                refutation_tests_passed=4 if robust else 0,
            )
            collector.update_interpretation(
                signal,
                interpretation_depth="standard",
                narrative_length=200,
                key_findings_count=3,
                recommendations_count=2,
                total_latency_ms=10000,
                confidence_score=0.8,
            )

        robust_examples = collector.get_robust_examples(limit=10)

        # Should only include robust ones
        assert len(robust_examples) == 2
        for ex in robust_examples:
            assert ex["refutation"]["overall_robust"] is True

    def test_clear_buffer(self):
        """Test buffer clearing."""
        collector = CausalImpactSignalCollector()

        # Add a signal
        signal = CausalAnalysisTrainingSignal()
        collector.update_interpretation(
            signal,
            interpretation_depth="minimal",
            narrative_length=50,
            key_findings_count=1,
            recommendations_count=1,
            total_latency_ms=5000,
            confidence_score=0.5,
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
        collector1 = get_causal_impact_signal_collector()
        collector2 = get_causal_impact_signal_collector()

        assert collector1 is collector2

    def test_reset_dspy_integration(self):
        """Test singleton reset."""
        collector1 = get_causal_impact_signal_collector()

        reset_dspy_integration()

        collector2 = get_causal_impact_signal_collector()

        assert collector1 is not collector2


class TestDSPySignatures:
    """Tests for DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE reflects actual availability."""
        # This depends on whether dspy is installed
        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_causal_graph_signature_fields(self):
        """Test CausalGraphSignature has expected fields."""
        # Verify it's a DSPy signature
        import dspy

        from src.agents.causal_impact.dspy_integration import CausalGraphSignature

        assert issubclass(CausalGraphSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_evidence_synthesis_signature_fields(self):
        """Test EvidenceSynthesisSignature has expected fields."""
        import dspy

        from src.agents.causal_impact.dspy_integration import EvidenceSynthesisSignature

        assert issubclass(EvidenceSynthesisSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_causal_interpretation_signature_fields(self):
        """Test CausalInterpretationSignature has expected fields."""
        import dspy

        from src.agents.causal_impact.dspy_integration import CausalInterpretationSignature

        assert issubclass(CausalInterpretationSignature, dspy.Signature)


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
        collector = get_causal_impact_signal_collector()

        # Phase 1: Initialize
        signal = collector.collect_analysis_signal(
            session_id="lifecycle-test",
            query="What causes increased sales?",
            treatment_var="marketing_spend",
            outcome_var="sales",
            confounders_count=3,
        )

        # Phase 2: Graph building
        signal = collector.update_graph_building(
            signal,
            dag_nodes_count=8,
            dag_edges_count=12,
            adjustment_sets_found=2,
            graph_confidence=0.9,
        )

        # Phase 3: Estimation
        signal = collector.update_estimation(
            signal,
            method="double_ml",
            ate_estimate=0.12,
            ate_ci_lower=0.08,
            ate_ci_upper=0.16,
            statistical_significance=True,
            effect_size="small",
            sample_size=5000,
        )

        # Phase 4: Refutation
        signal = collector.update_refutation(
            signal,
            tests_passed=3,
            tests_failed=1,
            overall_robust=True,
        )

        # Phase 5: Sensitivity
        signal = collector.update_sensitivity(
            signal,
            e_value=2.0,
            robust_to_confounding=True,
        )

        # Phase 6: Interpretation (this adds to buffer)
        signal = collector.update_interpretation(
            signal,
            interpretation_depth="deep",
            narrative_length=800,
            key_findings_count=4,
            recommendations_count=3,
            total_latency_ms=12000,
            confidence_score=0.82,
        )

        # Phase 7: Delayed feedback
        signal = collector.update_with_feedback(signal, user_satisfaction=4.5)

        # Verify final state
        assert signal.session_id == "lifecycle-test"
        assert signal.dag_nodes_count == 8
        assert signal.ate_estimate == 0.12
        assert signal.refutation_tests_passed == 3
        assert signal.e_value == 2.0
        assert signal.interpretation_depth == "deep"
        assert signal.user_satisfaction == 4.5

        # Verify in buffer
        signals = collector.get_signals_for_training()
        assert len(signals) == 1

        # Verify reward is reasonable
        reward = signal.compute_reward()
        assert reward > 0.5  # Should be decent with good metrics
