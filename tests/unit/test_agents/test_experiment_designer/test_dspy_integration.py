"""
Tests for Experiment Designer Agent DSPy Integration.

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

from src.agents.experiment_designer.dspy_integration import (
    DSPY_AVAILABLE,
    ExperimentDesignerSignalCollector,
    ExperimentDesignTrainingSignal,
    get_experiment_designer_signal_collector,
    reset_dspy_integration,
)


class TestExperimentDesignTrainingSignal:
    """Tests for ExperimentDesignTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with defaults."""
        signal = ExperimentDesignTrainingSignal()

        assert signal.signal_id == ""
        assert signal.session_id == ""
        assert signal.business_question == ""
        assert signal.preregistration_formality == ""
        assert signal.max_redesign_iterations == 3
        assert signal.design_type_chosen == ""
        assert signal.treatments_count == 0
        assert signal.outcomes_count == 0
        assert signal.achieved_power == 0.0
        assert signal.validity_threats_identified == 0
        assert signal.experiment_approved is None
        assert signal.user_satisfaction is None
        assert signal.created_at is not None

    def test_custom_initialization(self):
        """Test signal with custom values."""
        signal = ExperimentDesignTrainingSignal(
            session_id="exp-session-123",
            business_question="Does increased HCP engagement improve TRx?",
            preregistration_formality="medium",
            max_redesign_iterations=5,
        )

        assert signal.session_id == "exp-session-123"
        assert signal.preregistration_formality == "medium"
        assert signal.max_redesign_iterations == 5

    def test_compute_reward_minimal(self):
        """Test reward with minimal data."""
        signal = ExperimentDesignTrainingSignal()
        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0
        assert reward < 0.5  # Minimal data should yield low reward

    def test_compute_reward_high_quality(self):
        """Test reward with high-quality experiment design."""
        signal = ExperimentDesignTrainingSignal(
            design_type_chosen="RCT",
            treatments_count=2,  # Control + treatment
            outcomes_count=3,  # Primary + 2 secondary
            randomization_unit="individual",
            required_sample_size=500,
            achieved_power=0.85,
            minimum_detectable_effect=0.1,
            duration_estimate_days=90,
            validity_threats_identified=4,
            critical_threats=2,
            mitigations_proposed=2,
            overall_validity_score=0.8,
            template_generated=True,
            causal_graph_generated=True,
            analysis_code_generated=True,
            experiment_approved=True,
        )
        reward = signal.compute_reward()

        assert reward > 0.7  # High-quality should score well
        assert reward <= 1.0

    def test_compute_reward_design_quality_impact(self):
        """Test that design quality affects reward."""
        # Good design
        signal_good = ExperimentDesignTrainingSignal(
            design_type_chosen="RCT",
            treatments_count=2,
            outcomes_count=3,
        )

        # Missing design
        signal_no_design = ExperimentDesignTrainingSignal(
            design_type_chosen="",
            treatments_count=0,
            outcomes_count=0,
        )

        assert signal_good.compute_reward() > signal_no_design.compute_reward()

    def test_compute_reward_power_quality_impact(self):
        """Test that power quality affects reward."""
        # Adequate power
        signal_powered = ExperimentDesignTrainingSignal(
            achieved_power=0.85,
            required_sample_size=500,
            duration_estimate_days=60,
        )

        # Underpowered
        signal_underpowered = ExperimentDesignTrainingSignal(
            achieved_power=0.50,
            required_sample_size=50,
            duration_estimate_days=7,
        )

        assert signal_powered.compute_reward() > signal_underpowered.compute_reward()

    def test_compute_reward_validity_handling(self):
        """Test that validity handling affects reward."""
        # Good threat mitigation
        signal_mitigated = ExperimentDesignTrainingSignal(
            validity_threats_identified=5,
            critical_threats=3,
            mitigations_proposed=3,
            overall_validity_score=0.8,
        )

        # No threats identified (may have missed some)
        signal_no_threats = ExperimentDesignTrainingSignal(
            validity_threats_identified=0,
            critical_threats=0,
            mitigations_proposed=0,
            overall_validity_score=0.0,
        )

        assert signal_mitigated.compute_reward() > signal_no_threats.compute_reward()

    def test_compute_reward_completeness_impact(self):
        """Test that completeness affects reward."""
        # All artifacts generated
        signal_complete = ExperimentDesignTrainingSignal(
            template_generated=True,
            causal_graph_generated=True,
            analysis_code_generated=True,
        )

        # No artifacts
        signal_incomplete = ExperimentDesignTrainingSignal(
            template_generated=False,
            causal_graph_generated=False,
            analysis_code_generated=False,
        )

        assert signal_complete.compute_reward() > signal_incomplete.compute_reward()

    def test_compute_reward_approval_impact(self):
        """Test that approval status affects reward."""
        signal_approved = ExperimentDesignTrainingSignal(
            experiment_approved=True,
        )

        signal_rejected = ExperimentDesignTrainingSignal(
            experiment_approved=False,
        )

        assert signal_approved.compute_reward() > signal_rejected.compute_reward()

    def test_compute_reward_satisfaction_as_proxy(self):
        """Test that satisfaction works as approval proxy."""
        signal = ExperimentDesignTrainingSignal(
            user_satisfaction=5.0,  # High satisfaction
        )
        reward = signal.compute_reward()

        # Satisfaction should contribute to reward
        assert reward > ExperimentDesignTrainingSignal().compute_reward()

    def test_to_dict_structure(self):
        """Test dictionary serialization structure."""
        signal = ExperimentDesignTrainingSignal(
            session_id="sess-123",
            design_type_chosen="RCT",
            achieved_power=0.8,
            validity_threats_identified=3,
        )

        result = signal.to_dict()

        assert result["source_agent"] == "experiment_designer"
        assert result["dspy_type"] == "sender"
        assert "input_context" in result
        assert "design_reasoning" in result
        assert "power_analysis" in result
        assert "validity_audit" in result
        assert "template_generation" in result
        assert "outcome" in result
        assert "reward" in result

    def test_to_dict_question_truncation(self):
        """Test long business question truncation."""
        signal = ExperimentDesignTrainingSignal(business_question="A" * 600)
        result = signal.to_dict()

        assert len(result["input_context"]["business_question"]) <= 500


class TestExperimentDesignerSignalCollector:
    """Tests for ExperimentDesignerSignalCollector."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = ExperimentDesignerSignalCollector()

        assert collector.dspy_type == "sender"
        assert collector._signals_buffer == []
        assert collector._buffer_limit == 100

    def test_collect_design_signal(self):
        """Test signal collection initiation."""
        collector = ExperimentDesignerSignalCollector()

        signal = collector.collect_design_signal(
            session_id="exp-sess-123",
            business_question="Does rep visit frequency impact TRx?",
            preregistration_formality="heavy",
            max_redesign_iterations=5,
        )

        assert isinstance(signal, ExperimentDesignTrainingSignal)
        assert signal.session_id == "exp-sess-123"
        assert signal.preregistration_formality == "heavy"
        assert signal.max_redesign_iterations == 5

    def test_update_design_reasoning(self):
        """Test design reasoning phase update."""
        collector = ExperimentDesignerSignalCollector()
        signal = ExperimentDesignTrainingSignal()

        updated = collector.update_design_reasoning(
            signal,
            design_type_chosen="quasi_experiment",
            treatments_count=3,
            outcomes_count=2,
            randomization_unit="cluster",
        )

        assert updated.design_type_chosen == "quasi_experiment"
        assert updated.treatments_count == 3
        assert updated.outcomes_count == 2
        assert updated.randomization_unit == "cluster"

    def test_update_power_analysis(self):
        """Test power analysis phase update."""
        collector = ExperimentDesignerSignalCollector()
        signal = ExperimentDesignTrainingSignal()

        updated = collector.update_power_analysis(
            signal,
            required_sample_size=1000,
            achieved_power=0.82,
            minimum_detectable_effect=0.05,
            duration_estimate_days=120,
        )

        assert updated.required_sample_size == 1000
        assert updated.achieved_power == 0.82
        assert updated.minimum_detectable_effect == 0.05
        assert updated.duration_estimate_days == 120

    def test_update_validity_audit(self):
        """Test validity audit phase update."""
        collector = ExperimentDesignerSignalCollector()
        signal = ExperimentDesignTrainingSignal()

        updated = collector.update_validity_audit(
            signal,
            validity_threats_identified=6,
            critical_threats=2,
            mitigations_proposed=2,
            overall_validity_score=0.75,
            redesign_iterations=1,
        )

        assert updated.validity_threats_identified == 6
        assert updated.critical_threats == 2
        assert updated.mitigations_proposed == 2
        assert updated.overall_validity_score == 0.75
        assert updated.redesign_iterations == 1

    def test_update_template_generation_adds_to_buffer(self):
        """Test template generation update adds signal to buffer."""
        collector = ExperimentDesignerSignalCollector()
        signal = ExperimentDesignTrainingSignal()

        assert len(collector._signals_buffer) == 0

        collector.update_template_generation(
            signal,
            template_generated=True,
            causal_graph_generated=True,
            analysis_code_generated=True,
            total_llm_tokens_used=5000,
            total_latency_ms=8000,
        )

        assert len(collector._signals_buffer) == 1
        assert signal.template_generated is True
        assert signal.causal_graph_generated is True

    def test_buffer_limit_enforcement(self):
        """Test buffer respects limit."""
        collector = ExperimentDesignerSignalCollector()
        collector._buffer_limit = 5

        for i in range(7):
            signal = ExperimentDesignTrainingSignal(session_id=f"exp-{i}")
            collector.update_template_generation(
                signal,
                template_generated=True,
                causal_graph_generated=False,
                analysis_code_generated=False,
                total_llm_tokens_used=1000,
                total_latency_ms=3000,
            )

        assert len(collector._signals_buffer) == 5
        assert collector._signals_buffer[0].session_id == "exp-2"

    def test_update_with_approval(self):
        """Test delayed approval update."""
        collector = ExperimentDesignerSignalCollector()
        signal = ExperimentDesignTrainingSignal()

        updated = collector.update_with_approval(
            signal,
            experiment_approved=True,
            user_satisfaction=4.5,
        )

        assert updated.experiment_approved is True
        assert updated.user_satisfaction == 4.5

    def test_get_signals_for_training(self):
        """Test signal retrieval for training."""
        collector = ExperimentDesignerSignalCollector()

        for i, power in enumerate([0.85, 0.70, 0.90]):
            signal = ExperimentDesignTrainingSignal(
                session_id=f"exp-{i}",
                achieved_power=power,
            )
            collector.update_template_generation(
                signal,
                template_generated=True,
                causal_graph_generated=True,
                analysis_code_generated=True,
                total_llm_tokens_used=2000,
                total_latency_ms=5000,
            )

        all_signals = collector.get_signals_for_training(min_reward=0.0)
        assert len(all_signals) == 3

    def test_get_approved_examples(self):
        """Test retrieval of approved experiment examples only."""
        collector = ExperimentDesignerSignalCollector()

        for i, approved in enumerate([True, False, True, None]):
            signal = ExperimentDesignTrainingSignal(session_id=f"exp-{i}")
            collector.update_template_generation(
                signal,
                template_generated=True,
                causal_graph_generated=False,
                analysis_code_generated=False,
                total_llm_tokens_used=1000,
                total_latency_ms=3000,
            )
            if approved is not None:
                collector.update_with_approval(signal, experiment_approved=approved)

        approved = collector.get_approved_examples(limit=10)

        # Only approved experiments (2 with True)
        assert len(approved) == 2
        for ex in approved:
            assert ex["outcome"]["experiment_approved"] is True

    def test_clear_buffer(self):
        """Test buffer clearing."""
        collector = ExperimentDesignerSignalCollector()

        signal = ExperimentDesignTrainingSignal()
        collector.update_template_generation(
            signal,
            template_generated=True,
            causal_graph_generated=False,
            analysis_code_generated=False,
            total_llm_tokens_used=1000,
            total_latency_ms=2000,
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
        collector1 = get_experiment_designer_signal_collector()
        collector2 = get_experiment_designer_signal_collector()

        assert collector1 is collector2

    def test_reset_dspy_integration(self):
        """Test singleton reset."""
        collector1 = get_experiment_designer_signal_collector()
        reset_dspy_integration()
        collector2 = get_experiment_designer_signal_collector()

        assert collector1 is not collector2


class TestDSPySignatures:
    """Tests for DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE reflects actual availability."""
        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_design_reasoning_signature(self):
        """Test DesignReasoningSignature exists."""
        import dspy

        from src.agents.experiment_designer.dspy_integration import DesignReasoningSignature

        assert issubclass(DesignReasoningSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_investigation_plan_signature(self):
        """Test InvestigationPlanSignature exists."""
        import dspy

        from src.agents.experiment_designer.dspy_integration import InvestigationPlanSignature

        assert issubclass(InvestigationPlanSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_validity_assessment_signature(self):
        """Test ValidityAssessmentSignature exists."""
        import dspy

        from src.agents.experiment_designer.dspy_integration import ValidityAssessmentSignature

        assert issubclass(ValidityAssessmentSignature, dspy.Signature)


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
        collector = get_experiment_designer_signal_collector()

        # Phase 1: Initialize
        signal = collector.collect_design_signal(
            session_id="lifecycle-exp-test",
            business_question="Does digital engagement drive TRx conversion?",
            preregistration_formality="medium",
            max_redesign_iterations=3,
        )

        # Phase 2: Design reasoning
        signal = collector.update_design_reasoning(
            signal,
            design_type_chosen="RCT",
            treatments_count=2,
            outcomes_count=3,
            randomization_unit="individual",
        )

        # Phase 3: Power analysis
        signal = collector.update_power_analysis(
            signal,
            required_sample_size=800,
            achieved_power=0.82,
            minimum_detectable_effect=0.08,
            duration_estimate_days=90,
        )

        # Phase 4: Validity audit
        signal = collector.update_validity_audit(
            signal,
            validity_threats_identified=5,
            critical_threats=2,
            mitigations_proposed=2,
            overall_validity_score=0.78,
            redesign_iterations=1,
        )

        # Phase 5: Template generation (adds to buffer)
        signal = collector.update_template_generation(
            signal,
            template_generated=True,
            causal_graph_generated=True,
            analysis_code_generated=True,
            total_llm_tokens_used=6000,
            total_latency_ms=12000,
        )

        # Phase 6: Delayed approval
        signal = collector.update_with_approval(
            signal,
            experiment_approved=True,
            user_satisfaction=4.5,
        )

        # Verify final state
        assert signal.session_id == "lifecycle-exp-test"
        assert signal.design_type_chosen == "RCT"
        assert signal.achieved_power == 0.82
        assert signal.validity_threats_identified == 5
        assert signal.template_generated is True
        assert signal.experiment_approved is True

        # Verify in buffer
        signals = collector.get_signals_for_training()
        assert len(signals) == 1

        # Verify reward is good
        reward = signal.compute_reward()
        assert reward > 0.5
