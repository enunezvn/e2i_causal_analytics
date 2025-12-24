"""
E2I Signal Flow Integration Tests - Batch 1: Sender Signal Generation

Tests that all Sender agents can generate valid training signals for DSPy optimization.

Sender agents:
- causal_impact
- gap_analyzer
- heterogeneous_optimizer
- drift_monitor
- experiment_designer
- prediction_synthesizer

Run: pytest tests/integration/test_signal_flow/test_sender_signals.py -v
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List


# =============================================================================
# SENDER SIGNAL IMPORTS
# =============================================================================


class TestCausalImpactSignals:
    """Test causal_impact agent signal generation."""

    def test_import_training_signal(self):
        """Verify CausalAnalysisTrainingSignal can be imported."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        assert CausalAnalysisTrainingSignal is not None

    def test_create_training_signal(self):
        """Create a valid training signal with default values."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        signal = CausalAnalysisTrainingSignal(
            signal_id="test_ci_001",
            session_id="session_123",
            query="What is the causal impact of marketing on sales?",
            treatment_var="marketing_spend",
            outcome_var="sales_revenue",
        )

        assert signal.signal_id == "test_ci_001"
        assert signal.session_id == "session_123"
        assert signal.treatment_var == "marketing_spend"
        assert signal.outcome_var == "sales_revenue"

    def test_compute_reward_zero_state(self):
        """Test reward computation with minimal data."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        signal = CausalAnalysisTrainingSignal()
        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0, "Reward should be in [0, 1] range"

    def test_compute_reward_high_quality(self):
        """Test reward computation with high-quality analysis."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        signal = CausalAnalysisTrainingSignal(
            signal_id="test_ci_002",
            session_id="session_456",
            query="Impact of pricing on conversion",
            treatment_var="price",
            outcome_var="conversion_rate",
            refutation_tests_passed=4,
            refutation_tests_failed=0,
            statistical_significance=True,
            ate_estimate=0.15,
            ate_ci_width=0.02,
            interpretation_depth="deep",
            key_findings_count=5,
            recommendations_count=3,
            total_latency_ms=10000,
            user_satisfaction=5.0,
        )

        reward = signal.compute_reward()

        assert reward >= 0.7, f"High quality signal should have reward >= 0.7, got {reward}"
        assert reward <= 1.0, "Reward should not exceed 1.0"

    def test_to_dict_serialization(self):
        """Test signal serialization to dictionary."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        signal = CausalAnalysisTrainingSignal(
            signal_id="test_ci_003",
            session_id="session_789",
            query="Test query",
            treatment_var="treatment",
            outcome_var="outcome",
        )

        signal_dict = signal.to_dict()

        assert isinstance(signal_dict, dict)
        assert signal_dict["source_agent"] == "causal_impact"
        assert signal_dict["dspy_type"] == "sender"
        assert "input_context" in signal_dict
        assert "timestamp" in signal_dict
        assert "reward" in signal_dict

    def test_signal_includes_dspy_type(self):
        """Verify signal includes dspy_type = 'sender'."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        signal = CausalAnalysisTrainingSignal()
        signal_dict = signal.to_dict()

        assert signal_dict["dspy_type"] == "sender"


class TestGapAnalyzerSignals:
    """Test gap_analyzer agent signal generation."""

    def test_import_training_signal(self):
        """Verify GapAnalysisTrainingSignal can be imported."""
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )

        assert GapAnalysisTrainingSignal is not None

    def test_create_training_signal(self):
        """Create a valid training signal."""
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )

        signal = GapAnalysisTrainingSignal(
            signal_id="test_ga_001",
            session_id="session_123",
            query="What are the ROI opportunities in the northeast region?",
        )

        assert signal.signal_id == "test_ga_001"
        assert signal.session_id == "session_123"

    def test_compute_reward(self):
        """Test reward computation."""
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )

        signal = GapAnalysisTrainingSignal(
            gaps_detected_count=5,
            roi_estimates_count=4,
            segments_analyzed=10,
            avg_expected_roi=2.5,
            total_latency_ms=8000,
        )

        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0

    def test_to_dict_has_required_fields(self):
        """Test serialization has required fields."""
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )

        signal = GapAnalysisTrainingSignal()
        signal_dict = signal.to_dict()

        assert signal_dict["source_agent"] == "gap_analyzer"
        assert signal_dict["dspy_type"] == "sender"


class TestHeterogeneousOptimizerSignals:
    """Test heterogeneous_optimizer agent signal generation."""

    def test_import_training_signal(self):
        """Verify HeterogeneousOptimizationTrainingSignal can be imported."""
        from src.agents.heterogeneous_optimizer.dspy_integration import (
            HeterogeneousOptimizationTrainingSignal,
        )

        assert HeterogeneousOptimizationTrainingSignal is not None

    def test_create_training_signal(self):
        """Create a valid training signal."""
        from src.agents.heterogeneous_optimizer.dspy_integration import (
            HeterogeneousOptimizationTrainingSignal,
        )

        signal = HeterogeneousOptimizationTrainingSignal(
            signal_id="test_ho_001",
            session_id="session_123",
            query="Which segments respond best to email campaigns?",
        )

        assert signal.signal_id == "test_ho_001"

    def test_compute_reward(self):
        """Test reward computation."""
        from src.agents.heterogeneous_optimizer.dspy_integration import (
            HeterogeneousOptimizationTrainingSignal,
        )

        signal = HeterogeneousOptimizationTrainingSignal(
            cate_segments_count=8,
            significant_cate_count=6,
            heterogeneity_score=0.5,
            high_responders_count=3,
            low_responders_count=2,
            policy_recommendations_count=4,
            actionable_policies=3,
            total_latency_ms=12000,
        )

        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0

    def test_to_dict_has_required_fields(self):
        """Test serialization has required fields."""
        from src.agents.heterogeneous_optimizer.dspy_integration import (
            HeterogeneousOptimizationTrainingSignal,
        )

        signal = HeterogeneousOptimizationTrainingSignal()
        signal_dict = signal.to_dict()

        assert signal_dict["source_agent"] == "heterogeneous_optimizer"
        assert signal_dict["dspy_type"] == "sender"


class TestDriftMonitorSignals:
    """Test drift_monitor agent signal generation."""

    def test_import_training_signal(self):
        """Verify DriftDetectionTrainingSignal can be imported."""
        from src.agents.drift_monitor.dspy_integration import (
            DriftDetectionTrainingSignal,
        )

        assert DriftDetectionTrainingSignal is not None

    def test_create_training_signal(self):
        """Create a valid training signal."""
        from src.agents.drift_monitor.dspy_integration import (
            DriftDetectionTrainingSignal,
        )

        signal = DriftDetectionTrainingSignal(
            signal_id="test_dm_001",
            session_id="session_123",
            query="Is there data drift in the churn model?",
        )

        assert signal.signal_id == "test_dm_001"

    def test_compute_reward(self):
        """Test reward computation."""
        from src.agents.drift_monitor.dspy_integration import (
            DriftDetectionTrainingSignal,
        )

        signal = DriftDetectionTrainingSignal(
            data_drift_count=2,
            model_drift_count=1,
            concept_drift_count=0,
            features_monitored=10,
            features_checked=10,
            alerts_generated=2,
            critical_alerts=1,
            recommended_actions_count=3,
            total_latency_ms=5000,
        )

        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0

    def test_to_dict_has_required_fields(self):
        """Test serialization has required fields."""
        from src.agents.drift_monitor.dspy_integration import (
            DriftDetectionTrainingSignal,
        )

        signal = DriftDetectionTrainingSignal()
        signal_dict = signal.to_dict()

        assert signal_dict["source_agent"] == "drift_monitor"
        assert signal_dict["dspy_type"] == "sender"


class TestExperimentDesignerSignals:
    """Test experiment_designer agent signal generation."""

    def test_import_training_signal(self):
        """Verify ExperimentDesignTrainingSignal can be imported."""
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )

        assert ExperimentDesignTrainingSignal is not None

    def test_create_training_signal(self):
        """Create a valid training signal."""
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )

        signal = ExperimentDesignTrainingSignal(
            signal_id="test_ed_001",
            session_id="session_123",
            business_question="Design an A/B test for the new pricing page",
        )

        assert signal.signal_id == "test_ed_001"

    def test_compute_reward(self):
        """Test reward computation."""
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )

        signal = ExperimentDesignTrainingSignal(
            design_type_chosen="RCT",
            treatments_count=2,
            outcomes_count=3,
            required_sample_size=500,
            achieved_power=0.85,
            duration_estimate_days=30,
            validity_threats_identified=4,
            critical_threats=1,
            mitigations_proposed=4,
            overall_validity_score=0.8,
            template_generated=True,
            causal_graph_generated=True,
            analysis_code_generated=True,
            total_latency_ms=20000,
        )

        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0

    def test_to_dict_has_required_fields(self):
        """Test serialization has required fields."""
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )

        signal = ExperimentDesignTrainingSignal()
        signal_dict = signal.to_dict()

        assert signal_dict["source_agent"] == "experiment_designer"
        assert signal_dict["dspy_type"] == "sender"


class TestPredictionSynthesizerSignals:
    """Test prediction_synthesizer agent signal generation."""

    def test_import_training_signal(self):
        """Verify PredictionSynthesisTrainingSignal can be imported."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        assert PredictionSynthesisTrainingSignal is not None

    def test_create_training_signal(self):
        """Create a valid training signal."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal(
            signal_id="test_ps_001",
            session_id="session_123",
            query="Predict churn probability for customer segment A",
        )

        assert signal.signal_id == "test_ps_001"

    def test_compute_reward(self):
        """Test reward computation."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal(
            models_requested=5,
            models_succeeded=4,
            models_failed=1,
            ensemble_method="weighted",
            point_estimate=0.72,
            prediction_interval_width=0.28,
            ensemble_confidence=0.85,
            model_agreement=0.90,
            similar_cases_found=5,
            feature_importance_calculated=True,
            trend_direction="stable",
            total_latency_ms=3000,
        )

        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0

    def test_to_dict_has_required_fields(self):
        """Test serialization has required fields."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal()
        signal_dict = signal.to_dict()

        assert signal_dict["source_agent"] == "prediction_synthesizer"
        assert signal_dict["dspy_type"] == "sender"


# =============================================================================
# CROSS-SENDER VALIDATION
# =============================================================================


class TestAllSendersConsistency:
    """Test consistency across all sender agent signals."""

    @pytest.fixture
    def all_sender_signals(self) -> List[Dict[str, Any]]:
        """Create sample signals from all sender agents."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )
        from src.agents.heterogeneous_optimizer.dspy_integration import (
            HeterogeneousOptimizationTrainingSignal,
        )
        from src.agents.drift_monitor.dspy_integration import (
            DriftDetectionTrainingSignal,
        )
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signals = [
            CausalAnalysisTrainingSignal(
                signal_id="ci_001", session_id="session_test"
            ).to_dict(),
            GapAnalysisTrainingSignal(
                signal_id="ga_001", session_id="session_test"
            ).to_dict(),
            HeterogeneousOptimizationTrainingSignal(
                signal_id="ho_001", session_id="session_test"
            ).to_dict(),
            DriftDetectionTrainingSignal(
                signal_id="dm_001", session_id="session_test"
            ).to_dict(),
            ExperimentDesignTrainingSignal(
                signal_id="ed_001", session_id="session_test"
            ).to_dict(),
            PredictionSynthesisTrainingSignal(
                signal_id="ps_001", session_id="session_test"
            ).to_dict(),
        ]

        return signals

    def test_all_signals_have_source_agent(self, all_sender_signals):
        """All signals should have source_agent field."""
        expected_agents = {
            "causal_impact",
            "gap_analyzer",
            "heterogeneous_optimizer",
            "drift_monitor",
            "experiment_designer",
            "prediction_synthesizer",
        }

        actual_agents = {s["source_agent"] for s in all_sender_signals}

        assert actual_agents == expected_agents

    def test_all_signals_have_dspy_type_sender(self, all_sender_signals):
        """All sender signals should have dspy_type='sender'."""
        for signal in all_sender_signals:
            assert signal["dspy_type"] == "sender", (
                f"Signal from {signal['source_agent']} should have dspy_type='sender'"
            )

    def test_all_signals_have_timestamp(self, all_sender_signals):
        """All signals should have timestamp."""
        for signal in all_sender_signals:
            assert "timestamp" in signal, (
                f"Signal from {signal['source_agent']} missing timestamp"
            )

    def test_all_signals_have_reward(self, all_sender_signals):
        """All signals should have computed reward."""
        for signal in all_sender_signals:
            assert "reward" in signal, (
                f"Signal from {signal['source_agent']} missing reward"
            )
            assert 0.0 <= signal["reward"] <= 1.0, (
                f"Signal from {signal['source_agent']} has invalid reward: {signal['reward']}"
            )

    def test_all_signals_have_input_context(self, all_sender_signals):
        """All signals should have input_context."""
        for signal in all_sender_signals:
            assert "input_context" in signal, (
                f"Signal from {signal['source_agent']} missing input_context"
            )

    def test_signal_count_matches_contract(self, all_sender_signals):
        """Should have exactly 6 sender agents per SignalFlowContract."""
        assert len(all_sender_signals) == 6


class TestSignalQualityThresholds:
    """Test signal quality against contract thresholds."""

    def test_min_signal_quality_threshold(self):
        """Signals below min_signal_quality should be filtered out."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        # Per SignalFlowContract: min_signal_quality = 0.6
        MIN_SIGNAL_QUALITY = 0.6

        # Low quality signal
        low_quality = CausalAnalysisTrainingSignal()
        assert low_quality.compute_reward() < MIN_SIGNAL_QUALITY

        # High quality signal
        high_quality = CausalAnalysisTrainingSignal(
            refutation_tests_passed=4,
            refutation_tests_failed=0,
            statistical_significance=True,
            ate_estimate=0.15,
            ate_ci_width=0.02,
            interpretation_depth="deep",
            key_findings_count=5,
            recommendations_count=3,
            total_latency_ms=10000,
            user_satisfaction=5.0,
        )
        assert high_quality.compute_reward() >= MIN_SIGNAL_QUALITY

    def test_signal_filtering_by_quality(self):
        """Test filtering signals by quality threshold."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        MIN_SIGNAL_QUALITY = 0.6

        signals = [
            CausalAnalysisTrainingSignal().to_dict(),  # Low quality
            CausalAnalysisTrainingSignal(
                refutation_tests_passed=4,
                statistical_significance=True,
                interpretation_depth="deep",
            ).to_dict(),  # Medium quality
            CausalAnalysisTrainingSignal(
                refutation_tests_passed=4,
                refutation_tests_failed=0,
                statistical_significance=True,
                ate_estimate=0.15,
                ate_ci_width=0.02,
                interpretation_depth="deep",
                key_findings_count=5,
                recommendations_count=3,
                total_latency_ms=10000,
                user_satisfaction=5.0,
            ).to_dict(),  # High quality
        ]

        high_quality_signals = [
            s for s in signals if s["reward"] >= MIN_SIGNAL_QUALITY
        ]

        # At least one should pass (high quality)
        assert len(high_quality_signals) >= 1
        # Not all should pass (low quality should fail)
        assert len(high_quality_signals) < len(signals)
