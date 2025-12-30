"""
Tests for Prediction Synthesizer DSPy Integration.

Tests the Sender role implementation including:
- Training signal generation and validation
- Signal collector functionality
- DSPy signature availability
- Singleton pattern for signal collector
- Full signal lifecycle
"""

import pytest
from unittest.mock import patch, MagicMock

# Mark all tests in this module as dspy_integration to group them
pytestmark = pytest.mark.xdist_group(name="dspy_integration")


class TestPredictionSynthesisTrainingSignal:
    """Test PredictionSynthesisTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with sensible defaults."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal()

        assert signal.signal_id == ""
        assert signal.session_id == ""
        assert signal.query == ""
        assert signal.models_requested == 0
        assert signal.models_succeeded == 0
        assert signal.point_estimate == 0.0
        assert signal.ensemble_confidence == 0.0
        assert signal.model_agreement == 0.0
        assert signal.total_latency_ms == 0.0
        assert signal.prediction_accuracy is None
        assert signal.user_satisfaction is None
        assert signal.created_at  # Should have a timestamp

    def test_custom_initialization(self):
        """Test signal with custom values."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal(
            signal_id="ps_test_1",
            session_id="session_123",
            query="What is the churn probability for HCP-456?",
            entity_id="hcp_456",
            entity_type="hcp",
            prediction_target="churn",
            time_horizon="30d",
            models_requested=3,
            models_succeeded=3,
            models_failed=0,
            ensemble_method="weighted",
            point_estimate=0.72,
            prediction_interval_width=0.28,
            ensemble_confidence=0.85,
            model_agreement=0.91,
            similar_cases_found=5,
            feature_importance_calculated=True,
            historical_accuracy=0.82,
            trend_direction="stable",
            total_latency_ms=450.0,
            orchestration_latency_ms=150.0,
            ensemble_latency_ms=300.0,
        )

        assert signal.signal_id == "ps_test_1"
        assert signal.models_requested == 3
        assert signal.ensemble_method == "weighted"
        assert signal.point_estimate == 0.72
        assert signal.ensemble_confidence == 0.85

    def test_compute_reward_high_quality(self):
        """Test reward computation for high-quality prediction."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal(
            models_requested=3,
            models_succeeded=3,
            ensemble_confidence=0.9,
            model_agreement=0.95,
            point_estimate=0.72,
            prediction_interval_width=0.2,  # Narrow interval
            total_latency_ms=2000,  # Fast
            similar_cases_found=5,
            feature_importance_calculated=True,
            trend_direction="stable",
            prediction_accuracy=0.85,
        )

        reward = signal.compute_reward()

        # Should be high reward (> 0.7)
        assert reward >= 0.7
        assert reward <= 1.0

    def test_compute_reward_low_quality(self):
        """Test reward computation for low-quality prediction."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal(
            models_requested=3,
            models_succeeded=1,  # Low success rate
            ensemble_confidence=0.3,  # Low confidence
            model_agreement=0.2,  # Low agreement
            point_estimate=0.5,
            prediction_interval_width=0.8,  # Wide interval
            total_latency_ms=10000,  # Slow
            similar_cases_found=0,
            feature_importance_calculated=False,
            trend_direction="",
        )

        reward = signal.compute_reward()

        # Should be lower reward
        assert reward < 0.5

    def test_compute_reward_with_user_satisfaction(self):
        """Test reward uses user satisfaction when accuracy not available."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal(
            models_requested=2,
            models_succeeded=2,
            ensemble_confidence=0.8,
            model_agreement=0.8,
            point_estimate=0.5,
            prediction_interval_width=0.2,
            total_latency_ms=3000,
            similar_cases_found=3,
            feature_importance_calculated=True,
            trend_direction="increasing",
            prediction_accuracy=None,
            user_satisfaction=5.0,  # High satisfaction
        )

        reward = signal.compute_reward()
        assert reward > 0.5

    def test_to_dict_structure(self):
        """Test to_dict produces correct structure."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        signal = PredictionSynthesisTrainingSignal(
            session_id="sess_123",
            query="Test query",
            entity_id="hcp_1",
            entity_type="hcp",
            prediction_target="churn",
            time_horizon="30d",
            models_requested=2,
            models_succeeded=2,
            ensemble_method="average",
            point_estimate=0.6,
            ensemble_confidence=0.75,
        )

        result = signal.to_dict()

        # Check structure
        assert result["source_agent"] == "prediction_synthesizer"
        assert result["dspy_type"] == "sender"
        assert "signal_id" in result
        assert "timestamp" in result
        assert "input_context" in result
        assert "model_orchestration" in result
        assert "ensemble_results" in result
        assert "context_enrichment" in result
        assert "outcome" in result
        assert "reward" in result

        # Check nested structure
        assert result["input_context"]["entity_id"] == "hcp_1"
        assert result["model_orchestration"]["ensemble_method"] == "average"
        assert result["ensemble_results"]["ensemble_confidence"] == 0.75

    def test_to_dict_truncates_long_query(self):
        """Test that long queries are truncated."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        long_query = "x" * 1000
        signal = PredictionSynthesisTrainingSignal(query=long_query)

        result = signal.to_dict()

        assert len(result["input_context"]["query"]) == 500


class TestPredictionSynthesizerSignalCollector:
    """Test PredictionSynthesizerSignalCollector class."""

    def test_initialization(self):
        """Test collector initializes with empty buffer."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
        )

        collector = PredictionSynthesizerSignalCollector()

        assert collector._signals_buffer == []
        assert collector._buffer_limit == 100

    def test_collect_synthesis_signal(self):
        """Test signal collection at synthesis start."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
        )

        collector = PredictionSynthesizerSignalCollector()

        signal = collector.collect_synthesis_signal(
            session_id="sess_123",
            query="Predict churn for HCP-456",
            entity_id="hcp_456",
            entity_type="hcp",
            prediction_target="churn",
            time_horizon="30d",
            models_requested=3,
        )

        assert signal.session_id == "sess_123"
        assert signal.query == "Predict churn for HCP-456"
        assert signal.models_requested == 3

    def test_update_model_orchestration(self):
        """Test updating signal with orchestration results."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            PredictionSynthesisTrainingSignal,
        )

        collector = PredictionSynthesizerSignalCollector()
        signal = PredictionSynthesisTrainingSignal(models_requested=3)

        updated = collector.update_model_orchestration(
            signal=signal,
            models_succeeded=2,
            models_failed=1,
            ensemble_method="weighted",
            orchestration_latency_ms=150.0,
        )

        assert updated.models_succeeded == 2
        assert updated.models_failed == 1
        assert updated.ensemble_method == "weighted"
        assert updated.orchestration_latency_ms == 150.0

    def test_update_ensemble_results(self):
        """Test updating signal with ensemble results."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            PredictionSynthesisTrainingSignal,
        )

        collector = PredictionSynthesizerSignalCollector()
        signal = PredictionSynthesisTrainingSignal()

        updated = collector.update_ensemble_results(
            signal=signal,
            point_estimate=0.72,
            prediction_interval_lower=0.58,
            prediction_interval_upper=0.86,
            ensemble_confidence=0.85,
            model_agreement=0.91,
            ensemble_latency_ms=300.0,
        )

        assert updated.point_estimate == 0.72
        assert updated.prediction_interval_width == 0.28  # 0.86 - 0.58
        assert updated.ensemble_confidence == 0.85
        assert updated.model_agreement == 0.91

    def test_update_context_enrichment_adds_to_buffer(self):
        """Test that context enrichment update adds signal to buffer."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            PredictionSynthesisTrainingSignal,
        )

        collector = PredictionSynthesizerSignalCollector()
        signal = PredictionSynthesisTrainingSignal()

        assert len(collector._signals_buffer) == 0

        collector.update_context_enrichment(
            signal=signal,
            similar_cases_found=5,
            feature_importance_calculated=True,
            historical_accuracy=0.82,
            trend_direction="stable",
            total_latency_ms=450.0,
        )

        assert len(collector._signals_buffer) == 1
        assert collector._signals_buffer[0].similar_cases_found == 5

    def test_buffer_limit_enforcement(self):
        """Test that buffer respects size limit."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            PredictionSynthesisTrainingSignal,
        )

        collector = PredictionSynthesizerSignalCollector()
        collector._buffer_limit = 5

        # Add 10 signals
        for i in range(10):
            signal = PredictionSynthesisTrainingSignal(session_id=f"sess_{i}")
            collector.update_context_enrichment(
                signal=signal,
                similar_cases_found=i,
                feature_importance_calculated=True,
                historical_accuracy=0.8,
                trend_direction="stable",
                total_latency_ms=100.0,
            )

        # Should only have 5 signals (the last 5)
        assert len(collector._signals_buffer) == 5
        assert collector._signals_buffer[0].session_id == "sess_5"
        assert collector._signals_buffer[-1].session_id == "sess_9"

    def test_update_with_accuracy(self):
        """Test delayed accuracy update."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            PredictionSynthesisTrainingSignal,
        )

        collector = PredictionSynthesizerSignalCollector()
        signal = PredictionSynthesisTrainingSignal()

        updated = collector.update_with_accuracy(
            signal=signal,
            prediction_accuracy=0.92,
            user_satisfaction=4.5,
        )

        assert updated.prediction_accuracy == 0.92
        assert updated.user_satisfaction == 4.5

    def test_get_signals_for_training(self):
        """Test getting signals filtered by min reward."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            PredictionSynthesisTrainingSignal,
        )

        collector = PredictionSynthesizerSignalCollector()

        # Add high-quality signal
        high_quality = PredictionSynthesisTrainingSignal(
            models_requested=3,
            models_succeeded=3,
            ensemble_confidence=0.9,
            model_agreement=0.9,
            point_estimate=0.7,
            prediction_interval_width=0.2,
            total_latency_ms=1000,
            similar_cases_found=5,
            feature_importance_calculated=True,
            trend_direction="stable",
            prediction_accuracy=0.9,
        )
        collector._signals_buffer.append(high_quality)

        # Add low-quality signal
        low_quality = PredictionSynthesisTrainingSignal(
            models_requested=3,
            models_succeeded=1,
            ensemble_confidence=0.3,
            model_agreement=0.2,
        )
        collector._signals_buffer.append(low_quality)

        # Get signals with min_reward=0.5
        signals = collector.get_signals_for_training(min_reward=0.5)

        # Should only return high-quality signal
        assert len(signals) >= 1

    def test_get_accurate_examples(self):
        """Test getting examples with high accuracy."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            PredictionSynthesisTrainingSignal,
        )

        collector = PredictionSynthesizerSignalCollector()

        # Add signal with high accuracy
        signal1 = PredictionSynthesisTrainingSignal(prediction_accuracy=0.95)
        # Add signal with low accuracy
        signal2 = PredictionSynthesisTrainingSignal(prediction_accuracy=0.5)
        # Add signal without accuracy
        signal3 = PredictionSynthesisTrainingSignal(prediction_accuracy=None)

        collector._signals_buffer = [signal1, signal2, signal3]

        examples = collector.get_accurate_examples(min_accuracy=0.8)

        assert len(examples) == 1

    def test_clear_buffer(self):
        """Test buffer clearing."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            PredictionSynthesisTrainingSignal,
        )

        collector = PredictionSynthesizerSignalCollector()
        collector._signals_buffer = [
            PredictionSynthesisTrainingSignal(),
            PredictionSynthesisTrainingSignal(),
        ]

        collector.clear_buffer()

        assert len(collector._signals_buffer) == 0


class TestSingletonAccess:
    """Test singleton pattern for signal collector."""

    def test_get_signal_collector_creates_singleton(self):
        """Test that getter creates singleton."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            get_prediction_synthesizer_signal_collector,
            reset_dspy_integration,
        )

        reset_dspy_integration()

        collector1 = get_prediction_synthesizer_signal_collector()
        collector2 = get_prediction_synthesizer_signal_collector()

        assert collector1 is collector2

    def test_reset_clears_singleton(self):
        """Test that reset clears singleton."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            get_prediction_synthesizer_signal_collector,
            reset_dspy_integration,
        )

        collector1 = get_prediction_synthesizer_signal_collector()
        reset_dspy_integration()
        collector2 = get_prediction_synthesizer_signal_collector()

        assert collector1 is not collector2


class TestDSPySignatures:
    """Test DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE flag."""
        from src.agents.prediction_synthesizer.dspy_integration import DSPY_AVAILABLE

        # DSPy should be available in test environment
        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_prediction_synthesis_signature(self):
        """Test PredictionSynthesisSignature is valid DSPy signature."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisSignature,
            DSPY_AVAILABLE,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(PredictionSynthesisSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_prediction_interpretation_signature(self):
        """Test PredictionInterpretationSignature is valid DSPy signature."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionInterpretationSignature,
            DSPY_AVAILABLE,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(PredictionInterpretationSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_uncertainty_quantification_signature(self):
        """Test UncertaintyQuantificationSignature is valid DSPy signature."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            UncertaintyQuantificationSignature,
            DSPY_AVAILABLE,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(UncertaintyQuantificationSignature, dspy.Signature)


class TestFullSignalLifecycle:
    """Test complete signal lifecycle for prediction synthesis."""

    def test_full_lifecycle(self):
        """Test signal through complete prediction lifecycle."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
            reset_dspy_integration,
        )

        reset_dspy_integration()
        collector = PredictionSynthesizerSignalCollector()

        # Phase 1: Start synthesis
        signal = collector.collect_synthesis_signal(
            session_id="lifecycle_test",
            query="Predict churn probability",
            entity_id="hcp_123",
            entity_type="hcp",
            prediction_target="churn",
            time_horizon="30d",
            models_requested=3,
        )

        assert signal.session_id == "lifecycle_test"
        assert signal.models_requested == 3

        # Phase 2: Model orchestration
        signal = collector.update_model_orchestration(
            signal=signal,
            models_succeeded=3,
            models_failed=0,
            ensemble_method="weighted",
            orchestration_latency_ms=150.0,
        )

        assert signal.models_succeeded == 3
        assert signal.ensemble_method == "weighted"

        # Phase 3: Ensemble combination
        signal = collector.update_ensemble_results(
            signal=signal,
            point_estimate=0.72,
            prediction_interval_lower=0.58,
            prediction_interval_upper=0.86,
            ensemble_confidence=0.85,
            model_agreement=0.91,
            ensemble_latency_ms=200.0,
        )

        assert signal.point_estimate == 0.72
        assert signal.ensemble_confidence == 0.85

        # Phase 4: Context enrichment (adds to buffer)
        signal = collector.update_context_enrichment(
            signal=signal,
            similar_cases_found=5,
            feature_importance_calculated=True,
            historical_accuracy=0.82,
            trend_direction="stable",
            total_latency_ms=450.0,
        )

        assert len(collector._signals_buffer) == 1

        # Phase 5: Delayed accuracy update
        signal = collector.update_with_accuracy(
            signal=signal,
            prediction_accuracy=0.88,
            user_satisfaction=4.5,
        )

        # Verify final signal
        final_dict = signal.to_dict()
        assert final_dict["source_agent"] == "prediction_synthesizer"
        assert final_dict["dspy_type"] == "sender"
        assert final_dict["reward"] > 0.5  # Should be good reward

    def test_lifecycle_with_failures(self):
        """Test signal lifecycle with model failures."""
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesizerSignalCollector,
        )

        collector = PredictionSynthesizerSignalCollector()

        signal = collector.collect_synthesis_signal(
            session_id="failure_test",
            query="Predict conversion",
            entity_id="hcp_456",
            entity_type="hcp",
            prediction_target="conversion",
            time_horizon="7d",
            models_requested=5,
        )

        # Some models fail
        signal = collector.update_model_orchestration(
            signal=signal,
            models_succeeded=2,
            models_failed=3,
            ensemble_method="average",
            orchestration_latency_ms=500.0,
        )

        # Lower quality ensemble
        signal = collector.update_ensemble_results(
            signal=signal,
            point_estimate=0.45,
            prediction_interval_lower=0.2,
            prediction_interval_upper=0.7,
            ensemble_confidence=0.5,
            model_agreement=0.4,
            ensemble_latency_ms=100.0,
        )

        signal = collector.update_context_enrichment(
            signal=signal,
            similar_cases_found=1,
            feature_importance_calculated=False,
            historical_accuracy=0.6,
            trend_direction="",
            total_latency_ms=600.0,
        )

        # Lower reward due to failures
        reward = signal.compute_reward()
        assert reward < 0.7  # Should be lower due to failures
