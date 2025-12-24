"""
Tests for Drift Monitor Agent DSPy Integration.

Tests the DSPy Sender role implementation including:
- Training signal dataclass
- Reward computation
- Signal collector
- Singleton access patterns
"""

import pytest
from datetime import datetime, timezone

from src.agents.drift_monitor.dspy_integration import (
    DriftDetectionTrainingSignal,
    DriftMonitorSignalCollector,
    get_drift_monitor_signal_collector,
    reset_dspy_integration,
    DSPY_AVAILABLE,
)


class TestDriftDetectionTrainingSignal:
    """Tests for DriftDetectionTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with defaults."""
        signal = DriftDetectionTrainingSignal()

        assert signal.signal_id == ""
        assert signal.session_id == ""
        assert signal.query == ""
        assert signal.model_id == ""
        assert signal.features_monitored == 0
        assert signal.time_window == ""
        assert signal.check_data_drift is True
        assert signal.check_model_drift is False
        assert signal.check_concept_drift is False
        assert signal.psi_threshold == 0.1
        assert signal.significance_level == 0.05
        assert signal.user_satisfaction is None
        assert signal.created_at is not None

    def test_custom_initialization(self):
        """Test signal with custom values."""
        signal = DriftDetectionTrainingSignal(
            session_id="drift-session-123",
            query="Check model drift for Remibrutinib predictor",
            model_id="model-v2.1",
            features_monitored=50,
            time_window="30d",
            check_model_drift=True,
            check_concept_drift=True,
        )

        assert signal.session_id == "drift-session-123"
        assert signal.model_id == "model-v2.1"
        assert signal.features_monitored == 50
        assert signal.check_model_drift is True
        assert signal.check_concept_drift is True

    def test_compute_reward_minimal(self):
        """Test reward with minimal data."""
        signal = DriftDetectionTrainingSignal()
        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0
        assert reward < 0.5  # Minimal data should yield low reward

    def test_compute_reward_high_quality(self):
        """Test reward with high-quality drift detection."""
        signal = DriftDetectionTrainingSignal(
            features_monitored=100,
            data_drift_count=10,  # 10% drift rate (ideal)
            model_drift_count=2,
            overall_drift_score=0.15,
            severity_distribution={"low": 5, "medium": 4, "high": 2, "critical": 1},
            alerts_generated=12,
            critical_alerts=2,  # ~17% critical (in ideal range)
            warnings=8,
            recommended_actions_count=10,
            features_checked=100,
            total_latency_ms=15000,  # 150ms per feature
            user_satisfaction=5.0,
            drift_correctly_identified=True,
        )
        reward = signal.compute_reward()

        assert reward > 0.7  # High-quality should score well
        assert reward <= 1.0

    def test_compute_reward_drift_rate_impact(self):
        """Test that drift rate affects reward."""
        # Ideal drift rate: 5-20%
        signal_ideal = DriftDetectionTrainingSignal(
            features_monitored=100,
            data_drift_count=10,  # 10% drift rate
        )

        # Too few drifts
        signal_low = DriftDetectionTrainingSignal(
            features_monitored=100,
            data_drift_count=2,  # 2% drift rate
        )

        # Too many drifts
        signal_high = DriftDetectionTrainingSignal(
            features_monitored=100,
            data_drift_count=50,  # 50% drift rate
        )

        # Ideal should score better than extremes
        assert signal_ideal.compute_reward() >= signal_low.compute_reward()
        assert signal_ideal.compute_reward() >= signal_high.compute_reward()

    def test_compute_reward_alerting_quality(self):
        """Test that alerting quality impacts reward."""
        # Good alerting
        signal_good = DriftDetectionTrainingSignal(
            data_drift_count=10,
            alerts_generated=10,  # 100% alert rate
            critical_alerts=2,  # 20% critical (in range)
        )

        # Poor alerting
        signal_poor = DriftDetectionTrainingSignal(
            data_drift_count=10,
            alerts_generated=2,  # 20% alert rate
            critical_alerts=0,
        )

        assert signal_good.compute_reward() > signal_poor.compute_reward()

    def test_compute_reward_efficiency_impact(self):
        """Test that efficiency affects reward."""
        # Fast detection
        signal_fast = DriftDetectionTrainingSignal(
            features_checked=100,
            total_latency_ms=10000,  # 100ms per feature
        )

        # Slow detection
        signal_slow = DriftDetectionTrainingSignal(
            features_checked=100,
            total_latency_ms=100000,  # 1000ms per feature
        )

        assert signal_fast.compute_reward() > signal_slow.compute_reward()

    def test_compute_reward_actionability_impact(self):
        """Test that actionability affects reward."""
        # Actionable recommendations
        signal_actionable = DriftDetectionTrainingSignal(
            alerts_generated=10,
            recommended_actions_count=10,  # 100% actionable
        )

        # No recommendations
        signal_no_action = DriftDetectionTrainingSignal(
            alerts_generated=10,
            recommended_actions_count=0,
        )

        assert signal_actionable.compute_reward() > signal_no_action.compute_reward()

    def test_compute_reward_validation_impact(self):
        """Test that validated drift detection impacts reward."""
        signal_correct = DriftDetectionTrainingSignal(
            drift_correctly_identified=True,
        )

        signal_incorrect = DriftDetectionTrainingSignal(
            drift_correctly_identified=False,
        )

        assert signal_correct.compute_reward() > signal_incorrect.compute_reward()

    def test_to_dict_structure(self):
        """Test dictionary serialization structure."""
        signal = DriftDetectionTrainingSignal(
            session_id="sess-123",
            model_id="model-v1",
            data_drift_count=5,
            overall_drift_score=0.2,
        )

        result = signal.to_dict()

        assert result["source_agent"] == "drift_monitor"
        assert result["dspy_type"] == "sender"
        assert "input_context" in result
        assert "configuration" in result
        assert "detection_results" in result
        assert "alerting" in result
        assert "outcome" in result
        assert "reward" in result

    def test_to_dict_query_truncation(self):
        """Test long query truncation."""
        signal = DriftDetectionTrainingSignal(query="A" * 600)
        result = signal.to_dict()

        assert len(result["input_context"]["query"]) <= 500

    def test_to_dict_severity_distribution(self):
        """Test severity distribution serialization."""
        signal = DriftDetectionTrainingSignal(
            severity_distribution={"low": 5, "medium": 3, "high": 2}
        )
        result = signal.to_dict()

        assert result["detection_results"]["severity_distribution"] == {
            "low": 5,
            "medium": 3,
            "high": 2,
        }


class TestDriftMonitorSignalCollector:
    """Tests for DriftMonitorSignalCollector."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = DriftMonitorSignalCollector()

        assert collector.dspy_type == "sender"
        assert collector._signals_buffer == []
        assert collector._buffer_limit == 100

    def test_collect_detection_signal(self):
        """Test signal collection initiation."""
        collector = DriftMonitorSignalCollector()

        signal = collector.collect_detection_signal(
            session_id="drift-sess-123",
            query="Monitor feature drift",
            model_id="churn-model-v2",
            features_monitored=50,
            time_window="7d",
            check_data_drift=True,
            check_model_drift=True,
            check_concept_drift=False,
        )

        assert isinstance(signal, DriftDetectionTrainingSignal)
        assert signal.session_id == "drift-sess-123"
        assert signal.model_id == "churn-model-v2"
        assert signal.features_monitored == 50
        assert signal.check_model_drift is True

    def test_update_detection_results(self):
        """Test detection results update."""
        collector = DriftMonitorSignalCollector()
        signal = DriftDetectionTrainingSignal()

        updated = collector.update_detection_results(
            signal,
            data_drift_count=8,
            model_drift_count=2,
            concept_drift_count=1,
            overall_drift_score=0.25,
            severity_distribution={"low": 4, "medium": 3, "high": 2, "critical": 2},
            features_checked=100,
        )

        assert updated.data_drift_count == 8
        assert updated.model_drift_count == 2
        assert updated.concept_drift_count == 1
        assert updated.overall_drift_score == 0.25
        assert updated.features_checked == 100

    def test_update_alerting_adds_to_buffer(self):
        """Test alerting update adds signal to buffer."""
        collector = DriftMonitorSignalCollector()
        signal = DriftDetectionTrainingSignal()

        assert len(collector._signals_buffer) == 0

        collector.update_alerting(
            signal,
            alerts_generated=5,
            critical_alerts=1,
            warnings=3,
            recommended_actions_count=4,
            total_latency_ms=5000,
        )

        assert len(collector._signals_buffer) == 1
        assert signal.alerts_generated == 5
        assert signal.critical_alerts == 1

    def test_buffer_limit_enforcement(self):
        """Test buffer respects limit."""
        collector = DriftMonitorSignalCollector()
        collector._buffer_limit = 5

        for i in range(7):
            signal = DriftDetectionTrainingSignal(session_id=f"drift-{i}")
            collector.update_alerting(
                signal,
                alerts_generated=2,
                critical_alerts=0,
                warnings=2,
                recommended_actions_count=1,
                total_latency_ms=3000,
            )

        assert len(collector._signals_buffer) == 5
        assert collector._signals_buffer[0].session_id == "drift-2"

    def test_update_with_validation(self):
        """Test delayed validation update."""
        collector = DriftMonitorSignalCollector()
        signal = DriftDetectionTrainingSignal()

        updated = collector.update_with_validation(
            signal,
            drift_correctly_identified=True,
            user_satisfaction=4.5,
        )

        assert updated.drift_correctly_identified is True
        assert updated.user_satisfaction == 4.5

    def test_get_signals_for_training(self):
        """Test signal retrieval for training."""
        collector = DriftMonitorSignalCollector()

        for i, drift_count in enumerate([10, 2, 15]):
            signal = DriftDetectionTrainingSignal(
                session_id=f"drift-{i}",
                features_monitored=100,
                data_drift_count=drift_count,
            )
            collector.update_alerting(
                signal,
                alerts_generated=drift_count,
                critical_alerts=1,
                warnings=drift_count - 1,
                recommended_actions_count=drift_count,
                total_latency_ms=5000,
            )

        all_signals = collector.get_signals_for_training(min_reward=0.0)
        assert len(all_signals) == 3

    def test_get_validated_examples(self):
        """Test retrieval of validated examples only."""
        collector = DriftMonitorSignalCollector()

        for i, correct in enumerate([True, False, True, None]):
            signal = DriftDetectionTrainingSignal(session_id=f"drift-{i}")
            collector.update_alerting(
                signal,
                alerts_generated=2,
                critical_alerts=0,
                warnings=2,
                recommended_actions_count=1,
                total_latency_ms=3000,
            )
            if correct is not None:
                collector.update_with_validation(signal, drift_correctly_identified=correct)

        validated = collector.get_validated_examples(limit=10)

        # Only correctly identified signals (2 with True)
        assert len(validated) == 2
        for ex in validated:
            assert ex["outcome"]["drift_correctly_identified"] is True

    def test_clear_buffer(self):
        """Test buffer clearing."""
        collector = DriftMonitorSignalCollector()

        signal = DriftDetectionTrainingSignal()
        collector.update_alerting(
            signal,
            alerts_generated=1,
            critical_alerts=0,
            warnings=1,
            recommended_actions_count=1,
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
        collector1 = get_drift_monitor_signal_collector()
        collector2 = get_drift_monitor_signal_collector()

        assert collector1 is collector2

    def test_reset_dspy_integration(self):
        """Test singleton reset."""
        collector1 = get_drift_monitor_signal_collector()
        reset_dspy_integration()
        collector2 = get_drift_monitor_signal_collector()

        assert collector1 is not collector2


class TestDSPySignatures:
    """Tests for DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE reflects actual availability."""
        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_drift_detection_signature(self):
        """Test DriftDetectionSignature exists."""
        from src.agents.drift_monitor.dspy_integration import DriftDetectionSignature

        import dspy
        assert issubclass(DriftDetectionSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_hop_decision_signature(self):
        """Test HopDecisionSignature exists."""
        from src.agents.drift_monitor.dspy_integration import HopDecisionSignature

        import dspy
        assert issubclass(HopDecisionSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_drift_interpretation_signature(self):
        """Test DriftInterpretationSignature exists."""
        from src.agents.drift_monitor.dspy_integration import DriftInterpretationSignature

        import dspy
        assert issubclass(DriftInterpretationSignature, dspy.Signature)


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
        collector = get_drift_monitor_signal_collector()

        # Phase 1: Initialize
        signal = collector.collect_detection_signal(
            session_id="lifecycle-drift-test",
            query="Monitor drift in Northeast region model",
            model_id="region-model-v3",
            features_monitored=75,
            time_window="14d",
            check_data_drift=True,
            check_model_drift=True,
            check_concept_drift=True,
        )

        # Phase 2: Detection results
        signal = collector.update_detection_results(
            signal,
            data_drift_count=8,
            model_drift_count=2,
            concept_drift_count=1,
            overall_drift_score=0.18,
            severity_distribution={"low": 4, "medium": 3, "high": 2, "critical": 2},
            features_checked=75,
        )

        # Phase 3: Alerting (adds to buffer)
        signal = collector.update_alerting(
            signal,
            alerts_generated=11,
            critical_alerts=2,
            warnings=6,
            recommended_actions_count=8,
            total_latency_ms=12000,
        )

        # Phase 4: Delayed validation
        signal = collector.update_with_validation(
            signal,
            drift_correctly_identified=True,
            user_satisfaction=4.0,
        )

        # Verify final state
        assert signal.session_id == "lifecycle-drift-test"
        assert signal.data_drift_count == 8
        assert signal.model_drift_count == 2
        assert signal.alerts_generated == 11
        assert signal.drift_correctly_identified is True

        # Verify in buffer
        signals = collector.get_signals_for_training()
        assert len(signals) == 1

        # Verify reward is good
        reward = signal.compute_reward()
        assert reward > 0.5
