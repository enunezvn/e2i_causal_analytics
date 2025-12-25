"""
Tests for Gap Analyzer Agent DSPy Integration.

Tests the DSPy Sender role implementation including:
- Training signal dataclass
- Reward computation
- Signal collector
- Singleton access patterns

Note: This module is marked to run sequentially (not in parallel) because
the dspy import has race conditions during parallel pytest-xdist execution.
"""

import pytest
from datetime import datetime, timezone

# Mark entire module to run on same worker - prevents import race conditions
pytestmark = pytest.mark.xdist_group(name="dspy_integration")

from src.agents.gap_analyzer.dspy_integration import (
    GapAnalysisTrainingSignal,
    GapAnalyzerSignalCollector,
    get_gap_analyzer_signal_collector,
    reset_dspy_integration,
    DSPY_AVAILABLE,
)


class TestGapAnalysisTrainingSignal:
    """Tests for GapAnalysisTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with defaults."""
        signal = GapAnalysisTrainingSignal()

        assert signal.signal_id == ""
        assert signal.session_id == ""
        assert signal.query == ""
        assert signal.brand == ""
        assert signal.metrics_analyzed == []
        assert signal.segments_analyzed == 0
        assert signal.gaps_detected_count == 0
        assert signal.roi_estimates_count == 0
        assert signal.user_satisfaction is None
        assert signal.created_at is not None

    def test_custom_initialization(self):
        """Test signal with custom values."""
        signal = GapAnalysisTrainingSignal(
            session_id="gap-session-123",
            query="Find gaps in Remibrutinib performance",
            brand="Remibrutinib",
            metrics_analyzed=["TRx", "NRx", "market_share"],
            segments_analyzed=10,
        )

        assert signal.session_id == "gap-session-123"
        assert signal.brand == "Remibrutinib"
        assert len(signal.metrics_analyzed) == 3
        assert signal.segments_analyzed == 10

    def test_compute_reward_minimal(self):
        """Test reward with minimal data."""
        signal = GapAnalysisTrainingSignal()
        reward = signal.compute_reward()

        assert 0.0 <= reward <= 1.0
        assert reward < 0.5  # Minimal data should yield low reward

    def test_compute_reward_high_quality(self):
        """Test reward with high-quality gap analysis."""
        signal = GapAnalysisTrainingSignal(
            segments_analyzed=10,
            gaps_detected_count=20,  # 2 gaps per segment (ideal)
            gap_types=["vs_target", "vs_benchmark"],
            roi_estimates_count=15,
            avg_expected_roi=2.5,  # Good ROI
            high_roi_count=8,
            quick_wins_count=4,
            strategic_bets_count=3,
            key_insights_count=5,
            total_latency_ms=6000,  # Under 8s target
            user_satisfaction=5.0,
        )
        reward = signal.compute_reward()

        assert reward > 0.7  # High-quality should score well
        assert reward <= 1.0

    def test_compute_reward_gap_ratio_impact(self):
        """Test that gap ratio affects reward."""
        # Ideal ratio: 1-3 gaps per segment
        signal_ideal = GapAnalysisTrainingSignal(
            segments_analyzed=10,
            gaps_detected_count=20,  # 2 per segment
        )

        # Too many gaps per segment
        signal_excessive = GapAnalysisTrainingSignal(
            segments_analyzed=10,
            gaps_detected_count=100,  # 10 per segment
        )

        assert signal_ideal.compute_reward() > signal_excessive.compute_reward()

    def test_compute_reward_roi_quality(self):
        """Test that ROI quality impacts reward."""
        # High ROI
        signal_high_roi = GapAnalysisTrainingSignal(
            roi_estimates_count=10,
            avg_expected_roi=3.0,
            high_roi_count=8,
        )

        # Low ROI
        signal_low_roi = GapAnalysisTrainingSignal(
            roi_estimates_count=10,
            avg_expected_roi=0.5,
            high_roi_count=1,
        )

        assert signal_high_roi.compute_reward() > signal_low_roi.compute_reward()

    def test_compute_reward_prioritization_impact(self):
        """Test that prioritization quality impacts reward."""
        # Good prioritization
        signal_good = GapAnalysisTrainingSignal(
            gaps_detected_count=10,
            quick_wins_count=3,
            strategic_bets_count=3,
            key_insights_count=5,
        )

        # Poor prioritization
        signal_poor = GapAnalysisTrainingSignal(
            gaps_detected_count=10,
            quick_wins_count=0,
            strategic_bets_count=0,
            key_insights_count=0,
        )

        assert signal_good.compute_reward() > signal_poor.compute_reward()

    def test_compute_reward_latency_impact(self):
        """Test that latency affects reward."""
        # Fast analysis
        signal_fast = GapAnalysisTrainingSignal(total_latency_ms=4000)

        # Slow analysis
        signal_slow = GapAnalysisTrainingSignal(total_latency_ms=30000)

        assert signal_fast.compute_reward() > signal_slow.compute_reward()

    def test_compute_reward_implementation_proxy(self):
        """Test that recommendations_implemented works as satisfaction proxy."""
        signal = GapAnalysisTrainingSignal(
            actionable_recommendations=5,
            recommendations_implemented=4,  # 80% implemented
        )
        reward = signal.compute_reward()

        # Implementation rate should contribute to reward
        assert reward > GapAnalysisTrainingSignal().compute_reward()

    def test_to_dict_structure(self):
        """Test dictionary serialization structure."""
        signal = GapAnalysisTrainingSignal(
            session_id="sess-123",
            brand="Kisqali",
            gaps_detected_count=5,
            avg_expected_roi=2.0,
        )

        result = signal.to_dict()

        assert result["source_agent"] == "gap_analyzer"
        assert result["dspy_type"] == "sender"
        assert "input_context" in result
        assert "detection" in result
        assert "roi" in result
        assert "prioritization" in result
        assert "output" in result
        assert "outcome" in result
        assert "reward" in result

    def test_to_dict_query_truncation(self):
        """Test long query truncation."""
        signal = GapAnalysisTrainingSignal(query="A" * 600)
        result = signal.to_dict()

        assert len(result["input_context"]["query"]) <= 500

    def test_to_dict_metrics_limit(self):
        """Test metrics list limiting."""
        signal = GapAnalysisTrainingSignal(
            metrics_analyzed=[f"metric_{i}" for i in range(15)]
        )
        result = signal.to_dict()

        assert len(result["input_context"]["metrics_analyzed"]) <= 10


class TestGapAnalyzerSignalCollector:
    """Tests for GapAnalyzerSignalCollector."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_dspy_integration()
        yield
        reset_dspy_integration()

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = GapAnalyzerSignalCollector()

        assert collector.dspy_type == "sender"
        assert collector._signals_buffer == []
        assert collector._buffer_limit == 100

    def test_collect_analysis_signal(self):
        """Test signal collection initiation."""
        collector = GapAnalyzerSignalCollector()

        signal = collector.collect_analysis_signal(
            session_id="gap-sess-123",
            query="Find ROI opportunities",
            brand="Fabhalta",
            metrics_analyzed=["TRx", "NRx"],
            segments_analyzed=15,
        )

        assert isinstance(signal, GapAnalysisTrainingSignal)
        assert signal.session_id == "gap-sess-123"
        assert signal.brand == "Fabhalta"
        assert signal.segments_analyzed == 15

    def test_update_detection(self):
        """Test detection phase update."""
        collector = GapAnalyzerSignalCollector()
        signal = GapAnalysisTrainingSignal()

        updated = collector.update_detection(
            signal,
            gaps_detected_count=8,
            total_gap_value=150000.0,
            gap_types=["vs_target", "vs_benchmark", "vs_prior"],
            detection_latency_ms=2500,
        )

        assert updated.gaps_detected_count == 8
        assert updated.total_gap_value == 150000.0
        assert len(updated.gap_types) == 3
        assert updated.detection_latency_ms == 2500

    def test_update_roi(self):
        """Test ROI phase update."""
        collector = GapAnalyzerSignalCollector()
        signal = GapAnalysisTrainingSignal()

        updated = collector.update_roi(
            signal,
            roi_estimates_count=8,
            total_addressable_value=500000.0,
            avg_expected_roi=2.3,
            high_roi_count=5,
            roi_latency_ms=3000,
        )

        assert updated.roi_estimates_count == 8
        assert updated.total_addressable_value == 500000.0
        assert updated.avg_expected_roi == 2.3
        assert updated.high_roi_count == 5
        assert updated.roi_latency_ms == 3000

    def test_update_prioritization_adds_to_buffer(self):
        """Test prioritization update adds signal to buffer."""
        collector = GapAnalyzerSignalCollector()
        signal = GapAnalysisTrainingSignal()

        assert len(collector._signals_buffer) == 0

        collector.update_prioritization(
            signal,
            quick_wins_count=3,
            strategic_bets_count=2,
            prioritization_confidence=0.85,
            executive_summary_length=250,
            key_insights_count=4,
            actionable_recommendations=5,
            total_latency_ms=7000,
        )

        assert len(collector._signals_buffer) == 1
        assert signal.quick_wins_count == 3
        assert signal.strategic_bets_count == 2

    def test_buffer_limit_enforcement(self):
        """Test buffer respects limit."""
        collector = GapAnalyzerSignalCollector()
        collector._buffer_limit = 5

        for i in range(7):
            signal = GapAnalysisTrainingSignal(session_id=f"gap-{i}")
            collector.update_prioritization(
                signal,
                quick_wins_count=1,
                strategic_bets_count=1,
                prioritization_confidence=0.5,
                executive_summary_length=100,
                key_insights_count=2,
                actionable_recommendations=2,
                total_latency_ms=5000,
            )

        assert len(collector._signals_buffer) == 5
        assert collector._signals_buffer[0].session_id == "gap-2"

    def test_update_with_feedback(self):
        """Test delayed feedback update."""
        collector = GapAnalyzerSignalCollector()
        signal = GapAnalysisTrainingSignal()

        updated = collector.update_with_feedback(
            signal,
            user_satisfaction=4.0,
            recommendations_implemented=3,
        )

        assert updated.user_satisfaction == 4.0
        assert updated.recommendations_implemented == 3

    def test_get_signals_for_training(self):
        """Test signal retrieval for training."""
        collector = GapAnalyzerSignalCollector()

        for i, roi in enumerate([2.5, 0.5, 3.0]):
            signal = GapAnalysisTrainingSignal(
                session_id=f"gap-{i}",
                avg_expected_roi=roi,
                high_roi_count=5 if roi > 2 else 1,
            )
            collector.update_prioritization(
                signal,
                quick_wins_count=3,
                strategic_bets_count=2,
                prioritization_confidence=0.8,
                executive_summary_length=200,
                key_insights_count=4,
                actionable_recommendations=4,
                total_latency_ms=6000,
            )

        all_signals = collector.get_signals_for_training(min_reward=0.0)
        assert len(all_signals) == 3

    def test_get_high_roi_examples(self):
        """Test retrieval of high ROI examples only."""
        collector = GapAnalyzerSignalCollector()

        for i, roi in enumerate([1.5, 2.5, 3.0, 1.0]):
            signal = GapAnalysisTrainingSignal(
                session_id=f"gap-{i}",
                avg_expected_roi=roi,
            )
            collector.update_prioritization(
                signal,
                quick_wins_count=2,
                strategic_bets_count=2,
                prioritization_confidence=0.7,
                executive_summary_length=150,
                key_insights_count=3,
                actionable_recommendations=3,
                total_latency_ms=6000,
            )

        high_roi = collector.get_high_roi_examples(min_avg_roi=2.0, limit=10)

        assert len(high_roi) == 2
        for ex in high_roi:
            assert ex["roi"]["avg_expected_roi"] >= 2.0

    def test_clear_buffer(self):
        """Test buffer clearing."""
        collector = GapAnalyzerSignalCollector()

        signal = GapAnalysisTrainingSignal()
        collector.update_prioritization(
            signal,
            quick_wins_count=1,
            strategic_bets_count=1,
            prioritization_confidence=0.5,
            executive_summary_length=100,
            key_insights_count=2,
            actionable_recommendations=2,
            total_latency_ms=5000,
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
        collector1 = get_gap_analyzer_signal_collector()
        collector2 = get_gap_analyzer_signal_collector()

        assert collector1 is collector2

    def test_reset_dspy_integration(self):
        """Test singleton reset."""
        collector1 = get_gap_analyzer_signal_collector()
        reset_dspy_integration()
        collector2 = get_gap_analyzer_signal_collector()

        assert collector1 is not collector2


class TestDSPySignatures:
    """Tests for DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE reflects actual availability."""
        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_gap_detection_signature(self):
        """Test GapDetectionSignature exists."""
        from src.agents.gap_analyzer.dspy_integration import GapDetectionSignature

        import dspy
        assert issubclass(GapDetectionSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_evidence_relevance_signature(self):
        """Test EvidenceRelevanceSignature exists."""
        from src.agents.gap_analyzer.dspy_integration import EvidenceRelevanceSignature

        import dspy
        assert issubclass(EvidenceRelevanceSignature, dspy.Signature)

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
    def test_gap_prioritization_signature(self):
        """Test GapPrioritizationSignature exists."""
        from src.agents.gap_analyzer.dspy_integration import GapPrioritizationSignature

        import dspy
        assert issubclass(GapPrioritizationSignature, dspy.Signature)


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
        collector = get_gap_analyzer_signal_collector()

        # Phase 1: Initialize
        signal = collector.collect_analysis_signal(
            session_id="lifecycle-gap-test",
            query="Find gaps in Northeast region performance",
            brand="Remibrutinib",
            metrics_analyzed=["TRx", "NRx", "market_share", "conversion"],
            segments_analyzed=12,
        )

        # Phase 2: Detection
        signal = collector.update_detection(
            signal,
            gaps_detected_count=18,
            total_gap_value=200000.0,
            gap_types=["vs_target", "vs_benchmark"],
            detection_latency_ms=2000,
        )

        # Phase 3: ROI
        signal = collector.update_roi(
            signal,
            roi_estimates_count=15,
            total_addressable_value=750000.0,
            avg_expected_roi=2.8,
            high_roi_count=9,
            roi_latency_ms=2500,
        )

        # Phase 4: Prioritization (adds to buffer)
        signal = collector.update_prioritization(
            signal,
            quick_wins_count=4,
            strategic_bets_count=3,
            prioritization_confidence=0.88,
            executive_summary_length=320,
            key_insights_count=5,
            actionable_recommendations=6,
            total_latency_ms=6500,
        )

        # Phase 5: Delayed feedback
        signal = collector.update_with_feedback(
            signal,
            user_satisfaction=4.5,
            recommendations_implemented=4,
        )

        # Verify final state
        assert signal.session_id == "lifecycle-gap-test"
        assert signal.gaps_detected_count == 18
        assert signal.avg_expected_roi == 2.8
        assert signal.quick_wins_count == 4
        assert signal.user_satisfaction == 4.5

        # Verify in buffer
        signals = collector.get_signals_for_training()
        assert len(signals) == 1

        # Verify reward is good
        reward = signal.compute_reward()
        assert reward > 0.5
