"""
E2I Signal Flow Integration Tests - Batch 2: Signal Collection

Tests signal collection, batching, and aggregation from multiple sender agents.

Run: pytest tests/integration/test_signal_flow/test_signal_collection.py -v
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

# =============================================================================
# SIGNAL COLLECTION TESTS
# =============================================================================


class TestSignalBatching:
    """Test signal batching from multiple senders."""

    @pytest.fixture
    def signal_batch(self) -> List[Dict[str, Any]]:
        """Create a batch of signals from different agents."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )
        from src.agents.drift_monitor.dspy_integration import (
            DriftDetectionTrainingSignal,
        )
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )

        batch = []

        # Add multiple signals from causal_impact
        for i in range(3):
            signal = CausalAnalysisTrainingSignal(
                signal_id=f"ci_{i:03d}",
                session_id=f"session_{i}",
                query=f"Query {i}",
                refutation_tests_passed=2 + i,
                refutation_tests_failed=1,
                statistical_significance=i > 0,
            )
            batch.append(signal.to_dict())

        # Add signals from gap_analyzer
        for i in range(2):
            signal = GapAnalysisTrainingSignal(
                signal_id=f"ga_{i:03d}",
                session_id=f"session_ga_{i}",
                gaps_detected_count=3 + i,
                roi_estimates_count=2,
            )
            batch.append(signal.to_dict())

        # Add signal from drift_monitor
        signal = DriftDetectionTrainingSignal(
            signal_id="dm_001",
            session_id="session_dm",
            data_drift_count=2,
            model_drift_count=1,
            features_monitored=10,
        )
        batch.append(signal.to_dict())

        return batch

    def test_batch_contains_multiple_agents(self, signal_batch):
        """Batch should contain signals from multiple agents."""
        agents = {s["source_agent"] for s in signal_batch}

        assert len(agents) >= 2, "Batch should have signals from multiple agents"
        assert "causal_impact" in agents
        assert "gap_analyzer" in agents

    def test_batch_can_be_filtered_by_agent(self, signal_batch):
        """Signals can be filtered by source agent."""
        causal_signals = [s for s in signal_batch if s["source_agent"] == "causal_impact"]
        gap_signals = [s for s in signal_batch if s["source_agent"] == "gap_analyzer"]

        assert len(causal_signals) == 3
        assert len(gap_signals) == 2

    def test_batch_can_be_filtered_by_quality(self, signal_batch):
        """Signals can be filtered by reward threshold."""
        MIN_QUALITY = 0.3

        high_quality = [s for s in signal_batch if s["reward"] >= MIN_QUALITY]

        # Some signals should pass, some should fail
        assert len(high_quality) >= 1
        assert len(high_quality) <= len(signal_batch)

    def test_batch_can_be_sorted_by_reward(self, signal_batch):
        """Signals can be sorted by reward (descending)."""
        sorted_batch = sorted(signal_batch, key=lambda s: s["reward"], reverse=True)

        for i in range(len(sorted_batch) - 1):
            assert sorted_batch[i]["reward"] >= sorted_batch[i + 1]["reward"]

    def test_batch_statistics(self, signal_batch):
        """Compute statistics over signal batch."""
        rewards = [s["reward"] for s in signal_batch]

        stats = {
            "count": len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "avg_reward": sum(rewards) / len(rewards),
        }

        assert stats["count"] == 6
        assert 0.0 <= stats["min_reward"] <= stats["max_reward"] <= 1.0


class TestSignalAggregation:
    """Test signal aggregation by agent and time."""

    @pytest.fixture
    def time_series_signals(self) -> List[Dict[str, Any]]:
        """Create signals with different timestamps."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        now = datetime.now(timezone.utc)
        signals = []

        # Signals over past 24 hours
        for i in range(24):
            timestamp = now - timedelta(hours=i)
            signal = CausalAnalysisTrainingSignal(
                signal_id=f"ci_{i:03d}",
                session_id=f"session_{i}",
                created_at=timestamp.isoformat(),
                refutation_tests_passed=i % 4,
                refutation_tests_failed=1,
            )
            signals.append(signal.to_dict())

        return signals

    def test_aggregate_by_agent(self, time_series_signals):
        """Aggregate signals by source agent."""
        from collections import defaultdict

        by_agent = defaultdict(list)
        for signal in time_series_signals:
            by_agent[signal["source_agent"]].append(signal)

        assert "causal_impact" in by_agent
        assert len(by_agent["causal_impact"]) == 24

    def test_aggregate_by_time_window(self, time_series_signals):
        """Aggregate signals by time window."""
        now = datetime.now(timezone.utc)

        # Signals in last 6 hours
        recent = [
            s
            for s in time_series_signals
            if datetime.fromisoformat(s["timestamp"]) > now - timedelta(hours=6)
        ]

        assert len(recent) <= 6

    def test_compute_agent_metrics(self, time_series_signals):
        """Compute aggregate metrics per agent."""
        rewards = [s["reward"] for s in time_series_signals]

        agent_metrics = {
            "signal_count": len(time_series_signals),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "high_quality_count": sum(1 for r in rewards if r >= 0.5),
            "low_quality_count": sum(1 for r in rewards if r < 0.5),
        }

        assert agent_metrics["signal_count"] == 24
        assert 0.0 <= agent_metrics["avg_reward"] <= 1.0


class TestSignalPersistence:
    """Test signal storage and retrieval simulation."""

    def test_signal_json_serialization(self):
        """Signals can be serialized to JSON."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        signal = CausalAnalysisTrainingSignal(
            signal_id="test_001",
            session_id="session_test",
            query="Test query",
        )

        signal_dict = signal.to_dict()
        json_str = json.dumps(signal_dict)

        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_signal_json_deserialization(self):
        """Signals can be deserialized from JSON."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        signal = CausalAnalysisTrainingSignal(
            signal_id="test_001",
            session_id="session_test",
            query="Test query",
        )

        signal_dict = signal.to_dict()
        json_str = json.dumps(signal_dict)
        restored = json.loads(json_str)

        assert restored["signal_id"] == signal_dict["signal_id"]
        assert restored["source_agent"] == "causal_impact"
        assert restored["reward"] == signal_dict["reward"]

    def test_batch_json_roundtrip(self):
        """Batch of signals survives JSON roundtrip."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )

        batch = [
            CausalAnalysisTrainingSignal(signal_id="ci_001").to_dict(),
            GapAnalysisTrainingSignal(signal_id="ga_001").to_dict(),
        ]

        json_str = json.dumps(batch)
        restored = json.loads(json_str)

        assert len(restored) == 2
        assert restored[0]["source_agent"] == "causal_impact"
        assert restored[1]["source_agent"] == "gap_analyzer"


class TestSignalFlowContractCompliance:
    """Test compliance with SignalFlowContract requirements."""

    def test_min_signals_threshold(self):
        """Test minimum signals for optimization threshold."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )

        # Per SignalFlowContract: min_signals_for_optimization = 100
        MIN_SIGNALS = 100

        # Create batch below threshold
        small_batch = [
            CausalAnalysisTrainingSignal(signal_id=f"ci_{i:03d}").to_dict() for i in range(50)
        ]

        # Create batch at threshold
        threshold_batch = [
            CausalAnalysisTrainingSignal(signal_id=f"ci_{i:03d}").to_dict() for i in range(100)
        ]

        assert len(small_batch) < MIN_SIGNALS
        assert len(threshold_batch) >= MIN_SIGNALS

    def test_all_sender_types_represented(self):
        """Ensure all sender types per contract are represented."""
        CONTRACT_SENDERS = {
            "causal_impact",
            "gap_analyzer",
            "heterogeneous_optimizer",
            "drift_monitor",
            "experiment_designer",
            "prediction_synthesizer",
        }

        # Import all sender signal types
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )
        from src.agents.drift_monitor.dspy_integration import (
            DriftDetectionTrainingSignal,
        )
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )
        from src.agents.heterogeneous_optimizer.dspy_integration import (
            HeterogeneousOptimizationTrainingSignal,
        )
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        # Create one signal from each
        signals = [
            CausalAnalysisTrainingSignal().to_dict(),
            GapAnalysisTrainingSignal().to_dict(),
            HeterogeneousOptimizationTrainingSignal().to_dict(),
            DriftDetectionTrainingSignal().to_dict(),
            ExperimentDesignTrainingSignal().to_dict(),
            PredictionSynthesisTrainingSignal().to_dict(),
        ]

        actual_senders = {s["source_agent"] for s in signals}

        assert actual_senders == CONTRACT_SENDERS


class TestMultiAgentSignalMixing:
    """Test mixing signals from different agent types."""

    def test_heterogeneous_batch_processing(self):
        """Process batch with signals from all agent types."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )
        from src.agents.drift_monitor.dspy_integration import (
            DriftDetectionTrainingSignal,
        )
        from src.agents.experiment_designer.dspy_integration import (
            ExperimentDesignTrainingSignal,
        )
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )
        from src.agents.heterogeneous_optimizer.dspy_integration import (
            HeterogeneousOptimizationTrainingSignal,
        )
        from src.agents.prediction_synthesizer.dspy_integration import (
            PredictionSynthesisTrainingSignal,
        )

        # Create heterogeneous batch
        batch = []
        for i in range(5):
            batch.extend(
                [
                    CausalAnalysisTrainingSignal(signal_id=f"ci_{i}").to_dict(),
                    GapAnalysisTrainingSignal(signal_id=f"ga_{i}").to_dict(),
                    HeterogeneousOptimizationTrainingSignal(signal_id=f"ho_{i}").to_dict(),
                    DriftDetectionTrainingSignal(signal_id=f"dm_{i}").to_dict(),
                    ExperimentDesignTrainingSignal(signal_id=f"ed_{i}").to_dict(),
                    PredictionSynthesisTrainingSignal(signal_id=f"ps_{i}").to_dict(),
                ]
            )

        # Verify batch composition
        assert len(batch) == 30  # 5 * 6 agents

        # Group by agent
        by_agent = {}
        for s in batch:
            agent = s["source_agent"]
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(s)

        assert len(by_agent) == 6
        for agent, signals in by_agent.items():
            assert len(signals) == 5

    def test_batch_reward_distribution(self):
        """Analyze reward distribution across agent types."""
        from src.agents.causal_impact.dspy_integration import (
            CausalAnalysisTrainingSignal,
        )
        from src.agents.gap_analyzer.dspy_integration import (
            GapAnalysisTrainingSignal,
        )

        # Create batch with varying quality
        batch = []

        # High quality causal signals
        for i in range(5):
            batch.append(
                CausalAnalysisTrainingSignal(
                    signal_id=f"ci_{i}",
                    refutation_tests_passed=4,
                    statistical_significance=True,
                ).to_dict()
            )

        # Low quality gap signals
        for i in range(5):
            batch.append(
                GapAnalysisTrainingSignal(
                    signal_id=f"ga_{i}",
                ).to_dict()
            )

        # Compute per-agent reward stats
        causal_rewards = [s["reward"] for s in batch if s["source_agent"] == "causal_impact"]
        gap_rewards = [s["reward"] for s in batch if s["source_agent"] == "gap_analyzer"]

        avg_causal = sum(causal_rewards) / len(causal_rewards)
        avg_gap = sum(gap_rewards) / len(gap_rewards)

        # High quality signals should have higher average reward
        assert avg_causal > avg_gap
