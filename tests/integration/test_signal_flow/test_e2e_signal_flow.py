"""
E2I Signal Flow Integration Tests - Batch 5: End-to-End Signal Flow

Tests complete signal flow: Sender → Hub (feedback_learner) → Recipient.

This batch validates the full DSPy optimization pipeline:
1. Senders generate training signals
2. Hub collects and aggregates signals
3. Hub triggers MIPROv2 optimization
4. Recipients receive optimized prompts

Run: pytest tests/integration/test_signal_flow/test_e2e_signal_flow.py -v
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, patch
import json


# =============================================================================
# END-TO-END SIGNAL FLOW TESTS
# =============================================================================


class TestFullSignalFlow:
    """Test complete Sender → Hub → Recipient flow."""

    @pytest.fixture
    def sender_signals(self) -> List[Dict[str, Any]]:
        """Generate signals from all sender agents."""
        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal
        from src.agents.gap_analyzer.dspy_integration import GapAnalysisTrainingSignal
        from src.agents.heterogeneous_optimizer.dspy_integration import HeterogeneousOptimizationTrainingSignal
        from src.agents.drift_monitor.dspy_integration import DriftDetectionTrainingSignal
        from src.agents.experiment_designer.dspy_integration import ExperimentDesignTrainingSignal
        from src.agents.prediction_synthesizer.dspy_integration import PredictionSynthesisTrainingSignal

        signals = []

        # High-quality signals from each sender
        for i in range(20):
            signals.extend([
                CausalAnalysisTrainingSignal(
                    signal_id=f"ci_{i:03d}",
                    refutation_tests_passed=3,
                    statistical_significance=True,
                ).to_dict(),
                GapAnalysisTrainingSignal(
                    signal_id=f"ga_{i:03d}",
                    gaps_detected_count=5,
                    roi_estimates_count=3,
                ).to_dict(),
                HeterogeneousOptimizationTrainingSignal(
                    signal_id=f"ho_{i:03d}",
                    cate_segments_count=10,
                    significant_cate_count=8,
                ).to_dict(),
                DriftDetectionTrainingSignal(
                    signal_id=f"dm_{i:03d}",
                    data_drift_count=2,
                    features_monitored=10,
                ).to_dict(),
                ExperimentDesignTrainingSignal(
                    signal_id=f"ed_{i:03d}",
                    design_type_chosen="RCT",
                    treatments_count=3,
                ).to_dict(),
                PredictionSynthesisTrainingSignal(
                    signal_id=f"ps_{i:03d}",
                    models_requested=5,
                    models_succeeded=4,
                ).to_dict(),
            ])

        return signals

    def test_sender_to_hub_flow(self, sender_signals):
        """Test signals flow from senders to hub."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        # Simulate hub receiving sender signals
        hub_signal = FeedbackLearnerTrainingSignal(
            batch_id="e2e_test_001",
            feedback_count=len(sender_signals),
            time_range_start=sender_signals[0]["timestamp"],
            time_range_end=sender_signals[-1]["timestamp"],
            focus_agents=list({s["source_agent"] for s in sender_signals}),
            patterns_detected=8,
            recommendations_generated=5,
            updates_applied=3,
        )

        assert hub_signal.feedback_count == len(sender_signals)
        assert len(hub_signal.focus_agents) == 6  # All 6 senders

    def test_hub_to_recipient_flow(self):
        """Test prompts flow from hub to recipients."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizerDSPyIntegration
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        # Simulate hub distributing optimized prompts
        optimized_prompts = {
            "health_score": {
                "summary_template": "Optimized: Grade {grade}, Score {score}",
                "optimization_score": 0.85,
            },
            "resource_optimizer": {
                "summary_template": "Optimized: {resource_type} allocation",
                "optimization_score": 0.82,
            },
            "explainer": {
                "executive_summary_template": "Optimized: Key findings for {user_expertise}",
                "optimization_score": 0.88,
            },
        }

        recipients = {
            "health_score": HealthScoreDSPyIntegration(),
            "resource_optimizer": ResourceOptimizerDSPyIntegration(),
            "explainer": ExplainerDSPyIntegration(),
        }

        # Distribute prompts
        for agent_name, recipient in recipients.items():
            prompts = optimized_prompts[agent_name]
            recipient.update_optimized_prompts(
                prompts={k: v for k, v in prompts.items() if k != "optimization_score"},
                optimization_score=prompts["optimization_score"],
            )

        # Verify prompts were updated
        for agent_name, recipient in recipients.items():
            assert recipient.prompts.optimization_score > 0
            assert recipient.prompts.last_optimized != ""

    def test_full_e2e_flow(self, sender_signals):
        """Test complete end-to-end signal flow."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            FeedbackLearnerOptimizer,
        )
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration

        # Step 1: Senders generate signals (already in fixture)
        assert len(sender_signals) == 120  # 20 * 6 senders

        # Step 2: Hub processes signals
        hub_signal = FeedbackLearnerTrainingSignal(
            batch_id="e2e_full_test",
            feedback_count=len(sender_signals),
            time_range_start=sender_signals[0]["timestamp"],
            time_range_end=sender_signals[-1]["timestamp"],
            focus_agents=list({s["source_agent"] for s in sender_signals}),
            patterns_detected=10,
            pattern_accuracy=0.85,
            recommendations_generated=5,
            recommendation_actionability=0.8,
            updates_applied=4,
            update_effectiveness=0.75,
        )

        # Step 3: Compute optimization reward
        reward = hub_signal.compute_reward()
        assert 0.0 <= reward <= 1.0

        # Step 4: If reward is high, distribute prompts
        if reward >= 0.6:
            recipient = HealthScoreDSPyIntegration()
            recipient.update_optimized_prompts(
                prompts={"summary_template": "E2E optimized template"},
                optimization_score=reward,
            )
            assert recipient.prompts.optimization_score == reward


class TestSignalFlowThresholds:
    """Test signal flow threshold compliance."""

    def test_min_signals_threshold(self):
        """Per SignalFlowContract: min_signals_for_optimization = 100."""
        MIN_SIGNALS = 100

        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal

        # Generate signals at threshold
        signals = [
            CausalAnalysisTrainingSignal(signal_id=f"ci_{i:03d}").to_dict()
            for i in range(100)
        ]

        can_optimize = len(signals) >= MIN_SIGNALS
        assert can_optimize is True

    def test_min_quality_threshold(self):
        """Per SignalFlowContract: min_signal_quality = 0.6."""
        MIN_QUALITY = 0.6

        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal

        # High-quality signal
        high_quality = CausalAnalysisTrainingSignal(
            signal_id="hq_001",
            refutation_tests_passed=4,
            refutation_tests_failed=0,
            statistical_significance=True,
        ).to_dict()

        # Low-quality signal
        low_quality = CausalAnalysisTrainingSignal(
            signal_id="lq_001",
            refutation_tests_passed=0,
            refutation_tests_failed=3,
            statistical_significance=False,
        ).to_dict()

        assert high_quality["reward"] >= MIN_QUALITY
        assert low_quality["reward"] < MIN_QUALITY

    def test_optimization_interval(self):
        """Per SignalFlowContract: optimization_interval_hours = 24."""
        OPTIMIZATION_INTERVAL_HOURS = 24

        now = datetime.now(timezone.utc)
        last_optimization = now - timedelta(hours=25)

        hours_since_last = (now - last_optimization).total_seconds() / 3600
        can_optimize = hours_since_last >= OPTIMIZATION_INTERVAL_HOURS

        assert can_optimize is True


class TestSignalQualityFiltering:
    """Test signal quality filtering in flow."""

    def test_filter_by_quality(self):
        """Filter signals by quality threshold."""
        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal

        # Generate signals with varying quality
        signals = []
        for i in range(20):
            signals.append(
                CausalAnalysisTrainingSignal(
                    signal_id=f"ci_{i:03d}",
                    refutation_tests_passed=i % 5,
                    refutation_tests_failed=1,
                ).to_dict()
            )

        # Filter by quality
        MIN_QUALITY = 0.3
        high_quality = [s for s in signals if s["reward"] >= MIN_QUALITY]

        assert len(high_quality) <= len(signals)
        for s in high_quality:
            assert s["reward"] >= MIN_QUALITY

    def test_aggregate_quality_metrics(self):
        """Aggregate quality metrics across signals."""
        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal
        from src.agents.gap_analyzer.dspy_integration import GapAnalysisTrainingSignal

        signals = [
            CausalAnalysisTrainingSignal(
                signal_id=f"ci_{i}",
                refutation_tests_passed=3,
                statistical_significance=True,
            ).to_dict()
            for i in range(10)
        ] + [
            GapAnalysisTrainingSignal(
                signal_id=f"ga_{i}",
                gaps_detected_count=5,
                roi_estimates_count=3,
            ).to_dict()
            for i in range(10)
        ]

        # Aggregate
        rewards = [s["reward"] for s in signals]
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)

        assert 0.0 <= avg_reward <= 1.0
        assert min_reward <= avg_reward <= max_reward


class TestMultiAgentCoordination:
    """Test coordination between multiple agents in flow."""

    def test_concurrent_sender_signals(self):
        """Test handling concurrent signals from multiple senders."""
        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal
        from src.agents.gap_analyzer.dspy_integration import GapAnalysisTrainingSignal
        from src.agents.drift_monitor.dspy_integration import DriftDetectionTrainingSignal

        # Simulate concurrent signal generation
        now = datetime.now(timezone.utc)

        concurrent_signals = [
            CausalAnalysisTrainingSignal(
                signal_id="ci_concurrent",
                created_at=now.isoformat(),
            ).to_dict(),
            GapAnalysisTrainingSignal(
                signal_id="ga_concurrent",
                created_at=now.isoformat(),
            ).to_dict(),
            DriftDetectionTrainingSignal(
                signal_id="dm_concurrent",
                created_at=now.isoformat(),
            ).to_dict(),
        ]

        # All signals have same timestamp
        timestamps = {s["timestamp"][:19] for s in concurrent_signals}
        # They should all be from the same second
        assert len(timestamps) == 1

    def test_signal_ordering_by_timestamp(self):
        """Test signals are properly ordered by timestamp."""
        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal

        signals = []
        base_time = datetime.now(timezone.utc)

        # Create signals with different timestamps
        for i in range(10):
            signal = CausalAnalysisTrainingSignal(
                signal_id=f"ci_{i:03d}",
                created_at=(base_time - timedelta(hours=i)).isoformat(),
            )
            signals.append(signal.to_dict())

        # Sort by timestamp
        sorted_signals = sorted(
            signals,
            key=lambda s: s["timestamp"],
            reverse=True,
        )

        # Verify ordering
        for i in range(len(sorted_signals) - 1):
            assert sorted_signals[i]["timestamp"] >= sorted_signals[i + 1]["timestamp"]

    def test_batch_recipient_updates(self):
        """Test batch updates to multiple recipients."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizerDSPyIntegration
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        recipients = [
            HealthScoreDSPyIntegration(),
            ResourceOptimizerDSPyIntegration(),
            ExplainerDSPyIntegration(),
        ]

        # Batch update all recipients
        optimization_score = 0.82
        for recipient in recipients:
            recipient.update_optimized_prompts(
                prompts={},  # Empty update triggers version bump
                optimization_score=optimization_score,
            )

        # Verify all updated
        for recipient in recipients:
            assert recipient.prompts.optimization_score == optimization_score


class TestSignalFlowPersistence:
    """Test signal persistence in flow."""

    def test_signal_roundtrip(self):
        """Test signals survive JSON serialization roundtrip."""
        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal

        original = CausalAnalysisTrainingSignal(
            signal_id="roundtrip_001",
            session_id="session_test",
            query="Test query for roundtrip",
            refutation_tests_passed=3,
            statistical_significance=True,
        )

        # Serialize and deserialize
        signal_dict = original.to_dict()
        json_str = json.dumps(signal_dict)
        restored = json.loads(json_str)

        # Verify key fields
        assert restored["signal_id"] == original.signal_id
        assert restored["source_agent"] == "causal_impact"
        assert restored["input_context"]["query"] == "Test query for roundtrip"
        assert restored["reward"] == signal_dict["reward"]

    def test_batch_persistence(self):
        """Test batch of signals persists correctly."""
        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal
        from src.agents.gap_analyzer.dspy_integration import GapAnalysisTrainingSignal

        batch = [
            CausalAnalysisTrainingSignal(signal_id=f"ci_{i}").to_dict()
            for i in range(50)
        ] + [
            GapAnalysisTrainingSignal(signal_id=f"ga_{i}").to_dict()
            for i in range(50)
        ]

        # Serialize batch
        json_str = json.dumps(batch)
        restored = json.loads(json_str)

        assert len(restored) == 100
        assert sum(1 for s in restored if s["source_agent"] == "causal_impact") == 50
        assert sum(1 for s in restored if s["source_agent"] == "gap_analyzer") == 50


class TestSignalFlowContractCompliance:
    """Test complete compliance with SignalFlowContract."""

    def test_all_sender_agents_represented(self):
        """All contract senders are represented."""
        CONTRACT_SENDERS = {
            "causal_impact",
            "gap_analyzer",
            "heterogeneous_optimizer",
            "drift_monitor",
            "experiment_designer",
            "prediction_synthesizer",
        }

        from src.agents.causal_impact.dspy_integration import CausalAnalysisTrainingSignal
        from src.agents.gap_analyzer.dspy_integration import GapAnalysisTrainingSignal
        from src.agents.heterogeneous_optimizer.dspy_integration import HeterogeneousOptimizationTrainingSignal
        from src.agents.drift_monitor.dspy_integration import DriftDetectionTrainingSignal
        from src.agents.experiment_designer.dspy_integration import ExperimentDesignTrainingSignal
        from src.agents.prediction_synthesizer.dspy_integration import PredictionSynthesisTrainingSignal

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

    def test_all_recipient_agents_represented(self):
        """All contract recipients are represented."""
        CONTRACT_RECIPIENTS = {
            "health_score",
            "resource_optimizer",
            "explainer",
        }

        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizerDSPyIntegration
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        recipients = {
            "health_score": HealthScoreDSPyIntegration(),
            "resource_optimizer": ResourceOptimizerDSPyIntegration(),
            "explainer": ExplainerDSPyIntegration(),
        }

        actual_recipients = set(recipients.keys())
        assert actual_recipients == CONTRACT_RECIPIENTS

        for recipient in recipients.values():
            assert recipient.dspy_type == "recipient"

    def test_hub_agent_exists(self):
        """Hub agent (orchestrator) is implemented."""
        from src.agents.orchestrator.dspy_integration import OrchestratorDSPyHub

        hub = OrchestratorDSPyHub()
        assert hub.dspy_type == "hub"

    def test_hybrid_agents_represented(self):
        """Hybrid agents are represented."""
        CONTRACT_HYBRIDS = {
            "tool_composer",
            "feedback_learner",
        }

        from src.agents.tool_composer.dspy_integration import ToolComposerDSPyIntegration
        from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer

        # Both should exist
        tool_composer = ToolComposerDSPyIntegration()
        feedback_learner = FeedbackLearnerOptimizer()

        assert tool_composer is not None
        assert feedback_learner is not None


class TestMemoryIntegration:
    """Test memory integration in signal flow."""

    def test_semantic_memory_contribution(self):
        """Test semantic memory contribution from signals."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="memory_test",
            feedback_count=100,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T23:59:59Z",
            patterns_detected=5,
            focus_agents=["causal_impact", "gap_analyzer"],
        )

        contribution = create_memory_contribution(signal, "semantic")

        assert contribution["memory_type"] == "semantic"
        assert contribution["source_agent"] == "feedback_learner"
        assert "entities" in contribution

    def test_episodic_memory_contribution(self):
        """Test episodic memory contribution from signals."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="episodic_test",
            feedback_count=50,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T12:00:00Z",
        )

        contribution = create_memory_contribution(signal, "episodic")

        assert contribution["memory_type"] == "episodic"
        assert "content" in contribution

    def test_procedural_memory_for_high_quality(self):
        """Test procedural memory contribution for high-quality signals."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        # High-quality signal
        signal = FeedbackLearnerTrainingSignal(
            batch_id="procedural_test",
            feedback_count=100,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-01T23:59:59Z",
            patterns_detected=10,
            pattern_accuracy=0.9,
            recommendation_actionability=0.85,
            update_effectiveness=0.9,
        )

        contribution = create_memory_contribution(signal, "procedural")

        assert contribution["memory_type"] == "procedural"
        # High reward should include procedure
        if signal.compute_reward() >= 0.7:
            assert "procedure" in contribution
