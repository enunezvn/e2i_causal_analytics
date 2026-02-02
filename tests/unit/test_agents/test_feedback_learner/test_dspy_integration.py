"""
Tests for Feedback Learner DSPy Integration.

Tests both Sender and optimizer role implementation including:
- Training signal generation and validation
- GEPA and MIPROv2 optimizer support
- DSPy signature availability
- Cognitive context handling
- Memory contribution helpers
"""

import pytest

# Mark all tests in this module as dspy_integration to group them
pytestmark = pytest.mark.xdist_group(name="dspy_integration")


class TestFeedbackLearnerTrainingSignal:
    """Test FeedbackLearnerTrainingSignal dataclass."""

    def test_default_initialization(self):
        """Test signal initializes with sensible defaults."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="test_batch",
            feedback_count=0,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        assert signal.batch_id == "test_batch"
        assert signal.feedback_count == 0
        assert signal.focus_agents == []
        assert signal.cognitive_context is None
        assert signal.patterns_detected == 0
        assert signal.recommendations_generated == 0
        assert signal.updates_applied == 0
        assert signal.pattern_accuracy == 0.0
        assert signal.rubric_weighted_score is None
        assert signal.created_at  # Should have a timestamp

    def test_custom_initialization(self):
        """Test signal with custom values."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="batch_123",
            feedback_count=50,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-07T00:00:00Z",
            focus_agents=["causal_impact", "gap_analyzer"],
            patterns_detected=5,
            recommendations_generated=3,
            updates_applied=2,
            pattern_accuracy=0.85,
            recommendation_actionability=0.9,
            update_effectiveness=0.75,
            total_latency_ms=15000.0,
            model_used="anthropic/claude-sonnet-4-20250514",
            llm_calls=10,
        )

        assert signal.feedback_count == 50
        assert len(signal.focus_agents) == 2
        assert signal.patterns_detected == 5
        assert signal.pattern_accuracy == 0.85

    def test_compute_reward_high_quality_without_rubric(self):
        """Test reward computation for high-quality signal without rubric."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="high_quality",
            feedback_count=100,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
            patterns_detected=10,  # Good coverage (10/100 = 0.1)
            recommendations_generated=5,
            updates_applied=3,
            pattern_accuracy=0.9,
            recommendation_actionability=0.85,
            update_effectiveness=0.8,
            total_latency_ms=20000.0,  # 100 items in 20s = 5 items/s (good)
        )

        reward = signal.compute_reward()

        # Should be high reward (> 0.7)
        assert reward >= 0.7
        assert reward <= 1.0

    def test_compute_reward_with_rubric(self):
        """Test reward computation with rubric evaluation."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="rubric_test",
            feedback_count=50,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
            patterns_detected=5,
            recommendations_generated=3,
            updates_applied=2,
            pattern_accuracy=0.8,
            recommendation_actionability=0.75,
            update_effectiveness=0.7,
            rubric_weighted_score=4.5,  # High rubric score
            rubric_decision="apply",
            rubric_pattern_flags=0,
            total_latency_ms=15000.0,
        )

        reward = signal.compute_reward()

        # Should include rubric contribution
        assert reward > 0.5

    def test_compute_reward_low_quality(self):
        """Test reward computation for low-quality signal."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="low_quality",
            feedback_count=100,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
            patterns_detected=0,  # No patterns
            recommendations_generated=0,
            updates_applied=0,
            pattern_accuracy=0.1,
            recommendation_actionability=0.1,
            update_effectiveness=0.0,
            total_latency_ms=120000.0,  # Very slow
        )

        reward = signal.compute_reward()

        # Should be low reward
        assert reward < 0.3

    def test_to_dict_structure(self):
        """Test to_dict produces correct structure."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="dict_test",
            feedback_count=25,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
            focus_agents=["causal_impact"],
            patterns_detected=3,
            recommendations_generated=2,
            updates_applied=1,
            pattern_accuracy=0.75,
            rubric_weighted_score=4.0,
            rubric_decision="apply",
            model_used="anthropic/claude-sonnet-4-20250514",
            llm_calls=5,
            total_tokens=2500,
        )

        result = signal.to_dict()

        # Check structure
        assert result["source_agent"] == "feedback_learner"
        assert "signal_id" in result
        assert result["signal_id"] == "fbl_dict_test"
        assert "timestamp" in result
        assert "input_context" in result
        assert "output" in result
        assert "quality_metrics" in result
        assert "rubric_evaluation" in result
        assert "latency" in result
        assert "llm_usage" in result
        assert "outcomes" in result
        assert "reward" in result

        # Check nested structure
        assert result["input_context"]["feedback_count"] == 25
        assert result["input_context"]["has_cognitive_context"] is False
        assert result["output"]["patterns_detected"] == 3
        assert result["quality_metrics"]["pattern_accuracy"] == 0.75
        assert result["rubric_evaluation"]["weighted_score"] == 4.0
        assert result["llm_usage"]["model"] == "anthropic/claude-sonnet-4-20250514"


class TestFeedbackLearnerCognitiveContext:
    """Test FeedbackLearnerCognitiveContext TypedDict."""

    def test_context_structure(self):
        """Test cognitive context can be created with expected fields."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerCognitiveContext,
        )

        # TypedDict is a type hint, not a class - create as dict
        context: FeedbackLearnerCognitiveContext = {
            "synthesized_summary": "Analysis shows improvement in causal reasoning",
            "historical_patterns": [{"pattern": "accuracy_improvement", "trend": "up"}],
            "optimization_examples": [{"example": "prompt_refinement", "score": 0.85}],
            "agent_baselines": {
                "causal_impact": {"accuracy": 0.8, "latency_ms": 500},
            },
            "prior_learnings": [{"learning": "context helps", "confidence": 0.9}],
            "correlation_insights": [{"insight": "feedback correlates with updates"}],
            "evidence_confidence": 0.87,
        }

        assert context["synthesized_summary"] is not None
        assert len(context["historical_patterns"]) == 1
        assert context["evidence_confidence"] == 0.87


class TestAgentTrainingSignal:
    """Test AgentTrainingSignal TypedDict."""

    def test_signal_structure(self):
        """Test agent training signal can be created with expected fields."""
        from src.agents.feedback_learner.dspy_integration import AgentTrainingSignal

        signal: AgentTrainingSignal = {
            "signal_id": "sig_123",
            "source_agent": "causal_impact",
            "timestamp": "2025-01-01T12:00:00Z",
            "input_context": {"query": "test query"},
            "output": {"result": "test result"},
            "user_feedback": {"rating": 4},
            "outcome_observed": None,
            "latency_ms": 250.0,
            "token_count": 1500,
            "cognitive_phase": "analysis",
        }

        assert signal["signal_id"] == "sig_123"
        assert signal["source_agent"] == "causal_impact"
        assert signal["latency_ms"] == 250.0


class TestFeedbackLearnerOptimizer:
    """Test FeedbackLearnerOptimizer class."""

    def test_initialization_gepa(self):
        """Test optimizer initializes with GEPA when available."""
        from src.agents.feedback_learner.dspy_integration import (
            DSPY_AVAILABLE,
            GEPA_AVAILABLE,
            FeedbackLearnerOptimizer,
        )

        optimizer = FeedbackLearnerOptimizer(optimizer_type="gepa")

        if GEPA_AVAILABLE:
            assert optimizer.optimizer_type == "gepa"
        elif DSPY_AVAILABLE:
            assert optimizer.optimizer_type == "miprov2"
        else:
            assert optimizer.optimizer_type is None

    def test_initialization_miprov2(self):
        """Test optimizer initializes with MIPROv2."""
        from src.agents.feedback_learner.dspy_integration import (
            DSPY_AVAILABLE,
            FeedbackLearnerOptimizer,
        )

        optimizer = FeedbackLearnerOptimizer(optimizer_type="miprov2")

        if DSPY_AVAILABLE:
            assert optimizer.optimizer_type == "miprov2"
        else:
            assert optimizer.optimizer_type is None

    def test_pattern_metric(self):
        """Test pattern detection metric."""
        from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer

        optimizer = FeedbackLearnerOptimizer()

        # Create mock prediction with good patterns
        class MockPrediction:
            patterns = [
                {"type": "accuracy_drop", "severity": "high", "affected_agents": ["causal_impact"]},
                {
                    "type": "latency_spike",
                    "severity": "medium",
                    "affected_agents": ["orchestrator"],
                    "root_cause_hypothesis": "LLM timeout",
                },
            ]
            confidence = 0.7
            root_causes = ["LLM timeout", "Data quality issue"]

        score = optimizer.pattern_metric(None, MockPrediction())

        # Should have positive score
        assert score > 0

    def test_pattern_metric_empty(self):
        """Test pattern metric with empty prediction."""
        from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer

        optimizer = FeedbackLearnerOptimizer()

        class MockPrediction:
            patterns = []
            confidence = 0.5
            root_causes = []

        score = optimizer.pattern_metric(None, MockPrediction())

        # Should handle empty patterns gracefully
        assert score >= 0

    def test_recommendation_metric(self):
        """Test recommendation generation metric."""
        from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer

        optimizer = FeedbackLearnerOptimizer()

        class MockPrediction:
            recommendations = [
                {"category": "prompt_update", "expected_impact": "10% accuracy improvement"},
                {"category": "config_change", "expected_impact": "20% latency reduction"},
            ]
            implementation_order = ["prompt_update", "config_change"]
            risk_assessment = (
                "Low risk changes. Rollback possible if metrics degrade significantly."
            )

        score = optimizer.recommendation_metric(None, MockPrediction())

        # Should have positive score
        assert score > 0

    def test_signals_to_examples_empty(self):
        """Test converting empty signals to examples."""
        from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer

        optimizer = FeedbackLearnerOptimizer()

        examples = optimizer._signals_to_examples([], "pattern")

        assert examples == []

    def test_signals_to_examples_filters_low_reward(self):
        """Test that signals with low reward are filtered."""
        from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer

        optimizer = FeedbackLearnerOptimizer()

        signals = [
            {"source_agent": "feedback_learner", "reward": 0.3},  # Too low
            {"source_agent": "other_agent", "reward": 0.8},  # Wrong agent
        ]

        examples = optimizer._signals_to_examples(signals, "pattern")

        assert examples == []


class TestDSPySignatures:
    """Test DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE flag."""
        from src.agents.feedback_learner.dspy_integration import DSPY_AVAILABLE

        assert isinstance(DSPY_AVAILABLE, bool)

    def test_gepa_available_flag(self):
        """Test GEPA_AVAILABLE flag."""
        from src.agents.feedback_learner.dspy_integration import GEPA_AVAILABLE

        assert isinstance(GEPA_AVAILABLE, bool)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_pattern_detection_signature(self):
        """Test PatternDetectionSignature is valid DSPy signature."""
        from src.agents.feedback_learner.dspy_integration import (
            DSPY_AVAILABLE,
            PatternDetectionSignature,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(PatternDetectionSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_recommendation_generation_signature(self):
        """Test RecommendationGenerationSignature is valid DSPy signature."""
        from src.agents.feedback_learner.dspy_integration import (
            DSPY_AVAILABLE,
            RecommendationGenerationSignature,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(RecommendationGenerationSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_knowledge_update_signature(self):
        """Test KnowledgeUpdateSignature is valid DSPy signature."""
        from src.agents.feedback_learner.dspy_integration import (
            DSPY_AVAILABLE,
            KnowledgeUpdateSignature,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(KnowledgeUpdateSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_learning_summary_signature(self):
        """Test LearningSummarySignature is valid DSPy signature."""
        from src.agents.feedback_learner.dspy_integration import (
            DSPY_AVAILABLE,
            LearningSummarySignature,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(LearningSummarySignature, dspy.Signature)


class TestMemoryContribution:
    """Test memory contribution helper."""

    def test_create_semantic_memory_contribution(self):
        """Test creating semantic memory contribution."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="memory_test",
            feedback_count=50,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
            focus_agents=["causal_impact", "gap_analyzer"],
            patterns_detected=5,
            recommendations_generated=3,
            updates_applied=2,
        )

        contribution = create_memory_contribution(signal, memory_type="semantic")

        assert contribution["source_agent"] == "feedback_learner"
        assert contribution["memory_type"] == "semantic"
        assert contribution["index"] == "learning_outcomes"
        assert contribution["ttl_days"] == 365
        assert "entities" in contribution
        assert "relationships" in contribution

        # Check entity structure
        entity = contribution["entities"][0]
        assert entity["type"] == "LearningCycle"
        assert entity["id"] == "memory_test"
        assert entity["properties"]["patterns_detected"] == 5

        # Check relationships (one for each focus agent)
        assert len(contribution["relationships"]) == 2

    def test_create_episodic_memory_contribution(self):
        """Test creating episodic memory contribution."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="episodic_test",
            feedback_count=30,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
            patterns_detected=3,
            recommendations_generated=2,
        )

        contribution = create_memory_contribution(signal, memory_type="episodic")

        assert contribution["source_agent"] == "feedback_learner"
        assert contribution["memory_type"] == "episodic"
        assert contribution["index"] == "learning_experiences"
        assert contribution["ttl_days"] == 180
        assert "content" in contribution
        assert "30 feedback items" in contribution["content"]["summary"]

    def test_create_procedural_memory_contribution_high_quality(self):
        """Test creating procedural memory contribution for high-quality signal."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="procedural_test",
            feedback_count=100,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
            focus_agents=["causal_impact"],
            patterns_detected=10,
            recommendations_generated=5,
            updates_applied=3,
            pattern_accuracy=0.9,
            recommendation_actionability=0.85,
            update_effectiveness=0.8,
            total_latency_ms=20000.0,
        )

        contribution = create_memory_contribution(signal, memory_type="procedural")

        assert contribution["source_agent"] == "feedback_learner"
        assert contribution["memory_type"] == "procedural"
        assert contribution["index"] == "learning_procedures"
        assert contribution["ttl_days"] == 365
        assert "procedure" in contribution

        procedure = contribution["procedure"]
        assert procedure["trigger"] == "feedback_batch"
        assert "collect_feedback" in procedure["steps"]

    def test_create_procedural_memory_contribution_low_quality(self):
        """Test that low-quality signals don't create procedural memory."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerTrainingSignal,
            create_memory_contribution,
        )

        signal = FeedbackLearnerTrainingSignal(
            batch_id="low_quality",
            feedback_count=10,
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
            patterns_detected=0,
            recommendations_generated=0,
            updates_applied=0,
            pattern_accuracy=0.1,
            recommendation_actionability=0.1,
            update_effectiveness=0.0,
        )

        contribution = create_memory_contribution(signal, memory_type="procedural")

        # Should not include procedure for low-quality signal
        assert "procedure" not in contribution


class TestOptimizerTypeAlias:
    """Test OptimizerType type alias."""

    def test_optimizer_type_values(self):
        """Test OptimizerType accepts valid values."""
        from src.agents.feedback_learner.dspy_integration import OptimizerType

        # These should be valid
        valid_types: list[OptimizerType] = ["miprov2", "gepa"]

        for opt_type in valid_types:
            assert opt_type in ["miprov2", "gepa"]


class TestExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Test all expected exports are available."""
        from src.agents.feedback_learner import dspy_integration

        expected_exports = [
            "FeedbackLearnerCognitiveContext",
            "AgentTrainingSignal",
            "FeedbackLearnerTrainingSignal",
            "PatternDetectionSignature",
            "RecommendationGenerationSignature",
            "KnowledgeUpdateSignature",
            "LearningSummarySignature",
            "FeedbackLearnerOptimizer",
            "DSPY_AVAILABLE",
            "GEPA_AVAILABLE",
            "OptimizerType",
            "create_memory_contribution",
        ]

        for export in expected_exports:
            assert hasattr(dspy_integration, export), f"Missing export: {export}"
