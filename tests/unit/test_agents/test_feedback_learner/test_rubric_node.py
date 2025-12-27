"""
Tests for RubricNode in the Feedback Learner pipeline.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.feedback_learner.evaluation import (
    CriterionScore,
    EvaluationContext,
    ImprovementDecision,
    PatternFlag,
    RubricEvaluation,
    RubricEvaluator,
)
from src.agents.feedback_learner.nodes.rubric_node import RubricNode
from src.agents.feedback_learner.state import FeedbackLearnerState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_evaluation_context():
    """Sample evaluation context for testing."""
    return EvaluationContext(
        user_query="Why did Kisqali adoption increase in Q3?",
        agent_outputs={
            "causal_impact": {
                "effect_size": 0.23,
                "confidence_interval": [0.18, 0.28],
                "p_value": 0.01,
            }
        },
        final_response="Kisqali adoption increased by 23% in Q3, primarily driven by expanded HCP engagement.",
        session_id="test-session-001",
        agent_names=["causal_impact", "explainer"],
        messages_evaluated=5,
    )


@pytest.fixture
def sample_rubric_evaluation():
    """Sample rubric evaluation result."""
    return RubricEvaluation(
        weighted_score=4.2,
        criterion_scores=[
            CriterionScore(
                criterion="causal_validity",
                score=4.5,
                reasoning="Clear distinction between causal claims and correlations.",
            ),
            CriterionScore(
                criterion="actionability",
                score=4.0,
                reasoning="Provides specific next steps.",
            ),
            CriterionScore(
                criterion="evidence_chain",
                score=4.0,
                reasoning="Good data provenance.",
            ),
            CriterionScore(
                criterion="regulatory_awareness",
                score=4.5,
                reasoning="Stays within commercial scope.",
            ),
            CriterionScore(
                criterion="uncertainty_communication",
                score=4.0,
                reasoning="Appropriate hedging language.",
            ),
        ],
        decision=ImprovementDecision.ACCEPTABLE,
        overall_analysis="Response meets quality standards across all criteria.",
        pattern_flags=[],
        improvement_suggestion=None,
    )


@pytest.fixture
def sample_state() -> FeedbackLearnerState:
    """Sample pipeline state with rubric evaluation context."""
    return {
        "batch_id": "batch-001",
        "time_range_start": "2024-01-01",
        "time_range_end": "2024-01-31",
        "focus_agents": ["causal_impact"],
        "cognitive_context": None,
        "training_signal": None,
        "feedback_items": [],
        "feedback_summary": None,
        "detected_patterns": [],
        "pattern_clusters": None,
        "learning_recommendations": [],
        "priority_improvements": None,
        "proposed_updates": [],
        "applied_updates": [],
        "learning_summary": None,
        "metrics_before": None,
        "metrics_after": None,
        "rubric_evaluation_context": {
            "user_query": "Why did Kisqali adoption increase in Q3?",
            "agent_outputs": {"causal_impact": {"effect_size": 0.23}},
            "final_response": "Kisqali adoption increased by 23%...",
            "session_id": "test-session-001",
            "agent_names": ["causal_impact"],
            "messages_evaluated": 3,
        },
        "rubric_evaluation": None,
        "rubric_weighted_score": None,
        "rubric_decision": None,
        "rubric_pattern_flags": None,
        "rubric_improvement_suggestion": None,
        "rubric_latency_ms": None,
        "rubric_error": None,
        "collection_latency_ms": 0,
        "analysis_latency_ms": 0,
        "extraction_latency_ms": 0,
        "update_latency_ms": 0,
        "total_latency_ms": 0,
        "model_used": None,
        "errors": [],
        "warnings": [],
        "status": "analyzing",
    }


@pytest.fixture
def mock_evaluator(sample_rubric_evaluation):
    """Mock evaluator that returns sample evaluation."""
    evaluator = MagicMock(spec=RubricEvaluator)
    evaluator.evaluate = AsyncMock(return_value=sample_rubric_evaluation)
    return evaluator


# =============================================================================
# RubricNode Initialization Tests
# =============================================================================


class TestRubricNodeInit:
    """Tests for RubricNode initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            node = RubricNode()
            assert node.evaluator is not None
            assert node.db_client is None

    def test_init_with_evaluator(self, mock_evaluator):
        """Test initialization with custom evaluator."""
        node = RubricNode(evaluator=mock_evaluator)
        assert node.evaluator is mock_evaluator

    def test_init_with_db_client(self, mock_evaluator):
        """Test initialization with database client."""
        db_client = MagicMock()
        node = RubricNode(evaluator=mock_evaluator, db_client=db_client)
        assert node.db_client is db_client

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            node = RubricNode(model="claude-opus-4-20250514")
            # Just verify it doesn't crash
            assert node.evaluator is not None


# =============================================================================
# RubricNode Execute Tests
# =============================================================================


class TestRubricNodeExecute:
    """Tests for RubricNode.execute()."""

    @pytest.mark.asyncio
    async def test_execute_success(self, sample_state, mock_evaluator, sample_rubric_evaluation):
        """Test successful rubric evaluation."""
        node = RubricNode(evaluator=mock_evaluator)

        result = await node.execute(sample_state)

        # Verify evaluator was called
        mock_evaluator.evaluate.assert_called_once()

        # Verify result state
        assert result["rubric_evaluation"] is not None
        assert result["rubric_weighted_score"] == sample_rubric_evaluation.weighted_score
        assert result["rubric_decision"] == sample_rubric_evaluation.decision.value
        assert result["rubric_latency_ms"] is not None
        assert result["rubric_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_no_context(self, sample_state, mock_evaluator):
        """Test execute when no evaluation context is provided."""
        sample_state["rubric_evaluation_context"] = None
        node = RubricNode(evaluator=mock_evaluator)

        result = await node.execute(sample_state)

        # Evaluator should not be called
        mock_evaluator.evaluate.assert_not_called()

        # Result should indicate no evaluation
        assert result["rubric_evaluation"] is None
        assert result["rubric_latency_ms"] is not None

    @pytest.mark.asyncio
    async def test_execute_with_failed_status(self, sample_state, mock_evaluator):
        """Test that node skips evaluation when status is failed."""
        sample_state["status"] = "failed"
        node = RubricNode(evaluator=mock_evaluator)

        result = await node.execute(sample_state)

        # Should return state unchanged
        mock_evaluator.evaluate.assert_not_called()
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_with_dict_context(self, sample_state, mock_evaluator):
        """Test execute with dict evaluation context (converts to EvaluationContext)."""
        node = RubricNode(evaluator=mock_evaluator)

        result = await node.execute(sample_state)

        # Should still work
        mock_evaluator.evaluate.assert_called_once()
        assert result["rubric_evaluation"] is not None

    @pytest.mark.asyncio
    async def test_execute_handles_error(self, sample_state):
        """Test that execute handles evaluation errors gracefully."""
        mock_evaluator = MagicMock(spec=RubricEvaluator)
        mock_evaluator.evaluate = AsyncMock(side_effect=Exception("API error"))

        node = RubricNode(evaluator=mock_evaluator)
        result = await node.execute(sample_state)

        # Should capture error
        assert result["rubric_evaluation"] is None
        assert result["rubric_error"] is not None
        assert "API error" in result["rubric_error"]
        assert len(result["errors"]) > 0
        assert len(result["warnings"]) > 0


# =============================================================================
# RubricNode Store Evaluation Tests
# =============================================================================


class TestRubricNodeStoreEvaluation:
    """Tests for RubricNode._store_evaluation()."""

    @pytest.mark.asyncio
    async def test_store_evaluation_with_db_client(
        self, sample_state, mock_evaluator, sample_rubric_evaluation, sample_evaluation_context
    ):
        """Test that evaluation is stored when db_client is provided."""
        db_client = MagicMock()
        db_client.table = MagicMock(return_value=MagicMock())
        db_client.table.return_value.insert = MagicMock(return_value=MagicMock())
        db_client.table.return_value.insert.return_value.execute = AsyncMock()

        node = RubricNode(evaluator=mock_evaluator, db_client=db_client)

        await node._store_evaluation(sample_rubric_evaluation, sample_evaluation_context)

        # Verify table was called
        db_client.table.assert_called_once_with("learning_signals")

    @pytest.mark.asyncio
    async def test_store_evaluation_no_db_client(
        self, sample_rubric_evaluation, sample_evaluation_context, mock_evaluator
    ):
        """Test that store does nothing when no db_client."""
        node = RubricNode(evaluator=mock_evaluator, db_client=None)

        # Should not raise
        await node._store_evaluation(sample_rubric_evaluation, sample_evaluation_context)

    @pytest.mark.asyncio
    async def test_store_evaluation_handles_error(
        self, sample_rubric_evaluation, sample_evaluation_context, mock_evaluator
    ):
        """Test that store handles database errors gracefully."""
        db_client = MagicMock()
        db_client.table = MagicMock(side_effect=Exception("DB error"))

        node = RubricNode(evaluator=mock_evaluator, db_client=db_client)

        # Should not raise
        await node._store_evaluation(sample_rubric_evaluation, sample_evaluation_context)


# =============================================================================
# Improvement Type and Priority Tests
# =============================================================================


class TestRubricNodeDetermineImprovementType:
    """Tests for RubricNode._determine_improvement_type()."""

    def test_acceptable_returns_none(self, mock_evaluator, sample_rubric_evaluation):
        """Test that acceptable evaluations return 'none' improvement type."""
        node = RubricNode(evaluator=mock_evaluator)
        result = node._determine_improvement_type(sample_rubric_evaluation)
        assert result == "none"

    def test_low_causal_validity_returns_prompt(self, mock_evaluator):
        """Test that low causal_validity returns 'prompt' improvement type."""
        evaluation = RubricEvaluation(
            weighted_score=2.5,
            criterion_scores=[
                CriterionScore(criterion="causal_validity", score=1.5, reasoning="Poor"),
                CriterionScore(criterion="actionability", score=3.0, reasoning="OK"),
                CriterionScore(criterion="evidence_chain", score=3.0, reasoning="OK"),
                CriterionScore(criterion="regulatory_awareness", score=3.0, reasoning="OK"),
                CriterionScore(criterion="uncertainty_communication", score=3.0, reasoning="OK"),
            ],
            decision=ImprovementDecision.SUGGESTION,
            overall_analysis="Needs work on causal validity.",
            pattern_flags=[],
        )

        node = RubricNode(evaluator=mock_evaluator)
        result = node._determine_improvement_type(evaluation)
        assert result == "prompt"

    def test_low_evidence_chain_returns_retrieval(self, mock_evaluator):
        """Test that low evidence_chain returns 'retrieval' improvement type."""
        evaluation = RubricEvaluation(
            weighted_score=2.5,
            criterion_scores=[
                CriterionScore(criterion="causal_validity", score=3.0, reasoning="OK"),
                CriterionScore(criterion="actionability", score=3.0, reasoning="OK"),
                CriterionScore(criterion="evidence_chain", score=1.5, reasoning="Poor"),
                CriterionScore(criterion="regulatory_awareness", score=3.0, reasoning="OK"),
                CriterionScore(criterion="uncertainty_communication", score=3.0, reasoning="OK"),
            ],
            decision=ImprovementDecision.SUGGESTION,
            overall_analysis="Needs work on evidence chain.",
            pattern_flags=[],
        )

        node = RubricNode(evaluator=mock_evaluator)
        result = node._determine_improvement_type(evaluation)
        assert result == "retrieval"


class TestRubricNodeDeterminePriority:
    """Tests for RubricNode._determine_priority()."""

    def test_acceptable_returns_low(self, mock_evaluator, sample_rubric_evaluation):
        """Test that acceptable evaluations return 'low' priority."""
        node = RubricNode(evaluator=mock_evaluator)
        result = node._determine_priority(sample_rubric_evaluation)
        assert result == "low"

    def test_suggestion_returns_medium(self, mock_evaluator):
        """Test that suggestion decision returns 'medium' priority."""
        evaluation = RubricEvaluation(
            weighted_score=3.2,
            criterion_scores=[],
            decision=ImprovementDecision.SUGGESTION,
            overall_analysis="Minor improvements needed.",
            pattern_flags=[],
        )

        node = RubricNode(evaluator=mock_evaluator)
        result = node._determine_priority(evaluation)
        assert result == "medium"

    def test_auto_update_returns_high(self, mock_evaluator):
        """Test that auto_update decision returns 'high' priority."""
        evaluation = RubricEvaluation(
            weighted_score=2.5,
            criterion_scores=[],
            decision=ImprovementDecision.AUTO_UPDATE,
            overall_analysis="Significant improvements needed.",
            pattern_flags=[],
        )

        node = RubricNode(evaluator=mock_evaluator)
        result = node._determine_priority(evaluation)
        assert result == "high"

    def test_escalate_returns_critical(self, mock_evaluator):
        """Test that escalate decision returns 'critical' priority."""
        evaluation = RubricEvaluation(
            weighted_score=1.5,
            criterion_scores=[],
            decision=ImprovementDecision.ESCALATE,
            overall_analysis="Serious issues requiring escalation.",
            pattern_flags=[],
        )

        node = RubricNode(evaluator=mock_evaluator)
        result = node._determine_priority(evaluation)
        assert result == "critical"


# =============================================================================
# Standalone Evaluation Tests
# =============================================================================


class TestRubricNodeEvaluateAndDecide:
    """Tests for RubricNode.evaluate_and_decide()."""

    @pytest.mark.asyncio
    async def test_evaluate_and_decide(self, sample_evaluation_context, mock_evaluator, sample_rubric_evaluation):
        """Test the standalone evaluation method."""
        node = RubricNode(evaluator=mock_evaluator)

        result = await node.evaluate_and_decide(sample_evaluation_context)

        assert "weighted_score" in result
        assert result["weighted_score"] == sample_rubric_evaluation.weighted_score
        assert "decision" in result
        assert result["decision"] == sample_rubric_evaluation.decision.value
        assert "is_acceptable" in result
        assert "needs_action" in result
        assert "criterion_scores" in result
        assert "pattern_flags" in result
        assert "overall_analysis" in result


# =============================================================================
# Integration with Graph Tests
# =============================================================================


class TestRubricNodeGraphIntegration:
    """Tests for RubricNode integration with the feedback learner graph."""

    def test_node_can_be_imported_from_nodes(self):
        """Test that RubricNode can be imported from the nodes package."""
        from src.agents.feedback_learner.nodes.rubric_node import RubricNode

        assert RubricNode is not None

    def test_node_can_be_imported_from_graph(self):
        """Test that RubricNode is used in the graph module."""
        from src.agents.feedback_learner.graph import build_feedback_learner_graph

        # Should not raise
        assert build_feedback_learner_graph is not None
