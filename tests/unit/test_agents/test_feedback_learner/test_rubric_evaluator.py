"""
Tests for the rubric evaluator module.

Tests verify:
1. Criteria definitions and weights
2. Score calculation logic
3. Decision determination
4. Pattern detection
5. Pydantic models
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.agents.feedback_learner.evaluation.criteria import (
    DEFAULT_CRITERIA,
    DEFAULT_OVERRIDE_CONDITIONS,
    DEFAULT_THRESHOLDS,
    get_criterion_by_name,
    get_default_criteria,
    get_total_weight,
    validate_weights,
)
from src.agents.feedback_learner.evaluation.models import (
    CriterionScore,
    EvaluationContext,
    ImprovementDecision,
    PatternFlag,
    RubricEvaluation,
)
from src.agents.feedback_learner.evaluation.rubric_evaluator import RubricEvaluator

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_context() -> EvaluationContext:
    """Sample evaluation context for testing."""
    return EvaluationContext(
        user_query="Why did Kisqali adoption increase in the Northeast?",
        agent_outputs={
            "causal_impact": {"effect": 0.23, "confidence": 0.87},
            "orchestrator": {"routing_decision": "causal_analysis"},
        },
        final_response=(
            "Kisqali adoption increased by 23% in the Northeast region. "
            "This increase is correlated with the new promotional campaign launched in Q3. "
            "The causal analysis suggests the campaign contributed to improved HCP engagement."
        ),
        session_id="test-session-001",
        agent_names=["orchestrator", "causal_impact", "explainer"],
        messages_evaluated=1,
        retrieved_contexts=["HCP engagement data Q3 2024"],
    )


@pytest.fixture
def sample_criterion_scores() -> list[CriterionScore]:
    """Sample criterion scores for testing."""
    return [
        CriterionScore(
            criterion="causal_validity",
            score=4.0,
            reasoning="Uses appropriate causal language",
            evidence="'contributed to' rather than 'caused'",
        ),
        CriterionScore(
            criterion="actionability",
            score=3.5,
            reasoning="Some actionable content but lacks specifics",
            evidence="Mentions campaign but no next steps",
        ),
        CriterionScore(
            criterion="evidence_chain",
            score=4.5,
            reasoning="Clear data references",
            evidence="References Q3 data and HCP engagement",
        ),
        CriterionScore(
            criterion="regulatory_awareness",
            score=5.0,
            reasoning="Stays within commercial scope",
            evidence="No medical claims",
        ),
        CriterionScore(
            criterion="uncertainty_communication",
            score=3.0,
            reasoning="Could be more precise about confidence",
            evidence="Missing confidence intervals",
        ),
    ]


@pytest.fixture
def evaluator_no_client() -> RubricEvaluator:
    """Evaluator without API client (for testing fallback behavior)."""
    return RubricEvaluator()


# ═══════════════════════════════════════════════════════════════════════════════
# CRITERIA TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCriteriaDefinitions:
    """Test rubric criteria definitions."""

    def test_default_criteria_count(self):
        """Should have exactly 5 criteria."""
        assert len(DEFAULT_CRITERIA) == 5

    def test_criteria_names(self):
        """Should have expected criterion names."""
        names = {c.name for c in DEFAULT_CRITERIA}
        expected = {
            "causal_validity",
            "actionability",
            "evidence_chain",
            "regulatory_awareness",
            "uncertainty_communication",
        }
        assert names == expected

    def test_weights_sum_to_one(self):
        """Total weights should sum to 1.0."""
        assert validate_weights() is True
        assert abs(get_total_weight() - 1.0) < 0.001

    def test_individual_weights(self):
        """Check individual criterion weights."""
        weights = {c.name: c.weight for c in DEFAULT_CRITERIA}
        assert weights["causal_validity"] == 0.25
        assert weights["actionability"] == 0.25
        assert weights["evidence_chain"] == 0.20
        assert weights["regulatory_awareness"] == 0.15
        assert weights["uncertainty_communication"] == 0.15

    def test_get_default_criteria_returns_copy(self):
        """get_default_criteria should return a copy."""
        criteria1 = get_default_criteria()
        criteria2 = get_default_criteria()
        assert criteria1 is not criteria2

    def test_get_criterion_by_name(self):
        """Should get specific criterion by name."""
        criterion = get_criterion_by_name("causal_validity")
        assert criterion.name == "causal_validity"
        assert criterion.weight == 0.25

    def test_get_criterion_by_name_unknown_raises(self):
        """Should raise for unknown criterion name."""
        with pytest.raises(ValueError, match="Unknown criterion"):
            get_criterion_by_name("unknown_criterion")

    def test_scoring_guides_complete(self):
        """Each criterion should have scoring guide for 1-5."""
        for criterion in DEFAULT_CRITERIA:
            for score in [1, 2, 3, 4, 5]:
                assert score in criterion.scoring_guide


class TestDefaultThresholds:
    """Test decision thresholds."""

    def test_threshold_values(self):
        """Check threshold values."""
        assert DEFAULT_THRESHOLDS["acceptable"] == 4.0
        assert DEFAULT_THRESHOLDS["suggestion"] == 3.0
        assert DEFAULT_THRESHOLDS["auto_update"] == 2.0

    def test_override_conditions_exist(self):
        """Should have override conditions defined."""
        assert len(DEFAULT_OVERRIDE_CONDITIONS) > 0

    def test_override_condition_structure(self):
        """Override conditions should have required keys."""
        for condition in DEFAULT_OVERRIDE_CONDITIONS:
            assert "condition" in condition
            assert "threshold" in condition
            assert "action" in condition


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCriterionScore:
    """Test CriterionScore model."""

    def test_valid_score(self):
        """Should accept valid scores."""
        score = CriterionScore(
            criterion="causal_validity",
            score=4.0,
            reasoning="Good causal language",
        )
        assert score.score == 4.0

    def test_score_bounds(self):
        """Should enforce score bounds 1-5."""
        with pytest.raises(ValueError):
            CriterionScore(criterion="test", score=0.5, reasoning="Too low")
        with pytest.raises(ValueError):
            CriterionScore(criterion="test", score=5.5, reasoning="Too high")

    def test_optional_evidence(self):
        """Evidence should be optional."""
        score = CriterionScore(
            criterion="test",
            score=3.0,
            reasoning="No evidence provided",
        )
        assert score.evidence is None


class TestEvaluationContext:
    """Test EvaluationContext model."""

    def test_minimal_context(self):
        """Should work with minimal fields."""
        context = EvaluationContext(
            user_query="Test query",
            final_response="Test response",
        )
        assert context.user_query == "Test query"
        assert context.agent_outputs == {}
        assert context.agent_names == []

    def test_full_context(self, sample_context):
        """Should handle full context."""
        assert sample_context.session_id == "test-session-001"
        assert len(sample_context.agent_names) == 3
        assert "causal_impact" in sample_context.agent_outputs


class TestImprovementDecision:
    """Test ImprovementDecision enum."""

    def test_enum_values(self):
        """Check enum values."""
        assert ImprovementDecision.ACCEPTABLE.value == "acceptable"
        assert ImprovementDecision.SUGGESTION.value == "suggestion"
        assert ImprovementDecision.AUTO_UPDATE.value == "auto_update"
        assert ImprovementDecision.ESCALATE.value == "escalate"


class TestRubricEvaluation:
    """Test RubricEvaluation model."""

    def test_evaluation_creation(self, sample_criterion_scores):
        """Should create valid evaluation."""
        evaluation = RubricEvaluation(
            weighted_score=4.05,
            criterion_scores=sample_criterion_scores,
            decision=ImprovementDecision.ACCEPTABLE,
            overall_analysis="Good overall performance",
        )
        assert evaluation.weighted_score == 4.05
        assert len(evaluation.criterion_scores) == 5

    def test_is_acceptable_property(self, sample_criterion_scores):
        """Should correctly identify acceptable evaluations."""
        acceptable = RubricEvaluation(
            weighted_score=4.5,
            criterion_scores=sample_criterion_scores,
            decision=ImprovementDecision.ACCEPTABLE,
            overall_analysis="Good",
        )
        assert acceptable.is_acceptable is True

        not_acceptable = RubricEvaluation(
            weighted_score=3.0,
            criterion_scores=sample_criterion_scores,
            decision=ImprovementDecision.SUGGESTION,
            overall_analysis="Needs work",
        )
        assert not_acceptable.is_acceptable is False

    def test_needs_action_property(self, sample_criterion_scores):
        """Should correctly identify evaluations needing action."""
        acceptable = RubricEvaluation(
            weighted_score=4.5,
            criterion_scores=sample_criterion_scores,
            decision=ImprovementDecision.ACCEPTABLE,
            overall_analysis="Good",
        )
        assert acceptable.needs_action is False

        suggestion = RubricEvaluation(
            weighted_score=3.5,
            criterion_scores=sample_criterion_scores,
            decision=ImprovementDecision.SUGGESTION,
            overall_analysis="Needs work",
        )
        assert suggestion.needs_action is True

    def test_to_learning_signal_format(self, sample_criterion_scores):
        """Should convert to learning signal format."""
        evaluation = RubricEvaluation(
            weighted_score=4.0,
            criterion_scores=sample_criterion_scores,
            decision=ImprovementDecision.ACCEPTABLE,
            overall_analysis="Good",
        )
        signal_format = evaluation.to_learning_signal_format()

        assert "rubric_scores" in signal_format
        assert "rubric_total" in signal_format
        assert "improvement_details" in signal_format
        assert signal_format["rubric_total"] == 4.0


class TestPatternFlag:
    """Test PatternFlag model."""

    def test_pattern_flag_creation(self):
        """Should create valid pattern flag."""
        flag = PatternFlag(
            pattern_type="low_causal_validity",
            score=2.0,
            reasoning="Missing causal evidence",
            criterion="causal_validity",
        )
        assert flag.pattern_type == "low_causal_validity"
        assert flag.criterion == "causal_validity"


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRubricEvaluatorInit:
    """Test RubricEvaluator initialization."""

    def test_default_initialization(self, evaluator_no_client):
        """Should initialize with default values."""
        assert len(evaluator_no_client.criteria) == 5
        assert evaluator_no_client.thresholds["acceptable"] == 4.0
        assert evaluator_no_client.model == "claude-sonnet-4-20250514"

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        custom_thresholds = {"acceptable": 4.5, "suggestion": 3.5, "auto_update": 2.5}
        evaluator = RubricEvaluator(thresholds=custom_thresholds)
        assert evaluator.thresholds["acceptable"] == 4.5


class TestWeightedScoreCalculation:
    """Test weighted score calculation."""

    def test_calculate_weighted_score(self, evaluator_no_client, sample_criterion_scores):
        """Should calculate correct weighted score."""
        score = evaluator_no_client._calculate_weighted_score(sample_criterion_scores)
        # 4.0*0.25 + 3.5*0.25 + 4.5*0.20 + 5.0*0.15 + 3.0*0.15
        # = 1.0 + 0.875 + 0.9 + 0.75 + 0.45 = 3.975
        expected = 3.98  # Rounded
        assert abs(score - expected) < 0.01

    def test_all_fives(self, evaluator_no_client):
        """All 5s should give 5.0."""
        scores = [
            CriterionScore(criterion=c.name, score=5.0, reasoning="Perfect")
            for c in DEFAULT_CRITERIA
        ]
        result = evaluator_no_client._calculate_weighted_score(scores)
        assert result == 5.0

    def test_all_ones(self, evaluator_no_client):
        """All 1s should give 1.0."""
        scores = [
            CriterionScore(criterion=c.name, score=1.0, reasoning="Poor") for c in DEFAULT_CRITERIA
        ]
        result = evaluator_no_client._calculate_weighted_score(scores)
        assert result == 1.0


class TestDecisionDetermination:
    """Test decision determination logic."""

    def test_acceptable_decision(self, evaluator_no_client):
        """Score >= 4.0 should be ACCEPTABLE."""
        scores = [
            CriterionScore(criterion=c.name, score=4.5, reasoning="Good") for c in DEFAULT_CRITERIA
        ]
        decision = evaluator_no_client._determine_decision(4.5, scores)
        assert decision == ImprovementDecision.ACCEPTABLE

    def test_suggestion_decision(self, evaluator_no_client):
        """Score 3.0-3.9 should be SUGGESTION."""
        scores = [
            CriterionScore(criterion=c.name, score=3.5, reasoning="OK") for c in DEFAULT_CRITERIA
        ]
        decision = evaluator_no_client._determine_decision(3.5, scores)
        assert decision == ImprovementDecision.SUGGESTION

    def test_auto_update_decision(self, evaluator_no_client):
        """Score 2.0-2.9 should be AUTO_UPDATE."""
        scores = [
            CriterionScore(criterion=c.name, score=2.5, reasoning="Needs work")
            for c in DEFAULT_CRITERIA
        ]
        decision = evaluator_no_client._determine_decision(2.5, scores)
        assert decision == ImprovementDecision.AUTO_UPDATE

    def test_escalate_decision(self):
        """Score < 2.0 should be ESCALATE (without override conditions)."""
        # Create evaluator without override conditions to test pure threshold logic
        evaluator = RubricEvaluator(override_conditions=[])
        scores = [
            CriterionScore(criterion=c.name, score=1.5, reasoning="Poor") for c in DEFAULT_CRITERIA
        ]
        decision = evaluator._determine_decision(1.5, scores)
        assert decision == ImprovementDecision.ESCALATE

    def test_override_any_below_threshold(self, evaluator_no_client):
        """Any criterion below 2.0 should trigger SUGGESTION."""
        scores = [
            CriterionScore(criterion="causal_validity", score=1.5, reasoning="Bad"),
            CriterionScore(criterion="actionability", score=4.5, reasoning="Good"),
            CriterionScore(criterion="evidence_chain", score=4.5, reasoning="Good"),
            CriterionScore(criterion="regulatory_awareness", score=4.5, reasoning="Good"),
            CriterionScore(criterion="uncertainty_communication", score=4.5, reasoning="Good"),
        ]
        # Weighted score would be high, but override should trigger
        weighted = evaluator_no_client._calculate_weighted_score(scores)
        decision = evaluator_no_client._determine_decision(weighted, scores)
        assert decision == ImprovementDecision.SUGGESTION


class TestPatternDetection:
    """Test pattern detection logic."""

    def test_detect_low_score_patterns(self, evaluator_no_client):
        """Should detect patterns for scores below 3.0."""
        scores = [
            CriterionScore(criterion="causal_validity", score=2.0, reasoning="Poor causal"),
            CriterionScore(criterion="actionability", score=4.0, reasoning="Good"),
            CriterionScore(criterion="evidence_chain", score=2.5, reasoning="Weak evidence"),
            CriterionScore(criterion="regulatory_awareness", score=4.5, reasoning="Good"),
            CriterionScore(criterion="uncertainty_communication", score=3.5, reasoning="OK"),
        ]
        patterns = evaluator_no_client._detect_patterns(scores)
        assert len(patterns) == 2
        pattern_types = {p.pattern_type for p in patterns}
        assert "low_causal_validity" in pattern_types
        assert "low_evidence_chain" in pattern_types

    def test_no_patterns_for_good_scores(self, evaluator_no_client):
        """Should detect no patterns for all scores >= 3.0."""
        scores = [
            CriterionScore(criterion=c.name, score=4.0, reasoning="Good") for c in DEFAULT_CRITERIA
        ]
        patterns = evaluator_no_client._detect_patterns(scores)
        assert len(patterns) == 0


class TestFallbackEvaluation:
    """Test fallback evaluation behavior."""

    def test_fallback_returns_neutral_scores(self, evaluator_no_client):
        """Fallback should return neutral (3.0) scores."""
        scores, analysis = evaluator_no_client._fallback_evaluation()
        assert len(scores) == 5
        for score in scores:
            assert score.score == 3.0
        assert "Fallback" in analysis


class TestSummarizeEvaluation:
    """Test evaluation summarization."""

    def test_summarize_with_strengths_and_weaknesses(self, evaluator_no_client):
        """Should summarize strengths and weaknesses."""
        scores = [
            CriterionScore(criterion="causal_validity", score=4.5, reasoning="Good"),
            CriterionScore(criterion="actionability", score=2.5, reasoning="Poor"),
            CriterionScore(criterion="evidence_chain", score=4.0, reasoning="Good"),
            CriterionScore(criterion="regulatory_awareness", score=2.0, reasoning="Poor"),
            CriterionScore(criterion="uncertainty_communication", score=3.5, reasoning="OK"),
        ]
        summary = evaluator_no_client._summarize_evaluation(scores)
        assert "Strengths" in summary
        assert "Weaknesses" in summary

    def test_summarize_adequate(self, evaluator_no_client):
        """Should show 'Adequate' when no extreme scores."""
        scores = [
            CriterionScore(criterion=c.name, score=3.5, reasoning="OK") for c in DEFAULT_CRITERIA
        ]
        summary = evaluator_no_client._summarize_evaluation(scores)
        assert "Adequate" in summary


class TestEvaluateAsync:
    """Test async evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_without_client(self, sample_context):
        """Should return fallback evaluation when no client."""
        # Patch environment to ensure no API key
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            evaluator = RubricEvaluator()
            assert evaluator.client is None
            result = await evaluator.evaluate(sample_context)
            assert isinstance(result, RubricEvaluation)
            assert result.weighted_score == 3.0  # All neutral scores
            assert result.decision == ImprovementDecision.SUGGESTION

    @pytest.mark.asyncio
    async def test_evaluate_with_mock_client(self, sample_context):
        """Should parse AI response correctly."""
        evaluator = RubricEvaluator()

        # Mock the Anthropic client
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""```json
{
    "causal_validity": {"score": 4.5, "reasoning": "Good causal language", "evidence": "Uses 'contributed to'"},
    "actionability": {"score": 4.0, "reasoning": "Clear next steps", "evidence": "Mentions campaign"},
    "evidence_chain": {"score": 4.0, "reasoning": "Data referenced", "evidence": "Q3 data"},
    "regulatory_awareness": {"score": 5.0, "reasoning": "Within scope", "evidence": "No medical claims"},
    "uncertainty_communication": {"score": 3.5, "reasoning": "Some hedging", "evidence": "Uses 'suggests'"},
    "overall_analysis": "Strong response with minor improvements needed"
}
```"""
            )
        ]

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        evaluator.client = mock_client

        result = await evaluator.evaluate(sample_context)

        assert isinstance(result, RubricEvaluation)
        assert result.weighted_score > 3.0
        assert result.decision == ImprovementDecision.ACCEPTABLE
        assert "Strong response" in result.overall_analysis


class TestParseEvaluationResponse:
    """Test response parsing."""

    def test_parse_valid_json(self, evaluator_no_client):
        """Should parse valid JSON response."""
        response = """Here is my evaluation:
```json
{
    "causal_validity": {"score": 4, "reasoning": "Good", "evidence": "test"},
    "actionability": {"score": 3, "reasoning": "OK", "evidence": "test"},
    "evidence_chain": {"score": 5, "reasoning": "Great", "evidence": "test"},
    "regulatory_awareness": {"score": 4, "reasoning": "Good", "evidence": "test"},
    "uncertainty_communication": {"score": 3, "reasoning": "OK", "evidence": "test"}
}
```"""
        scores, analysis = evaluator_no_client._parse_evaluation_response(response)
        assert len(scores) == 5
        assert scores[0].score == 4.0

    def test_parse_invalid_json_fallback(self, evaluator_no_client):
        """Should fallback on invalid JSON."""
        response = "This is not valid JSON at all"
        scores, analysis = evaluator_no_client._parse_evaluation_response(response)
        assert len(scores) == 5
        for score in scores:
            assert score.score == 3.0

    def test_parse_clamps_out_of_range_scores(self, evaluator_no_client):
        """Should clamp scores to 1-5 range."""
        response = """
{
    "causal_validity": {"score": 10, "reasoning": "Too high"},
    "actionability": {"score": -5, "reasoning": "Too low"},
    "evidence_chain": {"score": 3, "reasoning": "Normal"},
    "regulatory_awareness": {"score": 4, "reasoning": "Good"},
    "uncertainty_communication": {"score": 3, "reasoning": "OK"}
}"""
        scores, _ = evaluator_no_client._parse_evaluation_response(response)
        score_values = {s.criterion: s.score for s in scores}
        assert score_values["causal_validity"] == 5.0  # Clamped to max
        assert score_values["actionability"] == 1.0  # Clamped to min
