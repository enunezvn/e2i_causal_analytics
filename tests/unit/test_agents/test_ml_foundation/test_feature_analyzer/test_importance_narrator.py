"""Tests for importance narrator node (LLM interpretation)."""

import pytest
from unittest.mock import Mock, patch
from src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator import (
    narrate_importance,
    _prepare_interpretation_context,
    _parse_interpretation_response,
    _interpret_interactions,
    _build_complete_interpretation,
)


@pytest.mark.asyncio
class TestNarrateImportance:
    """Test NL interpretation node."""

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_generates_interpretation(self, mock_anthropic_class):
        """Should generate natural language interpretation using Claude."""
        # Setup
        state = {
            "global_importance_ranked": [
                ("call_frequency", 0.23),
                ("recency_days", 0.18),
                ("competitor_share", 0.15),
            ],
            "feature_directions": {
                "call_frequency": "positive",
                "recency_days": "negative",
                "competitor_share": "negative",
            },
            "top_interactions_raw": [
                ("call_frequency", "recency_days", 0.05),
            ],
            "experiment_id": "exp_001",
            "model_version": "v1",
        }

        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [
            Mock(text='{"executive_summary": "Model relies on engagement metrics", "feature_explanations": {"call_frequency": "Drives prescriptions"}, "key_insights": ["Engagement matters"], "recommendations": ["Focus on high-engagement HCPs"], "cautions": ["Watch for confounders"]}')
        ]
        mock_response.usage = Mock(input_tokens=500, output_tokens=300)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await narrate_importance(state)

        # Assert
        assert "error" not in result
        assert "executive_summary" in result
        assert "feature_explanations" in result
        assert "key_insights" in result
        assert "recommendations" in result
        assert "cautions" in result
        assert "interpretation" in result
        assert result["interpretation_model"] == "claude-sonnet-4-20250514"
        assert result["interpretation_tokens"] == 800  # 500 + 300

    async def test_error_when_missing_shap_results(self):
        """Should return error when SHAP results are missing."""
        state = {
            "experiment_id": "exp_002",
            "global_importance_ranked": [],
        }

        result = await narrate_importance(state)

        assert "error" in result
        assert result["error_type"] == "missing_shap_results"
        assert result["status"] == "failed"

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_generates_interaction_interpretations(self, mock_anthropic_class):
        """Should generate interpretations for feature interactions."""
        # Setup
        state = {
            "global_importance_ranked": [("feat_1", 0.5), ("feat_2", 0.3)],
            "feature_directions": {"feat_1": "positive", "feat_2": "negative"},
            "top_interactions_raw": [
                ("feat_1", "feat_2", 0.6),
                ("feat_1", "feat_3", -0.4),
            ],
            "experiment_id": "exp_003",
        }

        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [
            Mock(text='{"executive_summary": "Summary", "feature_explanations": {"feat_1": "Explanation"}, "key_insights": ["Insight"], "recommendations": ["Rec"], "cautions": ["Caution"]}')
        ]
        mock_response.usage = Mock(input_tokens=400, output_tokens=200)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await narrate_importance(state)

        # Assert
        assert "interaction_interpretations" in result
        assert len(result["interaction_interpretations"]) > 0

        # Check interaction interpretation structure
        interaction = result["interaction_interpretations"][0]
        assert "features" in interaction
        assert "interaction_strength" in interaction
        assert "interpretation" in interaction

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_records_computation_time(self, mock_anthropic_class):
        """Should record interpretation computation time."""
        # Setup
        state = {
            "global_importance_ranked": [("feat_1", 0.5)],
            "feature_directions": {"feat_1": "positive"},
            "top_interactions_raw": [],
            "experiment_id": "exp_004",
        }

        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [
            Mock(text='{"executive_summary": "Summary", "feature_explanations": {}, "key_insights": [], "recommendations": [], "cautions": []}')
        ]
        mock_response.usage = Mock(input_tokens=300, output_tokens=150)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await narrate_importance(state)

        # Assert
        assert "interpretation_time_seconds" in result
        assert result["interpretation_time_seconds"] >= 0


class TestPrepareInterpretationContext:
    """Test context preparation for LLM."""

    def test_prepares_context_with_features(self):
        """Should prepare context string with feature information."""
        global_importance_ranked = [
            ("call_frequency", 0.23),
            ("recency_days", 0.18),
        ]
        feature_directions = {
            "call_frequency": "positive",
            "recency_days": "negative",
        }
        top_interactions_raw = []
        experiment_id = "exp_001"
        model_version = "v1"

        context = _prepare_interpretation_context(
            global_importance_ranked,
            feature_directions,
            top_interactions_raw,
            experiment_id,
            model_version
        )

        assert "Experiment ID" in context
        assert "exp_001" in context
        assert "call_frequency" in context
        assert "0.23" in context
        assert "positive" in context

    def test_includes_interactions_when_present(self):
        """Should include interactions in context when available."""
        global_importance_ranked = [("feat_1", 0.5)]
        feature_directions = {"feat_1": "positive"}
        top_interactions_raw = [
            ("feat_1", "feat_2", 0.6),
        ]
        experiment_id = "exp_002"
        model_version = "v2"

        context = _prepare_interpretation_context(
            global_importance_ranked,
            feature_directions,
            top_interactions_raw,
            experiment_id,
            model_version
        )

        assert "Interaction" in context
        assert "feat_1" in context
        assert "feat_2" in context
        assert "0.6" in context

    def test_handles_empty_interactions(self):
        """Should handle empty interactions gracefully."""
        global_importance_ranked = [("feat_1", 0.5)]
        feature_directions = {"feat_1": "positive"}
        top_interactions_raw = []
        experiment_id = "exp_003"
        model_version = "v3"

        context = _prepare_interpretation_context(
            global_importance_ranked,
            feature_directions,
            top_interactions_raw,
            experiment_id,
            model_version
        )

        # Should still produce valid context
        assert "exp_003" in context
        assert "feat_1" in context


class TestParseInterpretationResponse:
    """Test LLM response parsing."""

    def test_parses_json_from_markdown(self):
        """Should parse JSON from markdown code blocks."""
        response_text = '''```json
{
  "executive_summary": "Summary",
  "feature_explanations": {"feat_1": "Explanation"},
  "key_insights": ["Insight 1"],
  "recommendations": ["Rec 1"],
  "cautions": ["Caution 1"]
}
```'''

        result = _parse_interpretation_response(response_text)

        assert result["executive_summary"] == "Summary"
        assert "feat_1" in result["feature_explanations"]
        assert len(result["key_insights"]) == 1

    def test_parses_plain_json(self):
        """Should parse plain JSON without markdown."""
        response_text = '{"executive_summary": "Summary", "feature_explanations": {}, "key_insights": [], "recommendations": [], "cautions": []}'

        result = _parse_interpretation_response(response_text)

        assert result["executive_summary"] == "Summary"

    def test_handles_parsing_error_gracefully(self):
        """Should handle parsing errors gracefully."""
        response_text = "This is not valid JSON"

        result = _parse_interpretation_response(response_text)

        # Should return fallback structure
        assert "executive_summary" in result
        assert "Interpretation parsing failed" in result["executive_summary"]
        assert "cautions" in result


class TestInterpretInteractions:
    """Test interaction interpretation generation."""

    def test_generates_interaction_interpretations(self):
        """Should generate interpretations for interactions."""
        top_interactions_raw = [
            ("feat_1", "feat_2", 0.6),
            ("feat_3", "feat_4", -0.4),
        ]
        feature_explanations = {
            "feat_1": "Engagement metric",
            "feat_2": "Recency metric",
        }

        interpretations = _interpret_interactions(
            top_interactions_raw,
            feature_explanations
        )

        assert len(interpretations) == 2

        # Check first interpretation (positive interaction)
        assert interpretations[0]["features"] == ["feat_1", "feat_2"]
        assert interpretations[0]["interaction_strength"] == 0.6
        assert "amplify" in interpretations[0]["interpretation"]

        # Check second interpretation (negative interaction)
        assert interpretations[1]["features"] == ["feat_3", "feat_4"]
        assert interpretations[1]["interaction_strength"] == -0.4
        assert "oppose" in interpretations[1]["interpretation"]

    def test_limits_to_top_3_interactions(self):
        """Should limit to top 3 interactions."""
        top_interactions_raw = [
            (f"feat_{i}", f"feat_{i+1}", 0.5 - i * 0.1)
            for i in range(10)
        ]
        feature_explanations = {}

        interpretations = _interpret_interactions(
            top_interactions_raw,
            feature_explanations
        )

        assert len(interpretations) == 3


class TestBuildCompleteInterpretation:
    """Test complete interpretation text building."""

    def test_builds_complete_interpretation(self):
        """Should build complete interpretation text from components."""
        executive_summary = "Model relies on engagement metrics"
        feature_explanations = {
            "call_frequency": "Drives prescriptions",
            "recency_days": "Affects targeting",
        }
        key_insights = ["Engagement matters", "Recency is important"]
        recommendations = ["Focus on high-engagement HCPs"]
        cautions = ["Watch for confounders"]

        interpretation = _build_complete_interpretation(
            executive_summary,
            feature_explanations,
            key_insights,
            recommendations,
            cautions
        )

        assert "Executive Summary" in interpretation
        assert executive_summary in interpretation
        assert "Key Insights" in interpretation
        assert "Engagement matters" in interpretation
        assert "Top Features" in interpretation
        assert "call_frequency" in interpretation
        assert "Recommendations" in interpretation
        assert "Focus on high-engagement HCPs" in interpretation
        assert "Cautions" in interpretation
        assert "Watch for confounders" in interpretation

    def test_handles_empty_sections(self):
        """Should handle empty sections gracefully."""
        executive_summary = "Summary"
        feature_explanations = {}
        key_insights = []
        recommendations = []
        cautions = []

        interpretation = _build_complete_interpretation(
            executive_summary,
            feature_explanations,
            key_insights,
            recommendations,
            cautions
        )

        # Should still have executive summary
        assert "Executive Summary" in interpretation
        assert "Summary" in interpretation
