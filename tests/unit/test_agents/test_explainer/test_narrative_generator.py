"""Tests for NarrativeGeneratorNode."""

import pytest

from src.agents.explainer.nodes.context_assembler import ContextAssemblerNode
from src.agents.explainer.nodes.deep_reasoner import DeepReasonerNode
from src.agents.explainer.nodes.narrative_generator import NarrativeGeneratorNode


class TestNarrativeGeneratorNode:
    """Tests for the NarrativeGeneratorNode."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_init_deterministic_mode(self):
        """Test node initialization in deterministic mode."""
        node = NarrativeGeneratorNode(use_llm=False)
        assert node.use_llm is False
        assert node.llm is None

    def test_init_llm_mode(self, mock_llm):
        """Test node initialization in LLM mode."""
        node = NarrativeGeneratorNode(use_llm=True, llm=mock_llm)
        assert node.use_llm is True
        assert node.llm is mock_llm

    # ========================================================================
    # HELPER: Get reasoned state
    # ========================================================================

    async def _get_reasoned_state(self, base_state, analysis_results):
        """Helper to get a reasoned state for testing generation."""
        assembler = ContextAssemblerNode()
        state = {**base_state, "analysis_results": analysis_results}
        assembled = await assembler.execute(state)

        reasoner = DeepReasonerNode(use_llm=False)
        return await reasoner.execute(assembled)

    # ========================================================================
    # BASIC EXECUTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_execute_generates_executive_summary(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that executive summary is generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["status"] == "completed"
        assert result["executive_summary"] is not None
        assert len(result["executive_summary"]) > 0

    @pytest.mark.asyncio
    async def test_execute_generates_detailed_explanation(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that detailed explanation is generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["detailed_explanation"] is not None
        assert len(result["detailed_explanation"]) > 0

    @pytest.mark.asyncio
    async def test_execute_generates_narrative_sections(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that narrative sections are generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["narrative_sections"] is not None
        assert len(result["narrative_sections"]) > 0

    # ========================================================================
    # OUTPUT FORMAT TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_narrative_format(self, base_explainer_state, sample_causal_analysis):
        """Test narrative output format."""
        state = {**base_explainer_state, "output_format": "narrative"}
        reasoned = await self._get_reasoned_state(state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Narrative should be prose
        assert result["detailed_explanation"] is not None
        assert len(result["detailed_explanation"]) > 50  # Substantial content

    @pytest.mark.asyncio
    async def test_brief_format(self, base_explainer_state, sample_causal_analysis):
        """Test brief output format."""
        state = {**base_explainer_state, "output_format": "brief"}
        reasoned = await self._get_reasoned_state(state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Brief should have summary
        assert result["executive_summary"] is not None
        assert len(result["executive_summary"]) > 0

    @pytest.mark.asyncio
    async def test_structured_format(self, base_explainer_state, sample_causal_analysis):
        """Test structured output format."""
        state = {**base_explainer_state, "output_format": "structured"}
        reasoned = await self._get_reasoned_state(state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Structured should have sections or explanation
        assert result["detailed_explanation"] is not None

    @pytest.mark.asyncio
    async def test_presentation_format(self, base_explainer_state, sample_causal_analysis):
        """Test presentation output format."""
        state = {**base_explainer_state, "output_format": "presentation"}
        reasoned = await self._get_reasoned_state(state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Presentation should have slides/sections
        assert result["narrative_sections"] is not None
        assert len(result["narrative_sections"]) > 0

    # ========================================================================
    # SECTION GENERATION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_sections_have_required_fields(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that narrative sections have required fields."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        for section in result["narrative_sections"]:
            assert "title" in section
            assert "content" in section

    # ========================================================================
    # VISUAL SUGGESTIONS TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_generates_visual_suggestions(self, base_explainer_state, sample_causal_analysis):
        """Test that visual suggestions are generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["visual_suggestions"] is not None
        assert len(result["visual_suggestions"]) > 0

    @pytest.mark.asyncio
    async def test_visual_suggestions_have_type(self, base_explainer_state, sample_causal_analysis):
        """Test that visual suggestions have type field."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        for visual in result["visual_suggestions"]:
            assert "type" in visual

    # ========================================================================
    # FOLLOW-UP QUESTIONS TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_generates_follow_up_questions(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that follow-up questions are generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["follow_up_questions"] is not None
        assert len(result["follow_up_questions"]) > 0

    @pytest.mark.asyncio
    async def test_follow_up_questions_are_questions(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that follow-ups are actual questions."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        for question in result["follow_up_questions"]:
            # Questions should end with ? or be interrogative
            is_question = (
                question.endswith("?")
                or question.lower().startswith("what")
                or question.lower().startswith("how")
                or question.lower().startswith("why")
                or question.lower().startswith("when")
                or question.lower().startswith("can")
            )
            assert is_question, f"Not a question: {question}"

    # ========================================================================
    # EDGE CASE TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_handles_failed_status(self, base_explainer_state):
        """Test that already-failed state is passed through."""
        node = NarrativeGeneratorNode(use_llm=False)
        state = {**base_explainer_state, "status": "failed"}

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_handles_empty_insights(self, base_explainer_state, sample_causal_analysis):
        """Test handling of empty insights list."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])
        # Override with empty insights
        reasoned["extracted_insights"] = []

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Should still complete
        assert result["status"] == "completed"
        assert result["executive_summary"] is not None

    # ========================================================================
    # LATENCY TRACKING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_tracks_generation_latency(self, base_explainer_state, sample_causal_analysis):
        """Test that generation latency is tracked."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["generation_latency_ms"] >= 0
        assert isinstance(result["generation_latency_ms"], int)

    @pytest.mark.asyncio
    async def test_calculates_total_latency(self, base_explainer_state, sample_causal_analysis):
        """Test that total latency is calculated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Total should include all phases
        total = result["total_latency_ms"]
        assert total >= result["generation_latency_ms"]

    # ========================================================================
    # MODEL TRACKING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_preserves_model_used(self, base_explainer_state, sample_causal_analysis):
        """Test that model used is preserved from reasoning phase."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["model_used"] == "deterministic"

    # ========================================================================
    # CONTENT QUALITY TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_summary_is_reasonable_length(self, base_explainer_state, sample_causal_analysis):
        """Test that executive summary has reasonable length."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Summary should be concise but informative
        summary_length = len(result["executive_summary"])
        assert 10 < summary_length < 5000  # Reasonable bounds

    @pytest.mark.asyncio
    async def test_explanation_has_substance(self, base_explainer_state, sample_causal_analysis):
        """Test that detailed explanation has substance."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Explanation should have meaningful content
        assert len(result["detailed_explanation"]) > 20

    # ========================================================================
    # V4.4: CAUSAL DISCOVERY NARRATIVE TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_gate_decision_translation_executive(self, base_explainer_state):
        """Test gate decision translation for executive audience."""
        node = NarrativeGeneratorNode(use_llm=False)

        # Test accept decision
        result = node._translate_gate_decision("accept", 0.92, "executive")
        assert "high confidence" in result.lower() or "reliable" in result.lower()
        assert "92%" in result or "0.92" in result

        # Test review decision
        result = node._translate_gate_decision("review", 0.65, "executive")
        assert "review" in result.lower() or "validation" in result.lower()

        # Test reject decision
        result = node._translate_gate_decision("reject", 0.3, "executive")
        assert "low confidence" in result.lower() or "validation" in result.lower()

    @pytest.mark.asyncio
    async def test_gate_decision_translation_data_scientist(self, base_explainer_state):
        """Test gate decision translation for data scientist audience."""
        node = NarrativeGeneratorNode(use_llm=False)

        # Test accept decision - should be more technical
        result = node._translate_gate_decision("accept", 0.92, "data_scientist")
        assert "DAG" in result or "structure" in result.lower()
        assert "0.92" in result or "92" in result

        # Test augment decision
        result = node._translate_gate_decision("augment", 0.75, "data_scientist")
        assert "augment" in result.lower() or "domain" in result.lower()

    @pytest.mark.asyncio
    async def test_gate_decision_translation_analyst(self, base_explainer_state):
        """Test gate decision translation for analyst audience."""
        node = NarrativeGeneratorNode(use_llm=False)

        result = node._translate_gate_decision("accept", 0.88, "analyst")
        # Analyst should get balanced explanation
        assert len(result) > 10  # Has meaningful content

    @pytest.mark.asyncio
    async def test_has_discovery_data_with_rankings(self, base_explainer_state):
        """Test _has_discovery_data with causal rankings."""
        node = NarrativeGeneratorNode(use_llm=False)

        state = {
            **base_explainer_state,
            "causal_rankings": [{"feature": "marketing_spend", "rank": 1}],
        }
        assert node._has_discovery_data(state) is True

    @pytest.mark.asyncio
    async def test_has_discovery_data_with_dag(self, base_explainer_state):
        """Test _has_discovery_data with DAG data."""
        node = NarrativeGeneratorNode(use_llm=False)

        state = {
            **base_explainer_state,
            "discovered_dag_adjacency": [[0, 1], [0, 0]],
            "discovered_dag_nodes": ["X", "Y"],
        }
        assert node._has_discovery_data(state) is True

    @pytest.mark.asyncio
    async def test_has_discovery_data_with_gate_decision(self, base_explainer_state):
        """Test _has_discovery_data with gate decision only."""
        node = NarrativeGeneratorNode(use_llm=False)

        state = {
            **base_explainer_state,
            "discovery_gate_decision": "accept",
        }
        assert node._has_discovery_data(state) is True

    @pytest.mark.asyncio
    async def test_has_discovery_data_empty(self, base_explainer_state):
        """Test _has_discovery_data with no discovery data."""
        node = NarrativeGeneratorNode(use_llm=False)

        assert node._has_discovery_data(base_explainer_state) is False

    @pytest.mark.asyncio
    async def test_ranking_comparison_generation(self, base_explainer_state):
        """Test ranking comparison narrative generation."""
        node = NarrativeGeneratorNode(use_llm=False)

        state = {
            **base_explainer_state,
            "causal_rankings": [
                {"feature": "marketing_spend", "rank": 1},
                {"feature": "rep_visits", "rank": 2},
            ],
            "predictive_rankings": [
                {"feature": "rep_visits", "rank": 1},
                {"feature": "marketing_spend", "rank": 2},
            ],
            "rank_correlation": 0.75,
        }

        result = node._generate_ranking_comparison(state, "analyst")
        assert result is not None
        assert "correlation" in result.lower() or "0.75" in result

    @pytest.mark.asyncio
    async def test_divergent_features_explanation(self, base_explainer_state):
        """Test divergent features explanation generation."""
        node = NarrativeGeneratorNode(use_llm=False)

        divergent = ["marketing_spend", "territory_size"]
        state = {
            **base_explainer_state,
            "divergent_features": divergent,
            "user_expertise": "analyst",
        }

        result = node._explain_divergent_features(divergent, state, "analyst")
        assert result is not None
        assert "marketing_spend" in result
        assert "territory_size" in result

    @pytest.mark.asyncio
    async def test_causal_only_features_explanation(self, base_explainer_state):
        """Test causal-only features explanation generation."""
        node = NarrativeGeneratorNode(use_llm=False)

        causal_only = ["regulatory_changes", "competitive_entry"]

        result = node._explain_causal_only_features(causal_only, "executive")
        assert result is not None
        # Should mention the features
        assert "regulatory_changes" in result or "competitive_entry" in result

    @pytest.mark.asyncio
    async def test_latent_confounder_extraction(self, base_explainer_state):
        """Test latent confounder extraction from edge types."""
        node = NarrativeGeneratorNode(use_llm=False)

        state = {
            **base_explainer_state,
            "discovered_dag_edge_types": {
                "marketing_spend->sales": "DIRECTED",
                "rep_visits<->territory_size": "BIDIRECTED",
                "pricing<->demand": "BIDIRECTED",
            },
            "discovered_dag_nodes": [
                "marketing_spend",
                "sales",
                "rep_visits",
                "territory_size",
                "pricing",
                "demand",
            ],
        }

        confounders = node._extract_latent_confounders(state)
        assert len(confounders) == 2
        # Should have the bidirected pairs
        assert any("rep_visits" in str(c) for c in confounders)
        assert any("pricing" in str(c) for c in confounders)

    @pytest.mark.asyncio
    async def test_latent_confounder_explanation_executive(self, base_explainer_state):
        """Test latent confounder explanation for executive."""
        node = NarrativeGeneratorNode(use_llm=False)

        # Confounders are edge keys from _extract_latent_confounders
        confounders = ["rep_visits<->territory_size", "pricing<->demand"]

        result = node._explain_latent_confounders(confounders, "executive")
        assert result is not None
        # Executive should get business-friendly explanation
        assert "unmeasured" in result.lower() or "external" in result.lower()

    @pytest.mark.asyncio
    async def test_latent_confounder_explanation_data_scientist(self, base_explainer_state):
        """Test latent confounder explanation for data scientist."""
        node = NarrativeGeneratorNode(use_llm=False)

        confounders = ["rep_visits<->territory_size"]

        result = node._explain_latent_confounders(confounders, "data_scientist")
        assert result is not None
        # Data scientist should get technical explanation
        assert "bidirected" in result.lower() or "latent" in result.lower()

    @pytest.mark.asyncio
    async def test_causal_discovery_section_generation(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test full causal discovery section generation."""
        node = NarrativeGeneratorNode(use_llm=False)

        state = {
            **base_explainer_state,
            "analysis_results": [sample_causal_analysis],
            "discovery_gate_decision": "accept",
            "discovery_gate_confidence": 0.88,
            "causal_rankings": [
                {"feature": "marketing_spend", "rank": 1, "score": 0.92},
                {"feature": "rep_visits", "rank": 2, "score": 0.78},
            ],
            "predictive_rankings": [
                {"feature": "rep_visits", "rank": 1, "score": 0.85},
                {"feature": "marketing_spend", "rank": 2, "score": 0.72},
            ],
            "rank_correlation": 0.65,
            "divergent_features": ["marketing_spend"],
            "user_expertise": "analyst",
        }

        section = node._generate_causal_discovery_section(state, "analyst")

        assert section is not None
        assert section["section_type"] == "causal_discovery"
        assert "Causal" in section["title"]
        assert len(section["content"]) > 50  # Has substantial content
        # Should have supporting data
        assert section["supporting_data"] is not None
        assert "gate_decision" in section["supporting_data"]

    @pytest.mark.asyncio
    async def test_causal_discovery_section_returns_none_without_data(self, base_explainer_state):
        """Test that causal discovery section returns None without discovery data."""
        node = NarrativeGeneratorNode(use_llm=False)

        section = node._generate_causal_discovery_section(base_explainer_state, "analyst")
        assert section is None

    @pytest.mark.asyncio
    async def test_narrative_includes_causal_discovery_section(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that narrative generation includes causal discovery section when data present."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        # Add discovery data to reasoned state
        reasoned["discovery_gate_decision"] = "accept"
        reasoned["discovery_gate_confidence"] = 0.9
        reasoned["causal_rankings"] = [{"feature": "test", "rank": 1}]

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Find causal discovery section
        causal_sections = [
            s for s in result["narrative_sections"] if s.get("section_type") == "causal_discovery"
        ]
        assert len(causal_sections) == 1
        assert "Causal" in causal_sections[0]["title"]
