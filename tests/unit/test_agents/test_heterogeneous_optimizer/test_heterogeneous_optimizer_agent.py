"""Tests for Heterogeneous Optimizer Agent."""

import pytest
from src.agents.heterogeneous_optimizer.agent import HeterogeneousOptimizerAgent
from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState


class TestHeterogeneousOptimizerAgent:
    """Test HeterogeneousOptimizerAgent."""

    @pytest.mark.asyncio
    async def test_run_complete_workflow(self):
        """Test complete heterogeneous optimizer workflow."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "Which HCP segments respond best to increased engagement?",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty", "region"],
            "effect_modifiers": ["hcp_tenure", "competitive_pressure", "formulary_status"],
            "data_source": "hcp_performance_metrics",
        }

        result = await agent.run(input_data)

        # Verify output structure
        assert "overall_ate" in result
        assert "heterogeneity_score" in result
        assert "high_responders" in result
        assert "low_responders" in result
        assert "policy_recommendations" in result

    @pytest.mark.asyncio
    async def test_input_validation_required_fields(self):
        """Test input validation for required fields."""
        agent = HeterogeneousOptimizerAgent()

        # Missing query
        with pytest.raises(ValueError, match="Missing required field: query"):
            await agent.run({
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "segment_vars": ["segment1"],
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
            })

    @pytest.mark.asyncio
    async def test_input_validation_missing_treatment_var(self):
        """Test missing treatment_var."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="Missing required field: treatment_var"):
            await agent.run({
                "query": "test",
                "outcome_var": "outcome",
                "segment_vars": ["segment1"],
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
            })

    @pytest.mark.asyncio
    async def test_input_validation_missing_outcome_var(self):
        """Test missing outcome_var."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="Missing required field: outcome_var"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "segment_vars": ["segment1"],
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
            })

    @pytest.mark.asyncio
    async def test_input_validation_missing_segment_vars(self):
        """Test missing segment_vars."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="Missing required field: segment_vars"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
            })

    @pytest.mark.asyncio
    async def test_input_validation_missing_effect_modifiers(self):
        """Test missing effect_modifiers."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="Missing required field: effect_modifiers"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "segment_vars": ["segment1"],
                "data_source": "test",
            })

    @pytest.mark.asyncio
    async def test_input_validation_empty_segment_vars(self):
        """Test empty segment_vars list."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="segment_vars must be a non-empty list"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "segment_vars": [],
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
            })

    @pytest.mark.asyncio
    async def test_input_validation_empty_effect_modifiers(self):
        """Test empty effect_modifiers list."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="effect_modifiers must be a non-empty list"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "segment_vars": ["segment1"],
                "effect_modifiers": [],
                "data_source": "test",
            })

    @pytest.mark.asyncio
    async def test_input_validation_n_estimators_range(self):
        """Test n_estimators range validation."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="n_estimators must be an integer between 50 and 500"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "segment_vars": ["segment1"],
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
                "n_estimators": 1000,  # Too high
            })

    @pytest.mark.asyncio
    async def test_input_validation_min_samples_leaf_range(self):
        """Test min_samples_leaf range validation."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="min_samples_leaf must be an integer between 5 and 100"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "segment_vars": ["segment1"],
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
                "min_samples_leaf": 200,  # Too high
            })

    @pytest.mark.asyncio
    async def test_input_validation_significance_level_range(self):
        """Test significance_level range validation."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="significance_level must be a number between 0.01 and 0.10"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "segment_vars": ["segment1"],
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
                "significance_level": 0.5,  # Too high
            })

    @pytest.mark.asyncio
    async def test_input_validation_top_segments_count_range(self):
        """Test top_segments_count range validation."""
        agent = HeterogeneousOptimizerAgent()

        with pytest.raises(ValueError, match="top_segments_count must be an integer between 5 and 50"):
            await agent.run({
                "query": "test",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "segment_vars": ["segment1"],
                "effect_modifiers": ["modifier1"],
                "data_source": "test",
                "top_segments_count": 100,  # Too high
            })

    @pytest.mark.asyncio
    async def test_output_has_required_fields(self):
        """Test output has all required contract fields."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        # Required fields from HeterogeneousOptimizerOutput contract
        required_fields = [
            "overall_ate",
            "heterogeneity_score",
            "high_responders",
            "low_responders",
            "cate_by_segment",
            "policy_recommendations",
            "expected_total_lift",
            "optimal_allocation_summary",
            "feature_importance",
            "executive_summary",
            "key_insights",
            "estimation_latency_ms",
            "analysis_latency_ms",
            "total_latency_ms",
            "confidence",
            "warnings",
            "requires_further_analysis",
            "suggested_next_agent",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_output_field_types(self):
        """Test output field types match contract."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        # Type checks
        assert isinstance(result["overall_ate"], float)
        assert isinstance(result["heterogeneity_score"], float)
        assert isinstance(result["high_responders"], list)
        assert isinstance(result["low_responders"], list)
        assert isinstance(result["cate_by_segment"], dict)
        assert isinstance(result["policy_recommendations"], list)
        assert isinstance(result["expected_total_lift"], float)
        assert isinstance(result["feature_importance"], dict)
        assert isinstance(result["executive_summary"], str)
        assert isinstance(result["key_insights"], list)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["requires_further_analysis"], bool)

    @pytest.mark.asyncio
    async def test_heterogeneity_score_range(self):
        """Test heterogeneity score is in valid range."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        assert 0.0 <= result["heterogeneity_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_range(self):
        """Test confidence is in valid range."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_classify_intent_in_scope(self):
        """Test intent classification for in-scope queries."""
        agent = HeterogeneousOptimizerAgent()

        in_scope_queries = [
            "Which segments respond best?",
            "Identify heterogeneous treatment effects",
            "Who are the high responders?",
            "Optimal treatment allocation",
            "CATE analysis by segment",
        ]

        for query in in_scope_queries:
            intent = agent.classify_intent(query)
            assert intent == "in_scope", f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_classify_intent_out_of_scope(self):
        """Test intent classification for out-of-scope queries."""
        agent = HeterogeneousOptimizerAgent()

        out_of_scope_queries = [
            "What is the weather?",
            "Calculate summary statistics",
            "Show me the data",
        ]

        for query in out_of_scope_queries:
            intent = agent.classify_intent(query)
            assert intent == "out_of_scope", f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_analyze_convenience_method(self):
        """Test analyze convenience method."""
        agent = HeterogeneousOptimizerAgent()

        result = await agent.analyze(
            query="Test query",
            treatment_var="hcp_engagement_frequency",
            outcome_var="trx_total",
            segment_vars=["hcp_specialty"],
            effect_modifiers=["hcp_tenure"],
            data_source="test",
        )

        assert "overall_ate" in result
        assert "heterogeneity_score" in result

    @pytest.mark.asyncio
    async def test_cate_by_segment_structure(self):
        """Test CATE by segment structure."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty", "region"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        cate_by_segment = result["cate_by_segment"]

        # Should have results for each segment variable
        assert "hcp_specialty" in cate_by_segment
        assert "region" in cate_by_segment

        # Each should be a list of CATE results
        for segment_var, results in cate_by_segment.items():
            assert isinstance(results, list)
            if results:
                assert "cate_estimate" in results[0]
                assert "segment_value" in results[0]

    @pytest.mark.asyncio
    async def test_performance_target(self):
        """Test performance meets latency target (<150s)."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        # With mock data, should be well under 150s (150000ms)
        assert result["total_latency_ms"] < 150000

    @pytest.mark.asyncio
    async def test_segment_profile_structure(self):
        """Test segment profile structure in output."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        if result["high_responders"]:
            profile = result["high_responders"][0]
            assert "segment_id" in profile
            assert "responder_type" in profile
            assert "cate_estimate" in profile
            assert "recommendation" in profile

    @pytest.mark.asyncio
    async def test_policy_recommendation_structure(self):
        """Test policy recommendation structure in output."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        if result["policy_recommendations"]:
            rec = result["policy_recommendations"][0]
            assert "segment" in rec
            assert "current_treatment_rate" in rec
            assert "recommended_treatment_rate" in rec
            assert "expected_incremental_outcome" in rec
            assert "confidence" in rec

    @pytest.mark.asyncio
    async def test_treatment_rate_bounds(self):
        """Test treatment rates are bounded between 0 and 1."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        for rec in result["policy_recommendations"]:
            assert 0.0 <= rec["current_treatment_rate"] <= 1.0
            assert 0.0 <= rec["recommended_treatment_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_optional_configuration_defaults(self):
        """Test optional configuration parameters use correct defaults."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
            # Omit optional config
        }

        result = await agent.run(input_data)

        # Should use defaults: n_estimators=100, min_samples_leaf=10, etc.
        assert result is not None


class TestHeterogeneousOptimizerAgentEdgeCases:
    """Test edge cases for heterogeneous optimizer agent."""

    @pytest.mark.asyncio
    async def test_single_segment_variable(self):
        """Test with single segment variable."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        assert len(result["cate_by_segment"]) == 1

    @pytest.mark.asyncio
    async def test_single_effect_modifier(self):
        """Test with single effect modifier."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
        }

        result = await agent.run(input_data)

        assert len(result["feature_importance"]) == 1

    @pytest.mark.asyncio
    async def test_with_filters(self):
        """Test with data filters."""
        agent = HeterogeneousOptimizerAgent()

        input_data = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure"],
            "data_source": "test",
            "filters": {"region": "Northeast"},
        }

        result = await agent.run(input_data)

        # Should run successfully with filters
        assert "overall_ate" in result
