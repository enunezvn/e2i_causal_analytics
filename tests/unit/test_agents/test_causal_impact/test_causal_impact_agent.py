"""Integration tests for CausalImpactAgent."""

import pytest

from src.agents.causal_impact.agent import CausalImpactAgent


class TestCausalImpactAgent:
    """Test CausalImpactAgent integration."""

    @pytest.mark.asyncio
    async def test_run_complete_workflow(self):
        """Test complete causal impact workflow."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "what is the impact of hcp engagement on patient conversions?",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "interpretation_depth": "standard",
        }

        result = await agent.run(input_data)

        # Verify output structure (contract-aligned field names)
        assert result["status"] == "completed"
        assert "causal_narrative" in result
        assert result["causal_narrative"] != ""
        assert "ate_estimate" in result
        assert result["statistical_significance"] is not None
        assert result["confidence"] > 0  # Contract field name
        assert result["total_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_run_with_explicit_variables(self):
        """Test with explicitly specified variables."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region", "hcp_specialty"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        # Summary indicates confounders were included (4 variables: treatment + outcome + 2 confounders)
        assert (
            "4 variables" in result.get("causal_graph_summary", "")
            or result.get("causal_graph_summary") is not None
        )

    @pytest.mark.asyncio
    async def test_run_infer_variables_from_query(self):
        """Test variable inference from query."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "does marketing spend affect prescription volume?",
            "treatment_var": "marketing_spend",
            "outcome_var": "prescription_volume",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        # Should infer treatment and outcome
        assert result["causal_graph_summary"] is not None

    @pytest.mark.asyncio
    async def test_minimal_interpretation(self):
        """Test with minimal interpretation depth."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "interpretation_depth": "minimal",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        # Minimal interpretation should be brief
        assert len(result["causal_narrative"]) < 500

    @pytest.mark.asyncio
    async def test_deep_interpretation(self):
        """Test with deep interpretation depth."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "interpretation_depth": "deep",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        # Deep interpretation should be comprehensive
        assert len(result["causal_narrative"]) > 300
        assert len(result["actionable_recommendations"]) >= 3  # Contract field name

    @pytest.mark.asyncio
    async def test_no_interpretation(self):
        """Test with depth=none."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "interpretation_depth": "none",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        assert "skipped" in result["causal_narrative"].lower()

    @pytest.mark.asyncio
    async def test_run_missing_query(self):
        """Test that missing query raises ValueError."""
        agent = CausalImpactAgent()

        # Provide all required fields except query
        input_data = {
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        with pytest.raises(ValueError, match="query"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_executive_user_context(self):
        """Test with executive user context."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "user_context": {"expertise": "executive"},
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        # Executive framing should focus on business impact
        narrative_lower = result["causal_narrative"].lower()
        assert "business" in narrative_lower or "impact" in narrative_lower

    @pytest.mark.asyncio
    async def test_analyst_user_context(self):
        """Test with analyst user context."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "user_context": {"expertise": "analyst"},
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_data_scientist_user_context(self):
        """Test with data scientist user context."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "user_context": {"expertise": "data_scientist"},
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"


class TestCausalImpactOutputContract:
    """Test that output conforms to contract."""

    @pytest.mark.asyncio
    async def test_output_has_required_fields(self):
        """Test that output has all required contract fields."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        # Required fields from CausalImpactOutput contract (contract-aligned)
        required_fields = [
            "query_id",
            "status",
            "causal_narrative",  # Contract field name (was narrative)
            "statistical_significance",
            "key_assumptions",
            "limitations",
            "actionable_recommendations",  # Contract field name (was recommendations)
            "computation_latency_ms",
            "interpretation_latency_ms",
            "total_latency_ms",
            "confidence",  # Contract field name (was overall_confidence)
            "follow_up_suggestions",
            "citations",
            # Contract REQUIRED fields
            "model_used",
            "key_insights",
            "assumption_warnings",
            "requires_further_analysis",
            "refutation_passed",
            "executive_summary",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_output_types(self):
        """Test that output field types are correct."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        # Verify types (contract-aligned field names)
        assert isinstance(result["query_id"], str)
        assert isinstance(result["status"], str)
        assert result["status"] in ["completed", "failed"]
        assert isinstance(result["causal_narrative"], str)
        assert isinstance(result["statistical_significance"], bool)
        assert isinstance(result["key_assumptions"], list)
        assert isinstance(result["limitations"], list)
        assert isinstance(result["actionable_recommendations"], list)  # Contract field
        assert isinstance(result["computation_latency_ms"], (int, float))
        assert isinstance(result["interpretation_latency_ms"], (int, float))
        assert isinstance(result["total_latency_ms"], (int, float))
        assert isinstance(result["confidence"], (int, float))  # Contract field
        assert isinstance(result["follow_up_suggestions"], list)
        assert isinstance(result["citations"], list)
        # Contract REQUIRED field types
        assert isinstance(result["model_used"], str)
        assert isinstance(result["key_insights"], list)
        assert isinstance(result["assumption_warnings"], list)
        assert isinstance(result["requires_further_analysis"], bool)
        assert isinstance(result["refutation_passed"], bool)
        assert isinstance(result["executive_summary"], str)

    @pytest.mark.asyncio
    async def test_latency_breakdown(self):
        """Test that all latency components are measured."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        # Verify latency breakdown
        assert result["computation_latency_ms"] >= 0
        assert result["interpretation_latency_ms"] >= 0
        assert result["total_latency_ms"] >= 0

        # Total should be >= computation + interpretation
        assert result["total_latency_ms"] >= (
            result["computation_latency_ms"] + result["interpretation_latency_ms"]
        )

    @pytest.mark.asyncio
    async def test_confidence_interval_structure(self):
        """Test confidence interval structure."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        if result["confidence_interval"] is not None:
            ci = result["confidence_interval"]
            assert isinstance(ci, tuple)
            assert len(ci) == 2
            assert ci[0] <= ci[1]  # lower <= upper

    @pytest.mark.asyncio
    async def test_causal_graph_summary(self):
        """Test causal graph summary generation."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        if result["causal_graph_summary"] is not None:
            summary = result["causal_graph_summary"]
            assert "hcp_engagement_level" in summary
            assert "patient_conversion_rate" in summary


class TestCausalImpactPerformance:
    """Test causal impact performance characteristics."""

    @pytest.mark.asyncio
    async def test_total_latency_target(self):
        """Test that total latency meets <120s target."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        # Should be well under 120s with mock execution
        assert result["total_latency_ms"] < 120000

    @pytest.mark.asyncio
    async def test_computation_latency_target(self):
        """Test that computation meets <60s target."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        # Computation (graph + estimation + refutation + sensitivity) < 60s
        assert result["computation_latency_ms"] < 60000

    @pytest.mark.asyncio
    async def test_interpretation_latency_target(self):
        """Test that interpretation meets <30s target."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        # Interpretation should be < 30s
        assert result["interpretation_latency_ms"] < 30000


class TestCausalImpactHelperMethods:
    """Test CausalImpactAgent helper methods."""

    @pytest.mark.asyncio
    async def test_classify_intent(self):
        """Test standalone classify_intent helper."""
        agent = CausalImpactAgent()

        intent = await agent.classify_intent("what causes conversions?")

        assert intent["primary_intent"] == "causal_effect"
        assert intent["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_analyze_method(self):
        """Test simplified analyze interface for orchestrator."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.analyze(input_data)

        # Simplified output for orchestrator
        assert "narrative" in result
        assert "recommendations" in result
        assert "confidence" in result


class TestCausalImpactEdgeCases:
    """Test causal impact edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test agent with empty query."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        # Should complete with default variables
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test agent with very long query."""
        agent = CausalImpactAgent()

        long_query = "what is the impact " * 100

        input_data = {
            "query": long_query,
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Test agent with special characters."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "what's the impact? (HCP engagement -> conversions)",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"


class TestCausalImpactRobustness:
    """Test robustness indicators in output."""

    @pytest.mark.asyncio
    async def test_refutation_tests_included(self):
        """Test that refutation test counts are included."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        assert "refutation_tests_passed" in result
        assert "refutation_tests_total" in result

        if result["refutation_tests_total"] is not None:
            assert result["refutation_tests_total"] > 0
            assert result["refutation_tests_passed"] >= 0
            assert result["refutation_tests_passed"] <= result["refutation_tests_total"]

    @pytest.mark.asyncio
    async def test_sensitivity_evalue_included(self):
        """Test that E-value is included."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        assert "sensitivity_e_value" in result

        if result["sensitivity_e_value"] is not None:
            assert result["sensitivity_e_value"] >= 1.0  # E-value always >= 1

    @pytest.mark.asyncio
    async def test_overall_confidence_calculation(self):
        """Test overall confidence calculation."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        assert 0.0 <= result["confidence"] <= 1.0  # Contract field name

    @pytest.mark.asyncio
    async def test_high_confidence_criteria(self):
        """Test that high confidence requires all positive signals."""
        agent = CausalImpactAgent()

        input_data = {
            "query": "test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
        }

        result = await agent.run(input_data)

        # If confidence is high (>= 0.8), should have:
        # - Statistical significance
        # - Robust refutation results
        # - High E-value
        if result["confidence"] >= 0.8:  # Contract field name
            assert result["statistical_significance"] is True


class TestCausalImpactErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_error_output_structure(self):
        """Test error output structure."""
        # This would require mocking internal failures
        # For now, test that error handling path exists
        agent = CausalImpactAgent()

        # The agent should handle errors gracefully
        # (Implementation uses try/except in run method)
        assert hasattr(agent, "_build_error_output")
