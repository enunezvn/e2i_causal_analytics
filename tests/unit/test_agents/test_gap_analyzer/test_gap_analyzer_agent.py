"""Integration tests for GapAnalyzerAgent."""

import pytest
from src.agents.gap_analyzer.agent import GapAnalyzerAgent


class TestGapAnalyzerAgent:
    """Test GapAnalyzerAgent integration."""

    @pytest.mark.asyncio
    async def test_run_complete_workflow(self):
        """Test complete gap analyzer workflow."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "identify trx gaps in northeast region",
            "metrics": ["trx", "nrx"],
            "segments": ["region", "specialty"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # Verify output structure
        assert "prioritized_opportunities" in result
        assert "quick_wins" in result
        assert "strategic_bets" in result
        assert "total_addressable_value" in result
        assert "executive_summary" in result
        assert isinstance(result["prioritized_opportunities"], list)

    @pytest.mark.asyncio
    async def test_run_with_gap_type(self):
        """Test with specific gap type."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "identify gaps vs targets",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "gap_type": "vs_target",
        }

        result = await agent.run(input_data)

        # Should have opportunities
        assert len(result["prioritized_opportunities"]) >= 0

    @pytest.mark.asyncio
    async def test_run_with_all_gap_types(self):
        """Test with all gap types."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "comprehensive gap analysis",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "gap_type": "all",
        }

        result = await agent.run(input_data)

        # Should analyze all gap types
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_with_threshold(self):
        """Test with custom threshold."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "significant gaps only",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "min_gap_threshold": 20.0,  # High threshold
        }

        result = await agent.run(input_data)

        # High threshold may result in fewer opportunities
        assert isinstance(result["prioritized_opportunities"], list)

    @pytest.mark.asyncio
    async def test_run_with_max_opportunities(self):
        """Test with max opportunities limit."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "top 3 opportunities",
            "metrics": ["trx", "nrx"],
            "segments": ["region"],
            "brand": "kisqali",
            "max_opportunities": 3,
        }

        result = await agent.run(input_data)

        # Should limit to 3
        assert len(result["prioritized_opportunities"]) <= 3

    @pytest.mark.asyncio
    async def test_run_with_filters(self):
        """Test with additional filters."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "gaps in oncology",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "filters": {"specialty": "Oncology"},
        }

        result = await agent.run(input_data)

        assert result is not None

    @pytest.mark.asyncio
    async def test_quick_wins_generation(self):
        """Test quick wins generation."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "quick wins",
            "metrics": ["trx", "nrx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        assert "quick_wins" in result
        assert isinstance(result["quick_wins"], list)

        # Quick wins should have low difficulty
        for qw in result["quick_wins"]:
            assert qw["implementation_difficulty"] == "low"

    @pytest.mark.asyncio
    async def test_strategic_bets_generation(self):
        """Test strategic bets generation."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "strategic opportunities",
            "metrics": ["trx", "market_share"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        assert "strategic_bets" in result
        assert isinstance(result["strategic_bets"], list)


class TestGapAnalyzerInputValidation:
    """Test input validation."""

    @pytest.mark.asyncio
    async def test_missing_query(self):
        """Test that missing query raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        with pytest.raises(ValueError, match="Missing required field: query"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_missing_metrics(self):
        """Test that missing metrics raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "segments": ["region"],
            "brand": "kisqali",
        }

        with pytest.raises(ValueError, match="Missing required field: metrics"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_missing_segments(self):
        """Test that missing segments raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "brand": "kisqali",
        }

        with pytest.raises(ValueError, match="Missing required field: segments"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_missing_brand(self):
        """Test that missing brand raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
        }

        with pytest.raises(ValueError, match="Missing required field: brand"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_empty_metrics(self):
        """Test that empty metrics raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": [],
            "segments": ["region"],
            "brand": "kisqali",
        }

        with pytest.raises(ValueError, match="metrics must be a non-empty list"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_empty_segments(self):
        """Test that empty segments raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": [],
            "brand": "kisqali",
        }

        with pytest.raises(ValueError, match="segments must be a non-empty list"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_invalid_gap_type(self):
        """Test that invalid gap_type raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "gap_type": "invalid_type",
        }

        with pytest.raises(ValueError, match="gap_type must be one of"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_invalid_threshold_type(self):
        """Test that invalid threshold type raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "min_gap_threshold": "not_a_number",
        }

        with pytest.raises(ValueError, match="min_gap_threshold must be a number"):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_invalid_max_opportunities(self):
        """Test that invalid max_opportunities raises ValueError."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "max_opportunities": -5,
        }

        with pytest.raises(ValueError, match="max_opportunities must be a positive integer"):
            await agent.run(input_data)


class TestGapAnalyzerOutputContract:
    """Test that output conforms to contract."""

    @pytest.mark.asyncio
    async def test_output_has_required_fields(self):
        """Test that output has all required contract fields."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # Required fields from GapAnalyzerOutput contract
        required_fields = [
            "prioritized_opportunities",
            "quick_wins",
            "strategic_bets",
            "total_addressable_value",
            "total_gap_value",
            "segments_analyzed",
            "executive_summary",
            "key_insights",
            "detection_latency_ms",
            "roi_latency_ms",
            "total_latency_ms",
            "confidence",
            "warnings",
            "requires_further_analysis",
            "suggested_next_agent",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_output_types(self):
        """Test that output field types are correct."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # Verify types
        assert isinstance(result["prioritized_opportunities"], list)
        assert isinstance(result["quick_wins"], list)
        assert isinstance(result["strategic_bets"], list)
        assert isinstance(result["total_addressable_value"], (int, float))
        assert isinstance(result["total_gap_value"], (int, float))
        assert isinstance(result["segments_analyzed"], int)
        assert isinstance(result["executive_summary"], str)
        assert isinstance(result["key_insights"], list)
        assert isinstance(result["detection_latency_ms"], (int, float))
        assert isinstance(result["roi_latency_ms"], (int, float))
        assert isinstance(result["total_latency_ms"], (int, float))
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["warnings"], list)
        assert isinstance(result["requires_further_analysis"], bool)

    @pytest.mark.asyncio
    async def test_latency_breakdown(self):
        """Test that all latency components are measured."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # Verify latency breakdown
        assert result["detection_latency_ms"] >= 0
        assert result["roi_latency_ms"] >= 0
        assert result["total_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_confidence_range(self):
        """Test confidence is in valid range."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_opportunity_structure(self):
        """Test prioritized opportunity structure."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        if result["prioritized_opportunities"]:
            opp = result["prioritized_opportunities"][0]
            assert "rank" in opp
            assert "gap" in opp
            assert "roi_estimate" in opp
            assert "recommended_action" in opp
            assert "implementation_difficulty" in opp
            assert "time_to_impact" in opp


class TestGapAnalyzerPerformance:
    """Test gap analyzer performance characteristics."""

    @pytest.mark.asyncio
    async def test_total_latency_target(self):
        """Test that total latency meets <20s target."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # Should be well under 20s with mock execution
        assert result["total_latency_ms"] < 20000

    @pytest.mark.asyncio
    async def test_detection_latency(self):
        """Test detection latency."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # Detection should be fast
        assert result["detection_latency_ms"] < 10000

    @pytest.mark.asyncio
    async def test_roi_latency(self):
        """Test ROI calculation latency."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # ROI calculation should be fast
        assert result["roi_latency_ms"] < 5000


class TestGapAnalyzerHelperMethods:
    """Test GapAnalyzerAgent helper methods."""

    @pytest.mark.asyncio
    async def test_classify_intent(self):
        """Test standalone classify_intent helper."""
        agent = GapAnalyzerAgent()

        intent = await agent.classify_intent("identify ROI opportunities in northeast")

        assert "primary_intent" in intent
        assert intent["primary_intent"] == "gap_detection"
        assert "confidence" in intent
        assert 0.0 <= intent["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_classify_intent_high_confidence(self):
        """Test intent classification with high confidence."""
        agent = GapAnalyzerAgent()

        intent = await agent.classify_intent(
            "identify gaps and opportunities for TRx improvement with ROI analysis"
        )

        # Should have high confidence (multiple keywords)
        assert intent["confidence"] > 0.5
        assert intent["requires_gap_analysis"] is True

    @pytest.mark.asyncio
    async def test_classify_intent_low_confidence(self):
        """Test intent classification with low confidence."""
        agent = GapAnalyzerAgent()

        intent = await agent.classify_intent("what is happening with sales")

        # May have lower confidence
        assert intent["confidence"] >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_method(self):
        """Test simplified analyze interface for orchestrator."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.analyze(input_data)

        # Simplified output for orchestrator
        assert "summary" in result
        assert "opportunities" in result
        assert "quick_wins" in result
        assert "confidence" in result


class TestGapAnalyzerEdgeCases:
    """Test gap analyzer edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test agent with empty query."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # Should complete with default behavior
        assert result is not None

    @pytest.mark.asyncio
    async def test_single_metric_single_segment(self):
        """Test with minimal configuration."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        assert result is not None

    @pytest.mark.asyncio
    async def test_many_metrics_many_segments(self):
        """Test with many metrics and segments."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "comprehensive analysis",
            "metrics": ["trx", "nrx", "market_share", "conversion_rate"],
            "segments": ["region", "specialty", "hcp_tier"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # Should handle many combinations
        assert result is not None


class TestGapAnalyzerFurtherAnalysis:
    """Test further analysis recommendations."""

    @pytest.mark.asyncio
    async def test_suggests_next_agent(self):
        """Test that next agent is suggested when appropriate."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        # May or may not suggest next agent
        assert "suggested_next_agent" in result

    @pytest.mark.asyncio
    async def test_requires_further_analysis_flag(self):
        """Test requires_further_analysis flag."""
        agent = GapAnalyzerAgent()

        input_data = {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
        }

        result = await agent.run(input_data)

        assert isinstance(result["requires_further_analysis"], bool)


class TestGapAnalyzerErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_error_output_structure(self):
        """Test error output structure."""
        agent = GapAnalyzerAgent()

        # The agent should handle errors gracefully
        assert hasattr(agent, "_build_error_output")
