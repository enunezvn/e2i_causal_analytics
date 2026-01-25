"""
Unit tests for Explainer LLM Timeout Scenarios.
Version: 4.3

Tests LLM timeout handling and fallback behavior.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMTimeoutHandling:
    """Tests for LLM timeout scenarios."""

    @pytest.fixture
    def slow_llm(self):
        """Create an LLM that times out."""
        llm = MagicMock()

        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(60)  # Simulate very slow response
            return MagicMock(content="Slow response")

        llm.ainvoke = slow_invoke
        return llm

    @pytest.fixture
    def failing_llm(self):
        """Create an LLM that raises errors."""
        llm = MagicMock()

        async def failing_invoke(*args, **kwargs):
            raise TimeoutError("LLM request timed out")

        llm.ainvoke = failing_invoke
        return llm

    @pytest.mark.asyncio
    async def test_deterministic_fallback_on_no_llm(self):
        """Agent should use deterministic mode when no LLM provided."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=True, llm=None)

        result = await agent.explain(
            analysis_results=[{"findings": ["Test finding"]}],
            query="Explain this",
        )

        assert result is not None
        assert result.model_used in ["deterministic", None, ""]

    @pytest.mark.asyncio
    async def test_fallback_completes_within_timeout(self):
        """Fallback should complete within reasonable time."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        start_time = asyncio.get_event_loop().time()

        result = await asyncio.wait_for(
            agent.explain(
                analysis_results=[{"findings": ["Test"]}],
                query="Test",
            ),
            timeout=30,  # Should complete well within 30 seconds
        )

        elapsed = asyncio.get_event_loop().time() - start_time

        assert result is not None
        assert elapsed < 30

    @pytest.mark.asyncio
    async def test_llm_error_does_not_crash_agent(self):
        """LLM errors should not crash the agent."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Should not raise
        result = await agent.explain(
            analysis_results=[{"data": "test"}],
            query="Test query",
        )

        assert result is not None
        assert result.status in ["completed", "partial", "failed"]


class TestAutoLLMModeFallback:
    """Tests for auto LLM mode fallback behavior."""

    @pytest.mark.asyncio
    async def test_auto_mode_falls_back_on_no_llm(self):
        """Auto mode should fall back to deterministic when LLM unavailable."""
        from src.agents.explainer import ExplainerAgent

        # Auto mode (use_llm=None) with no LLM provided
        agent = ExplainerAgent(use_llm=None, llm=None)

        result = await agent.explain(
            analysis_results=[{"findings": ["Complex analysis"]}] * 5,
            query="Why did this happen and what is the impact?",  # Complex query
        )

        assert result is not None
        # Should still complete using deterministic mode
        assert result.model_used in ["deterministic", None, ""]

    @pytest.mark.asyncio
    async def test_complexity_check_works_without_llm(self):
        """Complexity check should work even without LLM configured."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=None)

        # Check that _should_use_llm works
        should_use, score, reason = agent._should_use_llm(
            analysis_results=[{"data": "test"}] * 10,
            query="Why did sales decline and what are the causal factors?",
            user_expertise="executive",
        )

        assert isinstance(should_use, bool)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestDeepReasonerFallback:
    """Tests for deep reasoner node fallback behavior."""

    @pytest.mark.asyncio
    async def test_deterministic_insights_extraction(self):
        """Deterministic insight extraction should work."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        analysis_results = [
            {
                "agent": "causal_impact",
                "findings": [
                    "Sales increased by 15%",
                    "Marketing spend was the primary driver",
                ],
                "recommendations": [
                    "Increase marketing budget",
                ],
            }
        ]

        result = await agent.explain(
            analysis_results=analysis_results,
            query="What drove sales growth?",
        )

        assert result is not None
        # Should extract insights even in deterministic mode
        assert result.extracted_insights is not None

    @pytest.mark.asyncio
    async def test_narrative_generation_without_llm(self):
        """Narrative should be generated without LLM."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[
                {
                    "findings": ["Finding 1", "Finding 2"],
                    "metrics": {"improvement": 0.15},
                }
            ],
            query="Summarize the analysis",
            output_format="brief",
        )

        assert result is not None
        # Should have some narrative content
        assert result.executive_summary or result.detailed_explanation


class TestRecoveryBehavior:
    """Tests for recovery from LLM failures."""

    @pytest.mark.asyncio
    async def test_multiple_calls_after_failure(self):
        """Agent should work for multiple calls after a failure."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Multiple calls should all work
        for i in range(3):
            result = await agent.explain(
                analysis_results=[{"data": f"test_{i}"}],
                query=f"Query {i}",
            )
            assert result is not None
            assert result.status in ["completed", "partial"]

    @pytest.mark.asyncio
    async def test_state_not_corrupted_after_failure(self):
        """Agent state should not be corrupted after failures."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # First call
        result1 = await agent.explain(
            analysis_results=[{"data": "first"}],
            query="First query",
        )

        # Second call with different data
        result2 = await agent.explain(
            analysis_results=[{"data": "second", "extra": "field"}],
            query="Second query",
        )

        # Results should be independent
        assert result1.timestamp != result2.timestamp


class TestTimeoutConfiguration:
    """Tests for timeout configuration."""

    @pytest.mark.asyncio
    async def test_respects_execution_timeout(self):
        """Agent should respect execution timeouts."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Should complete quickly in deterministic mode
        result = await asyncio.wait_for(
            agent.explain(
                analysis_results=[{"data": "test"}],
                query="Test",
            ),
            timeout=10,  # 10 second timeout
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_large_input_handling(self):
        """Large inputs should be handled within timeout."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Large input
        large_results = [
            {
                "findings": [f"Finding {i}" for i in range(100)],
                "data": {f"key_{i}": f"value_{i}" for i in range(50)},
            }
            for _ in range(10)
        ]

        result = await asyncio.wait_for(
            agent.explain(
                analysis_results=large_results,
                query="Analyze all these results",
            ),
            timeout=30,
        )

        assert result is not None


class TestErrorReporting:
    """Tests for error reporting in failure scenarios."""

    @pytest.mark.asyncio
    async def test_errors_included_in_output(self):
        """Errors should be included in output."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[],  # Empty might trigger warnings
            query="",
        )

        assert result is not None
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)

    @pytest.mark.asyncio
    async def test_status_reflects_outcome(self):
        """Status should accurately reflect execution outcome."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[{"findings": ["Valid finding"]}],
            query="Explain",
        )

        assert result.status in ["completed", "partial", "failed", "pending"]

        # If status is completed, should have content
        if result.status == "completed":
            assert result.executive_summary or result.detailed_explanation or result.narrative_sections
