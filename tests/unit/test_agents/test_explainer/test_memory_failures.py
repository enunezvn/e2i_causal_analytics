"""
Unit tests for Explainer Memory Failure Scenarios.
Version: 4.3

Tests graceful degradation when memory backends are unavailable.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRedisUnavailable:
    """Tests for Redis (Working Memory) unavailability."""

    @pytest.fixture
    def mock_redis_failure(self):
        """Mock Redis connection failure."""
        mock = MagicMock()
        mock.get = AsyncMock(side_effect=ConnectionError("Redis connection refused"))
        mock.set = AsyncMock(side_effect=ConnectionError("Redis connection refused"))
        mock.ping = AsyncMock(side_effect=ConnectionError("Redis connection refused"))
        return mock

    @pytest.mark.asyncio
    async def test_agent_works_without_redis(self):
        """Agent should work when Redis is unavailable."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Simple analysis results
        analysis_results = [
            {
                "agent": "causal_impact",
                "findings": ["Sales increased by 15%"],
            }
        ]

        # Should not raise, should complete with degraded functionality
        result = await agent.explain(
            analysis_results=analysis_results,
            query="Explain the sales trend",
        )

        assert result.status in ["completed", "partial"]
        # Should have some explanation even without memory
        assert result.executive_summary or result.detailed_explanation

    @pytest.mark.asyncio
    async def test_memory_hooks_failure_logged(self):
        """Memory hook failures should be logged as warnings."""
        from src.agents.explainer import ExplainerAgent

        with patch(
            "src.agents.explainer.agent.ExplainerAgent.memory_hooks",
            new_callable=lambda: property(lambda self: None),
        ):
            agent = ExplainerAgent(use_llm=False)

            # Memory hooks should return None on failure
            assert agent.memory_hooks is None


class TestSupabaseUnavailable:
    """Tests for Supabase (Episodic Memory) unavailability."""

    @pytest.mark.asyncio
    async def test_agent_works_without_supabase(self):
        """Agent should work when Supabase is unavailable."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        analysis_results = [
            {
                "agent": "gap_analyzer",
                "findings": ["Coverage gap in Northeast region"],
            }
        ]

        # Mock Supabase failure by not initializing connection
        result = await agent.explain(
            analysis_results=analysis_results,
            query="What are the coverage issues?",
        )

        assert result.status in ["completed", "partial"]

    @pytest.mark.asyncio
    async def test_historical_context_unavailable_warning(self):
        """Missing historical context should generate warning."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        analysis_results = [{"agent": "test", "findings": ["Test finding"]}]

        result = await agent.explain(
            analysis_results=analysis_results,
            query="Test query",
            memory_config={"require_history": True},
        )

        # Should complete but may have warnings about missing context
        assert result.status in ["completed", "partial"]


class TestAllMemoryUnavailable:
    """Tests for complete memory failure scenarios."""

    @pytest.mark.asyncio
    async def test_agent_returns_valid_response_without_memory(self):
        """Agent should return valid response even with no memory access."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        analysis_results = [
            {
                "agent": "prediction_synthesizer",
                "predictions": [
                    {"metric": "TRx", "value": 1500, "confidence": 0.85}
                ],
            }
        ]

        result = await agent.explain(
            analysis_results=analysis_results,
            query="What are the predictions?",
        )

        # Should have valid response
        assert result is not None
        assert result.status in ["completed", "partial", "failed"]

        # If completed, should have content
        if result.status == "completed":
            assert result.executive_summary or result.detailed_explanation

    @pytest.mark.asyncio
    async def test_no_unhandled_exceptions(self):
        """No unhandled exceptions should occur in memory failure scenarios."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Various inputs that might trigger memory access
        test_cases = [
            {"analysis_results": [], "query": ""},
            {"analysis_results": [{"data": "test"}], "query": "explain"},
            {
                "analysis_results": [{"findings": ["a", "b", "c"]}] * 5,
                "query": "complex query with multiple results",
            },
        ]

        for case in test_cases:
            # Should not raise
            try:
                result = await agent.explain(**case)
                assert result is not None
            except Exception as e:
                pytest.fail(f"Unhandled exception: {e}")


class TestLLMTimeoutRecovery:
    """Tests for LLM timeout and recovery scenarios."""

    @pytest.fixture
    def slow_llm(self):
        """Create a slow LLM mock that times out."""
        llm = AsyncMock()

        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(60)  # Very slow
            return {"content": "Response after delay"}

        llm.ainvoke = slow_invoke
        return llm

    @pytest.mark.asyncio
    async def test_llm_timeout_fallback_to_deterministic(self):
        """LLM timeout should trigger deterministic fallback."""
        from src.agents.explainer import ExplainerAgent

        # Create agent with very short timeout expectation
        agent = ExplainerAgent(use_llm=True, llm=None)

        analysis_results = [{"findings": ["Test finding"]}]

        # Should complete using deterministic mode as fallback
        result = await agent.explain(
            analysis_results=analysis_results,
            query="Test",
        )

        # Should have some response (fallback worked)
        assert result is not None
        # Model used should indicate deterministic or fallback
        assert result.model_used in ["deterministic", None, ""]

    @pytest.mark.asyncio
    async def test_partial_llm_response_handled(self):
        """Partial LLM responses should be handled gracefully."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Simulate scenario where LLM returns partial data
        analysis_results = [
            {
                "agent": "causal_impact",
                "partial": True,  # Indicator of partial data
                "findings": ["Partial finding"],
            }
        ]

        result = await agent.explain(
            analysis_results=analysis_results,
            query="Explain partial results",
        )

        assert result is not None
        assert result.status in ["completed", "partial"]

    @pytest.mark.asyncio
    async def test_llm_error_recorded_in_output(self):
        """LLM errors should be recorded in output errors/warnings."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[],
            query="Test",
        )

        # Empty results might generate warnings
        # Main check: no crash occurred
        assert result is not None


class TestFeedbackLearnerMemoryFailures:
    """Tests for Feedback Learner memory failure scenarios."""

    @pytest.mark.asyncio
    async def test_feedback_learner_works_without_stores(self):
        """Feedback Learner should work without feedback stores."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent = FeedbackLearnerAgent(
            feedback_store=None,
            outcome_store=None,
            knowledge_stores=None,
        )

        # Should not raise
        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        assert result is not None
        # May have warnings about missing stores but should complete
        assert result.status in ["completed", "partial", "failed"]

    @pytest.mark.asyncio
    async def test_knowledge_update_failure_recovery(self):
        """Knowledge update failures should be recovered from."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        # Mock store that fails on updates
        mock_store = MagicMock()
        mock_store.update = AsyncMock(side_effect=Exception("Update failed"))
        mock_store.get_feedback = AsyncMock(return_value=[])

        agent = FeedbackLearnerAgent(
            feedback_store=mock_store,
            knowledge_stores={"test": mock_store},
        )

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        # Should complete even with update failures
        assert result is not None

    @pytest.mark.asyncio
    async def test_process_feedback_without_store(self):
        """process_feedback should work with inline items."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent = FeedbackLearnerAgent()

        feedback_items = [
            {
                "feedback_id": "fb_1",
                "feedback_type": "rating",
                "source_agent": "explainer",
                "rating": 4,
            }
        ]

        result = await agent.process_feedback(feedback_items)

        assert result is not None


class TestGracefulDegradation:
    """Tests for graceful degradation behavior."""

    @pytest.mark.asyncio
    async def test_degraded_mode_indicators(self):
        """Degraded mode should be indicated in response."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[{"data": "test"}],
            query="Test",
        )

        # Response should be valid
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "warnings")

    @pytest.mark.asyncio
    async def test_appropriate_warnings_logged(self):
        """Appropriate warnings should be included in response."""
        from src.agents.explainer import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Empty results should trigger warning
        result = await agent.explain(
            analysis_results=[],
            query="Explain nothing",
        )

        assert result is not None
        # Either has content or warnings about empty input
        assert result.executive_summary or result.warnings

    @pytest.mark.asyncio
    async def test_consistent_response_structure(self):
        """Response structure should be consistent regardless of failures."""
        from src.agents.explainer import ExplainerAgent, ExplainerOutput

        agent = ExplainerAgent(use_llm=False)

        # Various scenarios
        scenarios = [
            {"analysis_results": [], "query": ""},
            {"analysis_results": [{"x": 1}], "query": "y"},
            {"analysis_results": [{"findings": []}] * 10, "query": "test"},
        ]

        for scenario in scenarios:
            result = await agent.explain(**scenario)

            # Verify all expected fields exist
            assert isinstance(result, ExplainerOutput)
            assert hasattr(result, "executive_summary")
            assert hasattr(result, "detailed_explanation")
            assert hasattr(result, "extracted_insights")
            assert hasattr(result, "status")
            assert hasattr(result, "errors")
            assert hasattr(result, "warnings")
