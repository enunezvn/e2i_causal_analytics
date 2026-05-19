"""Tests for Experiment Monitor Memory Hooks.

Limited scope: covers the HybridRetriever wire-in (Phase 2 finishing, issue #373).
Other ExperimentMonitorMemoryHooks behavior is exercised via tests/unit/test_agents/test_experiment_monitor/test_integration.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.experiment_monitor.memory_hooks import (
    ExperimentMonitorMemoryHooks,
    get_experiment_monitor_memory_hooks,
    reset_memory_hooks,
)

# ============================================================================
# HYBRID RETRIEVER WIRE-IN TESTS (Phase 2 finishing, issue #373)
# ============================================================================


class TestGetHybridContext:
    """Tests for get_hybrid_context wire-in to HybridRetriever.

    Phase 2 finishing per .claude/plans/e2i_memory_subsystems_implementation_plan.md
    §Recommended-sequencing item 1. Closes the audit gap that
    experiment_monitor had zero hits for HybridRetriever/hybrid_search.
    """

    def setup_method(self):
        """Reset singleton before each test."""
        reset_memory_hooks()

    @pytest.mark.asyncio
    async def test_get_hybrid_context_calls_hybrid_search_with_freshness_default(self):
        """get_hybrid_context should call hybrid_search with max_staleness=0.0 + agent_name filter."""
        hooks = ExperimentMonitorMemoryHooks()
        sentinel = [MagicMock(spec=["source_id"])]

        with patch("src.rag.retriever.hybrid_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = sentinel
            result = await hooks.get_hybrid_context(query="any SRM alerts?")

        assert result is sentinel
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["query"] == "any SRM alerts?"
        assert call_kwargs["max_staleness"] == 0.0, (
            "Tier 3 agents default to fresh-only retrieval (max_staleness=0.0)"
        )
        assert call_kwargs["filters"]["agent_name"] == "experiment_monitor"

    def test_singleton_access(self):
        """get_experiment_monitor_memory_hooks should return singleton."""
        hooks1 = get_experiment_monitor_memory_hooks()
        hooks2 = get_experiment_monitor_memory_hooks()
        assert hooks1 is hooks2
