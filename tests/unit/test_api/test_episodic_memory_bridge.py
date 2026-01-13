"""
Unit tests for E2I Chatbot Episodic Memory Bridge.

Tests the significance scoring and episodic memory bridge functionality
that saves significant chatbot interactions to long-term episodic memory.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chatbot_graph import (
    SIGNIFICANCE_THRESHOLD,
    SIGNIFICANT_INTENTS,
    INTENT_TO_EVENT_TYPE,
    _calculate_significance_score,
    _save_to_episodic_memory,
)
from src.api.routes.chatbot_state import IntentType


class TestSignificanceScoring:
    """Tests for _calculate_significance_score function."""

    def test_baseline_score_is_zero(self):
        """Test that empty state has zero significance."""
        state = {}
        score = _calculate_significance_score(state)
        assert score == 0.0

    def test_tool_usage_adds_significance(self):
        """Test that tool usage increases significance score."""
        # One tool result
        state = {"tool_results": [{"tool": "e2i_data_query_tool"}]}
        score = _calculate_significance_score(state)
        assert score == 0.3

        # Two tool results
        state = {"tool_results": [{"tool": "tool1"}, {"tool": "tool2"}]}
        score = _calculate_significance_score(state)
        assert score == 0.6

        # Three tool results (should cap at 0.6)
        state = {"tool_results": [{"tool": "t1"}, {"tool": "t2"}, {"tool": "t3"}]}
        score = _calculate_significance_score(state)
        assert score == 0.6

    def test_significant_intent_adds_score(self):
        """Test that significant intents add to score."""
        for intent in SIGNIFICANT_INTENTS:
            state = {"intent": intent}
            score = _calculate_significance_score(state)
            assert score == 0.25, f"Expected 0.25 for intent {intent}"

    def test_non_significant_intent_no_score(self):
        """Test that non-significant intents don't add score."""
        state = {"intent": IntentType.GREETING}
        score = _calculate_significance_score(state)
        assert score == 0.0

        state = {"intent": IntentType.HELP}
        score = _calculate_significance_score(state)
        assert score == 0.0

    def test_brand_context_adds_score(self):
        """Test that brand context increases significance."""
        state = {"brand_context": "Kisqali"}
        score = _calculate_significance_score(state)
        assert score == 0.15

    def test_kpi_in_metadata_adds_score(self):
        """Test that KPI mention in metadata adds score."""
        state = {"metadata": {"kpi_name": "TRx"}}
        score = _calculate_significance_score(state)
        assert score == 0.15

    def test_rag_context_adds_score(self):
        """Test that RAG context retrieval adds significance."""
        state = {"rag_context": [{"doc": "some_context"}]}
        score = _calculate_significance_score(state)
        assert score == 0.1

    def test_long_response_adds_score(self):
        """Test that long responses add significance."""
        state = {"response_text": "x" * 501}  # Over 500 chars
        score = _calculate_significance_score(state)
        assert score == 0.1

        # Short response should not add score
        state = {"response_text": "short"}
        score = _calculate_significance_score(state)
        assert score == 0.0

    def test_combined_factors_accumulate(self):
        """Test that multiple factors accumulate correctly."""
        state = {
            "tool_results": [{"tool": "causal_analysis_tool"}],  # +0.3
            "intent": IntentType.CAUSAL_ANALYSIS,  # +0.25
            "brand_context": "Kisqali",  # +0.15
            "rag_context": [{"doc": "context"}],  # +0.1
            "response_text": "x" * 600,  # +0.1
        }
        score = _calculate_significance_score(state)
        assert score == 0.9

    def test_score_capped_at_one(self):
        """Test that score never exceeds 1.0."""
        state = {
            "tool_results": [{"t": "1"}, {"t": "2"}, {"t": "3"}],  # +0.6
            "intent": IntentType.CAUSAL_ANALYSIS,  # +0.25
            "brand_context": "Kisqali",  # +0.15
            "rag_context": [{"doc": "context"}],  # +0.1
            "response_text": "x" * 600,  # +0.1 = 1.2 total
        }
        score = _calculate_significance_score(state)
        assert score == 1.0

    def test_threshold_is_reasonable(self):
        """Test that the significance threshold makes sense."""
        # Threshold should be 0.6
        assert SIGNIFICANCE_THRESHOLD == 0.6

        # A simple KPI query with brand should pass
        state = {
            "intent": IntentType.KPI_QUERY,  # +0.25
            "brand_context": "Kisqali",  # +0.15
            "rag_context": [{"doc": "x"}],  # +0.1
            "response_text": "x" * 600,  # +0.1
        }
        score = _calculate_significance_score(state)
        assert score >= SIGNIFICANCE_THRESHOLD

        # A greeting should not pass
        state = {"intent": IntentType.GREETING}
        score = _calculate_significance_score(state)
        assert score < SIGNIFICANCE_THRESHOLD


class TestIntentToEventTypeMapping:
    """Tests for event type mapping from intents."""

    def test_causal_analysis_maps_to_causal_discovery(self):
        """Test causal analysis intent maps to causal_discovery event."""
        assert INTENT_TO_EVENT_TYPE[IntentType.CAUSAL_ANALYSIS] == "causal_discovery"

    def test_kpi_query_maps_to_user_query(self):
        """Test KPI query intent maps to user_query event."""
        assert INTENT_TO_EVENT_TYPE[IntentType.KPI_QUERY] == "user_query"

    def test_recommendation_maps_to_agent_action(self):
        """Test recommendation intent maps to agent_action event."""
        assert INTENT_TO_EVENT_TYPE[IntentType.RECOMMENDATION] == "agent_action"

    def test_search_maps_to_user_query(self):
        """Test search intent maps to user_query event."""
        assert INTENT_TO_EVENT_TYPE[IntentType.SEARCH] == "user_query"


class TestSaveToEpisodicMemory:
    """Tests for _save_to_episodic_memory function."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
    async def test_saves_memory_with_correct_event_type(self, mock_insert):
        """Test that memory is saved with correct event type."""
        mock_insert.return_value = "memory-id-123"

        state = {
            "query": "What caused TRx to drop for Kisqali?",
            "session_id": "user~session-123",
            "intent": IntentType.CAUSAL_ANALYSIS,
            "brand_context": "Kisqali",
            "region_context": None,
            "tool_results": [],
            "metadata": {},
        }
        tool_calls = [{"tool_name": "causal_analysis_tool", "args": {}}]

        memory_id = await _save_to_episodic_memory(
            state=state,
            response_text="TRx dropped due to market dynamics.",
            tool_calls=tool_calls,
            significance_score=0.75,
        )

        assert memory_id == "memory-id-123"
        mock_insert.assert_called_once()

        # Check the memory input
        call_args = mock_insert.call_args
        memory_input = call_args.kwargs["memory"]
        assert memory_input.event_type == "causal_discovery"
        assert memory_input.event_subtype == "chatbot_causal_query"
        assert memory_input.importance_score == 0.75

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
    async def test_includes_brand_in_entities(self, mock_insert):
        """Test that brand context is included in entities."""
        mock_insert.return_value = "memory-id-456"

        state = {
            "query": "What is TRx?",
            "session_id": "user~session-456",
            "intent": IntentType.KPI_QUERY,
            "brand_context": "Kisqali",
            "region_context": "northeast",
            "tool_results": [],
            "metadata": {},
        }

        await _save_to_episodic_memory(
            state=state,
            response_text="TRx is total prescriptions.",
            tool_calls=[],
            significance_score=0.65,
        )

        call_args = mock_insert.call_args
        memory_input = call_args.kwargs["memory"]
        assert memory_input.entities["brands"] == ["Kisqali"]
        assert memory_input.entities["regions"] == ["northeast"]

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
    async def test_truncates_long_query_in_description(self, mock_insert):
        """Test that long queries are truncated in description."""
        mock_insert.return_value = "memory-id-789"

        long_query = "x" * 300  # Over 200 chars
        state = {
            "query": long_query,
            "session_id": "user~session-789",
            "intent": IntentType.SEARCH,
            "brand_context": None,
            "region_context": None,
            "tool_results": [],
            "metadata": {},
        }

        await _save_to_episodic_memory(
            state=state,
            response_text="Results found.",
            tool_calls=[],
            significance_score=0.6,
        )

        call_args = mock_insert.call_args
        memory_input = call_args.kwargs["memory"]
        assert len(memory_input.description) < 300
        assert "..." in memory_input.description

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
    async def test_handles_data_query_tool(self, mock_insert):
        """Test data query tool sets correct subtype."""
        mock_insert.return_value = "memory-id-data"

        state = {
            "query": "Show me KPI data",
            "session_id": "user~session-data",
            "intent": IntentType.KPI_QUERY,
            "brand_context": None,
            "region_context": None,
            "tool_results": [{"result": "data"}],
            "metadata": {},
        }
        tool_calls = [{"tool_name": "e2i_data_query_tool", "args": {}}]

        await _save_to_episodic_memory(
            state=state,
            response_text="Here is the KPI data.",
            tool_calls=tool_calls,
            significance_score=0.7,
        )

        call_args = mock_insert.call_args
        memory_input = call_args.kwargs["memory"]
        assert memory_input.event_subtype == "chatbot_data_query"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
    async def test_handles_exception_gracefully(self, mock_insert):
        """Test that exceptions are handled without crashing."""
        mock_insert.side_effect = Exception("Database error")

        state = {
            "query": "Test query",
            "session_id": "user~session-err",
            "intent": IntentType.GENERAL,
            "brand_context": None,
            "region_context": None,
            "tool_results": [],
            "metadata": {},
        }

        # Should return None, not raise
        result = await _save_to_episodic_memory(
            state=state,
            response_text="Response",
            tool_calls=[],
            significance_score=0.6,
        )
        assert result is None

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
    async def test_session_id_passed_correctly(self, mock_insert):
        """Test that session ID is passed to insert function."""
        mock_insert.return_value = "memory-id-session"

        state = {
            "query": "Test query",
            "session_id": "user123~abc-def-ghi",
            "intent": IntentType.KPI_QUERY,
            "brand_context": None,
            "region_context": None,
            "tool_results": [],
            "metadata": {},
        }

        await _save_to_episodic_memory(
            state=state,
            response_text="Response text",
            tool_calls=[],
            significance_score=0.65,
        )

        call_args = mock_insert.call_args
        assert call_args.kwargs["session_id"] == "user123~abc-def-ghi"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
    async def test_raw_content_includes_query_and_response(self, mock_insert):
        """Test that raw_content includes query and response preview."""
        mock_insert.return_value = "memory-id-content"

        state = {
            "query": "What drives TRx?",
            "session_id": "user~session",
            "intent": IntentType.CAUSAL_ANALYSIS,
            "brand_context": None,
            "region_context": None,
            "tool_results": [{"result": "driver1"}],
            "metadata": {},
        }

        await _save_to_episodic_memory(
            state=state,
            response_text="The main driver is HCP engagement.",
            tool_calls=[{"tool_name": "causal_analysis_tool", "args": {}}],
            significance_score=0.8,
        )

        call_args = mock_insert.call_args
        memory_input = call_args.kwargs["memory"]
        raw_content = memory_input.raw_content

        assert raw_content["query"] == "What drives TRx?"
        assert "HCP engagement" in raw_content["response_preview"]
        assert raw_content["tools_used"] == ["causal_analysis_tool"]
        assert raw_content["tool_results_count"] == 1
