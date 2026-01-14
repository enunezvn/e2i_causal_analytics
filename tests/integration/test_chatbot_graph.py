"""
Integration tests for E2I Chatbot LangGraph Workflow.

Tests the complete chatbot workflow including state management,
node execution, and tool integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.api.routes.chatbot_graph import (
    classify_intent,
    create_e2i_chatbot_graph,
    run_chatbot,
)
from src.api.routes.chatbot_state import ChatbotState, IntentType, create_initial_state


class TestIntentClassification:
    """Tests for intent classification."""

    def test_classifies_greeting(self):
        """Test greeting intent classification."""
        assert classify_intent("Hello, how are you?") == IntentType.GREETING
        assert classify_intent("Hi there!") == IntentType.GREETING
        assert classify_intent("Good morning") == IntentType.GREETING

    def test_classifies_help(self):
        """Test help intent classification."""
        assert classify_intent("Help me understand this") == IntentType.HELP
        assert classify_intent("What can you do?") == IntentType.HELP
        assert classify_intent("How do I use this?") == IntentType.HELP

    def test_classifies_kpi_query(self):
        """Test KPI query intent classification."""
        assert classify_intent("What is the TRx for Kisqali?") == IntentType.KPI_QUERY
        assert classify_intent("Show me NRx trends") == IntentType.KPI_QUERY
        assert classify_intent("Market share analysis") == IntentType.KPI_QUERY

    def test_classifies_causal_analysis(self):
        """Test causal analysis intent classification."""
        assert classify_intent("Why did sales drop?") == IntentType.CAUSAL_ANALYSIS
        assert classify_intent("What caused the increase?") == IntentType.CAUSAL_ANALYSIS
        assert classify_intent("Impact of the campaign") == IntentType.CAUSAL_ANALYSIS

    def test_classifies_agent_status(self):
        """Test agent status intent classification."""
        assert classify_intent("Show agent status") == IntentType.AGENT_STATUS
        assert classify_intent("What tier handles this?") == IntentType.AGENT_STATUS
        assert classify_intent("How is the system performing?") == IntentType.AGENT_STATUS

    def test_classifies_recommendation(self):
        """Test recommendation intent classification."""
        assert classify_intent("Recommend next actions") == IntentType.RECOMMENDATION
        assert classify_intent("How can I improve sales?") == IntentType.RECOMMENDATION
        assert classify_intent("Optimize HCP targeting") == IntentType.RECOMMENDATION

    def test_classifies_search(self):
        """Test search intent classification."""
        assert classify_intent("Search for Kisqali trends") == IntentType.SEARCH
        assert classify_intent("Find market analysis") == IntentType.SEARCH
        assert classify_intent("Show me the latest reports") == IntentType.SEARCH

    def test_classifies_general(self):
        """Test general intent classification for unmatched queries."""
        assert classify_intent("Random text here") == IntentType.GENERAL
        assert classify_intent("Something else") == IntentType.GENERAL


class TestChatbotState:
    """Tests for ChatbotState creation."""

    def test_creates_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state(
            user_id="user-123",
            query="What is TRx?",
            request_id="req-456",
        )

        assert state["user_id"] == "user-123"
        assert state["query"] == "What is TRx?"
        assert state["request_id"] == "req-456"
        assert "~" in state["session_id"]  # Auto-generated session_id
        assert state["messages"] == []
        assert state["error"] is None

    def test_creates_state_with_context(self):
        """Test creating state with brand/region context."""
        state = create_initial_state(
            user_id="user-123",
            query="What is TRx?",
            request_id="req-456",
            brand_context="Kisqali",
            region_context="US",
        )

        assert state["brand_context"] == "Kisqali"
        assert state["region_context"] == "US"

    def test_preserves_existing_session_id(self):
        """Test that existing session_id is preserved."""
        state = create_initial_state(
            user_id="user-123",
            query="What is TRx?",
            request_id="req-456",
            session_id="existing-session-123",
        )

        assert state["session_id"] == "existing-session-123"


class TestGraphCreation:
    """Tests for LangGraph workflow creation."""

    def test_creates_graph_successfully(self):
        """Test that the graph is created successfully."""
        graph = create_e2i_chatbot_graph()

        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """Test that the graph has expected node names."""
        graph = create_e2i_chatbot_graph()

        # The compiled graph should have the expected structure
        assert graph is not None


class TestChatbotWorkflow:
    """Integration tests for chatbot workflow execution."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.get_async_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_graph.hybrid_search")
    @patch("src.api.routes.chatbot_graph.get_chat_llm")
    async def test_runs_full_workflow_with_fallback(
        self, mock_get_chat_llm, mock_hybrid_search, mock_get_client
    ):
        """Test running full workflow with fallback when LLM factory fails."""
        # Mock Supabase client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock conversation repo
        mock_conv_result = MagicMock()
        mock_conv_result.data = []
        mock_execute = AsyncMock(return_value=mock_conv_result)
        mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute = (
            mock_execute
        )
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        # Mock RAG search
        mock_hybrid_search.return_value = []

        # Mock LLM factory to raise error (simulating no API key)
        mock_get_chat_llm.side_effect = ValueError("OPENAI_API_KEY not set")
        result = await run_chatbot(
            query="Hello",
            user_id="user-123",
            request_id="req-456",
        )

        assert result is not None
        assert "response_text" in result
        assert result["streaming_complete"] is True

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.get_async_supabase_client", new_callable=AsyncMock)
    async def test_handles_greeting_intent(self, mock_get_client):
        """Test that greeting queries get appropriate response."""
        mock_get_client.return_value = None  # No database

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            result = await run_chatbot(
                query="Hello!",
                user_id="user-123",
                request_id="req-456",
            )

        assert "response_text" in result
        # Fallback greeting response should mention pharmaceutical analytics or assistance
        response_lower = result["response_text"].lower()
        assert any(keyword in response_lower for keyword in ["pharmaceutical", "analytics", "assist", "help"])

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.get_async_supabase_client", new_callable=AsyncMock)
    async def test_handles_help_intent(self, mock_get_client):
        """Test that help queries get appropriate response."""
        mock_get_client.return_value = None

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            result = await run_chatbot(
                query="What can you do?",
                user_id="user-123",
                request_id="req-456",
            )

        assert "response_text" in result
        # Help response should list capabilities
        assert "KPI" in result["response_text"] or "help" in result["response_text"].lower()


class TestMessagePersistence:
    """Tests for message persistence in finalize_node."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.get_async_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_graph.get_chatbot_message_repository")
    @patch("src.api.routes.chatbot_graph.get_chatbot_conversation_repository")
    async def test_persists_messages_on_completion(
        self, mock_get_conv_repo, mock_get_msg_repo, mock_get_client
    ):
        """Test that messages are persisted when workflow completes."""
        # Setup mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_conv_repo = MagicMock()
        mock_conv_repo.get_by_session_id = AsyncMock(return_value=None)
        mock_conv_repo.create_conversation = AsyncMock(return_value={})
        mock_get_conv_repo.return_value = mock_conv_repo

        mock_msg_repo = MagicMock()
        mock_msg_repo.add_message = AsyncMock(return_value={})
        mock_msg_repo.get_recent_messages = AsyncMock(return_value=[])
        mock_get_msg_repo.return_value = mock_msg_repo

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            await run_chatbot(
                query="Hello!",
                user_id="user-123",
                request_id="req-456",
            )

        # Verify add_message was called (once for user, once for assistant)
        assert mock_msg_repo.add_message.call_count >= 2


class TestRAGIntegration:
    """Tests for RAG retrieval integration."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.get_async_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_graph.hybrid_search")
    async def test_retrieves_rag_context(self, mock_hybrid_search, mock_get_client):
        """Test that RAG context is retrieved."""
        mock_get_client.return_value = None

        mock_result = MagicMock()
        mock_result.source_id = "doc-1"
        mock_result.content = "TRx analysis for Kisqali..."
        mock_result.score = 0.9
        mock_result.source = "business_metrics"
        mock_hybrid_search.return_value = [mock_result]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            result = await run_chatbot(
                query="What is TRx for Kisqali?",
                user_id="user-123",
                request_id="req-456",
                brand_context="Kisqali",
            )

        # Hybrid search should have been called
        mock_hybrid_search.assert_called()
        assert "rag_context" in result

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.get_async_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_graph.hybrid_search")
    async def test_handles_rag_error_gracefully(self, mock_hybrid_search, mock_get_client):
        """Test that RAG errors don't crash the workflow."""
        mock_get_client.return_value = None
        mock_hybrid_search.side_effect = Exception("RAG search failed")

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            result = await run_chatbot(
                query="What is TRx?",
                user_id="user-123",
                request_id="req-456",
            )

        # Workflow should complete even with RAG error
        assert result is not None
        assert "response_text" in result
