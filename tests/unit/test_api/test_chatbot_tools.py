"""
Unit tests for E2I Chatbot Tools.

Tests the LangGraph tools for the E2I chatbot including data queries,
causal analysis, agent routing, and document retrieval.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chatbot_tools import (
    E2I_CHATBOT_TOOLS,
    E2IQueryType,
    TimeRange,
    agent_routing_tool,
    causal_analysis_tool,
    conversation_memory_tool,
    document_retrieval_tool,
    e2i_data_query_tool,
)


class TestE2IDataQueryTool:
    """Tests for e2i_data_query_tool."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.BusinessMetricRepository")
    async def test_queries_kpi_data(self, mock_repo_class, mock_get_client):
        """Test querying KPI data."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_repo = MagicMock()
        mock_repo.get_many = AsyncMock(
            return_value=[
                {"kpi_name": "trx", "value": 1500, "brand": "Kisqali"},
                {"kpi_name": "trx", "value": 1600, "brand": "Kisqali"},
            ]
        )
        mock_repo_class.return_value = mock_repo

        result = await e2i_data_query_tool.ainvoke(
            {
                "query_type": E2IQueryType.KPI,
                "brand": "Kisqali",
                "limit": 10,
            }
        )

        assert result["success"] is True
        assert result["query_type"] == "kpi"
        assert result["count"] == 2

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.CausalPathRepository")
    async def test_queries_causal_chain_data(self, mock_repo_class, mock_get_client):
        """Test querying causal chain data."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_repo = MagicMock()
        mock_repo.get_many = AsyncMock(
            return_value=[
                {"path_id": "path-1", "source_node": "A", "target_node": "B", "confidence": 0.85},
            ]
        )
        mock_repo_class.return_value = mock_repo

        result = await e2i_data_query_tool.ainvoke(
            {
                "query_type": E2IQueryType.CAUSAL_CHAIN,
                "limit": 10,
            }
        )

        assert result["success"] is True
        assert result["query_type"] == "causal_chain"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.AgentActivityRepository")
    async def test_queries_agent_analysis_data(self, mock_repo_class, mock_get_client):
        """Test querying agent analysis data."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_repo = MagicMock()
        mock_repo.get_many = AsyncMock(
            return_value=[
                {"agent_name": "causal_impact", "analysis_type": "effect_estimation"},
            ]
        )
        mock_repo_class.return_value = mock_repo

        result = await e2i_data_query_tool.ainvoke(
            {
                "query_type": E2IQueryType.AGENT_ANALYSIS,
                "limit": 10,
            }
        )

        assert result["success"] is True
        assert result["query_type"] == "agent_analysis"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    async def test_queries_experiments_via_rag(self, mock_hybrid_search, mock_get_client):
        """Test that experiments query type uses RAG fallback."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_result = MagicMock()
        mock_result.source_id = "exp-1"
        mock_result.content = "A/B test results..."
        mock_result.score = 0.9
        mock_result.source = "experiments"
        mock_result.metadata = {}
        mock_hybrid_search.return_value = [mock_result]

        result = await e2i_data_query_tool.ainvoke(
            {
                "query_type": E2IQueryType.EXPERIMENTS,
                "brand": "Kisqali",
                "limit": 10,
            }
        )

        assert result["success"] is True
        assert result["query_type"] == "experiments"
        mock_hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.BusinessMetricRepository")
    async def test_handles_database_error(self, mock_repo_class, mock_get_client):
        """Test that database errors are handled gracefully."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_repo = MagicMock()
        mock_repo.get_many = AsyncMock(side_effect=Exception("Database connection failed"))
        mock_repo_class.return_value = mock_repo

        result = await e2i_data_query_tool.ainvoke(
            {
                "query_type": E2IQueryType.KPI,
                "limit": 10,
            }
        )

        assert result["success"] is False
        assert "error" in result


class TestCausalAnalysisTool:
    """Tests for causal_analysis_tool."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    async def test_searches_causal_paths(self, mock_hybrid_search):
        """Test searching for causal paths."""
        mock_result = MagicMock()
        mock_result.source_id = "causal-1"
        mock_result.content = "TRx is driven by HCP engagement..."
        mock_result.score = 0.92
        mock_result.source = "causal_paths"
        mock_result.metadata = {"confidence": 0.92, "effect_magnitude": 0.15}
        mock_hybrid_search.return_value = [mock_result]

        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Kisqali", "min_confidence": 0.7}
        )

        assert result["success"] is True
        assert result["kpi_analyzed"] == "TRx"
        assert len(result["results"]) == 1
        mock_hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    async def test_handles_empty_results(self, mock_hybrid_search):
        """Test handling of empty search results."""
        mock_hybrid_search.return_value = []

        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "obscure_kpi", "min_confidence": 0.9}
        )

        assert result["success"] is True
        assert result["results"] == []

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    async def test_handles_search_error(self, mock_hybrid_search):
        """Test that search errors are handled gracefully."""
        mock_hybrid_search.side_effect = Exception("Search failed")

        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "TRx", "min_confidence": 0.7}
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    async def test_filters_by_confidence(self, mock_hybrid_search):
        """Test that results are filtered by minimum confidence."""
        # Create results with different confidence scores
        high_conf = MagicMock()
        high_conf.source_id = "causal-1"
        high_conf.content = "High confidence result"
        high_conf.score = 0.95
        high_conf.metadata = {"confidence": 0.95}

        low_conf = MagicMock()
        low_conf.source_id = "causal-2"
        low_conf.content = "Low confidence result"
        low_conf.score = 0.5
        low_conf.metadata = {"confidence": 0.5}

        mock_hybrid_search.return_value = [high_conf, low_conf]

        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "TRx", "min_confidence": 0.8}
        )

        assert result["success"] is True
        # Only high confidence result should be included
        assert len(result["results"]) == 1
        assert result["results"][0]["confidence"] == 0.95


class TestAgentRoutingTool:
    """Tests for agent_routing_tool."""

    @pytest.mark.asyncio
    async def test_routes_to_causal_agent(self):
        """Test routing a causal analysis query."""
        result = await agent_routing_tool.ainvoke(
            {"query": "Why did market share drop? What is the cause?"}
        )

        assert result["success"] is True
        assert result["routed_to"] == "causal_impact"

    @pytest.mark.asyncio
    async def test_routes_to_experiment_designer(self):
        """Test routing an experiment-related query."""
        result = await agent_routing_tool.ainvoke(
            {"query": "Design an A/B test for the new campaign"}
        )

        assert result["success"] is True
        assert result["routed_to"] == "experiment_designer"

    @pytest.mark.asyncio
    async def test_routes_to_prediction_agent(self):
        """Test routing a prediction query."""
        result = await agent_routing_tool.ainvoke(
            {"query": "What is the forecast for Q3 sales?"}
        )

        assert result["success"] is True
        assert result["routed_to"] == "prediction_synthesizer"

    @pytest.mark.asyncio
    async def test_routes_general_query_to_explainer(self):
        """Test that general queries default to explainer."""
        result = await agent_routing_tool.ainvoke(
            {"query": "Random text without specific keywords"}
        )

        assert result["success"] is True
        assert result["routed_to"] == "explainer"
        assert "Default routing" in result["reason"]

    @pytest.mark.asyncio
    async def test_routes_to_specific_target_agent(self):
        """Test routing to explicitly specified agent."""
        result = await agent_routing_tool.ainvoke(
            {"query": "Any query", "target_agent": "drift_monitor"}
        )

        assert result["success"] is True
        assert result["routed_to"] == "drift_monitor"
        assert result["reason"] == "Explicit agent selection"

    @pytest.mark.asyncio
    async def test_rejects_unknown_target_agent(self):
        """Test rejection of unknown target agent."""
        result = await agent_routing_tool.ainvoke(
            {"query": "Any query", "target_agent": "unknown_agent"}
        )

        assert result["success"] is False
        assert "Unknown agent" in result["error"]


class TestConversationMemoryTool:
    """Tests for conversation_memory_tool."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.get_chatbot_message_repository")
    @patch("src.api.routes.chatbot_tools.get_chatbot_conversation_repository")
    async def test_retrieves_conversation_history(
        self, mock_get_conv_repo, mock_get_msg_repo, mock_get_client
    ):
        """Test retrieving conversation history."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_conv_repo = MagicMock()
        mock_conv_repo.get_by_session_id = AsyncMock(
            return_value={
                "session_id": "user-123~uuid-456",
                "title": "KPI Analysis",
                "brand_context": "Kisqali",
                "region_context": "US",
            }
        )
        mock_get_conv_repo.return_value = mock_conv_repo

        mock_msg_repo = MagicMock()
        mock_msg_repo.get_recent_messages = AsyncMock(
            return_value=[
                {"role": "user", "content": "What is TRx?", "agent_name": None},
                {"role": "assistant", "content": "TRx is total prescriptions...", "agent_name": "chatbot"},
            ]
        )
        mock_get_msg_repo.return_value = mock_msg_repo

        result = await conversation_memory_tool.ainvoke(
            {"session_id": "user-123~uuid-456", "message_count": 10}
        )

        assert result["success"] is True
        assert result["message_count"] == 2
        assert result["conversation_title"] == "KPI Analysis"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.get_chatbot_conversation_repository")
    async def test_returns_error_when_conversation_not_found(
        self, mock_get_conv_repo, mock_get_client
    ):
        """Test error handling when conversation not found."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_conv_repo = MagicMock()
        mock_conv_repo.get_by_session_id = AsyncMock(return_value=None)
        mock_get_conv_repo.return_value = mock_conv_repo

        result = await conversation_memory_tool.ainvoke(
            {"session_id": "nonexistent-session", "message_count": 10}
        )

        assert result["success"] is False
        assert "Conversation not found" in result["error"]


class TestDocumentRetrievalTool:
    """Tests for document_retrieval_tool."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    async def test_retrieves_documents(self, mock_hybrid_search):
        """Test retrieving documents via hybrid search."""
        mock_result = MagicMock()
        mock_result.source_id = "doc-1"
        mock_result.content = "Kisqali market analysis..."
        mock_result.score = 0.88
        mock_result.source = "agent_activities"
        mock_result.retrieval_method = "hybrid"
        mock_result.metadata = {}
        mock_hybrid_search.return_value = [mock_result]

        result = await document_retrieval_tool.ainvoke(
            {"query": "Kisqali market analysis", "k": 5}
        )

        assert result["success"] is True
        assert result["document_count"] == 1
        assert result["documents"][0]["relevance_score"] == 0.88

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    async def test_retrieves_documents_with_filters(self, mock_hybrid_search):
        """Test document retrieval with brand and KPI filters."""
        mock_result = MagicMock()
        mock_result.source_id = "doc-2"
        mock_result.content = "TRx trends for Kisqali..."
        mock_result.score = 0.92
        mock_result.source = "business_metrics"
        mock_result.retrieval_method = "hybrid"
        mock_result.metadata = {"brand": "Kisqali"}
        mock_hybrid_search.return_value = [mock_result]

        result = await document_retrieval_tool.ainvoke(
            {"query": "TRx trends", "k": 5, "brand": "Kisqali", "kpi_name": "TRx"}
        )

        assert result["success"] is True
        assert result["filters_applied"]["brand"] == "Kisqali"
        assert result["filters_applied"]["kpi_name"] == "TRx"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    async def test_handles_retrieval_error(self, mock_hybrid_search):
        """Test that retrieval errors are handled gracefully."""
        mock_hybrid_search.side_effect = Exception("Retrieval failed")

        result = await document_retrieval_tool.ainvoke(
            {"query": "Test query", "k": 5}
        )

        assert result["success"] is False
        assert "error" in result


class TestToolExports:
    """Tests for tool exports."""

    def test_all_tools_exported(self):
        """Test that all expected tools are exported."""
        tool_names = [tool.name for tool in E2I_CHATBOT_TOOLS]

        assert "e2i_data_query_tool" in tool_names
        assert "causal_analysis_tool" in tool_names
        assert "agent_routing_tool" in tool_names
        assert "conversation_memory_tool" in tool_names
        assert "document_retrieval_tool" in tool_names

    def test_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        for tool in E2I_CHATBOT_TOOLS:
            assert tool.description, f"Tool {tool.name} missing description"
            assert len(tool.description) > 10, f"Tool {tool.name} has short description"

    def test_tool_count(self):
        """Test expected number of tools."""
        assert len(E2I_CHATBOT_TOOLS) == 5
