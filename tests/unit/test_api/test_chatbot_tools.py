"""
Unit tests for E2I Chatbot Tools.

Tests the LangGraph tools for the E2I chatbot including data queries,
causal analysis, agent routing, and document retrieval.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chatbot_graph import _is_multi_faceted_query, classify_intent
from src.api.routes.chatbot_state import IntentType
from src.api.routes.chatbot_tools import (
    E2I_CHATBOT_TOOLS,
    E2IQueryType,
    agent_routing_tool,
    causal_analysis_tool,
    clinical_context_tool,
    conversation_memory_tool,
    document_retrieval_tool,
    e2i_data_query_tool,
    orchestrator_tool,
    tool_composer_tool,
)


class TestE2IDataQueryTool:
    """Tests for e2i_data_query_tool."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.BusinessMetricRepository")
    async def test_queries_kpi_data(self, mock_repo_class, mock_get_client):
        """Test querying KPI data."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_repo = MagicMock()
        mock_repo.query_metrics = AsyncMock(
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
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
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
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.AgentActivityRepository")
    async def test_queries_agent_analysis_data(self, mock_repo_class, mock_get_client):
        """Test querying agent analysis data."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_repo = MagicMock()
        # #1355 (commit 33740696): _query_agent_analysis now calls
        # repo.query_activities(agent_name, brand, since, limit) — get_many is
        # no longer on this path.
        mock_repo.query_activities = AsyncMock(
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
        assert result["count"] == 1

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
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
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
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
    @patch("src.api.routes.chatbot_tools._fetch_refutation_summaries", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.CausalPathRepository")
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
    async def test_searches_causal_paths(
        self, mock_get_client, mock_repo_class, mock_fetch_summaries
    ):
        """Test searching for causal paths.

        2026-07-07 rewire (commit 70446ea6): causal_analysis_tool queries the
        causal_paths registry via CausalPathRepository.search_paths_for_outcome
        (real 0-1 confidence_level), not hybrid_search RAG scores.
        """
        mock_fetch_summaries.return_value = {}
        mock_repo = MagicMock()
        mock_repo.search_paths_for_outcome = AsyncMock(
            return_value=[
                {
                    "path_id": "causal-1",
                    "start_node": "HCP engagement",
                    "end_node": "TRx",
                    "confidence_level": 0.92,
                    "causal_effect_size": 0.15,
                    "method_used": "dowhy",
                    "brand": "Kisqali",
                    "validation_status": "validated",
                },
            ]
        )
        mock_repo_class.return_value = mock_repo

        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "TRx", "brand": "Kisqali", "min_confidence": 0.7}
        )

        assert result["success"] is True
        assert result["kpi_analyzed"] == "TRx"
        assert len(result["results"]) == 1
        mock_repo.search_paths_for_outcome.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools._fetch_refutation_summaries", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.CausalPathRepository")
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
    async def test_handles_empty_results(
        self, mock_get_client, mock_repo_class, mock_fetch_summaries
    ):
        """Test handling of empty search results (substrate-coverage gap)."""
        mock_fetch_summaries.return_value = {}
        mock_repo = MagicMock()
        mock_repo.search_paths_for_outcome = AsyncMock(return_value=[])
        # Empty branch reports the outcomes the registry DOES model (src:920).
        mock_repo.get_distinct_outcomes = AsyncMock(return_value=["treatment_initiated", "TRx"])
        mock_repo_class.return_value = mock_repo

        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "obscure_kpi", "min_confidence": 0.9}
        )

        assert result["success"] is True
        assert result["results"] == []
        assert "substrate_coverage" in result

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools._fetch_refutation_summaries", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.CausalPathRepository")
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
    async def test_handles_search_error(
        self, mock_get_client, mock_repo_class, mock_fetch_summaries
    ):
        """Test that registry-query errors are handled gracefully.

        The error surface moved with the 70446ea6 rewire: the search now lives
        in repo.search_paths_for_outcome, so that is where a failure is raised
        and caught by the tool's fail-closed except clause.
        """
        mock_fetch_summaries.return_value = {}
        mock_repo = MagicMock()
        mock_repo.search_paths_for_outcome = AsyncMock(side_effect=Exception("Search failed"))
        mock_repo_class.return_value = mock_repo

        result = await causal_analysis_tool.ainvoke({"kpi_name": "TRx", "min_confidence": 0.7})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools._fetch_refutation_summaries", new_callable=AsyncMock)
    @patch("src.api.routes.chatbot_tools.CausalPathRepository")
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
    async def test_filters_by_confidence(
        self, mock_get_client, mock_repo_class, mock_fetch_summaries
    ):
        """Confidence filtering now lives in the repository query (.gte in
        causal_path.py:200): min_confidence is forwarded to
        search_paths_for_outcome, which returns only the passing rows."""
        mock_fetch_summaries.return_value = {}
        mock_repo = MagicMock()
        # Repo is the filtering layer: it returns ONLY the passing row.
        mock_repo.search_paths_for_outcome = AsyncMock(
            return_value=[
                {
                    "path_id": "causal-1",
                    "start_node": "x",
                    "end_node": "TRx",
                    "confidence_level": 0.95,
                    "method_used": "dowhy",
                },
            ]
        )
        mock_repo_class.return_value = mock_repo

        result = await causal_analysis_tool.ainvoke({"kpi_name": "TRx", "min_confidence": 0.8})

        assert result["success"] is True
        assert len(result["results"]) == 1
        # _format_causal_path maps confidence_level -> confidence (causal_path.py:451)
        assert result["results"][0]["confidence"] == 0.95
        # The filter intent lives at the forwarding boundary, not tool-side.
        mock_repo.search_paths_for_outcome.assert_called_once()
        assert mock_repo.search_paths_for_outcome.call_args.kwargs["min_confidence"] == 0.8


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
        result = await agent_routing_tool.ainvoke({"query": "What is the forecast for Q3 sales?"})

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
        # Rationale can be from DSPy (detailed) or hardcoded ("Default routing")
        assert "rationale" in result and len(result["rationale"]) > 0

    @pytest.mark.asyncio
    async def test_routes_to_specific_target_agent(self):
        """Test routing to explicitly specified agent."""
        result = await agent_routing_tool.ainvoke(
            {"query": "Any query", "target_agent": "drift_monitor"}
        )

        assert result["success"] is True
        assert result["routed_to"] == "drift_monitor"
        assert result["rationale"] == "Explicit agent selection"

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
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
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
                {
                    "role": "assistant",
                    "content": "TRx is total prescriptions...",
                    "agent_name": "chatbot",
                },
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
    @patch("src.api.routes.chatbot_tools.get_async_supabase_client", new_callable=AsyncMock)
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

        result = await document_retrieval_tool.ainvoke({"query": "Kisqali market analysis", "k": 5})

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

        result = await document_retrieval_tool.ainvoke({"query": "Test query", "k": 5})

        assert result["success"] is False
        assert "error" in result


class TestOrchestratorTool:
    """Tests for orchestrator_tool."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_executes_query_through_orchestrator(self, mock_get_orchestrator):
        """Test executing a query through the orchestrator."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "response_text": "TRx is primarily driven by HCP engagement...",
                "response_confidence": 0.92,
                "agents_dispatched": ["causal_impact"],
                "analysis_results": {
                    "causal_chains": [{"source": "HCP_engagement", "target": "TRx"}]
                },
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke(
            {"query": "Why is TRx declining for Kisqali?", "brand": "Kisqali"}
        )

        assert result["success"] is True
        assert result["fallback"] is False
        assert result["response"] == "TRx is primarily driven by HCP engagement..."
        assert result["confidence"] == 0.92
        assert "causal_impact" in result["agents_dispatched"]
        mock_orchestrator.run.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_passes_context_to_orchestrator(self, mock_get_orchestrator):
        """Test that brand and region context is passed to orchestrator."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "response_text": "Analysis complete",
                "response_confidence": 0.85,
                "agents_dispatched": [],
                "analysis_results": {},
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        await orchestrator_tool.ainvoke(
            {
                "query": "Analyze TRx trends",
                "brand": "Kisqali",
                "region": "US",
                "target_agent": "causal_impact",
            }
        )

        # Verify context was passed correctly
        call_args = mock_orchestrator.run.call_args[0][0]
        assert call_args["user_context"]["brand"] == "Kisqali"
        assert call_args["user_context"]["region"] == "US"
        assert call_args["user_context"]["target_agent"] == "causal_impact"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.hybrid_search")
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_falls_back_to_rag_when_orchestrator_unavailable(
        self, mock_get_orchestrator, mock_hybrid_search
    ):
        """Test fallback to RAG when orchestrator is unavailable."""
        mock_get_orchestrator.return_value = None

        mock_result = MagicMock()
        mock_result.content = "Fallback content from RAG"
        mock_result.score = 0.85
        mock_result.source = "causal_paths"
        mock_hybrid_search.return_value = [mock_result]

        result = await orchestrator_tool.ainvoke(
            {"query": "Why is TRx declining?", "brand": "Kisqali"}
        )

        assert result["success"] is True
        assert result["fallback"] is True
        assert "Orchestrator unavailable" in result["reason"]
        assert result["result_count"] == 1
        mock_hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_handles_orchestrator_error(self, mock_get_orchestrator):
        """Test handling of orchestrator errors."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(side_effect=Exception("Orchestrator failed"))
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "Test query"})

        assert result["success"] is False
        assert "error" in result
        assert result["fallback"] is True

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_generates_session_id_when_not_provided(self, mock_get_orchestrator):
        """Test that a session ID is generated when not provided."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "response_text": "OK",
                "response_confidence": 0.9,
                "agents_dispatched": [],
                "analysis_results": {},
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "Test query"})

        assert result["success"] is True
        assert result["context"]["session_id"].startswith("chatbot-")

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_uses_provided_session_id(self, mock_get_orchestrator):
        """Test that provided session ID is used."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "response_text": "OK",
                "response_confidence": 0.9,
                "agents_dispatched": [],
                "analysis_results": {},
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke(
            {"query": "Test query", "session_id": "custom-session-123"}
        )

        assert result["success"] is True
        assert result["context"]["session_id"] == "custom-session-123"

        # Verify session_id was passed to orchestrator
        call_args = mock_orchestrator.run.call_args[0][0]
        assert call_args["session_id"] == "custom-session-123"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_returns_agents_dispatched(self, mock_get_orchestrator):
        """Test that agents dispatched info is returned."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "response_text": "Multi-agent analysis complete",
                "response_confidence": 0.88,
                "agents_dispatched": ["causal_impact", "gap_analyzer", "explainer"],
                "analysis_results": {},
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "Full analysis of Kisqali performance"})

        assert result["success"] is True
        assert len(result["agents_dispatched"]) == 3
        assert "causal_impact" in result["agents_dispatched"]
        assert "gap_analyzer" in result["agents_dispatched"]
        assert "explainer" in result["agents_dispatched"]

    # ------------------------------------------------------------------
    # #1549 truthful envelope: success must reflect the orchestrator run's
    # REAL status. A fail-closed run (status "failed", zero successful
    # agents, synthesizer's "Please try again or rephrase your question."
    # abstention) was previously re-promoted to a hardcoded success=True,
    # so the AG-UI synthesis prompt presented the ask-back as grounded
    # evidence and tool_evidence/_grade_copilot_turn rewarded it.
    # ------------------------------------------------------------------

    _FAIL_CLOSED_TEXT = (
        "I was unable to complete the analysis due to the following errors:\n"
        "- explainer: explainer needs structured inputs that could not be "
        "grounded in real data; missing: analysis_results. Failing closed - "
        "no values were fabricated.\n\n"
        "Please try again or rephrase your question."
    )

    @staticmethod
    def _fail_closed_orchestrator_result():
        return {
            "status": "failed",
            "response_text": TestOrchestratorTool._FAIL_CLOSED_TEXT,
            "response_confidence": 0.0,
            "agents_dispatched": ["explainer"],
            "successful_agents": [],
            "failed_agents": ["explainer"],
            "has_partial_failure": False,
            "failure_details": [
                {
                    "agent_name": "explainer",
                    "error": (
                        "explainer needs structured inputs that could not be "
                        "grounded in real data; missing: analysis_results"
                    ),
                    "latency_ms": 12,
                    "user_action": (
                        "Run an analysis first (a causal, gap or segmentation "
                        "question), then ask me to explain it."
                    ),
                }
            ],
        }

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_failed_run_propagates_success_false(self, mock_get_orchestrator):
        """A fail-closed orchestrator run must NOT claim success=True; the
        honest abstention text is preserved so the synthesizer can relay it,
        and the failure metadata rides along."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value=self._fail_closed_orchestrator_result())
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "explain that"})

        assert result["success"] is False
        assert result["status"] == "failed"
        assert result["fallback"] is False
        # Honest abstention text preserved for the synthesizer to relay.
        assert result["response"] == self._FAIL_CLOSED_TEXT
        assert result["confidence"] == 0.0
        assert result["failed_agents"] == ["explainer"]
        assert result["failure_details"][0]["user_action"].startswith("Run an analysis first")

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_failed_run_payload_carries_no_evidence(self, mock_get_orchestrator):
        """Downstream pin: the fail-closed envelope is excluded from
        grounded-evidence counts (the #1257 rule keys on ``success``), so
        _grade_copilot_turn no longer rewards fail-closed turns."""
        import json

        from src.utils.tool_evidence import evidence_tool_count, payload_carries_evidence

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value=self._fail_closed_orchestrator_result())
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "explain that"})

        # langchain_core stringifies the dict payload into the ToolMessage.
        payload = json.dumps(result, default=str)
        assert payload_carries_evidence(payload) is False
        assert evidence_tool_count([{"tool": "orchestrator_tool", "result": payload}]) == 0

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_partial_failure_keeps_success_with_metadata(self, mock_get_orchestrator):
        """partial_success carries real evidence from the successful agents:
        success stays True, but the failure metadata is propagated so the
        synthesizer can caveat honestly."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "status": "partial_success",
                "response_text": "TRx is driven by HCP engagement (causal agent failed).",
                "response_confidence": 0.7,
                "agents_dispatched": ["gap_analyzer", "causal_impact"],
                "successful_agents": ["gap_analyzer"],
                "failed_agents": ["causal_impact"],
                "has_partial_failure": True,
                "failure_details": [
                    {
                        "agent_name": "causal_impact",
                        "error": "insufficient rows",
                        "latency_ms": 40,
                        "user_action": None,
                    }
                ],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "Why is TRx moving?"})

        assert result["success"] is True
        assert result["status"] == "partial_success"
        assert result["has_partial_failure"] is True
        assert result["failed_agents"] == ["causal_impact"]

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_partial_failure_details_are_projected_not_raw(self, mock_get_orchestrator):
        """#1549 iter-2 (codex MEDIUM): failure_details must be a TRIMMED
        projection — agent name + coarse category + the dispatcher-authored
        user_action — never the raw internal error string. The AG-UI synthesis
        prompt serializes tool payloads without redaction, and on the
        partial_success path response_text does NOT already narrate the
        errors, so a passthrough would leak internals it never leaked before.
        Mirrors /chat's surfaced-user_action pattern (chat_bridge #1451)."""
        import json

        raw_error = (
            "Traceback (most recent call last): ValueError: 22P02 invalid "
            "input value for enum region_t: 'northeastregion'"
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "status": "partial_success",
                "response_text": "TRx is driven by HCP engagement (causal agent failed).",
                "response_confidence": 0.7,
                "agents_dispatched": ["gap_analyzer", "causal_impact"],
                "successful_agents": ["gap_analyzer"],
                "failed_agents": ["causal_impact"],
                "has_partial_failure": True,
                "failure_details": [
                    {
                        "agent_name": "causal_impact",
                        "error": raw_error,
                        "latency_ms": 40,
                        "user_action": None,
                    }
                ],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "Why is TRx moving?"})

        serialized = json.dumps(result, default=str)
        assert raw_error not in serialized
        assert "22P02" not in serialized
        # Enough survives for an honest "the causal agent failed" caveat.
        assert result["failure_details"] == [
            {"agent_name": "causal_impact", "reason": "agent_error"}
        ]

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_failed_run_failure_details_are_projected(self, mock_get_orchestrator):
        """Projection applies on the all-failed path too: user_action and the
        category survive, raw error/latency internals are dropped (the honest
        abstention narrative already lives in response_text — synthesizer
        contract, out of this lane's scope)."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value=self._fail_closed_orchestrator_result())
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "explain that"})

        detail = result["failure_details"][0]
        assert detail["agent_name"] == "explainer"
        assert detail["reason"] == "missing_required_inputs"
        assert detail["user_action"].startswith("Run an analysis first")
        assert "error" not in detail
        assert "latency_ms" not in detail

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    async def test_completed_run_reports_status_and_omits_empty_analysis_results(
        self, mock_get_orchestrator
    ):
        """_build_output never emits ``analysis_results``, so the tool's
        ``analysis_results`` key was ALWAYS ``{}`` — silent evidence-loss noise
        presented to the synthesis LLM. With zero consumers repo-wide, the key
        is dropped rather than fabricated."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "status": "completed",
                "response_text": "TRx grew 12% on HCP engagement.",
                "response_confidence": 0.9,
                "agents_dispatched": ["causal_impact"],
                "successful_agents": ["causal_impact"],
                "failed_agents": [],
                "has_partial_failure": False,
                "failure_details": None,
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await orchestrator_tool.ainvoke({"query": "TRx drivers?"})

        assert result["success"] is True
        assert result["status"] == "completed"
        assert "analysis_results" not in result
        assert "failed_agents" not in result
        assert "failure_details" not in result


class TestToolComposerTool:
    """Tests for tool_composer_tool.

    NOTE (#1557): the former ``_mock_llm_client`` autouse fixture (patching
    ``chatbot_tools.get_chat_llm`` for CI hermeticity) is gone WITH the seam it
    guarded — the entry point no longer pre-builds an LLM client before
    ``compose_query``; the composer sizes clients per phase (#1365) and
    ``compose_query`` is mocked here, so no real client is ever constructed.
    """

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.compose_query")
    async def test_executes_multi_faceted_query(self, mock_compose_query):
        """Test executing a multi-faceted query through the Tool Composer."""
        # Create mock CompositionResult-like structure
        mock_sub_question = MagicMock()
        mock_sub_question.id = "sq_1"
        mock_sub_question.question = "What is the TRx trend for Kisqali?"
        mock_sub_question.intent = "kpi_query"

        mock_sub_question2 = MagicMock()
        mock_sub_question2.id = "sq_2"
        mock_sub_question2.question = "What is the TRx trend for Fabhalta?"
        mock_sub_question2.intent = "kpi_query"

        mock_sub_question3 = MagicMock()
        mock_sub_question3.id = "sq_3"
        mock_sub_question3.question = "What factors are driving these trends?"
        mock_sub_question3.intent = "causal_query"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.status.value = "COMPLETED"
        mock_result.decomposition.sub_questions = [
            mock_sub_question,
            mock_sub_question2,
            mock_sub_question3,
        ]
        mock_result.execution.tools_executed = [
            "e2i_data_query_tool",
            "e2i_data_query_tool",
            "causal_analysis_tool",
        ]
        mock_result.plan.get_execution_order.return_value = [1, 2, 3]
        mock_result.plan.parallel_groups = [[1, 2], [3]]
        mock_result.response.answer = (
            "Kisqali shows 15% TRx growth while Fabhalta shows 8% growth..."
        )
        mock_result.response.confidence = 0.88
        mock_result.execution.get_all_outputs.return_value = {"causal_impact": {"effect": 0.15}}

        mock_compose_query.return_value = mock_result

        result = await tool_composer_tool.ainvoke(
            {
                "query": "Compare TRx trends across Kisqali and Fabhalta and explain the causal factors",
                "brand": None,  # Multi-brand comparison
            }
        )

        assert result["success"] is True
        assert len(result["sub_questions"]) == 3
        assert len(result["tools_executed"]) == 3
        assert result["confidence"] == 0.88
        assert "Kisqali shows 15%" in result["synthesized_response"]
        mock_compose_query.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.compose_query")
    async def test_passes_context_to_composer(self, mock_compose_query):
        """Test that context is passed correctly to Tool Composer."""
        # Create CompositionResult-like mock structure
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.status.value = "COMPLETED"
        mock_result.decomposition.sub_questions = []
        mock_result.execution.tools_executed = []
        mock_result.plan.get_execution_order.return_value = []
        mock_result.plan.parallel_groups = []
        mock_result.response.answer = "OK"
        mock_result.response.confidence = 0.9
        mock_result.execution.get_all_outputs.return_value = {}

        mock_compose_query.return_value = mock_result

        await tool_composer_tool.ainvoke(
            {
                "query": "Complex query",
                "brand": "Kisqali",
                "region": "US",
                "max_parallel": 4,
            }
        )

        # Verify context was passed correctly
        call_args = mock_compose_query.call_args
        assert call_args.kwargs["context"]["brand"] == "Kisqali"
        assert call_args.kwargs["context"]["region"] == "US"
        assert call_args.kwargs["context"]["max_parallel"] == 4

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    @patch("src.api.routes.chatbot_tools.compose_query")
    async def test_falls_back_to_orchestrator_on_error(
        self, mock_compose_query, mock_get_orchestrator
    ):
        """Test fallback to orchestrator when Tool Composer fails."""
        mock_compose_query.side_effect = Exception("Composition failed")

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "response_text": "Fallback response from orchestrator",
                "response_confidence": 0.75,
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await tool_composer_tool.ainvoke({"query": "Complex multi-faceted query"})

        assert result["success"] is True
        assert result["fallback"] is True
        assert "Composition failed" in result["fallback_reason"]
        assert result["response"] == "Fallback response from orchestrator"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    @patch("src.api.routes.chatbot_tools.compose_query")
    async def test_handles_complete_failure(self, mock_compose_query, mock_get_orchestrator):
        """Test handling when both Tool Composer and orchestrator fail."""
        mock_compose_query.side_effect = Exception("Composition failed")
        mock_get_orchestrator.return_value = None

        result = await tool_composer_tool.ainvoke({"query": "Complex multi-faceted query"})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.compose_query")
    async def test_generates_session_id_when_not_provided(self, mock_compose_query):
        """Test that a session ID is generated when not provided."""
        # Create CompositionResult-like mock structure
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.status.value = "COMPLETED"
        mock_result.decomposition.sub_questions = []
        mock_result.execution.tools_executed = []
        mock_result.plan.get_execution_order.return_value = []
        mock_result.plan.parallel_groups = []
        mock_result.response.answer = "OK"
        mock_result.response.confidence = 0.9
        mock_result.execution.get_all_outputs.return_value = {}

        mock_compose_query.return_value = mock_result

        result = await tool_composer_tool.ainvoke({"query": "Test query"})

        assert result["success"] is True
        assert result["context"]["session_id"].startswith("composer-")

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.compose_query")
    async def test_total_tool_failure_returns_success_false(self, mock_compose_query):
        """F6 route fail-closed: a FAILED composition (0/N tools succeeded) must NOT
        be re-promoted to a success=True envelope by the route wrapper. The honest
        answer text + 0.0 confidence flow through; success/status reflect reality."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.status.value = "FAILED"
        mock_result.decomposition.sub_questions = []
        mock_result.execution.tools_executed = 2
        mock_result.plan.get_execution_order.return_value = []
        mock_result.plan.parallel_groups = []
        mock_result.response.answer = "Unable to complete analysis: all tool(s) failed."
        mock_result.response.confidence = 0.0
        mock_result.execution.get_all_outputs.return_value = {}
        mock_compose_query.return_value = mock_result

        result = await tool_composer_tool.ainvoke({"query": "multi-part failing query"})

        assert result["success"] is False
        assert result["status"] == "FAILED"
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    @patch("src.api.routes.chatbot_tools.compose_query")
    async def test_fallback_propagates_orchestrator_failure(
        self, mock_compose_query, mock_get_orchestrator
    ):
        """#1549: mirror of orchestrator_tool's truthful envelope in the
        composer's fallback branch — when the fallback orchestrator run itself
        fails closed, the payload must not claim success=True. The honest
        abstention text and failure metadata are preserved."""
        mock_compose_query.side_effect = Exception("Composition failed")

        fail_text = (
            "I was unable to complete the analysis due to the following errors:\n"
            "- explainer: no successful upstream agent results.\n\n"
            "Please try again or rephrase your question."
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "status": "failed",
                "response_text": fail_text,
                "response_confidence": 0.0,
                "agents_dispatched": ["explainer"],
                "failed_agents": ["explainer"],
                "has_partial_failure": False,
                "failure_details": [
                    {
                        "agent_name": "explainer",
                        "error": "no successful upstream agent results",
                        "latency_ms": 8,
                        "user_action": "Run an analysis first, then ask me to explain it.",
                    }
                ],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await tool_composer_tool.ainvoke({"query": "Complex multi-faceted query"})

        assert result["success"] is False
        assert result["fallback"] is True
        assert result["status"] == "failed"
        assert "Composition failed" in result["fallback_reason"]
        assert result["response"] == fail_text
        assert result["confidence"] == 0.0
        assert result["failed_agents"] == ["explainer"]
        assert result["failure_details"][0]["user_action"].startswith("Run an analysis first")

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_tools.get_orchestrator")
    @patch("src.api.routes.chatbot_tools.compose_query")
    async def test_fallback_mirrors_partial_failure_metadata(
        self, mock_compose_query, mock_get_orchestrator
    ):
        """#1549 iter-2 (codex LOW + MEDIUM): the composer fallback must mirror
        orchestrator_tool fully — has_partial_failure propagates (it was
        missing from the fallback in iter-1) and failure_details arrive as the
        same trimmed projection, never raw internal error strings."""
        import json

        mock_compose_query.side_effect = Exception("Composition failed")

        raw_error = (
            "RuntimeError: resource_optimizer needs per-entity allocation "
            "inputs; internal audit ref dispatcher.py:2853"
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(
            return_value={
                "status": "partial_success",
                "response_text": "Northeast snapshots below (optimizer failed closed).",
                "response_confidence": 0.6,
                "agents_dispatched": ["gap_analyzer", "resource_optimizer"],
                "successful_agents": ["gap_analyzer"],
                "failed_agents": ["resource_optimizer"],
                "has_partial_failure": True,
                "failure_details": [
                    {
                        "agent_name": "resource_optimizer",
                        "error": raw_error,
                        "latency_ms": 90,
                        "user_action": ("Provide per-entity response coefficients and a budget."),
                    }
                ],
            }
        )
        mock_get_orchestrator.return_value = mock_orchestrator

        result = await tool_composer_tool.ainvoke({"query": "Complex multi-faceted query"})

        assert result["success"] is True
        assert result["fallback"] is True
        assert result["status"] == "partial_success"
        assert result["has_partial_failure"] is True
        assert result["failed_agents"] == ["resource_optimizer"]
        serialized = json.dumps(result, default=str)
        assert raw_error not in serialized
        assert "dispatcher.py" not in serialized
        detail = result["failure_details"][0]
        assert detail["agent_name"] == "resource_optimizer"
        assert detail["reason"] == "agent_error"
        assert detail["user_action"].startswith("Provide per-entity")


class TestMultiFacetedQueryDetection:
    """Tests for multi-faceted query detection."""

    def test_detects_multi_kpi_comparison(self):
        """Test detection of queries comparing multiple KPIs."""
        query = "Compare TRx and NRx trends for Kisqali"
        assert _is_multi_faceted_query(query) is True

    def test_detects_cross_brand_analysis(self):
        """Test detection of queries spanning multiple brands."""
        query = "Compare Kisqali and Fabhalta market share trends and explain the differences"
        assert _is_multi_faceted_query(query) is True

    def test_detects_cross_agent_query(self):
        """Test detection of queries requiring multiple agents."""
        # Query has cross_agent (drift + experiment) AND conjunction_keywords (compare)
        query = "Compare the drift trends and experiment recommendations for Kisqali"
        assert _is_multi_faceted_query(query) is True

    def test_detects_analysis_and_recommendation(self):
        """Test detection of queries needing both analysis and recommendations."""
        # Query has analysis_and_recommendation (why + should) AND multiple_kpis (trx + market share)
        query = "Why did TRx and market share drop for Kisqali and what should we do about it?"
        assert _is_multi_faceted_query(query) is True

    def test_does_not_detect_simple_kpi_query(self):
        """Test that simple KPI queries are not flagged as multi-faceted."""
        query = "What is the TRx for Kisqali?"
        assert _is_multi_faceted_query(query) is False

    def test_does_not_detect_simple_causal_query(self):
        """Test that simple causal queries are not flagged as multi-faceted."""
        query = "Why did TRx drop?"
        assert _is_multi_faceted_query(query) is False

    def test_classify_intent_returns_multi_faceted(self):
        """Test that classify_intent returns MULTI_FACETED for complex queries."""
        query = (
            "Compare TRx and NRx trends across Kisqali and Fabhalta and explain the causal factors"
        )
        intent = classify_intent(query)
        assert intent == IntentType.MULTI_FACETED

    def test_classify_intent_simple_query_not_multi_faceted(self):
        """Test that simple queries don't get MULTI_FACETED intent."""
        query = "What is the TRx for Kisqali?"
        intent = classify_intent(query)
        assert intent != IntentType.MULTI_FACETED
        assert intent == IntentType.KPI_QUERY


class TestClinicalContextTool:
    """Tests for clinical_context_tool (FDA-label / mechanism / competitor context)."""

    _FABHALTA_PAYLOAD = {
        "brand": "Fabhalta",
        "drug_name": "iptacopan",
        "disease": "paroxysmal nocturnal hemoglobinuria",
        "our_outcome": "treatment_initiated",
        "mapped_endpoint": "Treatment initiation (complement-inhibitor start/switch)",
        "mechanism": {
            "mechanism_of_action": "complement Factor B inhibitor",
            "source": "chembl",
        },
        "pivotal_endpoints": {
            "endpoints": ["Transfusion avoidance"],
            "source": "clinicaltrials.gov",
        },
        "real_world_evidence": None,
        "approved_indications": {
            "indications": [
                "Paroxysmal nocturnal hemoglobinuria (PNH)",
                "Primary IgA nephropathy (IgAN), to reduce proteinuria",
            ],
            "limitations_of_use": None,
            "boxed_warning": "Serious infections caused by encapsulated bacteria can occur.",
            "source": "openfda",
        },
        "competitor_landscape": {
            "competitors": ["Soliris (eculizumab)", "Ultomiris (ravulizumab)"],
            "count": 2,
            "source": "curated",
        },
        "honesty_label": "Effect estimate = a SYNTHETIC patient cohort ...",
    }

    @pytest.mark.asyncio
    async def test_returns_openfda_label_indications(self):
        """Tool surfaces the real OpenFDA approved indications for a brand."""
        mock_service = MagicMock()
        mock_service.get_context = MagicMock(return_value=self._FABHALTA_PAYLOAD)

        with patch(
            "src.api.routes.chatbot_tools._get_clinical_context_service",
            return_value=mock_service,
        ):
            result = await clinical_context_tool.ainvoke({"brand": "Fabhalta"})

        assert result["success"] is True
        assert result["query_type"] == "clinical_context"
        assert result["brand"] == "Fabhalta"
        indications = result["clinical_context"]["approved_indications"]
        assert "Paroxysmal nocturnal hemoglobinuria (PNH)" in indications["indications"]
        assert indications["source"] == "openfda"
        # The bare "what's on the label" question defaults the outcome mapping.
        mock_service.get_context.assert_called_once_with("Fabhalta", "treatment_initiated")

    @pytest.mark.asyncio
    async def test_passes_through_explicit_outcome(self):
        """An explicit outcome is forwarded to the service for endpoint framing."""
        mock_service = MagicMock()
        mock_service.get_context = MagicMock(return_value=self._FABHALTA_PAYLOAD)

        with patch(
            "src.api.routes.chatbot_tools._get_clinical_context_service",
            return_value=mock_service,
        ):
            await clinical_context_tool.ainvoke({"brand": "Fabhalta", "outcome": "persistent_180d"})

        mock_service.get_context.assert_called_once_with("Fabhalta", "persistent_180d")

    @pytest.mark.asyncio
    async def test_unknown_brand_returns_error_without_raising(self):
        """A brand with no profile fails closed (success False), never raises."""
        mock_service = MagicMock()
        mock_service.get_context = MagicMock(side_effect=KeyError("Aspirin"))

        with patch(
            "src.api.routes.chatbot_tools._get_clinical_context_service",
            return_value=mock_service,
        ):
            result = await clinical_context_tool.ainvoke({"brand": "Aspirin"})

        assert result["success"] is False
        assert "Aspirin" in result["error"]

    @pytest.mark.asyncio
    async def test_upstream_failure_surfaced_not_fabricated(self):
        """A provider/exception surfaces as a tool error, never fake data."""
        mock_service = MagicMock()
        mock_service.get_context = MagicMock(side_effect=RuntimeError("openfda 503"))

        with patch(
            "src.api.routes.chatbot_tools._get_clinical_context_service",
            return_value=mock_service,
        ):
            result = await clinical_context_tool.ainvoke({"brand": "Fabhalta"})

        assert result["success"] is False
        assert "clinical_context" not in result
        assert "openfda 503" in result["error"]

    def test_registered_in_tool_list_and_map(self):
        """Tool is bound to the chatbot so the LLM can actually call it."""
        from src.api.routes.chatbot_tools import E2I_TOOL_MAP

        assert clinical_context_tool in E2I_CHATBOT_TOOLS
        assert E2I_TOOL_MAP["clinical_context_tool"] is clinical_context_tool


class TestToolExports:
    """Tests for tool exports."""

    def test_all_tools_exported(self):
        """Test that all expected tools are exported."""
        tool_names = [tool.name for tool in E2I_CHATBOT_TOOLS]

        assert "e2i_data_query_tool" in tool_names
        assert "kpi_calculate_tool" in tool_names
        assert "causal_analysis_tool" in tool_names
        assert "clinical_context_tool" in tool_names
        assert "agent_routing_tool" in tool_names
        assert "conversation_memory_tool" in tool_names
        assert "document_retrieval_tool" in tool_names
        assert "orchestrator_tool" in tool_names
        assert "tool_composer_tool" in tool_names
        assert "predict_hcp_segment_likelihood_tool" in tool_names  # #1354

    def test_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        for tool in E2I_CHATBOT_TOOLS:
            assert tool.description, f"Tool {tool.name} missing description"
            assert len(tool.description) > 10, f"Tool {tool.name} has short description"

    def test_tool_count(self):
        """Test expected number of tools."""
        assert len(E2I_CHATBOT_TOOLS) == 10  # #1354 added predict_hcp_segment_likelihood_tool
