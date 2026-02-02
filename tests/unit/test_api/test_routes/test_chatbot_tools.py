"""
Tests for src/api/routes/chatbot_tools.py

Covers:
- E2IQueryType and TimeRange enums
- Pydantic input models for all tools
- Helper functions (_get_time_filter, _query_*)
- LangGraph tools with mocked dependencies
- Tool exports and mappings
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.api.routes.chatbot_tools import (
    # Exports
    E2I_CHATBOT_TOOLS,
    E2I_TOOL_MAP,
    AgentRoutingInput,
    CausalAnalysisInput,
    ConversationMemoryInput,
    DocumentRetrievalInput,
    # Pydantic models
    E2IDataQueryInput,
    # Enums
    E2IQueryType,
    OrchestratorToolInput,
    TimeRange,
    ToolComposerToolInput,
    # Helper functions
    _get_time_filter,
    _query_agent_analysis,
    _query_causal_chains,
    _query_kpis,
    _query_triggers,
    _query_via_rag,
    agent_routing_tool,
    causal_analysis_tool,
    conversation_memory_tool,
    document_retrieval_tool,
    # Tools
    e2i_data_query_tool,
    get_e2i_chatbot_tools,
    get_tool_by_name,
    orchestrator_tool,
    tool_composer_tool,
)

# =============================================================================
# E2IQueryType Enum Tests
# =============================================================================


class TestE2IQueryType:
    """Tests for E2IQueryType enum."""

    def test_query_type_values(self):
        """Test that all expected query types exist."""
        assert E2IQueryType.KPI.value == "kpi"
        assert E2IQueryType.CAUSAL_CHAIN.value == "causal_chain"
        assert E2IQueryType.AGENT_ANALYSIS.value == "agent_analysis"
        assert E2IQueryType.TRIGGERS.value == "triggers"
        assert E2IQueryType.EXPERIMENTS.value == "experiments"
        assert E2IQueryType.PREDICTIONS.value == "predictions"
        assert E2IQueryType.RECOMMENDATIONS.value == "recommendations"
        assert E2IQueryType.DRIFT_REPORTS.value == "drift_reports"

    def test_query_type_count(self):
        """Test that we have exactly 8 query types."""
        assert len(E2IQueryType) == 8

    def test_query_type_is_string_enum(self):
        """Test that query type is a string enum."""
        assert isinstance(E2IQueryType.KPI.value, str)
        assert E2IQueryType.KPI == "kpi"


# =============================================================================
# TimeRange Enum Tests
# =============================================================================


class TestTimeRange:
    """Tests for TimeRange enum."""

    def test_time_range_values(self):
        """Test that all expected time ranges exist."""
        assert TimeRange.LAST_7_DAYS.value == "last_7_days"
        assert TimeRange.LAST_30_DAYS.value == "last_30_days"
        assert TimeRange.LAST_90_DAYS.value == "last_90_days"
        assert TimeRange.LAST_YEAR.value == "last_year"
        assert TimeRange.ALL_TIME.value == "all_time"

    def test_time_range_count(self):
        """Test that we have exactly 5 time ranges."""
        assert len(TimeRange) == 5


# =============================================================================
# E2IDataQueryInput Model Tests
# =============================================================================


class TestE2IDataQueryInput:
    """Tests for E2IDataQueryInput model."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        input_data = E2IDataQueryInput(query_type=E2IQueryType.KPI)
        assert input_data.query_type == E2IQueryType.KPI
        assert input_data.brand is None
        assert input_data.region is None
        assert input_data.time_range == TimeRange.LAST_30_DAYS  # default
        assert input_data.limit == 10  # default

    def test_create_full_input(self):
        """Test creating input with all fields."""
        input_data = E2IDataQueryInput(
            query_type=E2IQueryType.CAUSAL_CHAIN,
            brand="Kisqali",
            region="US",
            kpi_name="TRx",
            agent_name="causal_impact",
            time_range=TimeRange.LAST_90_DAYS,
            limit=50,
            filters={"status": "active"},
        )
        assert input_data.query_type == E2IQueryType.CAUSAL_CHAIN
        assert input_data.brand == "Kisqali"
        assert input_data.region == "US"
        assert input_data.kpi_name == "TRx"
        assert input_data.agent_name == "causal_impact"
        assert input_data.time_range == TimeRange.LAST_90_DAYS
        assert input_data.limit == 50
        assert input_data.filters == {"status": "active"}

    def test_limit_validation_min(self):
        """Test that limit must be at least 1."""
        with pytest.raises(ValidationError):
            E2IDataQueryInput(query_type=E2IQueryType.KPI, limit=0)

    def test_limit_validation_max(self):
        """Test that limit must be at most 100."""
        with pytest.raises(ValidationError):
            E2IDataQueryInput(query_type=E2IQueryType.KPI, limit=101)


# =============================================================================
# CausalAnalysisInput Model Tests
# =============================================================================


class TestCausalAnalysisInput:
    """Tests for CausalAnalysisInput model."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        input_data = CausalAnalysisInput(kpi_name="TRx")
        assert input_data.kpi_name == "TRx"
        assert input_data.brand is None
        assert input_data.min_confidence == 0.7  # default

    def test_create_full_input(self):
        """Test creating input with all fields."""
        input_data = CausalAnalysisInput(
            kpi_name="NRx",
            brand="Fabhalta",
            region="EU",
            time_period="last_90_days",
            min_confidence=0.85,
        )
        assert input_data.kpi_name == "NRx"
        assert input_data.brand == "Fabhalta"
        assert input_data.region == "EU"
        assert input_data.time_period == "last_90_days"
        assert input_data.min_confidence == 0.85

    def test_min_confidence_validation(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            CausalAnalysisInput(kpi_name="TRx", min_confidence=1.5)
        with pytest.raises(ValidationError):
            CausalAnalysisInput(kpi_name="TRx", min_confidence=-0.1)


# =============================================================================
# AgentRoutingInput Model Tests
# =============================================================================


class TestAgentRoutingInput:
    """Tests for AgentRoutingInput model."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        input_data = AgentRoutingInput(query="What is the TRx?")
        assert input_data.query == "What is the TRx?"
        assert input_data.target_agent is None
        assert input_data.context is None

    def test_create_full_input(self):
        """Test creating input with all fields."""
        input_data = AgentRoutingInput(
            query="Analyze causal factors",
            target_agent="causal_impact",
            context={"intent": "causal_analysis", "brand": "Kisqali"},
        )
        assert input_data.query == "Analyze causal factors"
        assert input_data.target_agent == "causal_impact"
        assert input_data.context == {"intent": "causal_analysis", "brand": "Kisqali"}


# =============================================================================
# ConversationMemoryInput Model Tests
# =============================================================================


class TestConversationMemoryInput:
    """Tests for ConversationMemoryInput model."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        input_data = ConversationMemoryInput(session_id="session-123")
        assert input_data.session_id == "session-123"
        assert input_data.message_count == 10  # default
        assert input_data.include_tool_calls is True  # default

    def test_create_full_input(self):
        """Test creating input with all fields."""
        input_data = ConversationMemoryInput(
            session_id="session-456",
            message_count=25,
            include_tool_calls=False,
        )
        assert input_data.session_id == "session-456"
        assert input_data.message_count == 25
        assert input_data.include_tool_calls is False

    def test_message_count_validation(self):
        """Test message count must be 1-50."""
        with pytest.raises(ValidationError):
            ConversationMemoryInput(session_id="s", message_count=0)
        with pytest.raises(ValidationError):
            ConversationMemoryInput(session_id="s", message_count=51)


# =============================================================================
# DocumentRetrievalInput Model Tests
# =============================================================================


class TestDocumentRetrievalInput:
    """Tests for DocumentRetrievalInput model."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        input_data = DocumentRetrievalInput(query="TRx trends")
        assert input_data.query == "TRx trends"
        assert input_data.k == 5  # default
        assert input_data.brand is None

    def test_create_full_input(self):
        """Test creating input with all fields."""
        input_data = DocumentRetrievalInput(
            query="causal factors for NRx",
            k=15,
            brand="Remibrutinib",
            kpi_name="NRx",
        )
        assert input_data.query == "causal factors for NRx"
        assert input_data.k == 15
        assert input_data.brand == "Remibrutinib"
        assert input_data.kpi_name == "NRx"

    def test_k_validation(self):
        """Test k must be 1-20."""
        with pytest.raises(ValidationError):
            DocumentRetrievalInput(query="test", k=0)
        with pytest.raises(ValidationError):
            DocumentRetrievalInput(query="test", k=21)


# =============================================================================
# OrchestratorToolInput Model Tests
# =============================================================================


class TestOrchestratorToolInput:
    """Tests for OrchestratorToolInput model."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        input_data = OrchestratorToolInput(query="Analyze TRx")
        assert input_data.query == "Analyze TRx"
        assert input_data.target_agent is None
        assert input_data.brand is None
        assert input_data.region is None
        assert input_data.session_id is None

    def test_create_full_input(self):
        """Test creating input with all fields."""
        input_data = OrchestratorToolInput(
            query="Run experiment design",
            target_agent="experiment_designer",
            brand="Kisqali",
            region="APAC",
            session_id="session-789",
        )
        assert input_data.query == "Run experiment design"
        assert input_data.target_agent == "experiment_designer"
        assert input_data.brand == "Kisqali"
        assert input_data.region == "APAC"
        assert input_data.session_id == "session-789"


# =============================================================================
# ToolComposerToolInput Model Tests
# =============================================================================


class TestToolComposerToolInput:
    """Tests for ToolComposerToolInput model."""

    def test_create_minimal_input(self):
        """Test creating input with only required fields."""
        input_data = ToolComposerToolInput(query="Compare TRx and NRx")
        assert input_data.query == "Compare TRx and NRx"
        assert input_data.max_parallel == 3  # default

    def test_create_full_input(self):
        """Test creating input with all fields."""
        input_data = ToolComposerToolInput(
            query="Multi-faceted analysis",
            brand="Fabhalta",
            region="EU",
            session_id="session-abc",
            max_parallel=5,
        )
        assert input_data.query == "Multi-faceted analysis"
        assert input_data.brand == "Fabhalta"
        assert input_data.region == "EU"
        assert input_data.session_id == "session-abc"
        assert input_data.max_parallel == 5

    def test_max_parallel_validation(self):
        """Test max_parallel must be 1-5."""
        with pytest.raises(ValidationError):
            ToolComposerToolInput(query="test", max_parallel=0)
        with pytest.raises(ValidationError):
            ToolComposerToolInput(query="test", max_parallel=6)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetTimeFilter:
    """Tests for _get_time_filter helper."""

    def test_last_7_days(self):
        """Test last 7 days filter."""
        result = _get_time_filter(TimeRange.LAST_7_DAYS)
        expected_min = datetime.now(timezone.utc) - timedelta(days=8)
        expected_max = datetime.now(timezone.utc) - timedelta(days=6)
        assert expected_min < result < expected_max

    def test_last_30_days(self):
        """Test last 30 days filter."""
        result = _get_time_filter(TimeRange.LAST_30_DAYS)
        expected_min = datetime.now(timezone.utc) - timedelta(days=31)
        expected_max = datetime.now(timezone.utc) - timedelta(days=29)
        assert expected_min < result < expected_max

    def test_last_90_days(self):
        """Test last 90 days filter."""
        result = _get_time_filter(TimeRange.LAST_90_DAYS)
        expected_min = datetime.now(timezone.utc) - timedelta(days=91)
        expected_max = datetime.now(timezone.utc) - timedelta(days=89)
        assert expected_min < result < expected_max

    def test_last_year(self):
        """Test last year filter."""
        result = _get_time_filter(TimeRange.LAST_YEAR)
        expected_min = datetime.now(timezone.utc) - timedelta(days=366)
        expected_max = datetime.now(timezone.utc) - timedelta(days=364)
        assert expected_min < result < expected_max

    def test_all_time(self):
        """Test all time filter."""
        result = _get_time_filter(TimeRange.ALL_TIME)
        assert result == datetime(2020, 1, 1)


# =============================================================================
# _query_kpis Tests
# =============================================================================


class TestQueryKpis:
    """Tests for _query_kpis helper."""

    @pytest.mark.asyncio
    async def test_query_kpis_success(self):
        """Test successful KPI query."""
        mock_metrics = [
            {"metric_name": "TRx", "value": 100, "brand": "Kisqali"},
            {"metric_name": "TRx", "value": 150, "brand": "Kisqali"},
        ]

        with patch("src.api.routes.chatbot_tools.get_async_supabase_client") as mock_client:
            mock_repo = AsyncMock()
            mock_repo.get_many.return_value = mock_metrics
            mock_client.return_value = MagicMock()

            with patch(
                "src.api.routes.chatbot_tools.BusinessMetricRepository",
                return_value=mock_repo,
            ):
                result = await _query_kpis(
                    brand="Kisqali",
                    region="US",
                    kpi_name="TRx",
                    since=datetime.now(timezone.utc),
                    limit=10,
                )

        assert result["success"] is True
        assert result["query_type"] == "kpi"
        assert result["count"] == 2
        assert result["data"] == mock_metrics

    @pytest.mark.asyncio
    async def test_query_kpis_with_filters(self):
        """Test KPI query applies filters correctly."""
        with patch("src.api.routes.chatbot_tools.get_async_supabase_client") as mock_client:
            mock_repo = AsyncMock()
            mock_repo.get_many.return_value = []
            mock_client.return_value = MagicMock()

            with patch(
                "src.api.routes.chatbot_tools.BusinessMetricRepository",
                return_value=mock_repo,
            ):
                await _query_kpis(
                    brand="Fabhalta",
                    region="EU",
                    kpi_name="NRx",
                    since=datetime.now(timezone.utc),
                    limit=5,
                )

        # Verify filters were passed
        mock_repo.get_many.assert_called_once()
        call_kwargs = mock_repo.get_many.call_args[1]
        assert call_kwargs["filters"]["brand"] == "Fabhalta"
        assert call_kwargs["filters"]["region"] == "EU"
        assert call_kwargs["filters"]["metric_name"] == "NRx"
        assert call_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_query_kpis_error_handling(self):
        """Test KPI query error handling."""
        with patch("src.api.routes.chatbot_tools.get_async_supabase_client") as mock_client:
            mock_client.side_effect = Exception("Database connection failed")

            result = await _query_kpis(
                brand=None,
                region=None,
                kpi_name=None,
                since=datetime.now(timezone.utc),
                limit=10,
            )

        assert result["success"] is False
        assert "error" in result
        assert "Database connection failed" in result["error"]


# =============================================================================
# _query_causal_chains Tests
# =============================================================================


class TestQueryCausalChains:
    """Tests for _query_causal_chains helper."""

    @pytest.mark.asyncio
    async def test_query_causal_chains_with_kpi(self):
        """Test causal chains query with KPI uses RAG."""
        mock_rag_result = MagicMock()
        mock_rag_result.source_id = "causal-1"
        mock_rag_result.content = "TRx affects NRx"
        mock_rag_result.score = 0.85
        mock_rag_result.metadata = {"confidence": 0.9}

        with patch(
            "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = [mock_rag_result]

            result = await _query_causal_chains(
                brand="Kisqali",
                kpi_name="TRx",
                since=datetime.now(timezone.utc),
                limit=10,
            )

        assert result["success"] is True
        assert result["query_type"] == "causal_chain"
        assert result["count"] == 1
        assert result["kpi_analyzed"] == "TRx"

    @pytest.mark.asyncio
    async def test_query_causal_chains_without_kpi(self):
        """Test causal chains query without KPI uses repository."""
        mock_paths = [{"id": 1, "source": "A", "target": "B"}]

        with patch("src.api.routes.chatbot_tools.get_async_supabase_client") as mock_client:
            mock_repo = AsyncMock()
            mock_repo.get_many.return_value = mock_paths
            mock_client.return_value = MagicMock()

            with patch(
                "src.api.routes.chatbot_tools.CausalPathRepository",
                return_value=mock_repo,
            ):
                result = await _query_causal_chains(
                    brand=None,
                    kpi_name=None,
                    since=datetime.now(timezone.utc),
                    limit=10,
                )

        assert result["success"] is True
        assert result["count"] == 1


# =============================================================================
# _query_agent_analysis Tests
# =============================================================================


class TestQueryAgentAnalysis:
    """Tests for _query_agent_analysis helper."""

    @pytest.mark.asyncio
    async def test_query_agent_analysis_success(self):
        """Test successful agent analysis query."""
        mock_activities = [
            {"agent_name": "causal_impact", "output": "Analysis result"},
        ]

        with patch("src.api.routes.chatbot_tools.get_async_supabase_client") as mock_client:
            mock_repo = AsyncMock()
            mock_repo.get_many.return_value = mock_activities
            mock_client.return_value = MagicMock()

            with patch(
                "src.api.routes.chatbot_tools.AgentActivityRepository",
                return_value=mock_repo,
            ):
                result = await _query_agent_analysis(
                    agent_name="causal_impact",
                    brand=None,
                    since=datetime.now(timezone.utc),
                    limit=10,
                )

        assert result["success"] is True
        assert result["query_type"] == "agent_analysis"
        assert result["agent_filter"] == "causal_impact"


# =============================================================================
# _query_triggers Tests
# =============================================================================


class TestQueryTriggers:
    """Tests for _query_triggers helper."""

    @pytest.mark.asyncio
    async def test_query_triggers_success(self):
        """Test successful triggers query."""
        mock_triggers = [{"id": 1, "type": "alert", "message": "TRx dropped"}]

        with patch("src.api.routes.chatbot_tools.get_async_supabase_client") as mock_client:
            mock_repo = AsyncMock()
            mock_repo.get_many.return_value = mock_triggers
            mock_client.return_value = MagicMock()

            with patch(
                "src.api.routes.chatbot_tools.TriggerRepository",
                return_value=mock_repo,
            ):
                result = await _query_triggers(
                    brand=None,
                    region=None,
                    since=datetime.now(timezone.utc),
                    limit=10,
                )

        assert result["success"] is True
        assert result["query_type"] == "triggers"
        assert result["count"] == 1


# =============================================================================
# _query_via_rag Tests
# =============================================================================


class TestQueryViaRag:
    """Tests for _query_via_rag helper."""

    @pytest.mark.asyncio
    async def test_query_via_rag_success(self):
        """Test successful RAG query."""
        mock_result = MagicMock()
        mock_result.source_id = "doc-1"
        mock_result.content = "Experiment results"
        mock_result.score = 0.9
        mock_result.source = "experiments"
        mock_result.metadata = {"type": "experiment"}

        with patch(
            "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = [mock_result]

            result = await _query_via_rag(
                query_type="experiments",
                query="recent experiments",
                filters={"brand": "Kisqali"},
                limit=5,
            )

        assert result["success"] is True
        assert result["query_type"] == "experiments"
        assert result["retrieval_method"] == "hybrid_rag"
        assert result["count"] == 1


# =============================================================================
# e2i_data_query_tool Tests
# =============================================================================


class TestE2IDataQueryTool:
    """Tests for e2i_data_query_tool."""

    @pytest.mark.asyncio
    async def test_kpi_query_routing(self):
        """Test that KPI queries route to _query_kpis."""
        with patch(
            "src.api.routes.chatbot_tools._query_kpis", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = {"success": True, "data": []}

            result = await e2i_data_query_tool.ainvoke(
                {
                    "query_type": E2IQueryType.KPI.value,
                    "brand": "Kisqali",
                    "kpi_name": "TRx",
                }
            )

        mock_query.assert_called_once()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_causal_chain_query_routing(self):
        """Test that causal chain queries route correctly."""
        with patch(
            "src.api.routes.chatbot_tools._query_causal_chains", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = {"success": True, "data": []}

            await e2i_data_query_tool.ainvoke(
                {
                    "query_type": E2IQueryType.CAUSAL_CHAIN.value,
                    "kpi_name": "NRx",
                }
            )

        mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_analysis_query_routing(self):
        """Test that agent analysis queries route correctly."""
        with patch(
            "src.api.routes.chatbot_tools._query_agent_analysis", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = {"success": True, "data": []}

            await e2i_data_query_tool.ainvoke(
                {
                    "query_type": E2IQueryType.AGENT_ANALYSIS.value,
                    "agent_name": "drift_monitor",
                }
            )

        mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_triggers_query_routing(self):
        """Test that triggers queries route correctly."""
        with patch(
            "src.api.routes.chatbot_tools._query_triggers", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = {"success": True, "data": []}

            await e2i_data_query_tool.ainvoke(
                {
                    "query_type": E2IQueryType.TRIGGERS.value,
                }
            )

        mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_experiments_query_uses_rag(self):
        """Test that experiments queries use RAG."""
        with patch(
            "src.api.routes.chatbot_tools._query_via_rag", new_callable=AsyncMock
        ) as mock_rag:
            mock_rag.return_value = {"success": True, "data": []}

            await e2i_data_query_tool.ainvoke(
                {
                    "query_type": E2IQueryType.EXPERIMENTS.value,
                    "brand": "Kisqali",
                }
            )

        mock_rag.assert_called_once()


# =============================================================================
# causal_analysis_tool Tests
# =============================================================================


class TestCausalAnalysisTool:
    """Tests for causal_analysis_tool."""

    @pytest.mark.asyncio
    async def test_causal_analysis_success(self):
        """Test successful causal analysis."""
        mock_result = MagicMock()
        mock_result.source_id = "causal-1"
        mock_result.content = "TRx driven by HCP engagement"
        mock_result.score = 0.85
        mock_result.metadata = {"confidence": 0.9, "effect_magnitude": 0.3}

        with patch(
            "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = [mock_result]

            result = await causal_analysis_tool.ainvoke(
                {
                    "kpi_name": "TRx",
                    "brand": "Kisqali",
                    "min_confidence": 0.7,
                }
            )

        assert result["success"] is True
        assert result["kpi_analyzed"] == "TRx"
        assert result["brand"] == "Kisqali"
        assert result["min_confidence_applied"] == 0.7
        assert result["causal_chains_found"] == 1

    @pytest.mark.asyncio
    async def test_causal_analysis_filters_by_confidence(self):
        """Test that results are filtered by min_confidence."""
        mock_low_conf = MagicMock()
        mock_low_conf.score = 0.5
        mock_low_conf.metadata = {"confidence": 0.5}

        mock_high_conf = MagicMock()
        mock_high_conf.source_id = "high"
        mock_high_conf.content = "High confidence result"
        mock_high_conf.score = 0.9
        mock_high_conf.metadata = {"confidence": 0.9}

        with patch(
            "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = [mock_low_conf, mock_high_conf]

            result = await causal_analysis_tool.ainvoke(
                {
                    "kpi_name": "TRx",
                    "min_confidence": 0.8,
                }
            )

        # Only high confidence result should be included
        assert result["causal_chains_found"] == 1

    @pytest.mark.asyncio
    async def test_causal_analysis_error_handling(self):
        """Test causal analysis error handling."""
        with patch(
            "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.side_effect = Exception("RAG unavailable")

            result = await causal_analysis_tool.ainvoke({"kpi_name": "TRx"})

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# agent_routing_tool Tests
# =============================================================================


class TestAgentRoutingTool:
    """Tests for agent_routing_tool."""

    @pytest.mark.asyncio
    async def test_explicit_agent_routing(self):
        """Test routing with explicit target agent."""
        result = await agent_routing_tool.ainvoke(
            {
                "query": "Analyze causal factors",
                "target_agent": "causal_impact",
            }
        )

        assert result["success"] is True
        assert result["routed_to"] == "causal_impact"
        assert result["routing_confidence"] == 1.0
        assert result["routing_method"] == "explicit"

    @pytest.mark.asyncio
    async def test_invalid_target_agent(self):
        """Test routing with invalid target agent."""
        result = await agent_routing_tool.ainvoke(
            {
                "query": "Test query",
                "target_agent": "invalid_agent",
            }
        )

        assert result["success"] is False
        assert "Unknown agent" in result["error"]
        assert "available_agents" in result

    @pytest.mark.asyncio
    async def test_dspy_routing(self):
        """Test DSPy-based routing."""
        with patch(
            "src.api.routes.chatbot_tools.route_agent_dspy", new_callable=AsyncMock
        ) as mock_dspy:
            mock_dspy.return_value = (
                "causal_impact",
                ["gap_analyzer"],
                0.85,
                "Causal query detected",
                "dspy",
            )

            result = await agent_routing_tool.ainvoke(
                {
                    "query": "Why did TRx drop?",
                    "context": {"intent": "causal_analysis"},
                }
            )

        assert result["success"] is True
        assert result["routed_to"] == "causal_impact"
        assert result["routing_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_routing_fallback_on_error(self):
        """Test fallback to hardcoded routing on DSPy error."""
        with patch(
            "src.api.routes.chatbot_tools.route_agent_dspy", new_callable=AsyncMock
        ) as mock_dspy:
            mock_dspy.side_effect = Exception("DSPy failed")

            with patch("src.api.routes.chatbot_tools.route_agent_hardcoded") as mock_hardcoded:
                mock_hardcoded.return_value = (
                    "explainer",
                    [],
                    0.6,
                    "Default routing",
                )

                result = await agent_routing_tool.ainvoke({"query": "General question"})

        assert result["success"] is True
        assert result["routing_method"] == "hardcoded_fallback"


# =============================================================================
# conversation_memory_tool Tests
# =============================================================================


class TestConversationMemoryTool:
    """Tests for conversation_memory_tool."""

    @pytest.mark.asyncio
    async def test_conversation_memory_success(self):
        """Test successful conversation memory retrieval."""
        mock_conversation = {
            "title": "TRx Analysis",
            "brand_context": "Kisqali",
            "region_context": "US",
        }
        mock_messages = [
            {
                "role": "user",
                "content": "What is TRx?",
                "created_at": "2024-01-01T00:00:00Z",
                "agent_name": None,
                "tool_calls": [],
                "tool_results": [],
            },
        ]

        with patch("src.api.routes.chatbot_tools.get_async_supabase_client") as mock_client:
            mock_conv_repo = AsyncMock()
            mock_conv_repo.get_by_session_id.return_value = mock_conversation
            mock_msg_repo = AsyncMock()
            mock_msg_repo.get_recent_messages.return_value = mock_messages
            mock_client.return_value = MagicMock()

            with patch(
                "src.api.routes.chatbot_tools.get_chatbot_conversation_repository",
                return_value=mock_conv_repo,
            ):
                with patch(
                    "src.api.routes.chatbot_tools.get_chatbot_message_repository",
                    return_value=mock_msg_repo,
                ):
                    result = await conversation_memory_tool.ainvoke(
                        {
                            "session_id": "session-123",
                            "message_count": 10,
                        }
                    )

        assert result["success"] is True
        assert result["session_id"] == "session-123"
        assert result["conversation_title"] == "TRx Analysis"
        assert result["message_count"] == 1

    @pytest.mark.asyncio
    async def test_conversation_not_found(self):
        """Test handling of missing conversation."""
        with patch("src.api.routes.chatbot_tools.get_async_supabase_client") as mock_client:
            mock_conv_repo = AsyncMock()
            mock_conv_repo.get_by_session_id.return_value = None
            mock_client.return_value = MagicMock()

            with patch(
                "src.api.routes.chatbot_tools.get_chatbot_conversation_repository",
                return_value=mock_conv_repo,
            ):
                result = await conversation_memory_tool.ainvoke({"session_id": "unknown-session"})

        assert result["success"] is False
        assert "Conversation not found" in result["error"]


# =============================================================================
# document_retrieval_tool Tests
# =============================================================================


class TestDocumentRetrievalTool:
    """Tests for document_retrieval_tool."""

    @pytest.mark.asyncio
    async def test_document_retrieval_success(self):
        """Test successful document retrieval."""
        mock_result = MagicMock()
        mock_result.source_id = "doc-1"
        mock_result.content = "TRx trends for Kisqali"
        mock_result.score = 0.92
        mock_result.source = "business_metrics"
        mock_result.retrieval_method = "hybrid"
        mock_result.metadata = {"brand": "Kisqali"}

        with patch(
            "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = [mock_result]

            result = await document_retrieval_tool.ainvoke(
                {
                    "query": "TRx trends",
                    "k": 5,
                    "brand": "Kisqali",
                }
            )

        assert result["success"] is True
        assert result["document_count"] == 1
        assert result["documents"][0]["source_id"] == "doc-1"
        assert result["filters_applied"]["brand"] == "Kisqali"

    @pytest.mark.asyncio
    async def test_document_retrieval_error(self):
        """Test document retrieval error handling."""
        with patch(
            "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.side_effect = Exception("Search failed")

            result = await document_retrieval_tool.ainvoke({"query": "test"})

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# orchestrator_tool Tests
# =============================================================================


class TestOrchestratorTool:
    """Tests for orchestrator_tool."""

    @pytest.mark.asyncio
    async def test_orchestrator_success(self):
        """Test successful orchestrator execution."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run.return_value = {
            "response_text": "Analysis complete",
            "response_confidence": 0.9,
            "agents_dispatched": ["causal_impact"],
            "analysis_results": {"key": "value"},
        }

        with patch("src.api.routes.chatbot_tools.get_orchestrator", return_value=mock_orchestrator):
            result = await orchestrator_tool.ainvoke(
                {
                    "query": "Analyze TRx",
                    "brand": "Kisqali",
                }
            )

        assert result["success"] is True
        assert result["fallback"] is False
        assert result["response"] == "Analysis complete"
        assert result["agents_dispatched"] == ["causal_impact"]

    @pytest.mark.asyncio
    async def test_orchestrator_fallback_when_unavailable(self):
        """Test fallback to RAG when orchestrator unavailable."""
        mock_rag_result = MagicMock()
        mock_rag_result.content = "RAG result"
        mock_rag_result.score = 0.8
        mock_rag_result.source = "documents"

        with patch("src.api.routes.chatbot_tools.get_orchestrator", return_value=None):
            with patch(
                "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = [mock_rag_result]

                result = await orchestrator_tool.ainvoke({"query": "Test query"})

        assert result["success"] is True
        assert result["fallback"] is True
        assert "Orchestrator unavailable" in result["reason"]


# =============================================================================
# tool_composer_tool Tests
# =============================================================================


class TestToolComposerTool:
    """Tests for tool_composer_tool."""

    @pytest.mark.asyncio
    async def test_tool_composer_success(self):
        """Test successful tool composer execution."""
        # Create mock composition result
        mock_sub_q = MagicMock()
        mock_sub_q.id = "sq-1"
        mock_sub_q.question = "What is TRx?"
        mock_sub_q.intent = "kpi_query"

        mock_decomposition = MagicMock()
        mock_decomposition.sub_questions = [mock_sub_q]

        mock_plan = MagicMock()
        mock_plan.get_execution_order.return_value = ["sq-1"]
        mock_plan.parallel_groups = [["sq-1"]]

        mock_execution = MagicMock()
        mock_execution.tools_executed = ["e2i_data_query_tool"]
        mock_execution.get_all_outputs.return_value = {"sq-1": {"result": "100"}}

        mock_response = MagicMock()
        mock_response.answer = "TRx is 100"
        mock_response.confidence = 0.9

        mock_result = MagicMock()
        mock_result.decomposition = mock_decomposition
        mock_result.plan = mock_plan
        mock_result.execution = mock_execution
        mock_result.response = mock_response

        with patch(
            "src.api.routes.chatbot_tools.compose_query", new_callable=AsyncMock
        ) as mock_compose:
            mock_compose.return_value = mock_result

            with patch("src.api.routes.chatbot_tools.get_chat_llm"):
                result = await tool_composer_tool.ainvoke(
                    {
                        "query": "What is TRx and why did it change?",
                        "brand": "Kisqali",
                    }
                )

        assert result["success"] is True
        assert result["synthesized_response"] == "TRx is 100"
        assert result["confidence"] == 0.9
        assert len(result["sub_questions"]) == 1

    @pytest.mark.asyncio
    async def test_tool_composer_error_with_orchestrator_fallback(self):
        """Test tool composer falls back to orchestrator on error."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run.return_value = {
            "response_text": "Fallback response",
            "response_confidence": 0.7,
        }

        with patch(
            "src.api.routes.chatbot_tools.compose_query", new_callable=AsyncMock
        ) as mock_compose:
            mock_compose.side_effect = Exception("Composition failed")

            with patch("src.api.routes.chatbot_tools.get_chat_llm"):
                with patch(
                    "src.api.routes.chatbot_tools.get_orchestrator",
                    return_value=mock_orchestrator,
                ):
                    result = await tool_composer_tool.ainvoke({"query": "Multi-faceted query"})

        assert result["success"] is True
        assert result["fallback"] is True
        assert result["response"] == "Fallback response"


# =============================================================================
# Tool Exports Tests
# =============================================================================


class TestToolExports:
    """Tests for tool exports and mappings."""

    def test_e2i_chatbot_tools_list(self):
        """Test E2I_CHATBOT_TOOLS contains all tools."""
        assert len(E2I_CHATBOT_TOOLS) == 7
        assert e2i_data_query_tool in E2I_CHATBOT_TOOLS
        assert causal_analysis_tool in E2I_CHATBOT_TOOLS
        assert agent_routing_tool in E2I_CHATBOT_TOOLS
        assert conversation_memory_tool in E2I_CHATBOT_TOOLS
        assert document_retrieval_tool in E2I_CHATBOT_TOOLS
        assert orchestrator_tool in E2I_CHATBOT_TOOLS
        assert tool_composer_tool in E2I_CHATBOT_TOOLS

    def test_e2i_tool_map(self):
        """Test E2I_TOOL_MAP contains all tools."""
        assert len(E2I_TOOL_MAP) == 7
        assert E2I_TOOL_MAP["e2i_data_query_tool"] == e2i_data_query_tool
        assert E2I_TOOL_MAP["causal_analysis_tool"] == causal_analysis_tool
        assert E2I_TOOL_MAP["agent_routing_tool"] == agent_routing_tool
        assert E2I_TOOL_MAP["conversation_memory_tool"] == conversation_memory_tool
        assert E2I_TOOL_MAP["document_retrieval_tool"] == document_retrieval_tool
        assert E2I_TOOL_MAP["orchestrator_tool"] == orchestrator_tool
        assert E2I_TOOL_MAP["tool_composer_tool"] == tool_composer_tool

    def test_get_e2i_chatbot_tools(self):
        """Test get_e2i_chatbot_tools function."""
        tools = get_e2i_chatbot_tools()
        assert tools == E2I_CHATBOT_TOOLS
        assert len(tools) == 7

    def test_get_tool_by_name_valid(self):
        """Test get_tool_by_name with valid names."""
        assert get_tool_by_name("e2i_data_query_tool") == e2i_data_query_tool
        assert get_tool_by_name("causal_analysis_tool") == causal_analysis_tool
        assert get_tool_by_name("orchestrator_tool") == orchestrator_tool

    def test_get_tool_by_name_invalid(self):
        """Test get_tool_by_name with invalid name."""
        assert get_tool_by_name("nonexistent_tool") is None
        assert get_tool_by_name("") is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for chatbot tools."""

    @pytest.mark.asyncio
    async def test_data_query_to_causal_analysis_flow(self):
        """Test flow from data query to causal analysis."""
        # First query KPIs
        with patch("src.api.routes.chatbot_tools._query_kpis", new_callable=AsyncMock) as mock_kpi:
            mock_kpi.return_value = {
                "success": True,
                "data": [{"metric_name": "TRx", "value": 100}],
            }

            kpi_result = await e2i_data_query_tool.ainvoke(
                {
                    "query_type": E2IQueryType.KPI.value,
                    "kpi_name": "TRx",
                }
            )

        assert kpi_result["success"] is True

        # Then run causal analysis on the same KPI
        mock_causal = MagicMock()
        mock_causal.source_id = "causal-1"
        mock_causal.content = "HCP engagement drives TRx"
        mock_causal.score = 0.9
        mock_causal.metadata = {"confidence": 0.9}

        with patch(
            "src.api.routes.chatbot_tools.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = [mock_causal]

            causal_result = await causal_analysis_tool.ainvoke(
                {
                    "kpi_name": "TRx",
                    "min_confidence": 0.7,
                }
            )

        assert causal_result["success"] is True
        assert causal_result["causal_chains_found"] == 1

    @pytest.mark.asyncio
    async def test_routing_to_orchestrator_flow(self):
        """Test routing decision followed by orchestrator execution."""
        # First, route the query
        with patch(
            "src.api.routes.chatbot_tools.route_agent_dspy", new_callable=AsyncMock
        ) as mock_route:
            mock_route.return_value = (
                "causal_impact",
                [],
                0.85,
                "Causal query",
                "dspy",
            )

            routing_result = await agent_routing_tool.ainvoke({"query": "Why did TRx drop?"})

        assert routing_result["routed_to"] == "causal_impact"

        # Then execute through orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run.return_value = {
            "response_text": "TRx dropped due to HCP engagement decline",
            "response_confidence": 0.9,
            "agents_dispatched": ["causal_impact"],
            "analysis_results": {},
        }

        with patch(
            "src.api.routes.chatbot_tools.get_orchestrator",
            return_value=mock_orchestrator,
        ):
            orch_result = await orchestrator_tool.ainvoke(
                {
                    "query": "Why did TRx drop?",
                    "target_agent": "causal_impact",
                }
            )

        assert orch_result["success"] is True
        assert "causal_impact" in orch_result["agents_dispatched"]
