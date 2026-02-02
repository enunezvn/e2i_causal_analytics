"""
Unit tests for E2I RAG Search Logger.

Tests for SearchLogger class and its integration with SearchStats.
All external dependencies are mocked.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from src.rag.search_logger import SearchLogger
from src.rag.types import SearchStats

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = MagicMock()
    return client


@pytest.fixture
def search_logger(mock_supabase_client):
    """Create a SearchLogger instance with mock client."""
    return SearchLogger(supabase_client=mock_supabase_client)


@pytest.fixture
def sample_search_stats():
    """Create sample SearchStats for testing."""
    return SearchStats(
        query="TRx conversion Remibrutinib",
        total_latency_ms=150.5,
        vector_count=5,
        fulltext_count=3,
        graph_count=2,
        fused_count=8,
        sources_used={"vector": True, "fulltext": True, "graph": True},
        vector_latency_ms=80.0,
        fulltext_latency_ms=30.0,
        graph_latency_ms=40.0,
        fusion_latency_ms=5.0,
        timestamp=datetime.now(timezone.utc),
        errors=[],
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSearchLoggerInit:
    """Tests for SearchLogger initialization."""

    def test_init_with_defaults(self, mock_supabase_client):
        """Test initialization with default settings."""
        logger = SearchLogger(supabase_client=mock_supabase_client)
        assert logger.client == mock_supabase_client
        assert logger.enabled is True

    def test_init_disabled(self, mock_supabase_client):
        """Test initialization with logging disabled."""
        logger = SearchLogger(supabase_client=mock_supabase_client, enabled=False)
        assert logger.enabled is False

    def test_repr(self, search_logger):
        """Test string representation."""
        repr_str = repr(search_logger)
        assert "SearchLogger" in repr_str
        assert "enabled=True" in repr_str


# ============================================================================
# Log Search Tests
# ============================================================================


class TestSearchLoggerLogSearch:
    """Tests for SearchLogger.log_search method."""

    @pytest.mark.asyncio
    async def test_log_search_success(
        self, mock_supabase_client, search_logger, sample_search_stats
    ):
        """Test successful search logging."""
        expected_log_id = str(uuid4())
        mock_response = MagicMock()
        mock_response.data = expected_log_id
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        log_id = await search_logger.log_search(sample_search_stats)

        assert log_id == UUID(expected_log_id)
        mock_supabase_client.rpc.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_search_with_session_and_user(
        self, mock_supabase_client, search_logger, sample_search_stats
    ):
        """Test logging with session and user context."""
        expected_log_id = str(uuid4())
        mock_response = MagicMock()
        mock_response.data = expected_log_id
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        session_id = uuid4()
        user_id = "user-123"

        log_id = await search_logger.log_search(
            sample_search_stats, session_id=session_id, user_id=user_id
        )

        assert log_id is not None
        # Verify RPC was called with correct parameters
        call_args = mock_supabase_client.rpc.call_args
        assert call_args[0][0] == "log_rag_search"
        params = call_args[0][1]
        assert params["p_session_id"] == str(session_id)
        assert params["p_user_id"] == user_id

    @pytest.mark.asyncio
    async def test_log_search_disabled(self, mock_supabase_client, sample_search_stats):
        """Test that logging is skipped when disabled."""
        logger = SearchLogger(supabase_client=mock_supabase_client, enabled=False)

        log_id = await logger.log_search(sample_search_stats)

        assert log_id is None
        mock_supabase_client.rpc.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_search_error_handling(
        self, mock_supabase_client, search_logger, sample_search_stats
    ):
        """Test that logging errors don't propagate."""
        mock_supabase_client.rpc.side_effect = Exception("Database error")

        # Should not raise, just return None
        log_id = await search_logger.log_search(sample_search_stats)

        assert log_id is None

    @pytest.mark.asyncio
    async def test_log_search_with_config_and_entities(
        self, mock_supabase_client, search_logger, sample_search_stats
    ):
        """Test logging with search configuration and extracted entities."""
        expected_log_id = str(uuid4())
        mock_response = MagicMock()
        mock_response.data = expected_log_id
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        config = {"vector_top_k": 20, "rrf_k": 60}
        entities = {"brands": ["Remibrutinib"], "kpis": ["TRx"]}

        log_id = await search_logger.log_search(
            sample_search_stats, config=config, extracted_entities=entities
        )

        assert log_id is not None
        call_args = mock_supabase_client.rpc.call_args
        params = call_args[0][1]
        assert params["p_config"] == config
        assert params["p_extracted_entities"] == entities


# ============================================================================
# Batch Logging Tests
# ============================================================================


class TestSearchLoggerBatchLog:
    """Tests for SearchLogger.log_search_batch method."""

    @pytest.mark.asyncio
    async def test_log_batch_success(self, mock_supabase_client, search_logger):
        """Test successful batch logging."""
        stats_list = [
            SearchStats(
                query=f"query {i}",
                total_latency_ms=100.0 + i * 10,
                vector_count=5,
                fulltext_count=3,
                graph_count=2,
                fused_count=8,
                sources_used={"vector": True, "fulltext": True, "graph": False},
            )
            for i in range(3)
        ]

        mock_response = MagicMock()
        mock_response.data = str(uuid4())
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        success_count = await search_logger.log_search_batch(stats_list)

        assert success_count == 3
        assert mock_supabase_client.rpc.call_count == 3

    @pytest.mark.asyncio
    async def test_log_batch_disabled(self, mock_supabase_client):
        """Test that batch logging is skipped when disabled."""
        logger = SearchLogger(supabase_client=mock_supabase_client, enabled=False)

        stats_list = [
            SearchStats(
                query="test",
                total_latency_ms=100.0,
                vector_count=1,
                fulltext_count=1,
                graph_count=0,
                fused_count=2,
                sources_used={},
            )
        ]

        success_count = await logger.log_search_batch(stats_list)

        assert success_count == 0
        mock_supabase_client.rpc.assert_not_called()


# ============================================================================
# Analytics Tests
# ============================================================================


class TestSearchLoggerAnalytics:
    """Tests for SearchLogger analytics methods."""

    @pytest.mark.asyncio
    async def test_get_slow_queries(self, mock_supabase_client, search_logger):
        """Test retrieving slow queries."""
        mock_response = MagicMock()
        mock_response.data = [
            {"log_id": str(uuid4()), "query": "slow query 1", "total_latency_ms": 2000.0},
            {"log_id": str(uuid4()), "query": "slow query 2", "total_latency_ms": 1500.0},
        ]
        mock_supabase_client.from_.return_value.select.return_value.gt.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        slow_queries = await search_logger.get_slow_queries(limit=10, threshold_ms=1000.0)

        assert len(slow_queries) == 2
        assert slow_queries[0]["total_latency_ms"] == 2000.0

    @pytest.mark.asyncio
    async def test_get_slow_queries_error(self, mock_supabase_client, search_logger):
        """Test that slow queries errors return empty list."""
        mock_supabase_client.from_.side_effect = Exception("Database error")

        slow_queries = await search_logger.get_slow_queries()

        assert slow_queries == []

    @pytest.mark.asyncio
    async def test_get_search_stats_summary(self, mock_supabase_client, search_logger):
        """Test retrieving search stats summary."""
        mock_response = MagicMock()
        mock_response.data = [
            {
                "hour": "2025-12-20T10:00:00Z",
                "query_count": 100,
                "avg_latency_ms": 150.0,
                "p95_latency_ms": 400.0,
                "error_count": 2,
            },
            {
                "hour": "2025-12-20T09:00:00Z",
                "query_count": 80,
                "avg_latency_ms": 120.0,
                "p95_latency_ms": 350.0,
                "error_count": 1,
            },
        ]
        mock_supabase_client.from_.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        summary = await search_logger.get_search_stats_summary(hours=24)

        assert summary["total_queries"] == 180
        assert summary["avg_latency_ms"] == 135.0  # (150 + 120) / 2
        assert summary["p95_latency_ms"] == 400.0  # max of 400, 350
        assert summary["error_rate"] == round(3 / 180 * 100, 2)

    @pytest.mark.asyncio
    async def test_get_search_stats_summary_empty(self, mock_supabase_client, search_logger):
        """Test stats summary with no data."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase_client.from_.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        summary = await search_logger.get_search_stats_summary(hours=24)

        assert summary["total_queries"] == 0
        assert summary["avg_latency_ms"] == 0
        assert summary["error_rate"] == 0

    @pytest.mark.asyncio
    async def test_get_search_stats_summary_error(self, mock_supabase_client, search_logger):
        """Test that stats summary errors return default values."""
        mock_supabase_client.from_.side_effect = Exception("Database error")

        summary = await search_logger.get_search_stats_summary()

        assert summary["total_queries"] == 0
        assert "error" in summary


# ============================================================================
# Integration with SearchStats Tests
# ============================================================================


class TestSearchStatsIntegration:
    """Tests for SearchStats dataclass integration."""

    def test_search_stats_to_dict(self, sample_search_stats):
        """Test SearchStats.to_dict for logging."""
        stats_dict = sample_search_stats.to_dict()

        assert stats_dict["query"] == "TRx conversion Remibrutinib"
        assert stats_dict["total_latency_ms"] == 150.5
        assert stats_dict["vector_count"] == 5
        assert stats_dict["fulltext_count"] == 3
        assert stats_dict["graph_count"] == 2
        assert stats_dict["fused_count"] == 8

    def test_search_stats_with_errors(self):
        """Test SearchStats with error tracking."""
        stats = SearchStats(
            query="failing query",
            total_latency_ms=5000.0,
            vector_count=0,
            fulltext_count=0,
            graph_count=0,
            fused_count=0,
            sources_used={"vector": False, "fulltext": False, "graph": False},
            errors=["Vector timeout", "Graph connection failed"],
        )

        assert len(stats.errors) == 2
        assert "Vector timeout" in stats.errors
