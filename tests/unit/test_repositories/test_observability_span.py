"""Tests for ObservabilitySpanRepository.

Version: 1.0.0
Tests the observability span repository CRUD operations.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.agents.ml_foundation.observability_connector.models import (
    AgentNameEnum,
    AgentTierEnum,
    ObservabilitySpan,
    SpanStatusEnum,
)
from src.repositories.observability_span import ObservabilitySpanRepository


class TestObservabilitySpanRepository:
    """Test ObservabilitySpanRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        repo = ObservabilitySpanRepository()
        repo.client = mock_client
        return repo

    @pytest.fixture
    def sample_span(self):
        """Create a sample span for testing."""
        return ObservabilitySpan(
            trace_id="trace-123",
            span_id="span-456",
            agent_name=AgentNameEnum.ORCHESTRATOR,
            agent_tier=AgentTierEnum.COORDINATION,
            started_at=datetime.now(timezone.utc),
            duration_ms=150,
            status=SpanStatusEnum.SUCCESS,
        )

    # =========================================================================
    # INSERT TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_insert_span(self, repo, mock_client, sample_span):
        """Test inserting a single span."""
        mock_execute = AsyncMock(
            return_value=MagicMock(data=[sample_span.to_db_dict()])
        )
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.insert_span(sample_span)

        assert result is not None
        mock_client.table.assert_called_with("ml_observability_spans")

    @pytest.mark.asyncio
    async def test_insert_span_without_client(self, sample_span):
        """Test insert_span returns None without client."""
        repo = ObservabilitySpanRepository()
        repo.client = None

        result = await repo.insert_span(sample_span)

        assert result is None

    @pytest.mark.asyncio
    async def test_insert_spans_batch(self, repo, mock_client):
        """Test inserting multiple spans."""
        spans = [
            ObservabilitySpan(
                trace_id=f"trace-{i}",
                span_id=f"span-{i}",
                agent_name=AgentNameEnum.ORCHESTRATOR,
                agent_tier=AgentTierEnum.COORDINATION,
                started_at=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        mock_data = [s.to_db_dict() for s in spans]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.insert_spans_batch(spans)

        assert result["success"] is True
        assert result["inserted_count"] == 3
        assert result["failed_count"] == 0

    @pytest.mark.asyncio
    async def test_insert_spans_batch_empty(self, repo):
        """Test inserting empty batch."""
        result = await repo.insert_spans_batch([])

        assert result["success"] is False
        assert result["inserted_count"] == 0

    @pytest.mark.asyncio
    async def test_insert_spans_batch_without_client(self):
        """Test batch insert without client."""
        repo = ObservabilitySpanRepository()
        repo.client = None

        spans = [
            ObservabilitySpan(
                trace_id="trace-1",
                span_id="span-1",
                agent_name=AgentNameEnum.ORCHESTRATOR,
                agent_tier=AgentTierEnum.COORDINATION,
                started_at=datetime.now(timezone.utc),
            )
        ]

        result = await repo.insert_spans_batch(spans)

        assert result["success"] is False
        assert result["failed_count"] == 1

    # =========================================================================
    # TIME-BASED QUERY TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_spans_by_time_window_24h(self, repo, mock_client):
        """Test getting spans from last 24 hours."""
        mock_data = [
            {
                "id": str(uuid4()),
                "trace_id": "trace-1",
                "span_id": "span-1",
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
            }
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value.limit.return_value = mock_query
        mock_client.table.return_value.select.return_value.gte.return_value = mock_query

        result = await repo.get_spans_by_time_window(window="24h")

        assert len(result) == 1
        mock_client.table.assert_called_with("ml_observability_spans")

    @pytest.mark.asyncio
    async def test_get_spans_by_time_window_with_agent_filter(self, repo, mock_client):
        """Test getting spans filtered by agent."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value.limit.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_client.table.return_value.select.return_value.gte.return_value = mock_query

        await repo.get_spans_by_time_window(window="1h", agent_name="orchestrator")

        # Verify agent filter was applied
        mock_query.eq.assert_called()

    def test_parse_time_window(self, repo):
        """Test time window parsing."""
        assert repo._parse_time_window("1h") == 1
        assert repo._parse_time_window("24h") == 24
        assert repo._parse_time_window("7d") == 168
        assert repo._parse_time_window("1w") == 168
        assert repo._parse_time_window("invalid") == 24  # Default

    # =========================================================================
    # TRACE QUERY TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_spans_by_trace_id(self, repo, mock_client):
        """Test getting all spans for a trace."""
        mock_data = [
            {
                "id": str(uuid4()),
                "trace_id": "trace-123",
                "span_id": f"span-{i}",
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
            }
            for i in range(3)
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value = mock_query

        result = await repo.get_spans_by_trace_id("trace-123")

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_spans_by_trace_id_empty(self, repo, mock_client):
        """Test getting spans for non-existent trace."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value = mock_query

        result = await repo.get_spans_by_trace_id("non-existent")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_trace_summary(self, repo, mock_client):
        """Test getting trace summary."""
        now = datetime.now(timezone.utc)
        mock_data = [
            {
                "id": str(uuid4()),
                "trace_id": "trace-123",
                "span_id": "span-1",
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "started_at": now.isoformat(),
                "ended_at": (now + timedelta(milliseconds=100)).isoformat(),
                "duration_ms": 100,
                "status": "success",
                "total_tokens": 500,
            },
            {
                "id": str(uuid4()),
                "trace_id": "trace-123",
                "span_id": "span-2",
                "agent_name": "causal_impact",
                "agent_tier": "causal_analytics",
                "started_at": now.isoformat(),
                "ended_at": (now + timedelta(milliseconds=200)).isoformat(),
                "duration_ms": 200,
                "status": "error",
                "total_tokens": 300,
            },
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value = mock_query

        result = await repo.get_trace_summary("trace-123")

        assert result["trace_id"] == "trace-123"
        assert result["span_count"] == 2
        assert result["agent_count"] == 2
        assert result["error_count"] == 1
        assert result["total_tokens"] == 800

    # =========================================================================
    # AGENT QUERY TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_spans_by_agent(self, repo, mock_client):
        """Test getting spans for specific agent."""
        mock_data = [
            {
                "id": str(uuid4()),
                "trace_id": "trace-1",
                "span_id": "span-1",
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
            }
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value.limit.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.gte.return_value = (
            mock_query
        )

        result = await repo.get_spans_by_agent("orchestrator")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_spans_by_tier(self, repo, mock_client):
        """Test getting spans for specific tier."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value.limit.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.gte.return_value = (
            mock_query
        )

        result = await repo.get_spans_by_tier("coordination")

        assert result == []

    # =========================================================================
    # LATENCY STATS TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_latency_stats(self, repo, mock_client):
        """Test getting latency stats from view."""
        mock_data = [
            {
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "total_spans": 100,
                "avg_duration_ms": 150.5,
                "p50_ms": 120.0,
                "p95_ms": 300.0,
                "p99_ms": 500.0,
                "error_rate": 0.02,
                "fallback_rate": 0.05,
                "total_tokens_used": 50000,
            }
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_client.table.return_value.select.return_value = mock_query

        result = await repo.get_latency_stats()

        assert len(result) == 1
        assert result[0].total_spans == 100
        assert result[0].p95_ms == 300.0

    @pytest.mark.asyncio
    async def test_get_latency_stats_with_filter(self, repo, mock_client):
        """Test getting latency stats filtered by agent."""
        mock_execute = AsyncMock(return_value=MagicMock(data=[]))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.eq.return_value = mock_query
        mock_client.table.return_value.select.return_value = mock_query

        await repo.get_latency_stats(agent_name="orchestrator")

        mock_query.eq.assert_called_with("agent_name", "orchestrator")

    @pytest.mark.asyncio
    async def test_get_quality_metrics(self, repo, mock_client):
        """Test computing quality metrics."""
        now = datetime.now(timezone.utc)
        mock_data = [
            {
                "id": str(uuid4()),
                "trace_id": f"trace-{i}",
                "span_id": f"span-{i}",
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "started_at": now.isoformat(),
                "ended_at": (now + timedelta(milliseconds=100 + i * 50)).isoformat(),
                "duration_ms": 100 + i * 50,
                "status": "success" if i < 9 else "error",
                "total_tokens": 100,
                "fallback_used": i == 5,
            }
            for i in range(10)
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value.limit.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_client.table.return_value.select.return_value.gte.return_value = mock_query

        result = await repo.get_quality_metrics(time_window="24h")

        assert result.total_spans == 10
        assert result.error_count == 1
        assert result.success_rate == 0.9
        assert result.quality_score > 0

    # =========================================================================
    # RETENTION CLEANUP TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_delete_old_spans(self, repo, mock_client):
        """Test deleting old spans."""
        mock_data = [{"id": str(uuid4())} for _ in range(5)]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.limit.return_value = mock_query
        mock_client.table.return_value.delete.return_value.lt.return_value = mock_query

        result = await repo.delete_old_spans(retention_days=30)

        assert result["deleted_count"] == 5
        assert result["cutoff_date"] != ""

    @pytest.mark.asyncio
    async def test_delete_old_spans_without_client(self):
        """Test delete without client."""
        repo = ObservabilitySpanRepository()
        repo.client = None

        result = await repo.delete_old_spans()

        assert result["deleted_count"] == 0

    # =========================================================================
    # ERROR ANALYSIS TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_error_spans(self, repo, mock_client):
        """Test getting error spans."""
        mock_data = [
            {
                "id": str(uuid4()),
                "trace_id": "trace-err",
                "span_id": "span-err",
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error_type": "ValidationError",
                "error_message": "Invalid input",
            }
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value.limit.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.gte.return_value = (
            mock_query
        )

        result = await repo.get_error_spans()

        assert len(result) == 1
        assert result[0].status == SpanStatusEnum.ERROR

    @pytest.mark.asyncio
    async def test_get_fallback_spans(self, repo, mock_client):
        """Test getting fallback spans."""
        mock_data = [
            {
                "id": str(uuid4()),
                "trace_id": "trace-fb",
                "span_id": "span-fb",
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "fallback_used": True,
                "fallback_chain": ["claude-3-5-sonnet", "gpt-4"],
            }
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=mock_data))
        mock_query = MagicMock()
        mock_query.execute = mock_execute
        mock_query.order.return_value.limit.return_value = mock_query
        mock_client.table.return_value.select.return_value.eq.return_value.gte.return_value = (
            mock_query
        )

        result = await repo.get_fallback_spans()

        assert len(result) == 1
        assert result[0].fallback_used is True


class TestObservabilitySpanRepositoryPercentile:
    """Test percentile calculation."""

    @pytest.fixture
    def repo(self):
        """Create repository without client."""
        return ObservabilitySpanRepository()

    def test_percentile_empty_list(self, repo):
        """Test percentile with empty list."""
        result = repo._percentile([], 0.5)
        assert result == 0.0

    def test_percentile_single_value(self, repo):
        """Test percentile with single value."""
        result = repo._percentile([100], 0.5)
        assert result == 100.0

    def test_percentile_p50(self, repo):
        """Test 50th percentile (median)."""
        result = repo._percentile([10, 20, 30, 40, 50], 0.5)
        assert result == 30.0

    def test_percentile_p95(self, repo):
        """Test 95th percentile."""
        data = list(range(1, 101))  # 1 to 100
        result = repo._percentile(data, 0.95)
        assert result >= 95.0

    def test_percentile_p99(self, repo):
        """Test 99th percentile."""
        data = list(range(1, 101))  # 1 to 100
        result = repo._percentile(data, 0.99)
        assert result >= 99.0


class TestObservabilitySpanRepositoryWithoutClient:
    """Test repository behavior without client."""

    @pytest.fixture
    def repo(self):
        """Create repository without client."""
        repo = ObservabilitySpanRepository()
        repo.client = None
        return repo

    @pytest.mark.asyncio
    async def test_get_spans_by_time_window_without_client(self, repo):
        """Test get_spans_by_time_window returns empty without client."""
        result = await repo.get_spans_by_time_window()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_spans_by_trace_id_without_client(self, repo):
        """Test get_spans_by_trace_id returns empty without client."""
        result = await repo.get_spans_by_trace_id("trace-123")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_spans_by_agent_without_client(self, repo):
        """Test get_spans_by_agent returns empty without client."""
        result = await repo.get_spans_by_agent("orchestrator")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_latency_stats_without_client(self, repo):
        """Test get_latency_stats returns empty without client."""
        result = await repo.get_latency_stats()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_error_spans_without_client(self, repo):
        """Test get_error_spans returns empty without client."""
        result = await repo.get_error_spans()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_fallback_spans_without_client(self, repo):
        """Test get_fallback_spans returns empty without client."""
        result = await repo.get_fallback_spans()
        assert result == []
