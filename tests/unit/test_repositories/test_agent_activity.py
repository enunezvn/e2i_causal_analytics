"""
Unit tests for AgentActivityRepository.

Tests agent output queries, analysis result retrieval, and activity tracking.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.repositories.agent_activity import AgentActivityRepository


@pytest.mark.unit
class TestAgentActivityRepository:
    """Tests for AgentActivityRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return AgentActivityRepository(supabase_client=mock_client)

    @pytest.fixture
    def sample_activities(self):
        """Sample agent activity data."""
        return [
            {
                "activity_id": str(uuid4()),
                "agent_name": "causal_impact",
                "agent_tier": "causal_analytics",
                "activity_timestamp": "2025-01-15T10:00:00Z",
                "workstream": "WS2",
                "analysis_results": {
                    "effect_size": 0.25,
                    "confidence": 0.85,
                    "interpretation": "Strong positive effect",
                },
            },
            {
                "activity_id": str(uuid4()),
                "agent_name": "gap_analyzer",
                "agent_tier": "causal_analytics",
                "activity_timestamp": "2025-01-15T09:00:00Z",
                "workstream": "WS2",
                "analysis_results": {
                    "gap_identified": True,
                    "opportunity_size": 150000,
                },
            },
        ]


@pytest.mark.unit
class TestGetByAgent(TestAgentActivityRepository):
    """Tests for get_by_agent method."""

    @pytest.mark.asyncio
    async def test_returns_activities_for_agent(self, repo, mock_client, sample_activities):
        """Test that activities for a specific agent are returned."""
        mock_result = MagicMock()
        mock_result.data = [sample_activities[0]]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_by_agent(agent_type="causal_impact")

        assert len(result) == 1
        mock_client.table.assert_called_with("agent_activities")

    @pytest.mark.asyncio
    async def test_orders_by_timestamp_descending(self, repo, mock_client, sample_activities):
        """Test that results are ordered by timestamp descending."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_eq = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_by_agent(agent_type="causal_impact")

        assert len(result) == 2
        mock_eq.order.assert_called_with("activity_timestamp", desc=True)

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_activities):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_activities[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_eq = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_by_agent(agent_type="causal_impact", limit=1)

        assert len(result) == 1
        mock_order.limit.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = AgentActivityRepository(supabase_client=None)
        result = await repo.get_by_agent(agent_type="any")
        assert result == []


@pytest.mark.unit
class TestGetByTier(TestAgentActivityRepository):
    """Tests for get_by_tier method."""

    @pytest.mark.asyncio
    async def test_returns_activities_for_tier(self, repo, mock_client, sample_activities):
        """Test that activities for all agents in a tier are returned."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_by_tier(tier="causal_analytics")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_orders_by_timestamp_descending(self, repo, mock_client, sample_activities):
        """Test that results are ordered by timestamp descending."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_eq = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        await repo.get_by_tier(tier="causal_analytics")

        mock_eq.order.assert_called_with("activity_timestamp", desc=True)

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_activities):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_activities[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_eq = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_by_tier(tier="causal_analytics", limit=1)

        assert len(result) == 1
        mock_order.limit.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = AgentActivityRepository(supabase_client=None)
        result = await repo.get_by_tier(tier="any")
        assert result == []


@pytest.mark.unit
class TestGetAnalysisResults(TestAgentActivityRepository):
    """Tests for get_analysis_results method."""

    @pytest.mark.asyncio
    async def test_returns_only_activities_with_results(self, repo, mock_client, sample_activities):
        """Test that only activities with non-null analysis_results are returned."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)

        # Create the mock chain for .not_.is_()
        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_not_is = MagicMock()
        mock_not_is.order.return_value = mock_order
        mock_not = MagicMock()
        mock_not.is_.return_value = mock_not_is
        mock_eq = MagicMock()
        mock_eq.not_ = mock_not
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_analysis_results(agent_type="causal_impact")

        assert len(result) == 2
        mock_not.is_.assert_called_with("analysis_results", "null")

    @pytest.mark.asyncio
    async def test_filters_by_agent_type(self, repo, mock_client, sample_activities):
        """Test that agent_type filter is applied."""
        mock_result = MagicMock()
        mock_result.data = [sample_activities[0]]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_not_is = MagicMock()
        mock_not_is.order.return_value = mock_order
        mock_not = MagicMock()
        mock_not.is_.return_value = mock_not_is
        mock_eq = MagicMock()
        mock_eq.not_ = mock_not
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_analysis_results(agent_type="causal_impact")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_activities):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_activities[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_not_is = MagicMock()
        mock_not_is.order.return_value = mock_order
        mock_not = MagicMock()
        mock_not.is_.return_value = mock_not_is
        mock_eq = MagicMock()
        mock_eq.not_ = mock_not
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_analysis_results(agent_type="causal_impact", limit=1)

        assert len(result) == 1
        mock_order.limit.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = AgentActivityRepository(supabase_client=None)
        result = await repo.get_analysis_results(agent_type="any")
        assert result == []


@pytest.mark.unit
class TestGetRecentActivities(TestAgentActivityRepository):
    """Tests for get_recent_activities method."""

    @pytest.mark.asyncio
    async def test_returns_recent_activities(self, repo, mock_client, sample_activities):
        """Test that recent activities are returned."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_recent_activities(hours=24)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_calculates_cutoff_time_correctly(self, repo, mock_client, sample_activities):
        """Test that cutoff time is calculated correctly."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_gte = MagicMock()
        mock_gte.order.return_value = mock_order
        mock_client.table.return_value.select.return_value.gte.return_value = mock_gte

        # Get current time before the call
        before_time = datetime.now(timezone.utc)
        await repo.get_recent_activities(hours=24)
        after_time = datetime.now(timezone.utc)

        # Verify gte was called with a timestamp roughly 24 hours ago
        call_args = mock_client.table.return_value.select.return_value.gte.call_args
        assert call_args[0][0] == "activity_timestamp"
        # The cutoff should be within a few seconds of 24 hours ago
        cutoff = datetime.fromisoformat(call_args[0][1].replace("Z", "+00:00"))
        expected_cutoff_before = before_time - timedelta(hours=24)
        expected_cutoff_after = after_time - timedelta(hours=24)
        assert expected_cutoff_before <= cutoff <= expected_cutoff_after

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_activities):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_activities[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_gte = MagicMock()
        mock_gte.order.return_value = mock_order
        mock_client.table.return_value.select.return_value.gte.return_value = mock_gte

        result = await repo.get_recent_activities(hours=24, limit=1)

        assert len(result) == 1
        mock_order.limit.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = AgentActivityRepository(supabase_client=None)
        result = await repo.get_recent_activities(hours=24)
        assert result == []


@pytest.mark.unit
class TestGetByWorkstream(TestAgentActivityRepository):
    """Tests for get_by_workstream method."""

    @pytest.mark.asyncio
    async def test_filters_by_workstream(self, repo, mock_client, sample_activities):
        """Test that workstream filter is applied."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_by_workstream(workstream="WS2")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_orders_by_timestamp_descending(self, repo, mock_client, sample_activities):
        """Test that results are ordered by timestamp descending."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_eq = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        await repo.get_by_workstream(workstream="WS2")

        mock_eq.order.assert_called_with("activity_timestamp", desc=True)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = AgentActivityRepository(supabase_client=None)
        result = await repo.get_by_workstream(workstream="WS2")
        assert result == []


@pytest.mark.unit
class TestGetAgentActivitySummary(TestAgentActivityRepository):
    """Tests for get_agent_activity_summary method."""

    @pytest.fixture
    def summary_activities(self):
        """Sample activities for summary testing."""
        return [
            {
                "agent_name": "causal_impact",
                "agent_tier": "causal_analytics",
                "analysis_results": {"effect_size": 0.25},
            },
            {
                "agent_name": "causal_impact",
                "agent_tier": "causal_analytics",
                "analysis_results": None,
            },
            {
                "agent_name": "gap_analyzer",
                "agent_tier": "causal_analytics",
                "analysis_results": {"gap_size": 100},
            },
            {
                "agent_name": "orchestrator",
                "agent_tier": "coordination",
                "analysis_results": {"route": "causal"},
            },
        ]

    @pytest.mark.asyncio
    async def test_calculates_total_activities(self, repo, mock_client, summary_activities):
        """Test that total activity count is calculated."""
        mock_result = MagicMock()
        mock_result.data = summary_activities
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.gte.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_agent_activity_summary(hours=24)

        assert result["total_activities"] == 4

    @pytest.mark.asyncio
    async def test_aggregates_by_agent(self, repo, mock_client, summary_activities):
        """Test that activities are aggregated by agent."""
        mock_result = MagicMock()
        mock_result.data = summary_activities
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.gte.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_agent_activity_summary(hours=24)

        assert result["by_agent"]["causal_impact"] == 2
        assert result["by_agent"]["gap_analyzer"] == 1
        assert result["by_agent"]["orchestrator"] == 1

    @pytest.mark.asyncio
    async def test_aggregates_by_tier(self, repo, mock_client, summary_activities):
        """Test that activities are aggregated by tier."""
        mock_result = MagicMock()
        mock_result.data = summary_activities
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.gte.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_agent_activity_summary(hours=24)

        assert result["by_tier"]["causal_analytics"] == 3
        assert result["by_tier"]["coordination"] == 1

    @pytest.mark.asyncio
    async def test_counts_activities_with_results(self, repo, mock_client, summary_activities):
        """Test that activities with results are counted."""
        mock_result = MagicMock()
        mock_result.data = summary_activities
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.gte.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_agent_activity_summary(hours=24)

        # 3 out of 4 have non-null analysis_results
        assert result["with_results"] == 3

    @pytest.mark.asyncio
    async def test_returns_default_without_client(self):
        """Test that default values are returned when client is None."""
        repo = AgentActivityRepository(supabase_client=None)
        result = await repo.get_agent_activity_summary(hours=24)

        assert result["total_activities"] == 0
        assert result["by_agent"] == {}
        assert result["by_tier"] == {}
        assert result["with_results"] == 0

    @pytest.mark.asyncio
    async def test_returns_default_when_no_data(self, repo, mock_client):
        """Test that default values are returned when no data exists."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.gte.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_agent_activity_summary(hours=24)

        assert result["total_activities"] == 0
        assert result["by_agent"] == {}
        assert result["by_tier"] == {}
        assert result["with_results"] == 0


@pytest.mark.unit
class TestGetActivitiesInRange(TestAgentActivityRepository):
    """Tests for get_activities_in_range method."""

    @pytest.mark.asyncio
    async def test_filters_by_time_range(self, repo, mock_client, sample_activities):
        """Test that time range filter is applied."""
        mock_result = MagicMock()
        mock_result.data = sample_activities
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_lte = MagicMock()
        mock_lte.order.return_value = mock_order
        mock_gte = MagicMock()
        mock_gte.lte.return_value = mock_lte
        mock_client.table.return_value.select.return_value.gte.return_value = mock_gte

        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)

        result = await repo.get_activities_in_range(start_time, end_time)

        assert len(result) == 2
        mock_client.table.return_value.select.return_value.gte.assert_called_with(
            "activity_timestamp", start_time.isoformat()
        )
        mock_gte.lte.assert_called_with("activity_timestamp", end_time.isoformat())

    @pytest.mark.asyncio
    async def test_filters_by_agent_type(self, repo, mock_client, sample_activities):
        """Test that agent_type filter is applied when provided."""
        mock_result = MagicMock()
        mock_result.data = [sample_activities[0]]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_eq = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_lte = MagicMock()
        mock_lte.eq.return_value = mock_eq
        mock_gte = MagicMock()
        mock_gte.lte.return_value = mock_lte
        mock_client.table.return_value.select.return_value.gte.return_value = mock_gte

        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)

        result = await repo.get_activities_in_range(
            start_time, end_time, agent_type="causal_impact"
        )

        assert len(result) == 1
        mock_lte.eq.assert_called_with("agent_name", "causal_impact")

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_activities):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_activities[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_lte = MagicMock()
        mock_lte.order.return_value = mock_order
        mock_gte = MagicMock()
        mock_gte.lte.return_value = mock_lte
        mock_client.table.return_value.select.return_value.gte.return_value = mock_gte

        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)

        result = await repo.get_activities_in_range(start_time, end_time, limit=1)

        assert len(result) == 1
        mock_order.limit.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = AgentActivityRepository(supabase_client=None)
        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)
        result = await repo.get_activities_in_range(start_time, end_time)
        assert result == []
