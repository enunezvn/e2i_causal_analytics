"""
Unit tests for AgentActivityRepository.

Tests agent output queries, analysis result retrieval, and activity tracking.

#894 mock refactor: the previous mocks pinned the exact pre-fix builder chain
(``table().select().eq().order().limit().execute``), so adding the provenance
predicate (``.eq("is_synthetic", False)`` — agent_activities is tagged by
migration 063) broke every chain. Rewritten with a self-chaining recorder
(the #893 ``test_causal_path.py`` idiom): every original test name and
assertion intent is preserved, and the filter assertions are strengthened
with explicit ``eq``/``order``/``limit`` checks.
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.repositories.agent_activity import AgentActivityRepository


class _RecordingQuery:
    """supabase-style fluent builder: records calls, chains itself."""

    def __init__(self, data: list | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict]] = []
        self._data = data or []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> "_RecordingQuery":
        self.calls.append((name, args, kwargs))
        return self

    def select(self, *a: Any, **kw: Any) -> "_RecordingQuery":
        return self._record("select", *a, **kw)

    def eq(self, *a: Any) -> "_RecordingQuery":
        return self._record("eq", *a)

    def gte(self, *a: Any) -> "_RecordingQuery":
        return self._record("gte", *a)

    def lte(self, *a: Any) -> "_RecordingQuery":
        return self._record("lte", *a)

    def is_(self, *a: Any) -> "_RecordingQuery":
        return self._record("is_", *a)

    @property
    def not_(self) -> "_RecordingQuery":
        return self._record("not_")

    def order(self, *a: Any, **kw: Any) -> "_RecordingQuery":
        return self._record("order", *a, **kw)

    def limit(self, *a: Any) -> "_RecordingQuery":
        return self._record("limit", *a)

    def execute(self) -> Any:
        result = MagicMock()
        result.data = list(self._data)
        return AsyncMock(return_value=result)()

    # ---- assertion helpers -------------------------------------------------
    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [args for (n, args, _kw) in self.calls if n == name]

    def assert_called(self, name: str, *args: Any, **kwargs: Any) -> None:
        matches = [(a, kw) for (n, a, kw) in self.calls if n == name and a == args and kw == kwargs]
        assert matches, f"{name}{args} not found in calls: {self.calls}"

    def assert_excludes_synthetic(self) -> None:
        assert ("is_synthetic", False) in self.named("eq"), (
            f"missing provenance predicate; eq calls: {self.named('eq')}"
        )


class _RecordingClient:
    def __init__(self, data: list | None = None) -> None:
        self._data = data or []
        self.tables: list[str] = []
        self.query: _RecordingQuery | None = None

    def table(self, name: str) -> _RecordingQuery:
        self.tables.append(name)
        self.query = _RecordingQuery(self._data)
        return self.query

    def assert_table(self, name: str) -> None:
        assert name in self.tables, f"table({name!r}) never called; saw {self.tables}"


@pytest.mark.unit
class TestAgentActivityRepository:
    """Tests for AgentActivityRepository."""

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

    @staticmethod
    def _repo(data: list | None = None) -> tuple[AgentActivityRepository, _RecordingClient]:
        client = _RecordingClient(data)
        return AgentActivityRepository(supabase_client=client), client


@pytest.mark.unit
class TestGetByAgent(TestAgentActivityRepository):
    """Tests for get_by_agent method."""

    @pytest.mark.asyncio
    async def test_returns_activities_for_agent(self, sample_activities):
        """Test that activities for a specific agent are returned."""
        repo, client = self._repo([sample_activities[0]])

        result = await repo.get_by_agent(agent_type="causal_impact")

        assert len(result) == 1
        client.assert_table("agent_activities")
        client.query.assert_called("eq", "agent_name", "causal_impact")
        client.query.assert_excludes_synthetic()

    @pytest.mark.asyncio
    async def test_orders_by_timestamp_descending(self, sample_activities):
        """Test that results are ordered by timestamp descending."""
        repo, client = self._repo(sample_activities)

        result = await repo.get_by_agent(agent_type="causal_impact")

        assert len(result) == 2
        client.query.assert_called("order", "activity_timestamp", desc=True)

    @pytest.mark.asyncio
    async def test_respects_limit(self, sample_activities):
        """Test that limit is respected."""
        repo, client = self._repo(sample_activities[:1])

        result = await repo.get_by_agent(agent_type="causal_impact", limit=1)

        assert len(result) == 1
        client.query.assert_called("limit", 1)

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
    async def test_returns_activities_for_tier(self, sample_activities):
        """Test that activities for all agents in a tier are returned."""
        repo, client = self._repo(sample_activities)

        result = await repo.get_by_tier(tier="causal_analytics")

        assert len(result) == 2
        client.query.assert_called("eq", "agent_tier", "causal_analytics")
        client.query.assert_excludes_synthetic()

    @pytest.mark.asyncio
    async def test_orders_by_timestamp_descending(self, sample_activities):
        """Test that results are ordered by timestamp descending."""
        repo, client = self._repo(sample_activities)

        await repo.get_by_tier(tier="causal_analytics")

        client.query.assert_called("order", "activity_timestamp", desc=True)

    @pytest.mark.asyncio
    async def test_respects_limit(self, sample_activities):
        """Test that limit is respected."""
        repo, client = self._repo(sample_activities[:1])

        result = await repo.get_by_tier(tier="causal_analytics", limit=1)

        assert len(result) == 1
        client.query.assert_called("limit", 1)

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
    async def test_returns_only_activities_with_results(self, sample_activities):
        """Test that only activities with non-null analysis_results are returned."""
        repo, client = self._repo(sample_activities)

        result = await repo.get_analysis_results(agent_type="causal_impact")

        assert len(result) == 2
        client.query.assert_called("is_", "analysis_results", "null")
        assert ("not_", (), {}) in client.query.calls
        client.query.assert_excludes_synthetic()

    @pytest.mark.asyncio
    async def test_filters_by_agent_type(self, sample_activities):
        """Test that agent_type filter is applied."""
        repo, client = self._repo([sample_activities[0]])

        result = await repo.get_analysis_results(agent_type="causal_impact")

        assert len(result) == 1
        client.query.assert_called("eq", "agent_name", "causal_impact")

    @pytest.mark.asyncio
    async def test_respects_limit(self, sample_activities):
        """Test that limit is respected."""
        repo, client = self._repo(sample_activities[:1])

        result = await repo.get_analysis_results(agent_type="causal_impact", limit=1)

        assert len(result) == 1
        client.query.assert_called("limit", 1)

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
    async def test_returns_recent_activities(self, sample_activities):
        """Test that recent activities are returned."""
        repo, client = self._repo(sample_activities)

        result = await repo.get_recent_activities(hours=24)

        assert len(result) == 2
        client.query.assert_excludes_synthetic()

    @pytest.mark.asyncio
    async def test_calculates_cutoff_time_correctly(self, sample_activities):
        """Test that cutoff time is calculated correctly."""
        repo, client = self._repo(sample_activities)

        before_time = datetime.now(timezone.utc)
        await repo.get_recent_activities(hours=24)
        after_time = datetime.now(timezone.utc)

        gte_calls = client.query.named("gte")
        assert gte_calls and gte_calls[0][0] == "activity_timestamp"
        cutoff = datetime.fromisoformat(gte_calls[0][1].replace("Z", "+00:00"))
        expected_cutoff_before = before_time - timedelta(hours=24)
        expected_cutoff_after = after_time - timedelta(hours=24)
        assert expected_cutoff_before <= cutoff <= expected_cutoff_after

    @pytest.mark.asyncio
    async def test_respects_limit(self, sample_activities):
        """Test that limit is respected."""
        repo, client = self._repo(sample_activities[:1])

        result = await repo.get_recent_activities(hours=24, limit=1)

        assert len(result) == 1
        client.query.assert_called("limit", 1)

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
    async def test_filters_by_workstream(self, sample_activities):
        """Test that workstream filter is applied."""
        repo, client = self._repo(sample_activities)

        result = await repo.get_by_workstream(workstream="WS2")

        assert len(result) == 2
        client.query.assert_called("eq", "workstream", "WS2")
        client.query.assert_excludes_synthetic()

    @pytest.mark.asyncio
    async def test_orders_by_timestamp_descending(self, sample_activities):
        """Test that results are ordered by timestamp descending."""
        repo, client = self._repo(sample_activities)

        await repo.get_by_workstream(workstream="WS2")

        client.query.assert_called("order", "activity_timestamp", desc=True)

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
    async def test_calculates_total_activities(self, summary_activities):
        """Test that total activity count is calculated."""
        repo, client = self._repo(summary_activities)

        result = await repo.get_agent_activity_summary(hours=24)

        assert result["total_activities"] == 4
        client.query.assert_excludes_synthetic()

    @pytest.mark.asyncio
    async def test_aggregates_by_agent(self, summary_activities):
        """Test that activities are aggregated by agent."""
        repo, _client = self._repo(summary_activities)

        result = await repo.get_agent_activity_summary(hours=24)

        assert result["by_agent"]["causal_impact"] == 2
        assert result["by_agent"]["gap_analyzer"] == 1
        assert result["by_agent"]["orchestrator"] == 1

    @pytest.mark.asyncio
    async def test_aggregates_by_tier(self, summary_activities):
        """Test that activities are aggregated by tier."""
        repo, _client = self._repo(summary_activities)

        result = await repo.get_agent_activity_summary(hours=24)

        assert result["by_tier"]["causal_analytics"] == 3
        assert result["by_tier"]["coordination"] == 1

    @pytest.mark.asyncio
    async def test_counts_activities_with_results(self, summary_activities):
        """Test that activities with results are counted."""
        repo, _client = self._repo(summary_activities)

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
    async def test_returns_default_when_no_data(self):
        """Test that default values are returned when no data exists."""
        repo, _client = self._repo([])

        result = await repo.get_agent_activity_summary(hours=24)

        assert result["total_activities"] == 0
        assert result["by_agent"] == {}
        assert result["by_tier"] == {}
        assert result["with_results"] == 0


@pytest.mark.unit
class TestGetActivitiesInRange(TestAgentActivityRepository):
    """Tests for get_activities_in_range method."""

    @pytest.mark.asyncio
    async def test_filters_by_time_range(self, sample_activities):
        """Test that time range filter is applied."""
        repo, client = self._repo(sample_activities)

        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)

        result = await repo.get_activities_in_range(start_time, end_time)

        assert len(result) == 2
        client.query.assert_called("gte", "activity_timestamp", start_time.isoformat())
        client.query.assert_called("lte", "activity_timestamp", end_time.isoformat())
        client.query.assert_excludes_synthetic()

    @pytest.mark.asyncio
    async def test_filters_by_agent_type(self, sample_activities):
        """Test that agent_type filter is applied when provided."""
        repo, client = self._repo([sample_activities[0]])

        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)

        result = await repo.get_activities_in_range(
            start_time, end_time, agent_type="causal_impact"
        )

        assert len(result) == 1
        client.query.assert_called("eq", "agent_name", "causal_impact")

    @pytest.mark.asyncio
    async def test_respects_limit(self, sample_activities):
        """Test that limit is respected."""
        repo, client = self._repo(sample_activities[:1])

        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)

        result = await repo.get_activities_in_range(start_time, end_time, limit=1)

        assert len(result) == 1
        client.query.assert_called("limit", 1)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = AgentActivityRepository(supabase_client=None)
        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=timezone.utc)
        result = await repo.get_activities_in_range(start_time, end_time)
        assert result == []
