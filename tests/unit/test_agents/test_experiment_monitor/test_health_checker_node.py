"""Tests for Health Checker Node.

Tests cover:
- Node initialization
- Execute method with real and mock clients
- Experiment fetching from database
- Health status determination logic
- Enrollment rate checking
- Error handling
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.experiment_monitor.nodes.health_checker import HealthCheckerNode
from src.agents.experiment_monitor.state import ExperimentMonitorState


class TestHealthCheckerNodeInit:
    """Tests for HealthCheckerNode initialization."""

    def test_node_initialization(self):
        """Test that node initializes correctly."""
        node = HealthCheckerNode()
        assert node is not None
        assert node._client is None

    def test_multiple_node_instances(self):
        """Test creating multiple node instances."""
        node1 = HealthCheckerNode()
        node2 = HealthCheckerNode()
        assert node1 is not node2
        assert node1._client is None
        assert node2._client is None


class TestHealthCheckerGetClient:
    """Tests for lazy client loading."""

    @pytest.mark.asyncio
    async def test_get_client_lazy_loads(self):
        """Test that client is lazily loaded."""
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            node = HealthCheckerNode()
            assert node._client is None

            client = await node._get_client()

            assert client is mock_client
            assert node._client is mock_client
            mock_get_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_caches_result(self):
        """Test that client is cached after first load."""
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            node = HealthCheckerNode()

            # First call
            client1 = await node._get_client()
            # Second call
            client2 = await node._get_client()

            assert client1 is client2
            # Should only be called once due to caching
            mock_get_client.assert_called_once()


class TestHealthCheckerExecute:
    """Tests for execute method."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Create a mock Supabase client with query builder pattern."""
        mock = MagicMock()

        # Setup experiments table mock
        exp_result = MagicMock()
        exp_result.data = [
            {
                "id": "exp-001",
                "name": "Test Experiment",
                "status": "running",
                "config": {"target_sample_size": 1000},
                "created_at": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
            }
        ]

        exp_query = MagicMock()
        exp_query.select = MagicMock(return_value=exp_query)
        exp_query.eq = MagicMock(return_value=exp_query)
        exp_query.in_ = MagicMock(return_value=exp_query)
        # Genuine-A/B predicate (migration 102): not_.is_ must return the same
        # builder or the rest of the chain derails onto a fresh MagicMock.
        exp_query.not_.is_ = MagicMock(return_value=exp_query)
        exp_query.order = MagicMock(return_value=exp_query)
        exp_query.limit = MagicMock(return_value=exp_query)
        exp_query.execute = AsyncMock(return_value=exp_result)

        # Setup assignments table mock
        assign_result = MagicMock()
        assign_result.count = 350

        assign_query = MagicMock()
        assign_query.select = MagicMock(return_value=assign_query)
        assign_query.eq = MagicMock(return_value=assign_query)
        assign_query.execute = AsyncMock(return_value=assign_result)

        def table_mock(name):
            if name == "ml_experiments":
                return exp_query
            elif name == "ab_experiment_assignments":
                return assign_query
            return MagicMock()

        mock.table = table_mock
        return mock

    @pytest.mark.asyncio
    async def test_execute_sets_status_to_checking(self, base_monitor_state):
        """Test that execute sets status to 'checking'."""
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None  # Trigger mock data path

            node = HealthCheckerNode()
            result = await node.execute(base_monitor_state)

            # Status should transition during execution
            assert result["status"] == "checking"

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_no_client(self, base_monitor_state):
        """No-mocking: when no DB client is available the node fails CLOSED --
        empty experiments + a recorded error -- never fabricated mock data."""
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            node = HealthCheckerNode()
            result = await node.execute(base_monitor_state)

            assert result["experiments"] == []
            errors = result.get("errors", [])
            assert any("client unavailable" in str(e.get("error", "")).lower() for e in errors), (
                errors
            )

    @pytest.mark.asyncio
    async def test_execute_with_real_client(self, base_monitor_state, mock_supabase_client):
        """Test execute with real database client."""
        node = HealthCheckerNode()
        node._client = mock_supabase_client

        result = await node.execute(base_monitor_state)

        assert result["experiments_checked"] == 1
        assert len(result["experiments"]) == 1
        assert result["experiments"][0]["experiment_id"] == "exp-001"

    @pytest.mark.asyncio
    async def test_execute_calculates_latency(self, base_monitor_state):
        """Test that latency is calculated."""
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            node = HealthCheckerNode()
            result = await node.execute(base_monitor_state)

            assert "check_latency_ms" in result
            assert result["check_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_handles_exceptions(self, base_monitor_state):
        """Test that exceptions are caught and recorded."""
        node = HealthCheckerNode()

        # Make _get_client raise an exception
        with patch.object(node, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Database connection failed")

            result = await node.execute(base_monitor_state)

            assert len(result["errors"]) >= 1
            assert "Database connection failed" in result["errors"][0]["error"]
            assert result["errors"][0]["node"] == "health_checker"

    @pytest.mark.asyncio
    async def test_execute_populates_experiments_list(self, base_monitor_state):
        """Test that experiments list is populated."""
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            node = HealthCheckerNode()
            result = await node.execute(base_monitor_state)

            for exp in result["experiments"]:
                assert "experiment_id" in exp
                assert "name" in exp
                assert "status" in exp
                assert "health_status" in exp
                assert "days_running" in exp
                assert "total_enrolled" in exp
                assert "enrollment_rate" in exp
                assert "current_information_fraction" in exp


class TestDedupeByName:
    """Tests for _dedupe_by_name (collapses duplicate-named running experiments).

    The synthetic generator leaves many perpetually-"running" rows sharing an
    experiment_name (e.g. "Kisqali - Predict prescribing" x252). Without dedup
    the newest-N slice surfaced many identical cards in the UI.
    """

    def test_keeps_most_recent_per_name(self):
        # Rows arrive created_at-desc, so the FIRST occurrence per name is newest.
        rows = [
            {"id": "1", "experiment_name": "A"},
            {"id": "2", "experiment_name": "B"},
            {"id": "3", "experiment_name": "A"},  # older dup of A — dropped
            {"id": "4", "experiment_name": "A"},  # older dup of A — dropped
            {"id": "5", "experiment_name": "C"},
        ]
        out = HealthCheckerNode._dedupe_by_name(rows, cap=10)
        assert [r["experiment_name"] for r in out] == ["A", "B", "C"]
        assert [r["id"] for r in out] == ["1", "2", "5"]

    def test_respects_cap(self):
        rows = [{"id": str(i), "experiment_name": f"E{i}"} for i in range(50)]
        out = HealthCheckerNode._dedupe_by_name(rows, cap=25)
        assert len(out) == 25

    def test_unnamed_rows_fall_back_to_id_and_are_not_collapsed(self):
        # Two distinct unnamed rows must not collapse to a single "None" key.
        rows = [{"id": "1"}, {"id": "2"}]
        out = HealthCheckerNode._dedupe_by_name(rows, cap=10)
        assert len(out) == 2

    def test_empty_input(self):
        assert HealthCheckerNode._dedupe_by_name([], cap=25) == []


class TestGetExperiments:
    """Tests for _get_experiments method."""

    @pytest.fixture
    def node(self):
        return HealthCheckerNode()

    @pytest.mark.asyncio
    async def test_get_experiments_check_all_active(self, node):
        """Test getting all active experiments."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "exp-1", "name": "Exp 1", "status": "running"},
            {"id": "exp-2", "name": "Exp 2", "status": "running"},
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        # Bounded interactive sweep (700+ running experiments would blow the 30s
        # client timeout): most-recent N via order+limit.
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        # Unscoped sweep restricts to the platform brand portfolio (2026-07-11).
        mock_query.in_ = MagicMock(return_value=mock_query)
        # Genuine-A/B predicate (migration 102): not_.is_ must return the same
        # builder or the rest of the chain derails onto a fresh MagicMock.
        mock_query.not_.is_ = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        state: ExperimentMonitorState = {
            "check_all_active": True,
            "experiment_ids": [],
            "query": "",
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        result = await node._get_experiments(mock_client, state)

        assert len(result) == 2
        mock_query.eq.assert_any_call("status", "running")
        # Migration 102: lineage rows (no intervention_channel) are not
        # experiments — the sweep keeps them out even if a future writer
        # inherits the DB-default 'running' status.
        mock_query.not_.is_.assert_called_once_with("intervention_channel", "null")
        # #894: the enumeration default-excludes synthetic experiments
        mock_query.eq.assert_any_call("is_synthetic", False)
        # 2026-07-11: unscoped = the 3-brand portfolio (scope_definer scaffolding
        # rows with brand NULL/'competitor' are not A/B experiments).
        mock_query.in_.assert_called_once_with("brand", ["Remibrutinib", "Kisqali", "Fabhalta"])
        # The sweep fetches a wide newest-first window (then collapses duplicate
        # names via _dedupe_by_name); only the deduped subset incurs checks.
        mock_query.order.assert_called_once_with("created_at", desc=True)
        mock_query.limit.assert_called_once_with(1000)

    @pytest.mark.asyncio
    async def test_get_experiments_brand_scoped(self, node):
        """A brand in state scopes the sweep with eq('brand', ...) and skips the
        portfolio in_ filter (2026-07-11 brand dropdown)."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "exp-1", "experiment_name": "K1", "status": "running", "brand": "Kisqali"},
        ]
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.in_ = MagicMock(return_value=mock_query)
        mock_query.not_.is_ = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value = mock_query

        state: ExperimentMonitorState = {
            "check_all_active": True,
            "experiment_ids": [],
            "brand": "Kisqali",
            "status": "pending",
        }

        result = await node._get_experiments(mock_client, state)

        assert [r["id"] for r in result] == ["exp-1"]
        mock_query.eq.assert_any_call("brand", "Kisqali")
        mock_query.in_.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_experiments_dedups_duplicate_names(self, node, monkeypatch):
        """check_all_active collapses duplicate-named rows + uses the wide fetch.

        Forces real-mode (deployment_includes_synthetic -> False) so the path is
        deterministic regardless of the ambient E2I_INCLUDE_SYNTHETIC flag, and
        verifies the wide fetch limit (1000) plus the name-dedup actually run
        inside _get_experiments (not just _dedupe_by_name in isolation).
        """
        import src.repositories.provenance as prov

        monkeypatch.setattr(prov, "deployment_includes_synthetic", lambda: False)

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "1", "experiment_name": "Dup", "status": "running"},
            {"id": "2", "experiment_name": "Dup", "status": "running"},  # collapsed
            {"id": "3", "experiment_name": "Unique", "status": "running"},
        ]
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.in_ = MagicMock(return_value=mock_query)
        mock_query.not_.is_ = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value = mock_query

        state: ExperimentMonitorState = {
            "check_all_active": True,
            "experiment_ids": [],
            "status": "pending",
        }

        result = await node._get_experiments(mock_client, state)

        # 3 rows in, 2 distinct names out (Dup collapsed to its first/newest row).
        assert [r["id"] for r in result] == ["1", "3"]
        # Wide newest-first fetch, then dedup (not a narrow .limit(25)).
        mock_query.limit.assert_called_once_with(1000)
        mock_query.eq.assert_any_call("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_experiments_reports_total_running(self, node):
        """The sweep records the exact scope-matching count in state so the UI
        can say "N running, 25 monitored" instead of presenting the roster cap
        as the portfolio size (2026-07-11 review: '25 seems hardcoded')."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "exp-1", "experiment_name": "E1", "status": "running"},
        ]
        mock_result.count = 955  # exact count from PostgREST Content-Range
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.in_ = MagicMock(return_value=mock_query)
        mock_query.not_.is_ = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value = mock_query

        state: ExperimentMonitorState = {
            "check_all_active": True,
            "experiment_ids": [],
            "status": "pending",
        }

        await node._get_experiments(mock_client, state)
        assert state["total_running"] == 955

        # A fake result without an int count falls back to the fetched window
        # size instead of fabricating (or crashing on) a Mock attribute.
        mock_result.count = None
        await node._get_experiments(mock_client, state)
        assert state["total_running"] == 1

    @pytest.mark.asyncio
    async def test_get_experiments_excludes_channel_less_lineage_rows(
        self, node, mock_supabase_client, monkeypatch
    ):
        """Rows without an intervention_channel are pipeline-lineage records
        (scope definitions, evals, deploys), not A/B experiments — the sweep
        and its exact total_running count must exclude them even when they
        carry status='running' (migration 102: 692 scope_definer rows sat on
        the DB-default status and inflated the portfolio 955 vs 360)."""
        import src.repositories.provenance as prov

        monkeypatch.setattr(prov, "deployment_includes_synthetic", lambda: False)
        now = datetime.now(timezone.utc)
        mock_supabase_client.set_mock_data(
            "ml_experiments",
            [
                {
                    "id": "ab-1",
                    "experiment_name": "Kisqali: Email Campaign",
                    "status": "running",
                    "brand": "Kisqali",
                    "is_synthetic": False,
                    "intervention_channel": "email_campaign",
                    "created_at": (now - timedelta(days=3)).isoformat(),
                },
                {
                    "id": "scope-1",
                    "experiment_name": "Kisqali - Predict prescribing",
                    "status": "running",  # inherited DB default, not an experiment
                    "brand": "Kisqali",
                    "is_synthetic": False,
                    "created_at": (now - timedelta(days=1)).isoformat(),
                },
            ],
        )
        state: ExperimentMonitorState = {
            "check_all_active": True,
            "experiment_ids": [],
            "status": "pending",
        }

        result = await node._get_experiments(mock_supabase_client, state)

        assert [r["id"] for r in result] == ["ab-1"]
        assert state["total_running"] == 1

    @pytest.mark.asyncio
    async def test_get_experiments_specific_ids(self, node):
        """Test getting specific experiments by ID."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"id": "exp-1", "name": "Exp 1", "status": "running"}]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.in_ = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)  # #894 provenance predicate
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        state: ExperimentMonitorState = {
            "check_all_active": False,
            "experiment_ids": ["exp-1"],
            "query": "",
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        result = await node._get_experiments(mock_client, state)

        assert len(result) == 1
        mock_query.in_.assert_called_with("id", ["exp-1"])
        # #894: a synthetic id must not resolve in real mode either
        mock_query.eq.assert_any_call("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_experiments_no_filters_returns_empty(self, node):
        """Test that no filters returns empty list."""
        mock_client = MagicMock()

        state: ExperimentMonitorState = {
            "check_all_active": False,
            "experiment_ids": [],
            "query": "",
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        result = await node._get_experiments(mock_client, state)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_experiments_handles_exception(self, node):
        """Test that exceptions return empty list."""
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(side_effect=Exception("DB Error"))
        mock_client.table = MagicMock(return_value=mock_query)

        state: ExperimentMonitorState = {
            "check_all_active": True,
            "experiment_ids": [],
            "query": "",
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        result = await node._get_experiments(mock_client, state)

        assert result == []


class TestCheckExperimentHealth:
    """Tests for _check_experiment_health method."""

    @pytest.fixture
    def node(self):
        return HealthCheckerNode()

    @pytest.mark.asyncio
    async def test_check_health_calculates_days_running(self, node):
        """Test that days running is calculated correctly."""
        experiment = {
            "id": "exp-001",
            "name": "Test",
            "status": "running",
            "config": {"target_sample_size": 1000},
            "created_at": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, None)

        assert summary["days_running"] == 7

    @pytest.mark.asyncio
    async def test_check_health_minimum_one_day(self, node):
        """Test that minimum days running is 1."""
        experiment = {
            "id": "exp-001",
            "name": "Test",
            "status": "running",
            "config": {"target_sample_size": 1000},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, None)

        assert summary["days_running"] >= 1

    @pytest.mark.asyncio
    async def test_check_health_with_client_gets_enrollment(self, node):
        """Test that enrollment is fetched from client."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.count = 500

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        experiment = {
            "id": "exp-001",
            "name": "Test",
            "status": "running",
            "config": {"target_sample_size": 1000},
            "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, mock_client)

        assert summary["total_enrolled"] == 500

    @pytest.mark.asyncio
    async def test_check_health_without_client_zero_enrollment(self, node):
        """Test that enrollment is 0 without client."""
        experiment = {
            "id": "exp-001",
            "name": "Test",
            "status": "running",
            "config": {"target_sample_size": 1000},
            "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, None)

        assert summary["total_enrolled"] == 0

    @pytest.mark.asyncio
    async def test_check_health_calculates_information_fraction(self, node):
        """Information fraction comes from the row's REAL recorded plan."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.count = 500

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        experiment = {
            "id": "exp-001",
            "name": "Test",
            "status": "running",
            "target_enrollment": 1000,
            "planned_duration_days": 60,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, mock_client)

        assert summary["current_information_fraction"] == 0.5  # 500/1000
        assert summary["target_enrollment"] == 1000
        assert summary["health_reason"]

    @pytest.mark.asyncio
    async def test_check_health_fraction_capped_at_one(self, node):
        """Enrollment past the target reports fraction 1.0, not >1."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.count = 1500

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        experiment = {
            "id": "exp-001",
            "name": "Test",
            "status": "running",
            "target_enrollment": 1000,
            "planned_duration_days": 60,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, mock_client)

        assert summary["current_information_fraction"] == 1.0

    @pytest.mark.asyncio
    async def test_check_health_no_plan_reports_unknown_fraction(self, node):
        """Regression (2026-07-11): a row WITHOUT a recorded plan reports
        fraction None — never a fraction of a fabricated default target."""
        experiment = {
            "id": "exp-001",
            "name": "Test",
            "status": "running",
            "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, None)

        assert summary["current_information_fraction"] is None
        assert summary["target_enrollment"] is None
        assert summary["health_reason"]


class TestDetermineHealthStatus:
    """Tests for _determine_health_status method.

    The method returns (status, reason). Plan-relative checks require a REAL
    recorded plan (migration 101) — a missing plan must never be substituted
    with a default target (the fabricated 1000-in-30-days default previously
    flagged the entire portfolio "warning"; live incident 2026-07-11).
    """

    @pytest.fixture
    def node(self):
        return HealthCheckerNode()

    def test_critical_low_enrollment_after_14_days(self, node):
        """Critical status for stalled enrollment after 14 days, plan or not."""
        status, reason = node._determine_health_status(
            enrollment_rate=1.5,  # < 2
            days_running=14,
            total_enrolled=21,
            target_enrollment=None,
            planned_duration_days=None,
        )
        assert status == "critical"
        assert "stalled" in reason.lower()

    def test_warning_low_enrollment_after_7_days(self, node):
        """Warning status for slow enrollment after 7 days."""
        status, reason = node._determine_health_status(
            enrollment_rate=3.0,  # < 5
            days_running=7,
            total_enrolled=21,
            target_enrollment=None,
            planned_duration_days=None,
        )
        assert status == "warning"
        assert "slow enrollment" in reason.lower()

    def test_no_plan_is_not_behind_schedule(self, node):
        """Regression (2026-07-11 all-warning incident): a healthy-pace
        experiment WITHOUT a recorded plan must be healthy — the old code
        fabricated target_sample_size=1000 and flagged it behind schedule."""
        status, reason = node._determine_health_status(
            enrollment_rate=10.0,
            days_running=10,
            total_enrolled=100,
            target_enrollment=None,
            planned_duration_days=None,
        )
        assert status == "healthy"
        assert "no enrollment plan recorded" in reason

    def test_target_reached_is_healthy(self, node):
        """Reaching the recorded target is healthy regardless of age."""
        status, reason = node._determine_health_status(
            enrollment_rate=30.0,
            days_running=20,
            total_enrolled=650,
            target_enrollment=600,
            planned_duration_days=60,
        )
        assert status == "healthy"
        assert "target reached" in reason.lower()

    def test_past_planned_duration_under_target_warns(self, node):
        """Running past the planned window while under target is a warning."""
        status, reason = node._determine_health_status(
            enrollment_rate=7.1,
            days_running=70,
            total_enrolled=500,
            target_enrollment=600,
            planned_duration_days=60,
        )
        assert status == "warning"
        assert "past planned duration" in reason.lower()

    def test_far_behind_plan_warns(self, node):
        """Under half the expected fraction of a real plan is a warning."""
        status, reason = node._determine_health_status(
            enrollment_rate=5.3,  # above the absolute floor
            days_running=30,
            total_enrolled=160,  # 8% of target vs 50% expected
            target_enrollment=2000,
            planned_duration_days=60,
        )
        assert status == "warning"
        assert "behind plan" in reason.lower()

    def test_on_pace_with_plan_is_healthy(self, node):
        """On-pace enrollment against a real plan is healthy."""
        status, reason = node._determine_health_status(
            enrollment_rate=10.0,
            days_running=30,
            total_enrolled=300,
            target_enrollment=600,
            planned_duration_days=60,
        )
        assert status == "healthy"
        assert "on pace" in reason.lower()

    def test_healthy_early_experiment(self, node):
        """A young experiment below the pace floors is still healthy."""
        status, _reason = node._determine_health_status(
            enrollment_rate=5.0,
            days_running=3,
            total_enrolled=15,
            target_enrollment=None,
            planned_duration_days=None,
        )
        assert status == "healthy"

    def test_critical_takes_precedence_over_plan_checks(self, node):
        """Stalled enrollment is critical even when a plan is present."""
        status, _reason = node._determine_health_status(
            enrollment_rate=1.0,
            days_running=14,
            total_enrolled=14,
            target_enrollment=600,
            planned_duration_days=60,
        )
        assert status == "critical"


class TestCheckEnrollmentRate:
    """Tests for _check_enrollment_rate method."""

    @pytest.fixture
    def node(self):
        return HealthCheckerNode()

    def test_no_issue_when_above_threshold(self, node, sample_summary_healthy, base_monitor_state):
        """Test no issue is returned when rate is above threshold."""
        # Healthy summary has enrollment_rate of 71.43
        result = node._check_enrollment_rate(
            experiment={},
            summary=sample_summary_healthy,
            state=base_monitor_state,
        )
        assert result is None

    def test_issue_when_below_threshold(self, node, base_monitor_state):
        """Test issue is returned when rate is below threshold."""
        low_enrollment_summary = {
            "experiment_id": "exp-low",
            "name": "Low Enrollment",
            "status": "running",
            "health_status": "warning",
            "days_running": 10,
            "total_enrolled": 30,
            "enrollment_rate": 3.0,  # Below 5.0 threshold
            "current_information_fraction": 0.03,
        }

        result = node._check_enrollment_rate(
            experiment={},
            summary=low_enrollment_summary,
            state=base_monitor_state,
        )

        assert result is not None
        assert result["experiment_id"] == "exp-low"
        assert result["current_rate"] == 3.0
        assert result["expected_rate"] == 5.0

    def test_severity_info_for_new_experiment(self, node, base_monitor_state):
        """Test info severity for experiments less than 7 days old."""
        summary = {
            "experiment_id": "exp-new",
            "name": "New Experiment",
            "status": "running",
            "health_status": "healthy",
            "days_running": 3,
            "total_enrolled": 10,
            "enrollment_rate": 3.0,
            "current_information_fraction": 0.01,
        }

        result = node._check_enrollment_rate(
            experiment={},
            summary=summary,
            state=base_monitor_state,
        )

        assert result is not None
        assert result["severity"] == "info"

    def test_severity_warning_for_7_day_experiment(self, node, base_monitor_state):
        """Test warning severity for experiments 7+ days old."""
        summary = {
            "experiment_id": "exp-week",
            "name": "Week Old",
            "status": "running",
            "health_status": "warning",
            "days_running": 7,
            "total_enrolled": 20,
            "enrollment_rate": 3.0,
            "current_information_fraction": 0.02,
        }

        result = node._check_enrollment_rate(
            experiment={},
            summary=summary,
            state=base_monitor_state,
        )

        assert result is not None
        assert result["severity"] == "warning"

    def test_severity_critical_for_14_day_experiment(self, node, base_monitor_state):
        """Test critical severity for experiments 14+ days old."""
        summary = {
            "experiment_id": "exp-old",
            "name": "Old Experiment",
            "status": "running",
            "health_status": "critical",
            "days_running": 14,
            "total_enrolled": 30,
            "enrollment_rate": 2.0,
            "current_information_fraction": 0.03,
        }

        result = node._check_enrollment_rate(
            experiment={},
            summary=summary,
            state=base_monitor_state,
        )

        assert result is not None
        assert result["severity"] == "critical"

    def test_custom_threshold_from_state(self, node):
        """Test that custom threshold from state is used."""
        state: ExperimentMonitorState = {
            "query": "",
            "check_all_active": True,
            "experiment_ids": [],
            "srm_threshold": 0.001,
            "enrollment_threshold": 10.0,  # Higher threshold
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        summary = {
            "experiment_id": "exp-test",
            "name": "Test",
            "status": "running",
            "health_status": "healthy",
            "days_running": 7,
            "total_enrolled": 50,
            "enrollment_rate": 7.0,  # Above 5.0 but below 10.0
            "current_information_fraction": 0.05,
        }

        result = node._check_enrollment_rate(
            experiment={},
            summary=summary,
            state=state,
        )

        assert result is not None
        assert result["expected_rate"] == 10.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def node(self):
        return HealthCheckerNode()

    @pytest.mark.asyncio
    async def test_empty_experiment_list(self, node, base_monitor_state):
        """Test handling of empty experiment list."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        node._client = mock_client
        result = await node.execute(base_monitor_state)

        assert result["experiments"] == []
        assert result["experiments_checked"] == 0

    @pytest.mark.asyncio
    async def test_experiment_with_zero_target_enrollment(self, node):
        """A zero recorded target is treated as no plan (fraction unknowable),
        never a division by zero or a fabricated fraction."""
        experiment = {
            "id": "exp-zero",
            "name": "Zero Target",
            "status": "running",
            "target_enrollment": 0,
            "planned_duration_days": 60,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, None)

        assert summary["current_information_fraction"] is None

    @pytest.mark.asyncio
    async def test_experiment_with_missing_plan(self, node):
        """A row without plan columns (real pre-101 experiments) still gets a
        summary — with fraction None, never a default-target fabrication."""
        experiment = {
            "id": "exp-no-plan",
            "name": "No Plan",
            "status": "running",
            "created_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, None)

        assert summary is not None
        assert summary["experiment_id"] == "exp-no-plan"
        assert summary["current_information_fraction"] is None

    @pytest.mark.asyncio
    async def test_experiment_with_iso_date_z_suffix(self, node):
        """Test experiment with ISO date containing Z suffix."""
        experiment = {
            "id": "exp-z",
            "name": "Z Suffix Date",
            "status": "running",
            "config": {"target_sample_size": 1000},
            "created_at": (datetime.now(timezone.utc) - timedelta(days=3))
            .isoformat()
            .replace("+00:00", "Z"),
        }

        summary = await node._check_experiment_health(experiment, None)

        assert summary["days_running"] == 3

    @pytest.mark.asyncio
    async def test_enrollment_query_exception_handled(self, node):
        """Test that enrollment query exceptions are handled gracefully."""
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(side_effect=Exception("Query failed"))
        mock_client.table = MagicMock(return_value=mock_query)

        experiment = {
            "id": "exp-error",
            "name": "Error Test",
            "status": "running",
            "config": {"target_sample_size": 1000},
            "created_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        }

        # Should not raise, just return 0 enrollment
        summary = await node._check_experiment_health(experiment, mock_client)

        assert summary["total_enrolled"] == 0

    def test_health_status_at_exact_boundaries(self, node):
        """Test health status at exact boundary conditions."""
        # Exactly at 14 days with rate exactly 2 (boundary)
        status, _ = node._determine_health_status(
            enrollment_rate=2.0,
            days_running=14,
            total_enrolled=28,
            target_enrollment=None,
            planned_duration_days=None,
        )
        # Rate < 2 is critical, rate == 2 is not critical
        assert status in ["healthy", "warning"]

        # Exactly at 7 days with rate exactly 5 (boundary)
        status, _ = node._determine_health_status(
            enrollment_rate=5.0,
            days_running=7,
            total_enrolled=35,
            target_enrollment=None,
            planned_duration_days=None,
        )
        # Rate < 5 is warning, rate == 5 is not warning
        assert status == "healthy"


class TestInterleaveByBrand:
    """_interleave_by_brand: brand-balanced roster (2026-07-11 review).

    Newest-first alone let one same-instant generation batch monopolize the
    capped slice (live incident: the top-25 was 25 Fabhalta rows from one
    burst); the interleave takes one row per brand per round.
    """

    def test_round_robins_across_brands(self):
        rows = (
            [{"id": f"f{i}", "brand": "Fabhalta"} for i in range(5)]
            + [{"id": f"k{i}", "brand": "Kisqali"} for i in range(5)]
            + [{"id": f"r{i}", "brand": "Remibrutinib"} for i in range(5)]
        )
        out = HealthCheckerNode._interleave_by_brand(rows, cap=6)
        assert [r["id"] for r in out] == ["f0", "k0", "r0", "f1", "k1", "r1"]

    def test_monoculture_burst_no_longer_monopolizes(self):
        # 25 newest same-brand rows followed by older other-brand rows: the cap
        # must still show every brand.
        rows = [{"id": f"f{i}", "brand": "Fabhalta"} for i in range(25)] + [
            {"id": "k0", "brand": "Kisqali"},
            {"id": "r0", "brand": "Remibrutinib"},
        ]
        out = HealthCheckerNode._interleave_by_brand(rows, cap=25)
        brands = {r["brand"] for r in out}
        assert brands == {"Fabhalta", "Kisqali", "Remibrutinib"}

    def test_preserves_newest_first_within_brand(self):
        rows = [
            {"id": "f0", "brand": "Fabhalta"},
            {"id": "k0", "brand": "Kisqali"},
            {"id": "f1", "brand": "Fabhalta"},
        ]
        out = HealthCheckerNode._interleave_by_brand(rows, cap=3)
        f_order = [r["id"] for r in out if r["brand"] == "Fabhalta"]
        assert f_order == ["f0", "f1"]

    def test_missing_brand_forms_own_group_and_empty_input(self):
        rows = [{"id": "a"}, {"id": "b", "brand": "Kisqali"}]
        out = HealthCheckerNode._interleave_by_brand(rows, cap=10)
        assert {r["id"] for r in out} == {"a", "b"}
        assert HealthCheckerNode._interleave_by_brand([], cap=25) == []

    def test_exhausted_groups_backfill_from_remaining(self):
        rows = [{"id": "k0", "brand": "Kisqali"}] + [
            {"id": f"f{i}", "brand": "Fabhalta"} for i in range(4)
        ]
        out = HealthCheckerNode._interleave_by_brand(rows, cap=5)
        assert len(out) == 5  # cap reached even though Kisqali ran dry


def _ab_row(id_, ci_lower=None, ci_upper=None, target=None, brand="Kisqali"):
    row = {"id": id_, "experiment_name": id_, "brand": brand}
    if ci_lower is not None or ci_upper is not None:
        row["ab_experiment_results"] = [
            {
                "effect_estimate": ((ci_lower or 0) + (ci_upper or 0)) / 2,
                "effect_ci_lower": ci_lower,
                "effect_ci_upper": ci_upper,
            }
        ]
    if target is not None:
        row["target_enrollment"] = target
    return row


class TestExpectedImpact:
    """_expected_impact: CI-shrunk OBSERVED effect x planned reach.

    The score must be earned with data — the CI bound nearer zero, never the
    point estimate or a ground-truth prior — so a noisy early read cannot jump
    the monitoring queue.
    """

    def test_positive_effect_uses_ci_lower(self):
        row = _ab_row("e", ci_lower=0.10, ci_upper=0.30, target=1000)
        assert HealthCheckerNode._expected_impact(row) == 100.0

    def test_confidently_harmful_uses_abs_ci_upper(self):
        # An experiment confidently making things worse deserves monitoring
        # priority too.
        row = _ab_row("e", ci_lower=-0.30, ci_upper=-0.10, target=500)
        assert HealthCheckerNode._expected_impact(row) == 50.0

    def test_ci_spanning_zero_scores_zero(self):
        row = _ab_row("e", ci_lower=-0.05, ci_upper=0.15, target=1000)
        assert HealthCheckerNode._expected_impact(row) == 0.0

    def test_missing_results_is_unscorable_not_zero(self):
        assert HealthCheckerNode._expected_impact(_ab_row("e", target=1000)) is None

    def test_missing_plan_is_unscorable_not_zero(self):
        row = _ab_row("e", ci_lower=0.10, ci_upper=0.30)
        assert HealthCheckerNode._expected_impact(row) is None

    def test_embedded_result_dict_accepted(self):
        # PostgREST returns a to-one embed as a dict on some client versions.
        row = {
            "id": "e",
            "target_enrollment": 100,
            "ab_experiment_results": {"effect_ci_lower": 0.2, "effect_ci_upper": 0.4},
        }
        assert HealthCheckerNode._expected_impact(row) == 20.0


class TestRankByImpact:
    """_rank_by_impact: top slots by expected impact + newest-first reserve,
    with evidence-based tiers stamped on every row."""

    def test_top_slots_by_impact_reserve_by_recency(self):
        # Input is created_at-desc: newest first. n1/n2 are new & unscorable;
        # winners w1..w3 are older but carry demonstrated impact.
        rows = [
            _ab_row("n1"),
            _ab_row("n2"),
            _ab_row("z1", ci_lower=-0.1, ci_upper=0.1, target=1000),  # score 0
            _ab_row("w3", ci_lower=0.05, ci_upper=0.2, target=1000),  # 50
            _ab_row("w1", ci_lower=0.20, ci_upper=0.4, target=1000),  # 200
            _ab_row("w2", ci_lower=0.10, ci_upper=0.3, target=1000),  # 100
        ]
        # cap 4 with _NEWEST_SLOTS=5 would leave 0 impact slots; exercise the
        # mix explicitly via a cap where both segments are non-empty.
        out = HealthCheckerNode._rank_by_impact(rows, cap=7)
        ids = [r["id"] for r in out]
        # 2 impact slots (cap 7 - 5 reserve): the two highest scores, in order.
        assert ids[:2] == ["w1", "w2"]
        # Reserve: newest-first among the rest.
        assert set(ids[2:]) == {"n1", "n2", "z1", "w3"}

    def test_tiers_are_evidence_based(self):
        rows = [
            _ab_row("hi", ci_lower=0.20, ci_upper=0.4, target=1000),  # 200
            _ab_row("mid", ci_lower=0.05, ci_upper=0.2, target=1000),  # 50
            _ab_row("null_effect", ci_lower=-0.1, ci_upper=0.1, target=1000),  # 0
            _ab_row("unscored"),
        ]
        HealthCheckerNode._rank_by_impact(rows, cap=25)
        tiers = {r["id"]: r["impact_tier"] for r in rows}
        assert tiers == {
            "hi": "high",
            "mid": "medium",
            "null_effect": "low",
            "unscored": None,
        }

    def test_all_unscorable_falls_back_to_newest_interleaved(self):
        rows = [_ab_row(f"e{i}") for i in range(6)]
        out = HealthCheckerNode._rank_by_impact(rows, cap=4)
        assert [r["id"] for r in out] == ["e0", "e1", "e2", "e3"]
        assert all(r["expected_impact"] is None for r in rows)

    def test_cap_respected(self):
        rows = [_ab_row(f"e{i}", ci_lower=0.1, ci_upper=0.2, target=100 + i) for i in range(30)]
        out = HealthCheckerNode._rank_by_impact(rows, cap=25)
        assert len(out) == 25


class TestStaleSeverityRelativeToThreshold:
    """_check_stale_data severity scales with the caller's threshold (2026-07-11).

    The old absolute tiers (48h/72h) pinned every experiment critical once the
    substrate refresh cadence was slower than 3 days (weekly synthetic refresh),
    regardless of the caller-chosen threshold.
    """

    @pytest.fixture
    def node(self):
        return HealthCheckerNode()

    def _client_with_last_assignment(self, hours_ago: float):
        result = MagicMock()
        result.data = [
            {"assigned_at": (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()}
        ]
        query = MagicMock()
        query.select.return_value = query
        query.eq.return_value = query
        query.order.return_value = query
        query.limit.return_value = query
        query.execute = AsyncMock(return_value=result)
        client = MagicMock()
        client.table.return_value = query
        return client

    @pytest.mark.asyncio
    async def test_breach_below_1_5x_is_info(self, node):
        # 240h stale vs a 192h (weekly-cadence) threshold: breach but < 1.5x.
        client = self._client_with_last_assignment(hours_ago=240)
        state: ExperimentMonitorState = {
            "stale_data_threshold_hours": 192.0,
            "include_synthetic": True,
            "status": "pending",
        }
        issue = await node._check_stale_data({"id": "e1"}, client, state)
        assert issue is not None
        assert issue["severity"] == "info"
        assert issue["threshold_hours"] == 192.0

    @pytest.mark.asyncio
    async def test_same_staleness_critical_under_default_threshold(self, node):
        # The SAME 240h staleness under the 24h live-feed default is > 3x: critical.
        client = self._client_with_last_assignment(hours_ago=240)
        state: ExperimentMonitorState = {
            "stale_data_threshold_hours": 24.0,
            "include_synthetic": True,
            "status": "pending",
        }
        issue = await node._check_stale_data({"id": "e1"}, client, state)
        assert issue is not None
        assert issue["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_within_threshold_no_issue(self, node):
        client = self._client_with_last_assignment(hours_ago=100)
        state: ExperimentMonitorState = {
            "stale_data_threshold_hours": 192.0,
            "include_synthetic": True,
            "status": "pending",
        }
        issue = await node._check_stale_data({"id": "e1"}, client, state)
        assert issue is None

    def _client_with_no_assignments(self):
        result = MagicMock()
        result.data = []
        query = MagicMock()
        query.select.return_value = query
        query.eq.return_value = query
        query.order.return_value = query
        query.limit.return_value = query
        query.execute = AsyncMock(return_value=result)
        client = MagicMock()
        client.table.return_value = query
        return client

    @pytest.mark.asyncio
    async def test_no_assignments_uses_the_same_ladder(self, node):
        """Never-enrolled experiments must follow the SAME documented contract
        (breach=info, 1.5x=warning, 3x=critical) — the branch used to jump to
        warning at breach and critical at 2x, so identical staleness got
        severities two tiers apart depending on whether any assignment existed."""
        client = self._client_with_no_assignments()
        state: ExperimentMonitorState = {
            "stale_data_threshold_hours": 192.0,
            "include_synthetic": True,
            "status": "pending",
        }
        for hours_ago, expected in ((240, "info"), (385, "warning"), (600, "critical")):
            created = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
            issue = await node._check_stale_data({"id": "e1", "created_at": created}, client, state)
            assert issue is not None, hours_ago
            assert issue["severity"] == expected, (hours_ago, issue["severity"])
            assert issue["last_data_timestamp"] == "N/A - No assignments"

    @pytest.mark.asyncio
    async def test_no_assignments_within_threshold_no_issue(self, node):
        client = self._client_with_no_assignments()
        state: ExperimentMonitorState = {
            "stale_data_threshold_hours": 192.0,
            "include_synthetic": True,
            "status": "pending",
        }
        created = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        issue = await node._check_stale_data({"id": "e1", "created_at": created}, client, state)
        assert issue is None
