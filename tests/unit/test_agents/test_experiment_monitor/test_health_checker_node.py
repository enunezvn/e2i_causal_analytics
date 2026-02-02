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
            "src.memory.services.factories.get_supabase_client",
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
            "src.memory.services.factories.get_supabase_client",
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
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None  # Trigger mock data path

            node = HealthCheckerNode()
            result = await node.execute(base_monitor_state)

            # Status should transition during execution
            assert result["status"] == "checking"

    @pytest.mark.asyncio
    async def test_execute_with_mock_data_when_no_client(self, base_monitor_state):
        """Test that mock data is used when no client available."""
        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            node = HealthCheckerNode()
            result = await node.execute(base_monitor_state)

            assert len(result["experiments"]) == 2  # Mock returns 2 experiments
            assert "No database client available" in str(result.get("warnings", []))

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
            "src.memory.services.factories.get_supabase_client",
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
            "src.memory.services.factories.get_supabase_client",
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
        mock_query.eq.assert_called_with("status", "running")

    @pytest.mark.asyncio
    async def test_get_experiments_specific_ids(self, node):
        """Test getting specific experiments by ID."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"id": "exp-1", "name": "Exp 1", "status": "running"}]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.in_ = MagicMock(return_value=mock_query)
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


class TestGetMockExperiments:
    """Tests for _get_mock_experiments method."""

    @pytest.mark.asyncio
    async def test_mock_experiments_returns_two(self, base_monitor_state):
        """Test that mock experiments returns 2 experiments."""
        node = HealthCheckerNode()
        result = await node._get_mock_experiments(base_monitor_state)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_mock_experiments_have_required_fields(self, base_monitor_state):
        """Test that mock experiments have required fields."""
        node = HealthCheckerNode()
        result = await node._get_mock_experiments(base_monitor_state)

        for exp in result:
            assert "id" in exp
            assert "name" in exp
            assert "status" in exp
            assert "config" in exp
            assert "created_at" in exp

    @pytest.mark.asyncio
    async def test_mock_experiments_have_valid_config(self, base_monitor_state):
        """Test that mock experiments have valid config."""
        node = HealthCheckerNode()
        result = await node._get_mock_experiments(base_monitor_state)

        for exp in result:
            config = exp["config"]
            assert "target_sample_size" in config
            assert "allocation_ratio" in config


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
        """Test that information fraction is calculated."""
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

        assert summary["current_information_fraction"] == 0.5  # 500/1000


class TestDetermineHealthStatus:
    """Tests for _determine_health_status method."""

    @pytest.fixture
    def node(self):
        return HealthCheckerNode()

    def test_critical_low_enrollment_after_14_days(self, node):
        """Test critical status for low enrollment after 14 days."""
        status = node._determine_health_status(
            enrollment_rate=1.5,  # < 2
            information_fraction=0.1,
            days_running=14,
        )
        assert status == "critical"

    def test_warning_low_enrollment_after_7_days(self, node):
        """Test warning status for low enrollment after 7 days."""
        status = node._determine_health_status(
            enrollment_rate=3.0,  # < 5
            information_fraction=0.2,
            days_running=7,
        )
        assert status == "warning"

    def test_warning_behind_schedule(self, node):
        """Test warning status when behind schedule."""
        status = node._determine_health_status(
            enrollment_rate=10.0,
            information_fraction=0.1,  # 10% when expected ~33% at day 10
            days_running=10,
        )
        assert status == "warning"

    def test_healthy_on_track(self, node):
        """Test healthy status when on track."""
        status = node._determine_health_status(
            enrollment_rate=15.0,
            information_fraction=0.5,
            days_running=10,
        )
        assert status == "healthy"

    def test_healthy_early_experiment(self, node):
        """Test healthy status for early experiment."""
        status = node._determine_health_status(
            enrollment_rate=5.0,
            information_fraction=0.1,
            days_running=3,
        )
        assert status == "healthy"

    def test_critical_takes_precedence_over_warning(self, node):
        """Test that critical status takes precedence."""
        # Very low enrollment after 14 days should be critical
        # even if information fraction is reasonable
        status = node._determine_health_status(
            enrollment_rate=1.0,  # Critical threshold
            information_fraction=0.4,
            days_running=14,
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
    async def test_experiment_with_zero_target_sample_size(self, node):
        """Test experiment with zero target sample size."""
        experiment = {
            "id": "exp-zero",
            "name": "Zero Target",
            "status": "running",
            "config": {"target_sample_size": 0},
            "created_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, None)

        # Should handle division by zero gracefully
        assert summary["current_information_fraction"] == 0

    @pytest.mark.asyncio
    async def test_experiment_with_missing_config(self, node):
        """Test experiment with missing config."""
        experiment = {
            "id": "exp-no-config",
            "name": "No Config",
            "status": "running",
            "created_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        }

        summary = await node._check_experiment_health(experiment, None)

        # Should use default target sample size of 1000
        assert summary is not None
        assert summary["experiment_id"] == "exp-no-config"

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
        status = node._determine_health_status(
            enrollment_rate=2.0,
            information_fraction=0.5,
            days_running=14,
        )
        # Rate < 2 is critical, rate == 2 is not critical
        assert status in ["healthy", "warning"]

        # Exactly at 7 days with rate exactly 5 (boundary)
        status = node._determine_health_status(
            enrollment_rate=5.0,
            information_fraction=0.5,
            days_running=7,
        )
        # Rate < 5 is warning, rate == 5 is not warning
        assert status == "healthy"
