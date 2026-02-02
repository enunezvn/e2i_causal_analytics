"""Tests for Fidelity Checker Node.

Tests cover:
- Node initialization
- Digital Twin fidelity checking
- Prediction error threshold comparison
- Severity determination based on error magnitude
- Calibration needed flag
- Edge cases and error handling
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.experiment_monitor.nodes.fidelity_checker import FidelityCheckerNode
from src.agents.experiment_monitor.state import FidelityIssue


class TestFidelityCheckerNodeInit:
    """Tests for FidelityCheckerNode initialization."""

    def test_node_initialization(self):
        """Test that node initializes correctly."""
        node = FidelityCheckerNode()
        assert node is not None
        assert node._client is None

    def test_multiple_node_instances(self):
        """Test creating multiple node instances."""
        node1 = FidelityCheckerNode()
        node2 = FidelityCheckerNode()
        assert node1 is not node2


class TestFidelityCheckerGetClient:
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

            node = FidelityCheckerNode()
            client = await node._get_client()

            assert client is mock_client
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

            node = FidelityCheckerNode()
            await node._get_client()
            await node._get_client()

            mock_get_client.assert_called_once()


class TestFidelityCheckerExecute:
    """Tests for execute method."""

    @pytest.fixture
    def state_with_experiments(self):
        """State with experiment summaries."""
        return {
            "query": "Check fidelity",
            "check_all_active": True,
            "experiment_ids": [],
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "experiments": [
                {
                    "experiment_id": "exp-001",
                    "name": "Test Experiment",
                    "status": "running",
                    "health_status": "healthy",
                    "days_running": 7,
                    "total_enrolled": 500,
                    "enrollment_rate": 71.43,
                    "current_information_fraction": 0.5,
                }
            ],
            "srm_issues": [],
            "enrollment_issues": [],
            "stale_data_issues": [],
            "fidelity_issues": [],
            "interim_triggers": [],
            "alerts": [],
            "monitor_summary": "",
            "recommended_actions": [],
            "check_latency_ms": 0,
            "experiments_checked": 1,
            "errors": [],
            "warnings": [],
            "status": "checking",
        }

    @pytest.mark.asyncio
    async def test_execute_with_no_experiments(self, base_monitor_state):
        """Test execute with empty experiments list."""
        node = FidelityCheckerNode()

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_client = MagicMock()
            mock_get.return_value = mock_client

            result = await node.execute(base_monitor_state)

            assert result["fidelity_issues"] == []

    @pytest.mark.asyncio
    async def test_execute_without_client(self, state_with_experiments):
        """Test execute adds warning when client unavailable."""
        node = FidelityCheckerNode()

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            result = await node.execute(state_with_experiments)

            assert result["fidelity_issues"] == []
            assert "No database client available for fidelity checks" in result["warnings"]

    @pytest.mark.asyncio
    async def test_execute_accumulates_latency(self, state_with_experiments):
        """Test that latency is accumulated."""
        node = FidelityCheckerNode()
        state_with_experiments["check_latency_ms"] = 100

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            result = await node.execute(state_with_experiments)

            assert result["check_latency_ms"] >= 100

    @pytest.mark.asyncio
    async def test_execute_handles_exceptions(self, state_with_experiments):
        """Test that exceptions are caught and recorded."""
        node = FidelityCheckerNode()

        with patch.object(node, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Fidelity check failed")

            result = await node.execute(state_with_experiments)

            assert len(result["errors"]) >= 1
            assert "Fidelity check failed" in result["errors"][-1]["error"]
            assert result["errors"][-1]["node"] == "fidelity_checker"

    @pytest.mark.asyncio
    async def test_execute_detects_fidelity_issue(self, state_with_experiments):
        """Test that fidelity issues are detected when error exceeds threshold."""
        node = FidelityCheckerNode()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-001",
                "simulated_ate": 0.15,
                "actual_ate": 0.08,
                "prediction_error": 0.467,  # > 0.2 threshold
                "fidelity_grade": "C",
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_client

            result = await node.execute(state_with_experiments)

            assert len(result["fidelity_issues"]) == 1
            issue = result["fidelity_issues"][0]
            assert issue["experiment_id"] == "exp-001"
            assert issue["prediction_error"] == 0.467

    @pytest.mark.asyncio
    async def test_execute_no_issue_below_threshold(self, state_with_experiments):
        """Test that no issue is created when error is below threshold."""
        node = FidelityCheckerNode()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-001",
                "simulated_ate": 0.10,
                "actual_ate": 0.09,
                "prediction_error": 0.10,  # < 0.2 threshold
                "fidelity_grade": "A",
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_client

            result = await node.execute(state_with_experiments)

            assert len(result["fidelity_issues"]) == 0


class TestCheckFidelity:
    """Tests for _check_fidelity method."""

    @pytest.fixture
    def node(self):
        return FidelityCheckerNode()

    @pytest.fixture
    def mock_client_with_data(self):
        """Create a mock client that returns fidelity data."""

        def create_mock(prediction_error, simulated_ate=0.15, actual_ate=0.08):
            mock_client = MagicMock()
            mock_result = MagicMock()
            mock_result.data = [
                {
                    "simulation_id": "sim-001",
                    "simulated_ate": simulated_ate,
                    "actual_ate": actual_ate,
                    "prediction_error": prediction_error,
                    "fidelity_grade": "C" if prediction_error > 0.2 else "A",
                }
            ]

            mock_query = MagicMock()
            mock_query.select = MagicMock(return_value=mock_query)
            mock_query.eq = MagicMock(return_value=mock_query)
            mock_query.order = MagicMock(return_value=mock_query)
            mock_query.limit = MagicMock(return_value=mock_query)
            mock_query.execute = AsyncMock(return_value=mock_result)
            mock_client.table = MagicMock(return_value=mock_query)

            return mock_client

        return create_mock

    @pytest.mark.asyncio
    async def test_returns_none_when_no_tracking_data(self, node):
        """Test returns None when no fidelity tracking data exists."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_error_below_threshold(self, node, mock_client_with_data):
        """Test returns None when prediction error is below threshold."""
        mock_client = mock_client_with_data(0.15)  # Below 0.2 threshold

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_issue_when_error_above_threshold(self, node, mock_client_with_data):
        """Test returns FidelityIssue when prediction error exceeds threshold."""
        mock_client = mock_client_with_data(0.25)  # Above 0.2 threshold

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is not None
        assert result["experiment_id"] == "exp-001"
        assert result["prediction_error"] == 0.25

    @pytest.mark.asyncio
    async def test_info_severity_for_moderate_error(self, node, mock_client_with_data):
        """Test info severity for error > threshold but < 2*threshold."""
        mock_client = mock_client_with_data(0.25)  # > 0.2 but < 0.4

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is not None
        assert result["severity"] == "info"
        assert result["calibration_needed"] is False

    @pytest.mark.asyncio
    async def test_warning_severity_for_severe_error(self, node, mock_client_with_data):
        """Test warning severity for error > 2*threshold."""
        mock_client = mock_client_with_data(0.50)  # > 0.4 (2*0.2)

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is not None
        assert result["severity"] == "warning"
        assert result["calibration_needed"] is True

    @pytest.mark.asyncio
    async def test_handles_negative_prediction_error(self, node, mock_client_with_data):
        """Test handles negative prediction errors (actual > predicted)."""
        mock_client = mock_client_with_data(-0.30, simulated_ate=0.08, actual_ate=0.15)

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is not None
        assert result["prediction_error"] == -0.30
        # abs(-0.30) = 0.30 > 0.2, so issue detected

    @pytest.mark.asyncio
    async def test_handles_database_exception(self, node):
        """Test gracefully handles database exceptions."""
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(side_effect=Exception("DB error"))
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        # Should return None instead of raising
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_correct_table(self, node):
        """Test queries the twin_fidelity_tracking table."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        await node._check_fidelity("exp-001", mock_client, 0.2)

        mock_client.table.assert_called_with("twin_fidelity_tracking")


class TestFidelityIssueSeverity:
    """Tests for fidelity issue severity determination."""

    @pytest.fixture
    def node(self):
        return FidelityCheckerNode()

    @pytest.fixture
    def create_mock_client(self):
        """Factory to create mock clients with specified prediction error."""

        def _create(prediction_error):
            mock_client = MagicMock()
            mock_result = MagicMock()
            mock_result.data = [
                {
                    "simulation_id": "sim-001",
                    "simulated_ate": 0.10,
                    "actual_ate": 0.10 + prediction_error,
                    "prediction_error": prediction_error,
                    "fidelity_grade": "C",
                }
            ]

            mock_query = MagicMock()
            mock_query.select = MagicMock(return_value=mock_query)
            mock_query.eq = MagicMock(return_value=mock_query)
            mock_query.order = MagicMock(return_value=mock_query)
            mock_query.limit = MagicMock(return_value=mock_query)
            mock_query.execute = AsyncMock(return_value=mock_result)
            mock_client.table = MagicMock(return_value=mock_query)

            return mock_client

        return _create

    @pytest.mark.asyncio
    async def test_threshold_boundary_below(self, node, create_mock_client):
        """Test at exactly threshold - should not trigger."""
        mock_client = create_mock_client(0.2)  # Exactly at threshold

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is None  # abs(0.2) > 0.2 is False

    @pytest.mark.asyncio
    async def test_threshold_boundary_above(self, node, create_mock_client):
        """Test just above threshold - should trigger with info severity."""
        mock_client = create_mock_client(0.21)  # Just above threshold

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is not None
        assert result["severity"] == "info"

    @pytest.mark.asyncio
    async def test_double_threshold_boundary(self, node, create_mock_client):
        """Test at exactly 2x threshold - warning severity."""
        mock_client = create_mock_client(0.41)  # Just above 2*0.2

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is not None
        assert result["severity"] == "warning"
        assert result["calibration_needed"] is True


class TestFidelityIssueStructure:
    """Tests for FidelityIssue structure and fields."""

    @pytest.fixture
    def node(self):
        return FidelityCheckerNode()

    @pytest.mark.asyncio
    async def test_issue_has_all_required_fields(self, node):
        """Test that FidelityIssue has all required fields."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-123",
                "simulated_ate": 0.15,
                "actual_ate": 0.08,
                "prediction_error": 0.467,
                "fidelity_grade": "C",
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_fidelity("exp-test", mock_client, 0.2)

        assert result is not None
        assert "experiment_id" in result
        assert "twin_simulation_id" in result
        assert "predicted_effect" in result
        assert "actual_effect" in result
        assert "prediction_error" in result
        assert "calibration_needed" in result
        assert "severity" in result

    @pytest.mark.asyncio
    async def test_issue_field_types(self, node):
        """Test that FidelityIssue fields have correct types."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-123",
                "simulated_ate": 0.15,
                "actual_ate": 0.08,
                "prediction_error": 0.467,
                "fidelity_grade": "C",
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_fidelity("exp-test", mock_client, 0.2)

        assert isinstance(result["experiment_id"], str)
        assert isinstance(result["twin_simulation_id"], str)
        assert isinstance(result["predicted_effect"], float)
        assert isinstance(result["actual_effect"], float)
        assert isinstance(result["prediction_error"], float)
        assert isinstance(result["calibration_needed"], bool)
        assert result["severity"] in ["info", "warning"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def node(self):
        return FidelityCheckerNode()

    @pytest.mark.asyncio
    async def test_zero_prediction_error(self, node):
        """Test with zero prediction error."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-001",
                "simulated_ate": 0.10,
                "actual_ate": 0.10,
                "prediction_error": 0.0,
                "fidelity_grade": "A+",
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is None  # Zero error is below any threshold

    @pytest.mark.asyncio
    async def test_missing_prediction_error_field(self, node):
        """Test handling of missing prediction_error field."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-001",
                "simulated_ate": 0.10,
                "actual_ate": 0.15,
                # Missing prediction_error
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        # Should handle gracefully (default to 0)
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_experiments_processed(self, node):
        """Test that all experiments are processed."""
        state = {
            "query": "",
            "check_all_active": True,
            "experiment_ids": [],
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "experiments": [
                {"experiment_id": "exp-1", "total_enrolled": 500},
                {"experiment_id": "exp-2", "total_enrolled": 600},
                {"experiment_id": "exp-3", "total_enrolled": 700},
            ],
            "srm_issues": [],
            "enrollment_issues": [],
            "stale_data_issues": [],
            "fidelity_issues": [],
            "interim_triggers": [],
            "alerts": [],
            "monitor_summary": "",
            "recommended_actions": [],
            "check_latency_ms": 0,
            "experiments_checked": 3,
            "errors": [],
            "warnings": [],
            "status": "checking",
        }

        # Track which experiments were checked
        checked_experiments = []

        async def mock_check_fidelity(exp_id, client, threshold):
            checked_experiments.append(exp_id)
            if exp_id == "exp-2":
                return FidelityIssue(
                    experiment_id=exp_id,
                    twin_simulation_id="sim-002",
                    predicted_effect=0.15,
                    actual_effect=0.08,
                    prediction_error=0.467,
                    calibration_needed=True,
                    severity="warning",
                )
            return None

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_client = MagicMock()
            mock_get.return_value = mock_client

            with patch.object(node, "_check_fidelity", side_effect=mock_check_fidelity):
                result = await node.execute(state)

        assert len(checked_experiments) == 3
        assert "exp-1" in checked_experiments
        assert "exp-2" in checked_experiments
        assert "exp-3" in checked_experiments
        assert len(result["fidelity_issues"]) == 1

    @pytest.mark.asyncio
    async def test_very_high_prediction_error(self, node):
        """Test with very high prediction error."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-001",
                "simulated_ate": 0.50,
                "actual_ate": 0.05,
                "prediction_error": 9.0,  # 900% error
                "fidelity_grade": "F",
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_fidelity("exp-001", mock_client, 0.2)

        assert result is not None
        assert result["severity"] == "warning"
        assert result["calibration_needed"] is True
        assert result["prediction_error"] == 9.0

    @pytest.mark.asyncio
    async def test_custom_threshold(self, node):
        """Test with custom fidelity threshold."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-001",
                "simulated_ate": 0.10,
                "actual_ate": 0.08,
                "prediction_error": 0.25,
                "fidelity_grade": "B",
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        # With threshold 0.3, error of 0.25 should not trigger
        result = await node._check_fidelity("exp-001", mock_client, 0.3)
        assert result is None

        # With threshold 0.2, error of 0.25 should trigger
        result = await node._check_fidelity("exp-001", mock_client, 0.2)
        assert result is not None


class TestAlternativeFidelitySource:
    """Tests for _get_fidelity_from_simulation_summary method."""

    @pytest.fixture
    def node(self):
        return FidelityCheckerNode()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_summary_data(self, node):
        """Test returns None when no simulation summary data exists."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._get_fidelity_from_simulation_summary("exp-001", mock_client, 0.2)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_prediction_error_is_none(self, node):
        """Test returns None when prediction_error is None in summary."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-001",
                "experiment_design_id": "exp-001",
                "simulated_ate": 0.10,
                "prediction_error": None,
                "fidelity_grade": None,
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._get_fidelity_from_simulation_summary("exp-001", mock_client, 0.2)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_issue_from_summary(self, node):
        """Test returns issue from simulation summary when error exceeds threshold."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "simulation_id": "sim-001",
                "experiment_design_id": "exp-001",
                "simulated_ate": 0.15,
                "prediction_error": 0.35,
                "fidelity_grade": "C",
            }
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._get_fidelity_from_simulation_summary("exp-001", mock_client, 0.2)

        assert result is not None
        assert result["experiment_id"] == "exp-001"
        assert result["actual_effect"] == 0.0  # Not available in summary

    @pytest.mark.asyncio
    async def test_summary_uses_correct_table(self, node):
        """Test queries the v_simulation_summary view."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        await node._get_fidelity_from_simulation_summary("exp-001", mock_client, 0.2)

        mock_client.table.assert_called_with("v_simulation_summary")

    @pytest.mark.asyncio
    async def test_summary_handles_exception(self, node):
        """Test handles database exception gracefully."""
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(side_effect=Exception("View not found"))
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._get_fidelity_from_simulation_summary("exp-001", mock_client, 0.2)

        assert result is None


class TestStateUpdates:
    """Tests for state update behavior."""

    @pytest.fixture
    def node(self):
        return FidelityCheckerNode()

    @pytest.mark.asyncio
    async def test_preserves_existing_state_fields(self, node):
        """Test that execute preserves existing state fields."""
        state = {
            "query": "Original query",
            "check_all_active": True,
            "experiment_ids": ["exp-existing"],
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "experiments": [],
            "srm_issues": [{"experiment_id": "exp-srm"}],
            "enrollment_issues": [{"experiment_id": "exp-enrollment"}],
            "stale_data_issues": [],
            "fidelity_issues": [],
            "interim_triggers": [],
            "alerts": [],
            "monitor_summary": "Previous summary",
            "recommended_actions": ["Action 1"],
            "check_latency_ms": 500,
            "experiments_checked": 5,
            "errors": [],
            "warnings": ["Previous warning"],
            "status": "checking",
        }

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            result = await node.execute(state)

        # Verify original fields preserved
        assert result["query"] == "Original query"
        assert result["srm_issues"] == [{"experiment_id": "exp-srm"}]
        assert result["enrollment_issues"] == [{"experiment_id": "exp-enrollment"}]
        assert result["monitor_summary"] == "Previous summary"
        assert result["recommended_actions"] == ["Action 1"]
        assert result["experiments_checked"] == 5

    @pytest.mark.asyncio
    async def test_updates_fidelity_issues_only(self, node):
        """Test that execute only updates fidelity-related fields."""
        state = {
            "query": "",
            "check_all_active": True,
            "experiment_ids": [],
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "experiments": [{"experiment_id": "exp-1"}],
            "srm_issues": [],
            "enrollment_issues": [],
            "stale_data_issues": [],
            "fidelity_issues": [{"old": "issue"}],  # Existing issue should be replaced
            "interim_triggers": [],
            "alerts": [],
            "monitor_summary": "",
            "recommended_actions": [],
            "check_latency_ms": 100,
            "experiments_checked": 1,
            "errors": [],
            "warnings": [],
            "status": "checking",
        }

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            result = await node.execute(state)

        # fidelity_issues should be overwritten, not appended
        assert result["fidelity_issues"] == []
        assert {"old": "issue"} not in result["fidelity_issues"]
